"""Bounded, failure-isolated parallel research execution.

This module adopts the useful part of supervisor/worker deep-research
architectures: independent research tasks run concurrently.  It deliberately
does *not* share a mutable candidate window, conversation transcript, or
unreserved global budget between workers.  Each task receives a task-scoped
plan and a fixed slice of the remaining envelope, then returns a typed result
for the supervisor to inspect.

The implementation is framework-free.  It wraps the existing capability-gated
``GeneralActionLoop`` rather than exposing raw tool calls to a new agent
runtime.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

from .controller import ActionLoopResult, Controller
from .schemas import CoverageState, ResearchPlan, ResearchTask
from .workflow import BudgetSnapshot, GeneralResearchCancelled, ResearchStageResult


class ParallelResearchError(ValueError):
    """A caller asked the parallel executor to violate its fixed contract."""


@runtime_checkable
class ActionLoop(Protocol):
    """The task-local action-loop surface needed by the coordinator."""

    def run(
        self,
        *,
        plan: ResearchPlan,
        coverage: CoverageState | None,
        budget: BudgetSnapshot,
        controller: Controller,
    ) -> ActionLoopResult:
        """Execute one bounded controller loop."""


LoopFactory = Callable[[ResearchTask], ActionLoop]
ControllerFactory = Callable[[ResearchTask], Controller]


@dataclass(frozen=True, slots=True)
class WorkerResearchResult:
    """One worker's independent outcome, including non-fatal failure state."""

    task: ResearchTask
    allocated_budget: BudgetSnapshot
    action_result: ActionLoopResult | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, ResearchTask):
            raise TypeError("task must be ResearchTask")
        if not isinstance(self.allocated_budget, BudgetSnapshot):
            raise TypeError("allocated_budget must be BudgetSnapshot")
        if self.action_result is not None and not isinstance(
            self.action_result, ActionLoopResult
        ):
            raise TypeError("action_result must be ActionLoopResult or None")
        if self.failure_code is not None:
            if not isinstance(self.failure_code, str) or not self.failure_code:
                raise ValueError("failure_code must be a non-empty string or None")
            if self.action_result is not None:
                raise ValueError("a failed worker must not also contain an action result")
        elif self.action_result is None:
            raise ValueError("a worker requires an action result or a failure code")

    @property
    def model_calls(self) -> int:
        return self.action_result.model_calls if self.action_result is not None else 0

    @property
    def tool_calls(self) -> int:
        return self.action_result.tool_calls if self.action_result is not None else 0

    @property
    def terminal_reason(self) -> str:
        return (
            self.action_result.terminal_reason
            if self.action_result is not None
            else self.failure_code or "worker_runtime_error"
        )


@dataclass(frozen=True, slots=True)
class ParallelResearchResult:
    """Merged, deterministic projection of one concurrent supervisor dispatch."""

    workers: tuple[WorkerResearchResult, ...]

    def __post_init__(self) -> None:
        workers = tuple(self.workers)
        if not workers:
            raise ValueError("ParallelResearchResult requires at least one worker")
        if not all(isinstance(worker, WorkerResearchResult) for worker in workers):
            raise TypeError("workers must contain WorkerResearchResult values")
        task_ids = [worker.task.task_id for worker in workers]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("workers must use unique task IDs")
        object.__setattr__(self, "workers", workers)

    @property
    def model_calls(self) -> int:
        return sum(worker.model_calls for worker in self.workers)

    @property
    def tool_calls(self) -> int:
        return sum(worker.tool_calls for worker in self.workers)

    @property
    def failure_task_ids(self) -> tuple[str, ...]:
        return tuple(
            worker.task.task_id
            for worker in self.workers
            if worker.failure_code is not None
        )

    @property
    def pages(self):
        """Merge snapshots deterministically, rejecting conflicting duplicates."""

        pages_by_source: dict[str, object] = {}
        for worker in self.workers:
            if worker.action_result is None:
                continue
            for page in worker.action_result.pages:
                existing = pages_by_source.get(page.source.source_id)
                if existing is None:
                    pages_by_source[page.source.source_id] = page
                    continue
                if existing.source.content_hash != page.source.content_hash:
                    raise ParallelResearchError(
                        "one source ID cannot identify two different worker snapshots"
                    )
        return tuple(pages_by_source[source_id] for source_id in sorted(pages_by_source))


def task_scoped_plan(plan: ResearchPlan, task: ResearchTask) -> ResearchPlan:
    """Project the frozen plan to exactly one assigned worker task."""

    if not isinstance(plan, ResearchPlan):
        raise TypeError("plan must be ResearchPlan")
    if not isinstance(task, ResearchTask):
        raise TypeError("task must be ResearchTask")
    items_by_id = {item.item_id: item for item in plan.items}
    unknown = [item_id for item_id in task.plan_item_ids if item_id not in items_by_id]
    if unknown:
        raise ParallelResearchError(
            "task refers to plan item(s) absent from the active plan: "
            + ", ".join(sorted(unknown))
        )
    return ResearchPlan(
        plan_id=plan.plan_id,
        run_id=plan.run_id,
        query=task.question,
        created_at=plan.created_at,
        items=tuple(items_by_id[item_id] for item_id in task.plan_item_ids),
    )


def _budget_slices(total: int, count: int) -> tuple[int, ...]:
    if total < count:
        raise ParallelResearchError(
            "remaining budget must allocate at least one unit to every dispatched worker"
        )
    quotient, remainder = divmod(total, count)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(count))


def _worker_budget(
    parent: BudgetSnapshot, *, model_calls: int, tool_calls: int
) -> BudgetSnapshot:
    return BudgetSnapshot(
        model_calls_used=parent.model_calls_used,
        tool_calls_used=parent.tool_calls_used,
        model_calls_remaining=model_calls,
        tool_calls_remaining=tool_calls,
    )


class ParallelResearchCoordinator:
    """Run one supervisor dispatch with fixed concurrency and budget partitions.

    The coordinator does not decide tasks itself; that remains the supervisor's
    responsibility.  It also does not raise a generic worker exception as a
    whole-run exception.  The caller receives one failure record per failed
    task and can decide whether to retry a gap in a subsequent round.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        loop_factory: LoopFactory,
        controller_factory: ControllerFactory,
    ) -> None:
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or not 1 <= max_workers <= 20
        ):
            raise ValueError("max_workers must be an integer between 1 and 20")
        if not callable(loop_factory) or not callable(controller_factory):
            raise TypeError("loop_factory and controller_factory must be callable")
        self.max_workers = max_workers
        self._loop_factory = loop_factory
        self._controller_factory = controller_factory

    def _run_one(
        self,
        *,
        task: ResearchTask,
        plan: ResearchPlan,
        coverage: CoverageState | None,
        budget: BudgetSnapshot,
    ) -> WorkerResearchResult:
        try:
            loop = self._loop_factory(task)
            if not isinstance(loop, ActionLoop):
                raise TypeError("loop_factory must return an ActionLoop")
            controller = self._controller_factory(task)
            if not callable(controller):
                raise TypeError("controller_factory must return a controller")
            return WorkerResearchResult(
                task=task,
                allocated_budget=budget,
                action_result=loop.run(
                    plan=task_scoped_plan(plan, task),
                    coverage=coverage,
                    budget=budget,
                    controller=controller,
                ),
            )
        except GeneralResearchCancelled:
            raise
        except Exception:
            return WorkerResearchResult(
                task=task,
                allocated_budget=budget,
                failure_code="worker_runtime_error",
            )

    def run(
        self,
        *,
        tasks: Iterable[ResearchTask],
        plan: ResearchPlan,
        coverage: CoverageState | None,
        budget: BudgetSnapshot,
    ) -> ParallelResearchResult:
        """Dispatch at most ``max_workers`` independent tasks concurrently."""

        task_list = tuple(tasks)
        if not task_list:
            raise ParallelResearchError("parallel research requires at least one task")
        if len(task_list) > self.max_workers:
            raise ParallelResearchError(
                "supervisor dispatched more tasks than the configured worker limit"
            )
        if not all(isinstance(task, ResearchTask) for task in task_list):
            raise TypeError("tasks must contain ResearchTask values")
        task_ids = [task.task_id for task in task_list]
        if len(task_ids) != len(set(task_ids)):
            raise ParallelResearchError("dispatched task IDs must be unique")
        if not isinstance(plan, ResearchPlan):
            raise TypeError("plan must be ResearchPlan")
        if coverage is not None and not isinstance(coverage, CoverageState):
            raise TypeError("coverage must be CoverageState or None")
        if not isinstance(budget, BudgetSnapshot):
            raise TypeError("budget must be BudgetSnapshot")

        count = len(task_list)
        model_slices = _budget_slices(budget.model_calls_remaining, count)
        tool_slices = _budget_slices(budget.tool_calls_remaining, count)
        allocations = tuple(
            _worker_budget(
                budget, model_calls=model_slices[index], tool_calls=tool_slices[index]
            )
            for index in range(count)
        )
        futures: dict[str, Future[WorkerResearchResult]] = {}
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="general-research") as executor:
            for task, allocation in zip(task_list, allocations, strict=True):
                futures[task.task_id] = executor.submit(
                    self._run_one,
                    task=task,
                    plan=plan,
                    coverage=coverage,
                    budget=allocation,
                )
            results = tuple(futures[task.task_id].result() for task in task_list)
        return ParallelResearchResult(workers=results)


class ParallelActionLoopResearcher:
    """Adapt a bounded coordinator to the workflow's research-stage contract.

    The adapter retains every worker's local action sequence under its task
    ID.  The workflow can therefore expose an interleaved fan-out as a
    replayable audit trail without pretending that parallel steps share one
    global sequence number.
    """

    def __init__(self, coordinator: ParallelResearchCoordinator) -> None:
        if not isinstance(coordinator, ParallelResearchCoordinator):
            raise TypeError("coordinator must be ParallelResearchCoordinator")
        self._coordinator = coordinator

    def __call__(
        self,
        plan: ResearchPlan,
        tasks: tuple[ResearchTask, ...],
        coverage: CoverageState | None,
        budget: BudgetSnapshot,
    ) -> ResearchStageResult:
        result = self._coordinator.run(
            tasks=tasks,
            plan=plan,
            coverage=coverage,
            budget=budget,
        )
        task_transitions = tuple(
            (
                worker.task.task_id,
                worker.action_result.transitions
                if worker.action_result is not None
                else (),
            )
            for worker in result.workers
        )
        terminal_reason = ";".join(
            f"{worker.task.task_id}:{worker.terminal_reason}" for worker in result.workers
        )
        return ResearchStageResult(
            pages=result.pages,
            model_calls=result.model_calls,
            tool_calls=result.tool_calls,
            worker_count=len(result.workers),
            controller_terminal_reason=terminal_reason,
            research_tasks=tasks,
            task_action_transitions=task_transitions,
        )


__all__ = [
    "ActionLoop",
    "ControllerFactory",
    "LoopFactory",
    "ParallelActionLoopResearcher",
    "ParallelResearchCoordinator",
    "ParallelResearchError",
    "ParallelResearchResult",
    "WorkerResearchResult",
    "task_scoped_plan",
]
