"""Budget and incomplete-work accounting adapted from deep-research-harness.

This is a small framework adapter, not an independently designed policy.
It ports the ``RunBudget.open_report_allowance`` and unresolved-work semantics
from ``deep_research/research.py`` and ``deep_research/models.py`` at
``393d907239ee649f85dd888de715d8379e8b4a87``.  The source project's complete
MIT license is in ``HARNESS_LICENSE`` and the exact mapping is in ``NOTICE.md``.

The local runtime has synchronous LangChain invocations rather than Pydantic
AI's token-aware ``RunUsage``.  It therefore preserves the upstream boundary
with model-call accounting: research has one finite allowance; immediately
before every write-up pass, the report receives a fresh allowance measured from
what research actually consumed.  Provider-reported token and cost telemetry
remain in ``ModelUsageLedger`` and are intentionally not guessed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ResearchBudgetExhausted(RuntimeError):
    """Raised before a role invocation when its phase allowance is exhausted."""


@dataclass(slots=True)
class OdrRunBudget:
    """One run-wide research allowance plus a guaranteed report allowance.

    Direct adaptation target: ``deep_research.research.RunBudget``.  The
    upstream implementation shares a token/USD/request accumulator.  Here a
    pre-call model-invocation counter is the enforceable common denominator,
    but the important semantics stay identical:

    * research calls draw only from ``research_model_call_limit``;
    * ``open_report_allowance`` records actual prior usage;
    * a report pass can make ``report_model_call_allowance`` calls on top of
      that usage, even if research already reached its ceiling;
    * each later rewrite pass receives a fresh allowance.

    ``None`` remains a deliberate comparison mode for research only.  Report
    calls are still explicitly bounded unless the caller changes the policy.
    """

    research_model_call_limit: int | None
    report_model_call_allowance: int
    total_model_calls: int = 0
    research_model_calls: int = 0
    report_model_calls: int = 0
    _report_call_limit: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.research_model_call_limit is not None and self.research_model_call_limit < 1:
            raise ValueError("research_model_call_limit must be positive or None")
        if self.report_model_call_allowance < 1:
            raise ValueError("report_model_call_allowance must be positive")

    def authorize_research_call(self) -> int:
        """Charge one research call against the run-wide research allowance."""

        if (
            self.research_model_call_limit is not None
            and self.research_model_calls >= self.research_model_call_limit
        ):
            raise ResearchBudgetExhausted("research_model_call_limit_reached")
        self.total_model_calls += 1
        self.research_model_calls += 1
        return self.total_model_calls

    def open_report_allowance(self) -> None:
        """Grant a fresh report allowance on top of actual completed usage.

        This intentionally never reserves a fixed fraction before research.
        A research overrun or a fully used research cap therefore cannot leave
        the user with paid-for research and no final report.
        """

        self._report_call_limit = self.total_model_calls + self.report_model_call_allowance

    def authorize_report_call(self) -> int:
        if self._report_call_limit is None:
            raise RuntimeError("open_report_allowance() must run before a report call")
        if self.total_model_calls >= self._report_call_limit:
            raise ResearchBudgetExhausted("report_model_call_allowance_reached")
        self.total_model_calls += 1
        self.report_model_calls += 1
        return self.total_model_calls

    def to_dict(self) -> dict[str, object]:
        return {
            "research_model_call_limit": self.research_model_call_limit,
            "report_model_call_allowance": self.report_model_call_allowance,
            "total_model_calls": self.total_model_calls,
            "research_model_calls": self.research_model_calls,
            "report_model_calls": self.report_model_calls,
            "report_call_limit": self._report_call_limit,
        }


@dataclass(slots=True)
class OdrResearchState:
    """Minimal persisted-state projection of the upstream ``ResearchState``.

    ``unresolved_tasks`` is deliberately separate from a task queue.  A queue
    is emptied while work progresses, so it cannot truthfully answer which
    planned tasks never received research when a breadth/depth/budget limit
    stops the run.
    """

    breadth_budget: int
    depth_budget: int
    research_tasks: list[str] = field(default_factory=list)
    research_notes: list[str] = field(default_factory=list)
    unresolved_tasks: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    round_index: int = 0
    status: str = "researching"

    def record_unresolved(self, tasks: list[str]) -> None:
        for task in tasks:
            normalized = task.strip() if isinstance(task, str) else ""
            if (
                normalized
                and normalized not in self.completed_tasks
                and normalized not in self.unresolved_tasks
            ):
                self.unresolved_tasks.append(normalized)

    def record_completed(self, task: str, note: str) -> None:
        if task not in self.completed_tasks:
            self.completed_tasks.append(task)
        self.unresolved_tasks = [item for item in self.unresolved_tasks if item != task]
        self.research_notes.append(note)

    @property
    def breadth_exhausted(self) -> bool:
        return len(self.research_tasks) >= self.breadth_budget

    @property
    def depth_exhausted(self) -> bool:
        """``depth_budget`` follows upstream: max follow-up rounds after initial work."""

        return self.round_index >= self.depth_budget

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "odr-harness-state/v1",
            "breadth_budget": self.breadth_budget,
            "depth_budget": self.depth_budget,
            "research_tasks": list(self.research_tasks),
            "research_notes_count": len(self.research_notes),
            "completed_tasks": list(self.completed_tasks),
            "unresolved_tasks": list(self.unresolved_tasks),
            "round_index": self.round_index,
            "status": self.status,
        }


__all__ = ["OdrResearchState", "OdrRunBudget", "ResearchBudgetExhausted"]
