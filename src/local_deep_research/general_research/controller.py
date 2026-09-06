"""Bounded controller loop for General Research V1.

The controller is deliberately a small state machine, not a generic function
calling agent.  It receives only a plan, coverage state, budget, safe candidate
views, and stable feedback codes.  Resource locators, connector exceptions,
credentials, and raw page contents stay outside this model-visible boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse

from .actions import (
    ActionGate,
    ActionGateError,
    ActionOutcome,
    ActionTransition,
    FetchAction,
    GeneralAction,
    RequestStopAction,
    SearchAction,
    SearchCandidate,
)
from .artifact_store import ContentArtifactStore
from .config import GeneralExecutionConfig
from .connectors import (
    ConnectorRegistry,
    DiscoveredResource,
    FetchedResource,
    make_candidate,
    materialize_fetched_page,
    source_id_for_candidate,
)
from .schemas import CoverageState, ResearchPlan
from .workflow import (
    BudgetSnapshot,
    FetchedPage,
    GeneralResearchCancelled,
    ResearchStageResult,
)


class ControllerContractError(ValueError):
    """The injected controller returned a result outside the fixed protocol."""


class _FetchedContentLimitError(ValueError):
    """A connector returned more content than this frozen run permits."""


@dataclass(frozen=True, slots=True)
class ControllerObservation:
    """The complete, locator-free state visible to the controller model."""

    plan: ResearchPlan
    coverage: CoverageState | None
    available_connector_ids: tuple[str, ...]
    candidates: tuple[dict[str, str], ...]
    fetched_candidate_ids: tuple[str, ...]
    failed_candidate_ids: tuple[str, ...]
    budget: BudgetSnapshot
    feedback_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ResearchPlan):
            raise TypeError("plan must be ResearchPlan")
        if self.coverage is not None and not isinstance(self.coverage, CoverageState):
            raise TypeError("coverage must be CoverageState or None")
        connectors = tuple(str(value).strip() for value in self.available_connector_ids)
        if not connectors or any(not value for value in connectors):
            raise ValueError("available_connector_ids must be non-empty")
        object.__setattr__(self, "available_connector_ids", tuple(dict.fromkeys(connectors)))
        normalized_candidates: list[dict[str, str]] = []
        for candidate in self.candidates:
            if not isinstance(candidate, dict):
                raise TypeError("candidates must contain model-view dictionaries")
            copied = {str(key): str(value) for key, value in candidate.items()}
            if "resource_locator" in copied:
                raise ValueError("controller candidates must not expose resource_locator")
            if not {"candidate_id", "connector_id", "source_channel", "title", "snippet"}.issubset(copied):
                raise ValueError("controller candidate view is incomplete")
            normalized_candidates.append(copied)
        object.__setattr__(self, "candidates", tuple(normalized_candidates))
        candidate_ids = {candidate["candidate_id"] for candidate in normalized_candidates}
        fetched = tuple(str(value).strip() for value in self.fetched_candidate_ids)
        if any(not value for value in fetched) or not set(fetched).issubset(candidate_ids):
            raise ValueError("fetched_candidate_ids must refer to observed candidates")
        object.__setattr__(self, "fetched_candidate_ids", tuple(dict.fromkeys(fetched)))
        failed = tuple(str(value).strip() for value in self.failed_candidate_ids)
        if any(not value for value in failed) or not set(failed).issubset(candidate_ids):
            raise ValueError("failed_candidate_ids must refer to observed candidates")
        if set(failed).intersection(fetched):
            raise ValueError("a candidate cannot be both fetched and failed")
        object.__setattr__(self, "failed_candidate_ids", tuple(dict.fromkeys(failed)))
        feedback = tuple(str(value).strip() for value in self.feedback_codes)
        if any(not value or len(value) > 128 for value in feedback):
            raise ValueError("feedback_codes must be short, non-empty identifiers")
        object.__setattr__(self, "feedback_codes", tuple(dict.fromkeys(feedback)))


@dataclass(frozen=True, slots=True)
class ControllerStageResult:
    """One model decision; parsing happens before it reaches the runtime loop."""

    action: GeneralAction
    model_calls: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.action, (SearchAction, FetchAction, RequestStopAction)):
            raise TypeError("action must be a GeneralAction")
        if (
            isinstance(self.model_calls, bool)
            or not isinstance(self.model_calls, int)
            or self.model_calls < 1
        ):
            raise ControllerContractError("controller must account for at least one model call")


Controller = Callable[[ControllerObservation], ControllerStageResult]
Clock = Callable[[], str]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class ActionLoopResult:
    """One bounded research pass suitable for ``GeneralResearchWorkflow``."""

    pages: tuple[FetchedPage, ...]
    transitions: tuple[ActionTransition, ...]
    model_calls: int
    tool_calls: int
    terminal_reason: str

    def __post_init__(self) -> None:
        if not all(isinstance(page, FetchedPage) for page in self.pages):
            raise TypeError("pages must contain FetchedPage values")
        if not all(isinstance(item, ActionTransition) for item in self.transitions):
            raise TypeError("transitions must contain ActionTransition values")
        if self.model_calls < 0 or self.tool_calls < 0:
            raise ValueError("call counts must be non-negative")
        if not isinstance(self.terminal_reason, str) or not self.terminal_reason:
            raise ValueError("terminal_reason must be non-empty")

    def as_research_stage(self) -> ResearchStageResult:
        return ResearchStageResult(
            pages=self.pages,
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            action_transitions=self.transitions,
            controller_terminal_reason=self.terminal_reason,
        )


@dataclass(frozen=True, slots=True)
class _CandidateBinding:
    candidate: SearchCandidate
    resource: DiscoveredResource


class GeneralActionLoop:
    """Run controller proposals through capability, budget, and evidence gates."""

    def __init__(
        self,
        *,
        config: GeneralExecutionConfig,
        registry: ConnectorRegistry,
        artifact_store: ContentArtifactStore,
        authorized_connector_ids: tuple[str, ...] = (),
        clock: Clock = _utc_timestamp,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(config, GeneralExecutionConfig):
            raise TypeError("config must be GeneralExecutionConfig")
        if not isinstance(registry, ConnectorRegistry):
            raise TypeError("registry must be ConnectorRegistry")
        if not isinstance(artifact_store, ContentArtifactStore):
            raise TypeError("artifact_store must be ContentArtifactStore")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if should_cancel is not None and not callable(should_cancel):
            raise TypeError("should_cancel must be callable or None")
        declared_ids = {item.connector_id for item in config.source_connectors}
        if registry.connector_ids != declared_ids:
            raise ValueError(
                "connector registry must bind exactly the connectors declared by the run config"
            )
        self.config = config
        self.registry = registry
        self.artifact_store = artifact_store
        self._clock = clock
        self._should_cancel = should_cancel
        self._gate = ActionGate(
            config.source_connectors,
            authorized_connector_ids=authorized_connector_ids,
        )
        authorized = frozenset(authorized_connector_ids)
        self._available_connector_ids = tuple(
            connector.connector_id
            for connector in config.source_connectors
            if not connector.requires_user_authorization
            or connector.connector_id in authorized
        )
        if not self._available_connector_ids:
            raise ValueError("no source connector is authorized for this run")

    def _ensure_not_cancelled(self) -> None:
        if self._should_cancel is not None and self._should_cancel():
            raise GeneralResearchCancelled("general_research_cancelled")

    def _resource_scheme_is_allowed(
        self, *, resource: DiscoveredResource, binding
    ) -> bool:
        """Apply run-owned URL policy even for a third-party connector.

        Public connector implementations may normalize their own results, but
        the action loop is the final generic capability boundary. Local
        collection locators use a separate declared channel and are therefore
        not compared to the public HTTP(S) allowlist.
        """

        if binding.config.source_channel.value != "public_web":
            return True
        return urlparse(resource.resource_locator).scheme.casefold() in set(
            self.config.run.allowed_url_schemes
        )

    def _observation(
        self,
        *,
        plan: ResearchPlan,
        coverage: CoverageState | None,
        candidates: dict[str, _CandidateBinding],
        fetched_candidate_ids: set[str],
        failed_candidate_ids: set[str],
        budget: BudgetSnapshot,
        model_calls_used: int,
        tool_calls_used: int,
        feedback_codes: tuple[str, ...],
    ) -> ControllerObservation:
        remaining_models = max(0, budget.model_calls_remaining - model_calls_used)
        remaining_tools = max(0, budget.tool_calls_remaining - tool_calls_used)
        return ControllerObservation(
            plan=plan,
            coverage=coverage,
            available_connector_ids=self._available_connector_ids,
            candidates=tuple(
                binding.candidate.model_view()
                for _, binding in sorted(candidates.items())
            ),
            fetched_candidate_ids=tuple(sorted(fetched_candidate_ids)),
            failed_candidate_ids=tuple(sorted(failed_candidate_ids)),
            budget=BudgetSnapshot(
                model_calls_used=budget.model_calls_used + model_calls_used,
                tool_calls_used=budget.tool_calls_used + tool_calls_used,
                model_calls_remaining=remaining_models,
                tool_calls_remaining=remaining_tools,
            ),
            feedback_codes=feedback_codes,
        )

    @staticmethod
    def _reject(
        transitions: list[ActionTransition],
        *,
        step: int,
        action: GeneralAction,
        error_code: str,
    ) -> tuple[str, ...]:
        transitions.append(
            ActionTransition.for_action(
                step=step,
                action=action,
                outcome=ActionOutcome.REJECTED,
                error_code=error_code,
            )
        )
        return (error_code,)

    @staticmethod
    def _fetch_failure_code(error: Exception) -> str:
        """Return a bounded connector code without retaining provider detail."""

        code = getattr(error, "failure_code", None)
        if code in {
            "public_fetch_authorization_failed",
            "public_fetch_empty",
            "public_fetch_timeout",
            "public_fetch_transport_failed",
        }:
            return code
        return "connector_fetch_failed"

    def run(
        self,
        *,
        plan: ResearchPlan,
        coverage: CoverageState | None,
        budget: BudgetSnapshot,
        controller: Controller,
    ) -> ActionLoopResult:
        """Execute until a safe stop request or one frozen budget becomes binding."""

        if not isinstance(plan, ResearchPlan):
            raise TypeError("plan must be ResearchPlan")
        if coverage is not None and not isinstance(coverage, CoverageState):
            raise TypeError("coverage must be CoverageState or None")
        if not isinstance(budget, BudgetSnapshot):
            raise TypeError("budget must be BudgetSnapshot")
        if not callable(controller):
            raise TypeError("controller must be callable")

        candidates: dict[str, _CandidateBinding] = {}
        resource_keys: set[tuple[str, str]] = set()
        fetched_source_ids: set[str] = set()
        fetched_candidate_ids: set[str] = set()
        failed_candidate_ids: set[str] = set()
        pages: list[FetchedPage] = []
        transitions: list[ActionTransition] = []
        feedback_codes: tuple[str, ...] = ()
        model_calls = 0
        tool_calls = 0
        step = 0

        while model_calls < budget.model_calls_remaining:
            self._ensure_not_cancelled()
            observation = self._observation(
                plan=plan,
                coverage=coverage,
                candidates=candidates,
                fetched_candidate_ids=fetched_candidate_ids,
                failed_candidate_ids=failed_candidate_ids,
                budget=budget,
                model_calls_used=model_calls,
                tool_calls_used=tool_calls,
                feedback_codes=feedback_codes,
            )
            try:
                stage = controller(observation)
            except GeneralResearchCancelled:
                raise
            except Exception:
                return ActionLoopResult(
                    pages=tuple(pages),
                    transitions=tuple(transitions),
                    model_calls=model_calls,
                    tool_calls=tool_calls,
                    terminal_reason="controller_runtime_error",
                )
            if not isinstance(stage, ControllerStageResult):
                raise ControllerContractError("controller must return ControllerStageResult")
            if model_calls + stage.model_calls > budget.model_calls_remaining:
                raise ControllerContractError("controller reported model calls beyond remaining budget")
            model_calls += stage.model_calls
            action = stage.action
            try:
                self._gate.validate(
                    action,
                    candidates={key: value.candidate for key, value in candidates.items()},
                )
            except ActionGateError:
                feedback_codes = self._reject(
                    transitions,
                    step=step,
                    action=action,
                    error_code="action_not_authorized_or_not_observed",
                )
                step += 1
                continue

            if isinstance(action, RequestStopAction):
                transitions.append(
                    ActionTransition.for_action(
                        step=step, action=action, outcome=ActionOutcome.STOPPED
                    )
                )
                return ActionLoopResult(
                    pages=tuple(pages),
                    transitions=tuple(transitions),
                    model_calls=model_calls,
                    tool_calls=tool_calls,
                    terminal_reason="controller_requested_stop",
                )

            if tool_calls >= budget.tool_calls_remaining:
                transitions.append(
                    ActionTransition.for_action(
                        step=step,
                        action=action,
                        outcome=ActionOutcome.REJECTED,
                        error_code="tool_budget_exhausted",
                    )
                )
                return ActionLoopResult(
                    pages=tuple(pages),
                    transitions=tuple(transitions),
                    model_calls=model_calls,
                    tool_calls=tool_calls,
                    terminal_reason="controller_tool_budget_exhausted",
                )

            if isinstance(action, SearchAction):
                if (
                    len(candidates)
                    >= self.config.research_control.max_total_candidates
                ):
                    feedback_codes = self._reject(
                        transitions,
                        step=step,
                        action=action,
                        error_code="candidate_window_full",
                    )
                    step += 1
                    continue
                try:
                    self._ensure_not_cancelled()
                    binding = self.registry.get(action.connector_id)
                    discovered = iter(binding.connector.search(action.query))
                except GeneralResearchCancelled:
                    raise
                except Exception:
                    tool_calls += 1
                    transitions.append(
                        ActionTransition.for_action(
                            step=step,
                            action=action,
                            outcome=ActionOutcome.FAILED,
                            error_code="connector_search_failed",
                        )
                    )
                    feedback_codes = ("connector_search_failed",)
                    step += 1
                    continue
                tool_calls += 1
                added = 0
                candidate_limit = min(
                    self.config.research_control.max_candidates_per_search,
                    self.config.research_control.max_total_candidates
                    - len(candidates),
                )
                # Search connectors are untrusted runtime dependencies.  Scan
                # a bounded multiple of the visible window so an endless or
                # duplicate-heavy iterator cannot grow controller state or
                # consume unbounded CPU before the model receives a result.
                max_scanned = candidate_limit * 4
                try:
                    for scanned, resource in enumerate(discovered, start=1):
                        if scanned > max_scanned:
                            break
                        if not isinstance(resource, DiscoveredResource):
                            raise TypeError(
                                "connector search must return DiscoveredResource values"
                            )
                        if not self._resource_scheme_is_allowed(
                            resource=resource, binding=binding
                        ):
                            continue
                        key = (action.connector_id, resource.resource_locator)
                        if key in resource_keys:
                            continue
                        candidate_id = f"candidate-{len(candidates) + 1:04d}"
                        candidate = make_candidate(
                            candidate_id=candidate_id,
                            binding=binding,
                            resource=resource,
                            observed_at=self._clock(),
                        )
                        candidates[candidate_id] = _CandidateBinding(
                            candidate=candidate, resource=resource
                        )
                        resource_keys.add(key)
                        added += 1
                        if added >= candidate_limit:
                            break
                except Exception:
                    transitions.append(
                        ActionTransition.for_action(
                            step=step,
                            action=action,
                            outcome=ActionOutcome.FAILED,
                            error_code="connector_search_invalid_result",
                        )
                    )
                    feedback_codes = ("connector_search_invalid_result",)
                    step += 1
                    continue
                transitions.append(
                    ActionTransition.for_action(
                        step=step,
                        action=action,
                        outcome=ActionOutcome.EXECUTED,
                        candidate_count=added,
                    )
                )
                feedback_codes = (
                    ("candidate_window_full",)
                    if len(candidates)
                    >= self.config.research_control.max_total_candidates
                    else ()
                )
                step += 1
                continue

            if isinstance(action, FetchAction):
                candidate_binding = candidates[action.candidate_id]
                if action.candidate_id in failed_candidate_ids:
                    feedback_codes = self._reject(
                        transitions,
                        step=step,
                        action=action,
                        error_code="candidate_previous_fetch_failed",
                    )
                    step += 1
                    continue
                if not self._resource_scheme_is_allowed(
                    resource=candidate_binding.resource,
                    binding=self.registry.get(
                        candidate_binding.candidate.connector_id
                    ),
                ):
                    feedback_codes = self._reject(
                        transitions,
                        step=step,
                        action=action,
                        error_code="candidate_scheme_not_allowed",
                    )
                    step += 1
                    continue
                source_id = source_id_for_candidate(candidate_binding.candidate)
                if source_id in fetched_source_ids:
                    feedback_codes = self._reject(
                        transitions,
                        step=step,
                        action=action,
                        error_code="candidate_already_fetched",
                    )
                    step += 1
                    continue
                try:
                    self._ensure_not_cancelled()
                    binding = self.registry.get(candidate_binding.candidate.connector_id)
                    fetched = binding.connector.fetch(candidate_binding.resource)
                    if not isinstance(fetched, FetchedResource):
                        raise TypeError("connector fetch must return FetchedResource")
                    if (
                        len(fetched.content)
                        > self.config.research_control.max_fetched_characters
                    ):
                        raise _FetchedContentLimitError(
                            "fetched content exceeds the General V1 limit"
                        )
                    page = materialize_fetched_page(
                        artifact_store=self.artifact_store,
                        candidate=candidate_binding.candidate,
                        fetched=fetched,
                    )
                except GeneralResearchCancelled:
                    raise
                except _FetchedContentLimitError:
                    tool_calls += 1
                    failed_candidate_ids.add(action.candidate_id)
                    transitions.append(
                        ActionTransition.for_action(
                            step=step,
                            action=action,
                            outcome=ActionOutcome.FAILED,
                            error_code="fetched_content_exceeds_limit",
                        )
                    )
                    feedback_codes = ("fetched_content_exceeds_limit",)
                    step += 1
                    continue
                except Exception as exc:
                    tool_calls += 1
                    failed_candidate_ids.add(action.candidate_id)
                    error_code = self._fetch_failure_code(exc)
                    transitions.append(
                        ActionTransition.for_action(
                            step=step,
                            action=action,
                            outcome=ActionOutcome.FAILED,
                            error_code=error_code,
                        )
                    )
                    feedback_codes = (error_code,)
                    step += 1
                    continue
                tool_calls += 1
                fetched_source_ids.add(source_id)
                fetched_candidate_ids.add(action.candidate_id)
                pages.append(page)
                transitions.append(
                    ActionTransition.for_action(
                        step=step, action=action, outcome=ActionOutcome.EXECUTED
                    )
                )
                feedback_codes = ()
                step += 1
                continue

            raise ControllerContractError("unsupported GeneralAction")

        return ActionLoopResult(
            pages=tuple(pages),
            transitions=tuple(transitions),
            model_calls=model_calls,
            tool_calls=tool_calls,
            terminal_reason="controller_model_budget_exhausted",
        )


class ActionLoopResearcher:
    """Adapt one ActionLoop into the workflow's bounded research-stage API."""

    def __init__(self, loop: GeneralActionLoop, controller: Controller) -> None:
        if not isinstance(loop, GeneralActionLoop):
            raise TypeError("loop must be GeneralActionLoop")
        if not callable(controller):
            raise TypeError("controller must be callable")
        self.loop = loop
        self.controller = controller

    def __call__(
        self,
        plan: ResearchPlan,
        coverage: CoverageState | None,
        budget: BudgetSnapshot,
    ) -> ResearchStageResult:
        return self.loop.run(
            plan=plan, coverage=coverage, budget=budget, controller=self.controller
        ).as_research_stage()


__all__ = [
    "ActionLoopResearcher",
    "ActionLoopResult",
    "Controller",
    "ControllerContractError",
    "ControllerObservation",
    "ControllerStageResult",
    "GeneralActionLoop",
]
