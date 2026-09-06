"""Budget-aware middleware and replay helpers for the lead Planner.

The runtime used to keep the hard tool-call cap outside the model context.  A
Planner could therefore propose a valid-looking parallel batch that crossed
the remaining budget, after which the runtime terminated the whole run.  This
module makes the contract part of every model request and gives one bounded
replan opportunity before failing closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage

from .request_lifecycle import (
    get_request_lifecycle,
    normalize_failure_category,
    response_server_request_id,
)


PLANNER_BUDGET_PROTOCOL = "planner-budget/v1"
_PROMPT_MARKER = "PLANNER BUDGET CONTRACT (authoritative)"
FORCED_STOP_FLAG = "planner_budget_forced_stop"
DEFAULT_MAX_RUNTIME_RECOVERY_ACTIONS = 1
PLANNER_STATE_ARM_LEGACY_COMPACT = "legacy_compact"
PLANNER_STATE_ARM_EXPANDED = "expanded"
_PLANNER_STATE_ARMS = {
    PLANNER_STATE_ARM_LEGACY_COMPACT,
    PLANNER_STATE_ARM_EXPANDED,
}


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
    else:
        calls = getattr(message, "tool_calls", None) or []
    return [call for call in calls if isinstance(call, dict)]


def count_prior_tool_calls(messages: list[Any]) -> int:
    """Count Planner calls already committed to the graph message history."""
    return sum(len(_message_tool_calls(message)) for message in messages)


def count_prior_named_tool_calls(messages: list[Any], name: str) -> int:
    """Count committed calls for one named tool without inferring task content."""
    return sum(
        1
        for message in messages
        for call in _message_tool_calls(message)
        if str(call.get("name") or "") == name
    )


@dataclass(frozen=True)
class PlannerBudgetState:
    """The exact bounded state exposed before one Planner model call."""

    max_tool_calls: int | None
    used_tool_calls: int
    remaining_tool_calls: int | None
    max_tool_calls_per_batch: int | None
    allowed_tool_calls_next_batch: int | None
    max_model_calls: int | None
    used_model_calls: int
    remaining_model_calls: int | None
    max_fetch_calls: int | None
    used_fetch_calls: int
    remaining_fetch_calls: int | None
    max_fetch_calls_per_batch: int | None
    allowed_fetch_calls_next_batch: int | None

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": PLANNER_BUDGET_PROTOCOL,
            "max_tool_calls": self.max_tool_calls,
            "used_tool_calls": self.used_tool_calls,
            "remaining_tool_calls": self.remaining_tool_calls,
            "max_tool_calls_per_batch": self.max_tool_calls_per_batch,
            "allowed_tool_calls_next_batch": (
                self.allowed_tool_calls_next_batch
            ),
            "max_model_calls": self.max_model_calls,
            "used_model_calls": self.used_model_calls,
            "remaining_model_calls": self.remaining_model_calls,
            "max_fetch_calls": self.max_fetch_calls,
            "used_fetch_calls": self.used_fetch_calls,
            "remaining_fetch_calls": self.remaining_fetch_calls,
            "max_fetch_calls_per_batch": self.max_fetch_calls_per_batch,
            "allowed_fetch_calls_next_batch": (
                self.allowed_fetch_calls_next_batch
            ),
        }


def planner_budget_state(
    *,
    used_tool_calls: int,
    used_model_calls: int,
    max_tool_calls: int | None,
    max_tool_calls_per_batch: int | None,
    max_model_calls: int | None,
    max_fetch_calls: int | None = None,
    used_fetch_calls: int = 0,
    max_fetch_calls_per_batch: int | None = None,
) -> PlannerBudgetState:
    remaining_tools = (
        max(0, max_tool_calls - used_tool_calls)
        if max_tool_calls is not None
        else None
    )
    allowed = max_tool_calls_per_batch
    if remaining_tools is not None:
        allowed = (
            remaining_tools
            if allowed is None
            else min(allowed, remaining_tools)
        )
    remaining_models = (
        max(0, max_model_calls - used_model_calls)
        if max_model_calls is not None
        else None
    )
    remaining_fetches = (
        max(0, max_fetch_calls - used_fetch_calls)
        if max_fetch_calls is not None
        else None
    )
    allowed_fetches = max_fetch_calls_per_batch
    if remaining_fetches is not None:
        allowed_fetches = (
            remaining_fetches
            if allowed_fetches is None
            else min(allowed_fetches, remaining_fetches)
        )
    return PlannerBudgetState(
        max_tool_calls=max_tool_calls,
        used_tool_calls=max(0, used_tool_calls),
        remaining_tool_calls=remaining_tools,
        max_tool_calls_per_batch=max_tool_calls_per_batch,
        allowed_tool_calls_next_batch=allowed,
        max_model_calls=max_model_calls,
        used_model_calls=max(0, used_model_calls),
        remaining_model_calls=remaining_models,
        max_fetch_calls=max_fetch_calls,
        used_fetch_calls=max(0, used_fetch_calls),
        remaining_fetch_calls=remaining_fetches,
        max_fetch_calls_per_batch=max_fetch_calls_per_batch,
        allowed_fetch_calls_next_batch=allowed_fetches,
    )


def render_budget_prompt(
    base_prompt: str,
    state: PlannerBudgetState,
    *,
    rejection: str | None = None,
    supplemental_prompt: str | None = None,
    include_fetch_state: bool = True,
) -> str:
    """Append a deterministic budget block to the base system prompt."""
    clean_base = base_prompt.split(f"\n\n{_PROMPT_MARKER}", 1)[0].rstrip()
    tool_limit = (
        "unbounded"
        if state.max_tool_calls is None
        else str(state.max_tool_calls)
    )
    remaining_tools = (
        "unbounded"
        if state.remaining_tool_calls is None
        else str(state.remaining_tool_calls)
    )
    allowed = (
        "unbounded"
        if state.allowed_tool_calls_next_batch is None
        else str(state.allowed_tool_calls_next_batch)
    )
    remaining_models = (
        "unbounded"
        if state.remaining_model_calls is None
        else str(state.remaining_model_calls)
    )
    lines = [
        clean_base,
        "",
        _PROMPT_MARKER,
        f"- tool calls used: {state.used_tool_calls}/{tool_limit}",
        f"- tool calls remaining: {remaining_tools}",
        f"- maximum tool calls allowed in your next batch: {allowed}",
        f"- model calls remaining including this decision: {remaining_models}",
        "Never emit more tool calls than the next-batch allowance.",
        "Prefer the smallest high-value batch; do not spend the remaining "
        "budget merely because it exists.",
    ]
    if include_fetch_state:
        fetch_limit = (
            "unbounded"
            if state.max_fetch_calls is None
            else str(state.max_fetch_calls)
        )
        remaining_fetches = (
            "unbounded"
            if state.remaining_fetch_calls is None
            else str(state.remaining_fetch_calls)
        )
        lines[7:7] = [
            f"- fetch calls used: {state.used_fetch_calls}/{fetch_limit}",
            f"- fetch calls remaining: {remaining_fetches}",
        ]
        if state.allowed_fetch_calls_next_batch is not None:
            lines[9:9] = [
                "- maximum fetch calls allowed in your next Planner turn: "
                f"{state.allowed_fetch_calls_next_batch}",
            ]
    if state.allowed_tool_calls_next_batch == 0:
        lines.append(
            "No tool call is allowed now. Finish the research response from "
            "the evidence already collected."
        )
    else:
        lines.append(
            "When the evidence is sufficient, stop and answer instead of "
            "issuing another tool batch."
        )
    if include_fetch_state and state.remaining_fetch_calls == 0:
        lines.append(
            "No fetch call is allowed now. Use only evidence already fetched or "
            "STOP when the evidence contract permits it."
        )
    if rejection:
        lines.extend(
            [
                "",
                "Your previous proposal was rejected by the runtime:",
                rejection,
                "Replan once now within the stated allowance, or stop and answer.",
            ]
        )
    if supplemental_prompt:
        lines.extend(["", supplemental_prompt.strip()])
    return "\n".join(lines)


def inject_budget_state_into_messages(
    messages: list[dict[str, Any]], state_metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rebuild the dynamic online prompt in an offline replay context."""
    copied = [dict(message) for message in messages]
    try:
        state = PlannerBudgetState(
            max_tool_calls=state_metadata.get("max_tool_calls"),
            used_tool_calls=int(state_metadata["used_tool_calls"]),
            remaining_tool_calls=state_metadata.get("remaining_tool_calls"),
            max_tool_calls_per_batch=state_metadata.get(
                "max_tool_calls_per_batch"
            ),
            allowed_tool_calls_next_batch=state_metadata.get(
                "allowed_tool_calls_next_batch"
            ),
            max_model_calls=state_metadata.get("max_model_calls"),
            used_model_calls=int(state_metadata["used_model_calls"]),
            remaining_model_calls=state_metadata.get("remaining_model_calls"),
            max_fetch_calls=state_metadata.get("max_fetch_calls"),
            used_fetch_calls=int(state_metadata.get("used_fetch_calls", 0)),
            remaining_fetch_calls=state_metadata.get("remaining_fetch_calls"),
            max_fetch_calls_per_batch=state_metadata.get(
                "max_fetch_calls_per_batch"
            ),
            allowed_fetch_calls_next_batch=state_metadata.get(
                "allowed_fetch_calls_next_batch"
            ),
        )
    except (KeyError, TypeError, ValueError):
        return copied
    for index, message in enumerate(copied):
        if message.get("role") == "system":
            content = str(message.get("content") or "")
            copied[index] = {
                **message,
                "content": render_budget_prompt(
                    content,
                    state,
                    rejection=state_metadata.get("replan_rejection"),
                ),
            }
            return copied
    copied.insert(
        0,
        {
            "role": "system",
            "content": render_budget_prompt(
                "",
                state,
                rejection=state_metadata.get("replan_rejection"),
            ).lstrip(),
        },
    )
    return copied


class PlannerBudgetMiddleware(AgentMiddleware):
    """Expose remaining budget and retry one over-budget Planner proposal."""

    tools = []

    def __init__(
        self,
        *,
        max_tool_calls: int | None,
        max_tool_calls_per_batch: int | None,
        max_model_calls: int | None,
        max_replans: int = 1,
        max_fetch_calls: int | None = None,
        max_fetch_calls_per_batch: int | None = None,
        max_stop_replans: int = 1,
        max_runtime_recovery_actions: int = DEFAULT_MAX_RUNTIME_RECOVERY_ACTIONS,
        supplemental_state_provider: Callable[[], str | None] | None = None,
        stop_guard_provider: Callable[[], Mapping[str, Any] | None] | None = None,
        recovery_action_provider: (
            Callable[[list[str]], Mapping[str, Any] | None] | None
        ) = None,
        allowed_fetch_urls: set[str] | None = None,
        planner_state_arm: str = PLANNER_STATE_ARM_EXPANDED,
    ) -> None:
        self.max_tool_calls = max_tool_calls
        self.max_tool_calls_per_batch = max_tool_calls_per_batch
        self.max_model_calls = max_model_calls
        self.max_replans = max(0, max_replans)
        self.max_fetch_calls = (
            max_fetch_calls if max_fetch_calls is not None and max_fetch_calls > 0 else None
        )
        self.max_fetch_calls_per_batch = (
            max_fetch_calls_per_batch
            if max_fetch_calls_per_batch is not None
            and max_fetch_calls_per_batch > 0
            else None
        )
        self.max_stop_replans = max(0, max_stop_replans)
        self.max_runtime_recovery_actions = max(0, max_runtime_recovery_actions)
        self.actual_model_calls = 0
        self.replan_attempts = 0
        self.rejected_batches = 0
        self.forced_stops = 0
        self.last_rejection: str | None = None
        self.forced_stop_reason: str | None = None
        self.stop_replan_attempts = 0
        self.legal_stop_accepted = 0
        self.illegal_stop_rejected = 0
        self.illegal_stop_recovered = 0
        self.policy_failure_illegal_stop = 0
        self.runtime_recovery_actions = 0
        self.supplemental_state_provider = supplemental_state_provider
        self.stop_guard_provider = stop_guard_provider
        self.recovery_action_provider = recovery_action_provider
        self.allowed_fetch_urls = (
            frozenset(
                str(url).strip()
                for url in allowed_fetch_urls
                if str(url).strip()
            )
            if allowed_fetch_urls is not None
            else None
        )
        self.rejected_unobserved_url_proposals = 0
        self.planner_state_arm = (
            planner_state_arm
            if planner_state_arm in _PLANNER_STATE_ARMS
            else PLANNER_STATE_ARM_EXPANDED
        )
        self.proposed_tool_batches: list[dict[str, Any]] = []

    def _state(self, messages: list[Any]) -> PlannerBudgetState:
        return planner_budget_state(
            used_tool_calls=count_prior_tool_calls(messages),
            used_model_calls=self.actual_model_calls,
            max_tool_calls=self.max_tool_calls,
            max_tool_calls_per_batch=self.max_tool_calls_per_batch,
            max_model_calls=self.max_model_calls,
            max_fetch_calls=self.max_fetch_calls,
            used_fetch_calls=count_prior_named_tool_calls(messages, "fetch_content"),
            max_fetch_calls_per_batch=self.max_fetch_calls_per_batch,
        )

    @staticmethod
    def _first_ai_message(response: ModelResponse[Any]) -> AIMessage | None:
        return next(
            (
                message
                for message in response.result
                if isinstance(message, AIMessage)
            ),
            None,
        )

    def _annotate(
        self,
        response: ModelResponse[Any],
        state: PlannerBudgetState,
        *,
        rejection: str | None = None,
    ) -> ModelResponse[Any]:
        result = []
        annotated = False
        for message in response.result:
            if isinstance(message, AIMessage) and not annotated:
                additional = dict(message.additional_kwargs or {})
                budget_metadata = state.metadata()
                if rejection:
                    budget_metadata["replan_rejection"] = rejection
                additional["planner_budget"] = budget_metadata
                additional["planner_stop_guard"] = self._stop_guard_summary()
                message = message.model_copy(
                    update={"additional_kwargs": additional}
                )
                annotated = True
            result.append(message)
        return ModelResponse(
            result=result,
            structured_response=response.structured_response,
        )

    def _forced_stop(
        self, state: PlannerBudgetState, reason: str
    ) -> ModelResponse[Any]:
        self.forced_stops += 1
        self.forced_stop_reason = reason
        budget_metadata = state.metadata()
        if self.last_rejection:
            budget_metadata["last_rejection"] = self.last_rejection
        return ModelResponse(
            result=[
                AIMessage(
                    content="Research complete.",
                    additional_kwargs={
                        FORCED_STOP_FLAG: True,
                        "planner_budget_stop_reason": reason,
                        "planner_budget": budget_metadata,
                        "planner_stop_guard": self._stop_guard_summary(),
                    },
                )
            ]
        )

    def _stop_guard_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.stop_guard_provider is not None,
            "max_stop_replans": self.max_stop_replans,
            "stop_replan_attempts": self.stop_replan_attempts,
            "legal_stop_accepted": self.legal_stop_accepted,
            "illegal_stop_rejected": self.illegal_stop_rejected,
            "illegal_stop_recovered": self.illegal_stop_recovered,
            "policy_failure_illegal_stop": self.policy_failure_illegal_stop,
            "max_runtime_recovery_actions": self.max_runtime_recovery_actions,
            "runtime_recovery_actions": self.runtime_recovery_actions,
        }

    @staticmethod
    def _tool_names(tools: list[Any]) -> list[str]:
        names: list[str] = []
        for tool in tools:
            if isinstance(tool, Mapping):
                name = str(tool.get("name") or "")
            else:
                name = str(getattr(tool, "name", "") or "")
            if name:
                names.append(name)
        return names

    def _runtime_recovery_action(
        self,
        request: ModelRequest[Any],
        state: PlannerBudgetState,
    ) -> ModelResponse[Any] | None:
        if (
            self.recovery_action_provider is None
            or self.runtime_recovery_actions >= self.max_runtime_recovery_actions
        ):
            return None
        action = dict(self.recovery_action_provider(self._tool_names(request.tools)) or {})
        name = str(action.get("tool_name") or "")
        args = action.get("tool_input")
        if not name or not isinstance(args, Mapping):
            return None
        self.runtime_recovery_actions += 1
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"runtime-recovery-{self.runtime_recovery_actions}",
                    "name": name,
                    "args": dict(args),
                }
            ],
            additional_kwargs={
                "planner_runtime_recovery": {
                    "family": str(action.get("family") or ""),
                    "reason": "repeated_illegal_stop",
                }
            },
        )
        return self._annotate(ModelResponse(result=[message]), state)

    @staticmethod
    def _incomplete_stop_reason(
        state: PlannerBudgetState, review: Mapping[str, Any]
    ) -> str | None:
        """Allow an auditable incomplete stop only when no legal work remains."""
        if bool(review.get("incomplete_stop_legal")):
            return "stop_incomplete_recovery_exhausted"
        if state.allowed_tool_calls_next_batch == 0:
            return "stop_incomplete_tool_budget"
        return None

    def _call_model(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
        state: PlannerBudgetState,
        *,
        rejection: str | None = None,
    ) -> ModelResponse[Any] | None:
        if (
            self.max_model_calls is not None
            and self.actual_model_calls >= self.max_model_calls
        ):
            return None
        base_prompt = (
            request.system_message.text if request.system_message else ""
        )
        supplemental_prompt = (
            self.supplemental_state_provider()
            if self.supplemental_state_provider is not None
            else None
        )
        prompt = render_budget_prompt(
            base_prompt,
            state,
            rejection=rejection,
            supplemental_prompt=supplemental_prompt,
            include_fetch_state=(
                self.planner_state_arm != PLANNER_STATE_ARM_LEGACY_COMPACT
            ),
        )
        available_tools = (
            [] if state.allowed_tool_calls_next_batch == 0 else request.tools
        )
        bounded_request = request.override(
            system_message=SystemMessage(content=prompt),
            tools=available_tools,
        )
        self.actual_model_calls += 1
        lifecycle = get_request_lifecycle()
        planner_call = self.actual_model_calls
        if lifecycle is not None:
            lifecycle.record(
                "planner_request_enqueued",
                planner_call=planner_call,
                remaining_model_calls=state.remaining_model_calls,
                remaining_tool_calls=state.remaining_tool_calls,
            )
            lifecycle.record(
                "planner_request_sent",
                planner_call=planner_call,
            )
        try:
            response = handler(bounded_request)
        except BaseException as exc:
            if lifecycle is not None:
                lifecycle.record(
                    "planner_request_error",
                    planner_call=planner_call,
                    failure_category=normalize_failure_category(
                        exc, stage="planner_request_sent"
                    ),
                    raw_error=str(exc),
                )
            raise
        if lifecycle is not None:
            server_request_id = response_server_request_id(response)
            # The runtime invokes non-streaming models, so true first-token
            # timing is unavailable.  Emit an explicit unavailable marker
            # rather than fabricating a token timestamp.
            lifecycle.record(
                "planner_first_token",
                planner_call=planner_call,
                available=False,
                server_request_id=server_request_id,
            )
            lifecycle.record(
                "planner_response_end",
                planner_call=planner_call,
                server_request_id=server_request_id,
            )
        return response

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        state = self._state(request.messages)
        response = self._call_model(request, handler, state)
        if response is None:
            return self._forced_stop(state, "model_call_budget")
        batch_replans = 0
        rejection: str | None = None

        while True:
            message = self._first_ai_message(response)
            tool_calls = _message_tool_calls(message)
            call_count = len(tool_calls)
            allowed = state.allowed_tool_calls_next_batch
            proposal = {
                "planner_call": self.actual_model_calls,
                "requested_tool_calls": call_count,
                "allowed_tool_calls_next_batch": allowed,
                "tool_calls": [
                    {
                        "name": str(call.get("name") or ""),
                        "args": dict(call.get("args") or {}),
                    }
                    for call in tool_calls
                ],
            }
            self.proposed_tool_batches.append(proposal)
            requested_fetches = sum(
                str(call.get("name") or "") == "fetch_content"
                for call in tool_calls
            )
            unobserved_fetch_urls = sorted(
                {
                    str((call.get("args") or {}).get("url") or "").strip()
                    for call in tool_calls
                    if str(call.get("name") or "") == "fetch_content"
                    and str((call.get("args") or {}).get("url") or "").strip()
                    and self.allowed_fetch_urls is not None
                    and str((call.get("args") or {}).get("url") or "").strip()
                    not in self.allowed_fetch_urls
                }
            )
            if unobserved_fetch_urls:
                proposal["unobserved_fetch_urls"] = unobserved_fetch_urls
                self.rejected_unobserved_url_proposals += len(
                    unobserved_fetch_urls
                )
            fetch_over_budget = (
                state.remaining_fetch_calls is not None
                and requested_fetches > state.remaining_fetch_calls
            )
            fetch_over_batch = (
                state.allowed_fetch_calls_next_batch is not None
                and requested_fetches > state.allowed_fetch_calls_next_batch
            )
            over_tool_budget = allowed is not None and call_count > allowed

            if (
                over_tool_budget
                or fetch_over_budget
                or fetch_over_batch
                or unobserved_fetch_urls
            ):
                proposal["disposition"] = "rejected"
                self.rejected_batches += 1
                if over_tool_budget:
                    rejection = (
                        f"requested {call_count} tool calls, but only {allowed} are "
                        "allowed in this batch"
                    )
                elif fetch_over_budget:
                    rejection = (
                        f"requested {requested_fetches} fetch calls, but only "
                        f"{state.remaining_fetch_calls} fetch calls remain"
                    )
                elif fetch_over_batch:
                    rejection = (
                        f"requested {requested_fetches} fetch calls, but only "
                        f"{state.allowed_fetch_calls_next_batch} fetch call(s) are "
                        "allowed in this Planner turn"
                    )
                else:
                    rejection = (
                        "requested unobserved fetch URL(s) outside the shared "
                        "frozen candidate pool: "
                        + ", ".join(unobserved_fetch_urls)
                    )
                self.last_rejection = rejection
                if batch_replans >= self.max_replans:
                    return self._forced_stop(
                        self._state(request.messages), "batch_contract_violation"
                    )
                batch_replans += 1
                self.replan_attempts += 1
                state = self._state(request.messages)
                response = self._call_model(
                    request, handler, state, rejection=rejection
                )
                if response is None:
                    return self._forced_stop(state, "model_call_budget")
                continue

            # A final answer is a STOP proposal.  When this optional guard is
            # active, source/evidence requirements are an executable contract,
            # not merely a sentence in the prompt.
            if not tool_calls and message is not None and self.stop_guard_provider:
                review = dict(self.stop_guard_provider() or {})
                if bool(review.get("legal")):
                    proposal["disposition"] = "accepted_stop"
                    self.legal_stop_accepted += 1
                    if self.illegal_stop_rejected:
                        self.illegal_stop_recovered += 1
                    return self._annotate(response, state, rejection=rejection)

                incomplete_stop_reason = self._incomplete_stop_reason(state, review)
                if incomplete_stop_reason:
                    proposal["disposition"] = "stop_incomplete"
                    guidance = review.get("action_guidance")
                    instruction = (
                        str(guidance.get("instruction") or "")
                        if isinstance(guidance, Mapping)
                        else ""
                    )
                    self.last_rejection = (
                        "STOP_SUCCESS is not legal because evidence remains "
                        "incomplete; runtime emitted STOP_INCOMPLETE."
                        + (f" {instruction}" if instruction else "")
                    )
                    return self._forced_stop(state, incomplete_stop_reason)

                self.illegal_stop_rejected += 1
                proposal["disposition"] = "illegal_stop_rejected"
                blockers = [str(item) for item in review.get("blockers") or []]
                guidance = review.get("action_guidance")
                rejection = (
                    "STOP rejected: required evidence is still missing: "
                    + ("; ".join(blockers) if blockers else "unknown blocker")
                )
                if (
                    self.planner_state_arm != PLANNER_STATE_ARM_LEGACY_COMPACT
                    and isinstance(guidance, Mapping)
                ):
                    preferred = str(guidance.get("preferred_action_family") or "")
                    instruction = str(guidance.get("instruction") or "")
                    if preferred:
                        rejection += f". Next legal action family: {preferred}."
                    if instruction:
                        rejection += f" {instruction}"
                self.last_rejection = rejection
                if self.stop_replan_attempts >= self.max_stop_replans:
                    recovered = self._runtime_recovery_action(request, state)
                    if recovered is not None:
                        proposal["disposition"] = "runtime_recovery"
                        return recovered
                    self.policy_failure_illegal_stop += 1
                    return self._forced_stop(
                        state, "policy_failure_illegal_stop"
                    )
                self.stop_replan_attempts += 1
                state = self._state(request.messages)
                response = self._call_model(
                    request, handler, state, rejection=rejection
                )
                if response is None:
                    return self._forced_stop(state, "model_call_budget")
                continue

            proposal["disposition"] = "accepted"
            return self._annotate(response, state, rejection=rejection)

    def trace_summary(self) -> dict[str, Any]:
        return {
            "protocol": PLANNER_BUDGET_PROTOCOL,
            "max_tool_calls": self.max_tool_calls,
            "max_tool_calls_per_batch": self.max_tool_calls_per_batch,
            "max_model_calls": self.max_model_calls,
            "max_replans": self.max_replans,
            "max_fetch_calls": self.max_fetch_calls,
            "max_fetch_calls_per_batch": self.max_fetch_calls_per_batch,
            "max_stop_replans": self.max_stop_replans,
            "actual_model_calls": self.actual_model_calls,
            "replan_attempts": self.replan_attempts,
            "rejected_batches": self.rejected_batches,
            "forced_stops": self.forced_stops,
            "last_rejection": self.last_rejection,
            "forced_stop_reason": self.forced_stop_reason,
            "runtime_recovery_actions": self.runtime_recovery_actions,
            "rejected_unobserved_url_proposals": (
                self.rejected_unobserved_url_proposals
            ),
            "planner_state_arm": self.planner_state_arm,
            "proposed_tool_batches": list(self.proposed_tool_batches),
            **self._stop_guard_summary(),
        }
