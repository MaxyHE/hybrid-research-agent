"""Versioned replay scenarios and Planner state-to-action exports."""

from __future__ import annotations

from enum import StrEnum
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .replay import (
    ObservationCache,
    ObservationCacheStore,
    canonical_tool_signature,
)
from .planner_budget import (
    PLANNER_BUDGET_PROTOCOL,
    inject_budget_state_into_messages,
)
from .schema import EventKind, RunOutcome, TraceEvent
from .sft import infer_tool_schemas
from .store import JsonlTraceStore


SCENARIO_SCHEMA_VERSION = "agent-replay-scenario/v1"
DECISION_POINT_SCHEMA_VERSION = "planner-decision-point/v1"


class ScenarioError(ValueError):
    """Raised when a trace cannot form a faithful replay scenario."""


class PlannerActionKind(StrEnum):
    TOOL_CALL = "tool_call"
    STOP = "stop"


class PlannerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PlannerActionKind
    message: dict[str, Any]
    tool_signatures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape(self) -> "PlannerAction":
        calls = self.message.get("tool_calls") or []
        if self.kind == PlannerActionKind.TOOL_CALL and not calls:
            raise ValueError("tool_call action requires tool_calls")
        if self.kind == PlannerActionKind.STOP and calls:
            raise ValueError("stop action cannot contain tool_calls")
        if self.kind == PlannerActionKind.STOP and not str(
            self.message.get("content") or ""
        ).strip():
            raise ValueError("stop action requires runtime assistant content")
        return self


class PlannerDecisionPoint(BaseModel):
    """One frozen Planner state and the next observed runtime action.

    ``split_group`` is always the source run id. Dataset splits must operate on
    this field, never on individual decision rows, or turns from one task leak
    across train and evaluation.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["planner-decision-point/v1"] = (
        DECISION_POINT_SCHEMA_VERSION
    )
    decision_id: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    split_group: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    source_event_ids: list[int] = Field(min_length=1)
    context_messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    target: PlannerAction
    prior_tool_signatures: list[str] = Field(default_factory=list)
    duplicate_with_history: bool = False
    cache_covered: bool = False
    sft_eligible: bool = False
    sft_exclusion_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_grouping_and_target(self) -> "PlannerDecisionPoint":
        if self.split_group != self.source_run_id:
            raise ValueError("split_group must equal source_run_id")
        if not self.context_messages:
            raise ValueError("decision point requires context messages")
        if self.target.kind == PlannerActionKind.STOP and self.sft_eligible:
            raise ValueError(
                "stop targets are evaluation-only until an action-only loss "
                "protocol is implemented"
            )
        if self.sft_eligible and self.sft_exclusion_reasons:
            raise ValueError(
                "SFT-eligible decision cannot have exclusion reasons"
            )
        return self


class ReplayScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent-replay-scenario/v1"] = (
        SCENARIO_SCHEMA_VERSION
    )
    scenario_id: str = Field(min_length=1)
    split_group: str = Field(min_length=1)
    query: str = Field(min_length=1)
    outcome: RunOutcome
    tools: list[dict[str, Any]]
    observation_cache: ObservationCache
    decision_points: list[PlannerDecisionPoint]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scenario_group(self) -> "ReplayScenario":
        if self.split_group != self.scenario_id:
            raise ValueError("scenario split_group must equal scenario_id")
        if any(
            point.source_run_id != self.scenario_id
            for point in self.decision_points
        ):
            raise ValueError("scenario contains a decision from another run")
        return self


def _tool_call_message(events: list[TraceEvent]) -> tuple[dict[str, Any], list[str]]:
    calls = []
    signatures = []
    for event in events:
        tool_input = event.tool_input or {}
        calls.append(
            {
                "id": event.tool_call_id,
                "type": "function",
                "function": {
                    "name": event.tool_name,
                    "arguments": tool_input,
                },
            }
        )
        signatures.append(
            canonical_tool_signature(event.tool_name or "", tool_input)
        )
    return {"role": "assistant", "content": "", "tool_calls": calls}, signatures


def trace_to_replay_scenario(
    events: Iterable[TraceEvent],
) -> ReplayScenario:
    """Convert one valid trace into a task-grouped offline scenario.

    Successful tool-call decisions are eligible for the v5 action dataset.
    STOP decisions retain the runtime final content for evaluation, but are not
    SFT eligible: training that prose with assistant-only loss would reintroduce
    the writer objective that this dataset is intended to isolate.
    """
    validated = JsonlTraceStore.validate(events)
    run_id = validated[0].run_id
    outcome = validated[-1].outcome or RunOutcome.ERROR
    query = next(
        (
            str(event.content)
            for event in validated
            if event.kind == EventKind.MESSAGE and event.role == "user"
        ),
        "",
    ).strip()
    if not query:
        raise ScenarioError("trace is missing a text user query")

    tools = infer_tool_schemas(
        validated, validated[0].metadata.get("fetch_mode")
    )
    cache = ObservationCacheStore.from_traces([validated])
    cache_signatures = {entry.signature for entry in cache.entries}
    context: list[dict[str, Any]] = []
    prior_signatures: list[str] = []
    decisions: list[PlannerDecisionPoint] = []
    pending_assistant_content = ""
    pending_assistant_event_id: int | None = None
    index = 1
    step_index = 0

    while index < len(validated) - 1:
        event = validated[index]
        next_kind = (
            validated[index + 1].kind
            if index + 1 < len(validated)
            else None
        )
        if event.kind == EventKind.MESSAGE:
            if event.role == "assistant" and next_kind == EventKind.TOOL_CALL:
                pending_assistant_content = str(event.content or "")
                pending_assistant_event_id = event.event_id
                index += 1
                continue
            if event.role == "assistant":
                target_message = {
                    "role": "assistant",
                    "content": event.content,
                }
                budget_state = event.metadata.get("planner_budget")
                decision_context = (
                    inject_budget_state_into_messages(context, budget_state)
                    if isinstance(budget_state, dict)
                    and budget_state.get("protocol")
                    == PLANNER_BUDGET_PROTOCOL
                    else list(context)
                )
                decision_metadata = {"trace_outcome": outcome.value}
                if isinstance(budget_state, dict):
                    decision_metadata.update(
                        {
                            "planner_budget_state": budget_state,
                            "planner_budget_state_visible": (
                                budget_state.get("protocol")
                                == PLANNER_BUDGET_PROTOCOL
                            ),
                        }
                    )
                decisions.append(
                    PlannerDecisionPoint(
                        decision_id=f"{run_id}:decision:{step_index}",
                        source_run_id=run_id,
                        split_group=run_id,
                        step_index=step_index,
                        source_event_ids=[event.event_id],
                        context_messages=decision_context,
                        tools=tools,
                        target=PlannerAction(
                            kind=PlannerActionKind.STOP,
                            message=target_message,
                        ),
                        prior_tool_signatures=list(prior_signatures),
                        cache_covered=True,
                        sft_eligible=False,
                        sft_exclusion_reasons=[
                            "stop_requires_action_only_loss"
                        ],
                        metadata=decision_metadata,
                    )
                )
                step_index += 1
                context.append(target_message)
                index += 1
                continue
            context.append({"role": event.role, "content": event.content})
            index += 1
            continue

        if event.kind == EventKind.TOOL_CALL:
            call_events = []
            while (
                index < len(validated) - 1
                and validated[index].kind == EventKind.TOOL_CALL
            ):
                call_events.append(validated[index])
                index += 1
            target_message, signatures = _tool_call_message(call_events)
            target_message["content"] = pending_assistant_content
            pending_assistant_content = ""
            source_event_ids = (
                [pending_assistant_event_id]
                if pending_assistant_event_id is not None
                else []
            ) + [item.event_id for item in call_events]
            pending_assistant_event_id = None
            duplicate_with_history = any(
                signature in prior_signatures for signature in signatures
            )
            cache_covered = all(
                signature in cache_signatures for signature in signatures
            )
            exclusion_reasons = []
            if outcome != RunOutcome.SUCCESS:
                exclusion_reasons.append("trace_outcome_not_success")
            if not cache_covered:
                exclusion_reasons.append("observation_not_cached")
            if duplicate_with_history:
                exclusion_reasons.append("duplicate_tool_call")
            budget_state = next(
                (
                    item.metadata.get("planner_budget")
                    for item in call_events
                    if isinstance(item.metadata.get("planner_budget"), dict)
                ),
                None,
            )
            decision_context = (
                inject_budget_state_into_messages(context, budget_state)
                if isinstance(budget_state, dict)
                and budget_state.get("protocol") == PLANNER_BUDGET_PROTOCOL
                else list(context)
            )
            decision_metadata = {"trace_outcome": outcome.value}
            if isinstance(budget_state, dict):
                decision_metadata.update(
                    {
                        "planner_budget_state": budget_state,
                        "planner_budget_state_visible": (
                            budget_state.get("protocol")
                            == PLANNER_BUDGET_PROTOCOL
                        ),
                    }
                )
            decisions.append(
                PlannerDecisionPoint(
                    decision_id=f"{run_id}:decision:{step_index}",
                    source_run_id=run_id,
                    split_group=run_id,
                    step_index=step_index,
                    source_event_ids=source_event_ids,
                    context_messages=decision_context,
                    tools=tools,
                    target=PlannerAction(
                        kind=PlannerActionKind.TOOL_CALL,
                        message=target_message,
                        tool_signatures=signatures,
                    ),
                    prior_tool_signatures=list(prior_signatures),
                    duplicate_with_history=duplicate_with_history,
                    cache_covered=cache_covered,
                    sft_eligible=not exclusion_reasons,
                    sft_exclusion_reasons=exclusion_reasons,
                    metadata=decision_metadata,
                )
            )
            step_index += 1
            context.append(target_message)
            prior_signatures.extend(signatures)
            continue

        if event.kind == EventKind.TOOL_OBSERVATION:
            context.append(
                {
                    "role": "tool",
                    "tool_call_id": event.tool_call_id,
                    "name": event.tool_name,
                    "content": event.content,
                }
            )
        index += 1

    return ReplayScenario(
        scenario_id=run_id,
        split_group=run_id,
        query=query,
        outcome=outcome,
        tools=tools,
        observation_cache=cache,
        decision_points=decisions,
        metadata={
            "source_trace_schema": validated[0].schema_version,
            "runtime": validated[0].metadata.get("runtime"),
            "model": validated[0].metadata.get("model"),
            "strategy": validated[0].metadata.get("strategy"),
            "public_fetch_fallback": validated[0].metadata.get(
                "public_fetch_fallback"
            ),
            "planner_budget_protocol": validated[0].metadata.get(
                "planner_budget_protocol"
            ),
            "max_model_calls": validated[0].metadata.get("max_model_calls"),
            "max_tool_calls": validated[0].metadata.get("max_tool_calls"),
            "max_tool_calls_per_batch": validated[0].metadata.get(
                "max_tool_calls_per_batch"
            ),
        },
    )


class ReplayScenarioStore:
    @staticmethod
    def write(path: str | Path, scenario: ReplayScenario) -> Path:
        return _atomic_json_write(path, scenario.model_dump(mode="json"))

    @staticmethod
    def read(path: str | Path) -> ReplayScenario:
        return ReplayScenario.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )


def write_decision_points_jsonl(
    path: str | Path,
    scenarios: Iterable[ReplayScenario],
) -> Path:
    """Write decisions while rejecting duplicate task groups."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    seen_groups: set[str] = set()
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for scenario in scenarios:
            if scenario.split_group in seen_groups:
                raise ScenarioError(
                    f"duplicate split group: {scenario.split_group}"
                )
            seen_groups.add(scenario.split_group)
            for point in scenario.decision_points:
                handle.write(
                    json.dumps(
                        point.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def _atomic_json_write(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target
