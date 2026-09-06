"""Action-only evaluation for frozen Planner decision points."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from .replay import ObservationCache, canonical_tool_signature
from .scenario import (
    PlannerActionKind,
    PlannerDecisionPoint,
    ReplayScenario,
)


class PlannerPredictionError(ValueError):
    """Raised when a model message cannot represent a Planner action."""


def tool_family(tool_name: str) -> str:
    if tool_name == "fetch_content":
        return "fetch"
    if tool_name == "research_subtopic":
        return "subtopic"
    if tool_name.startswith("search_collection_") or tool_name in {
        "search_library",
        "search_local",
    }:
        return "collection"
    if tool_name == "web_search" or tool_name.startswith("search_"):
        return "web"
    return "other"


class ParsedPlannerPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PlannerActionKind
    message: dict[str, Any]
    tool_names: list[str] = Field(default_factory=list)
    tool_families: list[str] = Field(default_factory=list)
    tool_signatures: list[str] = Field(default_factory=list)


def parse_planner_prediction(message: dict[str, Any]) -> ParsedPlannerPrediction:
    """Parse one OpenAI-compatible assistant message into an action."""
    if not isinstance(message, dict):
        raise PlannerPredictionError("prediction message must be an object")
    if message.get("role") not in (None, "assistant"):
        raise PlannerPredictionError("prediction role must be assistant")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise PlannerPredictionError("tool_calls must be an array")
    if not raw_calls:
        if not str(message.get("content") or "").strip():
            raise PlannerPredictionError(
                "prediction must contain tool_calls or final content"
            )
        return ParsedPlannerPrediction(
            kind=PlannerActionKind.STOP,
            message=message,
        )

    names: list[str] = []
    families: list[str] = []
    signatures: list[str] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            raise PlannerPredictionError("each tool call must be an object")
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise PlannerPredictionError("tool call is missing function")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise PlannerPredictionError("tool call function name is missing")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise PlannerPredictionError(
                    "tool call arguments are not valid JSON"
                ) from exc
        if not isinstance(arguments, dict):
            raise PlannerPredictionError(
                "tool call arguments must decode to an object"
            )
        names.append(name)
        families.append(tool_family(name))
        signatures.append(canonical_tool_signature(name, arguments))
    return ParsedPlannerPrediction(
        kind=PlannerActionKind.TOOL_CALL,
        message=message,
        tool_names=names,
        tool_families=families,
        tool_signatures=signatures,
    )


class PlannerDecisionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    split_group: str
    target_kind: PlannerActionKind
    predicted_kind: PlannerActionKind | None = None
    schema_valid: bool
    action_kind_match: bool = False
    exact_action_match: bool = False
    tool_name_match: bool = False
    tool_name_set_match: bool = False
    tool_family_match: bool = False
    tool_family_set_match: bool = False
    tool_call_count_match: bool = False
    stop_correct: bool = False
    premature_stop: bool = False
    target_tool_calls: int = 0
    predicted_tool_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    duplicate_tool_calls: int = 0
    budget_state_visible: bool = False
    budget_compliant: bool | None = None
    tool_calls_over_budget: int = 0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlannerPredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "planner-prediction/v1"
    decision_id: str = Field(min_length=1)
    split_group: str = Field(min_length=1)
    model: str = Field(min_length=1)
    message: dict[str, Any] | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _target_names(point: PlannerDecisionPoint) -> list[str]:
    return [
        str(call.get("function", {}).get("name") or "")
        for call in point.target.message.get("tool_calls", [])
    ]


def _next_batch_allowance(point: PlannerDecisionPoint) -> int | None:
    if point.metadata.get("planner_budget_state_visible") is not True:
        return None
    state = point.metadata.get("planner_budget_state")
    if not isinstance(state, dict):
        return None
    value = state.get("allowed_tool_calls_next_batch")
    return value if isinstance(value, int) and value >= 0 else None


def evaluate_planner_prediction(
    point: PlannerDecisionPoint,
    message: dict[str, Any],
    cache: ObservationCache,
) -> PlannerDecisionEvaluation:
    target_tool_calls = len(point.target.message.get("tool_calls", []))
    batch_allowance = _next_batch_allowance(point)
    budget_visible = batch_allowance is not None
    try:
        prediction = parse_planner_prediction(message)
    except PlannerPredictionError as exc:
        return PlannerDecisionEvaluation(
            decision_id=point.decision_id,
            split_group=point.split_group,
            target_kind=point.target.kind,
            schema_valid=False,
            target_tool_calls=target_tool_calls,
            budget_state_visible=budget_visible,
            budget_compliant=False if budget_visible else None,
            error=str(exc),
        )

    kind_match = prediction.kind == point.target.kind
    if prediction.kind == PlannerActionKind.STOP:
        premature_stop = point.target.kind == PlannerActionKind.TOOL_CALL
        return PlannerDecisionEvaluation(
            decision_id=point.decision_id,
            split_group=point.split_group,
            target_kind=point.target.kind,
            predicted_kind=prediction.kind,
            schema_valid=True,
            action_kind_match=kind_match,
            exact_action_match=kind_match,
            tool_name_match=kind_match,
            tool_name_set_match=kind_match,
            tool_family_match=kind_match,
            tool_family_set_match=kind_match,
            tool_call_count_match=kind_match,
            stop_correct=kind_match,
            premature_stop=premature_stop,
            target_tool_calls=target_tool_calls,
            budget_state_visible=budget_visible,
            budget_compliant=True if budget_visible else None,
        )

    cache_signatures = {entry.signature for entry in cache.entries}
    hits = sum(
        signature in cache_signatures
        for signature in prediction.tool_signatures
    )
    misses = len(prediction.tool_signatures) - hits
    seen_signatures = set(point.prior_tool_signatures)
    duplicates = 0
    for signature in prediction.tool_signatures:
        if signature in seen_signatures:
            duplicates += 1
        seen_signatures.add(signature)
    target_names = _target_names(point)
    target_families = [tool_family(name) for name in target_names]
    exact = kind_match and Counter(prediction.tool_signatures) == Counter(
        point.target.tool_signatures
    )
    predicted_count = len(prediction.tool_signatures)
    over_budget = (
        max(0, predicted_count - batch_allowance)
        if batch_allowance is not None
        else 0
    )
    return PlannerDecisionEvaluation(
        decision_id=point.decision_id,
        split_group=point.split_group,
        target_kind=point.target.kind,
        predicted_kind=prediction.kind,
        schema_valid=True,
        action_kind_match=kind_match,
        exact_action_match=exact,
        tool_name_match=(
            kind_match and Counter(prediction.tool_names) == Counter(target_names)
        ),
        tool_name_set_match=(
            kind_match and set(prediction.tool_names) == set(target_names)
        ),
        tool_family_match=(
            kind_match
            and Counter(prediction.tool_families) == Counter(target_families)
        ),
        tool_family_set_match=(
            kind_match
            and set(prediction.tool_families) == set(target_families)
        ),
        tool_call_count_match=(
            kind_match and len(prediction.tool_names) == len(target_names)
        ),
        stop_correct=False,
        target_tool_calls=target_tool_calls,
        predicted_tool_calls=predicted_count,
        cache_hits=hits,
        cache_misses=misses,
        duplicate_tool_calls=duplicates,
        budget_state_visible=budget_visible,
        budget_compliant=(over_budget == 0) if budget_visible else None,
        tool_calls_over_budget=over_budget,
    )


def aggregate_planner_evaluations(
    evaluations: Iterable[PlannerDecisionEvaluation],
) -> dict[str, Any]:
    rows = list(evaluations)
    if not rows:
        raise ValueError("cannot aggregate an empty Planner evaluation")
    total = len(rows)
    tool_calls = sum(row.predicted_tool_calls for row in rows)
    target_tool_calls = sum(row.target_tool_calls for row in rows)
    stop_targets = sum(
        row.target_kind == PlannerActionKind.STOP for row in rows
    )
    tool_targets = total - stop_targets
    budget_rows = [row for row in rows if row.budget_state_visible]

    def rate(count: int, denominator: int = total) -> float | None:
        return round(count / denominator, 4) if denominator else None

    return {
        "decision_points": total,
        "task_groups": len({row.split_group for row in rows}),
        "schema_valid": sum(row.schema_valid for row in rows),
        "schema_valid_rate": rate(sum(row.schema_valid for row in rows)),
        "action_kind_matches": sum(row.action_kind_match for row in rows),
        "action_kind_accuracy": rate(
            sum(row.action_kind_match for row in rows)
        ),
        "exact_action_matches": sum(row.exact_action_match for row in rows),
        "exact_action_accuracy": rate(
            sum(row.exact_action_match for row in rows)
        ),
        "tool_name_matches": sum(row.tool_name_match for row in rows),
        "tool_name_accuracy": rate(sum(row.tool_name_match for row in rows)),
        "tool_name_set_matches": sum(
            row.tool_name_set_match for row in rows
        ),
        "tool_name_set_accuracy": rate(
            sum(row.tool_name_set_match for row in rows)
        ),
        "tool_family_matches": sum(row.tool_family_match for row in rows),
        "tool_family_accuracy": rate(
            sum(row.tool_family_match for row in rows)
        ),
        "tool_family_set_matches": sum(
            row.tool_family_set_match for row in rows
        ),
        "tool_family_set_accuracy": rate(
            sum(row.tool_family_set_match for row in rows)
        ),
        "tool_call_count_matches": sum(
            row.tool_call_count_match for row in rows
        ),
        "tool_call_count_accuracy": rate(
            sum(row.tool_call_count_match for row in rows)
        ),
        "stop_targets": stop_targets,
        "correct_stops": sum(row.stop_correct for row in rows),
        "stop_accuracy": rate(
            sum(row.stop_correct for row in rows), stop_targets
        ),
        "premature_stops": sum(row.premature_stop for row in rows),
        "premature_stop_rate": rate(
            sum(row.premature_stop for row in rows), tool_targets
        ),
        "predicted_tool_calls": tool_calls,
        "target_tool_calls": target_tool_calls,
        "tool_call_delta": tool_calls - target_tool_calls,
        "tool_call_inflation_rate": rate(
            tool_calls - target_tool_calls, target_tool_calls
        ),
        "cache_hits": sum(row.cache_hits for row in rows),
        "cache_misses": sum(row.cache_misses for row in rows),
        "cache_miss_rate": rate(
            sum(row.cache_misses for row in rows), tool_calls
        ),
        "duplicate_tool_calls": sum(
            row.duplicate_tool_calls for row in rows
        ),
        "duplicate_tool_call_rate": rate(
            sum(row.duplicate_tool_calls for row in rows), tool_calls
        ),
        "budget_aware_decisions": len(budget_rows),
        "budget_compliant_decisions": sum(
            row.budget_compliant is True for row in budget_rows
        ),
        "budget_compliance_rate": rate(
            sum(row.budget_compliant is True for row in budget_rows),
            len(budget_rows),
        ),
        "tool_calls_over_budget": sum(
            row.tool_calls_over_budget for row in budget_rows
        ),
    }


def evaluate_prediction_records(
    scenarios: Iterable[ReplayScenario],
    records: Iterable[PlannerPredictionRecord],
) -> tuple[list[PlannerDecisionEvaluation], dict[str, Any]]:
    """Evaluate a complete prediction set without shrinking denominators."""
    scenario_list = list(scenarios)
    point_by_id: dict[str, tuple[PlannerDecisionPoint, ObservationCache]] = {}
    for scenario in scenario_list:
        for point in scenario.decision_points:
            if point.decision_id in point_by_id:
                raise ValueError(f"duplicate decision id: {point.decision_id}")
            point_by_id[point.decision_id] = (
                point,
                scenario.observation_cache,
            )

    record_by_id: dict[str, PlannerPredictionRecord] = {}
    for record in records:
        if record.decision_id in record_by_id:
            raise ValueError(
                f"duplicate prediction decision id: {record.decision_id}"
            )
        if record.decision_id not in point_by_id:
            raise ValueError(
                f"prediction references unknown decision: {record.decision_id}"
            )
        record_by_id[record.decision_id] = record

    rows = []
    missing = 0
    for decision_id, (point, cache) in point_by_id.items():
        record = record_by_id.get(decision_id)
        if record is None:
            missing += 1
            budget_visible = _next_batch_allowance(point) is not None
            rows.append(
                PlannerDecisionEvaluation(
                    decision_id=decision_id,
                    split_group=point.split_group,
                    target_kind=point.target.kind,
                    schema_valid=False,
                    target_tool_calls=len(
                        point.target.message.get("tool_calls", [])
                    ),
                    budget_state_visible=budget_visible,
                    budget_compliant=False if budget_visible else None,
                    error="missing prediction",
                )
            )
            continue
        if record.message is None:
            budget_visible = _next_batch_allowance(point) is not None
            rows.append(
                PlannerDecisionEvaluation(
                    decision_id=decision_id,
                    split_group=point.split_group,
                    target_kind=point.target.kind,
                    schema_valid=False,
                    target_tool_calls=len(
                        point.target.message.get("tool_calls", [])
                    ),
                    budget_state_visible=budget_visible,
                    budget_compliant=False if budget_visible else None,
                    error=record.error or "prediction has no message",
                    metadata={
                        "model": record.model,
                        "latency_ms": record.latency_ms,
                    },
                )
            )
            continue
        evaluation = evaluate_planner_prediction(point, record.message, cache)
        evaluation.metadata.update(
            {"model": record.model, "latency_ms": record.latency_ms}
        )
        rows.append(evaluation)
    aggregate = aggregate_planner_evaluations(rows)
    aggregate.update(
        {
            "scenario_count": len(scenario_list),
            "prediction_records": len(record_by_id),
            "missing_predictions": missing,
        }
    )
    return rows, aggregate
