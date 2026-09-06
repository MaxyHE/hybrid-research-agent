"""Quality gates for externally collected, strong-teacher trajectories.

The gates deliberately evaluate the executable trace, rather than trusting a
collector exit code or a fluent final answer.  They are used before a trace
can become supervision for the Planner's ``continue / fetch / STOP``
decisions.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from .planner_eval import tool_family
from .planner_budget import PLANNER_BUDGET_PROTOCOL
from .scenario import PlannerActionKind, trace_to_replay_scenario
from .schema import EventKind, RunOutcome, TraceEvent
from .store import JsonlTraceStore


PUBLIC_TEACHER_EGRESS_SCOPE = "public_only"
PUBLIC_TEACHER_SOURCE_KINDS = frozenset({"public_web", "public_collection"})


def task_fingerprint(task: Mapping[str, Any]) -> str:
    """Return a non-reversible identifier for task-level split auditing."""
    query = str(task.get("query") or "").strip()
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def validate_external_teacher_task(task: Mapping[str, Any]) -> list[str]:
    """Validate the explicit public-data attestation required before egress.

    This validation intentionally does not infer whether material is public
    from a task's wording.  The task author must opt in with the three fields
    below, making an eventual cloud call auditable and fail-closed by default.
    """
    errors: list[str] = []
    if task.get("external_teacher_eligible") is not True:
        errors.append("external_teacher_eligible_not_true")
    if task.get("data_classification") != "public":
        errors.append("data_classification_not_public")
    source_kinds = task.get("allowed_source_kinds")
    if not isinstance(source_kinds, list) or not source_kinds:
        errors.append("allowed_source_kinds_missing")
    elif set(source_kinds) - PUBLIC_TEACHER_SOURCE_KINDS:
        errors.append("allowed_source_kinds_not_public")
    return errors


def validate_external_teacher_batch(
    tasks: Iterable[Mapping[str, Any]],
    hybrid_manifest: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Return task-id keyed preflight failures for a cloud-teacher batch."""
    failures: dict[str, list[str]] = {}
    if hybrid_manifest.get("egress_scope") != PUBLIC_TEACHER_EGRESS_SCOPE:
        failures["__batch__"] = ["hybrid_manifest_not_public_only"]
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        if not task_id:
            failures.setdefault("__batch__", []).append("task_id_missing")
            continue
        errors = validate_external_teacher_task(task)
        if errors:
            failures[task_id] = errors
    return failures


def _terminal_stop_is_runtime_action(events: list[TraceEvent]) -> bool:
    """Check that STOP comes from the Planner, not post-processing/writer text."""
    scenario = trace_to_replay_scenario(events)
    return bool(
        scenario.decision_points
        and scenario.decision_points[-1].target.kind == PlannerActionKind.STOP
    )


def _search_query_repeats(calls: list[TraceEvent]) -> int:
    queries = [
        str((call.tool_input or {}).get("query") or "").strip().casefold()
        for call in calls
        if call.tool_name
        and "search" in call.tool_name
        and str((call.tool_input or {}).get("query") or "").strip()
    ]
    return len(queries) - len(set(queries))


def _observation_has_evidence(event: TraceEvent) -> bool:
    content = str(event.content or "").strip().casefold()
    if not content:
        return False
    return not content.startswith(
        (
            "no results",
            "no results from",
            "failed to fetch",
            "cannot fetch",
            "not relevant",
            "no registered citation",
            "error:",
        )
    )


def audit_teacher_trace(
    events: Iterable[TraceEvent],
    task: Mapping[str, Any],
    *,
    require_budget_protocol: bool = False,
) -> dict[str, Any]:
    """Return deterministic strong-teacher gates and non-sensitive metrics.

    ``task`` may carry routing contract fields already used by the legacy
    collector: ``required_tool_families``, ``forbidden_tool_families``,
    ``expected_first_tool_family``, ``min_tool_calls`` and
    ``max_tool_calls``.  No query or observation text is copied into output.
    """
    validated = JsonlTraceStore.validate(events)
    calls = [event for event in validated if event.kind == EventKind.TOOL_CALL]
    observations = [
        event for event in validated if event.kind == EventKind.TOOL_OBSERVATION
    ]
    errors = [event for event in validated if event.kind == EventKind.ERROR]
    final_answers = [
        event
        for event in validated
        if event.kind == EventKind.FINAL_ANSWER
        and isinstance(event.content, str)
        and event.content.strip()
    ]
    terminal = validated[-1]
    terminal_metadata = terminal.metadata
    families = [tool_family(call.tool_name) for call in calls]
    observed_families = set(families)
    required_families = set(task.get("required_tool_families") or [])
    forbidden_families = set(task.get("forbidden_tool_families") or [])
    expected_first = task.get("expected_first_tool_family")
    min_calls = int(task.get("min_tool_calls", 2))
    max_calls = int(task.get("max_tool_calls", 14))
    observation_ids = {event.tool_call_id for event in observations}
    call_ids = {event.tool_call_id for event in calls}
    evidence_observations = [
        event for event in observations if _observation_has_evidence(event)
    ]
    evidence_families = {
        tool_family(event.tool_name) for event in evidence_observations
    }
    required_url_substrings = [
        str(value).strip().casefold()
        for value in task.get("required_url_substrings", [])
        if str(value).strip()
    ]
    evidence_text = "\n".join(
        str(event.content or "").casefold()
        for event in evidence_observations
    )
    missing_required_sources = [
        value for value in required_url_substrings if value not in evidence_text
    ]
    budget_stopped = bool(
        terminal_metadata.get("stopped_by_call_budget")
        or terminal_metadata.get("stopped_by_tool_call_budget")
    )

    try:
        scenario = trace_to_replay_scenario(validated)
        runtime_stop = bool(
            scenario.decision_points
            and scenario.decision_points[-1].target.kind
            == PlannerActionKind.STOP
        )
    except Exception:
        scenario = None
        runtime_stop = False
    budget_protocol_visible = bool(
        validated[0].metadata.get("planner_budget_protocol")
        == PLANNER_BUDGET_PROTOCOL
        and scenario is not None
        and scenario.decision_points
        and all(
            point.metadata.get("planner_budget_state_visible") is True
            for point in scenario.decision_points
        )
    )

    gates = {
        "terminal_success": terminal.outcome == RunOutcome.SUCCESS,
        "not_budget_stopped": not budget_stopped,
        "runtime_stop_action": runtime_stop,
        "no_trace_error": not errors,
        "matched_tool_observations": call_ids == observation_ids,
        "has_final_answer": bool(final_answers),
        "tool_call_range": min_calls <= len(calls) <= max_calls,
        "required_families": required_families.issubset(observed_families),
        "required_family_evidence": required_families.issubset(
            evidence_families
        ),
        "required_sources": not missing_required_sources,
        "forbidden_families": not (forbidden_families & observed_families),
        "first_family": (
            expected_first is None
            or bool(families)
            and families[0] == expected_first
        ),
        "no_exact_search_repeat": _search_query_repeats(calls) == 0,
        "budget_protocol_visible": (
            budget_protocol_visible if require_budget_protocol else True
        ),
    }
    passed = all(gates.values())
    return {
        "id": str(task.get("id") or ""),
        "status": "strong_teacher_candidate" if passed else "rejected",
        "teacher_model": validated[0].metadata.get("model"),
        "trace_outcome": terminal.outcome.value if terminal.outcome else None,
        "budget_stopped": budget_stopped,
        "budget_protocol_visible": budget_protocol_visible,
        "budget_replans": (
            (terminal_metadata.get("planner_budget") or {}).get(
                "replan_attempts"
            )
        ),
        "iterations": terminal_metadata.get("iterations"),
        "tool_calls": len(calls),
        "tool_family_counts": {
            family: families.count(family)
            for family in ("web", "collection", "fetch", "other")
        },
        "observed_tool_families": sorted(observed_families),
        "evidence_tool_families": sorted(evidence_families),
        "missing_required_sources": missing_required_sources,
        "exact_search_repeats": _search_query_repeats(calls),
        "task_fingerprint": task_fingerprint(task),
        "gates": gates,
    }


def compact_audit_record(
    audit: Mapping[str, Any], *, trace_path: str
) -> dict[str, Any]:
    """Attach only a trace reference; never copy task/observation content."""
    record = dict(audit)
    record["trace"] = trace_path
    return record
