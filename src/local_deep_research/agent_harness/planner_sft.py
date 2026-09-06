"""Convert gated Planner decisions into action-only Tool-SFT examples."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from .scenario import (
    PlannerActionKind,
    PlannerDecisionPoint,
    ReplayScenario,
)
from .sft import compact_tool_observation


class PlannerSftConversionError(ValueError):
    """Raised when a decision is not safe for action-only supervision."""


CANONICAL_PLANNER_STOP = "Research complete."


def _is_safe_canonical_stop(point: PlannerDecisionPoint) -> bool:
    """Return whether a runtime STOP can be reduced to a Planner-only action."""
    return (
        point.target.kind == PlannerActionKind.STOP
        and not point.sft_eligible
        and point.metadata.get("trace_outcome") == "success"
        and point.cache_covered
        and not point.duplicate_with_history
        and point.sft_exclusion_reasons == ["stop_requires_action_only_loss"]
    )


def assistant_supervision_indices(example: dict[str, Any]) -> tuple[int, ...]:
    """Return assistant message indices selected by the example's loss scope."""
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise PlannerSftConversionError("example messages must be an array")
    assistant_indices = tuple(
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    )
    loss_scope = (example.get("metadata") or {}).get("loss_scope")
    if loss_scope in (None, "all_assistant_turns"):
        return assistant_indices
    if loss_scope not in {
        "assistant_tool_action_only",
        "assistant_action_only",
    }:
        raise PlannerSftConversionError(f"unsupported loss_scope: {loss_scope}")
    if not messages:
        raise PlannerSftConversionError("action-only example has no messages")
    target = messages[-1]
    if not isinstance(target, dict) or target.get("role") != "assistant":
        raise PlannerSftConversionError(
            "action-only example must end with an assistant action"
        )
    if loss_scope == "assistant_tool_action_only" and not target.get(
        "tool_calls"
    ):
        raise PlannerSftConversionError(
            "action-only example must end with an assistant tool call"
        )
    if loss_scope == "assistant_action_only":
        has_calls = bool(target.get("tool_calls"))
        has_content = bool(str(target.get("content") or "").strip())
        if not has_calls and not has_content:
            raise PlannerSftConversionError(
                "Planner action target must contain tool_calls or canonical "
                "STOP content"
            )
    return (len(messages) - 1,)


def planner_decision_to_sft_example(
    point: PlannerDecisionPoint,
    *,
    max_tool_observation_chars: int | None = None,
    include_stop_target: bool = False,
) -> dict[str, Any]:
    canonical_stop = include_stop_target and _is_safe_canonical_stop(point)
    if not point.sft_eligible and not canonical_stop:
        raise PlannerSftConversionError(
            f"decision is not SFT eligible: {point.sft_exclusion_reasons}"
        )
    if point.target.kind not in {
        PlannerActionKind.TOOL_CALL,
        PlannerActionKind.STOP,
    }:
        raise PlannerSftConversionError(
            f"unsupported Planner action kind: {point.target.kind}"
        )
    messages = deepcopy(point.context_messages)
    target = (
        {"role": "assistant", "content": CANONICAL_PLANNER_STOP}
        if canonical_stop
        else deepcopy(point.target.message)
    )
    messages.append(target)
    if messages[-1].get("role") != "assistant" or (
        not canonical_stop and not messages[-1].get("tool_calls")
    ):
        raise PlannerSftConversionError(
            "training target must be an assistant Planner action"
        )

    compacted = 0
    original_chars = 0
    output_chars = 0
    if max_tool_observation_chars is not None:
        if max_tool_observation_chars < 160:
            raise PlannerSftConversionError(
                "max_tool_observation_chars must be at least 160"
            )
        for message in messages[:-1]:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            original_chars += len(content)
            if len(content) > max_tool_observation_chars:
                message["content"] = compact_tool_observation(
                    content, max_tool_observation_chars
                )
                compacted += 1
            output_chars += len(message["content"])

    return {
        "messages": messages,
        "tools": deepcopy(point.tools),
        "metadata": {
            "schema_version": point.schema_version,
            "decision_id": point.decision_id,
            "source_run_id": point.source_run_id,
            "split_group": point.split_group,
            "step_index": point.step_index,
            "loss_scope": (
                "assistant_action_only"
                if include_stop_target
                else "assistant_tool_action_only"
            ),
            "target_kind": point.target.kind.value,
            "canonical_stop": canonical_stop,
            "planner_budget_state_visible": point.metadata.get(
                "planner_budget_state_visible", False
            ),
            "planner_budget_state": point.metadata.get("planner_budget_state"),
            "tool_observation_compaction": (
                {
                    "enabled": True,
                    "max_chars": max_tool_observation_chars,
                    "compacted_messages": compacted,
                    "original_chars": original_chars,
                    "compacted_chars": output_chars,
                    "method": "head_tail_with_explicit_marker",
                }
                if max_tool_observation_chars is not None
                else {"enabled": False}
            ),
        },
    }


def write_planner_sft_jsonl(
    path: str | Path,
    scenarios: Iterable[ReplayScenario],
    *,
    max_tool_observation_chars: int | None = None,
    include_stop_targets: bool = False,
) -> tuple[Path, dict[str, int]]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    seen_groups: set[str] = set()
    written = 0
    excluded = 0
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
                raise PlannerSftConversionError(
                    f"duplicate split group: {scenario.split_group}"
                )
            seen_groups.add(scenario.split_group)
            for point in scenario.decision_points:
                if not point.sft_eligible and not (
                    include_stop_targets and _is_safe_canonical_stop(point)
                ):
                    excluded += 1
                    continue
                example = planner_decision_to_sft_example(
                    point,
                    max_tool_observation_chars=max_tool_observation_chars,
                    include_stop_target=include_stop_targets,
                )
                handle.write(
                    json.dumps(
                        example,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
                written += 1
        handle.flush()
        os.fsync(handle.fileno())
    if not written:
        temporary.unlink(missing_ok=True)
        raise PlannerSftConversionError("no SFT-eligible Planner decisions")
    os.replace(temporary, target)
    return target, {
        "task_groups": len(seen_groups),
        "examples": written,
        "excluded_decisions": excluded,
    }
