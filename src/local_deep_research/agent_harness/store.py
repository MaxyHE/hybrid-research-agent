"""Deterministic JSONL storage and trajectory-level validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from pydantic import ValidationError

from .schema import EventKind, TraceEvent


class TraceValidationError(ValueError):
    """Raised when individually valid events do not form a valid run."""


class JsonlTraceStore:
    @staticmethod
    def validate(events: Iterable[TraceEvent]) -> list[TraceEvent]:
        materialized = list(events)
        if not materialized:
            raise TraceValidationError("trajectory is empty")

        run_id = materialized[0].run_id
        observed_calls: set[str] = set()
        observed_results: set[str] = set()
        for expected_id, event in enumerate(materialized):
            if event.run_id != run_id:
                raise TraceValidationError(
                    "trajectory contains multiple run_ids"
                )
            if event.event_id != expected_id:
                raise TraceValidationError(
                    f"event_id must be contiguous: expected {expected_id}, "
                    f"got {event.event_id}"
                )
            if event.kind == EventKind.TOOL_CALL:
                if event.tool_call_id in observed_calls:
                    raise TraceValidationError(
                        f"duplicate tool_call_id: {event.tool_call_id}"
                    )
                observed_calls.add(event.tool_call_id or "")
            elif event.kind == EventKind.TOOL_OBSERVATION:
                if event.tool_call_id not in observed_calls:
                    raise TraceValidationError(
                        "tool_observation references an unseen tool_call_id: "
                        f"{event.tool_call_id}"
                    )
                if event.tool_call_id in observed_results:
                    raise TraceValidationError(
                        "multiple tool_observations for tool_call_id: "
                        f"{event.tool_call_id}"
                    )
                observed_results.add(event.tool_call_id or "")

        if materialized[0].kind != EventKind.RUN_START:
            raise TraceValidationError("first event must be run_start")
        if materialized[-1].kind != EventKind.RUN_END:
            raise TraceValidationError("last event must be run_end")
        if (
            materialized[-1].outcome is not None
            and materialized[-1].outcome.value == "success"
            and observed_calls != observed_results
        ):
            missing = sorted(observed_calls - observed_results)
            raise TraceValidationError(
                f"successful trajectory has unobserved tool calls: {missing}"
            )
        return materialized

    @classmethod
    def write(cls, path: str | Path, events: Iterable[TraceEvent]) -> Path:
        target = Path(path)
        validated = cls.validate(events)
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
            for event in validated:
                handle.write(
                    json.dumps(
                        event.model_dump(mode="json"),
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

    @classmethod
    def read(cls, path: str | Path) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        events.append(TraceEvent.model_validate_json(line))
                    except ValidationError as exc:
                        raise TraceValidationError(
                            f"invalid event at line {line_number}: {exc}"
                        ) from exc
        except json.JSONDecodeError as exc:
            raise TraceValidationError(f"invalid JSONL: {exc}") from exc
        return cls.validate(events)
