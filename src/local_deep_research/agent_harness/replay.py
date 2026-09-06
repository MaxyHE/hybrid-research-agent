"""Offline transcript and cached-observation replay primitives.

The replay layer deliberately does not know about LangGraph or concrete search
engines.  It consumes validated ``agent-trace/v1`` events and never falls back
to a live tool call, which keeps planner comparisons reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import EventKind, RunOutcome, TraceEvent
from .store import JsonlTraceStore


CACHE_SCHEMA_VERSION = "agent-observation-cache/v1"


class ReplayError(ValueError):
    """Base class for deterministic replay failures."""


class ReplayCacheMiss(ReplayError):
    """Raised when strict replay receives an unseen tool invocation."""

    def __init__(self, tool_name: str, tool_input: dict[str, Any], signature: str):
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.signature = signature
        super().__init__(
            f"observation cache miss for {tool_name!r} ({signature})"
        )


class ReplayCacheConflict(ReplayError):
    """Raised when one tool signature maps to different observations."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ReplayError("replay values must be JSON serializable") from exc


def content_digest(content: Any) -> str:
    """Return a stable digest for a JSON-compatible observation."""
    return sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def canonical_tool_signature(
    tool_name: str,
    tool_input: dict[str, Any],
) -> str:
    """Identify a tool invocation without changing string semantics."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ReplayError("tool_name must be a non-empty string")
    if not isinstance(tool_input, dict):
        raise ReplayError("tool_input must be an object")
    payload = {"tool_name": tool_name, "tool_input": tool_input}
    digest = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


class ObservationCacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signature: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_input: dict[str, Any]
    content: Any
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_digests(self) -> "ObservationCacheEntry":
        expected_signature = canonical_tool_signature(
            self.tool_name, self.tool_input
        )
        if self.signature != expected_signature:
            raise ValueError("cache entry signature does not match tool input")
        if self.content_digest != content_digest(self.content):
            raise ValueError("cache entry content digest does not match content")
        return self


class ObservationCache(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent-observation-cache/v1"] = (
        CACHE_SCHEMA_VERSION
    )
    entries: list[ObservationCacheEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_signatures(self) -> "ObservationCache":
        seen: dict[str, str] = {}
        for entry in self.entries:
            previous = seen.get(entry.signature)
            if previous is not None and previous != entry.content_digest:
                raise ValueError(
                    "one tool signature maps to conflicting observations"
                )
            if previous is not None:
                raise ValueError(f"duplicate cache signature: {entry.signature}")
            seen[entry.signature] = entry.content_digest
        return self


class ObservationCacheStore:
    """Build and persist versioned caches from validated trajectories."""

    @staticmethod
    def from_traces(
        traces: Iterable[Iterable[TraceEvent]],
    ) -> ObservationCache:
        by_signature: dict[str, ObservationCacheEntry] = {}
        trace_count = 0
        for raw_events in traces:
            events = JsonlTraceStore.validate(raw_events)
            trace_count += 1
            calls = {
                event.tool_call_id: event
                for event in events
                if event.kind == EventKind.TOOL_CALL
            }
            for observation in events:
                if observation.kind != EventKind.TOOL_OBSERVATION:
                    continue
                call = calls[observation.tool_call_id]
                tool_input = call.tool_input or {}
                tool_name = call.tool_name or ""
                signature = canonical_tool_signature(tool_name, tool_input)
                digest = content_digest(observation.content)
                existing = by_signature.get(signature)
                if existing is not None:
                    if existing.content_digest != digest:
                        raise ReplayCacheConflict(
                            "one tool signature produced different observations: "
                            f"{signature}"
                        )
                    if event_run_id := observation.run_id:
                        if event_run_id not in existing.source_run_ids:
                            existing.source_run_ids.append(event_run_id)
                    continue
                by_signature[signature] = ObservationCacheEntry(
                    signature=signature,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    content=observation.content,
                    content_digest=digest,
                    source_run_ids=[observation.run_id],
                    metadata={"status": observation.metadata.get("status")},
                )
        return ObservationCache(
            entries=list(by_signature.values()),
            metadata={"source_trace_count": trace_count},
        )

    @staticmethod
    def write(path: str | Path, cache: ObservationCache) -> Path:
        target = Path(path)
        validated = ObservationCache.model_validate(cache.model_dump())
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
                    validated.model_dump(mode="json"),
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

    @staticmethod
    def read(path: str | Path) -> ObservationCache:
        return ObservationCache.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )


class ReplaySession:
    """Strict, offline lookup session for frozen tool observations."""

    def __init__(self, cache: ObservationCache) -> None:
        self._entries = {entry.signature: entry for entry in cache.entries}
        self.hits = 0
        self.misses = 0
        self.calls: list[dict[str, Any]] = []

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        signature = canonical_tool_signature(tool_name, tool_input)
        entry = self._entries.get(signature)
        if entry is None:
            self.misses += 1
            self.calls.append(
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "signature": signature,
                    "hit": False,
                }
            )
            raise ReplayCacheMiss(tool_name, tool_input, signature)
        self.hits += 1
        self.calls.append(
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "signature": signature,
                "hit": True,
                "content_digest": entry.content_digest,
            }
        )
        return entry.content

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class ReplayTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    outcome: RunOutcome
    messages: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


def rebuild_transcript(events: Iterable[TraceEvent]) -> ReplayTranscript:
    """Reconstruct ordered model/tool messages from any valid trajectory."""
    validated = JsonlTraceStore.validate(events)
    messages: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    index = 1
    while index < len(validated) - 1:
        event = validated[index]
        if event.kind == EventKind.MESSAGE:
            messages.append({"role": event.role, "content": event.content})
            index += 1
            continue
        if event.kind == EventKind.TOOL_CALL:
            calls = []
            while (
                index < len(validated) - 1
                and validated[index].kind == EventKind.TOOL_CALL
            ):
                call = validated[index]
                calls.append(
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": call.tool_input or {},
                        },
                    }
                )
                index += 1
            messages.append(
                {"role": "assistant", "content": "", "tool_calls": calls}
            )
            continue
        if event.kind == EventKind.TOOL_OBSERVATION:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.tool_call_id,
                    "name": event.tool_name,
                    "content": event.content,
                }
            )
        elif event.kind == EventKind.FINAL_ANSWER:
            messages.append(
                {
                    "role": "assistant",
                    "content": event.content,
                    "metadata": {"trace_kind": "final_answer"},
                }
            )
        elif event.kind == EventKind.ERROR:
            errors.append(
                {
                    "event_id": event.event_id,
                    "error_type": event.error_type,
                    "content": event.content,
                }
            )
        index += 1
    end = validated[-1]
    return ReplayTranscript(
        run_id=end.run_id,
        outcome=end.outcome or RunOutcome.ERROR,
        messages=messages,
        errors=errors,
        metadata={
            "source_schema_version": validated[0].schema_version,
            "event_count": len(validated),
        },
    )
