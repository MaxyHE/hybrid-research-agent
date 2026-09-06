"""Durable request-lifecycle tracing for online Agent reliability runs.

Agent action traces begin only after the strategy has been constructed.  That
is too late to diagnose a blocked model request or a preflight failure.  This
small sidecar trace deliberately has a simpler schema and is flushed after
every event so a killed process still leaves the last known execution stage.

It carries runtime-only metadata (request IDs, stages, raw errors) and never
contains user answers, evaluator annotations, or credentials.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping
from uuid import uuid4


REQUEST_LIFECYCLE_PROTOCOL = "agent-request-lifecycle/v1"

_ACTIVE_REQUEST_LIFECYCLE: ContextVar["RequestLifecycleTrace | None"] = (
    ContextVar("active_request_lifecycle", default=None)
)


def normalize_failure_category(
    raw_error: object | None,
    *,
    stage: str | None = None,
) -> str | None:
    """Map runtime-visible errors to a compact, reportable category.

    The original error remains in the lifecycle event.  This classification is
    intentionally conservative: it does not infer an HTTP failure type that
    the runtime did not observe.
    """

    text = str(raw_error or "").lower()
    normalized_stage = str(stage or "").lower()
    if "queue" in text or "queue" in normalized_stage:
        return "service_queue_timeout"
    if "timeout" in text or "timed out" in text:
        if "preflight" in normalized_stage or "warmup" in normalized_stage:
            return "preflight_timeout"
        if "planner" in normalized_stage or "model" in normalized_stage:
            return "model_request_timeout"
        return "timeout"
    if "403" in text or "forbidden" in text or "blocked" in text:
        return "blocked"
    if "empty body" in text or not text.strip():
        return "empty_body"
    if "landing page" in text:
        return "landing_page"
    if "source role" in text and "mismatch" in text:
        return "source_role_mismatch"
    if "irrelevant" in text or "not relevant" in text:
        return "irrelevant_body"
    return None


class RequestLifecycleTrace:
    """Append-only sidecar trace for the pre-Planner and model lifecycle."""

    def __init__(
        self,
        path: str | Path,
        *,
        client_request_id: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client_request_id = client_request_id or f"ldr-{uuid4().hex}"
        self._started_monotonic = time.perf_counter()
        self._sequence = 0
        self.last_stage: str | None = None
        self.last_server_request_id: str | None = None

    def record(self, stage: str, **metadata: Any) -> dict[str, Any]:
        """Append and flush one event before control returns to the caller."""

        self._sequence += 1
        server_request_id = metadata.pop("server_request_id", None)
        if server_request_id:
            self.last_server_request_id = str(server_request_id)
        event = {
            "protocol": REQUEST_LIFECYCLE_PROTOCOL,
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round(
                (time.perf_counter() - self._started_monotonic) * 1000, 3
            ),
            "stage": stage,
            "client_request_id": self.client_request_id,
            "server_request_id": self.last_server_request_id,
            **metadata,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
        self.last_stage = stage
        return event

    def timeout(self, raw_error: object, *, stage: str | None = None) -> None:
        timeout_stage = stage or self.last_stage or "unknown"
        self.record(
            "timeout",
            timeout_stage=timeout_stage,
            failure_category=normalize_failure_category(
                raw_error, stage=timeout_stage
            ),
            raw_error=str(raw_error),
        )

    def close(
        self,
        outcome: str,
        *,
        raw_error: object | None = None,
        timeout_stage: str | None = None,
    ) -> None:
        self.record(
            "request_end",
            outcome=outcome,
            timeout_stage=timeout_stage,
            failure_category=normalize_failure_category(
                raw_error, stage=timeout_stage
            )
            if raw_error is not None
            else None,
            raw_error=str(raw_error) if raw_error is not None else None,
        )


def set_request_lifecycle(
    trace: RequestLifecycleTrace | None,
) -> Token[RequestLifecycleTrace | None]:
    """Make a lifecycle trace available to the current planner invocation."""

    return _ACTIVE_REQUEST_LIFECYCLE.set(trace)


def reset_request_lifecycle(token: Token[RequestLifecycleTrace | None]) -> None:
    _ACTIVE_REQUEST_LIFECYCLE.reset(token)


def get_request_lifecycle() -> RequestLifecycleTrace | None:
    return _ACTIVE_REQUEST_LIFECYCLE.get()


def lifecycle_headers() -> Mapping[str, str]:
    """Expose an opaque task request ID to compatible local model servers."""

    trace = get_request_lifecycle()
    return (
        {"X-LDR-Client-Request-Id": trace.client_request_id}
        if trace is not None
        else {}
    )


def response_server_request_id(response: object) -> str | None:
    """Best-effort extraction without assuming one OpenAI-compatible server."""

    candidates: list[object] = []
    result = getattr(response, "result", None)
    if isinstance(result, list):
        candidates.extend(result)
    candidates.append(response)
    for candidate in candidates:
        metadata = getattr(candidate, "response_metadata", None)
        if isinstance(metadata, Mapping):
            for key in ("id", "request_id", "x_request_id"):
                value = metadata.get(key)
                if value:
                    return str(value)
        additional = getattr(candidate, "additional_kwargs", None)
        if isinstance(additional, Mapping):
            for key in ("id", "request_id", "x_request_id"):
                value = additional.get(key)
                if value:
                    return str(value)
    return None


__all__ = [
    "REQUEST_LIFECYCLE_PROTOCOL",
    "RequestLifecycleTrace",
    "get_request_lifecycle",
    "lifecycle_headers",
    "normalize_failure_category",
    "reset_request_lifecycle",
    "response_server_request_id",
    "set_request_lifecycle",
]
