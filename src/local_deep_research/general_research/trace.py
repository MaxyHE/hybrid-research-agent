"""Structured, JSONL-safe audit events for General Research Agent runs.

The events are deliberately separate from the generic agent trajectory schema.
They can be embedded in ``TraceRecorder`` metadata today and stored as their
own JSONL evidence ledger without changing legacy Hybrid traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from typing import Any, Mapping
from urllib.parse import urlparse

from .config import GeneralExecutionConfig
from .schemas import SourceChannel


GENERAL_AUDIT_EVENT_SCHEMA_VERSION = "general-research-audit-event/v1"
GENERAL_AUDIT_JSONL_SCHEMA_VERSION = "general-research-audit-jsonl/v1"


class GeneralAuditEventKind(StrEnum):
    BRIEF = "brief"
    PLAN = "plan"
    SUPERVISION = "supervision"
    ACTION = "action"
    SOURCE = "source"
    EVIDENCE = "evidence"
    COVERAGE = "coverage"
    MEMO = "memo"
    CITATION_AUDIT = "citation_audit"


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _json_object(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    """Copy and validate untrusted event data into deterministic JSON values."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain JSON-compatible values") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # defensive; dict input always decodes to dict
        raise TypeError(f"{field_name} must encode as a JSON object")
    return decoded


def _http_url(value: object, *, field_name: str) -> str:
    url = _text(value, field_name=field_name)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    return url


def _local_collection_locator(value: object, *, field_name: str) -> str:
    locator = _text(value, field_name=field_name)
    parsed = urlparse(locator)
    if (
        parsed.scheme != "local"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.params
        or parsed.query
        or parsed.fragment
        or any(segment in {".", ".."} for segment in parsed.path.split("/"))
    ):
        raise ValueError(
            f"{field_name} must be local://<collection>/<document-id>"
        )
    return locator


def _event_timestamp(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class GeneralAuditEvent:
    """One versioned audit event emitted by a General Research Agent run."""

    run_id: str
    event_id: int
    kind: GeneralAuditEventKind
    data: Mapping[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = GENERAL_AUDIT_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.run_id, field_name="run_id")
        if not isinstance(self.event_id, int) or isinstance(self.event_id, bool):
            raise TypeError("event_id must be an integer")
        if self.event_id < 0:
            raise ValueError("event_id cannot be negative")
        if self.schema_version != GENERAL_AUDIT_EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported audit event schema: {self.schema_version!r}"
            )
        if not isinstance(self.kind, GeneralAuditEventKind):
            raise TypeError("kind must be GeneralAuditEventKind")
        timestamp = _event_timestamp(self.timestamp)
        normalized_data = _json_object(self.data, field_name="event data")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "data", normalized_data)

    @classmethod
    def brief(
        cls,
        *,
        run_id: str,
        event_id: int,
        brief_id: str,
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.BRIEF,
            data={"brief_id": _text(brief_id, field_name="brief_id"), **extra},
            timestamp=_event_timestamp(timestamp),
        )

    @classmethod
    def plan(
        cls,
        *,
        run_id: str,
        event_id: int,
        plan_id: str,
        claim_ids: list[str] | tuple[str, ...],
        status: str = "created",
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        normalized_claims = [_text(claim, field_name="claim id") for claim in claim_ids]
        if not normalized_claims:
            raise ValueError("plan requires at least one claim_id")
        data = {
            "plan_id": _text(plan_id, field_name="plan_id"),
            "claim_ids": normalized_claims,
            "status": _text(status, field_name="plan status"),
            **extra,
        }
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.PLAN,
            data=data,
            timestamp=_event_timestamp(timestamp),
        )

    @classmethod
    def supervision(
        cls,
        *,
        run_id: str,
        event_id: int,
        round_index: int,
        task_ids: list[str] | tuple[str, ...],
        should_finish: bool,
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise ValueError("supervision round_index must be a non-negative integer")
        if not isinstance(should_finish, bool):
            raise TypeError("supervision should_finish must be a boolean")
        normalized_ids = [_text(task_id, field_name="task_id") for task_id in task_ids]
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("supervision task_ids must be unique")
        if should_finish and normalized_ids:
            raise ValueError("a finished supervision decision cannot include task_ids")
        if not should_finish and not normalized_ids:
            raise ValueError("a dispatch supervision decision requires task_ids")
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.SUPERVISION,
            data={
                "round_index": round_index,
                "task_ids": normalized_ids,
                "should_finish": should_finish,
                **extra,
            },
            timestamp=_event_timestamp(timestamp),
        )

    @classmethod
    def source(
        cls,
        *,
        run_id: str,
        event_id: int,
        source_id: str,
        url: str,
        source_class: str,
        fetch_status: str,
        source_channel: SourceChannel | str = SourceChannel.PUBLIC_WEB,
        source_connector_id: str = "public_web",
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        resolved_channel = SourceChannel(source_channel)
        locator = (
            _http_url(url, field_name="source URL")
            if resolved_channel == SourceChannel.PUBLIC_WEB
            else _local_collection_locator(url, field_name="source locator")
        )
        data = {
            "source_id": _text(source_id, field_name="source_id"),
            "url": locator,
            "source_channel": resolved_channel.value,
            "source_connector_id": _text(
                source_connector_id, field_name="source_connector_id"
            ),
            "source_class": _text(source_class, field_name="source_class"),
            "fetch_status": _text(fetch_status, field_name="fetch_status"),
            **extra,
        }
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.SOURCE,
            data=data,
            timestamp=_event_timestamp(timestamp),
        )

    @classmethod
    def action(
        cls,
        *,
        run_id: str,
        event_id: int,
        step: int,
        action_type: str,
        outcome: str,
        connector_id: str | None = None,
        candidate_id: str | None = None,
        query_sha256: str | None = None,
        candidate_count: int | None = None,
        reason: str | None = None,
        error_code: str | None = None,
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("action step must be a non-negative integer")
        if action_type not in {"search", "fetch", "request_stop"}:
            raise ValueError("action_type must be search, fetch, or request_stop")
        if outcome not in {"executed", "rejected", "failed", "stopped", "step_limit"}:
            raise ValueError("action outcome is invalid")
        if candidate_count is not None and (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < 0
        ):
            raise ValueError("candidate_count must be a non-negative integer")
        data: dict[str, Any] = {
            "step": step,
            "action_type": action_type,
            "outcome": outcome,
        }
        for field_name, value in (
            ("connector_id", connector_id),
            ("candidate_id", candidate_id),
            ("query_sha256", query_sha256),
            ("reason", reason),
            ("error_code", error_code),
        ):
            if value is not None:
                data[field_name] = _text(value, field_name=field_name)
        if candidate_count is not None:
            data["candidate_count"] = candidate_count
        data.update(extra)
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.ACTION,
            data=data,
            timestamp=_event_timestamp(timestamp),
        )

    @classmethod
    def evidence(
        cls,
        *,
        run_id: str,
        event_id: int,
        evidence_id: str,
        claim_id: str,
        source_id: str,
        excerpt: str,
        support: str,
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        if support not in {"supports", "partially_supports", "refutes"}:
            raise ValueError("support must be supports, partially_supports, or refutes")
        data = {
            "evidence_id": _text(evidence_id, field_name="evidence_id"),
            "claim_id": _text(claim_id, field_name="claim_id"),
            "source_id": _text(source_id, field_name="source_id"),
            "excerpt": _text(excerpt, field_name="evidence excerpt"),
            "support": support,
            **extra,
        }
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.EVIDENCE,
            data=data,
            timestamp=_event_timestamp(timestamp),
        )

    @classmethod
    def coverage(
        cls,
        *,
        run_id: str,
        event_id: int,
        total_claims: int,
        covered_claims: int,
        uncovered_claim_ids: list[str] | tuple[str, ...],
        decision: str,
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        if total_claims < 0 or covered_claims < 0 or covered_claims > total_claims:
            raise ValueError("coverage counts must satisfy 0 <= covered <= total")
        if decision not in {"continue", "synthesize", "blocked"}:
            raise ValueError(
                "coverage decision must be continue, synthesize, or blocked"
            )
        data = {
            "total_claims": total_claims,
            "covered_claims": covered_claims,
            "uncovered_claim_ids": [
                _text(claim, field_name="uncovered claim id")
                for claim in uncovered_claim_ids
            ],
            "decision": decision,
            **extra,
        }
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.COVERAGE,
            data=data,
            timestamp=_event_timestamp(timestamp),
        )

    @classmethod
    def memo(
        cls,
        *,
        run_id: str,
        event_id: int,
        task_id: str,
        evidence_ids: list[str] | tuple[str, ...],
        unresolved_count: int,
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        if isinstance(unresolved_count, bool) or not isinstance(unresolved_count, int) or unresolved_count < 0:
            raise ValueError("memo unresolved_count must be a non-negative integer")
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.MEMO,
            data={
                "task_id": _text(task_id, field_name="task_id"),
                "evidence_ids": [
                    _text(evidence_id, field_name="evidence_id")
                    for evidence_id in evidence_ids
                ],
                "unresolved_count": unresolved_count,
                **extra,
            },
            timestamp=_event_timestamp(timestamp),
        )

    @classmethod
    def citation_audit(
        cls,
        *,
        run_id: str,
        event_id: int,
        citation_count: int,
        supported_citation_count: int,
        unsupported_citation_ids: list[str] | tuple[str, ...],
        verdict: str,
        timestamp: datetime | None = None,
        **extra: Any,
    ) -> "GeneralAuditEvent":
        if citation_count < 0 or supported_citation_count < 0:
            raise ValueError("citation counts cannot be negative")
        if supported_citation_count > citation_count:
            raise ValueError("supported citations cannot exceed all citations")
        if verdict not in {"pass", "warning", "fail"}:
            raise ValueError("citation audit verdict must be pass, warning, or fail")
        normalized_unsupported = [
            _text(citation_id, field_name="unsupported citation id")
            for citation_id in unsupported_citation_ids
        ]
        if verdict == "pass" and normalized_unsupported:
            raise ValueError(
                "a passing citation audit cannot have unsupported citations"
            )
        data = {
            "citation_count": citation_count,
            "supported_citation_count": supported_citation_count,
            "unsupported_citation_ids": normalized_unsupported,
            "verdict": verdict,
            **extra,
        }
        return cls(
            run_id=run_id,
            event_id=event_id,
            kind=GeneralAuditEventKind.CITATION_AUDIT,
            data=data,
            timestamp=_event_timestamp(timestamp),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "kind": self.kind.value,
            "data": _json_object(self.data, field_name="event data"),
        }


def general_event_jsonl_payload(
    event: GeneralAuditEvent, config: GeneralExecutionConfig
) -> dict[str, Any]:
    """Build one standalone JSONL record linked to an immutable run config."""
    if event.run_id != config.run.run_id:
        raise ValueError("event.run_id must match config.run.run_id")
    return {
        "schema_version": GENERAL_AUDIT_JSONL_SCHEMA_VERSION,
        "run_id": event.run_id,
        "config_digest_sha256": config.digest(),
        "event": event.canonical_dict(),
    }


def general_event_jsonl_line(
    event: GeneralAuditEvent, config: GeneralExecutionConfig
) -> str:
    """Serialize one deterministic JSONL line (without a trailing newline)."""
    return json.dumps(
        general_event_jsonl_payload(event, config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def general_event_trace_metadata(
    event: GeneralAuditEvent, config: GeneralExecutionConfig | None = None
) -> dict[str, Any]:
    """Return metadata suitable for ``TraceRecorder.record_*`` methods.

    Passing the config is recommended for a self-contained event.  Run-start
    metadata should separately use ``config.trace_metadata()``.
    """
    metadata: dict[str, Any] = {"general_research_event": event.canonical_dict()}
    if config is not None:
        if event.run_id != config.run.run_id:
            raise ValueError("event.run_id must match config.run.run_id")
        metadata["general_research_config_digest_sha256"] = config.digest()
    return metadata
