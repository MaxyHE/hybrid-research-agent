"""Typed, capability-gated actions for the General V1 research controller.

The controller never receives a callable tool object and never emits a raw URL
to fetch.  It chooses a connector for search, then chooses an opaque candidate
ID returned by that connector.  The runtime owns candidate-to-resource
resolution and authorization, which prevents URL invention and keeps private
collection locators out of model-visible state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Iterable, Mapping
from urllib.parse import urlparse

from .config import SourceConnectorConfig
from .schemas import SourceChannel


class ActionType(StrEnum):
    SEARCH = "search"
    FETCH = "fetch"
    REQUEST_STOP = "request_stop"


class ActionGateError(ValueError):
    """A proposed action exceeds the run's declared capability boundary."""


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if any(char.isspace() for char in normalized):
        raise ValueError(f"{field_name} must not contain whitespace")
    return normalized


def _text(value: object, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} exceeds {max_length} characters")
    return normalized


def _timestamp(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """A connector-produced candidate; the resource locator stays runtime-only."""

    candidate_id: str
    connector_id: str
    source_channel: SourceChannel
    resource_locator: str
    title: str
    snippet: str
    observed_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, field_name="candidate_id"))
        object.__setattr__(self, "connector_id", _identifier(self.connector_id, field_name="connector_id"))
        object.__setattr__(self, "source_channel", SourceChannel(self.source_channel))
        object.__setattr__(
            self,
            "resource_locator",
            _text(self.resource_locator, field_name="resource_locator", max_length=4_096),
        )
        object.__setattr__(self, "title", _text(self.title, field_name="title", max_length=1_000))
        # Discovery snippets are model-visible triage context, not evidence.
        # A tight cap prevents one search provider response from dominating the
        # controller context; full text is available only after a Fetch action.
        object.__setattr__(self, "snippet", _text(self.snippet, field_name="snippet", max_length=1_500))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, field_name="observed_at"))
        parsed = urlparse(self.resource_locator)
        if self.source_channel == SourceChannel.PUBLIC_WEB:
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("public_web candidate must use an absolute HTTP(S) locator")
        elif self.source_channel == SourceChannel.LOCAL_COLLECTION:
            if (
                parsed.scheme != "local"
                or not parsed.netloc
                or not parsed.path.strip("/")
                or parsed.params
                or parsed.query
                or parsed.fragment
                or any(segment in {".", ".."} for segment in parsed.path.split("/"))
            ):
                raise ValueError("local_collection candidate must use a local:// locator")

    def model_view(self) -> dict[str, str]:
        """Return safe selection context without exposing a fetchable locator."""

        return {
            "candidate_id": self.candidate_id,
            "connector_id": self.connector_id,
            "source_channel": self.source_channel.value,
            "title": self.title,
            "snippet": self.snippet,
        }


@dataclass(frozen=True, slots=True)
class SearchAction:
    connector_id: str
    query: str
    action_type: ActionType = ActionType.SEARCH

    def __post_init__(self) -> None:
        object.__setattr__(self, "connector_id", _identifier(self.connector_id, field_name="connector_id"))
        object.__setattr__(self, "query", _text(self.query, field_name="query", max_length=512))
        if self.action_type != ActionType.SEARCH:
            raise ValueError("SearchAction.action_type must be search")


@dataclass(frozen=True, slots=True)
class FetchAction:
    candidate_id: str
    action_type: ActionType = ActionType.FETCH

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, field_name="candidate_id"))
        if self.action_type != ActionType.FETCH:
            raise ValueError("FetchAction.action_type must be fetch")


@dataclass(frozen=True, slots=True)
class RequestStopAction:
    reason: str
    action_type: ActionType = ActionType.REQUEST_STOP

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _text(self.reason, field_name="reason", max_length=512))
        if self.action_type != ActionType.REQUEST_STOP:
            raise ValueError("RequestStopAction.action_type must be request_stop")


GeneralAction = SearchAction | FetchAction | RequestStopAction


class ActionOutcome(StrEnum):
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"
    STOPPED = "stopped"
    STEP_LIMIT = "step_limit"


def _digest(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ActionTransition:
    """Replay-safe record of one controller proposal and runtime outcome.

    Search queries are hashed because a local-collection query may itself be
    sensitive.  Candidate IDs are opaque runtime handles, so they are safe to
    preserve for explaining the control flow without leaking a resource URL.
    """

    step: int
    action_type: ActionType
    outcome: ActionOutcome
    connector_id: str | None = None
    candidate_id: str | None = None
    query_sha256: str | None = None
    candidate_count: int | None = None
    reason: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        object.__setattr__(self, "action_type", ActionType(self.action_type))
        object.__setattr__(self, "outcome", ActionOutcome(self.outcome))
        for field_name in ("connector_id", "candidate_id", "reason", "error_code"):
            value = getattr(self, field_name)
            if value is not None:
                maximum = 512 if field_name == "reason" else 128
                object.__setattr__(
                    self,
                    field_name,
                    _text(value, field_name=field_name, max_length=maximum),
                )
        if self.query_sha256 is not None:
            digest = _text(self.query_sha256, field_name="query_sha256", max_length=80)
            if not digest.startswith("sha256:") or len(digest) != 71:
                raise ValueError("query_sha256 must be a sha256 digest")
            object.__setattr__(self, "query_sha256", digest)
        if self.candidate_count is not None:
            if (
                isinstance(self.candidate_count, bool)
                or not isinstance(self.candidate_count, int)
                or self.candidate_count < 0
            ):
                raise ValueError("candidate_count must be a non-negative integer")
        if self.action_type == ActionType.SEARCH:
            if self.connector_id is None or self.query_sha256 is None:
                raise ValueError("search transition requires connector_id and query_sha256")
        if self.action_type == ActionType.FETCH and self.candidate_id is None:
            raise ValueError("fetch transition requires candidate_id")
        if self.action_type == ActionType.REQUEST_STOP and self.reason is None:
            raise ValueError("stop transition requires a reason")

    @classmethod
    def for_action(
        cls,
        *,
        step: int,
        action: GeneralAction,
        outcome: ActionOutcome,
        candidate_count: int | None = None,
        error_code: str | None = None,
    ) -> "ActionTransition":
        if isinstance(action, SearchAction):
            return cls(
                step=step,
                action_type=action.action_type,
                outcome=outcome,
                connector_id=action.connector_id,
                query_sha256=_digest(action.query),
                candidate_count=candidate_count,
                error_code=error_code,
            )
        if isinstance(action, FetchAction):
            return cls(
                step=step,
                action_type=action.action_type,
                outcome=outcome,
                candidate_id=action.candidate_id,
                error_code=error_code,
            )
        return cls(
            step=step,
            action_type=action.action_type,
            outcome=outcome,
            reason=action.reason,
            error_code=error_code,
        )


class ActionGate:
    """Validate each action against declared connectors and observed candidates."""

    def __init__(
        self,
        connectors: Iterable[SourceConnectorConfig],
        *,
        authorized_connector_ids: Iterable[str] = (),
    ) -> None:
        connector_map: dict[str, SourceConnectorConfig] = {}
        for connector in connectors:
            if not isinstance(connector, SourceConnectorConfig):
                raise TypeError("connectors must contain SourceConnectorConfig values")
            if connector.connector_id in connector_map:
                raise ValueError("connectors must use unique connector_id values")
            connector_map[connector.connector_id] = connector
        if not connector_map:
            raise ValueError("ActionGate requires at least one connector")
        self._connectors = connector_map
        self._authorized_connector_ids = frozenset(
            _identifier(connector_id, field_name="authorized connector_id")
            for connector_id in authorized_connector_ids
        )

    def _connector(self, connector_id: str) -> SourceConnectorConfig:
        connector = self._connectors.get(connector_id)
        if connector is None:
            raise ActionGateError("action names a connector not enabled for this run")
        if (
            connector.requires_user_authorization
            and connector.connector_id not in self._authorized_connector_ids
        ):
            raise ActionGateError("connector is not authorized for this user/run")
        return connector

    def validate(
        self,
        action: GeneralAction,
        *,
        candidates: Mapping[str, SearchCandidate],
    ) -> None:
        """Accept a legal action or raise a stable, non-secret error code."""

        if isinstance(action, SearchAction):
            connector = self._connector(action.connector_id)
            if not connector.supports_search:
                raise ActionGateError("connector does not support search")
            return
        if isinstance(action, FetchAction):
            candidate = candidates.get(action.candidate_id)
            if candidate is None:
                raise ActionGateError("fetch requires an observed candidate_id")
            connector = self._connector(candidate.connector_id)
            if not connector.supports_fetch:
                raise ActionGateError("connector does not support fetch")
            if candidate.source_channel != connector.source_channel:
                raise ActionGateError("candidate channel does not match connector policy")
            return
        if isinstance(action, RequestStopAction):
            return
        raise TypeError("action must be a GeneralAction")


__all__ = [
    "ActionGate",
    "ActionGateError",
    "ActionOutcome",
    "ActionTransition",
    "ActionType",
    "FetchAction",
    "GeneralAction",
    "RequestStopAction",
    "SearchAction",
    "SearchCandidate",
]
