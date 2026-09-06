"""Immutable runtime provenance for one General Research Agent run.

The execution configuration says what the agent was *allowed* to do.  This
manifest says which non-secret runtime implementations and data snapshots it
actually used.  Keeping them separate is intentional: the same frozen
research policy can be evaluated with different search backends or collection
snapshots without pretending those runs are directly comparable.

This module adopts the useful shape of mature research-engine ``package`` /
``manifest`` artifacts, but is written for General's evidence-card protocol:
it never stores credentials, raw provider endpoints, or an unverifiable claim
that a live web run can be replayed bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from .config import GeneralExecutionConfig
from .schemas import SourceChannel


GENERAL_RUN_MANIFEST_SCHEMA_VERSION = "general-run-manifest/v1"
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")
_PURPOSES = frozenset({"library", "product", "benchmark"})


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name=field_name)


def _identifier(value: object, *, field_name: str) -> str:
    result = _text(value, field_name=field_name)
    if not _IDENTIFIER_RE.fullmatch(result):
        raise ValueError(f"{field_name} must use safe identifier syntax")
    return result


def _strict_mapping(
    value: object, *, field_name: str, allowed_keys: frozenset[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    unknown = set(value).difference(allowed_keys)
    if unknown:
        raise ValueError(f"{field_name} has unsupported keys: {sorted(unknown)}")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeConnectorManifest:
    """Credential-free identity of the implementation behind one connector.

    ``data_snapshot_fingerprint`` is optional for the public live web and
    required by benchmark policy only when an operator elects to evaluate a
    local corpus.  It is a caller-provided opaque fingerprint, never a local
    path or a listing of collection documents.
    """

    connector_id: str
    source_channel: SourceChannel
    implementation_id: str
    data_snapshot_fingerprint: str | None = None
    request_timeout_seconds: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "connector_id", _identifier(self.connector_id, field_name="connector_id")
        )
        object.__setattr__(self, "source_channel", SourceChannel(self.source_channel))
        object.__setattr__(
            self,
            "implementation_id",
            _identifier(self.implementation_id, field_name="implementation_id"),
        )
        object.__setattr__(
            self,
            "data_snapshot_fingerprint",
            _optional_text(
                self.data_snapshot_fingerprint,
                field_name="data_snapshot_fingerprint",
            ),
        )
        if self.request_timeout_seconds is not None:
            timeout = self.request_timeout_seconds
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, int)
                or not 1 <= timeout <= 300
            ):
                raise ValueError(
                    "request_timeout_seconds must be an integer between 1 and 300"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConnectorManifest":
        data = _strict_mapping(
            value,
            field_name="runtime connector",
            allowed_keys=frozenset(
                {
                    "connector_id",
                    "source_channel",
                    "implementation_id",
                    "data_snapshot_fingerprint",
                    "request_timeout_seconds",
                }
            ),
        )
        required = {"connector_id", "source_channel", "implementation_id"}
        missing = required.difference(data)
        if missing:
            raise ValueError("runtime connector is missing: " + ", ".join(sorted(missing)))
        return cls(**dict(data))

    def to_dict(self) -> dict[str, object]:
        return {
            "connector_id": self.connector_id,
            "source_channel": self.source_channel.value,
            "implementation_id": self.implementation_id,
            "data_snapshot_fingerprint": self.data_snapshot_fingerprint,
            "request_timeout_seconds": self.request_timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class GeneralRunManifest:
    """Runtime provenance strongly bound to one frozen execution config.

    A manifest makes the run *auditable*, not deterministically repeatable.
    Live search ranking and model sampling can change.  The saved source
    snapshots and evidence cards are what can be replayed offline.
    """

    run_id: str
    execution_config_digest_sha256: str
    created_at: str
    code_revision: str
    execution_purpose: str
    connectors: tuple[RuntimeConnectorManifest, ...]
    schema_version: str = GENERAL_RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERAL_RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported run manifest schema: {self.schema_version!r}")
        object.__setattr__(self, "run_id", _identifier(self.run_id, field_name="run_id"))
        digest = _text(
            self.execution_config_digest_sha256,
            field_name="execution_config_digest_sha256",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("execution_config_digest_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "execution_config_digest_sha256", digest)
        object.__setattr__(self, "created_at", _text(self.created_at, field_name="created_at"))
        object.__setattr__(self, "code_revision", _identifier(self.code_revision, field_name="code_revision"))
        purpose = _text(self.execution_purpose, field_name="execution_purpose")
        if purpose not in _PURPOSES:
            raise ValueError("execution_purpose must be library, product, or benchmark")
        object.__setattr__(self, "execution_purpose", purpose)
        connectors = tuple(self.connectors)
        if not connectors or not all(
            isinstance(connector, RuntimeConnectorManifest) for connector in connectors
        ):
            raise TypeError("connectors must contain runtime connector manifests")
        connector_ids = [connector.connector_id for connector in connectors]
        if len(connector_ids) != len(set(connector_ids)):
            raise ValueError("runtime connector IDs must be unique")
        object.__setattr__(
            self, "connectors", tuple(sorted(connectors, key=lambda connector: connector.connector_id))
        )

    @classmethod
    def for_config(
        cls,
        config: GeneralExecutionConfig,
        *,
        code_revision: str = "unrecorded",
        execution_purpose: str = "library",
        implementations_by_connector: Mapping[str, str] | None = None,
        snapshot_fingerprints_by_connector: Mapping[str, str] | None = None,
        request_timeouts_by_connector: Mapping[str, int] | None = None,
    ) -> "GeneralRunManifest":
        """Build a minimal explicit manifest without inferring hidden runtime state."""

        if not isinstance(config, GeneralExecutionConfig):
            raise TypeError("config must be GeneralExecutionConfig")
        implementations = dict(implementations_by_connector or {})
        fingerprints = dict(snapshot_fingerprints_by_connector or {})
        timeouts = dict(request_timeouts_by_connector or {})
        supplied_connectors = set(implementations) | set(fingerprints) | set(timeouts)
        unknown = supplied_connectors.difference(
            connector.connector_id for connector in config.source_connectors
        )
        if unknown:
            raise ValueError("runtime manifest names undeclared connectors: " + ", ".join(sorted(unknown)))
        return cls(
            run_id=config.run.run_id,
            execution_config_digest_sha256=config.digest(),
            created_at=config.run.created_at,
            code_revision=code_revision,
            execution_purpose=execution_purpose,
            connectors=tuple(
                RuntimeConnectorManifest(
                    connector_id=connector.connector_id,
                    source_channel=connector.source_channel,
                    implementation_id=implementations.get(
                        connector.connector_id, "unspecified"
                    ),
                    data_snapshot_fingerprint=fingerprints.get(connector.connector_id),
                    request_timeout_seconds=timeouts.get(connector.connector_id),
                )
                for connector in config.source_connectors
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GeneralRunManifest":
        data = _strict_mapping(
            value,
            field_name="general run manifest",
            allowed_keys=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "execution_config_digest_sha256",
                    "created_at",
                    "code_revision",
                    "execution_purpose",
                    "connectors",
                }
            ),
        )
        required = {
            "run_id",
            "execution_config_digest_sha256",
            "created_at",
            "code_revision",
            "execution_purpose",
            "connectors",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError("general run manifest is missing: " + ", ".join(sorted(missing)))
        raw_connectors = data["connectors"]
        if not isinstance(raw_connectors, list):
            raise TypeError("manifest connectors must be a list")
        return cls(
            schema_version=data.get("schema_version", GENERAL_RUN_MANIFEST_SCHEMA_VERSION),
            run_id=data["run_id"],
            execution_config_digest_sha256=data["execution_config_digest_sha256"],
            created_at=data["created_at"],
            code_revision=data["code_revision"],
            execution_purpose=data["execution_purpose"],
            connectors=tuple(
                RuntimeConnectorManifest.from_mapping(connector)
                for connector in raw_connectors
            ),
        )

    def validate_for_config(self, config: GeneralExecutionConfig) -> None:
        """Reject a manifest that differs from the frozen config capability set."""

        if not isinstance(config, GeneralExecutionConfig):
            raise TypeError("config must be GeneralExecutionConfig")
        if self.run_id != config.run.run_id:
            raise ValueError("manifest run_id differs from execution config")
        if self.execution_config_digest_sha256 != config.digest():
            raise ValueError("manifest digest differs from execution config")
        expected = {
            (connector.connector_id, connector.source_channel.value)
            for connector in config.source_connectors
        }
        observed = {
            (connector.connector_id, connector.source_channel.value)
            for connector in self.connectors
        }
        if observed != expected:
            raise ValueError("manifest connectors differ from execution config")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "execution_config_digest_sha256": self.execution_config_digest_sha256,
            "created_at": self.created_at,
            "code_revision": self.code_revision,
            "execution_purpose": self.execution_purpose,
            "connectors": [connector.to_dict() for connector in self.connectors],
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_dict())


__all__ = [
    "GENERAL_RUN_MANIFEST_SCHEMA_VERSION",
    "GeneralRunManifest",
    "RuntimeConnectorManifest",
]
