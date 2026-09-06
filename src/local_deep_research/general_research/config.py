"""Strict runtime configuration and provenance helpers for General Research.

``schemas.GeneralRunConfig`` is the single run-level contract. This module
adds runtime-only model, worker, tool, and citation settings around it; it
does not duplicate the run schema or perform environment lookup. Credentials,
provider endpoints, and implicit MCP discovery never enter an audit artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from .schemas import GeneralRunConfig, SourceChannel


GENERAL_EXECUTION_CONFIG_SCHEMA_VERSION = "general-execution-config/v1"
RESEARCH_CONTROL_POLICY_VERSION = "research-control-policy/v1"
_BASE_ROLE_NAMES = frozenset(
    {"planner", "controller", "evidence_extractor", "writer", "semantic_auditor"}
)
_AGENTIC_ROLE_NAMES = frozenset({"brief", "supervisor"})
_ROLE_NAMES = _BASE_ROLE_NAMES | _AGENTIC_ROLE_NAMES
_REQUIRED_GENERAL_TOOLS = frozenset({"search_sources", "fetch_source"})
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STRUCTURED_OUTPUT_MODES = frozenset({"prompted_json", "json_object"})


def _nonempty_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _strict_mapping(
    value: object, *, field_name: str, allowed_keys: frozenset[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    unknown = set(value) - allowed_keys
    if unknown:
        raise ValueError(
            f"{field_name} contains unsupported keys: {sorted(unknown)}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ModelRoleConfig:
    """A credential-free model identity for one explicit runtime role."""

    provider: str = "unconfigured"
    model: str = "unconfigured"
    temperature: float | None = None
    structured_output_mode: str = "prompted_json"

    def __post_init__(self) -> None:
        _nonempty_text(self.provider, field_name="model provider")
        _nonempty_text(self.model, field_name="model")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.structured_output_mode not in _STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                "structured_output_mode must be prompted_json or json_object"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelRoleConfig":
        data = _strict_mapping(
            value,
            field_name="model role",
            allowed_keys=frozenset(
                {"provider", "model", "temperature", "structured_output_mode"}
            ),
        )
        return cls(
            provider=data.get("provider", "unconfigured"),
            model=data.get("model", "unconfigured"),
            temperature=data.get("temperature"),
            structured_output_mode=data.get("structured_output_mode", "prompted_json"),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "structured_output_mode": self.structured_output_mode,
        }


@dataclass(frozen=True, slots=True)
class SourceConnectorConfig:
    """One source adapter exposed to an individual General run.

    Connector declarations are capability records, not evidence-quality
    claims. Runtime authorization is checked separately for connectors that
    may expose user-local material, so a model never gains collection access
    merely because it emitted a connector name.
    """

    connector_id: str
    source_channel: SourceChannel
    supports_search: bool = True
    supports_fetch: bool = True
    requires_user_authorization: bool = False

    def __post_init__(self) -> None:
        connector_id = _nonempty_text(self.connector_id, field_name="connector_id")
        if not _TOOL_NAME_RE.fullmatch(connector_id):
            raise ValueError("connector_id must use lowercase identifier syntax")
        object.__setattr__(self, "connector_id", connector_id)
        object.__setattr__(self, "source_channel", SourceChannel(self.source_channel))
        for field_name in (
            "supports_search",
            "supports_fetch",
            "requires_user_authorization",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if not self.supports_search and not self.supports_fetch:
            raise ValueError("a source connector must support search or fetch")
        if (
            self.source_channel == SourceChannel.LOCAL_COLLECTION
            and not self.requires_user_authorization
        ):
            raise ValueError("local_collection connectors require user authorization")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceConnectorConfig":
        data = _strict_mapping(
            value,
            field_name="source connector",
            allowed_keys=frozenset(
                {
                    "connector_id",
                    "source_channel",
                    "supports_search",
                    "supports_fetch",
                    "requires_user_authorization",
                }
            ),
        )
        if "connector_id" not in data or "source_channel" not in data:
            raise ValueError("source connector requires connector_id and source_channel")
        return cls(**dict(data))

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "source_channel": self.source_channel.value,
            "supports_search": self.supports_search,
            "supports_fetch": self.supports_fetch,
            "requires_user_authorization": self.requires_user_authorization,
        }


def _default_source_connectors() -> tuple[SourceConnectorConfig, ...]:
    return (
        SourceConnectorConfig(
            connector_id="public_web",
            source_channel=SourceChannel.PUBLIC_WEB,
        ),
    )


@dataclass(frozen=True, slots=True)
class ResearchControlPolicy:
    """Frozen, runtime-owned limits and acceptance floors for General V1.

    The planner may decide *which* atomic questions are worth investigating.
    It must not decide what counts as enough evidence, loosen source quality,
    or enlarge its own context and execution budget.  Those decisions are
    operational policy and are captured in the execution-config digest.
    """

    version: str = RESEARCH_CONTROL_POLICY_VERSION
    # Six is still bounded by the two-pass, three-worker topology, while
    # letting a comparative request keep its explicitly named alternatives
    # visible to the supervisor instead of collapsing them into one generic
    # "vendor comparison" item.
    max_plan_items: int = 6
    min_evidence_cards_per_item: int = 1
    min_distinct_source_groups_per_item: int = 1
    min_source_quality_score: int = 40
    max_candidates_per_search: int = 8
    max_total_candidates: int = 16
    max_fetched_characters: int = 1_000_000
    max_research_passes: int = 2

    def __post_init__(self) -> None:
        if self.version != RESEARCH_CONTROL_POLICY_VERSION:
            raise ValueError(
                f"unsupported research control policy: {self.version!r}"
            )
        for field_name in (
            "max_plan_items",
            "min_evidence_cards_per_item",
            "min_distinct_source_groups_per_item",
            "min_source_quality_score",
            "max_candidates_per_search",
            "max_total_candidates",
            "max_fetched_characters",
            "max_research_passes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if not 1 <= self.max_plan_items <= 6:
            raise ValueError("max_plan_items must be between 1 and 6")
        if not 1 <= self.min_evidence_cards_per_item <= 3:
            raise ValueError("min_evidence_cards_per_item must be between 1 and 3")
        if not 1 <= self.min_distinct_source_groups_per_item <= 3:
            raise ValueError(
                "min_distinct_source_groups_per_item must be between 1 and 3"
            )
        # 40 is the score of a verified unknown source (or verified local
        # collection item) under source-policy/v1.  Lower settings could turn
        # a mere discovery snippet or unverified source into a STOP condition.
        if not 40 <= self.min_source_quality_score <= 100:
            raise ValueError("min_source_quality_score must be between 40 and 100")
        if not 1 <= self.max_candidates_per_search <= 10:
            raise ValueError("max_candidates_per_search must be between 1 and 10")
        if not self.max_candidates_per_search <= self.max_total_candidates <= 24:
            raise ValueError(
                "max_total_candidates must be at least max_candidates_per_search "
                "and at most 24"
            )
        if not 10_000 <= self.max_fetched_characters <= 2_000_000:
            raise ValueError(
                "max_fetched_characters must be between 10,000 and 2,000,000"
            )
        if not 1 <= self.max_research_passes <= 3:
            raise ValueError("max_research_passes must be between 1 and 3")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResearchControlPolicy":
        data = _strict_mapping(
            value,
            field_name="research_control",
            allowed_keys=frozenset(
                {
                    "version",
                    "max_plan_items",
                    "min_evidence_cards_per_item",
                    "min_distinct_source_groups_per_item",
                    "min_source_quality_score",
                    "max_candidates_per_search",
                    "max_total_candidates",
                    "max_fetched_characters",
                    "max_research_passes",
                }
            ),
        )
        return cls(**dict(data))

    def canonical_dict(self) -> dict[str, int | str]:
        return {
            "version": self.version,
            "max_plan_items": self.max_plan_items,
            "min_evidence_cards_per_item": self.min_evidence_cards_per_item,
            "min_distinct_source_groups_per_item": self.min_distinct_source_groups_per_item,
            "min_source_quality_score": self.min_source_quality_score,
            "max_candidates_per_search": self.max_candidates_per_search,
            "max_total_candidates": self.max_total_candidates,
            "max_fetched_characters": self.max_fetched_characters,
            "max_research_passes": self.max_research_passes,
        }


@dataclass(frozen=True, slots=True)
class EvidenceExtractionPolicy:
    """Deterministic text-window limits recorded in every General run.

    Character offsets are used instead of provider-token offsets because they
    can be verified against the exact UTF-8 text snapshot without relying on a
    model-specific tokenizer. The values are calibration parameters, but their
    complete policy is part of the execution-config digest.
    """

    max_chunk_characters: int = 6_000
    chunk_overlap_characters: int = 400
    max_chunks_per_source: int = 3
    max_span_characters: int = 1_000

    def __post_init__(self) -> None:
        for field_name in (
            "max_chunk_characters",
            "chunk_overlap_characters",
            "max_chunks_per_source",
            "max_span_characters",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if not 1_000 <= self.max_chunk_characters <= 16_000:
            raise ValueError("max_chunk_characters must be between 1,000 and 16,000")
        if not 0 <= self.chunk_overlap_characters < self.max_chunk_characters:
            raise ValueError("chunk_overlap_characters must be below chunk length")
        if not 1 <= self.max_chunks_per_source <= 10:
            raise ValueError("max_chunks_per_source must be between 1 and 10")
        if not 256 <= self.max_span_characters <= self.max_chunk_characters:
            raise ValueError(
                "max_span_characters must be between 256 and max_chunk_characters"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceExtractionPolicy":
        data = _strict_mapping(
            value,
            field_name="evidence_extraction",
            allowed_keys=frozenset(
                {
                    "max_chunk_characters",
                    "chunk_overlap_characters",
                    "max_chunks_per_source",
                    "max_span_characters",
                }
            ),
        )
        return cls(**dict(data))

    def canonical_dict(self) -> dict[str, int]:
        return {
            "max_chunk_characters": self.max_chunk_characters,
            "chunk_overlap_characters": self.chunk_overlap_characters,
            "max_chunks_per_source": self.max_chunks_per_source,
            "max_span_characters": self.max_span_characters,
        }


@dataclass(frozen=True, slots=True)
class WorkerToolPolicy:
    """Explicit worker and built-in tool surface; V1 is single-agent by default."""

    max_research_workers: int = 1
    max_parallel_tool_calls: int = 1
    allow_subagents: bool = False
    tool_allowlist: frozenset[str] = field(
        default_factory=lambda: frozenset({"search_sources", "fetch_source"})
    )
    allow_external_mcp: bool = False
    external_mcp_servers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_research_workers, int)
            or isinstance(self.max_research_workers, bool)
            or self.max_research_workers < 1
        ):
            raise ValueError("max_research_workers must be a positive integer")
        if (
            not isinstance(self.max_parallel_tool_calls, int)
            or isinstance(self.max_parallel_tool_calls, bool)
            or self.max_parallel_tool_calls < 1
        ):
            raise ValueError("max_parallel_tool_calls must be a positive integer")
        if not isinstance(self.allow_subagents, bool):
            raise TypeError("allow_subagents must be a boolean")
        if not isinstance(self.allow_external_mcp, bool):
            raise TypeError("allow_external_mcp must be a boolean")
        if isinstance(self.tool_allowlist, (str, bytes)):
            raise TypeError("tool_allowlist must be an iterable of tool names")
        if isinstance(self.external_mcp_servers, (str, bytes)):
            raise TypeError(
                "external_mcp_servers must be an iterable of server names"
            )

        normalized_tools = frozenset(
            _nonempty_text(name, field_name="tool allowlist entry")
            for name in self.tool_allowlist
        )
        if not normalized_tools:
            raise ValueError("tool_allowlist cannot be empty")
        invalid_names = sorted(
            name for name in normalized_tools if not _TOOL_NAME_RE.fullmatch(name)
        )
        if invalid_names:
            raise ValueError(f"invalid tool names: {invalid_names}")
        if not _REQUIRED_GENERAL_TOOLS.issubset(normalized_tools):
            raise ValueError(
                "General Research requires search_sources and fetch_source in "
                "tool_allowlist"
            )
        if "research_subtopic" in normalized_tools and not self.allow_subagents:
            raise ValueError(
                "research_subtopic requires allow_subagents=True"
            )
        if not self.allow_subagents and self.max_research_workers != 1:
            raise ValueError(
                "max_research_workers must be 1 when subagents are disabled"
            )
        if self.allow_subagents and self.max_research_workers < 2:
            raise ValueError(
                "allow_subagents=True requires at least two research workers"
            )

        normalized_servers = tuple(
            sorted(
                {
                    _nonempty_text(server, field_name="external MCP server")
                    for server in self.external_mcp_servers
                }
            )
        )
        if normalized_servers and not self.allow_external_mcp:
            raise ValueError(
                "external MCP servers require allow_external_mcp=True"
            )
        if self.allow_external_mcp and not normalized_servers:
            raise ValueError(
                "allow_external_mcp=True requires explicit external_mcp_servers"
            )
        # V1 does not treat arbitrary strings as executable MCP tools. A later
        # adapter must map an explicitly approved server to a named capability.
        if any("mcp" in name for name in normalized_tools):
            raise ValueError(
                "MCP tools are not valid General V1 tool_allowlist entries"
            )

        object.__setattr__(self, "tool_allowlist", normalized_tools)
        object.__setattr__(self, "external_mcp_servers", normalized_servers)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkerToolPolicy":
        data = _strict_mapping(
            value,
            field_name="workers",
            allowed_keys=frozenset(
                {
                    "max_research_workers",
                    "max_parallel_tool_calls",
                    "allow_subagents",
                    "tool_allowlist",
                    "allow_external_mcp",
                    "external_mcp_servers",
                }
            ),
        )
        normalized = dict(data)
        if "tool_allowlist" in normalized:
            if isinstance(normalized["tool_allowlist"], (str, bytes)):
                raise TypeError("tool_allowlist must be an iterable of tool names")
            normalized["tool_allowlist"] = frozenset(normalized["tool_allowlist"])
        if "external_mcp_servers" in normalized:
            if isinstance(normalized["external_mcp_servers"], (str, bytes)):
                raise TypeError(
                    "external_mcp_servers must be an iterable of server names"
                )
            normalized["external_mcp_servers"] = tuple(
                normalized["external_mcp_servers"]
            )
        return cls(**normalized)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "max_research_workers": self.max_research_workers,
            "max_parallel_tool_calls": self.max_parallel_tool_calls,
            "allow_subagents": self.allow_subagents,
            "tool_allowlist": sorted(self.tool_allowlist),
            "allow_external_mcp": self.allow_external_mcp,
            "external_mcp_servers": list(self.external_mcp_servers),
        }


@dataclass(frozen=True, slots=True)
class CitationAuditPolicy:
    """Provenance audit and optional model-semantic review policy.

    Source locators and verified quotes are deterministic publication
    requirements.  The post-synthesis semantic model review is an optional
    diagnostic: it should not turn an otherwise traceable benchmark report
    into a blank result merely because a second model disagrees.
    """

    version: str = "citation-policy/v1"
    require_source_locator: bool = True
    require_quote_or_span: bool = True
    require_post_synthesis_audit: bool = False
    fail_on_unsupported_claim: bool = False
    max_writer_repairs: int = 1

    def __post_init__(self) -> None:
        _nonempty_text(self.version, field_name="citation policy version")
        if not self.require_source_locator:
            raise ValueError("General V1 requires a source locator per citation")
        if not self.require_quote_or_span:
            raise ValueError("General V1 requires a quote or span per citation")
        if self.fail_on_unsupported_claim and not self.require_post_synthesis_audit:
            raise ValueError(
                "fail_on_unsupported_claim requires require_post_synthesis_audit"
            )
        if (
            isinstance(self.max_writer_repairs, bool)
            or not isinstance(self.max_writer_repairs, int)
            or not 0 <= self.max_writer_repairs <= 1
        ):
            raise ValueError("General V1 permits zero or one writer repair")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CitationAuditPolicy":
        data = _strict_mapping(
            value,
            field_name="citation_policy",
            allowed_keys=frozenset(
                {
                    "version",
                    "require_source_locator",
                    "require_quote_or_span",
                    "require_post_synthesis_audit",
                    "fail_on_unsupported_claim",
                    "max_writer_repairs",
                }
            ),
        )
        return cls(**dict(data))

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "require_source_locator": self.require_source_locator,
            "require_quote_or_span": self.require_quote_or_span,
            "require_post_synthesis_audit": self.require_post_synthesis_audit,
            "fail_on_unsupported_claim": self.fail_on_unsupported_claim,
            "max_writer_repairs": self.max_writer_repairs,
        }


@dataclass(frozen=True, slots=True)
class AgenticOrchestrationPolicy:
    """Frozen execution tier for the supervisor/parallel-worker topology.

    The tier values deliberately follow the public quick/standard/deep
    progression used by ZenML's deep-research engine: 2, 5, and 10 supervisor
    rounds.  General's ``standard`` tier uses three concurrent workers, the
    same fan-out that upstream uses for non-exhaustive tiers.  These are
    execution capacities, not evidence-quality thresholds; the model cannot
    alter them during a run.
    """

    version: str = "agentic-orchestration-policy/v1"
    tier: str = "standard"
    max_supervisor_rounds: int = 5
    max_parallel_workers: int = 3

    _TIER_VALUES = {
        "quick": (2, 3),
        "standard": (5, 3),
        "deep": (10, 3),
    }

    def __post_init__(self) -> None:
        if self.version != "agentic-orchestration-policy/v1":
            raise ValueError(f"unsupported agentic orchestration policy: {self.version!r}")
        expected = self._TIER_VALUES.get(self.tier)
        if expected is None:
            raise ValueError("tier must be quick, standard, or deep")
        if (
            isinstance(self.max_supervisor_rounds, bool)
            or isinstance(self.max_parallel_workers, bool)
            or not isinstance(self.max_supervisor_rounds, int)
            or not isinstance(self.max_parallel_workers, int)
        ):
            raise TypeError("agentic orchestration capacities must be integers")
        if (self.max_supervisor_rounds, self.max_parallel_workers) != expected:
            raise ValueError(
                "agentic orchestration capacities must match the frozen tier definition"
            )

    @classmethod
    def for_tier(cls, tier: str = "standard") -> "AgenticOrchestrationPolicy":
        normalized = _nonempty_text(tier, field_name="agentic orchestration tier")
        expected = cls._TIER_VALUES.get(normalized)
        if expected is None:
            raise ValueError("tier must be quick, standard, or deep")
        return cls(
            tier=normalized,
            max_supervisor_rounds=expected[0],
            max_parallel_workers=expected[1],
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgenticOrchestrationPolicy":
        data = _strict_mapping(
            value,
            field_name="agentic orchestration",
            allowed_keys=frozenset(
                {"version", "tier", "max_supervisor_rounds", "max_parallel_workers"}
            ),
        )
        tier = data.get("tier", "standard")
        if (
            "max_supervisor_rounds" not in data
            and "max_parallel_workers" not in data
        ):
            return cls.for_tier(tier)
        if "max_supervisor_rounds" not in data or "max_parallel_workers" not in data:
            raise ValueError(
                "agentic orchestration capacity overrides must be specified together"
            )
        return cls(**dict(data))

    def canonical_dict(self) -> dict[str, int | str]:
        return {
            "version": self.version,
            "tier": self.tier,
            "max_supervisor_rounds": self.max_supervisor_rounds,
            "max_parallel_workers": self.max_parallel_workers,
        }


def _default_model_roles(
    *, include_semantic_auditor: bool = False
) -> tuple[tuple[str, ModelRoleConfig], ...]:
    roles: tuple[tuple[str, ModelRoleConfig], ...] = (
        ("planner", ModelRoleConfig()),
        ("controller", ModelRoleConfig()),
        ("evidence_extractor", ModelRoleConfig()),
        ("writer", ModelRoleConfig()),
    )
    if include_semantic_auditor:
        return roles + (("semantic_auditor", ModelRoleConfig()),)
    return roles


@dataclass(frozen=True, slots=True)
class GeneralExecutionConfig:
    """Complete config linked to one immutable ``schemas.GeneralRunConfig``."""

    run: GeneralRunConfig
    model_roles: tuple[tuple[str, ModelRoleConfig], ...] = field(
        default_factory=_default_model_roles
    )
    workers: WorkerToolPolicy = field(default_factory=WorkerToolPolicy)
    source_connectors: tuple[SourceConnectorConfig, ...] = field(
        default_factory=_default_source_connectors
    )
    research_control: ResearchControlPolicy = field(
        default_factory=ResearchControlPolicy
    )
    evidence_extraction: EvidenceExtractionPolicy = field(
        default_factory=EvidenceExtractionPolicy
    )
    citation_policy: CitationAuditPolicy = field(default_factory=CitationAuditPolicy)
    orchestration: AgenticOrchestrationPolicy | None = None
    schema_version: str = GENERAL_EXECUTION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERAL_EXECUTION_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported execution config schema: {self.schema_version!r}"
            )
        if not isinstance(self.run, GeneralRunConfig):
            raise TypeError("run must be schemas.GeneralRunConfig")
        if not isinstance(self.workers, WorkerToolPolicy):
            raise TypeError("workers must be WorkerToolPolicy")
        connectors = tuple(self.source_connectors)
        if not connectors:
            raise ValueError("General V1 requires at least one source connector")
        if not all(
            isinstance(connector, SourceConnectorConfig)
            for connector in connectors
        ):
            raise TypeError("source_connectors must contain SourceConnectorConfig values")
        connector_ids = [connector.connector_id for connector in connectors]
        if len(connector_ids) != len(set(connector_ids)):
            raise ValueError("source_connectors must use unique connector_id values")
        if not isinstance(self.research_control, ResearchControlPolicy):
            raise TypeError("research_control must be ResearchControlPolicy")
        if not isinstance(self.evidence_extraction, EvidenceExtractionPolicy):
            raise TypeError("evidence_extraction must be EvidenceExtractionPolicy")
        if not isinstance(self.citation_policy, CitationAuditPolicy):
            raise TypeError("citation_policy must be CitationAuditPolicy")
        if self.orchestration is not None and not isinstance(
            self.orchestration, AgenticOrchestrationPolicy
        ):
            raise TypeError("orchestration must be AgenticOrchestrationPolicy or None")
        normalized_roles: list[tuple[str, ModelRoleConfig]] = []
        seen_roles: set[str] = set()
        for role, model_config in self.model_roles:
            role = _nonempty_text(role, field_name="model role")
            if role not in _ROLE_NAMES:
                raise ValueError(f"unsupported General V1 model role: {role}")
            if role in seen_roles:
                raise ValueError(f"duplicate model role: {role}")
            if not isinstance(model_config, ModelRoleConfig):
                raise TypeError(f"model role {role} must use ModelRoleConfig")
            seen_roles.add(role)
            normalized_roles.append((role, model_config))
        required_roles = (
            _BASE_ROLE_NAMES
            if self.citation_policy.require_post_synthesis_audit
            else _BASE_ROLE_NAMES - {"semantic_auditor"}
        ) | (
            _AGENTIC_ROLE_NAMES if self.orchestration is not None else frozenset()
        )
        missing_roles = required_roles - seen_roles
        if missing_roles:
            raise ValueError(
                f"General V1 requires model roles: {sorted(missing_roles)}"
            )
        inactive_roles = seen_roles - required_roles
        if inactive_roles:
            raise ValueError(
                "agentic model roles require an explicit agentic orchestration policy: "
                f"{sorted(inactive_roles)}"
            )
        if self.orchestration is None and (
            self.workers.allow_subagents
            or self.workers.max_research_workers != 1
            or self.run.max_parallel_subagents != 0
        ):
            raise ValueError(
                "multi-worker settings require an explicit agentic orchestration policy"
            )
        if self.orchestration is not None and (
            not self.workers.allow_subagents
            or self.workers.max_research_workers
            != self.orchestration.max_parallel_workers
            or self.run.max_parallel_subagents
            != self.orchestration.max_parallel_workers
        ):
            raise ValueError(
                "agentic orchestration requires matching run and worker concurrency"
            )
        if self.workers.max_parallel_tool_calls != 1:
            raise ValueError("each General worker executes one tool call at a time")
        if self.workers.allow_external_mcp or self.workers.external_mcp_servers:
            raise ValueError("General V1 does not permit external MCP")
        object.__setattr__(self, "model_roles", tuple(sorted(normalized_roles)))
        object.__setattr__(
            self,
            "source_connectors",
            tuple(
                sorted(connectors, key=lambda connector: connector.connector_id)
            ),
        )

    @property
    def models_by_role(self) -> dict[str, ModelRoleConfig]:
        """Return a defensive mapping suitable for runtime construction."""
        return dict(self.model_roles)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GeneralExecutionConfig":
        data = _strict_mapping(
            value,
            field_name="general execution config",
            allowed_keys=frozenset(
                {
                    "schema_version",
                    "run",
                    "models",
                    "workers",
                    "source_connectors",
                    "research_control",
                    "evidence_extraction",
                    "citation_policy",
                    "orchestration",
                }
            ),
        )
        if "run" not in data:
            raise ValueError("general execution config requires a run mapping")
        raw_run = _strict_mapping(
            data["run"],
            field_name="run",
            allowed_keys=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "query",
                    "created_at",
                    "profile",
                    "max_model_calls",
                    "max_tool_calls",
                    "max_parallel_subagents",
                    "evidence_only_synthesis",
                    "source_policy_version",
                    "coverage_policy_version",
                    "allowed_url_schemes",
                }
            ),
        )
        citation_policy = (
            CitationAuditPolicy.from_mapping(data["citation_policy"])
            if "citation_policy" in data
            else CitationAuditPolicy()
        )
        raw_models = data.get("models")
        if raw_models is None:
            model_roles = _default_model_roles(
                include_semantic_auditor=citation_policy.require_post_synthesis_audit
            )
        else:
            model_mapping = _strict_mapping(
                raw_models,
                field_name="models",
                allowed_keys=_ROLE_NAMES,
            )
            model_roles = tuple(
                (role, ModelRoleConfig.from_mapping(raw_model))
                for role, raw_model in model_mapping.items()
            )
        raw_connectors = data.get("source_connectors")
        if raw_connectors is None:
            source_connectors = _default_source_connectors()
        else:
            if not isinstance(raw_connectors, list):
                raise TypeError("source_connectors must be a JSON list")
            source_connectors = tuple(
                SourceConnectorConfig.from_mapping(connector)
                for connector in raw_connectors
            )
        evidence_extraction = EvidenceExtractionPolicy.from_mapping(
            data["evidence_extraction"]
        ) if "evidence_extraction" in data else EvidenceExtractionPolicy()
        return cls(
            schema_version=data.get(
                "schema_version", GENERAL_EXECUTION_CONFIG_SCHEMA_VERSION
            ),
            run=GeneralRunConfig(**dict(raw_run)),
            model_roles=model_roles,
            workers=WorkerToolPolicy.from_mapping(data["workers"])
            if "workers" in data
            else WorkerToolPolicy(),
            source_connectors=source_connectors,
            research_control=ResearchControlPolicy.from_mapping(
                data["research_control"]
            ) if "research_control" in data else ResearchControlPolicy(),
            evidence_extraction=evidence_extraction,
            citation_policy=citation_policy,
            orchestration=(
                AgenticOrchestrationPolicy.from_mapping(data["orchestration"])
                if data.get("orchestration") is not None
                else None
            ),
        )

    def canonical_dict(self) -> dict[str, Any]:
        """Return a fresh deterministic representation with no secret fields."""
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "models": {
                role: model.canonical_dict() for role, model in self.model_roles
            },
            "workers": self.workers.canonical_dict(),
            "source_connectors": [
                connector.canonical_dict() for connector in self.source_connectors
            ],
            "research_control": self.research_control.canonical_dict(),
            "evidence_extraction": self.evidence_extraction.canonical_dict(),
            "citation_policy": self.citation_policy.canonical_dict(),
            "orchestration": (
                self.orchestration.canonical_dict()
                if self.orchestration is not None
                else None
            ),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def trace_metadata(self) -> dict[str, Any]:
        """Metadata safe to pass to ``TraceRecorder(..., metadata=...)``."""
        return {
            "general_research": {
                "execution_config": self.canonical_dict(),
                "config_digest_sha256": self.digest(),
                "research_control_policy_version": self.research_control.version,
                "source_policy_version": self.run.source_policy_version,
                "coverage_policy_version": self.run.coverage_policy_version,
                "citation_policy_version": self.citation_policy.version,
            }
        }
