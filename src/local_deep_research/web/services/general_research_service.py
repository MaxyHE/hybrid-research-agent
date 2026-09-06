"""Product adapter for the independent General Research Agent V1.

This module deliberately sits *between* the existing Flask research lifecycle
and :mod:`local_deep_research.general_research`.  It reuses the application's
configured chat model, public search engine, egress policy and report/source
persistence.  It does not invoke ``AdvancedSearchSystem`` or a Hybrid
strategy, and it never turns a failed General evidence gate into legacy model
prose.

The adapter is intentionally small.  The General package owns all research
decisions and evidence artifacts; this module owns only application settings,
credential-free execution metadata, and conversion of verified public sources
to the application's existing resource-store shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from uuid import UUID

from ...general_research.collection_connector import (
    CollectionDescriptor,
    project_local_collection_connector,
)
from ...general_research.config import (
    AgenticOrchestrationPolicy,
    CitationAuditPolicy,
    EvidenceExtractionPolicy,
    GeneralExecutionConfig,
    ModelRoleConfig,
    SourceConnectorConfig,
    WorkerToolPolicy,
)
from ...general_research.connectors import ConnectorBinding, ConnectorRegistry
from ...general_research.model_adapters import LangChainJsonGateway
from ...general_research.runner import GeneralResearchRunner, GeneralRunArtifacts
from ...general_research.run_manifest import GeneralRunManifest
from ...general_research.schemas import GeneralRunConfig, SourceChannel, SourceRecord
from ...general_research.web_connector import project_public_web_connector
from ...general_research.trace import GeneralAuditEvent, GeneralAuditEventKind
from ...general_research.workflow import GeneralWorkflowResult
from ...general_research.writer import WriterInput, render_report_document
from ...general_research.usage import ModelUsageLedger


# Standard agentic runs reserve extraction and synthesis budget before any
# worker fan-out. Twenty model calls permit a substantive first dispatch plus
# bounded recovery, while remaining a development default rather than a
# performance claim. Formal evaluation freezes its own envelope.
GENERAL_WEB_DEFAULT_MAX_MODEL_CALLS = 20
# Three workers can each search and fetch once (six calls); the remaining six
# permit recovery or a follow-up round without an unbounded retry loop.
GENERAL_WEB_DEFAULT_MAX_TOOL_CALLS = 12
_MAX_GENERAL_WEB_BUDGET = 64
GENERAL_WEB_DEFAULT_FETCH_TIMEOUT_SECONDS = 30
_GENERAL_ROLE_NAMES = (
    "brief",
    "planner",
    "supervisor",
    "controller",
    "evidence_extractor",
    "writer",
)
_GENERAL_SERIAL_ROLE_NAMES = (
    "planner",
    "controller",
    "evidence_extractor",
    "writer",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _snapshot_text(
    settings_snapshot: Mapping[str, Any], key: str, fallback: str
) -> str:
    """Read one non-secret settings value without touching the database."""

    raw = settings_snapshot.get(key, fallback)
    if isinstance(raw, Mapping):
        raw = raw.get("value", fallback)
    if not isinstance(raw, str) or not raw.strip():
        return fallback
    return raw.strip()


def _positive_env_budget(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not 1 <= value <= _MAX_GENERAL_WEB_BUDGET:
        raise ValueError(
            f"{name} must be between 1 and {_MAX_GENERAL_WEB_BUDGET}"
        )
    return value


@dataclass(frozen=True, slots=True)
class GeneralWebRunSettings:
    """Explicit, credential-free product limits for one General web run."""

    max_model_calls: int = GENERAL_WEB_DEFAULT_MAX_MODEL_CALLS
    max_tool_calls: int = GENERAL_WEB_DEFAULT_MAX_TOOL_CALLS
    fetch_timeout_seconds: int = GENERAL_WEB_DEFAULT_FETCH_TIMEOUT_SECONDS
    structured_output_mode: str = "prompted_json"
    orchestration_mode: str = "agentic"
    evidence_max_chunks_per_source: int = 3

    def __post_init__(self) -> None:
        for field_name in ("max_model_calls", "max_tool_calls"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if not 1 <= value <= _MAX_GENERAL_WEB_BUDGET:
                raise ValueError(
                    f"{field_name} must be between 1 and {_MAX_GENERAL_WEB_BUDGET}"
                )
        if (
            isinstance(self.fetch_timeout_seconds, bool)
            or not isinstance(self.fetch_timeout_seconds, int)
            or not 1 <= self.fetch_timeout_seconds <= 300
        ):
            raise ValueError("fetch_timeout_seconds must be between 1 and 300")
        if self.structured_output_mode not in {"prompted_json", "json_object"}:
            raise ValueError(
                "structured_output_mode must be prompted_json or json_object"
            )
        if self.orchestration_mode not in {"serial", "agentic"}:
            raise ValueError("orchestration_mode must be serial or agentic")
        if (
            isinstance(self.evidence_max_chunks_per_source, bool)
            or not isinstance(self.evidence_max_chunks_per_source, int)
            or not 1 <= self.evidence_max_chunks_per_source <= 10
        ):
            raise ValueError("evidence_max_chunks_per_source must be between 1 and 10")

    @classmethod
    def from_environment(cls) -> "GeneralWebRunSettings":
        return cls(
            max_model_calls=_positive_env_budget(
                "LDR_GENERAL_MAX_MODEL_CALLS", GENERAL_WEB_DEFAULT_MAX_MODEL_CALLS
            ),
            max_tool_calls=_positive_env_budget(
                "LDR_GENERAL_MAX_TOOL_CALLS", GENERAL_WEB_DEFAULT_MAX_TOOL_CALLS
            ),
            fetch_timeout_seconds=_positive_env_budget(
                "LDR_GENERAL_FETCH_TIMEOUT_SECONDS",
                GENERAL_WEB_DEFAULT_FETCH_TIMEOUT_SECONDS,
            ),
        )


@dataclass(frozen=True, slots=True)
class GeneralWebRunResult:
    """The only General outcome consumed by the legacy report persistence path."""

    workflow_result: GeneralWorkflowResult
    artifacts: GeneralRunArtifacts
    report_body_markdown: str
    source_rows: tuple[dict[str, str], ...]
    metadata: dict[str, object]

    @property
    def is_publishable(self) -> bool:
        return self.workflow_result.is_publishable


@dataclass(frozen=True, slots=True)
class GeneralResearchProgress:
    """A browser-safe projection of one immutable General audit event.

    This is deliberately not a copy of the trace event.  Trace events may
    contain local locators or verified quotes needed by offline audit; live UI
    receives only stable stage labels, bounded counters, and tool categories.
    """

    phase: str
    message: str
    progress_percent: int
    metadata: dict[str, object]


def general_progress_from_audit_event(
    event: GeneralAuditEvent,
) -> GeneralResearchProgress:
    """Project one audit event to non-sensitive product observability data."""

    if not isinstance(event, GeneralAuditEvent):
        raise TypeError("event must be GeneralAuditEvent")
    data = event.data
    if event.kind == GeneralAuditEventKind.PLAN:
        return GeneralResearchProgress(
            phase="search_planning",
            message="General research plan created.",
            progress_percent=30,
            metadata={"plan_item_count": len(data.get("claim_ids", ()))},
        )
    if event.kind == GeneralAuditEventKind.BRIEF:
        return GeneralResearchProgress(
            phase="research_brief",
            message="Clarified the research objective and scope.",
            progress_percent=15,
            metadata={"brief_created": True},
        )
    if event.kind == GeneralAuditEventKind.SUPERVISION:
        task_count = len(data.get("task_ids", ()))
        return GeneralResearchProgress(
            phase="supervision",
            message=(
                "Supervisor finished research dispatch."
                if data.get("should_finish")
                else f"Supervisor dispatched {task_count} bounded research task(s)."
            ),
            progress_percent=38,
            metadata={
                "round_index": int(data.get("round_index", 0)),
                "task_count": task_count,
            },
        )
    if event.kind == GeneralAuditEventKind.ACTION:
        action_type = data.get("action_type")
        connector_id = data.get("connector_id")
        if action_type == "search":
            tool = (
                "general_search_local_collection"
                if connector_id == "local_collection"
                else "general_search_public_web"
            )
        elif action_type == "fetch":
            tool = "general_fetch_verified_candidate"
        else:
            tool = "general_request_stop"
        outcome = str(data.get("outcome", "recorded"))
        return GeneralResearchProgress(
            phase="tool_call",
            message=f"General {action_type or 'controller'} action {outcome}.",
            progress_percent=45,
            metadata={
                "tool": tool,
                "arguments": {"connector_id": connector_id}
                if connector_id in {"public_web", "local_collection"}
                else {},
                "task_id": str(data["task_id"]) if "task_id" in data else None,
            },
        )
    if event.kind == GeneralAuditEventKind.SOURCE:
        channel = str(data.get("source_channel", "source"))
        return GeneralResearchProgress(
            phase="observation",
            message=f"Verified {channel} source snapshot.",
            progress_percent=55,
            metadata={"source_channel": channel},
        )
    if event.kind == GeneralAuditEventKind.EVIDENCE:
        return GeneralResearchProgress(
            phase="observation",
            message="Recorded quote-level evidence.",
            progress_percent=65,
            metadata={"evidence_recorded": True},
        )
    if event.kind == GeneralAuditEventKind.COVERAGE:
        return GeneralResearchProgress(
            phase="coverage_audit",
            message="Audited evidence coverage.",
            progress_percent=75,
            metadata={
                "covered_claims": int(data.get("covered_claims", 0)),
                "total_claims": int(data.get("total_claims", 0)),
                "decision": str(data.get("decision", "blocked")),
            },
        )
    if event.kind == GeneralAuditEventKind.MEMO:
        return GeneralResearchProgress(
            phase="supervision",
            message="Compressed a worker handoff with provenance links.",
            progress_percent=72,
            metadata={
                "memo_recorded": True,
                "unresolved_count": int(data.get("unresolved_count", 0)),
            },
        )
    if event.kind == GeneralAuditEventKind.CITATION_AUDIT:
        return GeneralResearchProgress(
            phase="output_generation",
            message="Audited claim citations and semantic support.",
            progress_percent=80,
            metadata={"citation_audit": str(data.get("verdict", "recorded"))},
        )
    raise ValueError(f"unsupported General audit event kind: {event.kind!r}")


def general_artifact_root() -> Path:
    """Return an absolute, operator-controlled General artifact root.

    This is intentionally separate from the public report database.  Raw page
    snapshots and quote-level audit artifacts must not become browser paths or
    be inferred from a user-controlled run identifier.
    """

    configured = os.getenv("LDR_GENERAL_ARTIFACT_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("LDR_GENERAL_ARTIFACT_DIR must be an absolute path")
        return root.resolve()
    return (Path.cwd() / "outputs" / "general_research").resolve()


def general_source_mode(
    collection: CollectionDescriptor | None,
) -> str:
    """Return the only three supported source-capability modes for V1.

    A collection's visibility is application-owned metadata, never a model
    assertion.  Private collections therefore have no public-web connector at
    all; the caller couples that mode to the egress policy before the LLM is
    constructed.  A public collection may supplement, but never silently
    replace, public web evidence.
    """

    if collection is None:
        return "web"
    if not isinstance(collection, CollectionDescriptor):
        raise TypeError("collection must be CollectionDescriptor or None")
    return "web_plus_public_collection" if collection.is_public else "private_collection"


def _source_connectors_for_mode(
    source_mode: str,
) -> tuple[SourceConnectorConfig, ...]:
    public_web = SourceConnectorConfig(
        connector_id="public_web",
        source_channel=SourceChannel.PUBLIC_WEB,
    )
    local_collection = SourceConnectorConfig(
        connector_id="local_collection",
        source_channel=SourceChannel.LOCAL_COLLECTION,
        requires_user_authorization=True,
    )
    if source_mode == "web":
        return (public_web,)
    if source_mode == "web_plus_public_collection":
        return (public_web, local_collection)
    if source_mode == "private_collection":
        return (local_collection,)
    raise ValueError(f"unsupported General source mode: {source_mode!r}")


def build_general_execution_config(
    *,
    run_id: str,
    query: str,
    settings_snapshot: Mapping[str, Any],
    model_provider: str | None,
    model_name: str | None,
    settings: GeneralWebRunSettings,
    collection: CollectionDescriptor | None = None,
    created_at: str | None = None,
) -> GeneralExecutionConfig:
    """Build the full General config before any model or connector call.

    ``ModelRoleConfig`` records an identity for every prompt role even when
    the application maps all roles to one physical model.  This keeps the
    audit trail honest about logical roles without claiming a multi-model
    architecture.
    """

    if not isinstance(settings_snapshot, Mapping):
        raise TypeError("settings_snapshot must be a mapping")
    if not isinstance(settings, GeneralWebRunSettings):
        raise TypeError("settings must be GeneralWebRunSettings")
    provider = (
        model_provider.strip()
        if isinstance(model_provider, str) and model_provider.strip()
        else _snapshot_text(settings_snapshot, "llm.provider", "configured")
    )
    model = (
        model_name.strip()
        if isinstance(model_name, str) and model_name.strip()
        else _snapshot_text(settings_snapshot, "llm.model", "configured")
    )
    raw_temperature = settings_snapshot.get("llm.temperature")
    temperature = (
        float(raw_temperature)
        if isinstance(raw_temperature, (int, float))
        and not isinstance(raw_temperature, bool)
        else None
    )
    role_config = ModelRoleConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        structured_output_mode=settings.structured_output_mode,
    )
    source_mode = general_source_mode(collection)
    agentic = settings.orchestration_mode == "agentic"
    role_names = _GENERAL_ROLE_NAMES if agentic else _GENERAL_SERIAL_ROLE_NAMES
    # Agentic is the product and benchmark baseline. Serial remains available
    # only as an explicit topology/cost ablation.
    return GeneralExecutionConfig(
        run=GeneralRunConfig(
            run_id=run_id,
            query=query,
            created_at=created_at or _timestamp(),
            max_model_calls=settings.max_model_calls,
            max_tool_calls=settings.max_tool_calls,
            max_parallel_subagents=3 if agentic else 0,
        ),
        model_roles=tuple((role, role_config) for role in role_names),
        workers=(
            WorkerToolPolicy(allow_subagents=True, max_research_workers=3)
            if agentic
            else WorkerToolPolicy(allow_subagents=False, max_research_workers=1)
        ),
        source_connectors=_source_connectors_for_mode(source_mode),
        evidence_extraction=EvidenceExtractionPolicy(
            max_chunks_per_source=settings.evidence_max_chunks_per_source
        ),
        citation_policy=CitationAuditPolicy(
            require_post_synthesis_audit=False,
            fail_on_unsupported_claim=False,
            max_writer_repairs=1,
        ),
        orchestration=AgenticOrchestrationPolicy.for_tier("standard") if agentic else None,
    )


def _collection_document_display_url(source: SourceRecord) -> str:
    """Map an audited local locator to the existing protected document view.

    The report and artifacts retain the ``local://`` locator verbatim.  This
    user-interface link exists only for the current user's resource list, so
    it cannot become an alternative provenance identifier.
    """

    parsed = urlparse(source.url)
    document_id = parsed.path.strip("/")
    try:
        normalized_document_id = str(UUID(document_id))
    except ValueError as exc:
        raise ValueError("General local source locator has no document UUID") from exc
    if (
        parsed.scheme != "local"
        or not parsed.netloc
        or not document_id
        or "/" in document_id
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("General local source locator is malformed")
    return f"/library/document/{normalized_document_id}"


def _legacy_source_row(source: SourceRecord, *, index: int) -> dict[str, str]:
    """Expose source metadata only; raw snapshot text stays in artifacts."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError("source index must be a positive integer")
    if source.source_channel == SourceChannel.PUBLIC_WEB:
        url = source.url
        source_type = "web"
    elif source.source_channel == SourceChannel.LOCAL_COLLECTION:
        url = _collection_document_display_url(source)
        source_type = "local_collection"
    else:  # defensive for future source channels
        raise ValueError("General source channel cannot be persisted")
    return {
        "url": url,
        "title": source.title,
        "snippet": "Verified full-page source retained in General audit artifacts.",
        "source_type": source_type,
        "general_source_id": source.source_id,
        "general_content_hash": source.content_hash or "",
        "general_source_channel": source.source_channel.value,
        "general_source_connector_id": source.source_connector_id,
        # Existing report assembly uses this index to keep its source list in
        # the same order as General's deterministic [N] claim citations.
        "index": str(index),
    }


def _legacy_public_source_row(source: SourceRecord, *, index: int) -> dict[str, str]:
    """Compatibility helper for callers that explicitly require public sources."""

    if source.source_channel != SourceChannel.PUBLIC_WEB:
        raise ValueError("General public source row requires a public web source")
    return _legacy_source_row(source, index=index)


def _legacy_source_rows(
    result: GeneralWorkflowResult,
) -> tuple[dict[str, str], ...]:
    """Persist exactly the report's cited source order when it is publishable."""

    sources_by_id = {source.source_id: source for source in result.sources}
    if result.is_publishable:
        assert result.citation_audit is not None
        ordered_source_ids: list[str] = []
        for trace in result.citation_audit.citation_traces:
            if trace.source_id not in ordered_source_ids:
                ordered_source_ids.append(trace.source_id)
        return tuple(
            _legacy_source_row(sources_by_id[source_id], index=index)
            for index, source_id in enumerate(ordered_source_ids, start=1)
        )
    return tuple(
        _legacy_source_row(source, index=index)
        for index, source in enumerate(result.sources, start=1)
    )


def _report_body(result: GeneralWorkflowResult) -> str:
    """Return an audited body only; never strip a model-authored source block."""

    if not result.is_publishable:
        return result.report_markdown
    assert result.plan is not None
    assert result.report_document is not None
    assert result.citation_audit is not None
    writer_input = WriterInput(
        plan=result.plan,
        sources=result.sources,
        evidence_cards=result.evidence_cards,
    )
    return render_report_document(
        writer_input,
        result.report_document,
        result.citation_audit,
        include_sources=False,
    )


def run_general_research(
    *,
    run_id: str,
    query: str,
    llm: Any,
    search_engine_name: str | None,
    settings_snapshot: Mapping[str, Any],
    username: str,
    model_provider: str | None,
    model_name: str | None,
    egress_context: Any = None,
    settings: GeneralWebRunSettings | None = None,
    collection: CollectionDescriptor | None = None,
    artifact_root: Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[GeneralResearchProgress], None] | None = None,
    execution_purpose: str = "product",
    code_revision: str | None = None,
    usage_ledger: ModelUsageLedger | None = None,
) -> GeneralWebRunResult:
    """Run an authorized General V1 capability mode with no Hybrid fallback.

    The caller must already have built the normal application egress context
    and configured LLM.  ``project_public_web_connector`` repeats URL-level
    SSRF/egress checks immediately before every full fetch, so authorization
    is not merely a run-start check.
    """

    if not isinstance(username, str) or not username.strip():
        raise ValueError("username is required for General research")
    if not callable(getattr(llm, "invoke", None)):
        raise TypeError("General research requires an invoke-capable LLM")
    if on_progress is not None and not callable(on_progress):
        raise TypeError("on_progress must be callable or None")
    if usage_ledger is not None and not isinstance(usage_ledger, ModelUsageLedger):
        raise TypeError("usage_ledger must be ModelUsageLedger or None")
    effective_settings = settings or GeneralWebRunSettings.from_environment()
    source_mode = general_source_mode(collection)
    requires_public_web = source_mode in {"web", "web_plus_public_collection"}
    if requires_public_web and (
        not isinstance(search_engine_name, str) or not search_engine_name.strip()
    ):
        raise ValueError("General public-web modes require a public search engine")
    config = build_general_execution_config(
        run_id=run_id,
        query=query,
        settings_snapshot=settings_snapshot,
        model_provider=model_provider,
        model_name=model_name,
        settings=effective_settings,
        collection=collection,
    )
    connector_configs = {
        connector.connector_id: connector for connector in config.source_connectors
    }
    bindings: list[ConnectorBinding] = []
    if requires_public_web:
        public_connector = project_public_web_connector(
            search_engine_name=search_engine_name.strip(),
            settings_snapshot=settings_snapshot,
            username=username,
            egress_context=egress_context,
            fetch_timeout_seconds=effective_settings.fetch_timeout_seconds,
        )
        bindings.append(
            ConnectorBinding(
                config=connector_configs["public_web"], connector=public_connector
            )
        )
    if collection is not None:
        local_connector = project_local_collection_connector(
            descriptor=collection,
            username=username,
            settings_snapshot=dict(settings_snapshot),
        )
        bindings.append(
            ConnectorBinding(
                config=connector_configs["local_collection"], connector=local_connector
            )
        )
    runtime_implementations = {
        "public_web": f"project_public_web:{search_engine_name.strip()}"
        for connector in config.source_connectors
        if connector.connector_id == "public_web"
    }
    if collection is not None:
        runtime_implementations["local_collection"] = "project_local_collection"
    runtime_manifest = GeneralRunManifest.for_config(
        config,
        # A deployment should supply its immutable revision.  The explicit
        # fallback is intentionally honest for local product development and
        # makes such runs ineligible for benchmark comparison until recorded.
        code_revision=(
            code_revision.strip()
            if isinstance(code_revision, str) and code_revision.strip()
            else os.getenv("LDR_GENERAL_CODE_REVISION", "").strip() or "unrecorded"
        ),
        execution_purpose=execution_purpose,
        implementations_by_connector=runtime_implementations,
        request_timeouts_by_connector=(
            {"public_web": effective_settings.fetch_timeout_seconds}
            if requires_public_web
            else {}
        ),
    )
    runner = GeneralResearchRunner(
        config=config,
        gateway=LangChainJsonGateway(
            {role: llm for role in config.models_by_role},
            usage_ledger=usage_ledger,
            structured_output_modes_by_role={
                role: model_config.structured_output_mode
                for role, model_config in config.model_roles
            },
        ),
        connector_registry=ConnectorRegistry(bindings),
        artifact_root=artifact_root or general_artifact_root(),
        authorized_connector_ids=("local_collection",) if collection is not None else (),
        should_cancel=should_cancel,
        event_observer=(
            (lambda event: on_progress(general_progress_from_audit_event(event)))
            if on_progress is not None
            else None
        ),
        runtime_manifest=runtime_manifest,
    )
    workflow_result, artifacts = runner.run()
    return GeneralWebRunResult(
        workflow_result=workflow_result,
        artifacts=artifacts,
        report_body_markdown=_report_body(workflow_result),
        source_rows=_legacy_source_rows(workflow_result),
        metadata={
            "general_research": {
                "schema_version": "general-research-profile/v1",
                "run_id": config.run.run_id,
                "source_mode": source_mode,
                "declared_connector_ids": [
                    connector.connector_id for connector in config.source_connectors
                ],
                "execution_config_digest_sha256": config.digest(),
                "run_manifest_digest_sha256": runtime_manifest.digest(),
                "code_revision": runtime_manifest.code_revision,
                "execution_purpose": runtime_manifest.execution_purpose,
                "workflow_status": workflow_result.status.value,
                "publishable": workflow_result.is_publishable,
                "terminal_reason": workflow_result.reason,
                "model_calls_used": workflow_result.budget.model_calls_used,
                "tool_calls_used": workflow_result.budget.tool_calls_used,
                "source_count": len(workflow_result.sources),
                "evidence_card_count": len(workflow_result.evidence_cards),
                "orchestration_tier": (
                    config.orchestration.tier if config.orchestration is not None else None
                ),
                "supervisor_decision_count": len(workflow_result.supervisor_decisions),
                "research_memo_count": len(workflow_result.research_memos),
            }
        },
    )


def run_general_web_research(
    *,
    run_id: str,
    query: str,
    llm: Any,
    search_engine_name: str,
    settings_snapshot: Mapping[str, Any],
    username: str,
    model_provider: str | None,
    model_name: str | None,
    egress_context: Any = None,
    settings: GeneralWebRunSettings | None = None,
    artifact_root: Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> GeneralWebRunResult:
    """Compatibility entry point for the public-web-only General mode."""

    return run_general_research(
        run_id=run_id,
        query=query,
        llm=llm,
        search_engine_name=search_engine_name,
        settings_snapshot=settings_snapshot,
        username=username,
        model_provider=model_provider,
        model_name=model_name,
        egress_context=egress_context,
        settings=settings,
        artifact_root=artifact_root,
        should_cancel=should_cancel,
    )


__all__ = [
    "GENERAL_WEB_DEFAULT_MAX_MODEL_CALLS",
    "GENERAL_WEB_DEFAULT_MAX_TOOL_CALLS",
    "GeneralWebRunResult",
    "GeneralWebRunSettings",
    "GeneralResearchProgress",
    "build_general_execution_config",
    "general_progress_from_audit_event",
    "general_source_mode",
    "general_artifact_root",
    "run_general_research",
    "run_general_web_research",
]
