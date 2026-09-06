"""One independent, replayable General V1 run entry point.

The runner composes only General contracts, connectors, model gateway, content
snapshots, action loop and workflow. It never calls a Hybrid strategy or takes
a task contract, gold URL, expected answer, or legacy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from .artifact_store import ContentArtifactStore
from .config import GeneralExecutionConfig
from .connectors import ConnectorRegistry
from .controller import ActionLoopResearcher, GeneralActionLoop
from .model_adapters import GeneralModelAdapters, JsonModelGateway
from .parallel_research import ParallelActionLoopResearcher, ParallelResearchCoordinator
from .run_manifest import GeneralRunManifest
from .trace import GeneralAuditEvent, general_event_jsonl_line
from .workflow import GeneralResearchWorkflow, GeneralWorkflowResult


class GeneralRunArtifactError(ValueError):
    """The selected General run directory is invalid or already occupied."""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _result_payload(result: GeneralWorkflowResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "reason": result.reason,
        "plan": result.plan.to_dict() if result.plan is not None else None,
        "brief": result.brief.to_dict() if result.brief is not None else None,
        "plan_adequacy": (
            result.plan_adequacy.to_dict()
            if result.plan_adequacy is not None
            else None
        ),
        "supervisor_decisions": [
            decision.to_dict() for decision in result.supervisor_decisions
        ],
        "research_memos": [memo.to_dict() for memo in result.research_memos],
        "sources": [source.to_dict() for source in result.sources],
        "evidence_cards": [card.to_dict() for card in result.evidence_cards],
        "coverage": result.coverage.to_dict() if result.coverage is not None else None,
        "report_document": (
            result.report_document.to_dict() if result.report_document is not None else None
        ),
        "citation_audit": {
            "passed": result.citation_audit.passed,
            "writer_input_fingerprint": result.citation_audit.writer_input_fingerprint,
            "citation_traces": [
                {
                    "claim_id": trace.claim_id,
                    "evidence_id": trace.evidence_id,
                    "source_id": trace.source_id,
                    "source_locator": trace.source_locator,
                    "source_channel": trace.source_channel.value,
                    "source_title": trace.source_title,
                    "verbatim_quote": trace.verbatim_quote,
                    "locator": trace.locator,
                    "stance": trace.stance.value,
                }
                for trace in result.citation_audit.citation_traces
            ],
            "coverage_gaps": [
                {
                    "code": gap.code,
                    "message": gap.message,
                    "plan_item_id": gap.plan_item_id,
                    "claim_id": gap.claim_id,
                    "evidence_id": gap.evidence_id,
                }
                for gap in result.citation_audit.coverage_gaps
            ],
        }
        if result.citation_audit is not None
        else None,
        "semantic_audit": (
            result.semantic_audit.to_dict() if result.semantic_audit is not None else None
        ),
        "semantic_audit_required": result.semantic_audit_required,
        "budget": {
            "model_calls_used": result.budget.model_calls_used,
            "tool_calls_used": result.budget.tool_calls_used,
            "model_calls_remaining": result.budget.model_calls_remaining,
            "tool_calls_remaining": result.budget.tool_calls_remaining,
        },
        "event_count": len(result.events),
    }


@dataclass(frozen=True, slots=True)
class GeneralRunArtifacts:
    """Operator-facing paths for all artifacts created by one immutable run."""

    run_dir: Path
    config_path: Path
    manifest_path: Path
    result_path: Path
    report_path: Path
    audit_jsonl_path: Path


class GeneralRunArtifactStore:
    """Create a non-overwriting run directory plus replayable General artifacts."""

    def __init__(self, root_dir: str | Path) -> None:
        root = Path(root_dir).expanduser()
        if not root.is_absolute():
            raise GeneralRunArtifactError("artifact root must be an absolute path")
        self._root = root.resolve()

    def create(self, config: GeneralExecutionConfig) -> GeneralRunArtifacts:
        if not isinstance(config, GeneralExecutionConfig):
            raise TypeError("config must be GeneralExecutionConfig")
        # Run IDs are user/model data and may contain path syntax under the
        # general identifier contract. A digest preserves association without
        # treating that input as a filesystem component.
        run_digest = sha256(config.run.run_id.encode("utf-8")).hexdigest()[:24]
        run_dir = self._root / f"general-{run_digest}"
        if run_dir.exists():
            raise GeneralRunArtifactError(
                "run artifact directory already exists; use a new run_id rather than overwrite evidence"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        return GeneralRunArtifacts(
            run_dir=run_dir,
            config_path=run_dir / "general_execution_config.json",
            manifest_path=run_dir / "general_run_manifest.json",
            result_path=run_dir / "general_result.json",
            report_path=run_dir / "report.md",
            audit_jsonl_path=run_dir / "general_audit.jsonl",
        )

    @staticmethod
    def write_config(
        *, artifacts: GeneralRunArtifacts, config: GeneralExecutionConfig
    ) -> None:
        _atomic_write(artifacts.config_path, _json_bytes(config.canonical_dict()))

    @staticmethod
    def write_manifest(
        *, artifacts: GeneralRunArtifacts, manifest: GeneralRunManifest
    ) -> None:
        _atomic_write(artifacts.manifest_path, _json_bytes(manifest.to_dict()))

    @staticmethod
    def write(
        *,
        artifacts: GeneralRunArtifacts,
        config: GeneralExecutionConfig,
        manifest: GeneralRunManifest,
        result: GeneralWorkflowResult,
    ) -> None:
        GeneralRunArtifactStore.write_config(artifacts=artifacts, config=config)
        GeneralRunArtifactStore.write_manifest(artifacts=artifacts, manifest=manifest)
        _atomic_write(artifacts.result_path, _json_bytes(_result_payload(result)))
        _atomic_write(artifacts.report_path, result.report_markdown.encode("utf-8"))
        lines = "".join(
            general_event_jsonl_line(event, config) + "\n" for event in result.events
        )
        _atomic_write(artifacts.audit_jsonl_path, lines.encode("utf-8"))


class GeneralResearchRunner:
    """Compose and execute one General V1 run without legacy-agent fallback."""

    def __init__(
        self,
        *,
        config: GeneralExecutionConfig,
        gateway: JsonModelGateway,
        connector_registry: ConnectorRegistry,
        artifact_root: str | Path,
        authorized_connector_ids: tuple[str, ...] = (),
        clock: Callable[[], str] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        event_observer: Callable[[GeneralAuditEvent], None] | None = None,
        runtime_manifest: GeneralRunManifest | None = None,
    ) -> None:
        if not isinstance(config, GeneralExecutionConfig):
            raise TypeError("config must be GeneralExecutionConfig")
        if not isinstance(gateway, JsonModelGateway):
            raise TypeError("gateway must implement JsonModelGateway")
        if not isinstance(connector_registry, ConnectorRegistry):
            raise TypeError("connector_registry must be ConnectorRegistry")
        declared_connector_ids = {
            connector.connector_id for connector in config.source_connectors
        }
        if connector_registry.connector_ids != declared_connector_ids:
            raise ValueError(
                "connector_registry must bind exactly the connectors declared by config"
            )
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if should_cancel is not None and not callable(should_cancel):
            raise TypeError("should_cancel must be callable or None")
        if event_observer is not None and not callable(event_observer):
            raise TypeError("event_observer must be callable or None")
        if runtime_manifest is not None and not isinstance(
            runtime_manifest, GeneralRunManifest
        ):
            raise TypeError("runtime_manifest must be GeneralRunManifest or None")
        manifest = runtime_manifest or GeneralRunManifest.for_config(config)
        manifest.validate_for_config(config)
        self.config = config
        self.gateway = gateway
        self.connector_registry = connector_registry
        self._artifact_store = GeneralRunArtifactStore(artifact_root)
        self._authorized_connector_ids = authorized_connector_ids
        self._clock = clock
        self._should_cancel = should_cancel
        self._event_observer = event_observer
        self.runtime_manifest = manifest

    def run(self) -> tuple[GeneralWorkflowResult, GeneralRunArtifacts]:
        artifacts = self._artifact_store.create(self.config)
        GeneralRunArtifactStore.write_config(artifacts=artifacts, config=self.config)
        GeneralRunArtifactStore.write_manifest(
            artifacts=artifacts, manifest=self.runtime_manifest
        )
        content_store = ContentArtifactStore(artifacts.run_dir)
        loop_kwargs: dict[str, Any] = {
            "config": self.config,
            "registry": self.connector_registry,
            "artifact_store": content_store,
            "authorized_connector_ids": self._authorized_connector_ids,
            "should_cancel": self._should_cancel,
        }
        if self._clock is not None:
            loop_kwargs["clock"] = self._clock
        model_kwargs: dict[str, Any] = {}
        if self._clock is not None:
            model_kwargs["clock"] = self._clock
        models = GeneralModelAdapters(self.gateway, **model_kwargs)
        if self.config.orchestration is None:
            loop = GeneralActionLoop(**loop_kwargs)
            workflow_adapters = models.workflow_adapters(
                execution_config=self.config,
                researcher=ActionLoopResearcher(loop, models.controller),
            )
        else:
            coordinator = ParallelResearchCoordinator(
                max_workers=self.config.orchestration.max_parallel_workers,
                loop_factory=lambda task: GeneralActionLoop(**loop_kwargs),
                controller_factory=models.controller_for_task,
            )
            workflow_adapters = models.workflow_adapters(
                execution_config=self.config,
                agentic_researcher=ParallelActionLoopResearcher(coordinator),
            )
        workflow = GeneralResearchWorkflow(
            self.config,
            workflow_adapters,
            should_cancel=self._should_cancel,
            event_observer=self._event_observer,
        )
        result = workflow.run()
        GeneralRunArtifactStore.write(
            artifacts=artifacts,
            config=self.config,
            manifest=self.runtime_manifest,
            result=result,
        )
        return result, artifacts


__all__ = [
    "GeneralResearchRunner",
    "GeneralRunArtifactError",
    "GeneralRunArtifactStore",
    "GeneralRunArtifacts",
]
