"""Offline integrity audit for one immutable General V1 run artifact.

This is deliberately an *artifact* checker, not a second research agent and
not an LLM judge.  It independently replays every deterministic invariant that
can be verified after a run: configuration identity, event linkage, content
snapshots, quote spans, coverage, citation provenance, semantic-gate shape and
the rendered report.  A passing result says the saved artifact obeys the V1
protocol; it never upgrades the model's semantic review into a truth claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Mapping

from .artifact_store import ContentArtifactError, ContentArtifactStore
from .config import GeneralExecutionConfig
from .coverage import CoverageAuditor
from .plan_adequacy import (
    PlanAdequacyAudit,
    RequirementPlanDecision,
    audit_plan_adequacy,
)
from .run_manifest import GeneralRunManifest
from .schemas import (
    CoverageDecision,
    CoverageState,
    EvidenceCard,
    MemoFinding,
    PlanItem,
    ResearchBrief,
    ResearchMemo,
    ResearchPlan,
    ResearchRequirement,
    ResearchTask,
    SourceRecord,
    SupervisorDecision,
    validate_research_memo,
    validate_supervisor_decision,
)
from .semantic_audit import (
    SemanticAuditResult,
    SemanticClaimReview,
    validate_semantic_audit,
)
from .trace import (
    GeneralAuditEvent,
    GeneralAuditEventKind,
    general_event_jsonl_payload,
)
from .writer import (
    ReportClaim,
    ReportDocument,
    WriterInput,
    audit_report_citations,
    render_report_document,
)


GENERAL_ARTIFACT_AUDIT_SCHEMA_VERSION = "general-artifact-audit/v1"


@dataclass(frozen=True, slots=True)
class ArtifactAuditFinding:
    """One deterministic protocol finding; messages never contain page text."""

    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True, slots=True)
class GeneralArtifactAudit:
    """An offline audit result that distinguishes integrity from publication."""

    run_dir: Path
    run_id: str | None
    execution_config_digest_sha256: str | None
    artifact_integrity_passed: bool
    publishable: bool
    findings: tuple[ArtifactAuditFinding, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GENERAL_ARTIFACT_AUDIT_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "execution_config_digest_sha256": self.execution_config_digest_sha256,
            "artifact_integrity_passed": self.artifact_integrity_passed,
            "publishable": self.publishable,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("artifact file is missing") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("artifact file is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("artifact JSON root must be an object")
    return value


def _as_list(value: object, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must be a JSON list of objects")
    return list(value)


def _plan_from_payload(payload: Mapping[str, Any]) -> ResearchPlan:
    raw_items = _as_list(payload.get("items"), field_name="plan.items")
    return ResearchPlan(
        **{
            **dict(payload),
            "items": tuple(PlanItem(**item) for item in raw_items),
        }
    )


def _brief_from_payload(payload: Mapping[str, Any]) -> ResearchBrief:
    raw_requirements = _as_list(
        payload.get("requirements"), field_name="brief.requirements"
    )
    return ResearchBrief(
        **{
            **dict(payload),
            "requirements": tuple(
                ResearchRequirement(**item) for item in raw_requirements
            ),
        }
    )


def _plan_adequacy_from_payload(payload: Mapping[str, Any]) -> PlanAdequacyAudit:
    raw_decisions = _as_list(payload.get("decisions"), field_name="plan_adequacy.decisions")
    allowed = {"schema_version", "brief_id", "plan_id", "decisions", "missing_required_requirement_ids", "is_adequate"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ValueError("plan_adequacy has unsupported keys: " + ", ".join(sorted(unknown)))
    raw_audit = {
        key: value
        for key, value in payload.items()
        if key not in {"missing_required_requirement_ids", "is_adequate"}
    }
    return PlanAdequacyAudit(
        **{
            **raw_audit,
            "decisions": tuple(
                RequirementPlanDecision(
                    requirement_id=item["requirement_id"],
                    required=item["required"],
                    plan_item_ids=tuple(item["plan_item_ids"]),
                )
                for item in raw_decisions
            ),
        }
    )


def _coverage_from_payload(payload: Mapping[str, Any]) -> CoverageState:
    raw_decisions = _as_list(payload.get("decisions"), field_name="coverage.decisions")
    return CoverageState(
        **{
            **dict(payload),
            "decisions": tuple(CoverageDecision(**item) for item in raw_decisions),
        }
    )


def _document_from_payload(payload: Mapping[str, Any]) -> ReportDocument:
    raw_claims = _as_list(payload.get("claims"), field_name="report_document.claims")
    return ReportDocument(
        **{
            **dict(payload),
            "claims": tuple(ReportClaim(**item) for item in raw_claims),
        }
    )


def _semantic_from_payload(payload: Mapping[str, Any]) -> SemanticAuditResult:
    raw_reviews = _as_list(payload.get("reviews"), field_name="semantic_audit.reviews")
    return SemanticAuditResult(
        **{
            **dict(payload),
            "reviews": tuple(SemanticClaimReview(**item) for item in raw_reviews),
        }
    )


def _supervisor_decision_from_payload(payload: Mapping[str, Any]) -> SupervisorDecision:
    raw_tasks = _as_list(payload.get("tasks"), field_name="supervisor_decision.tasks")
    return SupervisorDecision(
        **{
            **dict(payload),
            "tasks": tuple(ResearchTask(**task) for task in raw_tasks),
        }
    )


def _memo_from_payload(payload: Mapping[str, Any]) -> ResearchMemo:
    raw_findings = _as_list(payload.get("findings"), field_name="research_memo.findings")
    return ResearchMemo(
        **{
            **dict(payload),
            "findings": tuple(MemoFinding(**finding) for finding in raw_findings),
        }
    )


def _citation_subset(audit) -> dict[str, object]:
    """Serialize precisely the CitationAudit fields retained in result JSON."""

    return {
        "passed": audit.passed,
        "writer_input_fingerprint": audit.writer_input_fingerprint,
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
            for trace in audit.citation_traces
        ],
        "coverage_gaps": [
            {
                "code": gap.code,
                "message": gap.message,
                "plan_item_id": gap.plan_item_id,
                "claim_id": gap.claim_id,
                "evidence_id": gap.evidence_id,
            }
            for gap in audit.coverage_gaps
        ],
    }


class _AuditCollector:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.findings: list[ArtifactAuditFinding] = []

    def fail(self, code: str, message: str, *, path: str | None = None) -> None:
        self.findings.append(ArtifactAuditFinding(code, message, path))

    def attempt(self, code: str, path: str, callback):
        try:
            return callback()
        except (TypeError, ValueError, KeyError, ContentArtifactError) as exc:
            self.fail(code, str(exc), path=path)
        return None


def _audit_events(
    collector: _AuditCollector,
    *,
    audit_path: Path,
    config: GeneralExecutionConfig,
    expected_event_count: object,
) -> None:
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        collector.fail("audit_jsonl_unreadable", str(exc), path=audit_path.name)
        return
    if expected_event_count != len(lines):
        collector.fail(
            "event_count_mismatch",
            "general_result event_count does not equal JSONL record count",
            path="general_result.json:event_count",
        )
    for expected_id, line in enumerate(lines):
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("JSONL record must be an object")
            raw_event = raw["event"]
            if not isinstance(raw_event, dict):
                raise ValueError("JSONL event must be an object")
            event = GeneralAuditEvent(
                run_id=raw_event["run_id"],
                event_id=raw_event["event_id"],
                kind=GeneralAuditEventKind(raw_event["kind"]),
                data=raw_event["data"],
                timestamp=datetime.fromisoformat(
                    str(raw_event["timestamp"]).replace("Z", "+00:00")
                ),
                schema_version=raw_event["schema_version"],
            )
            if event.event_id != expected_id:
                raise ValueError("event_id values must start at zero and be contiguous")
            if raw != general_event_jsonl_payload(event, config):
                raise ValueError("JSONL record does not match its event/config payload")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            collector.fail(
                "invalid_audit_event",
                str(exc),
                path=f"general_audit.jsonl:{expected_id + 1}",
            )


def audit_general_run_artifact(run_dir: str | Path) -> GeneralArtifactAudit:
    """Audit one General V1 run without calling any model, connector, or network.

    The function returns findings rather than throwing for malformed artifacts,
    making it suitable for batch evaluation.  ``artifact_integrity_passed`` is
    independent of ``publishable``: an incomplete run can be a correctly
    recorded safe refusal, while a complete run whose snapshot was altered is
    an integrity failure.
    """

    root = Path(run_dir).expanduser().resolve()
    collector = _AuditCollector(root)
    config_path = root / "general_execution_config.json"
    manifest_path = root / "general_run_manifest.json"
    result_path = root / "general_result.json"
    report_path = root / "report.md"
    audit_path = root / "general_audit.jsonl"
    config = collector.attempt(
        "invalid_execution_config",
        config_path.name,
        lambda: GeneralExecutionConfig.from_mapping(_load_object(config_path)),
    )
    manifest = collector.attempt(
        "invalid_run_manifest",
        manifest_path.name,
        lambda: GeneralRunManifest.from_mapping(_load_object(manifest_path)),
    )
    result = collector.attempt("invalid_result", result_path.name, lambda: _load_object(result_path))
    if config is None or manifest is None or result is None:
        return GeneralArtifactAudit(
            run_dir=root,
            run_id=None,
            execution_config_digest_sha256=None,
            artifact_integrity_passed=False,
            publishable=False,
            findings=tuple(collector.findings),
        )

    run_id = config.run.run_id
    digest = config.digest()
    try:
        manifest.validate_for_config(config)
    except (TypeError, ValueError) as exc:
        collector.fail(
            "run_manifest_config_mismatch",
            str(exc),
            path=manifest_path.name,
        )
    _audit_events(
        collector,
        audit_path=audit_path,
        config=config,
        expected_event_count=result.get("event_count"),
    )
    if not report_path.is_file():
        collector.fail("report_missing", "report.md is missing", path=report_path.name)

    plan_payload = result.get("plan")
    plan = (
        collector.attempt("invalid_plan", "general_result.json:plan", lambda: _plan_from_payload(plan_payload))
        if isinstance(plan_payload, dict)
        else None
    )
    if plan is not None and plan.run_id != run_id:
        collector.fail("plan_run_id_mismatch", "plan.run_id differs from execution config", path="general_result.json:plan")
    status = result.get("status")
    valid_statuses = {
        "complete",
        "incomplete",
        "budget_exhausted",
        "invalid_artifact",
        "runtime_error",
    }
    if status not in valid_statuses:
        collector.fail(
            "invalid_workflow_status",
            "general_result status is not a supported terminal status",
            path="general_result.json:status",
        )
    complete = status == "complete"
    sources = collector.attempt(
        "invalid_sources",
        "general_result.json:sources",
        lambda: tuple(SourceRecord(**item) for item in _as_list(result.get("sources"), field_name="sources")),
    )
    cards = collector.attempt(
        "invalid_evidence_cards",
        "general_result.json:evidence_cards",
        lambda: tuple(EvidenceCard(**item) for item in _as_list(result.get("evidence_cards"), field_name="evidence_cards")),
    )
    if sources is None or cards is None:
        return GeneralArtifactAudit(root, run_id, digest, False, False, tuple(collector.findings))

    source_by_id = {source.source_id: source for source in sources}
    if len(source_by_id) != len(sources):
        collector.fail("duplicate_source_id", "source IDs must be unique", path="general_result.json:sources")
    content_store = ContentArtifactStore(root)
    snapshots: dict[str, str] = {}
    for source in sources:
        if not source.content_verified:
            continue
        if not source.content_artifact_id or not source.content_hash:
            collector.fail(
                "verified_source_missing_snapshot_reference",
                "verified source must declare content artifact ID and hash",
                path=f"source:{source.source_id}",
            )
            continue
        snapshot = collector.attempt(
            "invalid_source_snapshot",
            f"source:{source.source_id}",
            lambda source=source: content_store.load(
                source.content_artifact_id,
                expected_content_hash=source.content_hash,
            ),
        )
        if snapshot is not None:
            snapshots[source.source_id] = snapshot
    for card in cards:
        source = source_by_id.get(card.source_id)
        snapshot = snapshots.get(card.source_id)
        if source is None:
            collector.fail("evidence_unknown_source", "evidence references unknown source", path=f"evidence:{card.evidence_id}")
            continue
        if not source.content_verified or snapshot is None:
            collector.fail("evidence_unverified_source", "evidence must use a verified snapshot", path=f"evidence:{card.evidence_id}")
            continue
        if card.source_content_hash != source.content_hash:
            collector.fail("evidence_hash_mismatch", "evidence hash differs from source hash", path=f"evidence:{card.evidence_id}")
        if snapshot[card.quote_start : card.quote_end] != card.verbatim_quote:
            collector.fail("evidence_quote_mismatch", "quote offsets do not reproduce verbatim_quote", path=f"evidence:{card.evidence_id}")

    if config.orchestration is not None:
        brief_payload = result.get("brief")
        brief = (
            collector.attempt(
                "invalid_research_brief",
                "general_result.json:brief",
                lambda: _brief_from_payload(brief_payload),
            )
            if isinstance(brief_payload, dict)
            else None
        )
        if brief is None:
            if plan is not None:
                collector.fail(
                    "missing_research_brief",
                    "agentic result requires a typed research brief",
                    path="general_result.json:brief",
                )
            if result.get("plan_adequacy") is not None:
                collector.fail(
                    "plan_adequacy_without_plan",
                    "plan adequacy cannot exist without a research plan",
                    path="general_result.json:plan_adequacy",
                )
        elif brief.run_id != run_id or brief.user_query != config.run.query:
            collector.fail(
                "research_brief_identity_mismatch",
                "research brief does not belong to the frozen run/query",
                path="general_result.json:brief",
            )
        elif plan is not None:
            stored_adequacy_payload = result.get("plan_adequacy")
            stored_adequacy = (
                collector.attempt(
                    "invalid_plan_adequacy",
                    "general_result.json:plan_adequacy",
                    lambda: _plan_adequacy_from_payload(stored_adequacy_payload),
                )
                if isinstance(stored_adequacy_payload, dict)
                else None
            )
            if stored_adequacy is None:
                collector.fail(
                    "missing_plan_adequacy",
                    "agentic result requires a Brief -> Plan adequacy audit",
                    path="general_result.json:plan_adequacy",
                )
            else:
                recomputed_adequacy = collector.attempt(
                    "plan_adequacy_recomputation_failed",
                    "general_result.json:plan_adequacy",
                    lambda: audit_plan_adequacy(brief, plan),
                )
                if (
                    recomputed_adequacy is not None
                    and stored_adequacy.to_dict() != recomputed_adequacy.to_dict()
                ):
                    collector.fail(
                        "plan_adequacy_mismatch",
                        "stored plan adequacy differs from deterministic recomputation",
                        path="general_result.json:plan_adequacy",
                    )
                elif complete and not stored_adequacy.is_adequate:
                    collector.fail(
                        "required_brief_requirement_unplanned",
                        "complete agentic result contains an unplanned required brief requirement",
                        path="general_result.json:plan_adequacy",
                    )
        elif result.get("plan_adequacy") is not None:
            collector.fail(
                "plan_adequacy_without_plan",
                "plan adequacy cannot exist without a research plan",
                path="general_result.json:plan_adequacy",
            )

        decisions = collector.attempt(
            "invalid_supervisor_decisions",
            "general_result.json:supervisor_decisions",
            lambda: tuple(
                _supervisor_decision_from_payload(item)
                for item in _as_list(
                    result.get("supervisor_decisions"),
                    field_name="supervisor_decisions",
                )
            ),
        )
        tasks_by_id: dict[str, ResearchTask] = {}
        if decisions is not None:
            if not decisions and complete:
                collector.fail(
                    "missing_supervisor_decision",
                    "complete agentic result requires at least one supervisor decision",
                    path="general_result.json:supervisor_decisions",
                )
            elif decisions and plan is None:
                collector.fail(
                    "supervisor_decision_without_plan",
                    "supervisor decisions require a research plan",
                    path="general_result.json:supervisor_decisions",
                )
            for expected_round, decision in enumerate(decisions):
                try:
                    if decision.round_index != expected_round:
                        raise ValueError("supervisor round_index values must start at zero and be contiguous")
                    if decision.round_index >= config.orchestration.max_supervisor_rounds:
                        raise ValueError("supervisor decision exceeds the configured round cap")
                    if decision.should_finish and expected_round != len(decisions) - 1:
                        raise ValueError("a finish decision must be the final supervisor decision")
                    if plan is None:
                        raise ValueError("agentic supervisor decisions require a research plan")
                    validate_supervisor_decision(decision, plan)
                    for task in decision.tasks:
                        if task.task_id in tasks_by_id:
                            raise ValueError("supervisor task IDs must be unique across one run")
                        tasks_by_id[task.task_id] = task
                except (TypeError, ValueError) as exc:
                    collector.fail(
                        "invalid_supervisor_decision",
                        str(exc),
                        path=f"general_result.json:supervisor_decisions:{expected_round}",
                    )

        memos = collector.attempt(
            "invalid_research_memos",
            "general_result.json:research_memos",
            lambda: tuple(
                _memo_from_payload(item)
                for item in _as_list(
                    result.get("research_memos"), field_name="research_memos"
                )
            ),
        )
        if memos is not None:
            memo_task_ids = [memo.task_id for memo in memos]
            if len(memo_task_ids) != len(set(memo_task_ids)):
                collector.fail(
                    "duplicate_research_memo_task",
                    "research memos must use unique task IDs",
                    path="general_result.json:research_memos",
                )
            for memo in memos:
                task = tasks_by_id.get(memo.task_id)
                if task is None:
                    collector.fail(
                        "research_memo_unknown_task",
                        "research memo refers to no supervisor task",
                        path=f"research_memo:{memo.task_id}",
                    )
                    continue
                try:
                    validate_research_memo(memo, task, cards, sources)
                except (TypeError, ValueError) as exc:
                    collector.fail(
                        "invalid_research_memo",
                        str(exc),
                        path=f"research_memo:{memo.task_id}",
                    )
            missing_memos = sorted(set(tasks_by_id).difference(memo_task_ids))
            if missing_memos and complete:
                collector.fail(
                    "missing_research_memo",
                    "complete run requires one evidence-linked or unresolved memo per dispatched task",
                    path="general_result.json:research_memos",
                )
    elif (
        result.get("brief") is not None
        or result.get("plan_adequacy") is not None
        or result.get("supervisor_decisions") not in (None, [])
        or result.get("research_memos") not in (None, [])
    ):
        collector.fail(
            "serial_result_contains_agentic_artifacts",
            "serial General result cannot contain agentic orchestration artifacts",
            path="general_result.json",
        )

    coverage = None
    coverage_payload = result.get("coverage")
    if coverage_payload is not None:
        if not isinstance(coverage_payload, dict):
            collector.fail("invalid_coverage", "coverage must be an object or null", path="general_result.json:coverage")
        else:
            coverage = collector.attempt(
                "invalid_coverage", "general_result.json:coverage", lambda: _coverage_from_payload(coverage_payload)
            )
            if coverage is not None and plan is not None:
                recomputed = collector.attempt(
                    "coverage_recomputation_failed",
                    "general_result.json:coverage",
                    lambda: CoverageAuditor(
                        as_of=coverage.audited_at[:10], audited_at=coverage.audited_at
                    ).audit(plan, sources=sources, cards=cards),
                )
                if recomputed is not None and recomputed.to_dict() != coverage.to_dict():
                    collector.fail("coverage_mismatch", "stored coverage differs from deterministic recomputation", path="general_result.json:coverage")

    budget = result.get("budget")
    if not isinstance(budget, dict):
        collector.fail("invalid_budget", "budget must be an object", path="general_result.json:budget")
    else:
        model_used, tool_used = budget.get("model_calls_used"), budget.get("tool_calls_used")
        model_remaining, tool_remaining = budget.get("model_calls_remaining"), budget.get("tool_calls_remaining")
        valid_budget = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (model_used, tool_used, model_remaining, tool_remaining))
        if not valid_budget:
            collector.fail("invalid_budget", "budget values must be non-negative integers", path="general_result.json:budget")
        elif model_used + model_remaining != config.run.max_model_calls or tool_used + tool_remaining != config.run.max_tool_calls:
            collector.fail("budget_mismatch", "budget used/remaining does not match frozen config", path="general_result.json:budget")

    publishable = False
    document_payload = result.get("report_document")
    citation_payload = result.get("citation_audit")
    semantic_payload = result.get("semantic_audit")
    semantic_audit_required = config.citation_policy.require_post_synthesis_audit
    if complete:
        if plan is None or coverage is None or not coverage.ready_to_stop:
            collector.fail("complete_without_coverage", "complete result requires a STOP coverage state", path="general_result.json")
        if not isinstance(document_payload, dict):
            collector.fail("complete_without_document", "complete result requires report_document", path="general_result.json:report_document")
        if not isinstance(citation_payload, dict):
            collector.fail("complete_without_citation_audit", "complete result requires citation_audit", path="general_result.json:citation_audit")
        if semantic_audit_required and not isinstance(semantic_payload, dict):
            collector.fail("complete_without_semantic_audit", "complete result requires semantic_audit", path="general_result.json:semantic_audit")
        document = collector.attempt(
            "invalid_report_document", "general_result.json:report_document", lambda: _document_from_payload(document_payload)
        ) if isinstance(document_payload, dict) else None
        semantic = collector.attempt(
            "invalid_semantic_audit", "general_result.json:semantic_audit", lambda: _semantic_from_payload(semantic_payload)
        ) if isinstance(semantic_payload, dict) else None
        if plan is not None and document is not None and isinstance(citation_payload, dict):
            writer_input = collector.attempt(
                "invalid_writer_input", "general_result.json", lambda: WriterInput(plan=plan, sources=sources, evidence_cards=cards)
            )
            if writer_input is not None:
                recomputed_citation = audit_report_citations(writer_input, document.claims)
                if _citation_subset(recomputed_citation) != citation_payload:
                    collector.fail("citation_audit_mismatch", "stored citation audit differs from deterministic recomputation", path="general_result.json:citation_audit")
                if semantic_audit_required and (
                    semantic is None
                    or not validate_semantic_audit(
                        writer_input, document, recomputed_citation, semantic
                    )
                ):
                    collector.fail("semantic_gate_failed", "semantic audit does not support every final claim", path="general_result.json:semantic_audit")
                try:
                    rendered = render_report_document(writer_input, document, recomputed_citation)
                    stored_report = report_path.read_text(encoding="utf-8")
                    if stored_report != rendered:
                        collector.fail("report_render_mismatch", "report.md differs from deterministic renderer", path=report_path.name)
                except (OSError, ValueError) as exc:
                    collector.fail("report_replay_failed", str(exc), path=report_path.name)
                publishable = not collector.findings
    elif document_payload is not None:
        collector.fail("incomplete_has_final_document", "non-complete result must not retain a final report document", path="general_result.json:report_document")

    return GeneralArtifactAudit(
        run_dir=root,
        run_id=run_id,
        execution_config_digest_sha256=digest,
        artifact_integrity_passed=not collector.findings,
        publishable=publishable,
        findings=tuple(collector.findings),
    )


__all__ = [
    "ArtifactAuditFinding",
    "GENERAL_ARTIFACT_AUDIT_SCHEMA_VERSION",
    "GeneralArtifactAudit",
    "audit_general_run_artifact",
]
