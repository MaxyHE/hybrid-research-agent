"""Independent evidence-first control loop for General Research Agent V1.

The module deliberately does not import the legacy Hybrid strategy.  Runtime
adapters may call an LLM, the existing search registry, and the existing full
page fetcher, but must return the typed stage results below.  This keeps the
control decisions (budget, snapshot validation, coverage, and publication)
outside any single prompt or tool implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Callable, Iterable

from .actions import ActionTransition
from .config import GeneralExecutionConfig
from .chunks import (
    EvidenceChunk,
    EvidenceSpan,
    chunk_source,
    select_evidence_chunks,
    span_evidence_chunks,
)
from .coverage import CoverageAuditor
from .plan_adequacy import PlanAdequacyAudit, audit_plan_adequacy
from .schemas import (
    CoverageState,
    EvidenceCard,
    ResearchBrief,
    ResearchMemo,
    ResearchPlan,
    ResearchTask,
    SourceRecord,
    SupervisorDecision,
    validate_research_memo,
    validate_supervisor_decision,
)
from .semantic_audit import (
    SemanticAuditResult,
    SemanticVerdict,
    validate_semantic_audit,
)
from .trace import GeneralAuditEvent
from .writer import (
    CitationAuditResult,
    ReportDocument,
    ReportClaim,
    WriterInput,
    audit_report_citations,
    make_report_document,
    render_report_document,
)


class WorkflowStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INVALID_ARTIFACT = "invalid_artifact"
    RUNTIME_ERROR = "runtime_error"


class WorkflowArtifactError(ValueError):
    """A stage returned data that cannot enter the General evidence ledger."""


class WorkflowBudgetExhausted(RuntimeError):
    """A stage declared work that exceeds the frozen run budget."""


class ModelOutputContractError(RuntimeError):
    """A completed model call failed a typed General stage contract.

    Raw model text may contain user data or untrusted page content, so it is
    intentionally not retained. The completed-call count still crosses this
    boundary, preventing a rejected completion from disappearing from budget
    accounting.
    """

    def __init__(self, *, stage: str, model_calls: int) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        if (
            isinstance(model_calls, bool)
            or not isinstance(model_calls, int)
            or model_calls < 1
        ):
            raise ValueError("model_calls must be a positive integer")
        self.stage = stage.strip()
        self.model_calls = model_calls
        self.reason_code = f"invalid_{self.stage}_structured_output"
        super().__init__(self.reason_code)


@dataclass(frozen=True, slots=True)
class WriterRevisionFeedback:
    """A repair target bound to the exact rejected report claim.

    A semantic review alone identifies a claim by its transient document ID.
    That is not enough for a stateless writer role to repair a report: on the
    next invocation it does not otherwise receive the previous claim text or
    the evidence selection it made. Preserve that bounded context so a repair
    can narrow or remove the rejected claim rather than blindly regenerating
    the whole report.
    """

    claim_id: str
    claim_text: str
    plan_item_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    verdict: SemanticVerdict
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise ValueError("claim_id must be a non-empty string")
        if not isinstance(self.claim_text, str) or not self.claim_text.strip():
            raise ValueError("claim_text must be a non-empty string")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("reason_code must be a non-empty string")
        object.__setattr__(self, "claim_id", self.claim_id.strip())
        object.__setattr__(self, "claim_text", self.claim_text.strip())
        object.__setattr__(self, "plan_item_ids", tuple(self.plan_item_ids))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "verdict", SemanticVerdict(self.verdict))
        object.__setattr__(self, "reason_code", self.reason_code.strip())


class GeneralResearchCancelled(RuntimeError):
    """The application asked the General run to stop at a safe boundary."""


def content_sha256(content: str) -> str:
    """Return the canonical hash recorded for a fetched text snapshot."""

    if not isinstance(content, str) or not content:
        raise WorkflowArtifactError("fetched page content must be a non-empty string")
    return "sha256:" + sha256(content.encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Read-only budget view given to a runtime adapter before it acts."""

    model_calls_used: int
    tool_calls_used: int
    model_calls_remaining: int
    tool_calls_remaining: int


@dataclass(slots=True)
class _BudgetLedger:
    max_model_calls: int
    max_tool_calls: int
    model_calls_used: int = 0
    tool_calls_used: int = 0

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            model_calls_used=self.model_calls_used,
            tool_calls_used=self.tool_calls_used,
            model_calls_remaining=self.max_model_calls - self.model_calls_used,
            tool_calls_remaining=self.max_tool_calls - self.tool_calls_used,
        )

    def consume(self, *, model_calls: int = 0, tool_calls: int = 0) -> None:
        if (
            isinstance(model_calls, bool)
            or not isinstance(model_calls, int)
            or isinstance(tool_calls, bool)
            or not isinstance(tool_calls, int)
            or model_calls < 0
            or tool_calls < 0
        ):
            raise WorkflowArtifactError("stage call counts must be non-negative integers")
        if self.model_calls_used + model_calls > self.max_model_calls:
            raise WorkflowBudgetExhausted("General model-call budget exhausted")
        if self.tool_calls_used + tool_calls > self.max_tool_calls:
            raise WorkflowBudgetExhausted("General tool-call budget exhausted")
        self.model_calls_used += model_calls
        self.tool_calls_used += tool_calls


@dataclass(frozen=True, slots=True)
class PlanStageResult:
    """The only planner output accepted by the workflow."""

    plan: ResearchPlan
    model_calls: int = 1
    brief: ResearchBrief | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ResearchPlan):
            raise TypeError("plan must be ResearchPlan")
        if self.model_calls < 1:
            raise WorkflowArtifactError("planner must account for at least one model call")
        if self.brief is not None and not isinstance(self.brief, ResearchBrief):
            raise TypeError("brief must be ResearchBrief or None")


@dataclass(frozen=True, slots=True)
class SupervisorStageResult:
    """One model-owned, policy-bounded supervisor decision."""

    decision: SupervisorDecision
    model_calls: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.decision, SupervisorDecision):
            raise TypeError("decision must be SupervisorDecision")
        if self.model_calls < 1:
            raise WorkflowArtifactError(
                "supervisor must account for at least one model call"
            )


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """A raw, full-text page that can be used for extractive evidence only."""

    source: SourceRecord
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRecord):
            raise TypeError("source must be SourceRecord")
        actual_hash = content_sha256(self.content)
        if self.source.is_snippet or not self.source.content_verified:
            raise WorkflowArtifactError("General evidence requires a verified full fetch")
        if not self.source.content_artifact_id:
            raise WorkflowArtifactError(
                "General full fetches require a content_artifact_id for replay"
            )
        if self.source.content_hash != actual_hash:
            raise WorkflowArtifactError(
                "SourceRecord.content_hash does not match fetched page content"
            )


@dataclass(frozen=True, slots=True)
class ResearchStageResult:
    """All search/fetch activity from one bounded research pass."""

    pages: tuple[FetchedPage, ...]
    model_calls: int
    tool_calls: int
    worker_count: int = 1
    action_transitions: tuple[ActionTransition, ...] = ()
    controller_terminal_reason: str | None = None
    research_tasks: tuple[ResearchTask, ...] = ()
    task_action_transitions: tuple[tuple[str, tuple[ActionTransition, ...]], ...] = ()

    def __post_init__(self) -> None:
        pages = tuple(self.pages)
        if not all(isinstance(page, FetchedPage) for page in pages):
            raise TypeError("pages must contain FetchedPage objects")
        if self.model_calls < 0 or self.tool_calls < 0:
            raise WorkflowArtifactError("research call counts cannot be negative")
        if isinstance(self.worker_count, bool) or not isinstance(self.worker_count, int):
            raise WorkflowArtifactError("worker_count must be a positive integer")
        if not 1 <= self.worker_count <= 20:
            raise WorkflowArtifactError("worker_count must be between 1 and 20")
        source_ids = [page.source.source_id for page in pages]
        if len(source_ids) != len(set(source_ids)):
            raise WorkflowArtifactError("one research pass cannot return duplicate source_id values")
        object.__setattr__(self, "pages", pages)
        transitions = tuple(self.action_transitions)
        if not all(isinstance(item, ActionTransition) for item in transitions):
            raise TypeError("action_transitions must contain ActionTransition objects")
        steps = [item.step for item in transitions]
        if steps != list(range(len(steps))):
            raise WorkflowArtifactError("action transition steps must start at zero and be contiguous")
        object.__setattr__(self, "action_transitions", transitions)
        if self.controller_terminal_reason is not None and (
            not isinstance(self.controller_terminal_reason, str)
            or not self.controller_terminal_reason.strip()
        ):
            raise WorkflowArtifactError("controller_terminal_reason must be non-empty")
        tasks = tuple(self.research_tasks)
        if not all(isinstance(task, ResearchTask) for task in tasks):
            raise TypeError("research_tasks must contain ResearchTask objects")
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise WorkflowArtifactError("research_tasks must use unique task IDs")
        if tasks and self.worker_count != len(tasks):
            raise WorkflowArtifactError(
                "worker_count must equal the number of dispatched research_tasks"
            )
        object.__setattr__(self, "research_tasks", tasks)
        task_transitions = tuple(self.task_action_transitions)
        normalized_task_transitions: list[tuple[str, tuple[ActionTransition, ...]]] = []
        seen_task_ids: set[str] = set()
        for task_id, task_steps in task_transitions:
            if not isinstance(task_id, str) or not task_id:
                raise WorkflowArtifactError("task transition task_id must be non-empty")
            if task_id in seen_task_ids:
                raise WorkflowArtifactError("task transitions must use unique task IDs")
            if task_id not in set(task_ids):
                raise WorkflowArtifactError(
                    "task transitions must refer to dispatched research_tasks"
                )
            steps_for_task = tuple(task_steps)
            if not all(isinstance(item, ActionTransition) for item in steps_for_task):
                raise TypeError("task transitions must contain ActionTransition objects")
            if [item.step for item in steps_for_task] != list(range(len(steps_for_task))):
                raise WorkflowArtifactError(
                    "each task action transition sequence must start at zero and be contiguous"
                )
            seen_task_ids.add(task_id)
            normalized_task_transitions.append((task_id, steps_for_task))
        object.__setattr__(self, "task_action_transitions", tuple(normalized_task_transitions))


@dataclass(frozen=True, slots=True)
class EvidenceStageResult:
    """Extractive evidence produced against one increment of fetched pages."""

    cards: tuple[EvidenceCard, ...]
    model_calls: int = 1
    rejected_card_count: int = 0
    rejection_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cards = tuple(self.cards)
        if not all(isinstance(card, EvidenceCard) for card in cards):
            raise TypeError("cards must contain EvidenceCard objects")
        if self.model_calls < 1:
            raise WorkflowArtifactError("evidence extraction must account for a model call")
        if (
            isinstance(self.rejected_card_count, bool)
            or not isinstance(self.rejected_card_count, int)
            or self.rejected_card_count < 0
        ):
            raise WorkflowArtifactError(
                "rejected_card_count must be a non-negative integer"
            )
        rejection_codes = tuple(self.rejection_codes)
        if len(rejection_codes) != self.rejected_card_count:
            raise WorkflowArtifactError(
                "rejection_codes must contain one code per rejected card"
            )
        if not all(isinstance(code, str) and code for code in rejection_codes):
            raise WorkflowArtifactError("rejection_codes must contain non-empty strings")
        identifiers = [card.evidence_id for card in cards]
        if len(identifiers) != len(set(identifiers)):
            raise WorkflowArtifactError("one evidence pass cannot return duplicate evidence_id values")
        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "rejection_codes", rejection_codes)


@dataclass(frozen=True, slots=True)
class EvidenceExtractionContext:
    """Frozen, minimal context for one extractor call over selected chunks only."""

    plan: ResearchPlan
    chunks: tuple[EvidenceChunk, ...]
    sources: tuple[SourceRecord, ...]
    max_span_characters: int = 1_000

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ResearchPlan):
            raise TypeError("plan must be ResearchPlan")
        chunks = tuple(self.chunks)
        sources = tuple(self.sources)
        if not chunks or not all(isinstance(chunk, EvidenceChunk) for chunk in chunks):
            raise ValueError("chunks must contain at least one EvidenceChunk")
        if not all(isinstance(source, SourceRecord) for source in sources):
            raise TypeError("sources must contain SourceRecord values")
        if (
            isinstance(self.max_span_characters, bool)
            or not isinstance(self.max_span_characters, int)
            or self.max_span_characters < 1
        ):
            raise ValueError("max_span_characters must be a positive integer")
        source_by_id = {source.source_id: source for source in sources}
        if len(source_by_id) != len(sources):
            raise ValueError("sources must use unique source_id values")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunks must use unique chunk_id values")
        for chunk in chunks:
            source = source_by_id.get(chunk.source_id)
            if source is None or source.content_hash != chunk.source_content_hash:
                raise WorkflowArtifactError(
                    "every evidence chunk must match a source in the extraction context"
                )
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "sources", sources)

    @property
    def chunks_by_id(self) -> dict[str, EvidenceChunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}

    @property
    def sources_by_id(self) -> dict[str, SourceRecord]:
        return {source.source_id: source for source in self.sources}

    @property
    def spans(self) -> tuple[EvidenceSpan, ...]:
        return span_evidence_chunks(
            self.chunks, max_span_characters=self.max_span_characters
        )


@dataclass(frozen=True, slots=True)
class WriteStageResult:
    """A writer result whose claims are the only factual report input."""

    claims: tuple[ReportClaim, ...]
    model_calls: int = 1

    def __post_init__(self) -> None:
        claims = tuple(self.claims)
        if not all(isinstance(claim, ReportClaim) for claim in claims):
            raise TypeError("claims must contain ReportClaim objects")
        if not claims:
            raise WorkflowArtifactError("writer must return at least one ReportClaim")
        if self.model_calls < 1:
            raise WorkflowArtifactError("writer must account for at least one model call")
        object.__setattr__(self, "claims", claims)


@dataclass(frozen=True, slots=True)
class SemanticAuditStageResult:
    """One bounded semantic review over frozen claims and cited excerpts."""

    audit: SemanticAuditResult
    model_calls: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.audit, SemanticAuditResult):
            raise TypeError("audit must be SemanticAuditResult")
        if self.model_calls < 1:
            raise WorkflowArtifactError(
                "semantic audit must account for at least one model call"
            )


Planner = Callable[[BudgetSnapshot], PlanStageResult]
Researcher = Callable[[ResearchPlan, CoverageState | None, BudgetSnapshot], ResearchStageResult]
Supervisor = Callable[
    [ResearchPlan, CoverageState | None, tuple[ResearchMemo, ...], int, BudgetSnapshot],
    SupervisorStageResult,
]
AgenticResearcher = Callable[
    [ResearchPlan, tuple[ResearchTask, ...], CoverageState | None, BudgetSnapshot],
    ResearchStageResult,
]
MemoCompiler = Callable[
    [tuple[ResearchTask, ...], tuple[EvidenceCard, ...], str], tuple[ResearchMemo, ...]
]
EvidenceExtractor = Callable[[EvidenceExtractionContext, BudgetSnapshot], EvidenceStageResult]
Writer = Callable[
    [WriterInput, tuple[WriterRevisionFeedback, ...], BudgetSnapshot], WriteStageResult
]
SemanticAuditor = Callable[
    [WriterInput, ReportDocument, CitationAuditResult, BudgetSnapshot],
    SemanticAuditStageResult,
]


@dataclass(frozen=True, slots=True)
class GeneralWorkflowAdapters:
    """Runtime adapters; each stage must declare its own resource use."""

    planner: Planner
    researcher: Researcher | None
    evidence_extractor: EvidenceExtractor
    writer: Writer
    semantic_auditor: SemanticAuditor | None = None
    supervisor: Supervisor | None = None
    agentic_researcher: AgenticResearcher | None = None
    memo_compiler: MemoCompiler | None = None


@dataclass(frozen=True, slots=True)
class GeneralWorkflowResult:
    """One terminal General run, including evidence artifacts and audit trail."""

    status: WorkflowStatus
    plan: ResearchPlan | None
    sources: tuple[SourceRecord, ...]
    evidence_cards: tuple[EvidenceCard, ...]
    coverage: CoverageState | None
    report_document: ReportDocument | None
    report_markdown: str
    citation_audit: CitationAuditResult | None
    semantic_audit: SemanticAuditResult | None
    events: tuple[GeneralAuditEvent, ...]
    budget: BudgetSnapshot
    reason: str | None = None
    brief: ResearchBrief | None = None
    supervisor_decisions: tuple[SupervisorDecision, ...] = ()
    research_memos: tuple[ResearchMemo, ...] = ()
    plan_adequacy: PlanAdequacyAudit | None = None
    semantic_audit_required: bool = True

    @property
    def is_publishable(self) -> bool:
        base_publishable = (
            self.status == WorkflowStatus.COMPLETE
            and self.plan is not None
            and self.coverage is not None
            and self.coverage.ready_to_stop
            and self.citation_audit is not None
            and self.citation_audit.should_publish
            and self.report_document is not None
        )
        if not base_publishable:
            return False
        if not self.semantic_audit_required:
            return True
        return self.semantic_audit is not None and validate_semantic_audit(
                WriterInput(
                    plan=self.plan,
                    sources=self.sources,
                    evidence_cards=self.evidence_cards,
                ),
                self.report_document,
                self.citation_audit,
                self.semantic_audit,
            )


def _incomplete_report(
    plan: ResearchPlan | None,
    coverage: CoverageState | None,
    reason: str,
) -> str:
    lines = [
        "Research status: a complete evidence-grounded report is not ready.",
        f"Reason: {reason}",
    ]
    if plan is not None:
        lines.append(f"Plan ID: {plan.plan_id}")
    if coverage is not None:
        blockers = ", ".join(coverage.stop_blockers) or "none recorded"
        lines.append(f"Coverage verdict: {coverage.verdict.value}")
        lines.append(f"Coverage blockers: {blockers}")
    return "\n".join(lines)


class GeneralResearchWorkflow:
    """Execute Plan → Evidence → Coverage → Write using immutable artifacts."""

    def __init__(
        self,
        config: GeneralExecutionConfig,
        adapters: GeneralWorkflowAdapters,
        *,
        should_cancel: Callable[[], bool] | None = None,
        event_observer: Callable[[GeneralAuditEvent], None] | None = None,
    ) -> None:
        if not isinstance(config, GeneralExecutionConfig):
            raise TypeError("config must be GeneralExecutionConfig")
        if not isinstance(adapters, GeneralWorkflowAdapters):
            raise TypeError("adapters must be GeneralWorkflowAdapters")
        self.config = config
        self.adapters = adapters
        if should_cancel is not None and not callable(should_cancel):
            raise TypeError("should_cancel must be callable or None")
        if event_observer is not None and not callable(event_observer):
            raise TypeError("event_observer must be callable or None")
        self._should_cancel = should_cancel
        self._event_observer = event_observer
        if config.orchestration is None and (
            config.workers.allow_subagents or config.workers.max_research_workers != 1
        ):
            raise ValueError("serial General V1 requires one research worker")
        if config.orchestration is None and adapters.researcher is None:
            raise ValueError("serial General V1 requires a researcher adapter")
        if config.orchestration is not None and (
            adapters.supervisor is None
            or adapters.agentic_researcher is None
            or adapters.memo_compiler is None
        ):
            raise ValueError(
                "agentic orchestration requires supervisor, parallel researcher, and memo compiler"
            )
        if config.workers.allow_external_mcp:
            raise ValueError("GeneralResearchWorkflow does not permit external MCP")

    def _ensure_not_cancelled(self) -> None:
        if self._should_cancel is not None and self._should_cancel():
            raise GeneralResearchCancelled("general_research_cancelled")

    def _coverage_auditor(self) -> CoverageAuditor:
        now = _timestamp()
        return CoverageAuditor(
            as_of=now[:10],
            audited_at=now,
            source_policy_version=self.config.run.source_policy_version,
            coverage_policy_version=self.config.run.coverage_policy_version,
        )

    def _event(
        self,
        events: list[GeneralAuditEvent],
        factory: Callable[..., GeneralAuditEvent],
        **kwargs: object,
    ) -> None:
        event = factory(
            run_id=self.config.run.run_id,
            event_id=len(events),
            **kwargs,
        )
        events.append(event)
        # Observability is deliberately non-authoritative: a dashboard/socket
        # failure must not change evidence, coverage, or publication.  The
        # immutable event list and JSONL artifact remain the source of truth.
        if self._event_observer is not None:
            try:
                self._event_observer(event)
            except Exception:
                pass

    def _validate_plan(self, plan: ResearchPlan) -> None:
        if plan.run_id != self.config.run.run_id:
            raise WorkflowArtifactError("ResearchPlan.run_id must match GeneralRunConfig")
        if plan.query != self.config.run.query:
            raise WorkflowArtifactError("ResearchPlan.query must match GeneralRunConfig")

    def _validate_brief(self, brief: ResearchBrief) -> None:
        if brief.run_id != self.config.run.run_id:
            raise WorkflowArtifactError("ResearchBrief.run_id must match GeneralRunConfig")
        if brief.user_query != self.config.run.query:
            raise WorkflowArtifactError("ResearchBrief.user_query must match GeneralRunConfig")

    def _synthesis_model_reserve(self) -> int:
        """Reserve extractor plus worst-case writer/auditor repair calls.

        Dispatch workers receive only the remainder.  This prevents an
        apparently successful supervisor fan-out from consuming the complete
        run envelope before evidence extraction or synthesis can happen.
        """

        if not self.config.citation_policy.require_post_synthesis_audit:
            return 1 + self.config.citation_policy.max_writer_repairs + 1
        return 1 + 2 * (self.config.citation_policy.max_writer_repairs + 1)

    @staticmethod
    def _agentic_recovery_reserve(
        plan: ResearchPlan,
        tasks: tuple[ResearchTask, ...],
        coverage: CoverageState | None,
    ) -> tuple[int, int, tuple[str, ...]]:
        """Reserve one bounded follow-up before a fan-out spends its envelope.

        A first dispatch can intentionally defer an obligation because it
        depends on earlier findings; even a fully parallel dispatch can lose a
        card to a fetch failure or extraction miss.  Reserve the minimum for a
        supervisor handoff, one search/fetch pair per recoverable task, and a
        second extraction pass.  The caller may decline the reserve when its
        envelope cannot still give every current worker one action.
        """

        assigned_item_ids = {
            item_id for task in tasks for item_id in task.plan_item_ids
        }
        if coverage is None:
            future_item_ids = tuple(
                item.item_id for item in plan.items if item.item_id not in assigned_item_ids
            )
            # Keep one recovery opportunity even when the first fan-out covers
            # every item: a failed fetch or rejected extraction is observable
            # only after that pass finishes.
            recovery_task_count = max(1, len(future_item_ids))
        else:
            future_item_ids = tuple(
                decision.plan_item_id
                for decision in coverage.decisions
                if not decision.is_sufficient
                and decision.plan_item_id not in assigned_item_ids
            )
            recovery_task_count = len(future_item_ids)
        recovery_task_count = min(
            recovery_task_count,
            len(plan.items),
        )
        if recovery_task_count == 0:
            return 0, 0, future_item_ids
        # One future supervisor + one next-pass extractor + two controller
        # calls (search, fetch) per recovery task.
        return 2 * recovery_task_count + 2, 2 * recovery_task_count, future_item_ids

    @staticmethod
    def _citation_revision_feedback(
        writer_input: WriterInput,
        document: ReportDocument,
        citation_audit: CitationAuditResult,
    ) -> tuple[WriterRevisionFeedback, ...]:
        """Turn deterministic citation gaps into one bounded writer repair.

        This is deliberately not a second semantic judge.  It tells the
        writer which existing claim has an invalid binding and, crucially,
        which evidence-backed plan item it omitted altogether.
        """

        claims_by_id = {claim.claim_id: claim for claim in document.claims}
        feedback: list[WriterRevisionFeedback] = []
        for claim_audit in citation_audit.claim_audits:
            if claim_audit.valid:
                continue
            claim = claims_by_id[claim_audit.claim_id]
            feedback.append(
                WriterRevisionFeedback(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    plan_item_ids=claim.plan_item_ids,
                    evidence_ids=claim.evidence_ids,
                    verdict=SemanticVerdict.PARTIAL,
                    reason_code=(
                        claim_audit.coverage_gaps[0].code
                        if claim_audit.coverage_gaps
                        else "citation_binding_invalid"
                    ),
                )
            )

        plan_items_by_id = {item.item_id: item for item in writer_input.plan.items}
        evidence_by_item: dict[str, tuple[str, ...]] = {
            item.item_id: tuple(
                card.evidence_id
                for card in writer_input.evidence_cards
                if item.item_id in card.plan_item_ids
            )
            for item in writer_input.plan.items
        }
        missing_item_ids = {
            gap.plan_item_id
            for gap in citation_audit.coverage_gaps
            if gap.code == "planned_item_not_cited" and gap.plan_item_id is not None
        }
        for item_id in sorted(missing_item_ids):
            evidence_ids = evidence_by_item[item_id]
            if not evidence_ids:
                continue
            feedback.append(
                WriterRevisionFeedback(
                    claim_id=f"missing-{item_id}",
                    claim_text=(
                        "Add a grounded report claim for: "
                        + plan_items_by_id[item_id].question
                    ),
                    plan_item_ids=(item_id,),
                    evidence_ids=evidence_ids,
                    verdict=SemanticVerdict.PARTIAL,
                    reason_code="planned_item_not_cited",
                )
            )
        return tuple(feedback)

    @staticmethod
    def _validate_cards(
        cards: Iterable[EvidenceCard],
        *,
        plan: ResearchPlan,
        pages_by_source_id: dict[str, FetchedPage],
        chunks_by_id: dict[str, EvidenceChunk],
        existing_evidence_ids: set[str],
    ) -> tuple[EvidenceCard, ...]:
        plan_item_ids = {item.item_id for item in plan.items}
        accepted: list[EvidenceCard] = []
        seen_ids = set(existing_evidence_ids)
        for card in cards:
            if card.evidence_id in seen_ids:
                raise WorkflowArtifactError(f"duplicate evidence_id: {card.evidence_id}")
            unknown_plan_items = set(card.plan_item_ids).difference(plan_item_ids)
            if unknown_plan_items:
                raise WorkflowArtifactError(
                    "EvidenceCard refers to unknown plan items: "
                    + ", ".join(sorted(unknown_plan_items))
                )
            page = pages_by_source_id.get(card.source_id)
            if page is None:
                raise WorkflowArtifactError(
                    "EvidenceCard must cite a page fetched in this General run"
                )
            if card.source_content_hash != page.source.content_hash:
                raise WorkflowArtifactError(
                    "EvidenceCard source_content_hash does not match fetched snapshot"
                )
            chunk = chunks_by_id.get(card.evidence_chunk_id)
            if chunk is None:
                raise WorkflowArtifactError(
                    "EvidenceCard must cite a chunk selected for this General run"
                )
            if (
                chunk.source_id != card.source_id
                or chunk.source_content_hash != card.source_content_hash
            ):
                raise WorkflowArtifactError(
                    "EvidenceCard evidence_chunk_id does not match its source snapshot"
                )
            if card.quote_start < chunk.start or card.quote_end > chunk.end:
                raise WorkflowArtifactError(
                    "EvidenceCard quote offsets exceed its selected evidence chunk"
                )
            if (
                chunk.content[
                    card.quote_start - chunk.start : card.quote_end - chunk.start
                ]
                != card.verbatim_quote
            ):
                raise WorkflowArtifactError(
                    "EvidenceCard verbatim_quote does not match selected evidence chunk"
                )
            if page.content[card.quote_start : card.quote_end] != card.verbatim_quote:
                raise WorkflowArtifactError(
                    "EvidenceCard verbatim_quote does not match fetched content offsets"
                )
            accepted.append(card)
            seen_ids.add(card.evidence_id)
        return tuple(accepted)

    def _compile_agentic_memos(
        self,
        *,
        events: list[GeneralAuditEvent],
        tasks: tuple[ResearchTask, ...],
        evidence_cards: tuple[EvidenceCard, ...],
        sources: Iterable[SourceRecord],
    ) -> tuple[ResearchMemo, ...]:
        """Build, validate, and audit one memo for every dispatched task."""

        if self.adapters.memo_compiler is None:
            raise WorkflowArtifactError("agentic orchestration requires a memo compiler")
        round_memos = self.adapters.memo_compiler(
            tasks,
            evidence_cards,
            _timestamp(),
        )
        if len(round_memos) != len(tasks):
            raise WorkflowArtifactError(
                "memo compiler must return exactly one memo per task"
            )
        tasks_by_id = {task.task_id: task for task in tasks}
        for memo in round_memos:
            task = tasks_by_id.get(memo.task_id)
            if task is None:
                raise WorkflowArtifactError(
                    "memo compiler returned a memo for an undispatched task"
                )
            validate_research_memo(memo, task, evidence_cards, sources)
            self._event(
                events,
                GeneralAuditEvent.memo,
                task_id=memo.task_id,
                evidence_ids=[
                    evidence_id
                    for finding in memo.findings
                    for evidence_id in finding.evidence_ids
                ],
                unresolved_count=len(memo.unresolved_questions),
            )
        return round_memos

    def run(self) -> GeneralWorkflowResult:
        ledger = _BudgetLedger(
            max_model_calls=self.config.run.max_model_calls,
            max_tool_calls=self.config.run.max_tool_calls,
        )
        events: list[GeneralAuditEvent] = []
        plan: ResearchPlan | None = None
        brief: ResearchBrief | None = None
        supervisor_decisions: list[SupervisorDecision] = []
        research_memos: list[ResearchMemo] = []
        plan_adequacy: PlanAdequacyAudit | None = None
        source_by_id: dict[str, SourceRecord] = {}
        pages_by_source_id: dict[str, FetchedPage] = {}
        evidence_cards: list[EvidenceCard] = []
        coverage: CoverageState | None = None

        try:
            self._ensure_not_cancelled()
            plan_stage = self.adapters.planner(ledger.snapshot())
            ledger.consume(model_calls=plan_stage.model_calls)
            plan = plan_stage.plan
            self._validate_plan(plan)
            brief = plan_stage.brief
            if self.config.orchestration is not None and brief is None:
                raise WorkflowArtifactError(
                    "agentic orchestration requires a typed research brief"
                )
            if brief is not None:
                self._validate_brief(brief)
                if self.config.orchestration is not None and not brief.requirements:
                    raise WorkflowArtifactError(
                        "agentic research brief requires explicit requirements"
                    )
                plan_adequacy = audit_plan_adequacy(brief, plan)
                if not plan_adequacy.is_adequate:
                    raise WorkflowArtifactError(
                        "required brief requirements are absent from the research plan: "
                        + ", ".join(plan_adequacy.missing_required_requirement_ids)
                    )
                self._event(
                    events,
                    GeneralAuditEvent.brief,
                    brief_id=brief.brief_id,
                )
            self._event(
                events,
                GeneralAuditEvent.plan,
                plan_id=plan.plan_id,
                claim_ids=[item.item_id for item in plan.items],
                requirement_ids=(
                    [requirement.requirement_id for requirement in brief.requirements]
                    if brief is not None
                    else []
                ),
            )

            max_research_rounds = (
                self.config.orchestration.max_supervisor_rounds
                if self.config.orchestration is not None
                else self.config.research_control.max_research_passes
            )
            for pass_index in range(max_research_rounds):
                self._ensure_not_cancelled()
                dispatched_tasks: tuple[ResearchTask, ...] = ()
                if self.config.orchestration is None:
                    assert self.adapters.researcher is not None
                    research = self.adapters.researcher(
                        plan, coverage, ledger.snapshot()
                    )
                else:
                    assert self.adapters.supervisor is not None
                    assert self.adapters.agentic_researcher is not None
                    supervision = self.adapters.supervisor(
                        plan,
                        coverage,
                        tuple(research_memos),
                        pass_index,
                        ledger.snapshot(),
                    )
                    ledger.consume(model_calls=supervision.model_calls)
                    decision = supervision.decision
                    validate_supervisor_decision(decision, plan)
                    prior_task_ids = {
                        task.task_id
                        for earlier in supervisor_decisions
                        for task in earlier.tasks
                    }
                    duplicate_task_ids = prior_task_ids.intersection(
                        task.task_id for task in decision.tasks
                    )
                    if duplicate_task_ids:
                        raise WorkflowArtifactError(
                            "supervisor task IDs must be unique across one run: "
                            + ", ".join(sorted(duplicate_task_ids))
                        )
                    supervisor_decisions.append(decision)
                    recovery_model_reserve = 0
                    recovery_tool_reserve = 0
                    recovery_item_ids: tuple[str, ...] = ()
                    recovery_reservation_applied = False
                    if not decision.should_finish:
                        (
                            recovery_model_reserve,
                            recovery_tool_reserve,
                            recovery_item_ids,
                        ) = self._agentic_recovery_reserve(
                            plan, decision.tasks, coverage
                        )
                        worker_snapshot = ledger.snapshot()
                        available_worker_models = (
                            worker_snapshot.model_calls_remaining
                            - self._synthesis_model_reserve()
                        )
                        available_worker_tools = worker_snapshot.tool_calls_remaining
                        if (
                            available_worker_models - recovery_model_reserve
                            >= len(decision.tasks)
                            and available_worker_tools - recovery_tool_reserve
                            >= len(decision.tasks)
                        ):
                            recovery_reservation_applied = (
                                recovery_model_reserve > 0
                                or recovery_tool_reserve > 0
                            )
                        else:
                            recovery_model_reserve = 0
                            recovery_tool_reserve = 0
                    self._event(
                        events,
                        GeneralAuditEvent.supervision,
                        round_index=decision.round_index,
                        task_ids=[task.task_id for task in decision.tasks],
                        should_finish=decision.should_finish,
                        recovery_model_reserve=recovery_model_reserve,
                        recovery_tool_reserve=recovery_tool_reserve,
                        recovery_item_ids=list(recovery_item_ids),
                        recovery_reservation_applied=recovery_reservation_applied,
                    )
                    if decision.should_finish:
                        break
                    dispatched_tasks = decision.tasks
                    # Workers can never consume the calls needed for the
                    # extractor and the bounded writer/auditor repair loop.
                    worker_snapshot = ledger.snapshot()
                    worker_model_budget = (
                        worker_snapshot.model_calls_remaining
                        - self._synthesis_model_reserve()
                        - recovery_model_reserve
                    )
                    worker_tool_budget = (
                        worker_snapshot.tool_calls_remaining - recovery_tool_reserve
                    )
                    if (
                        worker_model_budget < len(dispatched_tasks)
                        or worker_tool_budget < len(dispatched_tasks)
                    ):
                        research_memos.extend(
                            self._compile_agentic_memos(
                                events=events,
                                tasks=dispatched_tasks,
                                evidence_cards=tuple(evidence_cards),
                                sources=source_by_id.values(),
                            )
                        )
                        break
                    research = self.adapters.agentic_researcher(
                        plan,
                        dispatched_tasks,
                        coverage,
                        BudgetSnapshot(
                            model_calls_used=worker_snapshot.model_calls_used,
                            tool_calls_used=worker_snapshot.tool_calls_used,
                            model_calls_remaining=worker_model_budget,
                            tool_calls_remaining=worker_tool_budget,
                        ),
                    )
                ledger.consume(
                    model_calls=research.model_calls, tool_calls=research.tool_calls
                )
                if research.task_action_transitions:
                    transition_groups = research.task_action_transitions
                else:
                    transition_groups = ((None, research.action_transitions),)
                for task_id, transitions in transition_groups:
                    for transition in transitions:
                        self._event(
                            events,
                            GeneralAuditEvent.action,
                            step=transition.step,
                            action_type=transition.action_type.value,
                            outcome=transition.outcome.value,
                            connector_id=transition.connector_id,
                            candidate_id=transition.candidate_id,
                            query_sha256=transition.query_sha256,
                            candidate_count=transition.candidate_count,
                            reason=transition.reason,
                            error_code=transition.error_code,
                            research_pass=pass_index,
                            task_id=task_id,
                        )
                if not research.pages:
                    if self.config.orchestration is not None:
                        research_memos.extend(
                            self._compile_agentic_memos(
                                events=events,
                                tasks=dispatched_tasks,
                                evidence_cards=tuple(evidence_cards),
                                sources=source_by_id.values(),
                            )
                        )
                        continue
                    break
                new_pages: list[FetchedPage] = []
                for page in research.pages:
                    existing = pages_by_source_id.get(page.source.source_id)
                    if existing is not None and existing.source.content_hash != page.source.content_hash:
                        raise WorkflowArtifactError(
                            "source_id cannot identify two different fetched snapshots"
                        )
                    if existing is None:
                        pages_by_source_id[page.source.source_id] = page
                        source_by_id[page.source.source_id] = page.source
                        new_pages.append(page)
                        self._event(
                            events,
                            GeneralAuditEvent.source,
                            source_id=page.source.source_id,
                            url=page.source.url,
                            source_class=page.source.source_class.value,
                            source_channel=page.source.source_channel,
                            source_connector_id=page.source.source_connector_id,
                            fetch_status="success",
                            content_hash=page.source.content_hash,
                            content_artifact_id=page.source.content_artifact_id,
                        )

                # Every immutable source snapshot enters extraction once.  A
                # later worker may deliberately revisit it, but re-extracting
                # it would make a normal multi-round run fail on duplicate
                # evidence IDs.  Coverage remains cumulative and is always
                # re-evaluated below.
                accepted_cards: tuple[EvidenceCard, ...] = ()
                if new_pages:
                    all_chunks = tuple(
                        chunk
                        for page in new_pages
                        for chunk in chunk_source(
                            page.source, page.content, self.config.evidence_extraction
                        )
                    )
                    selected_chunks = select_evidence_chunks(
                        plan, all_chunks, self.config.evidence_extraction
                    )
                    extraction_context = EvidenceExtractionContext(
                        plan=plan,
                        chunks=selected_chunks,
                        sources=tuple(
                            {
                                chunk.source_id: source_by_id[chunk.source_id]
                                for chunk in selected_chunks
                                if chunk.source_id in source_by_id
                            }.values()
                        ),
                        max_span_characters=self.config.evidence_extraction.max_span_characters,
                    )
                    self._ensure_not_cancelled()
                    extraction = self.adapters.evidence_extractor(
                        extraction_context, ledger.snapshot()
                    )
                    ledger.consume(model_calls=extraction.model_calls)
                    accepted_cards = self._validate_cards(
                        extraction.cards,
                        plan=plan,
                        pages_by_source_id=pages_by_source_id,
                        chunks_by_id=extraction_context.chunks_by_id,
                        existing_evidence_ids={card.evidence_id for card in evidence_cards},
                    )
                evidence_cards.extend(accepted_cards)
                for card in accepted_cards:
                    support = "refutes" if card.stance.value == "contradicts" else "supports"
                    self._event(
                        events,
                        GeneralAuditEvent.evidence,
                        evidence_id=card.evidence_id,
                        claim_id=card.plan_item_ids[0],
                        source_id=card.source_id,
                        excerpt=card.verbatim_quote,
                        support=support,
                        quote_start=card.quote_start,
                        quote_end=card.quote_end,
                        evidence_chunk_id=card.evidence_chunk_id,
                        source_content_hash=card.source_content_hash,
                    )
                coverage = self._coverage_auditor().audit(
                    plan,
                    sources=tuple(source_by_id.values()),
                    cards=tuple(evidence_cards),
                )
                next_decision = (
                    "synthesize"
                    if coverage.ready_to_stop
                    else "continue"
                    if pass_index + 1 < max_research_rounds
                    else "blocked"
                )
                self._event(
                    events,
                    GeneralAuditEvent.coverage,
                    total_claims=len(plan.items),
                    covered_claims=sum(
                        decision.is_sufficient for decision in coverage.decisions
                    ),
                    uncovered_claim_ids=[
                        decision.plan_item_id
                        for decision in coverage.decisions
                        if not decision.is_sufficient
                    ],
                    decision=next_decision,
                    verdict=coverage.verdict.value,
                    extraction_rejected_card_count=(
                        extraction.rejected_card_count if new_pages else 0
                    ),
                    extraction_rejection_codes=(
                        list(extraction.rejection_codes) if new_pages else []
                    ),
                )
                if self.config.orchestration is not None:
                    research_memos.extend(
                        self._compile_agentic_memos(
                            events=events,
                            tasks=dispatched_tasks,
                            evidence_cards=tuple(evidence_cards),
                            sources=source_by_id.values(),
                        )
                    )
                if coverage.ready_to_stop:
                    break

            if coverage is None or not coverage.ready_to_stop:
                return GeneralWorkflowResult(
                    status=WorkflowStatus.INCOMPLETE,
                    plan=plan,
                    sources=tuple(source_by_id.values()),
                    evidence_cards=tuple(evidence_cards),
                    coverage=coverage,
                    report_document=None,
                    report_markdown=_incomplete_report(
                        plan, coverage, "coverage requirements were not met"
                    ),
                    citation_audit=None,
                    semantic_audit=None,
                    events=tuple(events),
                    budget=ledger.snapshot(),
                    reason="coverage_requirements_not_met",
                    brief=brief,
                    supervisor_decisions=tuple(supervisor_decisions),
                    research_memos=tuple(research_memos),
                    plan_adequacy=plan_adequacy,
                )

            writer_input = WriterInput(
                plan=plan,
                sources=tuple(source_by_id.values()),
                evidence_cards=tuple(evidence_cards),
            )
            revision_feedback: tuple[WriterRevisionFeedback, ...] = ()
            for writer_attempt in range(
                self.config.citation_policy.max_writer_repairs + 1
            ):
                self._ensure_not_cancelled()
                writing = self.adapters.writer(
                    writer_input, revision_feedback, ledger.snapshot()
                )
                ledger.consume(model_calls=writing.model_calls)
                report_document = make_report_document(writer_input, writing.claims)
                citation_audit = audit_report_citations(
                    writer_input, report_document.claims
                )
                if not citation_audit.passed:
                    self._event(
                        events,
                        GeneralAuditEvent.citation_audit,
                        citation_count=len(citation_audit.citation_traces),
                        supported_citation_count=0,
                        unsupported_citation_ids=[
                            gap.evidence_id
                            for gap in citation_audit.coverage_gaps
                            if gap.evidence_id is not None
                        ],
                        verdict="fail",
                        writer_input_fingerprint=citation_audit.writer_input_fingerprint,
                        writer_attempt=writer_attempt,
                        semantic_audit_status="not_run",
                    )
                    if writer_attempt < self.config.citation_policy.max_writer_repairs:
                        revision_feedback = self._citation_revision_feedback(
                            writer_input, report_document, citation_audit
                        )
                        if revision_feedback:
                            continue
                    return GeneralWorkflowResult(
                        status=WorkflowStatus.INCOMPLETE,
                        plan=plan,
                        sources=tuple(source_by_id.values()),
                        evidence_cards=tuple(evidence_cards),
                        coverage=coverage,
                        report_document=None,
                        report_markdown=_incomplete_report(
                            plan,
                            coverage,
                            "citation audit rejected the writer output",
                        ),
                        citation_audit=citation_audit,
                        semantic_audit=None,
                        events=tuple(events),
                        budget=ledger.snapshot(),
                        reason="citation_audit_failed",
                        brief=brief,
                        supervisor_decisions=tuple(supervisor_decisions),
                        research_memos=tuple(research_memos),
                        plan_adequacy=plan_adequacy,
                    )

                if not self.config.citation_policy.require_post_synthesis_audit:
                    self._event(
                        events,
                        GeneralAuditEvent.citation_audit,
                        citation_count=len(citation_audit.citation_traces),
                        supported_citation_count=len(citation_audit.citation_traces),
                        unsupported_citation_ids=[],
                        verdict="pass",
                        writer_input_fingerprint=citation_audit.writer_input_fingerprint,
                        writer_attempt=writer_attempt,
                        semantic_audit_status="not_requested",
                    )
                    return GeneralWorkflowResult(
                        status=WorkflowStatus.COMPLETE,
                        plan=plan,
                        sources=tuple(source_by_id.values()),
                        evidence_cards=tuple(evidence_cards),
                        coverage=coverage,
                        report_document=report_document,
                        report_markdown=render_report_document(
                            writer_input, report_document, citation_audit
                        ),
                        citation_audit=citation_audit,
                        semantic_audit=None,
                        events=tuple(events),
                        budget=ledger.snapshot(),
                        brief=brief,
                        supervisor_decisions=tuple(supervisor_decisions),
                        research_memos=tuple(research_memos),
                        plan_adequacy=plan_adequacy,
                        semantic_audit_required=False,
                    )

                if self.adapters.semantic_auditor is None:
                    raise WorkflowArtifactError(
                        "semantic auditor is required by the configured citation policy"
                    )
                self._ensure_not_cancelled()
                semantic_stage = self.adapters.semantic_auditor(
                    writer_input, report_document, citation_audit, ledger.snapshot()
                )
                ledger.consume(model_calls=semantic_stage.model_calls)
                semantic_audit = semantic_stage.audit
                semantic_passed = validate_semantic_audit(
                    writer_input, report_document, citation_audit, semantic_audit
                )
                self._event(
                    events,
                    GeneralAuditEvent.citation_audit,
                    citation_count=len(citation_audit.citation_traces),
                    supported_citation_count=(
                        len(citation_audit.citation_traces) if semantic_passed else 0
                    ),
                    unsupported_citation_ids=[
                        gap.evidence_id
                        for gap in citation_audit.coverage_gaps
                        if gap.evidence_id is not None
                    ],
                    verdict="pass" if semantic_passed else "fail",
                    writer_input_fingerprint=citation_audit.writer_input_fingerprint,
                    writer_attempt=writer_attempt,
                    semantic_audit=semantic_audit.to_dict(),
                )
                if semantic_passed:
                    return GeneralWorkflowResult(
                        status=WorkflowStatus.COMPLETE,
                        plan=plan,
                        sources=tuple(source_by_id.values()),
                        evidence_cards=tuple(evidence_cards),
                        coverage=coverage,
                        report_document=report_document,
                        report_markdown=render_report_document(
                            writer_input, report_document, citation_audit
                        ),
                        citation_audit=citation_audit,
                        semantic_audit=semantic_audit,
                        events=tuple(events),
                        budget=ledger.snapshot(),
                        brief=brief,
                        supervisor_decisions=tuple(supervisor_decisions),
                        research_memos=tuple(research_memos),
                        plan_adequacy=plan_adequacy,
                    )
                claims_by_id = {
                    claim.claim_id: claim for claim in report_document.claims
                }
                revision_feedback = tuple(
                    WriterRevisionFeedback(
                        claim_id=review.claim_id,
                        claim_text=claims_by_id[review.claim_id].text,
                        plan_item_ids=claims_by_id[review.claim_id].plan_item_ids,
                        evidence_ids=claims_by_id[review.claim_id].evidence_ids,
                        verdict=review.verdict,
                        reason_code=review.reason_code,
                    )
                    for review in semantic_audit.reviews
                    if review.verdict != SemanticVerdict.SUPPORTED
                )

            return GeneralWorkflowResult(
                status=WorkflowStatus.INCOMPLETE,
                plan=plan,
                sources=tuple(source_by_id.values()),
                evidence_cards=tuple(evidence_cards),
                coverage=coverage,
                report_document=None,
                report_markdown=_incomplete_report(
                    plan, coverage, "semantic audit rejected the writer output"
                ),
                citation_audit=citation_audit,
                semantic_audit=semantic_audit,
                events=tuple(events),
                budget=ledger.snapshot(),
                reason="semantic_audit_failed",
                brief=brief,
                supervisor_decisions=tuple(supervisor_decisions),
                research_memos=tuple(research_memos),
                plan_adequacy=plan_adequacy,
            )
        except GeneralResearchCancelled:
            # Cancellation is lifecycle control, not an incomplete or failed
            # research result. The product boundary maps it to its existing
            # SUSPENDED cleanup path and intentionally persists no report.
            raise
        except WorkflowBudgetExhausted as exc:
            return GeneralWorkflowResult(
                status=WorkflowStatus.BUDGET_EXHAUSTED,
                plan=plan,
                sources=tuple(source_by_id.values()),
                evidence_cards=tuple(evidence_cards),
                coverage=coverage,
                report_document=None,
                report_markdown=_incomplete_report(plan, coverage, str(exc)),
                citation_audit=None,
                semantic_audit=None,
                events=tuple(events),
                budget=ledger.snapshot(),
                reason=str(exc),
                brief=brief,
                supervisor_decisions=tuple(supervisor_decisions),
                research_memos=tuple(research_memos),
                plan_adequacy=plan_adequacy,
            )
        except WorkflowArtifactError as exc:
            return GeneralWorkflowResult(
                status=WorkflowStatus.INVALID_ARTIFACT,
                plan=plan,
                sources=tuple(source_by_id.values()),
                evidence_cards=tuple(evidence_cards),
                coverage=coverage,
                report_document=None,
                report_markdown=_incomplete_report(plan, coverage, str(exc)),
                citation_audit=None,
                semantic_audit=None,
                events=tuple(events),
                budget=ledger.snapshot(),
                reason=str(exc),
                brief=brief,
                supervisor_decisions=tuple(supervisor_decisions),
                research_memos=tuple(research_memos),
                plan_adequacy=plan_adequacy,
            )
        except ModelOutputContractError as exc:
            # A completion arrived from the provider, but the typed boundary
            # rejected it. Charge that completed call and terminate safely;
            # neither prompt nor raw completion is persisted in artifacts.
            try:
                ledger.consume(model_calls=exc.model_calls)
            except WorkflowBudgetExhausted as budget_error:
                return GeneralWorkflowResult(
                    status=WorkflowStatus.BUDGET_EXHAUSTED,
                    plan=plan,
                    sources=tuple(source_by_id.values()),
                    evidence_cards=tuple(evidence_cards),
                    coverage=coverage,
                    report_document=None,
                    report_markdown=_incomplete_report(plan, coverage, str(budget_error)),
                    citation_audit=None,
                    semantic_audit=None,
                    events=tuple(events),
                    budget=ledger.snapshot(),
                    reason="model_output_contract_error_after_budget_exhaustion",
                    brief=brief,
                    supervisor_decisions=tuple(supervisor_decisions),
                    research_memos=tuple(research_memos),
                    plan_adequacy=plan_adequacy,
                )
            reason = f"runtime_stage_error:{exc.reason_code}"
            return GeneralWorkflowResult(
                status=WorkflowStatus.RUNTIME_ERROR,
                plan=plan,
                sources=tuple(source_by_id.values()),
                evidence_cards=tuple(evidence_cards),
                coverage=coverage,
                report_document=None,
                report_markdown=_incomplete_report(plan, coverage, reason),
                citation_audit=None,
                semantic_audit=None,
                events=tuple(events),
                budget=ledger.snapshot(),
                reason=reason,
                brief=brief,
                supervisor_decisions=tuple(supervisor_decisions),
                research_memos=tuple(research_memos),
                plan_adequacy=plan_adequacy,
            )
        except Exception as exc:
            # Adapter failures must not turn into an uncited best-effort
            # answer or leak a provider/tool exception into the report. The
            # concrete exception belongs in the application log at the
            # integration boundary; this core exposes only a stable reason.
            reason = f"runtime_stage_error:{type(exc).__name__}"
            return GeneralWorkflowResult(
                status=WorkflowStatus.RUNTIME_ERROR,
                plan=plan,
                sources=tuple(source_by_id.values()),
                evidence_cards=tuple(evidence_cards),
                coverage=coverage,
                report_document=None,
                report_markdown=_incomplete_report(plan, coverage, reason),
                citation_audit=None,
                semantic_audit=None,
                events=tuple(events),
                budget=ledger.snapshot(),
                reason=reason,
                brief=brief,
                supervisor_decisions=tuple(supervisor_decisions),
                research_memos=tuple(research_memos),
                plan_adequacy=plan_adequacy,
            )


__all__ = [
    "BudgetSnapshot",
    "EvidenceStageResult",
    "EvidenceExtractionContext",
    "FetchedPage",
    "GeneralResearchWorkflow",
    "GeneralResearchCancelled",
    "ModelOutputContractError",
    "GeneralWorkflowAdapters",
    "GeneralWorkflowResult",
    "PlanStageResult",
    "ResearchStageResult",
    "SemanticAuditStageResult",
    "SupervisorStageResult",
    "WorkflowArtifactError",
    "WorkflowBudgetExhausted",
    "WorkflowStatus",
    "WriteStageResult",
    "content_sha256",
]
