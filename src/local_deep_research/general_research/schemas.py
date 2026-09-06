"""Versioned, immutable data contracts for General Research Agent v1.

The contracts in this module intentionally contain observations and declared
links only.  They do not make semantic claims about whether a quoted passage
*actually* entails a plan item; that is a separate verifier/evaluation
responsibility.  Keeping that boundary explicit prevents a deterministic
coverage counter from being mistaken for a factuality oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Iterable


GENERAL_RESEARCH_SCHEMA_VERSION = "general-research/v1"


class SourceClass(StrEnum):
    """Explicit source classes; authority is never inferred from a TLD."""

    PRIMARY = "primary"
    GOVERNMENT = "government"
    REGULATOR = "regulator"
    STANDARDS_BODY = "standards_body"
    ACADEMIC = "academic"
    OFFICIAL = "official"
    INDUSTRY = "industry"
    NEWS = "news"
    SECONDARY = "secondary"
    COMMUNITY = "community"
    SEARCH_SNIPPET = "search_snippet"
    UNKNOWN = "unknown"


class SourceClassificationBasis(StrEnum):
    """How a source class was assigned.

    A model's reading of a page is not sufficient evidence that the publisher
    is official, primary, or authoritative. General V1 therefore preserves
    the basis for every non-unknown class.
    """

    UNVERIFIED = "unverified"
    HOST_ALLOWLIST = "host_allowlist"
    PUBLISHER_REGISTRY = "publisher_registry"
    MANUAL_REVIEW = "manual_review"


class SourceChannel(StrEnum):
    """Data boundary through which a source entered a General run.

    Source channel is deliberately separate from ``SourceClass``.  A local
    collection document might be a primary paper, while a public-web page
    might be official.  Keeping the two axes separate prevents a report from
    falsely presenting private collection material as open-web corroboration.
    """

    PUBLIC_WEB = "public_web"
    LOCAL_COLLECTION = "local_collection"


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"


class CoverageStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"
    STALE = "stale"
    LOW_QUALITY = "low_quality"


class CoverageVerdict(StrEnum):
    """A research-loop verdict, not a verdict that an answer is factually true."""

    STOP = "stop"
    INCOMPLETE = "incomplete"


def _required_text(value: str, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _identifier(value: str, field_name: str) -> str:
    cleaned = _required_text(value, field_name)
    if any(char.isspace() for char in cleaned):
        raise ValueError(f"{field_name} must not contain whitespace")
    return cleaned


def _unique_texts(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_value in values:
        value = _required_text(str(raw_value), field_name)
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return tuple(normalized)


def _normalise_timestamp(value: str, field_name: str) -> str:
    raw_value = _required_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a UTC offset")
    normalized = parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
    return normalized.replace("+00:00", "Z")


def _normalise_date(value: str, field_name: str) -> str:
    raw_value = _required_text(value, field_name)
    try:
        return date.fromisoformat(raw_value).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD") from exc


@dataclass(frozen=True)
class GeneralRunConfig:
    """Reproducible run-level constraints for an open-web research run."""

    schema_version: str = "general-run-config/v1"
    run_id: str = ""
    query: str = ""
    created_at: str = ""
    profile: str = "general"
    # The product envelope needs room for two bounded research passes plus
    # planning, extraction, writing, and semantic audit. Formal evaluation
    # still freezes an explicit budget instead of relying on this default.
    max_model_calls: int = 12
    max_tool_calls: int = 8
    max_parallel_subagents: int = 0
    evidence_only_synthesis: bool = True
    source_policy_version: str = "source-policy/v1"
    coverage_policy_version: str = "coverage-policy/v1"
    allowed_url_schemes: tuple[str, ...] = ("https", "http")

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "query", _required_text(self.query, "query"))
        object.__setattr__(self, "created_at", _normalise_timestamp(self.created_at, "created_at"))
        if self.profile != "general":
            raise ValueError("GeneralRunConfig.profile must be 'general'")
        if self.max_model_calls < 1 or self.max_tool_calls < 1:
            raise ValueError("model and tool call budgets must be positive")
        if self.max_parallel_subagents < 0:
            raise ValueError("max_parallel_subagents must be non-negative")
        if not self.evidence_only_synthesis:
            raise ValueError("general runs require evidence_only_synthesis=True")
        schemes = tuple(
            scheme.lower().strip().rstrip(":")
            for scheme in self.allowed_url_schemes
            if str(scheme).strip()
        )
        if not schemes or any(scheme not in {"http", "https"} for scheme in schemes):
            raise ValueError("allowed_url_schemes may contain only http or https")
        object.__setattr__(self, "allowed_url_schemes", _unique_texts(schemes, "allowed_url_schemes"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "query": self.query,
            "created_at": self.created_at,
            "profile": self.profile,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_parallel_subagents": self.max_parallel_subagents,
            "evidence_only_synthesis": self.evidence_only_synthesis,
            "source_policy_version": self.source_policy_version,
            "coverage_policy_version": self.coverage_policy_version,
            "allowed_url_schemes": list(self.allowed_url_schemes),
        }


@dataclass(frozen=True)
class PlanItem:
    """One atomic research obligation created by the planning stage."""

    schema_version: str = "research-plan-item/v2"
    item_id: str = ""
    question: str = ""
    # Agentic planning must connect every plan item to a requirement made
    # explicit by the preceding brief.  The empty tuple remains valid for the
    # small serial compatibility path, which has no brief stage to audit.
    requirement_ids: tuple[str, ...] = ()
    required: bool = True
    min_evidence_cards: int = 1
    min_distinct_source_groups: int = 1
    # 40 is the minimum source-policy/v1 score of a verified full fetch whose
    # publisher is still unknown.  Discovery snippets and unverified sources
    # cannot satisfy a General V1 plan item.
    min_quality_score: int = 40
    max_age_days: int | None = None
    required_source_classes: tuple[SourceClass, ...] = ()
    required_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _identifier(self.item_id, "item_id"))
        object.__setattr__(self, "question", _required_text(self.question, "question"))
        requirement_ids = tuple(
            _identifier(value, "requirement_ids") for value in self.requirement_ids
        )
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("requirement_ids must be unique")
        object.__setattr__(self, "requirement_ids", requirement_ids)
        if self.min_evidence_cards < 1:
            raise ValueError("min_evidence_cards must be at least 1")
        if self.min_distinct_source_groups < 1:
            raise ValueError("min_distinct_source_groups must be at least 1")
        if not 0 <= self.min_quality_score <= 100:
            raise ValueError("min_quality_score must be between 0 and 100")
        if self.max_age_days is not None and self.max_age_days < 0:
            raise ValueError("max_age_days must be non-negative")
        classes = tuple(SourceClass(value) for value in self.required_source_classes)
        object.__setattr__(self, "required_source_classes", tuple(dict.fromkeys(classes)))
        object.__setattr__(self, "required_terms", _unique_texts(self.required_terms, "required_terms"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "question": self.question,
            "requirement_ids": list(self.requirement_ids),
            "required": self.required,
            "min_evidence_cards": self.min_evidence_cards,
            "min_distinct_source_groups": self.min_distinct_source_groups,
            "min_quality_score": self.min_quality_score,
            "max_age_days": self.max_age_days,
            "required_source_classes": [value.value for value in self.required_source_classes],
            "required_terms": list(self.required_terms),
        }


@dataclass(frozen=True)
class ResearchPlan:
    """Versioned plan that can be frozen alongside a trace and report."""

    schema_version: str = "research-plan/v2"
    plan_id: str = ""
    run_id: str = ""
    query: str = ""
    created_at: str = ""
    items: tuple[PlanItem, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "query", _required_text(self.query, "query"))
        object.__setattr__(self, "created_at", _normalise_timestamp(self.created_at, "created_at"))
        items = tuple(self.items)
        if not items:
            raise ValueError("ResearchPlan.items must not be empty")
        if not all(isinstance(item, PlanItem) for item in items):
            raise TypeError("ResearchPlan.items must contain PlanItem objects")
        identifiers = [item.item_id for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ResearchPlan item_id values must be unique")
        object.__setattr__(self, "items", items)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "query": self.query,
            "created_at": self.created_at,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class ResearchRequirement:
    """One user-facing obligation captured before planning.

    A requirement is an intent-level target, not a truth claim and not a
    source constraint.  It exists solely so a later deterministic audit can
    prove that planning did not silently drop a requested deliverable.
    """

    schema_version: str = "research-requirement/v1"
    requirement_id: str = ""
    text: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _identifier(self.requirement_id, "requirement_id"),
        )
        object.__setattr__(self, "text", _required_text(self.text, "requirement text"))
        if not isinstance(self.required, bool):
            raise TypeError("requirement required must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "requirement_id": self.requirement_id,
            "text": self.text,
            "required": self.required,
        }


@dataclass(frozen=True)
class ResearchBrief:
    """A bounded interpretation of the user's request before decomposition.

    This is the typed equivalent of the research-brief stage used by mature
    deep-research agents.  It contains *intent*, not findings: no sources,
    evidence, or factual answer may enter the brief.  The runtime binds its
    identity and timestamp to the active run rather than trusting a model to
    create them.
    """

    schema_version: str = "general-research-brief/v2"
    brief_id: str = ""
    run_id: str = ""
    user_query: str = ""
    objective: str = ""
    scope: str = ""
    deliverable: str = ""
    requirements: tuple[ResearchRequirement, ...] = ()
    assumptions: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "brief_id", _identifier(self.brief_id, "brief_id"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        for field_name in ("user_query", "objective", "scope", "deliverable"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self, "assumptions", _unique_texts(self.assumptions, "assumptions")
        )
        requirements = tuple(self.requirements)
        if not all(isinstance(item, ResearchRequirement) for item in requirements):
            raise TypeError("requirements must contain ResearchRequirement objects")
        identifiers = [item.requirement_id for item in requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("research brief requirement IDs must be unique")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(
            self, "created_at", _normalise_timestamp(self.created_at, "created_at")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "brief_id": self.brief_id,
            "run_id": self.run_id,
            "user_query": self.user_query,
            "objective": self.objective,
            "scope": self.scope,
            "deliverable": self.deliverable,
            "requirements": [item.to_dict() for item in self.requirements],
            "assumptions": list(self.assumptions),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ResearchTask:
    """One supervisor-assigned unit of research work.

    A task must name the plan obligations it serves.  This prevents the
    supervisor from manufacturing an untracked branch of work, while still
    allowing one worker to investigate a tightly related group of obligations.
    ``research_focus`` is an instruction for the worker, never a claim of
    fact or a source-quality decision.
    """

    schema_version: str = "general-research-task/v1"
    task_id: str = ""
    plan_item_ids: tuple[str, ...] = ()
    question: str = ""
    research_focus: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        item_ids = tuple(
            _identifier(value, "plan_item_ids") for value in self.plan_item_ids
        )
        if not item_ids:
            raise ValueError("ResearchTask.plan_item_ids must not be empty")
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("ResearchTask.plan_item_ids must not contain duplicates")
        object.__setattr__(self, "plan_item_ids", item_ids)
        object.__setattr__(self, "question", _required_text(self.question, "question"))
        object.__setattr__(
            self, "research_focus", _required_text(self.research_focus, "research_focus")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "plan_item_ids": list(self.plan_item_ids),
            "question": self.question,
            "research_focus": self.research_focus,
        }


@dataclass(frozen=True)
class SupervisorDecision:
    """One bounded decision to dispatch tasks or end the research loop.

    The supervisor can decide *where to spend* a pre-reserved research round.
    It cannot change the source policy, worker budget, or publication gate.
    A finish decision must be task-free; a dispatch decision must contain at
    least one task.  The runtime validates task-to-plan links separately.
    """

    schema_version: str = "general-supervisor-decision/v1"
    run_id: str = ""
    plan_id: str = ""
    round_index: int = 0
    tasks: tuple[ResearchTask, ...] = ()
    should_finish: bool = False
    reason: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("round_index must be a non-negative integer")
        if not isinstance(self.should_finish, bool):
            raise TypeError("should_finish must be a boolean")
        tasks = tuple(self.tasks)
        if not all(isinstance(task, ResearchTask) for task in tasks):
            raise TypeError("tasks must contain ResearchTask objects")
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("SupervisorDecision task_id values must be unique")
        assigned_items = [item_id for task in tasks for item_id in task.plan_item_ids]
        if len(assigned_items) != len(set(assigned_items)):
            raise ValueError(
                "a SupervisorDecision may assign a plan item to only one task"
            )
        if self.should_finish and tasks:
            raise ValueError("a finish decision must not dispatch tasks")
        if not self.should_finish and not tasks:
            raise ValueError("a dispatch decision requires at least one task")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(
            self, "created_at", _normalise_timestamp(self.created_at, "created_at")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "round_index": self.round_index,
            "tasks": [task.to_dict() for task in self.tasks],
            "should_finish": self.should_finish,
            "reason": self.reason,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class MemoFinding:
    """A compressed worker finding with retained evidence links.

    The text is useful worker-to-supervisor context, not a new source of
    truth.  Every finding therefore names the evidence cards from which it was
    condensed; link validation happens once the full evidence ledger is known.
    """

    schema_version: str = "general-research-memo-finding/v1"
    finding_id: str = ""
    text: str = ""
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "finding_id", _identifier(self.finding_id, "finding_id"))
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        evidence_ids = tuple(
            _identifier(value, "evidence_ids") for value in self.evidence_ids
        )
        if not evidence_ids:
            raise ValueError("MemoFinding.evidence_ids must not be empty")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("MemoFinding.evidence_ids must not contain duplicates")
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "finding_id": self.finding_id,
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ResearchMemo:
    """A worker handoff artifact for supervisor context management.

    A memo may say that a task failed to find sufficient material, provided it
    records unresolved questions.  It may not be an empty success-shaped
    string: it contains either evidence-linked findings or explicit gaps.
    """

    schema_version: str = "general-research-memo/v1"
    task_id: str = ""
    plan_item_ids: tuple[str, ...] = ()
    findings: tuple[MemoFinding, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        plan_item_ids = tuple(
            _identifier(value, "plan_item_ids") for value in self.plan_item_ids
        )
        if not plan_item_ids:
            raise ValueError("ResearchMemo.plan_item_ids must not be empty")
        if len(plan_item_ids) != len(set(plan_item_ids)):
            raise ValueError("ResearchMemo.plan_item_ids must not contain duplicates")
        object.__setattr__(self, "plan_item_ids", plan_item_ids)
        findings = tuple(self.findings)
        if not all(isinstance(finding, MemoFinding) for finding in findings):
            raise TypeError("findings must contain MemoFinding objects")
        finding_ids = [finding.finding_id for finding in findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("ResearchMemo finding_id values must be unique")
        unresolved = _unique_texts(
            self.unresolved_questions, "unresolved_questions"
        )
        if not findings and not unresolved:
            raise ValueError(
                "ResearchMemo requires an evidence-linked finding or unresolved question"
            )
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "unresolved_questions", unresolved)
        object.__setattr__(
            self, "created_at", _normalise_timestamp(self.created_at, "created_at")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "plan_item_ids": list(self.plan_item_ids),
            "findings": [finding.to_dict() for finding in self.findings],
            "unresolved_questions": list(self.unresolved_questions),
            "created_at": self.created_at,
        }


def validate_supervisor_decision(
    decision: SupervisorDecision, plan: ResearchPlan
) -> None:
    """Verify that a supervisor decision belongs to this frozen plan/run."""

    if not isinstance(decision, SupervisorDecision):
        raise TypeError("decision must be SupervisorDecision")
    if not isinstance(plan, ResearchPlan):
        raise TypeError("plan must be ResearchPlan")
    if decision.run_id != plan.run_id or decision.plan_id != plan.plan_id:
        raise ValueError("SupervisorDecision must belong to the active plan/run")
    known_items = {item.item_id for item in plan.items}
    unknown_items = sorted(
        {
            item_id
            for task in decision.tasks
            for item_id in task.plan_item_ids
            if item_id not in known_items
        }
    )
    if unknown_items:
        raise ValueError(
            "SupervisorDecision refers to unknown plan items: "
            + ", ".join(unknown_items)
        )


def validate_research_memo(
    memo: ResearchMemo,
    task: ResearchTask,
    evidence_cards: Iterable["EvidenceCard"],
    sources: Iterable["SourceRecord"],
) -> None:
    """Verify a compressed memo against the authoritative evidence ledger."""

    if not isinstance(memo, ResearchMemo):
        raise TypeError("memo must be ResearchMemo")
    if not isinstance(task, ResearchTask):
        raise TypeError("task must be ResearchTask")
    if memo.task_id != task.task_id:
        raise ValueError("ResearchMemo task_id must match its ResearchTask")
    if memo.plan_item_ids != task.plan_item_ids:
        raise ValueError("ResearchMemo plan_item_ids must match its ResearchTask")
    cards = tuple(evidence_cards)
    source_ids = {source.source_id for source in sources}
    card_by_id = {card.evidence_id: card for card in cards}
    if len(card_by_id) != len(cards):
        raise ValueError("evidence_cards must use unique evidence IDs")
    for finding in memo.findings:
        for evidence_id in finding.evidence_ids:
            card = card_by_id.get(evidence_id)
            if card is None:
                raise ValueError("MemoFinding references evidence absent from the ledger")
            if card.source_id not in source_ids:
                raise ValueError("MemoFinding evidence source is absent from the ledger")
            if not set(card.plan_item_ids).intersection(memo.plan_item_ids):
                raise ValueError(
                    "MemoFinding evidence does not support the memo's assigned plan items"
                )


@dataclass(frozen=True)
class SourceRecord:
    """A fetched source with explicit provenance and non-semantic metadata."""

    schema_version: str = "general-source-record/v1"
    source_id: str = ""
    url: str = ""
    canonical_url: str = ""
    source_host: str = ""
    source_group: str = ""
    source_channel: SourceChannel = SourceChannel.PUBLIC_WEB
    source_connector_id: str = "public_web"
    source_class: SourceClass = SourceClass.UNKNOWN
    source_classification_basis: SourceClassificationBasis = (
        SourceClassificationBasis.UNVERIFIED
    )
    title: str = ""
    publisher_id: str | None = None
    published_on: str | None = None
    retrieved_at: str = ""
    content_hash: str | None = None
    content_artifact_id: str | None = None
    is_snippet: bool = False
    content_verified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "url", _required_text(self.url, "url"))
        object.__setattr__(self, "canonical_url", _required_text(self.canonical_url, "canonical_url"))
        object.__setattr__(self, "source_host", _required_text(self.source_host, "source_host").casefold())
        group = self.source_group.strip().casefold() or self.source_host
        object.__setattr__(self, "source_group", group)
        object.__setattr__(self, "source_channel", SourceChannel(self.source_channel))
        object.__setattr__(
            self,
            "source_connector_id",
            _identifier(self.source_connector_id, "source_connector_id"),
        )
        object.__setattr__(self, "source_class", SourceClass(self.source_class))
        object.__setattr__(
            self,
            "source_classification_basis",
            SourceClassificationBasis(self.source_classification_basis),
        )
        object.__setattr__(self, "title", _required_text(self.title, "title"))
        object.__setattr__(self, "retrieved_at", _normalise_timestamp(self.retrieved_at, "retrieved_at"))
        if self.publisher_id is not None:
            object.__setattr__(self, "publisher_id", _required_text(self.publisher_id, "publisher_id"))
        if self.published_on is not None:
            object.__setattr__(self, "published_on", _normalise_date(self.published_on, "published_on"))
        if self.content_hash is not None:
            object.__setattr__(self, "content_hash", _required_text(self.content_hash, "content_hash"))
        if self.content_artifact_id is not None:
            object.__setattr__(
                self,
                "content_artifact_id",
                _identifier(self.content_artifact_id, "content_artifact_id"),
            )
        if self.is_snippet and self.content_verified:
            raise ValueError("a search snippet cannot be marked content_verified")
        if self.content_verified and not self.content_hash:
            raise ValueError("content_verified sources require a content_hash")
        if self.source_class != SourceClass.UNKNOWN and (
            self.source_classification_basis
            == SourceClassificationBasis.UNVERIFIED
        ):
            raise ValueError(
                "a non-unknown source_class requires a verified classification basis"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "source_host": self.source_host,
            "source_group": self.source_group,
            "source_channel": self.source_channel.value,
            "source_connector_id": self.source_connector_id,
            "source_class": self.source_class.value,
            "source_classification_basis": self.source_classification_basis.value,
            "title": self.title,
            "publisher_id": self.publisher_id,
            "published_on": self.published_on,
            "retrieved_at": self.retrieved_at,
            "content_hash": self.content_hash,
            "content_artifact_id": self.content_artifact_id,
            "is_snippet": self.is_snippet,
            "content_verified": self.content_verified,
        }


@dataclass(frozen=True)
class EvidenceCard:
    """An extractive, source-linked item of evidence for one or more plan items."""

    schema_version: str = "general-evidence-card/v1"
    evidence_id: str = ""
    source_id: str = ""
    plan_item_ids: tuple[str, ...] = ()
    claim: str = ""
    verbatim_quote: str = ""
    locator: str = ""
    evidence_chunk_id: str = ""
    source_content_hash: str = ""
    quote_start: int = -1
    quote_end: int = -1
    stance: EvidenceStance = EvidenceStance.SUPPORTS
    observed_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        plan_item_ids = tuple(_identifier(value, "plan_item_ids") for value in self.plan_item_ids)
        if not plan_item_ids:
            raise ValueError("EvidenceCard.plan_item_ids must not be empty")
        object.__setattr__(self, "plan_item_ids", tuple(dict.fromkeys(plan_item_ids)))
        object.__setattr__(self, "claim", _required_text(self.claim, "claim"))
        object.__setattr__(self, "verbatim_quote", _required_text(self.verbatim_quote, "verbatim_quote"))
        object.__setattr__(self, "locator", _required_text(self.locator, "locator"))
        object.__setattr__(
            self,
            "evidence_chunk_id",
            _identifier(self.evidence_chunk_id, "evidence_chunk_id"),
        )
        object.__setattr__(
            self,
            "source_content_hash",
            _required_text(self.source_content_hash, "source_content_hash"),
        )
        if isinstance(self.quote_start, bool) or not isinstance(self.quote_start, int):
            raise TypeError("quote_start must be an integer")
        if isinstance(self.quote_end, bool) or not isinstance(self.quote_end, int):
            raise TypeError("quote_end must be an integer")
        if self.quote_start < 0 or self.quote_end <= self.quote_start:
            raise ValueError("quote offsets must satisfy 0 <= quote_start < quote_end")
        if self.quote_end - self.quote_start < len(self.verbatim_quote):
            raise ValueError("quote offsets cannot be shorter than verbatim_quote")
        object.__setattr__(self, "stance", EvidenceStance(self.stance))
        object.__setattr__(self, "observed_at", _normalise_timestamp(self.observed_at, "observed_at"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "plan_item_ids": list(self.plan_item_ids),
            "claim": self.claim,
            "verbatim_quote": self.verbatim_quote,
            "locator": self.locator,
            "evidence_chunk_id": self.evidence_chunk_id,
            "source_content_hash": self.source_content_hash,
            "quote_start": self.quote_start,
            "quote_end": self.quote_end,
            "stance": self.stance.value,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class CoverageDecision:
    """Deterministic audit result for exactly one plan item."""

    schema_version: str = "general-coverage-decision/v1"
    plan_item_id: str = ""
    status: CoverageStatus = CoverageStatus.UNSUPPORTED
    reason_codes: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    qualifying_evidence_ids: tuple[str, ...] = ()
    contradictory_evidence_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    source_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_item_id", _identifier(self.plan_item_id, "plan_item_id"))
        object.__setattr__(self, "status", CoverageStatus(self.status))
        for field_name in (
            "reason_codes",
            "evidence_ids",
            "qualifying_evidence_ids",
            "contradictory_evidence_ids",
            "source_ids",
            "source_groups",
        ):
            object.__setattr__(self, field_name, _unique_texts(getattr(self, field_name), field_name))

    @property
    def is_sufficient(self) -> bool:
        return self.status == CoverageStatus.SUPPORTED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_item_id": self.plan_item_id,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "evidence_ids": list(self.evidence_ids),
            "qualifying_evidence_ids": list(self.qualifying_evidence_ids),
            "contradictory_evidence_ids": list(self.contradictory_evidence_ids),
            "source_ids": list(self.source_ids),
            "source_groups": list(self.source_groups),
        }


@dataclass(frozen=True)
class CoverageState:
    """Frozen coverage result used to decide whether research may stop."""

    schema_version: str = "general-coverage-state/v1"
    run_id: str = ""
    plan_id: str = ""
    audited_at: str = ""
    source_policy_version: str = "source-policy/v1"
    coverage_policy_version: str = "coverage-policy/v1"
    decisions: tuple[CoverageDecision, ...] = ()
    verdict: CoverageVerdict = CoverageVerdict.INCOMPLETE
    stop_blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "audited_at", _normalise_timestamp(self.audited_at, "audited_at"))
        decisions = tuple(self.decisions)
        if not decisions:
            raise ValueError("CoverageState.decisions must not be empty")
        if not all(isinstance(decision, CoverageDecision) for decision in decisions):
            raise TypeError("CoverageState.decisions must contain CoverageDecision objects")
        identifiers = [decision.plan_item_id for decision in decisions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("CoverageState plan_item_id values must be unique")
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "verdict", CoverageVerdict(self.verdict))
        object.__setattr__(self, "stop_blockers", _unique_texts(self.stop_blockers, "stop_blockers"))

    @property
    def ready_to_stop(self) -> bool:
        return self.verdict == CoverageVerdict.STOP

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "audited_at": self.audited_at,
            "source_policy_version": self.source_policy_version,
            "coverage_policy_version": self.coverage_policy_version,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "verdict": self.verdict.value,
            "stop_blockers": list(self.stop_blockers),
        }
