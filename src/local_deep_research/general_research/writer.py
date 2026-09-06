"""Deterministic evidence-only boundary for General Research writing.

The writer boundary consumes the shared General Research contracts directly.
It does not duplicate planning, source, or evidence schemas and it never
accepts planner prose or search snippets as a substitute for verified source
content. This module deliberately makes no model calls: it validates the
writer context and audits structured report claims after a future writer has
generated them.

The audit is provenance and coverage based, not semantic-entailment based. A
separate verifier must decide whether an evidence quotation actually entails a
claim. The deterministic checks here guarantee that every report citation is
traceable to the frozen WriterInput and that incomplete coverage is surfaced as
a gap rather than replaced with untrusted planner prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable

from .schemas import (
    EvidenceCard,
    EvidenceStance,
    ResearchPlan,
    SourceClass,
    SourceChannel,
    SourceRecord,
)


def _required_text(value: str, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _normalized_unique_ids(
    values: Iterable[str], *, field_name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(
        _required_text(value, field_name=field_name) for value in values
    )
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one identifier")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class WriterInput:
    """The only information a General writer is allowed to consume.

    Every evidence card must link to a registered, fetched source with verified
    content. Search snippets are rejected even if a planner has attached a
    plausible URL. Both supporting and contradictory cards are deliberately
    admitted: a grounded report must be able to represent genuine disagreement
    instead of silently dropping it.
    """

    plan: ResearchPlan
    sources: tuple[SourceRecord, ...]
    evidence_cards: tuple[EvidenceCard, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ResearchPlan):
            raise TypeError("plan must be a ResearchPlan")
        sources = tuple(self.sources)
        cards = tuple(self.evidence_cards)
        if not all(isinstance(source, SourceRecord) for source in sources):
            raise TypeError("sources must contain only SourceRecord")
        if not all(isinstance(card, EvidenceCard) for card in cards):
            raise TypeError("evidence_cards must contain only EvidenceCard")

        source_ids = _normalized_unique_ids(
            (source.source_id for source in sources), field_name="sources"
        )
        if len(source_ids) != len(sources):
            raise ValueError("sources must use unique source_id values")
        cards_ids = _normalized_unique_ids(
            (card.evidence_id for card in cards),
            field_name="evidence_cards",
            allow_empty=True,
        )
        if len(cards_ids) != len(cards):
            raise ValueError("evidence_cards must use unique evidence_id values")

        source_by_id = {source.source_id: source for source in sources}
        plan_item_ids = {item.item_id for item in self.plan.items}
        for card in cards:
            source = source_by_id.get(card.source_id)
            if source is None:
                raise ValueError(
                    "EvidenceCard source_id must reference a registered SourceRecord: "
                    f"{card.source_id}"
                )
            if not source.content_verified:
                raise ValueError(
                    "General writer accepts only content_verified SourceRecord values"
                )
            if source.is_snippet or source.source_class == SourceClass.SEARCH_SNIPPET:
                raise ValueError(
                    "General writer rejects search snippets; fetch and verify the page first"
                )
            if not source.content_hash:
                raise ValueError(
                    "General writer requires a content_hash for every evidence source"
                )
            if card.source_content_hash != source.content_hash:
                raise ValueError(
                    "EvidenceCard source_content_hash must match its SourceRecord"
                )
            unknown_item_ids = sorted(
                set(card.plan_item_ids).difference(plan_item_ids)
            )
            if unknown_item_ids:
                raise ValueError(
                    "EvidenceCard refers to plan item(s) absent from ResearchPlan: "
                    + ", ".join(unknown_item_ids)
                )
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "evidence_cards", cards)

    @property
    def fingerprint(self) -> str:
        """Hash the exact plan, source registry, and evidence writer context."""
        payload = {
            "plan": self.plan.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "evidence_cards": [card.to_dict() for card in self.evidence_cards],
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReportClaim:
    """One structured final-report claim emitted by a future writer model.

    ``evidence_ids`` remains permissive at construction time so a missing or
    invented citation becomes an auditable coverage gap rather than a runtime
    exception or an ungrounded fallback to planner prose.
    """

    claim_id: str
    text: str
    plan_item_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "claim_id", _required_text(self.claim_id, field_name="claim_id")
        )
        object.__setattr__(self, "text", _required_text(self.text, field_name="text"))
        object.__setattr__(
            self,
            "plan_item_ids",
            _normalized_unique_ids(
                self.plan_item_ids, field_name="plan_item_ids", allow_empty=True
            ),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _normalized_unique_ids(
                self.evidence_ids, field_name="evidence_ids", allow_empty=True
            ),
        )
        if len(self.plan_item_ids) > 1:
            raise ValueError(
                "ReportClaim must map to one atomic plan item; split combined claims"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "plan_item_ids": list(self.plan_item_ids),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class ReportDocument:
    """The factual report before deterministic presentation rendering.

    The writer model returns only ``ReportClaim`` values. It must not return
    free-form markdown in parallel: that would create an unaudited second
    factual channel which could diverge from the claim/citation audit. The
    runtime binds the document to one frozen WriterInput and renders headings,
    citations, and source list itself.
    """

    plan_id: str
    writer_input_fingerprint: str
    claims: tuple[ReportClaim, ...]
    schema_version: str = "general-report-document/v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "plan_id", _required_text(self.plan_id, field_name="plan_id")
        )
        object.__setattr__(
            self,
            "writer_input_fingerprint",
            _required_text(
                self.writer_input_fingerprint,
                field_name="writer_input_fingerprint",
            ),
        )
        claims = tuple(self.claims)
        if not claims:
            raise ValueError("ReportDocument.claims must not be empty")
        if not all(isinstance(claim, ReportClaim) for claim in claims):
            raise TypeError("ReportDocument.claims must contain ReportClaim objects")
        claim_ids = [claim.claim_id for claim in claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("ReportDocument.claim_id values must be unique")
        object.__setattr__(self, "claims", claims)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "writer_input_fingerprint": self.writer_input_fingerprint,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def make_report_document(
    writer_input: WriterInput, claims: Iterable[ReportClaim]
) -> ReportDocument:
    """Bind model-proposed claims to the exact evidence context they saw."""

    if not isinstance(writer_input, WriterInput):
        raise TypeError("writer_input must be a WriterInput")
    return ReportDocument(
        plan_id=writer_input.plan.plan_id,
        writer_input_fingerprint=writer_input.fingerprint,
        claims=tuple(claims),
    )


@dataclass(frozen=True, slots=True)
class CoverageGap:
    """An explicit reason a report is unsafe to publish as grounded output."""

    code: str
    message: str
    plan_item_id: str | None = None
    claim_id: str | None = None
    evidence_id: str | None = None


@dataclass(frozen=True, slots=True)
class CitationTrace:
    """Exact shared-schema provenance for one evidence card cited by a claim."""

    claim_id: str
    evidence_id: str
    source_id: str
    source_locator: str
    source_channel: SourceChannel
    source_title: str
    verbatim_quote: str
    locator: str
    stance: EvidenceStance
    writer_input_fingerprint: str


@dataclass(frozen=True, slots=True)
class ClaimCitationAudit:
    """Deterministic per-claim audit result suitable for a run trace."""

    claim_id: str
    cited_evidence_ids: tuple[str, ...]
    valid: bool
    coverage_gaps: tuple[CoverageGap, ...]


@dataclass(frozen=True, slots=True)
class CitationAuditResult:
    """The publish gate result for a proposed structured final report."""

    passed: bool
    writer_input_fingerprint: str
    claim_audits: tuple[ClaimCitationAudit, ...]
    coverage_gaps: tuple[CoverageGap, ...]
    citation_traces: tuple[CitationTrace, ...]

    @property
    def should_publish(self) -> bool:
        """Publish only when every planned item has grounded citation coverage."""
        return self.passed


def _gap(
    code: str,
    message: str,
    *,
    plan_item_id: str | None = None,
    claim_id: str | None = None,
    evidence_id: str | None = None,
) -> CoverageGap:
    return CoverageGap(
        code=code,
        message=message,
        plan_item_id=plan_item_id,
        claim_id=claim_id,
        evidence_id=evidence_id,
    )


def audit_report_citations(
    writer_input: WriterInput, report_claims: Iterable[ReportClaim]
) -> CitationAuditResult:
    """Audit structured report citations against exactly the WriterInput.

    A claim fails when it has no evidence ID, cites evidence absent from the
    frozen input, names an unknown plan item, or maps a plan item to evidence
    that was not collected for it. Both ``SUPPORTS`` and ``CONTRADICTS`` cards
    may be cited, and the trace preserves the stance for a final writer or UI.
    Missing coverage is returned as data so the orchestrator can research more
    or ask the writer to repair its structured output.
    """
    if not isinstance(writer_input, WriterInput):
        raise TypeError("writer_input must be a WriterInput")
    claims = tuple(report_claims)
    if not all(isinstance(claim, ReportClaim) for claim in claims):
        raise TypeError("report_claims must contain only ReportClaim")
    claim_ids = [claim.claim_id for claim in claims]
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("report_claims must use unique claim_id values")

    cards_by_id = {card.evidence_id: card for card in writer_input.evidence_cards}
    sources_by_id = {source.source_id: source for source in writer_input.sources}
    planned_item_ids = {item.item_id for item in writer_input.plan.items}
    fingerprint = writer_input.fingerprint
    all_gaps: list[CoverageGap] = []
    claim_audits: list[ClaimCitationAudit] = []
    citation_traces: list[CitationTrace] = []
    cited_plan_item_ids: set[str] = set()

    for claim in claims:
        claim_gaps: list[CoverageGap] = []
        known_cards: list[EvidenceCard] = []
        if not claim.evidence_ids:
            claim_gaps.append(
                _gap(
                    "missing_claim_citation",
                    "Every report claim must cite at least one EvidenceCard.",
                    claim_id=claim.claim_id,
                )
            )
        for evidence_id in claim.evidence_ids:
            card = cards_by_id.get(evidence_id)
            if card is None:
                claim_gaps.append(
                    _gap(
                        "unknown_evidence_id",
                        "Report claim cites evidence absent from WriterInput.",
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                    )
                )
                continue
            source = sources_by_id[card.source_id]
            known_cards.append(card)
            citation_traces.append(
                CitationTrace(
                    claim_id=claim.claim_id,
                    evidence_id=card.evidence_id,
                    source_id=source.source_id,
                    source_locator=source.url,
                    source_channel=source.source_channel,
                    source_title=source.title,
                    verbatim_quote=card.verbatim_quote,
                    locator=card.locator,
                    stance=card.stance,
                    writer_input_fingerprint=fingerprint,
                )
            )

        for plan_item_id in claim.plan_item_ids:
            if plan_item_id not in planned_item_ids:
                claim_gaps.append(
                    _gap(
                        "unknown_plan_item_id",
                        "Report claim names a plan item absent from ResearchPlan.",
                        claim_id=claim.claim_id,
                        plan_item_id=plan_item_id,
                    )
                )
                continue
            if not known_cards:
                continue
            if not any(plan_item_id in card.plan_item_ids for card in known_cards):
                claim_gaps.append(
                    _gap(
                        "citation_not_mapped_to_plan_item",
                        "Claim citation was not collected for this plan item.",
                        claim_id=claim.claim_id,
                        plan_item_id=plan_item_id,
                    )
                )
                continue
            cited_plan_item_ids.add(plan_item_id)

        if not claim.plan_item_ids:
            claim_gaps.append(
                _gap(
                    "claim_missing_plan_item",
                    "Every report claim must map to at least one planned item.",
                    claim_id=claim.claim_id,
                )
            )
        claim_audits.append(
            ClaimCitationAudit(
                claim_id=claim.claim_id,
                cited_evidence_ids=tuple(card.evidence_id for card in known_cards),
                valid=not claim_gaps,
                coverage_gaps=tuple(claim_gaps),
            )
        )
        all_gaps.extend(claim_gaps)

    for item in writer_input.plan.items:
        if item.item_id not in cited_plan_item_ids:
            all_gaps.append(
                _gap(
                    "planned_item_not_cited",
                    "No grounded report claim cites evidence for this planned item.",
                    plan_item_id=item.item_id,
                )
            )

    return CitationAuditResult(
        passed=not all_gaps,
        writer_input_fingerprint=fingerprint,
        claim_audits=tuple(claim_audits),
        coverage_gaps=tuple(all_gaps),
        citation_traces=tuple(citation_traces),
    )


def render_report_document(
    writer_input: WriterInput,
    document: ReportDocument,
    citation_audit: CitationAuditResult,
    *,
    include_sources: bool = True,
) -> str:
    """Render the final report from audited claims and provenance only.

    This renderer is intentionally the sole markdown producer for factual
    output.  It has no input channel for planner notes, raw model prose, or
    search snippets.  The resulting markdown therefore cannot contain a
    factual sentence that escaped ``audit_report_citations``.
    """

    if not isinstance(writer_input, WriterInput):
        raise TypeError("writer_input must be a WriterInput")
    if not isinstance(document, ReportDocument):
        raise TypeError("document must be a ReportDocument")
    if not isinstance(citation_audit, CitationAuditResult):
        raise TypeError("citation_audit must be a CitationAuditResult")
    if document.plan_id != writer_input.plan.plan_id:
        raise ValueError("ReportDocument.plan_id must match WriterInput")
    if document.writer_input_fingerprint != writer_input.fingerprint:
        raise ValueError("ReportDocument fingerprint must match WriterInput")
    if citation_audit.writer_input_fingerprint != writer_input.fingerprint:
        raise ValueError("Citation audit fingerprint must match WriterInput")
    if not citation_audit.should_publish:
        raise ValueError("cannot render a report that failed citation audit")

    traces_by_claim: dict[str, list[CitationTrace]] = {}
    for trace in citation_audit.citation_traces:
        traces_by_claim.setdefault(trace.claim_id, []).append(trace)
    source_numbers: dict[str, int] = {}
    source_details: list[CitationTrace] = []
    for trace in citation_audit.citation_traces:
        if trace.source_id not in source_numbers:
            source_numbers[trace.source_id] = len(source_numbers) + 1
            source_details.append(trace)

    claims_by_item: dict[str, list[ReportClaim]] = {
        item.item_id: [] for item in writer_input.plan.items
    }
    for claim in document.claims:
        for item_id in claim.plan_item_ids:
            claims_by_item[item_id].append(claim)

    lines = ["# Research report"]
    for item in writer_input.plan.items:
        lines.extend(("", f"## {item.question}"))
        for claim in claims_by_item[item.item_id]:
            references = sorted(
                {
                    source_numbers[trace.source_id]
                    for trace in traces_by_claim.get(claim.claim_id, [])
                }
            )
            if not references:
                raise ValueError("audited claim has no citation trace")
            markers = " ".join(f"[{reference}]" for reference in references)
            lines.append(f"- {claim.text} {markers}")
    if include_sources:
        lines.extend(("", "## Sources"))
        for trace in source_details:
            number = source_numbers[trace.source_id]
            lines.append(
                f"[{number}] {trace.source_title} — {trace.source_locator} "
                f"({trace.source_channel.value})"
            )
    return "\n".join(lines)


def coverage_gap_report(
    writer_input: WriterInput, audit: CitationAuditResult
) -> str:
    """Render a deterministic non-answer when citation coverage is incomplete.

    The response contains gap metadata only. In particular it never exposes a
    planner conclusion or a search snippet as an emergency factual answer.
    """
    if audit.passed:
        return "All planned items have grounded citation coverage."
    items_by_id = {item.item_id: item for item in writer_input.plan.items}
    lines = [
        "Research status: a complete evidence-grounded report is not ready.",
        f"Writer input fingerprint: {audit.writer_input_fingerprint}",
        "Coverage gaps requiring further research or a corrected writer output:",
    ]
    for gap in audit.coverage_gaps:
        location = ""
        if gap.plan_item_id is not None:
            item = items_by_id.get(gap.plan_item_id)
            label = item.question if item is not None else gap.plan_item_id
            location = f" [plan item: {label}]"
        elif gap.claim_id is not None:
            location = f" [claim: {gap.claim_id}]"
        elif gap.evidence_id is not None:
            location = f" [evidence: {gap.evidence_id}]"
        lines.append(f"- {gap.code}{location}: {gap.message}")
    return "\n".join(lines)
