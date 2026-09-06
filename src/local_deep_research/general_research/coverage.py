"""Deterministic evidence coverage audit for General Research Agent v1.

The auditor checks the provenance, quality, freshness, diversity, and declared
plan links of evidence cards.  It does not use an LLM and cannot establish
semantic entailment from a quote to a claim.  Its ``stop`` verdict therefore
means "the configured research obligations have auditable coverage", never
"the final answer is true".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

from .schemas import (
    CoverageDecision,
    CoverageState,
    CoverageStatus,
    CoverageVerdict,
    EvidenceCard,
    EvidenceStance,
    PlanItem,
    ResearchPlan,
    SourceRecord,
)
from .source_policy import (
    SOURCE_POLICY_VERSION,
    assess_source_quality,
    source_age_days,
)


COVERAGE_POLICY_VERSION = "coverage-policy/v1"


def _parse_as_of(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of must be a date or YYYY-MM-DD") from exc


def _unique_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


@dataclass(frozen=True)
class CoverageAuditor:
    """Apply one frozen coverage policy against a plan and evidence registry."""

    as_of: date | str
    audited_at: str
    source_policy_version: str = SOURCE_POLICY_VERSION
    coverage_policy_version: str = COVERAGE_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _parse_as_of(self.as_of))
        # CoverageState owns strict ISO-8601 validation.  Keep the original
        # timestamp here so audit() has one obvious construction path.
        if not str(self.audited_at).strip():
            raise ValueError("audited_at must be non-empty")

    def policy_snapshot(self) -> dict[str, str]:
        """Return the deterministic configuration captured in an audit artifact."""

        return {
            "source_policy_version": self.source_policy_version,
            "coverage_policy_version": self.coverage_policy_version,
            "as_of": self.as_of.isoformat(),
            "audited_at": self.audited_at,
        }

    @staticmethod
    def _index_sources(sources: Iterable[SourceRecord]) -> dict[str, SourceRecord]:
        indexed: dict[str, SourceRecord] = {}
        for source in sources:
            if not isinstance(source, SourceRecord):
                raise TypeError("sources must contain SourceRecord objects")
            if source.source_id in indexed:
                raise ValueError(f"duplicate source_id: {source.source_id}")
            indexed[source.source_id] = source
        return indexed

    @staticmethod
    def _validate_cards(cards: Iterable[EvidenceCard]) -> tuple[EvidenceCard, ...]:
        normalized = tuple(cards)
        identifiers: set[str] = set()
        for card in normalized:
            if not isinstance(card, EvidenceCard):
                raise TypeError("cards must contain EvidenceCard objects")
            if card.evidence_id in identifiers:
                raise ValueError(f"duplicate evidence_id: {card.evidence_id}")
            identifiers.add(card.evidence_id)
        return normalized

    def _terms_present(self, item: PlanItem, card: EvidenceCard) -> bool:
        if not item.required_terms:
            return True
        searchable = f"{card.claim} {card.verbatim_quote}".casefold()
        return all(term.casefold() in searchable for term in item.required_terms)

    @staticmethod
    def _effective_source_groups(
        pairs: Iterable[tuple[EvidenceCard, SourceRecord]],
    ) -> tuple[str, ...]:
        """Count a byte-identical cross-connector page only once for diversity.

        ``source_group`` expresses independent publishers.  A public-web page
        and a copy of the same public page inside a collection are not two
        publishers merely because they entered through two connectors.  We
        collapse only verified equal content hashes; near duplicates still
        remain separate and should be resolved by future content-similarity
        policy rather than a hidden heuristic here.
        """

        normalized = tuple(pairs)
        source_ids_by_hash: dict[str, set[str]] = {}
        for _, source in normalized:
            if source.content_hash:
                source_ids_by_hash.setdefault(source.content_hash, set()).add(
                    source.source_id
                )
        groups: list[str] = []
        for _, source in normalized:
            if (
                source.content_hash
                and len(source_ids_by_hash.get(source.content_hash, ())) > 1
            ):
                group = f"duplicate-content:{source.content_hash}"
            else:
                group = source.source_group
            if group not in groups:
                groups.append(group)
        return tuple(groups)

    def _decision_for_item(
        self,
        item: PlanItem,
        *,
        sources: dict[str, SourceRecord],
        cards: tuple[EvidenceCard, ...],
    ) -> CoverageDecision:
        item_cards = tuple(card for card in cards if item.item_id in card.plan_item_ids)
        evidence_ids = tuple(card.evidence_id for card in item_cards)
        known_pairs = tuple(
            (card, sources[card.source_id])
            for card in item_cards
            if card.source_id in sources
        )
        source_ids = _unique_preserving_order(source.source_id for _, source in known_pairs)
        source_groups = self._effective_source_groups(known_pairs)

        if not item_cards:
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.UNSUPPORTED,
                reason_codes=("no_evidence_cards",),
            )
        if not known_pairs:
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.UNSUPPORTED,
                reason_codes=("evidence_source_not_registered",),
                evidence_ids=evidence_ids,
            )

        snapshot_bound_pairs = tuple(
            (card, source)
            for card, source in known_pairs
            if source.content_verified
            and source.content_hash is not None
            and card.source_content_hash == source.content_hash
        )
        if not snapshot_bound_pairs:
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.UNSUPPORTED,
                reason_codes=("evidence_not_bound_to_verified_source_snapshot",),
                evidence_ids=evidence_ids,
                source_ids=source_ids,
                source_groups=source_groups,
            )

        supporting = tuple(
            pair
            for pair in snapshot_bound_pairs
            if pair[0].stance == EvidenceStance.SUPPORTS
        )
        if not supporting:
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.UNSUPPORTED,
                reason_codes=("no_supporting_evidence",),
                evidence_ids=evidence_ids,
                source_ids=source_ids,
                source_groups=source_groups,
            )

        def class_matches(source: SourceRecord) -> bool:
            return not item.required_source_classes or source.source_class in item.required_source_classes

        def quality_matches(source: SourceRecord) -> bool:
            return assess_source_quality(source).score >= item.min_quality_score

        def freshness_matches(source: SourceRecord) -> bool:
            if item.max_age_days is None:
                return True
            age = source_age_days(source, as_of=self.as_of)
            return age is not None and 0 <= age <= item.max_age_days

        class_compatible = tuple(pair for pair in supporting if class_matches(pair[1]))
        if not class_compatible:
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.LOW_QUALITY,
                reason_codes=("required_source_class_missing",),
                evidence_ids=evidence_ids,
                source_ids=source_ids,
                source_groups=source_groups,
            )

        quality_compatible = tuple(pair for pair in class_compatible if quality_matches(pair[1]))
        if not quality_compatible:
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.LOW_QUALITY,
                reason_codes=("supporting_evidence_below_quality_threshold",),
                evidence_ids=evidence_ids,
                source_ids=source_ids,
                source_groups=source_groups,
            )

        freshness_compatible = tuple(pair for pair in quality_compatible if freshness_matches(pair[1]))
        if not freshness_compatible:
            reason = (
                "publication_date_missing_for_freshness_policy"
                if any(source.published_on is None for _, source in quality_compatible)
                else "supporting_evidence_exceeds_freshness_policy"
            )
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.STALE,
                reason_codes=(reason,),
                evidence_ids=evidence_ids,
                source_ids=source_ids,
                source_groups=source_groups,
            )

        qualifying = tuple(
            pair
            for pair in freshness_compatible
            if self._terms_present(item, pair[0])
        )
        qualifying_ids = tuple(card.evidence_id for card, _ in qualifying)
        qualifying_source_ids = _unique_preserving_order(source.source_id for _, source in qualifying)
        qualifying_source_groups = self._effective_source_groups(qualifying)

        # A contradiction becomes blocking only when it meets the same source
        # class, quality, and freshness policy.  A random low-quality page
        # cannot suppress otherwise supported research by itself.
        contradictions = tuple(
            (card, source)
            for card, source in snapshot_bound_pairs
            if card.stance == EvidenceStance.CONTRADICTS
            and class_matches(source)
            and quality_matches(source)
            and freshness_matches(source)
            and self._terms_present(item, card)
        )
        contradiction_ids = tuple(card.evidence_id for card, _ in contradictions)
        if contradictions:
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.CONFLICTING,
                reason_codes=("qualifying_contradictory_evidence",),
                evidence_ids=evidence_ids,
                qualifying_evidence_ids=qualifying_ids,
                contradictory_evidence_ids=contradiction_ids,
                source_ids=qualifying_source_ids,
                source_groups=qualifying_source_groups,
            )

        missing_terms = bool(freshness_compatible) and not qualifying
        if len(qualifying) < item.min_evidence_cards:
            reason = "required_terms_missing" if missing_terms else "insufficient_qualifying_evidence_cards"
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.PARTIAL,
                reason_codes=(reason,),
                evidence_ids=evidence_ids,
                qualifying_evidence_ids=qualifying_ids,
                source_ids=qualifying_source_ids,
                source_groups=qualifying_source_groups,
            )
        if len(qualifying_source_groups) < item.min_distinct_source_groups:
            return CoverageDecision(
                plan_item_id=item.item_id,
                status=CoverageStatus.PARTIAL,
                reason_codes=("insufficient_distinct_source_groups",),
                evidence_ids=evidence_ids,
                qualifying_evidence_ids=qualifying_ids,
                source_ids=qualifying_source_ids,
                source_groups=qualifying_source_groups,
            )
        return CoverageDecision(
            plan_item_id=item.item_id,
            status=CoverageStatus.SUPPORTED,
            reason_codes=("coverage_requirements_satisfied",),
            evidence_ids=evidence_ids,
            qualifying_evidence_ids=qualifying_ids,
            source_ids=qualifying_source_ids,
            source_groups=qualifying_source_groups,
        )

    def audit(
        self,
        plan: ResearchPlan,
        *,
        sources: Iterable[SourceRecord],
        cards: Iterable[EvidenceCard],
    ) -> CoverageState:
        """Audit every plan item and emit an explicit STOP/INCOMPLETE decision."""

        if not isinstance(plan, ResearchPlan):
            raise TypeError("plan must be a ResearchPlan")
        source_index = self._index_sources(sources)
        normalized_cards = self._validate_cards(cards)
        decisions = tuple(
            self._decision_for_item(item, sources=source_index, cards=normalized_cards)
            for item in plan.items
        )
        blockers = tuple(
            f"{item.item_id}:{decision.status.value}"
            for item, decision in zip(plan.items, decisions)
            if item.required and not decision.is_sufficient
        )
        return CoverageState(
            run_id=plan.run_id,
            plan_id=plan.plan_id,
            audited_at=self.audited_at,
            source_policy_version=self.source_policy_version,
            coverage_policy_version=self.coverage_policy_version,
            decisions=decisions,
            verdict=CoverageVerdict.STOP if not blockers else CoverageVerdict.INCOMPLETE,
            stop_blockers=blockers,
        )
