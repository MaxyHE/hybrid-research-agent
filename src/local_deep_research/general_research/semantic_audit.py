"""Contracts for the bounded semantic publish gate in General Research V1.

Deterministic citation audit establishes provenance, not entailment.  This
module represents a separate model (or later human) review of each final
claim against only the evidence excerpts it cites.  The reviewer cannot add
facts, sources, or report prose; it can only classify whether the claim is
supported by that already frozen evidence bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from .writer import CitationAuditResult, ReportDocument, WriterInput


class SemanticVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class SemanticClaimReview:
    """One bounded verifier judgment for one final structured claim."""

    claim_id: str
    verdict: SemanticVerdict
    reason_code: str

    def __post_init__(self) -> None:
        for field_name in ("claim_id", "reason_code"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if len(value.strip()) > 256:
                raise ValueError(f"{field_name} exceeds 256 characters")
            object.__setattr__(self, field_name, value.strip())
        object.__setattr__(self, "verdict", SemanticVerdict(self.verdict))


@dataclass(frozen=True, slots=True)
class SemanticAuditResult:
    """Verifier output bound to one frozen ReportDocument fingerprint."""

    writer_input_fingerprint: str
    reviews: tuple[SemanticClaimReview, ...]
    schema_version: str = "general-semantic-audit/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.writer_input_fingerprint, str) or not self.writer_input_fingerprint.strip():
            raise ValueError("writer_input_fingerprint must be a non-empty string")
        object.__setattr__(
            self, "writer_input_fingerprint", self.writer_input_fingerprint.strip()
        )
        reviews = tuple(self.reviews)
        if not reviews:
            raise ValueError("SemanticAuditResult.reviews must not be empty")
        if not all(isinstance(review, SemanticClaimReview) for review in reviews):
            raise TypeError("reviews must contain SemanticClaimReview objects")
        claim_ids = [review.claim_id for review in reviews]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("semantic reviews must use unique claim_id values")
        object.__setattr__(self, "reviews", reviews)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "writer_input_fingerprint": self.writer_input_fingerprint,
            "reviews": [
                {
                    "claim_id": review.claim_id,
                    "verdict": review.verdict.value,
                    "reason_code": review.reason_code,
                }
                for review in self.reviews
            ],
        }


def validate_semantic_audit(
    writer_input: WriterInput,
    document: ReportDocument,
    citation_audit: CitationAuditResult,
    semantic_audit: SemanticAuditResult,
) -> bool:
    """Return whether every audited report claim received a support verdict.

    A partial or unsupported verdict is a publish blocker.  Genuine source
    disagreement can still be published when the final claim accurately says
    that evidence conflicts; the reviewer should mark that *claim* supported.
    """

    if not citation_audit.should_publish:
        return False
    fingerprint = writer_input.fingerprint
    if document.writer_input_fingerprint != fingerprint:
        return False
    if citation_audit.writer_input_fingerprint != fingerprint:
        return False
    if semantic_audit.writer_input_fingerprint != fingerprint:
        return False
    document_claim_ids = {claim.claim_id for claim in document.claims}
    reviews_by_claim = {review.claim_id: review for review in semantic_audit.reviews}
    if set(reviews_by_claim) != document_claim_ids:
        return False
    return all(
        review.verdict == SemanticVerdict.SUPPORTED
        for review in semantic_audit.reviews
    )


__all__ = [
    "SemanticAuditResult",
    "SemanticClaimReview",
    "SemanticVerdict",
    "validate_semantic_audit",
]
