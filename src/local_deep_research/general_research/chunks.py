"""Replayable chunking and deterministic relevance selection for evidence.

The module operates on exact fetched text. It never normalizes whitespace or
rewrites content, because evidence offsets must remain valid against the stored
snapshot.  The selector is deliberately lexical and deterministic; it narrows
large pages before the extractor model sees them but never becomes a hidden
semantic judge or a second agent planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Iterable

from .config import EvidenceExtractionPolicy
from .schemas import ResearchPlan, SourceRecord


class ChunkingError(ValueError):
    """The fetched text or chunk policy cannot produce verifiable chunks."""


def _chunk_id(source_id: str, source_content_hash: str, start: int, end: int) -> str:
    material = f"{source_id}\n{source_content_hash}\n{start}\n{end}"
    return "chunk-" + sha256(material.encode("utf-8")).hexdigest()[:20]


def _span_id(chunk_id: str, start: int, end: int) -> str:
    material = f"{chunk_id}\n{start}\n{end}"
    return "span-" + sha256(material.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """One exact text range from a fetched SourceRecord snapshot."""

    chunk_id: str
    source_id: str
    source_content_hash: str
    start: int
    end: int
    content: str

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "source_id", "source_content_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ChunkingError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        for field_name in ("start", "end"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ChunkingError(f"{field_name} must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ChunkingError("chunk offsets must satisfy 0 <= start < end")
        if not isinstance(self.content, str) or not self.content:
            raise ChunkingError("chunk content must be a non-empty string")
        if len(self.content) != self.end - self.start:
            raise ChunkingError("chunk content length must match chunk offsets")

    def model_view(self) -> dict[str, object]:
        """The only source text form provided to the evidence extractor."""

        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_content_hash": self.source_content_hash,
            "start": self.start,
            "end": self.end,
            "untrusted_source_text": self.content,
        }


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """One runtime-created, exact excerpt a model may select as evidence.

    The extractor selects this stable reference rather than reproducing source
    text character-for-character.  The runtime owns the text and offsets, so
    every resulting card remains verifiable against the fetched snapshot.
    """

    span_id: str
    evidence_chunk_id: str
    source_id: str
    source_content_hash: str
    start: int
    end: int
    content: str

    def __post_init__(self) -> None:
        for field_name in (
            "span_id",
            "evidence_chunk_id",
            "source_id",
            "source_content_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ChunkingError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        for field_name in ("start", "end"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ChunkingError(f"{field_name} must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ChunkingError("span offsets must satisfy 0 <= start < end")
        if not isinstance(self.content, str) or not self.content:
            raise ChunkingError("span content must be a non-empty string")
        if len(self.content) != self.end - self.start:
            raise ChunkingError("span content length must match span offsets")


def chunk_source(
    source: SourceRecord, content: str, policy: EvidenceExtractionPolicy
) -> tuple[EvidenceChunk, ...]:
    """Split a verified page into fixed, overlapping ranges without mutation."""

    if not isinstance(source, SourceRecord):
        raise TypeError("source must be SourceRecord")
    if not isinstance(policy, EvidenceExtractionPolicy):
        raise TypeError("policy must be EvidenceExtractionPolicy")
    if not source.content_verified or not source.content_hash:
        raise ChunkingError("only verified snapshots can be chunked")
    if not isinstance(content, str) or not content:
        raise ChunkingError("content must be a non-empty string")
    if source.content_hash != "sha256:" + sha256(content.encode("utf-8")).hexdigest():
        raise ChunkingError("source content_hash does not match supplied snapshot text")
    chunk_length = policy.max_chunk_characters
    step = chunk_length - policy.chunk_overlap_characters
    chunks: list[EvidenceChunk] = []
    for start in range(0, len(content), step):
        end = min(len(content), start + chunk_length)
        chunk_content = content[start:end]
        chunks.append(
            EvidenceChunk(
                chunk_id=_chunk_id(source.source_id, source.content_hash, start, end),
                source_id=source.source_id,
                source_content_hash=source.content_hash,
                start=start,
                end=end,
                content=chunk_content,
            )
        )
        if end == len(content):
            break
    return tuple(chunks)


def span_evidence_chunks(
    chunks: Iterable[EvidenceChunk], *, max_span_characters: int
) -> tuple[EvidenceSpan, ...]:
    """Create exact selectable spans from already selected evidence chunks.

    Splitting is deterministic and preserves every source character.  A span
    can therefore be used directly as a verified quote while avoiding the
    brittle instruction for a model to hand-copy a long source substring.
    """

    if (
        isinstance(max_span_characters, bool)
        or not isinstance(max_span_characters, int)
        or max_span_characters < 1
    ):
        raise ChunkingError("max_span_characters must be a positive integer")
    spans: list[EvidenceSpan] = []
    seen_span_ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, EvidenceChunk):
            raise TypeError("chunks must contain EvidenceChunk objects")
        for start in range(chunk.start, chunk.end, max_span_characters):
            end = min(chunk.end, start + max_span_characters)
            span_id = _span_id(chunk.chunk_id, start, end)
            if span_id in seen_span_ids:
                raise ChunkingError("evidence spans must use unique span_id values")
            spans.append(
                EvidenceSpan(
                    span_id=span_id,
                    evidence_chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_content_hash=chunk.source_content_hash,
                    start=start,
                    end=end,
                    content=chunk.content[start - chunk.start : end - chunk.start],
                )
            )
            seen_span_ids.add(span_id)
    if not spans:
        raise ChunkingError("at least one evidence chunk is required to create spans")
    return tuple(spans)


_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _query_terms(value: str) -> frozenset[str]:
    folded = value.casefold()
    terms = set(_ASCII_TOKEN_RE.findall(folded))
    characters = _CJK_CHAR_RE.findall(folded)
    terms.update("".join(characters[index : index + 2]) for index in range(len(characters) - 1))
    if len(characters) == 1:
        terms.add(characters[0])
    return frozenset(terms)


def _score_chunk(chunk: EvidenceChunk, query_terms: frozenset[str]) -> int:
    if not query_terms:
        return 0
    return len(_query_terms(chunk.content).intersection(query_terms))


@dataclass(frozen=True, slots=True)
class ChunkSelection:
    """Selected evidence windows and transparent deterministic score records."""

    source_id: str
    selected_chunk_ids: tuple[str, ...]
    scores_by_chunk_id: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.selected_chunk_ids:
            raise ChunkingError("ChunkSelection must contain at least one chunk")
        if len(self.selected_chunk_ids) != len(set(self.selected_chunk_ids)):
            raise ChunkingError("selected_chunk_ids must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "scores_by_chunk_id": [list(value) for value in self.scores_by_chunk_id],
        }


def select_evidence_chunks(
    plan: ResearchPlan,
    chunks: Iterable[EvidenceChunk],
    policy: EvidenceExtractionPolicy,
) -> tuple[EvidenceChunk, ...]:
    """Allocate chunks to plan obligations before filling general relevance.

    A multi-part question should not lose its solution section merely because
    one earlier window overlaps more words from the *whole* plan.  First give
    every plan item its highest-ranked eligible candidate; then fill each
    source's remaining window capacity using full-plan relevance.
    """

    if not isinstance(plan, ResearchPlan):
        raise TypeError("plan must be ResearchPlan")
    if not isinstance(policy, EvidenceExtractionPolicy):
        raise TypeError("policy must be EvidenceExtractionPolicy")
    by_source: dict[str, list[EvidenceChunk]] = {}
    for chunk in chunks:
        if not isinstance(chunk, EvidenceChunk):
            raise TypeError("chunks must contain EvidenceChunk objects")
        by_source.setdefault(chunk.source_id, []).append(chunk)
    per_source_count = {source_id: 0 for source_id in by_source}
    selected: list[EvidenceChunk] = []
    selected_ids: set[str] = set()

    def select_if_eligible(chunk: EvidenceChunk) -> bool:
        if chunk.chunk_id in selected_ids:
            return False
        if per_source_count[chunk.source_id] >= policy.max_chunks_per_source:
            return False
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        per_source_count[chunk.source_id] += 1
        return True

    for item in plan.items:
        item_text = " ".join((plan.query, item.question, *item.required_terms))
        item_terms = _query_terms(item_text)
        ranked = sorted(
            (chunk for source_chunks in by_source.values() for chunk in source_chunks),
            key=lambda chunk: (
                -_score_chunk(chunk, item_terms),
                chunk.source_id,
                chunk.start,
                chunk.chunk_id,
            ),
        )
        for chunk in ranked:
            if select_if_eligible(chunk):
                break

    query_text = " ".join(
        (
            plan.query,
            *(item.question for item in plan.items),
            *(term for item in plan.items for term in item.required_terms),
        )
    )
    terms = _query_terms(query_text)
    for source_id in sorted(by_source):
        ranked = sorted(
            by_source[source_id],
            key=lambda chunk: (-_score_chunk(chunk, terms), chunk.start, chunk.chunk_id),
        )
        for chunk in ranked:
            select_if_eligible(chunk)
    return tuple(selected)


__all__ = [
    "ChunkSelection",
    "ChunkingError",
    "EvidenceChunk",
    "EvidenceSpan",
    "chunk_source",
    "select_evidence_chunks",
    "span_evidence_chunks",
]
