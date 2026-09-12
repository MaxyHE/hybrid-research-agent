"""Small Open Deep Research-style runtime adapted to the local web connector.

The control-flow shape and prompt fragments are adapted from
``langchain-ai/open_deep_research`` (MIT; see ``NOTICE.md`` and ``LICENSE``).
This module intentionally does *not* reuse General V1's plan/card/writer
schemas.  It is a separately identifiable baseline so that a mature research
loop can be evaluated before importing additional constraints into it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from math import log1p
from pathlib import Path
from threading import Event, RLock
from time import monotonic
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit
from uuid import uuid4
from local_deep_research.exceptions import ResearchTerminatedException

from langchain_core.messages import message_to_dict, messages_to_dict
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from local_deep_research.odr_baseline.harness import (
    OdrResearchState,
    OdrRunBudget,
    ResearchBudgetExhausted,
)
from local_deep_research.odr_baseline.project_sources import safe_fetch_failure_code
from local_deep_research.odr_baseline.sources import (
    DiscoveredResource,
    FetchedResource,
    SourceConnector,
)


_MARKDOWN_SOURCE_RE = re.compile(
    r"\[[^\]]*\]\((https?://[^)\s]+|/[^)\s]+)\)"
)
_HTTP_URL_RE = re.compile(r"https?://[^\s]+")
_SOURCE_ID_CITATION_RE = re.compile(r"\[(source-\d+)\](?!\()")
_MAX_COMPRESSION_ATTEMPTS = 3
_TRANSCRIPT_SECRET_LINE_RE = re.compile(
    r"(?im)^(\s*(?:authorization|proxy-authorization|x-api-key|api[-_ ]?key|"
    r"access[-_ ]?token|refresh[-_ ]?token|client[-_ ]?secret|password)"
    r"\s*[:=]\s*)\S.*$"
)
_TRANSCRIPT_SECRET_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "apikey",
        "access-token",
        "refresh-token",
        "client-secret",
        "password",
        "headers",
    }
)
_OBSERVED_THINKING_MODES = frozenset(
    {"provider_default", "enabled", "disabled"}
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content or "").strip()


def _message_input_metrics(messages: Iterable[object]) -> tuple[int, int, dict[str, int]]:
    """Return small, content-only diagnostics for one model request.

    This intentionally measures visible message content rather than estimating
    provider tokens or serialising a provider-specific wire format.  It gives a
    replay trace enough information to locate unexpectedly large researcher,
    compression, or writer requests without retaining a second copy of source
    text or changing the request itself.
    """

    message_count = 0
    content_characters = 0
    message_types: dict[str, int] = {}
    for message in messages:
        message_count += 1
        type_name = type(message).__name__
        message_types[type_name] = message_types.get(type_name, 0) + 1
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content_characters += sum(
                len(str(item.get("text", ""))) if isinstance(item, Mapping) else len(str(item))
                for item in content
            )
        else:
            content_characters += len(str(content or ""))
    return message_count, content_characters, message_types


def _optional_usage_count(value: object) -> int | None:
    """Accept only provider-reported non-negative token counts."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _response_observability(response: object) -> dict[str, object]:
    """Return non-content completion facts from common LangChain response shapes.

    The ODR trace and live progress stream must remain safe to keep next to a
    remote experiment. This observer therefore records only provider metadata,
    never a prompt, response body, source, URL, or exception message. Missing
    fields remain ``None`` rather than being estimated from characters.
    """

    usage_metadata = _mapping(getattr(response, "usage_metadata", None))
    response_metadata = _mapping(getattr(response, "response_metadata", None))
    usage = (
        usage_metadata
        or _mapping(response_metadata.get("token_usage"))
        or _mapping(response_metadata.get("usage"))
    )
    completion_details = _mapping(usage.get("completion_tokens_details"))
    finish_reason = response_metadata.get("finish_reason")
    additional_kwargs = _mapping(getattr(response, "additional_kwargs", None))
    return {
        "input_tokens": _optional_usage_count(
            usage.get("input_tokens", usage.get("prompt_tokens"))
        ),
        "output_tokens": _optional_usage_count(
            usage.get("output_tokens", usage.get("completion_tokens"))
        ),
        "total_tokens": _optional_usage_count(usage.get("total_tokens")),
        "reasoning_tokens": _optional_usage_count(
            usage.get(
                "reasoning_tokens", completion_details.get("reasoning_tokens")
            )
        ),
        "finish_reason": finish_reason
        if isinstance(finish_reason, str)
        else None,
        "reasoning_content_present": isinstance(
            additional_kwargs.get("reasoning_content"), str
        )
        and bool(additional_kwargs["reasoning_content"]),
    }


def _redact_transcript_secrets(value: object) -> object:
    """Redact credential/header values before writing a development transcript.

    ODR model calls receive messages and tool definitions, never transport
    headers.  This small recursive pass also protects against a credential
    accidentally appearing in message metadata or visible text.  It is scoped
    to the opt-in development export and does not affect model inputs.
    """

    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower().replace("_", "-") in _TRANSCRIPT_SECRET_KEYS
                else _redact_transcript_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_transcript_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_transcript_secrets(item) for item in value]
    if isinstance(value, str):
        return _TRANSCRIPT_SECRET_LINE_RE.sub(r"\1[REDACTED]", value)
    return value


def _safe_identifier(value: str, *, field_name: str) -> str:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", cleaned):
        raise ValueError(f"{field_name} must be a safe non-empty identifier")
    return cleaned


def _source_excerpt(content: str, *, maximum: int) -> str:
    """Expose a bounded, non-single-window source view to the researcher.

    General V1's previous fixed leading window could omit the relevant later
    section of a long source.  This baseline does not infer a winning passage:
    it retains the beginning, centre, and end and writes the complete immutable
    snapshot to disk for later human review.
    """

    if len(content) <= maximum:
        return content
    window = maximum // 3
    middle_start = max(0, len(content) // 2 - window // 2)
    tail_start = max(0, len(content) - window)
    return (
        content[:window]
        + "\n\n[... middle of this fetched page ...]\n\n"
        + content[middle_start : middle_start + window]
        + "\n\n[... end of this fetched page ...]\n\n"
        + content[tail_start:]
    )


def _extractive_evidence_span(
    content: str,
    *,
    requirement: str,
    retrieval_query: str | None = None,
    source_snippet: str | None = None,
    maximum: int | None,
) -> tuple[str, int, int]:
    """Project one source into a requirement-relevant, verbatim evidence span.

    Public-Web adapters may flatten HTML into a single line, so paragraph
    boundaries are not a reliable retrieval unit. Reuse the sentence-window
    projection used by extractive handoff: it selects a generic lexical match,
    then preserves a contiguous snapshot substring and its offsets. This is
    still retrieval projection only; the coverage reviewer decides whether the
    resulting evidence directly supports the requirement.  The requirement is
    often written in the user's language while a public-Web source is not, so
    include the requirement's already-recorded retrieval query when available.
    A search snippet is also a recorded, source-specific lexical anchor, but
    never evidence: the selected span is still copied only from ``content``.
    This stays tied to the ledger entry rather than introducing source- or
    topic-specific extraction rules.
    """

    focus = "\n".join(
        value.strip()
        for value in (requirement, retrieval_query)
        if isinstance(value, str) and value.strip()
    )
    return _extractive_handoff_window(
        content,
        focus=focus,
        secondary_focus=source_snippet or "",
        maximum=len(content) if maximum is None else maximum,
    )


def _collection_paper_id(source: "OdrSource") -> str | None:
    """Return the Collection paper identifier embedded in its frozen document header."""

    if source.channel != "collection" or not source.content:
        return None
    match = re.search(
        r"(?mi)^Paper id:\s*([A-Za-z0-9][A-Za-z0-9_-]{0,127})\s*$",
        source.content[:4096],
    )
    return match.group(1).casefold() if match is not None else None


def _verbatim_quote_from_candidate(
    candidate_excerpt: str,
    supplied_quote: str,
) -> tuple[str, bool] | None:
    """Recover an exact snapshot quote when a model only normalizes punctuation.

    Coverage review may preserve every lexical token in a quote while changing
    PDF-only surface formatting: punctuation, whitespace, compatibility
    ligatures, or a line-break hyphen inside a word. The ledger must still
    store only a verbatim source substring, so accept that response only when
    its normalized alphanumeric sequence appears unchanged and in order in the
    supplied candidate. The returned quote is always copied from the immutable
    candidate excerpt; changed words, numbers, or order do not match.
    """

    quote = supplied_quote.strip()
    if not quote:
        return None
    if quote in candidate_excerpt:
        return quote, False

    terms = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", quote)
    if not terms:
        return None
    pattern = (
        r"(?<![A-Za-z0-9])"
        + r"[\s\W_]*".join(re.escape(term) for term in terms)
        + r"(?![A-Za-z0-9])"
    )
    match = re.search(pattern, candidate_excerpt, flags=re.IGNORECASE)
    if match is None:
        quote_stream = "".join(
            character.casefold()
            for character in unicodedata.normalize("NFKC", quote)
            if character.isalnum()
        )
        if not quote_stream:
            return None
        candidate_stream_parts: list[str] = []
        source_starts: list[int] = []
        source_ends: list[int] = []
        for index, character in enumerate(candidate_excerpt):
            for normalized in unicodedata.normalize("NFKC", character):
                for folded in normalized.casefold():
                    if folded.isalnum():
                        candidate_stream_parts.append(folded)
                        source_starts.append(index)
                        source_ends.append(index + 1)
        candidate_stream = "".join(candidate_stream_parts)
        normalized_start = candidate_stream.find(quote_stream)
        if normalized_start < 0:
            return None
        normalized_end = normalized_start + len(quote_stream)
        return (
            candidate_excerpt[
                source_starts[normalized_start] : source_ends[normalized_end - 1]
            ],
            True,
        )

    start, end = match.span()
    if start > 0 and candidate_excerpt[start - 1] in "([{\"“":
        closing = {
            "(": ")",
            "[": "]",
            "{": "}",
            "\"": "\"",
            "“": "”",
        }[candidate_excerpt[start - 1]]
        if closing in candidate_excerpt[start:end]:
            start -= 1
    if re.search(r"[^\w\s]$", quote):
        while (
            end < len(candidate_excerpt)
            and not candidate_excerpt[end].isalnum()
            and not candidate_excerpt[end].isspace()
        ):
            end += 1
    return candidate_excerpt[start:end], True


def _extractive_handoff_window(
    content: str,
    *,
    focus: str,
    maximum: int,
    secondary_focus: str = "",
) -> tuple[str, int, int]:
    """Project one source into a task-relevant, verbatim snapshot window.

    Public-Web adapters may collapse HTML paragraphs into one line, so a
    paragraph-only extractor would silently select the beginning of a long
    page. Rank complete windows rather than choosing a single sentence first:
    useful evidence often spans several individually low-scoring sentences.
    Repeated document vocabulary is downweighted, and search snippets supply
    secondary anchors. Matching normalizes PDF typography; returned text and
    offsets always refer to the unchanged snapshot. This is retrieval, not a
    semantic coverage judgment.
    """

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("maximum must be a positive integer")
    if len(content) <= maximum:
        return content, 0, len(content)

    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
        return re.sub(r"\s+", " ", text).casefold()

    def focus_terms(text: str) -> set[str]:
        # A search restriction selects a source, not an answer within it.
        text = re.sub(r"\bsite:\S+", "", text, flags=re.IGNORECASE)
        terms = set(re.findall(r"[a-z0-9][a-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", normalize(text)))
        for phrase in tuple(terms):
            if re.fullmatch(r"[\u4e00-\u9fff]{3,}", phrase):
                terms.update(phrase[index : index + 2] for index in range(len(phrase) - 1))
        return terms

    primary_terms = focus_terms(focus)
    terms = primary_terms | focus_terms(secondary_focus)

    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in re.finditer(r"[.!?。！？]+(?:\s+|$)", content):
        end = boundary.end()
        while start < end and content[start].isspace():
            start += 1
        while end > start and content[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
        start = boundary.end()
    end = len(content)
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))
    if not spans:
        return "", 0, 0

    starts = {start for start, _ in spans}
    # Also anchor at sentence ends so a long preceding sentence does not
    # consume the budget before a short, useful follow-up sentence.
    starts.update(max(0, end - maximum) for _, end in spans)
    for start, end in spans:
        if end - start > maximum:
            starts.update(range(start, end, max(1, maximum // 3)))
            passage = content[start:end].lower()
            for term in terms:
                starts.update(
                    max(0, min(len(content) - maximum, start + match.start() - maximum // 3))
                    for match in re.finditer(re.escape(term), passage)
                )
    sentence_ends = [end for _, end in spans]
    windows: list[tuple[int, int, set[str]]] = []
    for start in sorted(starts):
        end = min(len(content), start + maximum)
        boundary_index = bisect_right(sentence_ends, end) - 1
        if boundary_index >= 0 and sentence_ends[boundary_index] - start >= maximum // 2:
            end = sentence_ends[boundary_index]
        passage = normalize(content[start:end])
        windows.append((start, end, {term for term in terms if term in passage}))

    frequencies = {term: 0 for term in terms}
    for _, _, matched in windows:
        for term in matched:
            frequencies[term] += 1
    weights = {
        term: log1p(len(windows) / (1 + frequencies[term]))
        * (1.0 if term in primary_terms else 0.25)
        for term in terms
    }
    selected_start, selected_end, _ = max(
        windows,
        key=lambda window: (
            sum(weights[term] for term in sorted(window[2])),
            -window[0],
        ),
    )
    return content[selected_start:selected_end], selected_start, selected_end


def _citation_locators(markdown: str) -> tuple[str, ...]:
    urls = list(_MARKDOWN_SOURCE_RE.findall(markdown))
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_url in urls:
        url = raw_url.rstrip(".,;:!?）】")
        if url and url not in seen:
            normalized.append(url)
            seen.add(url)
    return tuple(normalized)


def _citation_aliases(values: Iterable[object]) -> tuple[str, ...]:
    """Keep explicit, well-formed provenance URLs available to the audit only."""

    return tuple(
        value.strip()
        for value in values
        if isinstance(value, str) and _HTTP_URL_RE.fullmatch(value.strip())
    )


class ConductResearch(BaseModel):
    """Open Deep Research-compatible supervisor delegation contract."""

    research_topic: str = Field(
        description=(
            "A standalone, focused research question. Include the precise facts, "
            "comparison members, and source types the researcher should seek."
        )
    )


class ResearchComplete(BaseModel):
    """Signal that the supervisor or researcher has no further tool work."""


class OnePassResearchPlan(BaseModel):
    """One pre-retrieval research plan for the static workflow baseline."""

    research_topics: list[str] = Field(
        description=(
            "Focused research topics to execute once, before any source result is "
            "observed."
        )
    )


class CoverageRepairPlan(BaseModel):
    """One evidence-driven repair task after fixed retrieval leaves a gap."""

    research_topic: str | None = Field(
        default=None,
        description=(
            "One standalone focused repair task, or null when the observed "
            "coverage gap cannot be recovered with one additional task."
        ),
    )
    repaired_initial_task_ids: list[str] = Field(
        default_factory=list,
        description=(
            "The initial zero-source task IDs this repair is intended to recover. "
            "Use only IDs supplied by the runtime ledger."
        ),
    )


class EvidenceRequirement(BaseModel):
    """One user-facing requirement and its source-discovery query."""

    requirement: str = Field(
        description="One atomic fact, comparison, constraint, or question the report must address."
    )
    retrieval_query: str = Field(
        description="A focused source-discovery query for this requirement."
    )


class EvidenceRequirementPlan(BaseModel):
    """Pre-retrieval decomposition for the evidence-ledger workflow."""

    requirements: list[EvidenceRequirement] = Field(
        description=(
            "Atomic, non-overlapping user requirements. Each requirement must include "
            "one focused retrieval query and must be planned before seeing sources."
        )
    )


class EvidenceCoverageSelection(BaseModel):
    """One requirement that a supplied fetched-source extract directly supports."""

    requirement_id: str = Field(
        description="One requirement ID supplied by the runtime candidate ledger."
    )
    candidate_key: str = Field(
        description=(
            "The exact immutable candidate key shown in this requirement's candidate "
            "block. Candidate keys are unique across the full ledger."
        )
    )
    support_quote: str = Field(
        description=(
            "A non-empty contiguous quote copied exactly from that candidate's supplied "
            "extract which directly supports this requirement."
        )
    )


class EvidenceCoverageReview(BaseModel):
    """Source-bound coverage decision, with at most one repair proposal."""

    covered_evidence: list[EvidenceCoverageSelection] = Field(
        default_factory=list,
        description=(
            "At most one source selection for each requirement. Select only when the "
            "corresponding supplied extract directly supports the requirement."
        ),
    )
    research_topic: str | None = Field(
        default=None,
        description=(
            "One focused repair query for unresolved requirements, or null when no "
            "single repair can recover them."
        ),
    )
    repaired_requirement_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Unresolved requirement IDs the one repair query is intended to recover. "
            "Use only IDs supplied by the runtime ledger."
        ),
    )


class EvidenceBriefSection(BaseModel):
    """One reviewed card, optionally with its evidence-linked Chinese explanation."""

    requirement_id: str = Field(
        description="One requirement ID from the supplied reviewed evidence cards."
    )
    evidence_id: str = Field(
        description="The exact reviewed evidence ID supplied for that requirement."
    )
    explanation_zh: str | None = Field(
        default=None,
        description=(
            "For an evidence-linked narrative brief only: one or two Chinese "
            "sentences explaining the supplied reviewed quote."
        ),
    )


class EvidenceBriefPlan(BaseModel):
    """Reviewed-card plan for an extractive or evidence-linked narrative brief."""

    sections: list[EvidenceBriefSection] = Field(
        default_factory=list,
        description=(
            "Every supplied reviewed evidence card exactly once, in the order for the "
            "brief. An evidence-linked narrative brief requires explanation_zh on each "
            "section; an extractive evidence brief omits it."
        ),
    )


@dataclass(frozen=True, slots=True)
class OdrBaselinePolicy:
    """Bounded execution envelope, using deep-research-harness terminology.

    ``breadth_budget`` is the maximum total focused tasks (initial and follow
    ups); ``depth_budget`` is the maximum number of follow-up supervisor
    rounds after the initial pass.  Both are generic resource controls, never
    a topic taxonomy.  One worker remains the stabilization default.

    Research and report calls are intentionally separate.  With no global
    research cap, the copied upstream breadth, depth, researcher-turn, and
    report envelopes remain bounded.
    """

    research_model_call_limit: int | None = None
    report_model_call_allowance: int = 1
    max_tool_calls: int | None = None
    breadth_budget: int = 6
    depth_budget: int = 5
    max_researcher_turns: int = 10
    max_concurrent_research_units: int = 1
    initial_delegation_strategy: str = "adaptive"
    max_web_actions_per_research_turn: int = 1
    reserve_tools_per_pending_task: int = 2
    max_sources_per_search: int = 5
    max_fetches_per_research_unit: int = 3
    source_view_max_chars: int = 50_000
    evidence_excerpt_max_chars: int | None = None
    source_linked_working_notes: bool = False
    researcher_working_memory: bool = False
    source_handle_evidence_handoff: bool = False
    evidence_handoff_mode: str = "generative"
    evidence_handoff_excerpt_max_chars: int | None = None
    budget_aware_prompting: bool = False
    evidence_brief_enabled: bool = False
    evidence_narrative_brief_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "report_model_call_allowance",
            "breadth_budget",
            "max_researcher_turns",
            "max_concurrent_research_units",
            "max_web_actions_per_research_turn",
            "reserve_tools_per_pending_task",
            "max_sources_per_search",
            "max_fetches_per_research_unit",
            "source_view_max_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.depth_budget, bool)
            or not isinstance(self.depth_budget, int)
            or self.depth_budget < 0
        ):
            raise ValueError("depth_budget must be a non-negative integer")
        if self.initial_delegation_strategy not in {"adaptive", "parallel_first"}:
            raise ValueError(
                "initial_delegation_strategy must be 'adaptive' or 'parallel_first'"
            )
        for name in (
            "source_linked_working_notes",
            "researcher_working_memory",
            "source_handle_evidence_handoff",
            "budget_aware_prompting",
            "evidence_brief_enabled",
            "evidence_narrative_brief_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in ("research_model_call_limit", "max_tool_calls"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.evidence_excerpt_max_chars is not None and (
            isinstance(self.evidence_excerpt_max_chars, bool)
            or not isinstance(self.evidence_excerpt_max_chars, int)
            or self.evidence_excerpt_max_chars < 1
        ):
            raise ValueError("evidence_excerpt_max_chars must be a positive integer or None")
        if self.evidence_handoff_mode not in {"generative", "extractive"}:
            raise ValueError(
                "evidence_handoff_mode must be 'generative' or 'extractive'"
            )
        if self.evidence_handoff_excerpt_max_chars is not None and (
            isinstance(self.evidence_handoff_excerpt_max_chars, bool)
            or not isinstance(self.evidence_handoff_excerpt_max_chars, int)
            or self.evidence_handoff_excerpt_max_chars < 1
        ):
            raise ValueError(
                "evidence_handoff_excerpt_max_chars must be a positive integer or None"
            )
        if self.evidence_handoff_mode == "extractive":
            if not self.source_handle_evidence_handoff:
                raise ValueError(
                    "extractive evidence handoff requires source_handle_evidence_handoff"
                )
            if self.evidence_handoff_excerpt_max_chars is None:
                raise ValueError(
                    "extractive evidence handoff requires explicit "
                    "evidence_handoff_excerpt_max_chars"
                )
        elif self.evidence_handoff_excerpt_max_chars is not None:
            raise ValueError(
                "evidence_handoff_excerpt_max_chars is only valid for extractive handoff"
            )


@dataclass(slots=True)
class OdrSource:
    source_id: str
    url: str
    title: str
    snippet: str
    channel: str
    discovered_for: list[str] = field(default_factory=list)
    fetched_for: list[str] = field(default_factory=list)
    retrieved_at: str | None = None
    content: str | None = None
    content_sha256: str | None = None
    citation_aliases: tuple[str, ...] = ()
    retrieval_method: str | None = None
    fetch_error: str | None = None
    _fetch_lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def public_view(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "channel": self.channel,
            "discovered_for": list(self.discovered_for),
            "fetched_for": list(self.fetched_for),
            "retrieved_at": self.retrieved_at,
            "content_sha256": self.content_sha256,
            "content_characters": len(self.content or ""),
            "retrieval_method": self.retrieval_method,
            "fetch_error": self.fetch_error,
        }


@dataclass(slots=True)
class OdrEvidenceLedgerEntry:
    """A requirement-level pointer to an immutable source snapshot span.

    The excerpt is always copied directly from ``OdrSource.content`` and its
    offsets are retained so a reviewer can check it against the snapshot. A
    ``covered`` status is a model coverage judgement over that extract.  The
    selected candidate key and exact reviewer quote are retained separately so
    an extractive final artifact can show precisely what was reviewed instead
    of asking a writer to restate it from memory.
    """

    requirement_id: str
    requirement: str
    retrieval_query: str
    source_channel: str = "web"
    expected_collection_paper_id: str | None = None
    status: str = "pending"
    source_id: str | None = None
    excerpt: str | None = None
    source_start: int | None = None
    source_end: int | None = None
    candidate_key: str | None = None
    support_quote: str | None = None
    support_start: int | None = None
    support_end: int | None = None

    def public_view(self) -> dict[str, object]:
        view: dict[str, object] = {
            "requirement_id": self.requirement_id,
            "requirement": self.requirement,
            "retrieval_query": self.retrieval_query,
            "source_channel": self.source_channel,
            "status": self.status,
            "source_id": self.source_id,
            "excerpt": self.excerpt,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "candidate_key": self.candidate_key,
            "support_quote": self.support_quote,
            "support_start": self.support_start,
            "support_end": self.support_end,
        }
        if self.expected_collection_paper_id is not None:
            view["expected_collection_paper_id"] = self.expected_collection_paper_id
        return view


@dataclass(frozen=True, slots=True)
class _EvidenceCandidate:
    """One renderable, requirement-scoped immutable evidence candidate."""

    candidate_key: str
    source_id: str
    excerpt: str
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class OdrTraceEvent:
    sequence: int
    timestamp: str
    kind: str
    data: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "odr-baseline-trace/v1",
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "data": self.data,
        }


@dataclass(frozen=True, slots=True)
class OdrCitationAudit:
    cited_urls: tuple[str, ...]
    known_urls: tuple[str, ...]
    unknown_urls: tuple[str, ...]
    unresolved_source_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.known_urls) and not self.unknown_urls and not self.unresolved_source_ids

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "cited_urls": list(self.cited_urls),
            "known_urls": list(self.known_urls),
            "unknown_urls": list(self.unknown_urls),
            "unresolved_source_ids": list(self.unresolved_source_ids),
        }


@dataclass(frozen=True, slots=True)
class OdrClaimSupportAudit:
    """Deterministic integrity audit for reviewed-card output.

    This is deliberately narrower than semantic entailment.  It proves every
    rendered card or brief section has a verbatim quote with a requirement-scoped
    candidate key and an immutable source offset. For an evidence-linked narrative
    brief, it does not prove the Chinese explanation is semantically entailed.
    """

    mode: str
    expected_requirement_ids: tuple[str, ...]
    rendered_requirement_ids: tuple[str, ...]
    invalid_cards: tuple[tuple[str, str], ...]

    @property
    def passed(self) -> bool:
        return (
            not self.invalid_cards
            and self.rendered_requirement_ids == self.expected_requirement_ids
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "passed": self.passed,
            "expected_requirement_ids": list(self.expected_requirement_ids),
            "rendered_requirement_ids": list(self.rendered_requirement_ids),
            "invalid_cards": [
                {"requirement_id": requirement_id, "reason": reason}
                for requirement_id, reason in self.invalid_cards
            ],
        }


@dataclass(frozen=True, slots=True)
class OdrBaselineResult:
    run_id: str
    query: str
    execution_mode: str
    status: str
    terminal_reason: str
    report_markdown: str
    research_tasks: tuple[str, ...]
    research_notes: tuple[str, ...]
    unresolved_tasks: tuple[str, ...]
    sources: tuple[OdrSource, ...]
    trace: tuple[OdrTraceEvent, ...]
    model_calls_used: int
    tool_calls_used: int
    citation_audit: OdrCitationAudit
    claim_support_audit: OdrClaimSupportAudit | None
    harness_state: dict[str, object]
    budget: dict[str, object]
    evidence_ledger: tuple[OdrEvidenceLedgerEntry, ...]
    elapsed_seconds: float

    @property
    def is_publishable(self) -> bool:
        return (
            self.status == "complete"
            and self.citation_audit.passed
            and (
                self.claim_support_audit is None
                or self.claim_support_audit.passed
            )
        )


@dataclass(frozen=True, slots=True)
class OdrBaselineArtifacts:
    run_dir: Path
    report_path: Path
    trace_path: Path
    run_path: Path
    sources_path: Path
    notes_path: Path
    state_path: Path
    evidence_ledger_path: Path


@dataclass(slots=True)
class _ResearchTaskToolBudget:
    """A temporary tool allowance for one focused task.

    It is deliberately about resources, not subject matter: when a finite
    global budget is shared by several already-delegated tasks, reserving a
    small number of web actions for the tasks that have not started prevents
    the first task from silently starving them.  In an uncapped comparison run
    no such object is created.
    """

    maximum_tool_calls: int
    used_tool_calls: int = 0


class OdrBaselineRunner:
    """A bounded, source-snapshotting Open Deep Research-style runner."""

    def __init__(
        self,
        *,
        run_id: str,
        query: str,
        llm: Any,
        connector: SourceConnector,
        collection_connector: SourceConnector | None = None,
        collection_context: str | None = None,
        policy: OdrBaselinePolicy | None = None,
        usage_ledger: Any | None = None,
        research_llms: Iterable[Any] | None = None,
        development_transcript_path: str | Path | None = None,
        progress_path: str | Path | None = None,
        thinking_mode: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_event: Callable[[OdrTraceEvent], None] | None = None,
    ) -> None:
        self.run_id = _safe_identifier(run_id, field_name="run_id")
        self._cancelled = Event()
        self._should_cancel = should_cancel
        self._on_event = on_event
        self.query = query.strip() if isinstance(query, str) else ""
        if not self.query:
            raise ValueError("query must be non-empty")
        if not callable(getattr(llm, "invoke", None)):
            raise TypeError("llm must provide invoke()")
        if not callable(getattr(llm, "bind_tools", None)):
            raise TypeError("llm must provide bind_tools() for the baseline")
        selected_research_llms = (llm,) if research_llms is None else tuple(research_llms)
        if not selected_research_llms:
            raise ValueError("research_llms must contain at least one model")
        if any(
            not callable(getattr(model, "invoke", None))
            or not callable(getattr(model, "bind_tools", None))
            for model in selected_research_llms
        ):
            raise TypeError("each research_llms model must provide invoke() and bind_tools()")
        if not isinstance(connector, SourceConnector):
            raise TypeError("connector must implement SourceConnector")
        if collection_connector is not None and not isinstance(collection_connector, SourceConnector):
            raise TypeError("collection_connector must implement SourceConnector")
        self.llm = llm
        self._research_llms = selected_research_llms
        self.connector = connector
        self.collection_connector = collection_connector
        self.collection_context = (
            collection_context.strip()
            if isinstance(collection_context, str) and collection_context.strip()
            else None
        )
        self.policy = policy or OdrBaselinePolicy()
        self.usage_ledger = usage_ledger
        if (
            thinking_mode is not None
            and thinking_mode not in _OBSERVED_THINKING_MODES
        ):
            raise ValueError(
                "thinking_mode must be provider_default, enabled, disabled, or None"
            )
        self._thinking_mode = thinking_mode
        if development_transcript_path is None:
            self._development_transcript_path: Path | None = None
        else:
            transcript_path = Path(development_transcript_path).expanduser()
            if not transcript_path.is_absolute():
                raise ValueError(
                    "development_transcript_path must be an absolute path"
                )
            self._development_transcript_path = transcript_path.resolve()
            self._development_transcript_path.parent.mkdir(
                parents=True, exist_ok=True
            )
        if progress_path is None:
            self._progress_path: Path | None = None
        else:
            resolved_progress_path = Path(progress_path).expanduser()
            if not resolved_progress_path.is_absolute():
                raise ValueError("progress_path must be an absolute path")
            self._progress_path = resolved_progress_path.resolve()
            self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_budget = OdrRunBudget(
            research_model_call_limit=self.policy.research_model_call_limit,
            report_model_call_allowance=self.policy.report_model_call_allowance,
        )
        self._lock = RLock()
        self._model_calls_used = 0
        self._tool_calls_used = 0
        self._trace: list[OdrTraceEvent] = []
        self._sources_by_url: dict[tuple[str, str], OdrSource] = {}
        self._sources_by_id: dict[str, OdrSource] = {}
        self._next_source_number = 1
        self._research_state = OdrResearchState(
            breadth_budget=self.policy.breadth_budget,
            depth_budget=self.policy.depth_budget,
        )

    def _check_cancelled(self) -> None:
        if self._should_cancel is not None and self._should_cancel():
            self._cancelled.set()
        if self._cancelled.is_set():
            raise ResearchTerminatedException("Hybrid research cancelled by user")

    def _event(self, kind: str, **data: object) -> None:
        with self._lock:
            event = OdrTraceEvent(
                sequence=len(self._trace) + 1,
                timestamp=_timestamp(),
                kind=kind,
                data=dict(data),
            )
            self._trace.append(event)
            self._capture_progress_event(event)
        # Release this event's lock before notifying consumers. Budget-blocked
        # events can still originate under an outer lock; UI adapters ignore them.
        if self._on_event is not None:
            self._check_cancelled()
            try:
                self._on_event(event)
            except ResearchTerminatedException:
                # Web cleanup clears the shared flag; queued workers must still stop.
                self._cancelled.set()
                raise

    def _capture_progress_event(self, event: OdrTraceEvent) -> None:
        """Append safe, low-volume liveness facts for an opt-in live monitor.

        The full trace is still written only with the final artifacts.  This
        stream deliberately omits prompt/query text, model prose, URLs and
        source bodies so an operator can distinguish an active model request
        from a dead runner without retaining a second research transcript.
        """

        if self._progress_path is None:
            return
        data = event.data
        if event.kind == "model_request_started":
            safe_data = {
                key: data.get(key)
                for key in (
                    "role",
                    "call_index",
                    "phase",
                    "input_message_count",
                    "input_content_characters",
                    "input_message_types",
                    "thinking_mode",
                )
            }
        elif event.kind == "model_response":
            tool_calls = data.get("tool_calls")
            safe_data = {
                "role": data.get("role"),
                "call_index": data.get("call_index"),
                "phase": data.get("phase"),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "thinking_mode": data.get("thinking_mode"),
                "input_tokens": data.get("input_tokens"),
                "output_tokens": data.get("output_tokens"),
                "total_tokens": data.get("total_tokens"),
                "reasoning_tokens": data.get("reasoning_tokens"),
                "finish_reason": data.get("finish_reason"),
                "reasoning_content_present": data.get("reasoning_content_present"),
                "tool_call_names": [
                    str(call.get("name", ""))
                    for call in tool_calls
                    if isinstance(call, Mapping)
                ]
                if isinstance(tool_calls, list)
                else [],
            }
        elif event.kind == "search_completed":
            result_source_ids = data.get("result_source_ids")
            safe_data = {
                "channel": data.get("channel"),
                "result_count": len(result_source_ids)
                if isinstance(result_source_ids, list)
                else 0,
            }
        elif event.kind in {"fetch_completed", "fetch_failed"}:
            safe_data = {
                key: data.get(key)
                for key in (
                    "source_id",
                    "channel",
                    "content_characters",
                    "error",
                    "error_type",
                )
                if key in data
            }
        elif event.kind == "extractive_evidence_handoff_created":
            source_ids = data.get("source_ids")
            excerpts = data.get("excerpts")
            safe_data = {
                "source_ids": source_ids if isinstance(source_ids, list) else [],
                "excerpt_max_chars": data.get("excerpt_max_chars"),
                "excerpt_count": len(excerpts) if isinstance(excerpts, list) else 0,
            }
        elif event.kind in {
            "model_failure",
            "researcher_stopped",
            "budget_blocked",
            "tool_call_deferred",
            "run_finished",
        }:
            safe_data = {
                key: data.get(key)
                for key in (
                    "role",
                    "call_index",
                    "phase",
                    "elapsed_seconds",
                    "error_type",
                    "resource",
                    "tool",
                    "reason",
                    "status",
                    "terminal_reason",
                    "model_calls_used",
                    "tool_calls_used",
                    "fetched_source_count",
                )
                if key in data
            }
        elif event.kind == "researcher_turn":
            safe_data = {"turn": data.get("turn")}
        else:
            safe_data = {}
        record = {
            "schema_version": "odr-live-progress/v1",
            "run_id": self.run_id,
            "sequence": event.sequence,
            "timestamp": event.timestamp,
            "kind": event.kind,
            "data": safe_data,
        }
        try:
            with self._progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                handle.flush()
        except OSError:
            # Diagnostics must not alter the experimental execution path.
            return

    def _capture_development_transcript(
        self,
        *,
        role: str,
        call_index: int,
        decision_stream_id: str | None,
        report: bool,
        messages: list[Any],
        tool_definitions: Iterable[object] | None,
        tool_binding_options: Mapping[str, object] | None,
        response: Any | None = None,
        error_type: str | None = None,
    ) -> None:
        """Append one opt-in ODR model-call transcript for development SFT work.

        This is deliberately a runner-local file export rather than a trace or
        telemetry extension.  It captures the current ODR call contract only
        when a caller explicitly supplies an absolute output path.
        """

        if self._development_transcript_path is None or role not in {
            "supervisor",
            "researcher",
        }:
            return
        if not decision_stream_id:
            raise ValueError(
                "decision_stream_id is required for captured policy calls"
            )
        record = _redact_transcript_secrets(
            {
                "schema_version": "odr-training-transcript/v1",
                "run_id": self.run_id,
                "call_index": call_index,
                "role": role,
                "decision_stream_id": decision_stream_id,
                "phase": "report" if report else "research",
                "messages": messages_to_dict(messages),
                "tools": [
                    convert_to_openai_tool(tool)
                    for tool in (tool_definitions or ())
                ],
                "tool_binding_options": dict(tool_binding_options or {}),
                "response": message_to_dict(response)
                if response is not None
                else None,
                "error_type": error_type,
            }
        )
        with self._lock:
            with self._development_transcript_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
                handle.write(encoded + "\n")

    def _invoke(
        self,
        *,
        role: str,
        model: Any,
        messages: list[Any],
        report: bool = False,
        tool_definitions: Iterable[object] | None = None,
        tool_binding_options: Mapping[str, object] | None = None,
        decision_stream_id: str | None = None,
    ) -> Any:
        self._check_cancelled()
        with self._lock:
            try:
                call_index = (
                    self._run_budget.authorize_report_call()
                    if report
                    else self._run_budget.authorize_research_call()
                )
            except ResearchBudgetExhausted as exc:
                self._event(
                    "budget_blocked",
                    resource="model",
                    role=role,
                    phase="report" if report else "research",
                    reason=str(exc),
                )
                raise
            self._model_calls_used = self._run_budget.total_model_calls
        message_count, input_content_characters, message_types = _message_input_metrics(
            messages
        )
        started = monotonic()
        self._event(
            "model_request_started",
            role=role,
            call_index=call_index,
            phase="report" if report else "research",
            input_message_count=message_count,
            input_content_characters=input_content_characters,
            input_message_types=message_types,
            thinking_mode=self._thinking_mode,
        )
        try:
            self._check_cancelled()
            response = model.invoke(messages)
        except Exception as exc:
            self._capture_development_transcript(
                role=role,
                call_index=call_index,
                decision_stream_id=decision_stream_id,
                report=report,
                messages=messages,
                tool_definitions=tool_definitions,
                tool_binding_options=tool_binding_options,
                error_type=type(exc).__name__,
            )
            if self.usage_ledger is not None:
                self.usage_ledger.record_failure(role=role)
            self._event(
                "model_failure",
                role=role,
                call_index=call_index,
                error_type=type(exc).__name__,
                elapsed_seconds=round(monotonic() - started, 3),
                thinking_mode=self._thinking_mode,
            )
            raise
        self._capture_development_transcript(
            role=role,
            call_index=call_index,
            decision_stream_id=decision_stream_id,
            report=report,
            messages=messages,
            tool_definitions=tool_definitions,
            tool_binding_options=tool_binding_options,
            response=response,
        )
        if self.usage_ledger is not None:
            self.usage_ledger.record_response(role=role, response=response)
        calls = list(getattr(response, "tool_calls", None) or [])
        self._event(
            "model_response",
            role=role,
            call_index=call_index,
            phase="report" if report else "research",
            elapsed_seconds=round(monotonic() - started, 3),
            thinking_mode=self._thinking_mode,
            **_response_observability(response),
            response=_message_text(response),
            tool_calls=[
                {"name": str(call.get("name", "")), "args": dict(call.get("args") or {})}
                for call in calls
                if isinstance(call, Mapping)
            ],
        )
        self._check_cancelled()
        return response

    def _spend_tool(
        self,
        *,
        name: str,
        task: str,
        task_budget: _ResearchTaskToolBudget | None,
    ) -> bool:
        self._check_cancelled()
        with self._lock:
            if (
                self.policy.max_tool_calls is not None
                and self._tool_calls_used >= self.policy.max_tool_calls
            ):
                self._event("budget_blocked", resource="tool", tool=name, task=task)
                return False
            if task_budget is not None and task_budget.used_tool_calls >= task_budget.maximum_tool_calls:
                self._event(
                    "budget_blocked",
                    resource="tool",
                    tool=name,
                    task=task,
                    reason="task_allowance_reserved_for_pending_tasks",
                )
                return False
            self._tool_calls_used += 1
            if task_budget is not None:
                task_budget.used_tool_calls += 1
            return True

    def _register_discovery(self, resource: DiscoveredResource, *, task: str) -> OdrSource:
        with self._lock:
            source_key = (resource.channel, resource.resource_locator)
            source = self._sources_by_url.get(source_key)
            if source is None:
                source = OdrSource(
                    source_id=f"source-{self._next_source_number:03d}",
                    url=resource.resource_locator,
                    title=resource.title,
                    snippet=resource.snippet,
                    channel=resource.channel,
                )
                self._next_source_number += 1
                self._sources_by_url[source_key] = source
                self._sources_by_id[source.source_id] = source
            if task not in source.discovered_for:
                source.discovered_for.append(task)
            return source

    def _search(
        self,
        *,
        query: str,
        task: str,
        task_budget: _ResearchTaskToolBudget | None,
        connector: SourceConnector,
        channel: str,
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            return "Search query was empty."
        if not self._spend_tool(name=f"search_{channel}", task=task, task_budget=task_budget):
            return "This task's research-action allowance is exhausted; use fetched sources and finish research."
        try:
            resources = tuple(connector.search(query.strip()))
        except Exception as exc:
            self._event(
                "search_failed",
                task=task,
                channel=channel,
                error_type=type(exc).__name__,
            )
            return f"{channel} search failed without a usable result."
        sources = [
            self._register_discovery(resource, task=task)
            for resource in resources[: self.policy.max_sources_per_search]
        ]
        self._event(
            "search_completed",
            task=task,
            channel=channel,
            query=query.strip(),
            result_source_ids=[source.source_id for source in sources],
        )
        if not sources:
            return "No usable search results were returned. Try another query."
        return "\n\n".join(
            f"[{source.source_id}] {source.title}\nURL: {source.url}\nSnippet: {source.snippet}"
            for source in sources
        )

    def _read_source(
        self,
        *,
        source_id: str,
        task: str,
        task_budget: _ResearchTaskToolBudget | None,
        start: int | None = None,
        end: int | None = None,
    ) -> str:
        source = self._sources_by_id.get(source_id)
        if source is None:
            return "Unknown source id. Use only IDs returned by a search tool."
        if start is not None or end is not None:
            if source.content is None:
                return "Read this source without a range first to obtain its snapshot."
            left = 0 if start is None else start
            right = min(len(source.content), left + self.policy.source_view_max_chars) if end is None else end
            if not 0 <= left < right <= len(source.content):
                return f"Use 0 <= start < end <= {len(source.content)} (end exclusive)."
            right = min(right, left + self.policy.source_view_max_chars)
            if not self._spend_tool(name="read_source", task=task, task_budget=task_budget):
                return "This task's research-action allowance is exhausted; use fetched sources and finish research."
            if task not in source.fetched_for:
                source.fetched_for.append(task)
            self._event("source_range_read", task=task, source_id=source_id,
                        start=left, end=right, content_characters=right - left)
            return (f"[{source.source_id}] {source.title}\nURL: {source.url}\n"
                    f"Snapshot characters {left}:{right} of {len(source.content)} (end exclusive):\n\n"
                    + source.content[left:right])
        # Concurrent research units may discover the same canonical URL.  A
        # per-source lock makes its full fetch and snapshot single-flight;
        # another worker then reuses exactly the same immutable text instead
        # of spending a duplicate tool call or racing the source state.
        with source._fetch_lock:
            if source.content is None and source.fetch_error in (None, "public_fetch_dns_failed"):
                if not self._spend_tool(name="read_source", task=task, task_budget=task_budget):
                    return "This task's research-action allowance is exhausted; use fetched sources and finish research."
                try:
                    connector = (
                        self.collection_connector
                        if source.channel == "collection"
                        else self.connector
                    )
                    if connector is None:
                        raise RuntimeError("collection_connector_unavailable")
                    fetched = connector.fetch(
                        DiscoveredResource(
                            resource_locator=source.url,
                            title=source.title,
                            snippet=source.snippet,
                            channel=source.channel,
                        )
                    )
                    if not isinstance(fetched, FetchedResource):
                        raise TypeError("connector.fetch returned an invalid resource")
                    source.content = fetched.content
                    source.fetch_error = None
                    source.retrieved_at = fetched.retrieved_at
                    source.content_sha256 = sha256(fetched.content.encode("utf-8")).hexdigest()
                    source.title = fetched.title or source.title
                    source.citation_aliases = _citation_aliases(fetched.citation_aliases)
                    source.retrieval_method = fetched.retrieval_method
                    self._event(
                        "fetch_completed",
                        task=task,
                        source_id=source.source_id,
                        channel=source.channel,
                        url=source.url,
                        content_sha256=source.content_sha256,
                        content_characters=len(fetched.content),
                        retrieval_method=source.retrieval_method,
                    )
                except Exception as exc:
                    failure_code = safe_fetch_failure_code(exc)
                    source.fetch_error = failure_code
                    self._event(
                        "fetch_failed",
                        task=task,
                        source_id=source.source_id,
                        channel=source.channel,
                        error=failure_code,
                        error_type=type(exc).__name__,
                    )
        if source.content is None:
            return f"[{source.source_id}] could not be fetched: {source.fetch_error or 'unknown failure'}."
        if task not in source.fetched_for:
            source.fetched_for.append(task)
        return (
            f"[{source.source_id}] {source.title}\nURL: {source.url}\n"
            f"Retrieved: {source.retrieved_at}\n\n"
            f"{_source_excerpt(source.content, maximum=self.policy.source_view_max_chars)}"
        )

    def _search_in_source(
        self, *, source_id: str, query: str, task: str,
        task_budget: _ResearchTaskToolBudget | None,
    ) -> str:
        source = self._sources_by_id.get(source_id)
        if source is None or source.content is None:
            return "Read this source first; document search uses an already fetched snapshot."
        words = query.split()
        if not words:
            return "Supply a keyword or short phrase to locate in this document."
        if not self._spend_tool(name="search_in_source", task=task, task_budget=task_budget):
            return "This task's research-action allowance is exhausted; use fetched sources and finish research."
        # Match across PDF-extracted line breaks while preserving snapshot offsets.
        pattern = re.compile(r"\s+".join(re.escape(word) for word in words), re.IGNORECASE)
        matches = []
        for match in pattern.finditer(source.content):
            matches.append((match.start(), match.end()))
            if len(matches) == 5:
                break
        self._event("source_text_searched", task=task, source_id=source_id,
                    query=query, matches=[{"start": a, "end": b} for a, b in matches])
        header = f"[{source_id}] {source.title}\nURL: {source.url}\nSnapshot length: {len(source.content)} characters.\n"
        if not matches:
            return header + "No literal phrase match. Try a shorter keyword or alternate wording; this does not establish that the topic is absent."
        excerpts = []
        for start, end in matches:
            left, right = max(0, start - 300), min(len(source.content), end + 300)
            excerpts.append(f"Match {start}:{end}; context {left}:{right}:\n{source.content[left:right]}")
        return header + "Up to 5 matches; use read_source(start, end) to expand context.\n\n" + "\n\n".join(excerpts)


    def _research_tools(
        self,
        *,
        task: str,
        task_budget: _ResearchTaskToolBudget | None,
    ) -> dict[str, Any]:
        from langchain_core.tools import tool

        @tool
        def search_web(query: str) -> str:
            """Search the authorized public web and return source IDs, URLs, and snippets."""

            return self._search(
                query=query,
                task=task,
                task_budget=task_budget,
                connector=self.connector,
                channel="web",
            )

        tools: dict[str, Any] = {"search_web": search_web}

        if self.collection_connector is not None:

            @tool
            def search_collection(query: str) -> str:
                """Search the user-selected local Collection and return source IDs and snippets."""

                return self._search(
                    query=query,
                    task=task,
                    task_budget=task_budget,
                    connector=self.collection_connector,
                    channel="collection",
                )

            tools["search_collection"] = search_collection

        @tool
        def read_source(source_id: str, start: int | None = None, end: int | None = None) -> str:
            """Fetch a source overview, or expand a fetched snapshot using zero-based character start/end (end exclusive). Long overviews omit text; use search_in_source to locate missing sections."""

            return self._read_source(
                source_id=source_id,
                task=task,
                task_budget=task_budget,
                start=start,
                end=end,
            )

        @tool
        def search_in_source(source_id: str, query: str) -> str:
            """Find a keyword or literal phrase in an already fetched full snapshot; return original excerpts and character positions, without a network request."""
            return self._search_in_source(source_id=source_id, query=query,
                                          task=task, task_budget=task_budget)

        @tool
        def finish_research() -> str:
            """Signal that this focused research task is complete."""

            return "Research complete."

        tools["read_source"] = read_source
        tools["search_in_source"] = search_in_source
        tools["finish_research"] = finish_research
        return tools

    def _tool_messages(
        self,
        response: Any,
        tools: Mapping[str, Any],
        *,
        task: str,
    ) -> tuple[list[Any], bool, bool]:
        """Execute one web action per turn, including provider fallback.

        Open Deep Research requests ``parallel_tool_calls=False`` when it
        binds tools.  Some OpenAI-compatible providers accept that parameter
        yet still return multiple calls in one response.  Returning a tool
        result for every call preserves the provider's message protocol, but
        only the first permitted web action is executed; the rest are deferred
        until the model has seen the actual result.  This is the local,
        provider-compatibility implementation of the upstream's serial tool
        semantics, not a relevance ranking or a topic rule.
        """
        from langchain_core.messages import ToolMessage

        tool_messages: list[Any] = []
        completed = False
        executed_web_actions = 0
        for call in list(getattr(response, "tool_calls", None) or []):
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("name", ""))
            call_id = str(call.get("id", uuid4().hex))
            if name == "finish_research":
                if executed_web_actions:
                    content = (
                        "Research completion deferred: inspect the web-action result from this "
                        "turn before deciding whether research is complete."
                    )
                    self._event(
                        "tool_call_deferred",
                        task=task,
                        tool=name,
                        reason="web_action_already_executed_this_turn",
                    )
                else:
                    completed = True
                    content = "Research complete."
                tool_messages.append(
                    ToolMessage(content=content, name=name, tool_call_id=call_id)
                )
                continue
            tool = tools.get(name)
            if tool is None:
                content = f"Unknown tool {name!r}."
            elif executed_web_actions >= self.policy.max_web_actions_per_research_turn:
                content = (
                    "Research action deferred: this run executes one source action per research turn. "
                    "Use the returned result to choose the next action."
                )
                self._event(
                    "tool_call_deferred",
                    task=task,
                    tool=name,
                    reason="source_action_turn_limit",
                )
            else:
                try:
                    content = str(tool.invoke(dict(call.get("args") or {})))
                    executed_web_actions += 1
                except Exception as exc:
                    content = f"Tool execution failed: {type(exc).__name__}."
            tool_messages.append(ToolMessage(content=content, name=name, tool_call_id=call_id))
        return tool_messages, completed, bool(executed_web_actions)

    def _source_linked_working_note(
        self,
        *,
        task: str,
        source_id: str,
        source_view: str,
        model: Any,
    ) -> str:
        """Turn one fetched-source view into bounded, traceable working context.

        The complete fetched body remains on ``OdrSource.content`` and is
        written as a source snapshot.  This helper changes only what later
        researcher turns and focused-trail compression receive: a model reads
        one source view once, then later calls receive its source-ID/URL-linked
        note instead of another copy of that view.
        """
        source = self._sources_by_id.get(source_id)
        if source is None or source.content is None:
            return source_view
        from langchain_core.messages import HumanMessage, SystemMessage

        header = f"[{source.source_id}] {source.title}\nURL: {source.url}"
        prompt = """Create a compact internal evidence note for a focused researcher.
Use only the fetched source provided. Preserve the exact source ID and URL supplied
by the runtime, the facts relevant to the focused task, material caveats, and any
short quotation needed to keep a claim checkable. Do not add facts, sources, or
medical/product advice from memory. This note will replace the source view only in
later working context; the complete source snapshot remains available for audit.
"""
        try:
            response = self._invoke(
                role="source_note",
                model=model,
                messages=[
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            f"Focused task:\n{task}\n\n"
                            f"Fetched source view:\n{source_view}"
                        )
                    ),
                ],
            )
        except ResearchBudgetExhausted:
            self._event(
                "source_note_unavailable",
                task=task,
                source_id=source.source_id,
                reason="research_model_budget_exhausted",
            )
            return f"{header}\n\nNo working note was produced; inspect the retained source snapshot."
        except Exception as exc:
            self._event(
                "source_note_unavailable",
                task=task,
                source_id=source.source_id,
                reason="model_failure",
                error_type=type(exc).__name__,
            )
            return f"{header}\n\nNo working note was produced; inspect the retained source snapshot."

        note = _message_text(response)
        self._event(
            "source_note_created",
            task=task,
            source_id=source.source_id,
            url=source.url,
            source_view_characters=len(source_view),
            note_characters=len(note),
        )
        if not note:
            return f"{header}\n\nNo usable working note was returned; inspect the retained source snapshot."
        return f"{header}\n\nEvidence note:\n{note}"

    def _run_researcher(
        self,
        task: str,
        task_budget: _ResearchTaskToolBudget | None = None,
        researcher_slot: int = 0,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        tools = self._research_tools(task=task, task_budget=task_budget)
        # Directly adapted from Open Deep Research's multi-agent binding
        # (``legacy/multi_agent.py``): one source action per model response.
        # Without this, an otherwise capable model can emit a burst of search
        # calls in one turn, spending a task's finite retrieval slice before
        # it receives even one result to decide what to fetch.
        researcher_model_index = researcher_slot % len(self._research_llms)
        self._event(
            "researcher_model_selected",
            task=task,
            model_pool_index=researcher_model_index,
            model_pool_size=len(self._research_llms),
        )
        model = self._research_llms[researcher_model_index].bind_tools(
            list(tools.values()),
            parallel_tool_calls=False,
        )
        collection_note = (
            f"\nThe selected local source pack is: {self.collection_context}. "
            "It is a fixed local corpus; use its document links when it is useful.\n"
            if self.collection_context
            else ""
        )
        budget_note = ""
        if self.policy.budget_aware_prompting:
            shared_tools = (
                f"{self.policy.max_tool_calls} source actions shared across all focused tasks"
                if self.policy.max_tool_calls is not None
                else "a shared source-action budget across focused tasks"
            )
            budget_note = (
                f"\nThis experimental profile allows at most {self.policy.max_researcher_turns} "
                f"research turns for this task and has {shared_tools}. This runtime executes "
                "at most one source action in each turn. Choose the next highest-value search "
                "or read only after seeing the previous result; prioritize enough fetched "
                "primary evidence to support the assigned question, and label an evidence gap "
                "rather than expanding scope or listing unverified links.\n"
            )
        finding_handoff = (
            """response, list concise findings with the exact fetched `[source-###]` handles that
support them. Do not treat a search-result snippet or an unread candidate URL as
evidence, and do not copy raw URLs into the findings. Do not invent source links or
product abilities.
"""
            if self.policy.source_handle_evidence_handoff
            else """response, list concise findings with the exact fetched source links that support them.
Do not invent source links or product abilities.
"""
        )
        prompt = """You are a focused web researcher. Use the available tools to answer the
assigned research question from primary or official sources where possible.

Long source overviews omit parts of the saved full text. If a relevant section is
missing from an overview, use search_in_source with a keyword or short phrase,
then read_source with character start/end to expand its context before searching
for another copy of the document. These actions use the existing tool budget.

Work in a short tool-calling loop: search the public web, the local Collection, or
both when they are useful to the assigned question; then read the most relevant
returned sources. Do not rely on a search snippet as evidence. Stop when you have
enough specific facts for a later writer, and call finish_research. In your last prose
""" + finding_handoff + collection_note + budget_note
        base_messages: list[Any] = [SystemMessage(content=prompt), HumanMessage(content=task)]
        messages = list(base_messages)
        working_messages: list[str] = []
        working_notes_by_source: dict[str, str] = {}
        executed_search_queries: list[str] = []
        fetched_source_ids: list[str] = []
        failed_sources_by_id: dict[str, str] = {}
        for turn in range(self.policy.max_researcher_turns):
            try:
                response = self._invoke(
                    role="researcher",
                    model=model,
                    messages=messages,
                    tool_definitions=tools.values(),
                    tool_binding_options={"parallel_tool_calls": False},
                    decision_stream_id=(
                        "researcher:"
                        + sha256(task.encode("utf-8")).hexdigest()
                    ),
                )
            except ResearchBudgetExhausted:
                self._event("researcher_stopped", task=task, reason="model_budget")
                break
            except Exception:
                self._event("researcher_stopped", task=task, reason="model_failure")
                break
            messages.append(response)
            response_text = _message_text(response)
            if response_text and not self.policy.source_handle_evidence_handoff:
                working_messages.append(response_text)
            tool_messages, completed, _ = self._tool_messages(
                response,
                tools,
                task=task,
            )
            calls_by_id = {
                str(call.get("id", "")): call
                for call in list(getattr(response, "tool_calls", None) or [])
                if isinstance(call, Mapping)
            }
            source_id_by_call_id = {
                call_id: str((call.get("args") or {}).get("source_id", ""))
                for call_id, call in calls_by_id.items()
                if str(call.get("name", "")) == "read_source"
            }
            if tool_messages:
                context_messages = tool_messages
                if (
                    self.policy.source_linked_working_notes
                    or self.policy.researcher_working_memory
                ):
                    context_messages = []
                    for message in tool_messages:
                        source_id = source_id_by_call_id.get(str(message.tool_call_id), "")
                        if message.name != "read_source" or not source_id:
                            context_messages.append(message)
                            continue
                        context_messages.append(
                            ToolMessage(
                                content=self._source_linked_working_note(
                                    task=task,
                                    source_id=source_id,
                                    source_view=str(message.content),
                                    model=self._research_llms[researcher_model_index],
                                ),
                                name=message.name,
                                tool_call_id=message.tool_call_id,
                            )
                        )
                messages.extend(context_messages)
                if self.policy.source_handle_evidence_handoff:
                    # H is an evidence boundary, not merely a writer instruction.
                    # Search results and researcher prose are useful navigation
                    # context, but can contain unread candidate handles/snippets.
                    # Only a successful source read may enter compression.  Keep
                    # the existing N/M source-linked projection when it is opted
                    # in; otherwise preserve the successful source view itself.
                    evidence_messages = (
                        context_messages
                        if (
                            self.policy.source_linked_working_notes
                            or self.policy.researcher_working_memory
                        )
                        else tool_messages
                    )
                    for message in evidence_messages:
                        source_id = source_id_by_call_id.get(
                            str(message.tool_call_id), ""
                        )
                        source = self._sources_by_id.get(source_id)
                        if (
                            message.name != "read_source"
                            or source is None
                            or source.content is None
                        ):
                            continue
                        if source.source_id not in fetched_source_ids:
                            fetched_source_ids.append(source.source_id)
                        if self.policy.evidence_handoff_mode == "generative":
                            working_messages.append(str(message.content))
                else:
                    working_messages.extend(str(message.content) for message in context_messages)
                if self.policy.researcher_working_memory:
                    context_by_call_id = {
                        str(message.tool_call_id): message for message in context_messages
                    }
                    for message in tool_messages:
                        call = calls_by_id.get(str(message.tool_call_id))
                        if call is None or str(message.content).startswith(
                            "Research action deferred:"
                        ):
                            continue
                        name = str(call.get("name", ""))
                        args = call.get("args") or {}
                        if name in {"search_web", "search_collection"}:
                            query = str(args.get("query", "")).strip()
                            if query and query not in executed_search_queries:
                                executed_search_queries.append(query)
                        if name != "read_source":
                            continue
                        source_id = str(args.get("source_id", "")).strip()
                        source = self._sources_by_id.get(source_id)
                        if source is None:
                            continue
                        if source.content is None:
                            failed_sources_by_id[source_id] = source.fetch_error or "unavailable"
                            continue
                        if source_id not in fetched_source_ids:
                            fetched_source_ids.append(source_id)
                        note_message = context_by_call_id.get(str(message.tool_call_id))
                        if note_message is not None:
                            working_notes_by_source[source_id] = str(note_message.content)
            if completed or not tool_messages:
                break
            if self.policy.researcher_working_memory:
                evidence_notes: list[str] = []
                for source_id in fetched_source_ids:
                    source = self._sources_by_id[source_id]
                    note = working_notes_by_source.get(source_id, "")
                    if "Evidence note:\n" not in note:
                        note = (
                            f"[{source.source_id}] {source.title}\nURL: {source.url}\n\n"
                            "Evidence note unavailable; inspect the retained source snapshot."
                        )
                    evidence_notes.append(note)
                failed_sources = [
                    (
                        f"[{source_id}] URL: {self._sources_by_id[source_id].url} — "
                        f"note unavailable ({status})."
                    )
                    for source_id, status in failed_sources_by_id.items()
                ]
                memory = "\n\n".join(
                    (
                        "Working memory for this focused task. This is a state projection, not source text.",
                        "Executed search queries:\n"
                        + ("\n".join(f"- {query}" for query in executed_search_queries) or "- none yet"),
                        "Fetched source IDs:\n"
                        + ("\n".join(f"- {source_id}" for source_id in fetched_source_ids) or "- none yet"),
                        "Evidence notes from successfully fetched sources:\n"
                        + ("\n\n".join(evidence_notes) or "- none yet"),
                        "Fetch failures:\n"
                        + ("\n".join(failed_sources) or "- none yet"),
                    )
                )
                messages = [
                    *base_messages,
                    HumanMessage(content=memory),
                    response,
                    *context_messages,
                ]
                self._event(
                    "researcher_working_memory_rebuilt",
                    task=task,
                    turn=turn + 1,
                    source_note_count=len(working_notes_by_source),
                    search_query_count=len(executed_search_queries),
                    fetched_source_count=len(fetched_source_ids),
                    failed_source_count=len(failed_sources_by_id),
                    memory_characters=len(memory),
                )
            self._event("researcher_turn", task=task, turn=turn + 1)

        if self.policy.evidence_handoff_mode == "extractive":
            return self._build_extractive_evidence_handoff(
                task=task,
                source_ids=fetched_source_ids,
            )

        # The default remains Open Deep Research's raw focused trail.  The
        # explicit source-linked-notes profile above instead makes this a
        # source-addressable working trail while complete source snapshots
        # stay unchanged in the audit layer.
        return self._compress_research_trail(
            task=task,
            working_trail="\n\n".join(working_messages),
        )

    def _build_extractive_evidence_handoff(
        self,
        *,
        task: str,
        source_ids: Iterable[str],
    ) -> str:
        """Create a model-free, source-addressable research-to-writer handoff."""

        maximum = self.policy.evidence_handoff_excerpt_max_chars
        if maximum is None:
            raise RuntimeError(
                "extractive evidence handoff missing explicit excerpt maximum"
            )
        selected_source_ids = list(dict.fromkeys(source_ids))
        sections: list[str] = []
        spans: list[dict[str, object]] = []
        for source_id in selected_source_ids:
            source = self._sources_by_id.get(source_id)
            if source is None or source.content is None:
                continue
            excerpt, start, end = _extractive_handoff_window(
                source.content,
                focus=task,
                maximum=maximum,
            )
            if not excerpt:
                continue
            sections.append(
                f"[{source.source_id}] {source.title}\n"
                f"Snapshot characters {start}:{end} (verbatim):\n{excerpt}"
            )
            spans.append(
                {
                    "source_id": source.source_id,
                    "source_start": start,
                    "source_end": end,
                    "excerpt_characters": len(excerpt),
                }
            )
        self._event(
            "extractive_evidence_handoff_created",
            task=task,
            source_ids=[span["source_id"] for span in spans],
            excerpt_max_chars=maximum,
            excerpts=spans,
        )
        if not sections:
            return (
                "Extractive evidence handoff. No source snapshot could be projected; "
                "do not infer facts beyond the retained source registry."
            )
        return "\n\n".join(
            (
                "Extractive evidence handoff. Every passage below is a verbatim "
                "substring of a successfully fetched immutable snapshot. Source "
                "handles and character offsets are the audit boundary; do not infer "
                "support beyond the supplied passage.",
                f"Full user question:\n{self.query}",
                f"Focused task:\n{task}",
                "Evidence passages:\n" + "\n\n".join(sections),
            )
        )

    def _compress_research_trail(self, *, task: str, working_trail: str) -> str:
        """Compress fetched evidence without scheduling any further source action."""

        from langchain_core.messages import HumanMessage, SystemMessage

        compression_prompt = (
            """You are compressing one focused research trail for a later report writer.
Preserve specific facts, caveats, named products, limits, and the exact `[source-###]`
handle from each fetched page. A search result or snippet is navigation context, not
evidence: do not carry facts, raw URLs, or source handles forward from an unread
candidate. Organize by finding, attach fetched source handles to the findings they
support, and explicitly label an evidence gap when the trail lacks a fetched source.
Do not add facts from memory or invent citations.
"""
            if self.policy.source_handle_evidence_handoff
            else """You are compressing one focused research trail for a later report writer.
Preserve specific facts, caveats, named products, limits, and exact URLs from fetched
pages. Organize by finding. Do not add facts from memory and do not replace URLs
with invented citations. If the trail lacks evidence for a requested part, say so.
"""
        )
        for attempt in range(1, _MAX_COMPRESSION_ATTEMPTS + 1):
            try:
                response = self._invoke(
                    role="compression",
                    model=self.llm,
                    messages=[
                        SystemMessage(content=compression_prompt),
                        HumanMessage(
                            content=f"Focused task:\n{task}\n\nResearch trail:\n{working_trail}"
                        ),
                    ],
                )
                return _message_text(response) or "No usable compressed findings were returned."
            except ResearchBudgetExhausted:
                self._event(
                    "compression_stopped",
                    task=task,
                    reason="research_model_budget_exhausted",
                )
                return working_trail or "Research stopped before compression."
            except Exception as exc:
                self._event(
                    "compression_retry",
                    task=task,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                )
        return working_trail or "Research failed before compression."

    def _supervisor_prompt(self) -> str:
        # Adapted from Open Deep Research's current lead-researcher prompt.
        initial_delegation_instruction = (
            "For your first delegation, when the question has independent directions, "
            f"issue up to {self.policy.max_concurrent_research_units} distinct "
            "ConductResearch calls together. Do not serialize independent work merely "
            "to wait for earlier findings; reserve later rounds for evidence-driven gaps."
            if self.policy.initial_delegation_strategy == "parallel_first"
            else "For the first delegation, choose the number of focused tasks adaptively "
            "from the question and wait for findings before assigning follow-up work."
        )
        budget_note = ""
        if self.policy.budget_aware_prompting:
            shared_tools = (
                f"{self.policy.max_tool_calls} source actions shared across all tasks"
                if self.policy.max_tool_calls is not None
                else "a shared source-action budget across all tasks"
            )
            budget_note = (
                f"\nThis experimental profile has {shared_tools}. Decompose only to a scope "
                "that leaves each focused task enough searches and reads to obtain evidence. "
                "Do not turn a concise question into an exhaustive source checklist; preserve "
                "material gaps explicitly when the available evidence is insufficient."
            )
        return f"""You are supervising a deep-research report.  Decompose the user's question into
clear, non-overlapping focused research tasks.  Call ConductResearch once for each
task you need researched.  Each task must be standalone and state the facts,
alternatives, and source types to seek.  Do not search yourself and do not write the
final answer.

Think like a research manager with limited time and resources. Before delegating,
decide whether the question has independent directions; after every returned finding,
assess what is still missing and whether another task is material. Prefer one focused
researcher for simple questions. For a comparison explicitly requested by the user,
delegate clear, distinct, non-overlapping subtopics rather than assuming that one
source establishes every alternative.

Match the research scope to the user's actual request. Do not turn a concise
explanatory or practical question into a systematic review. Introduce a subquestion
only when it is needed for an explicit user requirement or for a material evidence
gap discovered during research. Do not invent unrequested comparison axes,
jurisdictions, population statistics, effect-size estimates, source lists, or
ancillary subjects. Prefer the smallest focused task set that can support a grounded
answer; use broader decomposition when the user explicitly asks for broad coverage.

This run supports at most {self.policy.breadth_budget} focused tasks in total and at
most {self.policy.max_concurrent_research_units} task(s) in a single delegation
round. You, not the runtime, must choose the decomposition from the user's question
and the evidence returned so far. Do not silently drop or replace a user requirement
to fit capacity: retain it as a labelled gap and use a later round if it remains
material.

{initial_delegation_instruction}

After you receive research results, identify genuine gaps and either delegate
follow-up work or call ResearchComplete. You have at most {self.policy.depth_budget}
follow-up round(s) after the initial research pass. You must use ConductResearch
before giving up.{budget_note}
"""

    def _serial_task_tool_budget(
        self,
        *,
        pending_task_count: int,
    ) -> _ResearchTaskToolBudget | None:
        """Reserve a generic web-action floor for later serial tasks.

        This intentionally does not inspect a task's topic or guess its
        importance.  The supervisor owns that choice.  If a prior task exits
        early, the global remainder automatically becomes available to the next
        task; only the minimum amount needed to keep not-yet-started tasks from
        having zero retrieval opportunities is held back.
        """

        if self.policy.max_tool_calls is None:
            return None
        if pending_task_count < 1:
            raise ValueError("pending_task_count must be positive")
        with self._lock:
            remaining = max(0, self.policy.max_tool_calls - self._tool_calls_used)
        per_pending_floor = min(
            self.policy.reserve_tools_per_pending_task,
            remaining // pending_task_count,
        )
        ceiling = max(0, remaining - per_pending_floor * (pending_task_count - 1))
        return _ResearchTaskToolBudget(maximum_tool_calls=ceiling)

    def _parallel_task_tool_budgets(
        self,
        *,
        task_count: int,
    ) -> list[_ResearchTaskToolBudget | None]:
        """Give simultaneous tasks fair fixed slices of a finite remainder.

        A serial run can return unused capacity to later tasks dynamically.  In
        a concurrent run, there is no stable "later", so equal slices prevent a
        race from deciding which independent task receives all retrieval work.
        """

        if self.policy.max_tool_calls is None:
            return [None] * task_count
        with self._lock:
            remaining = max(0, self.policy.max_tool_calls - self._tool_calls_used)
        base, extra = divmod(remaining, task_count)
        return [
            _ResearchTaskToolBudget(maximum_tool_calls=base + (1 if index < extra else 0))
            for index in range(task_count)
        ]

    def _record_task_tool_budget(
        self,
        *,
        task: str,
        task_budget: _ResearchTaskToolBudget | None,
        scheduling: str,
        pending_task_count: int,
    ) -> None:
        self._event(
            "task_tool_budget_assigned",
            task=task,
            scheduling=scheduling,
            pending_task_count=pending_task_count,
            maximum_tool_calls=(
                task_budget.maximum_tool_calls if task_budget is not None else None
            ),
        )

    def _supervisor_tasks(self) -> tuple[list[str], list[str]]:
        """Run initial work plus bounded supervisor-proposed follow-up rounds.

        The loop shape follows ``deep-research-harness.run_research_pipeline``:
        breadth limits the *total* work accepted, depth limits follow-ups after
        the initial pass, and every accepted task which fails to acquire a
        fetched source becomes an explicit unresolved item rather than an
        apparently covered empty note.
        """
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        model = self.llm.bind_tools([ConductResearch, ResearchComplete])
        messages: list[Any] = [
            SystemMessage(content=self._supervisor_prompt()),
            HumanMessage(content=self.query),
        ]
        state = self._research_state
        tasks = state.research_tasks
        notes = state.research_notes
        unexecuted_overflow_tasks: list[str] = []
        for round_index in range(self.policy.depth_budget + 1):
            try:
                response = self._invoke(
                    role="supervisor",
                    model=model,
                    messages=messages,
                    tool_definitions=(ConductResearch, ResearchComplete),
                    decision_stream_id="supervisor",
                )
            except ResearchBudgetExhausted:
                break
            except Exception:
                break
            messages.append(response)
            calls = [
                call
                for call in list(getattr(response, "tool_calls", None) or [])
                if isinstance(call, Mapping)
            ]
            delegated = [
                call for call in calls if str(call.get("name", "")) == "ConductResearch"
            ]
            if not delegated:
                if not tasks:
                    fallback = self.query
                    tasks.append(fallback)
                    self._event(
                        "fallback_delegation",
                        reason="supervisor_returned_no_research_call",
                        task=fallback,
                    )
                break
            remaining_task_slots = max(0, self.policy.breadth_budget - len(tasks))
            round_tasks = [
                str(call.get("args", {}).get("research_topic", "")).strip()
                for call in delegated
            ]
            round_tasks = [task for task in round_tasks if task and task not in tasks]
            if not round_tasks:
                break
            if len(round_tasks) > remaining_task_slots:
                # Never silently take the first N tasks: their order is a
                # model artifact, not a valid semantic priority.  Return
                # every unexecuted request to the supervisor, which must
                # choose a query-specific merge or a later prioritisation.
                for call in delegated:
                    messages.append(
                        ToolMessage(
                            content=(
                                "No research task was executed in this round. "
                                f"Only {remaining_task_slots} task slots remain. "
                                "Replan the complete user request into that many "
                                "focused tasks; retain distinct labelled subparts "
                                "for any requirements you merge."
                            ),
                            name="ConductResearch",
                            tool_call_id=str(call.get("id", uuid4().hex)),
                        )
                    )
                self._event(
                    "supervisor_replan_required",
                    round_index=round_index + 1,
                    proposed_task_count=len(round_tasks),
                    remaining_task_slots=remaining_task_slots,
                )
                # Do not mark these immediately: the next supervisor turn may
                # legitimately merge them into a smaller, grounded task.  If
                # that never happens, they are recorded below rather than
                # vanishing with the ephemeral tool message.
                unexecuted_overflow_tasks = round_tasks
                continue
            tasks.extend(round_tasks)
            unexecuted_overflow_tasks = []
            state.round_index = round_index
            self._event(
                "supervisor_delegated",
                round_index=round_index + 1,
                tasks=round_tasks,
            )
            if self.policy.max_concurrent_research_units == 1 or len(round_tasks) == 1:
                reports: list[str] = []
                for index, task in enumerate(round_tasks):
                    task_budget = self._serial_task_tool_budget(
                        pending_task_count=len(round_tasks) - index
                    )
                    self._record_task_tool_budget(
                        task=task,
                        task_budget=task_budget,
                        scheduling="serial",
                        pending_task_count=len(round_tasks) - index,
                    )
                    reports.append(self._run_researcher(task, task_budget))
            else:
                task_budgets = self._parallel_task_tool_budgets(task_count=len(round_tasks))
                for task, task_budget in zip(round_tasks, task_budgets):
                    self._record_task_tool_budget(
                        task=task,
                        task_budget=task_budget,
                        scheduling="parallel",
                        pending_task_count=len(round_tasks),
                    )
                with ThreadPoolExecutor(
                    max_workers=self.policy.max_concurrent_research_units,
                    thread_name_prefix="odr-researcher",
                ) as executor:
                    reports = list(
                        executor.map(
                            self._run_researcher,
                            round_tasks,
                            task_budgets,
                            range(len(round_tasks)),
                        )
                    )
            for task, report in zip(round_tasks, reports):
                has_fetched_source = any(
                    task in source.fetched_for and source.content is not None
                    for source in self._sources_by_id.values()
                )
                if has_fetched_source:
                    state.record_completed(task, report)
                else:
                    # Ported from ``_record_unresolved``: a worker outcome
                    # without retrieved evidence is not an empty success.
                    state.record_unresolved([task])
                    notes.append(report)
                    self._event(
                        "research_task_unresolved",
                        task=task,
                        reason="no_fetched_source",
                    )
            for call, task, report in zip(delegated, round_tasks, reports):
                messages.append(
                    ToolMessage(
                        content=(
                            f"{report}\n\n"
                            "Harness remaining capacity: "
                            f"{max(0, self.policy.breadth_budget - len(tasks))} focused task slot(s), "
                            f"{max(0, self.policy.depth_budget - round_index)} follow-up round(s)."
                        ),
                        name="ConductResearch",
                        tool_call_id=str(call.get("id", uuid4().hex)),
                    )
                )
            if round_index >= self.policy.depth_budget:
                break
        if not notes and tasks:
            for task in tasks:
                report = self._run_researcher(task)
                has_fetched_source = any(
                    task in source.fetched_for and source.content is not None
                    for source in self._sources_by_id.values()
                )
                if has_fetched_source:
                    state.record_completed(task, report)
                else:
                    state.record_unresolved([task])
                    notes.append(report)
                    self._event(
                        "research_task_unresolved",
                        task=task,
                        reason="no_fetched_source",
                    )
        state.record_unresolved(unexecuted_overflow_tasks)
        state.status = "research_complete"
        return tasks, notes

    def _static_one_pass_tasks(self, *, task_limit: int | None = None) -> list[str]:
        """Plan once before retrieval for the workflow counterfactual.

        The plan may use the same model as the agentic path, but it receives no
        source result and is never called again.  This preserves request
        decomposition while removing evidence-driven delegation and replanning.
        """

        from langchain_core.messages import HumanMessage, SystemMessage

        accepted_task_limit = self.policy.breadth_budget if task_limit is None else task_limit
        if accepted_task_limit < 1 or accepted_task_limit > self.policy.breadth_budget:
            raise ValueError("task_limit must be within the configured breadth budget")

        prompt = f"""You are planning a one-pass research workflow. Before any source result is
available, decompose the user's request into the smallest set of independent,
standalone focused research topics needed for a grounded answer. Return them in one
OnePassResearchPlan call. You will not receive search results and cannot add or revise
topics later, so do not invent ancillary topics. Use at most {accepted_task_limit}
topics and preserve every material user requirement within those topics.
"""
        model = self.llm.bind_tools([OnePassResearchPlan], parallel_tool_calls=False)
        raw_topics: list[str] = []
        fallback_reason: str | None = None
        try:
            response = self._invoke(
                role="workflow_planner",
                model=model,
                messages=[SystemMessage(content=prompt), HumanMessage(content=self.query)],
                tool_definitions=(OnePassResearchPlan,),
                tool_binding_options={"parallel_tool_calls": False},
            )
            for call in list(getattr(response, "tool_calls", None) or []):
                if not isinstance(call, Mapping):
                    continue
                if str(call.get("name", "")) != "OnePassResearchPlan":
                    continue
                values = (call.get("args") or {}).get("research_topics")
                if isinstance(values, list):
                    raw_topics = [str(value).strip() for value in values]
                break
            if not raw_topics:
                fallback_reason = "planner_returned_no_valid_plan"
        except ResearchBudgetExhausted:
            fallback_reason = "research_model_budget_exhausted"
        except Exception as exc:
            fallback_reason = f"planner_failure:{type(exc).__name__}"

        planned_topics: list[str] = []
        for topic in raw_topics:
            if topic and topic not in planned_topics:
                planned_topics.append(topic)
        overflow = planned_topics[accepted_task_limit:]
        tasks = planned_topics[:accepted_task_limit]
        if not tasks:
            tasks = [self.query]
            self._event(
                "workflow_plan_fallback",
                reason=fallback_reason or "planner_returned_empty_plan",
                task=self.query,
            )
        if overflow:
            self._research_state.record_unresolved(overflow)
            self._event(
                "workflow_plan_overflow",
                planned_task_count=len(planned_topics),
                accepted_task_count=len(tasks),
                unexecuted_tasks=overflow,
            )
        self._research_state.research_tasks.extend(tasks)
        self._event(
            "workflow_one_pass_plan",
            task_count=len(tasks),
            tasks=tasks,
            accepted_task_limit=accepted_task_limit,
            source_observations_available=False,
        )
        return tasks

    def _plan_evidence_requirements(
        self,
        *,
        requirement_limit: int,
    ) -> list[OdrEvidenceLedgerEntry]:
        """Decompose the request into traceable requirements before retrieval."""

        from langchain_core.messages import HumanMessage, SystemMessage

        if requirement_limit < 1 or requirement_limit > self.policy.breadth_budget:
            raise ValueError("requirement_limit must be within the configured breadth budget")
        prompt = f"""You are planning an evidence-ledger research workflow. Before any source
result is available, decompose the user's request into the smallest set of atomic,
user-facing requirements needed for a grounded answer. For every requirement, provide
one focused source-discovery query. Return one EvidenceRequirementPlan call with at
most {requirement_limit} requirements. Do not add ancillary requirements and do not
combine unrelated claims merely to reduce the count. If the user asks about multiple
independent cases, conditions, or comparison members, make each independently
checkable claim its own requirement; an example for one case does not cover the others.
"""
        raw_requirements: list[Mapping[str, object]] = []
        fallback_reason: str | None = None
        model = self.llm.bind_tools([EvidenceRequirementPlan], parallel_tool_calls=False)
        try:
            response = self._invoke(
                role="evidence_planner",
                model=model,
                messages=[SystemMessage(content=prompt), HumanMessage(content=self.query)],
                tool_definitions=(EvidenceRequirementPlan,),
                tool_binding_options={"parallel_tool_calls": False},
                decision_stream_id="evidence_planner",
            )
            for call in list(getattr(response, "tool_calls", None) or []):
                if not isinstance(call, Mapping):
                    continue
                if str(call.get("name", "")) != "EvidenceRequirementPlan":
                    continue
                values = (call.get("args") or {}).get("requirements")
                if isinstance(values, list):
                    raw_requirements = [value for value in values if isinstance(value, Mapping)]
                break
            if not raw_requirements:
                fallback_reason = "planner_returned_no_valid_plan"
        except ResearchBudgetExhausted:
            fallback_reason = "research_model_budget_exhausted"
        except Exception as exc:
            fallback_reason = f"planner_failure:{type(exc).__name__}"

        planned: list[tuple[str, str, str, str | None]] = []
        for raw_requirement in raw_requirements:
            requirement = str(raw_requirement.get("requirement") or "").strip()
            retrieval_query = str(raw_requirement.get("retrieval_query") or "").strip()
            if requirement and retrieval_query and requirement not in {
                existing_requirement for existing_requirement, _, _, _ in planned
            }:
                planned.append((requirement, retrieval_query, "web", None))
        if not planned:
            planned = [(self.query, self.query, "web", None)]
            self._event(
                "workflow_requirement_plan_fallback",
                reason=fallback_reason or "planner_returned_empty_plan",
                requirement=self.query,
            )
        overflow = planned[requirement_limit:]
        accepted = planned[:requirement_limit]
        return self._build_evidence_requirement_entries(
            accepted=accepted,
            overflow=overflow,
            plan_origin="model",
        )

    def _build_evidence_requirement_entries(
        self,
        *,
        accepted: list[tuple[str, str, str, str | None]],
        overflow: list[tuple[str, str, str, str | None]],
        plan_origin: str,
    ) -> list[OdrEvidenceLedgerEntry]:
        """Assign stable IDs to a validated pre-retrieval requirement plan."""

        entries = [
            OdrEvidenceLedgerEntry(
                requirement_id=f"req-{index:03d}",
                requirement=requirement,
                retrieval_query=retrieval_query,
                source_channel=source_channel,
                expected_collection_paper_id=expected_collection_paper_id,
            )
            for index, (
                requirement,
                retrieval_query,
                source_channel,
                expected_collection_paper_id,
            ) in enumerate(
                accepted, 1
            )
        ]
        if overflow:
            self._research_state.record_unresolved(
                [requirement for requirement, _, _, _ in overflow]
            )
            self._event(
                "workflow_requirement_plan_overflow",
                planned_requirement_count=len(entries) + len(overflow),
                accepted_requirement_count=len(entries),
                unexecuted_requirements=[requirement for requirement, _, _, _ in overflow],
            )
        self._research_state.research_tasks.extend(
            entry.requirement for entry in entries
        )
        self._event(
            "workflow_requirement_plan",
            requirement_count=len(entries),
            requirement_ids=[entry.requirement_id for entry in entries],
            requirements=[entry.requirement for entry in entries],
            source_channels=[entry.source_channel for entry in entries],
            expected_collection_paper_ids=[
                entry.expected_collection_paper_id for entry in entries
            ],
            source_observations_available=False,
            plan_origin=plan_origin,
        )
        return entries

    def _frozen_evidence_requirements(
        self,
        requirements: Iterable[Mapping[str, object]],
        *,
        requirement_limit: int,
    ) -> list[OdrEvidenceLedgerEntry]:
        """Load one externally frozen, validated requirement plan for a paired run."""

        if requirement_limit < 1 or requirement_limit > self.policy.breadth_budget:
            raise ValueError("requirement_limit must be within the configured breadth budget")
        planned: list[tuple[str, str, str, str | None]] = []
        channel_by_query: dict[str, str] = {}
        for raw_requirement in requirements:
            if not isinstance(raw_requirement, Mapping):
                raise TypeError("frozen evidence requirements must be mappings")
            requirement = str(raw_requirement.get("requirement") or "").strip()
            retrieval_query = str(raw_requirement.get("retrieval_query") or "").strip()
            source_channel = str(raw_requirement.get("source_channel") or "web").strip()
            expected_collection_paper_id = str(
                raw_requirement.get("expected_collection_paper_id") or ""
            ).strip()
            if not requirement or not retrieval_query:
                raise ValueError(
                    "each frozen evidence requirement needs non-empty requirement and retrieval_query"
                )
            if source_channel not in {"collection", "web"}:
                raise ValueError(
                    "frozen evidence requirement source_channel must be collection or web"
                )
            if requirement in {
                existing_requirement for existing_requirement, _, _, _ in planned
            }:
                raise ValueError("frozen evidence requirements must be unique")
            prior_channel = channel_by_query.setdefault(retrieval_query, source_channel)
            if prior_channel != source_channel:
                raise ValueError(
                    "one frozen retrieval_query cannot be assigned to both source channels"
                )
            planned.append(
                (
                    requirement,
                    retrieval_query,
                    source_channel,
                    expected_collection_paper_id or None,
                )
            )
        if not planned:
            raise ValueError("frozen evidence requirements must not be empty")
        if len(planned) > requirement_limit:
            raise ValueError("frozen evidence requirements exceed the workflow breadth budget")
        return self._build_evidence_requirement_entries(
            accepted=planned,
            overflow=[],
            plan_origin="frozen",
        )

    def _fetch_ranked_source_batch(
        self,
        *,
        query: str,
        task: str,
        task_budget: _ResearchTaskToolBudget | None,
        source_channel: str | None = None,
    ) -> list[str]:
        """Fetch one bounded, rank-ordered batch of immutable source snapshots.

        The number of attempted candidates is the existing policy's explicit
        ``max_fetches_per_research_unit`` and is emitted in the trace and attempt
        artifact. Every candidate in the fixed batch is attempted in rank order;
        success never authorizes a new query or an Agent replanning turn.
        """

        if source_channel == "web":
            channels: list[tuple[str, SourceConnector]] = [("web", self.connector)]
        elif source_channel == "collection":
            if self.collection_connector is None:
                raise RuntimeError("collection_connector_unavailable")
            channels = [("collection", self.collection_connector)]
        elif source_channel is None:
            channels = [("web", self.connector)]
            if self.collection_connector is not None:
                channels.append(("collection", self.collection_connector))
        else:
            raise ValueError("source_channel must be collection, web, or None")
        selected_source_ids: list[str] = []
        fetched_source_ids: list[str] = []
        for channel, connector in channels:
            remaining_fetches = max(
                0, self.policy.max_fetches_per_research_unit - len(selected_source_ids)
            )
            if not remaining_fetches:
                break
            ranked_source_ids = self._static_search_source_ids(
                query=query,
                task=task,
                task_budget=task_budget,
                connector=connector,
                channel=channel,
            )
            for source_id in ranked_source_ids[:remaining_fetches]:
                selected_source_ids.append(source_id)
                self._read_source(
                    source_id=source_id,
                    task=task,
                    task_budget=task_budget,
                )
                source = self._sources_by_id.get(source_id)
                if source is not None and source.content is not None:
                    fetched_source_ids.append(source_id)
                    self._event(
                        "workflow_batch_fetch_succeeded",
                        task=task,
                        query=query,
                        source_id=source_id,
                        attempted_source_ids=selected_source_ids,
                    )
        self._event(
            "workflow_fixed_retrieval",
            task=task,
            channels=[name for name, _ in channels],
            required_source_channel=source_channel,
            selection_policy="connector_rank_order_all_fetches",
            selected_source_ids=selected_source_ids,
            fetched_source_ids=fetched_source_ids,
        )
        return fetched_source_ids

    def _format_evidence_candidates(
        self,
        entries: list[OdrEvidenceLedgerEntry],
        *,
        candidate_source_ids_by_requirement: Mapping[str, list[str]],
    ) -> tuple[dict[str, list[_EvidenceCandidate]], dict[str, list[str]], str]:
        """Render all fixed-batch candidates for one source-bound coverage review."""

        candidates_by_requirement: dict[str, list[_EvidenceCandidate]] = {}
        candidate_ids_by_requirement: dict[str, list[str]] = {}
        ledger_sections: list[str] = []
        for entry in entries:
            candidate_ids = list(
                dict.fromkeys(
                    candidate_source_ids_by_requirement.get(entry.requirement_id, [])
                )
            )
            candidate_ids_by_requirement[entry.requirement_id] = candidate_ids
            candidates_by_requirement[entry.requirement_id] = []
            rendered_candidates: list[str] = []
            for source_id in candidate_ids:
                source = self._sources_by_id.get(source_id)
                if source is None or source.content is None:
                    continue
                expected_paper_id = entry.expected_collection_paper_id
                if expected_paper_id is not None:
                    actual_paper_id = _collection_paper_id(source)
                    if actual_paper_id != expected_paper_id:
                        self._event(
                            "workflow_evidence_candidate_excluded",
                            requirement_id=entry.requirement_id,
                            source_id=source.source_id,
                            source_channel=source.channel,
                            expected_collection_paper_id=expected_paper_id,
                            actual_collection_paper_id=actual_paper_id,
                            reason="collection_paper_id_mismatch",
                        )
                        continue
                excerpt, start, end = _extractive_evidence_span(
                    source.content,
                    requirement=entry.requirement,
                    retrieval_query=entry.retrieval_query,
                    # A source-pinned Collection requirement has already selected its
                    # document. Reusing the vector hit's stale chunk as a lexical anchor
                    # can pull the evidence span away from the frozen requirement.
                    source_snippet=(
                        None
                        if entry.expected_collection_paper_id is not None
                        else source.snippet
                    ),
                    maximum=self.policy.evidence_excerpt_max_chars,
                )
                if not excerpt:
                    continue
                host = urlsplit(source.url).hostname or "unknown-host"
                candidate_index = len(candidates_by_requirement[entry.requirement_id]) + 1
                candidate = _EvidenceCandidate(
                    candidate_key=(
                        f"{entry.requirement_id}-candidate-{candidate_index:02d}"
                    ),
                    source_id=source.source_id,
                    excerpt=excerpt,
                    source_start=start,
                    source_end=end,
                )
                candidates_by_requirement[entry.requirement_id].append(candidate)
                rendered_candidates.append(
                    f"- candidate key {candidate.candidate_key} [{source.source_id}] "
                    f"provenance: "
                    f"channel={source.channel}; "
                    f"host={host}; title={source.title}\n"
                    f"  snapshot characters {start}:{end}:\n{excerpt}"
                )
            if rendered_candidates:
                ledger_sections.append(
                    f"{entry.requirement_id}: {entry.requirement}\n"
                    + "\n".join(rendered_candidates)
                )

        return (
            candidates_by_requirement,
            candidate_ids_by_requirement,
            "\n\n".join(ledger_sections) or "(none)",
        )

    def _attach_extract_to_requirement(
        self,
        entry: OdrEvidenceLedgerEntry,
        *,
        candidate: _EvidenceCandidate | None,
        support_quote: str | None = None,
    ) -> None:
        """Attach one reviewer-selected candidate and its exact quote to a ledger entry."""

        entry.source_id = None
        entry.excerpt = None
        entry.source_start = None
        entry.source_end = None
        entry.candidate_key = None
        entry.support_quote = None
        entry.support_start = None
        entry.support_end = None
        if candidate is None:
            entry.status = "no_fetched_source"
            return
        source = self._sources_by_id.get(candidate.source_id)
        if source is None or source.content is None:
            entry.status = "no_fetched_source"
            return
        if source.content[candidate.source_start : candidate.source_end] != candidate.excerpt:
            entry.status = "candidate_snapshot_mismatch"
            return
        quote = (support_quote or "").strip()
        quote_offset = candidate.excerpt.find(quote) if quote else -1
        if quote_offset < 0:
            entry.status = "no_extractable_span"
            return
        entry.source_id = source.source_id
        entry.excerpt = candidate.excerpt
        entry.source_start = candidate.source_start
        entry.source_end = candidate.source_end
        entry.candidate_key = candidate.candidate_key
        entry.support_quote = quote
        entry.support_start = candidate.source_start + quote_offset
        entry.support_end = entry.support_start + len(quote)
        entry.status = "evidence_extracted"
        if entry.requirement not in source.fetched_for:
            source.fetched_for.append(entry.requirement)
        self._event(
            "workflow_evidence_extract_selected",
            requirement_id=entry.requirement_id,
            requirement=entry.requirement,
            source_id=source.source_id,
            candidate_key=candidate.candidate_key,
            source_start=candidate.source_start,
            source_end=candidate.source_end,
            support_start=entry.support_start,
            support_end=entry.support_end,
            excerpt_characters=len(candidate.excerpt),
        )

    def _run_evidence_ledger_initial_fetches(
        self,
        entries: list[OdrEvidenceLedgerEntry],
    ) -> dict[str, list[str]]:
        """Run one fixed ranked-candidate batch per planned requirement."""

        task_budgets = self._parallel_task_tool_budgets(task_count=len(entries))
        for entry, task_budget in zip(entries, task_budgets):
            self._record_task_tool_budget(
                task=entry.requirement,
                task_budget=task_budget,
                scheduling="evidence_ledger_initial",
                pending_task_count=len(entries),
            )
        if self.policy.max_concurrent_research_units == 1 or len(entries) == 1:
            candidate_source_ids = [
                self._fetch_ranked_source_batch(
                    query=entry.retrieval_query,
                    task=entry.requirement,
                    task_budget=task_budget,
                    source_channel=entry.source_channel,
                )
                for entry, task_budget in zip(entries, task_budgets)
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.policy.max_concurrent_research_units, len(entries)),
                thread_name_prefix="odr-evidence-ledger",
            ) as executor:
                futures = [
                    executor.submit(
                        self._fetch_ranked_source_batch,
                        query=entry.retrieval_query,
                        task=entry.requirement,
                        task_budget=task_budget,
                        source_channel=entry.source_channel,
                    )
                    for entry, task_budget in zip(entries, task_budgets)
                ]
                candidate_source_ids = [future.result() for future in futures]
        return {
            entry.requirement_id: source_ids
            for entry, source_ids in zip(entries, candidate_source_ids)
        }

    def _review_evidence_ledger(
        self,
        entries: list[OdrEvidenceLedgerEntry],
        *,
        candidate_source_ids_by_requirement: Mapping[str, list[str]],
        phase: str,
        allow_repair: bool,
        disallow_web_repair: bool = False,
    ) -> tuple[str, list[OdrEvidenceLedgerEntry], str] | None:
        """Classify requirement support and optionally authorize one repair."""

        from langchain_core.messages import HumanMessage, SystemMessage

        valid_ids = {entry.requirement_id for entry in entries}
        (
            candidates_by_requirement,
            candidate_ids_by_requirement,
            candidate_ledger,
        ) = self._format_evidence_candidates(
            entries,
            candidate_source_ids_by_requirement=candidate_source_ids_by_requirement,
        )
        repair_instruction = (
            "You may propose exactly one focused repair query for unresolved requirements."
            if allow_repair
            else "This is the final review. research_topic must be null and repaired_requirement_ids empty."
        )
        prompt = """You are reviewing a requirement-level evidence ledger for a research report.
Each candidate extract is verbatim text from an immutable snapshot; its source handle and
character offsets make it independently checkable. Return a candidate_key and a contiguous
support_quote for a requirement only when that candidate's supplied extract directly supports
it. A fetched source, title, or plausible inference is not enough. For a multi-clause
requirement, the selected extract must directly support every material clause; support for one
listed case never establishes the others. For a requirement about source authority, provenance
may establish the source identity, but it never establishes a factual claim. Requirements you
do not list remain unresolved. Candidate keys are unique across this full ledger: copy the
exact key from the same requirement block and copy support_quote exactly from its extract. Do
not return source IDs, write the report, or invent facts.
""" + repair_instruction
        model = self.llm.bind_tools([EvidenceCoverageReview], parallel_tool_calls=False)
        try:
            response = self._invoke(
                role="coverage_reviewer",
                model=model,
                messages=[
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            f"User request:\n{self.query}\n\n"
                            f"Fetched evidence candidates ({phase}):\n{candidate_ledger}"
                        )
                    ),
                ],
                tool_definitions=(EvidenceCoverageReview,),
                tool_binding_options={"parallel_tool_calls": False},
                decision_stream_id=f"coverage_reviewer_{phase}",
            )
        except ResearchBudgetExhausted:
            self._event(
                "workflow_evidence_coverage_review_failed",
                phase=phase,
                reason="research_model_budget_exhausted",
            )
            for entry in entries:
                entry.status = "coverage_review_unavailable"
            return None
        except Exception as exc:
            self._event(
                "workflow_evidence_coverage_review_failed",
                phase=phase,
                reason=f"reviewer_failure:{type(exc).__name__}",
            )
            for entry in entries:
                entry.status = "coverage_review_unavailable"
            return None

        selected_call: Mapping[str, object] | None = None
        for call in list(getattr(response, "tool_calls", None) or []):
            if isinstance(call, Mapping) and str(call.get("name", "")) == "EvidenceCoverageReview":
                selected_call = call
                break
        arguments = selected_call.get("args") if selected_call is not None else {}
        arguments = arguments if isinstance(arguments, Mapping) else {}
        selected_by_requirement: dict[str, tuple[_EvidenceCandidate, str]] = {}
        for selection in arguments.get("covered_evidence", []):
            if not isinstance(selection, Mapping):
                continue
            requirement_id = str(selection.get("requirement_id") or "").strip()
            candidate_key = str(selection.get("candidate_key") or "").strip()
            support_quote = str(selection.get("support_quote") or "").strip()
            if requirement_id not in valid_ids:
                self._event(
                    "workflow_evidence_selection_rejected",
                    phase=phase,
                    requirement_id=requirement_id,
                    candidate_key=candidate_key,
                    reason="unknown_requirement_id",
                )
                continue
            candidates = candidates_by_requirement[requirement_id]
            candidate_ids = candidate_ids_by_requirement[requirement_id]
            candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.candidate_key == candidate_key
                ),
                None,
            )
            if candidate is None:
                self._event(
                    "workflow_evidence_selection_rejected",
                    phase=phase,
                    requirement_id=requirement_id,
                    candidate_key=candidate_key,
                    candidate_source_ids=candidate_ids,
                    candidate_keys=[candidate.candidate_key for candidate in candidates],
                    candidate_count=len(candidates),
                    reason="candidate_key_not_in_requirement_block",
                )
                continue
            if requirement_id in selected_by_requirement:
                self._event(
                    "workflow_evidence_selection_rejected",
                    phase=phase,
                    requirement_id=requirement_id,
                    candidate_key=candidate_key,
                    candidate_source_ids=candidate_ids,
                    reason="duplicate_requirement_selection",
                )
                continue
            matched_quote = _verbatim_quote_from_candidate(
                candidate.excerpt,
                support_quote,
            )
            if matched_quote is None:
                self._event(
                    "workflow_evidence_selection_rejected",
                    phase=phase,
                    requirement_id=requirement_id,
                    candidate_key=candidate_key,
                    candidate_source_id=candidate.source_id,
                    candidate_source_ids=candidate_ids,
                    support_quote_characters=len(support_quote),
                    reason="support_quote_not_in_candidate_excerpt",
                )
                continue
            support_quote, normalized_to_snapshot = matched_quote
            if normalized_to_snapshot:
                self._event(
                    "workflow_evidence_quote_normalized_to_snapshot",
                    phase=phase,
                    requirement_id=requirement_id,
                    candidate_key=candidate_key,
                    candidate_source_id=candidate.source_id,
                    supplied_quote_characters=len(
                        str(selection.get("support_quote") or "").strip()
                    ),
                    snapshot_quote_characters=len(support_quote),
                    normalization="formatting_only",
                )
            selected_by_requirement[requirement_id] = (candidate, support_quote)
        covered_ids: set[str] = set()
        for entry in entries:
            candidates = candidates_by_requirement[entry.requirement_id]
            candidate_ids = candidate_ids_by_requirement[entry.requirement_id]
            selected = selected_by_requirement.get(entry.requirement_id)
            if selected is not None:
                candidate, support_quote = selected
                self._attach_extract_to_requirement(
                    entry,
                    candidate=candidate,
                    support_quote=support_quote,
                )
                if (
                    entry.source_id == candidate.source_id
                    and entry.excerpt is not None
                    and entry.support_quote == support_quote
                ):
                    entry.status = "covered"
                    covered_ids.add(entry.requirement_id)
                else:
                    entry.status = "unresolved"
            else:
                self._attach_extract_to_requirement(entry, candidate=None)
                entry.status = "unresolved" if candidate_ids else "no_fetched_source"
            self._event(
                "workflow_evidence_source_selected",
                phase=phase,
                requirement_id=entry.requirement_id,
                candidate_source_ids=candidate_ids,
                rendered_candidate_count=len(candidates),
                selected_candidate_key=(candidate.candidate_key if selected is not None else None),
                selected_source_id=(candidate.source_id if selected is not None else None),
                status=entry.status,
            )
        uncovered_entries = [
            entry for entry in entries if entry.requirement_id not in covered_ids
        ]
        self._event(
            "workflow_evidence_coverage_checked",
            phase=phase,
            covered_requirement_ids=sorted(covered_ids),
            unresolved_requirement_ids=[entry.requirement_id for entry in uncovered_entries],
        )
        if not allow_repair:
            return None
        candidate = str(arguments.get("research_topic") or "").strip()
        uncovered_ids = {entry.requirement_id for entry in uncovered_entries}
        repaired_ids = [
            str(requirement_id).strip()
            for requirement_id in arguments.get("repaired_requirement_ids", [])
            if isinstance(requirement_id, str)
            and str(requirement_id).strip() in uncovered_ids
        ]
        repaired_ids = list(dict.fromkeys(repaired_ids))
        if not candidate or not repaired_ids:
            self._event(
                "workflow_evidence_repair_skipped",
                phase=phase,
                reason="reviewer_returned_no_mapped_repair",
            )
            return None
        repaired_entries = [
            entry for entry in entries if entry.requirement_id in repaired_ids
        ]
        if disallow_web_repair and any(
            entry.source_channel == "web" for entry in repaired_entries
        ):
            self._event(
                "workflow_evidence_repair_skipped",
                phase=phase,
                reason="frozen_web_plan_disallows_dynamic_repair",
                repaired_requirement_ids=repaired_ids,
            )
            return None
        repair_channels = {entry.source_channel for entry in repaired_entries}
        if len(repair_channels) != 1:
            self._event(
                "workflow_evidence_repair_skipped",
                phase=phase,
                reason="reviewer_repair_spans_source_channels",
                repaired_requirement_ids=repaired_ids,
            )
            return None
        repair_channel = next(iter(repair_channels))
        known_channel = next(
            (
                entry.source_channel
                for entry in entries
                if entry.retrieval_query == candidate
            ),
            repair_channel,
        )
        if known_channel != repair_channel:
            self._event(
                "workflow_evidence_repair_skipped",
                phase=phase,
                reason="repair_query_assigned_to_other_source_channel",
                repair_query=candidate,
                source_channel=repair_channel,
            )
            return None
        self._event(
            "workflow_evidence_repair_planned",
            phase=phase,
            repair_query=candidate,
            repaired_requirement_ids=repaired_ids,
            source_channel=repair_channel,
        )
        return candidate, repaired_entries, repair_channel

    def _ledger_notes(self, entries: Iterable[OdrEvidenceLedgerEntry]) -> list[str]:
        """Render only reviewed ledger evidence for the report writer."""

        notes: list[str] = []
        for entry in entries:
            if (
                entry.status == "covered"
                and entry.source_id is not None
                and entry.excerpt is not None
                and entry.source_start is not None
                and entry.source_end is not None
                and entry.candidate_key is not None
                and entry.support_quote is not None
                and entry.support_start is not None
                and entry.support_end is not None
            ):
                notes.append(
                    f"Requirement {entry.requirement_id}: {entry.requirement}\n"
                    f"Reviewed candidate key: {entry.candidate_key}\n"
                    f"Reviewed support quote at snapshot characters "
                    f"{entry.support_start}:{entry.support_end}:\n"
                    f"{entry.support_quote}\n"
                    f"Evidence [{entry.source_id}] at snapshot characters "
                    f"{entry.source_start}:{entry.source_end}:\n{entry.excerpt}"
                )
        return notes

    def _plan_evidence_brief(
        self,
        entries: Iterable[OdrEvidenceLedgerEntry],
    ) -> tuple[EvidenceBriefSection, ...] | None:
        """Ask the model to organize reviewed cards, optionally with Chinese explanations."""

        from langchain_core.messages import HumanMessage, SystemMessage

        cards = [
            entry
            for entry in entries
            if entry.status == "covered"
            and entry.candidate_key is not None
            and entry.support_quote is not None
        ]
        if not cards:
            self._event(
                (
                    "workflow_evidence_linked_narrative_brief_plan_skipped"
                    if self.policy.evidence_narrative_brief_enabled
                    else "workflow_evidence_brief_plan_skipped"
                ),
                reason="no_covered_evidence_cards",
            )
            return None
        card_by_evidence_id = {entry.candidate_key: entry for entry in cards}
        if len(card_by_evidence_id) != len(cards):
            self._event(
                (
                    "workflow_evidence_linked_narrative_brief_plan_skipped"
                    if self.policy.evidence_narrative_brief_enabled
                    else "workflow_evidence_brief_plan_skipped"
                ),
                reason="duplicate_evidence_id",
            )
            return None
        card_ledger = "\n\n".join(
            (
                f"Evidence ID: {entry.candidate_key}\n"
                f"Requirement ID: {entry.requirement_id}\n"
                f"Requirement: {entry.requirement}\n"
                f"Reviewed quote: {entry.support_quote}\n"
                f"Source ID: {entry.source_id}\n"
                f"Snapshot offsets: {entry.support_start}:{entry.support_end}"
            )
            for entry in cards
        )
        narrative = self.policy.evidence_narrative_brief_enabled
        prompt = (
            """You are writing an evidence-linked Chinese research brief from reviewed cards.
Return only an EvidenceBriefPlan tool call. Each section must contain exactly one supplied
requirement_id, its matching supplied evidence_id, and one or two Chinese sentences in
explanation_zh. Include every supplied card exactly once. Explain only what the matching reviewed
quote supports; preserve stated conditions, dates, applicability, uncertainty, and conflicts. Do
not use outside knowledge, search, read webpages, introduce a URL or source, merge cards, or
adjudicate a conflict. Never invent or alter an evidence ID or requirement ID. The renderer will
add the fixed English quote, source URL, and snapshot offsets after each explanation.
"""
            if narrative
            else """You are organizing reviewed evidence cards into a short research brief.
Return only an EvidenceBriefPlan tool call. Each section must contain exactly one supplied
requirement_id and its matching supplied evidence_id. Include every supplied card exactly once;
choose only their presentation order. Do not write claims, summaries, titles, recommendations,
or explanations: the renderer will add fixed labels and verbatim quotes itself. Never invent or
alter an evidence ID or requirement ID.
"""
        )
        self._run_budget.open_report_allowance()
        model = self.llm.bind_tools([EvidenceBriefPlan], parallel_tool_calls=False)
        try:
            response = self._invoke(
                role=(
                    "evidence_linked_narrative_brief_planner"
                    if narrative
                    else "evidence_brief_planner"
                ),
                model=model,
                messages=[
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            f"User request:\n{self.query}\n\n"
                            f"Reviewed evidence cards:\n{card_ledger}"
                        )
                    ),
                ],
                report=True,
                tool_definitions=(EvidenceBriefPlan,),
                tool_binding_options={"parallel_tool_calls": False},
                decision_stream_id=(
                    "evidence_linked_narrative_brief_planner"
                    if narrative
                    else "evidence_brief_planner"
                ),
            )
        except ResearchBudgetExhausted:
            self._event(
                (
                    "workflow_evidence_linked_narrative_brief_plan_rejected"
                    if narrative
                    else "workflow_evidence_brief_plan_rejected"
                ),
                reason="report_model_budget_exhausted",
            )
            return None
        except Exception as exc:
            self._event(
                (
                    "workflow_evidence_linked_narrative_brief_plan_rejected"
                    if narrative
                    else "workflow_evidence_brief_plan_rejected"
                ),
                reason=f"planner_failure:{type(exc).__name__}",
            )
            return None

        selected_call = next(
            (
                call
                for call in list(getattr(response, "tool_calls", None) or [])
                if isinstance(call, Mapping)
                and str(call.get("name", "")) == "EvidenceBriefPlan"
            ),
            None,
        )
        arguments = selected_call.get("args") if selected_call is not None else {}
        arguments = arguments if isinstance(arguments, Mapping) else {}
        raw_sections = arguments.get("sections")
        if not isinstance(raw_sections, list):
            self._event(
                (
                    "workflow_evidence_linked_narrative_brief_plan_rejected"
                    if narrative
                    else "workflow_evidence_brief_plan_rejected"
                ),
                reason="missing_sections",
            )
            return None
        sections: list[EvidenceBriefSection] = []
        reasons: list[str] = []
        seen_requirement_ids: set[str] = set()
        seen_evidence_ids: set[str] = set()
        for raw_section in raw_sections:
            if not isinstance(raw_section, Mapping):
                reasons.append("section_not_an_object")
                continue
            requirement_id = str(raw_section.get("requirement_id") or "").strip()
            evidence_id = str(raw_section.get("evidence_id") or "").strip()
            explanation_zh = str(raw_section.get("explanation_zh") or "").strip()
            entry = card_by_evidence_id.get(evidence_id)
            if entry is None:
                reasons.append("unknown_evidence_id")
            elif entry.requirement_id != requirement_id:
                reasons.append("evidence_id_not_bound_to_requirement")
            elif narrative and not explanation_zh:
                reasons.append("missing_explanation_zh")
            elif requirement_id in seen_requirement_ids:
                reasons.append("duplicate_requirement_id")
            elif evidence_id in seen_evidence_ids:
                reasons.append("duplicate_evidence_id")
            else:
                sections.append(
                    EvidenceBriefSection(
                        requirement_id=requirement_id,
                        evidence_id=evidence_id,
                        explanation_zh=explanation_zh or None,
                    )
                )
                seen_requirement_ids.add(requirement_id)
                seen_evidence_ids.add(evidence_id)
        expected_requirement_ids = {entry.requirement_id for entry in cards}
        expected_evidence_ids = set(card_by_evidence_id)
        if (
            reasons
            or seen_requirement_ids != expected_requirement_ids
            or seen_evidence_ids != expected_evidence_ids
        ):
            self._event(
                (
                    "workflow_evidence_linked_narrative_brief_plan_rejected"
                    if narrative
                    else "workflow_evidence_brief_plan_rejected"
                ),
                reason=("invalid_narrative_plan" if narrative else "invalid_evidence_plan"),
                validation_reasons=sorted(set(reasons)),
                expected_requirement_ids=sorted(expected_requirement_ids),
                rendered_requirement_ids=[section.requirement_id for section in sections],
                expected_evidence_ids=sorted(expected_evidence_ids),
                rendered_evidence_ids=[section.evidence_id for section in sections],
            )
            return None
        self._event(
            (
                "workflow_evidence_linked_narrative_brief_plan_accepted"
                if narrative
                else "workflow_evidence_brief_plan_accepted"
            ),
            sections=[
                (
                    {
                        "requirement_id": section.requirement_id,
                        "evidence_id": section.evidence_id,
                        "explanation_zh": section.explanation_zh,
                    }
                    if narrative
                    else {
                        "requirement_id": section.requirement_id,
                        "evidence_id": section.evidence_id,
                    }
                )
                for section in sections
            ],
            **(
                {"evidence_anchor_binding": "one_to_one"}
                if narrative
                else {"free_text_claims_allowed": False}
            ),
        )
        return tuple(sections)

    def _validated_extractive_entries(
        self,
        entries: Iterable[OdrEvidenceLedgerEntry],
    ) -> tuple[
        list[str],
        list[tuple[OdrEvidenceLedgerEntry, OdrSource]],
        list[tuple[str, str]],
    ]:
        """Return covered ledger entries whose selected quotes still match snapshots."""

        expected_ids: list[str] = []
        valid_entries: list[tuple[OdrEvidenceLedgerEntry, OdrSource]] = []
        invalid_cards: list[tuple[str, str]] = []
        for entry in entries:
            if entry.status != "covered":
                continue
            expected_ids.append(entry.requirement_id)
            source = (
                self._sources_by_id.get(entry.source_id)
                if entry.source_id is not None
                else None
            )
            reason: str | None = None
            if source is None or source.content is None:
                reason = "selected_source_not_fetched"
            elif not entry.candidate_key:
                reason = "missing_candidate_key"
            elif not entry.support_quote:
                reason = "missing_support_quote"
            elif entry.support_start is None or entry.support_end is None:
                reason = "missing_support_offsets"
            elif entry.source_start is None or entry.source_end is None:
                reason = "missing_excerpt_offsets"
            elif not (
                entry.source_start
                <= entry.support_start
                <= entry.support_end
                <= entry.source_end
            ):
                reason = "support_offsets_outside_selected_excerpt"
            elif source.content[entry.support_start : entry.support_end] != entry.support_quote:
                reason = "support_quote_not_in_source_snapshot"
            if reason is not None:
                invalid_cards.append((entry.requirement_id, reason))
                continue
            valid_entries.append((entry, source))
        return expected_ids, valid_entries, invalid_cards

    def _render_extractive_evidence_cards(
        self,
        entries: Iterable[OdrEvidenceLedgerEntry],
        *,
        unresolved_tasks: Iterable[str],
    ) -> tuple[str, OdrClaimSupportAudit]:
        """Render a non-narrative final artifact from validated ledger quotes.

        This is the current evidence-ledger writer boundary.  It intentionally
        does not ask the model to paraphrase a reviewed quote, because source
        selection and URL provenance alone cannot prove a new paraphrase did
        not add obligation, scope, or recommendation.  A future narrative
        writer must provide a separately validated claim contract before it
        can replace these cards.
        """

        expected_ids, valid_entries, invalid_cards = self._validated_extractive_entries(
            entries
        )
        rendered_ids: list[str] = []
        sections: list[str] = [
            "# Reviewed evidence cards",
            "",
            "This artifact deliberately renders only requirement-scoped, verbatim "
            "support quotes from immutable source snapshots. It is not a synthesized "
            "research answer or recommendation.",
        ]
        for entry, source in valid_entries:
            title = re.sub(r"[\[\]]", "", source.title).strip() or source.source_id
            quote = entry.support_quote.replace("\n", "\n> ")
            sections.extend(
                (
                    "",
                    f"## {entry.requirement_id}",
                    "",
                    f"**Requirement supplied to the workflow:** {entry.requirement}",
                    "",
                    "**Verbatim reviewed support quote:**",
                    f"> {quote}",
                    "",
                    f"**Provenance:** [{title}]({source.url}); "
                    f"candidate `{entry.candidate_key}`; snapshot characters "
                    f"{entry.support_start}:{entry.support_end}.",
                )
            )
            rendered_ids.append(entry.requirement_id)
        audit = OdrClaimSupportAudit(
            mode="extractive_evidence_cards/v1",
            expected_requirement_ids=tuple(expected_ids),
            rendered_requirement_ids=tuple(rendered_ids),
            invalid_cards=tuple(invalid_cards),
        )
        self._event(
            "workflow_claim_support_cards_rendered",
            mode=audit.mode,
            expected_requirement_ids=list(audit.expected_requirement_ids),
            rendered_requirement_ids=list(audit.rendered_requirement_ids),
            invalid_cards=[
                {"requirement_id": requirement_id, "reason": reason}
                for requirement_id, reason in audit.invalid_cards
            ],
            writer_bypassed=True,
        )
        return self._append_unresolved_tasks("\n".join(sections), unresolved_tasks), audit

    def _render_evidence_constrained_brief(
        self,
        entries: Iterable[OdrEvidenceLedgerEntry],
        *,
        plan: Iterable[EvidenceBriefSection],
        unresolved_tasks: Iterable[str],
        narrative: bool = False,
    ) -> tuple[str, OdrClaimSupportAudit]:
        """Render reviewed cards as an extractive or evidence-linked narrative brief."""

        expected_ids, valid_entries, invalid_cards = self._validated_extractive_entries(
            entries
        )
        entry_by_evidence_id = {
            entry.candidate_key: (entry, source)
            for entry, source in valid_entries
            if entry.candidate_key is not None
        }
        sections: list[str] = [
            (
                "# Evidence-linked narrative brief"
                if narrative
                else "# Evidence-constrained research brief"
            ),
            "",
            (
                "每段中文解释都绑定一张已审阅证据卡；其后的完整英文 candidate excerpt、来源与快照位置可用于回放核查。"
                if narrative
                else "This brief is compiled from reviewed evidence cards. Each section renders "
                "a verbatim quote selected by the evidence-ID-only plan."
            ),
        ]
        rendered_ids: list[str] = []
        for section in plan:
            selected = entry_by_evidence_id.get(section.evidence_id)
            if selected is None:
                invalid_cards.append(
                    (section.requirement_id, "brief_plan_evidence_not_renderable")
                )
                continue
            entry, source = selected
            if entry.requirement_id != section.requirement_id:
                invalid_cards.append(
                    (section.requirement_id, "brief_plan_requirement_mismatch")
                )
                continue
            if narrative and not section.explanation_zh:
                invalid_cards.append(
                    (section.requirement_id, "narrative_plan_missing_explanation")
                )
                continue
            title = re.sub(r"[\[\]]", "", source.title).strip() or source.source_id
            displayed_excerpt = (
                entry.excerpt if narrative else (entry.support_quote or "")
            ).replace("\n", "\n> ")
            displayed_start = entry.source_start if narrative else entry.support_start
            displayed_end = entry.source_end if narrative else entry.support_end
            sections.extend(
                (
                    "",
                    f"## {entry.requirement}",
                    "",
                    *(
                        (f"**中文解释：** {section.explanation_zh}", "")
                        if narrative
                        else ()
                    ),
                    (
                        "**已审 candidate excerpt（英文原文）：**"
                        if narrative
                        else "**Reviewed source statement:**"
                    ),
                    f"> {displayed_excerpt}",
                    "",
                    (
                        f"**证据锚点：** `{section.evidence_id}`；[{title}]({source.url})；"
                        f"snapshot characters {displayed_start}:{displayed_end}。"
                        if narrative
                        else f"**Evidence:** `{section.evidence_id}`; [{title}]({source.url}); "
                        f"snapshot characters {entry.support_start}:{entry.support_end}."
                    ),
                )
            )
            rendered_ids.append(entry.requirement_id)
        audit = OdrClaimSupportAudit(
            mode=(
                "evidence_linked_narrative_brief/v1"
                if narrative
                else "extractive_evidence_brief/v1"
            ),
            expected_requirement_ids=tuple(expected_ids),
            rendered_requirement_ids=tuple(rendered_ids),
            invalid_cards=tuple(invalid_cards),
        )
        self._event(
            (
                "workflow_evidence_linked_narrative_brief_rendered"
                if narrative
                else "workflow_evidence_constrained_brief_rendered"
            ),
            mode=audit.mode,
            expected_requirement_ids=list(audit.expected_requirement_ids),
            rendered_requirement_ids=list(audit.rendered_requirement_ids),
            invalid_cards=[
                {"requirement_id": requirement_id, "reason": reason}
                for requirement_id, reason in audit.invalid_cards
            ],
            **(
                {"evidence_anchor_binding": "one_to_one"}
                if narrative
                else {"free_text_claims_rendered": False}
            ),
        )
        return self._append_unresolved_tasks("\n".join(sections), unresolved_tasks), audit

    def _static_search_source_ids(
        self,
        *,
        query: str,
        task: str,
        task_budget: _ResearchTaskToolBudget | None,
        connector: SourceConnector,
        channel: str,
    ) -> list[str]:
        """Run one fixed search and recover its trace-recorded rank order."""

        with self._lock:
            first_sequence = len(self._trace) + 1
        self._search(
            query=query,
            task=task,
            task_budget=task_budget,
            connector=connector,
            channel=channel,
        )
        with self._lock:
            events = tuple(self._trace[first_sequence - 1 :])
        for event in reversed(events):
            if (
                event.kind == "search_completed"
                and event.data.get("task") == task
                and event.data.get("channel") == channel
            ):
                source_ids = event.data.get("result_source_ids")
                return [str(source_id) for source_id in source_ids] if isinstance(source_ids, list) else []
        return []

    def _run_static_one_pass_task(
        self,
        task: str,
        task_budget: _ResearchTaskToolBudget | None,
    ) -> str:
        """Execute fixed discovery and rank-order fetches without replanning."""

        channels: list[tuple[str, SourceConnector]] = [("web", self.connector)]
        if self.collection_connector is not None:
            channels.append(("collection", self.collection_connector))
        selected_source_ids: list[str] = []
        fetched_source_ids: list[str] = []
        trail_sections: list[str] = []
        for channel, connector in channels:
            ranked_source_ids = self._static_search_source_ids(
                query=task,
                task=task,
                task_budget=task_budget,
                connector=connector,
                channel=channel,
            )
            for source_id in ranked_source_ids[: self.policy.max_fetches_per_research_unit]:
                selected_source_ids.append(source_id)
                source_view = self._read_source(
                    source_id=source_id,
                    task=task,
                    task_budget=task_budget,
                )
                source = self._sources_by_id.get(source_id)
                if source is not None and source.content is not None:
                    fetched_source_ids.append(source_id)
                    trail_sections.append(source_view)
        self._event(
            "workflow_fixed_retrieval",
            task=task,
            channels=[channel for channel, _ in channels],
            selection_policy="connector_rank_order",
            selected_source_ids=selected_source_ids,
            fetched_source_ids=fetched_source_ids,
        )
        if not trail_sections:
            return "No source could be fetched by the fixed one-pass workflow."
        return self._compress_research_trail(
            task=task,
            working_trail="\n\n".join(trail_sections),
        )

    def _static_one_pass_research(self, tasks: list[str]) -> list[str]:
        """Execute every preplanned task once, with the P1 worker envelope."""

        if not tasks:
            self._research_state.status = "research_complete"
            return self._research_state.research_notes
        reports = self._run_fixed_retrieval_tasks(
            tasks,
            scheduling="static_one_pass_parallel",
        )
        for task, report in zip(tasks, reports):
            has_fetched_source = any(
                task in source.fetched_for and source.content is not None
                for source in self._sources_by_id.values()
            )
            if has_fetched_source:
                self._research_state.record_completed(task, report)
            else:
                self._research_state.record_unresolved([task])
                self._research_state.research_notes.append(report)
                self._event(
                    "research_task_unresolved",
                    task=task,
                    reason="no_fetched_source",
                )
        self._research_state.status = "research_complete"
        return self._research_state.research_notes

    def _run_fixed_retrieval_tasks(
        self,
        tasks: list[str],
        *,
        scheduling: str,
    ) -> list[str]:
        """Run a known task set once without delegating another planner turn."""

        if not tasks:
            return []
        task_budgets = self._parallel_task_tool_budgets(task_count=len(tasks))
        for task, task_budget in zip(tasks, task_budgets):
            self._record_task_tool_budget(
                task=task,
                task_budget=task_budget,
                scheduling=scheduling,
                pending_task_count=len(tasks),
            )
        max_workers = min(self.policy.max_concurrent_research_units, len(tasks))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="odr-static-workflow",
        ) as executor:
            return list(
                executor.map(
                    self._run_static_one_pass_task,
                    tasks,
                    task_budgets,
                )
            )

    def _task_fetched_source_ids(self, task: str) -> list[str]:
        """Return the immutable-source coverage that actually reached one task."""

        return [
            source.source_id
            for source in self._sources_by_id.values()
            if task in source.fetched_for and source.content is not None
        ]

    def _record_coverage_state(self, *, task: str, phase: str) -> bool:
        """Record source-backed task coverage without inferring claim correctness."""

        source_ids = self._task_fetched_source_ids(task)
        covered = bool(source_ids)
        self._event(
            "workflow_coverage_checked",
            task=task,
            phase=phase,
            status="source_backed" if covered else "no_fetched_source",
            fetched_source_ids=source_ids,
        )
        return covered

    def _plan_one_coverage_repair(
        self,
        *,
        initial_tasks: Iterable[str],
    ) -> tuple[str, list[str]] | None:
        """Ask once for a focused repair only after observed source coverage fails.

        This deliberately receives the runtime's source-coverage ledger rather
        than full researcher trails. It may recover a zero-source task, but it
        cannot restart a supervisor loop or claim that fetched text supports a
        particular statement.
        """

        from langchain_core.messages import HumanMessage, SystemMessage

        uncovered = [
            task for task in initial_tasks if not self._task_fetched_source_ids(task)
        ]
        if not uncovered:
            self._event(
                "workflow_coverage_repair_skipped",
                reason="all_initial_tasks_have_fetched_source",
            )
            return None
        if len(self._research_state.research_tasks) >= self.policy.breadth_budget:
            self._event(
                "workflow_coverage_repair_skipped",
                reason="breadth_budget_exhausted",
                uncovered_tasks=uncovered,
            )
            return None

        task_ids = {
            f"task-{index:03d}": task
            for index, task in enumerate(initial_tasks, 1)
        }
        uncovered_ids = {
            task_id for task_id, task in task_ids.items() if task in uncovered
        }
        ledger = "\n".join(
            (
                f"- {task_id}: {task}\n"
                f"  fetched source IDs: {', '.join(self._task_fetched_source_ids(task)) or 'none'}"
            )
            for task_id, task in task_ids.items()
        )
        prompt = """You are reviewing an evidence-coverage ledger for a research report.
The runtime completed one fixed retrieval pass. A task with no fetched source is an
explicit gap, not evidence that the task is impossible. You may propose exactly one
standalone repair task when it can materially recover an uncovered user requirement.
Merge related gaps if one focused task can address them; otherwise return null rather
than silently dropping requirements. Do not repeat a completed task, invent facts, or
write the report. This is the only repair opportunity in the workflow.
"""
        model = self.llm.bind_tools([CoverageRepairPlan], parallel_tool_calls=False)
        try:
            response = self._invoke(
                role="coverage_reviewer",
                model=model,
                messages=[
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            f"User request:\n{self.query}\n\n"
                            f"Initial source-coverage ledger:\n{ledger}"
                        )
                    ),
                ],
                tool_definitions=(CoverageRepairPlan,),
                tool_binding_options={"parallel_tool_calls": False},
                decision_stream_id="coverage_reviewer",
            )
        except ResearchBudgetExhausted:
            self._event(
                "workflow_coverage_repair_skipped",
                reason="research_model_budget_exhausted",
                uncovered_tasks=uncovered,
            )
            return None
        except Exception as exc:
            self._event(
                "workflow_coverage_repair_skipped",
                reason="coverage_reviewer_failure",
                error_type=type(exc).__name__,
                uncovered_tasks=uncovered,
            )
            return None

        for call in list(getattr(response, "tool_calls", None) or []):
            if not isinstance(call, Mapping) or str(call.get("name", "")) != "CoverageRepairPlan":
                continue
            candidate = str((call.get("args") or {}).get("research_topic") or "").strip()
            if candidate and candidate not in self._research_state.research_tasks:
                repaired_task_ids = [
                    str(task_id).strip()
                    for task_id in (call.get("args") or {}).get(
                        "repaired_initial_task_ids", []
                    )
                    if isinstance(task_id, str)
                    and str(task_id).strip() in uncovered_ids
                ]
                repaired_task_ids = list(dict.fromkeys(repaired_task_ids))
                if repaired_task_ids:
                    repaired_tasks = [task_ids[task_id] for task_id in repaired_task_ids]
                    self._event(
                        "workflow_coverage_repair_planned",
                        task=candidate,
                        repaired_initial_task_ids=repaired_task_ids,
                        repaired_initial_tasks=repaired_tasks,
                        uncovered_tasks=uncovered,
                    )
                    return candidate, repaired_tasks
        self._event(
            "workflow_coverage_repair_skipped",
            reason="coverage_reviewer_returned_no_mapped_repair",
            uncovered_tasks=uncovered,
        )
        return None

    def _write_report(
        self,
        *,
        tasks: Iterable[str],
        notes: Iterable[str],
        unresolved_tasks: Iterable[str],
        writer_source_ids: Iterable[str] | None = None,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        allowed_source_ids = (
            None if writer_source_ids is None else set(writer_source_ids)
        )
        sources = [
            source
            for source in self._sources_by_id.values()
            if source.content is not None
            and (allowed_source_ids is None or source.source_id in allowed_source_ids)
        ]
        unresolved = list(unresolved_tasks)
        if allowed_source_ids is not None and not sources:
            self._event(
                "writer_skipped_no_reviewed_evidence",
                unresolved_task_count=len(unresolved),
            )
            return self._append_unresolved_tasks(
                "# Research report\n\n"
                "No fetched source passed requirement-level evidence review, so the "
                "runtime did not synthesize an ungrounded answer.",
                unresolved,
            )
        if allowed_source_ids is not None and unresolved:
            self._event(
                "writer_skipped_unresolved_evidence_requirements",
                unresolved_task_count=len(unresolved),
            )
            return self._append_unresolved_tasks(
                "# Research report\n\n"
                "Requirement-level evidence review did not cover every requested claim, so "
                "the runtime did not synthesize a partial answer.",
                unresolved,
            )
        source_list = "\n".join(
            f"- [{source.source_id}] {source.title}: {source.url}" for source in sources
        )
        citation_instruction = (
            """Every material factual conclusion must have an inline source handle in this exact form:
`[source-###]`. Use only handles from the fetched-source allowlist, never a raw URL,
Markdown link, search-result candidate, or source title reconstructed from memory. The
runtime deterministically turns valid handles into reader-facing clickable citations
after writing; a handle is therefore the only citation format you should emit. If the
fetched material does not establish a requested alternative, limit, or capability,
say so instead of citing an unfetched source. Distinguish facts from recommendations.
Include a final `### Sources` list containing only the fetched source handles actually
used in the report.
"""
            if self.policy.source_handle_evidence_handoff or getattr(self, "_writer_source_handles", False)
            else """Every material factual conclusion must have an inline clickable Markdown citation
in this form: `[descriptive source title](source-link)`. The `source-###`
identifiers in the allowlist and research trail are internal handles, not reader-facing
citations: never emit `[source-###]` in the report. Do not cite a URL that is not in
the fetched allowlist. Distinguish facts from recommendations and explicitly say when
the retrieved sources do not establish a requested alternative, limit, or capability.
Include a final `### Sources` list containing only clickable Markdown links to fetched
URLs actually used.
"""
        )
        evidence_ledger_writer_boundary = (
            """You are writing only from requirement-level reviewed excerpts. Do not turn an
example, permitted method, option, or source-specific date into a universal requirement,
exclusive method, or recommendation. In particular, do not use words such as “must”, “only”,
“all”, or their equivalent unless the reviewed excerpt directly states that same obligation or
scope. If the excerpt establishes only a narrower fact, state only that narrower fact.

"""
            if allowed_source_ids is not None
            else ""
        )
        final_prompt = """Based on the research findings, write a comprehensive, well-structured
answer to the user's question in the same language as the question.  Use normal,
free-form Markdown: headings, tables, and prose are allowed whenever they improve
clarity.  Do not mention this orchestration process.

""" + citation_instruction + """
""" + evidence_ledger_writer_boundary + """
This is a report-writing step, not a JSON schema task.  Do not flatten related ideas
into one-claim-per-bullet output merely because the research had multiple tasks.
"""
        findings = "\n\n".join(notes)
        context = (
            f"User question:\n{self.query}\n\n"
            f"Research tasks:\n" + "\n".join(f"- {task}" for task in tasks)
            + f"\n\nFetched source allowlist:\n{source_list}\n\n"
            + f"Research findings:\n{findings}\n\n"
            + "Tasks the harness could not ground with reviewed evidence. Do not invent an "
            "answer for these, and do not claim that the search or fetched-source corpus lacks "
            "information: say only that the supplied reviewed evidence does not establish the "
            "answer, or omit the task. Do not list unresolved tasks yourself; the harness appends "
            "them verbatim after writing so this limitation cannot be silently omitted.\n"
            + ("\n".join(f"- {task}" for task in unresolved) or "(none)")
        )
        try:
            # Direct adaptation of RunBudget.open_report_allowance: reserve is
            # measured from actual research consumption, not a fraction guessed
            # before researchers run.
            self._run_budget.open_report_allowance()
            response = self._invoke(
                role="writer",
                model=self.llm,
                messages=[SystemMessage(content=final_prompt), HumanMessage(content=context)],
                report=True,
            )
            return self._append_unresolved_tasks(
                self._materialize_internal_citations(
                    _message_text(response),
                    allowed_source_ids=allowed_source_ids,
                ),
                unresolved,
            )
        except ResearchBudgetExhausted:
            return self._append_unresolved_tasks(
                "# Research report\n\nThe report allowance was exhausted before final synthesis.",
                unresolved,
            )
        except Exception:
            return self._append_unresolved_tasks(
                "# Research report\n\nFinal synthesis failed; inspect the retained research trace.",
                unresolved,
            )

    @staticmethod
    def _append_unresolved_tasks(report: str, unresolved_tasks: Iterable[str]) -> str:
        """Make budget-truncated work visible without relying on writer compliance.

        This is the same post-synthesis honesty rule as the upstream harness:
        the model is told about gaps for context, but the runtime is the source
        of truth for whether scheduled work actually ran.
        """

        tasks = [task for task in unresolved_tasks if isinstance(task, str) and task.strip()]
        if not tasks:
            return report
        heading = "## Unresolved research tasks"
        suffix = "\n\n".join(
            (
                heading,
                "\n".join(
                    f"- Not established by the workflow's evidence gate: {task}"
                    for task in tasks
                ),
            )
        )
        return report.rstrip() + "\n\n" + suffix + "\n"

    def _materialize_internal_citations(
        self,
        report: str,
        *,
        allowed_source_ids: set[str] | None = None,
    ) -> str:
        """Render model-visible source handles as reader-facing Markdown links.

        Focused research notes use stable source IDs so that a model can refer
        to retrieved evidence without reproducing a long URL.  Those handles
        must not leak into the final report as non-clickable pseudo-citations.
        This transformation is mechanical: it only substitutes an ID for the
        exact title and URL of a source fetched in this run; it neither adds a
        citation nor claims support that was not in the writer output.
        """

        def replacement(match: re.Match[str]) -> str:
            source_id = match.group(1)
            if allowed_source_ids is not None and source_id not in allowed_source_ids:
                return match.group(0)
            source = self._sources_by_id.get(source_id)
            if source is None or source.content is None:
                return match.group(0)
            title = re.sub(r"[\[\]]", "", source.title).strip() or source.source_id
            return f"[{title}]({source.url})"

        report = _SOURCE_ID_CITATION_RE.sub(replacement, report)
        # Models sometimes turn a relative library locator into a web host.
        # Repair only exact destinations of documents read in this run.
        for source in self._sources_by_id.values():
            if (
                source.content
                and source.channel == "collection"
                and source.url.startswith("/library/document/")
                and (allowed_source_ids is None or source.source_id in allowed_source_ids)
            ):
                for scheme in ("https://", "http://"):
                    report = report.replace(
                        f"]({scheme}{source.url.lstrip('/')})",
                        f"]({source.url})",
                    )
        return report

    def _audit_citations(
        self,
        report: str,
        *,
        allowed_source_ids: set[str] | None = None,
    ) -> OdrCitationAudit:
        cited = _citation_locators(report)
        known = {
            locator
            for source in self._sources_by_id.values()
            if source.content
            and (allowed_source_ids is None or source.source_id in allowed_source_ids)
            for locator in (source.url, *source.citation_aliases)
        }
        known_urls = tuple(url for url in cited if url in known)
        unknown_urls = tuple(url for url in cited if url not in known)
        unresolved_source_ids = tuple(
            sorted(
                {
                    match.group(1)
                    for match in _SOURCE_ID_CITATION_RE.finditer(report)
                    if (
                        (source := self._sources_by_id.get(match.group(1))) is None
                        or source.content is None
                        or (
                            allowed_source_ids is not None
                            and source.source_id not in allowed_source_ids
                        )
                    )
                }
            )
        )
        return OdrCitationAudit(
            cited_urls=cited,
            known_urls=known_urls,
            unknown_urls=unknown_urls,
            unresolved_source_ids=unresolved_source_ids,
        )

    def _finish_run(
        self,
        *,
        started: float,
        tasks: list[str],
        notes: list[str],
        execution_mode: str,
        evidence_ledger: Iterable[OdrEvidenceLedgerEntry] = (),
        writer_source_ids: Iterable[str] | None = None,
    ) -> OdrBaselineResult:
        ledger_entries = tuple(evidence_ledger)
        writer_source_ids = (
            None
            if writer_source_ids is None
            else tuple(dict.fromkeys(writer_source_ids))
        )
        claim_support_audit: OdrClaimSupportAudit | None = None
        brief_plan_fallback = False
        narrative_brief_fallback = False
        if execution_mode == "evidence_ledger_repair":
            brief_requested = (
                self.policy.evidence_brief_enabled
                or self.policy.evidence_narrative_brief_enabled
            )
            brief_plan = (
                self._plan_evidence_brief(ledger_entries)
                if brief_requested and not self._research_state.unresolved_tasks
                else None
            )
            if brief_plan is not None:
                report, claim_support_audit = self._render_evidence_constrained_brief(
                    ledger_entries,
                    plan=brief_plan,
                    unresolved_tasks=self._research_state.unresolved_tasks,
                    narrative=self.policy.evidence_narrative_brief_enabled,
                )
            elif self.policy.evidence_narrative_brief_enabled:
                narrative_brief_fallback = True
                report, claim_support_audit = self._render_evidence_constrained_brief(
                    ledger_entries,
                    plan=tuple(
                        EvidenceBriefSection(
                            requirement_id=entry.requirement_id,
                            evidence_id=entry.candidate_key,
                        )
                        for entry in ledger_entries
                        if entry.status == "covered" and entry.candidate_key is not None
                    ),
                    unresolved_tasks=self._research_state.unresolved_tasks,
                )
            else:
                brief_plan_fallback = self.policy.evidence_brief_enabled
                report, claim_support_audit = self._render_extractive_evidence_cards(
                    ledger_entries,
                    unresolved_tasks=self._research_state.unresolved_tasks,
                )
        else:
            report = self._write_report(
                tasks=tasks,
                notes=notes,
                unresolved_tasks=self._research_state.unresolved_tasks,
                writer_source_ids=writer_source_ids,
            )
        allowed_source_ids = (
            None if writer_source_ids is None else set(writer_source_ids)
        )
        audit = self._audit_citations(report, allowed_source_ids=allowed_source_ids)
        source_count = sum(1 for source in self._sources_by_id.values() if source.content)
        if not source_count:
            status, reason = "incomplete", "no_fetched_sources"
        elif audit.unknown_urls:
            status, reason = "needs_review", "report_cites_unknown_url"
        elif audit.unresolved_source_ids:
            status, reason = "needs_review", "report_references_unknown_source_handle"
        elif self._research_state.unresolved_tasks:
            status, reason = "incomplete", "unresolved_research_tasks"
        elif claim_support_audit is not None and not claim_support_audit.passed:
            status, reason = "needs_review", "claim_support_card_integrity_failed"
        elif not audit.known_urls:
            status, reason = "needs_review", "report_has_no_fetched_source_citation"
        else:
            if claim_support_audit is None:
                status, reason = "complete", "citation_url_audit_passed"
            elif claim_support_audit.mode == "evidence_linked_narrative_brief/v1":
                status, reason = "complete", "evidence_linked_narrative_brief_integrity_passed"
            elif narrative_brief_fallback:
                status, reason = "complete", "evidence_linked_narrative_brief_fallback_to_extractive_brief"
            elif claim_support_audit.mode == "extractive_evidence_brief/v1":
                status, reason = "complete", "evidence_constrained_brief_audit_passed"
            elif brief_plan_fallback:
                status, reason = "complete", "evidence_brief_fallback_to_cards"
            else:
                status, reason = "complete", "extractive_claim_card_audit_passed"
        self._research_state.status = status
        self._event(
            "run_finished",
            execution_mode=execution_mode,
            status=status,
            terminal_reason=reason,
            model_calls_used=self._model_calls_used,
            tool_calls_used=self._tool_calls_used,
            fetched_source_count=source_count,
        )
        return OdrBaselineResult(
            run_id=self.run_id,
            query=self.query,
            execution_mode=execution_mode,
            status=status,
            terminal_reason=reason,
            report_markdown=report,
            research_tasks=tuple(tasks),
            research_notes=tuple(notes),
            unresolved_tasks=tuple(self._research_state.unresolved_tasks),
            sources=tuple(self._sources_by_id.values()),
            trace=tuple(self._trace),
            model_calls_used=self._model_calls_used,
            tool_calls_used=self._tool_calls_used,
            citation_audit=audit,
            claim_support_audit=claim_support_audit,
            harness_state=self._research_state.to_dict(),
            budget=self._run_budget.to_dict(),
            evidence_ledger=ledger_entries,
            elapsed_seconds=round(monotonic() - started, 3),
        )

    def run(self) -> OdrBaselineResult:
        """Run the adaptive supervisor/researcher ODR path."""

        started = monotonic()
        self._event(
            "run_started",
            execution_mode="agentic",
            query=self.query,
            policy=asdict(self.policy),
            collection_context=self.collection_context,
            upstream=(
                "langchain-ai/open_deep_research (MIT adaptation); "
                "jmlon/deep-research-harness@393d907 (MIT harness adaptation)"
            ),
        )
        tasks, notes = self._supervisor_tasks()
        return self._finish_run(
            started=started,
            tasks=tasks,
            notes=notes,
            execution_mode="agentic",
        )

    def run_static_one_pass_workflow(self) -> OdrBaselineResult:
        """Run the fixed one-pass workflow counterfactual for ODR evaluation.

        This is evaluation-only: it performs one pre-retrieval plan and fixed
        rank-order source reads, then uses the normal compression, writer, H
        rendering, citation audit, and artifact contract. It never observes a
        source result to schedule another search or research task.
        """

        started = monotonic()
        self._event(
            "run_started",
            execution_mode="static_one_pass",
            query=self.query,
            policy=asdict(self.policy),
            collection_context=self.collection_context,
            upstream=(
                "langchain-ai/open_deep_research (MIT adaptation); "
                "jmlon/deep-research-harness@393d907 (MIT harness adaptation)"
            ),
        )
        tasks = self._static_one_pass_tasks()
        notes = self._static_one_pass_research(tasks)
        return self._finish_run(
            started=started,
            tasks=tasks,
            notes=notes,
            execution_mode="static_one_pass",
        )

    def run_coverage_repair_workflow(self) -> OdrBaselineResult:
        """Run fixed retrieval with at most one source-coverage-driven repair.

        Unlike the adaptive supervisor path, this workflow never feeds a full
        researcher report back into an open-ended planner loop. It reserves one
        breadth slot before retrieval, and uses it only when a planned task
        acquired no fetched source. The resulting ledger is source coverage,
        not a semantic claim-support verdict.
        """

        started = monotonic()
        self._event(
            "run_started",
            execution_mode="coverage_repair",
            query=self.query,
            policy=asdict(self.policy),
            collection_context=self.collection_context,
            upstream=(
                "langchain-ai/open_deep_research (MIT adaptation); "
                "jmlon/deep-research-harness@393d907 (MIT harness adaptation)"
            ),
        )
        initial_task_limit = max(1, self.policy.breadth_budget - 1)
        tasks = self._static_one_pass_tasks(task_limit=initial_task_limit)
        notes = self._research_state.research_notes
        initial_reports = self._run_fixed_retrieval_tasks(
            tasks,
            scheduling="coverage_repair_initial",
        )
        unresolved_initial_reports: dict[str, str] = {}
        for task, report in zip(tasks, initial_reports):
            if self._task_fetched_source_ids(task):
                self._research_state.record_completed(task, report)
            else:
                self._research_state.record_unresolved([task])
                unresolved_initial_reports[task] = report
                self._event(
                    "research_task_unresolved",
                    task=task,
                    reason="no_fetched_source",
                )
        for task in tasks:
            self._record_coverage_state(task=task, phase="initial")

        repair = self._plan_one_coverage_repair(initial_tasks=tasks)
        if repair is not None:
            repair_task, repaired_initial_tasks = repair
            self._research_state.research_tasks.append(repair_task)
            task_budget = self._serial_task_tool_budget(pending_task_count=1)
            self._record_task_tool_budget(
                task=repair_task,
                task_budget=task_budget,
                scheduling="coverage_repair",
                pending_task_count=1,
            )
            report = self._run_static_one_pass_task(repair_task, task_budget)
            if self._record_coverage_state(task=repair_task, phase="repair"):
                repair_source_ids = self._task_fetched_source_ids(repair_task)
                for task in repaired_initial_tasks:
                    for source_id in repair_source_ids:
                        source = self._sources_by_id[source_id]
                        if task not in source.fetched_for:
                            source.fetched_for.append(task)
                    if task not in self._research_state.completed_tasks:
                        self._research_state.completed_tasks.append(task)
                    self._research_state.unresolved_tasks = [
                        unresolved
                        for unresolved in self._research_state.unresolved_tasks
                        if unresolved != task
                    ]
                    unresolved_initial_reports.pop(task, None)
                notes.append(report)
                self._event(
                    "workflow_coverage_repair_applied",
                    task=repair_task,
                    repaired_initial_tasks=repaired_initial_tasks,
                    fetched_source_ids=repair_source_ids,
                )
            else:
                self._research_state.record_unresolved([repair_task])
                notes.append(report)
            tasks.append(repair_task)

        notes.extend(unresolved_initial_reports.values())

        self._research_state.status = "research_complete"
        return self._finish_run(
            started=started,
            tasks=tasks,
            notes=notes,
            execution_mode="coverage_repair",
        )

    def run_evidence_ledger_repair_workflow(
        self,
        *,
        frozen_requirements: Iterable[Mapping[str, object]] | None = None,
    ) -> OdrBaselineResult:
        """Run requirement-led fixed retrieval with one evidence-grounded repair.

        This is intentionally distinct from ``coverage_repair``: it retains a
        stable requirement-to-extract ledger, makes the writer consume only
        reviewed ledger entries, and lets a reviewer request one repair for
        unmet requirements. It still does not resume a supervisor loop.
        """

        if self.policy.evidence_excerpt_max_chars is None:
            raise ValueError(
                "evidence_ledger_repair requires explicit evidence_excerpt_max_chars"
            )
        started = monotonic()
        self._event(
            "run_started",
            execution_mode="evidence_ledger_repair",
            query=self.query,
            policy=asdict(self.policy),
            collection_context=self.collection_context,
            upstream=(
                "langchain-ai/open_deep_research (MIT adaptation); "
                "jmlon/deep-research-harness@393d907 (MIT harness adaptation)"
            ),
        )
        initial_requirement_limit = max(1, self.policy.breadth_budget - 1)
        if frozen_requirements is None:
            entries = self._plan_evidence_requirements(
                requirement_limit=initial_requirement_limit
            )
        else:
            # A supplied plan is already an explicit pre-retrieval experiment
            # input. Do not force unrelated atomic requirements back together
            # merely to reserve a repair slot. A full-breadth frozen plan runs
            # its fixed retrieval pass without repair instead.
            entries = self._frozen_evidence_requirements(
                frozen_requirements,
                requirement_limit=self.policy.breadth_budget,
            )
        candidate_source_ids_by_requirement = self._run_evidence_ledger_initial_fetches(
            entries
        )
        allow_repair = len(entries) < self.policy.breadth_budget
        if not allow_repair:
            self._event(
                "workflow_evidence_repair_skipped",
                phase="initial",
                reason="breadth_budget_reserved_by_frozen_plan",
                requirement_count=len(entries),
            )
        repair = self._review_evidence_ledger(
            entries,
            candidate_source_ids_by_requirement=candidate_source_ids_by_requirement,
            phase="initial",
            allow_repair=allow_repair,
            disallow_web_repair=frozen_requirements is not None,
        )
        if repair is not None:
            repair_query, repaired_entries, repair_channel = repair
            if repair_query not in self._research_state.research_tasks:
                self._research_state.research_tasks.append(repair_query)
            task_budget = self._serial_task_tool_budget(pending_task_count=1)
            self._record_task_tool_budget(
                task=repair_query,
                task_budget=task_budget,
                scheduling="evidence_ledger_repair",
                pending_task_count=1,
            )
            repair_source_ids = self._fetch_ranked_source_batch(
                query=repair_query,
                task=repair_query,
                task_budget=task_budget,
                source_channel=repair_channel,
            )
            for entry in repaired_entries:
                current_candidates = candidate_source_ids_by_requirement.get(
                    entry.requirement_id, []
                )
                candidate_source_ids_by_requirement[entry.requirement_id] = list(
                    dict.fromkeys([*current_candidates, *repair_source_ids])
                )
            self._event(
                "workflow_evidence_repair_fetched",
                repair_query=repair_query,
                repaired_requirement_ids=[entry.requirement_id for entry in repaired_entries],
                fetched_source_ids=repair_source_ids,
                candidate_source_ids_by_requirement={
                    entry.requirement_id: candidate_source_ids_by_requirement[
                        entry.requirement_id
                    ]
                    for entry in repaired_entries
                },
            )
            self._review_evidence_ledger(
                repaired_entries,
                candidate_source_ids_by_requirement={
                    entry.requirement_id: candidate_source_ids_by_requirement[
                        entry.requirement_id
                    ]
                    for entry in repaired_entries
                },
                phase="post_repair",
                allow_repair=False,
            )

        notes = self._ledger_notes(entries)
        for entry in entries:
            if entry.status == "covered":
                self._research_state.record_completed(
                    entry.requirement,
                    f"Reviewed evidence ledger entry {entry.requirement_id}.",
                )
            else:
                self._research_state.record_unresolved([entry.requirement])
                self._event(
                    "research_task_unresolved",
                    task=entry.requirement,
                    reason=f"evidence_ledger:{entry.status}",
                )
        self._research_state.status = "research_complete"
        writer_source_ids = [
            entry.source_id
            for entry in entries
            if entry.status == "covered" and entry.source_id is not None
        ]
        return self._finish_run(
            started=started,
            tasks=[entry.requirement for entry in entries],
            notes=notes,
            execution_mode="evidence_ledger_repair",
            evidence_ledger=entries,
            writer_source_ids=writer_source_ids,
        )

    @staticmethod
    def write_artifacts(result: OdrBaselineResult, *, artifact_root: str | Path) -> OdrBaselineArtifacts:
        if not isinstance(result, OdrBaselineResult):
            raise TypeError("result must be OdrBaselineResult")
        root = Path(artifact_root).expanduser()
        if not root.is_absolute():
            raise ValueError("artifact_root must be an absolute path")
        run_dir = root.resolve() / result.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        report_path = run_dir / "report.md"
        trace_path = run_dir / "trace.jsonl"
        run_path = run_dir / "run.json"
        sources_path = run_dir / "sources.json"
        notes_path = run_dir / "research_notes.md"
        state_path = run_dir / "harness_state.json"
        evidence_ledger_path = run_dir / "evidence_ledger.json"
        report_path.write_text(result.report_markdown + "\n", encoding="utf-8", newline="\n")
        trace_path.write_text(
            "".join(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for event in result.trace),
            encoding="utf-8",
            newline="\n",
        )
        sources_path.write_text(
            json.dumps([source.public_view() for source in result.sources], ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        notes_path.write_text(
            "\n\n".join(
                f"## Research task {index}\n\n{note}"
                for index, note in enumerate(result.research_notes, start=1)
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        state_path.write_text(
            json.dumps(result.harness_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        evidence_ledger_path.write_text(
            json.dumps(
                [entry.public_view() for entry in result.evidence_ledger],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        snapshots = run_dir / "source_snapshots"
        snapshots.mkdir()
        for source in result.sources:
            if source.content is not None:
                (snapshots / f"{source.source_id}.txt").write_text(
                    source.content, encoding="utf-8", newline="\n"
                )
        run_path.write_text(
            json.dumps(
                {
                    "schema_version": "odr-baseline-run/v1",
                    "upstream": "langchain-ai/open_deep_research",
                    "upstream_license": "MIT",
                    "run_id": result.run_id,
                    "query": result.query,
                    "execution_mode": result.execution_mode,
                    "status": result.status,
                    "terminal_reason": result.terminal_reason,
                    "publishable": result.is_publishable,
                    "research_tasks": list(result.research_tasks),
                    "unresolved_tasks": list(result.unresolved_tasks),
                    "model_calls_used": result.model_calls_used,
                    "tool_calls_used": result.tool_calls_used,
                    "budget": result.budget,
                    "elapsed_seconds": result.elapsed_seconds,
                    "citation_audit": result.citation_audit.to_dict(),
                    "claim_support_audit": (
                        result.claim_support_audit.to_dict()
                        if result.claim_support_audit is not None
                        else None
                    ),
                    "evidence_ledger_path": evidence_ledger_path.name,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return OdrBaselineArtifacts(
            run_dir=run_dir,
            report_path=report_path,
            trace_path=trace_path,
            run_path=run_path,
            sources_path=sources_path,
            notes_path=notes_path,
            state_path=state_path,
            evidence_ledger_path=evidence_ledger_path,
        )


__all__ = [
    "CoverageRepairPlan",
    "ConductResearch",
    "EvidenceCoverageSelection",
    "EvidenceCoverageReview",
    "EvidenceRequirement",
    "EvidenceRequirementPlan",
    "OdrClaimSupportAudit",
    "OdrBaselineArtifacts",
    "OdrEvidenceLedgerEntry",
    "OdrBaselinePolicy",
    "OdrBaselineResult",
    "OdrBaselineRunner",
    "ResearchComplete",
]
