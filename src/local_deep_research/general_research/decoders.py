"""Fail-closed structured-output decoders for General Research model stages.

Models may propose a plan, extract evidence, and draft report claims, but they
do not control run identity, fetched snapshot hashes, or audit timestamps.
Those values are attached by the runtime after strict JSON decoding.  No
markdown fence stripping or prose salvage is performed: a malformed stage is
an incomplete run, not an invitation to treat model prose as evidence.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from .actions import FetchAction, GeneralAction, RequestStopAction, SearchAction
from .chunks import EvidenceChunk, EvidenceSpan
from .config import ResearchControlPolicy
from .schemas import (
    EvidenceCard,
    EvidenceStance,
    GeneralRunConfig,
    PlanItem,
    ResearchBrief,
    ResearchRequirement,
    ResearchPlan,
    ResearchTask,
    SourceRecord,
    SupervisorDecision,
    validate_supervisor_decision,
)
from .semantic_audit import (
    SemanticAuditResult,
    SemanticClaimReview,
    SemanticVerdict,
)
from .writer import ReportClaim


class StructuredOutputError(ValueError):
    """A model response failed the General V1 structured-output contract."""


def _as_object(value: str | Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
                raise StructuredOutputError(
                    f"{stage} must return exactly one JSON object"
                ) from exc
    if not isinstance(value, Mapping):
        raise StructuredOutputError(f"{stage} must return one JSON object")
    return dict(value)


def _strict_object(
    value: object,
    *,
    stage: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuredOutputError(f"{stage} item must be an object")
    result = dict(value)
    unknown = set(result).difference(allowed)
    missing = required.difference(result)
    if unknown:
        raise StructuredOutputError(f"{stage} item has unsupported fields: {sorted(unknown)}")
    if missing:
        raise StructuredOutputError(f"{stage} item is missing fields: {sorted(missing)}")
    return result


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredOutputError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise StructuredOutputError(f"{field_name} must be a JSON list")
    result = tuple(_text(item, field_name=field_name) for item in value)
    if not result:
        raise StructuredOutputError(f"{field_name} must not be empty")
    if len(set(result)) != len(result):
        raise StructuredOutputError(f"{field_name} must not contain duplicates")
    return result


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StructuredOutputError(f"{field_name} must be a non-negative integer")
    return value


def _plan_id(config: GeneralRunConfig) -> str:
    digest = sha256(f"{config.run_id}\n{config.query}".encode("utf-8")).hexdigest()
    return f"plan-{digest[:20]}"


def _brief_id(config: GeneralRunConfig) -> str:
    digest = sha256(f"{config.run_id}\n{config.query}\nbrief".encode("utf-8")).hexdigest()
    return f"brief-{digest[:20]}"


def _optional_string_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise StructuredOutputError(f"{field_name} must be a JSON list")
    items = tuple(_text(item, field_name=field_name) for item in value)
    if len(items) != len(set(items)):
        raise StructuredOutputError(f"{field_name} must not contain duplicates")
    return items


def decode_research_brief(
    value: str | Mapping[str, Any], *, config: GeneralRunConfig
) -> ResearchBrief:
    """Decode intent and scope while the runtime owns run identity.

    The brief intentionally excludes facts, sources, and planning constraints.
    It gives the later planner a sharper target without creating a hidden
    second answer channel before research begins.
    """

    if not isinstance(config, GeneralRunConfig):
        raise TypeError("config must be GeneralRunConfig")
    payload = _as_object(value, stage="research brief")
    item = _strict_object(
        payload,
        stage="research brief",
        allowed=frozenset(
            {"objective", "scope", "deliverable", "requirements", "assumptions"}
        ),
        required=frozenset({"objective", "scope", "deliverable", "requirements"}),
    )
    try:
        raw_requirements = _optional_string_list(
            item.get("requirements"), field_name="brief requirements"
        )
        if not raw_requirements:
            raise StructuredOutputError("brief requirements must not be empty")
        return ResearchBrief(
            brief_id=_brief_id(config),
            run_id=config.run_id,
            user_query=config.query,
            objective=_text(item["objective"], field_name="brief objective"),
            scope=_text(item["scope"], field_name="brief scope"),
            deliverable=_text(item["deliverable"], field_name="brief deliverable"),
            requirements=tuple(
                ResearchRequirement(
                    requirement_id=f"requirement-{index}", text=requirement
                )
                for index, requirement in enumerate(raw_requirements, start=1)
            ),
            assumptions=_optional_string_list(
                item.get("assumptions"), field_name="brief assumptions"
            ),
            created_at=config.created_at,
        )
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError(f"research brief is invalid: {exc}") from exc


def decode_research_plan(
    value: str | Mapping[str, Any],
    *,
    config: GeneralRunConfig,
    control_policy: ResearchControlPolicy | None = None,
    brief: ResearchBrief | None = None,
) -> ResearchPlan:
    """Decode questions while deriving run identity and acceptance gates locally.

    A planner may decompose the user's request but must not mark an obligation
    optional or lower its source/evidence requirements.  The runtime attaches
    every completion condition from the frozen ``ResearchControlPolicy``.
    """

    if not isinstance(config, GeneralRunConfig):
        raise TypeError("config must be GeneralRunConfig")
    if control_policy is None:
        control_policy = ResearchControlPolicy()
    if not isinstance(control_policy, ResearchControlPolicy):
        raise TypeError("control_policy must be ResearchControlPolicy")
    if brief is not None:
        if not isinstance(brief, ResearchBrief):
            raise TypeError("brief must be ResearchBrief or None")
        if brief.run_id != config.run_id or brief.user_query != config.query:
            raise ValueError("brief must belong to the active run and query")
    payload = _as_object(value, stage="planner")
    if set(payload) != {"items"}:
        raise StructuredOutputError("planner output must contain exactly: items")
    raw_items = payload["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise StructuredOutputError("planner items must be a non-empty JSON list")
    if len(raw_items) > control_policy.max_plan_items:
        raise StructuredOutputError(
            "planner emitted more items than the frozen research control policy permits"
        )
    items: list[PlanItem] = []
    for raw_item in raw_items:
        item = _strict_object(
            raw_item,
            stage="planner",
            allowed=frozenset({"item_id", "question", "requirement_ids"}),
            required=(
                frozenset({"item_id", "question", "requirement_ids"})
                if brief is not None
                else frozenset({"item_id", "question"})
            ),
        )
        try:
            requirement_ids = _optional_string_list(
                item.get("requirement_ids"), field_name="planner requirement_ids"
            )
            if brief is not None:
                known_requirement_ids = {
                    requirement.requirement_id for requirement in brief.requirements
                }
                if not requirement_ids:
                    raise StructuredOutputError(
                        "agentic planner must link each item to brief requirements"
                    )
                unknown_requirement_ids = set(requirement_ids).difference(
                    known_requirement_ids
                )
                if unknown_requirement_ids:
                    raise StructuredOutputError(
                        "planner names unknown brief requirements: "
                        + ", ".join(sorted(unknown_requirement_ids))
                    )
            items.append(
                PlanItem(
                    item_id=_text(item["item_id"], field_name="planner item_id"),
                    question=_text(item["question"], field_name="planner question"),
                    requirement_ids=requirement_ids,
                    required=True,
                    min_evidence_cards=control_policy.min_evidence_cards_per_item,
                    min_distinct_source_groups=(
                        control_policy.min_distinct_source_groups_per_item
                    ),
                    min_quality_score=control_policy.min_source_quality_score,
                )
            )
        except (TypeError, ValueError) as exc:
            raise StructuredOutputError(f"planner item is invalid: {exc}") from exc
    try:
        return ResearchPlan(
            plan_id=_plan_id(config),
            run_id=config.run_id,
            query=config.query,
            created_at=config.created_at,
            items=tuple(items),
        )
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError(f"planner output is invalid: {exc}") from exc


def decode_supervisor_decision(
    value: str | Mapping[str, Any],
    *,
    plan: ResearchPlan,
    round_index: int,
    created_at: str,
    max_tasks: int,
) -> SupervisorDecision:
    """Decode one bounded dispatch without letting the supervisor alter policy."""

    if not isinstance(plan, ResearchPlan):
        raise TypeError("plan must be ResearchPlan")
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer")
    if isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or max_tasks < 1:
        raise ValueError("max_tasks must be a positive integer")
    payload = _as_object(value, stage="supervisor")
    decision = payload.get("decision")
    if decision == "finish":
        item = _strict_object(
            payload,
            stage="supervisor finish",
            allowed=frozenset({"decision", "reason"}),
            required=frozenset({"decision", "reason"}),
        )
        try:
            result = SupervisorDecision(
                run_id=plan.run_id,
                plan_id=plan.plan_id,
                round_index=round_index,
                should_finish=True,
                reason=_text(item["reason"], field_name="supervisor finish reason"),
                created_at=created_at,
            )
            validate_supervisor_decision(result, plan)
            return result
        except (TypeError, ValueError) as exc:
            raise StructuredOutputError(f"supervisor finish is invalid: {exc}") from exc
    if decision != "dispatch":
        raise StructuredOutputError("supervisor decision must be dispatch or finish")
    item = _strict_object(
        payload,
        stage="supervisor dispatch",
        allowed=frozenset({"decision", "tasks", "reason"}),
        required=frozenset({"decision", "tasks", "reason"}),
    )
    raw_tasks = item["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise StructuredOutputError("supervisor tasks must be a non-empty JSON list")
    if len(raw_tasks) > max_tasks:
        raise StructuredOutputError("supervisor emitted more tasks than this round permits")
    tasks: list[ResearchTask] = []
    for raw_task in raw_tasks:
        task = _strict_object(
            raw_task,
            stage="supervisor task",
            allowed=frozenset(
                {"task_id", "plan_item_ids", "question", "research_focus"}
            ),
            required=frozenset(
                {"task_id", "plan_item_ids", "question", "research_focus"}
            ),
        )
        try:
            tasks.append(
                ResearchTask(
                    task_id=_text(task["task_id"], field_name="supervisor task_id"),
                    plan_item_ids=_string_list(
                        task["plan_item_ids"], field_name="supervisor plan_item_ids"
                    ),
                    question=_text(task["question"], field_name="supervisor question"),
                    research_focus=_text(
                        task["research_focus"], field_name="supervisor research_focus"
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise StructuredOutputError(f"supervisor task is invalid: {exc}") from exc
    try:
        result = SupervisorDecision(
            run_id=plan.run_id,
            plan_id=plan.plan_id,
            round_index=round_index,
            tasks=tuple(tasks),
            reason=_text(item["reason"], field_name="supervisor dispatch reason"),
            created_at=created_at,
        )
        validate_supervisor_decision(result, plan)
        return result
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError(f"supervisor dispatch is invalid: {exc}") from exc


def decode_controller_action(value: str | Mapping[str, Any]) -> GeneralAction:
    """Decode exactly one controller action without URL or tool-name escape hatches."""

    payload = _as_object(value, stage="controller")
    action_name = payload.get("action")
    if action_name == "search":
        item = _strict_object(
            payload,
            stage="controller search",
            allowed=frozenset({"action", "connector_id", "query"}),
            required=frozenset({"action", "connector_id", "query"}),
        )
        try:
            return SearchAction(
                connector_id=_text(item["connector_id"], field_name="controller connector_id"),
                query=_text(item["query"], field_name="controller query"),
            )
        except ValueError as exc:
            raise StructuredOutputError(f"controller search is invalid: {exc}") from exc
    if action_name == "fetch":
        item = _strict_object(
            payload,
            stage="controller fetch",
            allowed=frozenset({"action", "candidate_id"}),
            required=frozenset({"action", "candidate_id"}),
        )
        try:
            return FetchAction(
                candidate_id=_text(item["candidate_id"], field_name="controller candidate_id")
            )
        except ValueError as exc:
            raise StructuredOutputError(f"controller fetch is invalid: {exc}") from exc
    if action_name == "request_stop":
        item = _strict_object(
            payload,
            stage="controller stop",
            allowed=frozenset({"action", "reason"}),
            required=frozenset({"action", "reason"}),
        )
        try:
            return RequestStopAction(
                reason=_text(item["reason"], field_name="controller stop reason")
            )
        except ValueError as exc:
            raise StructuredOutputError(f"controller stop is invalid: {exc}") from exc
    raise StructuredOutputError(
        "controller action must be one of: search, fetch, request_stop"
    )


def _evidence_id(
    source_id: str,
    plan_item_ids: tuple[str, ...],
    quote_start: int,
    quote_end: int,
) -> str:
    material = "\n".join((source_id, *plan_item_ids, str(quote_start), str(quote_end)))
    return "e-" + sha256(material.encode("utf-8")).hexdigest()[:20]


def _verified_quote_offsets(*, chunk: EvidenceChunk, quote: str) -> tuple[int, int]:
    """Locate one exact model quote in its immutable chunk deterministically.

    Models choose the semantic span; the runtime, not the model, establishes
    character offsets.  Repeated text is deliberately rejected rather than
    silently choosing a different occurrence.
    """

    relative_start = chunk.content.find(quote)
    if relative_start < 0:
        raise StructuredOutputError(
            "evidence verbatim_quote is absent from the selected source chunk"
        )
    if chunk.content.find(quote, relative_start + 1) >= 0:
        raise StructuredOutputError(
            "evidence verbatim_quote is ambiguous within the selected source chunk"
        )
    quote_start = chunk.start + relative_start
    return quote_start, quote_start + len(quote)


def decode_evidence_cards(
    value: str | Mapping[str, Any],
    *,
    plan: ResearchPlan,
    sources_by_id: Mapping[str, SourceRecord],
    chunks_by_id: Mapping[str, EvidenceChunk],
    observed_at: str,
    chunk_ids_by_ref: Mapping[str, str] | None = None,
    spans_by_id: Mapping[str, EvidenceSpan] | None = None,
    span_ids_by_ref: Mapping[str, str] | None = None,
    plan_item_ids_by_ref: Mapping[str, str] | None = None,
    discard_invalid_cards: bool = False,
    rejection_codes: list[str] | None = None,
) -> tuple[EvidenceCard, ...]:
    """Decode lightweight evidence proposals into verified evidence cards.

    The default legacy boundary accepts a short chunk reference and an exact
    quotation.  The runtime path supplies pre-cut spans instead: the model
    selects ``span_ref`` and the runtime derives the quote, source identity,
    snapshot binding, locator, and offsets.  That removes brittle long-string
    copying while keeping all published evidence tied to immutable fetched text.

    At the runtime boundary, one malformed proposal may be discarded without
    admitting it as evidence.  If none survive, the whole extraction still
    fails and may use the bounded retry path.  Direct callers remain strict by
    default to make schema regressions visible in tests.
    """

    if not isinstance(plan, ResearchPlan):
        raise TypeError("plan must be ResearchPlan")
    payload = _as_object(value, stage="evidence extractor")
    if set(payload) != {"cards"}:
        raise StructuredOutputError("evidence output must contain exactly: cards")
    raw_cards = payload["cards"]
    if not isinstance(raw_cards, list):
        raise StructuredOutputError("evidence cards must be a JSON list")
    known_plan_item_ids = {item.item_id for item in plan.items}
    resolved_chunk_ids_by_ref = dict(
        chunk_ids_by_ref
        if chunk_ids_by_ref is not None
        else {chunk_id: chunk_id for chunk_id in chunks_by_id}
    )
    resolved_spans_by_id = dict(spans_by_id or {})
    resolved_span_ids_by_ref = dict(
        span_ids_by_ref
        if span_ids_by_ref is not None
        else {span_id: span_id for span_id in resolved_spans_by_id}
    )
    resolved_plan_item_ids_by_ref = dict(
        plan_item_ids_by_ref
        if plan_item_ids_by_ref is not None
        else {item_id: item_id for item_id in known_plan_item_ids}
    )
    if not all(
        isinstance(ref, str) and ref.strip()
        and isinstance(chunk_id, str) and chunk_id in chunks_by_id
        for ref, chunk_id in resolved_chunk_ids_by_ref.items()
    ):
        raise ValueError("chunk_ids_by_ref must point to supplied chunks")
    if not all(
        isinstance(ref, str) and ref.strip()
        and isinstance(span_id, str) and span_id in resolved_spans_by_id
        for ref, span_id in resolved_span_ids_by_ref.items()
    ):
        raise ValueError("span_ids_by_ref must point to supplied spans")
    for span in resolved_spans_by_id.values():
        if span.evidence_chunk_id not in chunks_by_id:
            raise ValueError("spans_by_id must point to supplied chunks")
    if not all(
        isinstance(ref, str) and ref.strip()
        and isinstance(item_id, str) and item_id in known_plan_item_ids
        for ref, item_id in resolved_plan_item_ids_by_ref.items()
    ):
        raise ValueError("plan_item_ids_by_ref must point to the supplied plan")
    cards: list[EvidenceCard] = []
    known_ids: set[str] = set()
    for raw_card in raw_cards:
        try:
            uses_spans = bool(resolved_spans_by_id)
            card = _strict_object(
                raw_card,
                stage="evidence",
                allowed=(
                    frozenset({"span_ref", "plan_item_ref", "claim"})
                    if uses_spans
                    else frozenset(
                        {"chunk_ref", "plan_item_ref", "claim", "verbatim_quote"}
                    )
                ),
                required=(
                    frozenset({"span_ref", "plan_item_ref", "claim"})
                    if uses_spans
                    else frozenset(
                        {"chunk_ref", "plan_item_ref", "claim", "verbatim_quote"}
                    )
                ),
            )
            span: EvidenceSpan | None = None
            if uses_spans:
                span_ref = _text(card["span_ref"], field_name="evidence span_ref")
                span_id = resolved_span_ids_by_ref.get(span_ref)
                span = resolved_spans_by_id.get(span_id) if span_id is not None else None
                if span is None:
                    raise StructuredOutputError("evidence span_ref is not available")
                chunk_id = span.evidence_chunk_id
            else:
                chunk_ref = _text(card["chunk_ref"], field_name="evidence chunk_ref")
                chunk_id = resolved_chunk_ids_by_ref.get(chunk_ref)
            chunk = chunks_by_id.get(chunk_id) if chunk_id is not None else None
            if chunk is None:
                raise StructuredOutputError("evidence chunk_ref is not available")
            if span is not None and (
                span.source_id != chunk.source_id
                or span.source_content_hash != chunk.source_content_hash
                or span.start < chunk.start
                or span.end > chunk.end
                or chunk.content[span.start - chunk.start : span.end - chunk.start]
                != span.content
            ):
                raise StructuredOutputError("evidence span does not match selected chunk")
            source_id = chunk.source_id
            source = sources_by_id.get(source_id)
            if source is None:
                raise StructuredOutputError("evidence chunk source was not fetched")
            if not source.content_verified or not source.content_hash:
                raise StructuredOutputError("evidence source is not a verified full fetch")
            if chunk.source_content_hash != source.content_hash:
                raise StructuredOutputError("evidence chunk does not match source snapshot")
            plan_ref = _text(
                card["plan_item_ref"], field_name="evidence plan_item_ref"
            )
            try:
                claimed_plan_items = (resolved_plan_item_ids_by_ref[plan_ref],)
            except KeyError as exc:
                raise StructuredOutputError(
                    "evidence plan_item_ref is not available"
                ) from exc
            if span is not None:
                # EvidenceCard normalizes surrounding whitespace. Derive the
                # same normalized exact substring and its adjusted offsets
                # here, rather than retaining span-edge whitespace with stale
                # offsets that the runtime would rightly reject later.
                quote = _text(
                    span.content, field_name="runtime evidence span content"
                )
                leading_whitespace = len(span.content) - len(span.content.lstrip())
                quote_start = span.start + leading_whitespace
                quote_end = quote_start + len(quote)
            else:
                quote = _text(
                    card["verbatim_quote"], field_name="evidence verbatim_quote"
                )
                quote_start, quote_end = _verified_quote_offsets(chunk=chunk, quote=quote)
            evidence_id = _evidence_id(
                source_id, claimed_plan_items, quote_start, quote_end
            )
            if evidence_id in known_ids:
                raise StructuredOutputError("evidence extractor returned duplicate evidence")
            cards.append(
                EvidenceCard(
                    evidence_id=evidence_id,
                    source_id=source_id,
                    plan_item_ids=claimed_plan_items,
                    claim=_text(card["claim"], field_name="evidence claim"),
                    verbatim_quote=quote,
                    locator=f"{chunk.chunk_id}, characters {quote_start}-{quote_end}",
                    source_content_hash=source.content_hash,
                    evidence_chunk_id=chunk.chunk_id,
                    quote_start=quote_start,
                    quote_end=quote_end,
                    stance=EvidenceStance.SUPPORTS,
                    observed_at=observed_at,
                )
            )
            known_ids.add(evidence_id)
        except (TypeError, ValueError, StructuredOutputError) as exc:
            if discard_invalid_cards:
                if rejection_codes is not None:
                    message = str(exc)
                    if "span_ref" in message:
                        rejection_codes.append("unavailable_span_ref")
                    elif "chunk_ref" in message:
                        rejection_codes.append("unavailable_chunk_ref")
                    elif "plan_item_ref" in message:
                        rejection_codes.append("unavailable_plan_item_ref")
                    elif "quote" in message:
                        rejection_codes.append("quote_verification_failed")
                    elif "duplicate" in message:
                        rejection_codes.append("duplicate_evidence")
                    elif "source" in message or "snapshot" in message:
                        rejection_codes.append("source_verification_failed")
                    else:
                        rejection_codes.append("invalid_card_schema")
                continue
            if isinstance(exc, StructuredOutputError):
                raise
            raise StructuredOutputError(f"evidence card is invalid: {exc}") from exc
    if not cards:
        raise StructuredOutputError("evidence extractor produced no valid cards")
    return tuple(cards)


def decode_writer_claims(
    value: str | Mapping[str, Any],
    *,
    evidence_ids_by_ref: Mapping[str, str],
    plan_item_id_by_evidence_id: Mapping[str, str],
) -> tuple[ReportClaim, ...]:
    """Decode writer claims without accepting a parallel prose report.

    The writer uses short evidence aliases only. The runtime maps those aliases
    to frozen cards and derives the sole plan item from each card, preventing a
    writer from accidentally binding a valid citation to the wrong obligation.
    Presentation markdown is deliberately absent from this schema, so no second
    factual output channel can bypass citation or semantic audit.
    """

    payload = _as_object(value, stage="writer")
    if set(payload) != {"claims"}:
        raise StructuredOutputError("writer output must contain exactly: claims")
    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        raise StructuredOutputError("writer claims must be a JSON list")
    resolved_evidence_ids_by_ref = dict(evidence_ids_by_ref)
    resolved_plan_item_id_by_evidence_id = dict(plan_item_id_by_evidence_id)
    if not all(
        isinstance(ref, str) and ref.strip()
        and isinstance(evidence_id, str)
        and evidence_id in resolved_plan_item_id_by_evidence_id
        for ref, evidence_id in resolved_evidence_ids_by_ref.items()
    ):
        raise ValueError(
            "evidence_ids_by_ref must map short references to known evidence"
        )
    claims: list[ReportClaim] = []
    known_ids: set[str] = set()
    for index, raw_claim in enumerate(raw_claims, start=1):
        claim = _strict_object(
            raw_claim,
            stage="writer",
            allowed=frozenset({"text", "evidence_refs"}),
            required=frozenset({"text", "evidence_refs"}),
        )
        claim_id = f"claim-{index}"
        if claim_id in known_ids:
            raise StructuredOutputError("writer returned duplicate claim identity")
        known_ids.add(claim_id)
        evidence_refs = _string_list(
            claim["evidence_refs"], field_name="writer evidence_refs"
        )
        try:
            evidence_ids = tuple(
                resolved_evidence_ids_by_ref[ref] for ref in evidence_refs
            )
        except KeyError as exc:
            raise StructuredOutputError(
                "writer evidence_ref is not available"
            ) from exc
        plan_item_ids = {
            resolved_plan_item_id_by_evidence_id[evidence_id]
            for evidence_id in evidence_ids
        }
        if len(plan_item_ids) != 1:
            raise StructuredOutputError(
                "writer claim evidence must map to exactly one plan item"
            )
        claims.append(
            ReportClaim(
                claim_id=claim_id,
                text=_text(claim["text"], field_name="writer claim text"),
                plan_item_ids=tuple(plan_item_ids),
                evidence_ids=evidence_ids,
            )
        )
    return tuple(claims)


def decode_semantic_audit(
    value: str | Mapping[str, Any], *, writer_input_fingerprint: str
) -> SemanticAuditResult:
    """Decode bounded semantic reviews while deriving the context binding locally."""

    payload = _as_object(value, stage="semantic auditor")
    if set(payload) != {"reviews"}:
        raise StructuredOutputError("semantic audit output must contain exactly: reviews")
    raw_reviews = payload["reviews"]
    if not isinstance(raw_reviews, list) or not raw_reviews:
        raise StructuredOutputError("semantic audit reviews must be a non-empty JSON list")
    reviews: list[SemanticClaimReview] = []
    for raw_review in raw_reviews:
        review = _strict_object(
            raw_review,
            stage="semantic audit",
            allowed=frozenset({"claim_id", "verdict", "reason_code"}),
            required=frozenset({"claim_id", "verdict", "reason_code"}),
        )
        try:
            reviews.append(
                SemanticClaimReview(
                    claim_id=_text(
                        review["claim_id"], field_name="semantic audit claim_id"
                    ),
                    verdict=SemanticVerdict(
                        _text(review["verdict"], field_name="semantic audit verdict")
                    ),
                    reason_code=_text(
                        review["reason_code"], field_name="semantic audit reason_code"
                    ),
                )
            )
        except ValueError as exc:
            raise StructuredOutputError("semantic audit review is invalid") from exc
    try:
        return SemanticAuditResult(
            writer_input_fingerprint=_text(
                writer_input_fingerprint, field_name="writer_input_fingerprint"
            ),
            reviews=tuple(reviews),
        )
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError("semantic audit output is invalid") from exc


__all__ = [
    "StructuredOutputError",
    "decode_controller_action",
    "decode_evidence_cards",
    "decode_research_plan",
    "decode_semantic_audit",
    "decode_writer_claims",
]
