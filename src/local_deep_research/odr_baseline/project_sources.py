"""Project adapters for the two Hybrid-ODR source channels.

The adapters call the existing project search/fetch primitives.  They do not
perform source scoring, query routing, evidence extraction, or answer
validation; those decisions stay with the ODR researcher and writer.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import os
import re
import time
from typing import Any, Generator, Iterable, Mapping
from urllib.parse import urlsplit

from .collection_query_translation import CollectionQueryTranslator
from .sources import DiscoveredResource, FetchedResource


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _result_text(value: object, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


_SAFE_FETCH_FAILURE_CODES = frozenset(
    {
        "collection_connector_unavailable",
        "collection_document_empty",
        "collection_document_unavailable",
        "invalid_fetched_resource",
        "public_fetch_egress_denied",
        "public_fetch_empty",
        "public_fetch_url_denied",
        "public_fetch_mirror_failed",
        "public_fetch_source_family_denied",
    }
)

_PUBLIC_FETCH_FALLBACKS = frozenset({"disabled", "jina"})
_COLLECTION_RAW_CANDIDATE_MULTIPLIER = 8
_COLLECTION_RERANK_ENV = "LDR_COLLECTION_CROSS_ENCODER_RERANK"
_COLLECTION_RERANK_CACHE_ENV = "LDR_COLLECTION_CROSS_ENCODER_CACHE"
_COLLECTION_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_COLLECTION_RERANK_MAX_LENGTH = 512
_COLLECTION_RERANK_BATCH_SIZE = 16
_CHINESE_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def safe_fetch_failure_code(exc: Exception) -> str:
    """Return a trace-safe, stable reason without retaining exception text."""

    candidate = str(exc).strip()
    if candidate in _SAFE_FETCH_FAILURE_CODES:
        return candidate
    return "source_fetch_failed"


def _is_public_only_context(egress_context: Any) -> bool:
    scope = getattr(egress_context, "scope", None)
    return getattr(scope, "value", scope) == "public_only"


def _normalized_host_suffixes(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized = tuple(
        dict.fromkeys(
            value.strip().lower().lstrip(".")
            for value in values
            if isinstance(value, str) and value.strip()
        )
    )
    if any("." not in value or "/" in value for value in normalized):
        raise ValueError("allowed source host suffixes must be hostnames")
    return normalized


def _matches_host_suffix(url: str, allowed_suffixes: tuple[str, ...]) -> bool:
    if not allowed_suffixes:
        return True
    host = (urlsplit(url).hostname or "").lower()
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_suffixes)


def _distinct_collection_document_results(
    raw_results: object, *, max_documents: int | None
) -> tuple[Mapping[str, Any], ...]:
    """Keep the first chunk hit for each Document in vector rank order."""

    if not isinstance(raw_results, (list, tuple)):
        return ()
    distinct: list[Mapping[str, Any]] = []
    seen_document_ids: set[str] = set()
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            continue
        locator = raw.get("link") or raw.get("url")
        if not isinstance(locator, str) or not locator.startswith(
            "/library/document/"
        ):
            continue
        document_id = locator.removeprefix("/library/document/").split("/", 1)[
            0
        ]
        if not document_id or document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        distinct.append(raw)
        if max_documents is not None and len(distinct) == max_documents:
            break
    return tuple(distinct)


def _collection_rerank_enabled() -> bool:
    return os.environ.get(_COLLECTION_RERANK_ENV, "").lower() in {
        "1", "true", "yes", "on"
    }


@lru_cache(maxsize=1)
def _load_collection_cross_encoder():
    """Load the explicit local-only reranker once per process."""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = (
        "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    cache_dir = os.environ.get(_COLLECTION_RERANK_CACHE_ENV) or None
    tokenizer = AutoTokenizer.from_pretrained(
        _COLLECTION_RERANK_MODEL, cache_dir=cache_dir, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        _COLLECTION_RERANK_MODEL, cache_dir=cache_dir, local_files_only=True
    )
    model.to(device)
    model.eval()
    return tokenizer, model, torch, device


class ProjectPublicWebConnector:
    """Serper discovery plus direct extraction and an explicit public fallback."""

    def __init__(
        self,
        *,
        settings_snapshot: Mapping[str, Any],
        username: str | None = None,
        egress_context: Any = None,
        search_engine_name: str = "serper",
        fetch_timeout_seconds: int = 30,
        public_fetch_fallback: str = "disabled",
        allowed_source_host_suffixes: Iterable[str] | None = None,
    ) -> None:
        if public_fetch_fallback not in _PUBLIC_FETCH_FALLBACKS:
            raise ValueError(
                "unknown public fetch fallback; expected 'disabled' or 'jina'"
            )
        if public_fetch_fallback == "jina" and not _is_public_only_context(
            egress_context
        ):
            raise ValueError(
                "public fetch fallback 'jina' requires a public_only egress context"
            )
        self._settings_snapshot = dict(settings_snapshot)
        self._username = username
        self._egress_context = egress_context
        self._search_engine_name = search_engine_name
        self._fetch_timeout_seconds = fetch_timeout_seconds
        self._public_fetch_fallback = public_fetch_fallback
        self._allowed_source_host_suffixes = _normalized_host_suffixes(
            allowed_source_host_suffixes
        )

    def search(self, query: str) -> Iterable[DiscoveredResource]:
        from local_deep_research.web_search_engines.search_engine_factory import (
            create_search_engine,
        )

        engine = create_search_engine(
            engine_name=self._search_engine_name,
            llm=None,
            username=self._username,
            settings_snapshot=self._settings_snapshot,
            programmatic_mode=True,
        )
        if engine is None:
            raise RuntimeError("search_engine_unavailable")
        try:
            raw_results = engine.run(query)
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        if not isinstance(raw_results, list):
            return ()
        return tuple(
            DiscoveredResource(
                resource_locator=url,
                title=_result_text(raw.get("title"), url),
                snippet=_result_text(
                    raw.get("snippet") or raw.get("content") or raw.get("description"),
                    "No search snippet returned.",
                ),
                channel="web",
            )
            for raw in raw_results
            if isinstance(raw, Mapping)
            and isinstance((url := raw.get("link") or raw.get("url")), str)
            and url.strip()
            and _matches_host_suffix(url, self._allowed_source_host_suffixes)
        )

    def fetch(self, resource: DiscoveredResource) -> FetchedResource:
        from local_deep_research.research_library.downloaders.extraction import (
            batch_fetch_and_extract,
        )
        from local_deep_research.security.ssrf_validator import validate_url
        from local_deep_research.utilities.js_rendering import (
            read_js_rendering_setting,
        )

        if not _matches_host_suffix(
            resource.resource_locator, self._allowed_source_host_suffixes
        ):
            raise RuntimeError("public_fetch_source_family_denied")
        if not validate_url(resource.resource_locator):
            raise RuntimeError("public_fetch_url_denied")
        if self._egress_context is not None:
            from local_deep_research.security.egress.policy import evaluate_url

            if not evaluate_url(resource.resource_locator, self._egress_context).allowed:
                raise RuntimeError("public_fetch_egress_denied")
        result = batch_fetch_and_extract(
            [resource.resource_locator],
            timeout=self._fetch_timeout_seconds,
            enable_js_rendering=read_js_rendering_setting(self._settings_snapshot),
        )
        content = result.get(resource.resource_locator) if isinstance(result, Mapping) else None
        retrieval_method = "direct_html"
        if (
            (not isinstance(content, str) or not content.strip())
            and self._public_fetch_fallback == "jina"
        ):
            from local_deep_research.advanced_search_system.tools.fetch import (
                fetch_via_public_mirror,
            )

            mirrored = fetch_via_public_mirror(
                resource.resource_locator,
                egress_context=self._egress_context,
            )
            mirror_content = mirrored.get("content")
            if (
                mirrored.get("status") == "success"
                and isinstance(mirror_content, str)
                and mirror_content.strip()
            ):
                content = mirror_content
                retrieval_method = "jina_public_mirror"
            else:
                raise RuntimeError("public_fetch_mirror_failed")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("public_fetch_empty")
        return FetchedResource(
            content=content,
            retrieved_at=_timestamp(),
            title=resource.title,
            retrieval_method=retrieval_method,
        )


class ProjectCollectionConnector:
    """One user-selected Collection through its existing RAG index."""

    def __init__(
        self,
        *,
        collection_id: str,
        collection_name: str,
        username: str,
        user_password: str | None = None,
        settings_snapshot: Mapping[str, Any],
        max_results: int = 8,
        query_translator: CollectionQueryTranslator | None = None,
    ) -> None:
        self._collection_id = collection_id
        self._collection_name = collection_name
        self._username = username
        self._user_password = user_password
        self._settings_snapshot = {
            **dict(settings_snapshot),
            "_username": username,
            "search.tool": f"collection_{collection_id}",
            "policy.egress_scope": "adaptive",
        }
        self._max_results = max_results
        self._query_translator = (
            query_translator or CollectionQueryTranslator.from_environment()
        )
        self._last_query_translation_metadata: dict[str, Any] | None = None
        self._last_rerank_metadata: dict[str, Any] | None = None

    @property
    def last_query_translation_metadata(self) -> dict[str, Any] | None:
        """Compact metadata from the most recent Collection search translation."""

        return (
            dict(self._last_query_translation_metadata)
            if self._last_query_translation_metadata is not None
            else None
        )

    @property
    def last_rerank_metadata(self) -> dict[str, Any] | None:
        """Compact metadata from the most recent optional Collection rerank."""

        return (
            dict(self._last_rerank_metadata)
            if self._last_rerank_metadata is not None
            else None
        )

    def _rerank_collection_documents(
        self,
        *,
        query: str,
        original_query: str,
        translation_status: str,
        candidates: tuple[Mapping[str, Any], ...],
    ) -> tuple[Mapping[str, Any], ...]:
        if not _collection_rerank_enabled():
            self._last_rerank_metadata = {"enabled": False, "applied": False}
            return candidates
        if _CHINESE_TEXT.search(original_query) and translation_status != "translated":
            self._last_rerank_metadata = {
                "enabled": True,
                "applied": False,
                "reason": "untranslated_chinese_query",
                "model": _COLLECTION_RERANK_MODEL,
            }
            return candidates

        from local_deep_research.advanced_search_system.tools.fetch.library_resolver import (
            resolve_library_document,
        )

        fetch_started = time.perf_counter()
        scored: list[tuple[int, Mapping[str, Any], str]] = []
        unavailable: list[tuple[int, Mapping[str, Any]]] = []
        for position, candidate in enumerate(candidates):
            locator = candidate.get("link") or candidate.get("url")
            if not isinstance(locator, str):
                unavailable.append((position, candidate))
                continue
            document = resolve_library_document(locator, self._username)
            content = document.get("content") if isinstance(document, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                unavailable.append((position, candidate))
                continue
            title = _result_text(document.get("title") if isinstance(document, Mapping) else None, _result_text(candidate.get("title"), "Collection document"))
            scored.append((position, candidate, f"{title}\n\n{content}"))

        load_started = time.perf_counter()
        tokenizer, model, torch, device = _load_collection_cross_encoder()
        model_load_seconds = time.perf_counter() - load_started
        truncated = sum(
            len(tokenizer(query, passage, add_special_tokens=True, truncation=False)["input_ids"])
            > _COLLECTION_RERANK_MAX_LENGTH
            for _, _, passage in scored
        )
        scoring_started = time.perf_counter()
        scores: list[float] = []
        for start in range(0, len(scored), _COLLECTION_RERANK_BATCH_SIZE):
            batch = scored[start : start + _COLLECTION_RERANK_BATCH_SIZE]
            features = tokenizer(
                [query] * len(batch),
                [passage for _, _, passage in batch],
                padding=True,
                truncation="only_second",
                max_length=_COLLECTION_RERANK_MAX_LENGTH,
                return_tensors="pt",
            )
            features = {name: value.to(device) for name, value in features.items()}
            with torch.inference_mode():
                scores.extend(
                    float(value)
                    for value in model(**features).logits.reshape(-1).detach().float().cpu().tolist()
                )
        scoring_seconds = time.perf_counter() - scoring_started
        ranked = sorted(
            zip(scored, scores), key=lambda item: (-item[1], item[0][0])
        )
        self._last_rerank_metadata = {
            "enabled": True,
            "applied": True,
            "model": _COLLECTION_RERANK_MODEL,
            "device": device,
            "candidate_documents": len(candidates),
            "scored_documents": len(scored),
            "unavailable_documents": len(unavailable),
            "pair_truncation": "only_second",
            "max_length": _COLLECTION_RERANK_MAX_LENGTH,
            "truncated_pairs": truncated,
            "model_load_seconds": model_load_seconds,
            "fetch_and_input_prepare_seconds": scoring_started - fetch_started - model_load_seconds,
            "scoring_seconds": scoring_seconds,
            "extra_seconds": time.perf_counter() - fetch_started,
        }
        return tuple(
            [candidate for (_, candidate, _), _ in ranked]
            + [candidate for _, candidate in unavailable]
        )

    @contextmanager
    def _collection_runtime(self) -> Generator[None, None, None]:
        """Give one local retrieval its normal DB and egress context.

        The ODR P1 scheduler uses standard worker threads.  Establishing this
        context at the Collection boundary keeps the selected user's encrypted
        library available in those workers without changing ODR scheduling.
        """

        from local_deep_research.security.egress.audit_hook import (
            active_egress_context,
        )
        from local_deep_research.security.egress.policy import (
            context_from_snapshot,
        )
        from local_deep_research.utilities.thread_context import (
            clear_search_context,
            get_search_context,
            set_search_context,
        )

        previous_context = get_search_context()
        context = dict(previous_context or {})
        context["username"] = self._username
        if self._user_password:
            context["user_password"] = self._user_password
        set_search_context(context)
        try:
            egress_context = context_from_snapshot(
                self._settings_snapshot,
                f"collection_{self._collection_id}",
                username=self._username,
            )
            with active_egress_context(egress_context):
                yield
        finally:
            if previous_context is None:
                clear_search_context()
            else:
                set_search_context(previous_context)

    def search(self, query: str) -> Iterable[DiscoveredResource]:
        from local_deep_research.web_search_engines.engines.search_engine_collection import (
            CollectionSearchEngine,
        )

        translation = self._query_translator.translate(query)
        self._last_query_translation_metadata = translation.metadata()
        with self._collection_runtime():
            engine = CollectionSearchEngine(
                collection_id=self._collection_id,
                collection_name=self._collection_name,
                llm=None,
                max_results=self._max_results,
                settings_snapshot=self._settings_snapshot,
            )
            try:
                # The RAG index ranks chunks, but this adapter hands an Agent
                # Documents to decide whether to read.  Retrieve a bounded
                # chunk pool, then retain the earliest hit for each Document;
                # the public contract remains max_results Documents.
                raw_results = engine.search(
                    translation.query,
                    limit=self._max_results
                    * _COLLECTION_RAW_CANDIDATE_MULTIPLIER,
                )
            finally:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
            reranked_results = self._rerank_collection_documents(
                query=translation.query,
                original_query=query,
                translation_status=translation.status,
                candidates=_distinct_collection_document_results(
                    raw_results, max_documents=None
                ),
            )
        return tuple(
            DiscoveredResource(
                resource_locator=url,
                title=_result_text(raw.get("title"), "Collection document"),
                snippet=_result_text(raw.get("snippet") or raw.get("content"), "No snippet returned."),
                channel="collection",
            )
            for raw in _distinct_collection_document_results(
                reranked_results,
                max_documents=self._max_results,
            )
            if isinstance((url := raw.get("link") or raw.get("url")), str)
            and url.startswith("/library/document/")
        )

    def fetch(self, resource: DiscoveredResource) -> FetchedResource:
        from local_deep_research.advanced_search_system.tools.fetch.library_resolver import (
            resolve_library_document,
        )

        with self._collection_runtime():
            document = resolve_library_document(
                resource.resource_locator, self._username
            )
        if document is None:
            raise RuntimeError("collection_document_unavailable")
        content = document.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("collection_document_empty")
        original_url = document.get("original_url")
        citation_aliases = (
            (original_url.strip(),)
            if isinstance(original_url, str)
            else ()
        )
        return FetchedResource(
            content=content,
            retrieved_at=_timestamp(),
            title=_result_text(document.get("title"), resource.title),
            citation_aliases=citation_aliases,
            retrieval_method="collection_document",
        )


__all__ = [
    "ProjectCollectionConnector",
    "ProjectPublicWebConnector",
    "safe_fetch_failure_code",
]
