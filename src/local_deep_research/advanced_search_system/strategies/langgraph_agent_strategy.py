"""
LangGraph agent-based research strategy with parallel subagent support.

Uses LangChain's create_agent() to build a tool-calling agent that autonomously
decides what to search, when to dig deeper, and when to synthesize. Complex
questions can be decomposed into subtopics researched in parallel by subagents.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.errors import GraphRecursionError
from loguru import logger

from ...agent_harness.evidence_policy import (
    EvidencePolicyDecision,
    EvidencePolicyGuard,
    EvidencePolicyMode,
    EvidencePolicyVerdict,
)
from ...agent_harness.planner_budget import (
    DEFAULT_MAX_RUNTIME_RECOVERY_ACTIONS,
    FORCED_STOP_FLAG,
    PLANNER_STATE_ARM_EXPANDED,
    PLANNER_STATE_ARM_LEGACY_COMPACT,
    PLANNER_BUDGET_PROTOCOL,
    PlannerBudgetMiddleware,
)
from ...agent_harness.uncertainty_state import (
    PLANNER_UNCERTAINTY_STATE_PROTOCOL,
    UncertaintyStateTracker,
)
from ...agent_harness.routing import (
    AUTO_ROUTING_MODE,
    ROUTING_MODES as PRODUCT_ROUTING_MODES,
    choose_routing_mode,
)
from ...citation_handler import CitationHandler
from ...security.egress import EngineClassification, classify_engine
from ...security import sanitize_error_for_client
from ...utilities.thread_context import get_search_context, search_context
from ...database.thread_local_session import thread_cleanup
from ..tools.fetch import (
    FETCH_MODES,
    PUBLIC_FETCH_FALLBACKS,
    build_fetch_tool,
    make_library_resolver,
)
from .base_strategy import (
    BaseSearchStrategy,
    CHECK_CONTEXT_AGENT_STREAM,
    CHECK_CONTEXT_ENTRY,
    CHECK_CONTEXT_FALLBACK_SYNTHESIS,
)
from .primary_search_metadata import (
    NEUTRAL_PRIMARY_SEARCH_DESCRIPTION,
    PrimarySourceType,
    classify_primary_source,
    format_primary_search_description,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS = (
    50  # agent needs many more cycles than pipeline strategies
)
MIN_ITERATIONS = 10  # below this the agent can barely do anything useful
SUBAGENT_TIMEOUT_SECONDS = 1800  # 30 minutes per subagent, measured from
# each subagent's *actual* start time (not from the drain-loop start, which
# used to make queued subagents inherit the wall-clock of everything that
# ran before them -- see #5014).
# User-facing sentinel for "the agent produced nothing". _finalize must
# not run the accumulated-sources citation pass over it — that would
# dress the failure up as a cited synthesis instead of surfacing it.
NO_RESULTS_MESSAGE = (
    "Research could not produce results. Try a different query."
)
MAX_SUBTOPICS = 5  # must match the "pass 2-5" contract in the lead prompt
# and the research_subtopic tool docstring. Values above this are dropped
# at the call site, with the dropped subtopics named in the tool's reply
# so the lead agent can re-issue or avoid citing them.
MAX_SUBAGENT_WORKERS = 4  # default pool size when the user has not set
# ``langgraph_agent.max_subagent_workers``. Surplus subtopics queue and
# start as workers free up; each queued subagent keeps its own per-task
# budget from its actual start time.
SUBAGENT_TIMEOUT_OVERALL_MULTIPLIER = 2  # safety cap multiplier: the overall
# drain-loop wall-clock is bounded to ``SUBAGENT_TIMEOUT_SECONDS * multiplier``
# seconds so a pathological hang cannot block the lead forever. This is a
# backstop only -- the per-task deadline above is what actually kills an
# individual subagent. The multiplier is intentionally small (2) because the
# worst case (queued subtopics) is already covered by per-task deadlines; we
# only need the overall cap to bound genuinely deadlocked / hung tasks.
# CONTENT_FETCH_TIMEOUT and CONTENT_MAX_LENGTH live alongside the fetch
# tool builders in advanced_search_system/tools/fetch/.

# Cap for credential-scrubbed tool/agent error strings. Larger than the
# 200-char HTTP-client default of ``sanitize_error_for_client`` because these
# strings feed the agent's reasoning AND the ErrorReporter pattern map, where
# over-aggressive truncation drops the categorizable error signal. Credential
# scrubbing still runs first on the full untruncated string (#4633).
_TOOL_ERROR_MAX_LEN = 500

# ``auto`` turns the validated bounded two-source contract into the static
# route. Explicit modes remain available for experiments and power users.
ROUTING_MODES = frozenset(PRODUCT_ROUTING_MODES)

_COLLECTION_NORMALIZER_SYSTEM = (
    "You normalize one Chinese research request before it is sent to an English "
    "local medical Collection. Return only valid JSON with exactly one key: "
    "collection_query. Its value must be a short English retrieval query naming "
    "the most specific canonical medical topic. Do not answer the health question, "
    "add medical advice, citations, Markdown, or extra keys."
)


def _scrub_tool_error(message: str) -> str:
    """Scrub credentials from an LLM/agent-facing tool error string."""
    return sanitize_error_for_client(message, max_length=_TOOL_ERROR_MAX_LEN)


def _normalize_collection_query(
    query: str, *, enabled: bool = True
) -> tuple[str, str | None]:
    """Optionally normalize a Collection query through an isolated SFT endpoint.

    The feature is opt-in through environment variables so existing Collections
    and all non-medical runs preserve their prior behavior.  A malformed reply,
    endpoint error, or empty normalized query is a soft failure: the native RAG
    engine receives the original query.  This is intentionally a query rewrite
    hook, not a medical answer generator.
    """
    if not enabled:
        return query, None
    endpoint = os.getenv("LDR_COLLECTION_QUERY_NORMALIZER_ENDPOINT", "").strip()
    model = os.getenv("LDR_COLLECTION_QUERY_NORMALIZER_MODEL", "").strip()
    if not endpoint or not model:
        return query, None
    try:
        timeout_seconds = float(
            os.getenv("LDR_COLLECTION_QUERY_NORMALIZER_TIMEOUT_SECONDS", "20")
        )
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        import requests

        response = requests.post(
            endpoint.rstrip("/") + "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": _COLLECTION_NORMALIZER_SYSTEM,
                    },
                    {"role": "user", "content": query},
                ],
                "temperature": 0,
                "max_tokens": 96,
                "stream": False,
            },
            timeout=timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}")
        payload = response.json()
        content = str(
            ((payload.get("choices") or [{}])[0].get("message") or {}).get(
                "content", ""
            )
        )
        for candidate in re.findall(r"\{[^{}]*\}", content, flags=re.S):
            try:
                normalized = json.loads(candidate).get("collection_query")
            except json.JSONDecodeError:
                continue
            if not isinstance(normalized, str):
                continue
            normalized = " ".join(normalized.split())
            if normalized and normalized != query:
                return normalized[:256], None
        raise ValueError("response did not contain collection_query JSON")
    except Exception as exc:
        logger.warning(
            "Collection query normalizer fallback: {}", type(exc).__name__
        )
        return query, type(exc).__name__


# ---------------------------------------------------------------------------
# Thread-safe search result collector
# ---------------------------------------------------------------------------


class SearchResultsCollector:
    """Accumulates search results from the lead agent and subagents.

    Thread-safe: multiple subagent threads may call ``add_results``
    concurrently.  The ``_all_links`` reference points to the strategy's
    shared ``all_links_of_system`` list and is never reassigned.
    """

    def __init__(self, all_links: list | None = None) -> None:
        self._results: list[dict] = []
        self._sources: list[str] = []
        self._lock = threading.Lock()
        self._all_links = all_links if all_links is not None else []

    # -- public API ----------------------------------------------------------

    def add_results(
        self,
        results: list[dict],
        engine_name: str = "web",
    ) -> int:
        """Index *results* and append to the internal list **and** the shared
        ``all_links_of_system``.  Returns the starting citation index
        (0-based) assigned to the first result in this batch.

        The entire operation runs under a single lock acquisition so that
        citation indices are never duplicated.
        """
        if not results:
            return len(self._all_links)

        with self._lock:
            # Use global offset (all_links) not per-call offset (results)
            # so that indices are unique across sections in detailed reports.
            start_idx = len(self._all_links)
            for i, raw in enumerate(results):
                if not isinstance(raw, dict):
                    continue
                r = dict(raw)  # shallow copy to avoid mutating engine output
                r["index"] = str(start_idx + i + 1)
                r["source_engine"] = engine_name
                # Normalise URL key — citation handler expects "link"
                if "link" not in r and "url" in r:
                    r["link"] = r["url"]
                self._results.append(r)
                link = r.get("link", "")
                if link:
                    self._sources.append(link)
                self._all_links.append(r)
            return start_idx

    def add_results_deduplicated(
        self,
        results: list[dict],
        engine_name: str = "web",
    ) -> list[tuple[dict, int]]:
        """Add a search batch while reusing citation ids for repeated URLs.

        Agent queries are often semantically similar without being byte-for-byte
        identical, so the per-tool query cache cannot stop the same document
        from being returned again.  Reusing the first citation id keeps the
        agent's evidence set and the final citation-rewrite prompt bounded while
        still returning the newest snippet to the agent.

        Deduplication is scoped to the current ``analyze_topic`` call.  A
        collector reset starts a fresh set so detailed-report sections retain
        their existing globally monotonic citation behavior.
        """
        if not results:
            return []

        indexed: list[tuple[dict, int]] = []
        with self._lock:
            current_by_link = {
                self._link_key(item): item
                for item in self._results
                if self._link_key(item)
            }
            for raw in results:
                if not isinstance(raw, dict):
                    continue
                result = dict(raw)
                if "link" not in result and "url" in result:
                    result["link"] = result["url"]
                link_key = self._link_key(result)
                existing = current_by_link.get(link_key) if link_key else None
                if existing is not None:
                    citation_idx = int(existing["index"])
                    self._merge_result_evidence(existing, result)
                    indexed.append((result, citation_idx))
                    continue

                citation_idx = len(self._all_links) + 1
                result["index"] = str(citation_idx)
                result["source_engine"] = engine_name
                self._results.append(result)
                link = result.get("link", "")
                if link:
                    self._sources.append(link)
                    current_by_link[link_key] = result
                self._all_links.append(result)
                indexed.append((result, citation_idx))
        return indexed

    @staticmethod
    def _link_key(result: dict) -> str:
        link = str(result.get("link", result.get("url", ""))).strip()
        return link.rstrip("/")

    @staticmethod
    def _merge_result_evidence(
        existing: dict,
        incoming: dict,
        max_chars: int = 6000,
    ) -> None:
        """Keep distinct snippets from repeat hits without unbounded growth."""
        for field in ("snippet", "body", "content"):
            new_text = incoming.get(field)
            if not isinstance(new_text, str) or not new_text.strip():
                continue
            old_text = existing.get(field)
            if not isinstance(old_text, str) or not old_text.strip():
                existing[field] = new_text[:max_chars]
            elif new_text not in old_text:
                existing[field] = (
                    f"{old_text}\n\nAdditional matching passage:\n{new_text}"
                )[:max_chars]
            return

    def find_by_url(self, url: str) -> int | None:
        """Return the 1-based citation index if *url* is already tracked, else ``None``."""
        target_key = self._link_key({"link": url})
        if not target_key:
            return None
        with self._lock:
            for r in self._all_links:
                if self._link_key(r) == target_key:
                    idx = r.get("index")
                    if idx is not None:
                        return int(idx)
                    return None
            return None

    def record_fetched_content(
        self,
        *,
        url: str,
        title: str,
        content: str,
        retrieval_method: str | None = None,
    ) -> int:
        """Store successful page content and mark its citation as fetched.

        Search normally discovers a URL before ``fetch_content`` reads it.
        Keeping that original record used to discard the fetched text and
        leave the evidence writer seeing only a search snippet. Upgrade the
        citation in place so its body and provenance reflect the fetch.
        """
        target_key = self._link_key({"link": url})
        if not target_key:
            raise ValueError("Fetched URL must not be empty")

        snippet = content[:200].strip()
        if len(content) > 200:
            snippet += "..."

        with self._lock:
            existing = next(
                (
                    item
                    for item in self._all_links
                    if self._link_key(item) == target_key
                ),
                None,
            )
            if existing is not None:
                existing.setdefault(
                    "discovered_via", existing.get("source_engine")
                )
                existing["source_engine"] = "fetch"
                existing["fetch_status"] = "success"
                existing["full_content"] = content
                existing["snippet"] = snippet
                if title:
                    existing["title"] = title
                if retrieval_method:
                    existing["retrieval_method"] = retrieval_method
                if existing not in self._results:
                    self._results.append(existing)
                if url not in self._sources:
                    self._sources.append(url)
                return int(existing["index"])

            citation_idx = len(self._all_links) + 1
            fetched = {
                "index": str(citation_idx),
                "title": title,
                "link": url,
                "snippet": snippet,
                "full_content": content,
                "source_engine": "fetch",
                "fetch_status": "success",
            }
            if retrieval_method:
                fetched["retrieval_method"] = retrieval_method
            self._results.append(fetched)
            self._all_links.append(fetched)
            self._sources.append(url)
            return citation_idx

    def find_by_index(self, idx: int) -> dict | None:
        """Return the result dict for a 1-based citation index, or ``None``.

        Reverse of ``find_by_url``: given ``[N]`` (the citation marker the
        LLM sees in the search-results block), look up the source it
        references so the fetch tool can resolve a confused "fetch [1062]"
        call to the real URL. Thread-safe via the collector lock; uses
        ``_all_links`` (the shared, monotonic list) so a citation registered
        by the lead agent is also resolvable by a pooled subagent.
        """
        if not isinstance(idx, int) or idx < 1:
            return None
        with self._lock:
            for r in self._all_links:
                stored = r.get("index")
                if stored is not None and int(stored) == idx:
                    return r
        return None

    def reset(self) -> None:
        """Clear per-call state.  ``_all_links`` is intentionally kept."""
        with self._lock:
            self._results.clear()
            self._sources.clear()

    @property
    def results(self) -> list[dict]:
        with self._lock:
            return list(self._results)

    @property
    def sources(self) -> list[str]:
        with self._lock:
            return list(self._sources)


# ---------------------------------------------------------------------------
# Tool factory helpers
# ---------------------------------------------------------------------------


# User-facing names for the agent's tools — used in the live milestone
# messages so the chat thinking-text reads "Searching PubMed for …"
# instead of "Tool: search_pubmed — …". Falls back to title-casing the
# raw tool name for tools without an explicit entry, so newly added
# engines work cleanly without a code change.
_TOOL_DISPLAY_NAMES = {
    "web_search": "the web",
    "search_pubmed": "PubMed",
    "search_arxiv": "arXiv",
    "search_semantic_scholar": "Semantic Scholar",
    "search_openalex": "OpenAlex",
    "search_searxng": "the web (SearXNG)",
    "search_google_scholar": "Google Scholar",
    "search_brave": "Brave Search",
    "search_duckduckgo": "DuckDuckGo",
    "search_serper": "Google (Serper)",
    "search_scaleserp": "Google (ScaleSERP)",
    "search_wikipedia": "Wikipedia",
    "search_github": "GitHub",
    "search_stackexchange": "Stack Exchange",
    "search_openlibrary": "Open Library",
    "search_gutenberg": "Project Gutenberg",
    "search_pubchem": "PubChem",
    "search_zenodo": "Zenodo",
    "search_nasa_ads": "NASA ADS",
    "search_local": "your library",
    "fetch_content": "the page",
    "research_subtopic": "subtopic researcher",
}

# The step heartbeat lists tools as a comma list ("selecting next action
# from X, Y, Z…"). The sentence-fragment display names of the two
# non-search tools read wrong in that context ("the page", "subtopic
# researcher"), so the heartbeat uses these list-friendly labels instead;
# every other tool falls through to ``_display_tool_name``.
_HEARTBEAT_TOOL_LABELS = {
    "fetch_content": "page fetching",
    "research_subtopic": "subtopic research",
}

# Bounds for observation progress events. The one-line preview feeds the
# log panel / current-task line / thinking bubble; the detail
# (``metadata["content"]``) is persisted per chat step and emitted per
# socket event, so it must stay bounded — but large enough to show what a
# search or page fetch actually returned when the user expands the step.
# Detail is only attached when the output exceeds the preview, so short
# results ("No results.") aren't shown twice in the expanded step.
_OBSERVATION_PREVIEW_MAX_CHARS = 150
_OBSERVATION_DETAIL_MAX_CHARS = 4000


def _truncate_arg(value: str, limit: int = 80) -> str:
    """Cap a tool-call arg for the one-line progress message.

    Marks the cut with an ellipsis so a shortened query/URL doesn't read
    as if it were the complete value.
    """
    return value[:limit] + "…" if len(value) > limit else value


def _tool_display_name(name: str) -> str:
    """Friendly name for a tool, falling back to a cleaned raw name."""
    if name in _TOOL_DISPLAY_NAMES:
        return _TOOL_DISPLAY_NAMES[name]
    # Strip leading "search_" and title-case for unknown engines.
    cleaned = name[len("search_") :] if name.startswith("search_") else name
    return cleaned.replace("_", " ").title()


def _format_results(results: list[dict], start_idx: int) -> str:
    """Format search results as ``[N] Title (URL)\\nSnippet``."""
    lines = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        idx = start_idx + i + 1
        title = r.get("title", "No title")
        link = r.get("link", r.get("url", ""))
        snippet = r.get("snippet", r.get("body", ""))
        lines.append(f"[{idx}] {title} ({link})\n{snippet}")
    return "\n\n".join(lines) if lines else "No results."


def _format_indexed_results(results: list[tuple[dict, int]]) -> str:
    """Format results whose citation ids may be reused after URL dedup."""
    lines = []
    for result, idx in results:
        title = result.get("title", "No title")
        link = result.get("link", result.get("url", ""))
        snippet = result.get("snippet", result.get("body", ""))
        lines.append(f"[{idx}] {title} ({link})\n{snippet}")
    return "\n\n".join(lines) if lines else "No results."


def _normalized_candidate_pool(value: Any) -> list[dict[str, str]]:
    """Normalize the explicitly supplied Candidate-v2 pool.

    Candidate metadata is runtime-visible by design, but evaluator fields and
    oracle URLs are not read here.  Normalizing at the boundary also means a
    malformed fixture degrades to the normal live-search path instead of
    creating a hidden selection preference.
    """
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        normalized.append(
            {
                "url": url,
                "link": url,
                "title": str(item.get("title") or "Untitled candidate").strip(),
                "snippet": str(item.get("snippet") or "").strip(),
            }
        )
    return normalized


def _candidate_pool_digest(candidates: list[dict[str, str]]) -> str | None:
    if not candidates:
        return None
    payload = json.dumps(candidates, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _one_shot_candidate_urls(
    response: Any, candidates: list[dict[str, Any]], *, limit: int = 2
) -> list[str]:
    """Parse a bounded one-shot selector reply without accepting new URLs."""
    raw_content = getattr(response, "content", response)
    if isinstance(raw_content, list):
        raw_content = "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in raw_content
        )
    if not isinstance(raw_content, str):
        return []
    payload: Any = None
    for match in re.finditer(r"\{.*?\}", raw_content, flags=re.DOTALL):
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if not isinstance(payload, dict):
        return []
    selected = payload.get("selected_candidate_ids")
    if not isinstance(selected, list):
        return []
    by_id = {
        f"C{index + 1}": str(candidate.get("link") or candidate.get("url") or "")
        for index, candidate in enumerate(candidates)
    }
    chosen: list[str] = []
    for item in selected:
        url = by_id.get(str(item).strip().upper())
        if url and url not in chosen:
            chosen.append(url)
        if len(chosen) >= limit:
            break
    return chosen


def _make_web_search_tool(
    search_engine_name: str,
    model: BaseChatModel,
    settings_snapshot: dict,
    collector: SearchResultsCollector,
    programmatic_mode: bool = False,
    description: str = NEUTRAL_PRIMARY_SEARCH_DESCRIPTION,
    fixed_candidates: list[dict[str, str]] | None = None,
):
    """Create a ``web_search`` tool that instantiates a fresh engine per call."""

    result_cache: dict[str, str] = {}
    in_flight: dict[str, threading.Event] = {}
    cache_lock = threading.Lock()

    def execute_search(query: str) -> str:
        if fixed_candidates:
            indexed = collector.add_results_deduplicated(
                fixed_candidates, engine_name="candidate_pool"
            )
            return _format_indexed_results(indexed)
        from local_deep_research.utilities.resource_utils import safe_close
        from local_deep_research.web_search_engines.search_engine_factory import (
            create_search_engine,
        )

        engine = create_search_engine(
            engine_name=search_engine_name,
            llm=model,
            settings_snapshot=settings_snapshot,
            programmatic_mode=programmatic_mode,
        )
        if engine is None:
            return f"Failed to create search engine '{search_engine_name}'."
        try:
            results = engine.run(query)
            if not isinstance(results, list) or not results:
                return f"No results found for '{query}'. Try rephrasing."
            indexed = collector.add_results_deduplicated(
                results, engine_name=search_engine_name
            )
            return _format_indexed_results(indexed)
        except Exception as exc:
            logger.exception("web_search tool error")
            # Scrub credentials: a search-engine exception can embed the
            # request URL, which may carry an API key. Full detail is logged
            # server-side above.
            return _scrub_tool_error(f"Search error: {exc}")
        finally:
            safe_close(engine, "web search engine")

    @tool
    def web_search(query: str) -> str:
        """Search the selected source and return result snippets with source indices."""
        cache_key = " ".join(query.split()).casefold()
        with cache_lock:
            cached = result_cache.get(cache_key)
            if cached is not None:
                return cached
            event = in_flight.get(cache_key)
            owns_request = event is None
            if owns_request:
                event = threading.Event()
                in_flight[cache_key] = event

        if not owns_request:
            assert event is not None
            event.wait(timeout=40)
            with cache_lock:
                cached = result_cache.get(cache_key)
            if cached is not None:
                return cached
            return "Duplicate search could not reuse the in-flight result."

        try:
            output = execute_search(query)
        except BaseException:
            with cache_lock:
                in_flight.pop(cache_key, None)
                assert event is not None
                event.set()
            raise

        with cache_lock:
            result_cache[cache_key] = output
            in_flight.pop(cache_key, None)
            assert event is not None
            event.set()
        return output

    web_search.description = description
    return web_search


_FROZEN_OBSERVATION_MISS_PREFIX = "FROZEN_OBSERVATION_CACHE_MISS"
_FROZEN_SEARCH_RESULT = re.compile(
    r"^\[(?P<index>\d+)\]\s+(?P<title>.+?)\s+\((?P<url>[^\n)]+)\)\n"
    r"(?P<snippet>.*?)(?=^\[\d+\]\s+|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
_FROZEN_FETCH_RESULT = re.compile(
    r"^\[(?P<index>\d+)\]\s+Title:\s*(?P<title>.*?)\n"
    r"URL:\s*(?P<url>[^\n]+)\n"
    r"(?:(?:Retrieval:\s*(?P<retrieval>[^\n]+))\n)?\n"
    r"(?P<content>.+)\Z",
    flags=re.DOTALL,
)


def _apply_frozen_observation_to_collector(
    collector: SearchResultsCollector,
    tool_name: str,
    tool_input: dict[str, Any],
    content: Any,
) -> None:
    """Replay only the collector side effect of a cached tool observation.

    Frozen policy evaluation keeps the original observation text as the model
    sees it.  The writer and uncertainty tracker also consume the collector,
    so successful cached searches/fetches must recreate that public runtime
    state without invoking a live engine or URL fetcher.
    """
    if not isinstance(content, str):
        return
    if tool_name == "fetch_content":
        match = _FROZEN_FETCH_RESULT.match(content)
        if match is None:
            return
        url = str(tool_input.get("url") or match.group("url")).strip()
        if not url:
            return
        retrieval = match.group("retrieval")
        collector.record_fetched_content(
            url=url,
            title=match.group("title").strip(),
            content=match.group("content"),
            retrieval_method=(
                "jina_public_mirror"
                if retrieval and "public text mirror" in retrieval.lower()
                else None
            ),
        )
        return
    if tool_name != "web_search" and not tool_name.startswith("search_"):
        return
    results = [
        {
            "title": match.group("title").strip(),
            "link": match.group("url").strip(),
            "url": match.group("url").strip(),
            "snippet": match.group("snippet").strip(),
        }
        for match in _FROZEN_SEARCH_RESULT.finditer(content)
        if match.group("url").strip()
    ]
    if results:
        collector.add_results_deduplicated(results, engine_name=tool_name)


def _wrap_tools_with_frozen_observations(
    tools: list,
    *,
    collector: SearchResultsCollector,
    session: Any,
) -> list:
    """Return strict offline tool wrappers for a frozen policy-eval run.

    A cache miss is an explicit deterministic observation, never a live call.
    The wrapper deliberately leaves the Planner's tool names/schemas unchanged.
    """
    from local_deep_research.agent_harness import ReplayCacheMiss

    wrapped_tools = []
    for original in tools:
        tool_name = str(getattr(original, "name", ""))
        if not tool_name:
            wrapped_tools.append(original)
            continue

        def replay_tool(
            _tool_name: str = tool_name,
            **tool_input: Any,
        ) -> Any:
            """Return the strict cached observation for this tool invocation."""
            try:
                observation = session.execute(_tool_name, tool_input)
            except ReplayCacheMiss as exc:
                # ReplayCacheMiss is intentionally rendered as a tool result:
                # the Planner can react, but no uncached action reaches live
                # search/fetch infrastructure.
                return f"{_FROZEN_OBSERVATION_MISS_PREFIX}: {exc}"
            _apply_frozen_observation_to_collector(
                collector, _tool_name, tool_input, observation
            )
            return observation

        wrapped_tools.append(
            StructuredTool.from_function(
                replay_tool,
                name=tool_name,
                description=str(getattr(original, "description", "")),
                args_schema=getattr(original, "args_schema", None),
                return_direct=bool(getattr(original, "return_direct", False)),
            )
        )
    return wrapped_tools


# Fetch tool builders (full / summary_focus / summary_focus_query / disabled)
# live in ``advanced_search_system.tools.fetch``; see ``build_fetch_tool``.


def _make_specialized_search_tool(
    engine_name: str,
    description: str,
    model: BaseChatModel,
    settings_snapshot: dict,
    collector: SearchResultsCollector,
    programmatic_mode: bool = False,
    query_rewrite_enabled: bool = True,
):
    """Create a ``search_{engine}`` tool for a specific search engine."""

    @tool
    def specialized_search(query: str) -> str:
        """Search a specialized engine."""  # overridden below
        from local_deep_research.utilities.resource_utils import safe_close
        from local_deep_research.web_search_engines.search_engine_factory import (
            create_search_engine,
        )

        engine = create_search_engine(
            engine_name=engine_name,
            llm=model,
            settings_snapshot=settings_snapshot,
            programmatic_mode=programmatic_mode,
        )
        if engine is None:
            return f"Failed to create {engine_name} engine."
        try:
            effective_query = query
            normalization_error = None
            if engine_name.startswith("collection_"):
                effective_query, normalization_error = (
                    _normalize_collection_query(
                        query, enabled=query_rewrite_enabled
                    )
                )
            results = engine.run(effective_query)
            # A normalizer is an optional retrieval preprocessor, never a
            # reason to hide a useful native hit.  Retry the original query
            # only when the rewritten one returns nothing.
            used_raw_fallback = False
            if (
                not results
                and effective_query != query
                and engine_name.startswith("collection_")
            ):
                results = engine.run(query)
                used_raw_fallback = bool(results)
            if not isinstance(results, list) or not results:
                return f"No results from {engine_name} for '{query}'. Try rephrasing."
            indexed = collector.add_results_deduplicated(
                results, engine_name=engine_name
            )
            rendered = _format_indexed_results(indexed)
            if effective_query != query:
                rendered = (
                    "[collection-query-rewrite] "
                    + json.dumps(
                        {
                            "original_query": query,
                            "normalized_query": effective_query,
                            "raw_fallback": used_raw_fallback,
                        },
                        ensure_ascii=False,
                    )
                    + f"\n{rendered}"
                )
            elif normalization_error is not None:
                rendered = (
                    "[collection-query-rewrite] "
                    + json.dumps(
                        {
                            "original_query": query,
                            "normalized_query": query,
                            "fallback": normalization_error,
                        },
                        ensure_ascii=False,
                    )
                    + f"\n{rendered}"
                )
            return rendered
        except Exception as exc:
            logger.exception(f"search_{engine_name} tool error")
            return _scrub_tool_error(f"Search error ({engine_name}): {exc}")
        finally:
            safe_close(engine, f"{engine_name} search engine")

    # Override name and description after decoration
    specialized_search.name = f"search_{engine_name}"
    specialized_search.description = description
    return specialized_search


def _load_specialized_engine_tools(
    skip_engine: str | None,
    model: BaseChatModel,
    settings_snapshot: dict,
    collector: SearchResultsCollector,
    programmatic_mode: bool = False,
    egress_context=None,
    engine_allowlist: set[str] | None = None,
    query_rewrite_enabled: bool = True,
) -> list:
    """Load tools for all available specialized search engines, filtered by
    egress policy and per-engine ``agent_enabled`` flag.

    ``skip_engine`` names the engine already exposed through the caller's
    generic ``web_search`` tool, so it isn't double-registered; pass ``None``
    when no ``web_search`` tool exists — every allowed engine then stays
    reachable as a specialized tool.

    Shared by ``_build_tools`` (lead agent) and subagent tool setup so both
    layers apply the SAME policy/enrichment logic — see the inline-block
    comment in ``_build_tools`` for why this pre-filtering matters (the
    factory PEP catches violations at instantiation time but the LLM still
    SEES forbidden tool names in the schema, leaking policy state).

    Each returned tool is a closure that creates a fresh engine per
    invocation, so the tool objects themselves are safe to reuse across
    threads (e.g. when a ``research_subtopic`` call fans out to parallel
    subagents that share one tool list).
    """
    tools: list = []
    try:
        from local_deep_research.web_search_engines.search_engines_config import (
            list_eligible_engine_configs,
        )
        from local_deep_research.security.egress.policy import (
            EgressScope,
            evaluate_engine,
            evaluate_retriever,
        )
        from local_deep_research.web_search_engines.retriever_registry import (
            retriever_registry,
        )

        # Discover the candidate pool INDEPENDENT of ``use_in_auto_search``.
        # The agent's "specialized tool surface" is governed by per-engine
        # ``agent_enabled``, credentials, and egress policy — NOT the
        # auto-search-mode toggle (which controls the non-agent ``auto``
        # search surface only). The two settings were siblings on the same
        # settings UI prior to #5015, but they actually control two
        # different surfaces; conflating them made it impossible to expose
        # Tavily / Google PSE (default ``use_in_auto_search=false``) to the
        # agent without also re-enabling them in the ``auto`` search path.
        # Dynamic ``collection_*`` engines never had a ``use_in_auto_search``
        # setting, so the old path locked them out of the agent entirely
        # — this discovery change restores symmetry with #4453.
        eligible = list_eligible_engine_configs(
            settings_snapshot=settings_snapshot,
            egress_context=egress_context,
            check_agent_enabled=True,
        )
    except Exception:
        logger.exception(
            "Failed to discover specialized search engines",
            skip_engine=skip_engine,
        )
        return tools

    for name, config in eligible.items():
        try:
            if name == skip_engine:
                continue
            if engine_allowlist is not None and name not in engine_allowlist:
                logger.debug(
                    "specialized tool skipped: engine not in run allowlist",
                    engine=name,
                )
                continue

            # Per-engine usability switch (independent of egress). Collection
            # configs carry their DB flag; built-in engines receive the
            # flattened search.engine.web.<name>.agent_enabled setting.
            # Note: Re-checked here to log debug info for specialized tools.
            # Missing flags default to available for backward compatibility.
            # The primary engine was skipped above and remains reachable
            # through the caller's generic web_search tool.
            if not config.get("agent_enabled", True):
                logger.debug(
                    "specialized tool skipped: engine disabled for "
                    "the research agent",
                    engine=name,
                )
                continue

            # Under STRICT, register no specialized engines at all — the
            # agent gets only its primary web_search tool. (Note: STRICT-scope
            # blanket-skip is intentionally enforced here as list_eligible_engine_configs
            # does not blanket-enforce STRICT).
            if (
                egress_context is not None
                and egress_context.scope == EgressScope.STRICT
            ):
                continue

            # Under PUBLIC_ONLY / PRIVATE_ONLY, ask the PDP whether this
            # engine fits the scope. Re-evaluating here produces policy_audit
            # log entries for specialized tools filtered by egress policy.
            # Retrievers route to evaluate_retriever
            # (engine-PDP returns engine_unknown for them); plain engines
            # route to evaluate_engine.
            if egress_context is not None:
                try:
                    if config.get("is_retriever"):
                        try:
                            meta = retriever_registry.get_metadata(name)
                        except AttributeError:
                            meta = None
                        decision = evaluate_retriever(
                            name, egress_context, metadata=meta
                        )
                    else:
                        # Pass the engine config as metadata so a per-collection
                        # is_public classification is honored without a
                        # redundant DB lookup per collection.
                        decision = evaluate_engine(
                            name,
                            egress_context,
                            settings_snapshot=settings_snapshot,
                            metadata=config,
                        )
                except Exception:
                    logger.bind(policy_audit=True).exception(
                        "specialized tool skipped: policy evaluation failed",
                        engine=name,
                        scope=egress_context.scope.value,
                    )
                    continue
                if not decision.allowed:
                    logger.bind(policy_audit=True).info(
                        "specialized tool filtered by egress policy",
                        engine=name,
                        scope=egress_context.scope.value,
                        reason=decision.reason,
                    )
                    continue

            desc = config.get("description", f"Search using {name}")
            strengths = config.get("strengths", [])
            if strengths:
                desc += f" Best for: {', '.join(strengths[:2])}."
            tools.append(
                _make_specialized_search_tool(
                    name,
                    desc,
                    model,
                    settings_snapshot,
                    collector,
                    programmatic_mode=programmatic_mode,
                    query_rewrite_enabled=query_rewrite_enabled,
                )
            )
        except Exception:
            logger.exception(
                "Failed to load specialized search engine",
                engine=name,
            )
    return tools


def _make_research_subtopic_tool(
    search_engine_name: str,
    model: BaseChatModel,
    settings_snapshot: dict,
    collector: SearchResultsCollector,
    max_sub_iterations: int,
    search_enabled: bool = True,
    progress_callback=None,
    programmatic_mode: bool = False,
    fetch_mode: str = "summary_focus_query",
    overall_query: str = "",
    egress_context=None,
    max_subagent_workers: int = MAX_SUBAGENT_WORKERS,
    library_resolver: Any = None,
    web_search_description: str = NEUTRAL_PRIMARY_SEARCH_DESCRIPTION,
    search_engine_allowlist: set[str] | None = None,
    query_rewrite_enabled: bool = True,
    include_specialized_tools: bool = True,
    public_fetch_fallback: str = "disabled",
    require_observed_fetch_urls: bool = False,
):
    """Create the ``research_subtopic`` tool that spawns parallel subagents.

    ``overall_query`` is the original user query passed by the lead agent's
    strategy; it's forwarded to summary-mode fetch tools so the per-page
    extractor sees both the agent's per-fetch focus and the original
    research question.

    ``max_subagent_workers`` bounds the pool size. Surplus subtopics queue
    and start as workers free up; each queued subagent gets its own per-task
    deadline measured from when *it* actually begins executing (not from
    drain-loop start -- see #5014). The value comes from the user setting
    ``langgraph_agent.max_subagent_workers`` and falls back to
    ``MAX_SUBAGENT_WORKERS`` when unset / invalid.

    ``library_resolver`` is threaded into the subagent's fetch tool so a
    subagent researching a library-derived subtopic can also resolve
    ``/library/document/<uuid>`` URLs and ``[N]`` citation markers instead
    of burning the egress-denial quota on them (A3). When ``None``,
    library / citation URLs fall through to the egress gate unchanged.
    """

    @tool(
        description=(
            f"Delegate parallel research on multiple subtopics. Each subtopic is "
            f"investigated by a separate agent. Pass 2-{MAX_SUBTOPICS} focused "
            f"research questions."
        )
    )
    def research_subtopic(subtopics: list[str]) -> str:
        """Description passed via ``@tool(description=...)`` above — f-strings can't be docstrings."""
        from langchain.agents import create_agent

        if not subtopics:
            return "No subtopics provided."

        requested_count = len(subtopics)
        truncated_from: int | None = None
        dropped_subtopics: list[str] = []
        if requested_count > MAX_SUBTOPICS:
            logger.warning(
                "research_subtopic received {} subtopics; truncating to MAX_SUBTOPICS={}",
                requested_count,
                MAX_SUBTOPICS,
            )
            dropped_subtopics = subtopics[MAX_SUBTOPICS:]
            subtopics = subtopics[:MAX_SUBTOPICS]
            truncated_from = requested_count

        # Build subagent tools ONCE per ``research_subtopic`` call — reused
        # across all parallel subagent invocations. Each tool factory creates
        # a fresh engine per invocation and ``SearchResultsCollector`` is
        # lock-protected, so sharing the tool objects across pool workers
        # is safe. ``research_subtopic`` is itself excluded so subagents
        # cannot recurse.
        sub_tools: list = []
        if search_enabled:
            sub_tools.append(
                _make_web_search_tool(
                    search_engine_name,
                    model,
                    settings_snapshot,
                    collector,
                    programmatic_mode=programmatic_mode,
                    description=web_search_description,
                )
            )
        sub_fetch = build_fetch_tool(
            fetch_mode,
            collector,
            model=model,
            overall_query=overall_query,
            settings_snapshot=settings_snapshot,
            egress_context=egress_context,
            library_resolver=library_resolver,
            public_fetch_fallback=public_fetch_fallback,
            require_observed_urls=require_observed_fetch_urls,
        )
        if sub_fetch is not None:
            sub_tools.append(sub_fetch)
        # Give subagents the same specialized-engine set the lead agent
        # gets, filtered by the same egress policy / agent_enabled gate via
        # the shared helper — without this, a subagent researching a
        # medical topic couldn't call PubMed directly and would fall back
        # to the generic web_search.
        if include_specialized_tools:
            sub_tools.extend(
                _load_specialized_engine_tools(
                    # Skip the primary only when web_search above exposes it —
                    # with search_enabled=False the subagent has no web_search,
                    # so skipping would make the primary engine unreachable.
                    search_engine_name if search_enabled else None,
                    model,
                    settings_snapshot,
                    collector,
                    programmatic_mode=programmatic_mode,
                    egress_context=egress_context,
                    engine_allowlist=search_engine_allowlist,
                    query_rewrite_enabled=query_rewrite_enabled,
                )
            )

        if not sub_tools:
            # No primary search engine, fetching disabled, and every
            # specialized engine filtered out (e.g. STRICT egress scope):
            # a tool-less subagent would return un-grounded LLM text
            # dressed up as research findings. Refuse instead — before the
            # milestone below, so the UI never announces sub-research that
            # won't run. (_build_tools drops research_subtopic entirely
            # when the lead toolbox is otherwise empty; this guard covers
            # any remaining divergence between the two layers' gating.)
            logger.warning(
                "research_subtopic invoked with no tools available; "
                "refusing to run tool-less subagents"
            )
            return (
                "research_subtopic is unavailable: no research tools "
                "(search, fetch, or specialized engines) are permitted in "
                "this configuration. Answer from sources already gathered."
            )

        # Emit progress for UI
        if progress_callback:
            meta = {
                "phase": "sub_research",
                "type": "milestone",
                "subtopics": subtopics,
            }
            # Only surface the truncation key when subtopics were actually
            # dropped, so UI consumers don't have to special-case a None value.
            if truncated_from is not None:
                meta["truncated_from"] = truncated_from
            progress_callback(
                f"Researching {len(subtopics)} subtopics in parallel",
                None,
                meta,
            )

        current_date = datetime.now(UTC).strftime("%Y-%m-%d")
        subagent_prompt = (
            f"You are a focused research assistant. Today's date: {current_date}. "
            "Search thoroughly and return a concise factual summary. "
            "Reference sources by their [N] index numbers. "
            "Do NOT ask clarifying questions — provide your findings directly."
        )
        specialized_names = [
            t.name
            for t in sub_tools
            if isinstance(getattr(t, "name", None), str)
            and t.name.startswith("search_")
        ]
        if specialized_names:
            # Name only the tools actually registered — a static example
            # list could advertise policy-filtered engines the subagent
            # must never learn about.
            subagent_prompt += (
                " Prefer these domain-specific search tools when one "
                f"matches the topic: {', '.join(specialized_names)}."
            )

        def run_subagent(topic: str) -> str:
            try:
                # create_agent() calls model.bind_tools(); ProcessingLLMWrapper
                # (config/llm_config.py) overrides bind_tools to re-wrap the
                # bound model, so the wrapper's <think>-tag stripping survives
                # the agent loop and runs on every model call here (fix #4804).
                # Scope note: other Runnable transforms still escape the wrapper
                # — with_config/bind/stream delegate via __getattr__ to the
                # unwrapped base model (silent, unstripped), and `|` raises
                # TypeError (the wrapper defines no __or__). None are on this
                # create_agent path. Closing that whole class would need a full
                # Runnable-subclass wrapper.
                agent = create_agent(
                    model=model,
                    tools=sub_tools,
                    system_prompt=subagent_prompt,
                )
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": topic}]},
                    {"recursion_limit": max_sub_iterations * 2 + 1},
                )
                messages = result.get("messages", [])
                if messages:
                    last = messages[-1]
                    content = getattr(last, "content", str(last))
                    if content:
                        return content
                return f"No findings for: {topic}"
            except GraphRecursionError:
                return f"Research on '{topic}' reached iteration limit. Partial findings above."
            except Exception as exc:
                logger.exception(f"Subagent failed for: {topic[:80]}")
                return _scrub_tool_error(f"Research on '{topic}' failed: {exc}")

        # Capture the lead thread's search context (it carries the user's DB
        # password) so each pool worker can open the per-user ENCRYPTED database
        # when a subagent re-creates a search engine / registers the user's
        # document collections. stdlib ThreadPoolExecutor does NOT propagate the
        # ContextVar — without this, a collection/library primary fails inside a
        # subagent with "Unknown search engine 'collection_…'". This is the same
        # gap sibling strategies (source_based, focused_iteration) close with
        # @preserve_research_context; captured ONCE here on the lead thread.
        captured_search_context = get_search_context()

        # Worker lifecycle timestamps are written under a condition so the
        # drain loop wakes both when a queued task actually starts and when a
        # running task finishes. Submit time is deliberately not tracked: it
        # includes time spent waiting for a free worker and caused #5014. Task
        # IDs keep duplicate subtopic strings independent.
        task_state_changed = threading.Condition()
        task_start_times: dict[int, float] = {}
        task_end_times: dict[int, float] = {}

        def _run_subagent_with_egress(task: tuple[int, str]) -> str:
            task_id, topic = task
            with task_state_changed:
                task_start_times[task_id] = time.monotonic()
                task_state_changed.notify_all()

            try:
                with thread_cleanup():
                    # threading.local is NOT inherited by ThreadPoolExecutor
                    # workers, so re-arm the PEP-578 audit-hook backstop for the
                    # subagent's lifetime.
                    from ...security.egress.audit_hook import (
                        active_egress_context,
                    )

                    with active_egress_context(egress_context):
                        # search_context sets the password ContextVar for this
                        # worker and clears it on exit, preventing pool reuse from
                        # leaking credentials between tasks.
                        if captured_search_context is not None:
                            with search_context(captured_search_context):
                                return run_subagent(topic)
                        return run_subagent(topic)
            finally:
                with task_state_changed:
                    task_end_times[task_id] = time.monotonic()
                    task_state_changed.notify_all()

        ordered_results: dict[int, str] = {}
        # Clamp to >=1 so a misconfigured setting cannot produce a 0-worker
        # pool that silently deadlocks the drain loop. len(subtopics) >= 1
        # is guaranteed by the early-return at the top of the tool body.
        effective_workers = max(1, min(max_subagent_workers, len(subtopics)))
        # Overall safety cap for the drain loop. Sized to cover the worst-
        # case queue wait (ceil(subtopics/workers) waves) plus slack. This is
        # only a backstop; each started task has its own earlier deadline.
        overall_timeout_seconds = (
            SUBAGENT_TIMEOUT_SECONDS
            * max(1, math.ceil(len(subtopics) / effective_workers))
            * SUBAGENT_TIMEOUT_OVERALL_MULTIPLIER
        )
        drain_start = time.monotonic()
        overall_deadline = drain_start + overall_timeout_seconds

        def _record_per_task_timeout(
            task_id: int, topic: str, elapsed: float
        ) -> None:
            logger.warning(
                f"Subagent timed out (per-task): {topic[:80]} "
                f"ran {elapsed:.1f}s of "
                f"{SUBAGENT_TIMEOUT_SECONDS}s budget"
            )
            ordered_results[task_id] = (
                f"Research on '{topic}' timed out after {elapsed:.1f}s "
                f"(per-subagent budget is {SUBAGENT_TIMEOUT_SECONDS}s)."
            )

        executor = ThreadPoolExecutor(max_workers=effective_workers)
        futures = {}
        try:
            futures = {
                executor.submit(_run_subagent_with_egress, (task_id, topic)): (
                    task_id,
                    topic,
                )
                for task_id, topic in enumerate(subtopics)
            }
            pending = set(futures)

            while pending:
                completed = []
                expired = []
                overall_expired = False

                with task_state_changed:
                    while True:
                        now = time.monotonic()
                        completed = [
                            future
                            for future in pending
                            if futures[future][0] in task_end_times
                        ]
                        if completed:
                            break

                        expired = [
                            future
                            for future in pending
                            if (
                                futures[future][0] in task_start_times
                                and now - task_start_times[futures[future][0]]
                                >= SUBAGENT_TIMEOUT_SECONDS
                            )
                        ]
                        if expired:
                            break

                        if now >= overall_deadline:
                            overall_expired = True
                            break

                        deadlines = [overall_deadline]
                        deadlines.extend(
                            task_start_times[futures[future][0]]
                            + SUBAGENT_TIMEOUT_SECONDS
                            for future in pending
                            if futures[future][0] in task_start_times
                        )
                        task_state_changed.wait(
                            timeout=max(0.0, min(deadlines) - now)
                        )

                # A completed future can still have exceeded its own budget.
                # Check the worker-recorded duration before accepting its
                # result; future.result(timeout=...) cannot do this after a
                # completion iterator has already yielded the future.
                for future in completed:
                    pending.remove(future)
                    task_id, topic = futures[future]
                    start = task_start_times[task_id]
                    elapsed = max(0.0, task_end_times[task_id] - start)
                    if elapsed >= SUBAGENT_TIMEOUT_SECONDS:
                        _record_per_task_timeout(task_id, topic, elapsed)
                        continue
                    try:
                        ordered_results[task_id] = future.result()
                    except Exception as exc:
                        logger.exception(f"Subagent failed for: {topic[:80]}")
                        ordered_results[task_id] = _scrub_tool_error(
                            f"Research on '{topic}' failed: {exc}"
                        )

                # These futures are still running, but their individual
                # deadline has arrived. ThreadPoolExecutor cannot preempt a
                # running Python callable, so ignore its eventual result and
                # let shutdown(wait=False) release the lead agent promptly.
                now = time.monotonic()
                for future in expired:
                    pending.remove(future)
                    task_id, topic = futures[future]
                    elapsed = max(0.0, now - task_start_times[task_id])
                    _record_per_task_timeout(task_id, topic, elapsed)
                    future.cancel()

                if overall_expired:
                    now = time.monotonic()
                    for future in pending:
                        task_id, topic = futures[future]
                        start = task_start_times.get(task_id)
                        if start is None:
                            queued_elapsed = max(0.0, now - drain_start)
                            logger.warning(
                                f"Subagent exceeded overall safety cap: "
                                f"{topic[:80]} remained queued for "
                                f"{queued_elapsed:.1f}s and never started"
                            )
                            ordered_results[task_id] = (
                                f"Research on '{topic}' did not start before "
                                f"the overall safety budget of "
                                f"{overall_timeout_seconds}s expired "
                                f"(queued for {queued_elapsed:.1f}s; its "
                                f"{SUBAGENT_TIMEOUT_SECONDS}s per-subagent "
                                f"budget begins at task start)."
                            )
                        else:
                            elapsed = max(0.0, now - start)
                            logger.warning(
                                f"Subagent exceeded overall safety cap: "
                                f"{topic[:80]} ran {elapsed:.1f}s of "
                                f"{overall_timeout_seconds}s overall / "
                                f"{SUBAGENT_TIMEOUT_SECONDS}s per-task budget"
                            )
                            ordered_results[task_id] = (
                                f"Research on '{topic}' did not finish within "
                                f"the overall safety budget of "
                                f"{overall_timeout_seconds}s (ran "
                                f"{elapsed:.1f}s; per-subagent budget is "
                                f"{SUBAGENT_TIMEOUT_SECONDS}s)."
                            )
                        future.cancel()
                    pending.clear()
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

        # Return results in original order
        parts = []
        for task_id, topic in enumerate(subtopics):
            parts.append(
                f"## {topic}\n{ordered_results.get(task_id, 'No results')}"
            )
        result_text = "\n\n---\n\n".join(parts)

        # Surface the truncation to the lead agent itself (not just logs/UI):
        # otherwise the model believes the dropped subtopics were investigated
        # and may cite them (#5012).
        if dropped_subtopics:
            result_text += (
                f"\n\nNote: {len(dropped_subtopics)} subtopic(s) beyond the limit "
                f"of {MAX_SUBTOPICS} were not investigated: "
                f"{', '.join(repr(s) for s in dropped_subtopics)}"
            )
        return result_text

    return research_subtopic


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class LangGraphAgentStrategy(BaseSearchStrategy):
    """Research strategy using LangGraph agents with parallel subagent support.

    The lead agent autonomously decides what to search, when to dig deeper
    (via subagents), and when to synthesize — replacing the manual ReAct loop
    in the MCP strategy.
    """

    def __init__(
        self,
        model: BaseChatModel,
        search,
        citation_handler=None,
        synthesis_model: BaseChatModel | None = None,
        max_iterations: int = 50,
        max_sub_iterations: int = 8,
        include_sub_research: bool = True,
        all_links_of_system: list | None = None,
        settings_snapshot: dict | None = None,
        programmatic_mode: bool = False,
        trace_output_dir: str | None = None,
        primary_only_tools: bool = False,
        search_engine_allowlist: list[str]
        | tuple[str, ...]
        | set[str]
        | None = None,
        research_profile: str = "default",
        query_rewrite_enabled: bool = True,
        routing_mode: str = AUTO_ROUTING_MODE,
        static_official_domain: str | None = None,
        task_contract: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            all_links_of_system=all_links_of_system,
            settings_snapshot=settings_snapshot,
            **kwargs,
        )
        self.model = model
        self.synthesis_model = synthesis_model or model
        self.evidence_only_synthesis = synthesis_model is not None
        self.search = search
        # Whether the parent AdvancedSearchSystem is running in programmatic
        # mode (no DB metrics/rate-limit persistence). Threaded into the
        # tool factory closures so engines created per tool call inherit it.
        self.programmatic_mode = programmatic_mode
        # Explicitly opt in: production users may not want research prompts
        # and tool observations written outside the normal application DB.
        self.trace_output_dir = trace_output_dir or os.getenv(
            "LDR_AGENT_TRACE_DIR"
        )
        # search.iterations (typically 1-5) controls pipeline strategies.
        # For an agent, each "iteration" is one LLM→tool round-trip, so we
        # need many more.  Treat any value below the agent minimum as "use
        # default" rather than clamping to a uselessly low number.
        self.max_iterations = (
            int(max_iterations)
            if int(max_iterations) >= MIN_ITERATIONS
            else DEFAULT_MAX_ITERATIONS
        )
        self.max_sub_iterations = int(max_sub_iterations)
        self.include_sub_research = include_sub_research
        self.primary_only_tools = primary_only_tools
        self.search_engine_allowlist = (
            set(search_engine_allowlist)
            if search_engine_allowlist is not None
            else None
        )
        self.research_profile = research_profile.strip().lower() or "default"
        self.query_rewrite_enabled = bool(query_rewrite_enabled)
        if self.research_profile not in {"default", "hybrid"}:
            raise ValueError(
                "Unknown agent research profile "
                f"{research_profile!r}; expected 'default' or 'hybrid'."
            )
        self.task_contract = (
            dict(task_contract) if isinstance(task_contract, dict) else None
        )
        self.candidate_pool = _normalized_candidate_pool(
            self.task_contract.get("candidate_pool")
            if self.task_contract is not None
            else None
        )
        self.candidate_pool_digest = _candidate_pool_digest(self.candidate_pool)
        raw_showcase_failure = (
            self.task_contract.get("showcase_failure_injection")
            if self.task_contract is not None
            and self.task_contract.get("showcase_development") is True
            else None
        )
        if (
            isinstance(raw_showcase_failure, dict)
            and raw_showcase_failure.get("kind")
            == "first_observed_public_fetch"
        ):
            reason = str(raw_showcase_failure.get("reason") or "empty_body")
            self.showcase_fetch_failure_injection = {
                "kind": "first_observed_public_fetch",
                "reason": reason
                if reason in {"empty_body", "timeout", "blocked"}
                else "empty_body",
            }
        else:
            self.showcase_fetch_failure_injection = None
        raw_prompt_variant = (
            self.task_contract.get("planner_prompt_variant")
            if self.task_contract is not None
            else None
        )
        self.planner_prompt_variant = (
            str(raw_prompt_variant).strip().lower()
            if isinstance(raw_prompt_variant, str)
            and raw_prompt_variant.strip().lower() in {"vanilla", "strong"}
            else "vanilla"
        )
        self.planner_uncertainty_state_enabled = bool(
            self.task_contract
            and self.task_contract.get("planner_uncertainty_state_enabled")
        )
        raw_planner_state_arm = (
            self.task_contract.get("planner_state_arm")
            if self.task_contract is not None
            else None
        )
        normalized_planner_state_arm = str(raw_planner_state_arm).strip().lower()
        self.planner_state_arm = (
            normalized_planner_state_arm
            if normalized_planner_state_arm
            in {PLANNER_STATE_ARM_LEGACY_COMPACT, PLANNER_STATE_ARM_EXPANDED}
            else PLANNER_STATE_ARM_EXPANDED
        )
        self.planner_expanded_state_visible = bool(
            self.planner_uncertainty_state_enabled
            and self.planner_state_arm == PLANNER_STATE_ARM_EXPANDED
        )
        self.planner_stop_guard_enabled = bool(
            self.task_contract
            and self.task_contract.get("planner_stop_guard_enabled")
        )
        raw_max_fetch_calls = (
            self.task_contract.get("max_fetch_calls")
            if self.task_contract is not None
            else None
        )
        try:
            parsed_max_fetch_calls = int(raw_max_fetch_calls)
        except (TypeError, ValueError):
            parsed_max_fetch_calls = 0
        self.max_fetch_calls = (
            parsed_max_fetch_calls if parsed_max_fetch_calls > 0 else None
        )
        raw_fetches_per_turn = (
            self.task_contract.get("max_fetch_calls_per_planner_turn")
            if self.task_contract is not None
            else None
        )
        try:
            parsed_fetches_per_turn = int(raw_fetches_per_turn)
        except (TypeError, ValueError):
            parsed_fetches_per_turn = 0
        self.max_fetch_calls_per_planner_turn = (
            parsed_fetches_per_turn
            if parsed_fetches_per_turn > 0
            else None
        )
        self.candidate_evaluation_mode = str(
            self.task_contract.get("candidate_evaluation_mode") or ""
        ) if self.task_contract is not None else ""
        self.routing_decision = choose_routing_mode(
            requested_mode=routing_mode,
            research_profile=self.research_profile,
            task_contract=self.task_contract,
            static_official_domain=static_official_domain,
        )
        self.requested_routing_mode = self.routing_decision.requested_mode
        self.routing_mode = self.routing_decision.selected_mode
        if (
            self.routing_mode
            in {
                "collection_only",
                "static_dual",
                "static_dual_fetch",
                "static_top1",
                "static_topk",
                "static_ranked_fetch_2",
                "one_shot_llm_selector",
            }
            and self.research_profile != "hybrid"
        ):
            raise ValueError(
                f"routing mode {self.routing_mode!r} requires research_profile='hybrid'."
            )
        requested_static_domain = (
            static_official_domain.strip().lower().lstrip(".")
            if isinstance(static_official_domain, str)
            and static_official_domain.strip()
            else None
        )
        self.static_official_domain = (
            self.routing_decision.official_domain or requested_static_domain
        )
        if (
            self.routing_mode == "static_dual_fetch"
            and not self.static_official_domain
        ):
            raise ValueError(
                "routing mode 'static_dual_fetch' requires static_official_domain"
            )
        raw_evidence_policy_mode = os.getenv(
            "LDR_HYBRID_POLICY_GUARD", EvidencePolicyMode.OFF.value
        )
        self.evidence_policy_mode = EvidencePolicyMode.parse(
            raw_evidence_policy_mode
        )
        if raw_evidence_policy_mode.strip().lower() not in {
            mode.value for mode in EvidencePolicyMode
        }:
            logger.warning(
                "Unknown LDR_HYBRID_POLICY_GUARD=%r; defaulting to 'off'",
                raw_evidence_policy_mode,
            )
        if self.research_profile != "hybrid":
            # This is intentionally a Hybrid research experiment. Keeping the
            # default profile out of scope prevents a new optional control
            # plane from changing upstream-compatible behavior.
            self.evidence_policy_mode = EvidencePolicyMode.OFF
        elif self.evidence_policy_mode == EvidencePolicyMode.ENFORCE:
            # v1 has been designed and evaluated as a shadow guard. Exact
            # duplicate/budget stopping is already enforced below; adding a
            # new hard fetch rule before it has a full quality evaluation
            # would make the configuration label misleading and risky.
            logger.warning(
                "LDR_HYBRID_POLICY_GUARD=enforce is not enabled in v1; "
                "using shadow mode instead."
            )
            self.evidence_policy_mode = EvidencePolicyMode.SHADOW
        raw_model_call_cap = os.getenv("LDR_AGENT_MAX_MODEL_CALLS")
        try:
            parsed_model_call_cap = (
                int(raw_model_call_cap) if raw_model_call_cap else None
            )
        except ValueError:
            parsed_model_call_cap = None
        self.max_model_calls = (
            parsed_model_call_cap
            if parsed_model_call_cap is not None and parsed_model_call_cap > 0
            else None
        )
        raw_tool_call_cap = os.getenv("LDR_AGENT_MAX_TOOL_CALLS")
        try:
            parsed_tool_call_cap = (
                int(raw_tool_call_cap) if raw_tool_call_cap else None
            )
        except ValueError:
            parsed_tool_call_cap = None
        self.max_tool_calls = (
            parsed_tool_call_cap
            if parsed_tool_call_cap is not None and parsed_tool_call_cap > 0
            else None
        )
        raw_batch_cap = os.getenv("LDR_AGENT_MAX_TOOL_CALLS_PER_BATCH")
        try:
            parsed_batch_cap = int(raw_batch_cap) if raw_batch_cap else None
        except ValueError:
            parsed_batch_cap = None
        self.max_tool_calls_per_batch = (
            parsed_batch_cap
            if parsed_batch_cap is not None and parsed_batch_cap > 0
            else (
                2
                if self.research_profile == "hybrid"
                and self.max_tool_calls is not None
                else None
            )
        )
        raw_budget_replans = os.getenv("LDR_AGENT_MAX_BUDGET_REPLANS", "1")
        try:
            parsed_budget_replans = int(raw_budget_replans)
        except ValueError:
            parsed_budget_replans = 1
        self.max_budget_replans = max(0, parsed_budget_replans)
        if citation_handler is not None:
            self.citation_handler = citation_handler
        elif self.evidence_only_synthesis:
            from local_deep_research.agent_harness.evidence_synthesis import (
                EvidenceOnlyCitationHandler,
            )

            self.citation_handler = EvidenceOnlyCitationHandler(
                self.synthesis_model,
                settings_snapshot=settings_snapshot,
            )
        else:
            self.citation_handler = CitationHandler(
                self.synthesis_model,
                handler_type="standard",
                settings_snapshot=settings_snapshot,
            )
        self.collector = SearchResultsCollector(self.all_links_of_system)
        # Evaluation-only, intentionally outside task_contract: cache
        # provenance must not become Planner-visible state.
        frozen_cache_path = os.getenv("LDR_FROZEN_OBSERVATION_CACHE", "").strip()
        self.frozen_observation_cache_path = frozen_cache_path or None
        self.frozen_observation_session = None
        if self.frozen_observation_cache_path is not None:
            from local_deep_research.agent_harness import (
                ObservationCacheStore,
                ReplaySession,
            )

            cache = ObservationCacheStore.read(self.frozen_observation_cache_path)
            self.frozen_observation_session = ReplaySession(cache)
        # This view is observation-derived.  It intentionally ignores
        # evaluator-only contract fields and showcase fixture annotations.
        self.uncertainty_tracker = UncertaintyStateTracker(self.task_contract)

        fetch_mode = self.get_setting(
            "search.fetch.mode", "summary_focus_query"
        )
        if fetch_mode not in FETCH_MODES:
            logger.warning(
                f"Unknown search.fetch.mode={fetch_mode!r}, falling back to "
                f"'summary_focus_query'. Valid modes: {FETCH_MODES}"
            )
            fetch_mode = "summary_focus_query"
        self.fetch_mode = fetch_mode
        public_fetch_fallback = self.get_setting(
            "search.fetch.public_mirror_fallback", "disabled"
        )
        self.contract_fetch_fallback = (
            self.task_contract.get("public_fetch_fallback")
            if self.task_contract is not None
            else None
        )
        if self.contract_fetch_fallback in PUBLIC_FETCH_FALLBACKS:
            public_fetch_fallback = self.contract_fetch_fallback
        if public_fetch_fallback not in PUBLIC_FETCH_FALLBACKS:
            logger.warning(
                "Unknown search.fetch.public_mirror_fallback={!r}; "
                "falling back to 'disabled'. Valid values: {}",
                public_fetch_fallback,
                PUBLIC_FETCH_FALLBACKS,
            )
            public_fetch_fallback = "disabled"
        self.public_fetch_fallback = public_fetch_fallback
        logger.info(f"LangGraph agent fetch_mode={self.fetch_mode}")

        # User-tunable pool size for parallel subagents (follow-up to #5014).
        # Lets users match their LLM backend's parallel-request capacity --
        # Ollama / LMStudio / llama.cpp ``OLLAMA_NUM_PARALLEL``, an OpenAI
        # tier limit, etc. -- without code changes. Falls back to the
        # constant default on missing/invalid input so a misconfigured
        # setting cannot silently break the drain loop.
        raw_max_workers = self.get_setting(
            "langgraph_agent.max_subagent_workers", MAX_SUBAGENT_WORKERS
        )
        self.max_subagent_workers = self._coerce_max_subagent_workers(
            raw_max_workers
        )

        # Derive the search engine name for creating fresh instances
        self._search_engine_name = self._resolve_engine_name()

    @staticmethod
    def _coerce_max_subagent_workers(raw: Any) -> int:
        """Validate and clamp the user-supplied pool size.

        - Non-numeric, non-finite, or unparsable values fall back to
          ``MAX_SUBAGENT_WORKERS`` with a warning so a misconfigured setting
          cannot crash the constructor or silently produce a 0-worker pool.
        - Values below 1 are clamped up to 1; a 0-worker pool would deadlock
          the drain loop waiting on tasks no thread will ever run.
        - Values above 32 are clamped down -- past that point the bottleneck
          is almost always the LLM/search backend, not pool size, and an
          unbounded value invites accidental denial-of-service against the
          user's own infrastructure.
        """
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError):
            logger.warning(
                f"langgraph_agent.max_subagent_workers={raw!r} is not an "
                f"integer; falling back to MAX_SUBAGENT_WORKERS="
                f"{MAX_SUBAGENT_WORKERS}"
            )
            return MAX_SUBAGENT_WORKERS
        if value < 1:
            logger.warning(
                f"langgraph_agent.max_subagent_workers={value} is below 1; "
                f"clamping to 1 to avoid a deadlocked pool"
            )
            return 1
        if value > 32:
            logger.warning(
                f"langgraph_agent.max_subagent_workers={value} is above the "
                f"32-thread soft cap; clamping to 32. Set via env or "
                f"settings only if your LLM/search backend explicitly "
                f"supports it."
            )
            return 32
        return value

    def _resolve_engine_name(self) -> Optional[str]:
        """Best-effort extraction of the configured engine name.

        Returns a CANONICAL engine id (the same string that
        ``search_config`` / ``engine_registry`` use as a dict key, e.g.
        ``semantic_scholar`` rather than the class-derived
        ``semanticscholar``). The class-name fallback historically returned
        a lowercased, ``SearchEngine``-stripped variant that was almost
        but not quite canonical — e.g. ``DuckDuckGoSearchEngine`` →
        ``"duckduckgo"`` instead of the registry key ``"ddg"``, and
        ``SemanticScholarSearchEngine`` → ``"semanticscholar"`` instead of
        ``"semantic_scholar"``. That mismatch leaked the configured primary
        engine back into the specialised-tools loop as a redundant
        ``search_<engine>`` tool alongside ``web_search``. Reverse-lookup
        via ``ENGINE_REGISTRY`` so both sides of the
        ``if name == skip_engine: continue`` comparison agree on the
        canonical id (#5015 follow-up).

        Returns ``None`` when no canonical id can be derived — callers
        then fall through and the helper's primary-skip never matches
        (which is the only safe behaviour, since the factory would also
        fail to resolve an unknown engine).
        """
        # Try settings first — `search.tool` is already the canonical id
        # the user typed in the UI / env var.
        tool_setting = self.get_setting("search.tool", None)
        if tool_setting and isinstance(tool_setting, str):
            return tool_setting
        # Fall back to the engine CLASS, but resolve via the registry so
        # we land on the canonical id (``semantic_scholar``,
        # ``ddg``, ...) rather than the class-derived heuristic. The
        # registry is the single source of truth for which Python class
        # implements which canonical id.
        if self.search is not None and hasattr(self.search, "__class__"):
            cls_name = self.search.__class__.__name__
            # Lazy import: ``engine_registry`` already lives in the same
            # package tree the strategy uses for ``ENGINE_REGISTRY``
            # construction; keeping it inside the function avoids a
            # ``from . import engine_registry`` cycle at module load.
            from local_deep_research.web_search_engines.engine_registry import (
                ENGINE_REGISTRY,
            )

            for name, entry in ENGINE_REGISTRY.items():
                if entry.class_name == cls_name:
                    return name
        return None

    def _display_tool_name(self, tool_name: str) -> str:
        """Return a user-friendly display name for a tool.

        ``web_search`` is a generic wrapper around the user's configured
        engine. Resolve it through the same curated ``_TOOL_DISPLAY_NAMES``
        map as the specialized search tools (keyed by ``search_<engine>``)
        so the UI shows brand-correct names like "DuckDuckGo" or
        "the web (SearXNG)" instead of the raw lowercase engine id
        (e.g. "searxng"). Other tools use the map directly.
        """
        if tool_name == "web_search":
            return _tool_display_name(f"search_{self._search_engine_name}")
        return _tool_display_name(tool_name)

    def _format_tool_call_progress(self, tc, display_name: str) -> str:
        """Format a single tool call as a user-facing progress message.

        Extracted from the ``analyze_topic`` stream loop so the per-tool-type
        emoji + argument-extraction can be unit-tested without driving the full
        LangGraph stream. Behavior is preserved from the original inline block.

        - ``fetch_content`` → ``📖 Reading the page: "<url>"`` (URL arg).
        - ``research_subtopic`` → ``🔬 Investigating subtopic: "<…>"``
          (accepts ``subtopics`` list, ``subtopic``, or ``query`` for
          forward-compat with older signatures).
        - All other tools (specialized engines, ``web_search``) →
          ``🔍 Searching <Display Name>: "<query>"`` (falls back to
          ``url`` if ``query`` is absent).

        Arg extractions are truncated to 80 chars (marked with an ellipsis
        so a cut arg doesn't read as complete) to keep the chat-progress
        line bounded. Subtopic lists cap per item instead of cutting the
        joined string, so every subtopic stays visible — the collapsed
        step row ellipsizes via CSS and expands on click.
        """
        tc_args = tc.get("args", {})
        raw_name = tc.get("name", "")
        # `fetch_content` carries a URL arg; the search tools carry a query
        # arg. Either way, show the meaningful arg in quotes so the user
        # sees what the agent is actually looking up.
        if raw_name == "fetch_content":
            target = _truncate_arg(str(tc_args.get("url", "")))
            return f'📖 Reading {display_name}: "{target}"'
        if raw_name == "research_subtopic":
            # Tool signature is `subtopics: list[str]`. Accept either key for
            # forward-compat and stringify a list as a comma list.
            raw_sub = tc_args.get(
                "subtopics",
                tc_args.get(
                    "subtopic",
                    tc_args.get("query", ""),
                ),
            )
            if isinstance(raw_sub, list):
                sub = ", ".join(_truncate_arg(str(s)) for s in raw_sub)
            else:
                sub = _truncate_arg(str(raw_sub))
            return f'🔬 Investigating subtopic: "{sub}"'
        # Search-style tool — query arg (or URL if query missing). Use a
        # loop-local name here — do NOT reassign the `query` parameter, which
        # is still needed downstream by _synthesize_from_collector()/
        # _finalize() as the original research question.
        tc_query = _truncate_arg(
            str(
                tc_args.get(
                    "query",
                    tc_args.get("url", ""),
                )
            )
        )
        return f'🔍 Searching {display_name}: "{tc_query}"'

    def _observation_event(self, msg) -> tuple[str, dict]:
        """Build the (message, metadata) pair for a tool-result observation.

        The message stays a one-line 150-char preview — it feeds the log
        panel, the classic progress page's current-task line, and the chat
        thinking bubble, none of which can absorb a full tool result. When
        the output is longer than the preview, the (bounded) full output
        rides along in ``metadata["content"]``: the chat route persists it
        beneath the message so the click-to-expand step row shows what was
        actually fetched, and the classic progress page's agent-thinking
        panel appends it to its RESULT entry. Outputs the preview already
        shows verbatim attach no detail — the expanded step would just
        repeat the line ("No results." twice).

        Extracted from the ``analyze_topic`` stream loop so it can be
        unit-tested without driving the full LangGraph stream.
        """
        tool_name = getattr(msg, "name", "tool")
        display_name = self._display_tool_name(tool_name)
        raw = str(getattr(msg, "content", ""))

        # Suppress the misleading "📄 From the page: Cannot fetch <url>: ..."
        # pattern for ``fetch_content`` denial/error observations. The fetch
        # tool returns a "Cannot fetch …: blocked by egress policy (…)"
        # string when the egress gate refuses the URL (policy.py:_record_denial
        # already emits a WARNING with the same URL), and the chat panel
        # would re-emit that string under the "From the page:" label — which
        # reads to the user as if the page was read and its content is a
        # denial. The WARNING in the persisted log is the audit signal;
        # suppressing the MILESTONE here keeps the chat UI truthful. The
        # caller skips the _update_progress call when this returns None.
        # Gated on the tool name (not the content prefix alone) so a
        # non-fetch tool whose result happens to start with "Cannot fetch "
        # still surfaces normally — the suppression is about the
        # ``fetch_content`` denial framing, not a generic string-substring
        # match.
        if tool_name == "fetch_content" and (
            raw.startswith("Cannot fetch ") or raw.startswith("Error fetching ")
        ):
            return None

        preview = raw[:_OBSERVATION_PREVIEW_MAX_CHARS].replace("\n", " ")
        # Keep the stable tool id in metadata; the friendly label
        # already lives in the message.
        metadata = {"phase": "observation", "tool": tool_name}
        if raw.startswith("[collection-query-rewrite] "):
            first_line = raw.splitlines()[0]
            try:
                rewrite = json.loads(
                    first_line.removeprefix("[collection-query-rewrite] ")
                )
            except (json.JSONDecodeError, TypeError):
                rewrite = None
            if isinstance(rewrite, dict):
                metadata["query_rewrite"] = {
                    key: rewrite[key]
                    for key in (
                        "original_query",
                        "normalized_query",
                        "raw_fallback",
                        "fallback",
                    )
                    if key in rewrite
                }
        # Attach detail only when it adds something beyond the preview —
        # longer output, or short multi-line output whose newlines the
        # preview flattened. Outputs identical to the preview would just
        # be repeated in the expanded step.
        if raw != preview:
            detail = raw[:_OBSERVATION_DETAIL_MAX_CHARS]
            if len(raw) > _OBSERVATION_DETAIL_MAX_CHARS:
                detail += " …"
            metadata["content"] = detail
        return (f"📄 From {display_name}: {preview}", metadata)

    def _heartbeat_message(self, iteration: int) -> str:
        """Build the between-steps heartbeat line for the given iteration.

        Before any source is gathered the agent is still planning, so the
        line reports the size of its toolbox. Afterwards it lists EVERY
        enabled tool by friendly name — an earlier 3-name sample with
        "+N more" hid most engines, which users read as the agent having
        fewer options than it does.

        Extracted from the ``analyze_topic`` stream loop so it can be
        unit-tested without driving the full LangGraph stream.
        """
        sources_so_far = len(self.all_links_of_system)
        names = getattr(self, "_tool_names", []) or []
        if sources_so_far == 0:
            return (
                f"Step {iteration} · planning approach "
                f"with {len(names)} research tool"
                f"{'s' if len(names) != 1 else ''} available…"
            )
        listing = ", ".join(
            _HEARTBEAT_TOOL_LABELS.get(n) or self._display_tool_name(n)
            for n in names
        )
        return (
            f"Step {iteration} · {sources_so_far} source"
            f"{'s' if sources_so_far != 1 else ''} gathered · "
            f"selecting next action from {listing}…"
        )

    def _build_egress_context(self):
        """Construct the frozen ``EgressContext`` for this run.

        Returns ``None`` if a context can't be built (no snapshot, or
        invariant violation) — callers fall through to current behavior
        rather than crashing. Lazy import to avoid pulling the security
        module at strategy-class import time.
        """
        if not self.settings_snapshot:
            return None
        from local_deep_research.security.egress.policy import (
            PolicyDeniedError,
            context_from_snapshot,
            resolve_run_primary_engine,
        )

        try:
            # Derive the primary engine the SAME way the factory PEP does —
            # from ``search.tool`` — NOT from the engine class name. Under the
            # default ADAPTIVE scope the primary IS what resolves the concrete
            # scope, so a divergent primary here silently under-filters the
            # agent's tool list: a private collection primary classified via the
            # class heuristic ("libraryrag" -> unknown -> BOTH) left public
            # engines visible, which the factory then hard-denied mid-run
            # (scope_mismatch_private_only). resolve_run_primary_engine raises
            # ValueError when no primary is configured; this advisory filter
            # then degrades to unfiltered (the factory PEP still enforces) —
            # research_service has already failed the run closed by that point.
            primary = resolve_run_primary_engine(self.settings_snapshot)
            return context_from_snapshot(self.settings_snapshot, primary)
        except PolicyDeniedError:
            # Corrupted/invalid policy.egress_scope — re-raise so the
            # caller fails closed instead of silently running unfiltered.
            raise
        except (ValueError, KeyError, TypeError):
            logger.debug(
                "Could not build EgressContext for langgraph agent — "
                "falling back to unfiltered tool list"
            )
            return None

    def _build_library_resolver(self):
        """Build a ``library_resolver`` callable for the fetch tool.

        Returns ``None`` when no user is associated with the run (programmatic
        mode, benchmarks, news). The fetch tool then behaves exactly as it
        did before the A3 fix: library / citation URLs fall through to the
        egress gate and are rejected as ``unsupported_scheme``. Returning
        ``None`` keeps those callers' behaviour identical.
        """
        username = None
        if self.settings_snapshot:
            # The snapshot carries the username under the ``_username`` key
            # injected by ``AdvancedSearchSystem._ensure_snapshot_username``;
            # the strategy also checks ``self._username`` attribute as a fallback
            # for non-snapshot callers (tests, programmatic API).
            username = self.settings_snapshot.get("_username")
        if not username:
            username = getattr(self, "_username", None)
        if not username:
            return None
        return make_library_resolver(username)

    def _build_tools(self, overall_query: str = "") -> list:
        """Build the LangChain tool list for the lead agent.

        ``overall_query`` is the original user query; it's threaded into
        summary-mode fetch tools so the per-page extractor sees both the
        agent's per-fetch focus and the original research question.
        """
        tools = []

        # Compute the policy context ONCE for this run. Threaded through
        # every tool builder so subagent threads — which don't inherit
        # thread-local state — get the same context as the lead agent.
        policy_ctx = self._build_egress_context()
        # Same lifetime rule as policy_ctx: build ONCE, thread through every
        # tool so the lead agent and pooled subagents resolve the same
        # library documents. Without this, every fetch call on a library
        # doc URL is rejected by the egress policy as ``unsupported_scheme``
        # (A3 — 26 of 26 "Reading the page" milestones produced no content
        # in the f3045c5b run).
        library_resolver = self._build_library_resolver()
        primary_search_description = NEUTRAL_PRIMARY_SEARCH_DESCRIPTION

        # Web search is omitted in the collection-only ablation. This is a
        # tool-surface constraint, not a natural-language instruction.
        web_search_enabled = (
            self.search is not None and self.routing_mode != "collection_only"
        )
        if web_search_enabled:
            try:
                from ...web_search_engines.search_engines_config import (
                    search_config,
                )

                primary_source_config = search_config(
                    settings_snapshot=self.settings_snapshot
                ).get(self._search_engine_name)
                primary_source_type = PrimarySourceType.SEARCH
                primary_engine_classification: EngineClassification | None = (
                    None
                )
                if self._search_engine_name == "library":
                    primary_source_type = PrimarySourceType.LIBRARY
                if self._search_engine_name.startswith("collection_"):
                    primary_source_type = PrimarySourceType.COLLECTION
                if (
                    primary_source_config is not None
                    and primary_source_config.get("is_retriever") is True
                ):
                    from ...web_search_engines.retriever_registry import (
                        retriever_registry,
                    )

                    primary_source_type = PrimarySourceType.RETRIEVER
                    retriever_metadata = retriever_registry.get_metadata(
                        self._search_engine_name
                    )
                    retriever_is_local = (
                        retriever_metadata.get("is_local")
                        if retriever_metadata is not None
                        else None
                    )
                    primary_engine_classification = EngineClassification(
                        is_public=(
                            not retriever_is_local
                            if isinstance(retriever_is_local, bool)
                            else None
                        ),
                        is_local=(
                            retriever_is_local
                            if isinstance(retriever_is_local, bool)
                            else None
                        ),
                    )
                elif policy_ctx is not None:
                    primary_engine_classification = classify_engine(
                        self._search_engine_name,
                        policy_ctx,
                        settings_snapshot=self.settings_snapshot,
                        metadata=primary_source_config,
                    )
                if primary_engine_classification is not None:
                    primary_source_classification = classify_primary_source(
                        primary_source_type,
                        primary_engine_classification,
                    )
                    primary_search_description = (
                        format_primary_search_description(
                            primary_source_classification
                        )
                    )
            except Exception:
                logger.debug(
                    "Could not resolve primary search metadata; using neutral "
                    "tool description"
                )
            tools.append(
                _make_web_search_tool(
                    self._search_engine_name,
                    self.model,
                    self.settings_snapshot,
                    self.collector,
                    programmatic_mode=self.programmatic_mode,
                    description=primary_search_description,
                    fixed_candidates=self.candidate_pool or None,
                )
            )

        # Content fetcher (returns None when fetch_mode == 'disabled')
        fetch = build_fetch_tool(
            self.fetch_mode,
            self.collector,
            model=self.model,
            overall_query=overall_query,
            settings_snapshot=self.settings_snapshot,
            egress_context=policy_ctx,
            library_resolver=library_resolver,
            public_fetch_fallback=self.public_fetch_fallback,
            require_observed_urls=(self.research_profile == "hybrid"),
            controlled_failure=self.showcase_fetch_failure_injection,
        )
        if fetch is not None:
            tools.append(fetch)

        # Specialized search engines (pre-filtered by egress policy).
        #
        # This is the core fix for the original LangGraph silent-expansion
        # complaint. The factory PEP catches engines at instantiation time,
        # but that's a runtime check — the LLM still SEES the forbidden
        # tool names in the schema and the latency of a denied tool call
        # leaks policy state. Filtering the tool list HERE means the
        # forbidden tools never reach create_agent(), and the LLM never
        # learns they exist.
        specialized_search_enabled = (
            not self.primary_only_tools and self.routing_mode != "web_only"
        )
        if specialized_search_enabled:
            tools.extend(
                _load_specialized_engine_tools(
                    # Skip the engine web_search above already wraps — the same
                    # name it was built from, and only when it was built at all,
                    # else the primary engine would become unreachable.
                    (self._search_engine_name if web_search_enabled else None),
                    self.model,
                    self.settings_snapshot,
                    self.collector,
                    programmatic_mode=self.programmatic_mode,
                    egress_context=policy_ctx,
                    engine_allowlist=self.search_engine_allowlist,
                    query_rewrite_enabled=self.query_rewrite_enabled,
                )
            )

        # Subagent research tool — only when the toolbox already holds at
        # least one real research tool. Subagents are gated on the same
        # search/fetch/egress state as the lead, so with nothing else here
        # they'd have nothing either: a research_subtopic-only agent would
        # fan out tool-less subagents whose un-grounded text reads as
        # findings. Dropping it lets the empty-toolbox error below fire.
        if self.include_sub_research and tools:
            tools.append(
                _make_research_subtopic_tool(
                    self._search_engine_name,
                    self.model,
                    self.settings_snapshot,
                    self.collector,
                    self.max_sub_iterations,
                    search_enabled=web_search_enabled,
                    progress_callback=self.progress_callback,
                    programmatic_mode=self.programmatic_mode,
                    fetch_mode=self.fetch_mode,
                    overall_query=overall_query,
                    egress_context=policy_ctx,
                    max_subagent_workers=self.max_subagent_workers,
                    library_resolver=library_resolver,
                    web_search_description=primary_search_description,
                    search_engine_allowlist=(
                        set()
                        if self.primary_only_tools
                        else self.search_engine_allowlist
                    ),
                    query_rewrite_enabled=self.query_rewrite_enabled,
                    include_specialized_tools=specialized_search_enabled,
                    public_fetch_fallback=self.public_fetch_fallback,
                    require_observed_fetch_urls=(
                        self.research_profile == "hybrid"
                    ),
                )
            )

        if self.frozen_observation_session is not None:
            tools = _wrap_tools_with_frozen_observations(
                tools,
                collector=self.collector,
                session=self.frozen_observation_session,
            )
        return tools

    # -- Main entry point ---------------------------------------------------

    @staticmethod
    def _model_identifier(model: Any) -> str:
        for attribute in ("model_name", "model"):
            candidate = getattr(model, attribute, None)
            if isinstance(candidate, str) and candidate:
                return candidate
        return type(model).__name__

    def _start_trace(self, query: str):
        if not self.trace_output_dir:
            return None

        from local_deep_research.agent_harness import TraceRecorder

        recorder = TraceRecorder(
            runtime="local-deep-research",
            model=self._model_identifier(self.model),
            strategy="langgraph-agent",
            metadata={
                "fetch_mode": self.fetch_mode,
                "public_fetch_fallback": self.public_fetch_fallback,
                "task_contract_fetch_fallback": self.contract_fetch_fallback,
                "programmatic_mode": self.programmatic_mode,
                "max_model_calls": self.max_model_calls,
                "max_tool_calls": self.max_tool_calls,
                "max_tool_calls_per_batch": self.max_tool_calls_per_batch,
                "max_fetch_calls": self.max_fetch_calls,
                "planner_budget_protocol": (
                    PLANNER_BUDGET_PROTOCOL
                    if self.max_model_calls is not None
                    or self.max_tool_calls is not None
                    else None
                ),
                "synthesis_model": self._model_identifier(self.synthesis_model),
                "synthesis_evidence_only": self.evidence_only_synthesis,
                "primary_only_tools": self.primary_only_tools,
                "search_engine_allowlist": sorted(self.search_engine_allowlist)
                if self.search_engine_allowlist is not None
                else None,
                "research_profile": self.research_profile,
                "routing_mode": self.routing_mode,
                "routing_decision": self._routing_metadata(),
                "planner_uncertainty_state_protocol": (
                    PLANNER_UNCERTAINTY_STATE_PROTOCOL
                    if self.research_profile == "hybrid"
                    else None
                ),
                "controlled_fetch_failure_injection": (
                    dict(self.showcase_fetch_failure_injection)
                    if self.showcase_fetch_failure_injection is not None
                    else None
                ),
                "planner_prompt_variant": self.planner_prompt_variant,
                "planner_uncertainty_state_enabled": (
                    self.planner_uncertainty_state_enabled
                ),
                "planner_state_arm": self.planner_state_arm,
                "planner_state_prompt_version": (
                    "planner-state/v6-budget-2de298b"
                    if self.planner_state_arm == PLANNER_STATE_ARM_LEGACY_COMPACT
                    else "planner-state/sprint2-expanded-v1"
                ),
                "planner_stop_guard_enabled": self.planner_stop_guard_enabled,
                "static_official_domain": self.static_official_domain,
                "candidate_pool_protocol": (
                    "candidate-pool/fixed-v1" if self.candidate_pool else None
                ),
                "candidate_pool_size": len(self.candidate_pool),
                "candidate_pool_digest": self.candidate_pool_digest,
                "max_fetch_calls_per_planner_turn": (
                    self.max_fetch_calls_per_planner_turn
                ),
                "evidence_policy_mode": self.evidence_policy_mode.value,
                "collection_query_normalizer_enabled": bool(
                    self.query_rewrite_enabled
                    and os.getenv("LDR_COLLECTION_QUERY_NORMALIZER_ENDPOINT")
                    and os.getenv("LDR_COLLECTION_QUERY_NORMALIZER_MODEL")
                ),
                "observation_mode": (
                    "frozen_strict"
                    if self.frozen_observation_session is not None
                    else "live"
                ),
                "frozen_observation_cache": self.frozen_observation_cache_path,
            },
        )
        return recorder

    def _routing_metadata(self) -> dict[str, Any]:
        """Return route metadata, including for lightweight unit-test stubs."""
        decision = getattr(self, "routing_decision", None)
        if decision is not None:
            return decision.metadata()
        return {
            "requested_mode": getattr(self, "routing_mode", "unknown"),
            "selected_mode": getattr(self, "routing_mode", "unknown"),
            "reason": "legacy_or_test_stub",
            "official_domain": getattr(self, "static_official_domain", None),
        }

    def _run_static_dual_baseline(
        self,
        query: str,
        *,
        tools: list,
        trace_recorder,
        trace_started_at: float,
        nr_of_links: int,
    ) -> Dict[str, Any]:
        """Run one Web and one Collection search before the shared writer.

        This is an executable baseline: it does not ask the Planner to obey a
        static instruction. Search tools, the collector, and citation writer
        are exactly the same components used by the adaptive Hybrid condition.
        """
        from local_deep_research.agent_harness import RunOutcome

        web_tool = next(
            (
                item
                for item in tools
                if getattr(item, "name", "") == "web_search"
            ),
            None,
        )
        collection_tool = next(
            (
                item
                for item in tools
                if getattr(item, "name", "") == "search_library"
                or str(getattr(item, "name", "")).startswith(
                    "search_collection_"
                )
            ),
            None,
        )
        if web_tool is None or collection_tool is None:
            error = RuntimeError(
                "static_dual requires one web_search and one collection tool"
            )
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=error,
                metadata={"routing_mode": self.routing_mode},
            )
            return self._error_result(_scrub_tool_error(str(error)))

        if trace_recorder is not None:
            trace_recorder.record_message(
                "system",
                "Static dual retrieval baseline: execute one Web search and "
                "one Collection search before citation synthesis.",
            )
            trace_recorder.record_message("user", query)

        agent_messages = []
        try:
            for index, selected_tool in enumerate(
                (web_tool, collection_tool), start=1
            ):
                tool_name = str(selected_tool.name)
                call_id = f"static-dual-{index}"
                proposal = SimpleNamespace(
                    content="",
                    tool_calls=[
                        {
                            "id": call_id,
                            "name": tool_name,
                            "args": {"query": query},
                        }
                    ],
                )
                agent_messages.append(proposal)
                if trace_recorder is not None:
                    trace_recorder.record_assistant_message(proposal)
                self._update_progress(
                    f"Static dual retrieval: "
                    f"{self._display_tool_name(tool_name)}",
                    15 + index * 20,
                    {
                        "phase": "static_dual_tool_call",
                        "tool": tool_name,
                        "iteration": index,
                    },
                )
                observation = str(selected_tool.invoke({"query": query}))
                tool_message = SimpleNamespace(
                    tool_call_id=call_id,
                    name=tool_name,
                    content=observation,
                    status=None,
                )
                if trace_recorder is not None:
                    trace_recorder.record_tool_message(tool_message)

            final_content = self._synthesize_from_collector(query)
            result = self._finalize(
                query,
                final_content,
                2,
                nr_of_links,
                agent_messages,
            )
        except Exception as exc:
            logger.exception("Static dual retrieval baseline failed")
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=exc,
                metadata={"routing_mode": self.routing_mode},
            )
            return self._error_result(self._format_agent_error(exc))

        findings = result.get("findings") or []
        exported_answer = (
            findings[0].get("content", final_content)
            if findings and isinstance(findings[0], dict)
            else final_content
        )
        self._finish_trace(
            trace_recorder,
            outcome=RunOutcome.SUCCESS,
            started_at=trace_started_at,
            final_answer=exported_answer,
            metadata={
                "routing_mode": self.routing_mode,
                "routing_decision": self._routing_metadata(),
                "iterations": 2,
                "scheduled_tool_calls": 2,
                "final_report_status": result.get("final_report_status"),
                "stop_reason": "static_dual_complete",
                "evidence_policy": None,
            },
        )
        return result

    def _first_static_official_result(
        self, *, collection_urls: set[str]
    ) -> str | None:
        """Return the first newly observed Web result from the frozen domain.

        ``static_dual_fetch`` deliberately has no model-controlled candidate
        selection: it uses the original user request plus ``site:<domain>``,
        then fetches the first matching result in the search engine's returned
        order.  Excluding URLs already observed through Collection keeps the
        source routes separate even if a Collection record happens to contain
        the same link.
        """
        assert self.static_official_domain is not None
        for result in self.collector.results:
            url = str(result.get("link") or result.get("url") or "").strip()
            if not url or url in collection_urls:
                continue
            try:
                hostname = (urlparse(url).hostname or "").lower()
            except ValueError:
                continue
            if hostname == self.static_official_domain or hostname.endswith(
                f".{self.static_official_domain}"
            ):
                return url
        return None

    @staticmethod
    def _static_search_query_seed(query: str) -> str:
        """Extract the leading topic span from the frozen Chinese task form.

        The static baseline is deliberately not allowed to call a query-rewrite
        model.  The E2E contract phrases each task as ``围绕 <topic>，...``;
        extracting that explicit user-provided topic prevents instructional
        boilerplate from becoming the Web search while preserving a purely
        deterministic mapping from the same user request.  Other request
        shapes fall back to the complete query.
        """
        match = re.match(r"^\s*(?:围绕|关于)\s+(.+?)[，,]", query)
        if match and match.group(1).strip():
            return match.group(1).strip()
        return query

    def _run_static_dual_fetch_baseline(
        self,
        query: str,
        *,
        tools: list,
        trace_recorder,
        trace_started_at: float,
        nr_of_links: int,
    ) -> Dict[str, Any]:
        """Run Collection → domain-constrained Web → one Fetch, then write.

        This is the strong deterministic fallback used in the E2E report
        ablation.  It never calls the Planner: the only Web query is
        ``<topic parsed from user query> site:<frozen official domain>`` and
        the first matching observed Web result is fetched exactly once.  The
        tools, provenance
        collector, Fetch guard, and evidence-only Writer are shared with the
        adaptive Hybrid arms.
        """
        from local_deep_research.agent_harness import RunOutcome

        web_tool = next(
            (item for item in tools if getattr(item, "name", "") == "web_search"),
            None,
        )
        collection_tool = next(
            (
                item
                for item in tools
                if getattr(item, "name", "") == "search_library"
                or str(getattr(item, "name", "")).startswith(
                    "search_collection_"
                )
            ),
            None,
        )
        fetch_tool = next(
            (
                item
                for item in tools
                if getattr(item, "name", "") == "fetch_content"
            ),
            None,
        )
        if web_tool is None or collection_tool is None or fetch_tool is None:
            error = RuntimeError(
                "static_dual_fetch requires Collection, Web, and Fetch tools"
            )
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=error,
                metadata={"routing_mode": self.routing_mode},
            )
            return self._error_result(_scrub_tool_error(str(error)))

        assert self.static_official_domain is not None
        web_query = (
            f"{self._static_search_query_seed(query)} "
            f"site:{self.static_official_domain}"
        )
        if trace_recorder is not None:
            trace_recorder.record_message(
                "system",
                "Static dual-fetch baseline: execute Collection search, a "
                "domain-constrained Web search, then fetch the first matching "
                "official result before citation synthesis.",
            )
            trace_recorder.record_message("user", query)

        agent_messages = []
        scheduled_tool_calls = 0

        def invoke_static_tool(selected_tool, arguments: dict[str, str]) -> None:
            nonlocal scheduled_tool_calls
            scheduled_tool_calls += 1
            tool_name = str(selected_tool.name)
            call_id = f"static-dual-fetch-{scheduled_tool_calls}"
            proposal = SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": tool_name,
                        "args": arguments,
                    }
                ],
            )
            agent_messages.append(proposal)
            if trace_recorder is not None:
                trace_recorder.record_assistant_message(proposal)
            self._update_progress(
                "Static dual-fetch: "
                f"{self._display_tool_name(tool_name)}",
                15 + scheduled_tool_calls * 20,
                {
                    "phase": "static_dual_fetch_tool_call",
                    "tool": tool_name,
                    "iteration": scheduled_tool_calls,
                },
            )
            observation = str(selected_tool.invoke(arguments))
            tool_message = SimpleNamespace(
                tool_call_id=call_id,
                name=tool_name,
                content=observation,
                status=None,
            )
            if trace_recorder is not None:
                trace_recorder.record_tool_message(tool_message)

        try:
            invoke_static_tool(collection_tool, {"query": query})
            collection_urls = {
                str(result.get("link") or result.get("url") or "").strip()
                for result in self.collector.results
            }
            invoke_static_tool(web_tool, {"query": web_query})
            candidate_url = self._first_static_official_result(
                collection_urls=collection_urls
            )
            if candidate_url:
                invoke_static_tool(fetch_tool, {"url": candidate_url})

            final_content = self._synthesize_from_collector(query)
            result = self._finalize(
                query,
                final_content,
                scheduled_tool_calls,
                nr_of_links,
                agent_messages,
            )
        except Exception as exc:
            logger.exception("Static dual-fetch baseline failed")
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=exc,
                metadata={"routing_mode": self.routing_mode},
            )
            return self._error_result(self._format_agent_error(exc))

        findings = result.get("findings") or []
        exported_answer = (
            findings[0].get("content", final_content)
            if findings and isinstance(findings[0], dict)
            else final_content
        )
        self._finish_trace(
            trace_recorder,
            outcome=RunOutcome.SUCCESS,
            started_at=trace_started_at,
            final_answer=exported_answer,
            metadata={
                "routing_mode": self.routing_mode,
                "routing_decision": self._routing_metadata(),
                "iterations": scheduled_tool_calls,
                "scheduled_tool_calls": scheduled_tool_calls,
                "final_report_status": result.get("final_report_status"),
                "stop_reason": "static_dual_fetch_complete",
                "static_official_domain": self.static_official_domain,
                "static_web_query": web_query,
                "static_fetch_candidate_found": bool(candidate_url),
                "evidence_policy": None,
            },
        )
        return result

    @staticmethod
    def _static_candidate_score(result: dict[str, Any]) -> tuple[int, int]:
        """A public, task-agnostic heuristic for the fixed Top-k baseline."""
        url = str(result.get("link") or result.get("url") or "")
        title = str(result.get("title") or "").lower()
        snippet = str(result.get("snippet") or "").lower()
        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            host = ""
        score = 0
        if host.endswith(".gov") or host in {"who.int", "europa.eu", "un.org"}:
            score += 4
        if any(token in host for token in ("gov", "who", "un", "edu")):
            score += 1
        if any(token in title for token in ("guidance", "recommend", "faq", "policy")):
            score += 1
        if any(token in title + " " + snippet for token in ("news", "press release", "blog")):
            score -= 2
        return score, -int(result.get("index") or 0)

    def _generic_web_candidates(self, collection_urls: set[str]) -> list[dict[str, Any]]:
        """Return observed public candidates without hidden-domain knowledge."""
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in self.collector.results:
            url = str(result.get("link") or result.get("url") or "").strip()
            source_engine = str(result.get("source_engine") or "").lower()
            if (
                not url
                or url in collection_urls
                or url in seen
                or source_engine == "fetch"
                or source_engine.startswith("collection_")
            ):
                continue
            seen.add(url)
            candidates.append(dict(result))
        return candidates

    def _candidate_evaluation_candidates(
        self, collection_urls: set[str]
    ) -> list[dict[str, Any]]:
        """Return the fixed Candidate-v2 pool when the contract provides one.

        Static, one-shot, and Adaptive all receive the same public metadata.
        The direct baselines use this helper rather than collector order so a
        Collection result or repeated fixed-pool search cannot alter their
        selection universe.
        """
        if self.candidate_pool:
            return [
                {
                    "index": str(index + 1),
                    "link": candidate["url"],
                    "url": candidate["url"],
                    "title": candidate["title"],
                    "snippet": candidate["snippet"],
                    "source_engine": "candidate_pool",
                }
                for index, candidate in enumerate(self.candidate_pool)
            ]
        return self._generic_web_candidates(collection_urls)

    def _run_static_candidate_workflow(
        self,
        query: str,
        *,
        tools: list,
        trace_recorder,
        trace_started_at: float,
        nr_of_links: int,
        fetch_limit: int,
        heuristic: bool,
    ) -> Dict[str, Any]:
        """Execute a non-adaptive Web/Collection/Fetch candidate baseline.

        ``static_top1`` follows search order. ``static_topk`` applies a small,
        public ranking heuristic and fetches a fixed three candidates. Neither
        route observes an outcome to choose the next action, reformulates a
        query, or evaluates evidence sufficiency.
        """
        from local_deep_research.agent_harness import RunOutcome

        web_tool = next(
            (item for item in tools if getattr(item, "name", "") == "web_search"),
            None,
        )
        collection_tool = next(
            (
                item
                for item in tools
                if getattr(item, "name", "") == "search_library"
                or str(getattr(item, "name", "")).startswith("search_collection_")
            ),
            None,
        )
        fetch_tool = next(
            (
                item
                for item in tools
                if getattr(item, "name", "") == "fetch_content"
            ),
            None,
        )
        if web_tool is None or collection_tool is None or fetch_tool is None:
            error = RuntimeError(
                "static candidate workflow requires Collection, Web, and Fetch tools"
            )
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=error,
                metadata={"routing_mode": self.routing_mode},
            )
            return self._error_result(_scrub_tool_error(str(error)))

        policy = (
            "ranked_fixed_fetch_2"
            if self.routing_mode == "static_ranked_fetch_2"
            else "heuristic_top_k"
            if heuristic
            else "search_rank_top_1"
        )
        if trace_recorder is not None:
            trace_recorder.record_message(
                "system",
                "Static candidate baseline: run one Collection search and one "
                f"Web search, then fetch a fixed candidate set ({policy}).",
            )
            trace_recorder.record_message("user", query)

        static_contract = getattr(self, "task_contract", {})
        if not isinstance(static_contract, dict):
            static_contract = {}
        collection_query = str(
            static_contract.get("static_collection_query") or query
        ).strip()
        web_query = str(
            static_contract.get("static_web_query") or query
        ).strip()

        agent_messages: list[Any] = []
        scheduled_tool_calls = 0
        selected_urls: list[str] = []

        def invoke_static_tool(selected_tool, arguments: dict[str, str]) -> None:
            nonlocal scheduled_tool_calls
            scheduled_tool_calls += 1
            tool_name = str(selected_tool.name)
            call_id = f"static-candidate-{scheduled_tool_calls}"
            proposal = SimpleNamespace(
                content="",
                tool_calls=[{"id": call_id, "name": tool_name, "args": arguments}],
            )
            agent_messages.append(proposal)
            self.uncertainty_tracker.observe_tool_calls(proposal.tool_calls)
            if trace_recorder is not None:
                trace_recorder.record_assistant_message(proposal)
            self._update_progress(
                f"Static candidate workflow: {self._display_tool_name(tool_name)}",
                min(85, 15 + scheduled_tool_calls * 15),
                {
                    "phase": "static_candidate_tool_call",
                    "tool": tool_name,
                    "iteration": scheduled_tool_calls,
                },
            )
            observation = str(selected_tool.invoke(arguments))
            tool_message = SimpleNamespace(
                tool_call_id=call_id,
                name=tool_name,
                content=observation,
                status=None,
            )
            self.uncertainty_tracker.observe_tool_observation(tool_message)
            self.uncertainty_tracker.sync_results(self.collector.results)
            if trace_recorder is not None:
                trace_recorder.record_tool_message(tool_message)

        try:
            invoke_static_tool(collection_tool, {"query": collection_query})
            collection_urls = {
                str(result.get("link") or result.get("url") or "").strip()
                for result in self.collector.results
            }
            invoke_static_tool(web_tool, {"query": web_query})
            candidates = self._candidate_evaluation_candidates(collection_urls)
            if heuristic:
                candidates.sort(key=self._static_candidate_score, reverse=True)
            for candidate in candidates[:fetch_limit]:
                url = str(candidate.get("link") or candidate.get("url") or "").strip()
                if not url:
                    continue
                selected_urls.append(url)
                fetch_arguments = {"url": url}
                if self.fetch_mode in {"summary_focus", "summary_focus_query"}:
                    fetch_arguments["focus"] = query
                invoke_static_tool(fetch_tool, fetch_arguments)

            final_content = self._synthesize_from_collector(query)
            result = self._finalize(
                query,
                final_content,
                scheduled_tool_calls,
                nr_of_links,
                agent_messages,
            )
        except Exception as exc:
            logger.exception("Static candidate workflow failed")
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=exc,
                metadata={"routing_mode": self.routing_mode},
            )
            return self._error_result(self._format_agent_error(exc))

        findings = result.get("findings") or []
        exported_answer = (
            findings[0].get("content", final_content)
            if findings and isinstance(findings[0], dict)
            else final_content
        )
        self._finish_trace(
            trace_recorder,
            outcome=RunOutcome.SUCCESS,
            started_at=trace_started_at,
            final_answer=exported_answer,
            metadata={
                "routing_mode": self.routing_mode,
                "routing_decision": self._routing_metadata(),
                "iterations": scheduled_tool_calls,
                "scheduled_tool_calls": scheduled_tool_calls,
                "final_report_status": result.get("final_report_status"),
                "stop_reason": "static_candidate_workflow_complete",
                "static_candidate_policy": policy,
                "static_fetch_limit": fetch_limit,
                "static_candidate_urls": selected_urls,
                "static_collection_query": collection_query,
                "static_web_query": web_query,
                "uncertainty_state": self.uncertainty_tracker.trace_summary(),
                "evidence_policy": None,
            },
        )
        return result

    def _run_one_shot_candidate_selector(
        self,
        query: str,
        *,
        tools: list,
        trace_recorder,
        trace_started_at: float,
        nr_of_links: int,
        fetch_limit: int,
    ) -> Dict[str, Any]:
        """Select up to two observed candidates in one model call, then fetch.

        The selector sees the exact fixed pool given to every Candidate-v2
        arm.  Both URLs are committed before the first Fetch starts: no
        observation can change the second choice.
        """
        from local_deep_research.agent_harness import RunOutcome

        web_tool = next(
            (item for item in tools if getattr(item, "name", "") == "web_search"),
            None,
        )
        collection_tool = next(
            (
                item
                for item in tools
                if getattr(item, "name", "") == "search_library"
                or str(getattr(item, "name", "")).startswith("search_collection_")
            ),
            None,
        )
        fetch_tool = next(
            (item for item in tools if getattr(item, "name", "") == "fetch_content"),
            None,
        )
        if web_tool is None or collection_tool is None or fetch_tool is None:
            error = RuntimeError(
                "one-shot candidate selector requires Collection, Web, and Fetch tools"
            )
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=error,
                metadata={"routing_mode": self.routing_mode},
            )
            return self._error_result(_scrub_tool_error(str(error)))

        if trace_recorder is not None:
            trace_recorder.record_message(
                "system",
                "One-shot candidate selector: inspect the shared candidate pool once, "
                "choose up to two candidates, then fetch the committed choices.",
            )
            trace_recorder.record_message("user", query)

        static_contract = getattr(self, "task_contract", {})
        if not isinstance(static_contract, dict):
            static_contract = {}
        collection_query = str(
            static_contract.get("static_collection_query") or query
        ).strip()
        web_query = str(static_contract.get("static_web_query") or query).strip()
        agent_messages: list[Any] = []
        scheduled_tool_calls = 0
        selected_urls: list[str] = []
        selector_status = "not_run"
        selector_response: Any = None

        def invoke_tool(selected_tool, arguments: dict[str, str]) -> None:
            nonlocal scheduled_tool_calls
            scheduled_tool_calls += 1
            tool_name = str(selected_tool.name)
            call_id = f"one-shot-candidate-{scheduled_tool_calls}"
            proposal = SimpleNamespace(
                content="",
                tool_calls=[{"id": call_id, "name": tool_name, "args": arguments}],
            )
            agent_messages.append(proposal)
            self.uncertainty_tracker.observe_tool_calls(proposal.tool_calls)
            if trace_recorder is not None:
                trace_recorder.record_assistant_message(proposal)
            observation = str(selected_tool.invoke(arguments))
            tool_message = SimpleNamespace(
                tool_call_id=call_id,
                name=tool_name,
                content=observation,
                status=None,
            )
            self.uncertainty_tracker.observe_tool_observation(tool_message)
            self.uncertainty_tracker.sync_results(self.collector.results)
            if trace_recorder is not None:
                trace_recorder.record_tool_message(tool_message)

        try:
            invoke_tool(collection_tool, {"query": collection_query})
            collection_urls = {
                str(result.get("link") or result.get("url") or "").strip()
                for result in self.collector.results
            }
            invoke_tool(web_tool, {"query": web_query})
            candidates = self._candidate_evaluation_candidates(collection_urls)
            selector_view = [
                {
                    "id": f"C{index + 1}",
                    "title": str(candidate.get("title") or ""),
                    "url": str(candidate.get("link") or candidate.get("url") or ""),
                    "snippet": str(candidate.get("snippet") or ""),
                }
                for index, candidate in enumerate(candidates)
            ]
            selector_response = self.model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a one-shot candidate reranker. Select up to two "
                            "candidate IDs that are most likely to satisfy the user's "
                            "requested public-page role. You will receive no Fetch "
                            "observations before either choice is committed. Use only the "
                            "provided candidates. Return exactly JSON with one key "
                            "selected_candidate_ids, whose value is an array of up to two "
                            "IDs such as [\"C2\", \"C4\"]."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            {"user_request": query, "candidates": selector_view},
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            agent_messages.append(selector_response)
            if trace_recorder is not None:
                trace_recorder.record_assistant_message(selector_response)
            selected_urls = _one_shot_candidate_urls(
                selector_response, candidates, limit=fetch_limit
            )
            selector_status = "selected" if selected_urls else "invalid_selector_output"
            for url in selected_urls:
                arguments = {"url": url}
                if self.fetch_mode in {"summary_focus", "summary_focus_query"}:
                    arguments["focus"] = query
                invoke_tool(fetch_tool, arguments)

            final_content = self._synthesize_from_collector(query)
            result = self._finalize(
                query,
                final_content,
                scheduled_tool_calls,
                nr_of_links,
                agent_messages,
            )
        except Exception as exc:
            logger.exception("One-shot candidate selector failed")
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=exc,
                metadata={"routing_mode": self.routing_mode},
            )
            return self._error_result(self._format_agent_error(exc))

        findings = result.get("findings") or []
        exported_answer = (
            findings[0].get("content", final_content)
            if findings and isinstance(findings[0], dict)
            else final_content
        )
        self._finish_trace(
            trace_recorder,
            outcome=RunOutcome.SUCCESS,
            started_at=trace_started_at,
            final_answer=exported_answer,
            metadata={
                "routing_mode": self.routing_mode,
                "routing_decision": self._routing_metadata(),
                "iterations": scheduled_tool_calls,
                "scheduled_tool_calls": scheduled_tool_calls,
                "final_report_status": result.get("final_report_status"),
                "stop_reason": "one_shot_candidate_selector_complete",
                "one_shot_selector": {
                    "model_calls": 1,
                    "candidate_pool_size": len(candidates),
                    "selected_urls": selected_urls,
                    "status": selector_status,
                },
                "planner_policy_success": None,
                "runtime_assisted_success": None,
                "system_success": None,
                "uncertainty_state": self.uncertainty_tracker.trace_summary(),
                "evidence_policy": None,
            },
        )
        return result

    def _emit_evidence_policy_progress(
        self,
        decision: EvidencePolicyDecision | None,
        *,
        iteration: int,
        progress: int,
    ) -> None:
        """Expose non-allow shadow decisions without steering the agent."""
        if decision is None or decision.verdict == EvidencePolicyVerdict.ALLOW:
            return
        outcome = (
            "would block"
            if decision.verdict == EvidencePolicyVerdict.BLOCK
            else "notes"
        )
        self._update_progress(
            f"Evidence policy {outcome}: {decision.reason}",
            min(88, progress),
            {
                "phase": "evidence_policy",
                "iteration": iteration,
                **decision.trace_metadata(),
            },
        )

    def _finish_trace(
        self,
        recorder,
        *,
        outcome,
        started_at: float,
        final_answer: str | None = None,
        error: BaseException | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if recorder is None:
            return

        from local_deep_research.agent_harness import JsonlTraceStore

        try:
            if error is not None:
                recorder.record_error(error)
            if final_answer is not None:
                recorder.record_final_answer(final_answer)
            frozen_stats = (
                self.frozen_observation_session.stats
                if self.frozen_observation_session is not None
                else None
            )
            recorder.finish(
                outcome,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                metadata={
                    **(metadata or {}),
                    "observation_mode": (
                        "frozen_strict"
                        if self.frozen_observation_session is not None
                        else "live"
                    ),
                    "frozen_observation": frozen_stats,
                },
            )
            output = Path(self.trace_output_dir) / f"{recorder.run_id}.jsonl"
            JsonlTraceStore.write(output, recorder.events)
            logger.info(f"Agent trajectory exported: {output.name}")
        except Exception:
            # Observability must never turn a usable research result into a
            # failed run. The full exception stays in the local server log.
            logger.exception("Failed to export agent trajectory")

    def analyze_topic(self, query: str) -> Dict[str, Any]:
        from langchain.agents import create_agent
        from local_deep_research.agent_harness import RunOutcome

        logger.info(f"LangGraph agent research: {query[:100]}")
        trace_started_at = time.perf_counter()
        trace_recorder = self._start_trace(query)
        trace_outcome = RunOutcome.SUCCESS

        # Reset collector for fresh subsection call (detailed report mode)
        self.collector.reset()
        self.uncertainty_tracker.reset(query)
        nr_of_links = len(self.all_links_of_system)

        self._update_progress(
            f'Starting agent research: "{query[:80]}"',
            5,
            {"phase": "init", "type": "milestone", "query": query[:100]},
        )
        self.check_termination(CHECK_CONTEXT_ENTRY)

        # Build tools (overall_query feeds summary-mode fetch tools)
        tools = self._build_tools(overall_query=query)
        if not tools:
            if trace_recorder is not None:
                trace_recorder.record_message("user", query)
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=RuntimeError("No tools available"),
            )
            return self._error_result("No tools available")
        # Stash tool names for the per-step heartbeat — gives the user
        # concrete info ("from the web (SearXNG), PubMed, …") instead of
        # a vague spinner while the LLM picks its next move. The raw ids
        # are mapped to friendly names at render time via
        # ``_display_tool_name``.
        self._tool_names = [getattr(t, "name", "?") for t in tools]

        if self.routing_mode == "static_dual":
            return self._run_static_dual_baseline(
                query,
                tools=tools,
                trace_recorder=trace_recorder,
                trace_started_at=trace_started_at,
                nr_of_links=nr_of_links,
            )
        if self.routing_mode == "static_dual_fetch":
            return self._run_static_dual_fetch_baseline(
                query,
                tools=tools,
                trace_recorder=trace_recorder,
                trace_started_at=trace_started_at,
                nr_of_links=nr_of_links,
            )
        if self.routing_mode in {
            "static_top1",
            "static_topk",
            "static_ranked_fetch_2",
        }:
            default_fetch_limit = (
                1
                if self.routing_mode == "static_top1"
                else 2
                if self.routing_mode == "static_ranked_fetch_2"
                else 3
            )
            return self._run_static_candidate_workflow(
                query,
                tools=tools,
                trace_recorder=trace_recorder,
                trace_started_at=trace_started_at,
                nr_of_links=nr_of_links,
                fetch_limit=(
                    min(default_fetch_limit, self.max_fetch_calls)
                    if self.max_fetch_calls is not None
                    else default_fetch_limit
                ),
                heuristic=self.routing_mode in {"static_topk", "static_ranked_fetch_2"},
            )
        if self.routing_mode == "one_shot_llm_selector":
            return self._run_one_shot_candidate_selector(
                query,
                tools=tools,
                trace_recorder=trace_recorder,
                trace_started_at=trace_started_at,
                nr_of_links=nr_of_links,
                fetch_limit=(
                    min(2, self.max_fetch_calls)
                    if self.max_fetch_calls is not None
                    else 2
                ),
            )

        # Build system prompt — fetch_line wording mirrors the active mode
        # so the agent isn't told to use a tool that doesn't exist.
        current_date = datetime.now(UTC).strftime("%Y-%m-%d")
        if self.fetch_mode == "disabled":
            fetch_line = (
                "3. Rely on search snippets — full-page fetching is disabled "
                "for this run.\n"
            )
        elif self.fetch_mode in ("summary_focus", "summary_focus_query"):
            fetch_line = (
                "3. Use fetch_content(url, focus) when snippets aren't enough; "
                "always pass the specific question or claim you want answered "
                "as ``focus`` so the tool returns only the relevant facts.\n"
            )
        else:  # full
            fetch_line = "3. Use fetch_content to read full pages when snippets aren't enough.\n"
        # Build the policy addendum once so the system prompt can carry
        # explicit guidance to the LLM about which tools actually exist.
        # Closing the timing-leak attack requires both halves: the tool
        # list is pre-filtered above, AND the LLM is told what's
        # available, so it doesn't waste tokens probing for forbidden
        # engines and the latency of denial paths doesn't leak policy.
        policy_addendum = ""
        try:
            from local_deep_research.security.egress.policy import (
                EgressScope,
                PolicyDeniedError,
            )

            ctx = self._build_egress_context()
            if ctx is not None and ctx.scope == EgressScope.STRICT:
                policy_addendum = (
                    "\nRESTRICTED MODE: only the primary search tool is "
                    f"available ({ctx.primary_engine}). Do NOT reference or "
                    "attempt other search_* tools — they do not exist in "
                    "this session and will not work. Use web_search and "
                    "research_subtopic for everything.\n"
                )
            elif ctx is not None and ctx.scope == EgressScope.PRIVATE_ONLY:
                policy_addendum = (
                    "\nPRIVATE-ONLY MODE: public search engines (arxiv, "
                    "pubmed, brave, etc.) are not available in this "
                    "session. Use only local search tools.\n"
                )
            elif ctx is not None and ctx.scope == EgressScope.PUBLIC_ONLY:
                public_collection_tools = [
                    name
                    for name in self._tool_names
                    if name == "search_library"
                    or name.startswith("search_collection_")
                ]
                if public_collection_tools:
                    policy_addendum = (
                        "\nPUBLIC-ONLY MODE: private local sources are not "
                        "available. The collection tools present in this "
                        "session are explicitly marked public and ARE "
                        "available; use them when they contain relevant "
                        "curated evidence.\n"
                    )
                else:
                    policy_addendum = (
                        "\nPUBLIC-ONLY MODE: private local search tools are "
                        "not available in this session.\n"
                    )
        except PolicyDeniedError:
            # Corrupt/unknown scope must fail closed, never run unfiltered.
            # In practice _build_tools() (called above at the top of
            # analyze_topic) already raised this for the same snapshot, so
            # we never reach here with a bad scope — but re-raise rather
            # than swallow, so this stays correct if the call order changes.
            raise
        except Exception:
            logger.debug(
                "Could not derive policy addendum for system prompt — "
                "agent will see the unmodified prompt"
            )

        profile_addendum = ""
        if (
            self.research_profile == "hybrid"
            and self.routing_mode == "adaptive_hybrid"
        ):
            profile_addendum = (
                "\nHYBRID EVIDENCE MODE:\n"
                "- Search both the selected public web source and at least one "
                "available collection tool before answering, unless a side "
                "explicitly returns no relevant results.\n"
                "- Treat collection documents as curated project evidence and "
                "web pages as external corroboration; say which kind supports "
                "each important conclusion.\n"
                "- Prefer a few focused collection searches, then fetch the "
                "important documents. Do not repeatedly search for a document "
                "whose URL and citation are already known.\n"
                "- Fetch only an exact URL returned by a search tool (or its "
                "[N] citation marker); do not invent or reconstruct URLs.\n"
                "- When fetch_content is available, fetch at least one "
                "important collection document or public web page before "
                "answering. Do not add factual claims that are absent from "
                "the collected evidence. Stop researching once both internal "
                "and external evidence are sufficient.\n"
                "- Prefer a concise, decision-oriented final report. Obey an "
                "explicit user length limit, avoid repeating conclusions, and "
                "do not add a long source catalog beyond the cited sources.\n"
            )
            if self.planner_expanded_state_visible:
                profile_addendum += (
                    "- A dynamic uncertainty-state block lists only observed "
                    "requirements, candidates, and fetch outcomes. Do not STOP "
                    "while its stop blockers remain, and never present a search "
                    "snippet as fetched-page evidence.\n"
                )
            if self.planner_prompt_variant == "strong":
                profile_addendum += (
                    "PLANNER CONTROL CHECKLIST:\n"
                    "- Translate the request into evidence roles before calling "
                    "tools; do not assume the first result fills every role.\n"
                    "- Compare candidate title, snippet, and source role before "
                    "fetching. If a fetched page is empty, blocked, irrelevant, "
                    "or a landing page, select another observed candidate or "
                    "reformulate the query.\n"
                    "- For comparisons, acquire support for each requested side. "
                    "Before STOP, verify that every required evidence role has "
                    "fetched-page or collection support; otherwise keep the "
                    "remaining budget for the missing role.\n"
                )
        elif self.routing_mode == "collection_only":
            profile_addendum = (
                "\nCOLLECTION-ONLY ABLATION MODE:\n"
                "- Use only the available curated Collection search and local "
                "document fetch tools. Do not claim external corroboration.\n"
            )
        elif self.routing_mode == "web_only":
            profile_addendum = (
                "\nWEB-ONLY ABLATION MODE:\n"
                "- Use only the available public web search and page-fetch "
                "tools. Do not infer unavailable Collection evidence.\n"
            )

        if "research_subtopic" in self._tool_names:
            subtopic_line = (
                "2. For complex multi-faceted questions, use "
                "research_subtopic to investigate specific aspects in "
                f"parallel (pass 2-{MAX_SUBTOPICS} focused, non-overlapping "
                f"questions — at most {MAX_SUBTOPICS}).\n"
            )
        else:
            subtopic_line = (
                "2. Research directly with the available search and fetch "
                "tools; parallel subtopic delegation is disabled for this "
                "bounded run.\n"
            )
        initial_search_line = (
            "1. Start with an available Collection search tool for initial "
            "exploration.\n"
            if self.routing_mode == "collection_only"
            else "1. Start with web_search — it queries your selected primary "
            "source — for initial exploration.\n"
        )

        system_prompt = (
            f"You are a research assistant writing a research report. Today's date: {current_date}.\n"
            "This is NOT a chat conversation. Your only job is to research the "
            "given topic and produce a comprehensive, well-cited report.\n"
            "Do NOT ask clarifying questions, do NOT ask the user anything, "
            "do NOT offer to help further — just research and report.\n"
            "You MUST search the selected source before answering — never answer from memory alone.\n\n"
            "Strategy:\n"
            f"{initial_search_line}"
            f"{subtopic_line}"
            f"{fetch_line}"
            "4. When available, use specialized search_[engine] tools for domain-specific searches "
            "(search_arxiv for science, search_pubmed for medical, etc.).\n"
            "5. When you have enough information, provide a comprehensive answer "
            "citing sources as [1], [2], etc.\n"
            f"{policy_addendum}"
            f"{profile_addendum}"
        )
        if trace_recorder is not None:
            trace_recorder.record_message("system", system_prompt)
            trace_recorder.record_message("user", query)

        # Create agent — may fail if model doesn't support tool calling.
        # create_agent() calls model.bind_tools(); ProcessingLLMWrapper overrides
        # bind_tools to re-wrap the bound model, so the wrapper's <think>-tag
        # stripping survives the agent loop here (fix #4804). Other Runnable
        # transforms still escape the wrapper (with_config/bind/stream delegate
        # via __getattr__ unstripped; `|` raises TypeError, no __or__), but none
        # are on this create_agent path — see config/llm_config.py.
        planner_budget = (
            PlannerBudgetMiddleware(
                max_tool_calls=self.max_tool_calls,
                max_tool_calls_per_batch=self.max_tool_calls_per_batch,
                max_model_calls=self.max_model_calls,
                max_replans=self.max_budget_replans,
                max_fetch_calls=self.max_fetch_calls,
                max_fetch_calls_per_batch=self.max_fetch_calls_per_planner_turn,
                max_stop_replans=1,
                max_runtime_recovery_actions=DEFAULT_MAX_RUNTIME_RECOVERY_ACTIONS,
                supplemental_state_provider=(
                    self.uncertainty_tracker.render_prompt
                    if self.planner_expanded_state_visible
                    and self.research_profile == "hybrid"
                    and self.routing_mode == "adaptive_hybrid"
                    else None
                ),
                stop_guard_provider=(
                    self.uncertainty_tracker.review_stop_attempt
                    if self.planner_stop_guard_enabled
                    and self.planner_uncertainty_state_enabled
                    and self.research_profile == "hybrid"
                    and self.routing_mode == "adaptive_hybrid"
                    else None
                ),
                recovery_action_provider=(
                    self.uncertainty_tracker.recovery_action
                    if self.planner_stop_guard_enabled
                    and self.planner_uncertainty_state_enabled
                    and self.research_profile == "hybrid"
                    and self.routing_mode == "adaptive_hybrid"
                    else None
                ),
                allowed_fetch_urls=(
                    {candidate["url"] for candidate in self.candidate_pool}
                    if self.candidate_evaluation_mode == "candidate_selection_v2"
                    and self.candidate_pool
                    else None
                ),
                planner_state_arm=self.planner_state_arm,
            )
            if self.max_model_calls is not None
            or self.max_tool_calls is not None
            else None
        )
        try:
            agent_kwargs = {
                "model": self.model,
                "tools": tools,
                "system_prompt": system_prompt,
            }
            if planner_budget is not None:
                agent_kwargs["middleware"] = [planner_budget]
            agent = create_agent(**agent_kwargs)
        except Exception as exc:
            logger.exception("Failed to create LangGraph agent")
            self._finish_trace(
                trace_recorder,
                outcome=RunOutcome.ERROR,
                started_at=trace_started_at,
                error=exc,
            )
            return self._error_result(
                _scrub_tool_error(
                    f"Failed to create agent (model may not "
                    f"support tool calling): {exc}"
                )
            )

        # Stream agent execution
        effective_max = max(MIN_ITERATIONS, self.max_iterations)
        config = {"recursion_limit": effective_max * 2 + 1}
        iteration = 0
        final_content = ""
        agent_messages: list = []
        stopped_by_call_budget = False
        stopped_by_tool_call_budget = False
        stopped_by_batch_contract = False
        tool_call_budget_exhausted = False
        pending_tool_call_ids: set[str] = set()
        scheduled_tool_calls = 0
        seen_tool_call_signatures: set[tuple[str, str]] = set()
        stop_reason: str | None = None
        stop_agent_stream = False
        runtime_fallback_triggered = False
        evidence_policy = EvidencePolicyGuard(
            mode=self.evidence_policy_mode,
            require_fetch_before_stop=(
                self.research_profile == "hybrid"
                and self.fetch_mode != "disabled"
            ),
        )

        try:
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                config,
                stream_mode="updates",
            ):
                self.check_termination(CHECK_CONTEXT_AGENT_STREAM)

                if "agent" in chunk or "model" in chunk:
                    node_key = "agent" if "agent" in chunk else "model"
                    iteration += 1
                    progress = 10 + int((iteration / effective_max) * 75)
                    msgs = chunk[node_key].get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage):
                            content = msg.content or ""
                            tool_calls = getattr(msg, "tool_calls", [])
                            additional_kwargs = (
                                getattr(msg, "additional_kwargs", None) or {}
                            )
                            if additional_kwargs.get(FORCED_STOP_FLAG):
                                stopped_by_call_budget = (
                                    additional_kwargs.get(
                                        "planner_budget_stop_reason"
                                    )
                                    == "model_call_budget"
                                )
                                stopped_by_tool_call_budget = not (
                                    stopped_by_call_budget
                                )
                                stop_reason = str(
                                    additional_kwargs.get(
                                        "planner_budget_stop_reason"
                                    )
                                    or "planner_budget_guard"
                                )
                                stopped_by_batch_contract = (
                                    stop_reason == "batch_contract_violation"
                                )
                                tool_call_budget_exhausted = bool(
                                    self.max_tool_calls is not None
                                    and scheduled_tool_calls
                                    >= self.max_tool_calls
                                )
                                trace_outcome = RunOutcome.STOPPED
                                final_content = self._synthesize_from_collector(
                                    query
                                )
                                stop_agent_stream = True
                                break
                            call_signatures = [
                                (
                                    str(call.get("name") or ""),
                                    json.dumps(
                                        call.get("args") or {},
                                        ensure_ascii=False,
                                        sort_keys=True,
                                        default=str,
                                    ),
                                )
                                for call in tool_calls
                            ]
                            repeated_batch = bool(call_signatures) and all(
                                signature in seen_tool_call_signatures
                                for signature in call_signatures
                            )
                            over_tool_budget = (
                                self.max_tool_calls is not None
                                and scheduled_tool_calls + len(call_signatures)
                                > self.max_tool_calls
                            )
                            policy_decision = evidence_policy.review_tool_batch(
                                tool_calls,
                                evidence_count=len(self.collector.results),
                                scheduled_tool_calls=scheduled_tool_calls,
                                max_tool_calls=self.max_tool_calls,
                            )
                            self._emit_evidence_policy_progress(
                                policy_decision,
                                iteration=iteration,
                                progress=progress,
                            )
                            if self.collector.results and (
                                repeated_batch or over_tool_budget
                            ):
                                stopped_by_tool_call_budget = True
                                stop_reason = (
                                    "repeated_tool_call_batch"
                                    if repeated_batch
                                    else "tool_call_budget"
                                )
                                tool_call_budget_exhausted = bool(
                                    self.max_tool_calls is not None
                                    and scheduled_tool_calls
                                    >= self.max_tool_calls
                                )
                                trace_outcome = RunOutcome.STOPPED
                                self._update_progress(
                                    "Tool-call guard reached; synthesizing "
                                    "from evidence already collected",
                                    min(90, progress),
                                    {
                                        "phase": "tool_budget",
                                        "iteration": iteration,
                                        "reason": stop_reason,
                                        "scheduled_tool_calls": (
                                            scheduled_tool_calls
                                        ),
                                    },
                                )
                                final_content = self._synthesize_from_collector(
                                    query
                                )
                                stop_agent_stream = True
                                break

                            agent_messages.append(msg)
                            self.uncertainty_tracker.observe_tool_calls(
                                tool_calls
                            )
                            if trace_recorder is not None:
                                trace_recorder.record_assistant_message(msg)
                            evidence_policy.observe_calls(tool_calls)
                            scheduled_tool_calls += len(call_signatures)
                            seen_tool_call_signatures.update(call_signatures)
                            pending_tool_call_ids.update(
                                str(call.get("id"))
                                for call in tool_calls
                                if call.get("id")
                            )

                            # Surface the model's *thinking* output (the
                            # <think>…</think> reasoning) when reasoning
                            # mode is on. langchain-ollama puts the
                            # discarded thinking content into
                            # additional_kwargs["reasoning_content"]; we
                            # emit it as agent_reasoning so the thinking
                            # bubble shows the agent's actual rationale
                            # ("I should search for X because…") right
                            # before the next tool call fires. This is
                            # per-step (one emit per LLM round) —
                            # token-level streaming would require switching
                            # langgraph to stream_mode=["updates",
                            # "messages"] and capturing chunks inside agent
                            # nodes, which is a larger change.
                            reasoning_text = ""
                            if additional_kwargs:
                                reasoning_text = str(
                                    additional_kwargs.get(
                                        "reasoning_content", ""
                                    )
                                    or ""
                                ).strip()
                            # Fall back to msg.content when the model
                            # emitted prose alongside tool_calls (rare for
                            # tool-calling LLMs — most emit only the tool
                            # call), but harmless when both apply.
                            if not reasoning_text and content and tool_calls:
                                reasoning_text = str(content).strip()
                            if reasoning_text:
                                self._update_progress(
                                    reasoning_text[:280],
                                    min(85, progress),
                                    {
                                        "phase": "agent_reasoning",
                                        "iteration": iteration,
                                    },
                                )

                            if tool_calls:
                                for tc in tool_calls:
                                    raw_name = tc.get("name", "")
                                    display_name = self._display_tool_name(
                                        raw_name
                                    )
                                    msg_text = self._format_tool_call_progress(
                                        tc, display_name
                                    )
                                    self._update_progress(
                                        msg_text,
                                        min(85, progress),
                                        {
                                            "phase": "tool_call",
                                            # Keep the stable tool id in
                                            # metadata; the friendly label
                                            # already lives in msg_text.
                                            "tool": raw_name,
                                            "iteration": iteration,
                                            "arguments": {
                                                key: tc.get("args", {}).get(key)
                                                for key in ("query", "url")
                                                if tc.get("args", {}).get(key)
                                            },
                                            "scheduled_tool_calls": scheduled_tool_calls,
                                            "max_tool_calls": self.max_tool_calls,
                                        },
                                    )
                            elif content:
                                # No tool calls = final answer
                                if not self.planner_stop_guard_enabled:
                                    self.uncertainty_tracker.record_stop_attempt()
                                self._emit_evidence_policy_progress(
                                    evidence_policy.review_stop(
                                        evidence_count=len(
                                            self.collector.results
                                        )
                                    ),
                                    iteration=iteration,
                                    progress=progress,
                                )
                                final_content = content

                    if stop_agent_stream:
                        break

                elif "tools" in chunk:
                    msgs = chunk["tools"].get("messages", [])
                    for msg in msgs:
                        pending_tool_call_ids.discard(
                            str(getattr(msg, "tool_call_id", "") or "")
                        )
                        self.uncertainty_tracker.observe_tool_observation(msg)
                        self.uncertainty_tracker.sync_results(
                            self.collector.results
                        )
                        if trace_recorder is not None:
                            trace_recorder.record_tool_message(msg)
                        obs_event = self._observation_event(msg)
                        # _observation_event returns None when the tool
                        # result is a denial/error string ("Cannot fetch …"
                        # / "Error fetching …"); the WARNING in
                        # policy.py is the audit signal and the chat
                        # milestone would otherwise read as a successful
                        # page read whose content is a denial.
                        if obs_event is None:
                            continue
                        obs_message, obs_metadata = obs_event
                        self._update_progress(
                            obs_message,
                            min(
                                85,
                                10 + int((iteration / effective_max) * 75) + 3,
                            ),
                            obs_metadata,
                        )
                    # After every tool result, the agent immediately re-
                    # invokes the model to decide the next step. For
                    # thinking-mode LLMs (Qwen 3.x, deepseek-r1, etc.)
                    # that step can take 30+ seconds of silent <think>
                    # generation that gets stripped before display —
                    # leaving the last displayed line stale ("Result from
                    # web_search …") with no indication the agent is still
                    # working.
                    # Emit a contextual heartbeat so the user gets a real
                    # sense of progress (which iteration, how many sources
                    # collected, which tools are available) instead of
                    # a generic "Choosing next step…" spinner.
                    self._update_progress(
                        self._heartbeat_message(iteration),
                        min(
                            85,
                            10 + int((iteration / effective_max) * 75) + 4,
                        ),
                        {"phase": "agent_thinking", "iteration": iteration},
                    )
                    # A full tool budget no longer forces synthesis here.  The
                    # next model call sees remaining_tool_calls=0, receives no
                    # tool schema, and gets one explicit opportunity to STOP.
                    # PlannerBudgetMiddleware also counts internal replan calls,
                    # so the old outer-loop model-call check would undercount.

        except GraphRecursionError as exc:
            logger.warning(
                "LangGraph agent hit recursion limit, synthesizing partial results"
            )
            trace_outcome = RunOutcome.STOPPED
            if trace_recorder is not None:
                trace_recorder.record_error(exc)
            if not final_content:
                final_content = self._synthesize_from_collector(query)
        except Exception as exc:
            logger.exception("LangGraph agent error")
            trace_outcome = RunOutcome.ERROR
            if trace_recorder is not None:
                trace_recorder.record_error(exc)
            if not final_content:
                if self.collector.results:
                    final_content = self._synthesize_from_collector(query)
                else:
                    self._finish_trace(
                        trace_recorder,
                        outcome=trace_outcome,
                        started_at=trace_started_at,
                    )
                    return self._error_result(self._format_agent_error(exc))

        if not final_content:
            if self.collector.results:
                final_content = self._synthesize_from_collector(query)
            else:
                final_content = NO_RESULTS_MESSAGE

        result = self._finalize(
            query, final_content, iteration, nr_of_links, agent_messages
        )
        findings = result.get("findings") or []
        exported_answer = (
            findings[0].get("content", final_content)
            if findings and isinstance(findings[0], dict)
            else final_content
        )
        planner_summary = (
            planner_budget.trace_summary() if planner_budget is not None else None
        )
        policy_failure = bool(
            planner_summary
            and planner_summary.get("policy_failure_illegal_stop")
        )
        forced_stop_reason = str(
            (planner_summary or {}).get("forced_stop_reason") or ""
        )
        stop_incomplete = forced_stop_reason.startswith("stop_incomplete_")
        runtime_recovery_actions = int(
            (planner_summary or {}).get("runtime_recovery_actions") or 0
        )
        accepted_legal_stop = (
            bool(planner_summary.get("legal_stop_accepted"))
            if self.planner_stop_guard_enabled and planner_summary is not None
            else None
        )
        planner_policy_success = (
            False
            if policy_failure or stop_incomplete
            else (
                bool(accepted_legal_stop and runtime_recovery_actions == 0)
                if accepted_legal_stop is not None
                else None
            )
        )
        runtime_assisted_success = (
            False
            if policy_failure or stop_incomplete
            else (
                bool(accepted_legal_stop and runtime_recovery_actions > 0)
                if accepted_legal_stop is not None
                else None
            )
        )
        self._finish_trace(
            trace_recorder,
            outcome=trace_outcome,
            started_at=trace_started_at,
            final_answer=exported_answer,
            metadata={
                "iterations": iteration,
                "routing_decision": self._routing_metadata(),
                "planner_budget": (
                    planner_summary
                ),
                "stopped_by_call_budget": stopped_by_call_budget,
                "stopped_by_tool_call_budget": (stopped_by_tool_call_budget),
                "stopped_by_batch_contract": stopped_by_batch_contract,
                "tool_call_budget_exhausted": tool_call_budget_exhausted,
                "scheduled_tool_calls": scheduled_tool_calls,
                "stop_reason": stop_reason,
                "stop_incomplete": stop_incomplete,
                "runtime_fallback_triggered": runtime_fallback_triggered,
                "planner_policy_success": planner_policy_success,
                "runtime_assisted_success": runtime_assisted_success,
                # Compatibility alias. This now has the narrow planner-only
                # meaning and never credits a runtime-selected recovery action.
                "agent_policy_success": planner_policy_success,
                "system_success": (
                    False
                    if policy_failure or stop_incomplete
                    else bool(result.get("findings"))
                ),
                "final_report_status": result.get("final_report_status"),
                "evidence_policy": evidence_policy.trace_summary(),
                "uncertainty_state": self.uncertainty_tracker.trace_summary(),
            },
        )
        return result

    # -- Helpers ------------------------------------------------------------

    def _synthesize_from_collector(self, query: str) -> str:
        """Fallback synthesis when the agent was cut short."""
        # Check cancellation before any synthesis LLM work. This is the
        # fallback path for when the agent stream errored out; without an
        # early check, a cancel that arrived during the error path would
        # have to wait for the synthesis LLM call to complete before
        # terminating.
        self.check_termination(CHECK_CONTEXT_FALLBACK_SYNTHESIS)

        results = self.collector.results
        if not results:
            return "Research could not be completed within the iteration limit."
        if self.evidence_only_synthesis:
            # _finalize will give the separate synthesis model the real
            # collector results. Avoid a redundant model call and never
            # promote planner-authored prose into trusted evidence.
            return ""
        summaries = []
        for r in results[:20]:
            summaries.append(
                f"[{r.get('index', '?')}] {r.get('title', '')}: "
                f"{r.get('snippet', '')}"
            )
        prompt = (
            f"Synthesize a comprehensive answer to: {query}\n\n"
            f"Based on these sources:\n" + "\n".join(summaries)
        )
        try:
            response = self.synthesis_model.invoke(prompt)
            return (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
        except Exception as exc:
            logger.exception("Fallback synthesis failed")
            return _scrub_tool_error(
                f"Research collected {len(results)} sources but "
                f"synthesis failed: {exc}"
            )

    def _finalize(
        self,
        query: str,
        final_answer: str,
        iteration: int,
        nr_of_links: int,
        agent_messages: list,
    ) -> Dict[str, Any]:
        """Apply citation handling and build the return dict."""
        all_search_results = self.collector.results

        # A subsection call in detailed-report mode can answer purely
        # from previously-written sections without running new searches,
        # leaving the per-call collector empty. The citation pass below
        # is then skipped and the section is saved as raw agent prose
        # with no inline [N] markers even though ## Sources renders the
        # full accumulated bibliography (#4969). Running the pass against
        # all_links_of_system instead is NOT safe as-is: the widened
        # prompt overflows default local-model context windows, the
        # rewrite has no structure-preservation guarantees, and the
        # empty-collector condition is also reachable from chat
        # follow-ups. Until that is redesigned, make the skip loud so
        # affected runs are diagnosable from the server log.
        if (
            not all_search_results
            and self.all_links_of_system
            and final_answer != NO_RESULTS_MESSAGE
        ):
            logger.warning(
                f"Citation pass skipped: no new sources collected in this "
                f"call although {len(self.all_links_of_system)} are "
                f"accumulated — this answer will have no inline [N] "
                f"citations (#4969, query '{query[:80]}')"
            )

        # Emit synthesis milestone if it is not an agent failure
        if final_answer != NO_RESULTS_MESSAGE:
            if not all_search_results and self.all_links_of_system:
                self._update_progress(
                    f"Skipping citation synthesis (reusing {len(self.all_links_of_system)} accumulated sources)",
                    90,
                    {
                        "phase": "synthesis",
                        "type": "milestone",
                        "new_sources": 0,
                        "accumulated_sources": len(self.all_links_of_system),
                        "citation_pass_skipped": True,
                    },
                )
            elif not all_search_results and not self.all_links_of_system:
                self._update_progress(
                    "No sources available for citation synthesis",
                    90,
                    {
                        "phase": "synthesis",
                        "type": "milestone",
                        "new_sources": 0,
                        "accumulated_sources": 0,
                        "citation_pass_skipped": True,
                    },
                )
            else:
                self._update_progress(
                    f"Synthesizing {len(all_search_results)} sources with citations",
                    90,
                    {"phase": "synthesis", "type": "milestone"},
                )

        from local_deep_research.agent_harness.evidence_synthesis import (
            evidence_gap_report,
        )

        synthesized_content = final_answer
        documents: list = []
        final_report_status = "complete"

        # Citation handling — only if we have results
        if all_search_results:
            try:
                citation_result = self.citation_handler.analyze_followup(
                    query,
                    all_search_results,
                    previous_knowledge=(
                        "" if self.evidence_only_synthesis else final_answer
                    ),
                    nr_of_links=nr_of_links,
                )
                if isinstance(citation_result, dict):
                    synthesized_content = citation_result.get(
                        "content", citation_result.get("response", final_answer)
                    )
                    documents = citation_result.get("documents", [])
                    final_report_status = str(
                        citation_result.get("synthesis_status", "complete")
                    )
            except Exception:
                logger.warning(
                    "Citation handler failed, using raw agent answer"
                )
                final_report_status = "citation_handler_error_fallback"

        if not isinstance(synthesized_content, str) or not synthesized_content.strip():
            reason = (
                "citation synthesis failed"
                if final_report_status == "citation_handler_error_fallback"
                else "citation synthesis returned no report"
            )
            synthesized_content = evidence_gap_report(
                all_search_results, reason=reason
            )
            final_report_status = "runtime_nonempty_report_fallback"

        if all_search_results and not re.search(
            r"\[\d", synthesized_content or ""
        ):
            logger.warning(
                f"Synthesis produced no inline [N] citation markers "
                f"despite {len(all_search_results)} available sources — "
                f"the report body will show no citations for this "
                f"query ('{query[:80]}')"
            )

        # Format sources — delegate to base helper
        formatted_output = self._format_citations(
            synthesized_content, all_search_results
        )

        # Build reasoning trace from agent messages
        reasoning_trace = []
        for msg in agent_messages:
            entry: Dict[str, Any] = {"role": "assistant"}
            if hasattr(msg, "content") and msg.content:
                entry["content"] = msg.content
            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.get("name"), "args": tc.get("args", {})}
                    for tc in tool_calls
                ]
            reasoning_trace.append(entry)

        self._update_progress(
            "Research complete",
            100,
            {"phase": "complete", "type": "milestone", "iterations": iteration},
        )

        return {
            "findings": [
                {
                    "content": synthesized_content,
                    "question": query,
                    "search_results": all_search_results,
                    "documents": documents,
                }
            ],
            "iterations": iteration,
            "questions": {},
            "formatted_findings": formatted_output,
            "current_knowledge": synthesized_content,
            "sources": list(set(self.collector.sources)),
            "search_results": all_search_results,
            "documents": documents,
            "reasoning_trace": reasoning_trace,
            "tool_names": list(getattr(self, "_tool_names", [])),
            "research_profile": self.research_profile,
            "routing_mode": self.routing_mode,
            "routing_decision": self._routing_metadata(),
            "final_report_status": final_report_status,
            "error": None,
        }

    @staticmethod
    def _format_agent_error(exc: BaseException) -> str:
        """Prefix the exception type so downstream rendering (and the
        `ErrorReportGenerator` pattern map) have a consistent shape to match
        on. The bare `str(exc)` produced by the catch-all loses the type,
        which makes deep LangChain / LangGraph failures hard to recognise.
        """
        # Scrub credentials before this error is rendered to the user. The
        # "Agent error: <Type>:" prefix stays at the front (no secrets, ahead
        # of any truncation) so the ErrorReportGenerator pattern map still
        # matches on the exception type.
        return _scrub_tool_error(f"Agent error: {type(exc).__name__}: {exc}")

    def _error_result(self, error: str) -> Dict[str, Any]:
        logger.error(f"LangGraph agent strategy error: {error}")
        self._update_progress(
            f"Error: {error}",
            100,
            {"phase": "error", "error": error, "status": "failed"},
        )
        return {
            "findings": [],
            "iterations": 0,
            "questions": {},
            "formatted_findings": f"Error: {error}",
            "current_knowledge": "",
            "sources": [],
            "search_results": [],
            "documents": [],
            "reasoning_trace": [],
            "error": error,
        }

    def close(self):
        """Close a separately injected synthesis model, if present."""
        if self.synthesis_model is self.model:
            return
        from ...utilities.resource_utils import safe_close

        safe_close(self.synthesis_model, "agent synthesis model")
