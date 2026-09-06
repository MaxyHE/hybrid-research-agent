"""Public-web connector adapters that reuse infrastructure without Hybrid logic.

The generic controller only depends on the ``SourceConnector`` protocol.  This
module adapts the project's existing search-engine factory and full-text
extractor behind that protocol while keeping all URL checks and fetches outside
the model boundary.  Search results are discovery data only: even an engine
that returns ``full_content`` cannot bypass the explicit full-fetch snapshot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from .connectors import DiscoveredResource, FetchedResource
from .source_policy import canonicalize_url


SearchEngineFactory = Callable[[], Any]
FullTextFetcher = Callable[[str], str]
UrlAuthorizer = Callable[[str], None]


class PublicWebFetchError(RuntimeError):
    """One safe, action-loop-visible public fetch failure category."""

    _ALLOWED_CODES = frozenset(
        {
            "public_fetch_authorization_failed",
            "public_fetch_empty",
            "public_fetch_timeout",
            "public_fetch_transport_failed",
        }
    )

    def __init__(self, failure_code: str) -> None:
        if failure_code not in self._ALLOWED_CODES:
            raise ValueError("unsupported public web fetch failure code")
        self.failure_code = failure_code
        super().__init__(failure_code)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _text(value: object, *, fallback: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return value.strip()[:maximum]


class SearchEnginePublicWebConnector:
    """Adapt a concrete, non-agent search engine plus a separately gated fetcher."""

    def __init__(
        self,
        *,
        search_engine_factory: SearchEngineFactory,
        full_text_fetcher: FullTextFetcher,
        url_authorizer: UrlAuthorizer,
        clock: Callable[[], str] = _timestamp,
    ) -> None:
        if not callable(search_engine_factory):
            raise TypeError("search_engine_factory must be callable")
        if not callable(full_text_fetcher):
            raise TypeError("full_text_fetcher must be callable")
        if not callable(url_authorizer):
            raise TypeError("url_authorizer must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._search_engine_factory = search_engine_factory
        self._full_text_fetcher = full_text_fetcher
        self._url_authorizer = url_authorizer
        self._clock = clock

    def search(self, query: str) -> Iterable[DiscoveredResource]:
        engine = self._search_engine_factory()
        if not callable(getattr(engine, "run", None)):
            raise TypeError("search engine must provide run(query)")
        try:
            raw_results = engine.run(query)
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        if not isinstance(raw_results, list):
            raise TypeError("search engine must return a list of result mappings")
        discovered: list[DiscoveredResource] = []
        seen_urls: set[str] = set()
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                continue
            raw_url = raw.get("link") or raw.get("url")
            try:
                canonical = canonicalize_url(str(raw_url or ""))
            except ValueError:
                continue
            if canonical.canonical_url in seen_urls:
                continue
            seen_urls.add(canonical.canonical_url)
            discovered.append(
                DiscoveredResource(
                    resource_locator=canonical.canonical_url,
                    title=_text(
                        raw.get("title"),
                        fallback=canonical.canonical_url,
                        maximum=1_000,
                    ),
                    snippet=_text(
                        raw.get("snippet") or raw.get("content") or raw.get("description"),
                        fallback="No search snippet returned.",
                        maximum=1_500,
                    ),
                )
            )
        return tuple(discovered)

    def fetch(self, resource: DiscoveredResource) -> FetchedResource:
        if not isinstance(resource, DiscoveredResource):
            raise TypeError("resource must be DiscoveredResource")
        canonical = canonicalize_url(resource.resource_locator)
        try:
            self._url_authorizer(canonical.canonical_url)
        except Exception as exc:
            raise PublicWebFetchError("public_fetch_authorization_failed") from exc
        try:
            content = self._full_text_fetcher(canonical.canonical_url)
        except TimeoutError as exc:
            raise PublicWebFetchError("public_fetch_timeout") from exc
        except Exception as exc:
            raise PublicWebFetchError("public_fetch_transport_failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise PublicWebFetchError("public_fetch_empty")
        return FetchedResource(
            content=content,
            retrieved_at=self._clock(),
            title=resource.title,
        )


def project_public_web_connector(
    *,
    search_engine_name: str,
    settings_snapshot: Mapping[str, Any],
    username: str | None = None,
    egress_context: Any = None,
    fetch_timeout_seconds: int = 30,
    clock: Callable[[], str] = _timestamp,
) -> SearchEnginePublicWebConnector:
    """Build a public connector from existing project primitives, not Hybrid tools.

    Search engines are constructed with ``llm=None`` deliberately: a hidden
    query-rewrite or quality-model call would evade General's model-call ledger.
    Egress authorization occurs immediately before full fetch and is evaluated
    against the supplied run context when one exists.
    """

    snapshot = dict(settings_snapshot)
    if not isinstance(search_engine_name, str) or not search_engine_name.strip():
        raise ValueError("search_engine_name must be non-empty")
    if not snapshot:
        raise ValueError("settings_snapshot must be non-empty")
    if (
        isinstance(fetch_timeout_seconds, bool)
        or not isinstance(fetch_timeout_seconds, int)
        or not 1 <= fetch_timeout_seconds <= 300
    ):
        raise ValueError("fetch_timeout_seconds must be an integer between 1 and 300")

    def make_engine():
        from local_deep_research.web_search_engines.search_engine_factory import (
            create_search_engine,
        )

        engine = create_search_engine(
            engine_name=search_engine_name.strip(),
            llm=None,
            username=username,
            settings_snapshot=snapshot,
            programmatic_mode=True,
        )
        if engine is None:
            raise RuntimeError("search_engine_unavailable")
        return engine

    def authorize(url: str) -> None:
        from local_deep_research.security.ssrf_validator import validate_url

        if not validate_url(url):
            raise ValueError("public_fetch_ssrf_denied")
        if egress_context is not None:
            from local_deep_research.security.egress.policy import evaluate_url

            decision = evaluate_url(url, egress_context)
            if not decision.allowed:
                raise ValueError("public_fetch_egress_denied")

    def fetch_full_text(url: str) -> str:
        from local_deep_research.research_library.downloaders.extraction import (
            batch_fetch_and_extract,
        )
        from local_deep_research.utilities.js_rendering import (
            read_js_rendering_setting,
        )

        result = batch_fetch_and_extract(
            [url],
            timeout=fetch_timeout_seconds,
            enable_js_rendering=read_js_rendering_setting(snapshot),
        )
        return result.get(url) if isinstance(result, Mapping) else ""

    return SearchEnginePublicWebConnector(
        search_engine_factory=make_engine,
        full_text_fetcher=fetch_full_text,
        url_authorizer=authorize,
        clock=clock,
    )


__all__ = [
    "SearchEnginePublicWebConnector",
    "PublicWebFetchError",
    "project_public_web_connector",
]
