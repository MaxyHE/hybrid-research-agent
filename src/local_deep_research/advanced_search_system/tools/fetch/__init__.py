"""Agent-facing ``fetch_content`` tool builders.

Public API:
    FETCH_MODES         — tuple of valid mode strings.
    build_fetch_tool()  — returns a LangChain ``@tool`` (or ``None`` when
                          mode == "disabled" so the caller can skip
                          registration).

Modes:
    disabled              — fetch tool is not registered with the agent.
    full                  — return the full extracted page text (legacy
                            behavior; can flood small-model context with
                            boilerplate / metadata enrichment).
    summary_focus         — LLM extracts only spans relevant to a focus
                            question the agent supplies per call.
    summary_focus_query   — same as above, but the prompt also includes
                            the original research query (passed in
                            programmatically by the strategy) so the
                            extractor can disambiguate vague focuses.

Each tool registers fetched URLs in the strategy's
``SearchResultsCollector`` for citation tracking, returning the result as
``[N] Title: ...\\nURL: ...\\n\\n<body>`` exactly like the original
in-strategy implementation, so downstream prompt formatting is unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from loguru import logger

from local_deep_research.utilities.js_rendering import (
    read_js_rendering_setting as _read_js_rendering_setting,
)
from local_deep_research.security import (
    redact_url_for_log,
    sanitize_error_for_client,
)

from .library_resolver import (
    is_citation_reference,
    make_library_resolver,
    resolve_citation_reference as _resolve_citation_reference,
)
from .prompts import SUMMARY_FOCUS_PROMPT, SUMMARY_FOCUS_QUERY_PROMPT


# Per-call timeouts and caps. Kept here rather than in the strategy file
# because they are properties of the fetch tool, not of agent
# orchestration.
CONTENT_FETCH_TIMEOUT = 30
CONTENT_MAX_LENGTH = 10_000

# Cap for credential-scrubbed fetch-tool error strings. Larger than the
# 200-char HTTP-client default because these errors feed the agent's
# reasoning; credential scrubbing still runs first on the full string (#4633).
_TOOL_ERROR_MAX_LEN = 500

# Jina Reader can return HTTP 200 for its own response while embedding the
# target's 4xx/5xx status in the Markdown body.  Treating that body as a
# successful fetched page would create false evidence (notably a full
# government-site "Page Not Found" template), so inspect this documented
# Reader warning before collector registration.
_MIRROR_TARGET_HTTP_ERROR_RE = re.compile(
    r"^\s*Warning:\s*Target URL returned error\s+(?P<status>\d{3})\b",
    re.IGNORECASE | re.MULTILINE,
)

# Jina Reader normally returns Markdown as text/plain.  An HTML/JSON error
# wrapper is transport success, not an extractable page, so never promote it
# to fetched evidence.
_MIRROR_TEXTUAL_CONTENT_TYPES = ("text/plain", "text/markdown")


def _scrub_tool_error(message: str) -> str:
    """Scrub credentials from an LLM/agent-facing fetch-tool error string."""
    return sanitize_error_for_client(message, max_length=_TOOL_ERROR_MAX_LEN)


FETCH_MODES = (
    "disabled",
    "full",
    "summary_focus",
    "summary_focus_query",
)
PUBLIC_FETCH_FALLBACKS = ("disabled", "jina")


def _register_in_collector(
    collector: Any,
    url: str,
    title: str,
    snippet_source: str,
    retrieval_method: str | None = None,
) -> int:
    """Register a fetched URL in the collector and return its 1-based citation index.

    If the URL was already tracked (via a prior search hit) the existing
    index is reused so the agent sees a stable citation per URL.
    """
    record_fetched_content = getattr(collector, "record_fetched_content", None)
    if callable(record_fetched_content):
        return record_fetched_content(
            url=url,
            title=title,
            content=snippet_source,
            retrieval_method=retrieval_method,
        )
    existing_idx = collector.find_by_url(url)
    if existing_idx is not None:
        return existing_idx
    snippet = snippet_source[:200].strip()
    if len(snippet_source) > 200:
        snippet += "..."
    start = collector.add_results(
        [{"title": title, "link": url, "snippet": snippet}],
        engine_name="fetch",
    )
    return start + 1


def _is_observed_url(collector: Any, url: str) -> bool:
    """Return whether *url* came from an earlier tool observation."""
    find_by_url = getattr(collector, "find_by_url", None)
    return bool(callable(find_by_url) and find_by_url(url) is not None)


def _enforce_url_policy(url: str, egress_context: Any) -> None:
    """Run ``evaluate_url`` against ``egress_context`` and raise
    ``PolicyDeniedError`` on denial.

    No-op when no context is configured (callers without policy enforcement,
    e.g. legacy non-LangGraph strategies, see the legacy behavior).
    """
    if egress_context is None:
        return
    from local_deep_research.security.egress.policy import (
        PolicyDeniedError,
        evaluate_url,
    )

    decision = evaluate_url(url, egress_context)
    if not decision.allowed:
        raise PolicyDeniedError(decision, target=url)


def _public_mirror_allowed(
    fallback: str, egress_context: Any
) -> bool:
    """Allow third-party mirroring only for an explicit PUBLIC_ONLY run."""
    scope = getattr(egress_context, "scope", None)
    scope_value = getattr(scope, "value", scope)
    return fallback == "jina" and scope_value == "public_only"


def _failure_code_from_exception(exc: Exception) -> str:
    """Return a stable, non-sensitive transport failure code."""
    try:
        import requests

        if isinstance(exc, requests.exceptions.ProxyError):
            return "proxy_connection_failed"
        if isinstance(exc, requests.exceptions.SSLError):
            return "tls_error"
        if isinstance(
            exc,
            (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.Timeout,
            ),
        ):
            return "timeout"
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "connection_error"
        if isinstance(exc, requests.exceptions.HTTPError):
            response = exc.response
            if response is not None:
                return f"http_{response.status_code}"
    except ImportError:  # pragma: no cover - requests is a core dependency
        pass
    return "mirror_request_error"


def _direct_attempt_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize ContentFetcher's direct-attempt facts for later diagnostics."""
    metadata = dict(result.get("fetch_metadata") or {})
    metadata.setdefault("retrieval_method", "direct_content_fetcher")
    metadata.setdefault("outcome", result.get("status", "error"))
    if result.get("status") != "success":
        metadata.setdefault("failure_code", "direct_fetch_failed")
        if result.get("error"):
            metadata.setdefault(
                "error", _scrub_tool_error(str(result["error"]))
            )
    return metadata


def _should_try_public_mirror(result: Mapping[str, Any]) -> bool:
    """Mirror only an eligible failed direct fetch, never a policy denial."""
    if result.get("status") == "success":
        return False
    error = str(result.get("error") or "").lower()
    return not any(
        marker in error
        for marker in (
            "security validation",
            "egress policy",
            "invalid or unsupported url scheme",
        )
    )


def _validate_public_mirror_target(url: str, egress_context: Any) -> None:
    """Validate the original target before it is embedded in a mirror path.

    The Jina host exemption below applies only to the *fixed relay host*.
    It never grants the original observed target a bypass: it must pass the
    normal DNS-aware SSRF validator and the run's ordinary egress policy.
    """
    from local_deep_research.security import validate_url

    if not validate_url(url):
        raise ValueError(
            "URL failed security validation (blocked by SSRF protection)"
        )
    _enforce_url_policy(url, egress_context)


def _mirror_body_failure(
    content: str, content_type: object
) -> tuple[str, str] | None:
    """Return a stable evidence-hygiene failure for a Jina response body."""
    if not content:
        return ("mirror_empty_body", "Public mirror returned an empty body")
    if not isinstance(content_type, str) or not content_type.lower().startswith(
        _MIRROR_TEXTUAL_CONTENT_TYPES
    ):
        return (
            "mirror_invalid_content",
            "Public mirror returned a non-text response; no page evidence was registered",
        )
    return None


def _assert_trusted_mirror_response_url(response_url: object) -> None:
    """Reject a redirect response that did not remain at the fixed relay."""
    from local_deep_research.security.safe_requests import (
        is_trusted_jina_mirror_url,
    )

    if not is_trusted_jina_mirror_url(response_url):
        raise ValueError("Public mirror redirect left trusted host")


def fetch_via_public_mirror(
    url: str,
    *,
    egress_context: Any,
) -> dict[str, Any]:
    """Fetch one public page through Jina Reader after direct HTTP fails.

    The original URL remains the citation target. Query strings, fragments,
    and credentialed URLs are deliberately not forwarded to a third party.
    """
    from local_deep_research.constants import USER_AGENT
    from local_deep_research.security import SafeSession
    from local_deep_research.security.safe_requests import (
        TRUSTED_JINA_MIRROR_HOST,
    )

    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return {
            "status": "error",
            "url": url,
            "source_type": "jina_public_mirror",
            "error": "Public mirror requires a plain observed http(s) URL",
            "fetch_metadata": {
                "retrieval_method": "jina_public_mirror",
                "failure_code": "mirror_url_not_eligible",
            },
        }
    # Validate the source before it is embedded in the fixed relay path.  In
    # particular, a direct-fetch failure must not turn localhost/RFC1918/etc.
    # into a Reader request merely because the relay itself is trusted.
    try:
        _validate_public_mirror_target(url, egress_context)
    except Exception as exc:
        return {
            "status": "error",
            "url": url,
            "source_type": "jina_public_mirror",
            "error": _scrub_tool_error(str(exc)),
            "fetch_metadata": {
                "retrieval_method": "jina_public_mirror",
                "failure_code": "mirror_target_validation_failed",
                "exception_type": type(exc).__name__,
            },
        }

    mirror_url = f"https://{TRUSTED_JINA_MIRROR_HOST}/{url}"
    try:
        # Do not feed the relay through the normal DNS-aware validator: under
        # a configured proxy it may receive a synthetic Teredo answer before
        # requests reaches the proxy.  SafeSession's trusted-egress mode is
        # exact-host-only and rejects every redirect that leaves this relay.
        with SafeSession(
            trusted_egress_host=TRUSTED_JINA_MIRROR_HOST
        ) as session:
            session.headers.update(
                {
                    "Accept": "text/plain",
                    "User-Agent": USER_AGENT,
                }
            )
            response = session.get(
                mirror_url,
                timeout=CONTENT_FETCH_TIMEOUT,
                allow_redirects=True,
            )
            response.raise_for_status()
            content = response.text[:CONTENT_MAX_LENGTH].strip()
            content_type = response.headers.get("content-type", "")
            _assert_trusted_mirror_response_url(response.url)
            mirror_metadata = {
                "retrieval_method": "jina_public_mirror",
                "mirror_http_status": response.status_code,
                "mirror_final_origin": redact_url_for_log(response.url),
                "mirror_redirect_count": len(response.history),
                "mirror_content_type": content_type,
            }
        target_error = _MIRROR_TARGET_HTTP_ERROR_RE.search(content)
        if target_error is not None:
            target_status = int(target_error.group("status"))
            return {
                "status": "error",
                "url": url,
                "source_type": "jina_public_mirror",
                "error": (
                    "Public mirror reports target HTTP "
                    f"{target_status}; no page evidence was registered"
                ),
                "fetch_metadata": {
                    **mirror_metadata,
                    "failure_code": "mirror_target_http_error",
                    "target_http_status": target_status,
                },
            }
        body_failure = _mirror_body_failure(content, content_type)
        if body_failure is not None:
            failure_code, error = body_failure
            return {
                "status": "error",
                "url": url,
                "source_type": "jina_public_mirror",
                "error": error,
                "fetch_metadata": {
                    **mirror_metadata,
                    "failure_code": failure_code,
                },
            }
        logger.info(
            "[FETCH] direct retrieval failed; public mirror succeeded for {}",
            redact_url_for_log(url),
        )
        return {
            "status": "success",
            "title": parsed.hostname,
            "content": content,
            "url": url,
            "retrieval_method": "jina_public_mirror",
            "fetch_metadata": mirror_metadata,
        }
    except Exception as exc:
        failure_code = _failure_code_from_exception(exc)
        logger.debug(
            "Public mirror fallback failed for {}",
            redact_url_for_log(url),
            exc_info=True,
        )
        return {
            "status": "error",
            "url": url,
            "source_type": "jina_public_mirror",
            "error": _scrub_tool_error(str(exc)),
            "fetch_metadata": {
                "retrieval_method": "jina_public_mirror",
                "failure_code": failure_code,
                "exception_type": type(exc).__name__,
                "error": _scrub_tool_error(str(exc)),
            },
        }


def _denial_reason(exc: Any) -> str:
    """Best-effort egress-denial reason code for an agent-facing message."""
    return getattr(getattr(exc, "decision", None), "reason", "policy_denied")


# Shared instruction appended to every per-URL egress denial returned to the
# agent. Tells it WHY the fetch was refused and what to do instead, so it
# adapts (stays in-scope) rather than retrying the same out-of-scope URL.
_EGRESS_DENIAL_HINT = (
    "In this run only local collection/library documents can be fetched; "
    "skip external URLs."
)

# ---------------------------------------------------------------------------
# Pre-resolution: rewrite non-network URLs BEFORE the egress gate.
#
# Two URL-shaped strings reach the fetch tool that aren't actually network
# fetches and that the egress policy correctly rejects as
# ``unsupported_scheme``. Resolving them here lets the agent get the page
# content (or a helpful error) instead of a generic denial — A3 from
# research_f3045c5b_issue_analysis.md.
# ---------------------------------------------------------------------------

# ``_KIND_RESULT`` marks a pre-resolved local document; the fetch tool
# consumes the dict directly. ``_KIND_ERROR`` short-circuits the fetch
# tool with the message in ``payload`` (already routed through the
# credential scrubber by the caller). ``_KIND_REWRITTEN`` means "this
# URL was rewritten to a new one — use the new URL going forward but
# keep running the normal HTTP flow" (e.g. a ``[N]`` citation marker
# that resolves to an external URL). ``None`` means "no pre-resolution;
# run the normal HTTP path with the original URL."
_KIND_RESULT = "result"
_KIND_ERROR = "error"
_KIND_REWRITTEN = "rewritten"


def _try_resolve_url(
    url: str,
    library_resolver: Any,
    collector: Any,
    _visited: set[str] | None = None,
) -> tuple[str, Any] | None:
    """Pre-resolve a fetch URL that isn't a network URL.

    Returns one of:

    - ``(``_KIND_RESULT``, dict)`` — local-content payload shaped like a
      ``ContentFetcher.fetch()`` success result, ready for the post-fetch
      pipeline. The caller MUST skip the egress gate (it's a local read).
    - ``(``_KIND_ERROR``, str)`` — short-circuit with this error message.
    - ``(``_KIND_REWRITTEN``, str)`` — *url* was rewritten (e.g. a
      citation marker ``[N]`` that resolved to an external URL); use
      the new URL but otherwise run the normal HTTP flow (the egress
      gate still applies).
    - ``None`` — no pre-resolution; run the normal HTTP path with *url*.

    Handles:

    - ``[N]`` citation markers → ``SearchResultsCollector.find_by_index``,
      then recurse with the resolved URL (so a citation whose source is a
      library doc still hits the local fast path, while a citation whose
      source is external surfaces a ``_KIND_REWRITTEN`` URL for the gate).
    - ``/library/document/<uuid>[/pdf]`` → ``Document.text_content`` read
      from the user DB via the library resolver.
    """
    if not isinstance(url, str):
        return None

    if _visited is None:
        _visited = set()
    if url in _visited or len(_visited) >= 5:
        return (
            _KIND_ERROR,
            f"Circular citation reference detected for '{url}'.",
        )
    _visited.add(url)

    # 1. Citation marker [N] → resolve to citation's URL.
    citation = _resolve_citation_reference(url, collector)
    if citation is not None:
        resolved_url = citation.get("link") or citation.get("url") or ""
        if not resolved_url:
            return (
                _KIND_ERROR,
                (
                    f"Citation {url} has no URL field. "
                    "Use the citation's URL or a different source instead."
                ),
            )
        # First, check if the citation's URL is itself a library doc
        # (the recursive call sees the library path and returns _KIND_RESULT
        # directly). Otherwise surface the citation's URL as a rewrite so
        # the fetch tool updates its ``url`` variable before the egress
        # gate / HTTP fetch runs against the resolved URL — not the marker.
        recursive = _try_resolve_url(
            resolved_url, library_resolver, collector, _visited=_visited
        )
        if recursive is not None:
            return recursive
        return (_KIND_REWRITTEN, resolved_url)

    # 2. Citation marker that doesn't match a tracked citation.
    # ``_resolve_citation_reference`` only returns the citation for a
    # well-formed ``[N]`` marker; anything else falls through to the
    # library-resolver check below.
    if is_citation_reference(url) is not None:
        return (
            _KIND_ERROR,
            (
                f"No registered citation matches {url}. The agent's "
                "search results use citation markers; use the source URL "
                "(the link next to the marker) or a tracked citation "
                "instead of a raw marker."
            ),
        )

    # 3. Library document URL.
    if library_resolver is not None:
        content = library_resolver(url)
        if content is not None:
            text = content.get("content") or ""
            if len(text) > CONTENT_MAX_LENGTH:
                text = text[:CONTENT_MAX_LENGTH]
            return (
                _KIND_RESULT,
                {
                    "status": "success",
                    "title": content.get("title") or "",
                    "content": text,
                    "url": content.get("url") or url,
                },
            )

    return None


def _fetch_raw_content(
    url: str,
    library_resolver: Any,
    collector: Any,
    egress_context: Any,
    settings_snapshot: dict | None,
    mode_label: str,
    public_fetch_fallback: str,
    require_observed_urls: bool,
) -> tuple[dict | None, str, str | None]:
    """Pre-resolve non-network URL shapes or fetch via HTTP.

    Returns:
        (result, final_url, error_string)
        If error_string is not None, a pre-resolution error occurred (already
        scrubbed) and should be returned immediately by the fetch tool.
    """
    from local_deep_research.content_fetcher import ContentFetcher

    pre = _try_resolve_url(url, library_resolver, collector)
    result: dict | None = None
    if pre is not None:
        kind, payload = pre
        if kind == _KIND_ERROR:
            return None, url, _scrub_tool_error(payload)
        if kind == _KIND_RESULT:
            # A local library document was resolved directly from the user DB.
            # Skip both the egress gate (local read) and ContentFetcher (no HTTP).
            result = payload
            url = result.get("url") or url
            logger.info(
                f"[FETCH] mode={mode_label} source=library url={url} — "
                "resolved local library document directly"
            )
        else:  # _KIND_REWRITTEN — citation marker resolved to a URL
            url = payload

    if result is None:
        if require_observed_urls and not _is_observed_url(collector, url):
            return (
                None,
                url,
                (
                    "Cannot fetch an unobserved URL. Fetch only an exact URL "
                    "returned by an earlier search tool (or its [N] citation "
                    "marker); search first instead of inventing a URL."
                ),
            )
        # Either no pre-resolution or a citation-marker rewrite —
        # run the normal HTTP path. Per-URL egress gate (pre-fetch)
        # + ContentFetcher's own per-redirect gate both raise
        # PolicyDeniedError on an out-of-scope URL. Run the gate
        # INSIDE the try so the denial is returned as a recoverable
        # tool message.
        _enforce_url_policy(url, egress_context)
        enable_js = _read_js_rendering_setting(settings_snapshot)
        with ContentFetcher(
            timeout=CONTENT_FETCH_TIMEOUT,
            enable_js_rendering=enable_js,
            egress_context=egress_context,
        ) as fetcher:
            result = fetcher.fetch(url, max_length=CONTENT_MAX_LENGTH)
        if _should_try_public_mirror(result) and _public_mirror_allowed(
            public_fetch_fallback, egress_context
        ):
            direct_metadata = _direct_attempt_metadata(result)
            mirrored = _fetch_via_public_mirror(
                url,
                egress_context=egress_context,
            )
            mirror_metadata = dict(mirrored.get("fetch_metadata") or {})
            if mirrored.get("status") == "success":
                mirrored["fetch_metadata"] = {
                    "direct": direct_metadata,
                    "public_mirror": mirror_metadata,
                }
                result = mirrored
            else:
                result = {
                    "status": "error",
                    "url": url,
                    "source_type": result.get("source_type", "html"),
                    "error": (
                        "direct fetch failed: "
                        f"{_scrub_tool_error(str(result.get('error') or 'unknown error'))}; "
                        "public mirror fallback failed: "
                        f"{_scrub_tool_error(str(mirrored.get('error') or 'unknown error'))}"
                    ),
                    "fetch_metadata": {
                        "direct": direct_metadata,
                        "public_mirror": mirror_metadata,
                    },
                }

    return result, url, None


def _controlled_fetch_failure(
    collector: Any,
    url: str,
    specification: Mapping[str, Any] | None,
    state: dict[str, bool],
) -> str | None:
    """Apply one explicit showcase fault without encoding a preferred URL.

    This exists solely for development recovery tasks. It can only fail the
    first *observed public* fetch, so it neither tells the planner which URL is
    correct nor changes a production request without a caller opt-in.
    """
    if not specification or state.get("applied"):
        return None
    if specification.get("kind") != "first_observed_public_fetch":
        return None
    observed_public = False
    for result in getattr(collector, "results", []) or []:
        candidate = str(result.get("link") or result.get("url") or "").strip()
        engine = str(result.get("source_engine") or "").lower()
        if candidate == url and engine != "fetch" and not engine.startswith(
            "collection_"
        ):
            observed_public = True
            break
    if not observed_public:
        return None
    state["applied"] = True
    reason = str(specification.get("reason") or "empty_body")
    if reason == "timeout":
        return f"Failed to fetch {url}: controlled timeout (showcase fault injection)."
    if reason == "blocked":
        return f"Cannot fetch {url}: controlled blocked response (showcase fault injection)."
    return (
        f"NOT RELEVANT (no extractable content: controlled empty body for {url}; "
        "showcase fault injection)."
    )


def _make_full_fetch_tool(
    collector: Any,
    settings_snapshot: dict | None = None,
    egress_context: Any = None,
    library_resolver: Any = None,
    public_fetch_fallback: str = "disabled",
    require_observed_urls: bool = False,
    controlled_failure: Mapping[str, Any] | None = None,
):
    mode_label = "full"
    failure_state = {"applied": False}

    @tool
    def fetch_content(url: str) -> str:
        """Download and read the full text content from a URL. Use when search snippets aren't detailed enough."""
        from local_deep_research.security.egress.policy import (
            PolicyDeniedError,
        )

        try:
            injected = _controlled_fetch_failure(
                collector, url, controlled_failure, failure_state
            )
            if injected is not None:
                return injected
            result, url, err_msg = _fetch_raw_content(
                url,
                library_resolver,
                collector,
                egress_context,
                settings_snapshot,
                mode_label,
                public_fetch_fallback,
                require_observed_urls,
            )
            if err_msg is not None:
                return err_msg

            if result.get("status") == "success":
                title = result.get("title", "")
                content = result.get("content", "")
                cite_idx = _register_in_collector(
                    collector,
                    url,
                    title,
                    content,
                    retrieval_method=result.get("retrieval_method"),
                )
                retrieval = result.get("retrieval_method")
                retrieval_line = (
                    "\nRetrieval: public text mirror fallback"
                    if retrieval == "jina_public_mirror"
                    else ""
                )
                return (
                    f"[{cite_idx}] Title: {title}\nURL: {url}"
                    f"{retrieval_line}\n\n{content}"
                )
            # result['error'] comes from ContentFetcher, which returns a
            # raw str(exception) — scrub it (and the url) before this
            # reaches the agent/LLM and user-visible output (#4633).
            return _scrub_tool_error(
                f"Failed to fetch {url}: {result.get('error', 'unknown error')}"
            )
        except PolicyDeniedError as exc:
            # An out-of-scope URL is a RECOVERABLE, per-call decision (the agent
            # picked one bad URL among many). Return it as a tool message — like
            # the transient-error path below — so the lead agent and pooled
            # subagents handle it identically and the agent can adapt, instead
            # of re-raising (which aborts a subagent and depends on each agent's
            # tool-error layer). The URL was already NOT fetched; the policy
            # already enforced — only the REPORTING changes, not security.
            target_url = getattr(exc, "target", "") or url
            return _scrub_tool_error(
                f"Cannot fetch {target_url}: blocked by egress policy "
                f"({_denial_reason(exc)}). {_EGRESS_DENIAL_HINT}"
            )
        except Exception as exc:
            # Message carries the mode + a REDACTED scheme://host only (no
            # userinfo/path/query) so an operator can locate the failure without
            # the log line leaking credentials, query tokens, or page content.
            # The traceback follows the sink's diagnose setting (off by default;
            # see utilities/log_utils). The agent/user-facing return is scrubbed
            # separately below.
            target_url = getattr(exc, "target", "") or url
            logger.exception(
                "fetch_content tool error (mode={}, url={})",
                mode_label,
                redact_url_for_log(target_url),
            )
            return _scrub_tool_error(f"Error fetching {target_url}: {exc}")

    return fetch_content


def _make_summary_fetch_tool(
    collector: Any,
    model: BaseChatModel,
    overall_query: str | None,
    settings_snapshot: dict | None = None,
    egress_context: Any = None,
    library_resolver: Any = None,
    public_fetch_fallback: str = "disabled",
    require_observed_urls: bool = False,
    controlled_failure: Mapping[str, Any] | None = None,
):
    """Build the summary-mode fetch tool.

    overall_query=None → focus-only prompt (``summary_focus`` mode).
    overall_query=str  → focus + overall-query prompt (``summary_focus_query``).
    """
    use_query = bool(overall_query)
    template = SUMMARY_FOCUS_QUERY_PROMPT if use_query else SUMMARY_FOCUS_PROMPT

    mode_label = "summary_focus_query" if use_query else "summary_focus"
    failure_state = {"applied": False}

    @tool
    def fetch_content(url: str, focus: str) -> str:
        """Fetch a URL and return only the spans of text relevant to ``focus``.
        Pass the specific question or claim you want answered as ``focus`` —
        the tool will quote relevant facts verbatim and discard unrelated content.
        """
        from local_deep_research.security.egress.policy import (
            PolicyDeniedError,
        )

        try:
            injected = _controlled_fetch_failure(
                collector, url, controlled_failure, failure_state
            )
            if injected is not None:
                return injected
            result, url, err_msg = _fetch_raw_content(
                url,
                library_resolver,
                collector,
                egress_context,
                settings_snapshot,
                mode_label,
                public_fetch_fallback,
                require_observed_urls,
            )
            if err_msg is not None:
                return err_msg

            if result.get("status") != "success":
                # result['error'] comes from ContentFetcher, which returns a
                # raw str(exception) — scrub it (and the url) before this
                # reaches the agent/LLM / user output (#4633).
                return _scrub_tool_error(
                    f"Failed to fetch {url}: "
                    f"{result.get('error', 'unknown error')}"
                )

            title = result.get("title") or ""
            content = result.get("content") or ""

            # Guard 1 — empty page content (paywalls, JS-only SPAs that
            # static fetch can't render, deleted pages with HTTP 200).
            # Skipping the LLM call here means we don't pay the round-trip
            # to summarise nothing, AND we don't register an empty
            # citation in the collector — `_register_in_collector` caches
            # by URL, so an empty snippet would lock the URL in as
            # "already fetched, nothing here" and the agent would never
            # retry it under a different focus.
            if not content.strip():
                logger.info(
                    f"[FETCH] mode={mode_label} url={url} — "
                    "empty page content, returning NOT RELEVANT without "
                    "LLM call or collector registration"
                )
                return f"NOT RELEVANT (no extractable content at {url})"

            fmt_kwargs = {
                "focus": focus,
                "title": title,
                "url": url,
                "content": content,
            }
            if use_query:
                fmt_kwargs["overall_query"] = overall_query
            prompt = template.format(**fmt_kwargs)

            try:
                summary_msg = model.invoke(prompt)
                summary = getattr(
                    summary_msg, "content", str(summary_msg)
                ).strip()
            except Exception as exc:
                # Redacted scheme://host + mode only — no page content,
                # focus, or credentials in the message. Traceback follows
                # the sink's diagnose setting (off by default).
                logger.exception(
                    "fetch_content summary LLM error (mode={}, url={})",
                    mode_label,
                    redact_url_for_log(url),
                )
                return _scrub_tool_error(f"Error summarizing {url}: {exc}")

            # Diagnostic log: per-fetch input/output for evaluating the
            # summariser. Single multi-line block so it's atomic per call
            # and easy to grep with ``grep -A1000 "[FETCH] mode="``.
            log_lines = [
                f"[FETCH] mode={mode_label} url={url}",
                f"[FETCH] focus: {focus}",
            ]
            if use_query:
                log_lines.append(f"[FETCH] overall_query: {overall_query}")
            log_lines.extend(
                [
                    f"[FETCH] title: {title}",
                    f"[FETCH] page_text ({len(content)} chars):",
                    content,
                    f"[FETCH] summary returned ({len(summary)} chars):",
                    summary or "(empty)",
                    "[FETCH] ---",
                ]
            )
            logger.info("\n".join(log_lines))

            # Guard 2 — empty LLM summary. The model decided nothing on
            # the page answers the focus (or it returned a malformed/empty
            # response). Treat as NOT RELEVANT and skip collector
            # registration: the agent should be free to re-fetch the URL
            # later with a different focus instead of seeing it as
            # already-cached with an empty body.
            if not summary:
                return f"NOT RELEVANT (no spans matched focus at {url})"

            cite_idx = _register_in_collector(
                collector,
                url,
                title,
                summary,
                retrieval_method=result.get("retrieval_method"),
            )
            retrieval = result.get("retrieval_method")
            retrieval_line = (
                "\nRetrieval: public text mirror fallback"
                if retrieval == "jina_public_mirror"
                else ""
            )
            return (
                f"[{cite_idx}] Title: {title}\nURL: {url}"
                f"{retrieval_line}\n\n{summary}"
            )
        except PolicyDeniedError as exc:
            # Recoverable per-URL denial — return a tool message so both the
            # lead agent and pooled subagents handle it identically and the
            # agent stays in-scope. The URL was already NOT fetched; only the
            # reporting changes, not security. (See the full-fetch variant.)
            target_url = getattr(exc, "target", "") or url
            return _scrub_tool_error(
                f"Cannot fetch {target_url}: blocked by egress policy "
                f"({_denial_reason(exc)}). {_EGRESS_DENIAL_HINT}"
            )
        except Exception as exc:
            # Message carries the mode + a REDACTED scheme://host only (no
            # userinfo/path/query) so an operator can locate the failure without
            # the log line leaking credentials, query tokens, or page content.
            # The traceback follows the sink's diagnose setting (off by default;
            # see utilities/log_utils). The agent/user-facing return is scrubbed
            # separately below.
            target_url = getattr(exc, "target", "") or url
            logger.exception(
                "fetch_content tool error (mode={}, url={})",
                mode_label,
                redact_url_for_log(target_url),
            )
            return _scrub_tool_error(f"Error fetching {target_url}: {exc}")

    return fetch_content


def build_fetch_tool(
    mode: str,
    collector: Any,
    *,
    model: BaseChatModel | None = None,
    overall_query: str = "",
    settings_snapshot: dict | None = None,
    egress_context: Any = None,
    library_resolver: Any = None,
    public_fetch_fallback: str = "disabled",
    require_observed_urls: bool = False,
    controlled_failure: Mapping[str, Any] | None = None,
):
    """Build the agent-facing ``fetch_content`` tool for *mode*.

    Returns ``None`` when ``mode == 'disabled'``; the caller should not
    register the tool with the agent in that case (and the system prompt
    should also drop the corresponding instruction line so the agent
    isn't told to use a tool that doesn't exist).

    ``settings_snapshot`` is captured by the tool closure so the per-call
    JS-rendering toggle can be read on a worker thread (where
    ``threading.local`` context does not propagate).

    ``egress_context`` is captured by the closure so the per-call URL
    can be policy-gated; when ``None``, no policy enforcement runs
    (preserves legacy non-LangGraph callers).

    ``library_resolver`` is captured by the closure so a fetch call can
    short-circuit on ``/library/document/<uuid>[/pdf]`` URLs (a local DB
    read, not a network fetch) and on bare ``[N]`` citation markers
    (rewritten to the citation's URL via ``SearchResultsCollector``).
    When ``None``, the tool falls through to the egress gate unchanged
    (which, when ``egress_context`` is configured, rejects library / citation
    URLs as ``unsupported_scheme``) — same as the pre-fix behaviour.

    ``controlled_failure`` is an explicit development-only showcase fixture.
    The sole supported value fails the first observed public fetch and never
    contains a URL, answer, or preferred tool path.
    """
    if public_fetch_fallback not in PUBLIC_FETCH_FALLBACKS:
        raise ValueError(
            f"Unknown public fetch fallback {public_fetch_fallback!r}; "
            f"expected one of {PUBLIC_FETCH_FALLBACKS}"
        )
    if mode == "disabled":
        return None
    if mode == "full":
        return _make_full_fetch_tool(
            collector,
            settings_snapshot=settings_snapshot,
            egress_context=egress_context,
            library_resolver=library_resolver,
            public_fetch_fallback=public_fetch_fallback,
            require_observed_urls=require_observed_urls,
            controlled_failure=controlled_failure,
        )
    if mode == "summary_focus":
        if model is None:
            raise ValueError("summary_focus fetch mode requires a model")
        return _make_summary_fetch_tool(
            collector,
            model,
            overall_query=None,
            settings_snapshot=settings_snapshot,
            egress_context=egress_context,
            library_resolver=library_resolver,
            public_fetch_fallback=public_fetch_fallback,
            require_observed_urls=require_observed_urls,
            controlled_failure=controlled_failure,
        )
    if mode == "summary_focus_query":
        if model is None:
            raise ValueError("summary_focus_query fetch mode requires a model")
        # Empty overall_query falls back to focus-only behaviour at format
        # time; we keep the *_query mode label so logs stay diagnostic.
        return _make_summary_fetch_tool(
            collector,
            model,
            overall_query=overall_query or None,
            settings_snapshot=settings_snapshot,
            egress_context=egress_context,
            library_resolver=library_resolver,
            public_fetch_fallback=public_fetch_fallback,
            require_observed_urls=require_observed_urls,
            controlled_failure=controlled_failure,
        )
    raise ValueError(
        f"Unknown fetch mode {mode!r}; expected one of {FETCH_MODES}"
    )


_fetch_via_public_mirror = fetch_via_public_mirror


__all__ = [
    "FETCH_MODES",
    "PUBLIC_FETCH_FALLBACKS",
    "build_fetch_tool",
    "fetch_via_public_mirror",
    "make_library_resolver",
]
