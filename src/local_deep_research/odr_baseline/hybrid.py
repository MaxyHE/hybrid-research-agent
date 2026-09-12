"""Small application factory for a Hybrid-ODR run.

The request chooses an existing Collection by ID.  This module resolves that
choice from the logged-in user's database and connects it to the ODR loop; it
does not create, index, route, or score collections.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .project_sources import ProjectCollectionConnector, ProjectPublicWebConnector
from .runtime import OdrBaselinePolicy, OdrBaselineRunner


@dataclass(frozen=True, slots=True)
class CollectionSourcePack:
    """One existing Collection selected for one research run."""

    collection_id: str
    name: str
    description: str | None

    @property
    def researcher_context(self) -> str:
        description = self.description.strip() if isinstance(self.description, str) else ""
        return f"{self.name}. {description}".strip()


def resolve_collection_source_pack(
    *, collection_id: str, username: str, user_password: str | None = None
) -> CollectionSourcePack:
    """Load the current user's selected Collection without touching its index."""

    from local_deep_research.database.models.library import Collection
    from local_deep_research.database.session_context import get_user_db_session

    with get_user_db_session(username, password=user_password) as db_session:
        collection = db_session.query(Collection).filter_by(id=collection_id).first()
        if collection is None:
            raise ValueError("selected collection is unavailable")
        if collection.agent_enabled is False:
            raise ValueError("selected collection is disabled for research")
        return CollectionSourcePack(
            collection_id=collection.id,
            name=collection.name,
            description=collection.description,
        )


def odr_p1_deep_policy() -> OdrBaselinePolicy:
    """Return the recorded P1 DeepResearchBench execution envelope.

    This is a named experiment profile, rather than a new runtime policy.  Its
    values reproduce the successful Task 19 P1 run so a Hybrid run can be
    compared against it without relying on undocumented CLI overrides.
    """

    return OdrBaselinePolicy(
        research_model_call_limit=72,
        report_model_call_allowance=1,
        max_tool_calls=60,
        breadth_budget=6,
        depth_budget=5,
        max_researcher_turns=10,
        max_concurrent_research_units=3,
        initial_delegation_strategy="parallel_first",
        max_web_actions_per_research_turn=1,
        reserve_tools_per_pending_task=2,
    )


def odr_p1_deep_evidence_handoff_policy() -> OdrBaselinePolicy:
    """Return the historical P1-H profile with source-handle handoff.

    P1-H keeps the recorded Task 19 scheduling and budget envelope intact.  It
    changes only the source-to-writer boundary: researchers and the writer
    exchange fetched-source handles, which the harness renders and audits.
    Source-linked notes (N) and action working memory (M) remain disabled.
    """

    return replace(
        odr_p1_deep_policy(),
        source_handle_evidence_handoff=True,
    )


class UnavailableWeb:
    """Collection-only runs do not create a public Web connector."""

    def search(self, query):
        raise RuntimeError("Web is unavailable for collection-only research")

    def fetch(self, resource):
        raise RuntimeError("Web is unavailable for collection-only research")


def source_scoped_query(query, source_mode, allowed_web_host_suffixes=()):
    suffix = {
        "web_only": "本次研究仅使用公开 Web。",
        "collection_only": "本次研究仅使用用户选择的文库；没有 Web 搜索权限。",
        "hybrid_available": "本次研究可使用用户选择的文库与公开 Web，按问题要求区分来源。",
    }[source_mode]
    if allowed_web_host_suffixes and source_mode != "collection_only":
        suffix += " Web 来源仅限以下域及其子域：" + ", ".join(allowed_web_host_suffixes) + "。"
    return query + "\n\n" + suffix


def build_source_scoped_runner(
    runner_class=OdrBaselineRunner, *, query, source_mode, web_connector,
    collection_connector, allowed_web_host_suffixes=(), **kwargs,
):
    """Share the three-mode tool wiring between the product and experiments."""
    if source_mode not in {"web_only", "collection_only", "hybrid_available"}:
        raise ValueError("Unknown research source mode")
    if source_mode != "web_only" and collection_connector is None:
        raise ValueError("A selected Collection is required")
    if source_mode != "collection_only" and web_connector is None:
        raise ValueError("A Web connector is required")

    class ScopedRunner(runner_class):
        def _research_tools(self, *, task, task_budget):
            tools = super()._research_tools(task=task, task_budget=task_budget)
            if source_mode == "collection_only":
                tools.pop("search_web", None)
            return tools

    runner = ScopedRunner(
        query=source_scoped_query(query, source_mode, allowed_web_host_suffixes),
        connector=UnavailableWeb() if source_mode == "collection_only" else web_connector,
        collection_connector=None if source_mode == "web_only" else collection_connector,
        **kwargs,
    )
    runner.original_user_query = query
    return runner


def _with_official_web_source_scope(
    query: str, allowed_web_host_suffixes: Iterable[str] | None
) -> str:
    """Expose an already-enforced Web source boundary to the researcher.

    The connector remains the enforcement point.  Naming the same boundary in
    the research request prevents the planner from spending its finite search
    budget pursuing sources that the connector must reject.
    """

    allowed = tuple(
        dict.fromkeys(
            value.strip().lower().lstrip(".")
            for value in allowed_web_host_suffixes or ()
            if isinstance(value, str) and value.strip()
        )
    )
    if not allowed:
        return query
    domains = ", ".join(allowed)
    return (
        f"{query.rstrip()}\n\n"
        "本次研究的来源范围是硬约束：仅可检索、阅读和引用下列官方域名及其子域："
        f"{domains}。不要搜索、阅读或引用其他域名；若这些来源未直接支持某项信息，"
        "请明确说明该限制，而不要用域外材料补全。"
    )


def build_hybrid_odr_runner(
    *,
    run_id: str,
    query: str,
    llm: Any,
    username: str | None,
    user_password: str | None = None,
    settings_snapshot: Mapping[str, Any],
    collection_id: str | None = None,
    source_mode: str | None = None,
    policy: OdrBaselinePolicy | None = None,
    usage_ledger: Any = None,
    research_llms: Iterable[Any] | None = None,
    development_transcript_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    thinking_mode: str | None = None,
    egress_context: Any = None,
    search_engine_name: str = "serper",
    public_fetch_fallback: str = "disabled",
    allowed_web_host_suffixes: Iterable[str] | None = None,
    runner_class: type[OdrBaselineRunner] = OdrBaselineRunner,
    should_cancel=None,
    on_event=None,
) -> OdrBaselineRunner:
    """Build a Web, Collection-only, or Hybrid run using the shared source wiring."""

    explicit_source_mode = source_mode is not None
    source_mode = source_mode or ("hybrid_available" if collection_id else "web_only")
    if source_mode not in {"web_only", "collection_only", "hybrid_available"}:
        raise ValueError("Unknown research source mode")
    if source_mode != "web_only" and not collection_id:
        raise ValueError("A selected Collection is required")
    allowed_web_host_suffixes = tuple(allowed_web_host_suffixes or ())
    web_connector = (
        ProjectPublicWebConnector(
            settings_snapshot=settings_snapshot,
            username=username,
            egress_context=egress_context,
            search_engine_name=search_engine_name,
            public_fetch_fallback=public_fetch_fallback,
            allowed_source_host_suffixes=allowed_web_host_suffixes,
        )
        if source_mode != "collection_only" else None
    )
    pack = None
    if source_mode != "web_only" and isinstance(collection_id, str) and collection_id.strip():
        if not username:
            raise ValueError("username is required when selecting a Collection")
        pack = resolve_collection_source_pack(
            collection_id=collection_id.strip(),
            username=username,
            user_password=user_password,
        )
    collection_connector = (
        ProjectCollectionConnector(
            collection_id=pack.collection_id,
            collection_name=pack.name,
            username=username,
            user_password=user_password,
            settings_snapshot=settings_snapshot,
        )
        if pack is not None
        else None
    )
    runner_options = dict(
        run_id=run_id,
        llm=llm,
        collection_connector=collection_connector,
        collection_context=pack.researcher_context if pack is not None else None,
        policy=policy,
        usage_ledger=usage_ledger,
        research_llms=research_llms,
        development_transcript_path=development_transcript_path,
        progress_path=progress_path,
        thinking_mode=thinking_mode,
        should_cancel=should_cancel,
        on_event=on_event,
    )
    if not explicit_source_mode:
        # Existing Qwen and historical experiment callers keep their original
        # request text, runner class and policy. Only the new product entry
        # opts into the shared three-mode wiring.
        return runner_class(
            query=_with_official_web_source_scope(query, allowed_web_host_suffixes),
            connector=web_connector, **runner_options,
        )
    return build_source_scoped_runner(
        runner_class, query=query, source_mode=source_mode,
        allowed_web_host_suffixes=allowed_web_host_suffixes,
        web_connector=web_connector, **runner_options,
    )


__all__ = [
    "CollectionSourcePack",
    "build_hybrid_odr_runner",
    "build_source_scoped_runner",
    "source_scoped_query",
    "odr_p1_deep_evidence_handoff_policy",
    "odr_p1_deep_policy",
    "resolve_collection_source_pack",
]
