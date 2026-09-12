"""Web-service adapter for the compact Hybrid-ODR research loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from ...exceptions import ResearchTerminatedException
from .hybrid_task_control import HybridTaskProgress, save_cancelled_research

from ...odr_baseline import (
    build_hybrid_odr_runner,
    odr_p1_deep_policy,
)


@dataclass(frozen=True, slots=True)
class HybridOdrRunResult:
    report_markdown: str
    source_rows: tuple[dict[str, str], ...]
    metadata: dict[str, object]
    task_count: int


def run_hybrid_odr(
    *,
    run_id: str,
    query: str,
    llm: Any,
    username: str,
    settings_snapshot: Mapping[str, Any],
    output_root: Path,
    collection_id: str | None = None,
    source_mode: str | None = None,
    user_password: str | None = None,
    search_engine_name: str = "serper",
    egress_context: Any = None,
    should_cancel=None,
    on_progress=None,
) -> HybridOdrRunResult:
    """Run hosted research or the explicitly configured Qwen evidence workflow."""

    source_mode = source_mode or ("hybrid_available" if collection_id else "web_only")
    if should_cancel is not None and should_cancel():
        raise ResearchTerminatedException("Hybrid research cancelled before start")
    qwen_model = os.environ.get("LDR_HYBRID_QWEN_MODEL", "").strip()
    use_qwen = bool(qwen_model and getattr(llm, "model_name", None) == qwen_model)
    use_located = not use_qwen and os.environ.get("LDR_HYBRID_EVIDENCE_HANDOFF", "").strip().lower() == "located"
    policy = odr_p1_deep_policy()
    options = {}
    if use_qwen:
        from ...odr_baseline.qwen_candidate import QwenCandidateRunner

        catalogue = []
        if collection_id and source_mode != "web_only":
            from ...database.models.library import Document, DocumentCollection
            from ...database.session_context import get_user_db_session

            with get_user_db_session(username, password=user_password) as session:
                rows = (
                    session.query(Document.id, Document.title)
                    .join(DocumentCollection, DocumentCollection.document_id == Document.id)
                    .filter(DocumentCollection.collection_id == collection_id)
                    .order_by(Document.id).all()
                )
                catalogue = [{"document_id": row.id, "title": row.title or "Untitled"} for row in rows]

        class WebQwenRunner(QwenCandidateRunner):
            def __init__(self, **kwargs):
                super().__init__(
                    source_mode="hybrid" if source_mode == "hybrid_available" else source_mode,
                    collection_catalogue=catalogue,
                    **kwargs,
                )

        options = {"runner_class": WebQwenRunner, "thinking_mode": "disabled"}
        policy = replace(
            policy, max_concurrent_research_units=1,
            initial_delegation_strategy="adaptive",
            source_handle_evidence_handoff=True,
            evidence_excerpt_max_chars=1200,
            evidence_brief_enabled=False, evidence_narrative_brief_enabled=True,
        )
    if use_located:
        from ...odr_baseline.located_handoff import LocatedHandoffRunner

        class WebLocatedRunner(LocatedHandoffRunner):
            def __init__(self, **kwargs):
                super().__init__(writer_guidance=True, source_fact_handoff=True,
                                 attributed_handoff=True, **kwargs)

        options = {"runner_class": WebLocatedRunner}
    runner = build_hybrid_odr_runner(
        run_id=run_id,
        query=query,
        llm=llm,
        username=username,
        user_password=user_password,
        settings_snapshot=settings_snapshot,
        collection_id=collection_id,
        source_mode=source_mode,
        policy=policy,
        egress_context=egress_context,
        search_engine_name=search_engine_name,
        should_cancel=should_cancel,
        on_event=HybridTaskProgress(on_progress),
        **options,
    )
    try:
        result = runner.run_evidence_ledger_repair_workflow() if use_qwen else runner.run()
        runner._check_cancelled()
    except ResearchTerminatedException:
        try:
            save_cancelled_research(runner, output_root)
        except Exception:
            logger.exception("Could not save cancelled Hybrid research {}", run_id)
        raise
    # Keep snapshots, trace, and citation-audit artifacts on the server. Their
    # location is intentionally not copied into browser-visible run metadata.
    runner.write_artifacts(result, artifact_root=output_root)
    source_rows = tuple(
        {
            "url": source.url,
            "title": source.title,
            "snippet": source.snippet,
            "source_type": "local_collection" if source.channel == "collection" else "web",
            "hybrid_odr_source_id": source.source_id,
            "hybrid_odr_content_hash": source.content_sha256 or "",
            "hybrid_odr_source_channel": source.channel,
        }
        for source in result.sources
        if source.content
    )
    return HybridOdrRunResult(
        report_markdown=result.report_markdown,
        source_rows=source_rows,
        metadata={
            "hybrid_odr": {
                "status": result.status,
                "terminal_reason": result.terminal_reason,
                "model_calls_used": result.model_calls_used,
                "tool_calls_used": result.tool_calls_used,
                "task_count": len(result.research_tasks),
                "fetched_source_count": len(source_rows),
                "selected_collection_id": collection_id,
                "source_mode": source_mode,
                "execution_profile": "qwen-evidence-ledger" if use_qwen else "p1-deep",
                "writer_context_version": runner.candidate_implementation if use_qwen else ("located-attributed" if use_located else "default-h-off"),
            }
        },
        task_count=len(result.research_tasks),
    )


__all__ = ["HybridOdrRunResult", "run_hybrid_odr"]
