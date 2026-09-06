"""Web-service adapter for the compact Hybrid-ODR research loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
) -> HybridOdrRunResult:
    """Run the shared H-off/default writer route with the selected sources."""

    source_mode = source_mode or ("hybrid_available" if collection_id else "web_only")
    runner = build_hybrid_odr_runner(
        run_id=run_id,
        query=query,
        llm=llm,
        username=username,
        user_password=user_password,
        settings_snapshot=settings_snapshot,
        collection_id=collection_id,
        source_mode=source_mode,
        policy=odr_p1_deep_policy(),
        egress_context=egress_context,
        search_engine_name=search_engine_name,
    )
    result = runner.run()
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
                "execution_profile": "p1-deep",
                "writer_context_version": "default-h-off",
            }
        },
        task_count=len(result.research_tasks),
    )


__all__ = ["HybridOdrRunResult", "run_hybrid_odr"]
