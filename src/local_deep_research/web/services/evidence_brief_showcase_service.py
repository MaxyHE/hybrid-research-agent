"""Read-only presentation adapter for audited Hybrid ODR showcase runs.

The ordinary research result view deliberately keeps detailed artifacts on the
server.  This adapter exposes a *curated, fixed* successful run without
leaking its filesystem location or generic trace.  Its payload is derived from
the actual run JSON, source registry, and reviewed evidence ledger rather than
from hand-written demonstration copy.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


class EvidenceBriefShowcaseUnavailable(RuntimeError):
    """The local installation does not have the frozen showcase artifacts."""


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LIBRARY_DOCUMENT_URL_RE = re.compile(
    r"^/library/document/[A-Za-z0-9][A-Za-z0-9_-]*(?:/pdf)?$"
)

# Deliberately a closed catalogue: the route never accepts a filesystem path
# or arbitrary run id from the browser.  Each card title describes the source
# boundary, while all factual material comes from the immutable run artifacts.
_SHOWCASES: dict[str, dict[str, Any]] = {
    "gaia-hybrid-r3": {
        "relative_run_dir": Path(
            "hybrid_odr_v2_closed_hybrid/runs/"
            "odr-v2-closed-gaia-hybrid-evidence-ledger-r3-url-recovery-20260905"
        ),
        "title": "GAIA 复现准备：定义与当前入口分流",
        "subtitle": "同一个研究问题，论文定义只读资料包；当前数据与运行入口只读官方公开页。",
        "cards": {
            "req-001": {
                "heading": "论文如何定义任务与评分？",
                "boundary": "仅来自 Collection 中冻结的 GAIA 论文",
            },
            "req-002": {
                "heading": "现在从哪里拿数据并开始复现？",
                "boundary": "仅来自 GAIA 官方 Hugging Face dataset 页面",
            },
        },
    }
}


def _showcase_root() -> Path:
    configured_root = os.getenv("LDR_EVIDENCE_BRIEF_SHOWCASE_ROOT", "").strip()
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return _PROJECT_ROOT / "outputs"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBriefShowcaseUnavailable(
            "展示所需的审计产物不可用；请先在本机保留该次冻结运行。"
        ) from exc


def _safe_source_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    if _LIBRARY_DOCUMENT_URL_RE.fullmatch(url):
        return url
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc and not parsed.username:
        return url
    return None


def _format_elapsed(seconds: object) -> str:
    try:
        whole_seconds = round(float(seconds))
    except (TypeError, ValueError):
        return "—"
    minutes, remainder = divmod(max(whole_seconds, 0), 60)
    return f"{minutes} 分 {remainder:02d} 秒" if minutes else f"{remainder} 秒"


def _quote_preview(quote: str, *, limit: int = 620) -> str:
    compact = " ".join(quote.split())
    if len(compact) <= limit:
        return compact
    boundary = compact.rfind(" ", 0, limit)
    return f"{compact[: boundary if boundary > 0 else limit].rstrip()}…"


def _audit_row(label: str, audit: Mapping[str, object]) -> dict[str, str]:
    passed = bool(audit.get("passed"))
    if label == "URL provenance":
        numerator = len(audit.get("known_urls", []))
    else:
        numerator = len(audit.get("rendered_requirement_ids", []))
    return {
        "label": label,
        "value": f"{numerator} / {numerator}" if passed else "未通过",
        "detail": "通过" if passed else "需要复核",
        "tone": "passed" if passed else "review",
    }


def get_evidence_brief_showcase(
    showcase_id: str, *, output_root: Path | None = None
) -> dict[str, object]:
    """Return a browser-safe view of one reviewed evidence-card run.

    ``output_root`` exists for tests and controlled deployments.  It is never
    taken from HTTP input; browser-visible data contains no server path.
    """
    definition = _SHOWCASES.get(showcase_id)
    if definition is None:
        raise EvidenceBriefShowcaseUnavailable("该展示案例不存在。")

    root = (output_root or _showcase_root()).resolve()
    run_dir = (root / definition["relative_run_dir"]).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise EvidenceBriefShowcaseUnavailable("展示目录无效。") from exc

    attempt = _read_json(run_dir / "attempt.json")
    run = _read_json(run_dir / "run.json")
    ledger = _read_json(run_dir / "evidence_ledger.json")
    sources = _read_json(run_dir / "sources.json")
    metrics = _read_json(run_dir / "execution_metrics.json")
    if not all(isinstance(value, Mapping) for value in (attempt, run, metrics)):
        raise EvidenceBriefShowcaseUnavailable("展示产物格式不受支持。")
    if not isinstance(ledger, list) or not isinstance(sources, list):
        raise EvidenceBriefShowcaseUnavailable("展示证据账本格式不受支持。")

    source_by_id = {
        source.get("source_id"): source
        for source in sources
        if isinstance(source, Mapping) and isinstance(source.get("source_id"), str)
    }
    cards = []
    card_copy = definition["cards"]
    for entry in ledger:
        if not isinstance(entry, Mapping) or entry.get("status") != "covered":
            continue
        requirement_id = entry.get("requirement_id")
        presentation = card_copy.get(requirement_id)
        source = source_by_id.get(entry.get("source_id"))
        quote = entry.get("support_quote")
        if (
            not isinstance(presentation, Mapping)
            or not isinstance(source, Mapping)
            or not isinstance(quote, str)
            or not quote.strip()
        ):
            continue
        source_url = _safe_source_url(source.get("url"))
        if source_url is None:
            continue
        channel = entry.get("source_channel")
        if channel not in {"collection", "web"}:
            continue
        cards.append(
            {
                "id": requirement_id,
                "heading": presentation["heading"],
                "boundary": presentation["boundary"],
                "requirement": entry.get("requirement", ""),
                "channel": channel,
                "channel_label": "资料包 / 论文" if channel == "collection" else "官方公开 Web",
                "source_title": source.get("title", "未命名来源"),
                "source_url": source_url,
                "source_range": f"{entry.get('support_start', '—')} : {entry.get('support_end', '—')}",
                "quote_preview": _quote_preview(quote),
                "quote": quote.strip(),
            }
        )

    planned_count = len(definition["cards"])
    if len(cards) != planned_count:
        raise EvidenceBriefShowcaseUnavailable("展示运行没有完整的已审阅证据卡。")

    model_usage = metrics.get("model_usage")
    price_estimate = (
        model_usage.get("price_estimate", {})
        if isinstance(model_usage, Mapping)
        else {}
    )
    citation_audit = run.get("citation_audit", {})
    claim_audit = run.get("claim_support_audit", {})
    if not isinstance(citation_audit, Mapping) or not isinstance(claim_audit, Mapping):
        raise EvidenceBriefShowcaseUnavailable("展示运行缺少审计结果。")

    task = attempt.get("task")
    collection = attempt.get("collection")
    task_query = task.get("query", "") if isinstance(task, Mapping) else ""
    collection_name = (
        collection.get("collection_name", "Collection")
        if isinstance(collection, Mapping)
        else "Collection"
    )
    return {
        "id": showcase_id,
        "title": definition["title"],
        "subtitle": definition["subtitle"],
        "question": task_query,
        "status": run.get("status", "unknown"),
        "terminal_reason": run.get("terminal_reason", ""),
        "collection_name": collection_name,
        "cards": cards,
        "audits": (
            _audit_row("URL provenance", citation_audit),
            _audit_row("Claim support", claim_audit),
        ),
        "metrics": (
            {"label": "完成耗时", "value": _format_elapsed(run.get("elapsed_seconds"))},
            {"label": "模型调用", "value": str(run.get("model_calls_used", "—"))},
            {"label": "工具调用", "value": str(run.get("tool_calls_used", "—"))},
            {"label": "实际抓取", "value": str(attempt.get("fetched_source_count", "—"))},
            {
                "label": "成本估计",
                "value": (
                    f"¥{price_estimate.get('minimum')}–{price_estimate.get('maximum')}"
                    if isinstance(price_estimate, Mapping)
                    and price_estimate.get("minimum") is not None
                    and price_estimate.get("maximum") is not None
                    else "—"
                ),
            },
        ),
        "run_label": run.get("run_id", showcase_id),
        "run_date": str(attempt.get("finished_at", ""))[:10],
        "model_label": str(attempt.get("model", "")),
    }


__all__ = ["EvidenceBriefShowcaseUnavailable", "get_evidence_brief_showcase"]
