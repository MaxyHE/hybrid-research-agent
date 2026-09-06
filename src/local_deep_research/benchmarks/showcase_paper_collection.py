"""Frozen source catalogue and integrity helpers for the showcase papers.

The catalogue is deliberately small and curated.  It identifies the user-facing
paper pack; it is not a field-complete literature search or a relevance set.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SHOWCASE_PAPER_COLLECTION_NAMESPACE = uuid.UUID(
    "8d6293cc-b25c-4d49-8619-57bc1da51b5d"
)


@dataclass(frozen=True)
class ShowcasePaper:
    """A paper selected before any local retrieval run."""

    paper_id: str
    title: str
    arxiv_id: str


@dataclass(frozen=True)
class ShowcaseRetrievalTask:
    """One pre-import target-document query for the component baseline."""

    task_id: str
    query: str
    target_paper_id: str


PAPERS: tuple[ShowcasePaper, ...] = (
    ShowcasePaper(
        "react",
        "ReAct: Synergizing Reasoning and Acting in Language Models",
        "2210.03629",
    ),
    ShowcasePaper(
        "webgpt",
        "WebGPT: Browser-assisted question-answering with human feedback",
        "2112.09332",
    ),
    ShowcasePaper(
        "reflexion",
        "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "2303.11366",
    ),
    ShowcasePaper(
        "lats",
        "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models",
        "2310.04406",
    ),
    ShowcasePaper(
        "rewoo",
        "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models",
        "2305.18323",
    ),
    ShowcasePaper(
        "dpr",
        "Dense Passage Retrieval for Open-Domain Question Answering",
        "2004.04906",
    ),
    ShowcasePaper(
        "rag",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "2005.11401",
    ),
    ShowcasePaper(
        "flare",
        "Active Retrieval Augmented Generation",
        "2305.06983",
    ),
    ShowcasePaper(
        "self_rag",
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "2310.11511",
    ),
    ShowcasePaper(
        "crag",
        "Corrective Retrieval Augmented Generation",
        "2401.15884",
    ),
    ShowcasePaper(
        "raptor",
        "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
        "2401.18059",
    ),
    ShowcasePaper(
        "alce",
        "Enabling Large Language Models to Generate Text with Citations",
        "2305.14627",
    ),
    ShowcasePaper(
        "generative_agents",
        "Generative Agents: Interactive Simulacra of Human Behavior",
        "2304.03442",
    ),
    ShowcasePaper(
        "longmem",
        "Augmenting Language Models with Long-Term Memory",
        "2306.07174",
    ),
    ShowcasePaper(
        "memgpt",
        "MemGPT: Towards LLMs as Operating Systems",
        "2310.08560",
    ),
    ShowcasePaper(
        "gaia",
        "GAIA: a benchmark for General AI Assistants",
        "2311.12983",
    ),
    ShowcasePaper(
        "agentbench",
        "AgentBench: Evaluating LLMs as Agents",
        "2308.03688",
    ),
    ShowcasePaper(
        "agentboard",
        "AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents",
        "2401.13178",
    ),
    ShowcasePaper(
        "browsecomp",
        "BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents",
        "2504.12516",
    ),
    ShowcasePaper(
        "deepresearch_bench",
        "DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents",
        "2506.11763",
    ),
)


RETRIEVAL_TASKS: tuple[ShowcaseRetrievalTask, ...] = (
    ShowcaseRetrievalTask(
        "R1",
        "In a language-agent paper, what framework interleaves reasoning traces "
        "with actions so that actions can query an external environment?",
        "react",
    ),
    ShowcaseRetrievalTask(
        "R2",
        "Which browser-assisted question-answering system requires collecting "
        "references while browsing and uses human feedback to optimize answer quality?",
        "webgpt",
    ),
    ShowcaseRetrievalTask(
        "R3",
        "Which agent framework uses verbal feedback stored as episodic memory "
        "without updating model weights?",
        "reflexion",
    ),
    ShowcaseRetrievalTask(
        "R4",
        "Which augmented language-model approach separates reasoning from tool "
        "observations to reduce repeated context and token consumption?",
        "rewoo",
    ),
    ShowcaseRetrievalTask(
        "R5",
        "Which system manages hierarchical memory tiers and interrupts to extend "
        "context beyond an LLM context window?",
        "memgpt",
    ),
    ShowcaseRetrievalTask(
        "R6",
        "Which RAG architecture recursively clusters and summarizes chunks into "
        "a hierarchy for long-document retrieval?",
        "raptor",
    ),
    ShowcaseRetrievalTask(
        "R7",
        "Which retrieval-augmented model uses reflection tokens to decide when "
        "to retrieve and to critique generated content?",
        "self_rag",
    ),
    ShowcaseRetrievalTask(
        "R8",
        "Which multi-turn agent benchmark proposes a progress-rate metric instead "
        "of evaluating only final success?",
        "agentboard",
    ),
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a local source file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def showcase_document_uuid(paper_id: str) -> str:
    """Return the stable LDR document id for a frozen paper id."""

    normalized = str(paper_id).strip()
    if normalized not in {paper.paper_id for paper in PAPERS}:
        raise ValueError(f"unknown showcase paper id: {paper_id!r}")
    return str(uuid.uuid5(SHOWCASE_PAPER_COLLECTION_NAMESPACE, normalized))


def catalogue_digest() -> str:
    """Hash source identities, independent of Markdown table formatting."""

    payload = [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "arxiv_id": paper.arxiv_id,
        }
        for paper in PAPERS
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_materialized_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Load the local-only source ledger and verify its frozen identities."""

    expected = {paper.paper_id: paper for paper in PAPERS}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            paper_id = str(row.get("paper_id") or "").strip()
            paper = expected.get(paper_id)
            if paper is None:
                raise ValueError(f"{path}:{line_number}: unexpected paper_id")
            if paper_id in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate paper_id {paper_id}"
                )
            if str(row.get("title") or "").strip() != paper.title:
                raise ValueError(
                    f"{path}:{line_number}: title mismatch for {paper_id}"
                )
            if str(row.get("arxiv_id") or "").strip() != paper.arxiv_id:
                raise ValueError(
                    f"{path}:{line_number}: arXiv id mismatch for {paper_id}"
                )
            for field in ("pdf_sha256", "text_sha256", "text_path"):
                if not str(row.get(field) or "").strip():
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            seen.add(paper_id)
            records.append(row)
    if seen != set(expected):
        missing = sorted(set(expected).difference(seen))
        extra = sorted(seen.difference(expected))
        raise ValueError(
            f"{path}: expected {len(expected)} frozen papers; "
            f"missing={missing}, extra={extra}"
        )
    return tuple(sorted(records, key=lambda row: str(row["paper_id"])))


def validate_materialized_text(record: dict[str, Any]) -> str:
    """Return verified extracted text from one local-only materialization row."""

    text_path = Path(str(record["text_path"])).expanduser().resolve()
    if not text_path.is_file():
        raise ValueError(f"missing extracted text: {text_path}")
    expected_sha256 = str(record["text_sha256"])
    actual_sha256 = sha256_file(text_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"extracted text hash mismatch: {text_path}")
    text = text_path.read_text(encoding="utf-8").strip()
    if len(text) < 1_000 or "abstract" not in text.lower():
        raise ValueError(
            f"extracted text is not a usable full paper (missing length/abstract): {text_path}"
        )
    return text


def corpus_digest(records: Sequence[dict[str, Any]]) -> str:
    """Hash frozen identities and validated source bytes for one materialization."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row["paper_id"])):
        for field in ("paper_id", "arxiv_id", "pdf_sha256", "text_sha256"):
            digest.update(str(record[field]).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def extract_arxiv_revision(html: str, arxiv_id: str) -> str:
    """Read the frozen revision from an arXiv abstract-page payload."""

    revisions = re.findall(
        rf"arXiv:{re.escape(arxiv_id)}(v\d+)", html, flags=re.IGNORECASE
    )
    if not revisions:
        raise ValueError(f"could not find an arXiv revision for {arxiv_id}")
    return revisions[-1]


def papers() -> Iterable[ShowcasePaper]:
    """Yield frozen papers in their stable catalogue order."""

    return iter(PAPERS)


def score_frozen_retrieval_tasks(
    rankings: dict[str, Sequence[str]], *, top_k: int = 8
) -> dict[str, Any]:
    """Score fixed one-target queries without inventing graded relevance labels."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    expected_task_ids = {task.task_id for task in RETRIEVAL_TASKS}
    if set(rankings) != expected_task_ids:
        raise ValueError(
            "rankings must contain exactly the frozen retrieval tasks"
        )

    per_task: dict[str, dict[str, Any]] = {}
    recall_values: list[float] = []
    mrr_values: list[float] = []
    for task in RETRIEVAL_TASKS:
        raw_ranking = list(rankings[task.task_id][:top_k])
        target_rank = next(
            (
                rank
                for rank, paper_id in enumerate(raw_ranking, start=1)
                if paper_id == task.target_paper_id
            ),
            None,
        )
        target_recalled = target_rank is not None
        recall = float(target_recalled)
        reciprocal_rank = 1.0 / target_rank if target_rank else 0.0
        unique_paper_ids = list(dict.fromkeys(raw_ranking))
        recall_values.append(recall)
        mrr_values.append(reciprocal_rank)
        per_task[task.task_id] = {
            "target_paper_id": task.target_paper_id,
            "returned_candidates": len(raw_ranking),
            "unique_documents": len(unique_paper_ids),
            "duplicate_candidate_documents": len(raw_ranking)
            - len(unique_paper_ids),
            "target_document_recalled": target_recalled,
            "target_document_rank": target_rank,
            f"document_recall_at_{top_k}": recall,
            f"mrr_at_{top_k}": reciprocal_rank,
        }
    count = len(RETRIEVAL_TASKS)
    return {
        "top_k": top_k,
        "evaluated_queries": count,
        "metrics": {
            f"document_recall_at_{top_k}": sum(recall_values) / count,
            f"mrr_at_{top_k}": sum(mrr_values) / count,
        },
        "per_task": per_task,
    }
