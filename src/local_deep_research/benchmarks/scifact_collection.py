"""Small, dependency-free helpers for the BEIR SciFact Collection audit.

SciFact supplies document-level relevance labels, while LDR retrieves chunks.
The scorer intentionally keeps the original chunk positions and only credits a
document once.  A duplicate chunk in the product's top-8 therefore consumes a
rank instead of silently becoming an extra document candidate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCIFACT_COLLECTION_NAMESPACE = uuid.UUID("a3d8a408-d632-4bc5-8c72-5ce297c63405")


@dataclass(frozen=True)
class SciFactDocument:
    """One BEIR corpus record, kept at document rather than chunk granularity."""

    corpus_id: str
    title: str
    text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scifact_document_uuid(corpus_id: str) -> str:
    """Return the deterministic LDR document id used by the importer."""

    normalized = str(corpus_id).strip()
    if not normalized:
        raise ValueError("SciFact corpus id must be non-empty")
    return str(uuid.uuid5(SCIFACT_COLLECTION_NAMESPACE, normalized))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            yield row


def load_scifact_corpus(path: Path) -> tuple[SciFactDocument, ...]:
    documents: list[SciFactDocument] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        corpus_id = str(row.get("_id") or "").strip()
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        if not corpus_id or not title or not text:
            raise ValueError(
                f"{path}: each corpus record needs _id, title and text"
            )
        if corpus_id in seen:
            raise ValueError(f"{path}: duplicate corpus id {corpus_id}")
        seen.add(corpus_id)
        documents.append(
            SciFactDocument(corpus_id=corpus_id, title=title, text=text)
        )
    if not documents:
        raise ValueError(f"{path}: no usable SciFact corpus records")
    return tuple(documents)


def load_scifact_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for row in _read_jsonl(path):
        query_id = str(row.get("_id") or "").strip()
        query = str(row.get("text") or "").strip()
        if not query_id or not query:
            raise ValueError(f"{path}: each query needs _id and text")
        if query_id in queries:
            raise ValueError(f"{path}: duplicate query id {query_id}")
        queries[query_id] = query
    if not queries:
        raise ValueError(f"{path}: no usable SciFact queries")
    return queries


def load_scifact_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load a BEIR ``qrels/<split>.tsv`` file, retaining positive grades."""

    labels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not required.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path}: expected TSV columns {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            query_id = str(row.get("query-id") or "").strip()
            corpus_id = str(row.get("corpus-id") or "").strip()
            try:
                score = int(str(row.get("score") or "").strip())
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: score must be an integer"
                ) from exc
            if not query_id or not corpus_id:
                raise ValueError(
                    f"{path}:{line_number}: empty query-id or corpus-id"
                )
            if score > 0:
                labels[query_id][corpus_id] = score
    if not labels:
        raise ValueError(f"{path}: no positive relevance labels")
    return {query_id: dict(scores) for query_id, scores in labels.items()}


def corpus_digest(documents: Sequence[SciFactDocument]) -> str:
    """Hash the imported document identity and content, independent of JSON key order."""

    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.corpus_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.title.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _ranked_unique_documents(
    document_ids: Sequence[str], top_k: int
) -> Iterable[tuple[int, str]]:
    """Yield a document once at its first *raw chunk* rank in the top-k list."""

    seen: set[str] = set()
    for raw_rank, document_id in enumerate(document_ids[:top_k], start=1):
        if document_id in seen:
            continue
        seen.add(document_id)
        yield raw_rank, document_id


def score_document_rankings(
    *,
    qrels: Mapping[str, Mapping[str, int]],
    rankings: Mapping[str, Sequence[str]],
    top_k: int = 8,
) -> dict[str, Any]:
    """Score LDR's chunk-ranked output against SciFact document qrels.

    ``rankings`` retains every candidate in its returned order. Duplicate
    document hits do not receive extra credit, but do retain their occupied
    rank position.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    missing = sorted(set(qrels).difference(rankings))
    if missing:
        raise ValueError(
            f"missing retrieval output for qrels query ids: {missing[:5]}"
        )

    per_query: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for query_id in sorted(qrels):
        relevance = qrels[query_id]
        raw_ranking = rankings[query_id]
        unique_ranked = tuple(_ranked_unique_documents(raw_ranking, top_k))
        relevant_retrieved = [
            document_id
            for _, document_id in unique_ranked
            if document_id in relevance
        ]
        recall = len(relevant_retrieved) / len(relevance)
        first_relevant_rank = next(
            (
                raw_rank
                for raw_rank, document_id in unique_ranked
                if document_id in relevance
            ),
            None,
        )
        reciprocal_rank = (
            1.0 / first_relevant_rank if first_relevant_rank else 0.0
        )

        dcg = sum(
            (2 ** relevance[document_id] - 1) / math.log2(raw_rank + 1)
            for raw_rank, document_id in unique_ranked
            if document_id in relevance
        )
        ideal_grades = sorted(relevance.values(), reverse=True)[:top_k]
        ideal_dcg = sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(ideal_grades, start=1)
        )
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        ndcgs.append(ndcg)
        per_query[query_id] = {
            "relevant_documents": len(relevance),
            "returned_candidates": min(len(raw_ranking), top_k),
            "unique_documents": len(unique_ranked),
            "duplicate_candidate_documents": min(len(raw_ranking), top_k)
            - len(unique_ranked),
            "relevant_documents_retrieved": len(relevant_retrieved),
            f"recall_at_{top_k}": recall,
            f"mrr_at_{top_k}": reciprocal_rank,
            f"ndcg_at_{top_k}": ndcg,
        }

    count = len(per_query)
    return {
        "top_k": top_k,
        "evaluated_queries": count,
        "metrics": {
            f"recall_at_{top_k}": sum(recalls) / count,
            f"mrr_at_{top_k}": sum(reciprocal_ranks) / count,
            f"ndcg_at_{top_k}": sum(ndcgs) / count,
        },
        "per_query": per_query,
    }
