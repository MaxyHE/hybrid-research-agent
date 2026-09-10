"""Recompute published SciFact300 metrics from saved document rankings."""
import json
import math
from pathlib import Path


RANKINGS_PATH = Path(__file__).resolve().parents[1] / "docs/evaluation/scifact-300-rankings.json"
ARMS = ("raw8", "document8_baseline", "cross_encoder_rerank8")


def compute(rows, arm):
    hits = targets = 0
    reciprocal_ranks = ndcgs = 0.0
    for row in rows:
        qrels = {document_id: grade for document_id, grade in row["qrels"].items() if grade > 0}
        ranked = row["rankings"][arm][:8]
        hits += len(qrels.keys() & set(ranked))
        targets += len(qrels)
        first_rank = next((rank for rank, document_id in enumerate(ranked, 1) if document_id in qrels), None)
        reciprocal_ranks += 1 / first_rank if first_rank else 0.0
        dcg = sum((2 ** qrels.get(document_id, 0) - 1) / math.log2(rank + 1)
                  for rank, document_id in enumerate(ranked, 1))
        ideal_dcg = sum((2 ** grade - 1) / math.log2(rank + 1)
                        for rank, grade in enumerate(sorted(qrels.values(), reverse=True)[:8], 1))
        ndcgs += dcg / ideal_dcg if ideal_dcg else 0.0
    return hits, targets, reciprocal_ranks / len(rows), ndcgs / len(rows)


def main():
    payload = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    subsets = (("all300", rows), ("exclude_history10 (290)", [row for row in rows if not row["previously_used_in_component_audit"]]))
    for subset_name, subset in subsets:
        for arm in ARMS:
            hits, targets, mrr, ndcg = compute(subset, arm)
            print(f"{subset_name} {arm}: micro Recall@8={hits}/{targets} ({hits / targets:.2%}), "
                  f"MRR@8={mrr:.4f}, nDCG@8={ndcg:.4f}")


if __name__ == "__main__":
    main()
