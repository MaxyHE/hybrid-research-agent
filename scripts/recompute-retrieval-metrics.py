"""Recompute published retrieval metrics from saved document rankings; no API."""
import json
from pathlib import Path

rows = json.loads((Path(__file__).resolve().parents[1] / 'docs/evaluation/retrieval-60-rankings.json').read_text())
for name, subset in [('all60', rows), ('chinese12', [r for r in rows if r['language'] == 'zh'])]:
    for arm in ['original_query', 'translated_chinese_query']:
        hits, targets, reciprocal = 0, 0, 0.0
        for row in subset:
            wanted = set(row['targets'])
            ranked = row[arm][:8]
            hits += len(wanted.intersection(ranked))
            targets += len(wanted)
            ranks = [i + 1 for i, doc in enumerate(ranked) if doc in wanted]
            reciprocal += 1 / min(ranks) if ranks else 0
        print(f'{name} {arm}: Recall@8={hits}/{targets} ({hits/targets:.2%}), MRR@8={reciprocal/len(subset):.4f}')
