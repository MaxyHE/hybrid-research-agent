"""Recompute published 2026-10-14/16 retrieval results without models or APIs."""
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'docs/evaluation/retrieval-followup-20261016'


def load(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def metrics(rows, k=8):
    hits = relevant = 0
    rr = ndcg = macro = 0.0
    for row in rows:
        gold = {d: g for d, g in row['qrels'].items() if g > 0}
        ranked = row['ranking'][:k]
        assert len(set(ranked)) == len(ranked)
        found = len(set(ranked) & gold.keys())
        hits += found
        relevant += len(gold)
        macro += found / len(gold)
        first = next((i for i, d in enumerate(ranked, 1) if d in gold), None)
        rr += 1 / first if first else 0
        dcg = sum((2 ** gold.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked, 1))
        ideal = sum((2 ** g - 1) / math.log2(i + 1)
                    for i, g in enumerate(sorted(gold.values(), reverse=True)[:k], 1))
        ndcg += dcg / ideal if ideal else 0
    n = len(rows)
    return dict(queries=n, hits=hits, relevant=relevant, micro_recall=hits / relevant,
                macro_recall=macro / n, mrr=rr / n, ndcg=ndcg / n)


def assert_metrics(actual, expected):
    for field, value in actual.items():
        assert abs(value - expected[field]) < 1e-10, (field, value, expected[field])


def main():
    rankings = load('rankings.json')
    expected = load('results.json')['metrics']
    stages = load('stage_and_k_results.json')
    reference = None
    for arm, rows in rankings.items():
        assert len(rows) == len({r['query_id'] for r in rows}) == 300
        identity = {r['query_id']: (r['text'], r['qrels']) for r in rows}
        if reference is None:
            reference = identity
        assert identity == reference
        assert all(len(r['ranking']) == len(set(r['ranking'])) == 64 for r in rows)
        actual = metrics(rows)
        assert_metrics(actual, expected[arm]['8'])
        outside = sum(d not in r['ranking'] for r in rows for d, g in r['qrels'].items() if g > 0)
        inside = actual['relevant'] - actual['hits'] - outside
        assert outside == stages[arm]['outside_candidates']
        assert inside == stages[arm]['inside_candidates_below_8']
        for item in stages[arm]['k_curve']:
            m = metrics(rows, item['k'])
            assert m['hits'] == item['hits'] and abs(m['micro_recall'] - item['micro_recall']) < 1e-12
        print(f"{arm}: {actual['hits']}/{actual['relevant']} Recall@8={actual['micro_recall']:.4%}, "
              f"MRR@8={actual['mrr']:.4f}, nDCG@8={actual['ndcg']:.4f}; misses={outside}+{inside}")
    summaries = load('embedding_sweep_summaries.json')
    assert len(summaries) == 57
    for arm, rows in load('chinese_rankings.json').items():
        dev = [r for r in rows if r['phase'] == 'dev']
        heldout = [r for r in rows if r['phase'] == 'heldout']
        assert len(dev) == 300 and len(heldout) == 700
        assert not {r['query_id'] for r in dev} & {r['query_id'] for r in heldout}
        for phase, subset in [('dev', dev), ('heldout', heldout)]:
            actual = metrics(subset)
            assert_metrics(actual, summaries[arm]['metrics'][phase]['dense_budgets']['64']['8'])
        print(f"{arm} internal heldout: Recall@8={metrics(heldout)['micro_recall']:.4%}")
    print('Verified: five complete chains, stage/K curves, four Chinese arms and 57 aggregate snapshots.')
    print('Saved-ranking verification only; no fresh retrieval, encoding, GPU inference or latency measurement.')


if __name__ == '__main__':
    main()
