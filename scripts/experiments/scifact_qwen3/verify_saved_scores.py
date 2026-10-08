"""Verify published dev selection and test reranking using only the standard library."""
import json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'docs/evaluation/scifact-qwen3-20261007'

def load(name):
    return json.loads((DATA/name).read_text())

def metrics(rows, gold):
    hits = targets = 0
    reciprocal = []
    for q, ranked in rows.items():
        relevant = {d for d,g in gold[q].items() if g > 0}
        top = ranked[:8]
        ranks = [i+1 for i,d in enumerate(top) if d in relevant]
        hits += len(ranks)
        targets += len(relevant)
        reciprocal.append(1/min(ranks) if ranks else 0)
    return hits/targets, statistics.mean(reciprocal)

def rankings(phase, arm):
    candidates = {r['query_id']:r['candidate_ids'] for r in load(phase+'_candidates.json')['rows']}
    records = [json.loads(line) for line in (DATA/(phase+'_scores.jsonl')).read_text().splitlines()]
    assert len(records)==len(candidates)
    result = {}
    for row in records:
        q=row['query_id']; pool=candidates[q]
        assert q not in result and len(pool)==len(set(pool))==64
        scores = row['arms'][arm]['scores']
        values = {r['document_id']:r['logit_yes_minus_no'] for r in scores}
        assert len(values)==len(scores)==64 and set(values)==set(pool)
        assert all(math.isfinite(v) for v in values.values())
        tie={d:i for i,d in enumerate(pool)}
        result[q]=sorted(pool,key=lambda d:(-values[d],tie[d]))
    return result

def main():
    sel=load('selection.json')
    gold=load('dev_qrels.json')
    # The saved mapping is query_id -> document_id -> positive grade.
    dev={arm:metrics(rankings('dev',arm),gold) for arm in ('A','B')}
    rrf={r['query_id']:r['candidate_ids'] for r in load('dev_candidates.json')['rows']}
    baseline=metrics(rrf,gold)
    chosen=max(('A','B'),key=lambda a:(dev[a][1],dev[a][0],a=='A'))
    assert chosen==sel['selected_arm'] and all(a>b for a,b in zip(dev[chosen],baseline))
    for arm in ('A','B'):
        assert abs(dev[arm][0]-sel['dev_metrics'][arm]['recall_micro_at_8'])<1e-12
        assert abs(dev[arm][1]-sel['dev_metrics'][arm]['mrr_at_8'])<1e-12
    test=rankings('test',chosen)
    frozen=json.loads((ROOT/'docs/evaluation/scifact-300-rankings.json').read_text())['rows']
    assert set(test)=={r['query_id'] for r in frozen}
    assert not set(test)&set(sel['selected_query_ids'])
    for r in frozen:
        assert test[r['query_id']][:8]==r['rankings']['rrf_qwen3_rerank8']
    recall,mrr=metrics(test,{r['query_id']:r['qrels'] for r in frozen})
    assert abs(recall-load('summary.json')['metrics']['all_300']['Qwen']['recall_micro_at_8'])<1e-12
    assert abs(mrr-load('summary.json')['metrics']['all_300']['Qwen']['mrr_at_8'])<1e-12
    print(f'Dev-selected {chosen}; 300/300 saved test rankings match scores; Recall@8={recall:.4%}, MRR@8={mrr:.4f}')

if __name__=='__main__':
    main()
