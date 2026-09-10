# 离线重算检索成绩

## SciFact300 公开排名附件

```bash
python scripts/recompute-scifact-metrics.py
```

`scifact-300-rankings.json` 保存 300 个 SciFact 官方测试查询的 qrels、历史组件审计标记，以及三个实验臂最终的 Document 前 8 名。脚本只使用 Python 标准库，离线重算 micro Recall@8、MRR@8 与 nDCG@8；不会调用模型或 API。

全 300 题的结果依次为 raw8 / document8_baseline / cross_encoder_rerank8：micro Recall@8 为 74.34% / 77.29% / 79.06%，MRR@8 为 0.6115 / 0.6139 / 0.6520，nDCG@8 为 0.6441 / 0.6519 / 0.6815。脚本也报告排除历史组件审计 10 题后的 290 题子集。

该附件复现的是已保存排名上的评分算术，不是重新运行检索；它不包含检索语料、原始 trace 或模型权重。

## 既有 60 查询附件

```bash
python scripts/recompute-retrieval-metrics.py
```

`retrieval-60-rankings.json` 及其脚本保留用于同 60 查询的原文/自动翻译开发对照。该附件同样只复现评分算术，不重新运行检索。
