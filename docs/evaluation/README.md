# 离线重算检索成绩

## SciFact300 公开排名附件

```bash
python scripts/recompute-scifact-metrics.py
```

`scifact-300-rankings.json` 保存 300 个 SciFact 官方测试查询的 qrels、历史组件审计标记，以及七个实验臂最终的 Document 前 8 名。脚本只使用 Python 标准库，离线重算 micro Recall@8、MRR@8 与 nDCG@8；不会调用模型或 API。

全 300 题的结果依次为 raw8 / document8_baseline / cross_encoder_rerank8：micro Recall@8 为 74.34% / 77.29% / 79.06%，MRR@8 为 0.6115 / 0.6139 / 0.6520，nDCG@8 为 0.6441 / 0.6519 / 0.6815。脚本也报告排除历史组件审计 10 题后的 290 题子集。

该附件复现的是已保存排名上的评分算术，不是重新运行检索；它不包含检索语料、原始 trace 或模型权重。

## 新增混合召回与 Qwen3 重排附件

2026-10-07 的最新方案为 Recall@8 **86.73%**、MRR@8 **0.7544**、nDCG@8 **0.7784**；排除历史 10 条审计查询后分别为 **86.50% / 0.7543 / 0.7777**。上述重算脚本同时输出 BM25、RRF、RRF+旧 MiniLM 及 RRF+Qwen3 四个新增实验臂。

[scifact-qwen3-20261007/](scifact-qwen3-20261007/)保存开发选型、开发/测试候选池、逐对 yes/no 分数、运行摘要和来源哈希。query/document ID 均来自公开 SciFact；模型权重、全文语料和私有数据库不在附件中。独立检验已发布分数与排名的一致性：

```bash
python scripts/experiments/scifact_qwen3/verify_saved_scores.py
```

重新执行 GPU 打分与准备候选的要求见[复现步骤](../SCIFACT_QWEN3_RERANKING.md)。

## 既有 60 查询附件

```bash
python scripts/recompute-retrieval-metrics.py
```

`retrieval-60-rankings.json` 及其脚本保留用于同 60 查询的原文/自动翻译开发对照。该附件同样只复现评分算术，不重新运行检索。
