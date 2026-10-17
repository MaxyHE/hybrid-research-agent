# 2026-10-14／16检索追加实验附件

[实验问题、条件与结果](../../RETRIEVAL_FOLLOWUP.md) · [内容级失败分析](../../RETRIEVAL_FAILURE_ANALYSIS.md)

| 文件 | 内容与可复核范围 |
|---|---|
| rankings.json | 五条完整英文链路，300查询、qrels与64篇最终排名 |
| results.json / protocol.json | 聚合指标、Qwen追加评分的冻结条件与配对区间 |
| selection_by_dev.json | 第一批与Qwen探索各自的开发选择，不事后重选 |
| stage_and_k_results.json / all_missed_relations.json | 全部五臂的池内／池外关系和K敏感性 |
| semantic_cases.json | 八个诊断案例的query、文档ID、题名与名次；摘要从原数据获取 |
| chinese_rankings.json | 四个模型的300开发／700内部留出qrels与文档排名 |
| embedding_sweep_summaries.json / embedding-sweep.md | 57组聚合快照与可读表格，不含全部向量或逐对分数 |
| index_zh_e5.json | Flat／HNSW搜索计时、索引字节数和近邻重合率 |
| hit_transitions.json | Qwen与参照模型在相关关系上的新增与丢失 |

```bash
python scripts/recompute-retrieval-followup.py
```

此命令从保存排名重算指标，不重新执行模型、索引构建、ANN搜索或语义判读。图像、完整语料、权重、服务器配置与内部日志不在附件中。原始数据许可与使用条件沿用各数据提供方，不将数据集归为本仓库原创代码。
