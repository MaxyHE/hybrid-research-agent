# 离线重算检索成绩

```bash
python scripts/recompute-retrieval-metrics.py
```

不调用模型、不下载语料、不需要用户账号。输入为同60查询的原文/自动翻译两侧已保存Document排名、指定目标与查询；脚本重算micro Recall@8和Document MRR@8。

本附件复现的是评分算术，不是重新运行检索。重新运行需要原索引或按原协议准备的20篇论文与76篇SciFact摘要；这里不打包原文或个人数据库。该数据来自2026-09-06自动翻译开发对照，研发存档1ed877d。
