# SciFact 混合检索与 Qwen3 重排实验

本实验优化离线检索组件，覆盖完整 SciFact 的 5,183 篇文档、300 条官方测试查询和 339 个相关查询—文档对。向量与 BM25 双路召回、RRF 融合及 Qwen3-Reranker-0.6B 重排将 micro Recall@8 从原始检索的 74.34% 提升至 86.73%，MRR@8 从 0.6115 提升至 0.7544。实验完成于 2026-10-07。

## 方法与对照

沿用 all-MiniLM-L6-v2、14,940 chunk 的归一化 FAISS 索引，检索 64 个 chunk 并按文档首次命中去重。BM25 对标题与摘要使用小写英文/数字分词及 Porter stemming，k1=1.2、b=0.75，取 64 篇；两路等权 RRF，常数 60，融合后保留 64 篇。最终各方法均输出 8 篇文档。

| 策略 | micro Recall@8 | MRR@8 | nDCG@8 |
|---|---:|---:|---:|
| 原始 raw8、文档不补位 | 74.34% | 0.6115 | 0.6441 |
| raw64 文档去重 | 77.29% | 0.6139 | 0.6519 |
| 同 dense 池 MiniLM Cross-Encoder | 79.06% | 0.6520 | 0.6815 |
| BM25 | 76.70% | 0.6495 | 0.6800 |
| Dense + BM25 RRF | 81.12% | 0.6841 | 0.7160 |
| RRF 池 MiniLM Cross-Encoder | 78.76% | 0.6537 | 0.6815 |
| RRF 池 Qwen3-Reranker | **86.73%** | **0.7544** | **0.7784** |

新方案覆盖 294/339 个相关对；相较 RRF 新增 26、丢失 7，净增 19。第一名相关的查询为 208/300。micro Recall 按相关文档对汇总；MRR 按查询平均首个相关结果的倒数排名。排除历史 10 条审计查询后，290 条结果为 282/326=86.50%、MRR=0.7543、nDCG=0.7777。

## 开发集选型

从官方 train 的 809 条查询中，按 `sha256('rag-v2-dev:'+query_id)` 固定取 160 条，含 180 个相关对，与 test 查询 ID 不重叠。只比较两条预设指令，按开发 MRR、micro Recall、固定 A 优先的顺序选择，并要求两项指标都超过 RRF。

| 指令 | Dev Recall@8 | Dev MRR@8 |
|---|---:|---:|
| RRF 基线 | 84.44% | 0.7318 |
| A 通用网页检索 | 86.67% | 0.7645 |
| B 科学论断证据 | 85.56% | **0.7690** |

2026-10-07 21:24:38（北京时间）冻结 B 后才执行完整测试打分。B 为：

```text
Given a scientific claim, retrieve research papers that provide evidence supporting or refuting the claim.
```

测试评分输入只含查询、文献和候选 ID。test 已有历史使用，本轮属于同 benchmark 的后续优化；开发与测试 ID 隔离不等于从未接触过该 benchmark。公开预训练重排器的训练数据重叠情况未在本项目验证。

## 重排实现与成本

联合输入任务指令、query、标题、原 SciFact ID 和摘要，使用最后位置 `yes` 与 `no` 的 logit 差排序；左 padding、每查询按长度排序后 batch=8，BF16、SDPA、max_length=4096、use_cache=False。同分沿用 RRF 排序。完整测试 19,200 个 pair，最大输入 2,049 token，截断 0 个。

单 RTX 3090 的每查询 64 篇候选 GPU 前向平均 1.026 秒、P95 1.397 秒；排除检索、分词、加载和写文件，不是完整请求延迟。峰值 PyTorch 显存分配约 1.74 GB；不是整卡总占用。旧 MiniLM 没有在同卡测量，不能据此比较速度。

发布模型为 `Qwen/Qwen3-Reranker-0.6B`；权重 SHA256 为 `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`。来源记录在 [download_manifest.json](evaluation/scifact-qwen3-20261007/download_manifest.json)。参考环境为 Python 3.10、PyTorch 2.4.0+cu121、Transformers 5.5.4。

## 离线重算

以下命令只依赖 Python 标准库，不加载模型、不调用 API：

```bash
python scripts/recompute-scifact-metrics.py
python scripts/experiments/scifact_qwen3/verify_saved_scores.py
```

前者计算七个实验臂的 Recall/MRR/nDCG 与 290 查询子集；后者从公开逐对分数重建排序，检查开发集选型、查询隔离以及 300 条测试前 8 名与公开排名一致。附件位于 [evaluation/scifact-qwen3-20261007](evaluation/scifact-qwen3-20261007/)。

## 重新执行模型打分

自行准备公开 [BEIR SciFact](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) 的 `corpus.jsonl` 和 [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) 本地权重。附件已提供冻结的候选池，重跑分数时使用新的输出目录以保留原结果：

```bash
mkdir -p runs/scifact_qwen3/models
cp docs/evaluation/scifact-qwen3-20261007/{dev_candidates.json,test_candidates.json,selection.json} runs/scifact_qwen3/
cp /path/to/scifact/corpus.jsonl runs/scifact_qwen3/
ln -s /path/to/Qwen3-Reranker-0.6B runs/scifact_qwen3/models/Qwen3-Reranker-0.6B
python scripts/experiments/scifact_qwen3/score_candidates.py --root runs/scifact_qwen3 --phase dev
python scripts/experiments/scifact_qwen3/score_candidates.py --root runs/scifact_qwen3 --phase test
```

`score_candidates.py` 是本次实际运行的评分程序，原分数哈希与运行摘要保留在附件中。测试只执行冻结 B；开发阶段执行 A/B。共享集群应将 GPU 命令提交到计算节点。

若要重建召回候选，而非复用冻结池，还需原 14,940 chunk 的 FAISS 索引、`向量行号 -> 文档 ID` JSON 映射，以及本地 all-MiniLM-L6-v2 权重；只下载 SciFact 原始摘要不足以恢复原 chunk 索引。候选构造脚本为 `scripts/experiments/scifact_qwen3/prepare_candidates.py`，参数由 `--help` 查看。查询 ID 选择文件 `query-selection.json` 随附件公开。依赖 faiss-cpu、sentence-transformers、nltk、numpy；该脚本只用候选和查询 ID，不使用 qrels。重建后应按 ID 对照公开冻结池，不把新分块或不同 embedding 配置的成绩混入本表。

## 采用范围

该链路作为新增离线检索原型公开；当前网页默认仍为 FAISS 文档去重路线，可选网页 MiniLM 开关仍使用原模型。检索结果不直接代表报告质量或 Agent 完成率；完整 Agent 评测与以上组件成绩分别记录。
