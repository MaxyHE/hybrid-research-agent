# 检索选型、索引取舍与失败分析

2026-10-14至16日追加实验，2026-10-17整理发布。实验回答三个问题：换embedding后，收益能否保留到最终重排？语言与文本长度怎样影响召回？剩余错误发生在哪个阶段？

## 数据与不变条件

英文使用完整BEIR SciFact的5,183篇标题摘要、14,940个原chunk、300条历史官方测试查询和339个相关文档关系。160条开发查询来自官方train，与test ID隔离。该测试集已有历史使用，属于追加优化，不是新盲测。相关性按文档ID评分，不按chunk命中评分。

各完整链路分别以自己的向量召回和BM25生成文档候选，经等权RRF（常数60）融合到64篇，再由同一Qwen3-Reranker-0.6B按科学证据指令重排到8篇。候选池并集只用于复用逐对打分计算，不向任何单臂额外提供候选。评分器使用BF16、SDPA、batch=8、max_length=4096、use_cache=False。

BGE无query instruction版本由第一阶段开发MRR预选；标准指令版本作为推荐格式对照。随后Qwen的chunk与整文档视图也分别按既有开发MRR选定配置，不用本轮test结果重新挑指令或输出维度。Qwen“整文档”只指完整标题摘要，不是论文全文。

## 五条完整链路

| Embedding配置 | 向量维度 | Recall@8 | MRR@8 | nDCG@8 |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2，原chunk／256上限 | 384 | 294/339，86.73% | 0.7493 | 0.7748 |
| BGE-base-en-v1.5，无query instruction，开发预选 | 768 | **298/339，87.91%** | 0.7483 | 0.7768 |
| BGE-base-en-v1.5，标准query instruction | 768 | **298/339，87.91%** | 0.7513 | 0.7792 |
| Qwen3-Embedding-0.6B，原chunk／领域指令 | 1024 | 297/339，87.61% | 0.7478 | 0.7757 |
| Qwen3-Embedding-0.6B，标题摘要整文档／领域指令 | 1024 | 297/339，87.61% | 0.7471 | 0.7752 |

开发预选BGE比同环境MiniLM多覆盖4个关系，Recall增加1.18个百分点，MRR没有改善。因此采用较高覆盖工作点有依据，但不能说所有排序指标显著提高。两种Qwen配置都未超过BGE，MRR也未超过MiniLM；更大的embedding在本任务中没有带来更好的最终链路表现。

本轮Qwen相对MiniLM的Recall差值，经2000次配对查询bootstrap得到的95%区间约为−0.30至+2.26个百分点，区间跨0。完整配对区间见[结果附件](evaluation/retrieval-followup-20261016/results.json)。新评分新增5,553个唯一pair，并复核690个原缓存pair，logit最大绝对差为0。

**历史与本轮环境分开：**2026-10-07 MiniLM完整链路MRR为0.7544；本轮同环境重评分为0.7493，命中仍为294/339。上述五臂统一使用本轮分数，不将不同环境的最好指标拼接。旧组件增量与其约1秒/查询GPU前向计时保留在[历史实验](SCIFACT_QWEN3_RERANKING.md)；本轮没有新的全链路延迟排名。

## 分块与长度：57组探索的主要发现

探索共57组，覆盖MiniLM、E5、BGE和Qwen的分块、长度、前缀／指令、语言及输出维度。每组开发／测试指标、token长度、截断、向量大小、编码设备和精度见[聚合快照](evaluation/retrieval-followup-20261016/embedding_sweep_summaries.json)，[完整表格](evaluation/retrieval-followup-20261016/embedding-sweep.md)便于逐组阅读。

- 原1000字符、200重叠的14,940个chunk中，MiniLM有480个超过256 token（3.21%）。编码上限改512后，纯向量文档Recall@8仍为77.29%。长度风险真实存在，但不能将全部漏检归因于截断。
- 更短token分块能消除截断，未稳定提高召回；需同时考虑段落语义与重复候选。每组按相同文档标签评分。
- 纯向量BGE标准指令版为84.96%，MiniLM为77.29%；经同一重排器后的差距明显缩小。粗召回收益不等于最终收益。
- Qwen输出1024／512／384维使用前缀维度裁剪并重新归一化。减少存储与搜索维度不减少已执行的1024维编码前向计算。
- 编码设备与精度不同，增量缓存也不同；不能把本批构建耗时作为模型速度排行榜。索引大小是序列化字节数，不是进程总内存。

## 中文医学问答检索

使用公开C-MTEB MedicalRetrieval的100,999条问答文本记录作为检索库；不是十万篇论文。将官方dev按固定规则拆为本实验300条开发和700条内部留出查询，不冒称官方test或外部独立队列。每条查询有一个标注相关目标，各模型使用相同原文档；前64个向量命中按文档去重后取8篇，不加本轮英文重排器。

| 模型 | 700条内部留出 Recall@8 |
|---|---:|
| all-MiniLM-L6-v2 | 2.43% |
| multilingual-e5-small | 50.29% |
| bge-small-zh-v1.5 | 57.29% |
| bge-large-zh-v1.5 | 67.29% |

该结果支持当前中文问答任务上的语言选型差异；不能外推所有中文场景。MiniLM的语言适配与长度限制存在混杂，统一token分块的补充结果仍见57组表。四臂公开ID、qrels和排名见[chinese_rankings.json](evaluation/retrieval-followup-20261016/chinese_rankings.json)，不重新分发完整医学问答库。

## FAISS Flat与HNSW

同一批中文E5向量，100,999条、384维、CPU单线程、128条查询。只计索引搜索，不含embedding与重排。HNSW的M=32、efConstruction=80。

| 索引 | efSearch | P50 / P95 | 序列化大小 | 相对Flat前8近邻重合率 |
|---|---:|---:|---:|---:|
| Flat | — | 5.80 / 8.25 ms | 155.13 MB | 100% |
| HNSW | 32 | 0.156 / 0.203 ms | 182.63 MB | 72.46% |
| HNSW | 64 | 0.266 / 0.358 ms | 182.63 MB | 82.32% |
| HNSW | 128 | 0.503 / 0.689 ms | 182.63 MB | 89.94% |

HNSW通过更多空间和近邻损失降低搜索耗时。表中89.94%是ANN结果相对精确Flat的重合率，**不是语义相关文档Recall**。更小规模和完整数值见[index_zh_e5.json](evaluation/retrieval-followup-20261016/index_zh_e5.json)。没有据此把默认FAISS切换为HNSW。

## 失败定位与阅读预算

主BGE配置漏41个关系：14个不在64篇候选池、27个在池内但低于前8。MiniLM漏45个，分解为19+26，两者不可混用。

固定BGE的64篇排名，仅扩大最终K：4／8／12／16／32／64对应Recall为79.65%／87.91%／90.56%／92.33%／95.58%／95.87%。K=8到12找回9个关系，但增加阅读量；14个池外关系无法靠扩大K补回。此曲线是历史测试上的事后敏感性分析，不用于宣布新的最优K；没有重测阅读token或报告延迟。

[内容级失败分析](RETRIEVAL_FAILURE_ANALYSIS.md)逐条讨论表达差异、关系排序、实体与机制、多目标覆盖和标注语义。八个案例是定向开发诊断，不能推算全部漏检原因比例；改写查询、实体扩展等后续方案尚未执行。

## 离线重算与采用范围

```bash
python scripts/recompute-retrieval-followup.py
```

仅依赖Python标准库，从[公开附件](evaluation/retrieval-followup-20261016/README.md)重算五臂Recall／MRR／nDCG、池外／池内分解、K曲线，以及四条中文链路的开发和留出指标。57组完整聚合快照与索引测量为归档结果；仅随附排名的部分可通过此命令重新评分。

这批发布更新离线实验与复核附件。网页默认仍使用原FAISS文档去重接口，可选MiniLM重排沿用原开关；BGE与Qwen新链路没有自动替换产品默认配置。没有重测完整报告质量，也没有新增单／多Agent架构因果消融。

来源：[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)、[BGE](https://huggingface.co/BAAI/bge-base-en-v1.5)、[Qwen Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)、[Qwen Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)、[MedicalRetrieval](https://huggingface.co/datasets/C-MTEB/MedicalRetrieval)、[FAISS索引](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)。
