# Hybrid Research Agent

**结合本地文库与公开 Web 的研究 Agent。**

Evidence-grounded research across the Web and your document library.

从一个技术问题出发，拆解研究任务、检索文档与网页、读取正文，最终生成带来源引用的报告。适合论文比较、技术选型，以及从论文机制到官方实现的工程调研。

项目围绕四项工程工作展开：研究 Agent 的执行控制、文档级候选检索、原文证据交接，以及资源受限的本地模型适配。提供托管模型和本地 Qwen 两条执行路线，并通过网页管理研究任务与报告。

[研究方式](#研究方式) · [快速开始](#快速开始) · [系统架构](#系统架构) · [核心设计](#核心设计) · [实验结果](#实验结果) · [文档导航](#文档导航)

## 研究方式

研究支持三种来源范围：

| 模式 | 使用的来源 | 典型任务 |
| --- | --- | --- |
| Collection | 用户导入并建立索引的文库 | 比较多篇论文的机制与取舍 |
| Web | 搜索并实际读取的公开网页 | 核对工具配置、接口和官方实现 |
| Hybrid | 文库与 Web | 将论文原理与当前实现结合起来分析 |

网页支持查看研究阶段与执行计数、取消任务、查看已保存的研究材料，以及从历史记录重新打开报告。历史验证使用公开文档构建文库，运行时可导入自己有权使用的材料。

## 快速开始

准备 **Python 3.12–3.14、Node.js 24 或以上**，以及可用的模型服务与搜索服务。以下命令使用 Python 3.12；也可换成 3.13 或 3.14。

```bash
git clone https://github.com/MaxyHE/hybrid-research-agent.git
cd hybrid-research-agent
python3.12 -m venv .venv
.venv/bin/pip install -e .
npm ci
npm run build
LDR_WEB_HOST=127.0.0.1 .venv/bin/ldr-web
```

打开启动输出中的本地地址，创建账号并配置模型与搜索服务。使用 Collection 或 Hybrid 时，先导入文档并建立文库索引，再提交研究问题。模型 API 与搜索服务可能产生费用。

源码安装保留本仓库的研究改造。macOS 的 SQLCipher 依赖、数据目录、端口和已有文库启动器见[完整运行指南](docs/QUICKSTART.md)。

本地 Qwen 通过已有的 OpenAI-compatible 模型端点接入；模型服务由用户单独部署。[连接步骤](docs/QUICKSTART.md#接入本地-qwen)说明准确模型名与路线选择。当前提供本地运行源码、配置说明和评测附件，尚无公共在线服务。

## 系统架构

```text
用户问题 + 来源范围 + 模型
              │
              ├─ 托管路线
              │  Supervisor → 并行 Researchers → Writer
              │
              └─ 本地 Qwen 路线
                 需求计划 → 检索与阅读 → 有限补缺 → 多片段综合

两条路线共用
  来源工具：Web 搜索与读取 / Collection 检索与正文读取
  运行记录：调用预算、来源身份、正文快照、任务状态与 trace
  网页交付：报告、引用、执行进度与历史记录
```

检索先回答“应该读哪些材料”，正文阅读再提供写作依据。运行时负责工具执行、额度检查和来源记录，模型负责研究决策与综合。两条路线共享产品入口，根据模型与推理资源采用不同的研究流程。

技术栈：Python、Flask、LangChain 兼容模型与工具接口、FAISS、Transformers、bitsandbytes。双卡本地推理在 Linux / Slurm 环境验证。

## 核心设计

### Agent Harness 与研究执行控制

多阶段研究需要协调任务、工具和调用资源。在上游研究控制流与预算语义的基础上，本项目将混合来源工具和运行状态接入统一运行时：Supervisor 拆分任务，Researchers 在各自的工具循环中研究，Writer 汇总交付。

- **执行前检查额度**：研究模型调用共享总额度，各任务拥有工具额度；写作阶段单独授予模型调用额度。
- **控制单轮来源动作**：模型返回多个工具调用时，按单轮上限执行，并为每个调用返回对应的工具消息，让后续决策基于已返回的结果。
- **管理任务生命周期**：记录未完成事项、阶段与调用事件；支持按用户排队和协作式取消，取消后保留已完成的笔记与来源正文。

[实际执行实例](docs/EXECUTION_EXAMPLE.md)记录了一次三任务并行研究的分发、额度分配和 Writer 交接。[任务控制说明](docs/HYBRID_TASK_PROGRESS.md)介绍进度与取消材料。

代码：[runtime.py](src/local_deep_research/odr_baseline/runtime.py) · [harness.py](src/local_deep_research/odr_baseline/harness.py) · [hybrid_task_control.py](src/local_deep_research/web/services/hybrid_task_control.py)

### 文档级候选检索

多论文研究需要覆盖不同文档。同一文档的多个相似 chunk 会占用有限候选位，因此文库接口先扩大 chunk 候选池，再按文档去重，向 Agent 返回用于后续阅读的文档候选。

```text
FAISS 检索 64 个 chunk
        ↓ 按文档去重，保留首次命中顺序
最多 8 篇候选文档
        ↓ Agent 按需读取正文
用于研究与综合的来源证据
```

可选 Cross-Encoder 在同一候选池内重排文档，默认关闭；中文问题检索英文文库时，可启用本地 Qwen 查询翻译。两项能力的模型准备、配置与测量范围见[文档重排](docs/COLLECTION_RERANKING.md)和[查询翻译](docs/COLLECTION_QUERY_TRANSLATION.md)。

代码：[project_sources.py](src/local_deep_research/odr_baseline/project_sources.py)

### 原文证据交接与上下文组织

Researcher 到 Writer 的多阶段摘要可能丢失结论对象、适用条件和原文依据。可选原文交接模式将研究结论与支持片段一起传递，附上来源身份、原文位置、结论主体、信息角色和适用条件。Researcher 可以在已读正文中定位关键词并展开相邻段落，完整快照保留供回查。

后续研究上下文使用带来源 ID 与 URL 的工作笔记，减少重复携带正文；写作时按所选模式接收压缩笔记或带位置的原文证据。报告、来源和运行轨迹共同支持对发现、阅读、交接及写作环节的检查。

托管路线默认使用压缩笔记，设置 `LDR_HYBRID_EVIDENCE_HANDOFF=located` 后启用原文交接。本地 Qwen 使用自己的证据流程。用法与评测见[原文交接说明](docs/EVIDENCE_HANDOFF.md)。

代码：[located_handoff.py](src/local_deep_research/odr_baseline/located_handoff.py) · [sources.py](src/local_deep_research/odr_baseline/sources.py)

### 资源受限的本地模型适配

本地路线将用户问题拆成可独立回答的需求，为需求指定文库或 Web 来源，再执行检索、阅读和有限补缺。Writer 综合多个正文片段，并分别限制片段长度、单来源长度和总证据上下文，控制本地推理负担。

该流程在 **27B、双卡 4-bit** 部署环境进行验证，与托管路线的层级式多智能体编排分开实现。网页通过准确模型名选择 Qwen 路线，两条路线共用来源选择、报告与历史入口。

代码：[qwen_candidate.py](src/local_deep_research/odr_baseline/qwen_candidate.py) · [qwen_writer.py](src/local_deep_research/odr_baseline/qwen_writer.py)

## 实验结果

### 检索组件对照

完整 SciFact 语料包含 **5,183 篇文档、300 条官方测试查询**。使用同一 embedding 与 FAISS 索引，按最终文档排名计算指标：

| 策略 | micro Recall@8 | MRR@8 |
| --- | ---: | ---: |
| 原始 top 8 chunks，文档不补位 | 74.34% | 0.6115 |
| top 64 chunks → 文档去重 → 8 篇 | 77.29% | 0.6139 |
| 同一 top 64 池 → MiniLM Cross-Encoder → 8 篇 | 79.06% | 0.6520 |
| 向量 + BM25 → RRF → 8 篇 | 81.12% | 0.6841 |
| RRF 候选池 → Qwen3-Reranker → 8 篇 | **86.73%** | **0.7544** |

最新离线实验采用向量与 BM25 双路召回、文档去重、RRF 融合及 Qwen3-Reranker-0.6B 重排；相较原始检索，Recall@8 从 74.34% 提升至 **86.73%**，MRR@8 从 0.6115 提升至 **0.7544**。重排指令在 160 条与测试查询不重叠的开发查询上选定并冻结。该实验是检索组件原型，当前网页默认链路仍为上述 FAISS 文档去重路线；不据此推定完整 Agent 的报告质量。实验设置与 GPU 成本见[混合检索与 Qwen3 重排实验](docs/SCIFACT_QWEN3_RERANKING.md)。[结果说明](docs/RESULTS.md)包含 nDCG、60 查询开发对照、中文查询翻译及模型选型结果。

公开排名附件支持离线重算指标，只需 Python 标准库，无需模型或 API：

```bash
python3 scripts/recompute-scifact-metrics.py
```

该命令从保存的排名重新计算成绩；重新执行检索还需准备语料、索引和模型。详见[评测附件](docs/evaluation/README.md)。

### 研究交付与证据交接

原文交接候选在 12 个固定任务的保存证据上重新组织材料并写作，按 60 个原题检查项获得 **51.5/60（85.8%）**。评分采用对照保存来源的 LLM 评审，包含未生成报告的任务。已生成报告中的引用目标全部匹配已读来源目录，内容支持程度由质量评审另行检查。

另一次新增 Self-RAG 工程接入调研实际执行文库与 Web 检索、阅读和报告生成，来源复核中 **5/5 核心要求满足**。保存证据写作评测和新增端到端单题分别记录，见[证据交接评测](docs/EVIDENCE_HANDOFF.md)。

### 本地模型验证

冻结 Qwen v6.2 的 30 题固定评测覆盖 Web、Collection、Hybrid，各 10 题：

| 测量内容 | 结果 |
| --- | --- |
| 质量完成率，首次尝试 | 17/30（56.7%） |
| 质量完成率，指定连接故障恢复后 | 18/30（60.0%） |
| 10 个文库任务中的指定论文发现与读取 | 25/25 个任务—论文目标 |

这些结果分别描述报告质量和来源获取。当前网页接入的是后续 Qwen 流程，完整 Hybrid 交付仍待验收；30 题成绩对应上述冻结版本。版本、质量判定与恢复记录见[完整结果](docs/RESULTS.md)和[历史运行记录](docs/DEMO_RUNS.md)。

## 文档导航

| 你想了解什么 | 入口 |
| --- | --- |
| 安装、文库配置与本地模型连接 | [快速开始](docs/QUICKSTART.md) |
| 模块职责、贡献范围与代码入口 | [架构说明](docs/ARCHITECTURE.md) |
| 一次真实研究如何运行 | [执行实例](docs/EXECUTION_EXAMPLE.md) |
| 历史网页运行与内容复核 | [运行记录](docs/DEMO_RUNS.md) |
| 实验设置、指标与版本 | [结果说明](docs/RESULTS.md) · [离线重算](docs/evaluation/README.md) |
| 当前集成与验证状态 | [发布状态](docs/RELEASE_STATUS.md) |
| 仓库材料与公开附件范围 | [材料导航](docs/REPOSITORY_GUIDE.md) |

## 上游来源与许可

应用基础采用 [LearningCircuit/local-deep-research](https://github.com/LearningCircuit/local-deep-research)，研究控制流与部分提示词改编自 [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)，预算与未完成任务语义适配自 [jmlon/deep-research-harness](https://github.com/jmlon/deep-research-harness/tree/393d907239ee649f85dd888de715d8379e8b4a87)。

本项目的改造集中于混合来源运行时集成、文档级候选接口、证据交接、本地模型执行适配、网页任务控制及工程评测。上游作者和 MIT 许可信息保留在 [LICENSE](LICENSE)、[ODR LICENSE](src/local_deep_research/odr_baseline/LICENSE)、[Harness LICENSE](src/local_deep_research/odr_baseline/HARNESS_LICENSE) 与 [NOTICE](src/local_deep_research/odr_baseline/NOTICE.md) 中；同步来源见[源码来源记录](docs/PROVENANCE.md)。

仓库提供源码、配置说明、历史运行记录和精简评测附件。账号凭据、用户数据库、模型权重及完整来源正文快照保存在仓库之外。
