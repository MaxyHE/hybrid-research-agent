# Hybrid Research Agent

**面向技术调研与复现准备的混合来源 Research Agent。**

连接用户文库与公开 Web，完成从问题拆解、资料发现、正文阅读到引用报告的研究流程。围绕 **层级式 Multi-Agent、Agent Harness、文档级检索与资源受限推理**，构建托管大模型和本地 Qwen 两条执行路线。演示使用公开文档模拟用户文库，产品不限于论文研究。

**Agent Harness · Tool Calling · RAG / FAISS · Evidence-grounded Research · Local LLM**

[主要结果](#主要结果) · [工程亮点](#工程亮点) · [架构与代码](#架构与代码) · [运行准备](docs/QUICKSTART.md) · [评测口径](docs/RESULTS.md)

## 适合解决什么问题

技术研究往往需要两类信息：相对稳定的论文与文档，以及持续变化的代码、数据和官方说明。Hybrid Research 将两者组织到同一研究任务中。

| 研究方式 | 典型问题 | 交付目标 |
| --- | --- | --- |
| Collection | 比较 ReAct、ReWOO 与 LATS 的控制环及取舍 | 基于文库原文的多论文比较 |
| Hybrid | 理解一篇论文，并核对官方仓库的复现准备 | 论文机制与当前实现的来源关联报告 |
| Web | 核对技术工具的配置、接口与使用条件 | 基于实际读取网页的研究结论 |

用户关注 **问题、来源、模型**；运行时负责研究编排、工具执行、预算和来源记录。

## 主要结果

| 维度 | 实测成果 | 评测范围 |
| --- | --- | --- |
| **文档候选覆盖** | Recall@8 **74.34% → 77.29%** | 完整 SciFact：5,183 篇文档、300 条官方查询；raw8 → raw64 / Document8 |
| **固定候选池重排** | Recall@8 **77.29% → 79.06%**；MRR@8 **0.6139 → 0.6520** | 同一 raw64 池内 Cross-Encoder 重排，最终最多 8 篇文档 |
| **研究交付** | 质量完成率 **91.7%（11/12）** | 三种来源各 4 题，指定官方来源条件，按预选连接故障位置恢复 |
| **本地推理工程** | **27B、双卡 4-bit、25/25 指定论文目标发现并读取** | 30题评测中10个文库任务的任务—论文目标 |

检索指标由程序计算；报告质量按固定检查项与保存来源进行助手离线评审。上述重排结果来自研发评测，研发入口已可选接入，本发布副本尚未同步该功能。30 题表示本地评测规模。版本、分层结果、完整质量表现及成本见 [评测说明](docs/RESULTS.md)。

## 工程亮点

### 1. Agent Harness：把模型决策变成可管理的研究执行

将研究控制流、工具调用与运行状态组织在同一运行时中：

- **层级式多智能体**：Supervisor 派发任务，Researchers 在独立工具循环中研究，线程池支持并行，Writer 综合交付。
- **调用预算控制**：模型与工具执行前检查计数和额度，为写作单独授予调用额度；不是仅在提示词中要求节省调用。
- **工具执行控制**：限制单轮实际来源动作，按任务分配工具额度，记录未完成研究事项。
- **执行可追踪**：记录任务状态、未完成事项、模型与工具调用，为诊断和版本对照提供依据。

核心代码：[runtime.py](src/local_deep_research/odr_baseline/runtime.py) · [harness.py](src/local_deep_research/odr_baseline/harness.py)

### 2. Document 级检索：让候选位覆盖更多研究对象

多文档调研需要的是不同文档，而不仅是高相似度片段。将原始 chunk 候选池扩大至 64，按 Document 保留首次命中，再向 Agent 返回最多 8 篇文档；Agent 按需回读正文。

```text
向量检索：64 个 chunk
         ↓ 按 Document 去重，保留首次排名
研究候选：最多 8 篇不同文档
         ↓ 按需读取
正文证据：供后续研究与综合使用
```

在固定 60 查询对照中，新增 42 条查询的指定目标召回从 **72.7% 提升至 83.6%**；原 18 条开发查询保持目标召回。提升主要体现在补齐多文档研究需求。

核心代码：[project_sources.py](src/local_deep_research/odr_baseline/project_sources.py)

中文问题检索英文文库时，可启用独立的本地 Qwen 翻译。固定开发集指定目标 Recall@8 进一步达到 100%，翻译耗时中位数 4.44 秒；英文跳过、重复译文复用。见[启用说明](docs/COLLECTION_QUERY_TRANSLATION.md)。[公开排名与指标重算](docs/evaluation/README.md)无需 API 或私有数据库。

### 3. 证据交接：连接“读过什么”与“写出了什么”

来源身份、实际读取正文与报告引用贯穿研究过程。托管 H-off 路线由 Researcher 返回压缩研究笔记，Writer 接收原问题、任务列表、已读来源清单、笔记及未完成事项；Qwen 路线采用有限原文片段交接。完整来源快照留存供回查，笔记交接不等同于逐条事实校验。

报告、来源与轨迹形成一组可回看的交付产物，便于定位问题发生在发现、读取、信息交接还是最终写作。

核心代码：[sources.py](src/local_deep_research/odr_baseline/sources.py) · [qwen_writer.py](src/local_deep_research/odr_baseline/qwen_writer.py)

### 4. 本地模型适配：相同研究目标，不同执行策略

托管路线采用多阶段研究编排。本地 Qwen 路线针对推理资源组织需求计划、有限证据补充与多片段综合，配合双卡 4-bit 部署和固定任务评测，迭代上下文使用与研究流程。

核心代码：[qwen_candidate.py](src/local_deep_research/odr_baseline/qwen_candidate.py)

## 架构与代码

```text
用户问题 + 来源范围 + 模型
          │
          ├─ 托管：Supervisor → Researchers → Writer
          └─ 本地：需求计划 → 检索阅读 → 有限补缺 → 综合
                              │
                    Web / Collection 来源工具
                              │
                    正文快照 + 来源身份 + 预算
                              │
                       报告 + 引用 + trace
```

| 入口 | 看什么 |
| --- | --- |
| [研究运行时](src/local_deep_research/odr_baseline/) | 编排、预算、混合来源与本地策略 |
| [网页研究服务](src/local_deep_research/web/services/hybrid_odr_service.py) | 产品界面到研究流程的接线 |
| [架构说明](docs/ARCHITECTURE.md) | 模块职责与代码导航 |
| [结果说明](docs/RESULTS.md) | 指标定义、对照范围与版本 |
| [仓库内容规划](docs/REPOSITORY_GUIDE.md) | 当前材料与后续演示、复现附件 |

技术栈：**Python · Flask · LangChain 兼容模型/工具接口 · FAISS · Transformers · bitsandbytes · Linux / Slurm**。

## 运行与演示

[运行准备](docs/QUICKSTART.md)提供源码安装、前端构建和文库配置步骤。已完成独立 Python 环境安装、前端构建与浏览器登录/注册页访问；完整研究交付验证见[发布状态](docs/RELEASE_STATUS.md)。[真实报告节选](docs/examples/hybrid-report-excerpt.md)展示论文与官方实现相结合的交付形式。

网页默认使用托管研究控制流。设置 `LDR_HYBRID_QWEN_MODEL` 为本地端点的准确模型名后，选择该模型会进入 Qwen 需求驱动证据流程；两条路线共用来源选择、报告和历史入口。连接方法见[快速开始](docs/QUICKSTART.md)。本地 30 题成绩对应冻结 v6.2，后续开发代码与旧版成绩分别记录。

[真实网页演示与截图](docs/DEMO_RUNS.md)：RAPTOR 本地论文 × 官方 GitHub，包含完整执行记录、引用链接修复及报告复核说明。

## 开源来源与许可

应用基础采用 [local-deep-research](https://github.com/LearningCircuit/local-deep-research)，研究控制流与部分提示词改编自 [Open Deep Research](https://github.com/langchain-ai/open_deep_research)，部分预算语义参考 deep-research-harness。

本项目工作集中于混合来源研究运行时、Document 级候选接口、约束与证据交接、本地资源适配及工程评测。保留上游作者与许可证，见 [LICENSE](LICENSE)、[ODR LICENSE](src/local_deep_research/odr_baseline/LICENSE) 和 [来源记录](docs/PROVENANCE.md)。
