# Hybrid Research Agent

**面向技术调研与复现准备的混合来源 Research Agent。**

将用户文库中的论文、技术文档与公开 Web 上的官方实现连接起来，完成从问题拆解、资料发现、正文阅读到引用报告的研究流程。围绕 **Agent Harness、文档级检索与资源受限推理**，构建托管大模型和本地 Qwen 两条执行路线。

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
| **目标文档召回** | Recall@8 **80.3% → 88.2%（+7.9 个百分点）** | 两套语料、60 条固定查询，指定目标 micro Recall |
| **候选多样性** | 不同文档候选 **3.85 → 7.73，约 2 倍** | 同一查询集，输出候选上限为 8 |
| **候选去重** | 输出文档重复率 **51.9% → 0%** | 原 raw8 对比 raw64 → Document 去重 |
| **研究交付** | 质量完成率 **91.7%（11/12）** | 三种来源各 4 题，托管模型故障恢复视图 |
| **本地推理工程** | **27B、双卡 4-bit、30 题固定评测** | Web / Collection / Hybrid 各 10 题 |

检索指标由程序计算；报告质量按固定检查项与保存来源进行助手离线评审。30 题表示本地评测规模。版本、分层结果、完整质量表现及成本见 [评测说明](docs/RESULTS.md)。

## 工程亮点

### 1. Agent Harness：把模型决策变成可管理的研究执行

将研究控制流、工具调用与运行状态组织在同一运行时中：

- **多阶段编排**：Supervisor 派发研究任务，Researchers 搜索与读取，Writer 综合交付。
- **预算管理**：分别管理研究调用、写作 allowance、工具调用与任务容量。
- **约束传递**：将来源范围和工具能力同步到规划上下文，减少计划与执行脱节。
- **执行可追踪**：记录任务状态、未完成事项、模型与工具调用，为诊断和版本对照提供依据。

核心代码：[runtime.py](src/local_deep_research/odr_baseline/runtime.py) · [harness.py](src/local_deep_research/odr_baseline/harness.py)

### 2. Document 级检索：让候选位覆盖更多研究对象

多论文调研需要的是不同文档，而不仅是高相似度片段。将原始 chunk 候选池扩大至 64，按 Document 保留首次命中，再向 Agent 返回最多 8 篇文档；Agent 按需回读正文。

```text
向量检索：64 个 chunk
         ↓ 按 Document 去重，保留首次排名
研究候选：最多 8 篇不同文档
         ↓ 按需读取
正文证据：供后续研究与综合使用
```

在固定 60 查询对照中，新增 42 条查询的指定目标召回从 **72.7% 提升至 83.6%**；原 18 条开发查询保持目标召回。提升主要体现在补齐多文档研究需求。

核心代码：[project_sources.py](src/local_deep_research/odr_baseline/project_sources.py)

### 3. 证据交接：连接“读过什么”与“写出了什么”

来源身份、实际读取正文与报告引用贯穿研究过程。完整来源快照用于回查，有限证据窗口与多片段上下文用于模型综合；通过来源句柄与链接转换保留引用关系。

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

[运行准备](docs/QUICKSTART.md)提供源码安装、前端构建和文库配置步骤。当前为本地发布候选版，独立安装验证与可分享演示正在整理。

网页服务使用托管研究控制流；选择本地模型不自动切换到 Qwen 专用候选策略。本地 30 题成绩对应冻结 v6.2，后续开发代码与旧版成绩分别记录。

## 开源来源与许可

应用基础采用 [local-deep-research](https://github.com/LearningCircuit/local-deep-research)，研究控制流与部分提示词改编自 [Open Deep Research](https://github.com/langchain-ai/open_deep_research)，部分预算语义参考 deep-research-harness。

本项目工作集中于混合来源研究运行时、Document 级候选接口、约束与证据交接、本地资源适配及工程评测。保留上游作者与许可证，见 [LICENSE](LICENSE)、[ODR LICENSE](src/local_deep_research/odr_baseline/LICENSE) 和 [来源记录](docs/PROVENANCE.md)。
