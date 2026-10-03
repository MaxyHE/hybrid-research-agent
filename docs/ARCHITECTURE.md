# 架构与代码导航

本仓库在 local-deep-research 应用基础上集成混合来源研究流程。文库、设置、账号与网页等应用能力沿用上游基础；项目改造集中在研究编排与来源接线、文档候选检索、证据交接、本地模型适配及网页任务控制。具体来源与许可证见 [NOTICE](../src/local_deep_research/odr_baseline/NOTICE.md)。

## 两条研究路线

托管路线由 Supervisor 分解问题，Researchers 在各自的工具循环中获取资料，Writer 综合报告。本地 Qwen 路线按独立需求安排来源，执行检索、阅读与有限补缺，再通过限长的多片段上下文写作。

两条路线共用网页中的来源选择、报告与历史入口，也共用来源连接器和记录机制。来源发现提供阅读候选，正文读取提供证据，后续综合使用来源身份将报告引用关联到实际读取材料。

## 模块职责

| 模块 | 代码入口 | 职责 |
| --- | --- | --- |
| 托管研究运行时 | [runtime.py](../src/local_deep_research/odr_baseline/runtime.py) | 任务调度、模型与工具调用、工作上下文和报告交付 |
| 调用预算与未完成事项 | [harness.py](../src/local_deep_research/odr_baseline/harness.py) | 研究调用额度、写作额度及未完成任务语义 |
| 混合来源接线 | [hybrid.py](../src/local_deep_research/odr_baseline/hybrid.py) | Web 与 Collection 来源工具和策略配置 |
| 文库候选与读取 | [project_sources.py](../src/local_deep_research/odr_baseline/project_sources.py) | chunk 候选池、文档去重、可选重排与正文读取 |
| 来源身份与记录 | [sources.py](../src/local_deep_research/odr_baseline/sources.py) | 发现资源、已读来源与证据数据结构 |
| 原文证据交接 | [located_handoff.py](../src/local_deep_research/odr_baseline/located_handoff.py) | 已读正文定位、带位置的原文证据与 Writer 交接 |
| 本地需求流程 | [qwen_candidate.py](../src/local_deep_research/odr_baseline/qwen_candidate.py) | 需求计划、来源分配和有限证据补缺 |
| 本地多片段写作 | [qwen_writer.py](../src/local_deep_research/odr_baseline/qwen_writer.py) | 片段选择、来源与总上下文限长、综合报告 |
| 网页研究服务 | [hybrid_odr_service.py](../src/local_deep_research/web/services/hybrid_odr_service.py) | 用户问题、来源范围、模型路线与研究运行时接线 |
| 进度与取消材料 | [hybrid_task_control.py](../src/local_deep_research/web/services/hybrid_task_control.py) | 阶段事件、执行计数及取消后的材料保存与读取 |

## 默认配置与可选能力

托管路线默认以压缩笔记交接研究；原文交接通过 `LDR_HYBRID_EVIDENCE_HANDOFF=located` 启用。本地 Qwen 由准确模型名显式选择，使用独立的证据流程。文库候选默认采用文档去重，可选本地 Cross-Encoder 重排默认关闭。

配置步骤见[快速开始](QUICKSTART.md)。一次实际任务的分发、并行研究、预算和写作记录见[执行实例](EXECUTION_EXAMPLE.md)。引用链接检查与报告内容质量分别验证，测量方法见[结果说明](RESULTS.md)。
