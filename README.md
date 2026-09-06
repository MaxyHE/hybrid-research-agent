# Hybrid Research Agent

**融合用户文库与公开 Web，让技术调研从资料发现走向有据可查的研究交付。**

基于 Open Deep Research 与 local-deep-research 构建，支持 Web、Collection 和混合来源研究，探索托管模型与本地 Qwen 的差异化执行策略。

## 主要成果

| 能力 | 实测结果 |
| --- | --- |
| 混合来源研究交付 | 三类来源共 12 个固定任务，质量完成率 **91.7%（11/12）** |
| Document 级候选接口 | 两套语料、60 条查询，指定目标 Recall@8 **80.3% → 88.2%** |
| 候选多样性 | 输出文档重复率 **51.9% → 0%**，平均不同文档候选 **3.85 → 7.73** |
| 本地模型工程 | Qwen 27B 双卡 4-bit 部署，完成三类来源 **30 题**固定评测 |

[完整指标口径](docs/RESULTS.md) · [代码与贡献导航](docs/ARCHITECTURE.md) · [运行准备](docs/QUICKSTART.md)

## 核心设计

- **混合来源**：文库回答论文与方法问题，Web 核对官方实现、数据和发布信息；来源身份贯穿读取与引用。
- **Document 级检索**：扩大 chunk 候选池、按文档去重，再回读正文，让有限候选位覆盖更多不同文档。
- **分路线编排**：托管模型使用多阶段研究流程；本地候选使用需求驱动的有限证据流程和多片段综合。
- **可追溯产物**：保存报告、来源快照、调用轨迹与用量，支持结果回查和版本对照。

```text
问题 + 来源范围 + 模型
 ├─ 托管：Supervisor → Researchers → Writer
 └─ 本地：需求计划 → 检索阅读 → 有限补缺 → 综合
                    │
        Web / Collection → 正文与来源快照
                    │
              报告 + 引用 + trace
```

## 从这里开始看

1. [运行时](src/local_deep_research/odr_baseline/runtime.py)：研究调度、工具调用和产物。
2. [来源连接器](src/local_deep_research/odr_baseline/project_sources.py)：文库候选接口与正文读取。
3. [Qwen 候选](src/local_deep_research/odr_baseline/qwen_candidate.py)：本地需求计划与执行策略。
4. [Qwen writer](src/local_deep_research/odr_baseline/qwen_writer.py)：来源级多片段综合。
5. [网页服务接线](src/local_deep_research/web/services/hybrid_odr_service.py)：产品入口。

网页服务与本地专用优化入口是不同执行路径，选择本地模型不会自动启用全部 Qwen 候选策略。30 题成绩对应冻结 v6.2；本快照包含后续候选代码，尚不以其替代冻结成绩。

## 本地发布候选版

本目录是独立 Git 仓库，供查看、整理与后续公开；未配置远端。保留应用完整源码以维持依赖关系，未携带原研发历史、个人简历、文库数据库、模型权重与原始实验大文件。独立安装与演示验证待完成，当前不宣称开箱即用。

技术栈：Python · Flask · LangChain 兼容接口 · FAISS · Transformers · bitsandbytes · Slurm。

## 开源基础

基于 [local-deep-research](https://github.com/LearningCircuit/local-deep-research) 与 [Open Deep Research](https://github.com/langchain-ai/open_deep_research)。保留原作者与许可信息。项目贡献聚焦混合来源接线、Document 候选接口、信息传递、本地资源适配与工程评测。

见 [LICENSE](LICENSE)、[ODR LICENSE](src/local_deep_research/odr_baseline/LICENSE) 和 [来源记录](docs/PROVENANCE.md)。
