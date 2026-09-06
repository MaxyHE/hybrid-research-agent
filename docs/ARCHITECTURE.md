# 架构与贡献导航

本仓库保留完整 src 应用包，避免摘取几个 Agent 文件后失去文库、设置与网页依赖。目录规模不代表全部原创。

| 层次 | 代码 | 本项目关注点 |
| --- | --- | --- |
| 研究编排 | ../src/local_deep_research/odr_baseline/runtime.py | 调度、预算、证据与报告交付 |
| 混合来源 | ../src/local_deep_research/odr_baseline/hybrid.py | Web / Collection 工具接线 |
| 文库接口 | ../src/local_deep_research/odr_baseline/project_sources.py | 候选池、Document 去重、全文读取 |
| 本地候选 | ../src/local_deep_research/odr_baseline/qwen_candidate.py | 需求驱动执行 |
| 本地写作 | ../src/local_deep_research/odr_baseline/qwen_writer.py | 有限原文片段与综合 |
| 网页服务 | ../src/local_deep_research/web/services/hybrid_odr_service.py | 问题、来源、模型入口 |

来源发现与证据使用分离：检索候选引导阅读，读取后的正文与来源身份进入后续综合。引用来源存在性检查不等于正文事实正确，质量由评测单独核对。
