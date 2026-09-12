# 源码来源与发布边界

- 本地快照来源：hybrid-odr-minimal 研发 worktree。
- 基础源码 revision：67d7b99cb762520231d7580ad47a50036410687b；追加同步研发02f9629的查询翻译、Collection连接器和启动器配置（不复制机器配置）。未整体同步后续Qwen候选代码。
- 本仓库拥有独立 Git 历史，不是原研发仓库的 worktree，不携带原研发提交历史。
- 应用基础：LearningCircuit/local-deep-research，原作者、包名与许可信息保留。
- ODR 控制流与部分提示词：langchain-ai/open_deep_research，组件 LICENSE 保留。
- 预算及未完成任务语义适配 deep-research-harness；已补组件 `NOTICE.md` 与固定来源版本的 `HARNESS_LICENSE`，原作者信息保留。
- 没有设置 GitHub 远端或改变任何现有仓库可见性。

## 发布前剩余事项

已完成的安装、前端、演示与检索附件工作见 [发布状态](RELEASE_STATUS.md)。本次追加原文交接模块及其必要运行时方法，来源为研发 `99f9217` 所在版本；保留发布副本已有的文库链接修复，不整体覆盖研发树。原文交接为可选模式，结果与默认 H-off 分开。

公开前由用户提供目标仓库 URL，确认目标为空后推送独立发布历史。原始数据库、模型、凭据和完整来源快照保持本地；Qwen 历史固定评测不改称新版成绩。
