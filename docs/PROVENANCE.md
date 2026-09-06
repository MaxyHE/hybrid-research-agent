# 源码来源与发布边界

- 本地快照来源：hybrid-odr-minimal 研发 worktree。
- 固定源码 revision：67d7b99cb762520231d7580ad47a50036410687b。
- 本仓库拥有独立 Git 历史，不是原研发仓库的 worktree，不携带原研发提交历史。
- 应用基础：LearningCircuit/local-deep-research，原作者、包名与许可信息保留。
- ODR 控制流与部分提示词：langchain-ai/open_deep_research，组件 LICENSE 保留。
- 预算及未完成任务语义有 deep-research-harness 适配背景；完整采用范围和所需 NOTICE 的发布核对尚待完成。
- 没有设置 GitHub 远端或改变任何现有仓库可见性。

## 发布前剩余事项

1. 验证干净环境安装、文库导入与真实入口；补可分享演示。
2. 对实际发布的源码/元数据做完整敏感信息与第三方归属核查。
3. 补可分发的逐题结果及可复现检索配方；不公开原始文库全文。
4. 确定采用的 Qwen 候选版本，将开发代码与该版本实测结果对应。
