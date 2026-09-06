# 混合研究交付示例（真实报告节选）

来源：固定任务eval-h01，托管候选恢复尝试r2。以下截取已生成报告的一小段，未重新生成，不代表当前网页实时执行。省略其余章节和本地文库链接，不公开原论文全文快照。

原问题：我们计划评估Deep Research系统。分别说明文库中DeepResearch Bench论文的任务与交付物、RACE和FACT的关注点，以及官方GitHub当前提供的复现实验入口；结论附来源。

## 原报告节选：需要用户放入的模型输出

生成报告应放到：

```text
data/test_data/raw_data/<model_name>.jsonl
```

## 原报告节选：查询文件

查询文件位于：

```text
data/prompt_data/query.jsonl
```

这也是 `run_benchmark.sh` 中 `QUERY_DATA_PATH` 的默认值。[run_benchmark.sh](https://github.com/Ayanami0730/deep_research_bench/blob/main/run_benchmark.sh)

## 阅读方式

这个例子展示的是研究报告把任务说明进一步落实到数据路径和可点击的官方实现入口。以上为当次快照结论，官方main分支可能继续变化；不能将静态示例当作实时核查结果。
