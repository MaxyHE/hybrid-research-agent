# 可选本地文档重排

文库连接器先检索64个chunk，再按文档去重。启用重排后，对候选文档计算query–document相关性，返回前8篇，供Agent继续阅读。

```bash
export LDR_COLLECTION_CROSS_ENCODER_RERANK=1
export LDR_COLLECTION_CROSS_ENCODER_CACHE=/path/to/existing/hf_cache
```

在启动网页服务的同一环境设置以上变量即可。模型为`cross-encoder/ms-marco-MiniLM-L6-v2`，仅从已有本地Hugging Face缓存加载；开关不会自动下载权重。未提供模型缓存时先准备模型，再启用。默认关闭，不改变已有文档去重检索路线。

- 输入：实际检索query与候选文档标题、正文。
- 文档端截断至pair总长512 token，batch size 16；同分保留原候选顺序。
- 使用MPS（可用时）或CPU，同一进程复用模型。
- 中文查询经现有翻译功能得到英文译文时可重排；未译中文保持原文档排序。
- `ProjectCollectionConnector.last_rerank_metadata`记录是否应用、候选数、截断数、加载与排序耗时。

重排只改变候选顺序，不调用生成API，不改变研究任务、官网来源范围或Writer策略。长文档只使用截断后的前段，完整论文检索不等同于SciFact摘要检索。

全库组件对照见[结果](RESULTS.md)，已保存排名的指标重算见[评测附件](evaluation/README.md)。组件结果不等于端到端报告质量提升。
