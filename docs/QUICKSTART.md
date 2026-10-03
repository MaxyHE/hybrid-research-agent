# 快速开始

本指南说明如何从源码启动网页、配置来源与模型，并提交第一项研究。已有验证包括 macOS arm64 的独立 Python 3.14 环境安装、前端构建，以及托管与本地模型的实际网页运行；运行结果见[历史运行记录](DEMO_RUNS.md)。

## 环境准备

- Python 3.12–3.14。
- Node.js 24 或以上。
- 可用的模型服务；Web 或 Hybrid 研究还需要配置搜索服务。
- Collection 或 Hybrid 研究所需的文档，用户应拥有使用权限。

macOS 若 SQLCipher 编译或加载失败，先安装系统 SQLCipher 开发库，再安装 Python 依赖。Linux x86_64 的依赖提供二进制包。

## 安装源码与构建前端

```bash
git clone https://github.com/MaxyHE/hybrid-research-agent.git
cd hybrid-research-agent
python3.12 -m venv .venv
.venv/bin/pip install -e .
npm ci
npm run build
```

可将 `python3.12` 换成 3.13 或 3.14。安装本仓库源码才能使用这里的研究改造；上游 pip 包和 Docker 镜像对应上游应用。

`npm ci` 使用仓库的 `package-lock.json`。上述 Python 命令按 `pyproject.toml` 声明的版本范围解析依赖；仓库另保留 `pdm.lock`，该命令并非按它严格同步。

## 启动网页与提交研究

```bash
LDR_WEB_HOST=127.0.0.1 .venv/bin/ldr-web
```

1. 打开启动输出中的本地地址，创建账号。
2. 在应用中配置模型服务；使用 Web 或 Hybrid 时，同时配置搜索服务。
3. 使用 Collection 或 Hybrid 时，在文库界面导入文档并建立索引。
4. 在研究页选择来源范围与模型，输入问题并提交。
5. 查看阶段与执行计数，完成后打开报告、引用和历史记录。

建议第一项任务使用一篇已导入的论文，询问其核心机制与适用条件；验证文库读取后，再选择 Hybrid 核对该论文的官方实现。模型 API 和搜索服务的费用由相应服务计收。

可通过启动环境设置 `LDR_WEB_PORT` 和 `LDR_DATA_DIR`，隔离端口与数据目录。数据目录使用本机绝对路径，实际访问地址以启动输出为准。本指南用于本地运行。

## 接入本地 Qwen

先单独部署 OpenAI-compatible 模型服务。在启动网页服务前设置：

```bash
export LDR_LLM_OPENAI_ENDPOINT_URL=http://127.0.0.1:8000/v1
export LDR_HYBRID_QWEN_MODEL=your-served-model-name
LDR_WEB_HOST=127.0.0.1 .venv/bin/ldr-web
```

将端点地址和模型名替换成实际服务配置；模型名须与服务返回的名称一致。首页选择 OpenAI-Compatible Endpoint 和该模型名后，研究进入 Qwen 的需求计划、证据组织与多片段综合流程。其余模型使用默认托管控制流。

模型服务按部署配方关闭 thinking，并设置适合设备的输出上限。模型权重下载、服务启动及硬件准备由用户完成。

网页使用当前 Qwen candidate 与产品来源连接器，运行元数据记录实际版本。冻结 v6.2 的 30 题成绩单独保留在[结果说明](RESULTS.md)中。

## 可选研究能力

| 能力 | 启用方式 | 准备与说明 |
| --- | --- | --- |
| 文档重排 | `LDR_COLLECTION_CROSS_ENCODER_RERANK=1` | 默认关闭，先准备已有模型缓存；见[文档重排](COLLECTION_RERANKING.md) |
| 中文查询翻译 | 按查询翻译文档配置本地 Qwen | 将中文问题转换为英文文库查询；见[查询翻译](COLLECTION_QUERY_TRANSLATION.md) |
| 原文证据交接 | `LDR_HYBRID_EVIDENCE_HANDOFF=located` | 用于托管路线，默认使用压缩笔记；见[证据交接](EVIDENCE_HANDOFF.md) |

这些设置在启动服务前配置，修改后重启服务。重排使用本地模型，不增加生成模型 API 调用。

## 已有文库的工作区启动器

已有本地文库时，可使用 `scripts/start-hybrid-research.py` 管理配置。它读取本机 `local_only/web_app.json` 中的 `env_file`、`collection_manifest` 和 `port`：

```json
{
  "env_file": ".env",
  "collection_manifest": "local_only/collection/manifest.json",
  "port": 8766
}
```

manifest 需包含已有文库的 `data_dir` 与 `username`。先通过应用创建文库，再准备对应 manifest，随后运行：

```bash
.venv/bin/python scripts/start-hybrid-research.py
```

凭据通过本机 `.env` 或应用设置配置，账号数据、manifest 中的私有信息与密钥保持本地。

## 离线重算检索指标

在仓库根目录运行：

```bash
python3 scripts/recompute-scifact-metrics.py
```

只需 Python 标准库，按公开的保存排名重新计算 Recall、MRR 与 nDCG。准备重新执行检索时还需语料、索引与模型；附件范围见[评测说明](evaluation/README.md)。
