# 运行准备

已在macOS arm64独立Python 3.14环境完成源码安装、依赖一致性检查、前端构建及浏览器注册/首页检查。已实际运行 DeepSeek 与 Qwen 网页 Hybrid 演示，结果与尚待收尾内容见[演示记录](DEMO_RUNS.md)。不要使用上游pip包或上游Docker镜像代替本仓库。

## 依赖与前端

项目要求 Python 3.12–3.14、Node.js 24 或以上，以 pyproject.toml 与 package.json 为准。已有 pdm.lock 和 package-lock.json。

在本目录创建环境并安装源码，构建静态资源：

```bash
python3.14 -m venv .venv
.venv/bin/pip install -e .
npm ci
npm run build
```

也可选Python 3.12/3.13；先确认解释器版本，系统自带Python可能太旧。macOS若SQLCipher编译/加载失败，需要先安装系统SQLCipher开发库再安装Python依赖；Linux x86_64依赖中提供二进制包。前端依赖以package-lock.json为准，Python以上命令按声明版本范围解析，并非严格按pdm.lock同步。

## 普通应用入口

```bash
.venv/bin/ldr-web
```

需要隔离端口和数据时，在启动进程设置 `LDR_WEB_HOST=127.0.0.1`、`LDR_WEB_PORT=8767` 和 `LDR_DATA_DIR=/absolute/path/to/your/data`。默认只在本机演示；本指南不部署公网服务。

具体端口以启动输出为准。创建账号，在文库界面导入有权使用的文档并建立索引，配置模型与搜索服务。不要复制他人的数据库或账号凭据。

## 已有文库的工作区启动器

scripts/start-hybrid-research.py 使用本机 local_only/web_app.json，读取其中 env_file、collection_manifest 和 port。manifest 需包含已有文库的 data_dir 与 username。它不负责创建文库。

```json
{
  "env_file": ".env",
  "collection_manifest": "local_only/collection/manifest.json",
  "port": 8766
}
```

准备本机配置后：

```bash
.venv/bin/python scripts/start-hybrid-research.py
```

模型凭据通过本机 .env 或应用设置配置，不提交 Git。

### 可选本地重排

已有重排模型缓存时，启动前设置`LDR_COLLECTION_CROSS_ENCODER_RERANK=1`即可对文库候选排序。模型、缓存路径及语言处理见[文档重排](COLLECTION_RERANKING.md)。不需要额外生成API费用。

### 网页使用本地 Qwen 证据流程

在启动服务前配置 `LDR_LLM_OPENAI_ENDPOINT_URL` 为已有本地 OpenAI-compatible 服务的 `/v1` 地址，`LDR_HYBRID_QWEN_MODEL` 为该服务返回的准确模型名。首页选择 OpenAI-Compatible Endpoint 和相同模型名，即进入需求计划、证据卡与多片段综合流程；未匹配的模型仍走默认托管控制流。模型服务应按部署配方关闭 thinking，并设置适合设备的输出上限。本项目不自动下载或启动模型。

网页本地路线使用当前仓库的 Qwen candidate 与产品来源连接器，运行元数据记录实际版本；它不是冻结 v6.2 的 30 题评测入口，不继承旧版评测成绩。
