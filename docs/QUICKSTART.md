# 运行准备

当前是本地发布候选源码包，尚未在全新环境完成安装演示。不要使用上游 pip 包或上游 Docker 镜像来验证本仓库改造。

## 依赖与前端

项目要求 Python 3.12–3.14、Node.js 24 或以上，以 pyproject.toml 与 package.json 为准。已有 pdm.lock 和 package-lock.json。

在本目录创建环境并安装源码，构建静态资源：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
npm ci
npm run build
```

## 普通应用入口

```bash
.venv/bin/ldr-web
```

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

模型凭据通过本机 .env 或应用设置配置，不提交 Git。本地专用 Qwen 策略的独立运行配方尚待从研发入口整理，当前网页入口不能代替对应固定评测。
