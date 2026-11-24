# Nebula Backend (星云后端)

自托管的个人知识星云后端：核心检索服务 + 嵌入 + 重排三件套。
**模型已打包进镜像，`docker compose up -d` 一键启动**。数据落盘物理机，镜像内零持久化数据。

配套 DSH 插件：[dsh-nebula-memory](https://github.com/greatyingzi/dsh-nebula-memory)（装后自动召回、多工具检索、零内嵌端点）。

## 架构

| 服务 | 说明 | 端口 |
|---|---|---|
| nebula-core | Rust 单进程，REST + MCP | 9200 / 9300 |
| embed | BGE-M3 GGUF 向量嵌入 (llama.cpp)，模型内置 | 8300 |
| rerank | BGE-reranker GGUF 精排 (llama.cpp)，模型内置 | 8301 |

核心只通过 `NEBULA_EMBED_URL` / `NEBULA_RERANK_URL` 调用依赖，可替换为远程机或公开 API。

## 快速开始

```bash
cp .env.example .env
# 生成 token：
mkdir -p data secrets
openssl rand -hex 32 > secrets/token
# 一键构建+启动（首次 build 会下载约 840MB 模型进镜像）：
docker compose up -d
curl -H "Authorization: Bearer $(cat secrets/token)" http://127.0.0.1:9200/health
```

## 数据持久化

- `./data` — sqlite 数据库（记忆、向量、session-rag）
- `./secrets/token` — Bearer token（不进仓）
- 模型已内置镜像，无需挂载。

## DSH 星云插件接入

安装配套插件 [dsh-nebula-memory](https://github.com/greatyingzi/dsh-nebula-memory)，然后在 DSH 端设置：

```bash
export NEBULA_BASE_URL=http://<host>:9200
export NEBULA_SYNC_TOKEN=$(cat secrets/token)
```

插件通过 `nebula_search` / `nebula_session_search` 等工具自动检索，并将相关记忆注入会话上下文。

## 许可

见 LICENSE。
