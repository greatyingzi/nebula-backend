# Nebula Backend (星云后端 · 公开脱敏版)

自托管的个人知识星云后端：核心检索服务 + 嵌入 + 重排三件套，Docker 一键部署。
数据全部落盘物理机，镜像内零持久化数据。

## 架构

| 服务 | 说明 | 端口 |
|---|---|---|
| nebula-core | Rust 单进程，REST + MCP | 9200 / 9300 |
| embed | BGE-M3 GGUF 向量嵌入 (llama.cpp) | 8300 |
| rerank | BGE-reranker GGUF 精排 (llama.cpp) | 8301 |

核心只通过 `NEBULA_EMBED_URL` / `NEBULA_RERANK_URL` 调用依赖，可替换为远程机或公开 API。

## 快速开始

```bash
cp .env.example .env
# 1. 准备模型：把 bge-m3-Q4_K_M.gguf / rerank gguf 放到 ./models/
# 2. 生成 token：openssl rand -hex 32 > ./secrets/token
mkdir -p data models secrets
docker compose up -d
curl -H "Authorization: Bearer $(cat secrets/token)" http://127.0.0.1:9200/health
```

## 数据持久化

- `./data` — sqlite 数据库（记忆、向量、session-rag）
- `./models` — GGUF 模型文件（不进镜像）
- `./secrets/token` — Bearer token（不进仓）

## DSH 星云插件接入

在 DSH 端设置环境变量（或写 `~/.dsh/nebula-memory.json`）：

```bash
export NEBULA_BASE_URL=http://<host>:9200
export NEBULA_SYNC_TOKEN=$(cat secrets/token)
```

插件通过 `nebula_search` / `nebula_session_search` 等工具自动检索。

## 许可

见 LICENSE。
