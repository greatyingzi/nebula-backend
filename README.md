# Nebula Backend (星云后端)

自托管的个人知识星云后端：**单镜像**内置 core 检索服务 + 嵌入 + 重排三服务，Docker 一键部署。数据落盘物理机，镜像内零持久化数据。

配套 DSH 插件：[dsh-nebula-memory](https://github.com/greatyingzi/dsh-nebula-memory)（自动召回、多工具检索、零内嵌端点）。

## 架构

| 服务 | 说明 | 端口 |
|---|---|---|
| core | Rust 单进程，REST + MCP | 9200 / 9300 |
| embed | BGE-M3 GGUF 向量嵌入 (llama.cpp) | 8300 |
| rerank | mxbai-rerank-xsmall-v1 ONNX int8 精排 | 8301 |

一个容器跑三进程，entrypoint 依次拉起并等待健康。core 通过 `NEBULA_EMBED_URL` / `NEBULA_RERANK_URL` 调用依赖，默认指向容器内服务，也可指向远程/公开 API。

## 快速开始

```bash
# 1) 准备模型（放到 ./models/，不进 git）:
#    bge-m3-Q4_K_M.gguf  ← gpustack/bge-m3-GGUF
#    model_quantized.onnx + tokenizer.json  ← mixedbread-ai/mxbai-rerank-xsmall-v1/onnx
mkdir -p models data secrets
openssl rand -hex 32 > secrets/token
cp .env.example .env
# 2) 一键构建 + 启动:
docker compose up -d --build
curl -H "Authorization: Bearer $(cat secrets/token)" http://127.0.0.1:9200/health
```

## 数据持久化

- `./data` — sqlite 数据库（记忆、向量、session-rag）
- `./secrets/token` — Bearer token（不进仓）
- `./models` — GGUF/ONNX 模型（构建时 COPY 进镜像，不进 git）

## DSH 星云插件接入

安装 [dsh-nebula-memory](https://github.com/greatyingzi/dsh-nebula-memory) 后设置：

```bash
export NEBULA_BASE_URL=http://<host>:9200
export NEBULA_SYNC_TOKEN=$(cat secrets/token)
```

## 许可

见 LICENSE。
