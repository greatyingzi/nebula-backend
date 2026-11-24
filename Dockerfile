# 星云后端单镜像: core + embed(llama.cpp) + rerank(onnx) 三服务共处

# ---- 阶段1: 从源码编译 core (可复现, 无需预编译二进制) ----
FROM rust:1.85-bookworm AS core-builder
WORKDIR /build
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libsqlite3-dev pkg-config cmake && rm -rf /var/lib/apt/lists/*
COPY rust/Cargo.toml rust/Cargo.lock ./
COPY rust/nebula-sync-rs ./nebula-sync-rs
COPY rust/nebula-mcp-rs ./nebula-mcp-rs
RUN cargo build --release -p nebula-sync-rs -p nebula-mcp-rs

# ---- 阶段2: 运行时 (llama.cpp 基础自带 llama-server + 全部库) ----
FROM ghcr.io/ggml-org/llama.cpp:server-b6097

# Python 运行时给 rerank 服务
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip && rm -rf /var/lib/apt/lists/*
COPY wheels/ /wheels/
RUN pip3 install --no-cache-dir --no-index --find-links=/wheels onnxruntime tokenizers 'numpy==1.26.4'

# core 二进制 (从源码编译阶段复制)
COPY --from=core-builder /build/target/release/nebula-sync /usr/local/bin/nebula-sync

# 模型 (构建前放 ./models/, 见 README)
COPY models/bge-m3-Q4_K_M.gguf /models/bge-m3-Q4_K_M.gguf
COPY models/model_quantized.onnx /models/model_quantized.onnx
COPY models/tokenizer.json /models/tokenizer.json

COPY scripts/mxbai_rerank_server.py /app/mxbai_rerank_server.py
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /usr/local/bin/nebula-sync /app/entrypoint.sh

ENV NEBULA_DATA=/data NEBULA_EMBED_URL=http://127.0.0.1:8300/v1/embeddings NEBULA_RERANK_URL=http://127.0.0.1:8301/v1/rerank
VOLUME ["/data"]
EXPOSE 9200 9300 8300 8301
ENTRYPOINT ["/app/entrypoint.sh"]
