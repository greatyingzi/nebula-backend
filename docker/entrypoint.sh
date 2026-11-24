#!/bin/bash
set -u
echo "[entrypoint] starting nebula services..."

# 1) embed (llama.cpp)
/app/llama-server --model /models/bge-m3-Q4_K_M.gguf --embeddings --pooling mean --port 8300 -b 4096 -ub 4096 > /tmp/embed.log 2>&1 &
PID_EMBED=$!

# 2) rerank (onnx mxbai)
python3 /app/mxbai_rerank_server.py 8301 > /tmp/rerank.log 2>&1 &
PID_RERANK=$!

# wait for embed + rerank healthy
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8300/health >/dev/null 2>&1 && EMB=1 || EMB=0
  curl -sf http://127.0.0.1:8301/health >/dev/null 2>&1 && RR=1 || RR=0
  [ "$EMB" = "1" ] && [ "$RR" = "1" ] && break
  sleep 2
done
echo "[entrypoint] embed=$EMB rerank=$RR ready"

# 3) core (respect externally-set URLs; default to local services with full paths)
: ${NEBULA_EMBED_URL:=http://127.0.0.1:8300/v1/embeddings}
: ${NEBULA_RERANK_URL:=http://127.0.0.1:8301/v1/rerank}
export NEBULA_EMBED_URL NEBULA_RERANK_URL
/usr/local/bin/nebula-sync > /tmp/core.log 2>&1 &
PID_CORE=$!

# keep container alive; if any critical process dies, exit
while kill -0 $PID_CORE 2>/dev/null && kill -0 $PID_EMBED 2>/dev/null && kill -0 $PID_RERANK 2>/dev/null; do sleep 5; done
echo "[entrypoint] a service exited, shutting down"
kill $PID_CORE $PID_EMBED $PID_RERANK 2>/dev/null
wait 2>/dev/null
