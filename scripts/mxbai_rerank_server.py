#!/usr/bin/env python3
"""
mxbai-rerank-xsmall-v1 ONNX rerank service — drop-in replacement for llama-server --rerank
Exposes the same /v1/rerank API that nebula-sync-rs (hub) expects.

Usage: python3 mxbai_rerank_server.py [port]
Default port: 8301
"""
import json, time, sys
import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from tokenizers import Tokenizer
import onnxruntime as ort

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8301
MODEL_DIR = "/models"

# ---- Load model + tokenizer (once) ----
print(f"Loading mxbai-rerank-xsmall-v1 ONNX from {MODEL_DIR}/model_quantized.onnx ...")
tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
sess = ort.InferenceSession(f"{MODEL_DIR}/mxbai-xsmall.onnx",
                            providers=["CPUExecutionProvider"])
sess.intra_op_num_threads = 2  # 4核, 防超订: 2并发x2线程=4
cls_id = tok.token_to_id("[CLS]")
sep_id = tok.token_to_id("[SEP]")
pad_id = tok.token_to_id("[PAD]") or 0
MAXLEN = 512
import threading
_INFER_LOCK = threading.Lock()  # serialize ONNX session.run (avoids GIL contention jitter)
print(f"  special tokens: CLS={cls_id} SEP={sep_id} PAD={pad_id}")
print(f"  model loaded, listening on port {PORT} (threaded, inference-locked)")

def rerank(query: str, documents: list) -> list:
    """Score all (query, doc) pairs, return [{index, relevance_score}] sorted desc."""
    all_ids, all_mask = [], []
    for doc in documents:
        enc_q = tok.encode(query)
        enc_d = tok.encode(doc)
        ids = [cls_id] + enc_q.ids + [sep_id] + enc_d.ids + [sep_id]
        if len(ids) > MAXLEN:
            ids = ids[:MAXLEN]
        mask = [1] * len(ids)
        all_ids.append(ids)
        all_mask.append(mask)
    # pad
    ml = max(len(x) for x in all_ids)
    for i in range(len(all_ids)):
        p = ml - len(all_ids[i])
        if p > 0:
            all_ids[i] += [pad_id] * p
            all_mask[i] += [0] * p
    ids = np.array(all_ids, dtype=np.int64)
    mask = np.array(all_mask, dtype=np.int64)
    with _INFER_LOCK:
        out = sess.run(None, {"input_ids": ids, "attention_mask": mask})
    scores = list(np.asarray(out[0]).flatten())
    # sort by score desc (like llama-server)
    results = sorted(
        [{"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )
    return results

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/rerank":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            query = body.get("query", "")
            documents = body.get("documents", [])
            t0 = time.perf_counter()
            results = rerank(query, documents)
            ms = (time.perf_counter() - t0) * 1000
            resp = {
                "model": "mxbai-rerank-xsmall-v1",
                "object": "list",
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
                "results": results,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
            print(f"  [rerank] {len(documents)} docs in {ms:.0f}ms")
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # quiet

if __name__ == "__main__":
    print(f"Starting mxbai ONNX rerank server on 0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
