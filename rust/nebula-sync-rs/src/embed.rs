use once_cell::sync::Lazy;
use rusqlite::{params, params_from_iter};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::config::Config;
use crate::db::vec_db;

static EMBED_CLIENT: Lazy<reqwest::blocking::Client> =
    Lazy::new(|| reqwest::blocking::Client::new());
static EMBED_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
static RERANK_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
// query -> (ts, vec) — pure function cache; persistent part lives in query_embeds
static EMBED_CACHE: Lazy<Mutex<HashMap<String, (f64, Vec<f32>)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn f64(v: &Value) -> f32 {
    v.as_f64().unwrap_or(0.0) as f32
}

fn sha16(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::digest(s.as_bytes());
    let mut out = String::new();
    for b in h.iter().take(8) {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn now_f64() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn l2_normalize(v: &mut Vec<f32>) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn floor_cb(s: &str, n: usize) -> &str {
    if n >= s.len() {
        return s;
    }
    let mut i = n;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    &s[..i]
}

fn bytes_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks(4)
        .map(|c| {
            let a: [u8; 4] = c.try_into().unwrap_or([0u8; 4]);
            f32::from_le_bytes(a)
        })
        .collect()
}

fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// POST texts to llama.cpp /v1/embeddings (BGE-M3). Serialized via one lock:
/// llama.cpp is single-threaded & serial — concurrent calls queue & time out.
/// Returns L2-normalized vectors.
pub fn embed(cfg: &Config, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
    let _g = EMBED_LOCK
        .lock()
        .map_err(|e| format!("embed lock poisoned: {e}"))?;
    let body = json!({"model": "bge-m3", "input": texts});
    let d: Value = EMBED_CLIENT
        .post(&cfg.embed_url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(90))
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(arr) = d.get("data").and_then(|x| x.as_array()) {
        for x in arr {
            let emb = x.get("embedding").and_then(|e| e.as_array()).map_or_else(
                || Vec::new(),
                |a| a.iter().map(f64).collect(),
            );
            let mut v = emb;
            l2_normalize(&mut v);
            out.push(v);
        }
    }
    Ok(out)
}

/// Single-text embed with 2-level cache (in-mem + persistent query_embeds).
pub fn embed_one(cfg: &Config, t: &str) -> Result<Vec<f32>, String> {
    let key = floor_cb(t, 200);
    // 1) in-memory cache (no TTL — pure function)
    {
        let g = EMBED_CACHE.lock().unwrap();
        if let Some((_, v)) = g.get(key) {
            return Ok(v.clone());
        }
    }
    // 2) persistent cache
    let qhash = sha16(key);
    if let Ok(vc) = vec_db(cfg) {
        let cached: Option<Vec<u8>> = vc
            .query_row(
                "SELECT vec FROM query_embeds WHERE qhash=?",
                params![qhash],
                |r| r.get(0),
            )
            .ok()
            .flatten();
        if let Some(blob) = cached {
            let v = bytes_to_f32(&blob);
            EMBED_CACHE.lock().unwrap().insert(key.to_string(), (now_f64(), v.clone()));
            return Ok(v);
        }
    } else {
        // vec_db open err is non-fatal
    }
    // 3) compute + store both
    let v = embed(cfg, &[key.to_string()])?
        .into_iter()
        .next()
        .unwrap_or_default();
    {
        let mut g = EMBED_CACHE.lock().unwrap();
        g.insert(key.to_string(), (now_f64(), v.clone()));
        if g.len() > cfg.embed_cache_max {
            // FIFO-ish eviction: drop oldest quarter
            let mut sorted: Vec<(&String, f64)> = g
                .iter()
                .map(|(k, (ts, _))| (k, *ts))
                .collect();
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let n = (g.len() / 4).max(1);
            let keys: Vec<String> = sorted.iter().take(n).map(|(k, _)| (*k).clone()).collect();
            for k in keys {
                g.remove(&k);
            }
        }
    }
    if let Ok(vc) = vec_db(cfg) {
        vc.execute(
            "INSERT OR REPLACE INTO query_embeds(qhash,query,vec,updated_at) VALUES(?,?,?,?)",
            params![qhash, key, f32_slice_to_bytes(&v), now_f64()],
        )
        .ok();
    }
    Ok(v)
}

/// POST query+docs to llama.cpp /v1/rerank (bge-reranker-v2-m3). Serial (single slot).
/// Truncates docs to rerank_max_chars (llama.cpp batch limit ~512 tokens).
/// items: Vec<serde_json::Value>; returns same list re-sorted by relevance.
pub fn rerank(cfg: &Config, query: &str, items: &[Value], limit: usize) -> Vec<Value> {
    use serde_json::Map;
    let trunc = cfg.rerank_max_chars;
    let q = floor_cb(query, trunc);
    let docs: Vec<String> = items
        .iter()
        .map(|it| {
            let c = it
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("");
            floor_cb(c, trunc).to_string()
        })
        .collect();
    let body = json!({"model": "bge-reranker-v2-m3", "query": q, "documents": docs});
    let client = EMBED_CLIENT.clone();
    let resp = {
        let _g = RERANK_LOCK
            .lock()
            .map_err(|e| e.to_string());
        client
            .post(&cfg.rerank_url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(60))
            .send()
    };
    let d: Value = match resp {
        Ok(r) => match r.error_for_status() {
            Ok(r) => match r.json() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[rerank] err: {e}");
                    return items.iter().take(limit).cloned().collect();
                }
            },
            Err(e) => {
                eprintln!("[rerank] err: {e}");
                return items.iter().take(limit).cloned().collect();
            }
        },
        Err(e) => {
            eprintln!("[rerank] err: {e}");
            return items.iter().take(limit).cloned().collect();
        }
    };
    let Some(scores) = d.get("results").and_then(|x| x.as_array()) else {
        return items.iter().take(limit).cloned().collect();
    };
    if scores.is_empty() {
        return items.iter().take(limit).cloned().collect();
    }
    let mut scored: Vec<(Value, f64)> = Vec::new();
    for (it, s) in items.iter().zip(scores.iter()) {
        let sc = s
            .get("relevance_score")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        scored.push((it.clone(), sc));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    for (mut it, sc) in scored {
        if let Some(obj) = it.as_object_mut() {
            // round to 4 decimals
            let r = (sc * 10000.0).round() / 10000.0;
            obj.insert("score".into(), json!(r));
            obj.insert("matched_by".into(), json!("rerank"));
        }
        out.push(it);
    }
    out
}
