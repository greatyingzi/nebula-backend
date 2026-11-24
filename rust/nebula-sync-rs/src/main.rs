mod api_common;
mod atoms_api;
mod config;
mod dash_api;
mod db;
mod embed;
mod media_api;
mod web_api;
mod memories_api;
mod graph_api;
mod metrics_api;
mod ops;
mod session_rag;
mod vector_store;
mod ws_api;

use axum::{
    body::Body,
    extract::{Query, State},
    http::header,
    middleware,
    response::{IntoResponse, Response},
    routing,
    routing::{get, put, post},
    Router,
};

// note: `Query`/`put`/`routing` pre-existed at HEAD (some unused there); kept verbatim to avoid scope creep.
// `middleware` is the single new axum feature added by the LingHub 业务后端 (阶段二) CORS layer.
use once_cell::sync::Lazy;
use rusqlite::{params, params_from_iter, Connection};

/// Convert a Vec of rusqlite Values into a Params-compatible iterator
fn vparams(params: &[rusqlite::types::Value]) -> Vec<rusqlite::types::Value> {
    params.to_vec()
}

use rusqlite::types::{Value as RValue, ValueRef};
use serde_json::{json, Value};

/// Convert a borrowed rusqlite ValueRef into a serde_json::Value.
/// Text is parsed as JSON when it looks like JSON, else treated as a plain string.
fn value_ref_to_json(vr: ValueRef) -> Value {
    match vr {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => Value::from(i),
        ValueRef::Real(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        ValueRef::Text(b) => {
            let s = String::from_utf8_lossy(b).into_owned();
            // if it parses as JSON use that, otherwise plain string
            serde_json::from_str(&s).unwrap_or(Value::String(s))
        }
        ValueRef::Blob(b) => Value::String(b.iter().map(|x| format!("{:02x}", x)).collect()),
    }
}
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use config::Config;
use ops::now_f64;
use vector_store::VectorStore;

// ---------- search cache (minimal skeleton; invalidated on mem.* writes) ----------
#[derive(Clone)]
pub(crate) struct App {
    pub cfg: Arc<Config>,
    pub token: Arc<std::sync::RwLock<Option<String>>>,
}

// (q, scope, limit, rerank) -> (ts, skeleton)
static SEARCH_CACHE: Lazy<Mutex<HashMap<(String, String, i64, bool), (f64, Value)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub(crate) fn search_cache_clear() {
    SEARCH_CACHE.lock().unwrap().clear();
}

// ---------- the single VectorStore ----------
pub(crate) static VS: Lazy<std::sync::Arc<VectorStore>> = Lazy::new(|| std::sync::Arc::new(VectorStore::new()));
/// 记忆写/删后置位 → search 缓存与向量索引后台重载 (writeMemory/consolidate 用)
pub(crate) static VS_NOTIFY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn now_f64_() -> f64 {
    now_f64()
}

// ---------- helpers ----------
pub(crate) fn qval(q: &HashMap<String, Vec<String>>, k: &str) -> String {
    q.get(k).and_then(|v| v.first().cloned()).unwrap_or_default()
}
pub(crate) fn qint(q: &HashMap<String, Vec<String>>, k: &str, def: i64) -> i64 {
    qval(q, k)
        .parse()
        .unwrap_or(def)
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let h = (b[i + 1] as char).to_digit(16);
            let l = (b[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (h, l) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        if b[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(b[i]);
        }
        i += 1;
    }
    String::from_utf8(out).unwrap_or_default()
}

/// Faithful urllib.parse.parse_qs: key -> Vec<value>, single values work.
fn parse_qs(pairs: &[(String, String)]) -> HashMap<String, Vec<String>> {
    let mut m: HashMap<String, Vec<String>> = HashMap::new();
    for (k, v) in pairs {
        if k.is_empty() {
            continue;
        }
        m.entry(k.clone()).or_default().push(v.clone());
    }
    m
}

/// Parse the request URI query string (scalar-tolerant, like urllib.parse.parse_qs).
pub(crate) fn qparams(req: &axum::http::Request<Body>) -> HashMap<String, Vec<String>> {
    let raw = req.uri().query().unwrap_or("");
    let mut pairs: Vec<(String, String)> = Vec::new();
    for part in raw.split('&') {
        if part.is_empty() {
            continue;
        }
        let (k, v) = match part.split_once('=') {
            Some((k, v)) => (k, v),
            None => (part, ""),
        };
        pairs.push((url_decode(k), url_decode(v)));
    }
    parse_qs(&pairs)
}

pub(crate) fn auth_ok(app: &App, req: &axum::http::Request<Body>) -> bool {
    let tok = app.token.read().unwrap().clone();
    let Some(tok) = tok else {
        return true;
    };
    let hdr = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|x| x.to_str().ok())
        .unwrap_or("");
    // Python uses hmac.compare_digest(h, "Bearer "+tok)
    if subtle_eq(hdr, &format!("Bearer {tok}")) {
        return true;
    }
    // 三通道之二: 登录会话 cookie (看板浏览器, 对齐 CF checkAuth)
    let secret = if app.cfg.session_secret.is_empty() {
        app.cfg.admin_password.clone()
    } else {
        app.cfg.session_secret.clone()
    };
    let ck = req.headers().get(header::COOKIE).and_then(|x| x.to_str().ok());
    api_common::cookie_session_ok(ck, &secret)
}

fn subtle_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut x = 0u8;
    for i in 0..a.len() {
        x |= a[i] ^ b[i];
    }
    x == 0
}

/// Generic 404 fallback — matches Python do_GET's `return self._send(404,{"error":"not found"})`.
async fn not_found(State(app): State<App>, req: axum::http::Request<Body>) -> Response {
    if auth_ok(&app, &req) {
        json_resp(404, &json!({"error": "not found"}))
    } else {
        json_resp(401, &json!({"error": "unauthorized"}))
    }
}

pub(crate) fn json_resp(code: u16, v: &Value) -> Response {
    let b = py_dumps(v);
    let h = header::HeaderValue::from_static("application/json");
    let mut resp = Response::new(Body::from(b));
    *resp.status_mut() = axum::http::StatusCode::from_u16(code).unwrap();
    resp.headers_mut().insert(header::CONTENT_TYPE, h);
    resp
}

/// Serialize exactly like Python `json.dumps(obj, ensure_ascii=False)` with default
/// separators: space after ':' and ',', no newlines. Required for byte-identical
/// parity with the Python original.
fn py_dumps(v: &Value) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = write_json(v, &mut s);
    s
}

fn write_json(v: &Value, s: &mut String) {
    use std::fmt::Write;
    match v {
        Value::Null => {
            let _ = s.write_str("null");
        }
        Value::Bool(b) => {
            let _ = s.write_str(if *b { "true" } else { "false" });
        }
        Value::Number(n) => {
            let _ = write!(s, "{}", n);
        }
        Value::String(str) => {
            write_json_str(str, s);
        }
        Value::Array(a) => {
            let _ = s.write_str("[");
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    let _ = s.write_str(", ");
                }
                write_json(item, s);
            }
            let _ = s.write_str("]");
        }
        Value::Object(o) => {
            let _ = s.write_str("{");
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 {
                    let _ = s.write_str(", ");
                }
                write_json_str(k, s);
                let _ = s.write_str(": ");
                write_json(val, s);
            }
            let _ = s.write_str("}");
        }
    }
}

fn write_json_str(str: &str, s: &mut String) {
    use std::fmt::Write;
    let _ = s.write_str("\"");
    for c in str.chars() {
        match c {
            '"' => {
                let _ = s.write_str("\\\"");
            }
            '\\' => {
                let _ = s.write_str("\\\\");
            }
            '\n' => {
                let _ = s.write_str("\\n");
            }
            '\r' => {
                let _ = s.write_str("\\r");
            }
            '\t' => {
                let _ = s.write_str("\\t");
            }
            '\u{08}' => {
                let _ = s.write_str("\\b");
            }
            '\u{0c}' => {
                let _ = s.write_str("\\f");
            }
            c if (c as u32) < 0x20 => {
                let _ = write!(s, "\\u{:04x}", c as u32);
            }
            c => {
                let _ = s.write_char(c);
            }
        }
    }
    let _ = s.write_str("\"");
}

// ---------- routes ----------
async fn health(State(app): State<App>) -> Response {
    let cfg = &app.cfg;
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let n: i64 = c
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);
    let m: f64 = c
        .query_row("SELECT MAX(ts) FROM ops", [], |r| r.get(0))
        .unwrap_or(0.0);
    let a: i64 = c
        .query_row("SELECT COUNT(*) FROM atoms_v2", [], |r| r.get(0))
        .unwrap_or(0);
    let vectors = VS.id_count();
    json_resp(200, &json!({"ok": true, "live_memories": n, "max_ts": m, "vectors": vectors, "atoms": a}))
}

async fn ops_list(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let since = qval(&q, "since").parse::<f64>().unwrap_or(0.0);
    let mut st = match c.prepare(
        "SELECT id,ts,op,entity,payload,source FROM ops WHERE ts>? ORDER BY ts ASC",
    ) {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let rows: Vec<Value> = st
        .query_map(params![since], |r| {
            let payload: String = r.get(4)?;
            let payload_v: Value = serde_json::from_str(&payload).unwrap_or(Value::String(payload));
            Ok(json!({"id": r.get::<_, String>(0)?, "ts": r.get::<_, f64>(1)?, "op": r.get::<_, String>(2)?,
                "entity": r.get::<_, String>(3)?, "payload": payload_v, "source": r.get::<_, String>(5)?}))
        })
        .ok()
        .map(|mut m| {
            let mut out = Vec::new();
            while let Some(Ok(v)) = m.next() {
                out.push(v);
            }
            out
        })
        .unwrap_or_default();
    json_resp(200, &json!({"ok": true, "ops": rows}))
}

async fn full(State(app): State<App>) -> Response {
    let cfg = &app.cfg;
    match std::fs::read(&cfg.db) {
        Ok(raw) => {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            use std::io::Write;
            if let Err(e) = enc.write_all(&raw) {
                return json_resp(500, &json!({"error": e.to_string()}));
            }
            let g = match enc.finish() {
                Ok(g) => g,
                Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
            };
            let len = g.len();
            let mut resp = Response::new(Body::from(g));
            *resp.status_mut() = axum::http::StatusCode::OK;
            resp.headers_mut().insert(header::CONTENT_TYPE, header::HeaderValue::from_static("application/gzip"));
            if let Ok(h) = header::HeaderValue::from_str(&len.to_string()) {
                resp.headers_mut().insert(header::CONTENT_LENGTH, h);
            }
            resp
        }
        Err(e) => json_resp(500, &json!({"error": e.to_string()})),
    }
}

async fn memories(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let limit = qint(&q, "limit", 50);
    let offset = qint(&q, "offset", 0);
    let scope = qval(&q, "scope").trim().to_string();
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let (rows, total) = if !scope.is_empty() && scope != "all" {
        let mut st = match c.prepare(
            "SELECT id,content,scope,tags,updated_at,created_at FROM memories WHERE scope=? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        let r = st
            .query_map(params![scope, limit, offset], |r| {
                Ok(json!({"id": r.get::<_, String>(0)?, "content": r.get::<_, String>(1)?, "scope": r.get::<_, String>(2)?,
                    "tags": r.get::<_, String>(3)?, "updated_at": r.get::<_, String>(4)?, "created_at": r.get::<_, String>(5)?}))
            })
            .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>());
        let t: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE scope=?",
                params![scope],
                |r| r.get(0),
            )
            .unwrap_or(0);
        (r.unwrap_or_default(), t)
    } else {
        let mut st = match c.prepare(
            "SELECT id,content,scope,tags,updated_at,created_at FROM memories ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        let r = st
            .query_map(params![limit, offset], |r| {
                Ok(json!({"id": r.get::<_, String>(0)?, "content": r.get::<_, String>(1)?, "scope": r.get::<_, String>(2)?,
                    "tags": r.get::<_, String>(3)?, "updated_at": r.get::<_, String>(4)?, "created_at": r.get::<_, String>(5)?}))
            })
            .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>());
        let t: i64 = c
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0);
        (r.unwrap_or_default(), t)
    };
    json_resp(
        200,
        &json!({"ok": true, "total": total, "items": rows}),
    )
}

async fn search(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let query = qval(&q, "q").trim().to_string();
    let limit = qint(&q, "limit", 10);
    let scope = qval(&q, "scope").trim().to_string();
    // 服务端默认: 缺省跟随 cfg.rerank_enabled(env NEBULA_RERANK_ENABLED),
    // 显式 rerank=1/true/yes 开, rerank=0/false/no 关 (对齐 session_rag.rs)。
    let want_rerank = match qval(&q, "rerank").trim().to_lowercase().as_str() {
        "" => cfg.rerank_enabled,
        "1" | "true" | "yes" | "on" => true,
        _ => false,
    };
    // fingerprint heuristic (1:1 port): ASCII-only 1..=30
    let is_fp = {
        let s = query.trim();
        !s.is_empty()
            && s.len() <= 30
            && s.chars().all(|c| c.is_ascii_alphanumeric()
                || matches!(c, '_' | '-' | '.' | ':' | '/' | '\\'))
    };
    let want_rerank = want_rerank && !is_fp;
    if query.is_empty() {
        return json_resp(400, &json!({"error": "q required"}));
    }
    let ck = (query.clone(), scope.clone(), limit, want_rerank);
    // cache hit
    if let Some((_ts, cached)) = SEARCH_CACHE
        .lock()
        .unwrap()
        .get(&ck)
        .cloned()
    {
        let cc = match db::db(cfg) {
            Ok(c) => c,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        let mut rows = HashMap::new();
        if let Some(ids) = cached.get("ids").and_then(|x| x.as_array()) {
            for idv in ids {
                let mid = idv.as_str().unwrap_or_default();
                if let Some(r) = cc
                    .query_row(
                        "SELECT content,scope,tags,updated_at,created_at FROM memories WHERE id=?",
                        params![mid],
                        |r| {
                            Ok(json!({"content": r.get::<_, String>(0)?, "scope": r.get::<_, String>(1)?,
                                "tags": r.get::<_, String>(2)?, "updated_at": r.get::<_, String>(3)?,
                                "created_at": r.get::<_, String>(4)?}))
                        },
                    ).ok()
                {
                    rows.insert(mid.to_string(), r);
                }
            }
        }
        let mut items: Vec<Value> = Vec::new();
        if let Some(sk_items) = cached.get("items").and_then(|x| x.as_array()) {
            for sk in sk_items {
                let mid = sk.get("id").and_then(|x| x.as_str()).unwrap_or_default();
                if let Some(r) = rows.get(mid) {
                    let mut m = json!({});
                    if let Some(mo) = m.as_object_mut() {
                        if let Some(sko) = sk.as_object() {
                            for (k, v) in sko {
                                mo.insert(k.clone(), v.clone());
                            }
                        }
                        if let Some(ro) = r.as_object() {
                            for (k, v) in ro {
                                mo.insert(k.clone(), v.clone());
                            }
                        }
                    }
                    items.push(m);
                }
            }
        }
        return json_resp(
            200,
            &json!({"ok": true, "total": items.len(), "items": items,
                "legs": cached.get("legs"), "cached": true}),
        );
    }
    let t_open0 = std::time::Instant::now();
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let t_db = t_open0.elapsed().as_millis();
    // leg 1: vector
    let t_emb0 = std::time::Instant::now();
    let mut vec_items: Vec<Value> = Vec::new();
    let mut t_faiss = 0u128;
    let mut n_cand = 0usize;
    if let Ok(qv) = embed::embed_one(cfg, &query) {
        let t_f0 = std::time::Instant::now();
        let cands = VS.search(&qv, (limit * 2).max(8) as usize);
        t_faiss = t_f0.elapsed().as_millis();
        n_cand = cands.len();
        for (mid, score) in cands {
            if !scope.is_empty() && scope != "all" {
                let sc: Option<String> = c
                    .query_row(
                        "SELECT scope FROM memories WHERE id=?",
                        params![mid],
                        |r| r.get(0),
                    ).ok();
                if sc.as_deref() != Some(scope.as_str()) {
                    continue;
                }
            }
            if let Some(r) = c
                .query_row(
                    "SELECT content,scope,tags FROM memories WHERE id=?",
                    params![mid],
                    |r| {
                        Ok(json!({"content": r.get::<_, String>(0)?, "scope": r.get::<_, String>(1)?, "tags": r.get::<_, String>(2)?}))
                    },
                ).ok()
            {
                vec_items.push(json!({"id": mid, "score": (score * 10000.0).round() / 10000.0,
                    "matched_by": "vector"}));
                if let Some(vi) = vec_items.last_mut().and_then(|v| v.as_object_mut()) {
                    if let Some(ro) = r.as_object() {
                        for (k, v) in ro {
                            vi.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
        }
    }
    // leg 2: FTS (bigram + bm25; the second element is the bm25 rank position, 0=best)
    let mut fts_items: Vec<Value> = Vec::new();
    for (rowid, _rank) in ops::fts_search(&c, &query, limit as usize, &scope) {
        if let Some(r) = c
            .query_row(
                "SELECT id,content,scope,tags FROM memories WHERE rowid=?",
                params![rowid],
                |r| {
                    Ok(json!({"id": r.get::<_, String>(0)?, "content": r.get::<_, String>(1)?,
                        "scope": r.get::<_, String>(2)?, "tags": r.get::<_, String>(3)?}))
                },
            ).ok()
        {
            if !scope.is_empty() && scope != "all" {
                if r.get("scope").and_then(|x| x.as_str()) != Some(scope.as_str()) {
                    continue;
                }
            }
            fts_items.push(json!({"rowid": rowid, "score": _rank as f64, "matched_by": "fts"}));
            if let Some(fi) = fts_items.last_mut().and_then(|v| v.as_object_mut()) {
                if let Some(ro) = r.as_object() {
                    for (k, v) in ro {
                        fi.insert(k.clone(), v.clone());
                    }
                }
            }
        }
    }
    let t_rest = t_open0.elapsed().as_millis();
    eprintln!("[search-timing] db_open={}ms emb={}ms faiss={}ms(n={} cand) rest={}ms total={}ms limit={}",
        t_db, t_emb0.elapsed().as_millis(), t_faiss, n_cand, t_rest.saturating_sub(t_db.saturating_add(0)), t_open0.elapsed().as_millis(), limit);
    drop(c);
    // RRF merge
    let merged = ops::rrf_merge(&[vec_items.clone(), fts_items.clone()], cfg.rrf_k);
    if merged.is_empty() {
        return json_resp(
            200,
            &json!({"ok": true, "total": 0, "items": [],
                "legs": {"vector": vec_items.len(), "fts": fts_items.len()}}),
        );
    }
    // optional rerank (cap at 3 for CPU latency)
    let mut final_items;
    let legs = json!({"vector": vec_items.len(), "fts": fts_items.len(),
        "rerank": cfg.rerank_enabled && want_rerank});
    if cfg.rerank_enabled && want_rerank {
        let rr_top: Vec<Value> = merged.iter().take(cfg.rerank_max_docs.min(merged.len())).cloned().collect();
        let ranked = embed::rerank(cfg, &query, &rr_top, limit as usize);
        let rr_ids: std::collections::HashSet<String> = ranked
            .iter()
            .map(|x| x.get("id").and_then(|x| x.as_str()).unwrap_or_default().to_string())
            .collect();
        let rest: Vec<Value> = merged
            .iter()
            .filter(|it| !rr_ids.contains(it.get("id").and_then(|x| x.as_str()).unwrap_or_default()))
            .take((limit as usize).saturating_sub(ranked.len()))
            .cloned()
            .collect();
        let mut combined = ranked;
        combined.extend(rest);
        final_items = combined;
    } else {
        final_items = merged.into_iter().take(limit as usize).collect::<Vec<_>>();
    }
    // backfill authoring timestamps
    {
        let cc_opt = db::db(cfg).ok();
        if let Some(cc) = cc_opt {
            for it in final_items.iter_mut() {
            let mid = it.get("id").and_then(|x| x.as_str()).unwrap_or_default();
            if let Some(r) = cc
                .query_row(
                    "SELECT created_at,updated_at FROM memories WHERE id=?",
                    params![mid],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                ).ok()
            {
                if let Some(o) = it.as_object_mut() {
                    o.insert("created_at".into(), json!(r.0));
                    o.insert("updated_at".into(), json!(r.1));
                }
            }
        }        }

    }
    // cache minimal skeleton
    let skel_items: Vec<Value> = final_items
        .iter()
        .map(|it| {
            let mut o = json!({});
            if let Some(io) = o.as_object_mut() {
                for k in ["id", "score", "sources", "matched_by"] {
                    if let Some(v) = it.get(k) {
                        io.insert(k.into(), v.clone());
                    }
                }
            }
            o
        })
        .collect();
    let skel_ids: Vec<String> = final_items
        .iter()
        .filter_map(|it| it.get("id").and_then(|x| x.as_str()))
        .map(|s| s.to_string())
        .collect();
    {
        let mut sc = SEARCH_CACHE.lock().unwrap();
        sc.insert(
            ck,
            (
                now_f64_(),
                json!({"items": skel_items, "ids": skel_ids, "legs": legs.clone()}),
            ),
        );
        if sc.len() > cfg.search_cache_max {
            let mut sorted: Vec<((String, String, i64, bool), f64)> = sc
                .iter()
                .map(|(k, (ts, _))| (k.clone(), *ts))
                .collect();
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (k, _) in sorted.iter().take((sc.len() / 4).max(1)) {
                sc.remove(k);
            }
        }
    }
    json_resp(
        200,
        &json!({"ok": true, "total": final_items.len(), "items": final_items, "legs": legs}),
    )
}

async fn stats(State(app): State<App>, req: axum::http::Request<Body>) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let total: i64 = c
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);
    let mut st = match c.prepare("SELECT scope,COUNT(*) c FROM memories GROUP BY scope ORDER BY c DESC") {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let scopes: Vec<(String, i64)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect())
        .unwrap_or_default();
    let ents: i64 = c.query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0)).unwrap_or(0);
    let edges: i64 = c.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0)).unwrap_or(0);
    let atoms: i64 = c.query_row("SELECT COUNT(*) FROM atoms_v2", [], |r| r.get(0)).unwrap_or(0);
    let vectors = VS.id_count();
    json_resp(
        200,
        &json!({"ok": true, "total_memories": total,
            "by_scopes": scopes.iter().map(|(s, c)| json!([s, c])).collect::<Vec<_>>(),
            "entities": ents, "edges": edges, "atoms": atoms, "vectors": vectors}),
    )
}

async fn trace(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let node = qval(&q, "node").trim().to_string();
    let hops = qint(&q, "hops", 2).min(3);
    if node.is_empty() {
        return json_resp(400, &json!({"error": "node required"}));
    }
    let cfg = &app.cfg;
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let mut visited = std::collections::HashSet::new();
    let mut edges_found: Vec<Value> = Vec::new();
    let mut frontier = vec![node.clone()];
    for _ in 0..hops {
        let mut nf = Vec::new();
        for n in &frontier {
            for dir in ["from_id", "to_id"] {
                let sql = format!("SELECT from_id,rel,to_id,note FROM edges WHERE {dir}=?");
                if let Ok(mut st) = c.prepare(&sql) {
                    if let Ok(mut rr) = st.query_map(params![n], |r| {
                        Ok(json!({"from_id": r.get::<_, String>(0)?, "rel": r.get::<_, String>(1)?,
                            "to_id": r.get::<_, String>(2)?, "note": r.get::<_, String>(3)?}))
                    }) {
                        for e in rr.filter_map(|x| x.ok()) {
                            let to_id = e.get("to_id").and_then(|x| x.as_str()).unwrap_or_default().to_string();
                            let from_id = e.get("from_id").and_then(|x| x.as_str()).unwrap_or_default().to_string();
                            let cand = if dir == "from_id" { to_id } else { from_id };
                            if !visited.contains(&cand) {
                                edges_found.push(e);
                                nf.push(cand);
                            }
                        }
                    }
                }
            }
        }
        visited.extend(frontier.iter().cloned());
        frontier = nf;
    }
    let mut node_ids = std::collections::HashSet::new();
    node_ids.insert(node);
    for e in &edges_found {
        node_ids.insert(e.get("from_id").and_then(|x| x.as_str()).unwrap_or_default().to_string());
        node_ids.insert(e.get("to_id").and_then(|x| x.as_str()).unwrap_or_default().to_string());
    }
    let mut nodes = serde_json::Map::new();
    for nid in &node_ids {
        if nid.is_empty() {
            continue;
        }
        if let Some(r) = c
            .query_row(
                "SELECT content FROM memories WHERE id=?",
                params![nid],
                |r| r.get::<_, String>(0),
            ).ok()
        {
            let truncated = &r[..r.len().min(200)];
            nodes.insert(nid.clone(), json!({"type": "memory", "content": truncated}));
            continue;
        }
        if let Some(r) = c
            .query_row(
                "SELECT name,kind FROM entities WHERE id=?",
                params![nid],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            ).ok()
        {
            nodes.insert(nid.clone(), json!({"type": "entity", "name": r.0, "kind": r.1}));
            continue;
        }
        nodes.insert(nid.clone(), json!({"type": "unknown"}));
    }
    json_resp(200, &json!({"ok": true, "nodes": nodes, "edges": edges_found}))
}

async fn entities(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let name_f = qval(&q, "name").trim().to_string();
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let rows = if !name_f.is_empty() {
        let mut st = match c.prepare(
            "SELECT id,name,kind,description FROM entities WHERE id=? OR name LIKE ? LIMIT 20",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params![name_f, format!("%{name_f}%")], |r| {
            Ok(json!({"id": r.get::<_, String>(0)?, "name": r.get::<_, String>(1)?,
                "kind": r.get::<_, String>(2)?, "description": r.get::<_, String>(3)?}))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default()
    } else {
        let mut st = match c.prepare("SELECT id,name,kind,description FROM entities LIMIT 100") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map([], |r| {
            Ok(json!({"id": r.get::<_, String>(0)?, "name": r.get::<_, String>(1)?,
                "kind": r.get::<_, String>(2)?, "description": r.get::<_, String>(3)?}))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default()
    };
    json_resp(200, &json!({"ok": true, "items": rows}))
}

async fn edges(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let node = qval(&q, "node").trim().to_string();
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let rows = if !node.is_empty() {
        let mut st = match c.prepare(
            "SELECT from_id,rel,to_id,note FROM edges WHERE from_id=? OR to_id=?",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params![node, node], |r| {
            Ok(json!({"from_id": r.get::<_, String>(0)?, "rel": r.get::<_, String>(1)?,
                "to_id": r.get::<_, String>(2)?, "note": r.get::<_, String>(3)?}))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default()
    } else {
        let mut st = match c.prepare("SELECT from_id,rel,to_id,note FROM edges LIMIT 200") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map([], |r| {
            Ok(json!({"from_id": r.get::<_, String>(0)?, "rel": r.get::<_, String>(1)?,
                "to_id": r.get::<_, String>(2)?, "note": r.get::<_, String>(3)?}))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default()
    };
    json_resp(200, &json!({"ok": true, "items": rows}))
}

// ---- nebula routes ----
async fn nebula_skills(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let qf = qval(&q, "q").trim().to_lowercase();
    let mut items = match ops::list_skills(cfg) {
        Ok(v) => v,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    if !qf.is_empty() {
        let exact: Vec<Value> = items
            .iter()
            .filter(|i| {
                i.get("skill")
                    .and_then(|x| x.as_str())
                    .and_then(|s| s.rsplit('/').next())
                    == Some(qf.as_str())
            })
            .cloned()
            .collect();
        let substr: Vec<Value> = items
            .iter()
            .filter(|i| {
                i.get("skill")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_lowercase().contains(&qf))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        let name_toks: std::collections::HashSet<String> = qf
            .split(|c| c == '-' || c == '_' || c == '/' || c == '-')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        let token_overlap: Vec<Value> = items
            .iter()
            .filter(|i| {
                let toks: std::collections::HashSet<String> = i
                    .get("skill")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .split(|c| c == '-' || c == '_' || c == '/')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                (name_toks.intersection(&toks).count()) >= 2
            })
            .cloned()
            .collect();
        items = if !exact.is_empty() {
            exact
        } else if !substr.is_empty() {
            substr
        } else if !token_overlap.is_empty() {
            token_overlap
        } else {
            items.into_iter().take(50).collect()
        };
    }
    json_resp(200, &json!({"ok": true, "items": items}))
}

async fn nebula_compose(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let name = qval(&q, "skill").trim().to_string();
    if name.is_empty() {
        return json_resp(400, &json!({"error": "skill required"}));
    }
    match ops::compose_skill(cfg, &name) {
        Ok(r) => {
            let mut o = json!({"ok": true});
            if let Some(oobj) = o.as_object_mut() {
                if let Some(robj) = r.as_object() {
                    for (k, v) in robj {
                        oobj.insert(k.clone(), v.clone());
                    }
                }
            }
            json_resp(200, &o)
        }
        Err(e) => json_resp(500, &json!({"error": e.to_string()})),
    }
}

async fn nebula_atoms(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let qy = qval(&q, "q").trim().to_string();
    let kind = qval(&q, "kind").trim().to_string();
    let limit = qint(&q, "limit", 30).min(100).max(0) as usize;
    let offset = qint(&q, "offset", 0).max(0);
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let mut where_ = vec!["a.id NOT IN (SELECT old_id FROM atom_supersedes)".to_string()];
    let mut params: Vec<RValue> = Vec::new();
    if !kind.is_empty() {
        where_.push("a.kind=?".into());
        params.push(RValue::Text(kind));
    }
    let ql = qy.trim().to_lowercase();
    if !ql.is_empty() {
        where_.push("(LOWER(a.title) LIKE ? OR LOWER(a.content) LIKE ?)".into());
        params.push(RValue::Text(format!("%{ql}%")));
        params.push(RValue::Text(format!("%{ql}%")));
    }
    let w = format!("WHERE {}", where_.join(" AND "));
    let total: i64 = {
        let mut st = match c.prepare(&format!("SELECT COUNT(*) FROM atoms_v2 a {w}")) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params_from_iter(params.iter().cloned()), |r| r.get(0)).ok().and_then(|mut rr| rr.next().and_then(|r| r.ok())).unwrap_or(0)
    };
    let mut st = match c.prepare(&format!(
        "SELECT a.id,a.kind,a.title,a.content,\
         (SELECT GROUP_CONCAT(DISTINCT s.skill) FROM atom_skills s WHERE s.atom_id=a.id) skills,\
         (SELECT new_id FROM atom_supersedes x WHERE x.old_id=a.id) supersededBy \
         FROM atoms_v2 a {w} ORDER BY a.updated_at DESC LIMIT ? OFFSET ?"
    )) {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let mut all = params;
    all.push(RValue::Integer(limit as i64));
    all.push(RValue::Integer(offset as i64));
    let rows = st
        .query_map(params_from_iter(vparams(&all).into_iter()), |r| {
            let skills: Option<String> = r.get(4)?;
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, String>(3)?,
                skills,
                r.get::<_, Option<String>>(5)?,
            ))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default();
    let items: Vec<Value> = rows
        .into_iter()
        .map(|(id, kind, title, content, skills, sup)| {
            let sk: Vec<String> = skills
                .as_deref()
                .map(|s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect())
                .unwrap_or_default();
            json!({"id": id, "kind": kind, "title": title, "content": content,
                "skills": sk, "supersededBy": sup})
        })
        .collect();
    json_resp(200, &json!({"ok": true, "total": total, "items": items}))
}

// ---- linghub-compatible /api/nebula/* (forwarded) ----
async fn api_nebula_stats(State(app): State<App>, req: axum::http::Request<Body>) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let atoms: i64 = c.query_row("SELECT COUNT(*) FROM atoms_v2", [], |r| r.get(0)).unwrap_or(0);
    let ents: i64 = c.query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0)).unwrap_or(0);
    let edges: i64 = c.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0)).unwrap_or(0);
    let refs: i64 = c.query_row("SELECT COUNT(*) FROM atom_skills", [], |r| r.get(0)).unwrap_or(0);
    let mut st = match c.prepare("SELECT kind,COUNT(*) c FROM atoms_v2 GROUP BY kind ORDER BY c DESC") {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let kinds: Vec<(String, i64)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect())
        .unwrap_or_default();
    let harvest: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM atom_skills WHERE skill LIKE 'harvest%'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let mut st = match c.prepare(
        "SELECT e.id,e.name,e.kind,COUNT(*) c FROM edges ed JOIN entities e ON e.id=ed.to_id \
         WHERE ed.rel IN ('part_of','pitfall_of') GROUP BY e.id ORDER BY c DESC LIMIT 12",
    ) {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let top: Vec<(String, String, String, i64)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect())
        .unwrap_or_default();
    let reflux: Option<String> = c
        .query_row("SELECT MAX(updated_at) t FROM atom_skills", [], |r| r.get(0)).ok();
    json_resp(
        200,
        &json!({
            "atoms": atoms, "entities": ents, "edges": edges, "refs": refs,
            "kinds": kinds.iter().map(|(k, c)| json!({"kind": k, "c": c})).collect::<Vec<_>>(),
            "harvestAtoms": harvest,
            "topEntities": top.iter().map(|(id, name, kind, c)| json!({"id": id, "name": name, "kind": kind, "c": c})).collect::<Vec<_>>(),
            "lastReflux": reflux,
        }),
    )
}

async fn api_nebula_atoms(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    // NOTE: the Python original has a typo `int((q.get("limit",...)[0]), 100)` — base-100 —
    // which makes /api/nebula/atoms?limit=N crash with ValueError for ANY explicit limit.
    // The Rust port parses base-10 (correct) and caps at 100, so this endpoint actually works.
    // (Deliberate improvement over the original; no-limit still defaults to 30 as in Python.)
    let limit = qint(&q, "limit", 30).min(100).max(0) as usize;
    let offset = qint(&q, "offset", 0).max(0);
    let kind = qval(&q, "kind").trim().to_string();
    let qy = qval(&q, "q").trim().to_lowercase();
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let mut where_ = vec!["a.id NOT IN (SELECT old_id FROM atom_supersedes)".to_string()];
    let mut params: Vec<RValue> = Vec::new();
    if !kind.is_empty() {
        where_.push("a.kind=?".into());
        params.push(RValue::Text(kind));
    }
    if !qy.is_empty() {
        where_.push("(LOWER(a.title) LIKE ? OR LOWER(a.content) LIKE ?)".into());
        params.push(RValue::Text(format!("%{qy}%")));
        params.push(RValue::Text(format!("%{qy}%")));
    }
    let w = format!("WHERE {}", where_.join(" AND "));
    let total: i64 = {
        let mut st = match c.prepare(&format!("SELECT COUNT(*) FROM atoms_v2 a {w}")) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params_from_iter(params.iter().cloned()), |r| r.get(0))
            .ok().and_then(|mut rr| rr.next().and_then(|r| r.ok()))
            .unwrap_or(0)
    };
    let mut st = match c.prepare(&format!(
        "SELECT a.id,a.kind,a.title,a.content,a.updated_at,\
         (SELECT GROUP_CONCAT(DISTINCT s.skill) FROM atom_skills s WHERE s.atom_id=a.id) skills,\
         (SELECT COUNT(*) FROM edges ed WHERE ed.from_id=a.id) outEdges,\
         (SELECT new_id FROM atom_supersedes x WHERE x.old_id=a.id) supersededBy \
         FROM atoms_v2 a {w} ORDER BY a.updated_at DESC LIMIT ? OFFSET ?"
    )) {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let mut all = params;
    all.push(RValue::Integer(limit as i64));
    all.push(RValue::Integer(offset as i64));
    let rows = st
        .query_map(params_from_iter(vparams(&all).into_iter()), |r| {
            let skills: Option<String> = r.get(5)?;
            let oute: i64 = r.get(6)?;
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                skills,
                oute,
                r.get::<_, Option<String>>(7)?,
            ))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default();
    let items: Vec<Value> = rows
        .into_iter()
        .map(|(id, kind, title, content, updated, skills, oute, sup)| {
            let sk: Vec<String> = skills
                .as_deref()
                .map(|s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect())
                .unwrap_or_default();
            json!({"id": id, "kind": kind, "title": title, "content": content,
                "updated_at": updated, "skills": sk, "outEdges": oute, "supersededBy": sup,
                "live": sup.is_none()})
        })
        .collect();
    json_resp(200, &json!({"total": total, "items": items}))
}

async fn api_nebula_atom(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let aid = qval(&q, "id").trim().to_string();
    if aid.is_empty() {
        return json_resp(400, &json!({"error": "id required"}));
    }
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let a: Option<String> = c
        .query_row("SELECT id FROM atoms_v2 WHERE id=?", params![aid], |r| r.get(0)).ok();
    if a.is_none() {
        return json_resp(404, &json!({"error": "atom not found"}));
    }
    // full atom row via dynamic columns
    let cols: Vec<String> = {
        let mut st = match c.prepare("SELECT * FROM atoms_v2 WHERE id=?") {
            Ok(s) => s,
            Err(_) => return json_resp(500, &json!({"error": "cols"})),
        };
        let cn = st.column_names().iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut rr = match st.query(params![aid]) {
            Ok(r) => r,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        if rr.next().ok().and_then(|x| x).is_some() {
            cn
        } else {
            Vec::new()
        }
    };
    let mut atom = serde_json::Map::new();
    if !cols.is_empty() {
        let mut st = match c.prepare("SELECT * FROM atoms_v2 WHERE id=?") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        if let Some(row) = st.query_map(params![aid], |r| {
            let mut m = serde_json::Map::new();
            for (i, col) in cols.iter().enumerate() {
                let v: Value = value_ref_to_json(r.get_ref(i)?);
                m.insert(col.clone(), v);
            }
            Ok(m)
        })
        .ok().and_then(|mut rr| rr.next().and_then(|r| r.ok()))
        {
            atom = row;
        }
    }
    let refs: Vec<Value> = {
        let mut st = match c.prepare(
            "SELECT skill,section,position FROM atom_skills WHERE atom_id=? ORDER BY skill,position",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params![aid], |r| {
            Ok(json!({"skill": r.get::<_, String>(0)?, "section": r.get::<_, Option<String>>(1)?, "position": r.get::<_, i64>(2)?}))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    };
    let outE: Vec<Value> = {
        let mut st = match c.prepare("SELECT to_id,rel FROM edges WHERE from_id=? LIMIT 50") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params![aid], |r| {
            Ok(json!({"to_id": r.get::<_, String>(0)?, "rel": r.get::<_, String>(1)?}))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    };
    let inE: Vec<Value> = {
        let mut st = match c.prepare("SELECT from_id,rel FROM edges WHERE to_id=? LIMIT 50") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params![aid], |r| {
            Ok(json!({"from_id": r.get::<_, String>(0)?, "rel": r.get::<_, String>(1)?}))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    };
    let sup: Option<String> = c
        .query_row("SELECT new_id FROM atom_supersedes WHERE old_id=?", params![aid], |r| r.get(0)).ok();
    json_resp(
        200,
        &json!({"atom": atom, "refs": refs, "outEdges": outE, "inEdges": inE, "supersededBy": sup}),
    )
}

async fn api_nebula_skills(State(app): State<App>, req: axum::http::Request<Body>) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    match ops::list_skills(cfg) {
        Ok(items) => json_resp(200, &json!({"ok": true, "items": items})),
        Err(e) => json_resp(500, &json!({"error": e.to_string()})),
    }
}

async fn api_nebula_skill(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let name = qval(&q, "name").trim().to_string();
    if name.is_empty() {
        return json_resp(400, &json!({"error": "name required"}));
    }
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let bare = name.rsplit('/').next().unwrap_or(&name);
    let mut display = name.clone();
    let mut skills: Vec<String> = Vec::new();
    let mut auto = false;
    if ops::SCOPE_PREFIXES.contains(&bare) {
        skills = ops::SCOPE_PREFIXES
            .iter()
            .filter(|p| **p == bare || **p == "mem")
            .map(|s| s.to_string())
            .collect();
        display = "mem".into();
        auto = true;
    } else if name == bare && (bare.starts_with("mem/") || bare.starts_with("memories/")) {
        let slash = bare.find('/').unwrap();
        let sk = &bare[slash + 1..];
        // use internal scope_variants via compose path is overkill; call list path
        let variants = ops::scope_variants_public(&c, sk);
        skills = if variants.is_empty() {
            vec![name.clone()]
        } else {
            variants
        };
        display = if sk.is_empty() {
            "mem".into()
        } else {
            format!("mem/{sk}")
        };
        auto = true;
    } else {
        skills = vec![name.clone()];
    }
    let marks = vec!["?"; skills.len()].join(",");
    let mut st = match c.prepare(&format!(
        "SELECT a.id,a.kind,a.title,a.content,s.section,s.position FROM atom_skills s JOIN atoms_v2 a ON a.id=s.atom_id \
         WHERE s.skill IN ({marks}) ORDER BY s.section,s.position"
    )) {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let vals: Vec<RValue> = skills.iter().map(|s| RValue::Text(s.clone())).collect();
    let rows: Vec<(String, String, Option<String>, String, Option<String>, i64)> = st
        .query_map(params_from_iter(vparams(&vals).into_iter()), |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
            ))
        }).ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default();
    let mut secs: Vec<String> = Vec::new();
    let mut bySec: std::collections::HashMap<String, Vec<Value>> = Default::default();
    let mut seen = std::collections::HashSet::new();
    for (aid, kind, title, content, sec, pos) in rows {
        if seen.contains(&aid) {
            continue;
        }
        seen.insert(aid.clone());
        let sec = sec.as_deref().filter(|s| !s.is_empty()).unwrap_or("(top)").to_string();
        if !bySec.contains_key(&sec) {
            bySec.insert(sec.clone(), Vec::new());
            secs.push(sec.clone());
        }
        bySec.get_mut(&sec).unwrap().push(json!({
            "id": aid, "kind": kind, "title": title, "content": content,
            "section": sec, "position": pos,
        }));
    }
    let mut resp = json!({
        "skill": display,
        "sections": secs.iter().map(|s| json!({"section": s, "atoms": bySec.get(s).cloned().unwrap_or_default()})).collect::<Vec<_>>(),
    });
    if skills.len() > 1 || skills[0] != name {
        if let Some(r) = resp.as_object_mut() {
            r.insert("variants".into(), Value::Array(skills.iter().map(|s| json!(s)).collect()));
        }
    }
    if skills.first().map(|s| ops::is_scope_skill(s)).unwrap_or(false) {
        if let Some(r) = resp.as_object_mut() {
            r.insert("auto".into(), json!(true));
        }
    }
    json_resp(200, &resp)
}

async fn api_nebula_entities(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let limit = qint(&q, "limit", 50).min(200).max(0) as usize;
    let offset = qint(&q, "offset", 0).max(0);
    let eq = qval(&q, "q").trim().to_lowercase();
    let kind_f = qval(&q, "kind").trim().to_string();
    let status_f = qval(&q, "status").trim().to_string();
    let parent_f = qval(&q, "parent").trim().to_string();
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let mut where_: Vec<String> = Vec::new();
    let mut params: Vec<RValue> = Vec::new();
    if !eq.is_empty() {
        where_.push("(LOWER(e.name) LIKE ? OR LOWER(e.id) LIKE ? OR e.aliases LIKE ?)".into());
        params.push(RValue::Text(format!("%{eq}%")));
    }
    if !kind_f.is_empty() {
        where_.push("e.kind=?".into());
        params.push(RValue::Text(kind_f));
    }
    if !status_f.is_empty() {
        where_.push("e.status=?".into());
        params.push(RValue::Text(status_f));
    }
    if !parent_f.is_empty() {
        where_.push("e.parent_id=?".into());
        params.push(RValue::Text(parent_f));
    }
    let w = if where_.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_.join(" AND "))
    };
    let count: i64 = {
        let mut st = match c.prepare(&format!("SELECT COUNT(*) FROM entities e {w}")) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params_from_iter(params.iter().cloned()), |r| r.get(0))
            .ok().and_then(|mut rr| rr.next().and_then(|r| r.ok()))
            .unwrap_or(0)
    };
    let mut st = match c.prepare(&format!(
        "SELECT e.id,e.name,e.kind,e.description,e.parent_id,e.status,e.aliases,\
         (SELECT COUNT(*) FROM edges ed WHERE ed.to_id=e.id AND ed.rel IN ('part_of','pitfall_of')) atomCount \
         FROM entities e {w} ORDER BY e.name LIMIT ? OFFSET ?"
    )) {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let mut all = params;
    all.push(RValue::Integer(limit as i64));
    all.push(RValue::Integer(offset as i64));
    let rows = st
        .query_map(params_from_iter(vparams(&all).into_iter()), |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, Option<String>>(6)?,
                r.get::<_, i64>(7)?,
            ))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default();
    let items: Vec<Value> = rows
        .into_iter()
        .map(|(id, name, kind, desc, parent, status, aliases, atomcount)| {
            let al: Vec<Value> = aliases
                .as_deref()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();
            json!({"id": id, "name": name, "kind": kind, "description": desc,
                "parent_id": parent, "status": status, "aliases": al, "atomCount": atomcount})
        })
        .collect();
    json_resp(200, &json!({"total": count, "items": items}))
}

async fn api_nebula_entity(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let eid = qval(&q, "id").trim().to_string();
    if eid.is_empty() {
        return json_resp(400, &json!({"error": "id required"}));
    }
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    if let Some(skill) = eid.strip_prefix("skill:") {
        let bare = skill.rsplit('/').next().unwrap_or(skill);
        let mut display = skill.to_string();
        let mut skills: Vec<String> = Vec::new();
        let mut auto = false;
        if ops::SCOPE_PREFIXES.contains(&bare) {
            skills = ops::SCOPE_PREFIXES
                .iter()
                .filter(|p| **p == bare || **p == "mem")
                .map(|s| s.to_string())
                .collect();
            display = "mem".into();
            auto = true;
        } else if bare == skill && (bare.starts_with("mem/") || bare.starts_with("memories/")) {
            let slash = bare.find('/').unwrap();
            let sk = &bare[slash + 1..];
            let variants = ops::scope_variants_public(&c, sk);
            skills = if variants.is_empty() {
                vec![display.clone()]
            } else {
                variants
            };
            display = if sk.is_empty() {
                "mem".into()
            } else {
                format!("mem/{sk}")
            };
            auto = true;
        } else {
            skills = vec![skill.to_string()];
        }
        if skills.is_empty() {
            skills = vec![display.clone()];
        }
        let marks = vec!["?"; skills.len()].join(",");
        let vals: Vec<RValue> = skills.iter().map(|s| RValue::Text(s.clone())).collect();
        let mut st = match c.prepare(&format!(
            "SELECT a.id,a.kind,a.title,a.content,a.updated_at,s.skill,s.section \
             FROM atom_skills s JOIN atoms_v2 a ON a.id=s.atom_id \
             WHERE s.skill IN ({marks}) AND a.id NOT IN (SELECT old_id FROM atom_supersedes) \
             ORDER BY s.section LIMIT 800"
        )) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        let atoms: Vec<(String, String, Option<String>, String, Option<String>, String, Option<String>)> = st
            .query_map(params_from_iter(vparams(&vals).into_iter()), |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            }).ok()
            .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
            .unwrap_or_default();
        let mut uniq = std::collections::HashSet::new();
        let mut atomsOut: Vec<Value> = Vec::new();
        for (aid, kind, title, content, upd, sk, sec) in atoms {
            if !uniq.insert(aid.clone()) {
                continue;
            }
            atomsOut.push(json!({"id": aid, "kind": kind, "title": title, "content": content,
                "updated_at": upd, "skills": [sk], "section": sec}));
        }
        let secs: Vec<(Option<String>, i64)> = {
            let vals2: Vec<RValue> = skills.iter().map(|s| RValue::Text(s.clone())).collect();
            let mut st2 = match c.prepare(&format!(
                "SELECT section,COUNT(*) n FROM atom_skills WHERE skill IN ({marks}) GROUP BY section"
            )) {
                Ok(s) => s,
                Err(_) => return json_resp(500, &json!({"error": "secs"})),
            };
            st2.query_map(params_from_iter(vparams(&vals2).into_iter()), |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, i64>(1)?))).ok()
                .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
                .unwrap_or_default()
        };
        let mut resp = json!({
            "entity": {"id": eid, "name": display.rsplit('/').next().unwrap_or(&display),
                "kind": "skill", "description": format!("{display} · {} atoms{}", atomsOut.len(), if auto { " (auto)" } else { "" })},
            "atoms": atomsOut, "neighbors": [],
            "sections": secs.iter().map(|(s, n)| json!({"section": s, "n": n})).collect::<Vec<_>>(),
            "auto": auto,
        });
        if auto {
            if let Some(r) = resp.as_object_mut() {
                r.insert("variants".into(), Value::Array(skills.iter().map(|s| json!(s)).collect()));
            }
        }
        return json_resp(200, &resp);
    }
    // regular entity
    let e: Option<String> = c
        .query_row("SELECT id FROM entities WHERE id=?", params![eid], |r| r.get(0)).ok();
    if e.is_none() {
        return json_resp(404, &json!({"error": "entity not found"}));
    }
    let cols: Vec<String> = {
        let mut st = match c.prepare("SELECT * FROM entities WHERE id=?") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        let cn = st.column_names().iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut rr = match st.query(params![eid]) {
            Ok(r) => r,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        if rr.next().ok().and_then(|x| x).is_some() {
            cn
        } else {
            Vec::new()
        }
    };
    let mut edict = serde_json::Map::new();
    {
        let mut st = match c.prepare("SELECT * FROM entities WHERE id=?") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        if let Some(row) = st.query_map(params![eid], |r| {
            let mut m = serde_json::Map::new();
            for (i, col) in cols.iter().enumerate() {
                let v: Value = value_ref_to_json(r.get_ref(i)?);
                m.insert(col.clone(), v);
            }
            Ok(m)
        })
        .ok().and_then(|mut rr| rr.next().and_then(|r| r.ok()))
        {
            edict = row;
        }
    }
    edict.entry("parent_id".to_string()).or_insert_with(|| json!(""));
    edict.entry("status".to_string()).or_insert_with(|| json!("active"));
    let aliases: Vec<Value> = edict
        .get("aliases")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    edict.insert("aliases".to_string(), Value::Array(aliases));
    // atoms via edges
    let mut st = match c.prepare(
        "SELECT a.id,a.kind,a.title,a.content,a.updated_at,\
         (SELECT GROUP_CONCAT(DISTINCT s.skill) FROM atom_skills s WHERE s.atom_id=a.id) skills \
         FROM edges ed JOIN atoms_v2 a ON a.id=ed.from_id \
         WHERE ed.to_id=? AND ed.rel IN ('part_of','pitfall_of') AND a.id NOT IN (SELECT old_id FROM atom_supersedes) \
         ORDER BY a.updated_at DESC LIMIT 400",
    ) {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let atomRows: Vec<(String, String, Option<String>, String, Option<String>, Option<String>)> = st
        .query_map(params![eid], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
            ))
        }).ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default();
    let memRows: Vec<(String, String, String, String, Option<String>)> = {
        let mut st = match c.prepare(
            "SELECT m.id,'memory' kind,substr(m.content,1,80) title,m.content,m.updated_at \
             FROM edges ed JOIN memories m ON m.id=ed.from_id \
             WHERE ed.to_id=? AND ed.rel IN ('part_of','pitfall_of') \
             AND ed.from_id NOT IN (SELECT id FROM atoms_v2) \
             ORDER BY m.updated_at DESC LIMIT 400",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params![eid], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
            ))
        }).ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default()
    };
    let mut allAtoms: Vec<Value> = atomRows
        .into_iter()
        .map(|(id, kind, title, content, updated, skills)| {
            let sk: Vec<String> = skills
                .as_deref()
                .map(|s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect())
                .unwrap_or_default();
            json!({"id": id, "kind": kind, "title": title, "content": content, "updated_at": updated, "skills": sk})
        })
        .chain(
            memRows
                .into_iter()
                .map(|(id, kind, title, content, updated)| {
                    json!({"id": id, "kind": kind, "title": title, "content": content, "updated_at": updated, "skills": []})
                }),
        )
        .collect();
    allAtoms.sort_by(|a, b| {
        let au = a.get("updated_at").and_then(|x| x.as_str()).unwrap_or("");
        let bu = b.get("updated_at").and_then(|x| x.as_str()).unwrap_or("");
        bu.cmp(au)
    });
    let neigh: Vec<Value> = {
        let mut st = match c.prepare(
            "SELECT to_id,rel FROM edges WHERE from_id=? AND to_id LIKE 'ent:%' LIMIT 40",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params![eid], |r| {
            Ok(json!({"to_id": r.get::<_, String>(0)?, "rel": r.get::<_, String>(1)?}))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    };
    json_resp(
        200,
        &json!({"entity": {
            "id": edict.get("id").cloned().unwrap_or(json!(eid)),
            "name": edict.get("name").cloned().unwrap_or(json!("")),
            "kind": edict.get("kind").cloned().unwrap_or(json!("")),
            "description": edict.get("description").cloned().unwrap_or(json!("")),
        }, "atoms": allAtoms, "neighbors": neigh}),
    )
}

async fn api_nebula_graph(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let show_arch = qval(&q, "show_archived") == "1";
    let sf = if show_arch { "" } else { "AND e.status != 'archived'" };
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let mut st = match c.prepare(&format!(
        "SELECT e.id,e.name,e.kind,e.description,e.parent_id,e.status,\
         (SELECT COUNT(*) FROM edges ed WHERE ed.to_id=e.id AND ed.rel IN ('part_of','pitfall_of') \
         AND (EXISTS(SELECT 1 FROM atoms_v2 a WHERE a.id=ed.from_id) OR EXISTS(SELECT 1 FROM memories m WHERE m.id=ed.from_id))) atomCount \
         FROM entities e WHERE atomCount>0 {sf}"
    )) {
        Ok(s) => s,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let entRows: Vec<(String, String, String, String, Option<String>, Option<String>, i64)> = st
        .query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
                r.get(6)?,
            ))
        }).ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default();
    let skillRows: Vec<(String, String, String)> = {
        let mut st = match c.prepare(
            "SELECT 'skill:'||s.skill id,substr(s.skill,instr(s.skill,'/')+1) name,'skill' kind,COUNT(*) atomCount \
             FROM atom_skills s GROUP BY s.skill HAVING atomCount>0",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        // Faithful to Python: ent dict uses r[2] (kind) as atomCount -> string "skill"
        // (original quirk; r[3] the real count is unused for skill entries).
        st.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .ok()
        .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_default()
    };
    let entEdges: Vec<(String, String, String)> = {
        let mut st = match c.prepare(
            "SELECT from_id,to_id,rel FROM edges WHERE from_id LIKE 'ent:%' AND to_id LIKE 'ent:%' LIMIT 2000",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).ok()
            .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    let skillEdges: Vec<(String, String, String)> = {
        let mut st = match c.prepare(
            "SELECT ed.from_id,ed.to_id,ed.rel FROM edges ed JOIN atom_skills s ON s.atom_id=ed.from_id \
             WHERE ed.rel IN ('pitfall_of','part_of') AND ed.to_id LIKE 'skill:%' GROUP BY ed.from_id,ed.to_id LIMIT 3000",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).ok()
            .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    let entAtomEdges: Vec<(String, String, i64)> = {
        let mut st = match c.prepare(
            "SELECT ed.from_id,ed.to_id,COUNT(*) w FROM edges ed WHERE ed.rel IN ('pitfall_of','part_of') \
             AND ed.to_id LIKE 'ent:%' GROUP BY ed.from_id,ed.to_id LIMIT 3000",
        ) {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).ok()
            .map(|mut rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    let mut ents: Vec<Value> = Vec::new();
    for (id, name, kind, desc, parent, status, atomcount) in entRows {
        ents.push(json!({"id": id, "name": name, "kind": kind, "description": desc,
            "parent_id": parent, "status": status, "atomCount": atomcount}));
    }
    for (id, name, atomcount) in skillRows {
        ents.push(json!({"id": id, "name": name, "kind": "skill", "atomCount": atomcount}));
    }
    let edges: Vec<Value> = entEdges
        .into_iter()
        .map(|(f, t, r)| json!({"from_id": f, "to_id": t, "rel": r}))
        .chain(skillEdges.into_iter().map(|(f, t, r)| json!({"from_id": f, "to_id": t, "rel": r})))
        .chain(entAtomEdges.into_iter().map(|(f, t, w)| json!({"from_id": f, "to_id": t, "rel": "aggregated", "w": w})))
        .collect();
    json_resp(200, &json!({"entities": ents, "edges": edges}))
}

async fn api_nebula_entity_tree(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let eid = qval(&q, "id").trim().to_string();
    if eid.is_empty() {
        return json_resp(400, &json!({"error": "id required"}));
    }
    let c = match db::db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
    };
    let root: Option<String> = c
        .query_row("SELECT id FROM entities WHERE id=?", params![eid], |r| r.get(0)).ok();
    if root.is_none() {
        return json_resp(404, &json!({"error": "entity not found"}));
    }
    let cols: Vec<String> = {
        let mut st = match c.prepare("SELECT * FROM entities WHERE id=?") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        let cn = st.column_names().iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut rr = match st.query(params![eid]) {
            Ok(r) => r,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        if rr.next().ok().and_then(|x| x).is_some() {
            cn
        } else {
            Vec::new()
        }
    };
    let rootD = {
        let mut st = match c.prepare("SELECT * FROM entities WHERE id=?") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        st.query_map(params![eid], |r| {
            let mut m = serde_json::Map::new();
            for (i, col) in cols.iter().enumerate() {
                let v: Value = value_ref_to_json(r.get_ref(i)?);
                m.insert(col.clone(), v);
            }
            Ok(m)
        })
        .ok().and_then(|mut rr| rr.next().and_then(|r| r.ok()))
        .unwrap_or_default()
    };
    let mut children: Vec<Value> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(eid.clone());
    while let Some(cur) = queue.pop_front() {
        let mut st = match c.prepare("SELECT * FROM entities WHERE parent_id=? ORDER BY name") {
            Ok(s) => s,
            Err(e) => return json_resp(500, &json!({"error": e.to_string()})),
        };
        let rows: Vec<Value> = st
            .query_map(params![cur], |r| {
                let mut m = serde_json::Map::new();
                for (i, col) in cols.iter().enumerate() {
                    let v: Value = value_ref_to_json(r.get_ref(i)?);
                    m.insert(col.clone(), v);
                }
                Ok(m)
            })
            .ok()
            .map(|mut rr| rr.filter_map(|x| x.ok()).map(Value::Object).collect())
            .unwrap_or_default();
        for d in rows {
            children.push(d.clone());
            if let Some(id) = d.get("id").and_then(|x| x.as_str()) {
                queue.push_back(id.to_string());
            }
        }
    }
    json_resp(200, &json!({"root": rootD, "children": children}))
}

async fn ops_post(
    State(app): State<App>,
    req: axum::http::Request<Body>,
) -> Result<impl IntoResponse, axum::http::StatusCode> {
    if !auth_ok(&app, &req) {
        return Ok(json_resp(401, &json!({"error": "unauthorized"})));
    }
    let path = req.uri().path().to_string();
    if path != "/ops" {
        return Ok(json_resp(404, &json!({"error": "not found"})));
    }
    let body: Vec<u8> = match axum::body::to_bytes(req.into_body(), 1 << 22).await {
        Ok(b) => b.to_vec(),
        Err(_) => return Ok(json_resp(413, &json!({"error": "body too large"}))),
    };
    let data: Value = serde_json::from_slice(&body)
        .map_err(|_| axum::http::StatusCode::BAD_REQUEST)?;
    let ops = if data.is_object() {
        data.get("ops").cloned().unwrap_or(Value::Null)
    } else {
        data.clone()
    };
    let ops = match ops {
        Value::Array(v) => v,
        _ => return Ok(json_resp(400, &json!({"error": "expected ops array"}))),
    };
    let cfg = &app.cfg;
    let mut accepted = 0;
    for op in ops.iter().take(1000) {
        if ops::apply_op(cfg, &VS, op) {
            accepted += 1;
        }
    }
    let mx: f64 = {
        let c = match db::db(cfg) {
            Ok(c) => c,
            Err(_) => return Ok(json_resp(500, &json!({"error": "db"}))),
        };
        c.query_row("SELECT MAX(ts) FROM ops", [], |r| r.get(0))
            .unwrap_or(0.0)
    };
    let vectors = VS.id_count();
    Ok(json_resp(
        200,
        &json!({"ok": true, "accepted": accepted, "max_ts": mx, "vectors": vectors}),
    ))
}

#[tokio::main]
async fn main() {
    let cfg = Arc::new(Config::load());
    if !cfg.data.exists() {
        let _ = std::fs::create_dir_all(&cfg.data);
    }
    if let Err(e) = db::init_db(&cfg) {
        eprintln!("[init] {e}");
        std::process::exit(1);
    }
    db::backfill_fts(&cfg);
    let token = Arc::new(std::sync::RwLock::new(config::load_token(&cfg.token_file)));
    // 9/13: MCP 网关并入本进程 —— 9300 streamable-HTTP + Bearer (与核心 REST 同进程)。
    // 必须在 token 被 move 进 App 之前 spawn (Arc 非 Copy)。
    {
        let base = format!("http://127.0.0.1:{}", cfg.port);
        let mport = cfg.mcp_port;
        let tok = token.read().unwrap().clone().unwrap_or_default();
        if mport == 0 {
            eprintln!("[nebula] MCP 监听关闭 (NEBULA_MCP_PORT=0)");
        } else {
            let mcp = Arc::new(nebula_mcp_rs::McpServer::new_with(&base, &tok));
            let mcp_tok = tok;
            let host = cfg.mcp_host.clone();
            tokio::spawn(async move {
                nebula_mcp_rs::serve_mcp(mcp, &mcp_tok, &host, mport).await;
            });
        }
    }
    let app = App {
        cfg: cfg.clone(),
        token,
    };
    // load vectors + start bg embed + drain
    {
        let cfg = &app.cfg;
        VS.load_or_start(cfg);
    }
    // session RAG: 13k+ 段向量 + FTS 构建较重 → 后台线程加载, 不阻塞服务启动
    {
        let cfg2 = app.cfg.clone();
        std::thread::spawn(move || session_rag::load_or_start(&cfg2));
    }
    let port = cfg.port;
    let router = Router::new()
        .route("/health", get(health))
        .route("/ops", get(ops_list).post(ops_post))
        .route("/full", get(full))
        .route("/memories", get(memories))
        .route("/search", get(search))
        .route("/stats", get(stats))
        .route("/trace", get(trace))
        .route("/entities", get(entities))
        .route("/edges", get(edges))
        .route("/nebula/skills", get(nebula_skills))
        .route("/nebula/compose", get(nebula_compose))
        .route("/nebula/atoms", get(nebula_atoms))
        .route("/api/nebula/stats", get(api_nebula_stats))
        .route("/api/nebula/atoms", get(api_nebula_atoms))
        .route("/api/nebula/atom", get(api_nebula_atom))
        .route("/api/nebula/skills", get(api_nebula_skills))
        .route("/api/nebula/skill", get(api_nebula_skill))
        .route("/api/nebula/entities", get(api_nebula_entities))
        .route("/api/nebula/entity", get(api_nebula_entity))
        .route("/api/nebula/graph", get(api_nebula_graph))
        .route("/api/nebula/entity-tree", get(api_nebula_entity_tree))
        // ---------- LingHub 业务后端 (阶段二: 按域拆模块, 1:1 复刻 CF worker) ----------
        .merge(metrics_api::router())
        .merge(ws_api::router())
        .merge(graph_api::router())
        .merge(session_rag::router())
        .merge(atoms_api::router())
        .merge(memories_api::router())
        .merge(dash_api::router())
        .merge(media_api::router())
        .merge(web_api::router())
        .fallback(web_api::serve)
        .with_state(app.clone());
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let tok = load_token_label(&app.token);
    eprintln!(
        "nebula-sync v3(rust) listening :{port} token={tok}",
    );
    // OPTIONS 预检: 浏览器跨域 PUT/DELETE 前置, 统一回 204 + CORS 头
    let router = router.layer(middleware::from_fn(
        |req: axum::http::Request<axum::body::Body>, next: axum::middleware::Next| async move {
            let mut r: axum::response::Response = if req.method() == axum::http::Method::OPTIONS {
                let mut r = Response::new(Body::empty());
                *r.status_mut() = axum::http::StatusCode::NO_CONTENT;
                let h = r.headers_mut();
                h.insert("Access-Control-Allow-Origin", axum::http::HeaderValue::from_static("*"));
                h.insert("Access-Control-Allow-Methods", axum::http::HeaderValue::from_static("GET,POST,PUT,DELETE,OPTIONS"));
                h.insert("Access-Control-Allow-Headers", axum::http::HeaderValue::from_static("Content-Type,Authorization"));
                r
            } else {
                next.run(req).await
            };
            r.headers_mut().insert("Access-Control-Allow-Origin", axum::http::HeaderValue::from_static("*"));
            Ok::<_, std::convert::Infallible>(r)
        },
    ));
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("serve");
}

fn load_token_label(t: &std::sync::RwLock<Option<String>>) -> &'static str {
    if t.read().unwrap().as_deref().unwrap_or("").is_empty() {
        "none"
    } else {
        "set"
    }
}

// re-export for ops.rs
