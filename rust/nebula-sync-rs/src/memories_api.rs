//! 记忆写入域 (阶段三): Rust 版 CF writeMemory + /api/memories* 全套。
//! 1:1 复刻 CF linghub worker:
//!   - POST /api/memories       写单条 (hash 精确去重 → 向量召回候选 → LLM 判重 SAME/MERGE/DIFF)
//!   - POST /api/memories/batch 批量写 (每条独立走 writeMemory, 上限 100)
//!   - GET  /api/memories/{id}   单条
//!   - PUT  /api/memories/{id}   更新 content/scope/tags (重算 hash+向量)
//!   - DELETE /api/memories/{id} 删除 (含向量)
//!   - POST /api/memories/batch-delete
//!   - GET  /api/memories/search 向量语义检索
//! CF 是 Vectorize, Rust 侧等价 = 本进程 VS (mem_vectors) + llama embed 8300。
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post, put, delete};
use axum::Router;
use serde_json::{json, Value};

use crate::api_common::{bad_request, get_db, now_iso, ok200, ok201, db_err, gen_id, hash16};
use crate::{auth_ok, json_resp, qparams, qval};

// 对外暴露: main.rs 需访问 VS (crate::VS 改 pub(crate))
use crate::VS;

pub fn router() -> Router<crate::App> {
    Router::new()
        .route("/api/memories", post(mem_post))
        .route("/api/memories/batch", post(mem_batch_post))
        .route("/api/memories/batch-delete", post(mem_batch_delete))
        .route("/api/memories/search", get(mem_search))
        .route("/api/memories/{id}", get(mem_get_one).put(mem_put).delete(mem_delete))
}

// ---------- 写核心: writeMemory ----------

/// writeMemory: 返回 (Value 结果, 201/200)
async fn write_memory(
    app: &crate::App,
    content: &str,
    scope: &str,
    tags: &str,
    force: bool,
) -> (Value, u16) {
    let now = now_iso();
    // 向量 (force/非force 共用: 正常写入要用; 召回段若 force 跳过)
    let vector = crate::embed::embed_one(&app.cfg, content).ok();
    if !force {
        // 1. hash 精确去重
        let h = hash16(content);
        let dup = {
            let c = match get_db(&app.cfg) {
                Ok(c) => c,
                Err(_) => return (json!({"error": "db unavailable"}), 500),
            };
            c.query_row(
                "SELECT id FROM memories WHERE hash = ?",
                rusqlite::params![h],
                |r| r.get::<_, String>(0),
            )
            .ok()
        };
        if let Some(id) = dup {
            return (
                json!({"ok": true, "status": "dedup-exact", "id": id, "existing_id": id,
                       "message": "已存在完全相同记忆, 已剔除"}),
                200,
            );
        }
        // 2. 向量召回候选 (CF: Vectorize.query top8, score>=0.62, 同 scope)
        let _vector = crate::embed::embed_one(&app.cfg, content).ok();
        if let Some(v) = &vector {
            let cands = VS.search(v, 8);
            let c = match get_db(&app.cfg) {
                Ok(c) => c,
                Err(_) => return (json!({"error": "db unavailable"}), 500),
            };
            // 取 score 最高的合格候选 (Rust search 按相似度降序; CF score=相似度)
            for (mid, score) in cands.iter() {
                let (ms, mc) = match c.query_row(
                    "SELECT scope, content FROM memories WHERE id = ?",
                    rusqlite::params![mid],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                ) {
                    Ok(x) => x,
                    Err(_) => continue, // 向量孤儿, 跳过
                };
                if ms != scope || (*score as f64) < app.cfg.memory_dedup_vec_min {
                    continue;
                }
                // 3. LLM 判断
                let verdict = classify_duplicate(app, content, &mc, *score).await;
                match verdict {
                    "same" => {
                        return (
                            json!({"ok": true, "status": "dedup-semantic", "id": mid,
                                   "existing_id": mid, "score": score,
                                   "message": "语义相同, 已剔除"}),
                            200,
                        );
                    }
                    "merge" => {
                        let merged = merge_memories(app, content, &mc).await;
                        let merged_h = hash16(&merged);
                        if let Err(e) = c.execute(
                            "UPDATE memories SET content = ?, hash = ?, updated_at = ? WHERE id = ?",
                            rusqlite::params![merged, merged_h, now, mid],
                        ) {
                            return (json!({"error": e.to_string()}), 500);
                        }
                        // 向量更新 (失败忽略, 与 CF 一致)
                        if let Ok(mv) = crate::embed::embed_one(&app.cfg, &merged) {
                            let vb = crate::vector_store::f32_slice_to_bytes(&mv);
                            let _ = c.execute(
                                "INSERT OR REPLACE INTO mem_vectors (id, vec, updated_at) VALUES (?, ?, ?)",
                                rusqlite::params![mid, vb, crate::ops::now_f64()],
                            );
                            // 通知 VS 重载
                            crate::VS_NOTIFY.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        return (
                            json!({"ok": true, "status": "dedup-merged", "id": mid,
                                   "existing_id": mid, "score": score, "content": merged,
                                   "message": "已与相似记忆合并"}),
                            200,
                        );
                    }
                    _ => {}
                }
            }
        }
    }
    // 正常写入
    let id = gen_id();
    let h2 = hash16(content);
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(_) => return (json!({"error": "db unavailable"}), 500),
    };
    if let Err(e) = c.execute(
        "INSERT INTO memories (id, content, scope, tags, hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![id, content, scope, tags, h2, now, now],
    ) {
        return (json!({"error": e.to_string()}), 500);
    }
    let warning = if let Some(v) = &vector {
        let vb = crate::vector_store::f32_slice_to_bytes(v);
        let _ = c.execute(
            "INSERT OR REPLACE INTO mem_vectors (id, vec, updated_at) VALUES (?, ?, ?)",
            rusqlite::params![id, vb, crate::ops::now_f64()],
        );
        crate::VS_NOTIFY.store(true, std::sync::atomic::Ordering::Relaxed);
        None
    } else {
        Some("embedding-failed: 记忆已存但未向量化（embed 服务问题），语义搜索暂不可用")
    };
    drop(c);
    let mut result = json!({"ok": true, "status": "added", "id": id, "created_at": now});
    if let Some(w) = warning {
        result["warning"] = json!(w);
    }
    (result, 201)
}

// ---------- LLM 判重 (OpenAI 兼容 chat/completions, 默认 work-pc sglang) ----------

async fn llm_chat(app: &crate::App, prompt: &str, max_tokens: u32) -> String {
    let key = &app.cfg.llm_api_key;
    let client = reqwest::Client::new();
    let body = json!({
        "model": app.cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    });
    let req = client
        .post(&app.cfg.llm_url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(45));
    let resp = if key.is_empty() {
        req.send().await
    } else {
        req.bearer_auth(key).send().await
    };
    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[writeMemory] llm req err: {e}");
            return String::new();
        }
    };
    if !resp.status().is_success() {
        eprintln!("[writeMemory] llm status {}", resp.status());
        return String::new();
    }
    let v: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[writeMemory] llm json err: {e}");
            return String::new();
        }
    };
    v.pointer("/choices/0/message/content")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

/// SAME / MERGE / DIFF (CF: classifyDuplicate)
async fn classify_duplicate(app: &crate::App, a: &str, b: &str, score: f32) -> &'static str {
    let prompt = format!(
        "你是记忆库去重器。判断两条记忆是否表达同一事实。\n记忆A: {a}\n\n记忆B: {b}\n\n(向量相似度 {s:.3})\n只输出一个词, 不要解释:\n- SAME = 语义完全相同, 表达的是同一件事\n- MERGE = 相似/相关, 同一主题的不同方面, 适合合并成一条\n- DIFF = 不同记忆\n输出:",
        s = score
    );
    if app.cfg.llm_url.is_empty() {
        return "diff"; // LLM 未配置 → 降级正常写入 (CF: 查重失败降级)
    }
    let out = llm_chat(app, &prompt, 50).await;
    let t = out.trim().to_uppercase();
    if t.contains("SAME") {
        "same"
    } else if t.contains("MERGE") {
        "merge"
    } else {
        "diff"
    }
}

async fn merge_memories(app: &crate::App, a: &str, b: &str) -> String {
    let prompt = format!(
        "合并下面两条相关记忆为一条精炼完整的记忆, 保留所有关键信息, 去除重复部分, 保持原有风格(如开头的[项目名]标记)。\n记忆A: {a}\n\n记忆B: {b}\n\n只输出合并后的内容, 不要任何解释。"
    );
    let out = llm_chat(app, &prompt, 3000).await;
    out.trim().to_string().if_empty(a)
}

trait IfEmpty: Sized {
    fn if_empty(self, d: &str) -> Self;
}
impl IfEmpty for String {
    fn if_empty(self, d: &str) -> Self {
        if self.is_empty() {
            d.to_string()
        } else {
            self
        }
    }
}

// ---------- 标签解析 (CF: parseTags: 逗号/空格分隔 → JSON 数组) ----------

fn parse_tags(v: &Value) -> String {
    let parts: Vec<String> = if let Some(s) = v.as_str() {
        s.split([',', ' ', '\t', '\n'])
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect()
    } else if let Some(arr) = v.as_array() {
        arr.iter()
            .filter_map(|x| x.as_str())
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect()
    } else {
        Vec::new()
    };
    serde_json::to_string(&parts).unwrap_or_else(|_| "[]".to_string())
}

// ---------- handlers ----------

async fn mem_post(State(app): State<crate::App>, req: Request<Body>) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 256 * 1024)
        .await
    {
        Ok(b) => match serde_json::from_slice::<Value>(&b) {
            Ok(v) => v,
            Err(_) => return bad_request("JSON body required"),
        },
        Err(_) => return bad_request("JSON body required"),
    };
    let content = body
        .get("content")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim();
    if content.is_empty() {
        return bad_request("content is required");
    }
    let scope = body
        .get("scope")
        .and_then(|x| x.as_str())
        .unwrap_or("general");
    let tags = parse_tags(body.get("tags").unwrap_or(&Value::Null));
    let force = body.get("force").and_then(|x| x.as_bool()).unwrap_or(false);
    let (v, code) = write_memory(&app, content, scope, &tags, force).await;
    if code == 201 {
        ok201(&v)
    } else if code == 500 {
        db_err(v.get("error").and_then(|x| x.as_str()).unwrap_or("db error"))
    } else {
        ok200(&v)
    }
}

async fn mem_batch_post(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 1024 * 1024)
        .await
    {
        Ok(b) => match serde_json::from_slice::<Value>(&b) {
            Ok(v) => v,
            Err(_) => return bad_request("JSON body required"),
        },
        Err(_) => return bad_request("JSON body required"),
    };
    let items = match body.get("items").and_then(|x| x.as_array()) {
        Some(a) if !a.is_empty() => a,
        _ => return bad_request("items array required"),
    };
    let force = body.get("force").and_then(|x| x.as_bool()).unwrap_or(false);
    let mut results: Vec<Value> = Vec::new();
    for it in items.iter().take(100) {
        let content = it
            .get("content")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .trim();
        if content.is_empty() {
            results.push(json!({"index": results.len(), "status": "error", "message": "content 为空"}));
            continue;
        }
        let scope = it
            .get("scope")
            .and_then(|x| x.as_str())
            .unwrap_or("general");
        let tags = parse_tags(it.get("tags").unwrap_or(&Value::Null));
        let (mut r, _) = write_memory(&app, content, scope, &tags, force).await;
        if let Some(o) = r.as_object_mut() {
            o.remove("ok");
            o.insert("index".into(), json!(results.len()));
        }
        results.push(r);
    }
    let added = results
        .iter()
        .filter(|r| r.get("status") == Some(&json!("added")))
        .count();
    let errors = results
        .iter()
        .filter(|r| r.get("status") == Some(&json!("error")))
        .count();
    let deduped = results.len() - added - errors;
    ok201(&json!({"ok": true, "total": results.len(), "added": added,
                 "deduped": deduped, "results": results}))
}

async fn mem_get_one(
    State(app): State<crate::App>,
    axum::extract::Path(id): axum::extract::Path<String>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    match c.query_row(
        "SELECT id, content, scope, tags, created_at, updated_at FROM memories WHERE id = ?",
        rusqlite::params![id],
        |r| {
            Ok(json!({"id": r.get::<_, String>(0)?, "content": r.get::<_, String>(1)?,
                      "scope": r.get::<_, String>(2)?, "tags": r.get::<_, String>(3)?,
                      "created_at": r.get::<_, String>(4)?, "updated_at": r.get::<_, String>(5)?}))
        },
    ) {
        Ok(v) => {
            let mut m = v;
            if let Some(tags) = m.get("tags").and_then(|t| t.as_str()) {
                m["tags"] = serde_json::from_str(tags).unwrap_or(json!([]));
            }
            ok200(&m)
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => json_resp(
            404,
            &json!({"error": "not found", "id": id}),
        ),
        Err(e) => db_err(&e.to_string()),
    }
}

async fn mem_put(State(app): State<crate::App>, req: Request<Body>) -> axum::response::Response {
    // :id 走 query 参数 (与 CF path 解析等价, 看板传 /api/memories/<id>)
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 256 * 1024)
        .await
    {
        Ok(b) => match serde_json::from_slice::<Value>(&b) {
            Ok(v) => v,
            Err(_) => return bad_request("JSON body required"),
        },
        Err(_) => return bad_request("JSON body required"),
    };
    let id = body
        .get("id")
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .unwrap_or_default();
    if id.is_empty() {
        return bad_request("id required");
    }
    let now = now_iso();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    // CF: content 变更 → 重算 hash; scope/tags 可改
    let existing = c.query_row(
        "SELECT content FROM memories WHERE id = ?",
        rusqlite::params![id],
        |r| r.get::<_, String>(0),
    );
    match existing {
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return json_resp(404, &json!({"error": "not found", "id": id}))
        }
        Err(e) => return db_err(&e.to_string()),
        Ok(_) => {}
    }
    let mut fields: Vec<&str> = Vec::new();
    let mut vals: Vec<String> = Vec::new();
    if let Some(content) = body.get("content").and_then(|x| x.as_str()) {
        let h = hash16(content);
        fields.push("content");
        vals.push(content.to_string());
        fields.push("hash");
        vals.push(h);
    }
    if let Some(scope) = body.get("scope").and_then(|x| x.as_str()) {
        fields.push("scope");
        vals.push(scope.to_string());
    }
    if body.get("tags").is_some() {
        let tags = parse_tags(body.get("tags").unwrap_or(&Value::Null));
        fields.push("tags");
        vals.push(tags);
    }
    fields.push("updated_at");
    vals.push(now);
    if fields.is_empty() {
        return bad_request("no fields to update");
    }
    let set_sql = fields
        .iter()
        .map(|f| format!("{f} = ?"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut params: Vec<String> = vals;
    params.push(id.clone());
    match c.execute(
        &format!("UPDATE memories SET {set_sql} WHERE id = ?"),
        rusqlite::params_from_iter(params),
    ) {
        Ok(n) => {
            if n == 0 {
                return json_resp(404, &json!({"error": "not found", "id": id}));
            }
            ok200(&json!({"ok": true, "id": id}))
        }
        Err(e) => db_err(&e.to_string()),
    }
}

async fn mem_delete(
    State(app): State<crate::App>,
    axum::extract::Path(id): axum::extract::Path<String>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    match c.execute(
        "DELETE FROM memories WHERE id = ?",
        rusqlite::params![id],
    ) {
        Ok(n) => {
            let _ = c.execute(
                "DELETE FROM mem_vectors WHERE id = ?",
                rusqlite::params![id],
            );
            if n == 0 {
                return json_resp(404, &json!({"error": "not found", "id": id}));
            }
            crate::VS_NOTIFY.store(true, std::sync::atomic::Ordering::Relaxed);
            ok200(&json!({"ok": true, "id": id}))
        }
        Err(e) => db_err(&e.to_string()),
    }
}

async fn mem_batch_delete(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 1024 * 1024)
        .await
    {
        Ok(b) => match serde_json::from_slice::<Value>(&b) {
            Ok(v) => v,
            Err(_) => return bad_request("JSON body required"),
        },
        Err(_) => return bad_request("JSON body required"),
    };
    let ids: Vec<String> = body
        .get("ids")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if ids.is_empty() {
        return bad_request("ids array required");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let mut deleted = 0;
    for id in &ids {
        let n = c
            .execute("DELETE FROM memories WHERE id = ?", rusqlite::params![id])
            .unwrap_or(0);
        deleted += n;
        let _ = c.execute(
            "DELETE FROM mem_vectors WHERE id = ?",
            rusqlite::params![id],
        );
    }
    crate::VS_NOTIFY.store(true, std::sync::atomic::Ordering::Relaxed);
    ok200(&json!({"ok": true, "deleted": deleted}))
}

// 向量语义检索 (CF: /api/memories/search → Vectorize query)
async fn mem_search(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let query = qval(&q, "q").trim().to_string();
    let limit = crate::qint(&q, "limit", 10);
    if query.is_empty() {
        return bad_request("q required");
    }
    let qv = match crate::embed::embed_one(&app.cfg, &query) {
        Ok(v) => v,
        Err(e) => return db_err(&format!("embedding failed: {e}")),
    };
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let cands = VS.search(&qv, limit as usize);
    let mut items = Vec::new();
    for (mid, score) in cands {
        if let Ok((content, scope, tags_s, updated)) = c.query_row(
            "SELECT content, scope, tags, updated_at FROM memories WHERE id = ?",
            rusqlite::params![mid],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            },
        ) {
            items.push(json!({
                "id": mid, "score": (score * 10000.0).round() / 10000.0,
                "content": content, "scope": scope,
                "tags": serde_json::from_str::<Value>(&tags_s).unwrap_or(json!([])),
                "updated_at": updated,
            }));
        }
        if items.len() >= limit as usize {
            break;
        }
    }
    let _ = StatusCode::OK; // keep import alive
    ok200(&json!({"ok": true, "query": query, "total": items.len(), "items": items}))
}
