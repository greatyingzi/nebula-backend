//! 知识图谱域 (阶段二): 实体 upsert / 边批量增删查。
//! 1:1 复刻 CF cloud-memory worker 的 /api/graph/entities + /api/graph/edges。
//! 读端点已存在于核心 (/api/nebula/entities|graph), 这里只补写/删 + CF 形状。
use axum::extract::State;
use axum::body::Bytes;
use axum::http::Request;
use axum::body::Body;
use axum::routing::post;
use axum::Router;
use serde_json::{json, Value};

use crate::api_common::{bad_request, db_err, get_db, ok200, ok201, now_iso};
use crate::{auth_ok, json_resp, qparams, qval};

/// 允许的边关系 (CF cloud-memory 白名单)
const VALID_RELS: [&str; 8] = [
    "caused_by", "fixed_by", "supersedes", "depends_on",
    "pitfall_of", "same_root", "part_of", "related_to",
];

pub fn router() -> Router<crate::App> {
    Router::new()
        .route("/api/graph/entities", post(entities_post))
        .route(
            "/api/graph/edges",
            post(edges_post).get(edges_get).delete(edges_delete),
        )
}

async fn unauthorized() -> axum::response::Response {
    json_resp(401, &json!({"error": "unauthorized"}))
}

// ---------- POST /api/graph/entities ----------

async fn entities_post(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return unauthorized().await;
    }
    let body: Bytes = match axum::body::to_bytes(req.into_body(), 1 << 22).await {
        Ok(b) => b,
        Err(e) => return bad_request(&e.to_string()),
    };
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return bad_request("bad json"),
    };
    let name = body.get("name").and_then(|x| x.as_str()).unwrap_or("").trim();
    if name.is_empty() {
        return bad_request("name required");
    }
    let kind = body.get("kind").and_then(|x| x.as_str()).unwrap_or("component").trim();
    let description = body.get("description").and_then(|x| x.as_str()).unwrap_or("").trim();
    let parent_id = body.get("parent_id").and_then(|x| x.as_str()).unwrap_or("").trim();
    let status = match body.get("status").and_then(|x| x.as_str()) {
        Some(s) if ["active", "provisional", "archived"].contains(&s) => s,
        _ => "active",
    };
    let aliases = body
        .get("aliases")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let aliases_json = serde_json::to_string(&aliases).unwrap_or_else(|_| "[]".into());
    // id: 显式优先, 否则 ent:<slug> (CF: 小写 + 中文/数字/字母, 其余转 '-')
    let id = match body.get("id").and_then(|x| x.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => slug_entity(name),
    };
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    if let Err(e) = c.execute(
        "INSERT INTO entities (id, name, kind, description, parent_id, status, aliases) VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           name = excluded.name, kind = excluded.kind,
           description = CASE WHEN excluded.description != '' THEN excluded.description ELSE entities.description END,
           parent_id = excluded.parent_id, status = excluded.status, aliases = excluded.aliases",
        (id.as_str(), name, kind, description, parent_id, status, aliases_json.as_str()),
    ) {
        return db_err(&e.to_string());
    }
    let _ = now_iso();
    ok201(&json!({"ok": true, "id": id}))
}

fn slug_entity(name: &str) -> String {
    let mut out = String::new();
    for ch in name.to_lowercase().chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() || ('\u{4e00}'..='\u{9fa5}').contains(&ch) {
            out.push(ch);
        } else {
            out.push('-');
        }
    }
    let t = out.trim_matches('-');
    let core = if t.is_empty() { "ent".into() } else { t.chars().take(40).collect::<String>() };
    format!("ent:{core}")
}

// ---------- POST /api/graph/edges ----------

async fn edges_post(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return unauthorized().await;
    }
    let body: Bytes = match axum::body::to_bytes(req.into_body(), 1 << 22).await {
        Ok(b) => b,
        Err(e) => return crate::api_common::bad_request(&e.to_string()),
    };
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return bad_request("bad json"),
    };
    let items: Vec<Value> = match body.get("edges").and_then(|x| x.as_array()) {
        Some(a) => a.to_vec(),
        None => {
            // 单条: 直接 {from, rel, to, note}
            if body.get("from").is_some() || body.get("rel").is_some() || body.get("to").is_some() {
                vec![body]
            } else {
                return bad_request("edges required");
            }
        }
    };
    let now = now_iso();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let (mut added, mut skipped) = (0i64, 0i64);
    for e in items.iter().take(500) {
        let from = e.get("from").and_then(|x| x.as_str()).unwrap_or("").trim();
        let rel = e.get("rel").and_then(|x| x.as_str()).unwrap_or("").trim();
        let to = e.get("to").and_then(|x| x.as_str()).unwrap_or("").trim();
        let note = e.get("note").and_then(|x| x.as_str()).unwrap_or("");
        if from.is_empty() || to.is_empty() || !VALID_RELS.contains(&rel) {
            skipped += 1;
            continue;
        }
        match c.execute(
            "INSERT INTO edges (from_id, rel, to_id, note, created_at) VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(from_id, rel, to_id) DO UPDATE SET note = excluded.note",
            (from, rel, to, note, now.as_str()),
        ) {
            Ok(_) => added += 1,
            Err(_) => skipped += 1,
        }
    }
    ok201(&json!({"ok": true, "added": added, "skipped": skipped}))
}

// ---------- GET /api/graph/edges?node= ----------

async fn edges_get(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return unauthorized().await;
    }
    let q = qparams(&req);
    let node = qval(&q, "node").trim().to_string();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let (sql, params) = if node.is_empty() {
        (
            "SELECT from_id, rel, to_id, note, created_at FROM edges ORDER BY created_at DESC",
            Vec::<String>::new(),
        )
    } else {
        (
            "SELECT from_id, rel, to_id, note, created_at FROM edges WHERE from_id = ? OR to_id = ? ORDER BY created_at DESC",
            vec![node.clone(), node],
        )
    };
    let mut st = match c.prepare(sql) {
        Ok(s) => s,
        Err(e) => return db_err(&e.to_string()),
    };
    let rows = match st.query_map(rusqlite::params_from_iter(params.iter()), |r| {
        Ok(json!({
            "from": r.get::<_, String>(0)?,
            "rel": r.get::<_, String>(1)?,
            "to": r.get::<_, String>(2)?,
            "note": r.get::<_, String>(3)?,
            "created_at": r.get::<_, String>(4)?,
        }))
    }) {
        Ok(rows) => rows.filter_map(|x| x.ok()).collect::<Vec<_>>(),
        Err(e) => return db_err(&e.to_string()),
    };
    ok200(&json!({"ok": true, "items": rows}))
}

// ---------- DELETE /api/graph/edges ----------

async fn edges_delete(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return unauthorized().await;
    }
    let body: Bytes = match axum::body::to_bytes(req.into_body(), 1 << 22).await {
        Ok(b) => b,
        Err(e) => return crate::api_common::bad_request(&e.to_string()),
    };
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return bad_request("bad json"),
    };
    let items: Vec<Value> = body.get("edges").and_then(|x| x.as_array()).cloned().unwrap_or_default();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let mut deleted = 0i64;
    for e in items.iter().take(500) {
        let from = e.get("from").and_then(|x| x.as_str()).unwrap_or("");
        let rel = e.get("rel").and_then(|x| x.as_str()).unwrap_or("");
        let to = e.get("to").and_then(|x| x.as_str()).unwrap_or("");
        if let Ok(n) = c.execute("DELETE FROM edges WHERE from_id = ? AND rel = ? AND to_id = ?", (from, rel, to)) {
            deleted += n as i64;
        }
    }
    ok200(&json!({"ok": true, "deleted": deleted}))
}
