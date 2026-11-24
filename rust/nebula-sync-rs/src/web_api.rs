//! 静态资产域 (阶段四: 看板 SPA 独立部署, 彻底脱离 CF 域名)。
//! Rust 直接供 public/ 前端: 非 /api/ 路径 → 静态文件; 404 → SPA fallback index.html。
//! 前端全相对路径 (location.host), 登录走 /api/login (cookie 会话, 浏览器免 Bearer)。
//! 目录: NEBULA_WEB_ROOT (默认 {data_dir}/web); 不存在时非 API 路径回 404 JSON。
use std::path::Path;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::response::Response;
use serde_json::json;

use crate::api_common::get_db;

use serde_json::Value;

/// 扩展名 → Content-Type (看板只有这几类)
fn mime_of(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        "wasm" => "application/wasm",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn read_file_resp(p: &Path) -> Response {
    let bytes = match std::fs::read(p) {
        Ok(b) => b,
        Err(_) => {
            let mut r = Response::new(Body::from("not found".as_bytes().to_vec()));
            *r.status_mut() = StatusCode::NOT_FOUND;
            return r;
        }
    };
    let mime = mime_of(p.to_str().unwrap_or(""));
    let mut r = Response::new(Body::from(bytes));
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, mime.parse().unwrap());
    h.insert(header::CACHE_CONTROL, "no-cache".parse().unwrap());
    r
}

/// 非 API 路径: 静态文件 + SPA fallback (CF 同款语义)
pub async fn serve(State(app): State<crate::App>, req: Request<Body>) -> Response {
    let path = req.uri().path().to_string();

    if path.contains("..") {
        let mut r = Response::new(Body::from("bad path".as_bytes().to_vec()));
        *r.status_mut() = StatusCode::BAD_REQUEST;
        return r;
    }
    let rel = if path.is_empty() || path == "/" {
        "/index.html".to_string()
    } else {
        path
    };

    let base = std::path::PathBuf::from(&app.cfg.web_root);
    let full = base.join(rel.trim_start_matches('/'));

    if full.is_file() {
        return read_file_resp(&full);
    }
    // SPA fallback
    let idx = base.join("index.html");
    if !idx.is_file() {
        let mut r = Response::new(Body::from(
            json!({"error": "web assets not found (set NEBULA_WEB_ROOT)"}).to_string(),
        ));
        *r.status_mut() = StatusCode::NOT_FOUND;
        return r;
    }
    read_file_resp(&idx)
}


/// GET /api/memories: 列表(q 空) + 关键词 LIKE 搜索(q 有) — CF listMemories/searchMemories 1:1。
/// 契约: {total, offset, query?, items:[{id,content,scope,tags:[...],updated_at,created_at}]}
pub async fn memories_list(State(app): State<crate::App>, req: Request<Body>) -> Response {
    use crate::{auth_ok, qint, qparams, qval};
    if !auth_ok(&app, &req) {
        return crate::json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let query = qval(&q, "q").trim().to_string();
    let limit = (qint(&q, "limit", 20)).min(100).max(1);
    let offset = qint(&q, "offset", 0).max(0);
    let scope = qval(&q, "scope").trim().to_string();
    let like = format!("%{}%", query);
    let scoped = !scope.is_empty() && scope != "all";

    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };

    // 行 → json (tags csv → 数组)
    let map_row = |r: &rusqlite::Row| -> rusqlite::Result<Value> {
        let tags_s = r.get::<_, String>(3)?;
        let tags: Vec<String> = tags_s
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        Ok(json!({
            "id": r.get::<_, String>(0)?,
            "content": r.get::<_, String>(1)?,
            "scope": r.get::<_, String>(2)?,
            "tags": tags,
            "updated_at": r.get::<_, String>(4)?,
            "created_at": r.get::<_, String>(5)?,
        }))
    };

    // 用 params! 宏(项目 main.rs 已验证模式); 每分支内联 prepare+query
    let (items, total): (Vec<Value>, i64) = if scoped && query.is_empty() {
        let mut st = c.prepare(
            "SELECT id,content,scope,tags,updated_at,created_at FROM memories WHERE scope=? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        ).map_err(|e| e.to_string()).unwrap();
        let items = st.query_map(rusqlite::params![scope, limit, offset], |r| map_row(r))
            .map(|rows| rows.flatten().collect()).unwrap_or_else(|_| Vec::new());
        let total = c.query_row("SELECT COUNT(*) FROM memories WHERE scope=?",
            rusqlite::params![scope], |r| r.get(0)).unwrap_or(items.len() as i64);
        (items, total)
    } else if scoped {
        let mut st = c.prepare(
            "SELECT id,content,scope,tags,updated_at,created_at FROM memories WHERE scope=? AND (content LIKE ? OR tags LIKE ?) ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        ).map_err(|e| e.to_string()).unwrap();
        let items = st.query_map(rusqlite::params![scope, like, like, limit, offset], |r| map_row(r))
            .map(|rows| rows.flatten().collect()).unwrap_or_else(|_| Vec::new());
        let total = c.query_row("SELECT COUNT(*) FROM memories WHERE scope=? AND (content LIKE ? OR tags LIKE ?)",
            rusqlite::params![scope, like, like], |r| r.get(0)).unwrap_or(items.len() as i64);
        (items, total)
    } else if query.is_empty() {
        let mut st = c.prepare(
            "SELECT id,content,scope,tags,updated_at,created_at FROM memories ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        ).map_err(|e| e.to_string()).unwrap();
        let items = st.query_map(rusqlite::params![limit, offset], |r| map_row(r))
            .map(|rows| rows.flatten().collect()).unwrap_or_else(|_| Vec::new());
        let total = c.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0)).unwrap_or(items.len() as i64);
        (items, total)
    } else {
        let mut st = c.prepare(
            "SELECT id,content,scope,tags,updated_at,created_at FROM memories WHERE content LIKE ? OR tags LIKE ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        ).map_err(|e| e.to_string()).unwrap();
        let items = st.query_map(rusqlite::params![like, like, limit, offset], |r| map_row(r))
            .map(|rows| rows.flatten().collect()).unwrap_or_else(|_| Vec::new());
        let total = c.query_row("SELECT COUNT(*) FROM memories WHERE content LIKE ? OR tags LIKE ?",
            rusqlite::params![like, like], |r| r.get(0)).unwrap_or(items.len() as i64);
        (items, total)
    };

    let mut out = json!({"total": total, "offset": offset, "items": items});
    if !query.is_empty() {
        if let Some(o) = out.as_object_mut() {
            o.insert("query".into(), json!(query));
        }
    }
    crate::json_resp(200, &out)
}

/// /api/stats: 记忆统计 (看板 mem 视图 scope 下拉; CF: total + byScope)
pub async fn stats(State(app): State<crate::App>) -> Response {
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let total: i64 = c
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);
    let by_scope: Vec<Value> = match c
        .prepare("SELECT scope, COUNT(*) AS c FROM memories GROUP BY scope ORDER BY c DESC")
    {
        Ok(mut st) => {
            let mut out = Vec::new();
            if let Ok(rows) = st.query_map([], |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?.unwrap_or_else(|| "general".into()),
                    r.get::<_, i64>(1)?,
                ))
            }) {
                for row in rows.flatten() {
                    out.push(json!({"scope": row.0, "c": row.1}));
                }
            }
            out
        }
        Err(_) => Vec::new(),
    };
    crate::json_resp(200, &json!({"total": total, "byScope": by_scope}))
}

pub fn router() -> axum::Router<crate::App> {
    axum::Router::new()
        .route("/api/stats", axum::routing::get(stats))
        .route("/api/memories", axum::routing::get(memories_list))
}
