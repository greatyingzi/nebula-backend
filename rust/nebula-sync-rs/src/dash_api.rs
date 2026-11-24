//! 看板交互域 (阶段三): tasks/links/settings/login/logout/dashboard/export
//! + nebula 写端点 (merge/entity status/reparent/skills merge) + maintenance consolidate。
//! 1:1 复刻 CF linghub worker; 看板静态页 (public/) 后续由本进程直接服务,
//! nebula.example.invalid 域名指过来即完成迁移。
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::http::Request;
use axum::response::Response;
use axum::routing::{get, post, put, delete};
use axum::Router;
use serde_json::{json, Value};

use crate::api_common::{bad_request, get_db, now_iso, ok200, ok201, db_err, gen_id};
use crate::{auth_ok, json_resp, qparams, qval};

// ---------- 登录会话 (CF: signSession = base64(exp) + '.' + base64(hmac-sha256)) ----------

fn b64_no_pad(s: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD_NO_PAD.encode(s)
}

fn b64_decode_lenient(s: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    let mut v = s.to_string();
    while v.len() % 4 != 0 {
        v.push('=');
    }
    base64::engine::general_purpose::STANDARD.decode(v).ok()
}

fn hmac_sha256_b64(data: &str, secret: &str) -> String {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    type H = Hmac<Sha256>;
    let mut mac = H::new_from_slice(secret.as_bytes()).expect("hmac key any len");
    mac.update(data.as_bytes());
    b64_no_pad(&mac.finalize().into_bytes())
}

/// 7 天会话 (CF: exp = now + 7d; cookie lh_session)
fn sign_session(secret: &str) -> String {
    let exp = crate::api_common::now_ms() + 7 * 24 * 3600 * 1000;
    let payload = b64_no_pad(format!("{{\"exp\":{exp}}}").as_bytes());
    let sig = hmac_sha256_b64(&payload, secret);
    format!("{payload}.{sig}")
}

fn verify_session(token: &str, secret: &str) -> bool {
    let Some((payload, sig)) = token.split_once('.') else {
        return false;
    };
    let expect = hmac_sha256_b64(payload, secret);
    if sig != expect {
        return false;
    }
    let Some(raw) = b64_decode_lenient(payload) else {
        return false;
    };
    let Ok(v) = serde_json::from_slice::<Value>(&raw) else {
        return false;
    };
    v.get("exp").and_then(|x| x.as_i64()).unwrap_or(0) > crate::api_common::now_ms()
}

fn parse_cookies(hdr: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for part in hdr.split(';') {
        if let Some((k, v)) = part.trim().split_once('=') {
            m.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    m
}

/// CF checkAuth 三通道: Bearer token (机器) / ?token= (WS) / cookie (人)。
/// Rust 核心 auth_ok 只查 Bearer; 看板浏览器只带 cookie → 这里放宽。
fn auth_ok_ext(app: &crate::App, req: &Request<Body>) -> bool {
    if auth_ok(app, req) {
        return true;
    }
    let q = qparams(req);
    let qtok = qval(&q, "token");
    if let Some(tok) = app.token.read().unwrap().as_ref() {
        if !qtok.is_empty() && qtok == *tok {
            return true;
        }
    }
    let secret = if app.cfg.session_secret.is_empty() {
        app.cfg.admin_password.clone()
    } else {
        app.cfg.session_secret.clone()
    };
    if !secret.is_empty() {
        if let Some(cookie) = req.headers().get(header::COOKIE) {
            let cookies = parse_cookies(cookie.to_str().unwrap_or(""));
            if let Some(s) = cookies.get("lh_session") {
                if verify_session(s, &secret) {
                    return true;
                }
            }
        }
    }
    if app.cfg.admin_password.is_empty() {
        // 双通道都未配置 → 开放模式 (CF 同逻辑)
        return true;
    }
    false
}

// ---------- dashboard 60s 缓存 (CF: _statsCache) ----------

static DASH_CACHE: Lazy<Mutex<Option<(i64, Value)>>> = Lazy::new(|| Mutex::new(None));
use once_cell::sync::Lazy;

pub fn router() -> Router<crate::App> {
    Router::new()
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .route("/api/auth-status", get(auth_status))
        .route("/api/dashboard", get(dashboard))
        .route("/api/export", get(export))
        .route("/api/tasks", get(tasks_get).post(tasks_post))
        .route("/api/tasks/{id}", put(tasks_put).delete(tasks_delete))
        .route("/api/links", get(links_get).post(links_post))
        .route("/api/links/fetch-title", post(links_fetch_title))
        .route("/api/links/{id}", put(links_put).delete(links_delete))
        .route("/api/settings/dashboard", get(settings_get).put(settings_put))
        .route("/api/nebula/merge", post(nebula_merge))
        .route("/api/nebula/entity/status", post(entity_status))
        .route("/api/nebula/entity/reparent", post(entity_reparent))
        .route("/api/nebula/skills/merge", post(skills_merge))
        .route("/api/maintenance/consolidate", post(consolidate))
}

// ---------- 登录 ----------

async fn login(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if app.cfg.admin_password.is_empty() {
        return json_resp(500, &json!({"error": "认证未配置（缺少 ADMIN_PASSWORD）"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let password = body.get("password").and_then(|x| x.as_str()).unwrap_or("");
    if password != app.cfg.admin_password {
        return json_resp(401, &json!({"error": "密码错误"}));
    }
    let secret = if app.cfg.session_secret.is_empty() {
        app.cfg.admin_password.clone()
    } else {
        app.cfg.session_secret.clone()
    };
    let session = sign_session(&secret);
    let mut r = json_resp(200, &json!({"ok": true}));
    r.headers_mut().insert(
        header::SET_COOKIE,
        header::HeaderValue::from_str(&format!("lh_session={session}; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800")).unwrap(),
    );
    r
}

async fn logout(State(_app): State<crate::App>, _req: Request<Body>) -> Response {
    let mut r = json_resp(200, &json!({"ok": true}));
    r.headers_mut().insert(
        header::SET_COOKIE,
        header::HeaderValue::from_static("lh_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"),
    );
    r
}

async fn auth_status(State(app): State<crate::App>, req: Request<Body>) -> Response {
    let authenticated = auth_ok_ext(&app, &req);
    json_resp(
        200,
        &json!({
            "authenticated": authenticated,
            "cookieAuth": !app.cfg.admin_password.is_empty(),
            "tokenAuth": app.token.read().unwrap().is_some(),
        }),
    )
}

// ---------- dashboard ----------

async fn dashboard(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let now = crate::api_common::now_ms();
    if let Some((ts, data)) = DASH_CACHE.lock().unwrap().clone() {
        if now - ts < 60_000 {
            return ok200(&data);
        }
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let count = |sql: &str, p: Option<&str>| -> u64 {
        match p {
            Some(x) => c
                .query_row(sql, rusqlite::params![x], |r| r.get::<_, i64>(0))
                .map(|v| v as u64)
                .unwrap_or(0),
            None => c
                .query_row(sql, [], |r| r.get::<_, i64>(0))
                .map(|v| v as u64)
                .unwrap_or(0),
        }
    };
    let mem_count = count("SELECT COUNT(*) FROM memories", None);
    let task_total = count("SELECT COUNT(*) FROM tasks", None);
    let task_todo = count("SELECT COUNT(*) FROM tasks WHERE status = 'todo'", None);
    let task_done = count("SELECT COUNT(*) FROM tasks WHERE status = 'done'", None);
    let link_count = count("SELECT COUNT(*) FROM links", None);
    let by_scope: Vec<Value> = c
        .prepare("SELECT scope, COUNT(*) as c FROM memories GROUP BY scope ORDER BY c DESC LIMIT 8")
        .and_then(|mut st| {
            st.query_map([], |r| {
                Ok(json!({"scope": r.get::<_, String>(0)?, "c": r.get::<_, i64>(1)?}))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect::<Vec<_>>())
        })
        .unwrap_or_default();
    let recent_mem: Vec<Value> = c
        .prepare("SELECT id, content, scope, updated_at FROM memories ORDER BY updated_at DESC LIMIT 8")
        .and_then(|mut st| {
            st.query_map([], |r| {
                Ok(json!({"id": r.get::<_, String>(0)?, "content": r.get::<_, String>(1)?,
                          "scope": r.get::<_, String>(2)?, "updated_at": r.get::<_, String>(3)?}))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect::<Vec<_>>())
        })
        .unwrap_or_default();
    let recent_tasks: Vec<Value> = c
        .prepare("SELECT id, title, priority, status, updated_at FROM tasks ORDER BY updated_at DESC LIMIT 6")
        .and_then(|mut st| {
            st.query_map([], |r| {
                Ok(json!({"id": r.get::<_, String>(0)?, "title": r.get::<_, String>(1)?,
                          "priority": r.get::<_, String>(2)?, "status": r.get::<_, String>(3)?,
                          "updated_at": r.get::<_, String>(4)?}))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect::<Vec<_>>())
        })
        .unwrap_or_default();
    // 近 7 天趋势 (CF: days 7 桶 + sqlite datetime)
    let days: Vec<Value> = (0..7).map(|i| {
        let d = (now as u128 - i as u128 * 86400000) as i64;
        let key = crate::api_common::iso_date(d / 1000);
        let mo = key[5..7].to_string();
        let dd = key[8..10].to_string();
        json!({"date": key, "label": format!("{}/{:?}", mo, dd.parse::<u32>().unwrap_or(0)),
               "count": 0})
    }).collect();
    let trend_rows: Vec<Value> = c
        .prepare("SELECT substr(created_at, 1, 10) as day, COUNT(*) as c FROM memories WHERE created_at >= ?1 GROUP BY day")
        .and_then(|mut st| {
            let cutoff = crate::api_common::iso_date(crate::api_common::now_ms() as i64 / 1000 - 7 * 86400);
            st.query_map(rusqlite::params![cutoff], |r| {
                Ok(json!({"day": r.get::<_, String>(0)?, "c": r.get::<_, i64>(1)?}))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect::<Vec<_>>())
        })
        .unwrap_or_default();
    let mut days_mut = days;
    for tr in &trend_rows {
        let day = tr.get("day").and_then(|x| x.as_str()).unwrap_or("");
        let c_ = tr.get("c").and_then(|x| x.as_i64()).unwrap_or(0) as u64;
        if let Some(d) = days_mut.iter_mut().find(|d| d.get("date").and_then(|x| x.as_str()) == Some(day)) {
            d["count"] = json!(c_);
        }
    }
    let data = json!({
        "memories": mem_count,
        "tasks": {"total": task_total, "todo": task_todo, "done": task_done},
        "links": link_count,
        "trend": days_mut,
        "topScopes": by_scope,
        "recentMemories": recent_mem,
        "recentTasks": recent_tasks,
    });
    *DASH_CACHE.lock().unwrap() = Some((now, data.clone()));
    ok200(&data)
}

// ---------- export ----------

async fn export(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let scope = qval(&q, "scope");
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let sql = if scope.is_empty() || scope == "all" {
        "SELECT id, content, scope, tags, created_at, updated_at FROM memories ORDER BY created_at ASC"
    } else {
        "SELECT id, content, scope, tags, created_at, updated_at FROM memories WHERE scope = ? ORDER BY created_at ASC"
    };
    let items: Vec<Value> = if scope.is_empty() || scope == "all" {
        c.prepare(sql)
            .and_then(|mut st| {
                st.query_map([], |r| row_mem(r)).map(|it| it.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })
            .unwrap_or_default()
    } else {
        c.prepare(sql)
            .and_then(|mut st| {
                st.query_map(rusqlite::params![scope], |r| row_mem(r)).map(|it| it.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })
            .unwrap_or_default()
    };
    let data = json!({"exported_at": now_iso(), "total": items.len(), "items": items});
    let mut r = json_resp(200, &data);
    r.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        header::HeaderValue::from_static("attachment; filename=\"cloud-memory-export.json\""),
    );
    r
}

fn row_mem(r: &rusqlite::Row) -> rusqlite::Result<Value> {
    let tags = r.get::<_, String>(3)?;
    Ok(json!({"id": r.get::<_, String>(0)?, "content": r.get::<_, String>(1)?,
              "scope": r.get::<_, String>(2)?, "tags": serde_json::from_str::<Value>(&tags).unwrap_or(json!([])),
              "created_at": r.get::<_, String>(4)?, "updated_at": r.get::<_, String>(5)?}))
}

// ---------- tasks ----------

async fn tasks_get(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let status = qval(&q, "status");
    let limit = crate::qint(&q, "limit", 50).min(100) as i64;
    let offset = crate::qint(&q, "offset", 0).max(0) as i64;
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let (sql, csql, p) = if status.is_empty() || status == "all" {
        ("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
         "SELECT COUNT(*) FROM tasks", false)
    } else {
        ("SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
         "SELECT COUNT(*) FROM tasks WHERE status = ?", true)
    };
    let count = if p {
        c.query_row(csql, rusqlite::params![status], |r| r.get::<_, i64>(0))
            .map(|v| v as u64).unwrap_or(0)
    } else {
        c.query_row(csql, [], |r| r.get::<_, i64>(0))
            .map(|v| v as u64).unwrap_or(0)
    };
    let items: Vec<Value> = c.prepare(sql)
        .and_then(|mut st| {
            let params = if p {
                rusqlite::params![status, limit, offset]
            } else {
                rusqlite::params![limit, offset]
            };
            st.query_map(params, |r| {
                let tags = r.get::<_, String>(6)?;
                Ok(json!({"id": r.get::<_, String>(0)?, "title": r.get::<_, String>(1)?,
                          "description": r.get::<_, String>(2)?, "priority": r.get::<_, String>(3)?,
                          "status": r.get::<_, String>(4)?, "due_date": r.get::<_, String>(5)?,
                          "tags": serde_json::from_str::<Value>(&tags).unwrap_or(json!([])),
                          "created_at": r.get::<_, String>(7)?, "updated_at": r.get::<_, String>(8)?}))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect::<Vec<_>>())
        })
        .unwrap_or_default();
    ok200(&json!({"total": count, "items": items}))
}

async fn tasks_post(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let title = body.get("title").and_then(|x| x.as_str()).unwrap_or("").trim();
    if title.is_empty() {
        return bad_request("title is required");
    }
    let now = now_iso();
    let id = gen_id();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let tags = parse_tags_json(body.get("tags"));
    if let Err(e) = c.execute(
        "INSERT INTO tasks (id, title, description, priority, status, due_date, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            id, title,
            body.get("description").and_then(|x| x.as_str()).unwrap_or(""),
            body.get("priority").and_then(|x| x.as_str()).unwrap_or("medium"),
            body.get("status").and_then(|x| x.as_str()).unwrap_or("todo"),
            body.get("due_date").and_then(|x| x.as_str()).unwrap_or(""),
            tags, now, now
        ],
    ) {
        return db_err(&e.to_string());
    }
    ok201(&json!({"ok": true, "id": id, "created_at": now}))
}

async fn tasks_put(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let id = qval(&q, "id"); // 看板 PUT /api/tasks/{id} — Rust 路由捕获在 Path, 这里兜底
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let id = if id.is_empty() {
        body.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string()
    } else {
        id
    };
    let _ = tasks_update(&app, &id, &body).await;
    json_resp(200, &json!({"ok": true, "id": id}))
}

async fn tasks_update(app: &crate::App, id: &str, body: &Value) -> Result<(), String> {
    let now = now_iso();
    let mut fields: Vec<String> = Vec::new();
    let mut vals: Vec<String> = Vec::new();
    for k in ["title", "description", "priority", "status", "due_date"] {
        if let Some(v) = body.get(k).and_then(|x| x.as_str()) {
            fields.push(format!("{k} = ?"));
            vals.push(v.to_string());
        }
    }
    if body.get("tags").is_some() {
        fields.push("tags = ?".into());
        vals.push(parse_tags_json(body.get("tags")));
    }
    fields.push("updated_at = ?".into());
    vals.push(now);
    if fields.is_empty() {
        return Err("no fields to update".into());
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(_) => return Err("db".into()),
    };
    let sql = format!("UPDATE tasks SET {} WHERE id = ?", fields.join(", "));
    let mut params: Vec<rusqlite::types::Value> = vals.iter().map(|s| rusqlite::types::Value::Text(s.clone())).collect();
    params.push(rusqlite::types::Value::Text(id.to_string()));
    c.execute(&sql, rusqlite::params_from_iter(params)).map_err(|e| e.to_string())?;
    Ok(())
}

async fn tasks_delete(State(app): State<crate::App>, axum::extract::Path(id): axum::extract::Path<String>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let n = c.execute("DELETE FROM tasks WHERE id = ?", rusqlite::params![id]).unwrap_or(0);
    if n == 0 {
        return json_resp(404, &json!({"error": "not found", "id": id}));
    }
    ok200(&json!({"ok": true, "id": id}))
}

// ---------- links ----------

async fn links_get(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let limit = crate::qint(&q, "limit", 50).min(100) as i64;
    let offset = crate::qint(&q, "offset", 0).max(0) as i64;
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let count = c.query_row("SELECT COUNT(*) FROM links", [], |r| r.get::<_, i64>(0)).map(|v| v as u64).unwrap_or(0);
    let items: Vec<Value> = c
        .prepare("SELECT * FROM links ORDER BY created_at DESC LIMIT ? OFFSET ?")
        .and_then(|mut st| {
            st.query_map(rusqlite::params![limit, offset], |r| {
                let tags = r.get::<_, String>(4)?;
                Ok(json!({"id": r.get::<_, String>(0)?, "url": r.get::<_, String>(1)?,
                          "title": r.get::<_, String>(2)?, "description": r.get::<_, String>(3)?,
                          "tags": serde_json::from_str::<Value>(&tags).unwrap_or(json!([])),
                          "created_at": r.get::<_, String>(5)?, "updated_at": r.get::<_, String>(6)?}))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect::<Vec<_>>())
        })
        .unwrap_or_default();
    ok200(&json!({"total": count, "items": items}))
}

async fn links_post(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let url = body.get("url").and_then(|x| x.as_str()).unwrap_or("").trim();
    if url.is_empty() {
        return bad_request("url is required");
    }
    let now = now_iso();
    let id = gen_id();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    if let Err(e) = c.execute(
        "INSERT INTO links (id, url, title, description, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            id, url,
            body.get("title").and_then(|x| x.as_str()).unwrap_or(""),
            body.get("description").and_then(|x| x.as_str()).unwrap_or(""),
            parse_tags_json(body.get("tags")), now, now
        ],
    ) {
        return db_err(&e.to_string());
    }
    ok201(&json!({"ok": true, "id": id, "created_at": now}))
}

async fn links_fetch_title(State(_app): State<crate::App>, req: Request<Body>) -> Response {
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let url = body.get("url").and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
    if url.is_empty() || !(url.starts_with("http://") || url.starts_with("https://")) {
        return bad_request("valid http(s) url required");
    }
    let title = tokio::task::spawn_blocking({
        let url = url.clone();
        move || {
            let client = reqwest::blocking::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; LingHub/1.0)")
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .ok()?;
            let resp = client.get(&url).send().ok()?;
            if !resp.status().is_success() {
                return None;
            }
            let html = resp.text().ok()?;
            let m = regex_title(&html)?;
            Some(m)
        }
    })
    .await
    .ok()
    .flatten()
    .unwrap_or_default();
    ok200(&json!({"ok": true, "url": url, "title": title}))
}

fn regex_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let gt = lower[start..].find('>')? + start;
    let end = lower[gt + 1..].find("</title")? + gt + 1;
    if end <= gt {
        return None;
    }
    Some(html[gt + 1..end].trim().chars().take(200).collect::<String>())
}

async fn links_put(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let id = body.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
    if id.is_empty() {
        return bad_request("id required");
    }
    let now = now_iso();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let mut fields: Vec<String> = Vec::new();
    let mut vals: Vec<String> = Vec::new();
    for k in ["url", "title", "description"] {
        if let Some(v) = body.get(k).and_then(|x| x.as_str()) {
            fields.push(format!("{k} = ?"));
            vals.push(v.to_string());
        }
    }
    if body.get("tags").is_some() {
        fields.push("tags = ?".into());
        vals.push(parse_tags_json(body.get("tags")));
    }
    fields.push("updated_at = ?".into());
    vals.push(now);
    if fields.is_empty() {
        return bad_request("no fields to update");
    }
    let sql = format!("UPDATE links SET {} WHERE id = ?", fields.join(", "));
    let mut params: Vec<rusqlite::types::Value> = vals.iter().map(|s| rusqlite::types::Value::Text(s.clone())).collect();
    params.push(rusqlite::types::Value::Text(id.clone()));
    match c.execute(&sql, rusqlite::params_from_iter(params)) {
        Ok(n) if n == 0 => json_resp(404, &json!({"error": "not found", "id": id})),
        Ok(_) => ok200(&json!({"ok": true, "id": id})),
        Err(e) => db_err(&e.to_string()),
    }
}

async fn links_delete(State(app): State<crate::App>, axum::extract::Path(id): axum::extract::Path<String>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let n = c.execute("DELETE FROM links WHERE id = ?", rusqlite::params![id]).unwrap_or(0);
    if n == 0 {
        return json_resp(404, &json!({"error": "not found", "id": id}));
    }
    ok200(&json!({"ok": true, "id": id}))
}

// ---------- settings ----------

async fn settings_get(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let def = json!({"widgets": [
        {"type": "stats"}, {"type": "trend"}, {"type": "scopes"},
        {"type": "recent_tasks"}, {"type": "recent_mem"},
    ]});
    let Ok(c) = get_db(&app.cfg) else {
        return ok200(&def);
    };
    if let Some(v) = c.query_row(
        "SELECT value FROM settings WHERE key = 'dashboard_widgets'",
        [],
        |r| r.get::<_, Option<String>>(0),
    ).ok().flatten() {
        if let Ok(cfg) = serde_json::from_str::<Value>(&v) {
            if cfg.get("widgets").and_then(|x| x.as_array()).is_some() {
                return ok200(&cfg);
            }
        }
    }
    ok200(&def)
}

async fn settings_put(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let widgets = match body.get("widgets").and_then(|x| x.as_array()) {
        Some(w) if w.len() <= 12 => w,
        _ => return bad_request("widgets array required (max 12)"),
    };
    let now = now_iso();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    if let Err(e) = c.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('dashboard_widgets', ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        rusqlite::params![serde_json::to_string(&body).unwrap_or_default(), now],
    ) {
        return db_err(&e.to_string());
    }
    let _ = widgets;
    ok200(&json!({"ok": true}))
}

// ---------- nebula 写端点 (CF handleNebula 的写部分) ----------

async fn nebula_merge(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let from = body.get("from").and_then(|x| x.as_str()).unwrap_or("");
    let to = body.get("to").and_then(|x| x.as_str()).unwrap_or("");
    if from.is_empty() || to.is_empty() {
        return bad_request("{ from, to } required");
    }
    if from == to {
        return bad_request("cannot merge entity with itself");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let fetch = |id: &str| -> Option<Value> {
        c.query_row(
            "SELECT id, name, description, parent_id, aliases FROM entities WHERE id = ?",
            rusqlite::params![id],
            |r| {
                Ok(json!({"id": r.get::<_, String>(0)?, "name": r.get::<_, String>(1)?,
                          "description": r.get::<_, String>(2)?, "parent_id": r.get::<_, String>(3)?,
                          "aliases": r.get::<_, String>(4)?}))
            },
        )
        .ok()
    };
    let Some(f) = fetch(from) else {
        return json_resp(404, &json!({"error": format!("source {from} not found")}));
    };
    let Some(t) = fetch(to) else {
        return json_resp(404, &json!({"error": format!("target {to} not found")}));
    };
    let mut moved_edges = 0u64;
    // in-edges: to_id = from → to
    let in_edges: Vec<(String, String, String, String, String)> = c
        .prepare("SELECT from_id, rel, to_id, note, created_at FROM edges WHERE to_id = ?")
        .and_then(|mut st| {
            st.query_map(rusqlite::params![from], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect())
        })
        .unwrap_or_default();
    for (fid, rel, _tid, note, created) in in_edges {
        if fid == to {
            continue;
        }
        if c.execute(
            "INSERT INTO edges (from_id, rel, to_id, note, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(from_id, rel, to_id) DO UPDATE SET note = excluded.note",
            rusqlite::params![fid, rel, to, note, created],
        ).is_ok()
        {
            let _ = c.execute(
                "DELETE FROM edges WHERE from_id = ? AND rel = ? AND to_id = ?",
                rusqlite::params![fid, rel, from],
            );
            moved_edges += 1;
        }
    }
    // out-edges: from_id = from → to
    let out_edges: Vec<(String, String, String, String, String)> = c
        .prepare("SELECT from_id, rel, to_id, note, created_at FROM edges WHERE from_id = ?")
        .and_then(|mut st| {
            st.query_map(rusqlite::params![from], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect())
        })
        .unwrap_or_default();
    for (_fid, rel, tid, note, created) in out_edges {
        if tid == to {
            continue;
        }
        if c.execute(
            "INSERT INTO edges (from_id, rel, to_id, note, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(from_id, rel, to_id) DO UPDATE SET note = excluded.note",
            rusqlite::params![to, rel, tid, note, created],
        ).is_ok()
        {
            let _ = c.execute(
                "DELETE FROM edges WHERE from_id = ? AND rel = ? AND to_id = ?",
                rusqlite::params![from, rel, tid],
            );
            moved_edges += 1;
        }
    }
    let _ = c.execute("DELETE FROM edges WHERE from_id = ? AND to_id = ?", rusqlite::params![from, from]);
    let _ = c.execute("DELETE FROM edges WHERE from_id = ? AND to_id = ?", rusqlite::params![to, to]);
    // aliases 合并
    let mut aliases: Vec<String> = serde_json::from_str::<Vec<String>>(&t["aliases"].as_str().unwrap_or("[]")).unwrap_or_default();
    let fname = t["name"].as_str().unwrap_or("").to_string();
    let _f = &f;
    let src_name = f["name"].as_str().unwrap_or("").to_string();
    if !aliases.contains(&src_name) {
        aliases.push(src_name);
    }
    let new_parent = if t["parent_id"].as_str().unwrap_or("").is_empty() {
        f["parent_id"].as_str().unwrap_or("").to_string()
    } else {
        t["parent_id"].as_str().unwrap_or("").to_string()
    };
    let _ = c.execute(
        "UPDATE entities SET aliases = ?, parent_id = ?, description = CASE WHEN description != '' THEN description WHEN ? != '' THEN ? ELSE description END WHERE id = ?",
        rusqlite::params![
            serde_json::to_string(&aliases).unwrap_or_default(),
            new_parent,
            f["description"].as_str().unwrap_or(""),
            f["description"].as_str().unwrap_or(""),
            to
        ],
    );
    let _ = c.execute("DELETE FROM entities WHERE id = ?", rusqlite::params![from]);
    let _ = fname;
    ok200(&json!({"ok": true, "merged": {"from": from, "to": to},
                 "movedEdges": moved_edges, "aliases": aliases}))
}

async fn entity_status(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let ids: Vec<String> = body
        .get("ids")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    let status = body.get("status").and_then(|x| x.as_str()).unwrap_or("");
    if ids.is_empty() {
        return bad_request("ids[] required");
    }
    if !["active", "provisional", "archived"].contains(&status) {
        return bad_request("status must be active|provisional|archived");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let mut updated = 0;
    for id in ids.iter().take(100) {
        updated += c.execute("UPDATE entities SET status = ? WHERE id = ?",
            rusqlite::params![status, id]).unwrap_or(0);
    }
    ok200(&json!({"ok": true, "updated": updated}))
}

async fn entity_reparent(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let id = body.get("id").and_then(|x| x.as_str()).unwrap_or("");
    let parent_id = body.get("parent_id").and_then(|x| x.as_str()).unwrap_or("");
    if id.is_empty() {
        return bad_request("id required");
    }
    // 环检测: 从 parent 向上走, 遇到 id = 环
    if !parent_id.is_empty() {
        let c = match get_db(&app.cfg) {
            Ok(c) => c,
            Err(r) => return r,
        };
        let mut cur = parent_id.to_string();
        let mut visited = std::collections::HashSet::new();
        while !cur.is_empty() {
            if !visited.insert(cur.clone()) {
                return bad_request("cycle detected");
            }
            let next = c.query_row(
                "SELECT parent_id FROM entities WHERE id = ?",
                rusqlite::params![cur],
                |r| r.get::<_, String>(0),
            )
            .unwrap_or_default();
            cur = next;
        }
        if visited.contains(id) {
            return bad_request("cycle: parent chain contains the entity itself");
        }
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let n = c.execute("UPDATE entities SET parent_id = ? WHERE id = ?",
        rusqlite::params![parent_id, id]).unwrap_or(0);
    if n == 0 {
        return json_resp(404, &json!({"error": format!("entity {id} not found")}));
    }
    ok200(&json!({"ok": true, "id": id, "parent_id": parent_id}))
}

async fn skills_merge(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let dry = body.get("dry_run").and_then(|x| x.as_bool()).unwrap_or(false);
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    // scope 桶分组 (memories/xxx 与 mem/xxx 归并)
    let all: Vec<(String, i64)> = c
        .prepare("SELECT skill, COUNT(*) n FROM atom_skills GROUP BY skill")
        .and_then(|mut st| {
            st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .map(|it| it.filter_map(|x| x.ok()).collect())
        })
        .unwrap_or_default();
    let scope_key = |s: &str| -> Option<String> {
        if s == "memories" || s == "mem" {
            Some(String::new())
        } else if let Some(rest) = s.strip_prefix("memories/") {
            Some(rest.to_string())
        } else if let Some(rest) = s.strip_prefix("mem/") {
            Some(rest.to_string())
        } else {
            None
        }
    };
    let mut groups: HashMap<String, Vec<(String, i64)>> = HashMap::new();
    for (skill, n) in all {
        if let Some(k) = scope_key(&skill) {
            groups.entry(k).or_default().push((skill, n));
        }
    }
    for v in groups.values_mut() {
        v.sort_by(|a, b| b.1.cmp(&a.1));
    }
    let mut targets: Vec<(String, Vec<(String, i64)>)> = groups
        .into_iter()
        .filter(|(_, v)| v.len() > 1)
        .collect();
    if let Some(scope) = body.get("scope").and_then(|x| x.as_str()) {
        targets.retain(|(k, _)| k == scope || k.is_empty());
    }
    let mut report: Vec<Value> = Vec::new();
    let (mut moved, mut deduped) = (0u64, 0u64);
    for (sk, variants) in targets {
        let target = variants[0].0.clone();
        let sources: Vec<String> = variants[1..].iter().map(|v| v.0.clone()).collect();
        let mut entry = json!({"scope": sk, "target": target, "sources": sources});
        if dry {
            entry["plan"] = json!(format!("{} → merged into {}", sources.len(), target));
            report.push(entry);
            continue;
        }
        for src in &sources {
            let rows: Vec<(String, String, i64)> = c
                .prepare("SELECT atom_id, section, position FROM atom_skills WHERE skill = ? ORDER BY position")
                .and_then(|mut st| {
                    st.query_map(rusqlite::params![src], |r| {
                        Ok((
                            r.get::<_, String>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, i64>(2)?,
                        ))
                    })
                    .map(|it| it.filter_map(|x| x.ok()).collect())
                })
                .unwrap_or_default();
            for (atom_id, section, position) in rows {
                let exists = c.query_row(
                    "SELECT 1 FROM atom_skills WHERE atom_id = ? AND skill = ?",
                    rusqlite::params![atom_id, target],
                    |_| Ok(1),
                ).is_ok();
                if exists {
                    deduped += 1;
                } else {
                    if c.execute(
                        "INSERT INTO atom_skills (skill, atom_id, section, position) VALUES (?, ?, ?, ?)",
                        rusqlite::params![target, atom_id, section, position],
                    ).is_ok()
                    {
                        moved += 1;
                    }
                }
            }
            let _ = c.execute("DELETE FROM atom_skills WHERE skill = ?", rusqlite::params![src]);
        }
        report.push(entry);
    }
    let _ = crate::ops::now_f64();
    ok200(&json!({"ok": true, "dry_run": dry, "moved": moved, "deduped": deduped, "report": report}))
}

// ---------- maintenance consolidate (批处理去重合并) ----------

async fn consolidate(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok_ext(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let scope = qval(&q, "scope");
    let batch = crate::qint(&q, "batch", 20).min(60) as i64;
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let sql = if scope.is_empty() || scope == "all" {
        "SELECT id, content, scope, hash FROM memories ORDER BY created_at ASC LIMIT ?"
    } else {
        "SELECT id, content, scope, hash FROM memories WHERE scope = ? ORDER BY created_at ASC LIMIT ?"
    };
    let rows: Vec<(String, String, String, String)> = c
        .prepare(sql)
        .and_then(|mut st| {
            let p = if scope.is_empty() || scope == "all" {
                rusqlite::params![batch]
            } else {
                rusqlite::params![scope, batch]
            };
            st.query_map(p, |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect())
        })
        .unwrap_or_default();
    // 批内精确 hash 去重 (廉价)
    let mut seen: HashMap<String, String> = HashMap::new();
    let mut removed = 0u64;
    for (id, _content, _scope, hash) in &rows {
        if let Some(first) = seen.get(hash) {
            if first != id {
                let _ = c.execute("DELETE FROM memories WHERE id = ?", rusqlite::params![id]);
                let _ = c.execute("DELETE FROM mem_vectors WHERE id = ?", rusqlite::params![id]);
                removed += 1;
            }
        } else {
            seen.insert(hash.clone(), id.clone());
        }
    }
    let remaining = rows.len() as u64 - removed;
    if removed > 0 {
        crate::VS_NOTIFY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let _ = app;
    ok200(&json!({"ok": true, "batch": rows.len(), "removed": removed,
                 "remaining_in_batch": remaining,
                 "note": "精确 hash 去重已完成; 语义合并走 /api/memories 写入链路"}))
}

// ---------- 标签解析 (与 memories_api 同款, 域内独立避免跨域依赖) ----------

fn parse_tags_json(v: Option<&Value>) -> String {
    let parts: Vec<String> = match v {
        Some(s) if s.is_string() => s.as_str().unwrap_or("")
            .split([',', ' ', '\t', '\n'])
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect(),
        Some(a) if a.is_array() => a
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x.as_str())
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect(),
        _ => Vec::new(),
    };
    serde_json::to_string(&parts).unwrap_or_else(|_| "[]".to_string())
}
