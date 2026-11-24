//! 设备命令通道 (阶段二): Rust 版 CommandHub。
//! 1:1 复刻 CF linghub worker 的 Durable Object CommandHub + /api/ws + /api/command:
//!   - 每 host 一条 WS 长连 (设备侧: linghub-phone CommandChannel)
//!   - 指令暂存到 SQLite `commands` 表 (ack 追踪 + 离线兜底补发, 60s 过期)
//!   - 在线则即时推送 {type:'command', id, command, camera, roomId, ts}
//!   - 设备回 {type:'ping'}->pong 保活; {type:'ack', id} 删暂存
//! CF 是"每 host 一个 DO 实例", Rust 单进程用 `host 列` 等价隔离 (同一张表)。
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::body::Bytes;
use axum::http::Request;
use axum::body::Body;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::api_common::{bad_request, cmd_id, db_err, get_db, ok200};
use crate::db;
use crate::{auth_ok, json_resp, qparams, qval};

type Sender = mpsc::Sender<Value>;

/// host -> 活跃 WS 的发送端 (重连即替换)。metrics_api 用它判在线。
static CMDS: Lazy<RwLock<HashMap<String, Sender>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn device_online(host: &str) -> bool {
    CMDS.read()
        .unwrap()
        .get(host)
        .map(|t| !t.is_closed())
        .unwrap_or(false)
}

/// host 的活跃 WS 发送端克隆 (media_api 推 RTC/camera 命令用)
pub fn active_sender(host: &str) -> Option<Sender> {
    CMDS.read().unwrap().get(host).cloned()
}

pub fn router() -> Router<crate::App> {
    Router::new()
        .route("/api/ws", get(ws_handler))
        .route("/api/command", post(command_post))
        .route("/api/command/pending", get(command_pending_get))
}

// ---------- GET /api/ws ----------

async fn ws_handler(
    State(app): State<crate::App>,
    ws: WebSocketUpgrade,
    req: Request<Body>,
) -> axum::response::Response {
    let q = qparams(&req);
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let host = qval(&q, "host").trim().chars().take(40).collect::<String>();
    if host.is_empty() {
        return bad_request("host required");
    }
    let cfg = app.cfg.clone();
    let mut resp = ws.on_upgrade(move |socket| async move { session(cfg, host, socket).await; });
    let h = resp.headers_mut();
    h.insert(
        "Access-Control-Allow-Origin",
        axum::http::HeaderValue::from_static("*"),
    );
    resp
}

/// 单条设备 WS 会话: 补发暂存 -> 收/发双向循环 -> 清理。
async fn session(cfg: Arc<crate::config::Config>, host: String, mut socket: WebSocket) {
    let (tx, mut rx) = mpsc::channel::<Value>(32);
    {
        CMDS.write().unwrap().insert(host.clone(), tx.clone());
    }
    // 补发暂存指令 (CF flushPending: 60s 过期 + 同 command 只发最新)
    flush_pending(&cfg, &host, &tx).await;
    // 收/发循环 (axum 0.7: 单一 socket 同时 recv/send, &mut 串行无竞争)
    loop {
        tokio::select! {
            msg = socket.recv() => match msg {
                Some(Ok(m)) => {
                    if let Message::Text(t) = m {
                        if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                            match v.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                                "ping" => {
                                    let _ = tx.try_send(json!({"type": "pong", "ts": crate::api_common::now_ms()}));
                                }
                                "ack" => {
                                    if let Some(id) = v.get("id").and_then(|x| x.as_str()) {
                                        if let Ok(c) = db::db(&cfg) {
                                            let _ = c.execute(
                                                "DELETE FROM commands WHERE id = ? AND host = ?",
                                                (id, host.as_str()),
                                            );
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else if matches!(m, Message::Close(_)) {
                        break;
                    }
                }
                _ => break, // 流结束/错误
            },
            maybe_v = rx.recv() => match maybe_v {
                Some(v) => {
                    if socket.send(Message::Text(v.to_string().into())).await.is_err() {
                        break;
                    }
                }
                None => break,
            },
        }
    }
    CMDS.write().unwrap().remove(&host);
}

/// 把未 ack 的暂存指令推给刚连上的设备 (60s 过期 + 同 command 只发最新)
async fn flush_pending(cfg: &Arc<crate::config::Config>, host: &str, tx: &Sender) {
    let Ok(c) = db::db(cfg) else { return };
    let Ok(mut st) = c.prepare(
        "SELECT id, command, camera, room_id, ts FROM commands WHERE host = ? ORDER BY ts",
    ) else { return };
    let Ok(rows) = st.query_map((host,), |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<String>>(2)?,
            r.get::<_, Option<String>>(3)?,
            r.get::<_, i64>(4)?,
        ))
    }) else { return };
    let rows: Vec<_> = rows.filter_map(|x| x.ok()).collect();
    let now = crate::api_common::now_ms();
    let mut seen: HashMap<String, Value> = HashMap::new();
    for (id, command, camera, room_id, ts) in rows {
        if now.saturating_sub(ts) < 60000 {
            seen.insert(
                command.clone(),
                json!({
                    "type": "command", "id": id, "command": command,
                    "camera": camera.unwrap_or_else(|| "back".into()),
                    "roomId": room_id.unwrap_or_default(), "ts": ts,
                }),
            );
        } else {
            let _ = c.execute("DELETE FROM commands WHERE id = ?", (id,));
        }
    }
    for v in seen.values() {
        if tx.try_send(v.clone()).is_err() {
            return;
        }
    }
}

// ---------- POST /api/command ----------

const VALID_CMDS: [&str; 6] = [
    "photo",
    "ring",
    "start-stream",
    "stop-stream",
    "start-wt-stream",
    "stop-wt-stream",
];

async fn command_post(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let body: Bytes = match axum::body::to_bytes(req.into_body(), 1 << 22).await {
        Ok(b) => b,
        Err(e) => return bad_request(&e.to_string()),
    };
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return bad_request("bad json"),
    };
    let host = body
        .get("host")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim()
        .chars()
        .take(40)
        .collect::<String>();
    let command = body
        .get("command")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if host.is_empty() {
        return bad_request("host required");
    }
    if !VALID_CMDS.contains(&command.as_str()) {
        return bad_request("command must be photo|ring|start-stream|stop-stream|start-wt-stream|stop-wt-stream");
    }
    let camera = if body.get("camera").and_then(|x| x.as_str()) == Some("front") {
        "front".to_string()
    } else {
        "back".to_string()
    };
    // payload 语义随 command 变化 (CF: photo/ring=camera, *-stream=roomId)
    let room_id = match command.as_str() {
        "start-stream" | "start-wt-stream" => Some(
            body.get("payload")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .chars()
                .take(200)
                .collect::<String>(),
        ),
        _ => None,
    };
    let cmd = json!({
        "type": "command",
        "id": cmd_id(),
        "command": command,
        "camera": camera,
        "roomId": room_id.clone().unwrap_or_default(),
        "ts": crate::api_common::now_ms(),
    });
    // 暂存 (ack 追踪 + 离线兜底)
    if let Ok(c) = get_db(&app.cfg) {
        if let Err(e) = c.execute(
            "INSERT OR REPLACE INTO commands (id, host, command, camera, room_id, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (
                cmd["id"].as_str().unwrap_or(""),
                host.as_str(),
                command.as_str(),
                camera.as_str(),
                room_id.as_deref().unwrap_or(""),
                cmd["ts"].as_i64().unwrap_or(0),
            ),
        ) {
            return db_err(&e.to_string());
        }
    }
    // 在线则即时推送
    let online = CMDS
        .read()
        .unwrap()
        .get(&host)
        .map(|t| t.try_send(cmd.clone()).is_ok())
        .unwrap_or(false);
    ok200(&json!({
        "ok": true, "id": cmd["id"], "host": host,
        "command": command, "camera": camera, "online": online,
    }))
}

// ---------- GET /api/command/pending ----------

async fn command_pending_get(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let host = qval(&q, "host").trim().chars().take(40).collect::<String>();
    if host.is_empty() {
        return bad_request("host required");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let mut st = match c.prepare(
        "SELECT id, command, camera, room_id, ts FROM commands WHERE host = ? ORDER BY ts",
    ) {
        Ok(s) => s,
        Err(e) => return db_err(&e.to_string()),
    };
    let rows = match st.query_map((host.as_str(),), |r| {
        Ok(json!({
            "id": r.get::<_, String>(0)?,
            "command": r.get::<_, String>(1)?,
            "camera": r.get::<_, Option<String>>(2)?,
            "roomId": r.get::<_, Option<String>>(3)?,
            "ts": r.get::<_, i64>(4)?,
        }))
    }) {
        Ok(rows) => rows.filter_map(|x| x.ok()).collect::<Vec<_>>(),
        Err(e) => return db_err(&e.to_string()),
    };
    ok200(&json!({"commands": rows}))
}
