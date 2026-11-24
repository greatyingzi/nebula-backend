//! 媒体/实时域 (阶段三): camera 帧存取 + phone command + RTC 信令 + viewer presence。
//! 1:1 复刻 CF linghub worker 的三个 Durable Object (RtcRoom/PresenceHub) + camera R2 + phone command:
//!   - PUT  /api/camera/upload?cam=pi|mac|phone  帧写本地文件 (替代 R2), 无观看者跳过 (src=command 例外)
//!   - GET  /api/camera/last / status / mac/request / mac/status
//!   - GET  /api/rtc/credentials  TURN iceServers (coturn + CF RTC API 兜底, 内存缓存)
//!   - POST /api/rtc/room / start / start-wt / stop   (经命令通道下发设备)
//!   - GET  /api/rtc/ws?room=&role=   WebRTC 信令 (publisher 1 + viewer N, 转发 offer/answer/ice)
//!   - GET  /api/rtc/debug / sdp / POST rtc/log / GET rtc/log
//!   - GET  /api/viewer/ws?viewerId=  观看者 presence (0↔N 翻转 → 命令通道推 viewer-online/offline)
//!   - GET  /api/viewers
//!   - POST /api/phone/command / ack + GET  (settings 表轮询, 60s 有效)
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock};

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::response::Response;
use axum::routing::{get, post, put};
use axum::Router;
use once_cell::sync::Lazy;
use serde_json::{json, Value};

use crate::api_common::{bad_request, get_db, now_iso, now_ms, ok200, db_err};
use crate::{auth_ok, json_resp, qparams, qval};

// ---------- 全局状态 ----------

/// RTC 信令房间: room → 会话表
#[derive(Clone)]
struct RtcSession {
    role: &'static str, // "publisher" | "viewer"
    id: String,
    tx: tokio::sync::mpsc::UnboundedSender<Value>,
}
#[derive(Clone)]
struct RtcRoomState {
    host: String,
    mode: String,
    sessions: HashMap<String, RtcSession>,
    publisher: Option<String>, // session id
    viewer_count: u64,
}
type RtcRooms = RwLock<HashMap<String, Arc<RtcRoomState>>>;
static RTC_ROOMS: Lazy<RtcRooms> = Lazy::new(|| RwLock::new(HashMap::new()));

/// viewer presence (单实例 "global"): viewerId → tx
type Viewers = RwLock<HashMap<String, tokio::sync::mpsc::UnboundedSender<Value>>>;
static VIEWERS: Lazy<Viewers> = Lazy::new(|| RwLock::new(HashMap::new()));
/// 翻转状态 (0↔N 才通知)
static VIEWERS_ONLINE: Lazy<std::sync::atomic::AtomicBool> =
    Lazy::new(|| std::sync::atomic::AtomicBool::new(false));
/// 僵尸踢除任务开关 (presence 60s 巡检)
static PRESENCE_TICK: Lazy<std::sync::atomic::AtomicBool> =
    Lazy::new(|| std::sync::atomic::AtomicBool::new(false));

/// ICE servers 缓存 (CF: _iceCache, 600s)
struct IceCache {
    key: String,
    expire_ms: i64,
    data: Value,
}
static ICE_CACHE: Lazy<Mutex<Option<IceCache>>> = Lazy::new(|| Mutex::new(None));

fn cam_key(cam: &str) -> &'static str {
    match cam {
        "mac" => "mac_last.jpg",
        "phone" => "phone_last.jpg",
        _ => "last.jpg",
    }
}

/// 发送命令到设备 (复用 ws_api 的 CommandHub: 在线即推, 离线暂存)
async fn send_device_command(app: &crate::App, host: &str, command: &str, camera: &str, room_id: &str) -> Value {
    use crate::ws_api;
    if host.is_empty() {
        return json!({"ok": false, "error": "host required"});
    }
    let id = crate::api_common::cmd_id();
    let ts = now_ms();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(_) => return json!({"ok": false, "error": "db unavailable"}),
    };
    let _ = c.execute(
        "INSERT OR REPLACE INTO commands (id, host, command, camera, room_id, ts) VALUES (?, ?, ?, ?, ?, ?)",
        rusqlite::params![id, host, command, camera, room_id, ts],
    );
    if ws_api::device_online(host) {
        if let Some(tx) = ws_api::active_sender(host) {
            let _ = tx.send(json!({"type": "command", "id": id, "command": command,
                                    "camera": camera, "roomId": room_id, "ts": ts}));
        }
        json!({"ok": true, "id": id, "delivered": true})
    } else {
        json!({"ok": true, "id": id, "delivered": false, "pending": true})
    }
}

// ---------- router ----------

pub fn router() -> Router<crate::App> {
    Router::new()
        .route("/api/camera/upload", put(camera_upload))
        .route("/api/camera/last", get(camera_last))
        .route("/api/camera/status", get(camera_status))
        .route("/api/camera/mac/request", post(camera_mac_request))
        .route("/api/camera/mac/status", get(camera_mac_status))
        .route("/api/rtc/credentials", get(rtc_credentials))
        .route("/api/rtc/room", post(rtc_room))
        .route("/api/rtc/ws", get(rtc_ws))
        .route("/api/rtc/log", get(rtc_log_get).post(rtc_log_post))
        .route("/api/rtc/log/hosts", get(rtc_log_hosts))
        .route("/api/rtc/debug", get(rtc_debug))
        .route("/api/rtc/sdp", get(rtc_sdp))
        .route("/api/rtc/start", post(rtc_start))
        .route("/api/rtc/start-wt", post(rtc_start_wt))
        .route("/api/rtc/stop", post(rtc_stop))
        .route("/api/viewer/ws", get(viewer_ws))
        .route("/api/viewers", get(viewers_get))
        .route("/api/phone/command", get(phone_cmd_get).post(phone_cmd_post))
        .route("/api/phone/command/ack", post(phone_cmd_ack))
}

// ---------- camera ----------

async fn camera_upload(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let cam = qval(&q, "cam");
    let src = qval(&q, "src");
    // 无观看者 → 跳过 (src=command 例外: 用户主动抓拍)
    let viewers = VIEWERS.read().unwrap().len();
    if viewers == 0 && src != "command" {
        return ok200(&json!({"ok": false, "skipped": true, "reason": "no-viewer",
                             "cam": cam, "ts": now_iso()}));
    }
    let body = match axum::body::to_bytes(req.into_body(), 3 * 1024 * 1024).await {
        Ok(b) if b.len() >= 100 => b,
        Ok(_) => return bad_request("empty frame"),
        Err(e) => return db_err(&e.to_string()),
    };
    if body.len() > 2 * 1024 * 1024 {
        return bad_request("frame too large");
    }
    if let Err(e) = std::fs::create_dir_all(&app.cfg.camera_dir) {
        return db_err(&format!("camera dir: {e}"));
    }
    let path = std::path::Path::new(&app.cfg.camera_dir).join(cam_key(&cam));
    if let Err(e) = std::fs::write(&path, &body) {
        return db_err(&format!("write frame: {e}"));
    }
    let ts = now_iso();
    // 帧时间戳落 settings (CF: R2 uploaded 时间)
    if let Ok(c) = get_db(&app.cfg) {
        let _ = c.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES ('cam_ts', ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            rusqlite::params![format!("{}|{}", cam, ts), ts],
        );
    }
    ok200(&json!({"ok": true, "cam": cam, "bytes": body.len(), "ts": ts}))
}

fn frame_ts(app: &crate::App, cam: &str) -> Option<String> {
    let Ok(c) = get_db(&app.cfg) else {
        return None;
    };
    let Ok(v) = c.query_row("SELECT value FROM settings WHERE key = 'cam_ts'", [], |r| r.get::<_, String>(0)) else {
        return None;
    };
    let (c_cam, ts) = v.split_once('|')?;
    (c_cam == cam).then_some(ts.to_string())
}

async fn camera_last(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let cam = qval(&q, "cam");
    let path = std::path::Path::new(&app.cfg.camera_dir).join(cam_key(&cam));
    match std::fs::read(&path) {
        Ok(data) => {
            let ts = frame_ts(&app, &cam).unwrap_or_default();
            let mut r = Response::new(Body::from(data));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(header::CONTENT_TYPE, header::HeaderValue::from_static("image/jpeg"));
            r.headers_mut().insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-cache"));
            r.headers_mut().insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, header::HeaderValue::from_static("*"));
            if let Ok(h) = axum::http::HeaderValue::from_str(&ts) {
                r.headers_mut().insert("X-Frame-Ts", h);
            }
            r
        }
        Err(_) => json_resp(404, &json!({"error": "no frame yet"})),
    }
}

async fn camera_status(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let cam = qval(&q, "cam");
    let path = std::path::Path::new(&app.cfg.camera_dir).join(cam_key(&cam));
    let has = path.exists();
    let uploaded = if has {
        frame_ts(&app, &cam)
    } else {
        None
    };
    ok200(&json!({"cam": cam, "has_frame": has, "uploaded": uploaded}))
}

async fn camera_mac_request(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let dur = body.get("duration").and_then(|x| x.as_i64()).unwrap_or(90).min(300).max(5) as i64;
    let until_ms = now_ms() + dur * 1000;
    let until = crate::api_common::iso_date(until_ms / 1000);
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let _ = c.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('mac_camera_request', ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        rusqlite::params![serde_json::to_string(&json!({"active": true, "until": until})).unwrap_or_default(), now_iso()],
    );
    // 推送 cam-window-on 给 macbook (CF: 推模式)
    let res = send_device_command(&app, "macbook", "cam-window-on", "front", "").await;
    let mut out = json!({"ok": true, "active": true, "until": until, "duration": dur});
    if let Some(r) = out.as_object_mut() {
        r.insert("push".into(), res);
    }
    ok200(&out)
}

async fn camera_mac_status(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let (mut active, mut until) = (false, None);
    if let Ok(c) = get_db(&app.cfg) {
        if let Some(v) = c.query_row("SELECT value FROM settings WHERE key = 'mac_camera_request'", [], |r| r.get::<_, Option<String>>(0)).ok().flatten() {
            if let Ok(r) = serde_json::from_str::<Value>(&v) {
                let until_str = r.get("until").and_then(|x| x.as_str()).unwrap_or("");
                let now = now_ms();
                let then = crate::api_common::parse_iso_ms(until_str);
                active = r.get("active").and_then(|x| x.as_bool()).unwrap_or(false) && then > now;
                until = if active { Some(until_str.to_string()) } else { None };
            }
        }
    }
    let path = std::path::Path::new(&app.cfg.camera_dir).join("mac_last.jpg");
    let has_frame = path.exists();
    let uploaded = if has_frame { frame_ts(&app, "mac") } else { None };
    let _ = req;
    ok200(&json!({"active": active, "until": until, "has_frame": has_frame, "uploaded": uploaded}))
}

// ---------- RTC credentials (CF getIceServers 1:1) ----------

async fn rtc_credentials(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let mode = if qval(&q, "rpi") == "1" { "fast" } else { "normal" };
    // 缓存命中
    {
        let guard = ICE_CACHE.lock().unwrap();
        if let Some(c) = guard.as_ref() {
            if c.key == mode && c.expire_ms > now_ms() {
                return ok200(&json!({"iceServers": c.data}));
            }
        }
    }
    let servers: Vec<Value> = Vec::new();
    let mut out: Vec<Value> = Vec::new();
    // 树莓派国内 coturn
    if !app.cfg.rpi_turn_password.is_empty() {
        out.push(json!({"urls": ["stun:relay.example.invalid:3478", "stun:stun.cloudflare.com:3478"]}));
        if mode == "fast" {
            out.push(json!({
                "urls": ["turn:relay.example.invalid:3478?transport=udp", "turn:relay.example.invalid:3478?transport=tcp"],
                "username": "linghub", "credential": app.cfg.rpi_turn_password,
            }));
        }
    }
    // Cloudflare TURN 兜底
    if !app.cfg.turn_key_id.is_empty() && !app.cfg.turn_key_api_token.is_empty() {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!(
                "https://rtc.live.cloudflare.com/v1/turn/keys/{}/credentials/generate-ice-servers",
                app.cfg.turn_key_id
            ))
            .bearer_auth(&app.cfg.turn_key_api_token)
            .json(&json!({"ttl": 86400}))
            .timeout(std::time::Duration::from_secs(8))
            .send()
            .await;
        if let Ok(r) = resp {
            if r.status().is_success() {
                if let Ok(data) = r.json::<Value>().await {
                    for srv in data.get("iceServers").and_then(|x| x.as_array()).cloned().unwrap_or_default() {
                        let urls: Vec<String> = srv
                            .get("urls")
                            .and_then(|x| x.as_array())
                            .map(|a| {
                                a.iter()
                                    .filter_map(|u| {
                                        let u = u.as_str()?;
                                        if u.starts_with("stun:") || u.starts_with("turns:")
                                            || (u.starts_with("turn:") && !u.contains("transport=udp"))
                                        {
                                            Some(u.to_string())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        if !urls.is_empty() {
                            out.push(json!({"urls": urls}));
                        }
                    }
                }
            }
        }
    }
    let data = Value::Array(out);
    *ICE_CACHE.lock().unwrap() = Some(IceCache {
        key: mode.to_string(),
        expire_ms: now_ms() + 600_000,
        data,
    });
    let _ = servers;
    ok200(&json!({"iceServers": ICE_CACHE.lock().unwrap().as_ref().unwrap().data.clone()}))
}

// ---------- RTC room + 信令 ----------

async fn rtc_room(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let host = body.get("host").and_then(|x| x.as_str()).unwrap_or("").trim();
    if host.is_empty() {
        return bad_request("host required");
    }
    let mode = if body.get("mode").and_then(|x| x.as_str()) == Some("stream") {
        "stream"
    } else {
        "camera"
    };
    let room_id = crate::api_common::gen_id();
    {
        let mut rooms = RTC_ROOMS.write().unwrap();
        rooms.insert(
            room_id.clone(),
            Arc::new(RtcRoomState {
                host: host.chars().take(40).collect(),
                mode: mode.to_string(),
                sessions: HashMap::new(),
                publisher: None,
                viewer_count: 0,
            }),
        );
    }
    // 60s 自毁 (无会话)
    let arc = {
        let rooms = RTC_ROOMS.read().unwrap();
        rooms.get(&room_id).cloned()
    };
    let room_id2 = room_id.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(600)).await;
        let still_empty = arc.map(|a| a.sessions.is_empty()).unwrap_or(true);
        if still_empty {
            let mut rooms = RTC_ROOMS.write().unwrap();
            if rooms.get(&room_id2).map(|a| a.sessions.is_empty()).unwrap_or(true) {
                rooms.remove(&room_id2);
            }
        }
    });
    ok200(&json!({"ok": true, "roomId": room_id, "host": host, "mode": mode}))
}

async fn rtc_ws(
    State(app): State<crate::App>,
    ws: WebSocketUpgrade,
    req: Request<Body>,
) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let room = qval(&q, "room").trim().to_string();
    let role = if qval(&q, "role") == "publisher" { "publisher" } else { "viewer" };
    if room.is_empty() {
        return bad_request("room required");
    }
    let room_state = {
        let rooms = RTC_ROOMS.read().unwrap();
        match rooms.get(&room) {
            Some(s) if !s.publisher.is_some() || role != "publisher" => s.clone(),
            Some(s) => s.clone(),
            None => return bad_request("room not found (创建后 10min 内有效)"),
        }
    };
    let session_id = crate::api_common::gen_id();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let room_id = room.clone();
    let is_pub = role == "publisher";
    // 注册本会话 (写锁内 clone→改→replace; task 不持有状态 Arc, 无引用计数冲突)
    {
        let mut rooms = RTC_ROOMS.write().unwrap();
        if let Some(old_arc) = rooms.get(&room_id).cloned() {
            let mut st = (*old_arc).clone();
            if is_pub {
                if let Some(old) = st.publisher.clone() {
                    if let Some(os) = st.sessions.get(&old) {
                        let _ = os.tx.send(json!({"type": "replaced-by-new-publisher"}));
                    }
                    st.sessions.remove(&old);
                }
                st.publisher = Some(session_id.clone());
                for s in st.sessions.values() {
                    if s.role == "viewer" {
                        let _ = s.tx.send(json!({"type": "publisher-ready"}));
                    }
                }
            } else {
                st.viewer_count += 1;
                if let Some(p) = st.sessions.values().find(|s| s.role == "publisher") {
                    let _ = p.tx.send(json!({"type": "viewer-joined", "viewerId": session_id}));
                }
            }
            st.sessions.insert(
                session_id.clone(),
                RtcSession {
                    role,
                    id: session_id.clone(),
                    tx,
                },
            );
            rooms.insert(room_id.clone(), Arc::new(st));
        }
    }

    ws.on_upgrade(move |sock| rtc_ws_task(app, sock, session_id, role, rx, room_id))
}

async fn rtc_ws_task(
    app: crate::App,
    mut sock: WebSocket,
    session_id: String,
    role: &'static str,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Value>,
    room_id: String,
) {
    // 收/发双向循环 (ws_api 同款: 单 socket + select!)
    loop {
        tokio::select! {
            msg = sock.recv() => match msg {
                Some(Ok(m)) => {
                    if matches!(m, Message::Close(_)) { break; }
                    let Ok(text) = m.to_text() else { continue };
                    let Ok(v) = serde_json::from_str::<Value>(text) else { continue };
                    let mtype = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    if mtype == "ping" {
                        let _ = sock.send(Message::Text(serde_json::to_string(&json!({"type": "pong", "ts": now_ms()})).unwrap().into())).await;
                        continue;
                    }
                    // SDP 方向日志 (offer/answer, 诊断视频方向)
                    if mtype == "offer" || mtype == "answer" {
                        let sdp_val = v.get(mtype).unwrap_or(&v);
                        let sdp = if let Some(x) = sdp_val.as_str() {
                            x.to_string()
                        } else {
                            sdp_val.get("sdp").and_then(|x| x.as_str()).unwrap_or("").to_string()
                        };
                        log_sdp(&app, &room_id, role, mtype, &sdp);
                    }
                    // offer/answer/ice 转发: publisher 发 → 全体 viewer; viewer 发 → publisher
                    let forward = {
                        let mut f = v.clone();
                        if let Some(o) = f.as_object_mut() {
                            o.insert("from".into(), json!(session_id));
                        }
                        f
                    };
                    let state = {
                        let rooms = RTC_ROOMS.read().unwrap();
                        rooms.get(&room_id).map(|a| (*a).clone())
                    };
                    if let Some(st) = state {
                        if role == "publisher" {
                            for s_ in st.sessions.values() {
                                if s_.role == "viewer" {
                                    let _ = s_.tx.send(forward.clone());
                                }
                            }
                        } else if let Some(p) = st.sessions.values().find(|s| s.role == "publisher") {
                            let _ = p.tx.send(forward);
                        }
                    }
                }
                _ => break,
            },
            maybe_v = rx.recv() => match maybe_v {
                Some(v) => {
                    if sock.send(Message::Text(v.to_string().into())).await.is_err() {
                        break;
                    }
                }
                None => break,
            },
        }
    }
    // close 清理 (写锁内 clone→改→replace)
    {
        let mut rooms = RTC_ROOMS.write().unwrap();
        if let Some(old_arc) = rooms.get(&room_id).cloned() {
            let mut st = (*old_arc).clone();
            st.sessions.remove(&session_id);
            let mut stop_requested = false;
            if role == "publisher" {
                if st.publisher.as_deref() == Some(session_id.as_str()) {
                    st.publisher = None;
                }
                for s_ in st.sessions.values() {
                    let _ = s_.tx.send(json!({"type": "publisher-left"}));
                }
            } else {
                st.viewer_count = st.viewer_count.saturating_sub(1);
                if let Some(p) = st.sessions.values().find(|s| s.role == "publisher") {
                    let _ = p.tx.send(json!({"type": "viewer-left", "viewerId": session_id}));
                }
                if st.viewer_count == 0 && !st.host.is_empty() {
                    stop_requested = true;
                }
            }
            if st.sessions.is_empty() {
                rooms.remove(&room_id);
            } else {
                rooms.insert(room_id.clone(), Arc::new(st));
            }
            if stop_requested {
                let host = old_arc.host.clone();
                tokio::spawn(async move {
                    // 60s 宽限期: 到期仍无 viewer → 下发 stop-stream (CF: alarm 60s 停播去抖)
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    let still_zero = {
                        let rooms = RTC_ROOMS.read().unwrap();
                        rooms.get(&room_id).map(|a| a.viewer_count == 0).unwrap_or(false)
                    };
                    if still_zero {
                        send_device_command(&app, &host, "stop-stream", "back", "").await;
                    }
                });
            }
        }
    }
}

fn rx2tx(_rx: &tokio::sync::mpsc::UnboundedReceiver<Value>, _v: Value) -> Result<(), ()> {
    // 信令回复走发送端; 这里为简化返回 Ok (真正发送在 send 循环)
    Ok(())
}

async fn rtc_debug(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let room = qval(&q, "room");
    if room.is_empty() {
        return ok200(&json!({"error": "room required"}));
    }
    let rooms = RTC_ROOMS.read().unwrap();
    let Some(r) = rooms.get(&room) else {
        return ok200(&json!({"error": "room not found"}));
    };
    let viewers: Vec<String> = r
        .sessions
        .values()
        .filter(|s| s.role == "viewer")
        .map(|s| s.id.chars().take(8).collect())
        .collect();
    ok200(&json!({
        "host": r.host, "mode": r.mode, "sessions": r.sessions.len(),
        "hasPublisher": r.publisher.is_some(),
        "publisherId": r.publisher.as_ref().map(|p| p.chars().take(8).collect::<String>()),
        "viewerCount": r.viewer_count, "viewers": viewers,
    }))
}

async fn rtc_sdp(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let room = qval(&q, "room");
    if room.is_empty() {
        return bad_request("room required");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let rows: Vec<Value> = c
        .prepare("SELECT ts, role, type, mlines FROM rtc_sdp_log WHERE room = ? ORDER BY ts DESC LIMIT 10")
        .and_then(|mut st| {
            st.query_map(rusqlite::params![room], |r| {
                Ok(json!({"ts": r.get::<_, i64>(0)?, "role": r.get::<_, String>(1)?,
                          "type": r.get::<_, String>(2)?, "mlines": r.get::<_, String>(3)?}))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect())
        })
        .unwrap_or_default();
    let _ = app;
    ok200(&json!({"items": rows}))
}

// 信令日志表 (rtc_ws 里写) — 在 init_db 建
pub fn log_sdp(app: &crate::App, room: &str, role: &str, mtype: &str, sdp: &str) {
    let mlines: String = sdp
        .lines()
        .filter(|l| {
            l.starts_with("m=")
                || l.starts_with("a=sendonly")
                || l.starts_with("a=recvonly")
                || l.starts_with("a=sendrecv")
                || l.starts_with("a=inactive")
        })
        .collect::<Vec<_>>()
        .join(" | ");
    if let Ok(c) = get_db(&app.cfg) {
        let _ = c.execute(
            "CREATE TABLE IF NOT EXISTS rtc_sdp_log (ts INTEGER, room TEXT, role TEXT, type TEXT, mlines TEXT)",
            [],
        );
        let _ = c.execute(
            "INSERT INTO rtc_sdp_log (ts, room, role, type, mlines) VALUES (?, ?, ?, ?, ?)",
            rusqlite::params![now_ms(), room, role, mtype, mlines],
        );
    }
}

async fn rtc_log_post(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let host = body.get("host").and_then(|x| x.as_str()).unwrap_or("").trim();
    if host.is_empty() {
        return bad_request("host required");
    }
    let lines = body
        .get("lines")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .chars()
        .take(50000)
        .collect::<String>();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let _ = c.execute(
        "INSERT INTO device_logs (host, lines, ts) VALUES (?, ?, ?)",
        rusqlite::params![host, lines, now_ms()],
    );
    let _ = c.execute(
        "DELETE FROM device_logs WHERE host = ? AND ts < ?",
        rusqlite::params![host, now_ms() - 3600000],
    );
    ok200(&json!({"ok": true}))
}

async fn rtc_log_hosts(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let hosts: Vec<String> = c
        .prepare("SELECT DISTINCT host FROM device_logs ORDER BY ts DESC LIMIT 50")
        .and_then(|mut st| {
            st.query_map([], |r| r.get::<_, String>(0))
                .map(|it| it.filter_map(|x| x.ok()).collect())
        })
        .unwrap_or_default();
    ok200(&json!({"hosts": hosts}))
}

async fn rtc_log_get(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let host = qval(&q, "host");
    if host.is_empty() {
        return bad_request("host required");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let rows: Vec<Value> = c
        .prepare("SELECT lines, ts FROM device_logs WHERE host = ? ORDER BY ts DESC LIMIT 20")
        .and_then(|mut st| {
            st.query_map(rusqlite::params![host], |r| {
                Ok(json!({"lines": r.get::<_, String>(0)?, "ts": r.get::<_, i64>(1)?}))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect())
        })
        .unwrap_or_default();
    ok200(&json!({"host": host, "items": rows}))
}

// ---------- RTC 控制 (经命令通道) ----------

fn clamp(v: Option<f64>, lo: f64, hi: f64, def: f64) -> u64 {
    match v {
        Some(x) if x.is_finite() => (x.clamp(lo, hi)) as u64,
        _ => def as u64,
    }
}

async fn rtc_start(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let host = body.get("host").and_then(|x| x.as_str()).unwrap_or("").trim();
    let room = body.get("roomId").and_then(|x| x.as_str()).unwrap_or("").trim();
    if host.is_empty() || room.is_empty() {
        return bad_request("host and roomId required");
    }
    let params = json!({
        "roomId": room,
        "width": clamp(body.get("width").and_then(|x| x.as_f64()), 320.0, 1920.0, 1280.0),
        "height": clamp(body.get("height").and_then(|x| x.as_f64()), 180.0, 1080.0, 720.0),
        "fps": clamp(body.get("fps").and_then(|x| x.as_f64()), 5.0, 60.0, 30.0),
        "bitrate": clamp(body.get("bitrate").and_then(|x| x.as_f64()), 100000.0, 8000000.0, 1500000.0),
    });
    let res = send_device_command(&app, host, "start-stream", "back", &params.to_string()).await;
    ok200(&json!({"ok": true, "host": host, "roomId": room, "streamParams": params, "send": res}))
}

async fn rtc_start_wt(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let host = body.get("host").and_then(|x| x.as_str()).unwrap_or("").trim();
    let room = body.get("roomId").and_then(|x| x.as_str()).unwrap_or("").trim();
    if host.is_empty() || room.is_empty() {
        return bad_request("host and roomId required");
    }
    let relay = "relay.example.invalid:48000";
    let res = send_device_command(&app, host, "start-wt-stream", "back", &format!("{room}|{relay}")).await;
    ok200(&json!({"ok": true, "host": host, "roomId": room, "relayAddr": relay, "send": res}))
}

async fn rtc_stop(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let host = body.get("host").and_then(|x| x.as_str()).unwrap_or("").trim();
    if host.is_empty() {
        return bad_request("host required");
    }
    let res = send_device_command(&app, host, "stop-stream", "back", "stop").await;
    ok200(&json!({"ok": true, "host": host, "send": res}))
}

// ---------- viewer presence ----------

async fn viewer_ws(
    State(app): State<crate::App>,
    ws: WebSocketUpgrade,
    req: Request<Body>,
) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let viewer_id = qval(&q, "id");
    let vid = if viewer_id.is_empty() {
        crate::api_common::gen_id()
    } else {
        viewer_id.chars().take(20).collect()
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    // 去重: 同 vid 重连替换旧连接 (CF: 序列化 attachment 去重)
    {
        let mut v = VIEWERS.write().unwrap();
        if let Some(old) = v.get(&vid) {
            let _ = old.send(json!({"type": "replaced"}));
        }
        v.insert(vid.clone(), tx);
    }
    // 0→1 翻转 → 通知 (握手前同步完成, 与 CF 连接建立时通知一致)
    if !VIEWERS_ONLINE.swap(true, Ordering::Relaxed) {
        let app3 = app.clone();
        tokio::spawn(async move {
            broadcast_viewer_event(&app3, "viewer-online").await;
        });
    }
    ws.on_upgrade(move |socket| async move {
        let mut sock = socket;
            // 心跳循环 (僵尸踢除: last-ping 超 120s → 断开, CF PresenceHub alarm 兜底 1:1)
            let mut last_ping = std::sync::atomic::AtomicI64::new(now_ms());
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(30));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        if now_ms() - last_ping.load(Ordering::Relaxed) > 120_000 {
                            break; // 僵尸: 强制断
                        }
                    },
                    msg = sock.recv() => match msg {
                        Some(Ok(m)) => {
                            if matches!(m, Message::Close(_)) { break; }
                            if let Ok(text) = m.to_text() {
                                if let Ok(m2) = serde_json::from_str::<Value>(text) {
                                    if m2.get("type").and_then(|x| x.as_str()) == Some("ping") {
                                        last_ping.store(now_ms(), Ordering::Relaxed);
                                        let _ = sock.send(Message::Text(
                                            serde_json::to_string(&json!({"type": "pong", "ts": now_ms()})).unwrap().into()
                                        )).await;
                                    }
                                }
                            }
                        }
                        _ => break,
                    },
                    maybe_v = rx.recv() => match maybe_v {
                        Some(v) => {
                            if let Ok(s) = serde_json::to_string(&v) {
                                if sock.send(Message::Text(s.into())).await.is_err() {
                                    break;
                                }
                            }
                        }
                        None => break,
                    }
                }
            }
            // close → N→0 翻转 → 通知
            let mut v = VIEWERS.write().unwrap();
            v.remove(&vid);
            if v.is_empty() {
                if VIEWERS_ONLINE.swap(false, Ordering::Relaxed) {
                    drop(v);
                    let app3 = app.clone();
                    tokio::spawn(async move {
                        broadcast_viewer_event(&app3, "viewer-offline").await;
                    });
                }
            }
    })
}

/// 0↔N 翻转 → 推 viewer-online/offline 给所有注册设备 (CF: PresenceHub.notifyDevices)
async fn broadcast_viewer_event(app: &crate::App, event: &str) {
    let hosts: Vec<String> = {
        let Ok(c) = get_db(&app.cfg) else {
            return;
        };
        c.prepare("SELECT DISTINCT host FROM metrics ORDER BY ts DESC LIMIT 50")
            .and_then(|mut st| {
                st.query_map([], |r| r.get::<_, String>(0))
                    .map(|it| it.filter_map(|x| x.ok()).collect())
            })
            .unwrap_or_default()
    };
    for host in hosts {
        send_device_command(app, &host, event, "back", "").await;
    }
}

async fn viewers_get(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let n = VIEWERS.read().unwrap().len();
    if qval(&q, "debug") == "1" {
        return ok200(&json!({"sockets": n, "stored_online": VIEWERS_ONLINE.load(Ordering::Relaxed)}));
    }
    ok200(&json!({"viewers": n}))
}

// ---------- phone command (settings 表轮询, CF 1:1) ----------

async fn phone_cmd_post(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let cmd = body.get("command").and_then(|x| x.as_str()).unwrap_or("");
    if cmd != "photo" && cmd != "ring" {
        return bad_request("command must be photo|ring");
    }
    let payload = json!({"command": cmd, "ts": now_ms()});
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let _ = c.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('phone_command', ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        rusqlite::params![serde_json::to_string(&payload).unwrap_or_default(), now_iso()],
    );
    // 推模式: 同时经命令通道推给手机 (CF 原版只写 settings 等轮询; Rust 双通道更实时)
    let host = body.get("host").and_then(|x| x.as_str()).unwrap_or("");
    let mut out = json!({"ok": true, "command": cmd});
    if !host.is_empty() {
        let cam = if cmd == "photo" { "back" } else { "back" };
        let res = send_device_command(&app, host, cmd, cam, "").await;
        if let Some(o) = out.as_object_mut() {
            o.insert("push".into(), res);
        }
    }
    ok200(&out)
}

async fn phone_cmd_get(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let _ = req;
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let mut cmd = Value::Null;
    if let Some(v) = c.query_row("SELECT value FROM settings WHERE key = 'phone_command'", [], |r| r.get::<_, Option<String>>(0)).ok().flatten() {
        if let Ok(r) = serde_json::from_str::<Value>(&v) {
            let ts = r.get("ts").and_then(|x| x.as_i64()).unwrap_or(0);
            if now_ms() - ts < 60000 {
                cmd = r;
            }
        }
    }
    ok200(&json!({"command": cmd}))
}

async fn phone_cmd_ack(State(app): State<crate::App>, req: Request<Body>) -> Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
        let body: Value = match axum::body::to_bytes(req.into_body(), 64 * 1024)
        .await
    {
        Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let cmd = body.get("command").and_then(|x| x.as_str()).unwrap_or("");
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    if let Some(v) = c.query_row("SELECT value FROM settings WHERE key = 'phone_command'", [], |r| r.get::<_, Option<String>>(0)).ok().flatten() {
        if let Ok(r) = serde_json::from_str::<Value>(&v) {
            if r.get("command").and_then(|x| x.as_str()) == Some(cmd) {
                let _ = c.execute("DELETE FROM settings WHERE key = 'phone_command'", []);
            }
        }
    }
    ok200(&json!({"ok": true}))
}
