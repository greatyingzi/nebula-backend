//! 设备指标域 (阶段二): 各设备采集 / 趋势查询 / 设备聚合。
//! 1:1 复刻 CF linghub worker 的 /api/metrics + /api/devices。
//! 在线判定复用 ws_api 的 WS presence (CF: CommandHub.isConnected()), 失败退回 age_min。
use axum::extract::State;
use axum::http::Request;
use axum::body::{Body, Bytes};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};

use crate::api_common::{bad_request, db_err, get_db, now_iso, ok200, parse_iso_ms, now_ms};
use crate::ws_api;
use crate::{auth_ok, json_resp, qparams, qval};

/// 采集字段 (与 CF /api/metrics 完全一致)
const METRIC_KEYS: [&str; 12] = [
    "cpu", "mem", "temp", "disk_home", "disk_data",
    "gpu0_temp", "gpu0_util", "gpu0_mem", "gpu1_temp", "gpu1_util", "gpu1_mem", "uptime",
];

/// 模块路由表 (main.rs 只做 merge)
pub fn router() -> Router<crate::App> {
    Router::new()
        .route("/api/metrics", post(metrics_post).get(metrics_get))
        .route("/api/devices", get(devices))
}

async fn unauthorized() -> axum::response::Response {
    json_resp(401, &json!({"error": "unauthorized"}))
}

/// POST /api/metrics  设备采集 upsert (host+ts 冲突覆盖)
async fn metrics_post(
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
    let host = body.get("host").and_then(|x| x.as_str()).unwrap_or("").trim();
    let host = host.chars().take(40).collect::<String>();
    if host.is_empty() {
        return bad_request("host required");
    }
    let ts = body
        .get("ts")
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .filter(|t| !t.is_empty())
        .unwrap_or_else(now_iso);
    let num = |k: &str| -> f64 {
        body.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)
    };
    let extra = if body.get("extra").map(|v| v.is_object()).unwrap_or(false) {
        body.get("extra").cloned().map(|v| v.to_string()).unwrap_or_default()
    } else {
        "{}".into()
    };
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let ph = vec!["?"; METRIC_KEYS.len()].join(", ");
    let sets = METRIC_KEYS
        .iter()
        .map(|k| format!("{k} = excluded.{k}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "INSERT INTO metrics (host, ts, {}, extra) VALUES (?, ?, {}, ?)
         ON CONFLICT(host, ts) DO UPDATE SET {}, extra = excluded.extra",
        METRIC_KEYS.join(", "), ph, sets
    );
    let mut params: Vec<rusqlite::types::Value> =
        vec![rusqlite::types::Value::Text(host.clone()), rusqlite::types::Value::Text(ts.clone())];
    for k in METRIC_KEYS {
        params.push(rusqlite::types::Value::Real(num(k)));
    }
    params.push(rusqlite::types::Value::Text(extra));
    if let Err(e) = c.execute(&sql, rusqlite::params_from_iter(params)) {
        return db_err(&e.to_string());
    }
    ok200(&json!({"ok": true, "host": host, "ts": ts}))
}

/// GET /api/metrics?host=&hours=  趋势查询
async fn metrics_get(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return unauthorized().await;
    }
    let q = qparams(&req);
    let host = qval(&q, "host").trim().to_string();
    let hours = crate::qint(&q, "hours", 24).clamp(1, 168);
    if host.is_empty() {
        return bad_request("host required");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let cols = "host, ts, cpu, mem, temp, disk_home, disk_data, gpu0_temp, gpu0_util, gpu0_mem, gpu1_temp, gpu1_util, gpu1_mem, uptime, extra";
    let sql = format!(
        "SELECT {} FROM metrics WHERE host = ? AND ts >= datetime('now', ?) ORDER BY ts ASC",
        cols
    );
    let mut st = match c.prepare(&sql) {
        Ok(s) => s,
        Err(e) => return db_err(&e.to_string()),
    };
    let off = format!("-{} hours", hours);
    let rows = match st.query_map((host.as_str(), off.as_str()), |r| {
        Ok(json!({
            "host": r.get::<_, String>(0)?,
            "ts": r.get::<_, String>(1)?,
            "cpu": r.get::<_, f64>(2)?,
            "mem": r.get::<_, f64>(3)?,
            "temp": r.get::<_, f64>(4)?,
            "disk_home": r.get::<_, f64>(5)?,
            "disk_data": r.get::<_, f64>(6)?,
            "gpu0_temp": r.get::<_, f64>(7)?,
            "gpu0_util": r.get::<_, f64>(8)?,
            "gpu0_mem": r.get::<_, f64>(9)?,
            "gpu1_temp": r.get::<_, f64>(10)?,
            "gpu1_util": r.get::<_, f64>(11)?,
            "gpu1_mem": r.get::<_, f64>(12)?,
            "uptime": r.get::<_, f64>(13)?,
            "extra": r.get::<_, String>(14)?,
        }))
    }) {
        Ok(rows) => rows.filter_map(|x| x.ok()).collect::<Vec<Value>>(),
        Err(e) => return db_err(&e.to_string()),
    };
    ok200(&json!({"host": host, "items": rows}))
}

/// GET /api/devices  各设备最新状态聚合 (CF: 在线=WS存活, 失败退回 age<=10min)
async fn devices(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return unauthorized().await;
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let mut hosts: Vec<String> = Vec::new();
    {
        let mut st = match c.prepare("SELECT DISTINCT host FROM metrics ORDER BY host") {
            Ok(s) => s,
            Err(e) => return db_err(&e.to_string()),
        };
        if let Ok(rows) = st.query_map([], |r| r.get::<_, String>(0)) {
            hosts = rows.filter_map(|x| x.ok()).collect::<Vec<String>>();
        }
        drop(st);
    }
    let now = now_ms();
    let mut out: Vec<Value> = Vec::new();
    for h in &hosts {
        let row = c
            .query_row(
                "SELECT ts, cpu, mem, temp, disk_home, disk_data, gpu0_temp, gpu0_util, gpu0_mem, gpu1_temp, gpu1_util, gpu1_mem, uptime, extra \
                 FROM metrics WHERE host = ? ORDER BY ts DESC LIMIT 1",
                (h.as_str(),),
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, f64>(1)?,
                        r.get::<_, f64>(2)?,
                        r.get::<_, f64>(3)?,
                        r.get::<_, f64>(4)?,
                        r.get::<_, f64>(5)?,
                        r.get::<_, f64>(6)?,
                        r.get::<_, f64>(7)?,
                        r.get::<_, f64>(8)?,
                        r.get::<_, f64>(9)?,
                        r.get::<_, f64>(10)?,
                        r.get::<_, f64>(11)?,
                        r.get::<_, f64>(12)?,
                        r.get::<_, String>(13)?,
                    ))
                },
            )
            .ok();
        let Some(last) = row else { continue };
        let last_ts = parse_iso_ms(&last.0);
        let age_min = ((now.saturating_sub(last_ts)) / 60000) as i64;
        let online = ws_api::device_online(h) || age_min <= 10;
        let extra: Value = serde_json::from_str(&last.13).unwrap_or(json!({}));
        out.push(json!({
            "host": h, "online": online, "age_min": age_min, "ts": last.0,
            "cpu": last.1, "mem": last.2, "temp": last.3,
            "disk_home": last.4, "disk_data": last.5,
            "gpu0_temp": last.6, "gpu0_util": last.7, "gpu0_mem": last.8,
            "gpu1_temp": last.9, "gpu1_util": last.10, "gpu1_mem": last.11,
            "uptime": last.12, "extra": extra,
        }));
    }
    ok200(&json!({"devices": out}))
}
