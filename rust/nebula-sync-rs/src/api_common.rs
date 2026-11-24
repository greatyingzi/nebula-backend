//! LingHub API 共享工具 (阶段二)。
//! 各业务域模块 (metrics_api / atoms_api / graph_api / ws_api) 共用的小工具,
//! 行为与 CF worker 对齐:
//!   - 时间:  ISO8601 UTC 字符串 (new Date().toISOString() 格式)
//!   - hash:  sha256 前 16 hex (crypto.subtle.digest 对齐)
//!   - id:    短随机 id / uuid
//!   - CORS:  CF worker 对 /api/* 全开 (Access-Control-Allow-Origin: *)
use axum::response::Response;
use serde_json::{json, Value};

use crate::db;
use crate::config::Config;

/// ISO8601 UTC 秒级 (CF: new Date().toISOString(); 消费方按字符串比较, 秒级够用)
pub fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as u64;
    let days = secs / 86400;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil-from-days (Howard Hinnant)
    let z = days as i64 + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// "2026-09-13T10:41:47Z" -> epoch ms (CF: new Date(ts).getTime())
pub fn parse_iso_ms(s: &str) -> i64 {
    let b = s.as_bytes();
    let get = |a: usize, l: usize| -> i64 {
        b.get(a..a + l)
            .and_then(|x| std::str::from_utf8(x).ok())
            .and_then(|x| x.parse().ok())
            .unwrap_or(0)
    };
    let (y, mo, d) = (get(0, 4), get(5, 2), get(8, 2));
    let (h, m, sec) = (get(11, 2), get(14, 2), get(17, 2));
    let (mut yy, mut mm) = (y, mo);
    if mm <= 2 {
        yy -= 1;
        mm += 12;
    }
    let era = (if yy >= 0 { yy } else { yy - 399 }) / 400;
    let yoe = yy - era * 400;
    let doy = (153 * (mm - 3) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    days * 86400000 + (h * 3600 + m * 60 + sec) * 1000
}

/// sha256 前 8 字节 = 16 hex (与 CF crypto.subtle 对齐)
pub fn hash16(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let h = Sha256::digest(s.as_bytes());
    h.iter().take(8).map(|b| format!("{:02x}", b)).collect()
}

/// 4 位随机字母 (CF: Math.random().toString(36).slice(2,6))
static RAND_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x9E3779B97F4A7C15);
pub fn rand4() -> String {
    use std::sync::atomic::Ordering;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let x = RAND_SEED
        .fetch_add(t.wrapping_add(1), Ordering::Relaxed)
        .wrapping_mul(6364136223846793005)
        ^ t;
    let mut out = String::new();
    let mut v = x;
    for _ in 0..4 {
        out.push((b'a' + (v % 26) as u8) as char);
        v = v / 26 + 1;
    }
    out
}

/// 命令 id (CF: crypto.randomUUID(); 消费方只要求唯一)
pub fn cmd_id() -> String {
    format!("cmd-{}-{}", now_ms(), rand4())
}

/// 短 id (CF: genId())
pub fn gen_id() -> String {
    format!("m{}{}", now_ms(), rand4())
}

/// epoch 秒 → "YYYY-MM-DD" (CF: d.toISOString().slice(0,10))
pub fn iso_date(epoch_s: i64) -> String {
    let days = (epoch_s / 86400).max(0);
    let z = days as i64 + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, mo, d)
}

// ---------- 响应助手 ----------

fn set_cors(r: &mut Response) {
    let h = r.headers_mut();
    h.insert(
        "Access-Control-Allow-Origin",
        axum::http::HeaderValue::from_static("*"),
    );
    h.insert(
        "Access-Control-Allow-Methods",
        axum::http::HeaderValue::from_static("GET,POST,PUT,DELETE,OPTIONS"),
    );
    h.insert(
        "Access-Control-Allow-Headers",
        axum::http::HeaderValue::from_static("Content-Type,Authorization"),
    );
}

/// OPTIONS 预检响应 (CF: 204 + CORS 头)
pub async fn cors_preflight() -> Response {
    let mut r = Response::new(axum::body::Body::empty());
    *r.status_mut() = axum::http::StatusCode::NO_CONTENT;
    set_cors(&mut r);
    r
}

pub fn bad_request(msg: &str) -> Response {
    let mut r = crate::json_resp(400, &json!({"error": msg}));
    set_cors(&mut r);
    r
}

pub fn ok200(v: &Value) -> Response {
    let mut r = crate::json_resp(200, v);
    set_cors(&mut r);
    r
}

pub fn ok201(v: &Value) -> Response {
    let mut r = crate::json_resp(201, v);
    set_cors(&mut r);
    r
}

pub fn db_err(e: &str) -> Response {
    let mut r = crate::json_resp(500, &json!({"error": e}));
    set_cors(&mut r);
    r
}

pub fn get_db(cfg: &Config) -> Result<rusqlite::Connection, Response> {
    db::db(cfg).map_err(|e| db_err(&e))
}

/// 登录会话 cookie 验证 (阶段四: 核心 auth_ok 三通道之一, 对齐 CF checkAuth)。
/// session 格式: b64({"exp":ms}).hmac_sha256_b64(payload, secret)
pub fn verify_session_token(token: &str, secret: &str) -> bool {
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
    v.get("exp").and_then(|x| x.as_i64()).unwrap_or(0) > now_ms()
}

/// 从 Cookie 头取 lh_session 并验证 (secret 为空 → false)
pub fn cookie_session_ok(cookie_hdr: Option<&str>, secret: &str) -> bool {
    if secret.is_empty() {
        return false;
    }
    let Some(h) = cookie_hdr else { return false };
    let mut s = String::new();
    for part in h.split(';') {
        if let Some((k, v)) = part.trim().split_once('=') {
            if k.trim() == "lh_session" {
                s = v.trim().to_string();
            }
        }
    }
    if s.is_empty() {
        return false;
    }
    verify_session_token(&s, secret)
}

fn hmac_sha256_b64(data: &str, secret: &str) -> String {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("hmac key any len");
    mac.update(data.as_bytes());
    b64_no_pad(&mac.finalize().into_bytes())
}

fn b64_no_pad(s: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD_NO_PAD.encode(s)
}

fn b64_decode_lenient(s: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    let e = base64::engine::general_purpose::STANDARD_NO_PAD;
    // 补 padding 兼容
    let mut ss = s.to_string();
    while ss.len() % 4 != 0 {
        ss.push('=');
    }
    e.decode(ss).ok()
}
