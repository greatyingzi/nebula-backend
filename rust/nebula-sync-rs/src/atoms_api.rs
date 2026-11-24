//! 原子域 (阶段二): reflux 批量写入 / 去重 / 版本链 / 图谱挂靠 / compose 收集。
//! 1:1 复刻 CF cloud-memory worker 的 /api/atoms 系列。
//!   - POST /api/atoms          幂等批量: ident/content hash 去重 + 版本链 + 引用 upsert + 消失引用清理 + skill 注册实体
//!   - GET  /api/atoms?skill=   原子列表 (skill 过滤走引用表, 全局走 created_at)
//!   - GET  /api/atoms/refs     复用度 (被多个 skill 引用的原子)
//!   - POST /api/atoms/supersede 取代声明 + 引用迁移
//!   - POST /api/atoms/link     原子→图谱边
//!   - GET  /api/atoms/compose  节点 BFS 收集活原子 (skill:/ent:/mem 种子)
use axum::extract::State;
use axum::body::Bytes;
use axum::http::Request;
use axum::body::Body;
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};

use crate::api_common::{bad_request, db_err, get_db, hash16, now_iso, ok200, ok201, rand4};
use crate::{auth_ok, json_resp, qint, qparams, qval};

use std::collections::{HashMap, HashSet};
use rusqlite::{params_from_iter, types::Value as RValue};

pub fn router() -> Router<crate::App> {
    Router::new()
        .route("/api/atoms", get(atoms_get).post(atoms_post))
        .route("/api/atoms/refs", get(atoms_refs))
        .route("/api/atoms/supersede", post(atoms_supersede))
        .route("/api/atoms/link", post(atoms_link))
        .route("/api/atoms/compose", get(atoms_compose))
}

async fn unauthorized() -> axum::response::Response {
    json_resp(401, &json!({"error": "unauthorized"}))
}

// ---------- POST /api/atoms  reflux 幂等批量 ----------

async fn atoms_post(
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
    let skill_id = body.get("skill").and_then(|x| x.as_str()).unwrap_or("").trim();
    let items = match body.get("items").and_then(|x| x.as_array()) {
        Some(a) => a,
        None => return bad_request("skill and items required"),
    };
    let now = now_iso();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    // 上轮引用 (section||position -> atom_id)
    let mut old_by_pos: HashMap<String, String> = HashMap::new();
    {
        let old_rows: Vec<(String, String, i64)> = {
            let mut st = match c.prepare("SELECT atom_id, section, position FROM atom_skills WHERE skill = ?") {
                Ok(s) => s,
                Err(e) => return db_err(&e.to_string()),
            };
            let v = match st.query_map((skill_id,), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?))
            }) {
                Ok(rows) => rows.filter_map(|x| x.ok()).collect(),
                Err(_) => Vec::new(),
            };
            drop(st);
            v
        };
        for row in old_rows {
            old_by_pos.insert(format!("{}||{}", row.1, row.2), row.0);
        }
    }
    // 逐条算 hash (批内去重)
    let mut new_hashes: HashSet<String> = HashSet::new();
    let mut hashed: Vec<(String, &Value)> = Vec::new();
    for it in items.iter().take(2000) {
        let content = it.get("content").and_then(|x| x.as_str()).unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        let ident = it.get("ident").and_then(|x| x.as_str()).unwrap_or("");
        let h = hash16(if ident.is_empty() { content } else { ident });
        if !new_hashes.insert(h.clone()) {
            continue;
        }
        hashed.push((h, it));
    }
    // 批量查已有原子
    let mut exist_map: HashMap<String, String> = HashMap::new();
    if !hashed.is_empty() {
        let hs: Vec<String> = hashed.iter().map(|(h, _)| h.clone()).collect();
        let ph = vec!["?"; hs.len()].join(",");
        if let Ok(mut st) = c.prepare(&format!("SELECT id, content_hash FROM atoms_v2 WHERE content_hash IN ({ph})")) {
            if let Ok(rows) = st.query_map(
                params_from_iter(hs.iter().cloned().map(RValue::Text)),
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            ) {
                for (id, h) in rows.filter_map(|x| x.ok()) {
                    exist_map.insert(h, id);
                }
            }
        }
    }
    let (mut added, mut reused) = (0i64, 0i64);
    let mut seen: Vec<String> = Vec::new();
    let mut sup_batch: Vec<(String, String)> = Vec::new();
    let mut ref_batch: Vec<(String, String, String, i64)> = Vec::new();
    for (h, it) in &hashed {
        let atom_id = match exist_map.get(h) {
            Some(existing) => {
                reused += 1;
                existing.clone()
            }
            None => {
                added += 1;
                let id = format!("atom:{}-{}", h, rand4());
                let kind = it.get("kind").and_then(|x| x.as_str()).unwrap_or("note");
                let title = it.get("title").and_then(|x| x.as_str()).unwrap_or("");
                let content = it.get("content").and_then(|x| x.as_str()).unwrap_or("").trim();
                if let Err(e) = c.execute(
                    "INSERT OR IGNORE INTO atoms_v2 (id, content_hash, kind, title, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (id.as_str(), h.as_str(), kind, title, content, now.as_str(), now.as_str()),
                ) {
                    return db_err(&e.to_string());
                }
                exist_map.insert(h.clone(), id.clone());
                id
            }
        };
        let section = it.get("section").and_then(|x| x.as_str()).unwrap_or("");
        let section = section.chars().take(120).collect::<String>();
        let position = it.get("position").and_then(|x| x.as_i64()).unwrap_or(0);
        let pos_key = format!("{}||{}", section, position);
        if let Some(old) = old_by_pos.get(&pos_key) {
            if old != &atom_id {
                sup_batch.push((old.clone(), atom_id.clone()));
            }
        }
        ref_batch.push((atom_id.clone(), skill_id.to_string(), section, position));
        seen.push(atom_id);
    }
    let _ = c.execute_batch("BEGIN");
    for (old_id, new_id) in &sup_batch {
        let _ = c.execute(
            "INSERT INTO atom_supersedes (old_id, new_id, reason, created_at) VALUES (?, ?, 'content', ?)
             ON CONFLICT(old_id) DO UPDATE SET new_id = excluded.new_id, reason = excluded.reason",
            (old_id.as_str(), new_id.as_str(), now.as_str()),
        );
    }
    for (atom_id, skill, section, position) in &ref_batch {
        let _ = c.execute(
            "INSERT INTO atom_skills (atom_id, skill, section, position, updated_at) VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(atom_id, skill) DO UPDATE SET section = excluded.section, position = excluded.position, updated_at = excluded.updated_at",
            (atom_id.as_str(), skill.as_str(), section.as_str(), *position, now.as_str()),
        );
    }
    // 消失引用清理
    let seen_set: HashSet<&str> = seen.iter().map(|s| s.as_str()).collect();
    let mut dereferenced = 0i64;
    {
        let old_ids: Vec<String> = {
            let mut st = c.prepare("SELECT atom_id FROM atom_skills WHERE skill = ?")
                .expect("prepare atom_skills query");
            let v = match st.query_map((skill_id,), |r| r.get::<_, String>(0)) {
                Ok(rows) => rows.filter_map(|x| x.ok()).collect(),
                Err(_) => Vec::new(),
            };
            drop(st);
            v
        };
        for a in old_ids {
            if !seen_set.contains(a.as_str()) {
                if let Ok(n) = c.execute("DELETE FROM atom_skills WHERE atom_id = ? AND skill = ?", (a.as_str(), skill_id)) {
                    if n > 0 {
                        dereferenced += 1;
                    }
                }
            }
        }
    }
    // skill 注册为实体
    let skill_name = skill_id.split('/').next().unwrap_or(skill_id).to_string();
    let desc = body.get("description").and_then(|x| x.as_str()).unwrap_or("");
    let _ = c.execute(
        "INSERT INTO entities (id, name, kind, description) VALUES (?, ?, 'skill', ?)
         ON CONFLICT(id) DO UPDATE SET name = excluded.name",
        (format!("skill:{skill_id}").as_str(), skill_name.as_str(), desc),
    );
    let _ = c.execute_batch("COMMIT");
    ok201(&json!({"ok": true, "added": added, "reused": reused, "dereferenced": dereferenced}))
}

// ---------- GET /api/atoms?skill=&offset=&limit= ----------

async fn atoms_get(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return unauthorized().await;
    }
    let q = qparams(&req);
    let skill = qval(&q, "skill").trim().to_string();
    let offset = crate::qint(&q, "offset", 0).max(0);
    let limit = crate::qint(&q, "limit", 500).clamp(1, 1000) as usize;
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let rows: Vec<Value> = if skill.is_empty() {
        let sql = "SELECT id, content_hash, kind, title, content, created_at, updated_at FROM atoms_v2 ORDER BY created_at LIMIT ? OFFSET ?";
        let mut st = match c.prepare(sql) {
            Ok(s) => s,
            Err(e) => return db_err(&e.to_string()),
        };
        let r = match st.query_map((limit as i64, offset), |r| atom_json(r)) {
            Ok(rows) => rows.filter_map(|x| x.ok()).collect::<Vec<Value>>(),
            Err(e) => return db_err(&e.to_string()),
        };
        drop(st);
        r
    } else {
        let sql = "SELECT a.id, a.content_hash, a.kind, a.title, a.content, a.created_at, a.updated_at, s.section, s.position
                   FROM atoms_v2 a JOIN atom_skills s ON s.atom_id = a.id
                   WHERE s.skill = ? ORDER BY s.section, s.position LIMIT ? OFFSET ?";
        let mut st = match c.prepare(sql) {
            Ok(s) => s,
            Err(e) => return db_err(&e.to_string()),
        };
        let r = match st.query_map((skill.as_str(), limit as i64, offset), |r| {
            let mut v = atom_json(r)?;
            if let (Some(sec), Some(pos)) = (r.get::<_, String>(7).ok(), r.get::<_, i64>(8).ok()) {
                if let Some(o) = v.as_object_mut() {
                    o.insert("section".into(), sec.into());
                    o.insert("position".into(), pos.into());
                }
            }
            Ok(v)
        }) {
            Ok(rows) => rows.filter_map(|x| x.ok()).collect::<Vec<Value>>(),
            Err(e) => return db_err(&e.to_string()),
        };
        drop(st);
        r
    };
    ok200(&json!({"ok": true, "items": rows}))
}

fn atom_json(r: &rusqlite::Row<'_>) -> Result<Value, rusqlite::Error> {
    Ok(json!({
        "id": r.get::<_, String>(0)?,
        "content_hash": r.get::<_, String>(1)?,
        "kind": r.get::<_, String>(2)?,
        "title": r.get::<_, String>(3)?,
        "content": r.get::<_, String>(4)?,
        "created_at": r.get::<_, String>(5)?,
        "updated_at": r.get::<_, String>(6)?,
    }))
}

// ---------- GET /api/atoms/refs ----------

async fn atoms_refs(
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
    let sql = "SELECT a.content_hash, COUNT(DISTINCT s.skill), GROUP_CONCAT(DISTINCT s.skill), MAX(a.title)
               FROM atoms_v2 a JOIN atom_skills s ON s.atom_id = a.id
               GROUP BY a.id HAVING COUNT(DISTINCT s.skill) > 1
               ORDER BY COUNT(DISTINCT s.skill) DESC LIMIT 100";
    let mut st = match c.prepare(sql) {
        Ok(s) => s,
        Err(e) => return db_err(&e.to_string()),
    };
    let rows = match st.query_map([], |r| {
        Ok(json!({
            "content_hash": r.get::<_, String>(0)?,
            "n_skills": r.get::<_, i64>(1)?,
            "skills": r.get::<_, Option<String>>(2)?,
            "title": r.get::<_, Option<String>>(3)?,
        }))
    }) {
        Ok(rows) => rows.filter_map(|x| x.ok()).collect::<Vec<_>>(),
        Err(e) => return db_err(&e.to_string()),
    };
    ok200(&json!({"ok": true, "items": rows}))
}

// ---------- POST /api/atoms/supersede ----------

async fn atoms_supersede(
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
    let old = body.get("old").and_then(|x| x.as_str()).unwrap_or("").trim();
    let nw = body.get("new").and_then(|x| x.as_str()).unwrap_or("").trim();
    if old.is_empty() || nw.is_empty() || old == nw {
        return bad_request("old and new required");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let resolve = |q: &str| -> Result<String, String> {
        if q.contains('-') && q.len() > 25 {
            Ok(q.to_string())
        } else {
            let pat = format!("atom:{}", q.trim_start_matches("atom:"));
            c.query_row("SELECT id FROM atoms_v2 WHERE id LIKE ? LIMIT 1", (format!("{pat}%"),), |r| r.get::<_, String>(0))
                .map_err(|e| e.to_string())
        }
    };
    let old_id = match resolve(old) {
        Ok(x) => x,
        Err(_) => return bad_request("atom not found"),
    };
    let new_id = match resolve(nw) {
        Ok(x) => x,
        Err(_) => return bad_request("atom not found"),
    };
    let now = now_iso();
    let reason = body.get("reason").and_then(|x| x.as_str()).unwrap_or("merge");
    let _ = c.execute(
        "INSERT INTO atom_supersedes (old_id, new_id, reason, created_at) VALUES (?, ?, ?, ?)
         ON CONFLICT(old_id) DO UPDATE SET new_id = excluded.new_id, reason = excluded.reason",
        (old_id.as_str(), new_id.as_str(), reason, now.as_str()),
    );
    // 引用迁移: old 的引用转给 new
    let _ = c.execute(
        "INSERT INTO atom_skills (atom_id, skill, section, position, updated_at)
         SELECT ?, skill, section, position, ? FROM atom_skills WHERE atom_id = ?
         ON CONFLICT(atom_id, skill) DO UPDATE SET section = excluded.section, position = excluded.position",
        (new_id.as_str(), now.as_str(), old_id.as_str()),
    );
    let _ = c.execute("DELETE FROM atom_skills WHERE atom_id = ?", (old_id.as_str(),));
    ok200(&json!({"ok": true, "old": old_id, "new": new_id}))
}

// ---------- POST /api/atoms/link ----------

async fn atoms_link(
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
    const VALID: [&str; 5] = ["pitfall_of", "related_to", "depends_on", "part_of", "same_as"];
    let now = now_iso();
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let (mut added, mut skipped) = (0i64, 0i64);
    for e in items.iter().take(1000) {
        let atom = e.get("atom_id").and_then(|x| x.as_str()).unwrap_or("").trim();
        let rel = e.get("rel").and_then(|x| x.as_str()).unwrap_or("").trim();
        let entity = e.get("entity").and_then(|x| x.as_str()).unwrap_or("").trim();
        let note = e.get("note").and_then(|x| x.as_str()).unwrap_or("");
        if atom.is_empty() || entity.is_empty() || !VALID.contains(&rel) || atom == entity {
            skipped += 1;
            continue;
        }
        let atom_id = match c.query_row("SELECT id FROM atoms_v2 WHERE id LIKE ? LIMIT 1", (format!("{atom}%"),), |r| r.get::<_, String>(0)) {
            Ok(x) => x,
            Err(_) => atom.to_string(),
        };
        match c.execute(
            "INSERT INTO edges (from_id, rel, to_id, note, created_at) VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(from_id, rel, to_id) DO UPDATE SET note = excluded.note",
            (atom_id.as_str(), rel, entity, note, now.as_str()),
        ) {
            Ok(_) => added += 1,
            Err(_) => skipped += 1,
        }
    }
    ok201(&json!({"ok": true, "added": added, "skipped": skipped}))
}

// ---------- GET /api/atoms/compose?node=&hops= ----------

async fn atoms_compose(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return unauthorized().await;
    }
    let q = qparams(&req);
    let node = qval(&q, "node").trim().to_string();
    let hops = qint(&q, "hops", 1).clamp(1, 2);
    if node.is_empty() {
        return bad_request("node required");
    }
    let c = match get_db(&app.cfg) {
        Ok(c) => c,
        Err(r) => return r,
    };
    // 种子
    let mut visited: HashSet<String> = HashSet::new();
    let mut frontier: Vec<String> = Vec::new();
    let mut seed = |v: &mut Vec<String>, n: &str, vis: &mut HashSet<String>| {
        if !vis.contains(n) {
            vis.insert(n.to_string());
            v.push(n.to_string());
        }
    };
    if let Some(s) = node.strip_prefix("skill:") {
        if let Ok(mut st) = c.prepare("SELECT atom_id FROM atom_skills WHERE skill = ?") {
            if let Ok(rows) = st.query_map((s,), |r| r.get::<_, String>(0)) {
                for a in rows.filter_map(|x| x.ok()) {
                    seed(&mut frontier, &a, &mut visited);
                }
            }
        }
    }
    if !node.starts_with("ent:") && !node.starts_with("skill:") && !node.starts_with("atom:") {
        let h = hash16(&format!("mem:{node}:v1"));
        if let Ok(id) = c.query_row("SELECT id FROM atoms_v2 WHERE content_hash = ?", (h,), |r| r.get::<_, String>(0)) {
            seed(&mut frontier, &id, &mut visited);
        }
    }
    // BFS
    for _ in 0..hops {
        let cur = std::mem::take(&mut frontier);
        if cur.is_empty() {
            break;
        }
        let mut next: Vec<String> = Vec::new();
        for chunk_start in (0..cur.len()).step_by(40) {
            let chunk = &cur[chunk_start..(chunk_start + 40).min(cur.len())];
            let ph = vec!["?"; chunk.len()].join(",");
            let sql = format!(
                "SELECT to_id FROM edges WHERE from_id IN ({ph})
                 UNION SELECT from_id FROM edges WHERE to_id IN ({ph})"
            );
            if let Ok(mut st) = c.prepare(&sql) {
                let ps: Vec<RValue> = chunk.iter().flat_map(|x| vec![RValue::Text(x.clone()), RValue::Text(x.clone())]).collect();
                if let Ok(rows) = st.query_map(params_from_iter(ps), |r| r.get::<_, String>(0)) {
                    for n in rows.filter_map(|x| x.ok()) {
                        seed(&mut next, &n, &mut visited);
                    }
                }
            }
        }
        frontier = next;
    }
    // 收集: 活原子 + 实体节点 + 记忆节点
    let mut atoms: Vec<Value> = Vec::new();
    let atom_nodes: Vec<String> = visited.iter().filter(|n| n.starts_with("atom:")).cloned().collect();
    for chunk_start in (0..atom_nodes.len()).step_by(50) {
        let chunk = &atom_nodes[chunk_start..(chunk_start + 50).min(atom_nodes.len())];
        let ph = vec!["?"; chunk.len()].join(",");
        let sql = format!(
            "SELECT id, content_hash, kind, title, content, created_at, updated_at
             FROM atoms_v2 WHERE id IN ({ph}) AND id NOT IN (SELECT old_id FROM atom_supersedes)"
        );
        if let Ok(mut st) = c.prepare(&sql) {
            let ps: Vec<RValue> = chunk.iter().map(|x| RValue::Text(x.clone())).collect();
            if let Ok(rows) = st.query_map(params_from_iter(ps), |r| atom_json(r)) {
                atoms.extend(rows.filter_map(|x| x.ok()));
            }
        }
    }
    let mut nodes: HashMap<String, Value> = HashMap::new();
    for n in visited.iter() {
        if let Some(id) = n.strip_prefix("ent:") {
            if let Ok(r) = c.query_row(
                "SELECT id, name, kind, description FROM entities WHERE id LIKE ?",
                (format!("ent:{id}%"),),
                |r| Ok(json!({
                    "id": r.get::<_, String>(0)?,
                    "type": "entity",
                    "name": r.get::<_, String>(1)?,
                    "kind": r.get::<_, String>(2)?,
                    "description": r.get::<_, String>(3)?,
                })),
            ) {
                nodes.insert(n.clone(), r);
            }
        } else if !n.starts_with("atom:") && !n.starts_with("skill:") {
            if let Ok(r) = c.query_row(
                "SELECT id, content, scope FROM memories WHERE id = ?",
                (n.as_str(),),
                |r| Ok(json!({
                    "id": r.get::<_, String>(0)?,
                    "type": "memory",
                    "content": r.get::<_, String>(1)?,
                    "scope": r.get::<_, String>(2)?,
                })),
            ) {
                nodes.insert(n.clone(), r);
            }
        }
    }
    ok200(&json!({"ok": true, "root": node, "atoms": atoms, "nodes": nodes, "total": atoms.len()}))
}
