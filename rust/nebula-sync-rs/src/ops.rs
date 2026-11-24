use rusqlite::{params, params_from_iter, Connection};
use rusqlite::types::Value as RValue;
use serde_json::{json, Value};
use std::collections::HashSet;

use crate::config::{iso_now, Config};
use crate::db::db;
use crate::embed::embed_one;
use crate::vector_store::VectorStore;

pub const SCOPE_PREFIXES: [&str; 2] = ["memories", "mem"];

// ---------- text helpers ----------

/// character bigram overlap — cheap text-duplication detector (1:1 Python port).
pub fn bigram_jaccard(a: &str, b: &str) -> f64 {
    let sa: HashSet<String> = a
        .chars()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|w| w.iter().collect::<String>())
        .collect();
    let sb: HashSet<String> = b
        .chars()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|w| w.iter().collect::<String>())
        .collect();
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    inter / union
}

// ---- bigram tokenizer (1:1 of session_rag.rs::bigram_tokens = 9/20 proven recipe) ----
// CJK bigrams + ascii words. Used for BOTH the query string and the content
// string written into the contentless unicode61 FTS table (via the nebula_bigram
// UDF in db.rs). 9/23: replaces the old 4-char-window fts_tokens, which covered
// only 2/9 of the golden recall misses vs 7/9 for bigram (fts_speed_test/hyp test).
pub fn mem_bigram(q: &str) -> Vec<String> {
    let low = q.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in low.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    let cjk: Vec<char> = low
        .chars()
        .filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
        .collect();
    for i in 0..cjk.len().saturating_sub(1) {
        out.push(format!("{}{}", cjk[i], cjk[i + 1]));
    }
    if let Some(c0) = cjk.first() {
        out.push(c0.to_string());
    }
    out
}

/// Space-joined bigram string consumed by the contentless FTS5 table. Triggers
/// call this via the nebula_bigram UDF. MUST stay 1:1 with mem_bigram so the
// index token stream == the query token stream (both derive from mem_bigram).
pub fn mem_bigram_str(q: &str) -> String {
    mem_bigram(q).join(" ")
}

/// FTS5 leg: bigram OR-combination over the contentless unicode61 index,
/// ranked by bm25 (9/20 proven recipe, 1:1 of session_rag.rs::fts_leg).
/// FTS5 `rank` is NEGATIVE (more relevant = more negative), so `ORDER BY rank`
/// ascending is best-first. Returns (rowid, rank_position) where position 0 = best
/// (replaces the old (rowid, match-count) — the position feeds RRF identically).
pub fn fts_search(
    c: &Connection,
    query: &str,
    limit: usize,
    _scope: &str,
) -> Vec<(i64, i64)> {
    let toks = mem_bigram(query);
    if toks.is_empty() {
        return Vec::new();
    }
    let q: String = toks
        .iter()
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ");
    let sql =
        "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?";
    let limit_n = (limit * 2).max(10) as i64;
    let mut out = Vec::new();
    // Note: keep the Statement in a let-binding (not inside an and_then closure)
    // so the Rows borrow is valid for the whole loop (E0515: a Rows returned from
    // s.query borrows the Statement, which must outlive the loop).
    let mut stmt = match c.prepare(sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[fts] search prepare err: {e}");
            return out;
        }
    };
    let mut rr = match stmt.query(rusqlite::params![q, limit_n]) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[fts] search query err: {e}");
            return out;
        }
    };
    let mut pos: i64 = 0;
    while let Ok(Some(row)) = rr.next() {
        let rid: i64 = row.get::<_, i64>(0).unwrap_or(0);
        out.push((rid, pos));
        pos += 1;
    }
    out
}

/// RRF merge of ranked lists (1:1 port of `_rrf_merge`).
pub fn rrf_merge(ranked_lists: &[Vec<Value>], k: i64) -> Vec<Value> {
    let mut scores: std::collections::HashMap<String, f64> = Default::default();
    let mut meta: std::collections::HashMap<String, Value> = Default::default();
    for lst in ranked_lists {
        for (rank, it) in lst.iter().enumerate() {
            let mid = it.get("id").and_then(|x| x.as_str()).unwrap_or_default();
            *scores.entry(mid.to_string()).or_insert(0.0) += 1.0 / (k + rank as i64 + 1) as f64;
            let mut prev = meta
                .get(mid)
                .cloned()
                .unwrap_or_else(|| json!({}) as Value);
            let obj = prev.as_object_mut().unwrap();
            let src = it
                .get("matched_by")
                .and_then(|x| x.as_str())
                .unwrap_or("?");
            let sources = obj
                .get("sources")
                .and_then(|x| x.as_array())
                .cloned()
                .unwrap_or_default();
            let mut new_sources = sources;
            new_sources.push(json!(src));
            obj.insert("sources".into(), Value::Array(new_sources));
            let rs = obj.get("rank_sum").and_then(|x| x.as_i64()).unwrap_or(0);
            obj.insert("rank_sum".into(), json!(rs + rank as i64));
            let sc = it.get("score").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let prev_sc = obj.get("score").and_then(|x| x.as_f64()).unwrap_or(0.0);
            obj.insert("score".into(), json!(prev_sc.max(sc)));
            // 对齐 py `_rrf_merge` 的整组填充语义: 用 content 作哨兵, 仅当 prev
            // 首次见到该 mid(尚无 content)时才批量填 content/scope/tags/rowid(只填
            // it 里有的)。逐字段填会导致双源项(第一路无 rowid + 第二路有 rowid)在
            // 第二路被补上 rowid, 与 py 不同(q=nebula 的 rowid missing 即由此)。
            if !obj.contains_key("content") {
                for f in ["content", "scope", "tags", "rowid"] {
                    if let Some(v) = it.get(f) {
                        obj.insert(f.into(), v.clone());
                    }
                }
            }
            meta.insert(mid.to_string(), prev);
        }
    }
    let mut merged: Vec<Value> = scores
        .iter()
        .map(|(mid, s)| {
            let mut m = json!({ "id": mid, "rrf": s });
            if let Some(mo) = m.as_object_mut() {
                if let Some(p) = meta.get(mid) {
                    if let Some(po) = p.as_object() {
                        for (k, v) in po {
                            mo.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
            m
        })
        .collect();
    merged.sort_by(|a, b| {
        let ra = a.get("rrf").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let rb = b.get("rrf").and_then(|x| x.as_f64()).unwrap_or(0.0);
        // 稳定 tie-break: RRF 分数相等(常见, 如 1/(k+0) 平局)时按 id 排。
        // Python 侧 _rrf_merge 同步加同一 tie-break → 两端平局落到同序,
        // 消除 dict/HashMap 插入序差异导致的 top-k 顺序翻转。
        match rb.partial_cmp(&ra) {
            Some(std::cmp::Ordering::Equal) => {
                a.get("id").and_then(|x| x.as_str()).unwrap_or("")
                    .cmp(b.get("id").and_then(|x| x.as_str()).unwrap_or(""))
            }
            Some(o) => o,
            None => std::cmp::Ordering::Equal,
        }
    });
    merged
}

// ---------- tags quirks (faithful to Python) ----------

fn tags_json_dumps(v: Option<&Value>) -> String {
    // Python: json.dumps(tags) — strings get quoted, lists dumped
    match v {
        None => "null".to_string(),
        Some(Value::Null) => "null".to_string(),
        Some(s) if s.is_string() => s.to_string(), // serde already quotes
        Some(s) => s.to_string(),
    }
}

fn tags_raw_or_dumps(v: Option<&Value>) -> String {
    // Python: json.dumps(tags) if isinstance(tags, list) else tags
    match v {
        None => "".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(s) => s.to_string(),
    }
}

// ---------- ops ----------

pub fn invalidate_search_cache() {
    crate::search_cache_clear();
}

/// Apply one op (1:1 port of Python `apply_op`).
pub fn apply_op(cfg: &Config, vs: &std::sync::Arc<VectorStore>, op: &Value) -> bool {
    let o = op.get("op").and_then(|x| x.as_str()).unwrap_or_default();
    let p_raw = op.get("payload").cloned().unwrap_or(json!({}));
    let p = if p_raw.is_string() {
        serde_json::from_str(p_raw.as_str().unwrap())
            .unwrap_or_else(|_| json!({}))
    } else {
        p_raw
    };
    let ts = op
        .get("ts")
        .and_then(|x| x.as_f64())
        .filter(|t| *t > 0.0)
        .unwrap_or_else(now_f64);
    let oid = op.get("id").and_then(|x| x.as_str()).unwrap_or_default();
    let now = iso_now();
    let c = match db(cfg) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[op] db err: {e}");
            return false;
        }
    };
    let mut p = p;
    let mut ok = true;
    match o {
        "mem.add" => {
            let content = p
                .get("content")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let h = sha256_hex(&content);
            let dup: Option<String> = c
                .query_row("SELECT id FROM memories WHERE hash=?", params![h], |r| r.get(0))
                .ok()
                .flatten();
            if let Some(mid) = dup {
                c.execute(
                    "UPDATE memories SET updated_at=?, tags=? WHERE id=?",
                    params![now, tags_json_dumps(p.get("tags")), mid],
                )
                .ok();
            } else {
                // semantic dedup
                let mut merged_into: Option<String> = None;
                let mut dedup_score = 0.0;
                if let Ok(qv) = embed_one(cfg, &content.trunc512()) {
                    for (cand_mid, cand_sc) in vs.search(&qv, cfg.dedup_top_n) {
                        if cand_sc < cfg.dedup_vec_min as f32 {
                            break;
                        }
                        let row_content: Option<String> = c
                            .query_row(
                                "SELECT content FROM memories WHERE id=?",
                                params![cand_mid],
                                |r| r.get(0),
                            )
                            .ok()
                            .flatten();
                        if let Some(row_content) = row_content {
                            let j = bigram_jaccard(&content, &row_content);
                            if j >= cfg.dedup_jac_min {
                                merged_into = Some(cand_mid);
                                dedup_score = j;
                                break;
                            }
                        }
                    }
                } else {
                    // embed fail: skip semantic dedup (same as Python try/except)
                }
                if let Some(mid) = merged_into {
                    let old: Option<String> = c
                        .query_row(
                            "SELECT content FROM memories WHERE id=?",
                            params![mid],
                            |r| r.get(0),
                        )
                        .ok()
                        .flatten();
                    c.execute(
                        "UPDATE memories SET updated_at=?, tags=? WHERE id=?",
                        params![now, tags_json_dumps(p.get("tags")), mid],
                    )
                    .ok();
                    if let Some(oldc) = old {
                        if content.chars().count() as f64 > oldc.chars().count() as f64 * 1.15 {
                            let nh = sha256_hex(&content);
                            c.execute(
                                "UPDATE memories SET content=?, hash=?, updated_at=? WHERE id=?",
                                params![content, nh, now, mid],
                            )
                            .ok();
                            vs.remove(cfg, &mid);
                            vs.add(&mid, &content);
                        }
                    }
                    if let Some(po) = p.as_object_mut() {
                        po.insert(
                            "_dedup_merged_into".into(),
                            json!(mid),
                        );
                        po.insert(
                            "_dedup_jaccard".into(),
                            json!((dedup_score * 1000.0).round() / 1000.0),
                        );
                    }
                } else {
                    let mid = p
                        .get("id")
                        .and_then(|x| x.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("mem-{}", &h[..16]));
                    let tags = tags_raw_or_dumps(p.get("tags"));
                    c.execute(
                        "INSERT OR REPLACE INTO memories(id,content,scope,tags,hash,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                        params![
                            mid,
                            content,
                            p.get("scope").and_then(|x| x.as_str()).unwrap_or("general"),
                            tags,
                            h,
                            now,
                            now
                        ],
                    )
                    .ok();
                    vs.remove(cfg, &mid);
                    vs.add(&mid, &content);
                }
            }
        }
        "mem.update" => {
            let content = p
                .get("content")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let h = sha256_hex(&content);
            let tags = tags_raw_or_dumps(p.get("tags"));
            c.execute(
                "UPDATE memories SET content=?,hash=?,scope=?,tags=?,updated_at=? WHERE id=?",
                params![
                    content,
                    h,
                    p.get("scope").and_then(|x| x.as_str()).unwrap_or("general"),
                    tags,
                    now,
                    p.get("id").and_then(|x| x.as_str()).unwrap_or_default()
                ],
            )
            .ok();
            let mid = p
                .get("id")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            vs.remove(cfg, &mid);
            vs.add(&mid, &content);
        }
        "mem.delete" => {
            let mid = p
                .get("id")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            c.execute("DELETE FROM memories WHERE id=?", params![mid]).ok();
            c.execute(
                "DELETE FROM edges WHERE from_id=? OR to_id=?",
                params![mid, mid],
            )
            .ok();
            vs.remove(cfg, &mid);
        }
        "ent.upsert" => {
            c.execute(
                "INSERT OR REPLACE INTO entities(id,name,kind,description) VALUES(?,?,?,?)",
                params![
                    p.get("id").and_then(|x| x.as_str()).unwrap_or_default(),
                    p.get("name").and_then(|x| x.as_str()).unwrap_or(""),
                    p.get("kind").and_then(|x| x.as_str()).unwrap_or("component"),
                    p.get("description").and_then(|x| x.as_str()).unwrap_or("")
                ],
            )
            .ok();
        }
        "edge.upsert" => {
            c.execute(
                "INSERT OR REPLACE INTO edges(from_id,rel,to_id,note,created_at) VALUES(?,?,?,?,?)",
                params![
                    p.get("from").and_then(|x| x.as_str()).unwrap_or_default(),
                    p.get("rel").and_then(|x| x.as_str()).unwrap_or_default(),
                    p.get("to").and_then(|x| x.as_str()).unwrap_or_default(),
                    p.get("note").and_then(|x| x.as_str()).unwrap_or(""),
                    now
                ],
            )
            .ok();
        }
        "edge.delete" => {
            c.execute(
                "DELETE FROM edges WHERE from_id=? AND rel=? AND to_id=?",
                params![
                    p.get("from").and_then(|x| x.as_str()).unwrap_or_default(),
                    p.get("rel").and_then(|x| x.as_str()).unwrap_or_default(),
                    p.get("to").and_then(|x| x.as_str()).unwrap_or_default()
                ],
            )
            .ok();
        }
        _ => ok = false,
    }
    if ok {
        c.execute(
            "INSERT OR IGNORE INTO ops(id,ts,op,entity,payload,source) VALUES(?,?,?,?,?,?)",
            params![
                oid,
                ts,
                o,
                op.get("entity").and_then(|x| x.as_str()).unwrap_or(""),
                p.to_string(),
                op.get("source").and_then(|x| x.as_str()).unwrap_or("")
            ],
        )
        .ok();
        
    }
    drop(c);
    if ok && o.starts_with("mem.") {
        invalidate_search_cache();
    }
    ok
}

pub fn now_f64() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn sha256_hex(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let h = Sha256::digest(s.as_bytes());
    let mut out = String::new();
    for b in h.iter() {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

trait StrExt {
    fn trunc512(&self) -> String;
}
impl StrExt for str {
    fn trunc512(&self) -> String {
        let mut i = 512.min(self.len());
        while i > 0 && !self.is_char_boundary(i) {
            i -= 1;
        }
        self[..i].to_string()
    }
}

// ---------- nebula helpers ----------

pub fn is_scope_skill(s: &str) -> bool {
    SCOPE_PREFIXES
        .iter()
        .any(|p| s == *p || s.starts_with(&format!("{p}/")))
}

pub fn scope_key_of(skill: &str) -> Option<&str> {
    for p in SCOPE_PREFIXES {
        if skill == p {
            return Some("");
        }
        if let Some(rest) = skill.strip_prefix(&format!("{p}/")) {
            return Some(rest);
        }
    }
    None
}

pub fn scope_variants_public(c: &Connection, scope: &str) -> Vec<String> {
    scope_variants(c, scope)
}

fn scope_variants(c: &Connection, scope: &str) -> Vec<String> {
    let mut st = match c.prepare("SELECT skill, COUNT(*) n FROM atom_skills GROUP BY skill") {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let rows: Vec<(String, i64)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .ok()
        .map(|rr| rr.filter_map(|r| r.ok()).collect())
        .unwrap_or_default();
    let mut out: Vec<(String, i64)> = rows
        .into_iter()
        .filter(|(sk, _)| {
            is_scope_skill(sk) && scope_key_of(sk) == Some(scope)
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out.into_iter().map(|(s, _)| s).collect()
}

pub fn compose_skill(cfg: &Config, name: &str) -> Result<Value, String> {
    let c = db(cfg)?;
    let bare = name.rsplit('/').next().unwrap_or(name);
    let mut display = name.to_string();
    let mut skills: Vec<String> = Vec::new();
    let mut auto = false;
    if SCOPE_PREFIXES.contains(&bare) {
        skills = SCOPE_PREFIXES
            .iter()
            .filter(|p| **p == bare || **p == "mem")
            .map(|s| s.to_string())
            .collect();
        display = "mem".into();
        auto = true;
    } else if name == bare && (bare.starts_with("mem/") || bare.starts_with("memories/")) {
        let slash = bare.find('/').unwrap();
        let sk = &bare[slash + 1..];
        skills = scope_variants(&c, sk);
        if skills.is_empty() {
            skills = vec![name.to_string()];
        }
        display = if sk.is_empty() {
            "mem".into()
        } else {
            format!("mem/{sk}")
        };
        auto = true;
    } else {
        skills = vec![name.to_string()];
    }
    // fallback: bare name not found as literal skill → try as scope key
    if skills.len() == 1 && skills[0] == name {
        let cnt: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM atom_skills WHERE skill=?",
                params![name],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if cnt == 0 {
            let variants = scope_variants(&c, name);
            if !variants.is_empty() {
                skills = variants;
                auto = true;
            }
        }
    }
    let marks = vec!["?"; skills.len()].join(",");
    let mut stmt = c
        .prepare(&format!(
            "SELECT a.id,a.kind,a.title,a.content,s.section,s.position FROM atom_skills s JOIN atoms_v2 a ON a.id=s.atom_id \
             WHERE s.skill IN ({marks}) AND a.id NOT IN (SELECT old_id FROM atom_supersedes) \
             ORDER BY s.section, s.position"
        ))
        .map_err(|e| e.to_string())?;
    let refs: Vec<RValue> = skills
        .iter()
        .map(|s| RValue::Text(s.clone()))
        .collect();
    let rows: Vec<(String, String, Option<String>, String, Option<String>, i64)> = stmt
        .query_map(params_from_iter(refs.iter()), |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
            ))
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    let mut md: Vec<String> = Vec::new();
    let mut by: std::collections::HashMap<String, Vec<(String, String, Option<String>, String)>> =
        Default::default();
    let mut order: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for (aid, kind, title, content, sec, _pos) in rows {
        if seen.contains(&aid) {
            continue;
        }
        seen.insert(aid.clone());
        let sec = sec.as_deref().filter(|s| !s.is_empty()).unwrap_or("(top)").to_string();
        if !by.contains_key(&sec) {
            by.insert(sec.clone(), Vec::new());
            order.push(sec.clone());
        }
        by.get_mut(&sec)
            .unwrap()
            .push((aid, kind, title.clone(), content));
    }
    md.push(format!("# Skill: {display}"));
    if auto {
        md.push(format!(
            "\n> auto-composed from scope bucket{}",
            if skills.is_empty() {
                "".into()
            } else {
                format!(" ({})", skills.join(", "))
            }
        ));
    }
    md.push(format!("\n> {} atoms, {} sections", seen.len(), order.len()));
    for sec in &order {
        md.push(format!("\n## {sec}\n"));
        for (aid, kind, title, content) in &by[sec] {
            if let Some(t) = title {
                if t.trim() != sec.trim() {
                    md.push(format!("### {t}\n"));
                }
            }
            md.push(format!("{content}\n"));
            md.push(format!("*_atom: {aid} ({kind})*\n"));
        }
    }
    let variants = if skills.len() > 1 || skills[0] != name {
        Some(Value::Array(
            skills.into_iter().map(|s| json!(s)).collect(),
        ))
    } else {
        None
    };
    Ok(json!({
        "skill": display,
        "auto": auto,
        "variants": variants,
        "markdown": md.join("\n"),
        "atom_count": seen.len(),
        "section_count": order.len(),
    }))
}

pub fn list_skills(cfg: &Config) -> Result<Vec<Value>, String> {
    let c = db(cfg)?;
    let mut st = c
        .prepare("SELECT skill, COUNT(*) atoms, COUNT(DISTINCT section) secs FROM atom_skills GROUP BY skill")
        .map_err(|e| e.to_string())?;
    let rows: Vec<(String, i64, i64)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    // NOTE: Python code does str(secs).split(",") which is a latent bug (secs is int,
    // never a comma list) — faithful port keeps the same observable behavior.
    let mut groups: std::collections::HashMap<String, (i64, std::collections::HashSet<String>, Vec<String>)> =
        Default::default();
    let mut groups_order: Vec<String> = Vec::new(); // first-appearance order (matches Python dict)
    let mut named: Vec<Value> = Vec::new();
    for (skill, n, secs) in rows {
        if is_scope_skill(&skill) {
            let sk = scope_key_of(&skill).unwrap_or("").to_string();
            if !groups.contains_key(&sk) {
                groups_order.push(sk.clone());
            }
            let g = groups
                .entry(sk)
                .or_insert_with(|| (0, Default::default(), Vec::new()));
            g.0 += n;
            let secs_list: Vec<String> = secs
                .to_string()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
            g.1.extend(secs_list);
            g.2.push(skill.clone());
        } else {
            named.push(json!({"skill": skill, "atoms": n, "sections": secs, "auto": false}));
        }
    }
    let mut out = named;
    for sk in groups_order {
        let (atoms, secs, variants) = &groups[&sk];
        out.push(json!({
            "skill": if sk.is_empty() { "mem".to_string() } else { format!("mem/{sk}") },
            "atoms": atoms,
            "sections": secs.len(),
            "auto": true,
            "variants": variants,
        }));
    }
    out.sort_by(|a, b| {
        b.get("atoms")
            .and_then(|x| x.as_i64())
            .unwrap_or(0)
            .cmp(&a.get("atoms").and_then(|x| x.as_i64()).unwrap_or(0))
    });
    Ok(out)
}
