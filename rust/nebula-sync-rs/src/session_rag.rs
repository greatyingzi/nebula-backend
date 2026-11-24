//! Session RAG 域 — 会话段级检索 (CHUNKING.md DOC-0.3 定稿方案)。
//!
//! 语料: 四端 Hermes + 7 种 Agent (hermes/claude/codex/pi/opencode/crush/kimi),
//!       按会话切段: <3000 字整会话一段; 长会话 10min 时间硬边界 + 3000-8000 字段
//!       + 200 字重叠 + <1200 字小段合并。
//! 索引: doc_vectors (Mac MLX 4bit 产出的 1024 维归一化向量, 内积=余弦)
//!       + docs_fts (FTS5 trigram, 服务端自建) 双路。
//! 查询: 双路各 top50 → RRF(k=cfg.rrf_k) → bge-reranker 精排 top20 → top-k。
//!       实测 39 条标注 query: R@5/R@10/MRR = 1.00/1.00/1.000 (对比: 纯向量 0.95,
//!       消息级切片 0.92, 整会话喂 rerank 反降 0.97 —— 故用段切 + 双路 + 精排)。
//!
//! 端点:
//!   GET  /api/session-rag/search?q=&limit=&rerank=&device=&agent=&session_id=
//!   GET  /api/session-rag/status
//!   GET  /api/session-rag/session?session_id=&limit=
//!   POST /api/session-rag/ingest   (增量写入: {docs:[{doc_id,...,text,vec:[f32]}]})
//!
//! 数据文件: $NEBULA_DATA/session_rag.db — 与主库(nebula.db)/记忆向量库(vectors.db)
//! 物理分离 (沿用既有约定: 避免 WAL 单写者争锁)。docs/docs_fts/doc_vectors 三表。
//! 批量导入走离线文件 (Mac 构建 + scp), 增量走 HTTP ingest。
use axum::body::Body;
use axum::extract::State;
use axum::http::Request;
use axum::routing::{get, post};
use axum::Router;
use once_cell::sync::Lazy;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::api_common::bad_request;
use crate::config::Config;
use crate::embed;
use crate::vector_store::FlatIndex;
use crate::{auth_ok, json_resp, qint, qparams, qval};

const DIM: usize = 1024;
/// 双路召回各取多少候选进 RRF
const LEG_TOPK: usize = 50;
/// 进 rerank 的候选数 (CPU 精排延迟约束)
const RERANK_TOPK: usize = 20;

// ---------- 存储 ----------

pub struct DocStore {
    index: Mutex<FlatIndex>,
    ids: Mutex<Vec<String>>,
}

static SR: Lazy<Arc<DocStore>> = Lazy::new(|| {
    Arc::new(DocStore {
        index: Mutex::new(FlatIndex::new(DIM)),
        ids: Mutex::new(Vec::new()),
    })
});

fn sr_path(cfg: &Config) -> std::path::PathBuf {
    cfg.data.join("session_rag.db")
}

pub fn sr_db(cfg: &Config) -> Result<Connection, String> {
    let c = Connection::open(sr_path(cfg)).map_err(|e| e.to_string())?;
    c.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA busy_timeout=30000;
         CREATE TABLE IF NOT EXISTS docs(
           doc_id TEXT PRIMARY KEY, session_id TEXT, device TEXT, agent TEXT,
           kind TEXT, ts_iso TEXT, char_count INTEGER, anchors TEXT, text TEXT,
           updated_at REAL
         );
         CREATE INDEX IF NOT EXISTS idx_sr_session ON docs(session_id);
         CREATE INDEX IF NOT EXISTS idx_sr_device ON docs(device);
         CREATE INDEX IF NOT EXISTS idx_sr_agent ON docs(agent);
         CREATE TABLE IF NOT EXISTS doc_vectors(
           doc_id TEXT PRIMARY KEY, vec BLOB, updated_at REAL
         );",
    )
    .map_err(|e| e.to_string())?;
    Ok(c)
}

fn f32_to_blob(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn blob_to_f32(b: &[u8]) -> Vec<f32> {
    let n = b.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    v
}

/// 加载向量索引 (启动时一次)。FTS 表按需自建。
pub fn load_or_start(cfg: &Config) {
    if let Err(e) = sr_db(cfg) {
        eprintln!("[sr] db init err: {e}");
        return;
    }
    let c = match sr_db(cfg) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[sr] db open err: {e}");
            return;
        }
    };
    // 体检: 绝不在坏/半截库上跑重活。
    // 事故复盘 (2026-09-20): 140MB 的 session_rag.db 只传了 94MB 时服务被重启,
    // 打开截断文件后又去建 84MB 文本的 trigram FTS 索引, 把 4 核 i5-3210M 拖到
    // 整机无响应 (SSH/HTTP 全挂, 只剩 tailscaled 应答)。这里先做廉价体检:
    // 文件非空 + quick_check 通过 + docs 表可查, 任一不合格就放弃加载 (其余功能照常),
    // 并在日志大声报出 —— FTS 索引现由 Mac 侧预建 (build_srag_db.py), hub 只读。
    let sz = std::fs::metadata(sr_path(cfg)).map(|m| m.len()).unwrap_or(0);
    if sz == 0 {
        eprintln!("[sr] session_rag.db 缺失/为空 → 索引关闭, 等待数据落地");
        return;
    }
    match c.query_row("PRAGMA quick_check(1)", [], |r| r.get::<_, String>(0)) {
        Ok(v) if v == "ok" => {}
        Ok(v) => {
            eprintln!("[sr] ⚠ quick_check 失败({v}) → 拒绝加载, 请重新推送完整库");
            return;
        }
        Err(e) => {
            eprintln!("[sr] ⚠ 库不可读({e}) → 拒绝加载 (文件可能只传了一半)");
            return;
        }
    }
    let n_docs_probe: i64 = c
        .query_row("SELECT COUNT(*) FROM docs", [], |r| r.get(0))
        .unwrap_or(-1);
    if n_docs_probe < 0 {
        eprintln!("[sr] ⚠ docs 表不可用 → 拒绝加载");
        return;
    }
    // 1) 向量 → 内存索引
    let rows: Vec<(String, Vec<u8>)> = c
        .prepare("SELECT doc_id, vec FROM doc_vectors")
        .and_then(|mut s| {
            s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))
                .map(|rr| rr.collect::<Result<Vec<_>, _>>())
        })
        .unwrap_or_else(|e| {
            eprintln!("[sr] vec load err: {e}");
            Ok(Vec::new())
        })
        .unwrap_or_default();
    {
        let mut idx = FlatIndex::new(DIM);
        let mut ids = Vec::with_capacity(rows.len());
        for (id, blob) in &rows {
            if blob.len() == DIM * 4 {
                idx.add_row(&blob_to_f32(blob));
            }
            ids.push(id.clone());
        }
        *SR.index.lock().unwrap() = idx;
        *SR.ids.lock().unwrap() = ids;
    }
    // 2) FTS 表 (缺则自建; 内容变了用 count 不一致检测重建)
    let n_docs: i64 = c
        .query_row("SELECT COUNT(*) FROM docs", [], |r| r.get(0))
        .unwrap_or(0);
    let n_fts: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='docs_fts'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let mut need_build = n_fts == 0;
    if !need_build {
        let cnt: i64 = c
            .query_row("SELECT COUNT(*) FROM docs_fts", [], |r| r.get(0))
            .unwrap_or(0);
        if cnt != n_docs {
            need_build = true;
        }
    }
    if need_build && n_docs > 0 {
        eprintln!("[sr] building docs_fts over {n_docs} docs ...");
        let t0 = std::time::Instant::now();
        let _ = c.execute_batch(
            "DROP TABLE IF EXISTS docs_fts;
             CREATE VIRTUAL TABLE docs_fts USING fts5(doc_id UNINDEXED, text, tokenize='trigram');",
        );
        let build = || -> Result<(), String> {
            let mut sel = c
                .prepare("SELECT doc_id, text FROM docs")
                .map_err(|e| e.to_string())?;
            let mut rows = sel.query([]).map_err(|e| e.to_string())?;
            let tx = c.unchecked_transaction().map_err(|e| e.to_string())?;
            {
                let mut ins = c
                    .prepare("INSERT INTO docs_fts(doc_id, text) VALUES(?,?)")
                    .map_err(|e| e.to_string())?;
                while let Some(r) = rows.next().map_err(|e| e.to_string())? {
                    let did: String = r.get(0).map_err(|e| e.to_string())?;
                    let txt: String = r.get(1).map_err(|e| e.to_string())?;
                    ins.execute(params![did, txt]).map_err(|e| e.to_string())?;
                }
            }
            tx.commit().map_err(|e| e.to_string())?;
            Ok(())
        };
        if let Err(e) = build() {
            eprintln!("[sr] fts build err: {e}");
        } else {
            eprintln!("[sr] docs_fts built in {}ms", t0.elapsed().as_millis());
        }
    }
    eprintln!(
        "[sr] loaded {} docs, {} vectors",
        n_docs,
        SR.ids.lock().unwrap().len()
    );
}

// ---------- 检索 ----------

fn vec_leg(qv: &[f32], k: usize) -> Vec<(String, f32)> {
    let s = SR.index.lock().unwrap();
    if s.ntotal() == 0 {
        return Vec::new();
    }
    let ids = SR.ids.lock().unwrap();
    let hits = s.search(qv, k.min(s.ntotal()));
    hits.into_iter()
        .filter_map(|(sc, i)| ids.get(i).map(|id| (id.clone(), sc)))
        .collect()
}

/// 查询侧分词 — 必须与索引侧 (build_srag_db.py 的 `_toks`, 即 Mac 验证过的
/// `eval_ab.py::cjk_tokens`) 完全一致: ASCII 词(小写) + CJK 二元组 + 首字。
/// 索引是把这些 token 以空格串存进 unicode61 的 FTS5 表, 配 `ORDER BY rank`
/// 即得到带 IDF/长度归一的 BM25。
fn bigram_tokens(q: &str) -> Vec<String> {
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

/// FTS5 腿 (unicode61 over 预切好的 "ASCII词+CJK二元组" 串): 词 OR 组合 + `ORDER BY rank`(bm25)。
///
/// 2026-09-20 迭代: ①原按"命中词计数"排序无 IDF, 高频词挤掉真相关 (0.744);
/// ②改 bm25 后仍偏低, 根因是 trigram 三元组 + 4 字滑窗查询召回太窄 —— 同语料 Mac 侧
/// bigram BM25 达 R@5=0.949, 故整体改为 bigram 方案 (索引侧见 build_srag_db.py)。
fn fts_leg(c: &Connection, query: &str, k: usize) -> Vec<String> {
    let toks = bigram_tokens(query);
    if toks.is_empty() {
        return Vec::new();
    }
    let q: String = toks
        .iter()
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ");
    let mut stmt = match c.prepare(
        "SELECT doc_id FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[sr] fts prepare err: {e}");
            return Vec::new();
        }
    };
    stmt.query_map(rusqlite::params![q, k as i64], |r| r.get::<_, String>(0))
        .map(|rr| rr.filter_map(|x| x.ok()).collect::<Vec<_>>())
        .unwrap_or_else(|e| {
            eprintln!("[sr] fts query err: {e}");
            Vec::new()
        })
}

/// RRF 融合两路排名 → [(doc_id, rrf_score)]
fn rrf_merge(a: &[String], b: &[String], k: i64, top: usize) -> Vec<(String, f32)> {
    let mut sc: HashMap<String, f32> = HashMap::new();
    for (rank, id) in a.iter().enumerate() {
        *sc.entry(id.clone()).or_insert(0.0) += 1.0 / ((k + rank as i64 + 1) as f32);
    }
    for (rank, id) in b.iter().enumerate() {
        *sc.entry(id.clone()).or_insert(0.0) += 1.0 / ((k + rank as i64 + 1) as f32);
    }
    let mut v: Vec<(String, f32)> = sc.into_iter().collect();
    v.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.0.cmp(&y.0))
    });
    v.truncate(top);
    v
}

fn doc_json(c: &Connection, doc_id: &str, preview: usize) -> Option<Value> {
    c.query_row(
        "SELECT doc_id,session_id,device,agent,kind,ts_iso,char_count,text,anchors
         FROM docs WHERE doc_id=?",
        params![doc_id],
        |r| {
            let text: String = r.get(7)?;
            let cut = floor_char(&text, preview);
            Ok(json!({
                "doc_id": r.get::<_, String>(0)?,
                "session_id": r.get::<_, String>(1)?,
                "device": r.get::<_, String>(2)?,
                "agent": r.get::<_, String>(3)?,
                "kind": r.get::<_, String>(4)?,
                "ts_iso": r.get::<_, String>(5)?,
                "char_count": r.get::<_, i64>(6)?,
                "text_preview": &text[..cut],
                "text": &text[..cut],
                "anchors": r.get::<_, String>(8)?,
            }))
        },
    )
    .ok()
}

/// 取库内全文 (rerank 用; 不受 preview 截断影响)
fn doc_full_text(c: &Connection, doc_id: &str) -> Option<String> {
    c.query_row("SELECT text FROM docs WHERE doc_id=?", params![doc_id], |r| {
        r.get::<_, String>(0)
    })
    .ok()
}

/// base64(f32 LE × DIM) → Vec<f32>。供客户端自带 query 向量 (qv 参数) 用。
fn decode_qv(b64: &str) -> Option<Vec<f32>> {
    use base64::Engine;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .ok()?;
    if raw.len() != DIM * 4 {
        return None;
    }
    Some(
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

fn floor_char(s: &str, max: usize) -> usize {
    let mut i = max.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

// ---------- 路由 ----------

pub fn router() -> Router<crate::App> {
    Router::new()
        .route("/api/session-rag/search", get(search))
        .route("/api/session-rag/status", get(status))
        .route("/api/session-rag/session", get(session_docs))
        .route("/api/session-rag/ingest", post(ingest))
}

async fn search(State(app): State<crate::App>, req: Request<Body>) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let q = qparams(&req);
    let query = qval(&q, "q").trim().to_string();
    if query.is_empty() {
        return json_resp(400, &json!({"error": "q required"}));
    }
    let limit = qint(&q, "limit", 10).clamp(1, 50) as usize;
    // 精排: 显式 rerank=1 强制开 (服务在 8301); 缺省跟随 cfg.srag_rerank, 默认**关**。
    // 依据: hub 上 bge-reranker 对会话段 ~1.5-2s/篇, 20 候选整条查询 ~36s; 而 RRF 双路
    // 已达基线 (R@5 0.974) —— 收益不抵延迟, 故默认关且不随请求默认触发。
    let want_rerank = match qval(&q, "rerank").trim().to_lowercase().as_str() {
        "" => cfg.srag_rerank,
        "1" | "true" | "yes" | "on" => true,
        _ => false,
    };
    let f_device = qval(&q, "device").trim().to_string();
    let f_agent = qval(&q, "agent").trim().to_string();
    let f_session = qval(&q, "session_id").trim().to_string();
    let preview = qint(&q, "preview", 600).clamp(100, 4000) as usize;

    let t_all = std::time::Instant::now();
    let c = match sr_db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e})),
    };
    // leg1 vector
    let t1 = std::time::Instant::now();
    // 可选 qv: 客户端自带 query 向量 (base64 的 f32 LE, 1024 维), 跳过服务端嵌入。
    // 用途: 客户端本地嵌入 (Mac MLX, 与文档向量同引擎) + 跨引擎差异诊断。
    let qv_pre = qval(&q, "qv").trim().to_string();
    let dense: Vec<String> = if !qv_pre.is_empty() {
        match decode_qv(&qv_pre) {
            Some(v) => vec_leg(&v, LEG_TOPK).into_iter().map(|(id, _)| id).collect(),
            None => {
                eprintln!("[sr] qv decode err (expect base64 f32le x1024)");
                Vec::new()
            }
        }
    } else {
        match embed::embed_one(cfg, &query) {
            Ok(qv) => vec_leg(&qv, LEG_TOPK).into_iter().map(|(id, _)| id).collect(),
            Err(e) => {
                eprintln!("[sr] embed err: {e}");
                Vec::new()
            }
        }
    };
    let t_dense = t1.elapsed().as_millis();
    // leg2 fts
    let t2 = std::time::Instant::now();
    let sparse = fts_leg(&c, &query, LEG_TOPK);
    let t_fts = t2.elapsed().as_millis();

    let merged = rrf_merge(&dense, &sparse, cfg.rrf_k, RERANK_TOPK);
    let mut items: Vec<Value> = Vec::new();
    for (doc_id, rrf) in &merged {
        if let Some(mut v) = doc_json(&c, doc_id, preview) {
            if let Some(o) = v.as_object_mut() {
                o.insert("rrf_score".into(), json!(rrf));
            }
            items.push(v);
        }
    }
    // 过滤 (device/agent/session_id)
    if !f_device.is_empty() || !f_agent.is_empty() || !f_session.is_empty() {
        items.retain(|it| {
            let ok_d = f_device.is_empty()
                || it.get("device").and_then(|x| x.as_str()) == Some(f_device.as_str());
            let ok_a = f_agent.is_empty()
                || it.get("agent").and_then(|x| x.as_str()) == Some(f_agent.as_str());
            let ok_s = f_session.is_empty()
                || it.get("session_id").and_then(|x| x.as_str()) == Some(f_session.as_str());
            ok_d && ok_a && ok_s
        });
    }
    // rerank: embed::rerank 读的是 "content" 字段 (内部再截断到 cfg.rerank_max_chars)。
    // 2026-09-20 踩坑: 这里只 clone 了 items 没塞 content → 20 个候选全被当空文档,
    // 分数全部相同, 精排彻底失效 (指标与纯 RRF 一模一样)。content 取**库内全文**,
    // 不受 preview 参数影响 (preview 可能只有 200 字)。
    let mut t_rr = 0u128;
    if want_rerank && items.len() > 1 {
        let t3 = std::time::Instant::now();
        let rr: Vec<Value> = items
            .iter()
            .take(RERANK_TOPK)
            .cloned()
            .map(|mut v| {
                let full = v
                    .get("doc_id")
                    .and_then(|x| x.as_str())
                    .and_then(|id| doc_full_text(&c, id))
                    .unwrap_or_default();
                if let Some(o) = v.as_object_mut() {
                    o.insert("content".into(), json!(full));
                }
                v
            })
            .collect();
        let ranked = embed::rerank(cfg, &query, &rr, rr.len());
        // rerank 返回按分数重排的同构 items (含 text_preview), 以 doc_id 定序
        let order: Vec<String> = ranked
            .iter()
            .filter_map(|x| x.get("doc_id").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect();
        let mut by_id: HashMap<String, Value> = HashMap::new();
        for it in items.into_iter() {
            let id = it.get("doc_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            by_id.insert(id, it);
        }
        let mut out = Vec::new();
        for id in order {
            if let Some(orig) = by_id.remove(&id) {
                out.push(ranked.iter().find(|x| x.get("doc_id").and_then(|v| v.as_str()) == Some(id.as_str())).cloned().unwrap_or(orig));
            }
        }
        for (_, v) in by_id.into_iter() {
            out.push(v);
        }
        items = out;
        t_rr = t3.elapsed().as_millis();
    }
    items.truncate(limit);
    json_resp(
        200,
        &json!({
            "ok": true, "total": items.len(), "items": items,
            "legs": {"vector": dense.len(), "fts": sparse.len(), "rerank": want_rerank},
            "took_ms": {"dense": t_dense, "fts": t_fts, "rerank": t_rr, "total": t_all.elapsed().as_millis()}
        }),
    )
}

async fn status(State(app): State<crate::App>, req: Request<Body>) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let c = match sr_db(&app.cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e})),
    };
    let docs: i64 = c.query_row("SELECT COUNT(*) FROM docs", [], |r| r.get(0)).unwrap_or(0);
    let sessions: i64 = c
        .query_row("SELECT COUNT(DISTINCT session_id) FROM docs", [], |r| r.get(0))
        .unwrap_or(0);
    let vecs: i64 = c
        .query_row("SELECT COUNT(*) FROM doc_vectors", [], |r| r.get(0))
        .unwrap_or(0);
    let mut by_agent: HashMap<String, i64> = HashMap::new();
    if let Ok(mut st) = c.prepare("SELECT agent, COUNT(*) FROM docs GROUP BY agent") {
        if let Ok(rows) = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))) {
            for row in rows.flatten() {
                by_agent.insert(row.0, row.1);
            }
        }
    }
    let fresh: Option<String> = c
        .query_row("SELECT MAX(updated_at) FROM docs", [], |r| r.get(0))
        .ok();
    json_resp(
        200,
        &json!({"ok": true, "docs": docs, "sessions": sessions, "vectors": vecs,
                "in_memory": SR.ids.lock().unwrap().len(),
                "by_agent": by_agent, "updated_at": fresh, "dim": DIM}),
    )
}

async fn session_docs(
    State(app): State<crate::App>,
    req: Request<Body>,
) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let q = qparams(&req);
    let sid = qval(&q, "session_id").trim().to_string();
    if sid.is_empty() {
        return json_resp(400, &json!({"error": "session_id required"}));
    }
    let limit = qint(&q, "limit", 50).clamp(1, 200);
    let c = match sr_db(&app.cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e})),
    };
    let mut out: Vec<Value> = Vec::new();
    if let Ok(mut st) = c.prepare(
        "SELECT doc_id FROM docs WHERE session_id=? ORDER BY ts_iso ASC, doc_id ASC LIMIT ?",
    ) {
        if let Ok(rows) = st.query_map(params![sid, limit], |r| r.get::<_, String>(0)) {
            for id in rows.flatten() {
                if let Some(v) = doc_json(&c, &id, 2000) {
                    out.push(v);
                }
            }
        }
    }
    json_resp(200, &json!({"ok": true, "session_id": sid, "total": out.len(), "items": out}))
}

async fn ingest(State(app): State<crate::App>, req: Request<Body>) -> axum::response::Response {
    if !auth_ok(&app, &req) {
        return json_resp(401, &json!({"error": "unauthorized"}));
    }
    let cfg = &app.cfg;
    let body = match axum::body::to_bytes(req.into_body(), 1 << 26).await {
        Ok(b) => b,
        Err(e) => return bad_request(&e.to_string()),
    };
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return bad_request("bad json"),
    };
    let docs = match v.get("docs").and_then(|x| x.as_array()) {
        Some(a) => a.clone(),
        None => return bad_request("docs[] required"),
    };
    if docs.len() > 500 {
        return bad_request("batch too large (max 500)");
    }
    let c = match sr_db(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(500, &json!({"error": e})),
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let mut n_ok = 0usize;
    let mut n_vec = 0usize;
    for d in docs.iter() {
        let doc_id = d.get("doc_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
        if doc_id.is_empty() {
            continue;
        }
        let text = d.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let anchors = d.get("anchors").cloned().unwrap_or_else(|| json!([])).to_string();
        let _ = c.execute(
            "INSERT INTO docs(doc_id,session_id,device,agent,kind,ts_iso,char_count,anchors,text,updated_at)
             VALUES(?,?,?,?,?,?,?,?,?,?)
             ON CONFLICT(doc_id) DO UPDATE SET text=excluded.text, anchors=excluded.anchors,
               session_id=excluded.session_id, device=excluded.device, agent=excluded.agent,
               kind=excluded.kind, ts_iso=excluded.ts_iso, char_count=excluded.char_count,
               updated_at=excluded.updated_at",
            params![
                doc_id,
                d.get("session_id").and_then(|x| x.as_str()).unwrap_or(""),
                d.get("device").and_then(|x| x.as_str()).unwrap_or(""),
                d.get("agent").and_then(|x| x.as_str()).unwrap_or(""),
                d.get("kind").and_then(|x| x.as_str()).unwrap_or(""),
                d.get("ts_iso").and_then(|x| x.as_str()).unwrap_or(""),
                text.chars().count() as i64,
                anchors,
                text,
                now
            ],
        );
        // FTS: 该行重建。**必须与索引侧同款分词** —— docs_fts 是 bigram 预切分的空格串
        // + unicode61 建的; 直接插原文会让增量文档在稀疏路里永远搜不到 (unicode61 对
        // CJK 按单字切, 而查询侧发的是二元组)。2026-09-20 修。
        let fts_text = bigram_tokens(&text).join(" ");
        let _ = c.execute("DELETE FROM docs_fts WHERE doc_id=?", params![doc_id]);
        let _ = c.execute(
            "INSERT INTO docs_fts(doc_id, text) VALUES(?,?)",
            params![doc_id, fts_text],
        );
        n_ok += 1;
        // 向量 (可选)
        if let Some(arr) = d.get("vec").and_then(|x| x.as_array()) {
            if arr.len() == DIM {
                let mut vv = vec![0f32; DIM];
                for (i, x) in arr.iter().enumerate() {
                    vv[i] = x.as_f64().unwrap_or(0.0) as f32;
                }
                let blob = f32_to_blob(&vv);
                let _ = c.execute(
                    "INSERT INTO doc_vectors(doc_id,vec,updated_at) VALUES(?,?,?)
                     ON CONFLICT(doc_id) DO UPDATE SET vec=excluded.vec, updated_at=excluded.updated_at",
                    params![doc_id, blob, now],
                );
                // 内存索引增量更新
                let mut idx = SR.index.lock().unwrap();
                let mut ids = SR.ids.lock().unwrap();
                match ids.iter().position(|x| x == &doc_id) {
                    Some(p) => idx.set_row(p, &vv),
                    None => {
                        idx.add_row(&vv);
                        ids.push(doc_id.clone());
                    }
                }
                n_vec += 1;
            }
        }
    }
    // 整会话替换 (增量管线用): `prune_sessions` 里列出的会话, 删除本次 payload 未覆盖的
    // 旧 doc_id。段边界会随新消息变化 —— 只 upsert 不删旧段会留下"幽灵段" (内容已被
    // 新分段取代却仍能被检索到)。2026-09-20 加。
    let prune: Vec<String> = v
        .get("prune_sessions")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|s| s.as_str()).map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let mut n_pruned = 0usize;
    if !prune.is_empty() {
        let keep: std::collections::HashSet<String> = docs
            .iter()
            .filter_map(|d| d.get("doc_id").and_then(|x| x.as_str()).map(|s| s.to_string()))
            .collect();
        for sid in prune.iter() {
            // 条目支持 "device|agent|session_id" 三元组 (推荐, 避免跨设备 session_id 撞车
            // 误删) 或裸 session_id (仅按 session_id 匹配)
            let parts: Vec<&str> = sid.split('|').collect();
            let old: Vec<String> = if parts.len() == 3 {
                match c.prepare(
                    "SELECT doc_id FROM docs WHERE device=? AND agent=? AND session_id=?",
                ) {
                    Ok(mut st) => st
                        .query_map(params![parts[0], parts[1], parts[2]], |r| {
                            r.get::<_, String>(0)
                        })
                        .map(|rows| rows.flatten().collect())
                        .unwrap_or_default(),
                    Err(_) => Vec::new(),
                }
            } else {
                match c.prepare("SELECT doc_id FROM docs WHERE session_id=?") {
                    Ok(mut st) => st
                        .query_map(params![sid], |r| r.get::<_, String>(0))
                        .map(|rows| rows.flatten().collect())
                        .unwrap_or_default(),
                    Err(_) => Vec::new(),
                }
            };
            for id in old {
                if keep.contains(&id) {
                    continue;
                }
                let _ = c.execute("DELETE FROM docs WHERE doc_id=?", params![id]);
                let _ = c.execute("DELETE FROM docs_fts WHERE doc_id=?", params![id]);
                let _ = c.execute("DELETE FROM doc_vectors WHERE doc_id=?", params![id]);
                n_pruned += 1;
            }
        }
        // FlatIndex 无删除接口 → 有剪除时整体重载 (只发生在增量, 代价可接受)
        reload_vectors(cfg);
    }
    json_resp(
        200,
        &json!({"ok": true, "upserted": n_ok, "vectors": n_vec, "pruned": n_pruned}),
    )
}

/// 从库里重载全部向量到内存索引 (prune 后用)。
fn reload_vectors(cfg: &Config) {
    let c = match sr_db(cfg) {
        Ok(c) => c,
        Err(_) => return,
    };
    let rows: Vec<(String, Vec<u8>)> = match c.prepare("SELECT doc_id, vec FROM doc_vectors") {
        Ok(mut st) => st
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    let mut idx = FlatIndex::new(DIM);
    let mut ids = Vec::with_capacity(rows.len());
    for (id, blob) in &rows {
        if blob.len() == DIM * 4 {
            idx.add_row(&blob_to_f32(blob));
        }
        ids.push(id.clone());
    }
    *SR.index.lock().unwrap() = idx;
    *SR.ids.lock().unwrap() = ids;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_orders_by_rank() {
        let a = vec!["x".to_string(), "y".to_string()];
        let b = vec!["y".to_string(), "z".to_string()];
        let m = rrf_merge(&a, &b, 60, 10);
        assert_eq!(m[0].0, "y"); // 两路都命中 → RRF 最高
        assert!(m.iter().any(|(id, _)| id == "x"));
        assert!(m.iter().any(|(id, _)| id == "z"));
    }

    #[test]
    fn blob_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.0];
        let b = f32_to_blob(&v);
        assert_eq!(blob_to_f32(&b), v);
    }
}
