use rusqlite::{Connection, params, types::ValueRef};
use std::path::Path;

/// Main DB connection (mirrors Python db(): per-call, WAL, busy_timeout=30s).
pub fn db(cfg: &super::config::Config) -> Result<Connection, String> {
    let c = Connection::open(cfg.db.as_os_str()).map_err(|e| e.to_string())?;
    c.pragma_update(None, "journal_mode", "WAL").ok();
    c.pragma_update(None, "synchronous", "NORMAL").ok();
    c.busy_timeout(std::time::Duration::from_secs(30)).ok();
    // 9/23: scalar UDF so the contentless FTS5 table's triggers can compute the
    // bigram string from new/old.content. rusqlite 0.31 API = create_scalar_function
    // (closure FnMut(&Context) -> Result<T>); String: ToSql: SqlFnOutput.
    let _ = c.create_scalar_function(
        "nebula_bigram",
        1,
        rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let s = match ctx.get_raw(0) {
                ValueRef::Text(b) => String::from_utf8_lossy(b).to_string(),
                _ => String::new(),
            };
            Ok(super::ops::mem_bigram_str(&s))
        },
    );
    // one-shot trigram -> unicode61(bigram) migration per connection (idempotent).
    ensure_fts_bigram(&c);
    Ok(c)
}

/// 9/23 one-shot migration: if memories_fts is still the old trigram table, drop
/// it so the init_db schema batch recreates it as unicode61 (bigram). No-op once
/// migrated (the table reports unicode61). The triggers are re-created by the
/// batch too, so dropping the table here is safe. Repopulation is handled by
/// backfill_memories_fts() at startup.
fn ensure_fts_bigram(c: &Connection) {
    let tok: String = c
        .query_row(
            "SELECT COALESCE(sql, '') FROM sqlite_master WHERE type='table' AND name='memories_fts'",
            [],
            |r| r.get::<_, String>(0),
        )
        .unwrap_or_default();
    if tok.is_empty() || tok.contains("tokenize='unicode61'") {
        return; // not present (fresh) or already bigram+unicode61
    }
    if !tok.contains("tokenize='trigram'") {
        return; // unknown schema; leave it (will be recreated next startup if needed)
    }
    eprintln!("[fts] migrating memories_fts trigram -> unicode61(bigram) ...");
    c.execute_batch(
        "DROP TRIGGER IF EXISTS memories_fts_ai;
         DROP TRIGGER IF EXISTS memories_fts_ad;
         DROP TRIGGER IF EXISTS memories_fts_au;
         DROP TRIGGER IF EXISTS memories_bgram_ai;
         DROP TRIGGER IF EXISTS memories_bgram_au;
         DROP TABLE IF EXISTS memories_fts;",
    )
    .ok();
    eprintln!("[fts] memories_fts migrated (init_db will recreate as unicode61; backfill repopulates)");
}

/// Vector DB connection (separate file); attaches main DB as `maindb`
/// so the bg embedder can find memories lacking vectors cross-db.
pub fn vec_db(cfg: &super::config::Config) -> Result<Connection, String> {
    let c = Connection::open(cfg.vec_db.as_os_str()).map_err(|e| e.to_string())?;
    c.pragma_update(None, "journal_mode", "WAL").ok();
    c.pragma_update(None, "synchronous", "NORMAL").ok();
    c.busy_timeout(std::time::Duration::from_secs(30)).ok();
    if cfg.db.exists() {
        c.execute(
            "ATTACH DATABASE ? AS maindb",
            params![cfg.db.display().to_string()],
        )
        .ok();
    }
    Ok(c)
}

pub fn init_db(cfg: &super::config::Config) -> Result<(), String> {
    if !cfg.db.exists() {
        return Err(format!("DB missing: {} — seed it first (import_nebula.py)", cfg.db.display()));
    }
    let c = db(cfg)?;
    c.execute_batch(
        r#"
    -- LingHub 命令暂存 (Rust 单进程版: 用 host 列替代 CF 每 host 一个 DO 的隔离)
    CREATE TABLE IF NOT EXISTS commands(
      id TEXT PRIMARY KEY, host TEXT NOT NULL, command TEXT, camera TEXT DEFAULT 'back',
      room_id TEXT, ts INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_commands_host ON commands(host);
    -- 看板交互域 (阶段三: 从 CF D1 下沉; settings/device_logs 已存在, IF NOT EXISTS 幂等)
    CREATE TABLE IF NOT EXISTS settings(
      key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS tasks(
      id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT DEFAULT '',
      priority TEXT DEFAULT 'medium', status TEXT DEFAULT 'todo', due_date TEXT DEFAULT '',
      tags TEXT DEFAULT '[]', created_at TEXT, updated_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
    CREATE TABLE IF NOT EXISTS links(
      id TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT DEFAULT '', description TEXT DEFAULT '',
      tags TEXT DEFAULT '[]', created_at TEXT, updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS device_logs(
      host TEXT, lines TEXT, ts INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_device_logs_host ON device_logs(host);
    CREATE TABLE IF NOT EXISTS ops(
      id TEXT PRIMARY KEY, ts REAL, op TEXT, entity TEXT, payload TEXT, source TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ops_ts ON ops(ts);
    DROP VIEW IF EXISTS atoms_fts_view;
    CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts USING fts5(
      content, title, content='atoms_v2', content_rowid='rowid', tokenize='unicode61'
    );
    DROP TRIGGER IF EXISTS memories_fts_ai;
    DROP TRIGGER IF EXISTS memories_fts_ad;
    DROP TRIGGER IF EXISTS memories_fts_au;
    DROP TRIGGER IF EXISTS memories_bgram_ai;
    DROP TRIGGER IF EXISTS memories_bgram_au;
    -- 9/23: bigram + unicode61 + bm25 (9/20 proven session-rag recipe). Non-external-
    -- content FTS5 table stores the bigram string itself (no memories schema change);
    -- triggers maintain it via new/old.rowid (memories uses id TEXT PRIMARY KEY =
    -- implicit rowid). ensure_fts_bigram() handles the one-shot trigram->unicode61
    -- migration + drift self-heal. Verified: insert/update/delete sync, no ghost
    -- (fts_prod_schema.py ALL_OK).
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
      content, tokenize='unicode61'
    );
    CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, nebula_bigram(new.content));
    END;
    CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
      DELETE FROM memories_fts WHERE rowid = old.rowid;
    END;
    CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content ON memories BEGIN
      DELETE FROM memories_fts WHERE rowid = old.rowid;
      INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, nebula_bigram(new.content));
    END;
    CREATE TRIGGER IF NOT EXISTS atoms_ai AFTER INSERT ON atoms_v2 BEGIN
      INSERT INTO atoms_fts(rowid, content, title) VALUES (new.rowid, new.content, new.title);
    END;
    CREATE TRIGGER IF NOT EXISTS atoms_ad AFTER DELETE ON atoms_v2 BEGIN
      INSERT INTO atoms_fts(atoms_fts, rowid, content, title) VALUES('delete', old.rowid, old.content, old.title);
    END;
    "#,
    )
    .map_err(|e| e.to_string())?;

    let vc = vec_db(cfg)?;
    vc.execute_batch(
        r#"
    CREATE TABLE IF NOT EXISTS mem_vectors(
        id TEXT PRIMARY KEY, vec BLOB, updated_at REAL
    );
    CREATE TABLE IF NOT EXISTS query_embeds(
        qhash TEXT PRIMARY KEY, query TEXT, vec BLOB, updated_at REAL
    );
    "#,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Startup: if FTS empty while content table has data → one-shot backfill.
pub fn backfill_fts(cfg: &super::config::Config) {
    let c = match db(cfg) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[fts] backfill db err: {e}");
            return;
        }
    };
    // atoms_fts backfill
    let nf: i64 = c
        .query_row("SELECT COUNT(*) FROM atoms_fts", [], |r| r.get(0))
        .unwrap_or(0);
    let na: i64 = c
        .query_row("SELECT COUNT(*) FROM atoms_v2", [], |r| r.get(0))
        .unwrap_or(0);
    if nf == 0 && na > 0 {
        eprintln!("[fts] backfilling {na} atoms...");
        let mut st = c
            .prepare("INSERT INTO atoms_fts(rowid, content, title) VALUES (?,?,?)")
            .map_err(|e| e.to_string())
            .unwrap_or_else(|_| {
                // fall back if prepare fails
                std::process::exit(0)
            });
        let rows: Vec<(i64, String, Option<String>)> = c
            .prepare("SELECT rowid, content, title FROM atoms_v2")
            .and_then(|mut s| {
                let mut rr = s.query([])?;
                let mut out = Vec::new();
                while let Some(r) = rr.next()? {
                    out.push((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<String>>(2)?));
                }
                Ok(out)
            })
            .unwrap_or_default();
        for row in rows {
            st.execute(params![row.0, row.1, row.2]).ok();
        }
        eprintln!("[fts] done");
    } else if nf != na {
        eprintln!("[fts] count mismatch fts={nf} atoms={na}; incremental sync via triggers");
    }
    // memories FTS: bigram+bm25 full-repopulate (also the trigram->unicode61 migration)
    backfill_memories_fts(cfg);
}

/// 9/23 bigram+bm25: maintain the non-external-content memories_fts table.
/// The FTS table stores bigram strings itself (not external-content), so there is
/// no 'rebuild' command — instead we (re)populate by computing bigrams in Rust
/// from memories.content. Idempotent + self-healing: re-populates whenever the
/// row counts disagree (which also covers the one-shot trigram->unicode61
/// migration after ensure_fts_bigram() dropped the old table). Triggers keep it
/// in sync between startups; a count drift just means a one-time full repopulate
/// (~1s for a few thousand rows).
pub fn backfill_memories_fts(cfg: &super::config::Config) {
    let c = match db(cfg) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[fts] memories_fts backfill db err: {e}");
            return;
        }
    };
    let nmf: i64 = c
        .query_row("SELECT COUNT(*) FROM memories_fts", [], |r| r.get(0))
        .unwrap_or(0);
    let nm: i64 = c
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);
    if nmf == nm {
        return; // in sync; triggers will keep it that way
    }
    eprintln!("[fts] memories_fts drift ({nmf} fts vs {nm} memories) -> full repopulate");
    // Full reset: clear the FTS table, recompute bigrams in Rust, re-insert.
    // (Rust computes the bigram string so we don't depend on the UDF being
    // visible inside this statement's VALUES clause.)
    c.execute("DELETE FROM memories_fts", []).ok();
    let rows: Vec<(i64, String)> = {
        let mut out = Vec::new();
        if let Ok(mut st) = c.prepare("SELECT rowid, content FROM memories") {
            if let Ok(mut rr) = st.query([]) {
                while let Ok(Some(r)) = rr.next() {
                    let rid: i64 = r.get::<_, i64>(0).unwrap_or(0);
                    let content: String = r.get::<_, String>(1).unwrap_or_default();
                    out.push((rid, super::ops::mem_bigram_str(&content)));
                }
            }
        }
        out
    };
    let mut ins = match c.prepare(
        "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[fts] memories_fts repopulate prepare err: {e}");
            return;
        }
    };
    for (rid, big) in rows {
        ins.execute(params![rid, big]).ok();
    }
    c.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')", [])
        .ok();
    let nmf2: i64 = c
        .query_row("SELECT COUNT(*) FROM memories_fts", [], |r| r.get(0))
        .unwrap_or(0);
    eprintln!("[fts] memories_fts repopulated: {nmf2}/{nm} (bigram+bm25)");
}

/// Like Python `s[:n]` but never splits a UTF-8 char.
pub fn floor_char_boundary(s: &str, n: usize) -> usize {
    if n >= s.len() {
        return s.len();
    }
    let mut i = n;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}
