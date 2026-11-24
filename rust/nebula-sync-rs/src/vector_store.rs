use rusqlite::{params, Connection};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::Config;
use crate::db::vec_db;
use crate::embed::{embed, embed_one};

/// In-memory flat inner-product index over 1024-dim f32 vectors
/// (1:1 replacement for faiss.IndexFlatIP(1024)).
/// `pub(crate)`: session_rag 复用同一套 AVX 内积实现。
pub(crate) struct FlatIndex {
    dim: usize,
    mat: Vec<f32>,
}

impl FlatIndex {
    pub(crate) fn new(dim: usize) -> Self {
        Self { dim, mat: Vec::new() }
    }
    pub(crate) fn ntotal(&self) -> usize {
        self.mat.len() / self.dim.max(1)
    }
    pub(crate) fn add_row(&mut self, v: &[f32]) {
        self.mat.extend_from_slice(v);
    }
    pub(crate) fn set_row(&mut self, i: usize, v: &[f32]) {
        let d = self.dim;
        self.mat[i * d..(i + 1) * d].copy_from_slice(v);
    }
    fn remove_row(&mut self, i: usize) {
        let d = self.dim;
        let n = self.ntotal();
        if i < n {
            let last = n - 1;
            if i != last {
                let tail = self.mat[last * d..(last + 1) * d].to_vec();
                self.mat[i * d..(i + 1) * d].copy_from_slice(&tail);
            }
            self.mat.truncate(last * d);
        }
    }
    /// Top-k by inner product.
    /// 内积用显式 AVX `_mm256` (4×ymm 累加器, 每轮 32 float, = FAISS IndexFlatIP
    /// 同款手法)。实测 home-server i5-3210M (Ivy, 有 AVX 无 AVX2/FMA): 标量
    /// 8-way 展开 ~3.3ms, AVX ~1.0ms → ~3.3× 加速 (dot_bench microbench)。
    /// 运行时 `is_x86_feature_detected!("avx")` 门控: 无 AVX 的 CPU 走标量回退,
    /// 所以 +avx 构建拷到纯 SSE2 机器不会 SIGILL。
    pub(crate) fn search(&self, q: &[f32], k: usize) -> Vec<(f32, usize)> {
        let d = self.dim;
        let n = self.ntotal();
        if n == 0 {
            return Vec::new();
        }
        if std::is_x86_feature_detected!("avx") {
            return unsafe { self.search_avx(q, k) };
        }
        self.search_scalar(q, k)
    }
    /// 标量回退: 8-way 展开累加器 (无 AVX 的 CPU / SSE2 构建)。
    fn search_scalar(&self, q: &[f32], k: usize) -> Vec<(f32, usize)> {
        let d = self.dim;
        let n = self.ntotal();
        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * d;
            let mut s0: f32 = 0.0;
            let mut s1: f32 = 0.0;
            let mut s2: f32 = 0.0;
            let mut s3: f32 = 0.0;
            let mut s4: f32 = 0.0;
            let mut s5: f32 = 0.0;
            let mut s6: f32 = 0.0;
            let mut s7: f32 = 0.0;
            let lim = d - (d % 8);
            let mut j = 0usize;
            while j < lim {
                s0 += q[j] * self.mat[base + j];
                s1 += q[j + 1] * self.mat[base + j + 1];
                s2 += q[j + 2] * self.mat[base + j + 2];
                s3 += q[j + 3] * self.mat[base + j + 3];
                s4 += q[j + 4] * self.mat[base + j + 4];
                s5 += q[j + 5] * self.mat[base + j + 5];
                s6 += q[j + 6] * self.mat[base + j + 6];
                s7 += q[j + 7] * self.mat[base + j + 7];
                j += 8;
            }
            while j < d {
                s0 += q[j] * self.mat[base + j];
                j += 1;
            }
            scored.push((s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7, i));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
    /// AVX 256-bit 内积: 4 个 ymm 累加器, 每轮 32 float (FAISS 同款)。
    #[target_feature(enable = "avx")]
    fn search_avx(&self, q: &[f32], k: usize) -> Vec<(f32, usize)> {
        use std::arch::x86_64::*;
        let d = self.dim;
        let n = self.ntotal();
        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(n);
        let qptr = q.as_ptr();
        let mptr = self.mat.as_ptr();
        for i in 0..n {
            let base = i * d;
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            let dlim = d - (d % 32);
            let mut j = 0usize;
            while j < dlim {
                unsafe {
                    a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(qptr.add(j)), _mm256_loadu_ps(mptr.add(base + j))));
                    a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(qptr.add(j + 8)), _mm256_loadu_ps(mptr.add(base + j + 8))));
                    a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(qptr.add(j + 16)), _mm256_loadu_ps(mptr.add(base + j + 16))));
                    a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(qptr.add(j + 24)), _mm256_loadu_ps(mptr.add(base + j + 24))));
                }
                j += 32;
            }
            let mut s = 0.0f32;
            while j < d {
                s += q[j] * self.mat[base + j];
                j += 1;
            }
            let mut sum = a0;
            sum = _mm256_add_ps(sum, a1);
            sum = _mm256_add_ps(sum, a2);
            sum = _mm256_add_ps(sum, a3);
            let mut out = [0.0f32; 8];
            unsafe {
                _mm256_storeu_ps(out.as_mut_ptr(), sum);
            }
            scored.push((out.iter().sum::<f32>() + s, i));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

struct State {
    index: FlatIndex,
    ids: Vec<String>,
}

/// VectorStore: FAISS replacement backed by persistent SQLite vector cache.
/// Same lifecycle as the Python version:
///  - startup: load all cached vectors instantly;
///  - bg: find memories without vectors, embed in batches of 16, persist;
///  - add/remove: incremental update of index + SQLite;
///  - interactive add() enqueues to an async drain thread (non-blocking).
pub struct VectorStore {
    pub state: Mutex<State>,
    pending: Mutex<VecDeque<(String, String)>>,
    pub done: std::sync::atomic::AtomicBool,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                index: FlatIndex::new(1024),
                ids: Vec::new(),
            }),
            pending: Mutex::new(VecDeque::new()),
            done: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn id_count(&self) -> usize {
        self.state.lock().unwrap().ids.len()
    }

    fn load_cached(&self, cfg: &Config) -> Result<usize, String> {
        let c = vec_db(cfg)?;
        let mut st = c
            .prepare("SELECT id, vec FROM mem_vectors")
            .map_err(|e| e.to_string())?;
        let rows = st
            .query_map([], |r| {
                let id: String = r.get(0)?;
                let vec: Vec<u8> = r.get(1)?;
                Ok((id, vec))
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let n = rows.len();
        if n == 0 {
            eprintln!("[vs] no cached vectors yet");
            return Ok(0);
        }
        let dim = 1024;
        let mut index = FlatIndex::new(dim);
        let mut ids = Vec::with_capacity(n);
        for (id, blob) in &rows {
            if blob.len() == dim * 4 {
                let mut row = vec![0f32; dim];
                for i in 0..dim {
                    row[i] = f32::from_le_bytes(blob[i * 4..i * 4 + 4].try_into().unwrap());
                }
                index.add_row(&row);
            }
            ids.push(id.clone());
        }
        let mut s = self.state.lock().unwrap();
        s.index = index;
        s.ids = ids;
        eprintln!("[vs] loaded {} cached vectors", n);
        Ok(n)
    }

    fn ensure_all_embedded(&self, cfg: &Config) {
        loop {
            let missing = {
                let c = match vec_db(cfg) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("[vs] missing-scan err: {e}");
                        thread::sleep(std::time::Duration::from_secs(10));
                        continue;
                    }
                };
                let res = c
                    .prepare(
                        "SELECT m.id, m.content FROM maindb.memories m \
                         WHERE m.content IS NOT NULL AND m.content!='' \
                         AND m.id NOT IN (SELECT id FROM mem_vectors)",
                    )
                    .and_then(|mut s| {
                        s.query_map([], |r| {
                            let id: String = r.get(0)?;
                            let content: String = r.get(1)?;
                            Ok((id, content))
                        })
                        .map(|rr| rr.collect::<Result<Vec<_>, _>>().unwrap_or_default())
                    });
                match res {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[vs] missing-scan err: {e}");
                        thread::sleep(std::time::Duration::from_secs(10));
                        continue;
                    }
                }
            };
            if missing.is_empty() {
                self.done.store(true, std::sync::atomic::Ordering::SeqCst);
                eprintln!("[vs] all memories embedded ✓");
                return;
            }
            let total = missing.len();
            let mut done = 0;
            for chunk in missing.chunks(16) {
                let contents: Vec<String> = chunk
                    .iter()
                    .map(|(_, c)| trunc512(c))
                    .collect();
                let vecs = match embed(cfg, &contents) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[vs] embed batch err: {e}; retrying in 10s");
                        thread::sleep(std::time::Duration::from_secs(10));
                        break;
                    }
                };
                if let Ok(vc) = vec_db(cfg) {
                    let now = now_f64();
                    for (b, v) in chunk.iter().zip(vecs.iter()) {
                        let blob = f32_slice_to_bytes(v);
                        vc.execute(
                            "INSERT OR REPLACE INTO mem_vectors(id,vec,updated_at) VALUES(?,?,?)",
                            params![b.0, blob, now],
                        )
                        .ok();
                    }
                                    }
                {
                    let mut s = self.state.lock().unwrap();
                    for (b, v) in chunk.iter().zip(vecs.iter()) {
                        match s.ids.iter().position(|x| x == &b.0) {
                            Some(pos) => s.index.set_row(pos, v),
                            None => {
                                s.index.add_row(v);
                                s.ids.push(b.0.clone());
                            }
                        }
                    }
                }
                done += chunk.len();
                eprintln!("[vs] embedded {}/{} pending", done.min(total), total);
            }
            thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    pub fn load_or_start(self: &Arc<Self>, cfg: &Config) {
        let n = self.load_cached(cfg).unwrap_or(0);
        let miss: i64 = {
            let c = match vec_db(cfg) {
                Ok(c) => c,
                Err(_) => return,
            };
            c.query_row(
                "SELECT COUNT(*) FROM maindb.memories m WHERE m.content IS NOT NULL AND m.content!='' \
                 AND m.id NOT IN (SELECT id FROM mem_vectors)",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0)
        };
        if miss > 0 {
            eprintln!("[vs] {miss} memories need embedding; starting bg thread");
            let this: &VectorStore = self;
            let cfg2 = cfg.clone();
            let arc = Arc::clone(self);
            thread::spawn(move || {
                let this: &VectorStore = &arc;
                this.ensure_all_embedded(&cfg2);
            });
        } else {
            self.done.store(true, std::sync::atomic::Ordering::SeqCst);
            eprintln!("[vs] all {n} memories already cached");
        }
        let cfg3 = cfg.clone();
        let arc = Arc::clone(self);
        thread::spawn(move || {
            let this: &VectorStore = &arc;
            this.drain_pending(&cfg3);
        });
    }

    pub fn add(&self, mid: &str, content: &str) {
        if content.is_empty() {
            return;
        }
        let mut p = self.pending.lock().unwrap();
        p.retain(|x| &x.0 != mid);
        p.push_back((mid.to_string(), content.to_string()));
        let n = p.len();
        drop(p);
        if n <= 1 {
            eprintln!("[vs] enqueued embed for {mid}");
        }
    }

    fn do_embed(&self, cfg: &Config, mid: &str, content: &str) {
        let v = match embed_one(cfg, content) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[vs] embed err for {mid}: {e}");
                self.pending
                    .lock()
                    .unwrap()
                    .push_back((mid.to_string(), content.to_string()));
                return;
            }
        };
        let blob = f32_slice_to_bytes(&v);
        if let Ok(vc) = vec_db(cfg) {
            vc.execute(
                "INSERT OR REPLACE INTO mem_vectors(id,vec,updated_at) VALUES(?,?,?)",
                params![mid, blob, now_f64()],
            )
            .ok();
        }
        let mut s = self.state.lock().unwrap();
        match s.ids.iter().position(|x| x == mid) {
            Some(idx) => s.index.set_row(idx, &v),
            None => {
                s.index.add_row(&v);
                s.ids.push(mid.to_string());
            }
        }
        eprintln!("[vs] embedded {mid}");
    }

    fn drain_pending(&self, cfg: &Config) {
        loop {
            let item = self.pending.lock().unwrap().pop_front();
            let Some((mid, content)) = item else {
                thread::sleep(std::time::Duration::from_millis(500));
                continue;
            };
            self.do_embed(cfg, &mid, &content);
        }
    }

    pub fn remove(self: &Arc<Self>, cfg: &Config, mid: &str) {
        if let Ok(vc) = vec_db(cfg) {
            vc.execute("DELETE FROM mem_vectors WHERE id=?", params![mid]).ok();
        }
        let (len, cached) = {
            let mut s = self.state.lock().unwrap();
            if let Some(pos) = s.ids.iter().position(|x| x == mid) {
                s.index.remove_row(pos);
                s.ids.remove(pos);
            }
            let cached: i64 = match vec_db(cfg) {
                Ok(vc) => vc
                    .query_row("SELECT COUNT(*) FROM mem_vectors", params![], |r| r.get(0))
                    .unwrap_or(0),
                Err(_) => 0,
            };
            (s.ids.len(), cached)
        };
        if len.saturating_sub(cached as usize) > (20.max(len * 3 / 10)) {
            let arc = Arc::clone(self);
            let cfg = cfg.clone();
            thread::spawn(move || {
                thread::sleep(std::time::Duration::from_secs(1));
                let this: &VectorStore = &arc;
                {
                    let mut s = this.state.lock().unwrap();
                    s.index = FlatIndex::new(1024);
                    s.ids.clear();
                }
                let _ = this.load_cached(&cfg);
            });
        }
    }

    pub fn search(&self, qv: &[f32], k: usize) -> Vec<(String, f32)> {
        let s = self.state.lock().unwrap();
        if s.index.ntotal() == 0 {
            return Vec::new();
        }
        let fk = (k * 3).min(s.index.ntotal());
        let hits = s.index.search(qv, fk);
        let mut out = Vec::new();
        for (sc, idx) in hits {
            if idx < s.ids.len() {
                out.push((s.ids[idx].clone(), sc));
                if out.len() >= k {
                    break;
                }
            }
        }
        out
    }
}

fn now_f64() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn trunc512(s: &str) -> String {
    let mut i = 512.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    s[..i].to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn flat_index_search_basic() {
        let mut idx = super::FlatIndex::new(2);
        idx.add_row(&[1.0, 0.0]);
        idx.add_row(&[0.0, 1.0]);
        let hits = idx.search(&[0.9, 0.1], 1);
        assert_eq!(hits[0].1, 0);
    }
}
