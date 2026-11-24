// 内积 benchmark: 3087 x 1024 (12.65MB, 超 L3=3MB, 内存带宽瓶颈)
// 对比: 单累加器(标量) / 8路展开(标量) / AVX 256-bit 内积(=FAISS 做法) / 纯读带宽地板
// 目标 CPU: i5-3210M Ivy Bridge, 有 AVX(256bit) 无 AVX2 无 FMA
use std::time::Instant;
use std::hint::black_box;

struct FlatIndex {
    dim: usize,
    mat: Vec<f32>,
}

impl FlatIndex {
    fn new(n: usize, dim: usize) -> Self {
        let mut rng: f32 = 12345.0;
        let mut mat = Vec::with_capacity(n * dim);
        for _ in 0..n * dim {
            rng = (rng * 1103515245.0 + 12345.0) % 1_00000.0;
            mat.push(rng / 1_00000.0 - 0.5);
        }
        Self { dim, mat }
    }
    fn ntotal(&self) -> usize {
        self.mat.len() / self.dim
    }
    // A: 单累加器 (原版, 串行 FMA 依赖链)
    fn search_single(&self, q: &[f32], k: usize) -> Vec<(f32, usize)> {
        let d = self.dim;
        let n = self.ntotal();
        let mut scored = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * d;
            let mut dot = 0.0f32;
            for j in 0..d {
                dot += q[j] * self.mat[base + j];
            }
            scored.push((dot, i));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(k);
        scored
    }
    // B: AVX 256-bit 内积 (FAISS 同款手法: 4个 ymm 累加器, 每轮 32 float)
    #[target_feature(enable = "avx")]
    fn search_avx(&self, q: &[f32], k: usize) -> Vec<(f32, usize)> {
        use std::arch::x86_64::*;
        let d = self.dim;
        let n = self.ntotal();
        let mut scored = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * d;
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            let mut j = 0usize;
            let dlim = d - (d % 32);
            let qptr = q.as_ptr();
            let mptr = self.mat.as_ptr();
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
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(k);
        scored
    }
    // C: 纯读带宽地板 (只读一遍 mat, 不算乘法) — 理论下限
    fn scan_bw(&self) -> u64 {
        let mut s = 0u64;
        for i in 0..self.ntotal() {
            let base = i * self.dim;
            for j in 0..self.dim {
                s += (self.mat[base + j] as u32) as u64;
            }
        }
        s
    }
}

fn main() {
    let dim = 1024;
    let n = 3087;
    let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() / 1024f32.sqrt()).collect();
    let idx = FlatIndex::new(n, dim);
    // warm
    let use_avx = std::is_x86_feature_detected!("avx");
    for _ in 0..5 {
        idx.search_single(&q, 15);
        if use_avx { unsafe { idx.search_avx(&q, 15); } }
        idx.scan_bw();
    }
    let iters = 200;
    let mut xs_s = Vec::new();
    for _ in 0..iters {
        let t0 = Instant::now();
        black_box(idx.search_single(&q, 15));
        xs_s.push(t0.elapsed().as_micros());
    }
    let mut xs_a = Vec::new();
    for _ in 0..iters {
        let t0 = Instant::now();
        if use_avx {
            black_box(unsafe { idx.search_avx(&q, 15) });
        }
        xs_a.push(t0.elapsed().as_micros());
    }
    let mut xs_bw = Vec::new();
    for _ in 0..iters {
        let t0 = Instant::now();
        black_box(idx.scan_bw());
        xs_bw.push(t0.elapsed().as_micros());
    }
    xs_s.sort();
    xs_a.sort();
    xs_bw.sort();
    let mb = (n * dim * 4) as f64 / 1_000_000.0;
    println!("数据量 {mb:.2}MB (n={n}, d={dim})");
    println!("带宽地板(纯读一遍): p50={}us  => {}GB/s", xs_bw[iters / 2], (mb / (xs_bw[iters / 2] as f64 / 1e6)).round() as u64);
    println!("A 单累加器(标量):     p50={}us p95={}us", xs_s[iters / 2], xs_s[iters * 95 / 100]);
    if use_avx {
        println!("B AVX 256bit内积:     p50={}us p95={}us", xs_a[iters / 2], xs_a[iters * 95 / 100]);
        println!(
            "对比: A/地板={:.2}x  B/地板={:.2}x  A/B={:.2}x",
            xs_s[iters / 2] as f64 / xs_bw[iters / 2] as f64,
            xs_a[iters / 2] as f64 / xs_bw[iters / 2] as f64,
            xs_s[iters / 2] as f64 / xs_a[iters / 2] as f64
        );
    } else {
        println!("B AVX 256bit内积:     (CPU 无 AVX, 跳过)");
    }
}
