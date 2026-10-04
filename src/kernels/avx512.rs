//! AVX-512 kernels (16 x f32 per register), with VNNI `vpdpbusd` for the int8 paths.
//!
//! Same contract as the AVX2 module: only sound on CPUs with the features below,
//! which `Kernels::for_isa` checks at runtime.

use std::arch::x86_64::*;

use super::TILE;
use crate::quant::{BLOCK, Q8Slice};

/// Mask with the low `n` (< 16) lanes set, for loading a ragged tail.
#[inline]
fn tail_mask(n: usize) -> __mmask16 {
    ((1u32 << n) - 1) as __mmask16
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm512_setzero_ps(); 4];
    let mut i = 0;
    while i + 64 <= n {
        for (k, acc) in acc.iter_mut().enumerate() {
            let (x, y) = unsafe {
                (
                    _mm512_loadu_ps(pa.add(i + 16 * k)),
                    _mm512_loadu_ps(pb.add(i + 16 * k)),
                )
            };
            *acc = _mm512_fmadd_ps(x, y, *acc);
        }
        i += 64;
    }
    while i + 16 <= n {
        let (x, y) = unsafe { (_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i))) };
        acc[0] = _mm512_fmadd_ps(x, y, acc[0]);
        i += 16;
    }
    if i < n {
        // Masked loads never touch the lanes that are switched off.
        let m = tail_mask(n - i);
        let (x, y) = unsafe {
            (
                _mm512_maskz_loadu_ps(m, pa.add(i)),
                _mm512_maskz_loadu_ps(m, pb.add(i)),
            )
        };
        acc[1] = _mm512_fmadd_ps(x, y, acc[1]);
    }
    _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(acc[0], acc[1]),
        _mm512_add_ps(acc[2], acc[3]),
    ))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len());
    let n = y.len();
    let va = _mm512_set1_ps(a);
    let mut i = 0;
    while i < n {
        let m = if n - i >= 16 { !0 } else { tail_mask(n - i) };
        unsafe {
            let py = y.as_mut_ptr().add(i);
            let v = _mm512_fmadd_ps(
                va,
                _mm512_maskz_loadu_ps(m, x.as_ptr().add(i)),
                _mm512_maskz_loadu_ps(m, py),
            );
            _mm512_mask_storeu_ps(py, m, v);
        }
        i += 16;
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn widen_bf16(v: __m256i) -> __m512 {
    _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(v)))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    assert_eq!(w.len(), x.len());
    let n = w.len();
    let (pw, px) = (w.as_ptr(), x.as_ptr());
    let mut acc = [_mm512_setzero_ps(); 4];
    let mut i = 0;
    while i + 64 <= n {
        for (k, acc) in acc.iter_mut().enumerate() {
            let (wv, xv) = unsafe {
                (
                    _mm256_loadu_si256(pw.add(i + 16 * k).cast()),
                    _mm512_loadu_ps(px.add(i + 16 * k)),
                )
            };
            *acc = _mm512_fmadd_ps(widen_bf16(wv), xv, *acc);
        }
        i += 64;
    }
    while i + 16 <= n {
        let (wv, xv) = unsafe {
            (
                _mm256_loadu_si256(pw.add(i).cast()),
                _mm512_loadu_ps(px.add(i)),
            )
        };
        acc[0] = _mm512_fmadd_ps(widen_bf16(wv), xv, acc[0]);
        i += 16;
    }
    let sum = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
    _mm512_reduce_add_ps(sum) + super::scalar::dot_bf16(&w[i..], &x[i..])
}

/// Index vectors for `vpermd`/`vpermps`: entry `i` sends element `2i` to lanes 0..8
/// and element `2i + 1` to lanes 8..16, i.e. block pair `i` of a 16-block group.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn pair_indices() -> [__m512i; 8] {
    let base = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    std::array::from_fn(|i| _mm512_add_epi32(base, _mm512_set1_epi32(2 * i as i32)))
}

/// Shared loop for the quantized dot products.
///
/// `vpdpbusd` multiplies *unsigned* by signed bytes and adds groups of four into i32
/// lanes. Instead of moving signs around per iteration, weights are made unsigned by
/// an offset `c` (Q4 nibbles already are: `q = w + 8`; Q8 uses `w ^ 0x80 = w + 128`),
/// and the error is removed exactly, in integers: `Σ w·x = Σ (w + c)·x − c·Σ x`, where
/// `Σ x` per block comes precomputed with the activations. The `−c·Σ x` term seeds the
/// accumulator of the first i32 lane of each block, so it costs one masked permute.
///
/// Work proceeds in groups of 16 blocks so the 16 scale products are one vector
/// multiply; `load_pair(b)` returns the 64 unsigned weight bytes of blocks `b, b + 1`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn dot_quant_groups(
    ws: &[f32],
    x: Q8Slice,
    offset_shift: u32,
    load_pair: impl Fn(usize) -> __m512i,
) -> (f32, usize) {
    let nb = ws.len();
    let idx = pair_indices();
    let mut acc = _mm512_setzero_ps();
    let mut b = 0;
    while b + 16 <= nb {
        let (wsv, xsv, sums) = unsafe {
            (
                _mm512_loadu_ps(ws.as_ptr().add(b)),
                _mm512_loadu_ps(x.scales.as_ptr().add(b)),
                _mm512_loadu_si512(x.sums.as_ptr().add(b).cast()),
            )
        };
        let scales = _mm512_mul_ps(wsv, xsv);
        let corrections = _mm512_sub_epi32(
            _mm512_setzero_si512(),
            _mm512_sllv_epi32(sums, _mm512_set1_epi32(offset_shift as i32)),
        );
        for (i, idx) in idx.iter().enumerate() {
            let pair = b + 2 * i;
            let xv = unsafe { _mm512_loadu_si512(x.qs.as_ptr().add(pair * BLOCK).cast()) };
            let seed = _mm512_maskz_permutexvar_epi32(0x0101, *idx, corrections);
            let dot = _mm512_dpbusd_epi32(seed, load_pair(pair), xv);
            acc = _mm512_fmadd_ps(
                _mm512_permutexvar_ps(*idx, scales),
                _mm512_cvtepi32_ps(dot),
                acc,
            );
        }
        b += 16;
    }
    (_mm512_reduce_add_ps(acc), b)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn dot_q8(wq: &[i8], ws: &[f32], x: Q8Slice) -> f32 {
    super::check_q8(wq.len(), ws, x.qs, x.scales);
    let flip = _mm512_set1_epi8(i8::MIN);
    let (sum, b) = dot_quant_groups(ws, x, 7, |pair| {
        // SAFETY: `dot_quant_groups` only asks for pairs inside full 16-block groups.
        let w = unsafe { _mm512_loadu_si512(wq.as_ptr().add(pair * BLOCK).cast()) };
        _mm512_xor_si512(w, flip) // w + 128
    });
    let tail = b * BLOCK;
    sum + super::scalar::dot_q8(&wq[tail..], &ws[b..], &x.qs[tail..], &x.scales[b..])
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn dot_q4(wq: &[u8], ws: &[f32], x: Q8Slice) -> f32 {
    super::check_q8(wq.len() * 2, ws, x.qs, x.scales);
    let m4 = _mm512_set1_epi8(0x0F);
    let (sum, b) = dot_quant_groups(ws, x, 3, |pair| {
        // SAFETY: as above; a block pair is 32 packed bytes.
        let raw = unsafe { _mm256_loadu_si256(wq.as_ptr().add(pair * BLOCK / 2).cast()) };
        // raw = [pair block 0 | pair block 1], 16 bytes each. Duplicate each block into
        // two 128-bit lanes, shift the second copy by 4 and mask, giving
        // [lo nibbles 0, hi nibbles 0, lo nibbles 1, hi nibbles 1] = element order.
        let raw = _mm512_castsi256_si512(raw);
        let w = _mm512_shuffle_i64x2::<0b01_01_00_00>(raw, raw);
        _mm512_and_si512(_mm512_mask_srli_epi16::<4>(w, 0xFF00_FF00, w), m4)
    });
    let tail = b * BLOCK;
    sum + super::scalar::dot_q4(&wq[tail / 2..], &ws[b..], &x.qs[tail..], &x.scales[b..])
}

/// Register-tiled version of `dot_quant_groups` for batches: `TILE` weight rows ×
/// `TILE` tokens at once. Each loaded weight register feeds `TILE` tokens and each
/// activation register `TILE` rows, and the per-call overhead (setup, horizontal
/// sums, tails) is shared by `TILE²` dot products.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn tile_groups(
    ws: [&[f32]; TILE],
    x: [Q8Slice; TILE],
    offset_shift: u32,
    load_pair: impl Fn(usize, usize) -> __m512i,
) -> ([[f32; TILE]; TILE], usize) {
    let nb = ws[0].len();
    let idx = pair_indices();
    let mut acc = [[_mm512_setzero_ps(); TILE]; TILE];
    let mut b = 0;
    while b + 16 <= nb {
        // SAFETY (all loads): `b + 16 <= nb` and every slice was checked to hold `nb` blocks.
        let wsv: [__m512; TILE] =
            std::array::from_fn(|r| unsafe { _mm512_loadu_ps(ws[r].as_ptr().add(b)) });
        let xsv: [__m512; TILE] =
            std::array::from_fn(|t| unsafe { _mm512_loadu_ps(x[t].scales.as_ptr().add(b)) });
        let corrections: [__m512i; TILE] = std::array::from_fn(|t| {
            let sums = unsafe { _mm512_loadu_si512(x[t].sums.as_ptr().add(b).cast()) };
            _mm512_sub_epi32(
                _mm512_setzero_si512(),
                _mm512_sllv_epi32(sums, _mm512_set1_epi32(offset_shift as i32)),
            )
        });
        for (i, idx) in idx.iter().enumerate() {
            let pair = b + 2 * i;
            let w: [__m512i; TILE] = std::array::from_fn(|r| load_pair(r, pair));
            let w_scale: [__m512; TILE] =
                std::array::from_fn(|r| _mm512_permutexvar_ps(*idx, wsv[r]));
            for t in 0..TILE {
                let xv = unsafe { _mm512_loadu_si512(x[t].qs.as_ptr().add(pair * BLOCK).cast()) };
                let x_scale = _mm512_permutexvar_ps(*idx, xsv[t]);
                let seed = _mm512_maskz_permutexvar_epi32(0x0101, *idx, corrections[t]);
                for r in 0..TILE {
                    let dot = _mm512_dpbusd_epi32(seed, w[r], xv);
                    let scale = _mm512_mul_ps(w_scale[r], x_scale);
                    acc[r][t] = _mm512_fmadd_ps(scale, _mm512_cvtepi32_ps(dot), acc[r][t]);
                }
            }
        }
        b += 16;
    }
    let out = std::array::from_fn(|r| std::array::from_fn(|t| _mm512_reduce_add_ps(acc[r][t])));
    (out, b)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn tile_q8(w: [&[i8]; TILE], ws: [&[f32]; TILE], x: [Q8Slice; TILE]) -> [[f32; TILE]; TILE] {
    for r in 0..TILE {
        for xt in x {
            super::check_q8(w[r].len(), ws[r], xt.qs, xt.scales);
        }
    }
    let flip = _mm512_set1_epi8(i8::MIN);
    let (mut out, b) = tile_groups(ws, x, 7, |r, pair| {
        // SAFETY: only pairs inside full 16-block groups are requested.
        let v = unsafe { _mm512_loadu_si512(w[r].as_ptr().add(pair * BLOCK).cast()) };
        _mm512_xor_si512(v, flip)
    });
    let tail = b * BLOCK;
    for r in 0..TILE {
        for t in 0..TILE {
            out[r][t] += super::scalar::dot_q8(
                &w[r][tail..],
                &ws[r][b..],
                &x[t].qs[tail..],
                &x[t].scales[b..],
            );
        }
    }
    out
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn tile_q4(w: [&[u8]; TILE], ws: [&[f32]; TILE], x: [Q8Slice; TILE]) -> [[f32; TILE]; TILE] {
    for r in 0..TILE {
        for xt in x {
            super::check_q8(w[r].len() * 2, ws[r], xt.qs, xt.scales);
        }
    }
    let m4 = _mm512_set1_epi8(0x0F);
    let (mut out, b) = tile_groups(ws, x, 3, |r, pair| {
        // SAFETY: as above.
        let raw = unsafe { _mm256_loadu_si256(w[r].as_ptr().add(pair * BLOCK / 2).cast()) };
        let raw = _mm512_castsi256_si512(raw);
        let v = _mm512_shuffle_i64x2::<0b01_01_00_00>(raw, raw);
        _mm512_and_si512(_mm512_mask_srli_epi16::<4>(v, 0xFF00_FF00, v), m4)
    });
    let tail = b * BLOCK;
    for r in 0..TILE {
        for t in 0..TILE {
            out[r][t] += super::scalar::dot_q4(
                &w[r][tail / 2..],
                &ws[r][b..],
                &x[t].qs[tail..],
                &x[t].scales[b..],
            );
        }
    }
    out
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn quantize_q8(x: &[f32], qs: &mut [i8], scales: &mut [f32]) {
    assert!(x.len() == scales.len() * BLOCK && qs.len() == x.len());
    for (b, scale) in scales.iter_mut().enumerate() {
        let (lo, hi) = unsafe {
            let p = x.as_ptr().add(b * BLOCK);
            (_mm512_loadu_ps(p), _mm512_loadu_ps(p.add(16)))
        };
        let amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(lo), _mm512_abs_ps(hi)));
        *scale = amax / 127.0;
        let inv = _mm512_set1_ps(if amax > 0.0 { 127.0 / amax } else { 0.0 });
        let q = |v| _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(v, inv)));
        unsafe {
            let p = qs.as_mut_ptr().add(b * BLOCK);
            _mm_storeu_si128(p.cast(), q(lo));
            _mm_storeu_si128(p.add(16).cast(), q(hi));
        }
    }
}

/// `e^x` on 16 lanes; the vector form of [`super::attention::exp_approx`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn exp_ps(x: __m512) -> __m512 {
    let xc = _mm512_min_ps(
        _mm512_max_ps(x, _mm512_set1_ps(-87.0)),
        _mm512_set1_ps(88.0),
    );
    let k = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(xc, _mm512_set1_ps(std::f32::consts::LOG2_E)),
    );
    let r = _mm512_fnmadd_ps(k, _mm512_set1_ps(0.693_359_4), xc);
    let r = _mm512_fnmadd_ps(k, _mm512_set1_ps(-2.121_944_4e-4), r);
    let mut p = _mm512_set1_ps(1.987_569_1e-4);
    for c in [
        1.398_199_9e-3,
        8.333_452e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        0.5,
    ] {
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(c));
    }
    let y = _mm512_fmadd_ps(
        p,
        _mm512_mul_ps(r, r),
        _mm512_add_ps(r, _mm512_set1_ps(1.0)),
    );
    let bits = _mm512_add_epi32(
        _mm512_castps_si512(y),
        _mm512_slli_epi32::<23>(_mm512_cvtps_epi32(k)),
    );
    // Below -87 (including -inf from masking) the result is exactly 0.
    let keep = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(x, _mm512_set1_ps(-87.0));
    _mm512_maskz_mov_ps(keep, _mm512_castsi512_ps(bits))
}

/// Most queries scored together per key tile: each tile load feeds this many
/// queries, which also provide 8 independent FMA chains.
const QUERY_GROUP: usize = 4;

/// Scores `G` consecutive queries (rows of `hd` floats at `q`) against the 32 keys
/// of one transposed tile `kt`, into `s[..G]`. Returns `G`.
///
/// # Safety
/// `q` must point to `G * hd` floats and `kt` to `hd * 32`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
unsafe fn score_queries<const G: usize>(
    q: *const f32,
    kt: *const f32,
    hd: usize,
    scale: f32,
    s: &mut [[__m512; 2]; QUERY_GROUP],
) -> usize {
    use super::attention::TILE;
    for d in 0..hd {
        let (k0, k1) = unsafe {
            (
                _mm512_loadu_ps(kt.add(d * TILE)),
                _mm512_loadu_ps(kt.add(d * TILE + 16)),
            )
        };
        for (g, s) in s.iter_mut().enumerate().take(G) {
            let qd = _mm512_set1_ps(unsafe { *q.add(g * hd + d) } * scale);
            s[0] = _mm512_fmadd_ps(qd, k0, s[0]);
            s[1] = _mm512_fmadd_ps(qd, k1, s[1]);
        }
    }
    G
}

/// Hand-vectorized [`super::attention::attention_impl`] for head dims that are a
/// multiple of 16 (LLVM keeps the generic loops scalar). Same contract.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub fn attention(
    q: &[f32],
    kv: &dyn super::attention::KvTiles,
    limits: &[usize],
    scale: f32,
    out: &mut [f32],
) {
    use super::attention::{MAX_HEAD_DIM, MAX_QUERIES, TILE, attention_impl, exp_approx};
    let m = limits.len();
    assert!(
        (1..=MAX_QUERIES).contains(&m),
        "1..={MAX_QUERIES} queries per call"
    );
    let hd = q.len() / m;
    if !hd.is_multiple_of(16) {
        return attention_impl::<true>(q, kv, limits, scale, out);
    }
    assert!(hd <= MAX_HEAD_DIM && q.len() == m * hd && out.len() == m * hd);
    let n = limits.iter().copied().max().unwrap_or(0);

    let mut max = [f32::NEG_INFINITY; MAX_QUERIES];
    let mut sum = [0f32; MAX_QUERIES];
    out.fill(0.0);
    let (pq, po) = (q.as_ptr(), out.as_mut_ptr());

    for (tile, j0) in (0..n).step_by(TILE).enumerate() {
        let (kt, vt) = kv.tile(tile);
        assert!(
            kt.len() >= TILE * hd && vt.len() >= TILE * hd,
            "short KV tile"
        );
        // SAFETY (all raw accesses below): the tile lengths were just checked, query
        // rows are `< m`, `d < hd`, and value rows are `< visible <= TILE`.
        let (kt, pv) = (kt.as_ptr(), vt.as_ptr());
        let mut i0 = 0;
        while i0 < m {
            // Score up to QUERY_GROUP queries per pass over the tile.
            let mut s = [[_mm512_setzero_ps(); 2]; QUERY_GROUP];
            // SAFETY: rows i0..i0 + g are < m.
            let g = match m - i0 {
                1 => unsafe { score_queries::<1>(pq.add(i0 * hd), kt, hd, scale, &mut s) },
                2 | 3 => unsafe { score_queries::<2>(pq.add(i0 * hd), kt, hd, scale, &mut s) },
                _ => unsafe { score_queries::<4>(pq.add(i0 * hd), kt, hd, scale, &mut s) },
            };

            for (g, s) in s.iter().enumerate().take(g) {
                let i = i0 + g;
                let visible = limits[i].saturating_sub(j0).min(TILE);
                if visible == 0 {
                    continue;
                }
                let lanes = |from: usize| -> __mmask16 {
                    ((1u32 << visible.saturating_sub(from).min(16)) - 1) as __mmask16
                };
                let ninf = _mm512_set1_ps(f32::NEG_INFINITY);
                let s0 = _mm512_mask_mov_ps(ninf, lanes(0), s[0]);
                let s1 = _mm512_mask_mov_ps(ninf, lanes(16), s[1]);
                let tile_max = _mm512_reduce_max_ps(_mm512_max_ps(s0, s1));

                let o = unsafe { po.add(i * hd) };
                if tile_max > max[i] {
                    let r = _mm512_set1_ps(exp_approx(max[i] - tile_max));
                    for c in 0..hd / 16 {
                        unsafe {
                            _mm512_storeu_ps(
                                o.add(c * 16),
                                _mm512_mul_ps(r, _mm512_loadu_ps(o.add(c * 16))),
                            )
                        };
                    }
                    sum[i] *= exp_approx(max[i] - tile_max);
                    max[i] = tile_max;
                }
                let mx = _mm512_set1_ps(max[i]);
                let (p0, p1) = (exp_ps(_mm512_sub_ps(s0, mx)), exp_ps(_mm512_sub_ps(s1, mx)));
                sum[i] += _mm512_reduce_add_ps(_mm512_add_ps(p0, p1));
                let mut p = [0f32; TILE];
                unsafe {
                    _mm512_storeu_ps(p.as_mut_ptr(), p0);
                    _mm512_storeu_ps(p.as_mut_ptr().add(16), p1);
                }

                // o += Σ_j p_j v_j, eight 16-wide output chunks at a time so the FMA
                // chains are independent.
                let mut c0 = 0;
                while c0 < hd / 16 {
                    let width = (hd / 16 - c0).min(8);
                    let mut acc: [__m512; 8] = std::array::from_fn(|c| {
                        if c < width {
                            unsafe { _mm512_loadu_ps(o.add((c0 + c) * 16)) }
                        } else {
                            _mm512_setzero_ps()
                        }
                    });
                    for (j, &pj) in p[..visible].iter().enumerate() {
                        let v = unsafe { pv.add(j * hd + c0 * 16) };
                        let pj = _mm512_set1_ps(pj);
                        if width == 8 {
                            for (c, acc) in acc.iter_mut().enumerate() {
                                *acc = _mm512_fmadd_ps(
                                    pj,
                                    unsafe { _mm512_loadu_ps(v.add(c * 16)) },
                                    *acc,
                                );
                            }
                        } else {
                            for (c, acc) in acc.iter_mut().enumerate().take(width) {
                                *acc = _mm512_fmadd_ps(
                                    pj,
                                    unsafe { _mm512_loadu_ps(v.add(c * 16)) },
                                    *acc,
                                );
                            }
                        }
                    }
                    for (c, acc) in acc.iter().enumerate().take(width) {
                        unsafe { _mm512_storeu_ps(o.add((c0 + c) * 16), *acc) };
                    }
                    c0 += width;
                }
            }
            i0 += g;
        }
    }
    for (o, &sum) in out.chunks_exact_mut(hd).zip(&sum) {
        let inv = 1.0 / sum;
        o.iter_mut().for_each(|o| *o *= inv);
    }
}
