//! AVX-512 kernels (16 x f32 per register), with VNNI `vpdpbusd` for the int8 paths.
//!
//! Same contract as the AVX2 module: only sound on CPUs with the features below,
//! which `Kernels::for_isa` checks at runtime.

use std::arch::x86_64::*;

use crate::quant::{BLOCK, Q8Vec};

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
    x: &Q8Vec,
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
pub fn dot_q8(wq: &[i8], ws: &[f32], x: &Q8Vec) -> f32 {
    super::check_q8(wq.len(), ws, &x.qs, &x.scales);
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
pub fn dot_q4(wq: &[u8], ws: &[f32], x: &Q8Vec) -> f32 {
    super::check_q8(wq.len() * 2, ws, &x.qs, &x.scales);
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
