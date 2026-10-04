//! AVX2 + FMA kernels (8 x f32 per register).
//!
//! These are `#[target_feature]` functions: calling them is only sound on a CPU with
//! AVX2 and FMA, which `Kernels::for_isa` checks before handing them out. Lengths are
//! validated with `assert!`, so the raw-pointer loads below always stay in bounds.

use std::arch::x86_64::*;

use crate::quant::BLOCK;

#[inline]
#[target_feature(enable = "avx2,fma")]
pub fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

#[target_feature(enable = "avx2,fma")]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // Four independent accumulators hide the 4-cycle FMA latency.
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (k, acc) in acc.iter_mut().enumerate() {
            let (x, y) = unsafe {
                (
                    _mm256_loadu_ps(pa.add(i + 8 * k)),
                    _mm256_loadu_ps(pb.add(i + 8 * k)),
                )
            };
            *acc = _mm256_fmadd_ps(x, y, *acc);
        }
        i += 32;
    }
    while i + 8 <= n {
        let (x, y) = unsafe { (_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))) };
        acc[0] = _mm256_fmadd_ps(x, y, acc[0]);
        i += 8;
    }
    let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    hsum(sum) + super::scalar::dot_f32(&a[i..], &b[i..])
}

#[target_feature(enable = "avx2,fma")]
pub fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len());
    let n = y.len();
    let va = _mm256_set1_ps(a);
    let mut i = 0;
    while i + 8 <= n {
        unsafe {
            let py = y.as_mut_ptr().add(i);
            let v = _mm256_fmadd_ps(va, _mm256_loadu_ps(x.as_ptr().add(i)), _mm256_loadu_ps(py));
            _mm256_storeu_ps(py, v);
        }
        i += 8;
    }
    super::scalar::axpy_f32(&mut y[i..], a, &x[i..]);
}

/// Widens 8 bf16 values to f32: zero-extend to 32 bits, then shift into the high half.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn widen_bf16(v: __m128i) -> __m256 {
    _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(v)))
}

#[target_feature(enable = "avx2,fma")]
pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    assert_eq!(w.len(), x.len());
    let n = w.len();
    let (pw, px) = (w.as_ptr(), x.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (k, acc) in acc.iter_mut().enumerate() {
            let (wv, xv) = unsafe {
                (
                    _mm_loadu_si128(pw.add(i + 8 * k).cast()),
                    _mm256_loadu_ps(px.add(i + 8 * k)),
                )
            };
            *acc = _mm256_fmadd_ps(widen_bf16(wv), xv, *acc);
        }
        i += 32;
    }
    while i + 8 <= n {
        let (wv, xv) = unsafe {
            (
                _mm_loadu_si128(pw.add(i).cast()),
                _mm256_loadu_ps(px.add(i)),
            )
        };
        acc[0] = _mm256_fmadd_ps(widen_bf16(wv), xv, acc[0]);
        i += 8;
    }
    let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    hsum(sum) + super::scalar::dot_bf16(&w[i..], &x[i..])
}

/// Signed 8-bit dot product of one 32-byte block, as 8 partial i32 sums converted to f32.
///
/// `maddubs` needs one unsigned operand, so we move x's sign onto w:
/// `w * x == (w * sign(x)) * |x|`. Neither side is ever -128, so nothing saturates.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub fn mul_sum_i8(w: __m256i, x: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sw = _mm256_sign_epi8(w, x);
    let pairs = _mm256_maddubs_epi16(ax, sw);
    _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
}

/// Unpacks one Q4 block (16 bytes) into 32 signed bytes in `[-8, 7]`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub fn unpack_q4(raw: __m128i) -> __m256i {
    let m4 = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(raw, m4);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), m4);
    _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8))
}

#[target_feature(enable = "avx2,fma")]
pub fn dot_q8(wq: &[i8], ws: &[f32], xq: &[i8], xs: &[f32]) -> f32 {
    super::check_q8(wq.len(), ws, xq, xs);
    let mut acc = _mm256_setzero_ps();
    for b in 0..ws.len() {
        let (w, x) = unsafe {
            (
                _mm256_loadu_si256(wq.as_ptr().add(b * BLOCK).cast()),
                _mm256_loadu_si256(xq.as_ptr().add(b * BLOCK).cast()),
            )
        };
        acc = _mm256_fmadd_ps(_mm256_set1_ps(ws[b] * xs[b]), mul_sum_i8(w, x), acc);
    }
    hsum(acc)
}

#[target_feature(enable = "avx2,fma")]
pub fn dot_q4(wq: &[u8], ws: &[f32], xq: &[i8], xs: &[f32]) -> f32 {
    super::check_q8(wq.len() * 2, ws, xq, xs);
    let mut acc = _mm256_setzero_ps();
    for b in 0..ws.len() {
        let (raw, x) = unsafe {
            (
                _mm_loadu_si128(wq.as_ptr().add(b * BLOCK / 2).cast()),
                _mm256_loadu_si256(xq.as_ptr().add(b * BLOCK).cast()),
            )
        };
        acc = _mm256_fmadd_ps(
            _mm256_set1_ps(ws[b] * xs[b]),
            mul_sum_i8(unpack_q4(raw), x),
            acc,
        );
    }
    hsum(acc)
}

#[target_feature(enable = "avx2,fma")]
pub fn quantize_q8(x: &[f32], qs: &mut [i8], scales: &mut [f32]) {
    assert!(x.len() == scales.len() * BLOCK && qs.len() == x.len());
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    for (b, scale) in scales.iter_mut().enumerate() {
        let v: [__m256; 4] =
            std::array::from_fn(|k| unsafe { _mm256_loadu_ps(x.as_ptr().add(b * BLOCK + 8 * k)) });
        let m = _mm256_max_ps(
            _mm256_max_ps(_mm256_and_ps(v[0], abs_mask), _mm256_and_ps(v[1], abs_mask)),
            _mm256_max_ps(_mm256_and_ps(v[2], abs_mask), _mm256_and_ps(v[3], abs_mask)),
        );
        let m = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps::<1>(m));
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let amax = _mm_cvtss_f32(_mm_max_ss(m, _mm_movehdup_ps(m)));
        *scale = amax / 127.0;
        let inv = _mm256_set1_ps(if amax > 0.0 { 127.0 / amax } else { 0.0 });
        let i: [__m256i; 4] = std::array::from_fn(|k| _mm256_cvtps_epi32(_mm256_mul_ps(v[k], inv)));
        // The packs interleave 128-bit lanes; the final permute restores element order.
        let packed = _mm256_packs_epi16(
            _mm256_packs_epi32(i[0], i[1]),
            _mm256_packs_epi32(i[2], i[3]),
        );
        let ordered =
            _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        unsafe { _mm256_storeu_si256(qs.as_mut_ptr().add(b * BLOCK).cast(), ordered) };
    }
}

/// The generic attention kernel, compiled (and auto-vectorized) for this ISA.
#[target_feature(enable = "avx2,fma")]
pub fn attention(
    q: &[f32],
    kv: &dyn super::attention::KvTiles,
    limits: &[usize],
    scale: f32,
    out: &mut [f32],
) {
    super::attention::attention_impl::<true>(q, kv, limits, scale, out)
}
