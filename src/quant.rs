//! Block quantization formats.
//!
//! Both formats split a row into blocks of [`BLOCK`] values that share one scale,
//! stored struct-of-arrays (all quants, then all scales) so kernels stream both linearly.
//! Weight scales are f16, like llama.cpp's Q8_0 / Q4_0: decode streams every weight
//! once per token, so the scale bytes cost speed directly.
//!
//! - **Q8**: `w = q * d`, `q ∈ [-127, 127]`, 1.0625 bytes/weight.
//! - **Q4**: `w = (q - 8) * d`, `q ∈ [0, 15]`, 0.5625 bytes/weight. A block's 16 bytes
//!   hold element `j` in the low nibble and element `j + 16` in the high nibble of
//!   byte `j`, so SIMD code can unpack a whole block with one mask and one shift.
//!
//! Activations are quantized to Q8 on the fly (with f32 scales, since they never
//! leave the cache) so the inner loops are pure integer dot products (`vpdpbusd` with
//! AVX-512 VNNI).

pub const BLOCK: usize = 32;

/// Quantizes one block to Q8 and returns its scale.
fn q8_block(x: &[f32], qs: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    for (q, &v) in qs.iter_mut().zip(x) {
        // Ties-to-even matches what SIMD float->int conversion does.
        *q = (v * inv).round_ties_even() as i8;
    }
    amax / 127.0
}

/// Q8 with f32 scales: the activation format (see [`Q8Vec`]).
pub fn quantize_q8(x: &[f32], qs: &mut [i8], scales: &mut [f32]) {
    debug_assert_eq!(x.len() % BLOCK, 0);
    for ((x, qs), d) in x
        .chunks_exact(BLOCK)
        .zip(qs.chunks_exact_mut(BLOCK))
        .zip(scales.iter_mut())
    {
        *d = q8_block(x, qs);
    }
}

/// Q8 with f16 scales: the weight format.
pub fn quantize_q8_weights(x: &[f32], qs: &mut [i8], scales: &mut [u16]) {
    debug_assert_eq!(x.len() % BLOCK, 0);
    for ((x, qs), d) in x
        .chunks_exact(BLOCK)
        .zip(qs.chunks_exact_mut(BLOCK))
        .zip(scales.iter_mut())
    {
        *d = f32_to_f16(q8_block(x, qs));
    }
}

pub fn dequantize_q8(qs: &[i8], scales: &[u16], out: &mut [f32]) {
    for ((out, qs), &d) in out
        .chunks_exact_mut(BLOCK)
        .zip(qs.chunks_exact(BLOCK))
        .zip(scales)
    {
        let d = f16_to_f32(d);
        for (o, &q) in out.iter_mut().zip(qs) {
            *o = q as f32 * d;
        }
    }
}

/// Symmetric 4-bit quantization with an error-minimizing scale per block.
pub fn quantize_q4(x: &[f32], qs: &mut [u8], scales: &mut [u16]) {
    debug_assert_eq!(x.len() % BLOCK, 0);
    for ((x, qs), d) in x
        .chunks_exact(BLOCK)
        .zip(qs.chunks_exact_mut(BLOCK / 2))
        .zip(scales.iter_mut())
    {
        let mut levels = [0i8; BLOCK];
        *d = f32_to_f16(search_q4_scale(x, &mut levels));
        for j in 0..BLOCK / 2 {
            let q = |l: i8| (l + 8) as u8;
            qs[j] = q(levels[j]) | (q(levels[j + BLOCK / 2]) << 4);
        }
    }
}
/// Picks levels `l ∈ [-8, 7]` and a scale `d` so that `Σ (x - d·l)²` is small.
///
/// With only 16 levels, the obvious choice (largest-magnitude value → -8) wastes
/// precision on a single outlier. Instead, try a range of candidate grids around it; for
/// each, the least-squares scale is `d = Σxl / Σl²` with error `Σx² - (Σxl)²/Σl²`, so
/// keep the grid that maximizes `(Σxl)²/Σl²` (the approach of llama.cpp's k-quants).
/// Candidate 0 is the obvious grid, so the result is never worse than it.
fn search_q4_scale(x: &[f32], levels: &mut [i8; BLOCK]) -> f32 {
    let max = x
        .iter()
        .copied()
        .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
    if max == 0.0 {
        levels.fill(0);
        return 0.0;
    }
    let (mut best_score, mut best_d) = (-1.0, 0.0);
    let mut candidate = [0i8; BLOCK];
    for step in -9..=9 {
        let inv = -(8.0 + 0.1 * step as f32) / max;
        let (mut sum_xl, mut sum_l2) = (0.0, 0.0);
        for (l, &v) in candidate.iter_mut().zip(x) {
            *l = (v * inv).round_ties_even().clamp(-8.0, 7.0) as i8;
            sum_xl += v * *l as f32;
            sum_l2 += (*l as f32) * (*l as f32);
        }
        if sum_l2 > 0.0 && sum_xl * sum_xl / sum_l2 > best_score {
            best_score = sum_xl * sum_xl / sum_l2;
            best_d = sum_xl / sum_l2;
            *levels = candidate;
        }
    }
    best_d
}

pub fn dequantize_q4(qs: &[u8], scales: &[u16], out: &mut [f32]) {
    for ((out, qs), &d) in out
        .chunks_exact_mut(BLOCK)
        .zip(qs.chunks_exact(BLOCK / 2))
        .zip(scales)
    {
        let d = f16_to_f32(d);
        for (j, &b) in qs.iter().enumerate() {
            out[j] = ((b & 0x0F) as f32 - 8.0) * d;
            out[j + BLOCK / 2] = ((b >> 4) as f32 - 8.0) * d;
        }
    }
}

/// IEEE half-precision bits to f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = match exp {
        0 if mant == 0 => sign,
        0 => {
            // Subnormal: shift the leading 1 into the implicit-bit position.
            let shift = mant.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((mant << shift) & 0x3FF) << 13)
        }
        31 => sign | 0x7F80_0000 | (mant << 13),
        _ => sign | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE half-precision bits, rounding to nearest even; overflows to infinity.
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let mant = x & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal result: the value is m * 2^-24 with m = (1.mant) >> (14 - e).
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let rounded = (m + (1 << (shift - 1)) - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    // Round the 23-bit mantissa to 10 bits; a carry correctly bumps the exponent.
    let rounded = ((e as u32) << 23 | mant) + 0xFFF + ((mant >> 13) & 1);
    if rounded >> 23 >= 31 {
        return sign | 0x7C00;
    }
    sign | (rounded >> 13) as u16
}

/// Activations quantized to Q8, reused across matvecs. May hold several tokens'
/// vectors back to back; blocks never straddle two tokens since rows are whole blocks.
#[derive(Default)]
pub struct Q8Vec {
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
    /// Sum of the quants in each block, which lets kernels treat weights as unsigned:
    /// `Σ w·x = Σ (w + c)·x − c·Σ x`.
    pub sums: Vec<i32>,
}

impl Q8Vec {
    pub fn quantize(&mut self, x: &[f32]) {
        self.qs.resize(x.len(), 0);
        self.scales.resize(x.len() / BLOCK, 0.0);
        (crate::kernels::kernels().quantize_q8)(x, &mut self.qs, &mut self.scales);
        self.sums.clear();
        self.sums.extend(
            self.qs
                .chunks_exact(BLOCK)
                .map(|b| b.iter().map(|&q| q as i32).sum::<i32>()),
        );
    }

    pub fn as_slice(&self) -> Q8Slice<'_> {
        self.blocks(0, self.scales.len())
    }

    /// Blocks `start..start + len`, e.g. one token's row.
    pub fn blocks(&self, start: usize, len: usize) -> Q8Slice<'_> {
        Q8Slice {
            qs: &self.qs[start * BLOCK..][..len * BLOCK],
            scales: &self.scales[start..][..len],
            sums: &self.sums[start..][..len],
        }
    }
}

/// A borrowed run of blocks from a [`Q8Vec`].
#[derive(Clone, Copy)]
pub struct Q8Slice<'a> {
    pub qs: &'a [i8],
    pub scales: &'a [f32],
    pub sums: &'a [i32],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect()
    }

    #[test]
    fn q8_round_trip_error_is_bounded() {
        let x = sample(256);
        let (mut qs, mut sc, mut y) = (vec![0; 256], vec![0; 8], vec![0.0; 256]);
        quantize_q8_weights(&x, &mut qs, &mut sc);
        dequantize_q8(&qs, &sc, &mut y);
        for (b, (x, y)) in x.chunks(BLOCK).zip(y.chunks(BLOCK)).enumerate() {
            // Half a step of rounding, plus up to 127 * 2^-11 of a step from the f16 scale.
            let d = f16_to_f32(sc[b]);
            for (a, b2) in x.iter().zip(y) {
                assert!((a - b2).abs() <= 0.57 * d + 1e-6);
            }
        }
    }

    #[test]
    fn f16_round_trips_every_finite_value() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#06x} -> {f}");
            }
        }
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF); // largest finite
        assert_eq!(f32_to_f16(65520.0), 0x7C00); // rounds up to infinity
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3C00); // tie -> even
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3C02); // tie -> even
        assert_eq!(f32_to_f16(-2.0f32.powi(-24)), 0x8001); // smallest subnormal
        assert_eq!(f32_to_f16(2.0f32.powi(-26)), 0x0000); // underflow
    }

    #[test]
    fn q4_scale_search_beats_max_scaling() {
        let mut rng = crate::sampler::Rng::new(3);
        // Gaussian-ish weights with an occasional outlier, like real layers.
        let x: Vec<f32> = (0..BLOCK * 64)
            .map(|i| {
                let v = (0..4).map(|_| rng.next_f32() - 0.5).sum::<f32>();
                if i % 97 == 0 { v * 6.0 } else { v }
            })
            .collect();
        let (mut qs, mut sc, mut y) = (vec![0; x.len() / 2], vec![0; 64], vec![0.0; x.len()]);
        quantize_q4(&x, &mut qs, &mut sc);
        dequantize_q4(&qs, &sc, &mut y);
        let searched: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum();

        // The naive grid: largest magnitude -> -8, round to nearest.
        let naive: f32 = x
            .chunks(BLOCK)
            .map(|b| {
                let max = b
                    .iter()
                    .copied()
                    .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
                let d = max / -8.0;
                b.iter()
                    .map(|&v| v - (v / d).round().clamp(-8.0, 7.0) * d)
                    .map(|e| e * e)
                    .sum::<f32>()
            })
            .sum();
        assert!(searched <= naive, "{searched} vs {naive}");
        // Measured: ~9% less squared error on this distribution.
        assert!(
            searched < 0.95 * naive,
            "search should help noticeably: {searched} vs {naive}"
        );
    }

    #[test]
    fn zero_block_quantizes_to_zero() {
        let x = [0.0; BLOCK];
        let (mut qs, mut sc, mut y) = ([0u8; 16], [0x3C00], [1.0; BLOCK]);
        quantize_q4(&x, &mut qs, &mut sc);
        dequantize_q4(&qs, &sc, &mut y);
        assert_eq!(y, [0.0; BLOCK]);
    }
}
