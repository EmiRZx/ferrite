//! Block quantization formats.
//!
//! Both formats split a row into blocks of [`BLOCK`] values that share one f32 scale,
//! stored struct-of-arrays (all quants, then all scales) so kernels stream both linearly.
//!
//! - **Q8**: `w = q * d`, `q ∈ [-127, 127]`, 1.125 bytes/weight.
//! - **Q4**: `w = (q - 8) * d`, `q ∈ [0, 15]`, 0.625 bytes/weight. A block's 16 bytes
//!   hold element `j` in the low nibble and element `j + 16` in the high nibble of
//!   byte `j`, so SIMD code can unpack a whole block with one mask and one shift.
//!
//! Activations are quantized to Q8 on the fly so the inner loops are pure integer
//! dot products (`vpdpbusd` with AVX-512 VNNI).

pub const BLOCK: usize = 32;

pub fn quantize_q8(x: &[f32], qs: &mut [i8], scales: &mut [f32]) {
    debug_assert_eq!(x.len() % BLOCK, 0);
    for ((x, qs), d) in x
        .chunks_exact(BLOCK)
        .zip(qs.chunks_exact_mut(BLOCK))
        .zip(scales.iter_mut())
    {
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        *d = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for (q, &v) in qs.iter_mut().zip(x) {
            // Ties-to-even matches what SIMD float->int conversion does.
            *q = (v * inv).round_ties_even() as i8;
        }
    }
}

pub fn dequantize_q8(qs: &[i8], scales: &[f32], out: &mut [f32]) {
    for ((out, qs), &d) in out
        .chunks_exact_mut(BLOCK)
        .zip(qs.chunks_exact(BLOCK))
        .zip(scales)
    {
        for (o, &q) in out.iter_mut().zip(qs) {
            *o = q as f32 * d;
        }
    }
}

/// Symmetric 4-bit quantization with an error-minimizing scale per block.
pub fn quantize_q4(x: &[f32], qs: &mut [u8], scales: &mut [f32]) {
    debug_assert_eq!(x.len() % BLOCK, 0);
    for ((x, qs), d) in x
        .chunks_exact(BLOCK)
        .zip(qs.chunks_exact_mut(BLOCK / 2))
        .zip(scales.iter_mut())
    {
        let mut levels = [0i8; BLOCK];
        *d = search_q4_scale(x, &mut levels);
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

pub fn dequantize_q4(qs: &[u8], scales: &[f32], out: &mut [f32]) {
    for ((out, qs), &d) in out
        .chunks_exact_mut(BLOCK)
        .zip(qs.chunks_exact(BLOCK / 2))
        .zip(scales)
    {
        for (j, &b) in qs.iter().enumerate() {
            out[j] = ((b & 0x0F) as f32 - 8.0) * d;
            out[j + BLOCK / 2] = ((b >> 4) as f32 - 8.0) * d;
        }
    }
}

/// An activation vector quantized to Q8, reused across matvecs.
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
        let (mut qs, mut sc, mut y) = (vec![0; 256], vec![0.0; 8], vec![0.0; 256]);
        quantize_q8(&x, &mut qs, &mut sc);
        dequantize_q8(&qs, &sc, &mut y);
        for (b, (x, y)) in x.chunks(BLOCK).zip(y.chunks(BLOCK)).enumerate() {
            for (a, b2) in x.iter().zip(y) {
                assert!((a - b2).abs() <= sc[b] / 2.0 + 1e-6);
            }
        }
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
        let (mut qs, mut sc, mut y) = (vec![0; x.len() / 2], vec![0.0; 64], vec![0.0; x.len()]);
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
        let (mut qs, mut sc, mut y) = ([0u8; 16], [1.0], [1.0; BLOCK]);
        quantize_q4(&x, &mut qs, &mut sc);
        dequantize_q4(&qs, &sc, &mut y);
        assert_eq!(y, [0.0; BLOCK]);
    }
}
