//! Small element-wise ops. Dot products and attention come from `kernels`.

use crate::kernels::kernels;

pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let mean_sq = (kernels().dot_f32)(x, x) / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let mean_sq = (kernels().dot_f32)(x, x) / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for (v, &w) in x.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn add_inplace(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

/// Precomputed rotary embedding tables in the "rotate half" (NeoX) layout.
pub struct Rope {
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub fn new(head_dim: usize, max_seq: usize, theta: f64) -> Self {
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for pos in 0..max_seq {
            for i in 0..half {
                let inv_freq = theta.powf(-((2 * i) as f64) / head_dim as f64);
                let angle = pos as f64 * inv_freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self { half, cos, sin }
    }

    /// Rotates one head vector in place: pairs are `(x[i], x[i + half])`.
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let (cos, sin) = (
            &self.cos[pos * self.half..][..self.half],
            &self.sin[pos * self.half..][..self.half],
        );
        let (lo, hi) = x.split_at_mut(self.half);
        for i in 0..self.half {
            let (a, b) = (lo[i], hi[i]);
            lo[i] = a * cos[i] - b * sin[i];
            hi[i] = b * cos[i] + a * sin[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_zero_is_identity() {
        let rope = Rope::new(8, 4, 10000.0);
        let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let orig = x.clone();
        rope.apply(&mut x, 0);
        assert_eq!(x, orig);
    }
}
