//! Portable reference kernels. SIMD versions are tested against these.

use crate::{quant::BLOCK, safetensors::bf16_to_f32};

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len());
    y.iter_mut().zip(x).for_each(|(y, &x)| *y += a * x);
}

pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    assert_eq!(w.len(), x.len());
    w.iter().zip(x).map(|(&w, &x)| bf16_to_f32(w) * x).sum()
}

pub fn dot_q8(wq: &[i8], ws: &[f32], xq: &[i8], xs: &[f32]) -> f32 {
    super::check_q8(wq.len(), ws, xq, xs);
    let blocks = wq.chunks_exact(BLOCK).zip(xq.chunks_exact(BLOCK));
    blocks
        .zip(ws.iter().zip(xs))
        .map(|((w, x), (&wd, &xd))| {
            let s: i32 = w.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum();
            s as f32 * wd * xd
        })
        .sum()
}

pub fn dot_q4(wq: &[u8], ws: &[f32], xq: &[i8], xs: &[f32]) -> f32 {
    super::check_q8(wq.len() * 2, ws, xq, xs);
    let blocks = wq.chunks_exact(BLOCK / 2).zip(xq.chunks_exact(BLOCK));
    blocks
        .zip(ws.iter().zip(xs))
        .map(|((w, x), (&wd, &xd))| {
            let mut s = 0i32;
            for (j, &b) in w.iter().enumerate() {
                s += ((b & 0x0F) as i32 - 8) * x[j] as i32;
                s += ((b >> 4) as i32 - 8) * x[j + BLOCK / 2] as i32;
            }
            s as f32 * wd * xd
        })
        .sum()
}
