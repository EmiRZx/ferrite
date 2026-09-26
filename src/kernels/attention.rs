//! Tiled multi-query attention with an online softmax, written as plain loops that
//! LLVM vectorizes. Each ISA wrapper compiles [`attention_impl`] with its own target
//! features.
//!
//! Keys are stored transposed per tile of [`TILE`] positions (`[head_dim][TILE]`),
//! so scoring one query against a whole tile is `head_dim` vector FMAs across the 32
//! positions, with no horizontal sums. Several queries (the query heads of one GQA
//! group, over a block of tokens) share each tile while it is in L1.

/// Positions per key tile.
pub const TILE: usize = 32;
/// Largest supported head dimension.
pub const MAX_HEAD_DIM: usize = 256;
/// Largest number of queries per call.
pub const MAX_QUERIES: usize = 64;

/// `a * b + c`, fused when the target has FMA (otherwise `mul_add` is a libm call).
#[inline(always)]
fn madd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `e^x` to within a few ulp, built from float and integer arithmetic only, so it
/// vectorizes (unlike `f32::exp`). Returns 0 below -87 (including -inf).
///
/// Splits `x = k ln2 + r` with `|r| <= ln2 / 2`, evaluates Cephes' `expf` polynomial for
/// `e^r` and adds `k` to the exponent bits.
#[inline(always)]
pub fn exp_approx(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const ROUND: f32 = 12_582_912.0; // 1.5 * 2^23: adding it rounds to an integer
    let xc = x.clamp(-87.0, 88.0);
    let k = (xc * LOG2E + ROUND) - ROUND;
    let r = xc - k * LN2_HI - k * LN2_LO;
    let mut p = 1.987_569_1e-4;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 0.5;
    let y = p * r * r + r + 1.0;
    let scaled = f32::from_bits((y.to_bits() as i32 + ((k as i32) << 23)) as u32);
    if x < -87.0 { 0.0 } else { scaled }
}

/// `m = limits.len()` queries (`q`: `[m, head_dim]`) attend over one KV head. Query
/// `i` sees positions `0..limits[i]`. `keys` is tile-transposed (see module docs),
/// `values` is `[positions, head_dim]`, `out` is `[m, head_dim]`.
#[inline(always)]
pub fn attention_impl<const FMA: bool>(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    limits: &[usize],
    scale: f32,
    out: &mut [f32],
) {
    let m = limits.len();
    assert!(
        (1..=MAX_QUERIES).contains(&m),
        "1..={MAX_QUERIES} queries per call"
    );
    let hd = q.len() / m;
    assert!(hd <= MAX_HEAD_DIM && q.len() == m * hd && out.len() == m * hd);
    let n = limits.iter().copied().max().unwrap_or(0);
    assert!(keys.len() >= n.div_ceil(TILE) * TILE * hd && values.len() >= n * hd);

    let mut max = [f32::NEG_INFINITY; MAX_QUERIES];
    let mut sum = [0f32; MAX_QUERIES];
    out.fill(0.0);

    for (tile, j0) in (0..n).step_by(TILE).enumerate() {
        let kt = &keys[tile * TILE * hd..][..TILE * hd];
        for i in 0..m {
            let visible = limits[i].saturating_sub(j0).min(TILE);
            if visible == 0 {
                continue;
            }
            // Scores for all 32 positions of the tile at once.
            let mut s = [0f32; TILE];
            for (&qd, kd) in q[i * hd..][..hd].iter().zip(kt.chunks_exact(TILE)) {
                let qd = qd * scale;
                for (s, &k) in s.iter_mut().zip(kd) {
                    *s = madd::<FMA>(qd, k, *s);
                }
            }
            s[visible..].fill(f32::NEG_INFINITY);

            // Online softmax: rescale earlier output when the running max grows.
            let tile_max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let o = &mut out[i * hd..][..hd];
            if tile_max > max[i] {
                let r = exp_approx(max[i] - tile_max); // 0 on the first tile
                o.iter_mut().for_each(|o| *o *= r);
                sum[i] *= r;
                max[i] = tile_max;
            }
            let mut p = [0f32; TILE];
            for (p, &s) in p.iter_mut().zip(&s) {
                *p = exp_approx(s - max[i]);
            }
            sum[i] += p.iter().sum::<f32>();
            for (&p, v) in p[..visible].iter().zip(values[j0 * hd..].chunks_exact(hd)) {
                for (o, &v) in o.iter_mut().zip(v) {
                    *o = madd::<FMA>(p, v, *o);
                }
            }
        }
    }
    for (o, &sum) in out.chunks_exact_mut(hd).zip(&sum) {
        let inv = 1.0 / sum;
        o.iter_mut().for_each(|o| *o *= inv);
    }
}

/// Index of element `d` of position `pos` in a tile-transposed key buffer.
#[inline]
pub fn key_index(pos: usize, d: usize, head_dim: usize) -> usize {
    (pos / TILE) * TILE * head_dim + d * TILE + pos % TILE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::Rng;

    #[test]
    fn exp_approx_is_accurate() {
        let mut worst = 0f32;
        let mut x = -87.0f32;
        while x < 88.0 {
            let rel = ((exp_approx(x) - x.exp()) / x.exp()).abs();
            worst = worst.max(rel);
            x += 0.013;
        }
        assert!(worst < 2e-6, "worst relative error {worst}");
        assert_eq!(exp_approx(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp_approx(-100.0), 0.0);
    }

    #[test]
    fn matches_naive_attention_with_causal_limits() {
        let mut rng = Rng::new(9);
        let (hd, n) = (24, 75); // ragged: 2 full tiles + 11
        let mut rand =
            |len: usize| -> Vec<f32> { (0..len).map(|_| rng.next_f32() * 6.0 - 3.0).collect() };
        let (keys, values) = (rand(n * hd), rand(n * hd));
        let limits = [1, 31, 32, 33, 64, 75];
        let q = rand(limits.len() * hd);

        let mut tiled = vec![0.0; n.div_ceil(TILE) * TILE * hd];
        for pos in 0..n {
            for d in 0..hd {
                tiled[key_index(pos, d, hd)] = keys[pos * hd + d];
            }
        }
        let mut out = vec![0.0; q.len()];
        attention_impl::<false>(&q, &tiled, &values, &limits, 0.3, &mut out);

        for (i, &limit) in limits.iter().enumerate() {
            let qi = &q[i * hd..][..hd];
            let s: Vec<f32> = (0..limit)
                .map(|j| {
                    qi.iter()
                        .zip(&keys[j * hd..][..hd])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * 0.3
                })
                .collect();
            let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let w: Vec<f32> = s.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = w.iter().sum();
            for d in 0..hd {
                let want = (0..limit).map(|j| w[j] * values[j * hd + d]).sum::<f32>() / total;
                let got = out[i * hd + d];
                assert!(
                    (got - want).abs() < 1e-4,
                    "query {i} dim {d}: {got} vs {want}"
                );
            }
        }
    }
}
