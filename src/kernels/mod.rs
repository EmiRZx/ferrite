//! Compute kernels with runtime CPU feature dispatch.
//!
//! Each instruction set gets a [`Kernels`] table of plain function pointers. The best
//! table the CPU supports is picked once at startup; `FERRITE_ISA=scalar|avx2|avx512`
//! overrides the choice (handy for benchmarking one ISA against another).

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
pub mod scalar;

use std::{fmt, str::FromStr, sync::OnceLock};

use anyhow::{Result, bail};

use crate::quant::{BLOCK, Q8Vec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512,
}

impl Isa {
    pub const ALL: [Isa; 3] = [Isa::Scalar, Isa::Avx2, Isa::Avx512];

    pub fn is_supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                Isa::Avx2.is_supported()
                    && is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }

    pub fn best() -> Isa {
        Isa::ALL
            .into_iter()
            .rev()
            .find(|isa| isa.is_supported())
            .unwrap()
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
        })
    }
}

impl FromStr for Isa {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "scalar" => Isa::Scalar,
            "avx2" => Isa::Avx2,
            "avx512" => Isa::Avx512,
            _ => bail!("unknown ISA {s:?} (expected scalar, avx2 or avx512)"),
        })
    }
}

/// Dot product of block-quantized weights `(quants, scales)` with Q8 activations.
/// `W` is `i8` for Q8 and packed nibbles (`u8`) for Q4.
pub type QuantDot<W> = fn(&[W], &[f32], &Q8Vec) -> f32;

pub struct Kernels {
    pub isa: Isa,
    pub dot_f32: fn(&[f32], &[f32]) -> f32,
    /// `y += a * x`
    pub axpy_f32: fn(&mut [f32], f32, &[f32]),
    pub dot_bf16: fn(&[u16], &[f32]) -> f32,
    pub dot_q8: QuantDot<i8>,
    pub dot_q4: QuantDot<u8>,
    /// Quantizes activations to Q8 (bit-identical across ISAs).
    pub quantize_q8: fn(&[f32], &mut [i8], &mut [f32]),
}

static SCALAR: Kernels = Kernels {
    isa: Isa::Scalar,
    dot_f32: scalar::dot_f32,
    axpy_f32: scalar::axpy_f32,
    dot_bf16: scalar::dot_bf16,
    dot_q8: |wq, ws, x| scalar::dot_q8(wq, ws, &x.qs, &x.scales),
    dot_q4: |wq, ws, x| scalar::dot_q4(wq, ws, &x.qs, &x.scales),
    quantize_q8: crate::quant::quantize_q8,
};

// SAFETY (all `unsafe` blocks below): these tables are only reachable through
// `Kernels::for_isa`, which returns them only when the CPU supports the ISA.
#[cfg(target_arch = "x86_64")]
static AVX2: Kernels = Kernels {
    isa: Isa::Avx2,
    dot_f32: |a, b| unsafe { avx2::dot_f32(a, b) },
    axpy_f32: |y, a, x| unsafe { avx2::axpy_f32(y, a, x) },
    dot_bf16: |w, x| unsafe { avx2::dot_bf16(w, x) },
    dot_q8: |wq, ws, x| unsafe { avx2::dot_q8(wq, ws, &x.qs, &x.scales) },
    dot_q4: |wq, ws, x| unsafe { avx2::dot_q4(wq, ws, &x.qs, &x.scales) },
    quantize_q8: |x, qs, scales| unsafe { avx2::quantize_q8(x, qs, scales) },
};

#[cfg(target_arch = "x86_64")]
static AVX512: Kernels = Kernels {
    isa: Isa::Avx512,
    dot_f32: |a, b| unsafe { avx512::dot_f32(a, b) },
    axpy_f32: |y, a, x| unsafe { avx512::axpy_f32(y, a, x) },
    dot_bf16: |w, x| unsafe { avx512::dot_bf16(w, x) },
    dot_q8: |wq, ws, x| unsafe { avx512::dot_q8(wq, ws, x) },
    dot_q4: |wq, ws, x| unsafe { avx512::dot_q4(wq, ws, x) },
    quantize_q8: |x, qs, scales| unsafe { avx512::quantize_q8(x, qs, scales) },
};

impl Kernels {
    /// The kernel table for `isa`, or `None` if this CPU cannot run it.
    pub fn for_isa(isa: Isa) -> Option<&'static Kernels> {
        if !isa.is_supported() {
            return None;
        }
        match isa {
            Isa::Scalar => Some(&SCALAR),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => Some(&AVX2),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => Some(&AVX512),
            #[cfg(not(target_arch = "x86_64"))]
            _ => None,
        }
    }
}

/// The kernels used by the model: the best supported ISA unless `FERRITE_ISA` says otherwise.
pub fn kernels() -> &'static Kernels {
    static SELECTED: OnceLock<&'static Kernels> = OnceLock::new();
    SELECTED.get_or_init(|| {
        let requested = std::env::var("FERRITE_ISA").ok().map(|s| s.parse::<Isa>());
        let isa = match requested {
            Some(Ok(isa)) if isa.is_supported() => isa,
            Some(Ok(isa)) => panic!("FERRITE_ISA={isa} is not supported by this CPU"),
            Some(Err(e)) => panic!("{e}"),
            None => Isa::best(),
        };
        Kernels::for_isa(isa).unwrap()
    })
}

/// Shape check shared by the quantized dot products; `n` is the number of weights.
#[inline]
fn check_q8(n: usize, ws: &[f32], xq: &[i8], xs: &[f32]) {
    assert!(
        n == ws.len() * BLOCK && xq.len() == n && xs.len() == ws.len(),
        "quantized dot product shape mismatch"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        quant::{quantize_q4, quantize_q8},
        sampler::Rng,
    };

    fn random(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    fn close(a: f32, b: f32, what: &str) {
        let tol = 1e-4 * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
    }

    fn simd_kernels() -> Vec<&'static Kernels> {
        Isa::ALL[1..]
            .iter()
            .filter_map(|&i| Kernels::for_isa(i))
            .collect()
    }

    #[test]
    fn f32_kernels_match_scalar() {
        let mut rng = Rng::new(1);
        for n in [0, 1, 7, 8, 15, 16, 31, 33, 63, 64, 65, 100, 1024, 1027] {
            let (a, b) = (random(&mut rng, n), random(&mut rng, n));
            let bf: Vec<u16> = a.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            for k in simd_kernels() {
                let what = format!("{} n={n}", k.isa);
                close(
                    (k.dot_f32)(&a, &b),
                    scalar::dot_f32(&a, &b),
                    &format!("dot_f32 {what}"),
                );
                close(
                    (k.dot_bf16)(&bf, &b),
                    scalar::dot_bf16(&bf, &b),
                    &format!("dot_bf16 {what}"),
                );

                let (mut y1, mut y2) = (b.clone(), b.clone());
                (k.axpy_f32)(&mut y1, 0.37, &a);
                scalar::axpy_f32(&mut y2, 0.37, &a);
                for (u, v) in y1.iter().zip(&y2) {
                    close(*u, *v, &format!("axpy {what}"));
                }
            }
        }
    }

    #[test]
    fn quantized_kernels_match_scalar() {
        let mut rng = Rng::new(2);
        // AVX-512 works in groups of 16 blocks; the odd counts exercise its scalar tail.
        for blocks in [1, 2, 3, 8, 15, 16, 17, 31, 32, 96] {
            let n = blocks * BLOCK;
            let (w, x) = (random(&mut rng, n), random(&mut rng, n));
            let (mut w8, mut w8s) = (vec![0i8; n], vec![0.0; blocks]);
            let (mut w4, mut w4s) = (vec![0u8; n / 2], vec![0.0; blocks]);
            quantize_q8(&w, &mut w8, &mut w8s);
            quantize_q4(&w, &mut w4, &mut w4s);
            let mut xv = Q8Vec::default();
            xv.quantize(&x);
            let (x8, x8s) = (&xv.qs, &xv.scales);

            let want8 = scalar::dot_q8(&w8, &w8s, x8, x8s);
            let want4 = scalar::dot_q4(&w4, &w4s, x8, x8s);
            // Sanity check the formats themselves against the f32 dot product.
            let exact = scalar::dot_f32(&w, &x);
            assert!(
                (want8 - exact).abs() < 0.02 * (n as f32).sqrt(),
                "q8 {want8} vs {exact}"
            );
            assert!(
                (want4 - exact).abs() < 0.2 * (n as f32).sqrt(),
                "q4 {want4} vs {exact}"
            );

            for k in simd_kernels() {
                let what = format!("{} blocks={blocks}", k.isa);
                let (mut q, mut s) = (vec![0i8; n], vec![0.0; blocks]);
                (k.quantize_q8)(&x, &mut q, &mut s);
                assert_eq!((&q, &s), (x8, x8s), "quantize_q8 {what}");
                close((k.dot_q8)(&w8, &w8s, &xv), want8, &format!("dot_q8 {what}"));
                close((k.dot_q4)(&w4, &w4s, &xv), want4, &format!("dot_q4 {what}"));
            }
        }
    }
}
