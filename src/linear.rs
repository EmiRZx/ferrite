//! Bias-free linear layers (`y = W x`) over several weight formats.

use std::{fmt, str::FromStr};

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

use crate::{
    kernels::kernels,
    pool::pool,
    quant::{self, BLOCK, Q8Vec},
    safetensors::{SafeTensors, bf16_to_f32, f32_to_bf16},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    Q8,
    Q4,
}

impl DType {
    pub const ALL: [DType; 4] = [DType::F32, DType::Bf16, DType::Q8, DType::Q4];
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            DType::F32 => "f32",
            DType::Bf16 => "bf16",
            DType::Q8 => "q8",
            DType::Q4 => "q4",
        })
    }
}

impl FromStr for DType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" => DType::F32,
            "bf16" => DType::Bf16,
            "q8" => DType::Q8,
            "q4" => DType::Q4,
            _ => bail!("unknown dtype {s:?} (expected f32, bf16, q8 or q4)"),
        })
    }
}

enum Storage {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
    Q8 { qs: Vec<i8>, scales: Vec<f32> },
    Q4 { qs: Vec<u8>, scales: Vec<f32> },
}

/// A weight matrix stored row-major as `[rows, cols]` (the `nn.Linear` layout).
pub struct Linear {
    pub rows: usize,
    pub cols: usize,
    storage: Storage,
}

/// Rows claimed per work item; small enough to balance, large enough to amortize.
const ROWS_PER_TASK: usize = 16;

impl Linear {
    pub fn load(
        st: &SafeTensors,
        name: &str,
        rows: usize,
        cols: usize,
        dtype: DType,
    ) -> Result<Self> {
        let shape = [rows, cols];
        match dtype {
            // Keep bf16 checkpoints bit-exact instead of round-tripping through f32.
            DType::Bf16 => Ok(Self {
                rows,
                cols,
                storage: Storage::Bf16(st.load_bf16(name, &shape)?),
            }),
            _ => Self::from_f32(st.load_f32(name, &shape)?, rows, cols, dtype)
                .with_context(|| format!("loading {name}")),
        }
    }

    /// Loads several matrices with the same input and stacks their rows into one
    /// (e.g. Q/K/V), so a single matvec replaces several small ones.
    pub fn load_fused(
        st: &SafeTensors,
        parts: &[(&str, usize)],
        cols: usize,
        dtype: DType,
    ) -> Result<Self> {
        let mut w = Vec::new();
        for &(name, rows) in parts {
            w.extend(st.load_f32(name, &[rows, cols])?);
        }
        let rows = parts.iter().map(|p| p.1).sum();
        Self::from_f32(w, rows, cols, dtype)
    }

    pub fn from_f32(w: Vec<f32>, rows: usize, cols: usize, dtype: DType) -> Result<Self> {
        ensure!(
            w.len() == rows * cols,
            "expected {rows}x{cols} values, got {}",
            w.len()
        );
        if matches!(dtype, DType::Q8 | DType::Q4) {
            ensure!(
                cols.is_multiple_of(BLOCK),
                "{cols} columns is not a multiple of {BLOCK}"
            );
        }
        let storage = match dtype {
            DType::F32 => Storage::F32(w),
            DType::Bf16 => Storage::Bf16(w.iter().map(|&v| f32_to_bf16(v)).collect()),
            DType::Q8 => {
                let mut qs = vec![0i8; rows * cols];
                let mut scales = vec![0.0; rows * cols / BLOCK];
                qs.par_chunks_mut(cols)
                    .zip(scales.par_chunks_mut(cols / BLOCK))
                    .zip(w.par_chunks(cols))
                    .for_each(|((q, s), w)| quant::quantize_q8(w, q, s));
                Storage::Q8 { qs, scales }
            }
            DType::Q4 => {
                let mut qs = vec![0u8; rows * cols / 2];
                let mut scales = vec![0.0; rows * cols / BLOCK];
                qs.par_chunks_mut(cols / 2)
                    .zip(scales.par_chunks_mut(cols / BLOCK))
                    .zip(w.par_chunks(cols))
                    .for_each(|((q, s), w)| quant::quantize_q4(w, q, s));
                Storage::Q4 { qs, scales }
            }
        };
        Ok(Self {
            rows,
            cols,
            storage,
        })
    }

    pub fn dtype(&self) -> DType {
        match self.storage {
            Storage::F32(_) => DType::F32,
            Storage::Bf16(_) => DType::Bf16,
            Storage::Q8 { .. } => DType::Q8,
            Storage::Q4 { .. } => DType::Q4,
        }
    }

    /// `out = W x`. `xq` is scratch space for the quantized activations.
    pub fn forward(&self, out: &mut [f32], x: &[f32], xq: &mut Q8Vec) {
        assert_eq!(out.len(), self.rows);
        assert_eq!(x.len(), self.cols);
        let k = kernels();
        let (n, nb) = (self.cols, self.cols / BLOCK);
        match &self.storage {
            Storage::F32(w) => par_rows(out, |r| (k.dot_f32)(&w[r * n..][..n], x)),
            Storage::Bf16(w) => par_rows(out, |r| (k.dot_bf16)(&w[r * n..][..n], x)),
            Storage::Q8 { qs, scales } => {
                xq.quantize(x);
                let xq = &*xq;
                par_rows(out, |r| {
                    (k.dot_q8)(&qs[r * n..][..n], &scales[r * nb..][..nb], xq)
                });
            }
            Storage::Q4 { qs, scales } => {
                xq.quantize(x);
                let xq = &*xq;
                par_rows(out, |r| {
                    (k.dot_q4)(&qs[r * n / 2..][..n / 2], &scales[r * nb..][..nb], xq)
                });
            }
        }
    }

    /// Dequantizes row `r` into `out` (used for embedding lookups).
    pub fn row(&self, r: usize, out: &mut [f32]) {
        let (n, nb) = (self.cols, self.cols / BLOCK);
        match &self.storage {
            Storage::F32(w) => out.copy_from_slice(&w[r * n..][..n]),
            Storage::Bf16(w) => {
                for (o, &b) in out.iter_mut().zip(&w[r * n..][..n]) {
                    *o = bf16_to_f32(b);
                }
            }
            Storage::Q8 { qs, scales } => {
                quant::dequantize_q8(&qs[r * n..][..n], &scales[r * nb..][..nb], out)
            }
            Storage::Q4 { qs, scales } => {
                quant::dequantize_q4(&qs[r * n / 2..][..n / 2], &scales[r * nb..][..nb], out)
            }
        }
    }

    /// Bytes of weight data one forward pass streams from memory.
    pub fn bytes(&self) -> usize {
        match &self.storage {
            Storage::F32(w) => w.len() * 4,
            Storage::Bf16(w) => w.len() * 2,
            Storage::Q8 { qs, scales } => qs.len() + scales.len() * 4,
            Storage::Q4 { qs, scales } => qs.len() + scales.len() * 4,
        }
    }
}

fn par_rows(out: &mut [f32], f: impl Fn(usize) -> f32 + Sync) {
    pool().par_map(out, ROWS_PER_TASK, f);
}
