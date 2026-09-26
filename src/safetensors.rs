//! Minimal zero-copy reader for the `.safetensors` format.
//!
//! Layout: `[u64 LE header_len][JSON header][raw tensor bytes]`. The header maps
//! each tensor name to its dtype, shape and byte range inside the data section.
//! We memory-map the file so tensors are only paged in when they are touched.

use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    BF16,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Dtype::F32,
            "BF16" => Dtype::BF16,
            other => bail!("unsupported dtype {other}"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Deserialize)]
struct RawInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct SafeTensors {
    mmap: Mmap,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the file is opened read-only and we never hand out mutable views.
        let mmap = unsafe { Mmap::map(&file)? };
        ensure!(mmap.len() >= 8, "file too small to be safetensors");

        let header_len = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
        let data_start = 8 + header_len;
        ensure!(data_start <= mmap.len(), "header length exceeds file size");

        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..data_start]).context("parsing safetensors header")?;
        let data_len = mmap.len() - data_start;

        let mut tensors = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let raw: RawInfo = serde_json::from_value(value)
                .with_context(|| format!("bad header entry for {name}"))?;
            let [start, end] = raw.data_offsets;
            let info = TensorInfo {
                dtype: Dtype::parse(&raw.dtype)?,
                shape: raw.shape,
                start,
                end,
            };
            ensure!(
                start <= end && end <= data_len,
                "{name}: offsets out of bounds"
            );
            ensure!(
                end - start == info.numel() * info.dtype.size(),
                "{name}: byte size does not match shape"
            );
            tensors.insert(name, info);
        }
        Ok(Self {
            mmap,
            data_start,
            tensors,
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor {name} not found"))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        Ok(&self.mmap[self.data_start + info.start..self.data_start + info.end])
    }

    /// Loads a tensor as f32, checking that its shape matches `shape`.
    pub fn load_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        ensure!(
            info.shape == shape,
            "{name}: expected shape {shape:?}, found {:?}",
            info.shape
        );
        let bytes = self.bytes(name)?;
        Ok(match info.dtype {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| bf16_to_f32(u16::from_le_bytes(b)))
                .collect(),
        })
    }
}

/// bf16 is the upper half of an f32, so widening is a shift.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
