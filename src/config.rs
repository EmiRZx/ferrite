use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// The subset of a Hugging Face `config.json` that a Qwen3 dense model needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    head_dim: Option<usize>,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).context("parsing config.json")
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }
}
