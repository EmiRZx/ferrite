//! Qwen3 dense decoder, implemented directly on flat f32 buffers.
//!
//! Per layer:
//! ```text
//! h = x + o_proj(attn(rope(q_norm(q_proj(n1))), rope(k_norm(k_proj(n1))), v_proj(n1)))   n1 = rmsnorm(x)
//! x = h + down_proj(silu(gate_proj(n2)) * up_proj(n2))                                     n2 = rmsnorm(h)
//! ```

use std::path::Path;

use crate::{
    config::Config,
    kernels::kernels,
    linear::{DType, Linear},
    ops::{self, Rope},
    pool::pool,
    quant::Q8Vec,
    safetensors::SafeTensors,
};
use anyhow::{Result, ensure};

struct Layer {
    input_norm: Vec<f32>,
    /// q_proj, k_proj and v_proj stacked into one `[q_dim + 2 * kv_dim, hidden]` matrix.
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_norm: Vec<f32>,
    /// gate_proj and up_proj stacked into one `[2 * intermediate, hidden]` matrix.
    gate_up_proj: Linear,
    down_proj: Linear,
}

pub struct Model {
    pub config: Config,
    embed: Linear,
    lm_head: Option<Linear>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    rope: Rope,
    pub max_seq: usize,
}

impl Model {
    /// Loads the model with its linear layers converted to `dtype`. With Q4 the
    /// embedding / LM head stay at Q8: 4-bit there costs noticeably more accuracy.
    pub fn load(dir: impl AsRef<Path>, max_seq: usize, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::load(dir.join("config.json"))?;
        let st = SafeTensors::open(dir.join("model.safetensors"))?;
        let c = &config;
        let (d, hd) = (c.hidden_size, c.head_dim());
        let max_seq = max_seq.min(c.max_position_embeddings);
        let head_dtype = if dtype == DType::Q4 { DType::Q8 } else { dtype };

        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let p = format!("model.layers.{i}");
                let lin = |name: &str, rows, cols| {
                    Linear::load(&st, &format!("{p}.{name}"), rows, cols, dtype)
                };
                let fused = |parts: &[(&str, usize)]| {
                    let names: Vec<(String, usize)> = parts
                        .iter()
                        .map(|&(n, r)| (format!("{p}.{n}"), r))
                        .collect();
                    let names: Vec<(&str, usize)> =
                        names.iter().map(|(n, r)| (n.as_str(), *r)).collect();
                    Linear::load_fused(&st, &names, d, dtype)
                };
                let vec = |name: &str, len| st.load_f32(&format!("{p}.{name}"), &[len]);
                Ok(Layer {
                    input_norm: vec("input_layernorm.weight", d)?,
                    qkv_proj: fused(&[
                        ("self_attn.q_proj.weight", c.q_dim()),
                        ("self_attn.k_proj.weight", c.kv_dim()),
                        ("self_attn.v_proj.weight", c.kv_dim()),
                    ])?,
                    o_proj: lin("self_attn.o_proj.weight", d, c.q_dim())?,
                    q_norm: vec("self_attn.q_norm.weight", hd)?,
                    k_norm: vec("self_attn.k_norm.weight", hd)?,
                    post_norm: vec("post_attention_layernorm.weight", d)?,
                    gate_up_proj: fused(&[
                        ("mlp.gate_proj.weight", c.intermediate_size),
                        ("mlp.up_proj.weight", c.intermediate_size),
                    ])?,
                    down_proj: lin("mlp.down_proj.weight", d, c.intermediate_size)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let lm_head = if c.tie_word_embeddings && !st.contains("lm_head.weight") {
            None
        } else {
            Some(Linear::load(
                &st,
                "lm_head.weight",
                c.vocab_size,
                d,
                head_dtype,
            )?)
        };

        Ok(Self {
            embed: Linear::load(
                &st,
                "model.embed_tokens.weight",
                c.vocab_size,
                d,
                head_dtype,
            )?,
            lm_head,
            layers,
            final_norm: st.load_f32("model.norm.weight", &[d])?,
            rope: Rope::new(hd, max_seq, c.rope_theta),
            max_seq,
            config,
        })
    }

    pub fn dtype(&self) -> DType {
        self.layers[0].qkv_proj.dtype()
    }

    fn lm_head(&self) -> &Linear {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    pub fn weight_bytes(&self) -> usize {
        let per_layer: usize = self
            .layers
            .iter()
            .flat_map(|l| [&l.qkv_proj, &l.o_proj, &l.gate_up_proj, &l.down_proj])
            .map(Linear::bytes)
            .sum();
        per_layer + self.lm_head().bytes()
    }

    pub fn new_session(&self) -> Session<'_> {
        Session::new(self)
    }
}

/// Per-sequence KV cache, laid out `[layer][kv_head][pos][head_dim]` so that the
/// attention loop for one head streams through contiguous memory.
struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    max_seq: usize,
    n_kv: usize,
    hd: usize,
}

impl KvCache {
    fn new(c: &Config, max_seq: usize) -> Self {
        let len = c.num_hidden_layers * c.num_key_value_heads * max_seq * c.head_dim();
        Self {
            k: vec![0.0; len],
            v: vec![0.0; len],
            max_seq,
            n_kv: c.num_key_value_heads,
            hd: c.head_dim(),
        }
    }

    fn head_offset(&self, layer: usize, head: usize) -> usize {
        (layer * self.n_kv + head) * self.max_seq * self.hd
    }
}

/// Scratch buffers reused across forward passes.
struct Buffers {
    x: Vec<f32>,
    xb: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    scores: Vec<f32>,
    gate_up: Vec<f32>,
    logits: Vec<f32>,
    xq: Q8Vec,
}

/// A single generation stream: owns its KV cache and current position.
pub struct Session<'m> {
    model: &'m Model,
    cache: KvCache,
    buf: Buffers,
    pos: usize,
}

impl<'m> Session<'m> {
    fn new(model: &'m Model) -> Self {
        let c = &model.config;
        Self {
            model,
            cache: KvCache::new(c, model.max_seq),
            buf: Buffers {
                x: vec![0.0; c.hidden_size],
                xb: vec![0.0; c.hidden_size],
                qkv: vec![0.0; c.q_dim() + 2 * c.kv_dim()],
                att: vec![0.0; c.q_dim()],
                scores: vec![0.0; c.num_attention_heads * model.max_seq],
                gate_up: vec![0.0; 2 * c.intermediate_size],
                logits: vec![0.0; c.vocab_size],
                xq: Q8Vec::default(),
            },
            pos: 0,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Runs one token through the model at the current position and returns the
    /// logits for the next token.
    pub fn forward(&mut self, token: u32) -> Result<&[f32]> {
        let m = self.model;
        let c = &m.config;
        ensure!(
            self.pos < m.max_seq,
            "context window of {} tokens is full",
            m.max_seq
        );
        ensure!(
            (token as usize) < c.vocab_size,
            "token id {token} out of range"
        );

        let pos = self.pos;
        let hd = c.head_dim();
        let group = c.num_attention_heads / c.num_key_value_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = c.rms_norm_eps;
        let k = kernels();
        let b = &mut self.buf;

        m.embed.row(token as usize, &mut b.x);

        for (li, layer) in m.layers.iter().enumerate() {
            // --- attention ---
            ops::rmsnorm(&mut b.xb, &b.x, &layer.input_norm, eps);
            layer.qkv_proj.forward(&mut b.qkv, &b.xb, &mut b.xq);
            let (q, kv) = b.qkv.split_at_mut(c.q_dim());
            let (keys, values) = kv.split_at_mut(c.kv_dim());

            for q in q.chunks_exact_mut(hd) {
                ops::rmsnorm_inplace(q, &layer.q_norm, eps);
                m.rope.apply(q, pos);
            }
            for (h, (key, v)) in keys
                .chunks_exact_mut(hd)
                .zip(values.chunks_exact(hd))
                .enumerate()
            {
                ops::rmsnorm_inplace(key, &layer.k_norm, eps);
                m.rope.apply(key, pos);
                let off = self.cache.head_offset(li, h) + pos * hd;
                self.cache.k[off..off + hd].copy_from_slice(key);
                self.cache.v[off..off + hd].copy_from_slice(v);
            }

            let cache = &self.cache;
            let q = &*q;
            let seq = pos + 1;
            pool().par_chunks2(
                &mut b.att,
                hd,
                &mut b.scores,
                m.max_seq,
                |h, out, scores| {
                    let base = cache.head_offset(li, h / group);
                    let keys = &cache.k[base..base + seq * hd];
                    let values = &cache.v[base..base + seq * hd];
                    let q = &q[h * hd..][..hd];
                    let scores = &mut scores[..seq];
                    for (s, key) in scores.iter_mut().zip(keys.chunks_exact(hd)) {
                        *s = (k.dot_f32)(q, key) * scale;
                    }
                    ops::softmax_inplace(scores);
                    out.fill(0.0);
                    for (&p, v) in scores.iter().zip(values.chunks_exact(hd)) {
                        (k.axpy_f32)(out, p, v);
                    }
                },
            );

            layer.o_proj.forward(&mut b.xb, &b.att, &mut b.xq);
            ops::add_inplace(&mut b.x, &b.xb);

            // --- MLP (SwiGLU) ---
            ops::rmsnorm(&mut b.xb, &b.x, &layer.post_norm, eps);
            layer.gate_up_proj.forward(&mut b.gate_up, &b.xb, &mut b.xq);
            let (gate, up) = b.gate_up.split_at_mut(c.intermediate_size);
            for (g, &u) in gate.iter_mut().zip(&*up) {
                *g = ops::silu(*g) * u;
            }
            layer.down_proj.forward(&mut b.xb, gate, &mut b.xq);
            ops::add_inplace(&mut b.x, &b.xb);
        }

        ops::rmsnorm_inplace(&mut b.x, &m.final_norm, eps);
        m.lm_head().forward(&mut b.logits, &b.x, &mut b.xq);
        self.pos += 1;
        Ok(&b.logits)
    }
}
