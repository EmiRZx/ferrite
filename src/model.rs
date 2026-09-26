//! Qwen3 dense decoder, implemented directly on flat f32 buffers.
//!
//! Per layer:
//! ```text
//! h = x + o_proj(attn(rope(q_norm(q_proj(n1))), rope(k_norm(k_proj(n1))), v_proj(n1)))   n1 = rmsnorm(x)
//! x = h + down_proj(silu(gate_proj(n2)) * up_proj(n2))                                     n2 = rmsnorm(h)
//! ```

use std::path::Path;

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::{
    config::Config,
    ops::{self, Rope},
    safetensors::SafeTensors,
};

/// A dense `nn.Linear` without bias, stored row-major as `[rows, cols]`.
pub struct Linear {
    pub rows: usize,
    pub cols: usize,
    w: Vec<f32>,
}

impl Linear {
    fn load(st: &SafeTensors, name: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            rows,
            cols,
            w: st.load_f32(name, &[rows, cols])?,
        })
    }

    pub fn forward(&self, out: &mut [f32], x: &[f32]) {
        ops::matvec(out, &self.w, x);
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.w[i * self.cols..][..self.cols]
    }

    pub fn bytes(&self) -> usize {
        self.w.len() * size_of::<f32>()
    }
}

struct Layer {
    input_norm: Vec<f32>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_norm: Vec<f32>,
    gate_proj: Linear,
    up_proj: Linear,
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
    pub fn load(dir: impl AsRef<Path>, max_seq: usize) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::load(dir.join("config.json"))?;
        let st = SafeTensors::open(dir.join("model.safetensors"))?;
        let c = &config;
        let (d, hd) = (c.hidden_size, c.head_dim());
        let max_seq = max_seq.min(c.max_position_embeddings);

        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let p = format!("model.layers.{i}");
                let lin =
                    |name: &str, rows, cols| Linear::load(&st, &format!("{p}.{name}"), rows, cols);
                let vec = |name: &str, len| st.load_f32(&format!("{p}.{name}"), &[len]);
                Ok(Layer {
                    input_norm: vec("input_layernorm.weight", d)?,
                    q_proj: lin("self_attn.q_proj.weight", c.q_dim(), d)?,
                    k_proj: lin("self_attn.k_proj.weight", c.kv_dim(), d)?,
                    v_proj: lin("self_attn.v_proj.weight", c.kv_dim(), d)?,
                    o_proj: lin("self_attn.o_proj.weight", d, c.q_dim())?,
                    q_norm: vec("self_attn.q_norm.weight", hd)?,
                    k_norm: vec("self_attn.k_norm.weight", hd)?,
                    post_norm: vec("post_attention_layernorm.weight", d)?,
                    gate_proj: lin("mlp.gate_proj.weight", c.intermediate_size, d)?,
                    up_proj: lin("mlp.up_proj.weight", c.intermediate_size, d)?,
                    down_proj: lin("mlp.down_proj.weight", d, c.intermediate_size)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let lm_head = if c.tie_word_embeddings && !st.contains("lm_head.weight") {
            None
        } else {
            Some(Linear::load(&st, "lm_head.weight", c.vocab_size, d)?)
        };

        Ok(Self {
            embed: Linear::load(&st, "model.embed_tokens.weight", c.vocab_size, d)?,
            lm_head,
            layers,
            final_norm: st.load_f32("model.norm.weight", &[d])?,
            rope: Rope::new(hd, max_seq, c.rope_theta),
            max_seq,
            config,
        })
    }

    fn lm_head(&self) -> &Linear {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    pub fn weight_bytes(&self) -> usize {
        let per_layer: usize = self
            .layers
            .iter()
            .flat_map(|l| {
                [
                    &l.q_proj,
                    &l.k_proj,
                    &l.v_proj,
                    &l.o_proj,
                    &l.gate_proj,
                    &l.up_proj,
                    &l.down_proj,
                ]
            })
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
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    logits: Vec<f32>,
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
                q: vec![0.0; c.q_dim()],
                k: vec![0.0; c.kv_dim()],
                v: vec![0.0; c.kv_dim()],
                att: vec![0.0; c.q_dim()],
                scores: vec![0.0; c.num_attention_heads * model.max_seq],
                gate: vec![0.0; c.intermediate_size],
                up: vec![0.0; c.intermediate_size],
                logits: vec![0.0; c.vocab_size],
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
        let b = &mut self.buf;

        b.x.copy_from_slice(m.embed.row(token as usize));

        for (li, layer) in m.layers.iter().enumerate() {
            // --- attention ---
            ops::rmsnorm(&mut b.xb, &b.x, &layer.input_norm, eps);
            layer.q_proj.forward(&mut b.q, &b.xb);
            layer.k_proj.forward(&mut b.k, &b.xb);
            layer.v_proj.forward(&mut b.v, &b.xb);

            for q in b.q.chunks_exact_mut(hd) {
                ops::rmsnorm_inplace(q, &layer.q_norm, eps);
                m.rope.apply(q, pos);
            }
            for (h, (k, v)) in
                b.k.chunks_exact_mut(hd)
                    .zip(b.v.chunks_exact(hd))
                    .enumerate()
            {
                ops::rmsnorm_inplace(k, &layer.k_norm, eps);
                m.rope.apply(k, pos);
                let off = self.cache.head_offset(li, h) + pos * hd;
                self.cache.k[off..off + hd].copy_from_slice(k);
                self.cache.v[off..off + hd].copy_from_slice(v);
            }

            let cache = &self.cache;
            let seq = pos + 1;
            b.att
                .par_chunks_exact_mut(hd)
                .zip(b.scores.par_chunks_exact_mut(m.max_seq))
                .zip(b.q.par_chunks_exact(hd))
                .enumerate()
                .for_each(|(h, ((out, scores), q))| {
                    let base = cache.head_offset(li, h / group);
                    let keys = &cache.k[base..base + seq * hd];
                    let values = &cache.v[base..base + seq * hd];
                    let scores = &mut scores[..seq];
                    for (s, k) in scores.iter_mut().zip(keys.chunks_exact(hd)) {
                        *s = ops::dot(q, k) * scale;
                    }
                    ops::softmax_inplace(scores);
                    out.fill(0.0);
                    for (&p, v) in scores.iter().zip(values.chunks_exact(hd)) {
                        out.iter_mut().zip(v).for_each(|(o, &v)| *o += p * v);
                    }
                });

            layer.o_proj.forward(&mut b.xb, &b.att);
            ops::add_inplace(&mut b.x, &b.xb);

            // --- MLP (SwiGLU) ---
            ops::rmsnorm(&mut b.xb, &b.x, &layer.post_norm, eps);
            layer.gate_proj.forward(&mut b.gate, &b.xb);
            layer.up_proj.forward(&mut b.up, &b.xb);
            for (g, &u) in b.gate.iter_mut().zip(&b.up) {
                *g = ops::silu(*g) * u;
            }
            layer.down_proj.forward(&mut b.xb, &b.gate);
            ops::add_inplace(&mut b.x, &b.xb);
        }

        ops::rmsnorm_inplace(&mut b.x, &m.final_norm, eps);
        m.lm_head().forward(&mut b.logits, &b.x);
        self.pos += 1;
        Ok(&b.logits)
    }
}
