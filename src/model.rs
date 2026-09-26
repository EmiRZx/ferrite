//! Qwen3 dense decoder, implemented directly on flat f32 buffers.
//!
//! Per layer:
//! ```text
//! h = x + o_proj(attn(rope(q_norm(q_proj(n1))), rope(k_norm(k_proj(n1))), v_proj(n1)))   n1 = rmsnorm(x)
//! x = h + down_proj(silu(gate_proj(n2)) * up_proj(n2))                                     n2 = rmsnorm(h)
//! ```

use std::{cell::RefCell, path::Path};

use crate::{
    config::Config,
    kernels::{
        attention::{MAX_HEAD_DIM, MAX_QUERIES, TILE as ATT_TILE, key_index},
        kernels,
    },
    linear::{DType, Linear},
    ops::{self, Rope},
    pool::pool,
    profile::{self, Section},
    quant::Q8Vec,
    safetensors::SafeTensors,
};
use anyhow::{Result, bail, ensure};

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
        ensure!(hd <= MAX_HEAD_DIM, "head_dim {hd} exceeds {MAX_HEAD_DIM}");
        ensure!(
            c.num_attention_heads % c.num_key_value_heads == 0
                && c.num_attention_heads / c.num_key_value_heads <= MAX_QUERIES,
            "unsupported GQA grouping"
        );
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

/// Per-sequence KV cache with one contiguous region per `(layer, kv_head)`.
/// Values are `[pos][head_dim]`. Keys are transposed in tiles of 32 positions
/// (`[tile][head_dim][32]`, see `kernels::attention`) so attention can score a
/// whole tile with vector FMAs.
struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Capacity in positions, rounded up to whole key tiles.
    capacity: usize,
    n_kv: usize,
    hd: usize,
}

impl KvCache {
    fn new(c: &Config, max_seq: usize) -> Self {
        let capacity = max_seq.div_ceil(ATT_TILE) * ATT_TILE;
        let len = c.num_hidden_layers * c.num_key_value_heads * capacity * c.head_dim();
        Self {
            k: vec![0.0; len],
            v: vec![0.0; len],
            capacity,
            n_kv: c.num_key_value_heads,
            hd: c.head_dim(),
        }
    }

    fn head_offset(&self, layer: usize, head: usize) -> usize {
        (layer * self.n_kv + head) * self.capacity * self.hd
    }

    fn store(&mut self, layer: usize, head: usize, pos: usize, key: &[f32], value: &[f32]) {
        let base = self.head_offset(layer, head);
        for (d, &k) in key.iter().enumerate() {
            self.k[base + key_index(pos, d, self.hd)] = k;
        }
        self.v[base + pos * self.hd..][..self.hd].copy_from_slice(value);
    }
}

/// Queries per attention task. Fewer than the kernel's maximum so that a 64-token
/// prefill chunk still splits into enough tasks to keep every core busy.
const ATTENTION_BLOCK_QUERIES: usize = 16;

thread_local! {
    /// Per-thread gather / scatter buffers for attention queries and outputs.
    static ATTENTION_SCRATCH: RefCell<(Vec<f32>, Vec<f32>)> =
        RefCell::new((vec![0.0; MAX_QUERIES * MAX_HEAD_DIM], vec![0.0; MAX_QUERIES * MAX_HEAD_DIM]));
}

/// Tokens per forward pass during prefill. Larger batches reuse each weight more
/// times, but the activations stop fitting in L2.
pub const PREFILL_CHUNK: usize = 64;

/// Scratch buffers reused across forward passes, sized for [`PREFILL_CHUNK`] tokens.
/// Each is row-major `[tokens, width]`.
struct Buffers {
    x: Vec<f32>,
    xb: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    gate_up: Vec<f32>,
    mlp: Vec<f32>,
    /// `[vocab]` for the last token, or `[tokens, vocab]` when all logits are requested.
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
        let n = PREFILL_CHUNK;
        Self {
            model,
            cache: KvCache::new(c, model.max_seq),
            buf: Buffers {
                x: vec![0.0; n * c.hidden_size],
                xb: vec![0.0; n * c.hidden_size],
                qkv: vec![0.0; n * (c.q_dim() + 2 * c.kv_dim())],
                att: vec![0.0; n * c.q_dim()],
                gate_up: vec![0.0; n * 2 * c.intermediate_size],
                mlp: vec![0.0; n * c.intermediate_size],
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

    /// Runs one token at the current position; returns the next-token logits.
    pub fn forward(&mut self, token: u32) -> Result<&[f32]> {
        self.forward_batch(&[token], false)?;
        Ok(&self.buf.logits)
    }

    /// Feeds a prompt in batches of [`PREFILL_CHUNK`] tokens; returns the logits
    /// after its last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        ensure!(!tokens.is_empty(), "empty prompt");
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            self.forward_batch(chunk, false)?;
        }
        Ok(&self.buf.logits)
    }

    /// Runs up to [`PREFILL_CHUNK`] tokens and returns the logits after *each* of
    /// them, `[tokens, vocab]` (used for evaluation).
    pub fn forward_all(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        ensure!(
            tokens.len() <= PREFILL_CHUNK,
            "at most {PREFILL_CHUNK} tokens per call"
        );
        self.forward_batch(tokens, true)?;
        Ok(&self.buf.logits)
    }

    fn forward_batch(&mut self, tokens: &[u32], all_logits: bool) -> Result<()> {
        let m = self.model;
        let c = &m.config;
        let n = tokens.len();
        debug_assert!((1..=PREFILL_CHUNK).contains(&n));
        ensure!(
            self.pos + n <= m.max_seq,
            "context window of {} tokens is full",
            m.max_seq
        );
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= c.vocab_size) {
            bail!("token id {t} out of range");
        }

        let (d, hd, heads) = (c.hidden_size, c.head_dim(), c.num_attention_heads);
        let (q_dim, kv_dim, inter) = (c.q_dim(), c.kv_dim(), c.intermediate_size);
        let qkv_dim = q_dim + 2 * kv_dim;
        let n_kv = c.num_key_value_heads;
        let group = heads / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = c.rms_norm_eps;
        let k = kernels();
        let pos0 = self.pos;
        let b = &mut self.buf;
        let (x, xb) = (&mut b.x[..n * d], &mut b.xb[..n * d]);
        let (qkv, att) = (&mut b.qkv[..n * qkv_dim], &mut b.att[..n * q_dim]);
        let (gate_up, mlp) = (&mut b.gate_up[..n * 2 * inter], &mut b.mlp[..n * inter]);

        for (&token, x) in tokens.iter().zip(x.chunks_exact_mut(d)) {
            m.embed.row(token as usize, x);
        }

        // Per-token element-wise work runs in parallel over tokens (inline when n = 1).
        let per_token = |f: &(dyn Fn(usize) + Sync)| {
            profile::time(Section::Elementwise, || {
                pool().par_for(n, 1, |r| r.for_each(f))
            })
        };

        for (li, layer) in m.layers.iter().enumerate() {
            // --- attention ---
            pool_rows2(x, d, xb, d, &per_token, |_, x, xb| {
                ops::rmsnorm(xb, x, &layer.input_norm, eps)
            });
            layer.qkv_proj.forward_batch(qkv, xb, n, &mut b.xq);

            // QK-norm and RoPE, then append K and V to the cache.
            pool_rows(qkv, qkv_dim, &per_token, |t, row| {
                let (q, kv) = row.split_at_mut(q_dim);
                for (heads, norm) in [(q, &layer.q_norm), (&mut kv[..kv_dim], &layer.k_norm)] {
                    for head in heads.chunks_exact_mut(hd) {
                        ops::rmsnorm_inplace(head, norm, eps);
                        m.rope.apply(head, pos0 + t);
                    }
                }
            });
            profile::time(Section::Elementwise, || {
                for (t, row) in qkv.chunks_exact(qkv_dim).enumerate() {
                    let (keys, values) = row[q_dim..].split_at(kv_dim);
                    for (h, (key, v)) in keys
                        .chunks_exact(hd)
                        .zip(values.chunks_exact(hd))
                        .enumerate()
                    {
                        self.cache.store(li, h, pos0 + t, key, v);
                    }
                }
            });

            // One task per (KV head, block of tokens): the query heads of a GQA group
            // over the block share every key tile. Token t sees positions 0..=pos0 + t.
            let cache = &self.cache;
            let qkv_ro = &*qkv;
            let att_heads = RowPtr::new(att, hd); // row t * heads + h
            let block = (ATTENTION_BLOCK_QUERIES / group).max(1);
            let n_blocks = n.div_ceil(block);
            profile::time(Section::Attention, || {
                pool().par_for(n_kv * n_blocks, 1, |tasks| {
                    ATTENTION_SCRATCH.with_borrow_mut(|(qs, outs)| {
                        for task in tasks {
                            let (g, blk) = (task % n_kv, task / n_kv);
                            let tokens = blk * block..((blk + 1) * block).min(n);
                            let query_heads = g * group..(g + 1) * group;
                            let mut limits = [0usize; MAX_QUERIES];
                            let mut m = 0;
                            for t in tokens.clone() {
                                for h in query_heads.clone() {
                                    qs[m * hd..][..hd]
                                        .copy_from_slice(&qkv_ro[t * qkv_dim + h * hd..][..hd]);
                                    limits[m] = pos0 + t + 1;
                                    m += 1;
                                }
                            }
                            let seen = pos0 + tokens.end;
                            let base = cache.head_offset(li, g);
                            let keys = &cache.k[base..][..seen.div_ceil(ATT_TILE) * ATT_TILE * hd];
                            let values = &cache.v[base..][..seen * hd];
                            (k.attention)(
                                &qs[..m * hd],
                                keys,
                                values,
                                &limits[..m],
                                scale,
                                &mut outs[..m * hd],
                            );
                            let mut i = 0;
                            for t in tokens {
                                for h in query_heads.clone() {
                                    // SAFETY: each (t, h) belongs to exactly one task.
                                    unsafe { att_heads.row(t * heads + h) }
                                        .copy_from_slice(&outs[i * hd..][..hd]);
                                    i += 1;
                                }
                            }
                        }
                    })
                })
            });

            layer.o_proj.forward_batch(xb, att, n, &mut b.xq);

            // --- MLP (SwiGLU) ---
            pool_rows2(x, d, xb, d, &per_token, |_, x, xb| {
                ops::add_inplace(x, xb);
                ops::rmsnorm(xb, x, &layer.post_norm, eps);
            });
            layer.gate_up_proj.forward_batch(gate_up, xb, n, &mut b.xq);
            pool_rows2(mlp, inter, gate_up, 2 * inter, &per_token, |_, h, gu| {
                let (gate, up) = gu.split_at(inter);
                for ((h, &g), &u) in h.iter_mut().zip(gate).zip(up) {
                    *h = ops::silu(g) * u;
                }
            });
            layer.down_proj.forward_batch(xb, mlp, n, &mut b.xq);
            pool_rows2(x, d, xb, d, &per_token, |_, x, xb| ops::add_inplace(x, xb));
        }

        // Only compute the logits that are needed: the LM head is the largest matrix.
        let head = m.lm_head();
        let vocab = c.vocab_size;
        if all_logits {
            for x in x.chunks_exact_mut(d) {
                ops::rmsnorm_inplace(x, &m.final_norm, eps);
            }
            b.logits.resize(n * vocab, 0.0);
            head.forward_batch(&mut b.logits, x, n, &mut b.xq);
        } else {
            let last = &mut x[(n - 1) * d..];
            ops::rmsnorm_inplace(last, &m.final_norm, eps);
            b.logits.resize(vocab, 0.0);
            head.forward(&mut b.logits, last, &mut b.xq);
        }
        self.pos += n;
        Ok(())
    }
}

/// Runs a per-row closure for every row index, in whatever order and threads it likes.
type RowScheduler<'a> = dyn Fn(&(dyn Fn(usize) + Sync)) + 'a;

/// Runs `f(t, row_t)` for every row of the row-major buffer `a` (row length `len`)
/// through `run`, which decides how rows are scheduled.
fn pool_rows(
    a: &mut [f32],
    len: usize,
    run: &RowScheduler<'_>,
    f: impl Fn(usize, &mut [f32]) + Sync,
) {
    let a = RowPtr::new(a, len);
    // SAFETY: `run` calls the closure once per distinct row index.
    run(&|t| f(t, unsafe { a.row(t) }));
}

/// Like [`pool_rows`] over two buffers with the same number of rows.
fn pool_rows2(
    a: &mut [f32],
    la: usize,
    b: &mut [f32],
    lb: usize,
    run: &RowScheduler<'_>,
    f: impl Fn(usize, &mut [f32], &mut [f32]) + Sync,
) {
    assert_eq!(a.len() / la, b.len() / lb);
    let (a, b) = (RowPtr::new(a, la), RowPtr::new(b, lb));
    // SAFETY: as above.
    run(&|t| f(t, unsafe { a.row(t) }, unsafe { b.row(t) }));
}

/// Hands out disjoint rows of a buffer to parallel tasks.
struct RowPtr {
    ptr: *mut f32,
    len: usize,
    rows: usize,
}

// SAFETY: `row` is only called for distinct rows from different threads.
unsafe impl Sync for RowPtr {}

impl RowPtr {
    fn new(buf: &mut [f32], len: usize) -> Self {
        assert_eq!(buf.len() % len, 0);
        Self {
            ptr: buf.as_mut_ptr(),
            len,
            rows: buf.len() / len,
        }
    }

    /// # Safety
    /// No two live references to the same row may exist.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, t: usize) -> &mut [f32] {
        assert!(t < self.rows);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(t * self.len), self.len) }
    }
}
