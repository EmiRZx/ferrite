//! Qwen3 dense decoder, implemented directly on flat f32 buffers.
//!
//! Per layer:
//! ```text
//! h = x + o_proj(attn(rope(q_norm(q_proj(n1))), rope(k_norm(k_proj(n1))), v_proj(n1)))   n1 = rmsnorm(x)
//! x = h + down_proj(silu(gate_proj(n2)) * up_proj(n2))                                     n2 = rmsnorm(h)
//! ```

use std::{cell::RefCell, ops::Range, path::Path};

use crate::{
    config::Config,
    kernels::{
        attention::{MAX_HEAD_DIM, MAX_QUERIES},
        kernels,
    },
    kv::{KvPool, SeqCache},
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

/// One sequence's part of a forward pass: `tokens` are appended to `seq` at
/// positions `seq.len()..`. The sequence's KV pages must already be reserved.
pub struct Span<'a> {
    pub seq: &'a mut SeqCache,
    pub tokens: &'a [u32],
    /// Logits after every token instead of only after the last one.
    pub all_logits: bool,
}

/// Scratch buffers for forward passes of up to `max_rows` tokens in total. Each is
/// row-major `[rows, width]`.
pub struct Workspace {
    max_rows: usize,
    x: Vec<f32>,
    xb: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    gate_up: Vec<f32>,
    mlp: Vec<f32>,
    /// Final hidden states of the rows that need logits.
    head_in: Vec<f32>,
    logits: Vec<f32>,
    /// Rows of `logits` belonging to each span of the last forward pass.
    logit_rows: Vec<Range<usize>>,
    xq: Q8Vec,
}

impl Workspace {
    pub fn new(model: &Model, max_rows: usize) -> Self {
        let c = &model.config;
        Self {
            max_rows,
            x: vec![0.0; max_rows * c.hidden_size],
            xb: vec![0.0; max_rows * c.hidden_size],
            qkv: vec![0.0; max_rows * (c.q_dim() + 2 * c.kv_dim())],
            att: vec![0.0; max_rows * c.q_dim()],
            gate_up: vec![0.0; max_rows * 2 * c.intermediate_size],
            mlp: vec![0.0; max_rows * c.intermediate_size],
            head_in: Vec::new(),
            logits: Vec::new(),
            logit_rows: Vec::new(),
            xq: Q8Vec::default(),
        }
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Logits for span `i` of the last forward pass: `[rows, vocab]`, a single row
    /// unless that span asked for all of them.
    pub fn logits(&self, span: usize, vocab: usize) -> &[f32] {
        let rows = &self.logit_rows[span];
        &self.logits[rows.start * vocab..rows.end * vocab]
    }
}

impl Model {
    /// Runs one forward pass over a ragged batch: every span's tokens are processed
    /// together, so the matmuls stream each weight once for all of them, while
    /// attention reads each span's own KV pages. A batch can mix prompt chunks with
    /// single decode tokens of other sequences.
    pub fn forward(&self, ws: &mut Workspace, kv: &mut KvPool, spans: &mut [Span]) -> Result<()> {
        let c = &self.config;
        let n: usize = spans.iter().map(|s| s.tokens.len()).sum();
        ensure!(
            n > 0 && n <= ws.max_rows,
            "batch of {n} tokens exceeds the workspace ({})",
            ws.max_rows
        );
        for s in spans.iter() {
            let end = s.seq.len() + s.tokens.len();
            ensure!(!s.tokens.is_empty(), "empty span");
            ensure!(
                end <= self.max_seq,
                "context window of {} tokens is full",
                self.max_seq
            );
            ensure!(
                end <= s.seq.capacity(),
                "KV pages for {end} positions are not reserved"
            );
            if let Some(&t) = s.tokens.iter().find(|&&t| t as usize >= c.vocab_size) {
                bail!("token id {t} out of range");
            }
        }

        let (d, hd, heads) = (c.hidden_size, c.head_dim(), c.num_attention_heads);
        let (q_dim, kv_dim, inter) = (c.q_dim(), c.kv_dim(), c.intermediate_size);
        let qkv_dim = q_dim + 2 * kv_dim;
        let n_kv = c.num_key_value_heads;
        let group = heads / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = c.rms_norm_eps;
        let k = kernels();

        // Row t of the batch is token `rows[t].1` of span `rows[t].0`, at position `rows[t].2`.
        let mut rows = Vec::with_capacity(n);
        let mut span_rows = Vec::with_capacity(spans.len());
        for (si, s) in spans.iter().enumerate() {
            let start = rows.len();
            rows.extend(
                s.tokens
                    .iter()
                    .enumerate()
                    .map(|(i, &tok)| (si, tok, s.seq.len() + i)),
            );
            span_rows.push(start..rows.len());
        }

        // Attention work: per span, its tokens in blocks, times the KV heads.
        let block = (ATTENTION_BLOCK_QUERIES / group).max(1);
        let mut tasks = Vec::new();
        for (si, r) in span_rows.iter().enumerate() {
            for b0 in r.clone().step_by(block) {
                for g in 0..n_kv {
                    tasks.push((si, g, b0..(b0 + block).min(r.end)));
                }
            }
        }

        let Workspace {
            x,
            xb,
            qkv,
            att,
            gate_up,
            mlp,
            xq,
            ..
        } = ws;
        let (x, xb) = (&mut x[..n * d], &mut xb[..n * d]);
        let (qkv, att) = (&mut qkv[..n * qkv_dim], &mut att[..n * q_dim]);
        let (gate_up, mlp) = (&mut gate_up[..n * 2 * inter], &mut mlp[..n * inter]);

        for (&(_, token, _), x) in rows.iter().zip(x.chunks_exact_mut(d)) {
            self.embed.row(token as usize, x);
        }

        // Per-token element-wise work runs in parallel over tokens (inline when n = 1).
        let per_token = |f: &(dyn Fn(usize) + Sync)| {
            profile::time(Section::Elementwise, || {
                pool().par_for(n, 1, |r| r.for_each(f))
            })
        };

        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention ---
            pool_rows2(x, d, xb, d, &per_token, |_, x, xb| {
                ops::rmsnorm(xb, x, &layer.input_norm, eps)
            });
            layer.qkv_proj.forward_batch(qkv, xb, n, xq);

            // QK-norm and RoPE at each token's own position, then append K and V.
            pool_rows(qkv, qkv_dim, &per_token, |t, row| {
                let (q, kvs) = row.split_at_mut(q_dim);
                for (heads, norm) in [(q, &layer.q_norm), (&mut kvs[..kv_dim], &layer.k_norm)] {
                    for head in heads.chunks_exact_mut(hd) {
                        ops::rmsnorm_inplace(head, norm, eps);
                        self.rope.apply(head, rows[t].2);
                    }
                }
            });
            profile::time(Section::Elementwise, || {
                for (&(si, _, pos), row) in rows.iter().zip(qkv.chunks_exact(qkv_dim)) {
                    let (keys, values) = row[q_dim..].split_at(kv_dim);
                    for (h, (key, v)) in keys
                        .chunks_exact(hd)
                        .zip(values.chunks_exact(hd))
                        .enumerate()
                    {
                        kv.store(spans[si].seq, pos, li, h, key, v);
                    }
                }
            });

            // One task per (span, KV head, block of the span's tokens): the query heads
            // of a GQA group share every key tile. A token sees positions 0..=its own.
            let (kv_ro, spans_ro, qkv_ro) = (&*kv, &*spans, &*qkv);
            let att_heads = RowPtr::new(att, hd); // row t * heads + h
            profile::time(Section::Attention, || {
                pool().par_for(tasks.len(), 1, |range| {
                    ATTENTION_SCRATCH.with_borrow_mut(|(qs, outs)| {
                        for (si, g, block_rows) in range.map(|i| tasks[i].clone()) {
                            let query_heads = g * group..(g + 1) * group;
                            let mut limits = [0usize; MAX_QUERIES];
                            let mut m = 0;
                            for t in block_rows.clone() {
                                for h in query_heads.clone() {
                                    qs[m * hd..][..hd]
                                        .copy_from_slice(&qkv_ro[t * qkv_dim + h * hd..][..hd]);
                                    limits[m] = rows[t].2 + 1;
                                    m += 1;
                                }
                            }
                            let tiles = kv_ro.tiles(spans_ro[si].seq, li, g);
                            (k.attention)(
                                &qs[..m * hd],
                                &tiles,
                                &limits[..m],
                                scale,
                                &mut outs[..m * hd],
                            );
                            let mut i = 0;
                            for t in block_rows {
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

            layer.o_proj.forward_batch(xb, att, n, xq);

            // --- MLP (SwiGLU) ---
            pool_rows2(x, d, xb, d, &per_token, |_, x, xb| {
                ops::add_inplace(x, xb);
                ops::rmsnorm(xb, x, &layer.post_norm, eps);
            });
            layer.gate_up_proj.forward_batch(gate_up, xb, n, xq);
            pool_rows2(mlp, inter, gate_up, 2 * inter, &per_token, |_, h, gu| {
                let (gate, up) = gu.split_at(inter);
                for ((h, &g), &u) in h.iter_mut().zip(gate).zip(up) {
                    *h = ops::silu(g) * u;
                }
            });
            layer.down_proj.forward_batch(xb, mlp, n, xq);
            pool_rows2(x, d, xb, d, &per_token, |_, x, xb| ops::add_inplace(x, xb));
        }

        // Logits only for the rows that need them: the LM head is the largest matrix.
        let wanted: Vec<usize> = spans
            .iter()
            .zip(&span_rows)
            .flat_map(|(s, r)| {
                if s.all_logits {
                    r.clone()
                } else {
                    r.end - 1..r.end
                }
            })
            .collect();
        ws.logit_rows.clear();
        let mut next = 0;
        for (s, r) in spans.iter().zip(&span_rows) {
            let count = if s.all_logits { r.len() } else { 1 };
            ws.logit_rows.push(next..next + count);
            next += count;
        }
        let vocab = c.vocab_size;
        ws.head_in.resize(wanted.len() * d, 0.0);
        for (dst, &t) in ws.head_in.chunks_exact_mut(d).zip(&wanted) {
            dst.copy_from_slice(&x[t * d..][..d]);
            ops::rmsnorm_inplace(dst, &self.final_norm, eps);
        }
        ws.logits.resize(wanted.len() * vocab, 0.0);
        self.lm_head()
            .forward_batch(&mut ws.logits, &ws.head_in, wanted.len(), &mut ws.xq);

        for s in spans.iter_mut() {
            s.seq.advance(s.tokens.len());
        }
        Ok(())
    }
}

/// A single generation stream with its own KV pages; the simple API used by the
/// CLI, benchmarks and tests.
pub struct Session<'m> {
    model: &'m Model,
    kv: KvPool,
    seq: SeqCache,
    ws: Workspace,
}

impl<'m> Session<'m> {
    fn new(model: &'m Model) -> Self {
        Self {
            model,
            kv: KvPool::for_tokens(&model.config, model.max_seq),
            seq: SeqCache::default(),
            ws: Workspace::new(model, PREFILL_CHUNK),
        }
    }

    pub fn pos(&self) -> usize {
        self.seq.len()
    }

    pub fn reset(&mut self) {
        self.kv.release(&mut self.seq);
    }

    /// Runs one token at the current position; returns the next-token logits.
    pub fn forward(&mut self, token: u32) -> Result<&[f32]> {
        self.run(&[token], false)
    }

    /// Feeds a prompt in batches of [`PREFILL_CHUNK`] tokens; returns the logits
    /// after its last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        ensure!(!tokens.is_empty(), "empty prompt");
        let mut chunks = tokens.chunks(PREFILL_CHUNK);
        let last = chunks.next_back().unwrap();
        for chunk in chunks {
            self.run(chunk, false)?;
        }
        self.run(last, false)
    }

    /// Runs up to [`PREFILL_CHUNK`] tokens and returns the logits after *each* of
    /// them, `[tokens, vocab]` (used for evaluation).
    pub fn forward_all(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        ensure!(
            tokens.len() <= PREFILL_CHUNK,
            "at most {PREFILL_CHUNK} tokens per call"
        );
        self.run(tokens, true)
    }

    fn run(&mut self, tokens: &[u32], all_logits: bool) -> Result<&[f32]> {
        let end = self.seq.len() + tokens.len();
        ensure!(
            end <= self.model.max_seq,
            "context window of {} tokens is full",
            self.model.max_seq
        );
        self.kv.reserve(&mut self.seq, end)?;
        let span = Span {
            seq: &mut self.seq,
            tokens,
            all_logits,
        };
        self.model
            .forward(&mut self.ws, &mut self.kv, &mut [span])?;
        Ok(self.ws.logits(0, self.model.config.vocab_size))
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
