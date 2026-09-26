# ferrite

A from-scratch CPU inference engine for large language models, written in Rust.

No ML frameworks. The safetensors loader, byte-level BPE tokenizer, transformer
forward pass, quantization formats, SIMD kernels, thread pool and sampler are all
implemented in this repository, and every step is verified numerically against
Hugging Face `transformers`.

The target model is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). The goal is to
take it from a naive scalar implementation to a fast, quantized, SIMD-optimized engine
and document *why* each optimization works along the way.

```text
$ ferrite -d q4 run "What is the capital of France? Answer in one sentence." -t 0
loaded models/qwen3-0.6b as q4 in 4.08s (429 MiB of weights, avx512 kernels, 8 threads)
The capital of France is Paris.
```

## Quick start

```sh
# download the model (≈1.5 GB)
mkdir -p models/qwen3-0.6b && cd models/qwen3-0.6b
for f in config.json tokenizer.json model.safetensors; do
  curl -LO https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/$f
done
cd ../..

cargo run --release -- run "Explain Rust ownership in two sentences."
cargo run --release -- run --think "How many r's are in strawberry?"
cargo run --release -- -d q4 bench --prompt-len 128 --gen-len 128

# quality: perplexity on WikiText-2, KL divergence against the f32 model
uv run scripts/fetch_wikitext.py
cargo run --release -- -d q4 eval models/wikitext-2-test.txt --baseline f32

# kernel micro-benchmarks
cargo run --release --example kernels
```

Options: `-d/--dtype f32|bf16|q8|q4` (default `q8`), `--threads N` (default: physical
cores), `FERRITE_ISA=scalar|avx2|avx512` to force a kernel set, `FERRITE_PROFILE=1`
to print where `bench` spends its time (matmul / quantize / attention / element-wise).

## Results

Machine: AMD Ryzen 9 7940HS (Zen 4, 8C/16T, AVX-512 + VNNI), 16 GB DDR5-5600 (measured
streaming read: ~50 GB/s), Windows 11. Greedy decoding, 8 threads.

| weights | size | prefill tok/s | decode tok/s | weight bandwidth | WikiText-2 ppl | KL vs f32 | top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| f32 | 2274 MiB | 194 | 20.9 | 49.9 GB/s | 29.19 | – | – |
| bf16 | 1137 MiB | 223 | 40.3 | 48.0 GB/s | 29.19 | 0.000 | 100% |
| Q8 | 639 MiB | 449 | 67.3 | 45.1 GB/s | 29.07 | 0.003 | 96.9% |
| Q4 (head in Q8) | 429 MiB | 433 | 85.0 | 38.3 GB/s | 32.99 | 0.189 | 76.1% |

Prefill: 512-token prompt. Decode: 128 tokens after a short prompt. Both are medians of
repeated runs. This is a laptop, so expect ±10–30% between sessions depending on heat.
Quality: 4 windows of 512 tokens from the WikiText-2 test set.
bf16 is bit-identical to f32 because the checkpoint *is* bf16. Q8 is effectively
lossless. Q4 costs real quality on a model this small. Choosing each block's scale by
least-squares search instead of max-scaling cut KL from 0.210 to 0.189. Mixed precision
for sensitive layers is next (see roadmap).

## How it got fast

Decoding one token multiplies every weight matrix by a single vector, so each weight
is read exactly once and used for exactly one multiply-add. That is ~0.25 FLOP per
byte for f32, far below what the cores can do per byte of DRAM bandwidth: **decode is
memory-bound**, and tokens/s ≈ bandwidth / bytes of weights. Every step below
either moves fewer bytes or gets closer to the bandwidth ceiling.

Decode tok/s after each step, measured as the step landed (laptop numbers drift by
about ±10% between runs):

| step | f32 | bf16 | Q8 | Q4 | why |
|---|---:|---:|---:|---:|---|
| M0: scalar kernels, rayon | 13.4 | | | | baseline: ~32 GB/s already |
| + AVX2 / AVX-512 kernels | 17.3 | 28.1 | 39.4 | 52.1 | f32 at ~41 GB/s; halving the bytes (bf16) nearly doubles speed |
| + spin-waiting pool, fused QKV / gate-up | 20.2 | 38.8 | 61.8 | 86.1 | removes ~5 ms/token of thread wake-ups and dispatch |
| + SIMD activation quantization | | | 71.8 | 92.1 | `f32::round` was a libm call per element on baseline x86-64 |
| + unsigned-offset VNNI kernel | | | ± 0 | +5–10% | ~30% faster in cache (micro-benchmark), but decode is near the bandwidth limit |

Things that did *not* help, and why:

- **Static row partitioning** (one contiguous chunk per thread) was slower than
  work-stealing: a single thread that gets descheduled or runs at a lower clock holds
  up the whole matvec. The pool therefore hands out 16-row chunks through an atomic counter.
- **16 threads (SMT)**: SMT siblings share a core's load ports, so they add no
  bandwidth, and a spinning barrier is only as fast as its slowest thread. 16 threads
  ran ~40% *slower* than 8.
- **AVX-512 over AVX2** barely matters for decode: Zen 4 executes 512-bit ops as two
  256-bit halves, and the loop is waiting on memory anyway. The micro-benchmark shows
  both at ~37 GFLOP/s per core on in-cache data vs 2.7 for scalar code.

## Prefill: from memory-bound to compute-bound

Prefill processes the whole prompt at once, so each weight row can be multiplied with
64 tokens while it sits in L1. That is 64× more work per byte of DRAM traffic: the matmul
stops waiting on memory, and the speed of the arithmetic starts to matter.

Q8 prefill of a 512-token prompt, step by step:

| step | tok/s | what changed |
|---|---:|---|
| M2: token by token | ~60 | every prompt token is a separate memory-bound decode step |
| batched GEMM, one dot product per (row, token) | 247 | weights reused 64×. Now limited by per-call overhead (~3 MAC/cycle/core) |
| 4×4 register-tiled VNNI microkernel | 363 | each weight load feeds 4 tokens, each activation load 4 rows |
| element-wise work parallel over tokens | 441 | RMSNorm / RoPE / SiLU were 17% of the time on one thread (`FERRITE_PROFILE`) |
| tiled AVX-512 attention | 449–488 | attention at 1024 tokens: 1.67 s → 0.70 s |

After these steps, the matmul is ~83% of prefill time at 512 tokens and runs at
~350 GMAC/s (≈10 int8 MAC per cycle per core). The GEMM micro-benchmark shows f32 at
133 GMAC/s by comparison. Attention grows quadratically: 2% of prefill at 64 tokens,
15% at 1024.

More things that did *not* help:

- **Auto-vectorized attention.** The first tiled attention kernel was portable Rust
  meant for LLVM to vectorize. The assembly showed 32 *scalar* FMAs per tile row: the
  fixed-length loops were fully unrolled instead of vectorized. The AVX-512 path is now
  written with intrinsics, and the portable version remains as the scalar/AVX2 fallback
  and test reference.
- **Big attention blocks.** Putting 32 tokens × 2 query heads in one task maximizes key
  reuse, but a 64-token chunk then splits into only 16 unevenly sized (causal) tasks for
  8 cores. 8-token blocks were ~20% faster.

## Correctness

`scripts/reference.py` dumps tokenizer outputs, per-position top-10 logits and a greedy
continuation from `transformers` (fp32) into `tests/fixtures/`. `cargo test --release`
checks ferrite against them:

| check | result |
|---|---|
| tokenizer (12 cases: Unicode, NFC, code, whitespace, special tokens) | exact match |
| max \|logit − reference\| over top-10 logits at every prompt position | 4.8e-5 |
| same, through the batched prefill path | 4.8e-5 |
| 32-token greedy continuation | exact match |
| batched vs token-by-token logits for bf16 / Q8 / Q4 | bit-identical |

Every SIMD kernel is also tested against its scalar reference on ragged lengths
(including the tiled GEMM and attention kernels), and the thread pool has tests for
coverage, disjoint writes and panic propagation.

## Design notes

**Quantization formats** (`src/quant.rs`). Rows are split into blocks of 32 values
with one f32 scale each, stored struct-of-arrays so kernels stream quants and scales linearly.
Q8 stores `round(w / d)` in `[-127, 127]` (1.125 bytes/weight). Q4 stores 4-bit values
with the block's largest-magnitude weight mapped exactly to -8 (0.625 bytes/weight).
In Q4 mode the embedding / LM head (26% of all parameters) stays Q8.

**Integer dot products** (`src/kernels/`). Activations are quantized to Q8 on the fly
(once per matvec, together with each block's sum), so the inner loop is an int8 dot
product. With AVX-512 VNNI one `vpdpbusd` multiplies 64 byte pairs and accumulates groups
of four into 16 int32 lanes. It needs one *unsigned* operand. Instead of shuffling signs
every iteration, the weights are made unsigned by an offset `c` and the error is
removed exactly in integers: `Σ w·x = Σ (w + c)·x − c·Σx`. Q4 nibbles already are
`w + 8`. Q8 uses `w XOR 0x80 = w + 128`. The `−c·Σx` term seeds the accumulator, so the
correction costs one masked permute per two blocks, and scales are applied 16 blocks at a
time with vector permutes. Q4 nibbles are unpacked with a lane shuffle, a masked shift and an AND.
The AVX2 path uses the classic `maddubs` sign trick instead, because its 16-bit
intermediate would overflow with `w + 128`.

**Runtime dispatch** (`src/kernels/mod.rs`). Kernels are `#[target_feature]` functions
collected into per-ISA tables of function pointers. The best table the CPU supports is
chosen once at startup, so one binary runs everywhere and the scalar path remains
available as a reference.

**Spin-waiting thread pool** (`src/pool.rs`). A decode step issues ~140 parallel
regions of 10–50 µs each. Rayon parks idle workers, and waking them cost about as much as
the work (a 1024×1024 matvec: 21 µs with rayon vs 13 µs here). Workers spin on an
epoch counter between jobs (parking after ~1 ms idle), and the caller runs as thread 0.

**Fused projections**. Q/K/V and gate/up share their input, so they are stacked into
one matrix each: 4 matvecs per layer instead of 7.

**Batched forward pass** (`src/model.rs`, `src/linear.rs`). A single code path handles
any number of tokens; decode is the one-token case. Prompts run in chunks of 64 tokens.
For each matmul, tasks own 16 weight rows and sweep all tokens. With AVX-512 each call
computes a 4-row × 4-token tile, so every task writes whole 64-byte cache lines of the
output. Only the logits that are needed are computed: the LM head is the largest matrix.

**Attention** (`src/kernels/attention.rs`). Keys are cached transposed in tiles of 32
positions (`[head_dim][32]`), so scoring a query against a tile is `head_dim` vector
FMAs across 32 positions, with no horizontal sums. A task covers one KV head for a
block of tokens, so all query heads of a GQA group share each key tile while it is in
L1. The softmax is computed online (FlashAttention-style): a running max and sum per
query, with one rescale when a tile raises the max, so no buffer grows with the context.
`exp` is a Cephes-style polynomial that runs on 16 lanes, since `f32::exp` is a libm call.

## Roadmap

- [x] **M0**: safetensors (mmap), BPE tokenizer, Qwen3 forward pass (GQA, QK-norm, RoPE), KV cache, sampling, CLI, reference tests
- [x] **M1**: SIMD kernels (AVX2 / AVX-512) with runtime CPU feature dispatch, micro-benchmarks
- [x] **M2**: bf16 / Q8 / Q4 weights with VNNI integer dot products, spin-waiting thread pool, fused projections, perplexity + KL evaluation
- [x] **M3**: batched prefill: tiled VNNI GEMM, FlashAttention-style tiled attention, parallel element-wise ops, profiler
- [ ] Mixed-precision Q4 (sensitive layers such as `down_proj` / `v_proj` in Q8), in the spirit of llama.cpp's Q4_K_M
- [ ] bf16 / int8 KV cache: at 1 k tokens of context, decode drops from 68 to 55 tok/s reading the f32 cache
- [ ] **M4**: OpenAI-compatible HTTP server with streaming
- [ ] **M5**: comparison against llama.cpp, roofline analysis write-up

## Layout

```text
src/
  safetensors.rs   zero-copy .safetensors reader
  tokenizer.rs     byte-level BPE (HF tokenizer.json compatible) + streaming UTF-8 decoder
  model.rs         Qwen3 weights, KV cache, forward pass
  linear.rs        linear layers over f32 / bf16 / Q8 / Q4 weights
  quant.rs         block quantization formats
  kernels/         scalar, AVX2 and AVX-512 (VNNI) kernels + runtime dispatch
    attention.rs   tiled online-softmax attention (portable reference)
  pool.rs          low-latency spin-waiting thread pool
  profile.rs       opt-in per-section timing (FERRITE_PROFILE=1)
  ops.rs           RMSNorm, RoPE, SiLU
  sampler.rs       temperature / top-k / top-p sampling
  generate.rs      generation loop, chat template
  eval.rs          perplexity, KL divergence, top-1 agreement
  main.rs          CLI
examples/kernels.rs     matvec / GEMM / bandwidth micro-benchmarks
scripts/reference.py    dumps HF transformers outputs for the tests
```

## License

MIT
