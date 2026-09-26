# ferrite

A from-scratch CPU inference engine for large language models, written in Rust.

No ML frameworks. The safetensors loader, byte-level BPE tokenizer, transformer
forward pass, sampler and all compute kernels are implemented in this repository,
and every step is verified numerically against Hugging Face `transformers`.

The target model is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); the goal is to
take it from a naive scalar implementation to a fast, quantized, SIMD-optimized engine
and document *why* each optimization works along the way.

```text
$ ferrite run "What is the capital of France? Answer in one sentence." -t 0
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
cargo run --release -- bench --prompt-len 128 --gen-len 128
```

## Correctness

`scripts/reference.py` dumps tokenizer outputs, per-position top-10 logits and a greedy
continuation from `transformers` (fp32) into `tests/fixtures/`. `cargo test --release`
checks ferrite against them:

| check | result |
|---|---|
| tokenizer (12 cases: Unicode, NFC, code, whitespace, special tokens) | exact match |
| max \|logit − reference\| over top-10 logits at every prompt position | 1.0e-4 |
| 32-token greedy continuation | exact match |

## Benchmarks

Machine: AMD Ryzen 9 7940HS (Zen 4, 8C/16T, AVX-512), 16 GB DDR5, Windows 11.
Greedy decoding, 64-token prompt, 64 generated tokens.

| version | weights | prefill tok/s | decode tok/s | effective weight bandwidth |
|---|---|---:|---:|---:|
| M0: scalar f32, 1 thread | 2.27 GiB | 2.2 | 2.2 | 5 GB/s |
| M0: scalar f32, rayon (8 threads) | 2.27 GiB | 13.4 | 13.4 | 32 GB/s |

Decode on CPU is memory-bandwidth bound: every generated token streams every weight
through the cores once. With the naive kernels we already reach ~32 GB/s,
so the next big wins come from moving fewer bytes (bf16 / int8 / int4 weights), while
prefill will become compute-bound once it is batched into a GEMM.

## Roadmap

- [x] **M0**: safetensors (mmap), BPE tokenizer, Qwen3 forward pass (GQA, QK-norm, RoPE), KV cache, sampling, CLI, reference tests
- [ ] **M1**: SIMD kernels (AVX2 / AVX-512) with runtime CPU feature dispatch, micro-benchmarks
- [ ] **M2**: bf16 weights, then int8 / int4 weight-only quantization (VNNI dot products)
- [ ] **M3**: batched prefill with a cache-blocked GEMM
- [ ] **M4**: OpenAI-compatible HTTP server with streaming
- [ ] **M5**: comparison against llama.cpp, roofline analysis write-up

## Layout

```text
src/
  safetensors.rs   zero-copy .safetensors reader
  tokenizer.rs     byte-level BPE (HF tokenizer.json compatible) + streaming UTF-8 decoder
  model.rs         Qwen3 weights, KV cache, forward pass
  ops.rs           reference kernels: matvec, RMSNorm, RoPE, softmax
  sampler.rs       temperature / top-k / top-p sampling
  generate.rs      generation loop, chat template
  main.rs          CLI
scripts/reference.py   dumps HF transformers outputs for the tests
```

## License

MIT
