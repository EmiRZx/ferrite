# ferrite vs llama.cpp

How does a from-scratch engine compare with the most widely used CPU inference
engine, on the same machine, the same weights and the same quantization formats?

## Setup

- **Machine**: AMD Ryzen 9 7940HS (Zen 4, 8 cores / 16 threads, AVX-512 + VNNI),
  16 GB DDR5-5600 (~50 GB/s measured streaming read), Windows 11, mains power,
  "Balanced" power plan.
- **llama.cpp**: release `b11393` (commit `dbe4c3ed4`), official `win-cpu-x64` build,
  which loads its Zen 4 backend (`ggml-cpu-zen4.dll`) at runtime.
- **Model**: Qwen3-0.6B. The GGUF files were converted from the same
  `model.safetensors` that ferrite loads, with llama.cpp's own
  `convert_hf_to_gguf.py` (bf16), then quantized with `llama-quantize`.

| ferrite | llama.cpp | layers | LM head | bytes per weight (layers) |
|---|---|---|---|---|
| `bf16` | `BF16` | bf16 | bf16 | 2 / 2 |
| `q8` | `Q8_0` | 8-bit, blocks of 32 | 8-bit | 1.125 / 1.0625 |
| `q4` | `Q4_0 --output-tensor-type q8_0` | 4-bit, blocks of 32 | 8-bit | 0.625 / 0.5625 |

The formats are the same idea (symmetric blocks of 32 with one scale each), with one
difference: llama.cpp stores the scale as f16, ferrite (at the time of the main
measurements) as f32. ferrite therefore moved ~6% (Q8) and ~11% (Q4) more bytes per
token. Both keep the LM head in 8-bit for Q4, so per-token weight traffic is otherwise
the same. ferrite has since switched to f16 scales; see the update at the end.

## Method

**Speed**: `scripts/compare_llamacpp.py`. It measures prompt processing of 512 tokens
(pp512) and generation of 128 tokens from an empty context (tg128), the same tests as
`llama-bench`. Both engines were tuned first:

- llama.cpp: flash attention on (faster in every test), threads swept over 8 and 16.
- ferrite: threads swept over 8 and 16.

Both are fastest with 16 threads for prefill (compute-bound, where SMT siblings help)
and 8 threads for decode (bandwidth-bound, where extra threads only add
synchronization), so both settings are reported for prefill. The engines alternate
inside each round and swap order between rounds, so thermal drift hits both. The
table shows medians over the rounds.

**Quality**: perplexity on the first 4 × 512 tokens of WikiText-2 (test split),
scoring only the second half of each window, which is `llama-perplexity`'s method
(`ferrite eval --score-from 256`). KL divergence and top-1 agreement are measured
against each engine's own unquantized model on the same tokens.

## Results

### Speed

Medians of 3 rounds (each round itself a median of 3 runs per engine):

| weights | test | threads | ferrite tok/s | llama.cpp tok/s | ferrite / llama.cpp |
|---|---|---:|---:|---:|---:|
| bf16 | pp512 | 8 | 220 | 864 | 0.25× |
| bf16 | pp512 | 16 | 191 | 1007 | 0.19× |
| bf16 | tg128 | 8 | 37.8 | 35.9 | 1.05× |
| Q8 | pp512 | 8 | 436 | 410 | 1.06× |
| Q8 | pp512 | 16 | 483 | 502 | 0.96× |
| Q8 | tg128 | 8 | 67.3 | 69.2 | 0.97× |
| Q4 | pp512 | 8 | 422 | 750 | 0.56× |
| Q4 | pp512 | 16 | 306 | 857 | 0.36× |
| Q4 | tg128 | 8 | 83.1 | 90.9 | 0.91× |

### Quality

Perplexity over 4 × 512 tokens of WikiText-2, second half of each window scored
(1020 tokens). The perplexities carry an uncertainty of about ±2 at this sample size,
so only differences between the formats mean much. KL divergence and top-1 agreement
are paired per token, so they are far more precise.

| weights | ppl ferrite | ppl llama.cpp | KL ferrite | KL llama.cpp | top-1 ferrite | top-1 llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 18.88 | 18.89 | – | – | – | – |
| Q8 | 18.88 | 18.84 | 0.0029 | 0.0026 | 97.2% | 97.7% |
| Q4 | 20.79 | 20.75 | **0.179** | 0.206 | **78.5%** | 77.8% |

For reference, Hugging Face `transformers` in fp32 gives 18.88 on the same tokens
(`scripts/reference_ppl.py`). ferrite's f32 path matches it to four digits.

### Serving

`llama-server -t 8 -tb 16 -np 16 -c 8192 -fa on` (16 parallel slots, continuous
batching) against `ferrite serve --max-batch 16`, both Q8, driven by
`scripts/load_test.py`: streaming requests of 128 tokens with `ignore_eos`, Qwen3
with thinking off. llama-server ran before and after ferrite, and both runs are shown:

| concurrent requests | ferrite tok/s | llama-server tok/s | ferrite TTFT p50 | llama-server TTFT p50 |
|---:|---:|---:|---:|---:|
| 1 | 63.5 | 59.7 / 61.7 | 52 ms | 54 / 61 ms |
| 4 | 169 | 138 / 136 | 173 ms | 217 / 227 ms |
| 8 | 222 | 161 / 225 | 256 ms | 267 / 254 ms |
| 16 | 251 | 243 / 253 | 343 ms | 453 / 411 ms |

## What explains the differences

**Decode is a tie, because both engines hit the memory wall.** Generating a token
streams every weight once. In Q8, ferrite moves 639 MiB per token at 67 tok/s
(45 GB/s), llama.cpp ~600 MiB at 69 tok/s (44 GB/s). Both are close to the ~50 GB/s
the memory can deliver. In Q4, ferrite is 9% slower, which matches the 11% more bytes it
reads (f32 block scales instead of f16).

**Q8 prefill is a tie.** Both run int8 GEMMs on AVX-512 VNNI (`vpdpbusd`) with
activations quantized on the fly. ferrite's 4×4 register-tiled microkernel keeps pace
with llama.cpp's.

**bf16 prefill: llama.cpp is 4-5× faster.** Zen 4 has AVX512-BF16, whose `vdpbf16ps`
does 32 bf16 multiply-adds per instruction, and llama.cpp uses it inside a tiled GEMM.
ferrite has no bf16 GEMM microkernel: it widens bf16 to f32 and computes one dot
product per (row, token). This is the largest gap, and a contained one to close.

**Q4 prefill: llama.cpp is 2-3× faster.** When it loads Q4_0 weights, llama.cpp
*repacks* them into an interleaved layout built for its GEMM kernels, which makes its
Q4_0 prefill faster than its own Q8_0. ferrite unpacks nibbles inside the inner loop.
ferrite's 16-thread prefill also varies a lot between runs (292-489 tok/s for Q4). Its
spin-waiting pool was designed for 8 threads, and with SMT siblings a worker can be
starved by its twin.

**Quality: same for Q8, better for Q4.** ferrite's Q4 picks each block's scale by a
least-squares search instead of mapping the largest weight to -8, and keeps f32
scales. Its KL divergence to the unquantized model is 13% lower than Q4_0's.

**Serving: a tie at high load, ahead at moderate load.** With 4 concurrent requests
ferrite delivers ~22% more tokens per second and a lower time to first token. At 16
both are compute-bound on the batched matmuls (see the README's tokens-per-step table).

## A bug the comparison caught

The first quality comparison showed ferrite at a perplexity of 19.67 and llama.cpp at
18.89, on the same tokens. A 4% gap is too large for bf16 rounding. Bisecting the
context length showed the gap even at 32 tokens. Comparing tokenizations then showed
the cause: the dataset script wrote the file in Python text mode, which on Windows
turns every `\n` into `\r\n`. ferrite read the bytes as they were, while the Python
reference and llama-perplexity (C++ text mode) read `\n`. The engine was right, the data
was not. The script now writes bytes, and the README numbers were regenerated.

The detour exposed a real hole in the tests: the reference test used a 24-token prompt,
a single attention tile. The fixture now also holds a 266-token text that spans 9 KV
pages and 5 prefill chunks, and its logits match `transformers` to 8.6e-5.

## Closing the gaps

In order of payoff per effort:

1. ~~**f16 block scales**: 6-11% fewer bytes per token.~~ Done, see below.
2. **bf16 GEMM with `vdpbf16ps`**: the 4-5× bf16 prefill gap.
3. **Repacked Q4 layout** for prefill: the 2-3× Q4 prefill gap.
4. **SMT-aware scheduling** for compute-bound regions, so 16-thread prefill is stable.

## Update: f16 block scales

ferrite now stores weight scales as f16, like llama.cpp. AVX-512 converts 16 at a
time with `vcvtph2ps`, and AVX2 uses F16C. Its bytes per token now match llama.cpp's
(Q8: 604 MiB, Q4: 394 MiB). Decode re-measured the same way (3 alternating rounds):

| weights | ferrite tg128 | llama.cpp tg128 | before | after |
|---|---:|---:|---:|---:|
| Q8 | 77.4 | 75.0 | 0.97× | **1.03×** |
| Q4 | 104.1 | 110.9 | 0.91× | **0.94×** |

Quality is unchanged: Q8 KL 0.0025, Q4 KL 0.181 (second-half scoring as above). With
equal bytes, the remaining Q4 gap comes from the kernel. ferrite's Q4 decode reaches
42 GB/s against 45 GB/s for Q8, because unpacking nibbles costs instructions even when
the loop waits on memory.

## Reproducing

```sh
# llama.cpp release and GGUF files from the same checkpoint
mkdir -p third_party/llama.cpp && cd third_party/llama.cpp
curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b11393/llama-b11393-bin-win-cpu-x64.zip
unzip llama-b11393-bin-win-cpu-x64.zip -d bin
curl -L https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11393.tar.gz | tar -xz
uv run --python 3.12 --with ./llama.cpp-b11393/gguf-py --with torch --with transformers \
  --with sentencepiece --with safetensors python llama.cpp-b11393/convert_hf_to_gguf.py \
  ../../models/qwen3-0.6b --outtype bf16 --outfile models/qwen3-0.6b-bf16.gguf
bin/llama-quantize models/qwen3-0.6b-bf16.gguf models/qwen3-0.6b-q8_0.gguf Q8_0
bin/llama-quantize --token-embedding-type q8_0 --output-tensor-type q8_0 \
  models/qwen3-0.6b-bf16.gguf models/qwen3-0.6b-q4_0.gguf Q4_0
cd ../..

uv run scripts/compare_llamacpp.py                         # speed
cargo run --release -- -d q4 eval models/wikitext-2-test.txt \
  --score-from 256 --baseline f32                          # quality, ferrite
third_party/llama.cpp/bin/llama-perplexity -m <gguf> -f models/wikitext-2-test.txt \
  -c 512 --chunks 4                                        # quality, llama.cpp
uv run scripts/load_test.py --base-url http://127.0.0.1:8080/v1 --levels 1,4,8,16
```
