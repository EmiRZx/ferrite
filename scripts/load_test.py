# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.0"]
# ///
"""Load test for `ferrite serve`: throughput and latency versus concurrency.

For each concurrency level, sends 2 x concurrency streaming requests (that many in
flight at once) of `--max-tokens` tokens each, with `ignore_eos` so every request
generates the full length.

Usage: uv run scripts/load_test.py [--levels 1,2,4,8] [--max-tokens 128]
"""

import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI

PROMPTS = [
    "Write a short story about a lighthouse keeper.",
    "Explain how a CPU cache works.",
    "Describe the water cycle to a child.",
    "What are the main causes of the French Revolution?",
    "Give advice to a new software engineer.",
    "Describe a walk through a rainy city at night.",
    "Explain the difference between TCP and UDP.",
    "Write a poem about the ocean.",
]


async def one(client, model, prompt, max_tokens):
    start = time.perf_counter()
    first, tokens = None, 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True},
    )
    async for chunk in stream:
        if chunk.usage:
            tokens = chunk.usage.completion_tokens
        elif chunk.choices and chunk.choices[0].delta.content and first is None:
            first = time.perf_counter()
    end = time.perf_counter()
    decode_rate = (tokens - 1) / (end - first) if first and end > first else 0.0
    return first - start, decode_rate, tokens


async def level(client, model, concurrency, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    n = 2 * concurrency

    async def bounded(i):
        async with sem:
            return await one(client, model, PROMPTS[i % len(PROMPTS)], max_tokens)

    start = time.perf_counter()
    results = await asyncio.gather(*(bounded(i) for i in range(n)))
    wall = time.perf_counter() - start
    ttft = sorted(r[0] for r in results)
    tokens = sum(r[2] for r in results)
    return {
        "concurrency": concurrency,
        "requests": n,
        "throughput": tokens / wall,
        "ttft_p50": 1000 * statistics.median(ttft),
        "ttft_p90": 1000 * ttft[min(len(ttft) - 1, int(0.9 * len(ttft)))],
        "per_request": statistics.mean(r[1] for r in results),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--levels", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed", max_retries=0, timeout=600)
    model = (await client.models.list()).data[0].id
    await one(client, model, "Hi", 8)  # warm-up

    print(f"model {model}, {args.max_tokens} tokens per request\n")
    print("| concurrency | requests | total tok/s | per-request tok/s | TTFT p50 | TTFT p90 |")
    print("|---:|---:|---:|---:|---:|---:|")
    for c in (int(x) for x in args.levels.split(",")):
        r = await level(client, model, c, args.max_tokens)
        print(f"| {r['concurrency']} | {r['requests']} | {r['throughput']:.1f} | {r['per_request']:.1f} "
              f"| {r['ttft_p50']:.0f} ms | {r['ttft_p90']:.0f} ms |", flush=True)


asyncio.run(main())
