# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.0"]
# ///
"""Streams a chat completion from `ferrite serve` with the official OpenAI client.

Usage: uv run scripts/chat_client.py ["your question"] [--base-url http://127.0.0.1:8080/v1]
"""

import argparse
import time

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="Explain what a KV cache is in two sentences.")
parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
args = parser.parse_args()

client = OpenAI(base_url=args.base_url, api_key="not-needed")
model = client.models.list().data[0].id

start = time.perf_counter()
first = None
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": args.prompt}],
    stream=True,
    stream_options={"include_usage": True},
)
usage = None
for chunk in stream:
    if chunk.usage:
        usage = chunk.usage
    if chunk.choices and chunk.choices[0].delta.content:
        first = first or time.perf_counter()
        print(chunk.choices[0].delta.content, end="", flush=True)
end = time.perf_counter()
print()
if usage and first:
    rate = (usage.completion_tokens - 1) / (end - first) if end > first else 0.0
    print(f"\n[{model}] time to first token {1000 * (first - start):.0f} ms, "
          f"{usage.completion_tokens} tokens at {rate:.1f} tok/s")
