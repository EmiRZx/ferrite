# /// script
# requires-python = ">=3.10"
# ///
"""Benchmarks ferrite against llama.cpp on the same machine, model and quantization.

Prompt processing of 512 tokens (pp512) at 8 and 16 threads, generation of 128
tokens from an empty context (tg128) at 8 threads. The engines alternate within each
round (and swap order between rounds) so slow drift such as heat affects both; the
median over rounds is reported.

Setup (see README): llama.cpp binaries in third_party/llama.cpp/bin and GGUF files
converted from the same checkpoint in third_party/llama.cpp/models.

Usage: uv run scripts/compare_llamacpp.py [--rounds 3]
"""

import argparse
import json
import os
import re
import statistics
import subprocess

EXE = ".exe" if os.name == "nt" else ""
FERRITE = os.path.abspath("target/release/ferrite" + EXE)
LLAMA_BENCH = os.path.abspath("third_party/llama.cpp/bin/llama-bench" + EXE)
GGUF = "third_party/llama.cpp/models/qwen3-0.6b-{}.gguf"
FORMATS = [("bf16", "bf16"), ("q8", "q8_0"), ("q4", "q4_0")]  # ferrite dtype, GGUF type
TESTS = [("pp512", 8), ("pp512", 16), ("tg128", 8)]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def ferrite(dtype, test, threads):
    base = [FERRITE, "--threads", str(threads), "-d", dtype, "bench", "--runs", "3"]
    if test == "pp512":
        out = run(base + ["--prompt-len", "512", "--gen-len", "1"])
        return float(re.search(r"prefill:\s+([\d.]+)", out).group(1))
    out = run(base + ["--prompt-len", "1", "--gen-len", "128"])
    return float(re.search(r"decode:\s+([\d.]+)", out).group(1))


def llama(gguf_type, test, threads):
    p, n = ("512", "0") if test == "pp512" else ("0", "128")
    out = run([LLAMA_BENCH, "-m", GGUF.format(gguf_type), "-t", str(threads), "-fa", "1",
               "-p", p, "-n", n, "-r", "3", "-o", "json"])
    return json.loads(out)[0]["avg_ts"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    results = {}
    for r in range(args.rounds):
        for dtype, gguf_type in FORMATS:
            for test, threads in TESTS:
                engines = [("ferrite", lambda: ferrite(dtype, test, threads)),
                           ("llama.cpp", lambda: llama(gguf_type, test, threads))]
                for name, fn in engines if r % 2 == 0 else reversed(engines):
                    value = fn()
                    results.setdefault((dtype, test, threads, name), []).append(value)
                    print(f"round {r + 1}: {dtype:<4} {test}@{threads:<2} {name:<9} {value:7.1f} tok/s", flush=True)

    print("\n| weights | test | threads | ferrite tok/s | llama.cpp tok/s | ferrite / llama.cpp |")
    print("|---|---|---:|---:|---:|---:|")
    for dtype, _ in FORMATS:
        for test, threads in TESTS:
            f = statistics.median(results[(dtype, test, threads, "ferrite")])
            l = statistics.median(results[(dtype, test, threads, "llama.cpp")])
            print(f"| {dtype} | {test} | {threads} | {f:.1f} | {l:.1f} | {f / l:.2f}x |")


main()
