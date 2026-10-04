# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "transformers>=4.51"]
# ///
"""Perplexity with Hugging Face transformers (fp32), scored like `ferrite eval`.

Usage: uv run scripts/reference_ppl.py [--ctx 512] [--chunks 4] [--score-from 256]
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/qwen3-0.6b")
parser.add_argument("--file", default="models/wikitext-2-test.txt")
parser.add_argument("--ctx", type=int, default=512)
parser.add_argument("--chunks", type=int, default=4)
parser.add_argument("--score-from", type=int, default=0)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
ids = tok.encode(open(args.file, encoding="utf-8").read(), add_special_tokens=False)

nll, n = 0.0, 0
with torch.no_grad():
    for c in range(args.chunks):
        window = torch.tensor([ids[c * args.ctx:(c + 1) * args.ctx]])
        logp = model(window).logits[0].log_softmax(-1)
        for p in range(args.score_from, args.ctx - 1):
            nll -= logp[p, window[0, p + 1]].item()
            n += 1
print(f"transformers fp32: ppl {math.exp(nll / n):.3f} over {n} tokens")
