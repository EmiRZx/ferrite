# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "transformers>=4.51", "tokenizers"]
# ///
"""Dumps reference outputs from Hugging Face transformers for ferrite's tests.

Usage: uv run scripts/reference.py [model_dir] [out.json]
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3-0.6b"
out_path = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/qwen3-0.6b.json"

TOKENIZER_CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and trailing spaces   ",
    "line one\nline two\r\n\n\ttabbed",
    "Merhaba dünya! Çalışkan öğrenciler şöyle ığdır.",
    "日本語のテキストと中文混合",
    "emoji 🦀🔥 and combining é (NFC)",
    "numbers 1234567890 3.14159 1e-6",
    "fn main() {\n    println!(\"{}\", x + 1);\n}",
    "I'm sure it's what they'd've DONE",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "",
]

PROMPT = (
    "<|im_start|>user\nExplain what a KV cache is in one short paragraph.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
GREEDY_TOKENS = 32
TOP_K = 10

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
model.eval()

tokenizer_cases = [
    {"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in TOKENIZER_CASES
]

prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
with torch.no_grad():
    logits = model(torch.tensor([prompt_ids])).logits[0]
    generated = model.generate(
        torch.tensor([prompt_ids]),
        max_new_tokens=GREEDY_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )[0, len(prompt_ids):].tolist()

# Top-k logits at every prompt position.
positions = []
for row in logits:
    values, ids = row.topk(TOP_K)
    positions.append({"ids": ids.tolist(), "logits": values.tolist()})

out = {
    "model": model_dir,
    "torch": torch.__version__,
    "tokenizer": tokenizer_cases,
    "prompt_ids": prompt_ids,
    "positions": positions,
    "greedy": generated,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {out_path}: {len(prompt_ids)} prompt tokens, greedy -> {tok.decode(generated)!r}")
