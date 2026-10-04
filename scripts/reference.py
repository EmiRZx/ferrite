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

# A long, public-domain text (the opening of the US Declaration of Independence) for
# checking long contexts: ~300 tokens span many attention tiles, KV pages and
# prefill chunks.
LONG_TEXT = (
    "When in the Course of human events, it becomes necessary for one people to dissolve "
    "the political bands which have connected them with another, and to assume among the "
    "powers of the earth, the separate and equal station to which the Laws of Nature and of "
    "Nature's God entitle them, a decent respect to the opinions of mankind requires that "
    "they should declare the causes which impel them to the separation.\n\n"
    "We hold these truths to be self-evident, that all men are created equal, that they are "
    "endowed by their Creator with certain unalienable Rights, that among these are Life, "
    "Liberty and the pursuit of Happiness.--That to secure these rights, Governments are "
    "instituted among Men, deriving their just powers from the consent of the governed, "
    "--That whenever any Form of Government becomes destructive of these ends, it is the "
    "Right of the People to alter or to abolish it, and to institute new Government, laying "
    "its foundation on such principles and organizing its powers in such form, as to them "
    "shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will "
    "dictate that Governments long established should not be changed for light and "
    "transient causes; and accordingly all experience hath shewn, that mankind are more "
    "disposed to suffer, while evils are sufferable, than to right themselves by abolishing "
    "the forms to which they are accustomed."
)

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

def top_k(rows):
    """Top-k ids and logits at every position."""
    out = []
    for row in rows:
        values, ids = row.topk(TOP_K)
        out.append({"ids": ids.tolist(), "logits": values.tolist()})
    return out


positions = top_k(logits)

long_ids = tok.encode(LONG_TEXT, add_special_tokens=False)
with torch.no_grad():
    long_logits = model(torch.tensor([long_ids])).logits[0]

out = {
    "model": model_dir,
    "torch": torch.__version__,
    "tokenizer": tokenizer_cases,
    "prompt_ids": prompt_ids,
    "positions": positions,
    "greedy": generated,
    "long_ids": long_ids,
    "long_positions": top_k(long_logits),
}
with open(out_path, "w", encoding="utf-8", newline="\n") as f:  # no CRLF on Windows
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {out_path}: {len(prompt_ids)} prompt tokens, {len(long_ids)} long-context tokens, "
      f"greedy -> {tok.decode(generated)!r}")
