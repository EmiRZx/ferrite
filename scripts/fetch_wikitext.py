# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow"]
# ///
"""Downloads the WikiText-2 (raw) test split as plain text for `ferrite eval`.

Usage: uv run scripts/fetch_wikitext.py [out.txt]
"""

import io
import sys
import urllib.request

import pyarrow.parquet as pq

URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
out = sys.argv[1] if len(sys.argv) > 1 else "models/wikitext-2-test.txt"

data = urllib.request.urlopen(URL).read()
lines = pq.read_table(io.BytesIO(data)).column("text").to_pylist()
with open(out, "w", encoding="utf-8") as f:
    f.write("".join(lines))
print(f"wrote {out} ({len(lines)} lines)")
