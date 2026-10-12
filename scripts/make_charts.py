# /// script
# requires-python = ">=3.10"
# ///
"""Renders the README charts as SVG, in a light and a dark variant.

The numbers are the medians recorded in docs/llama-cpp-comparison.md.
Usage: uv run scripts/make_charts.py  (writes docs/assets/*-{light,dark}.svg)
"""

import os

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926"),
}
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'

# ferrite throughput / llama.cpp throughput, 8 threads each.
RATIOS = {
    "Decode (tg128)": [("bf16", 1.05), ("Q8", 1.03), ("Q4", 0.94)],
    "Prefill (pp512)": [("bf16", 0.25), ("Q8", 1.10), ("Q4", 0.64)],
}
# Total tok/s versus concurrent requests (Q8, 128-token streams).
CONCURRENCY = [1, 4, 8, 16]
FERRITE = [63.5, 169.1, 222.2, 251.1]
LLAMA_RUNS = [(59.7, 61.7), (138.2, 136.1), (160.6, 224.8), (242.6, 252.5)]


def text(x, y, s, fill, size=13, anchor="start", weight="normal", halo=None):
    """Text; `halo` paints a surface-colored outline so it stays legible over lines."""
    outline = f'stroke="{halo}" stroke-width="4" paint-order="stroke" ' if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" {outline}'
            f'text-anchor="{anchor}" font-weight="{weight}" {FONT}>{s}</text>')


def bar(x0, y, x1, h, fill, r=4):
    """Horizontal bar, square at the baseline, rounded at the data end."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{fill}"/>')


def svg(width, height, body, t, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
            f'<rect width="{width}" height="{height}" rx="8" fill="{t["surface"]}"/>{body}</svg>')


def ratio_chart(t):
    width, height = 760, 210
    panel_w, gap, left = 300, 70, 60
    body = [text(24, 30, "ferrite throughput relative to llama.cpp, 8 threads (1.0 = parity)", t["ink"], 15, weight="600")]
    for p, (title, rows) in enumerate(RATIOS.items()):
        x0 = left + p * (panel_w + gap)
        scale = lambda v: x0 + v / 1.2 * panel_w  # axis 0 .. 1.2
        top = 62
        body.append(text(x0, top - 10, title, t["ink2"], 13, weight="600"))
        for v in (0, 0.5, 1.0):
            x = scale(v)
            color = t["axis"] if v == 0 else (t["ink2"] if v == 1.0 else t["grid"])
            body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + 3 * 34 - 6}" stroke="{color}" stroke-width="1"/>')
            body.append(text(x, top + 3 * 34 + 12, f"{v:g}x", t["muted"], 12, "middle"))
        for i, (label, v) in enumerate(rows):
            y = top + i * 34 + 4
            body.append(text(x0 - 10, y + 15, label, t["ink2"], 13, "end"))
            body.append(bar(scale(0), y, scale(v), 20, t["s1"]))
            body.append(text(scale(v) + 6, y + 15, f"{v:.2f}x", t["ink"], 13, halo=t["surface"]))
    return svg(width, height, "".join(body), t, "ferrite throughput relative to llama.cpp")


def serving_chart(t):
    width, height = 760, 300
    left, right, top, bottom = 70, 600, 70, 250
    xs = {c: left + (right - left) * (i / (len(CONCURRENCY) - 1)) for i, c in enumerate(CONCURRENCY)}
    y = lambda v: bottom - v / 300 * (bottom - top)
    body = [text(24, 30, "Serving throughput: total tok/s vs concurrent requests (Q8)", t["ink"], 15, weight="600")]
    for v in (0, 100, 200, 300):
        color = t["axis"] if v == 0 else t["grid"]
        body.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{right}" y2="{y(v):.1f}" stroke="{color}" stroke-width="1"/>')
        body.append(text(left - 10, y(v) + 4, f"{v}", t["muted"], 12, "end"))
    for c in CONCURRENCY:
        body.append(text(xs[c], bottom + 20, str(c), t["muted"], 12, "middle"))
    body.append(text((left + right) / 2, bottom + 40, "concurrent requests", t["muted"], 12, "middle"))

    # llama-server: band between its two runs, line through their mean.
    lo = [min(r) for r in LLAMA_RUNS]
    hi = [max(r) for r in LLAMA_RUNS]
    band = " ".join(f"{xs[c]:.1f},{y(v):.1f}" for c, v in zip(CONCURRENCY, hi))
    band += " " + " ".join(f"{xs[c]:.1f},{y(v):.1f}" for c, v in reversed(list(zip(CONCURRENCY, lo))))
    body.append(f'<polygon points="{band}" fill="{t["s2"]}" fill-opacity="0.12"/>')
    for color, values in ((t["s2"], [sum(r) / 2 for r in LLAMA_RUNS]), (t["s1"], FERRITE)):
        pts = " ".join(f"{xs[c]:.1f},{y(v):.1f}" for c, v in zip(CONCURRENCY, values))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for c, v in zip(CONCURRENCY, values):
            body.append(f'<circle cx="{xs[c]:.1f}" cy="{y(v):.1f}" r="4.5" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')

    # Legend (two series), then a single direct label where the gap is largest.
    lx = right + 30
    for i, (name, color) in enumerate((("ferrite", t["s1"]), ("llama-server", t["s2"]))):
        yy = top + 10 + i * 24
        body.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 18}" y2="{yy}" stroke="{color}" stroke-width="2"/>')
        body.append(f'<circle cx="{lx + 9}" cy="{yy}" r="4" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
        body.append(text(lx + 26, yy + 4, name, t["ink2"], 13))
    body.append(text(lx, top + 66, "band: two llama-server", t["muted"], 11))
    body.append(text(lx, top + 80, "runs, line: their mean", t["muted"], 11))
    body.append(text(xs[4], y(FERRITE[1]) - 12, "169 vs 137 tok/s", t["ink"], 12, "middle", halo=t["surface"]))
    return svg(width, height, "".join(body), t, "Serving throughput versus concurrent requests")


def main():
    os.makedirs("docs/assets", exist_ok=True)
    for name, t in THEMES.items():
        for chart, fn in (("vs-llamacpp", ratio_chart), ("serving", serving_chart)):
            path = f"docs/assets/{chart}-{name}.svg"
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(fn(t))
            print("wrote", path)


main()
