# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Records a real `ferrite run` and renders it as a terminal GIF with the true timing
of the streamed tokens (only the typing of the command and the model load are
shortened).

Usage: uv run scripts/make_demo_gif.py [--out docs/assets/demo.gif]
"""

import argparse
import os
import subprocess
import textwrap
import threading
import time

from PIL import Image, ImageDraw, ImageFont

PROMPT = "Explain how a CPU cache works in three sentences."
COLS, ROWS = 92, 16
FONT_SIZE, PAD = 16, 14
BG, FG, DIM, ACCENT = (24, 26, 33), (220, 223, 228), (128, 134, 148), (126, 200, 160)


def record(exe):
    """Runs ferrite and returns [(seconds, stream, text)] in arrival order."""
    cmd = [exe, "-d", "q4", "run", PROMPT, "-t", "0", "-n", "160"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    start, events, lock = time.perf_counter(), [], threading.Lock()

    def pump(pipe, name):
        while chunk := os.read(pipe.fileno(), 256):
            with lock:
                events.append((time.perf_counter() - start, name, chunk))

    threads = [threading.Thread(target=pump, args=(p, n)) for p, n in ((proc.stdout, "out"), (proc.stderr, "err"))]
    for t in threads:
        t.start()
    proc.wait()
    for t in threads:
        t.join()
    events.sort(key=lambda e: e[0])
    return [(t, n, c.decode("utf-8", errors="replace").replace("\r", "")) for t, n, c in events]


def timeline(events):
    """Turns the recording into [(seconds, [(text, color)])] screen states."""
    command = f'$ ferrite -d q4 run "{PROMPT}" -t 0'
    states, t = [], 0.0
    for i in range(1, len(command) + 1, 3):  # quick typing
        states.append((t, [(command[:i], FG)]))
        t += 0.03
    lines = [(command, FG)]
    states.append((t, list(lines)))
    t += 0.5

    out, err = "", ""
    load_end = next((et for et, n, c in events if n == "err" and "loaded" in c), 0.0)
    for et, name, chunk in events:
        if name == "err":
            err += chunk
        else:
            out += chunk
        # Show the model load as 0.6 s; everything after it keeps its real timing.
        shown = t + 0.6 + max(0.0, et - load_end)
        screen = [(command, FG)]
        for line in err.splitlines():
            if line.startswith("loaded"):
                screen.append((line, DIM))
        for para in out.split("\n"):
            screen += [(w, FG) for w in (textwrap.wrap(para, COLS) or [""])]
        for line in err.splitlines():
            if line.startswith("prompt:"):
                screen.append((line, ACCENT))
        states.append((shown, screen))
    return states


def render(states, out_path, preview=False):
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    cw, ch = font.getbbox("M")[2], int(FONT_SIZE * 1.4)
    size = (COLS * cw + 2 * PAD, ROWS * ch + 2 * PAD)

    def draw(screen):
        img = Image.new("RGB", size, BG)
        d = ImageDraw.Draw(img)
        for row, (text, color) in enumerate(screen[-ROWS:]):
            d.text((PAD, PAD + row * ch), text, font=font, fill=color)
        return img

    # Sample the screen every FRAME_MS (what a viewer would see), merging repeats.
    FRAME_MS = 40
    end = states[-1][0] + 4.0  # hold the final screen
    frames, durations, shown, i = [], [], None, 0
    for k in range(int(end * 1000 / FRAME_MS) + 1):
        t = k * FRAME_MS / 1000
        while i + 1 < len(states) and states[i + 1][0] <= t:
            i += 1
        if states[i][1] == shown:
            durations[-1] += FRAME_MS
            continue
        shown = states[i][1]
        frames.append(draw(shown))
        durations.append(FRAME_MS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if preview:
        frames[len(frames) // 2].save(out_path.replace(".gif", "-mid.png"))
        frames[-1].save(out_path.replace(".gif", "-last.png"))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out_path}: {len(frames)} frames, {sum(durations) / 1000:.1f} s, {os.path.getsize(out_path) // 1024} KiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/assets/demo.gif")
    parser.add_argument("--preview", action="store_true", help="also save middle/last frames as PNG")
    args = parser.parse_args()
    exe = os.path.abspath("target/release/ferrite" + (".exe" if os.name == "nt" else ""))
    render(timeline(record(exe)), args.out, args.preview)


main()
