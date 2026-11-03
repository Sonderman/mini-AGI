#!/usr/bin/env python3
"""
The model changing shape as it reads, one character at a time.

Captured from the model - n_recur=1, n_coda=0 - where halting speaks after
every row, so the depth a character costs is whatever that character needed.
Nothing here is drawn by hand: the depth is the row at which halting's
cumulative mass crosses halt_thresh, and the experts are the ones each row
actually ran.

It reads HELD-OUT text. The first chunk is read in one forward, which admits
the experts its characters ask for most, up to the card's slots; every
character after it is read one at a time through the attention cache, and
each of those forwards admits the experts that character asks for, loading
whatever is not already on the card.

Two things move together:

  the stack     how many rows this character took, and which experts each row
                composed itself from. Colour is expert identity and is stable
                for the whole animation, so the same colour is always the
                same expert.
  the text      what has been read, with the current character marked, and the
                depth of the last few characters as a trace.

    python3 tools/animate_shape.py --weights weights --chars 160

assets/shape.gif is the first 15 seconds of that, at the README's width:

    ffmpeg -t 15 -i runs/shape.mp4 -vf "scale=820:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" -loop 0 assets/shape.gif
"""
import argparse
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T                                    # noqa: E402
from minagi.ingest import as_stream, collect         # noqa: E402
from minagi.precision import amp, set_compute_dtype  # noqa: E402
from minagi.pool import PooledMLP                    # noqa: E402
from minagi.tokenizer import ByteTokenizer           # noqa: E402

BG, FG, DIM = "#0d1117", "#e6edf3", "#6e7681"
AMBER, BLUE, GREEN = "#d29922", "#58a6ff", "#3fb950"


def capture(weights, chars, subject, device, thresh=None, root="data/val",
            warm=2048):
    set_compute_dtype("fp32" if device.type == "cpu" else "bf16")
    # read_only: the training run may own this directory, and paging an
    # expert in marks it dirty whether or not anything changed it
    model, cfg, pool, man = T.build_paged(weights, device, read_only=True)
    T.weights_store.load(model, weights, device=device)
    model.eval()
    thresh = thresh if thresh is not None else cfg.halt_thresh
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    for s in sites:
        s.record_weights = True

    files = [f for f in collect([root]) if f"/{subject}/" in f]
    data = as_stream(sorted(files)[0])
    ids = list(np.asarray(data[:warm + chars + 1], np.int64))
    tok = ByteTokenizer()

    # the first chunk is one forward, which admits the experts its characters
    # ask for most; after it, every character is a forward of its own
    caches = model.empty_caches()
    cur = torch.tensor([ids[:warm]], device=device)
    with torch.no_grad(), amp(device):
        model(cur, caches=caches, pos_offset=0)
    print(f"    the first {warm} characters admitted "
          f"{len(pool._admitted)} experts ({pool.loads} loaded)", flush=True)
    loads_after_warm = pool.loads

    frames = []
    off = warm
    for t in range(chars):
        loads_before = pool.loads
        rows = []
        hooks = [s.register_forward_hook(
            lambda m, i, o: rows.append(
                m.last_weight_one.detach().float().cpu().numpy().copy()))
            for s in sites]
        one = torch.tensor([[ids[warm + t]]], device=device)
        with torch.no_grad(), amp(device):
            logits, extra = model(one, caches=caches, pos_offset=off,
                                  collect=True)
        for h in hooks:
            h.remove()
        off += 1
        per = (extra or {}).get("per_step") or []
        cum = [p["cum"] for p in per]
        depth = next((i + 1 for i, c in enumerate(cum) if c >= thresh),
                     len(cum))
        slots = [int(s) for s in pool.slots]
        chosen = [sorted(slots[j] for j in np.nonzero(r)[0] if j < len(slots))
                  for r in rows[:depth]]
        frames.append(dict(depth=depth, chosen=chosen,
                           loaded=int(pool.loads - loads_before),
                           ch=tok.decode([ids[warm + t]]),
                           nxt=tok.decode([int(logits[0, -1].argmax())]),
                           cum=cum))
        if t % 20 == 0:
            print(f"    {t}/{chars}  depth {depth}", flush=True)
    print(f"    {pool.loads - loads_after_warm} loaded while reading the "
          f"{chars} characters shown", flush=True)
    return frames, cfg.max_steps, man["step"], sites[0].top_k, pool.resident


def draw(frames, max_steps, step, k, resident, out, fps, text, subject):
    ids = sorted({e for f in frames for row in f["chosen"] for e in row})
    base = (list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    col = {e: base[i % len(base)] for i, e in enumerate(ids)}
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                       f"shape_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    depths = [f["depth"] for f in frames]
    for t, f in enumerate(frames):
        fig = plt.figure(figsize=(15, 8.6), facecolor=BG)
        fig.text(0.5, 0.955, "the model choosing its own depth, character by "
                 "character", ha="center", color=FG, fontsize=20, weight="bold")
        fig.text(0.5, 0.917,
                 f"step {step:,}  ·  held-out {subject}  ·  one row at a "
                 f"time, up to {max_steps}  ·  each row picks {k} of the "
                 f"{resident} experts on the card  ·  colour = expert identity",
                 ha="center", color=DIM, fontsize=11)

        ax = fig.add_axes([0.06, 0.30, 0.42, 0.58], facecolor=BG)
        for r in range(max_steps):
            y = max_steps - 1 - r
            if r < len(f["chosen"]):
                for c, e in enumerate(f["chosen"][r][:k]):
                    ax.add_patch(plt.Rectangle((c, y), .9, .86, color=col[e],
                                               ec=BG, lw=.8))
            else:
                for c in range(k):
                    ax.add_patch(plt.Rectangle((c, y), .9, .86,
                                               color="#161b22", ec=BG, lw=.8))
        ax.axhline(max_steps - f["depth"] - .07, color=AMBER, lw=1.6)
        ax.text(k + .3, max_steps - f["depth"] + .4, f"halted after "
                f"{f['depth']} row{'s' if f['depth'] > 1 else ''}",
                color=AMBER, fontsize=11, va="center")
        ax.set_xlim(-0.3, k + 5); ax.set_ylim(0, max_steps)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        p = fig.add_axes([0.56, 0.30, 0.39, 0.58], facecolor=BG)
        lo = max(0, t - 80)
        p.plot(range(lo, t + 1), depths[lo:t + 1], color=GREEN, lw=1.4)
        p.scatter([t], [depths[t]], s=45, color=AMBER, zorder=5)
        p.set_ylim(0, max_steps + 1)
        p.set_xlim(lo, max(t + 1, lo + 80))
        p.set_ylabel("rows this character took", color=DIM, fontsize=10)
        p.set_xlabel("character", color=DIM, fontsize=10)
        p.tick_params(colors=DIM, labelsize=9)
        for s in p.spines.values():
            s.set_color(DIM)
        p.grid(alpha=.2)
        seen = depths[:t + 1]
        so_far = sum(g["loaded"] for g in frames[:t + 1])
        p.set_title(f"mean so far {np.mean(seen):.1f} of {max_steps}  ·  "
                    f"range {min(seen)}-{max(seen)}  ·  {so_far} loaded",
                    color=FG, fontsize=12)

        shown = text[max(0, t - 58):t + 1].replace("\n", " ")[-59:]
        fig.text(0.08, 0.20, shown, ha="left", color=FG, fontsize=16,
                 family="monospace")
        fig.text(0.08, 0.16, " " * (len(shown) - 1) + "^", ha="left",
                 color=AMBER, fontsize=16, family="monospace")
        fig.text(0.5, 0.075,
                 f"reading {f['ch']!r}   ·   {f['depth'] * k} expert "
                 f"evaluations, against {max_steps * k} at the ceiling   ·   "
                 f"{f['loaded']} loaded onto the card",
                 ha="center", color=FG, fontsize=12)
        fig.text(0.5, 0.03,
                 "empty rows are computation the model declined to spend  ·  "
                 "the whole text chooses the experts",
                 ha="center", color=DIM, fontsize=10.5)
        fig.savefig(f"{tmp}/{t:05d}.png", facecolor=BG, dpi=80)
        plt.close(fig)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                    str(fps), "-i", f"{tmp}/%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {len(frames)} frames -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--chars", type=int, default=160)
    ap.add_argument("--subject", default="stories")
    ap.add_argument("--root", default="data/val",
                    help="where the subject's text is; held-out by default")
    ap.add_argument("--warm", type=int, default=2048,
                    help="characters read as the first chunk, which starts "
                         "the text and admits its experts, before the "
                         "first frame")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thresh", type=float, default=None,
                    help="halt threshold; defaults to the model's own")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out", default="runs/shape.mp4")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    frames, ms, step, k, res = capture(a.weights, a.chars, a.subject, dev,
                                       a.thresh, a.root, a.warm)
    d = [f["depth"] for f in frames]
    ld = [f["loaded"] for f in frames]
    print(f"  depth over {len(d)} characters: mean {np.mean(d):.2f}, "
          f"range {min(d)}-{max(d)}, ceiling {ms}")
    print(f"  {np.mean(ld):.1f} experts loaded per character "
          f"({min(ld)}-{max(ld)}), {sum(1 for x in ld if x)} of {len(ld)} "
          f"characters loaded any")
    text = "".join(f["ch"] for f in frames)
    draw(frames, ms, step, k, res, a.out, a.fps, text, a.subject)


if __name__ == "__main__":
    main()
