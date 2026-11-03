#!/usr/bin/env python3
"""
The model changing shape as it WRITES, one character at a time.

The companion to animate_shape.py. That one teacher-forces held-out text and
shows what reading costs; this one primes the model with a passage and then
lets it continue, so every character on screen is the model's own.

Two things differ from reading, and both are visible in the animation:

  THE WHOLE TEXT CHOOSES THE EXPERTS. The prompt's forward admits the
  experts its characters ask for most, up to the card's slots; then every
  character of the reply is a forward of its own that adds its requests to
  the text's vote and routes among what the prompt and the reply so far have
  voted for, loading whatever is not already on the card. The animation says
  how many each character loaded.

  DECODING HOLDS OFF THE LOOP. Greedy argmax on this model repeats itself; the
  adaptation trace in minagi/decode.py is what the sample log calls the
  `[adapted]` reading, and it is what is used here. Nothing is sampled - the
  output is deterministic and will be the same text twice.

    python3 tools/animate_generate.py --weights weights --chars 140 --prime 2500

assets/generate.gif is the first 15 seconds of that, at the README's width:

    ffmpeg -t 15 -i runs/generate.mp4 -vf "scale=820:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" -loop 0 assets/generate.gif
"""
import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T                                    # noqa: E402
from minagi.decode import pick_next                  # noqa: E402
from minagi.ingest import as_stream, collect         # noqa: E402
from minagi.precision import amp, set_compute_dtype  # noqa: E402
from minagi.pool import PooledMLP                    # noqa: E402
from minagi.tokenizer import ByteTokenizer           # noqa: E402

BG, FG, DIM = "#0d1117", "#e6edf3", "#6e7681"
AMBER, BLUE, GREEN = "#d29922", "#58a6ff", "#3fb950"


def prime_text(subject, n):
    """
    A passage of HELD-OUT text to start from, so nothing is recited.

    Trimmed back to the last sentence that finished inside it. Cutting at an
    arbitrary byte leaves the model halfway through a word, and the first
    thing it writes is then the rest of that word - which says nothing about
    what it can do with a clean start.
    """
    files = [f for f in collect(["data/val"]) if f"/{subject}/" in f]
    data = as_stream(sorted(files)[0])
    s = ByteTokenizer().decode([int(c) for c in data[:n]])
    cut = max(s.rfind(". "), s.rfind(".\n"), s.rfind("! "), s.rfind("? "))
    return s[:cut + 1] + " " if cut > len(s) // 2 else s


def capture(weights, prompt, chars, device, thresh=None,
            strength=2.5, decay=0.88):
    set_compute_dtype("fp32" if device.type == "cpu" else "bf16")
    # read_only: paging an expert in marks it dirty whether or not anything
    # changed it, and the training run may own this directory
    model, cfg, pool, man = T.build_paged(weights, device, read_only=True)
    T.weights_store.load(model, weights, device=device)
    model.eval()
    thresh = thresh if thresh is not None else cfg.halt_thresh
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    for s in sites:
        s.record_weights = True

    tok = ByteTokenizer()
    ids = tok.encode(prompt).ids[-cfg.block:]
    out = torch.tensor([ids], device=device)

    # The prompt starts the text at position 0 and admits its experts, and
    # its last position predicts the first character of the reply - the
    # same order RecurrentLM.generate uses, so no prompt character is read
    # twice.
    caches = model.empty_caches()
    frames = []
    cur, off = out, 0
    for t in range(chars):
        loads_before = pool.loads
        rows = []
        hooks = [s.register_forward_hook(
            lambda m, i, o: rows.append(
                m.last_weight_one.detach().float().cpu().numpy().copy()))
            for s in sites]
        with torch.no_grad(), amp(device):
            logits, extra = model(cur, caches=caches, pos_offset=off,
                                  collect=True)
        for h in hooks:
            h.remove()
        off += cur.shape[1]
        loaded = int(pool.loads - loads_before)
        if t == 0:
            print(f"    the {cur.shape[1]}-character prompt admitted "
                  f"{len(pool._admitted)} experts ({loaded} loaded)",
                  flush=True)

        if cur.shape[1] > 1:
            # the halting mass in per_step is a mean over the positions, so
            # for the prompt the last position's own depth is read directly
            depth = int(extra["steps"][0, -1].item())
        else:
            per = (extra or {}).get("per_step") or []
            cum = [p["cum"] for p in per]
            depth = next((i + 1 for i, c in enumerate(cum) if c >= thresh),
                         len(cum))
        slots = [int(s) for s in pool.slots]
        chosen = [sorted(slots[j] for j in np.nonzero(r)[0] if j < len(slots))
                  for r in rows[:depth]]

        nxt = pick_next(logits[:, -1, :].float(), out, temperature=0.0,
                        adapt_strength=strength, adapt_decay=decay)
        out = torch.cat([out, nxt], dim=1)
        cur = nxt
        frames.append(dict(depth=depth, chosen=chosen, loaded=loaded,
                           ch=tok.decode([int(nxt.item())])))
        if t % 20 == 0:
            print(f"    {t}/{chars}  depth {depth}  loaded {loaded}",
                  flush=True)
    written = tok.decode([int(c) for c in out[0, len(ids):]])
    return frames, cfg.max_steps, man["step"], sites[0].top_k, pool.resident, \
        written


def draw(frames, max_steps, step, k, resident, out, fps, prompt, written):
    ids = sorted({e for f in frames for row in f["chosen"] for e in row})
    base = (list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    col = {e: base[i % len(base)] for i, e in enumerate(ids)}
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"gen_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    depths = [f["depth"] for f in frames]
    tail = prompt.replace("\n", " ")[-30:]

    for t, f in enumerate(frames):
        fig = plt.figure(figsize=(15, 8.6), facecolor=BG)
        fig.text(0.5, 0.955, "the model choosing its own depth while it writes",
                 ha="center", color=FG, fontsize=20, weight="bold")
        fig.text(0.5, 0.917,
                 f"step {step:,}  ·  one row at a time, up to {max_steps}  ·  "
                 f"each row picks {k} of the {resident} experts on the card  ·  "
                 f"colour = expert identity  ·  greedy, no sampling",
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
        ax.text(k + .3, max_steps - f["depth"] + .4,
                f"halted after {f['depth']} row{'s' if f['depth'] > 1 else ''}",
                color=AMBER, fontsize=11, va="center")
        ax.set_xlim(-0.3, k + 5); ax.set_ylim(0, max_steps)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        p = fig.add_axes([0.56, 0.30, 0.39, 0.58], facecolor=BG)
        lo = max(0, t - 80)
        p.plot(range(lo, t + 1), depths[lo:t + 1], color=BLUE, lw=1.4)
        p.scatter([t], [depths[t]], s=45, color=AMBER, zorder=5)
        p.set_ylim(0, max_steps + 1)
        p.set_xlim(lo, max(t + 1, lo + 80))
        p.set_ylabel("rows this character took", color=DIM, fontsize=10)
        p.set_xlabel("character written", color=DIM, fontsize=10)
        p.tick_params(colors=DIM, labelsize=9)
        for s in p.spines.values():
            s.set_color(DIM)
        p.grid(alpha=.2)
        seen = depths[:t + 1]
        # frame 0 is the prompt's own forward; what the reply loads starts
        # with the character after it
        so_far = sum(g["loaded"] for g in frames[1:t + 1])
        p.set_title(f"mean so far {np.mean(seen):.1f} of {max_steps}  ·  "
                    f"range {min(seen)}-{max(seen)}  ·  "
                    f"{so_far} loaded while writing", color=FG, fontsize=12)

        # the prompt stays dim and the model's own text is bright, so the
        # seam between what it was given and what it wrote is never in doubt
        # built from the frames rather than by slicing the decoded reply: a
        # marker is one symbol but decodes to several characters, and slicing
        # would leave the text behind the character the frame says it wrote
        made = "".join(g["ch"] for g in frames[:t + 1]).replace("\n", " ")
        head = (tail + made)[-64:]
        n_made = min(len(made), 64)
        n_head = len(head) - n_made
        # Both halves are drawn from the SAME x with the leading half blanked
        # out, so the monospace advance does the alignment. Positioning the
        # second half by an assumed character width leaves a visible seam,
        # and the seam moves with the font.
        fig.text(0.06, 0.20, head[:n_head], ha="left", color=DIM,
                 fontsize=15, family="monospace")
        fig.text(0.06, 0.20, " " * n_head + head[n_head:], ha="left",
                 color=GREEN, fontsize=15, family="monospace")
        fig.text(0.06, 0.16, " " * (len(head) - 1) + "^", ha="left",
                 color=AMBER, fontsize=15, family="monospace")

        fig.text(0.5, 0.075,
                 f"wrote {f['ch']!r}   ·   {f['depth'] * k} expert "
                 f"evaluations, against {max_steps * k} at the ceiling   ·   "
                 + (f"the prompt's forward loaded {f['loaded']}" if t == 0
                    else f"{f['loaded']} loaded onto the card"),
                 ha="center", color=FG, fontsize=12)
        fig.text(0.5, 0.03,
                 "grey is the prompt it was given, green is its own text  ·  "
                 "the whole text chooses the experts",
                 ha="center", color=DIM, fontsize=10.5)
        fig.savefig(f"{tmp}/{t:05d}.png", facecolor=BG, dpi=80)
        plt.close(fig)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                    str(fps), "-i", f"{tmp}/%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
    print(f"  {len(frames)} frames -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--chars", type=int, default=140,
                    help="characters to generate")
    ap.add_argument("--prime", type=int, default=480,
                    help="characters of held-out text to prime with")
    ap.add_argument("--subject", default="stories")
    ap.add_argument("--prompt", default=None,
                    help="prime with this text instead of a held-out passage")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--adapt-strength", type=float, default=2.5)
    ap.add_argument("--adapt-decay", type=float, default=0.88)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out", default="runs/generate.mp4")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    prompt = a.prompt or prime_text(a.subject, a.prime)
    print(f"  priming with {len(prompt)} characters of "
          f"{'given text' if a.prompt else 'held-out ' + a.subject}")
    frames, ms, step, k, res, written = capture(
        a.weights, prompt, a.chars, dev, a.thresh,
        a.adapt_strength, a.adapt_decay)
    d = [f["depth"] for f in frames]
    ld = [f["loaded"] for f in frames[1:]]
    print(f"\n  wrote {len(d)} characters, depth mean {np.mean(d):.2f}, "
          f"range {min(d)}-{max(d)}, ceiling {ms}")
    print(f"  after the prompt, {np.mean(ld):.1f} experts loaded per "
          f"character ({min(ld)}-{max(ld)}), {sum(1 for x in ld if x)} of "
          f"{len(ld)} characters loaded any")
    print(f"  ---\n{written}\n  ---")
    draw(frames, ms, step, k, res, a.out, a.fps, prompt, written)


if __name__ == "__main__":
    main()
