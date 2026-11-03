"""
Animate a routing trace recorded inside a real training run.

This tool has no model and cannot run one. Everything it draws came out of
train.py's own forward pass - the one whose loss was backpropagated - so
there is nothing here that could drift from what training did.

    python3 train.py read --trace-routes runs/trace.npz --trace-chunks 8 ...
    python3 tools/animate_trace.py runs/trace.npz --out runs/training.mp4

Colour is an expert's identity and is stable throughout, so the same colour
anywhere really is the same expert. The right-hand panel shows every expert
running for the current character, at all pooled depths, heaviest at left -
so persistence is visible as colour staying put rather than averaged away.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.tokenizer import ByteTokenizer                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default="runs/training.mp4")
    ap.add_argument("--trail", type=int, default=256)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1,
                    help="draw every Nth character, for a shorter animation")
    a = ap.parse_args()

    z = np.load(a.trace, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    chunks = meta["chunks"]
    ids = z["ids"]                       # [C, D, T, k] expert ids
    wts = z["weights"].astype(np.float32)
    text_ids = z["text"]

    C, D, Tn, K = ids.shape
    order = np.argsort(-wts, axis=-1)
    ids = np.take_along_axis(ids, order, axis=-1)
    flat = ids.transpose(1, 0, 2, 3).reshape(D, C * Tn, K)   # [D, C*T, K]
    best = flat[:, :, 0]
    total = C * Tn

    tok = ByteTokenizer()
    text = tok.decode([int(v) for v in text_ids.reshape(-1)])

    used = sorted(set(int(v) for v in flat.ravel()))
    # A colour is an EXPERT'S IDENTITY, so it may not repeat: cycling a
    # twenty-colour map gave two different experts the same colour as soon as
    # more than twenty appeared, and the docstring's promise above - that the
    # same colour anywhere is the same expert - was then false exactly where
    # the picture was interesting. Hues walk by the golden ratio so
    # consecutive indices land far apart on the wheel, with lightness and
    # saturation cycling underneath on periods that share no factor with it.
    import colorsys
    cmap = ListedColormap([
        colorsys.hls_to_rgb((i * 0.6180339887) % 1.0,
                            (0.58, 0.72, 0.46)[i % 3],
                            (0.85, 0.62, 0.95, 0.74)[i % 4])
        for i in range(max(len(used), 1))])
    cmap.set_bad("#f2f2f2")
    remap = {e: i for i, e in enumerate(used)}
    trace = np.vectorize(remap.get)(best).astype(float)
    grid = np.vectorize(remap.get)(flat).astype(float)

    # which experts are on the card for each chunk, and which of them are new
    res = [list(m["resident"]) for m in chunks]
    fresh = [set() if i == 0 else set(res[i]) - set(res[i - 1])
             for i in range(len(res))]
    n_res = max(len(r) for r in res)

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.4, 1.15],
                          height_ratios=[0.55, 5, 0.9],
                          hspace=0.34, wspace=0.09,
                          top=0.875, bottom=0.085, left=0.070, right=0.985)
    ax_txt = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_now = fig.add_subplot(gs[1, 1])
    ax_res = fig.add_subplot(gs[2, :])

    ax_txt.axis("off")
    t_prev = ax_txt.text(0.0, 0.55, "", family="monospace", fontsize=10.5,
                         color="#7f8c8d", va="center", transform=ax_txt.transAxes)
    t_head = ax_txt.text(0.0, 0.02, "", family="monospace", fontsize=11.5,
                         color="#c0392b", va="center", transform=ax_txt.transAxes)

    im = ax_map.imshow(np.zeros((D, a.trail)), aspect="auto", cmap=cmap,
                       vmin=0, vmax=max(len(used) - 1, 1), origin="lower",
                       interpolation="nearest")
    ax_map.set_ylabel("depth (pooled block application)")
    ax_map.set_xlabel(f"the last {a.trail} characters of the training read")
    ax_map.set_xticks([])
    ax_map.axvline(a.trail - 0.5, color="k", lw=2.0)

    # One column per RESIDENT expert, in a fixed order that matches the row
    # of cells at the bottom. A column that stays lit as the characters go by
    # is an expert that keeps being chosen - which is the thing a colour-by-
    # rank picture cannot show, because the heaviest of the same eight
    # changes constantly while the eight themselves do not.
    now = ax_now.imshow(np.full((D, n_res), np.nan), aspect="auto", cmap=cmap,
                        vmin=0, vmax=max(len(used) - 1, 1), origin="lower",
                        interpolation="nearest")
    ax_now.set_title(f"this character\nwhich of the {n_res} resident\n"
                     f"run at each depth\n({K} per depth; columns\n"
                     f"match the row below)", fontsize=8)
    ax_now.set_xticks([]); ax_now.set_yticks([])

    # The resident set: what is on the card. It changes when a training
    # step's forward admits, not once a character - the point of the panel is
    # that it sits still while everything above it flickers.
    ax_res.set_xlim(0, n_res)
    ax_res.set_ylim(0, 1)
    ax_res.set_xticks([])
    ax_res.set_yticks([])
    ax_res.set_ylabel("resident", fontsize=9)
    cells = [Rectangle((i, 0.18), 0.92, 0.64, lw=0) for i in range(n_res)]
    for c in cells:
        ax_res.add_patch(c)
    marks = [Rectangle((i, 0.18), 0.92, 0.64, fill=False, lw=0) 
             for i in range(n_res)]
    for m_ in marks:
        ax_res.add_patch(m_)
    labels = [ax_res.text(i + 0.46, 0.5, "", ha="center", va="center",
                          fontsize=7, color="#222222") for i in range(n_res)]
    res_cap = ax_res.text(0.5, -0.34, "", fontsize=8.5, color="#444444",
                          ha="center", transform=ax_res.transAxes)

    title = fig.suptitle("", fontsize=12, y=0.995)
    prov = (f"recorded inside train.py — {meta['precision']}, chunk "
            f"{meta['chunk']}, context {meta['context']:,} carried,\n"
            f"top-{K} of the resident set at each of {D} pooled depths, "
            f"gradient step taken after every chunk")
    fig.text(0.5, 0.955, prov, ha="center", va="top", fontsize=8.5,
             color="#666666")

    def frame(fi):
        t = min(fi * a.stride, total - 1)
        c = t // Tn
        m = chunks[c]
        lo = max(0, t - a.trail + 1)
        win = trace[:, lo:t + 1]
        if win.shape[1] < a.trail:
            win = np.pad(win, ((0, 0), (a.trail - win.shape[1], 0)),
                         constant_values=np.nan)
        im.set_data(np.ma.masked_invalid(win))
        # depth x resident slot: lit where that expert is one of this
        # character's K at that depth
        slot_of = {e: i for i, e in enumerate(res[c])}
        panel = np.full((D, n_res), np.nan)
        for d in range(D):
            for e in flat[d, t]:
                i = slot_of.get(int(e))
                if i is not None:
                    panel[d, i] = remap[int(e)]
        now.set_data(np.ma.masked_invalid(panel))
        s = max(0, t - 70)
        t_prev.set_text(repr(text[s:t])[1:-1][-115:])
        t_head.set_text(f"next: {text[t:t + 1]!r}")
        # the experts this step's forward admitted
        cur_res, new = res[c], fresh[c]
        for i, cell in enumerate(cells):
            if i < len(cur_res):
                e = cur_res[i]
                cell.set_visible(True)
                cell.set_facecolor(cmap(remap[e]) if e in remap else "#dddddd")
                labels[i].set_text(str(e))
                marks[i].set_visible(e in new)
                marks[i].set_edgecolor("#c0392b")
                marks[i].set_linewidth(2.2)
            else:
                cell.set_visible(False)
                marks[i].set_visible(False)
                labels[i].set_text("")
        res_cap.set_text(
            f"{len(cur_res)} experts on the card — admitted by this step's "
            f"forward, fixed once the card is full"
            + (f", {len(new)} outlined in red were loaded here" if new else "")
            + "\nthe colours above change every character because ROUTING is "
              "per character; this row changes only when a step admits")

        boundary = (t % Tn) == 0
        title.set_text(
            f"step {m['step']:,} · reading {m['subject']} · loss {m['loss']:.3f}"
            f" · {m['experts']} experts, {len(m['resident'])} resident"
            + (f"   —   a new step, {m['moved']} loaded"
               if boundary else "")
            )
        for sp in ax_now.spines.values():
            sp.set_edgecolor("#c0392b" if boundary else "#cccccc")
            sp.set_linewidth(2.5 if boundary else 0.8)
        return im, now, t_prev, t_head, title

    n_frames = total // a.stride
    print(f"{C} chunks x {Tn} characters, {D} depths, top-{K}, "
          f"{len(used)} distinct experts -> {n_frames} frames", flush=True)
    anim = animation.FuncAnimation(fig, frame, frames=n_frames,
                                   interval=1000 / a.fps, blit=False)
    anim.save(a.out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=2600))
    print(f"wrote {a.out}  ({n_frames / a.fps:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
