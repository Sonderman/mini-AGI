"""
Which expert runs where, character by character and depth by depth.

X is the text being read, Y is depth - one row per POOLED block-application,
so the bottom of the picture is the first thing that happens to a character
and the top is the last. Colour is the expert's identity, kept stable across
the whole picture, so the same colour in two places really is the same expert.

Only 4 of the 6 blocks route through the pool - the 3 recurrent ones and the
coda - so 6 latent passes give 24 rows. The 2 prelude blocks keep their own
dense feed-forward and have no expert to show.

Read it honestly: top_k experts run at every one of these points, not one.
The colour shows the highest-weighted of them, which is the single expert
contributing most to that position. The lower panels give what one colour
cannot: how much of the full choice carries over from one character to the
next, and which experts are resident at all.

    python3 tools/plot_routing.py --chars 2048 --out runs/routing.png

Runs on CPU by default so it can be taken while a training run has the GPU.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minagi.pool import capture_routes                       # noqa: E402
from minagi.ingest import collect, as_stream                 # noqa: E402
from minagi.precision import set_compute_dtype               # noqa: E402
import train as T                                            # noqa: E402


def gather(model, pool, data, chunk, n_chunks, device):
    """
    Read the text the way the trainer does, recording every routing decision.

    Each chunk is read as a new text, as each training window is, so it
    admits afresh and the loads this picture shows are the loads the rule
    really makes.
    """
    top, share, resident, bounds = [], [], [], []
    pos = 0
    for c in range(n_chunks):
        ids = torch.from_numpy(
            data[pos:pos + chunk].astype(np.int64)).to(device).unsqueeze(0)
        if ids.shape[1] < 2:
            break
        bounds.append(len(top[0]) if top else 0)
        loads_before = pool.loads
        with torch.no_grad(), capture_routes() as picks:
            model(ids, caches=None, pos_offset=0)      # admits, as in training
        resident.append((sorted(int(e) for e in pool.slots if e >= 0),
                         pool.loads - loads_before))
        # picks[d] is (ids[T,k], weights[T,k]) for depth d
        best = np.stack([p[0][np.arange(p[0].shape[0]),
                              p[1].argmax(-1)].numpy() for p in picks])
        cur = np.stack([p[0].numpy() for p in picks])   # [D, T, k]
        if top:
            prev = top[-1][1]
            carry = np.zeros(cur.shape[1])
            for t in range(cur.shape[1]):
                a = cur[:, t, :]
                b = prev[:, t - 1, :] if t else prev[:, -1, :]
                carry[t] = np.mean([len(set(a[d]) & set(b[d])) / a.shape[1]
                                    for d in range(a.shape[0])])
        else:
            carry = np.full(cur.shape[1], np.nan)
        top.append((best, cur))
        share.append(carry)
        pos += chunk
    return top, share, resident


def draw(top, share, resident, chunk, out, title):
    best = np.concatenate([b for b, _ in top], axis=1)      # [D, T]
    carry = np.concatenate(share)
    D, Tn = best.shape

    used = sorted(set(int(v) for v in best.ravel()))
    # a stable colour per expert id, so the same expert is the same colour
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    cmap = ListedColormap([base[i % 20] for i in range(len(used))])
    remap = {e: i for i, e in enumerate(used)}
    img = np.vectorize(remap.get)(best)

    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True,
                           gridspec_kw={"height_ratios": [4, 1.3, 1.3]})
    fig.suptitle(title, fontsize=13, y=0.98)

    ax[0].imshow(img, aspect="auto", interpolation="nearest", cmap=cmap,
                 origin="lower", extent=[0, Tn, -0.5, D - 0.5])
    ax[0].set_ylabel("depth\n(pooled block application)")
    ax[0].set_title(
        f"highest-weighted of the {top[0][1].shape[2]} experts running at each "
        f"point — {len(used)} distinct experts appear\n"
        f"{D} pooled applications = {D // 4} latent passes x (3 recurrent "
        f"blocks + 1 coda); the 2 prelude blocks keep a dense feed-forward "
        f"and have no expert to show", fontsize=9, pad=8)
    for b in range(chunk, Tn, chunk):
        ax[0].axvline(b, color="w", lw=1.4)
        ax[0].axvline(b, color="k", lw=0.6, ls=":")

    ax[1].plot(np.arange(Tn), carry, lw=0.8, color="#2c3e50")
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("carried over\nfrom previous\ncharacter")
    ax[1].grid(alpha=.25)
    for b in range(chunk, Tn, chunk):
        ax[1].axvline(b, color="k", lw=0.6, ls=":")

    # which experts were resident for each chunk, and how many had to be loaded
    for i, (res, moved) in enumerate(resident):
        for e in res:
            ax[2].plot([i * chunk, (i + 1) * chunk],
                       [remap.get(e, -1)] * 2,
                       lw=2.4, solid_capstyle="butt",
                       color=cmap(remap[e]) if e in remap else "#dddddd")
        ax[2].text(i * chunk + chunk * 0.5, len(used) + 0.5,
                   f"+{moved}", ha="center", fontsize=8, color="#c0392b")
    ax[2].set_ylabel("resident experts\n(one line each)")
    ax[2].set_xlabel("character read  —  dotted lines are chunk boundaries, "
                     "where a new text admits afresh; +n is how many "
                     "were loaded onto the card")
    ax[2].set_ylim(-1, len(used) + 2)
    ax[2].grid(alpha=.2, axis="x")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({D} depths x {Tn} characters, "
          f"{len(used)} distinct experts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--text", default=None, help="file to read; default picks "
                                                 "one from data/val")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--chars", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--out", default="runs/routing.png")
    a = ap.parse_args()

    device = torch.device(a.device)
    set_compute_dtype(a.precision)
    # read_only: the training run may own this directory, and paging an
    # expert in marks it dirty whether or not anything changed it
    model, cfg, pool, man = T.build_paged(a.weights, device, None, 96,
                                          read_only=True)
    model.eval()

    path = a.text
    if path is None:
        files = collect(["data/val"]) or collect(["data/train"])
        path = files[len(files) // 2]
    data = as_stream(path)
    n_chunks = max(1, a.chars // a.chunk)
    print(f"reading {a.chars} characters of {os.path.relpath(path)} "
          f"in {n_chunks} chunks of {a.chunk}, on {a.device}", flush=True)

    top, share, resident = gather(model, pool, data, a.chunk, n_chunks, device)
    draw(top, share, resident, a.chunk, a.out,
         f"mini-AGI — which expert runs at each depth, reading "
         f"{os.path.basename(os.path.dirname(path))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
