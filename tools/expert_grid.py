#!/usr/bin/env python3
"""
The model as a grid, for ONE character: every row is a block application,
every cell is an expert it ran.

Colour is expert IDENTITY and is stable across the whole figure, so the same
colour anywhere really is the same expert. That is the thing the picture is
for: seeing which experts a character keeps coming back to as it goes deeper,
and which appear once.

Rows are block applications in the order they run. With `depth` latent passes
there are depth x 4 of them - three recur blocks and a coda per pass. Columns
are the top-k experts that row chose, sorted by id so a column means nothing
and only the colour does.

Gate strength is deliberately NOT shown. This is about which experts ran
where, not how much each contributed.

    python3 tools/expert_grid.py --depths 1,4 --out runs/expert_grid.png
"""
import argparse
import os
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

BG, FG, DIM = "#0d1117", "#e6edf3", "#6e7681"


def capture(weights, depth, chars, subject, device):
    set_compute_dtype("fp32" if device.type == "cpu" else "bf16")
    # read_only: the training run may own this directory, and paging an
    # expert in marks it dirty whether or not anything changed it
    model, cfg, pool, man = T.build_paged(weights, device, read_only=True)
    T.weights_store.load(model, weights, device=device)
    model.eval()
    cfg.max_steps = depth
    cfg.min_steps = min(cfg.min_steps, depth)
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    for s in sites:
        s.record_weights = True
    rows = []
    hooks = [s.register_forward_hook(
        lambda m, i, o: rows.append(
            m.last_weight_one.detach().float().cpu().numpy().copy()))
        for s in sites]
    files = [f for f in collect(["data/train"]) if f"/{subject}/" in f]
    data = as_stream(sorted(files)[0])
    ids = torch.tensor(np.asarray(data[:chars], np.int64)[None], device=device)
    with torch.no_grad(), amp(device):
        logits, extra = model(ids, collect=True)
    for h in hooks:
        h.remove()
    slots = [int(s) for s in pool.slots]
    # rows[i] is a vector over SLOTS; turn it into the expert ids that ran
    chosen = [sorted(slots[j] for j in np.nonzero(r)[0] if j < len(slots))
              for r in rows]
    tok = __import__("minagi.tokenizer",
                     fromlist=["ByteTokenizer"]).ByteTokenizer()
    return dict(chosen=chosen, depth=depth, step=man["step"],
                top_k=sites[0].top_k, resident=pool.resident,
                n_experts=pool.n_experts(),
                per_step=(extra or {}).get("per_step") or [],
                text=tok.decode([int(v) for v in ids[0].tolist() if v < 256]),
                nxt=tok.decode([int(logits[0, -1].argmax())]))


def colours(ids):
    """A stable colour per expert id, distinguishable at a glance."""
    base = (list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    return {e: base[i % len(base)] for i, e in enumerate(sorted(ids))}


def draw(caps, out, title_extra=""):
    every = sorted({e for c in caps for row in c["chosen"] for e in row})
    col = colours(every)
    k = caps[0]["top_k"]
    deepest = max(len(c["chosen"]) for c in caps)
    n = len(caps)
    fig = plt.figure(figsize=(4.6 * n + 2.4, 1.05 + 0.42 * deepest),
                     facecolor=BG)
    fig.text(0.5, 0.965, "which experts ran, for one character",
             ha="center", color=FG, fontsize=19, weight="bold")
    c0 = caps[0]
    fig.text(0.5, 0.925,
             f"step {c0['step']:,}  ·  {len(every)} distinct experts across the "
             f"figure, from {c0['resident']} on the card of {c0['n_experts']}  "
             f"·  same colour = same expert{title_extra}",
             ha="center", color=DIM, fontsize=11)

    names = ["recur 0", "recur 1", "recur 2", "coda"]
    for ci, cap in enumerate(caps):
        nrows = len(cap["chosen"])
        left = 0.10 + ci * (0.88 / n)
        width = 0.88 / n - 0.10
        ax = fig.add_axes([left, 0.07, width, 0.80], facecolor=BG)
        ax.set_title(f"{cap['depth']} latent pass"
                     f"{'es' if cap['depth'] > 1 else ''}"
                     f"   —   {nrows} rows, {nrows * k} expert slots run",
                     color=FG, fontsize=13, pad=12)
        for r, row in enumerate(cap["chosen"]):
            y = deepest - 1 - r
            for c, e in enumerate(row[:k]):
                ax.add_patch(plt.Rectangle((c, y), .92, .88,
                                           color=col[e], ec=BG, lw=1.2))
                ax.text(c + .46, y + .44, str(e), ha="center", va="center",
                        fontsize=7.5, color="#0d1117", weight="bold")
            lab = f"pass {r // 4 + 1}  {names[r % 4]}"
            ax.text(-0.25, y + .44, lab, ha="right", va="center",
                    color=DIM if r % 4 != 3 else "#58a6ff", fontsize=9)
        ax.set_xlim(-0.3, k); ax.set_ylim(0, deepest)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        reuse = {}
        for row in cap["chosen"]:
            for e in row:
                reuse[e] = reuse.get(e, 0) + 1
        multi = sum(1 for v in reuse.values() if v > 1)
        fig.text(left + width / 2, 0.028,
                 f"{len(reuse)} distinct experts, {multi} of them used more "
                 f"than once   ·   predicts {cap['nxt']!r}",
                 ha="center", color=DIM, fontsize=10)
    fig.savefig(out, facecolor=BG, dpi=115)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--depths", default="1,4")
    ap.add_argument("--chars", type=int, default=48)
    ap.add_argument("--subject", default="stories")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/expert_grid.png")
    a = ap.parse_args()
    dev = torch.device(a.device)
    caps = [capture(a.weights, int(d), a.chars, a.subject, dev)
            for d in a.depths.split(",")]
    for c in caps:
        flat = [e for row in c["chosen"] for e in row]
        print(f"  depth {c['depth']}: {len(c['chosen'])} rows, {len(flat)} "
              f"slots run, {len(set(flat))} distinct experts", flush=True)
    draw(caps, a.out)


if __name__ == "__main__":
    main()
