#!/usr/bin/env python3
"""
The depth mechanism as it is, against the one-row-at-a-time proposal.

What is MEASURED here comes from the trained model reading real text: the
experts each row ran, and the halting probability it produced. What is
STRUCTURAL - the proposed side - is drawn from the config that would produce
it, not from a trained model, and is labelled as such. No routing is invented.

THE DIFFERENCE IS GRANULARITY, not the ceiling. Both designs top out at 24
block applications choosing 8 of the 32 experts on the card. Today those 24
arrive in bundles of four - three recur blocks and a coda - and halting can
only speak between bundles, so the model's choice of effort is one of
{4, 8, 12, 16, 20, 24}. With n_recur=1 and n_coda=0 a bundle is one row, and
the choice becomes any of {1 .. 24}.

    python3 tools/depth_proposal.py --out runs/depth_proposal.png
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
AMBER, BLUE, GREEN, RED = "#d29922", "#58a6ff", "#3fb950", "#f85149"


def capture(weights, chars, subject, device):
    set_compute_dtype("fp32" if device.type == "cpu" else "bf16")
    # read_only: the training run may own this directory, and paging an
    # expert in marks it dirty whether or not anything changed it
    model, cfg, pool, man = T.build_paged(weights, device, read_only=True)
    T.weights_store.load(model, weights, device=device)
    model.eval()
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
    chosen = [sorted(slots[j] for j in np.nonzero(r)[0] if j < len(slots))
              for r in rows]
    return dict(chosen=chosen, per_step=(extra or {}).get("per_step") or [],
                step=man["step"], resident=pool.resident, top_k=sites[0].top_k,
                n_experts=pool.n_experts(), max_steps=cfg.max_steps,
                per_pass=cfg.n_recur + cfg.n_coda)


def draw(cap, out):
    R = cap["max_steps"] * cap["per_pass"]          # 24 rows either way
    per = cap["per_pass"]
    k = cap["top_k"]
    ids = sorted({e for row in cap["chosen"] for e in row})
    base = (list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    col = {e: base[i % len(base)] for i, e in enumerate(ids)}

    fig = plt.figure(figsize=(15.5, 11.2), facecolor=BG)
    fig.text(0.5, 0.968, "how deep to go, and who gets to decide",
             ha="center", color=FG, fontsize=21, weight="bold")
    fig.text(0.5, 0.936,
             f"one character, model at step {cap['step']:,}  ·  both designs "
             f"top out at {R} rows of {k} experts from the {cap['resident']} on "
             f"the card  ·  only the granularity differs",
             ha="center", color=DIM, fontsize=11.5)

    halt = [p["halt_p"] for p in cap["per_step"]]
    cum = [p["cum"] for p in cap["per_step"]]

    # ---------------- left: as it is ---------------------------------------
    ax = fig.add_axes([0.055, 0.085, 0.40, 0.80], facecolor=BG)
    ax.set_title("now — bundles of four, halting between bundles",
                 color=AMBER, fontsize=14, pad=14)
    names = ["recur 0", "recur 1", "recur 2", "coda"]
    drawn = min(len(cap["chosen"]), R)
    for r in range(drawn):
        y = R - 1 - r
        for c, e in enumerate(cap["chosen"][r][:k]):
            ax.add_patch(plt.Rectangle((c, y), .9, .86, color=col[e], ec=BG,
                                       lw=.9))
        ax.text(-0.3, y + .43, names[r % per], ha="right", va="center",
                color=DIM if r % per != per - 1 else BLUE, fontsize=8)
        if r % per == per - 1:
            p = r // per
            ax.plot([-0.1, k + 2.6], [y - .07, y - .07], color="#30363d", lw=1)
            if p < len(halt):
                ax.text(k + 0.35, y + .43,
                        f"halt {halt[p]:.3f}   cum {cum[p]:.3f}",
                        ha="left", va="center", color=FG, fontsize=9)
    ax.set_xlim(-3.4, k + 5.2); ax.set_ylim(0, R)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-3.2, R - 2, "the model may stop\nonly at these lines",
            color=DIM, fontsize=9, va="top")
    for p in range(1, cap["max_steps"] + 1):
        ax.text(-3.2, R - p * per + .3, f"{p*per}", color=AMBER, fontsize=9,
                ha="left", va="center")

    # ---------------- right: the proposal -----------------------------------
    ax2 = fig.add_axes([0.55, 0.085, 0.40, 0.80], facecolor=BG)
    ax2.set_title("proposed — one row at a time, halting after every row",
                  color=GREEN, fontsize=14, pad=14)
    for r in range(R):
        y = R - 1 - r
        for c in range(k):
            ax2.add_patch(plt.Rectangle((c, y), .9, .86, color="#1f2937",
                                        ec=BG, lw=.9))
        ax2.plot([-0.1, k + 2.6], [y - .07, y - .07], color="#30363d", lw=.7)
        ax2.text(-0.3, y + .43, f"row {r+1}", ha="right", va="center",
                 color=DIM, fontsize=8)
        ax2.text(-3.2, y + .43, f"{r+1}", color=GREEN, fontsize=9,
                 ha="left", va="center")
    ax2.set_xlim(-3.4, k + 5.2); ax2.set_ylim(0, R)
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.text(k + 0.35, R - 1.5,
             "each row still picks its\nown 8 of the 32 resident.\n\n"
             "cells are left empty here:\nthis is the SHAPE, from the\n"
             "config that produces it —\nno routing is invented.",
             color=DIM, fontsize=9.5, va="top")
    ax2.text(-3.2, R + 0.9, "may stop\nafter any row", color=GREEN,
             fontsize=9, va="top")

    # ---------------- what it would have chosen -----------------------------
    spent = next((i + 1 for i, c in enumerate(cum) if c >= 0.9), len(cum))
    fig.text(0.055, 0.045,
             f"MEASURED: halting spent {cum[0]:.1%} of its mass after the first "
             f"bundle — the model wanted to stop at row {per}, the shallowest "
             f"it can.\nIt then ran {R - per} more rows, {(R-per)*k} expert "
             f"evaluations, weighted at {1-cum[0]:.1%}.",
             color=FG, fontsize=11, va="top")
    fig.text(0.55, 0.045,
             "STRUCTURAL: n_recur=1, n_coda=0, max_steps=24. One block applied\n"
             "up to 24 times. Verified to run; not trained, so nothing here is\n"
             "a claim about which experts it would pick.",
             color=DIM, fontsize=11, va="top")
    fig.savefig(out, facecolor=BG, dpi=112)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--chars", type=int, default=48)
    ap.add_argument("--subject", default="stories")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/depth_proposal.png")
    a = ap.parse_args()
    cap = capture(a.weights, a.chars, a.subject, torch.device(a.device))
    print(f"  {len(cap['chosen'])} rows captured, halting over "
          f"{len(cap['per_step'])} bundles", flush=True)
    draw(cap, a.out)


if __name__ == "__main__":
    main()
