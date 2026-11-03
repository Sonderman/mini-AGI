#!/usr/bin/env python3
"""
How one character flows through the latent passes, with the real experts.

Every number is captured from the trained model reading real text. The point
of the picture is two things the loop is easy to draw wrongly:

  x IS RE-INJECTED EVERY PASS. The prelude runs ONCE and produces x. Then
  h starts at zero and each pass computes adapter(cat([h, x])) - so every
  pass sees the original encoding again, not only the running state. It is
  not a plain loop over h.

  THE CODA IS A BRANCH, NOT A LINK. Each pass copies h into y, runs the coda
  on y, and reads logits and a halting probability off that. h continues to
  the next pass untouched by the coda. So the readout hangs off the side of
  the chain rather than sitting in it.

    python3 tools/latent_flow.py --depth 4 --out runs/latent_flow.png
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
BLUE, GREEN, AMBER, RED = "#58a6ff", "#3fb950", "#d29922", "#f85149"


def capture(weights, depth, chars, subject, device):
    set_compute_dtype("fp32" if device.type == "cpu" else "bf16")
    # read_only: the training run may own this directory, and paging an
    # expert in marks it dirty whether or not anything changed it
    model, cfg, pool, man = T.build_paged(weights, device, read_only=True)
    T.weights_store.load(model, weights, device=device)
    model.eval()
    cfg.max_steps = depth                 # eval always runs the ceiling
    cfg.min_steps = min(cfg.min_steps, depth)

    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    names = ["recur 0", "recur 1", "recur 2", "coda"][:len(sites)]
    for s in sites:
        s.record_weights = True

    # every call to a pooled site, in order: 4 sites x depth passes
    calls = []
    hooks = [s.register_forward_hook(
        lambda m, i, o, s=s: calls.append(
            None if m.last_weight is None else m.last_weight.detach()
            .float().cpu().numpy().copy()))
        for s in sites]

    files = [f for f in collect(["data/train"]) if f"/{subject}/" in f]
    data = as_stream(sorted(files)[0])
    ids = torch.tensor(np.asarray(data[:chars], np.int64)[None], device=device)

    with torch.no_grad(), amp(device):
        logits, extra = model(ids, collect=True)
    for h in hooks:
        h.remove()

    per_step = (extra or {}).get("per_step") or []
    tok = __import__("minagi.tokenizer", fromlist=["ByteTokenizer"]).ByteTokenizer()
    pred = [int(logits[0, -1].argmax())]
    return dict(calls=calls, names=names, per_step=per_step,
                slots=list(pool.slots), gate=pool.gate.detach().abs().cpu().numpy(),
                depth=depth, resident=pool.resident, top_k=sites[0].top_k,
                text=tok.decode([int(v) for v in ids[0].tolist() if v < 256]),
                nxt=tok.decode(pred), step=man["step"],
                n_experts=pool.n_experts())


def draw(caps, out):
    n = len(caps)
    fig = plt.figure(figsize=(9 + 5.2 * n, 9.5), facecolor=BG)
    fig.text(0.5, 0.965, "how one character flows through the latent passes",
             ha="center", color=FG, fontsize=22, weight="bold")
    c0 = caps[0]
    fig.text(0.5, 0.925,
             f"captured from the model at step {c0['step']:,}  ·  "
             f"{c0['n_experts']} experts, {c0['resident']} on the card, "
             f"top-{c0['top_k']} per site  ·  bar height is the gate-weighted "
             f"contribution", ha="center", color=DIM, fontsize=12)

    # one pass is the same height in every panel, so a shallow run does not
    # stretch to fill the frame and look like it is doing more work
    deepest = max(c["depth"] for c in caps)
    scale = max(float(np.percentile(np.concatenate(
        [c.ravel() for cap in caps for c in cap["calls"] if c is not None]), 97)),
        1e-6)

    for ci, cap in enumerate(caps):
        d = cap["depth"]
        left = 0.05 + ci * (0.92 / n)
        width = 0.92 / n - 0.05
        fig.text(left + width / 2, 0.885, f"{d} latent pass{'es' if d > 1 else ''}",
                 ha="center", color=AMBER, fontsize=17, weight="bold")

        # the prelude, once
        ax = fig.add_axes([left, 0.79, width, 0.055], facecolor="#161b22")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(DIM)
        ax.text(0.5, 0.5, "prelude — 2 blocks, DENSE, no experts.  runs ONCE  →  x",
                ha="center", va="center", color=BLUE, fontsize=11,
                transform=ax.transAxes)

        rows = len(cap["names"])
        top, bot = 0.75, 0.10
        ph = (top - bot) / deepest
        for p in range(d):
            y0 = top - (p + 1) * ph + 0.012
            box = fig.add_axes([left, y0, width, ph - 0.028], facecolor="#0f1620")
            box.set_xticks([]); box.set_yticks([])
            for s in box.spines.values():
                s.set_color("#30363d")
            # x is re-injected here, every pass
            box.text(0.012, 0.94, f"pass {p+1}:   h ← adapter( h , x )",
                     transform=box.transAxes, color=AMBER, fontsize=10.5,
                     va="top")
            for r in range(rows):
                v = cap["calls"][p * rows + r]
                a = fig.add_axes([left + 0.012,
                                  y0 + (rows - 1 - r) * (ph - 0.062) / rows + 0.012,
                                  width - 0.024, (ph - 0.062) / rows - 0.006],
                                 facecolor="#0f1620")
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_visible(False)
                if v is not None:
                    k = min(len(v), cap["resident"])
                    col = [GREEN if x > 0 else "#1f2937" for x in v[:k]]
                    if cap["names"][r] == "coda":
                        col = [BLUE if x > 0 else "#1f2937" for x in v[:k]]
                    a.bar(range(k), np.minimum(v[:k] / scale, 1.0), width=.82,
                          color=col)
                    a.set_ylim(0, 1.05); a.set_xlim(-1, cap["resident"])
                lab = cap["names"][r]
                a.text(-0.005, 0.5, lab, transform=a.transAxes, ha="right",
                       va="center", color=DIM if lab != "coda" else BLUE,
                       fontsize=9)
                if lab == "coda":
                    a.text(1.005, 0.5, "→ logits + halt", transform=a.transAxes,
                           ha="left", va="center", color=BLUE, fontsize=9)
            ps = cap["per_step"][p] if p < len(cap["per_step"]) else None
            if ps:
                box.text(0.988, 0.955,
                         f"halt {ps['halt_p']:.3f}   cumulative {ps['cum']:.3f}",
                         transform=box.transAxes, color=FG, fontsize=10.5,
                         ha="right", va="top")
            if p == d - 1 and d < deepest:
                fig.text(left + width / 2, y0 - 0.028,
                         f"halting had already spent "
                         f"{cap['per_step'][p]['cum']:.1%} of its mass",
                         ha="center", color=DIM, fontsize=10)
        fig.text(left + width / 2, 0.055,
                 f"predicts {cap['nxt']!r}", ha="center", color=FG, fontsize=13)

    fig.text(0.5, 0.017,
             "green = a pooled expert firing inside the chain   ·   "
             "blue = the coda, which BRANCHES off to read out and does not "
             "feed the next pass   ·   x is re-injected every pass",
             ha="center", color=DIM, fontsize=11)
    fig.savefig(out, facecolor=BG, dpi=110)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--depths", default="1,4")
    ap.add_argument("--chars", type=int, default=48)
    ap.add_argument("--subject", default="stories")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/latent_flow.png")
    a = ap.parse_args()
    dev = torch.device(a.device)
    caps = []
    for d in [int(x) for x in a.depths.split(",")]:
        caps.append(capture(a.weights, d, a.chars, a.subject, dev))
        c = caps[-1]
        lit = sum(int((v > 0).sum()) for v in c["calls"] if v is not None)
        print(f"  depth {d}: {len(c['calls'])} pooled calls, {lit} expert "
              f"activations, predicts {c['nxt']!r}", flush=True)
    draw(caps, a.out)


if __name__ == "__main__":
    main()
