#!/usr/bin/env python3
"""
The trunk: what every character passes through, and what it costs.

Two figures, both read from a real weights directory rather than written by
hand:

  trunk_forward.png    every parameter outside the pool, where it sits, what
                       it does, and how information moves through it
  trunk_backward.png   how gradient comes back - what is weighted by halting,
                       what is truncated, and what the pool actually receives

The distinction the first figure is built around is that the trunk runs for
EVERY character while the pool only lends 8 experts per row. The trunk is 1.5%
of the parameters and 100% of the traffic, which is also why forgetting lives
there - see runs/results/where_forgetting.json.

    python3 tools/trunk_anatomy.py --weights weights
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG, FG, DIM, GRID = "#0d1117", "#e6edf3", "#8b949e", "#30363d"
BLUE, GREEN, AMBER = "#58a6ff", "#3fb950", "#d29922"
PURPLE, RED, TEAL = "#bc8cff", "#f85149", "#39c5cf"


def box(ax, x, y, w, h, label, sub, colour, fs=11, face="#161b22", lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                facecolor=face, edgecolor=colour, lw=lw))
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), label, ha="center",
            va="center", color=colour, fontsize=fs, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                color=DIM, fontsize=fs - 3.0, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, colour=DIM, lw=1.7, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, color=colour,
        lw=lw, shrinkA=3, shrinkB=3,
        connectionstyle=f"arc3,rad={rad}"))


def elbow(ax, pts, colour=DIM, lw=1.7):
    """Orthogonal polyline with an arrowhead on the last leg."""
    for i in range(len(pts) - 2):
        ax.add_patch(FancyArrowPatch(pts[i], pts[i + 1], arrowstyle="-",
                                     color=colour, lw=lw, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                 mutation_scale=14, color=colour, lw=lw,
                                 shrinkA=0, shrinkB=3))


def inventory(wdir):
    """Every trunk tensor, grouped, with tied weights counted once."""
    core = np.load(os.path.join(wdir, "core.npz"))
    rout = np.load(os.path.join(wdir, "routers.npz"))
    man = json.load(open(os.path.join(wdir, "manifest.json")))
    cfg = man["cfg"]
    tied = bool(cfg.get("tie_embeddings"))
    g = {}
    for k in core.files:
        n = int(core[k].size)
        if k == "head.weight" and tied:
            continue                     # the same tensor as tok_emb
        if k.startswith("prelude."):
            g.setdefault(f"prelude.{k.split('.')[1]}", 0)
            g[f"prelude.{k.split('.')[1]}"] += n
        elif k.startswith("recur."):
            g["recur.0"] = g.get("recur.0", 0) + n
        else:
            g[k.split(".")[0]] = g.get(k.split(".")[0], 0) + n
    r = {k: int(rout[k].size) for k in rout.files}
    per_expert = 2 * cfg["pool_d_ff"] * cfg["d_model"] + \
        cfg["d_model"] * cfg["pool_d_ff"]
    return g, r, cfg, man, tied, per_expert


def measure_halting(wdir, chars, subject, device):
    """Real halting mass per row, from one forward over real text."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    import train as T
    from minagi.ingest import as_stream, collect
    from minagi.precision import amp, set_compute_dtype
    set_compute_dtype("fp32" if device == "cpu" else "bf16")
    dev = torch.device(device)
    model, cfg, pool, man = T.build_paged(wdir, dev, read_only=True)
    T.weights_store.load(model, wdir, device=dev)
    model.eval()
    files = [f for f in collect(["data/train"]) if f"/{subject}/" in f]
    data = as_stream(sorted(files)[0])
    ids = torch.tensor(np.asarray(data[:chars], np.int64)[None], device=dev)
    with torch.no_grad(), amp(dev):
        _, extra = model(ids, collect=True)
    return [p["halt_p"] for p in (extra or {}).get("per_step") or []]


# --------------------------------------------------------------------------
def forward_figure(g, r, cfg, man, tied, per_expert, out):
    M = 1e6
    prel = g["prelude.0"] + g["prelude.1"]
    rec = g["recur.0"]
    trunk = sum(g.values())
    routers = sum(r.values())
    n_exp = int(man["n_experts"])
    live = cfg["pool_top_k"] * per_expert

    fig = plt.figure(figsize=(19.5, 11.6), facecolor=BG)
    fig.text(0.5, 0.963, "the trunk — everything a character passes through",
             ha="center", color=FG, fontsize=25, weight="bold")
    fig.text(0.5, 0.929,
             f"{trunk/M:.2f}M parameters, run for every character  ·  against "
             f"{n_exp*per_expert/M:.0f}M in the pool of which {live/M:.1f}M run "
             f"per row  ·  the trunk is "
             f"{100*trunk/(trunk+routers+n_exp*per_expert):.1f}% of the model "
             f"and 100% of the traffic",
             ha="center", color=DIM, fontsize=12.5)

    ax = fig.add_axes([0.025, 0.055, 0.70, 0.855], facecolor=BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- band 1: once per character ---------------------------------------
    ax.add_patch(FancyBboxPatch((1.2, 74.5), 97, 22,
                                boxstyle="round,pad=0.4", facecolor="#10161f",
                                edgecolor=GRID, lw=1.2))
    ax.text(2.6, 94.2, "ONCE PER CHARACTER", color=BLUE, fontsize=10,
            weight="bold")

    box(ax, 3, 79, 12, 12, "bytes", "one of 256\n+ 9 markers", FG, fs=11)
    box(ax, 18.5, 79, 15, 12, "tok_emb",
        f"265 x {cfg['d_model']}\n{g['tok_emb']/M:.3f}M"
        + ("\nTIED to head" if tied else ""), TEAL)
    arrow(ax, 15, 85, 18.5, 85)

    for i, (x0, nm) in enumerate([(37.5, "prelude.0"), (60.5, "prelude.1")]):
        box(ax, x0, 79, 20, 12, nm,
            f"attn {cfg['n_head']} heads + SwiGLU {cfg['d_ff']}\n"
            f"{g[nm]/M:.3f}M", BLUE)
        arrow(ax, x0 - 4, 85, x0, 85)
    arrow(ax, 80.5, 85, 85.5, 85)
    box(ax, 85.5, 79, 12, 12, "x", "the reading.\nfixed from\nhere on", FG,
        fs=12)
    ax.text(50, 76.2, "two ordinary transformer blocks. this is the only place "
            "attention over the context happens before the loop",
            color=DIM, fontsize=9.5, ha="center")

    # ---- band 2: the loop, drawn RIGHT to LEFT so the flow serpentines ----
    ax.add_patch(FancyBboxPatch((1.2, 40), 97, 31,
                                boxstyle="round,pad=0.4", facecolor="#141a10",
                                edgecolor=GREEN, lw=1.6, linestyle="--"))
    ax.text(2.6, 68.2, f"UP TO {cfg['max_steps']} TIMES — one row per pass, "
            f"the model decides when to stop", color=GREEN, fontsize=10,
            weight="bold")

    box(ax, 75, 51, 20, 12, "adapter",
        f"[h | x] -> h\n{2*cfg['d_model']} -> {cfg['d_model']}\n"
        f"{g['adapter']/M:.3f}M", PURPLE)
    box(ax, 49, 51, 20, 12, "recur.0  attn",
        f"self-attention + norms\n{rec/M:.3f}M", BLUE)
    box(ax, 17, 51, 26, 12, "recur.0  mlp  ->  THE POOL",
        f"router picks {cfg['pool_top_k']} of {cfg['pool_resident']} resident, "
        f"of {n_exp}\n{live/M:.1f}M runs here — NOT trunk", AMBER)
    box(ax, 2, 51, 11, 12, "h", "the latent.\nnever\ndecoded", GREEN)

    arrow(ax, 75, 57, 69, 57)
    arrow(ax, 49, 57, 43, 57)
    arrow(ax, 17, 57, 13, 57)

    # x comes down from band 1 into the adapter's top
    elbow(ax, [(91.5, 79), (91.5, 67), (85, 67), (85, 63)], FG)
    ax.text(88, 69.0, "x re-injected every row —\nthe loop cannot drift from "
            "the text", color=FG, fontsize=9, ha="right", va="top",
            linespacing=1.4)

    # h returns to the adapter's bottom, in a clear channel below the boxes
    elbow(ax, [(10.5, 51), (10.5, 45), (85, 45), (85, 51)], GREEN, lw=2.0)
    ax.text(47, 42.4, "h feeds back into the next row — this is the "
            "recurrence. the SAME block, applied again", color=GREEN,
            fontsize=10.5, ha="center")

    # ---- band 3: the readout ----------------------------------------------
    ax.add_patch(FancyBboxPatch((1.2, 4), 97, 32,
                                boxstyle="round,pad=0.4", facecolor="#1a1214",
                                edgecolor=RED, lw=1.2))
    ax.text(24, 33.0, "AFTER EVERY SINGLE ROW — which is why halting can "
            "speak at any depth", color=RED, fontsize=10, weight="bold")

    box(ax, 6, 13, 13, 12, "ln_f", f"RMSNorm\n{g['ln_f']/M*1e3:.2f}K", TEAL)
    box(ax, 25, 13, 17, 12, "head",
        f"{cfg['d_model']} -> 265\n"
        + ("tied — 0 extra" if tied else f"{g.get('head',0)/M:.3f}M"), TEAL)
    box(ax, 48, 13, 17, 12, "logits_n", "what it would\nsay at THIS depth", FG)
    box(ax, 71, 13, 12, 12, "halt", f"{cfg['d_model']} -> 1\n513 params", RED)
    box(ax, 87, 13, 10, 12, "lam_n", "am I done?", RED)

    # the same h is also read out, every row
    elbow(ax, [(4.5, 51), (4.5, 30), (12.5, 30), (12.5, 25)], GREEN)
    ax.text(5.6, 33.4, "the same h is read out\nafter every row",
            color=GREEN, fontsize=8.5, va="top", linespacing=1.4)
    arrow(ax, 19, 19, 25, 19)
    arrow(ax, 42, 19, 48, 19)
    arrow(ax, 65, 19, 71, 19, RED)
    arrow(ax, 83, 19, 87, 19, RED)
    ax.text(50, 8.0,
            "513 parameters decide the depth of a 553M-parameter model",
            color=RED, fontsize=10.5, ha="center", style="italic")

    # ---- right: the budget -------------------------------------------------
    p = fig.add_axes([0.755, 0.545, 0.225, 0.335], facecolor=BG)
    items = [("prelude.0", g["prelude.0"], BLUE), ("prelude.1", g["prelude.1"], BLUE),
             ("recur.0 attn", rec, TEAL), ("adapter", g["adapter"], PURPLE),
             ("tok_emb / head", g["tok_emb"], GREEN),
             ("ln_f + halt", g["ln_f"] + g["halt"], RED)]
    ypos = np.arange(len(items))[::-1]
    p.barh(ypos, [i[1] / M for i in items], color=[i[2] for i in items],
           height=.68)
    for yv, (nm, n, _) in zip(ypos, items):
        p.text(n / M + 0.09, yv, f"{n/M:.3f}M  ({100*n/trunk:.0f}%)",
               va="center", color=DIM, fontsize=8.5)
    p.set_yticks(ypos); p.set_yticklabels([i[0] for i in items],
                                          color=DIM, fontsize=9)
    p.set_xlim(0, max(i[1] for i in items) / M * 1.55)
    p.set_title(f"where the {trunk/M:.2f}M sits", color=FG, fontsize=12)
    p.tick_params(colors=DIM, labelsize=8.5)
    for s in p.spines.values():
        s.set_color(GRID)
    p.grid(alpha=.18, axis="x")

    q = fig.add_axes([0.755, 0.075, 0.225, 0.335], facecolor=BG)
    tot = trunk + routers + n_exp * per_expert
    q.pie([trunk, routers, n_exp * per_expert],
          colors=[BLUE, PURPLE, AMBER], startangle=90,
          wedgeprops=dict(width=.42, edgecolor=BG, lw=1.6))
    q.text(0, 0.06, f"{tot/M:.0f}M", ha="center", color=FG, fontsize=19,
           weight="bold")
    q.text(0, -0.20, "total", ha="center", color=DIM, fontsize=10)
    q.set_title("trunk vs routers vs pool", color=FG, fontsize=12, pad=2)
    fig.text(0.868, 0.052,
             f"trunk {trunk/M:.1f}M   ·   routers {routers/M:.2f}M   ·   "
             f"pool {n_exp*per_expert/M:.0f}M",
             ha="center", color=DIM, fontsize=9)
    fig.savefig(out, facecolor=BG, dpi=104)
    plt.close(fig)
    print(f"  wrote {out}")
    return trunk, routers


# --------------------------------------------------------------------------
def backward_figure(g, r, cfg, man, per_expert, halt_p, out):
    M = 1e6
    W = int(cfg["bptt_window"])
    S = int(cfg["max_steps"])
    mean = float(cfg.get("train_steps_mean") or 0)
    n_exp = int(man["n_experts"])

    # the real distribution of n_steps: Poisson(mean)+1, clamped
    ks = np.arange(0, 60)
    pmf = np.exp(-mean + ks * math.log(mean) - np.array(
        [math.lgamma(k + 1) for k in ks])) if mean > 0 else None
    if pmf is not None:
        steps = np.clip(ks + 1, cfg["min_steps"], S)
        dist = np.zeros(S + 1)
        for s, w in zip(steps, pmf):
            dist[s] += w
        dist = dist / dist.sum()
    else:
        dist = np.zeros(S + 1); dist[S] = 1.0
    p_any = float(dist[W + 1:].sum()) if W < S else 0.0
    exp_det = float(sum(dist[s] * max(0, s - W) for s in range(S + 1)))

    fig = plt.figure(figsize=(19.5, 11.2), facecolor=BG)
    fig.text(0.5, 0.963, "how the gradient comes back", ha="center", color=FG,
             fontsize=25, weight="bold")
    fig.text(0.5, 0.929,
             "three different rules govern three different parts: the loop is "
             "weighted by halting, the depth is truncated, and the pool is "
             "sparse",
             ha="center", color=DIM, fontsize=12.5)

    # ---- A: the unrolled loop ---------------------------------------------
    ax = fig.add_axes([0.035, 0.105, 0.30, 0.785], facecolor=BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, S + 2.2); ax.axis("off")
    ax.text(5, S + 1.4, f"one character, unrolled  (n_steps = {S})",
            ha="center", color=FG, fontsize=13, weight="bold")
    shown = S
    # scale the bars to the space there is; they ran off the axis before
    hscale = (2.75 / max(halt_p)) if halt_p else 0.0
    for n in range(shown):
        y = shown - 1 - n
        det = n < shown - W
        c = "#21262d" if det else GREEN
        ax.add_patch(plt.Rectangle((1.6, y + .08), 5.2, .82,
                                   facecolor=c if not det else "#161b22",
                                   edgecolor=c, lw=1.4))
        ax.text(4.2, y + .5, f"row {n+1}", ha="center", va="center",
                color=DIM if det else "#0d1117", fontsize=8.5,
                weight="bold" if not det else "normal")
        ax.text(1.35, y + .5, f"{n+1}", ha="right", va="center", color=DIM,
                fontsize=7.5)
        if halt_p and n < len(halt_p):
            ax.add_patch(plt.Rectangle((7.0, y + .18), halt_p[n] * hscale,
                                       .62, facecolor=AMBER,
                                       edgecolor="none"))
    ax.text(7.0, shown + .35, "halting weight p_n", color=AMBER, fontsize=8.5)
    ax.plot([1.3, 9.9], [W - .06, W - .06], color=RED, lw=1.8, ls="--")
    ax.text(7.15, W + .30, f"detach line\nbptt_window = {W}", color=RED,
            fontsize=8.5, va="bottom", linespacing=1.3)
    cap = ("green rows carry gradient · grey are detached\n"
           "the loss is  sum_n  p_n * CE_n  — every row is trained, "
           "weighted by how\nmuch halting mass it holds")
    if halt_p:
        cap += (f"\n\nmeasured: row 1 alone holds {halt_p[0]:.0%} of the "
                f"halting mass on this text")
    ax.text(0.2, -0.6, cap, color=DIM, fontsize=9, va="top",
            linespacing=1.5)

    # ---- B: how often truncation actually fires ---------------------------
    p = fig.add_axes([0.395, 0.565, 0.265, 0.325], facecolor=BG)
    p.bar(range(S + 1), dist, color=BLUE, width=.85)
    p.axvline(W + .5, color=RED, ls="--", lw=1.6)
    p.text(W + .9, dist.max() * 1.0, f"truncation only\nbites above {W}",
           color=RED, fontsize=8.5, va="top")
    p.set_title(f"depth sampled per step: Poisson({mean:g}) + 1",
                color=FG, fontsize=12)
    p.set_xlabel("n_steps this training step", color=DIM, fontsize=9)
    p.tick_params(colors=DIM, labelsize=8.5)
    for s in p.spines.values():
        s.set_color(GRID)
    p.grid(alpha=.18, axis="y")
    p.text(0.02, 0.97, f"P(any detach) = {p_any:.0%}\n"
           f"mean rows detached\n  = {exp_det:.2f} of {mean+1:.1f}",
           transform=p.transAxes, color=AMBER, fontsize=9.5, va="top",
           linespacing=1.4)

    # ---- C: what the pool receives ----------------------------------------
    q = fig.add_axes([0.395, 0.105, 0.265, 0.415], facecolor=BG)
    q.axis("off")
    q.set_xlim(0, 10); q.set_ylim(0, 10)
    q.text(5, 9.6, "what the pool actually receives", ha="center", color=FG,
           fontsize=12.5, weight="bold")
    cols, rows_ = 20, 9
    for i in range(n_exp):
        cx, cy = i % cols, i // cols
        resident = i < cfg["pool_resident"]
        q.add_patch(plt.Rectangle((0.25 + cx * 0.48, 8.0 - cy * 0.48),
                                  0.40, 0.40,
                                  facecolor="#161b22" if not resident else BLUE,
                                  edgecolor=GRID, lw=.4))
    for i in range(cfg["pool_top_k"]):
        q.add_patch(plt.Rectangle((0.25 + i * 0.48, 8.0), 0.40, 0.40,
                                  facecolor=AMBER, edgecolor=BG, lw=.6))
    q.text(0.25, 3.15,
           f"all {n_exp} experts exist on disk\n"
           f"{cfg['pool_resident']} are on the card (blue) — only these CAN "
           f"receive gradient\n"
           f"{cfg['pool_top_k']} run per row (amber) — only these DO\n\n"
           f"The forward admits its experts at its first pass and keeps them\n"
           f"through the backward, so the experts that computed it are the\n"
           f"ones it updates - never whoever took a slot later.",
           color=DIM, fontsize=9.4, va="top", linespacing=1.6)

    # ---- D: the three rules ------------------------------------------------
    t = fig.add_axes([0.685, 0.075, 0.295, 0.815], facecolor=BG)
    t.axis("off"); t.set_xlim(0, 10); t.set_ylim(0, 10)
    t.text(0, 9.75, "the three rules", color=FG, fontsize=14, weight="bold")

    blocks = [
        (AMBER, "1 · halting weights the loss",
         "loss = sum_n p_n * CE_n  +  beta * KL(p || geometric)\n\n"
         "Training never halts. It runs every row and mixes what each row\n"
         "would have said, weighted by that row's halting mass. So there is\n"
         "no discrete stop decision anywhere for gradients to get stuck on —\n"
         "halting learns because being right EARLY is directly rewarded.\n\n"
         f"The KL term (beta = {cfg['ponder_beta']}) pulls toward a geometric\n"
         f"prior of {cfg['halt_prior']}, which is what stops it simply always\n"
         "running to the ceiling to be safe."),
        (RED, "2 · depth is truncated, in principle",
         f"h = h.detach()   when   n < n_steps - {W}\n\n"
         "This is what keeps memory from growing with depth. But look at\n"
         f"the numbers on the left: depth is sampled as Poisson({mean:g})+1 and\n"
         f"the window is {W}, so truncation fires on only {p_any:.0%} of steps and\n"
         f"detaches {exp_det:.2f} rows on average. In practice nearly every\n"
         "training step backpropagates through the WHOLE loop.\n\n"
         "The mechanism is real; it is just barely engaged at these settings."),
        (BLUE, "3 · the pool is sparse, the trunk is not",
         f"Every character passes through all {sum(g.values())/M:.2f}M trunk\n"
         f"parameters, so the trunk gets gradient from every character of\n"
         "every domain — and that is where the interference lives.\n\n"
         "Freezing experiments, runs/results/where_forgetting.json:\n\n"
         "    nothing frozen        21% of old knowledge kept\n"
         "    experts frozen        36%\n"
         "    the trunk frozen    99.7%   (and it learns almost nothing)\n\n"
         "Forgetting is a trunk problem, not a pool problem."),
    ]
    y = 9.0
    for colour, title, body in blocks:
        t.add_patch(FancyBboxPatch((0, y - 2.72), 10, 2.62,
                                   boxstyle="round,pad=0.06",
                                   facecolor="#161b22", edgecolor=colour,
                                   lw=1.6))
        t.text(0.3, y - 0.36, title, color=colour, fontsize=11.5,
               weight="bold")
        t.text(0.3, y - 0.72, body, color=DIM, fontsize=8.5, va="top",
               linespacing=1.5, family="monospace")
        y -= 3.0
    fig.savefig(out, facecolor=BG, dpi=104)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--forward-out", default="runs/trunk_forward.png")
    ap.add_argument("--backward-out", default="runs/trunk_backward.png")
    ap.add_argument("--measure", action="store_true",
                    help="run one forward to get real halting mass per row")
    ap.add_argument("--chars", type=int, default=96)
    ap.add_argument("--subject", default="stories")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    g, r, cfg, man, tied, per = inventory(a.weights)
    print(f"  {a.weights} at step {man['step']:,}, {man['n_experts']} experts")
    trunk, routers = forward_figure(g, r, cfg, man, tied, per, a.forward_out)
    halt_p = None
    if a.measure:
        print("  measuring halting mass...", flush=True)
        halt_p = measure_halting(a.weights, a.chars, a.subject, a.device)
        print(f"  {len(halt_p)} rows, mass on row 1 = {halt_p[0]:.3f}")
    backward_figure(g, r, cfg, man, per, halt_p, a.backward_out)
    print(f"  trunk {trunk/1e6:.3f}M  routers {routers/1e6:.3f}M  "
          f"pool {man['n_experts']*per/1e6:.1f}M"
          + ("   (head tied to tok_emb, counted once)" if tied else ""))


if __name__ == "__main__":
    main()
