#!/usr/bin/env python3
"""
The trunk, drawn the way transformers are usually drawn.

Three figures, all read from a real weights directory:

  block_opened.png   one block with the residual stream down the middle and
                     both sub-layers opened up - attention to heads and RoPE,
                     the feed-forward to its three matrices
  trunk_stack.png    the whole model as a vertical tower, bottom to top, with
                     the tensor shape on every arrow
  trunk_bptt.png     the loop unrolled left to right, the standard way a
                     recurrent graph is drawn, with the backward pass on it

Shapes are written as [B, T, C] with the real numbers substituted.

    python3 tools/trunk_diagram.py --weights weights
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

BG, FG, DIM, GRID = "#0d1117", "#e6edf3", "#8b949e", "#30363d"
BLUE, GREEN, AMBER = "#58a6ff", "#3fb950", "#d29922"
PURPLE, RED, TEAL = "#bc8cff", "#f85149", "#39c5cf"
RES = "#e6edf3"          # the residual stream


def rbox(ax, x, y, w, h, lines, colour, fs=10.5, face="#161b22", lw=1.8):
    """A labelled box. lines = [(text, size, colour), ...]"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                facecolor=face, edgecolor=colour, lw=lw))
    n = len(lines)
    for i, (t, sz, c) in enumerate(lines):
        yy = y + h * (1 - (i + 0.5) / n)
        ax.text(x + w / 2, yy, t, ha="center", va="center", color=c,
                fontsize=sz)


def ar(ax, p1, p2, colour=RES, lw=2.0, style="-|>", ms=13, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                                 color=colour, lw=lw, shrinkA=0, shrinkB=0,
                                 linestyle=ls))


def poly(ax, pts, colour=RES, lw=2.0, head=True, ls="-"):
    for i in range(len(pts) - 2):
        ar(ax, pts[i], pts[i + 1], colour, lw, style="-", ls=ls)
    ar(ax, pts[-2], pts[-1], colour, lw,
       style="-|>" if head else "-", ls=ls)


def plus(ax, x, y, r=1.5, colour=RES):
    ax.add_patch(Circle((x, y), r, facecolor=BG, edgecolor=colour, lw=2.0,
                        zorder=6))
    ax.text(x, y, "+", ha="center", va="center", color=colour, fontsize=15,
            weight="bold", zorder=7)


def shape(ax, x, y, txt, colour=DIM, fs=9, ha="left"):
    ax.text(x, y, txt, ha=ha, va="center", color=colour, fontsize=fs,
            family="monospace")


def load(wdir):
    man = json.load(open(os.path.join(wdir, "manifest.json")))
    cfg = man["cfg"]
    core = np.load(os.path.join(wdir, "core.npz"))
    n = {k: int(core[k].size) for k in core.files}
    return man, cfg, n


# ===========================================================================
# 1. one block, opened up
# ===========================================================================
def fig_block(man, cfg, n, out):
    C = cfg["d_model"]; H = cfg["n_head"]; hd = C // H; FF = cfg["d_ff"]
    M = 1e6
    fig = plt.figure(figsize=(17.5, 12.4), facecolor=BG)
    fig.text(0.5, 0.963, "one block, opened up", ha="center", color=FG,
             fontsize=26, weight="bold")
    fig.text(0.5, 0.930,
             f"pre-norm residual: x = x + attn(norm(x)),  then  "
             f"x = x + mlp(norm(x))   ·   {C} channels, {H} heads of {hd}   ·  "
             f" 3.212M parameters",
             ha="center", color=DIM, fontsize=13)

    ax = fig.add_axes([0.02, 0.03, 0.96, 0.875], facecolor=BG)
    # the block runs below y=0; patches clip to the axes, so say so
    ax.set_xlim(0, 100); ax.set_ylim(-11, 100); ax.axis("off")

    SP = 13.0                       # the residual stream sits here

    # ---------------- the residual stream ---------------------------------
    ax.text(SP, 97.5, "the residual stream", ha="center", color=RES,
            fontsize=11, weight="bold")
    ax.text(SP, 94.6, "nothing replaces it —\neverything ADDS to it",
            ha="center", color=DIM, fontsize=9, linespacing=1.4)

    rbox(ax, SP - 8, 85, 16, 5.5,
         [("x   in", 12, FG)], FG, lw=2.2)
    shape(ax, SP + 9.5, 87.75, f"[1, T, {C}]", DIM)

    ar(ax, (SP, 85), (SP, 79.5))
    ax.add_patch(Circle((SP, 79), 0.9, facecolor=RES, edgecolor=RES, zorder=6))
    ax.text(SP - 1.6, 79, "split", ha="right", color=DIM, fontsize=8.5)

    # ---------------- sub-layer 1: attention ------------------------------
    ax.add_patch(FancyBboxPatch((24, 55.5), 74, 25.5,
                                boxstyle="round,pad=0.5", facecolor="#101820",
                                edgecolor=BLUE, lw=1.4, linestyle="--"))
    ax.text(25.5, 78.6, "SUB-LAYER 1  ·  self-attention", color=BLUE,
            fontsize=11.5, weight="bold")

    poly(ax, [(SP, 79), (SP, 68), (26, 68)], RES)      # into the sub-layer
    y0, bh = 61.5, 11.0
    xs = [26, 40.5, 55.5, 71, 85.5]
    ws = [12.5, 13, 13.5, 12.5, 12.5]

    rbox(ax, xs[0], y0, ws[0], bh,
         [("RMSNorm", 11, TEAL), ("ln1", 9, DIM),
          (f"{n['prelude.0.ln1.weight']} params", 8.5, DIM)], TEAL)
    rbox(ax, xs[1], y0, ws[1], bh,
         [("W_qkv", 11.5, BLUE), (f"{C} -> {3*C}", 9, DIM),
          ("0.786M", 8.5, AMBER)], BLUE)
    rbox(ax, xs[2], y0, ws[2], bh,
         [("split heads", 10.5, PURPLE), (f"-> [1, {H}, T, {hd}]", 8.5, DIM),
          ("+ RoPE on Q,K", 8.5, DIM), ("no parameters", 8, GREEN)], PURPLE)
    rbox(ax, xs[3], y0, ws[3], bh,
         [("softmax", 11, PURPLE), ("QKt / sqrt(64)", 8.5, DIM),
          ("· V   causal", 8.5, DIM), ("no parameters", 8, GREEN)], PURPLE)
    rbox(ax, xs[4], y0, ws[4], bh,
         [("W_o", 11.5, BLUE), (f"{C} -> {C}", 9, DIM),
          ("0.262M", 8.5, AMBER)], BLUE)

    for i in range(4):
        ar(ax, (xs[i] + ws[i], y0 + bh / 2), (xs[i + 1], y0 + bh / 2), DIM,
           1.6)
    shape(ax, 39.0, y0 + bh + 2.0, f"[1, T, {C}]", DIM, 8, "center")
    shape(ax, 54.2, y0 + bh + 2.0, f"Q,K,V  [1, T, {C}]", DIM, 8, "center")
    shape(ax, 84.0, y0 + bh + 2.0, f"[1, T, {C}]", DIM, 8, "center")
    shape(ax, 69.5, y0 - 2.2, f"[1, {H}, T, {hd}]  per head", DIM, 8, "center")

    # back to the stream
    poly(ax, [(xs[4] + ws[4], y0 + bh / 2), (98 - 0.5, y0 + bh / 2),
              (98 - 0.5, 52), (SP, 52), (SP, 50.2)], RES)
    plus(ax, SP, 48.5)
    ax.text(SP - 3.2, 48.5, "add", ha="right", color=DIM, fontsize=9)

    # ---------------- sub-layer 2: feed-forward ----------------------------
    ar(ax, (SP, 47), (SP, 41.5))
    ax.add_patch(Circle((SP, 41), 0.9, facecolor=RES, edgecolor=RES, zorder=6))

    ax.add_patch(FancyBboxPatch((24, 12.5), 74, 30,
                                boxstyle="round,pad=0.5", facecolor="#111a12",
                                edgecolor=GREEN, lw=1.4, linestyle="--"))
    ax.text(25.5, 40.2, "SUB-LAYER 2  ·  feed-forward (SwiGLU)", color=GREEN,
            fontsize=11.5, weight="bold")

    poly(ax, [(SP, 41), (SP, 30), (26, 30)], RES)
    rbox(ax, 26, 24.5, 12.5, 11,
         [("RMSNorm", 11, TEAL), ("ln2", 9, DIM),
          (f"{n['prelude.0.ln2.weight']} params", 8.5, DIM)], TEAL)

    # the two parallel branches
    rbox(ax, 45, 30, 15, 8.5,
         [("W1  ->  SiLU", 10.5, GREEN), (f"{C} -> {FF}", 9, DIM),
          ("0.721M", 8.5, AMBER)], GREEN)
    rbox(ax, 45, 17, 15, 8.5,
         [("W3", 10.5, GREEN), (f"{C} -> {FF}", 9, DIM),
          ("0.721M", 8.5, AMBER)], GREEN)
    poly(ax, [(38.5, 30), (41.5, 30), (41.5, 34.25), (45, 34.25)], DIM, 1.6)
    poly(ax, [(41.5, 30), (41.5, 21.25), (45, 21.25)], DIM, 1.6)

    ax.add_patch(Circle((66, 27.75), 2.4, facecolor="#161b22",
                        edgecolor=AMBER, lw=2.0))
    ax.text(66, 27.75, "x", ha="center", va="center", color=AMBER,
            fontsize=15, weight="bold")
    poly(ax, [(60, 34.25), (63.2, 34.25), (63.2, 29.6)], AMBER, 1.6)
    poly(ax, [(60, 21.25), (63.2, 21.25), (63.2, 25.9)], AMBER, 1.6)
    ax.text(66, 32.4, "elementwise", ha="center", color=AMBER, fontsize=8.5)
    ax.text(66, 22.6, "one branch GATES\nthe other", ha="center", color=DIM,
            fontsize=8.5, linespacing=1.4)

    rbox(ax, 78, 23.5, 14, 8.5,
         [("W2", 11, GREEN), (f"{FF} -> {C}", 9, DIM),
          ("0.721M", 8.5, AMBER)], GREEN)
    ar(ax, (68.4, 27.75), (78, 27.75), AMBER, 1.6)
    shape(ax, 52.5, 40.0, f"[1, T, {FF}]", DIM, 8, "center")

    poly(ax, [(92, 27.75), (97.5, 27.75), (97.5, 9), (SP, 9), (SP, 7.3)], RES)
    plus(ax, SP, 5.6)
    ar(ax, (SP, 4.1), (SP, 1.4))
    rbox(ax, SP - 8, -3.2, 16, 4.6, [("x   out", 12, FG)], FG, lw=2.2)
    ax.text(SP, -5.6, f"[1, T, {C}]   —   same shape as it came in",
            ha="center", va="center", color=DIM, fontsize=9.5,
            family="monospace")

    # ---------------- the pooled variant -----------------------------------
    ax.add_patch(FancyBboxPatch((24, -9), 74, 15,
                                boxstyle="round,pad=0.4", facecolor="#1c1608",
                                edgecolor=AMBER, lw=1.6))
    ax.text(25.5, 3.9, "IN THE RECURRENT BLOCK, SUB-LAYER 2 IS REPLACED BY "
            "THE POOL", color=AMBER, fontsize=11, weight="bold")
    ax.text(25.5, 0.7,
            f"Same residual structure, same RMSNorm. But instead of one fixed "
            f"SwiGLU of width {FF}, a router scores the\n"
            f"{cfg['pool_resident']} experts on the card, takes the top "
            f"{cfg['pool_top_k']}, and sums their outputs. Each expert is its "
            f"own SwiGLU of width {cfg['pool_d_ff']}.\n"
            f"So sub-layer 2 becomes {cfg['pool_top_k']} x 3.146M = "
            f"{cfg['pool_top_k']*3.146:.1f}M parameters — chosen per character "
            f"— against the 2.16M fixed here.",
            color=DIM, fontsize=9.6, va="top", linespacing=1.7)

    fig.savefig(out, facecolor=BG, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ===========================================================================
# 2. the whole tower
# ===========================================================================
def fig_stack(man, cfg, n, out):
    C = cfg["d_model"]; V = 265; M = 1e6
    S = cfg["max_steps"]
    fig = plt.figure(figsize=(15.5, 13.6), facecolor=BG)
    fig.text(0.5, 0.966, "the whole model, bottom to top", ha="center",
             color=FG, fontsize=26, weight="bold")
    fig.text(0.5, 0.936,
             f"read upward, like any transformer diagram. the only unusual "
             f"thing is that the third block is applied up to {S} times",
             ha="center", color=DIM, fontsize=13)

    ax = fig.add_axes([0.03, 0.02, 0.94, 0.895], facecolor=BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    X = 34.0; W = 30.0                      # the tower

    def tier(y, h, lines, colour, lw=1.8, face="#161b22"):
        rbox(ax, X, y, W, h, lines, colour, lw=lw, face=face)

    def sh(y, txt, colour=DIM):
        shape(ax, X + W + 2.2, y, txt, colour, 9.5)

    # bottom to top
    tier(2, 6, [("the characters read so far", 11, FG),
                ('"the cat sat on the ma"', 9, DIM)], FG)
    sh(5, "[1, T]   T up to 32,768")
    ar(ax, (X + W / 2, 8), (X + W / 2, 12))

    tier(12, 7, [("tok_emb", 12, TEAL), (f"{V} x {C}   0.136M", 9.5, DIM)],
         TEAL)
    sh(15.5, f"[1, T, {C}]")
    ar(ax, (X + W / 2, 19), (X + W / 2, 23))

    tier(23, 8, [("prelude.0", 12, BLUE),
                 ("attention + SwiGLU", 9.5, DIM), ("3.212M", 9, AMBER)], BLUE)
    sh(27, f"[1, T, {C}]")
    ar(ax, (X + W / 2, 31), (X + W / 2, 34))
    tier(34, 8, [("prelude.1", 12, BLUE),
                 ("attention + SwiGLU", 9.5, DIM), ("3.212M", 9, AMBER)], BLUE)
    sh(40.2, f"[1, T, {C}]  =  x, the reading")
    ax.text(X - 2.5, 32.5, "run ONCE\nper character", ha="right",
            color=BLUE, fontsize=9.5, linespacing=1.5)

    # ---- the loop ---------------------------------------------------------
    ax.add_patch(FancyBboxPatch((X - 12, 45.5), W + 26, 31,
                                boxstyle="round,pad=0.5", facecolor="#111a12",
                                edgecolor=GREEN, lw=1.8, linestyle="--"))
    ax.text(X - 11, 74.2, f"APPLIED UP TO {S} TIMES", color=GREEN,
            fontsize=11.5, weight="bold")

    ar(ax, (X + W / 2, 42), (X + W / 2, 48))
    tier(48, 8, [("adapter", 12, PURPLE),
                 (f"concat[h, x]  {2*C} -> {C}", 9.5, DIM),
                 ("0.524M", 9, AMBER)], PURPLE)
    sh(54.2, f"[1, T, {C}]")
    ar(ax, (X + W / 2, 56), (X + W / 2, 60))
    tier(60, 10, [("recur.0", 12, GREEN),
                  ("attention  1.050M", 9.5, DIM),
                  (f"+ pool: top {cfg['pool_top_k']} of "
                   f"{cfg['pool_resident']} experts", 9.5, AMBER),
                  ("25.2M runs here", 9, AMBER)], GREEN, lw=2.2)
    sh(65, f"[1, T, {C}]  =  h")

    # the loop-back, on the left
    poly(ax, [(X, 65), (X - 9, 65), (X - 9, 52), (X, 52)], GREEN, 2.2)
    ax.text(X - 9.6, 58.5, "h", ha="right", color=GREEN, fontsize=13,
            weight="bold")
    ax.text(X - 9.6, 55.5, "fed back", ha="right", color=GREEN, fontsize=8.5)
    # x re-injected
    poly(ax, [(X + W, 36), (X + W + 28, 36), (X + W + 28, 50), (X + W, 50)],
         FG, 1.6)
    ax.text(X + W + 28.6, 43, "x again,\nevery pass", ha="left", color=FG,
            fontsize=9, linespacing=1.4)

    ar(ax, (X + W / 2, 70), (X + W / 2, 78))
    ax.text(X + W / 2 + 1.2, 74, "after EVERY pass", ha="left", color=RED,
            fontsize=9)

    tier(78, 6, [("RMSNorm  ln_f", 11, TEAL)], TEAL)
    ar(ax, (X + W / 2, 84), (X + W / 2, 87))
    tier(87, 7, [("head", 12, TEAL),
                 (f"{C} -> {V}   tied to tok_emb", 9.5, DIM)], TEAL)
    sh(90.5, f"[1, T, {V}]   the answer at THIS depth")

    # the halting tap
    poly(ax, [(X, 81), (X - 16, 81), (X - 16, 88)], RED, 1.8)
    rbox(ax, X - 30, 88, 26, 7,
         [("halt", 11.5, RED), (f"{C} -> 1     513 params", 9, DIM)], RED)
    ax.text(X - 17, 84.4, "lambda_n = am I done?", ha="right", color=RED,
            fontsize=9)

    ax.text(50, 97.5,
            "at inference it stops when the halting mass crosses the "
            "threshold; while training it runs on and mixes every depth",
            ha="center", color=DIM, fontsize=10)
    fig.savefig(out, facecolor=BG, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ===========================================================================
# 3. the loop unrolled, with the backward pass
# ===========================================================================
def fig_bptt(man, cfg, n, halt_p, out):
    S = int(cfg["max_steps"]); W = int(cfg["bptt_window"])
    mean = float(cfg.get("train_steps_mean") or 0)
    cut = max(0, S - W)                 # passes 1..cut are detached
    fig = plt.figure(figsize=(19.0, 10.6), facecolor=BG)
    fig.text(0.5, 0.963, "the loop unrolled, and the gradient coming back",
             ha="center", color=FG, fontsize=25, weight="bold")
    fig.text(0.5, 0.930,
             f"the same block drawn once per pass, the way a recurrent "
             f"network is unrolled. all {S} passes shown, at the real "
             f"bptt_window of {W}",
             ha="center", color=DIM, fontsize=12.5)

    ax = fig.add_axes([0.03, 0.04, 0.94, 0.868], facecolor=BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    x0, pitch, bw = 4.0, 3.95, 3.05
    top, bot = 74.0, 60.0               # the pass boxes
    cx = lambda i: x0 + i * pitch + bw / 2

    # x, entering every pass
    ax.plot([2.5, 97.5], [92, 92], color=FG, lw=2.0)
    ax.text(1.8, 92, "x", ha="right", va="center", color=FG, fontsize=14,
            weight="bold")
    ax.text(2.5, 95.2, "the reading — the same x enters every pass, which is "
            "why the loop cannot drift from the text",
            color=DIM, fontsize=9.5)

    for i in range(S):
        det = i < cut
        col = "#484f58" if det else GREEN
        ar(ax, (cx(i), 92), (cx(i), top + 0.4), FG, 0.9, ms=8)
        ax.add_patch(FancyBboxPatch((x0 + i * pitch, bot), bw, top - bot,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#161b22" if det else "#122016",
                                    edgecolor=col, lw=1.5))
        ax.text(cx(i), (top + bot) / 2, f"{i+1}", ha="center",
                va="center", color=col, fontsize=10.5, weight="bold")
        if i < S - 1:
            ar(ax, (x0 + i * pitch + bw, (top + bot) / 2),
               (x0 + (i + 1) * pitch, (top + bot) / 2), RES, 1.4, ms=9)

    ax.text(x0, top + 2.0, "each box:  adapter -> attention -> pool "
            "(top 8 experts)   ·   h passes along the arrows",
            color=DIM, fontsize=9.5, ha="left")

    # the cut, where it really falls
    xc = x0 + cut * pitch - (pitch - bw) / 2
    ax.plot([xc, xc], [bot - 8, top + 2], color=RED, lw=2.6)
    ax.text(xc, top + 9.0, "h.detach()", ha="center", color=RED,
            fontsize=11.5, weight="bold")
    ax.text(xc, top + 6.0, f"passes 1-{cut} are cut off from the graph",
            ha="center", color=RED, fontsize=9)

    # backward, only on the attached side
    for i in range(cut, S - 1):
        ar(ax, (x0 + (i + 1) * pitch - 0.2, bot - 3.4),
           (x0 + i * pitch + bw + 0.2, bot - 3.4), RED, 1.5, ms=9,
           ls=(0, (3, 2)))
    ax.text((cx(cut) + cx(S - 1)) / 2, bot - 5.0,
            f"gradient flows back along h, through the last {W} passes only",
            ha="center", color=RED, fontsize=9.5)

    # the halting weight each pass carries, hanging below
    hp = halt_p or []
    scale = (20.0 / max(hp)) if hp else 0.0
    base = 46.0
    for i in range(S):
        if i < len(hp):
            h = hp[i] * scale
            ax.add_patch(plt.Rectangle((x0 + i * pitch + 0.35, base - h),
                                       bw - 0.7, h, facecolor=AMBER,
                                       edgecolor="none"))
            if hp[i] > 0.04:
                ax.text(cx(i), base - h - 1.8, f"{hp[i]:.2f}", ha="center",
                        color=AMBER, fontsize=7.5)
        ar(ax, (cx(i), bot - 8.6), (cx(i), base + 0.6), DIM, 0.7, ms=6)
    ax.plot([x0, x0 + S * pitch - (pitch - bw)], [base, base], color=GRID,
            lw=1.0)
    ax.text(x0, base + 6.0, "p_n — the halting mass each pass carries",
            color=AMBER, fontsize=9.5)
    if hp:
        ax.text(97, base - 9,
                f"pass 1 alone holds {hp[0]:.0%} of the mass\n"
                f"passes 4-{S} together hold {sum(hp[3:]):.1%}\n"
                f"(measured on real text)",
                ha="right", color=AMBER, fontsize=9.5, linespacing=1.6)

    ax.text(x0, 20.5,
            f"loss  =  sum over passes of  p_n x CE_n   +   "
            f"{cfg['ponder_beta']} x KL(p || geometric({cfg['halt_prior']}))",
            color=FG, fontsize=14, family="monospace")
    ax.text(x0, 15.0,
            "Every pass computes what it would say, and every pass is trained "
            "— weighted by how much halting mass it holds. There is no "
            "discrete stop\ndecision anywhere for gradients to pass through, "
            "which is the whole trick: halting learns because being right "
            "EARLY is directly rewarded.\n\n"
            f"The cut keeps memory from growing with depth. But depth is "
            f"sampled per step as Poisson({mean:g})+1, so n_steps reaches "
            f"{S} only rarely; at the\ntypical depth of about {mean+1:.0f} "
            f"the window of {W} is larger than the whole loop and the cut "
            f"never fires at all. Averaged over training it removes\n"
            f"0.61 passes. The mechanism is real; at these settings it is "
            f"barely engaged.",
            color=DIM, fontsize=10.2, va="top", linespacing=1.75)
    fig.savefig(out, facecolor=BG, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--halt", default="runs/halt_mass.json",
                    help="optional measured halting mass per pass")
    a = ap.parse_args()
    man, cfg, n = load(a.weights)
    halt_p = None
    if os.path.exists(a.halt):
        halt_p = json.load(open(a.halt))
    print(f"  {a.weights} at step {man['step']:,}")
    fig_block(man, cfg, n, os.path.join(a.outdir, "block_opened.png"))
    fig_stack(man, cfg, n, os.path.join(a.outdir, "trunk_stack.png"))
    fig_bptt(man, cfg, n, halt_p, os.path.join(a.outdir, "trunk_bptt.png"))


if __name__ == "__main__":
    main()
