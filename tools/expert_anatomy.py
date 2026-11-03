#!/usr/bin/env python3
"""
One expert, drawn in full: what it is, what it computes, what it costs.

Shapes, parameter counts and weight statistics are read from a real expert
file in the weights directory, not written by hand.

    python3 tools/expert_anatomy.py --out runs/expert_anatomy.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG, FG, DIM = "#0d1117", "#e6edf3", "#8b949e"
BLUE, GREEN, AMBER, PURPLE, RED = ("#58a6ff", "#3fb950", "#d29922",
                                   "#bc8cff", "#f85149")


def box(ax, x, y, w, h, label, sub, colour, fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor="#161b22", edgecolor=colour, lw=2.0))
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
            color=colour, fontsize=fs, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                color=DIM, fontsize=fs - 3.2)


def arrow(ax, x1, y1, x2, y2, colour=DIM, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=15, color=colour, lw=lw,
                                 shrinkA=2, shrinkB=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="runs/expert_anatomy.png")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.weights, "manifest.json")))
    edir = os.path.join(a.weights, "experts")
    f0 = sorted(x for x in os.listdir(edir) if x.endswith(".npz"))[0]
    z = np.load(os.path.join(edir, f0))
    w1, w3, w2 = z["w1"], z["w3"], z["w2"]
    d_ff, d_model = w1.shape
    per = w1.size + w3.size + w2.size
    n = man["n_experts"]
    res = man["cfg"]["pool_resident"]
    k = man["cfg"]["pool_top_k"]
    gate = np.abs(np.array(man["telemetry"]["gate"], float))
    fsize = os.path.getsize(os.path.join(edir, f0))

    fig = plt.figure(figsize=(16.5, 9.6), facecolor=BG)
    fig.text(0.5, 0.955, "one expert", ha="center", color=FG, fontsize=26,
             weight="bold")
    fig.text(0.5, 0.915,
             f"a single SwiGLU block  ·  {per/1e6:.3f}M parameters  ·  "
             f"{fsize/1e6:.1f} MB on disk with its Adam moments  ·  "
             f"{n} of them exist, {res} fit on the card",
             ha="center", color=DIM, fontsize=12.5)

    ax = fig.add_axes([0.03, 0.20, 0.66, 0.68], facecolor=BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    box(ax, 2, 42, 13, 16, "x", f"{d_model} numbers\nthe hidden state", FG)

    box(ax, 26, 66, 19, 20, "w₁ x", f"{d_model} → {d_ff}\n"
        f"{w1.size/1e6:.2f}M weights", BLUE)
    box(ax, 26, 14, 19, 20, "w₃ x", f"{d_model} → {d_ff}\n"
        f"{w3.size/1e6:.2f}M weights", GREEN)
    arrow(ax, 15, 52, 26, 76, BLUE)
    arrow(ax, 15, 48, 26, 24, GREEN)
    ax.text(17.5, 92, "the same x goes\ninto both branches", color=DIM,
            fontsize=9.5, ha="center")

    box(ax, 51, 66, 14, 20, "SiLU", "x·σ(x)\nthe only\nnonlinearity", PURPLE)
    arrow(ax, 45, 76, 51, 76, BLUE)

    ax.add_patch(plt.Circle((74, 50), 4.6, facecolor="#161b22",
                            edgecolor=AMBER, lw=2.2))
    ax.text(74, 50, "×", ha="center", va="center", color=AMBER, fontsize=24,
            weight="bold")
    arrow(ax, 65, 76, 74, 55, PURPLE)
    arrow(ax, 45, 24, 74, 45, GREEN)
    ax.text(88, 66, f"elementwise,\nover all {d_ff}\nhidden units",
            color=AMBER, fontsize=9, ha="center")
    ax.text(50, 2, "one branch GATES the other — that is what makes this a "
            "SwiGLU\nrather than a plain feed-forward network",
            color=DIM, fontsize=10, ha="center")

    box(ax, 84, 42, 14, 16, "w₂", f"{d_ff} → {d_model}\n"
        f"{w2.size/1e6:.2f}M", RED)
    arrow(ax, 78.6, 50, 84, 50, AMBER)

    fig.text(0.505, 0.145,
             "y  =  w₂ ( SiLU(w₁ x)  ⊙  w₃ x )",
             ha="center", color=FG, fontsize=19, family="monospace")
    fig.text(0.505, 0.105,
             f"three matrices, not two. it widens {d_model}→{d_ff} twice in "
             f"parallel, gates one branch by the other, and comes back down.",
             ha="center", color=DIM, fontsize=11.5)

    # ---- right: where it sits, and what it is worth ----------------------
    p = fig.add_axes([0.725, 0.55, 0.245, 0.30], facecolor=BG)
    vals = [np.abs(w1).ravel()[::37], np.abs(w3).ravel()[::37],
            np.abs(w2).ravel()[::37]]
    p.boxplot(vals, tick_labels=["w₁", "w₃", "w₂"], vert=True, showfliers=False,
              patch_artist=True,
              boxprops=dict(facecolor="#161b22", color=DIM),
              medianprops=dict(color=AMBER), whiskerprops=dict(color=DIM),
              capprops=dict(color=DIM))
    p.set_title("|weights| in this expert", color=FG, fontsize=11)
    p.tick_params(colors=DIM, labelsize=9)
    for s in p.spines.values():
        s.set_color(DIM)
    p.grid(alpha=.2, axis="y")

    q = fig.add_axes([0.725, 0.145, 0.245, 0.30], facecolor=BG)
    q.hist(gate, bins=40, color=BLUE)
    q.axvline(0.005, color=RED, ls="--", lw=1.2)
    q.text(0.02, q.get_ylim()[1] * 0.82, "prune's\nmin_gate", color=RED,
           fontsize=8.5)
    q.set_title("every expert's gate", color=FG, fontsize=11)
    q.set_xlabel("gate", color=DIM, fontsize=9)
    q.tick_params(colors=DIM, labelsize=9)
    for s in q.spines.values():
        s.set_color(DIM)
    q.grid(alpha=.2, axis="y")

    fig.text(0.36, 0.052,
             "the output is then multiplied by the expert's GATE — one learned\n"
             "number deciding how much it contributes at all",
             ha="center", color=DIM, fontsize=11)
    fig.text(0.36, 0.012,
             f"a row picks {k} of the {res} resident, so ~{k*per/1e6:.0f}M "
             f"parameters run per row, against 8.1M for the entire trunk",
             ha="center", color=DIM, fontsize=11)
    fig.savefig(a.out, facecolor=BG, dpi=110)
    print(f"  wrote {a.out}")
    print(f"  {d_model}→{d_ff}→{d_model}, {per/1e6:.3f}M params, "
          f"{fsize/1e6:.1f} MB with moments")


if __name__ == "__main__":
    main()
