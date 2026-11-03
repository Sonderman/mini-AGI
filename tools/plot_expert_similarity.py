#!/usr/bin/env python3
"""Draw what tools/expert_similarity.py measured.

    python3 tools/expert_similarity.py --weights weights --out runs
    python3 tools/plot_expert_similarity.py --npz runs/expert_similarity.npz
"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import TwoSlopeNorm                    # noqa: E402
from scipy.cluster.hierarchy import linkage, leaves_list      # noqa: E402
from scipy.spatial.distance import squareform                 # noqa: E402

INK, MUTED, GRID = "#1b1b1f", "#6b6b76", "#e4e4ea"
ACC, WARN, COOL = "#c2492d", "#d9a326", "#2d6ec2"


def order_by_cluster(S):
    A = np.nan_to_num(S, nan=1.0)
    A = (A + A.T) / 2
    D = np.clip(1.0 - A, 0, None)
    np.fill_diagonal(D, 0.0)
    return leaves_list(linkage(squareform(D, checks=False), method="average"))


def panel_heat(ax, S, order, title, sub):
    M = np.nan_to_num(S, nan=1.0)[np.ix_(order, order)]
    v = max(0.05, float(np.nanpercentile(np.abs(S[~np.isnan(S)]), 99.5)))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v),
                   interpolation="nearest")
    ax.set_title(title, fontsize=10.5, color=INK, pad=10, loc="left",
                 fontweight="bold")
    ax.set_xlabel(sub, fontsize=7.8, color=MUTED, labelpad=6, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="runs/expert_similarity.npz")
    ap.add_argument("--out", default="runs/expert_similarity.png")
    ap.add_argument("--routers", default="weights/routers.npz")
    a = ap.parse_args()
    d = np.load(a.npz)
    raw, sub, keys = d["raw"], d["subspace"], d["keys"]
    gate, use, born = d["gate"], d["use"], d["born"]
    n = raw.shape[0]
    nrm = np.linalg.norm(
        np.load(a.routers)["recur.0.mlp.router.weight"].astype(np.float64), axis=1)

    fig = plt.figure(figsize=(15.5, 10.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.26,
                          left=0.045, right=0.975, top=0.885, bottom=0.075)
    order = order_by_cluster(sub)

    panel_heat(fig.add_subplot(gs[0, 0]), raw, order,
               "Weight space - are any two experts copies?",
               f"cosine between the stored weights - max pair {np.nanmax(raw):+.2f}")
    panel_heat(fig.add_subplot(gs[0, 1]), sub, order,
               "Function space - do any two do the same job?",
               "cosine between w1'w1 - blind to how each numbers its own neurons")
    panel_heat(fig.add_subplot(gs[0, 2]), keys, order,
               "Router - who does the router think they are?",
               "cosine between router rows - same expert order as the middle panel")

    # ---- nearest neighbour: does each expert have a twin? ----
    ax = fig.add_subplot(gs[1, 0])
    for S, lab, c in ((raw, "weights", COOL), (sub, "function", ACC),
                      (keys, "router", WARN)):
        nn = np.sort(np.nanmax(S, axis=1))
        ax.plot(nn, np.linspace(0, 100, len(nn)), color=c, lw=2, label=lab)
    ax.axvline(0.9, color=MUTED, ls=":", lw=1)
    ax.text(0.9, 4, " a twin", fontsize=7.5, color=MUTED)
    ax.set_xlabel("similarity to its most similar sibling", fontsize=8.5)
    ax.set_ylabel("% of experts below", fontsize=8.5)
    ax.set_title("How alone is each expert?", fontsize=10.5, loc="left",
                 color=INK, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=.3, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # ---- how many independent directions is the disk buying? ----
    ax = fig.add_subplot(gs[1, 1])
    for S, lab, c in ((raw, "weights", COOL), (sub, "function", ACC)):
        A = np.nan_to_num(S, nan=1.0); A = (A + A.T) / 2
        ev = np.clip(np.linalg.eigvalsh(A), 0, None)[::-1]
        ax.plot(np.arange(1, n + 1), np.cumsum(ev) / ev.sum() * 100,
                color=c, lw=2, label=lab)
    ax.plot([1, n], [0, 100], color=MUTED, ls="--", lw=1,
            label="if all were orthogonal")
    for q in (80, 95):
        ax.axhline(q, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("experts, most distinctive first", fontsize=8.5)
    ax.set_ylabel("% of the pool's variety explained", fontsize=8.5)
    ax.set_title("Is the disk buying variety?", fontsize=10.5, loc="left",
                 color=INK, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=.3, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # ---- the two populations ----
    ax = fig.add_subplot(gs[1, 2])
    rank = np.argsort(np.argsort(born))
    sc = ax.scatter(rank, np.maximum(nrm, 1e-5), c=gate, s=16,
                    cmap="viridis", norm=matplotlib.colors.Normalize(0, 1.2),
                    edgecolor="none", alpha=.85)
    ax.set_yscale("log")
    founders = int((born <= born.min()).sum())
    ax.axvline(founders, color=MUTED, ls=":", lw=1)
    ax.text(founders + 6, 3e-5, f" {founders} founding experts to the left",
            fontsize=7.5, color=MUTED)
    ax.set_xlabel("experts in birth order", fontsize=8.5)
    ax.set_ylabel("router row norm", fontsize=8.5)
    ax.set_title("Founders became always-on; the rest are routed",
                 fontsize=10.5, loc="left", color=INK, fontweight="bold")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("gate", fontsize=8, labelpad=2); cb.ax.tick_params(labelsize=7)
    ax.grid(alpha=.3, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    disk = n * 25.17 / 1000
    twins = int((np.nanmax(sub, axis=1) > 0.9).sum())
    fig.suptitle(f"{n} experts, {disk:.1f} GB on disk - how much of it is "
                 f"distinct?", x=0.045, y=0.965, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    fig.text(0.045, 0.925,
             f"No expert is a copy of another in weight space (highest pair "
             f"{np.nanmax(raw):+.2f}).  {twins} of {n} ({twins/n:.0%}) have a "
             f"functional twin above 0.9, worth {twins*25.17/1000:.2f} GB.",
             fontsize=9.5, color=MUTED, ha="left")
    fig.savefig(a.out, dpi=150, facecolor="white")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
