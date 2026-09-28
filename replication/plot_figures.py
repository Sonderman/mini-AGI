#!/usr/bin/env python3
"""
The continual-learning figures, drawn for a slide rather than for a notebook.

Two claims, two pictures:

  probe    reading ONE subject for half a million characters, does what the
           model knows about the other seven decay? Plotted as CHANGE from
           the baseline score, because the question is about movement and a
           panel of absolute losses hides it behind the spread between
           domains. The read subject carries the accent; the seven that are
           not being read stay neutral - which is the finding, so the chart
           should look like the finding.

  control  the same batch-1 stream given to a routed pool and to a dense
           model of the SAME parameter count, over a strictly sequential
           curriculum. The first domain's line is the one that matters: if
           reading moves on and that line bends upward, the model forgot.

    python3 replication/plot_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.ticker import FuncFormatter                  # noqa: E402

ACCENT = "#2a78d6"          # categorical slot 1
WARM = "#eb6834"            # slot 2, for the dense/forgetting case
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def R(name):
    """A shipped result."""
    return os.path.join(HERE, "results", name)


def A(name):
    """A figure the README shows."""
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    return os.path.join(ROOT, "assets", name)


INK = "#0b0b0b"
SUB = "#52514e"
MUTE = "#9a9892"
GRID = "#e4e3df"
SURF = "#fcfcfb"
BAND = "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": SUB,
    "xtick.color": SUB, "ytick.color": SUB,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True,
})
NICE = {"chat_hermes": "chat (hermes)", "self-knowledge": "self-knowledge"}


def label(d):
    return NICE.get(d, d)


def fig_probe(path, out):
    j = json.load(open(path))
    doms, rows = j["domains"], j["rows"]
    base = j["baseline"]
    lanes = [d for d in j.get("lanes", [j["domain"]]) if d in doms]
    rotating = bool(j.get("rotating"))
    read = j["domain"]
    xs = [r["chars"] / 1000 for r in rows]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.45, 1]})

    # ACCENT THE DOMAIN THE PROBE IS ABOUT, in both arms. Plotting the mean of
    # the read lanes was right when one lane was read and wrong the moment
    # several were: interleaved, the new domain's -0.31 averaged away against
    # seven subjects that barely moved, and the figure showed -0.046.
    others = [d for d in doms if d != read]
    lo = [min(r[d] - base[d] for d in others) for r in rows]
    hi = [max(r[d] - base[d] for d in others) for r in rows]
    ax.fill_between(xs, lo, hi, color=BAND, zorder=1,
                    label=f"range across the other {len(others)} subjects")
    for d in others:
        ax.plot(xs, [r[d] - base[d] for r in rows], color=MUTE,
                lw=1.3, zorder=2)
    readv = [r[read] - base[read] for r in rows]
    ax.plot(xs, readv, color=ACCENT, lw=2.4, marker="o", ms=5.5, zorder=4,
            label=f"{label(read)} - the new domain")
    ax.axhline(0, color=SUB, lw=1.1, zorder=3)

    # Labels at the right edge collide when several domains land together, so
    # they are pushed apart by a minimum gap in DATA units before drawing -
    # each keeps its line's order, none sits on top of another.
    last = rows[-1]
    ends = sorted(((last[d] - base[d], d) for d in others), reverse=True)
    # The minimum gap has to be a fraction of the AXIS, not of the spread of
    # the labels themselves. Interleaved, the other subjects sit within 0.02
    # of zero on an axis 0.8 tall, so a gap scaled to their own spread put
    # eight labels on top of one another.
    span = max(max(hi), abs(min(lo)), abs(readv[-1])) * 1.35
    gap = span * 2 * 0.043
    placed = []
    for v, d in ends:
        y = v if not placed else min(v, placed[-1][0] - gap)
        placed.append((y, d, v))
    for y, d, v in placed:
        ax.annotate(label(d), (xs[-1], y), xytext=(7, 0),
                    textcoords="offset points", va="center",
                    fontsize=8.5, color=SUB)
        if abs(y - v) > 1e-9:                # leader line back to the curve
            ax.plot([xs[-1], xs[-1] * 1.035], [v, y], color=GRID, lw=0.8,
                    zorder=1, clip_on=False)
    ax.annotate(label(read), (xs[-1], readv[-1]),
                xytext=(7, 0), textcoords="offset points", va="center",
                fontsize=9.5, color=ACCENT, weight="bold")
    ax.set_ylim(-span, span)
    ax.set_xlim(0, xs[-1] * 1.20)
    ax.set_ylabel("change in held-out loss (nats)")
    ax.set_xlabel("thousands of characters read, batch 1"
                  if rotating else
                  f"thousands of characters of {label(read)} read, batch 1")
    # 2 decimals collided on this axis - the band is a few thousandths wide,
    # so two different ticks both printed -0.01
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}"))
    ax.set_title("A. Held-out loss per subject, relative to baseline",
                 loc="left", color=INK, fontsize=12.5, weight="bold", pad=14)
    ax.text(0, 1.015, "higher = worse", transform=ax.transAxes, fontsize=8,
            color=SUB, ha="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9)

    # --- right: where the gradient actually went
    n_tot = j["experts_total"]
    touched = [r["experts_touched"] for r in rows]
    ax2.plot(xs, touched, color=ACCENT, lw=2.2, marker="o", ms=5)
    ax2.axhline(n_tot, color=SUB, lw=1.1, ls=(0, (4, 3)))
    ax2.annotate(f"all {n_tot} experts in the pool", (0, n_tot),
                 xytext=(4, -13), textcoords="offset points",
                 fontsize=9, color=SUB)
    ax2.set_ylim(0, n_tot * 1.14)
    ax2.set_xlim(0, xs[-1] * 1.02)
    ax2.set_xlabel("thousands of characters read")
    ax2.set_ylabel("experts having received a gradient (cumulative)")
    ax2.set_title("B. Experts receiving gradient",
                  loc="left", color=INK, fontsize=12.5, weight="bold", pad=14)
    ax2.annotate(f"{touched[-1]} of {n_tot}\n({touched[-1]/n_tot:.0%})",
                 (xs[-1], touched[-1]), xytext=(-8, 10),
                 textcoords="offset points", ha="right", fontsize=11,
                 color=ACCENT, weight="bold")

    # The heading states the arm and its two end points. It used to carry a
    # verdict in capitals, which is a conclusion rather than a measurement and
    # does not belong on the axis it is drawn from.
    n = rows[-1]["chars"]
    rest = float(np.mean([rows[-1][d] - base[d] for d in others]))
    if rotating:
        head = (f"Interleaved: {label(read)} read as one of {len(lanes)} "
                f"subjects, {n:,} characters at batch 1 - {label(read)} "
                f"{readv[-1]:+.4f} nats, the other {len(others)} {rest:+.4f}")
    else:
        head = (f"Massed exposure to {label(read)}, {n:,} characters at "
                f"batch 1 - {label(read)} {readv[-1]:+.4f} nats, "
                f"the {len(others)} withheld subjects {rest:+.4f}")
    fig.suptitle(head, x=0.008, ha="left", fontsize=13.5, weight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=170)
    print(f"  -> {out}")
    return j


def _mean_delta(j, which="other"):
    """Mean change from baseline across the read lanes, or across the rest."""
    base, doms = j["baseline"], j["domains"]
    lanes = j.get("lanes", [j["domain"]])
    sel = [d for d in doms if (d not in lanes if which == "other" else d in lanes)]
    if not sel:
        return None
    return [(r["chars"], float(np.mean([r[d] - base[d] for d in sel])))
            for r in j["rows"]]


def fig_ablation(out):
    """
    Three configurations of the same checkpoint, and the control.

    AdEMAMix was tested and dropped - see FINDINGS.md. What is left is one
    axis, the trunk learning rate, and one architectural ablation: frozen
    experts, which is the same model with the pool's defining property
    switched off. Right panel is progress retained against chance, because
    that is the number that decides whether "catastrophic" is the right word.
    """
    SPECS = [
        (R("static.json"),
         "experts FROZEN, trunk LR = expert LR", "#3a3a38", "-", 3.0),
        (R("none.json"),
         "swapping normally, trunk LR = expert LR", "#e34948", "-", 2.5),
        (R("trunk01.json"),
         "swapping normally, trunk at 0.1x   (the run)", "#2a78d6", "-", 2.7),
    ]
    RAND = float(np.log(265))
    _j0 = json.load(open(SPECS[-1][0]))
    READ = label(_j0["domain"])
    NOTH = len([d for d in _j0["domains"]
                if d not in _j0.get("lanes", [_j0["domain"]])])
    CHARS = _j0["rows"][-1]["chars"]

    fig, (ax, axl, ax2) = plt.subplots(
        1, 3, figsize=(17.6, 5.6),
        gridspec_kw={"width_ratios": [1.5, 1.15, 1]})

    if os.path.exists(R("control.json")):
        c = json.load(open(R("control.json")))
        rd = _mean_delta(c, "read")
        if rd:
            ax.plot([x / 1000 for x, _ in rd], [y for _, y in rd],
                    color=MUTE, lw=1.8, ls=(0, (2, 2)), zorder=3,
                    label=f"control: {len(c.get('lanes', []))} subjects read, "
                          f"interleaved\n(the regime the run actually uses)")
    names, ret, cols, xmax, ends = [], [], [], 1, []
    names, ret, cols, xmax = [], [], [], 1
    for path, name, col, dash, lw in SPECS:
        if not os.path.exists(path):
            continue
        j = json.load(open(path))
        d = _mean_delta(j)
        xs = [x / 1000 for x, _ in d]; ys = [y for _, y in d]
        xmax = max(xmax, xs[-1])
        ax.plot(xs, ys, color=col, lw=lw, marker="o", ms=5, zorder=4,
                label=name)
        ends.append((ys[-1], col))
        # WHAT IT LEARNED, beside what it forgot. An arm that forgets less
        # is only interesting if it still learns the thing it is reading -
        # otherwise "nothing was forgotten" is just "nothing happened".
        rd = _mean_delta(j, "read")
        axl.plot([x / 1000 for x, _ in rd], [y for _, y in rd],
                 color=col, lw=lw, marker="o", ms=5, zorder=4)
        axl.annotate(f"{rd[-1][1]:+.3f}", (rd[-1][0] / 1000, rd[-1][1]),
                     xytext=(8, 0), textcoords="offset points", va="center",
                     fontsize=10.5, color=col, weight="bold")
        base, doms = j["baseline"], j["domains"]
        lanes = j.get("lanes", [j["domain"]])
        oth = [x for x in doms if x not in lanes]
        last = j["rows"][-1]
        ret.append(float(np.mean([(RAND - last[x]) / (RAND - base[x])
                                  for x in oth])) * 100)
        names.append(name.split(",")[0] if "FROZEN" in name
                     else ("trunk at 0.1x" if "0.1x" in name
                           else "trunk LR = expert LR"))
        cols.append(col)

    # two arms can finish within a few thousandths of each other, and their
    # labels then print on top of one another - push them apart in data units
    gap = (max(v for v, _ in ends) - min(v for v, _ in ends) or 1.0) * 0.075
    placed = []
    for v, col in sorted(ends, reverse=True):
        y = v if not placed else min(v, placed[-1] - gap)
        placed.append(y)
        ax.annotate(f"{v:+.3f}", (xmax, y), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=10.5,
                    color=col, weight="bold", annotation_clip=False)

    ax.axhline(0, color=SUB, lw=1.1, zorder=2)
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel(f"thousands of characters of {READ} read, batch 1")
    ax.set_ylabel(f"mean change in held-out loss, the {NOTH} withheld "
                  f"subjects (nats)")
    ax.set_title("A. Withheld subjects", loc="left", color=INK,
                 fontsize=12.5, weight="bold", pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelspacing=0.8)

    axl.axhline(0, color=SUB, lw=1.1, zorder=2)
    axl.set_xlim(0, xmax * 1.20)
    axl.set_xlabel(f"thousands of characters of {READ} read, batch 1")
    axl.set_ylabel(f"change in held-out loss, {READ} (nats)")
    axl.set_title("B. Exposed subject", loc="left", color=INK,
                  fontsize=12.5, weight="bold", pad=14)
    axl.annotate("higher = worse", (0.012, 0.955), xycoords="axes fraction",
                 fontsize=8.5, color=SUB)
    axl.grid(alpha=.35, color=GRID); axl.set_axisbelow(True)

    y = np.arange(len(names))
    ax2.barh(y, ret, height=0.55, color=cols, zorder=3)
    ax2.axvline(100, color=SUB, lw=1.2, ls=(0, (4, 3)), zorder=4)
    for i, v in enumerate(ret):
        ax2.annotate(f"{v:.2f}%", (v, i), xytext=(8, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=12.5, weight="bold", color=cols[i])
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=9.5)
    ax2.invert_yaxis(); ax2.set_xlim(0, 132)
    ax2.set_xlabel("fraction of pre-probe progress retained (%),\n"
                   "measured against chance = ln(265) nats")
    ax2.set_title("C. Progress retained", loc="left", color=INK,
                  fontsize=12.5, weight="bold", pad=14)
    ax2.grid(axis="y", visible=False)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)

    fig.suptitle(f"Massed single-subject exposure: {CHARS:,} characters of "
                 f"{READ} at batch 1, three configurations",
                 x=0.008, ha="left", fontsize=15, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=170)
    print(f"  -> {out}")


def main():
    made = 0
    import numpy as _np
    globals()["np"] = _np
    # worst -> best. Validated as a set against the light surface:
    # red/yellow/aqua/blue, worst adjacent CVD dE 9.1, and every line carries a
    # direct value label, which is the relief the contrast WARN requires.
    fig_ablation(A("mitigations.png"))
    for src, fn, out in (
            (R("trunk01.json"), fig_probe, A("probe_massed.png")),
            (R("control.json"), fig_probe, A("probe_interleaved.png"))):
        if os.path.exists(src):
            fn(src, out)
            made += 1
        else:
            print(f"  (skipped, no {src})")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
