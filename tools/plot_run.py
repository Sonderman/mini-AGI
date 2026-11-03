#!/usr/bin/env python3
"""
Training curves for a run, read straight out of its log.

The trainer keeps no structured history - the log IS the history - so this
parses it back into numbers and draws them. Safe to run against a live run;
it just plots whatever has been written so far.

    python3 plot_run.py --log runs/agi.log --out runs/agi_curves.png
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, PANEL, EDGE = "#0B1014", "#131C22", "#243139"
INK, DIM, FAINT = "#DDE5E9", "#8298A4", "#5B6E79"
SIG, MEA, WRN, GRN = "#E2A35C", "#57B4C0", "#CE7A76", "#8FBF7A"
MONO = "DejaVu Sans Mono"

STEP = re.compile(
    r"^step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+lr\s+([\d.e+-]+)\s+"
    r"gn\s+([\d.]+)\s+ponder\s+([\d.]+)")
VAL = re.compile(r"val\s+([\d.]+)\s+ppl\s+([\d.]+)(?:\s+ponder\s+([\d.]+))?")
GREW = re.compile(r"GREW\s+\+(\d+)\s+->\s+(\d+)\s+experts")
PRUNE = re.compile(r"pruned\s+(\d+)")
ARMED = re.compile(r"pool starts at\s+(\d+)\s+experts")


def parse(path, eval_every):
    """Log lines back into arrays. Progress-bar noise is filtered out."""
    steps, loss, lr, gn, ponder = [], [], [], [], []
    vsteps, vloss, vppl = [], [], []
    grew, pruned = [], []
    total = None
    last = 0
    running = None          # true expert count, carried forward across growths
    for raw in open(path, errors="ignore"):
        line = raw.strip()
        if re.match(r"^W\d", line):          # warning spam
            continue
        m = ARMED.search(line)
        if m:
            running = int(m.group(1))
            continue
        m = STEP.match(line)
        if m:
            last = int(m.group(1))
            total = int(m.group(2))
            steps.append(last)
            loss.append(float(m.group(3)))
            lr.append(float(m.group(4)))
            gn.append(float(m.group(5)))
            ponder.append(float(m.group(6)))
            continue
        m = VAL.search(line)
        if m and "loaded" not in line:
            # an eval fires at the last step where (step+1) % eval_every == 0,
            # which is the next such step at or after the last one logged
            s = ((last // eval_every) + 1) * eval_every - 1
            vsteps.append(s)
            vloss.append(float(m.group(1)))
            vppl.append(float(m.group(2)))
            continue
        m = GREW.search(line)
        if m:
            # growth is decided inside an eval, so it lands on the eval step,
            # not on the last step whose line happened to be printed
            s_ = ((last // eval_every) + 1) * eval_every - 1
            k, n = int(m.group(1)), int(m.group(2))
            # the running trainer logs the count from BEFORE the addition, so
            # carry the true total forward from the armed line instead and use
            # the logged number only when it already agrees
            if running is not None:
                running += k
                if n != running:
                    n = running
            grew.append((s_, k, n))
            continue
        m = PRUNE.search(line)
        if m:
            k = int(m.group(1))
            if running is not None:
                running -= k
            # the log states the resulting count outright - trust it if present
            m2 = re.search(r"->\s*(\d+)\s+experts", line)
            if m2:
                running = int(m2.group(1))
            pruned.append((((last // eval_every) + 1) * eval_every - 1, k,
                           running))
    return dict(steps=np.array(steps), loss=np.array(loss), lr=np.array(lr),
                gn=np.array(gn), ponder=np.array(ponder),
                vsteps=np.array(vsteps), vloss=np.array(vloss),
                vppl=np.array(vppl), grew=grew, pruned=pruned, total=total)


def style(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL)
    ax.grid(True, color=EDGE, lw=.6, alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=FAINT, labelsize=8)
    for s in ax.spines.values():
        s.set_color(EDGE)
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, family=MONO, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, family=MONO, fontsize=9)


def mark_growth(ax, d, label=True):
    for i, (s, k, n) in enumerate(d["grew"]):
        ax.axvline(s, color=GRN, lw=1.1, ls="--", alpha=.75, zorder=1)
        if label:
            ax.annotate(f"+{k}", xy=(s, ax.get_ylim()[1]),
                        xytext=(2, -11), textcoords="offset points",
                        color=GRN, family=MONO, fontsize=8, va="top")
    for rec in d["pruned"]:
        ax.axvline(rec[0], color=WRN, lw=1.1, ls=":", alpha=.75, zorder=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/results/agi.log")
    ap.add_argument("--out", default="runs/agi_curves.png")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--title", default="mini-AGI — current run")
    args = ap.parse_args()

    d = parse(args.log, args.eval_every)
    if len(d["steps"]) == 0:
        print(f"no step lines in {args.log}", file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(13, 7.5), dpi=130, facecolor=BG)
    fig.text(.045, .955, args.title.upper(), color=INK, family=MONO,
             fontsize=13)
    span = f"steps {d['steps'][0]}–{d['steps'][-1]} of {d['total']}"
    fig.text(.045, .922, f"{span}   ·   {len(d['steps'])} training points, "
                         f"{len(d['vsteps'])} validation points",
             color=DIM, family=MONO, fontsize=9)

    # --- loss, train and val together --------------------------------------
    ax = fig.add_axes([.06, .40, .60, .47])
    style(ax, "step", "cross-entropy loss (nats/char)")
    ax.plot(d["steps"], d["loss"], color=SIG, lw=1.8, marker="o", ms=3.4,
            label="train (per-step batch)", zorder=3)
    if len(d["vsteps"]):
        st = "-" if len(d["vsteps"]) > 1 else "none"
        ax.plot(d["vsteps"], d["vloss"], color=MEA, lw=2.2, ls=st,
                marker="D", ms=6, label="validation (held out)", zorder=4)
        for s, v in zip(d["vsteps"], d["vloss"]):
            ax.annotate(f"{v:.4f}", xy=(s, v), xytext=(6, 7),
                        textcoords="offset points", color=MEA,
                        family=MONO, fontsize=8)
    mark_growth(ax, d)
    lg = ax.legend(facecolor=PANEL, edgecolor=EDGE, fontsize=8, loc="upper right")
    for t in lg.get_texts():
        t.set_color(INK); t.set_family(MONO)

    # --- perplexity --------------------------------------------------------
    ax2 = fig.add_axes([.72, .40, .245, .47])
    style(ax2, "step", "val perplexity (per character)")
    if len(d["vsteps"]):
        ax2.plot(d["vsteps"], d["vppl"], color=MEA, lw=2.2,
                 ls="-" if len(d["vsteps"]) > 1 else "none", marker="D", ms=6)
        for s, v in zip(d["vsteps"], d["vppl"]):
            ax2.annotate(f"{v:.2f}", xy=(s, v), xytext=(6, 7),
                         textcoords="offset points", color=MEA,
                         family=MONO, fontsize=8)
    else:
        ax2.text(.5, .5, "no eval yet", ha="center", va="center",
                 color=FAINT, family=MONO, fontsize=10, transform=ax2.transAxes)
    mark_growth(ax2, d, label=False)

    # --- the three diagnostics that say whether the run is healthy ----------
    for i, (key, col, lab) in enumerate([
            ("lr", FAINT, "learning rate"),
            ("gn", WRN, "gradient norm"),
            ("ponder", GRN, "ponder steps used")]):
        a = fig.add_axes([.06 + i * .315, .08, .245, .21])
        style(a, "step", lab)
        a.plot(d["steps"], d[key], color=col, lw=1.6, marker="o", ms=2.6)
        if key == "lr":
            a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            a.yaxis.get_offset_text().set_color(FAINT)
        mark_growth(a, d, label=False)

    if d["grew"] or d["pruned"]:
        seq = sorted([(s, f"+{k}", n) for s, k, n in d["grew"]] +
                     [(s, f"-{k}", n) for s, k, n in d["pruned"]])
        note = "  ".join(f"{s}:{t}→{n}" for s, t, n in seq)
        fig.text(.045, .345, "pool size (green dashed = grew, "
                             "red dotted = pruned)", color=DIM,
                 family=MONO, fontsize=8.5)
        fig.text(.045, .318, note[:150], color=INK, family=MONO, fontsize=8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor=BG)
    plt.close(fig)

    print(f"train loss  {d['loss'][0]:.4f} -> {d['loss'][-1]:.4f}"
          f"   ({len(d['steps'])} points, steps {d['steps'][0]}-{d['steps'][-1]})")
    if len(d["vsteps"]) > 1:
        print(f"val loss    {d['vloss'][0]:.4f} -> {d['vloss'][-1]:.4f}"
              f"   ppl {d['vppl'][0]:.2f} -> {d['vppl'][-1]:.2f}")
    elif len(d["vsteps"]) == 1:
        print(f"val loss    {d['vloss'][0]:.4f} at step {d['vsteps'][0]} "
              f"(ppl {d['vppl'][0]:.2f}) - one point, no trend yet")
    events = [(s, f"+{k}", n) for s, k, n in d["grew"]] + \
             [(s, f"-{k}", n) for s, k, n in d["pruned"]]
    for s, tag, n in sorted(events):
        print(f"pool        step {s}: {tag:>4} -> {n} experts")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
