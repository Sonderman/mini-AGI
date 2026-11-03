"""
Draw the training history that runs/samples.txt records.

The sample log is the only place the per-domain scores are kept, and it is
append-only across sessions, so it is the record of how the model got here
rather than of one run.

    python3 tools/plot_training.py                    # -> runs/training.png
    python3 tools/plot_training.py --log runs/samples.txt --out /tmp/x.png
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEAD = re.compile(r"^step ([\d,]+)\s+([\d.]+)M characters\s+(\d+) min"
                  r"(?:\s+(\d+) experts)?")
LOSS = re.compile(r"^held-out loss ([\d.]+)(?:\s*\+/-([\d.]+))?")
DOMS = re.compile(r"^\s+(\w+ [\d.]+(?:\s+\w+ [\d.]+)*)\s*$")
SESSION = re.compile(r"^# session started (.+?)\s+at step ([\d,]+)")


def parse(path):
    """Every sample block, plus where one session ended and the next began."""
    rows, marks, cur = [], [], None
    for ln in open(path):
        m = SESSION.match(ln)
        if m:
            marks.append({"step": int(m.group(2).replace(",", "")),
                          "when": m.group(1)})
            continue
        m = HEAD.match(ln)
        if m:
            if cur:
                rows.append(cur)
            cur = {"step": int(m.group(1).replace(",", "")),
                   "chars": float(m.group(2)), "min": int(m.group(3)),
                   "experts": int(m.group(4)) if m.group(4) else None}
            continue
        if cur is None:
            continue
        m = LOSS.match(ln)
        if m:
            cur["loss"] = float(m.group(1))
            cur["se"] = float(m.group(2) or 0.0)
            continue
        m = DOMS.match(ln)
        if m and "loss" in cur and "dom" not in cur:
            parts = m.group(1).split()
            cur["dom"] = {parts[i]: float(parts[i + 1])
                          for i in range(0, len(parts) - 1, 2)}
    if cur:
        rows.append(cur)
    return [r for r in rows if "loss" in r], marks


def draw(rows, marks, out, split=None):
    names = sorted({k for r in rows for k in r.get("dom", {})})
    x = [r["step"] for r in rows]

    fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 2, 1.4]})
    fig.suptitle("mini-AGI — held-out loss while reading", y=0.995,
                 fontsize=13)

    # where the reading regime changed, so the two halves are not confused
    def band(a):
        if split is not None:
            a.axvline(split, color="#c0392b", lw=1.2, ls="--", zorder=1)

    # 1. per domain
    for n in names:
        y = [r.get("dom", {}).get(n) for r in rows]
        ax[0].plot(x, y, marker="o", ms=4, lw=1.6, label=n)
    ax[0].set_ylabel("held-out loss per domain")
    ax[0].legend(ncol=len(names), fontsize=9, frameon=False)
    ax[0].grid(alpha=.25)
    band(ax[0])
    if split is not None:
        ax[0].annotate("whole files\n(200k chars per subject)",
                       xy=(split, ax[0].get_ylim()[1]), xytext=(-8, -34),
                       textcoords="offset points", ha="right", fontsize=9,
                       color="#7f8c8d")
        ax[0].annotate("interleaved windows\n(16k chars per subject)",
                       xy=(split, ax[0].get_ylim()[1]), xytext=(8, -34),
                       textcoords="offset points", ha="left", fontsize=9,
                       color="#c0392b")

    # 2. mean, with its error bar
    ax[1].errorbar(x, [r["loss"] for r in rows],
                   yerr=[r.get("se", 0) for r in rows],
                   marker="o", ms=4, lw=1.8, color="#2c3e50", capsize=3)
    ax[1].set_ylabel("mean held-out loss")
    ax[1].grid(alpha=.25)
    band(ax[1])

    # 3. spread - the number the interleaving was meant to move
    sp = [max(r["dom"].values()) - min(r["dom"].values())
          if r.get("dom") else None for r in rows]
    ax[2].plot(x, sp, marker="o", ms=4, lw=1.8, color="#8e44ad")
    ax[2].fill_between(x, 0, sp, color="#8e44ad", alpha=.12)
    ax[2].set_ylabel("spread across domains\n(worst - best)")
    ax[2].grid(alpha=.25)
    band(ax[2])

    # 4. the pool, which grows and prunes itself as it goes
    ex = [r.get("experts") for r in rows]
    if any(e is not None for e in ex):
        ax[3].step(x, ex, where="post", lw=1.8, color="#16a085")
        ax[3].set_ylabel("experts")
        ax[3].grid(alpha=.25)
        band(ax[3])
    ax[3].set_xlabel("step (cumulative across sessions)")

    for m in marks:
        ax[3].axvline(m["step"], color="#bdc3c7", lw=1, zorder=0)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({len(rows)} samples, steps {x[0]:,}-{x[-1]:,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/samples.txt")
    ap.add_argument("--out", default="runs/training.png")
    ap.add_argument("--split", type=int, default=None,
                    help="step at which the reading regime changed")
    a = ap.parse_args()
    rows, marks = parse(a.log)
    if not rows:
        print(f"no samples in {a.log}", file=sys.stderr)
        return 1
    draw(rows, marks, a.out, a.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
