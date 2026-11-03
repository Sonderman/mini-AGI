#!/usr/bin/env python3
"""
The pool at a glance, redrawn every time a sample is written.

One figure, six questions:

    who is doing the work      routing share over the last `survival_chars`
    who is close to dying      staleness as a fraction of the window prune
                               deletes on, with both thresholds marked
    how concentrated is it     cumulative share against rank
    does the gate mean use     it does not, and this is the panel that says so
    how has the pool moved     size and dying share over the run
    are new experts earning    usage against birth order

Reads the checkpoint and runs/expert_history.jsonl. No torch, no GPU - it is
meant to run while training owns the card.

    python3 tools/plot_experts.py --weights weights --out runs/experts.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

INK, MUTED, GRID = "#1b1b1f", "#6b6b76", "#e4e4ea"
WORK, WARM, COOL, DEAD = "#2ca25f", "#d9a326", "#2d6ec2", "#c2492d"

# 0 = nowhere near the line, 1 = deleted. Starts at a saturated blue rather
# than at white, because the overwhelming majority of experts sit at 0 and a
# map that fades to white there draws nothing at all.
DYING_MAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "dying", [(0.0, COOL), (0.45, "#8f9bb3"), (0.75, WARM), (1.0, DEAD)])


def _plural(k, one, many):
    return f"{k} {one if k == 1 else many}"


def cfg(root):
    """survival_chars, dying_at and chunk, with defaults that do not lie."""
    out = {"survival_chars": 0, "dying_at": 0.75, "chunk": 1536, "top_k": 8,
           "resident": 32}
    try:
        import yaml
        c = yaml.safe_load(open(os.path.join(root, "config.yaml")))
        out["survival_chars"] = int(str(c["prune"]["survival_chars"]).replace("_", ""))
        out["dying_at"] = float(c["prune"].get("dying_at", 0.75))
        out["chunk"] = int(c["training"]["chunk"])
        out["top_k"] = int(c["pool"].get("top_k", 8))
        out["resident"] = int(c["pool"].get("resident", 32))
    except Exception:
        pass
    return out


def history(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("use") is not None:
                    rows.append(d)
    except OSError:
        pass
    return rows


def window_use(rows, now, chars_back):
    """
    Routing hits per expert over the last `chars_back` characters.

    Matched by uid, not by position: pruning renumbers everything, so index i
    in an older row is a different expert.

    There is NO fallback to lifetime usage. An earlier version waited until
    the history spanned the full window and only then switched, which meant
    the panel silently showed a different quantity under the same label for
    the first stretch of a run and then changed completely in one step. It
    windows against the oldest row it has and reports the span it actually
    got, so the picture is the same kind of thing on the first day as on the
    thirtieth - just over a shorter reach.

    Returns (hits, span_in_millions).
    """
    cur_uid = now.get("uid")
    cur_use = np.asarray(now["use"], float)
    if not cur_uid:
        return np.zeros_like(cur_use), 0.0
    have = [r for r in rows if r.get("uid")]
    if not have:
        return np.zeros_like(cur_use), 0.0
    target = now["chars"] - chars_back
    older = [r for r in have if r["chars"] <= target]
    # The LATEST of them by character count, not the last one written. The
    # log is append-only and the counter is not monotonic across it: a run
    # resumed from an earlier checkpoint rewinds it, and a run pointed at a
    # different weights directory restarts it from zero while still writing
    # here. Taking the last line instead picked whichever of those landed
    # furthest down the file, which silently widened the window to the whole
    # run - the panel said 100M and showed 508M.
    past = max(older, key=lambda r: r["chars"]) if older else have[0]
    span = (now["chars"] - past["chars"]) / 1e6
    was = dict(zip(past["uid"], past["use"]))
    out = np.array([u - was.get(q, 0.0) for q, u in zip(cur_uid, cur_use)])
    return np.clip(out, 0, None), span


def dying(now, cfg_, mask_young=True):
    """
    Staleness as a share of the window prune deletes on.

    `mask_young` reads 0 for anything still inside its first survival window,
    which is what the RULE does - prune never touches those, so their distance
    to the line is not a live number. For a GRAPH it is the wrong default: for
    the whole first window of a run every expert is young, so the panel shows
    nothing at all while the staleness is quietly accumulating underneath.
    Measured at 13,524 steps into a fresh run, with everything masked to zero:
    the stalest expert had already gone 41.5M characters unasked out of a 100M
    window. Pass False to see that, and colour by `young` to show which of it
    prune may act on.
    """
    n = len(now["gate"])
    seg, step = float(now.get("segments") or 0), float(now.get("step") or 0)
    surv = cfg_["survival_chars"] / max(cfg_["chunk"], 1)
    if not (n and seg > 0 and step > 0 and surv > 0):
        return np.zeros(n), np.zeros(n, bool)
    window = surv * (seg / step)
    seen = now.get("last_seen") or now.get("since") or [seg] * n
    born = np.asarray((now.get("born") or [0] * n)[:n], float)
    # Clamped to the expert's own age: `last_seen` is 0 for a newborn, so
    # unclamped a brand new expert reads as having been ignored for the entire
    # run - a bar above the delete line that vanishes the moment it is first
    # admitted. See PagedPool.dying for the same clamp.
    age_seg = np.maximum(step - born, 0.0) * (seg / max(step, 1.0))
    idle = np.minimum(np.maximum(seg - np.asarray(seen[:n], float), 0.0), age_seg)
    young = (step - born) < surv
    frac = idle / max(window, 1e-9)
    return (np.where(young, 0.0, frac) if mask_young else frac), young


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--history", default="runs/expert_history.jsonl")
    ap.add_argument("--out", default="runs/experts.png")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    C = cfg(root)

    rows = history(a.history)
    try:
        man = json.load(open(os.path.join(a.weights, "manifest.json")))
    except OSError:
        print(f"  no checkpoint at {a.weights}")
        return 1
    t = man.get("telemetry") or {}
    if not t.get("gate"):
        print("  checkpoint carries no pool telemetry")
        return 1
    now = dict(t)
    now["chars"] = man.get("read_chars", 0)
    now["step"] = man.get("step", 0)

    g = np.abs(np.asarray(now["gate"], float))
    n = len(g)
    w, span = window_use(rows, now, C["survival_chars"])
    share = w / max(w.sum(), 1e-9)
    d, young = dying(now, C)
    born = np.asarray((now.get("born") or [0] * n)[:n], float)
    order = np.argsort(-share)

    fig = plt.figure(figsize=(15.5, 10.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.26,
                          left=0.055, right=0.975, top=0.875, bottom=0.075)
    want_m = C["survival_chars"] / 1e6
    win_lbl = (f"the last {span:.0f}M characters" if span >= want_m * 0.98
               else f"the last {span:.0f}M characters "
                    f"(all there is; the window is {want_m:.0f}M)")

    def frame(ax, title, sub=None):
        ax.set_title(title, fontsize=10.5, loc="left", color=INK,
                     fontweight="bold", pad=9)
        if sub:
            ax.set_xlabel(sub, fontsize=8, color=MUTED, labelpad=6, loc="left")
        ax.grid(alpha=.3, color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------------------------------------- 1. who does the work
    ax = fig.add_subplot(gs[0, 0])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=matplotlib.colors.Normalize(0, max(g.max(), 1e-9)))
    ax.bar(np.arange(n), 100 * share[order], width=1.0,
           color=plt.get_cmap("viridis")(g[order] / max(g.max(), 1e-9)))
    top = int(np.searchsorted(np.cumsum(share[order]), 0.99)) + 1
    ax.axvline(top - 0.5, color=DEAD, ls="--", lw=1.2)
    ax.text(top + 1, ax.get_ylim()[1] * 0.85,
            f" {top} experts\n carry 99%", fontsize=8, color=DEAD)
    frame(ax, "Who is doing the work",
          f"routing share over {win_lbl}, colour is the gate")
    ax.set_ylabel("% of all routing", fontsize=8.5)
    cb = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("gate", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # -------------------------------------------- 2. who is close to dying
    ax = fig.add_subplot(gs[0, 1])
    o2 = np.argsort(-d)
    cols = [DEAD if x >= 1.0 else WARM if x >= C["dying_at"] else WORK
            for x in d[o2]]
    cols = [COOL if y else c for c, y in zip(cols, young[o2])]
    ax.bar(np.arange(n), d[o2], width=1.0, color=cols)
    ax.axhline(C["dying_at"], color=WARM, ls="--", lw=1.2)
    ax.axhline(1.0, color=DEAD, ls="--", lw=1.2)
    # Right-hand side: the bars are sorted descending, so the left is where
    # they are tallest and a label there gets buried under them.
    ax.text(n * 0.98, C["dying_at"] + 0.02, "counts as dying", fontsize=7.5,
            color=WARM, ha="right")
    ax.text(n * 0.98, 1.02, "deleted", fontsize=7.5, color=DEAD, ha="right")
    ax.set_ylim(0, max(1.25, float(d.max()) * 1.1))
    frame(ax, f"How close to being deleted  "
              f"({int((d >= C['dying_at']).sum())} dying, {int(young.sum())} on trial)",
          "time since anything asked for it, as a share of the window")
    ax.set_ylabel("share of the survival window", fontsize=8.5)

    # ------------------------------------------------- 3. concentration
    ax = fig.add_subplot(gs[0, 2])
    cum = 100 * np.cumsum(share[order])
    ax.plot(np.arange(1, n + 1), cum, lw=2, color=WORK)
    ax.plot([1, n], [100 / n, 100], color=MUTED, ls="--", lw=1,
            label="if every expert carried the same")
    for q, c in ((50, COOL), (90, WARM), (99, DEAD)):
        k = int(np.searchsorted(cum, q)) + 1
        ax.plot([k], [q], "o", ms=5, color=c)
        ax.annotate(f"{q}% from {k}", (k, q), textcoords="offset points",
                    xytext=(8, -9), fontsize=8, color=c)
    frame(ax, "How concentrated the routing is",
          f"{n} experts, {C['top_k']} chosen per character, {C['resident']} on the card")
    ax.set_xlabel("experts, busiest first", fontsize=8.5)
    ax.set_ylabel("% of routing covered", fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # ------------------------------------ 4. the gate does not mean usage
    ax = fig.add_subplot(gs[1, 0])
    # YlOrRd put every point on its palest yellow and they vanished against
    # white: almost nothing is ever dying, so almost every point sits at 0 and
    # 0 is the lightest end of that map. The colour runs from the palette's
    # blue at "nothing wants it yet, but it is not close to the line" up
    # through amber to red, so the common case is visible and the rare case
    # still reads as alarm. A thin dark edge keeps single points legible where
    # they overlap.
    sc4 = ax.scatter(np.maximum(share * 100, 1e-4), np.maximum(g, 1e-4),
                     s=30, c=d, cmap=DYING_MAP,
                     norm=matplotlib.colors.Normalize(0, 1.0),
                     edgecolor=INK, linewidth=0.35, alpha=.92, zorder=3)
    cb4 = plt.colorbar(sc4, ax=ax, fraction=0.045, pad=0.02)
    cb4.set_label("share of the window", fontsize=8)
    cb4.ax.tick_params(labelsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    live = share > 0
    if live.sum() > 3:
        r = np.corrcoef(np.log(share[live] + 1e-12), np.log(g[live] + 1e-12))[0, 1]
        ax.text(0.03, 0.94, f"correlation {r:+.2f}", transform=ax.transAxes,
                fontsize=9, color=INK, fontweight="bold")
    frame(ax, "Gate against how much it is actually used",
          "blue is freshly wanted, red is at the delete line. A gate says how "
          "loudly an\nexpert speaks when chosen - a different question from "
          "whether it is chosen again")
    ax.set_xlabel("% of routing", fontsize=8.5)
    ax.set_ylabel("gate", fontsize=8.5)

    # --------------------------------------------- 5. the pool over time
    ax = fig.add_subplot(gs[1, 1])
    if rows:
        px = np.array([r["chars"] / 1e6 for r in rows])
        pe = np.array([len(r["gate"]) for r in rows], float)
        keep = np.r_[True, np.diff(px) >= 0]
        ax.plot(px[keep], pe[keep], lw=1.8, color=WORK)
        ax.set_ylabel("experts in the pool", color=WORK, fontsize=8.5)
        q = ax.twinx()
        dd = []
        for r in rows:
            r2 = dict(r)
            dv, _ = dying(r2, C)
            dd.append(100 * float((dv >= C["dying_at"]).mean()) if len(dv) else 0.0)
        q.plot(px[keep], np.array(dd)[keep], lw=1, color=WARM)
        q.set_ylabel("% dying", color=WARM, fontsize=8)
        q.tick_params(labelsize=7)
        for s in ("top",):
            q.spines[s].set_visible(False)
    frame(ax, "The pool over the run")
    ax.set_xlabel("million characters read", fontsize=8.5)

    # ------------------------------------- 6. are the new ones earning
    ax = fig.add_subplot(gs[1, 2])
    rank = np.argsort(np.argsort(born))
    ax.scatter(rank[~young], 100 * share[~young], s=24, color=WORK,
               edgecolor="none", alpha=.9, label="past its trial")
    if young.any():
        ax.scatter(rank[young], 100 * share[young], s=34, color=COOL,
                   edgecolor="none", alpha=.95, label="still on trial")
    founders = int((born <= born.min()).sum())
    if 0 < founders < n:
        ax.axvline(founders - 0.5, color=MUTED, ls=":", lw=1)
        ax.text(founders + 0.5, ax.get_ylim()[1] * 0.9,
                f" {founders} founders", fontsize=7.5, color=MUTED)
    frame(ax, "Are the newer experts earning their place?",
          "a newcomer with no share and no trial left is what prune removes")
    ax.set_xlabel("experts in birth order", fontsize=8.5)
    ax.set_ylabel("% of routing", fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    busiest = 100 * share[order[0]] if n else 0.0
    fig.suptitle(f"the pool  ·  {n} experts  ·  step {now['step']:,}  ·  "
                 f"{now['chars'] / 1e6:.0f}M characters read",
                 x=0.055, y=0.962, ha="left", fontsize=14.5, color=INK,
                 fontweight="bold")
    fig.text(0.055, 0.918,
             f"{top} of {n} experts carry 99% of the routing, the busiest takes "
             f"{busiest:.1f}%.  " + _plural(int((d >= C['dying_at']).sum()), "is dying",
                                            "are dying") + ", " +
             _plural(int(young.sum()), "is", "are") + " still on trial.  "
             f"Usage measured over {win_lbl}.",
             fontsize=9.5, color=MUTED, ha="left")
    fig.savefig(a.out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"  wrote {a.out}  ({n} experts, {top} carry 99%, "
          f"{int((d >= C['dying_at']).sum())} dying)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
