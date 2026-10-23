#!/usr/bin/env python3
"""
Where this model sits on the data-scaling curve, against published models.

    python3 tools/plot_scaling.py

Two panels. The left one puts the run's held-out bits-per-byte on the same
axes as every model whose BPB has been published, so the comparison is a
measurement rather than an assertion. The right one breaks the exponent out
per subject, because the aggregate hides that chess and wikipedia are on
completely different curves.

WHY BITS PER BYTE. It is the only loss that survives a change of tokenizer:
nats per token depends on how the text was cut up, and this model does not cut
it up at all. Models without a published BPB are left off rather than
converted, since converting requires knowing their tokenizer's bytes per token
on the same held-out set, and that is rarely reported.

ONE HELD-OUT SET PER COLOUR. Everything purple is scored on the PG19 test
split - this model's diamonds and the published models' circles - so it
shares a vertical axis and the comparison is a measurement rather than an
approximation. A model belongs on this figure only if it has a published
PG19 bits-per-byte: a BPB on a different held-out set is a different number
however the units are spelled. Everything green is enwik9 - this model's
squares and the published scores whose training data is known - and
compares with nothing else here. Filled marks are this model, hollow ones
published models.

This used to be a script in a scratchpad, which is how it nearly went missing.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import plot_progress as PP                                       # noqa: E402

LN2 = math.log(2)
INK, MUT, GRID = "#1b1b1f", "#6b6b76", "#e4e4ea"
VAL, DEAD, WARM, GRN, PUR = "#2b7bba", "#c2492d", "#d9a326", "#2ca25f", "#7b5ea7"

DOMS = ["arithmetic", "chat", "chat_hermes", "chess", "code", "reasoning",
        "stories", "wikipedia"]

E9_HIST = os.path.join(ROOT, "runs/scaling/enwik9_history.jsonl")
E9_OUT = os.path.join(ROOT, "runs/enwik9_measured.json")

# HOW FAR APART MEASUREMENTS MUST BE TO GIVE A RATE OF THEIR OWN: the last
# must have read this many times what the first had. Two enwik9
# measurements 0.4% apart - 2.320B and 2.329B characters read - differed by
# 0.6%, which put alpha at 1.5 and the line at 0.20 BPB a corpus pass on:
# the model's wobble from one day's reading to the next, not a rate. At a
# typical alpha of 0.15, a quarter more reading moves the score about 3%,
# five times that wobble. Until then a set carries on at the rate it would
# borrow with a single measurement, and says so.
OWN_SPAN = 1.25

# THE SQUARES ARE SPACED. Measurements a few days apart sit a hair apart on
# an axis that runs out to terabytes, and drawn as they came, their squares
# and values piled into one smudge. The first is always drawn, and each later
# one only once it stands this far clear of the last one drawn - in points,
# the width of the latest square and a little. Every measurement still counts
# in the rate the dotted line is carried on at.
ROOM = 14.0

# Published bits-per-byte. Subword token counts converted at 4 bytes/token,
# which is the standard approximation for English and is stated on the figure.
# MambaByte by its best model, the 972M, at the 75B bytes it had read when it
# scored this - the README's PG19 table has the same four.
BYTE = [("PerceiverAR-248M", 80e9, 1.104), ("Transformer-320M", 80e9, 1.057),
        ("MegaByte-1.02B", 80e9, 1.000), ("MambaByte-972M", 75e9, 0.883)]
MAMBA = next(b for n, b, _ in BYTE if n.startswith("MambaByte"))


def model_size(weights=None):
    """
    What the model is right now, counted from its files rather than a constant
    - minagi.store.count_params, the same count the README's table is made of.

    TWO NUMBERS, and the second is the one this chart should be read against.
    N_total counts every expert, most of which are on disk. What a byte
    actually costs is the trunk plus top_k of the RESIDENT experts, applied
    once per recurrent step - so the dense model that costs the same per byte
    is far smaller than N_total, and that is what the other points here are.
    """
    from minagi.store import count_params
    try:
        p = count_params(weights or os.path.join(ROOT, "weights"))
    except (OSError, KeyError, ValueError):
        return None
    return {"experts": p["n_experts"], "n_total": p["total"],
            "dense_equiv": p["dense_equiv"]}  # 6ND matches at N = active/byte


def fit(x, y):
    """
    Slope AND intercept, so the line is anchored by every point rather than
    by wherever the run happens to be standing. Returns (A, alpha, r2) for
    L = A * D^-alpha.
    """
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)
    r2 = 1 - np.sum((ly - (a * lx + b)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return math.exp(b), -a, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(ROOT, "runs/samples.txt"))
    ap.add_argument("--from-mb", type=float, default=20.0,
                    help="megabytes of reading before a point counts. The "
                         "warmup is not on the power law and including it "
                         "flattens the fit")
    ap.add_argument("--recent-mb", type=float, default=140.0,
                    help="a trailing window reported alongside the fit")
    a = ap.parse_args()

    rows = [r for r in PP.parse(a.log) if r.get("val")]
    if not rows:
        raise SystemExit(f"no evaluations in {a.log}")
    D = np.array([r["chars"] * 1e6 for r in rows], float)     # bytes
    L = np.array([r["val"] for r in rows], float)             # nats
    D0, L0 = D[-1], L[-1]

    keep = D >= a.from_mb * 1e6
    A, al, r2 = fit(D[keep], L[keep])

    # HOW MUCH alpha DEPENDS ON WHERE YOU START. The band used to be the
    # trailing window against the whole run, which was fine while the recent
    # trend was the steeper of the two and collapsed to nothing the moment it
    # was not. The spread across every reasonable window is the honest
    # quantity, and it does not depend on one window staying flattering.
    wins = {}
    for lo in (20, 40, 60, 80, 100, 150, 200):
        m = D >= lo * 1e6
        if m.sum() > 30:
            wins[f"from {lo}M"] = fit(D[m], L[m])[1]
    for w in (100, 140):
        m = D >= D0 - w * 1e6
        if m.sum() > 30:
            wins[f"last {w}M"] = fit(D[m], L[m])[1]
    loa, hia = min(wins.values()), max(wins.values())

    trend = {"AGGREGATE": {"A": A, "alpha": al, "r2": r2, "n": int(keep.sum())}}
    for d in DOMS:
        y = np.array([r["dom"].get(d, np.nan) for r in rows], float)
        ok = keep & np.isfinite(y)
        if ok.sum() > 8:
            dA, dal, dr2 = fit(D[ok], y[ok])
            trend[d] = {"A": dA, "alpha": dal, "r2": dr2, "n": int(ok.sum())}
    trend["meta"] = {"D_now": float(D0), "L_now": float(L0),
                     "experts": rows[-1].get("experts"),
                     "band": [loa, hia],
                     "windows": {k: float(v) for k, v in wins.items()}}
    # one pass of the corpus, as the log reports it: the projections stop
    # there, since nothing on this chart says what a second pass is worth
    END = float(rows[-1].get("corpus") or 7874.0) * 1e6
    e9 = enwik9_points()
    E = enwik9_rate(e9, D, rows)
    if E:
        E["end_D"] = END
        E["end_bpb"] = (E["bpb"] * (END / E["D"]) ** (-E["alpha"])
                        if E["alpha"] and E["alpha"] > 0 else None)
        trend["enwik9"] = E
    out_dir = os.path.join(ROOT, "runs/scaling")
    os.makedirs(out_dir, exist_ok=True)

    print(f"  {len(rows)} evaluations, {D[0]/1e6:.1f}M -> {D0/1e6:.1f}M bytes")
    print(f"  aggregate  alpha {al:.3f}  R2 {r2:.3f}  on {keep.sum()} points")
    print(f"  window spread  alpha {loa:.3f} to {hia:.3f}")
    for k, v in wins.items():
        print(f"     {k:<10} {v:.3f}")
    print(f"  now  {L0:.4f} nats = {L0/LN2:.4f} BPB   {rows[-1].get('experts')} experts")

    # The PG19 measurements, if tools/eval_pg19.py has been run. Read rather
    # than typed in, so the figure cannot drift from the numbers it plots.
    pgs = pg19_points()
    for q in pgs:
        print(f"  PG19 at {q['chars_read']/1e6:,.0f}M characters: {q['bpb']:.4f} BPB")
    size = model_size()
    if size:
        print(f"  {size['experts']} experts, N_total {size['n_total']/1e6:.0f}M, "
              f"dense-equivalent {size['dense_equiv']/1e6:.0f}M per byte")
    for d in e9:
        print(f"  enwik9 at {d['chars']/1e6:,.0f}M characters: {d['bpb']:.4f} BPB")
    if E and E["alpha"]:
        print(f"  enwik9 carried on at alpha {E['alpha']:.3f} ({E['how']})")
    known = enwik9_published() if e9 else []
    for name, seen, bpb, _ in known:
        print(f"  {name} on enwik9: {bpb:.2f} BPB after {seen/1e12:,.0f} TB")
    draw(D, L / LN2, trend, rows, size, pgs, e9, END, known)
    # after the chart: which enwik9 square is the last drawn is settled there,
    # and the README describes what the chart shows
    json.dump(trend, open(os.path.join(out_dir, "trend.json"), "w"), indent=1)


def spaced(ax, xs, ys, room=ROOM):
    """Which points to draw: the first, then each that stands `room` points
    clear of the last one drawn, on the axes as they are laid out."""
    if not len(xs):
        return []
    px = ax.transData.transform(np.column_stack([xs, ys]))
    far = room * ax.figure.dpi / 72.0
    keep = [0]
    for i in range(1, len(xs)):
        if math.hypot(*(px[i] - px[keep[-1]])) >= far:
            keep.append(i)
    return keep


def enwik9_published():
    """
    The published enwik9 scores whose models' training data is known, as
    (name, bytes trained on, bits per byte, window): the README's enwik9
    table keeps them, with where each number comes from, and only the ones
    whose makers said how much they trained on can be placed on this axis.
    """
    try:
        import refresh_readme as RR
        return RR.enwik9_known()
    except Exception:                                        # noqa: BLE001
        return []


def enwik9_points():
    """
    This model on enwik9, oldest first: every measurement
    tools/measure_enwik9.py has kept, one per set of weights - the one on the
    most pieces where the same weights were measured twice. The history and
    the latest measurement are both read, so a point the history never saw
    is not lost.
    """
    got = {}
    for path, many in ((E9_HIST, True), (E9_OUT, False)):
        try:
            with open(path) as f:
                ds = ([json.loads(line) for line in f if line.strip()]
                      if many else [json.load(f)])
        except (OSError, ValueError):
            continue
        for d in ds:
            if not (d.get("chars") and d.get("bpb")):
                continue
            key = (d.get("step"), d.get("experts"))
            if key not in got or d.get("pieces", 0) > got[key].get("pieces", 0):
                got[key] = d
    return sorted(got.values(), key=lambda d: d["chars"])


def enwik9_rate(e9, D, rows, back=1e9):
    """
    The rate enwik9 is carried on at past its latest point, and where it came
    from. Once its measurements span OWN_SPAN of reading it is the rate
    enwik9 itself improved at, fitted through all of them. Before that there
    is nothing of enwik9's own to go on, so it borrows the model's Wikipedia
    held-out - enwik9 is Wikipedia - over its last `back` characters read:
    the recent rate, not the whole run's, because this curve has been
    flattening and the whole run would carry the early, steep part forward.
    `span` is how far apart enwik9's measurements are, last over first.
    """
    if not e9:
        return None
    De = np.array([d["chars"] for d in e9], float)
    Le = np.array([d["bpb"] for d in e9], float)
    out = {"n": len(e9), "D": float(De[-1]), "bpb": float(Le[-1]),
           "alpha": None, "how": None, "span": float(De[-1] / De[0]),
           "own_span": OWN_SPAN}
    if len(e9) >= 2 and De[-1] >= OWN_SPAN * De[0]:
        out.update(alpha=fit(De, Le)[1], how="measured")
        return out
    W = np.array([r["dom"].get("wikipedia", np.nan) for r in rows], float)
    m = np.isfinite(W) & (D >= D[-1] - back)
    if m.sum() > 30:
        out.update(alpha=fit(D[m], W[m])[1], how="wikipedia", back=back)
    return out


def pg19_points():
    """
    Every PG19 measurement, oldest first: the history tools/eval_pg19.py
    appends to, or the one record it wrote before it kept a history.
    """
    out = []
    try:
        with open(os.path.join(ROOT, "runs/scaling/pg19_history.jsonl")) as f:
            out = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        out = []
    if not out:
        try:
            out = [json.load(open(os.path.join(ROOT, "runs/scaling/pg19.json")))]
        except (OSError, ValueError):
            out = []
    out = [q for q in out if q.get("chars_read") and q.get("bpb")]
    return sorted(out, key=lambda q: q["chars_read"])


def draw(D, Lb, T, rows, size=None, pgs=None, e9=None, END=7.874e9,
         known=None):
    M = T["meta"]
    A, al, r2 = T["AGGREGATE"]["A"], T["AGGREGATE"]["alpha"], T["AGGREGATE"]["r2"]
    loa, hia = M["band"]
    D0 = D[-1]
    Ltr = A * D0 ** (-al) / LN2            # the fit at today, not the point
    pgs = pgs or []
    pg = pgs[-1] if pgs else None
    pal = None                             # PG19's own exponent, once measured

    # THE MAIN PANEL GETS THE WIDTH. It runs out to the published enwik9
    # models whose training data is known - terabytes, where everything else
    # is gigabytes - and the side panel, the less important of the two, is
    # narrowed so that the main one keeps most of its scale.
    known = known or []
    XMAX = max([2e11] + [1.7 * k[1] for k in known])
    fig = plt.figure(figsize=(17.3, 8.0))
    fig.patch.set_facecolor("white")
    wide = XMAX > 2e11
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1] if wide else [1.95, 1],
                          wspace=.12 if wide else .17,
                          left=.045, right=.985, top=.92, bottom=.28)
    a = fig.add_subplot(gs[0, 0])
    # laid out first: where the enwik9 squares fall decides which are drawn
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(2.5e6 if wide else 2e6, XMAX); a.set_ylim(0.40, 4.4)

    a.plot(D, Lb, lw=1.7, color=VAL, zorder=4, alpha=.9)
    xf = np.logspace(math.log10(2e7), math.log10(D0), 80)
    a.plot(xf, A * xf ** (-al) / LN2, "-", lw=2.6, color=DEAD, zorder=6)

    xx = np.logspace(math.log10(D0), math.log10(END), 300)
    a.fill_between(xx, Ltr * (xx / D0) ** (-hia), Ltr * (xx / D0) ** (-loa),
                   color=DEAD, alpha=.11, zorder=1)
    a.plot(xx, Ltr * (xx / D0) ** (-al), "--", lw=2.0, color=DEAD, zorder=5,
           dashes=(5, 3))
    a.plot(xx, Ltr * (xx / D0) ** (-hia), ":", lw=1.0, color=DEAD, zorder=3)
    a.plot(xx, Ltr * (xx / D0) ** (-loa), ":", lw=1.0, color=DEAD, zorder=3)
    a.plot([END], [Ltr * (END / D0) ** (-al)], "|", ms=15, color=DEAD,
           mew=2.2, zorder=6)
    a.annotate("projection stops at\none corpus pass",
               (END, Ltr * (END / D0) ** (-al)), xytext=(10, 4),
               textcoords="offset points", fontsize=8.5, color=DEAD,
               va="bottom", ha="left", style="italic")

    # PG19, so purple like this model's diamonds; hollow, because they are
    # not this model - as the published enwik9 score is green and hollow
    for _, b_, bpb in BYTE:
        a.plot(b_, bpb, "o", ms=10, color="white", mec=PUR, mew=2.2, zorder=7)
    # Right of the three points when the axis runs on past them, where
    # nothing else is; on the shorter axis, in its empty upper right.
    a.annotate("MegaByte-1.02B  1.000\nTransformer-320M 1.057\n"
               "PerceiverAR-248M 1.104",
               (80e9, 1.05), xytext=(1.4e11, 1.05) if wide else (2.2e10, 3.2),
               fontsize=9, color=PUR, va="center",
               arrowprops=dict(arrowstyle="-", color=PUR, lw=.9, alpha=.6))
    # below and right of its point on the longer axis. On the shorter one,
    # right-aligned in axes fraction: in data coordinates it ran off the
    # right edge and landed on the other panel's tick labels.
    a.annotate("MambaByte-972M  0.883",
               (MAMBA, 0.883),
               **(dict(xytext=(1.1e11, 0.79), ha="left", va="center") if wide
                  else dict(xytext=(.985, .03), textcoords="axes fraction",
                            ha="right", va="bottom")),
               fontsize=9, color=PUR,
               arrowprops=dict(arrowstyle="-", color=PUR, lw=.9, alpha=.6))
    if pg:
        # WHAT THIS IS AND IS NOT. The diamonds are measured: this model scored
        # on the PG19 test split, the same set the other points are on. The
        # dotted line is not. With one diamond it was the exponent fitted on
        # the model's own held-out, carried across and anchored there - an
        # assumption, since nothing said PG19 improves at that rate. With two
        # or more it is the rate PG19 itself improved at between them, fitted
        # through them and continued from the latest: measured, if from few
        # points, and a run whose code changed between them - once they are
        # far enough apart to give a rate at all (OWN_SPAN).
        Dps = np.array([q["chars_read"] for q in pgs], float)
        Lps = np.array([q["bpb"] for q in pgs], float)
        Dp, Lp = Dps[-1], Lps[-1]
        if len(pgs) >= 2 and Dps[-1] >= OWN_SPAN * Dps[0]:
            pal = -float(np.polyfit(np.log(Dps), np.log(Lps), 1)[0])
            a.plot(Dps, Lps, "-", lw=2.0, color=PUR, zorder=9, alpha=.85)
        ex = pal if pal is not None else al
        if ex > 0:
            xp = np.logspace(math.log10(Dp), math.log10(END), 200)
            a.plot(xp, Lp * (xp / Dp) ** (-ex), ":", lw=2.2, color=PUR,
                   zorder=5)
        for k, (d_, l_) in enumerate(zip(Dps, Lps)):
            last = k == len(pgs) - 1
            a.plot(d_, l_, "D", ms=12 if last else 9.5, color=PUR, mec="white",
                   mew=1.6, zorder=10, alpha=1.0 if last else .75)
            if not last:
                a.annotate(f"{l_:.3f}", (d_, l_), fontsize=9.5, color=PUR,
                           va="bottom", ha="right", xytext=(-7, 7),
                           textcoords="offset points")
        # The whole test split is the assumption, so it goes unsaid. A number
        # scored on part of it is not, and says so.
        frac = pg.get("fraction", 1.0)
        note = ("" if frac >= 1.0
                else f"  ({pg['bytes_scored']/1e6:.1f}M bytes, "
                     f"{frac:g} of the split)")
        a.annotate(f"mini-AGI on PG19\n{Lp:.3f} BPB{note}\nout of distribution",
                   (Dp, Lp), fontsize=10.5, color=PUR, fontweight="bold",
                   va="bottom", ha="left", xytext=(10, 6),
                   textcoords="offset points")
        if ex > 0:
            end_v = Lp * (END / Dp) ** (-ex)
            how = ("at the rate PG19 improved\nbetween its measurements"
                   if pal is not None else
                   "if PG19 improves at\nthe same rate")
            a.annotate(f"{end_v:.2f} at one corpus pass\n{how}", (END, end_v),
                       fontsize=9, color=PUR, va="bottom", ha="left",
                       style="italic", xytext=(9, 4), textcoords="offset points")

    # Along the bottom, under the projection band and left of MambaByte's
    # label: higher up, every band of the figure now holds a projection -
    # PG19's, enwik9's and the held-out's.
    a.annotate("", xy=(MAMBA, 0.55), xytext=(D0, 0.55),
               arrowprops=dict(arrowstyle="<|-|>", color=DEAD, lw=1.6, alpha=.9))
    a.text(3.6e9, 0.565, f"{MAMBA/D0:.0f}x more data", fontsize=11,
           color=DEAD, ha="center", va="bottom", fontweight="bold")
    E = T.get("enwik9") or {}
    eal = E.get("alpha")
    if e9:
        # WHAT THIS IS AND IS NOT, as for PG19: the squares are measured, the
        # dotted line is not. It is enwik9's own rate once enwik9 has two
        # measurements, and until then the Wikipedia held-out's recent rate,
        # carried across and anchored at the square - an assumption.
        De = np.array([d["chars"] for d in e9], float)
        Le = np.array([d["bpb"] for d in e9], float)
        # the first square, and each later one once it stands clear of the
        # last one drawn (ROOM); the line runs from the last one drawn
        keep = spaced(a, De, Le)
        Dd, Ld = De[keep], Le[keep]
        if len(Dd) >= 2:
            a.plot(Dd, Ld, "-", lw=2.0, color=GRN, zorder=9, alpha=.85)
        end_e = None
        if eal and eal > 0:
            xe = np.logspace(math.log10(Dd[-1]), math.log10(END), 200)
            a.plot(xe, Ld[-1] * (xe / Dd[-1]) ** (-eal), ":", lw=2.2,
                   color=GRN, zorder=5)
            end_e = Ld[-1] * (END / Dd[-1]) ** (-eal)
            a.annotate(f"{end_e:.2f} at one corpus pass", (END, end_e),
                       fontsize=9, color=GRN, va="center", ha="left",
                       style="italic", xytext=(9, 0), textcoords="offset points")
        for k, (d_, l_) in enumerate(zip(Dd, Ld)):
            last = k == len(Dd) - 1
            a.plot(d_, l_, "s", ms=11 if last else 8.5, color=GRN, mec="white",
                   mew=1.6, zorder=10, alpha=1.0 if last else .75)
            if k == 0 and not last:
                # the first one's value; the last drawn has the headline
                a.annotate(f"{l_:.3f}", (d_, l_), fontsize=9.5, color=GRN,
                           va="top", ha="right", xytext=(-7, -7),
                           textcoords="offset points")
        # what the README describes is what is drawn, and what is not yet
        E.update(D=float(Dd[-1]), bpb=float(Ld[-1]), drawn=len(Dd),
                 since=len(De) - 1 - keep[-1], latest_D=float(De[-1]),
                 latest_bpb=float(Le[-1]))
        if end_e is not None:
            E["end_bpb"] = float(end_e)
        # Up and to the left, on a leader: the gap between the held-out
        # curve and PG19's diamonds is the one free space near the square.
        # Straight above it, PG19's projection comes down through the label
        # on the longer axis, and to the left is the held-out curve's spike.
        a.annotate(f"mini-AGI on enwik9\n{Ld[-1]:.3f} BPB", (Dd[-1], Ld[-1]),
                   xytext=(6e8, 1.78) if wide else (-14, 9),
                   textcoords="data" if wide else "offset points",
                   fontsize=10.5, color=GRN, fontweight="bold",
                   va="center" if wide else "bottom", ha="center",
                   arrowprops=(dict(arrowstyle="-", color=GRN, lw=.9, alpha=.7)
                               if wide else None))
        # the published ones whose training data is known, where they are:
        # hollow, because they are not this model; green, because they are
        # enwik9 and compare with the squares and nothing else here
        for name, seen, bpb, _ in known:
            a.plot(seen, bpb, "o", ms=10, color="white", mec=GRN, mew=2.2,
                   zorder=8)
            times = float(f"{seen / De[-1]:.2g}")
            a.annotate(f"{name} on enwik9  {bpb:.2f}\n~{seen / 1e12:,.0f} TB "
                       f"read, {times:,.0f}x mini-AGI", (seen, bpb),
                       fontsize=9.5, color=GRN, va="center", ha="right",
                       xytext=(-11, 0), textcoords="offset points")
    a.plot(D0, Lb[-1], "o", ms=11, color=DEAD, mec="white", mew=1.5, zorder=9)
    label = f"mini-AGI today\n{Lb[-1]:.3f} BPB, {D0/1e6:.0f}M bytes  "
    a.annotate(label, (D0, Lb[-1]), fontsize=11, color=DEAD, va="top",
               ha="right", fontweight="bold", xytext=(0, -6),
               textcoords="offset points")
    if size:
        # The pool grows while the run reads, so this is read from the
        # manifest every time the figure is drawn rather than written down.
        # ABOVE the point and to the right. Below it runs into the legend,
        # and directly right runs along the projection line.
        # Higher and smaller on the longer axis, where the same text spans
        # more of it and its second line ran into MambaByte's point.
        a.annotate(f"  {size['n_total']/1e6:.0f}M parameters, {size['experts']} experts\n"
                   f"  ~{size['dense_equiv']/1e6:.0f}M dense-equivalent per byte",
                   (D0, Lb[-1]), fontsize=9 if wide else 9.5, color=MUT,
                   va="bottom", ha="left", xytext=(8, 17) if wide else (8, 8),
                   textcoords="offset points")
    a.axvline(END, color=WARM, ls="--", lw=1.3, zorder=2, alpha=.8)
    a.text(END, 4.35, f"  corpus {END / 1e9:.2f}B", fontsize=9, color=WARM,
           rotation=90, va="top")

    # low enough that the optimistic edge of the band is not clipped, which
    # would hide exactly the part of the uncertainty that matters
    # Out to a little past the published PG19 models at 80B, or past the
    # furthest published enwik9 model whose training data is known. The axis
    # once ran to 3e13 for GPT-3 and Chinchilla with nothing out there but
    # them, which squeezed everything else; now the far side holds the legend
    # too.
    a.set_yticks([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
    a.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.tick_params(labelsize=10.5)
    a.set_xlabel("bytes of training data seen   (log)", fontsize=12, color=MUT)
    a.set_ylabel("bits per byte   (log)", fontsize=12, color=MUT)
    a.set_title("Data scaling on PG19 and enwik9: mini-AGI against the "
                "published byte-level models", fontsize=14.5, loc="left",
                fontweight="bold", color=INK)
    a.grid(alpha=.35, color=GRID); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.legend(handles=[
        Line2D([], [], color=VAL, lw=1.7, label="mini-AGI measured"),
        Line2D([], [], color=DEAD, lw=2.6,
               label=fr"power law fit  $\alpha$={al:.3f}, R$^2$={r2:.2f}  (D $\geq$ 20M)"),
        Line2D([], [], color=DEAD, lw=2.0, ls="--",
               label=fr"projection at $\alpha$={al:.2f}; band {loa:.2f}-{hia:.2f} across fitting windows"),
        Line2D([], [], color=PUR, marker="D", ls="", ms=8,
               label="mini-AGI measured on PG19 - out of distribution"),
        Line2D([], [], color=PUR, lw=2.2, ls=":",
               label=("PG19 did not improve between its measurements"
                      if pal is not None and pal <= 0 else
                      fr"PG19's own rate between its {len(pgs)} measurements, $\alpha$={pal:.3f}"
                      if pal is not None else
                      "the same exponent, anchored at that point — assumed, not fitted")),
        Line2D([], [], color=PUR, marker="o", mfc="white", mew=2.2, ls="",
               ms=8, label="published byte-level models on PG19"),
    ] + ([
        Line2D([], [], color=GRN, marker="s", ls="", ms=8,
               label="mini-AGI measured on enwik9 - Wikipedia, in distribution"),
        Line2D([], [], color=GRN, lw=2.2, ls=":",
               label=(fr"enwik9's own rate between its {E['n']} measurements, $\alpha$={eal:.3f}"
                      if E.get("how") == "measured" else
                      fr"its Wikipedia held-out's rate, last 1B chars, $\alpha$={eal:.3f} — assumed")),
    ] if e9 and eal else []) + ([
        Line2D([], [], color=GRN, marker="o", mfc="white", mew=2.2, ls="",
               ms=8, label="published on enwik9, with its training data known"),
    ] if e9 and known else []), fontsize=9, labelspacing=.35, frameon=False,
        loc="upper right" if wide else "lower left")
    own = (pg or {}).get("own_bpb")
    if own is None:
        own = (rows[-1]["val"] / LN2) if rows else float("nan")
    dotted = ("PG19 did not improve between its measurements, so nothing is projected from them."
              if pal is not None and pal <= 0 else
              f"The dotted line continues the rate PG19 itself improved at between the measured points, alpha {pal:.3f}, from the latest one."
              if pal is not None else
              "The dotted line carries this model's own fitted exponent across to PG19 and anchors it at the measured point: it assumes PG19 improves at the same rate,\n"
              "which has not been checked.")
    y0, dy = 0.172, 0.025             # under the side panel's two-line label
    fig.text(0.045, y0,
             ("Everything purple is bits-per-byte on the PG19 test split and compares directly"
              + ("; everything green is enwik9 and compares only with the rest of the green" if e9 else "")
              + ". Filled marks are this model, hollow ones published models; the blue line is this model's own held-out."),
             fontsize=9, color=MUT, va="top")
    fig.text(0.045, y0 - dy,
             "PG19 IS OUT OF DISTRIBUTION FOR THIS MODEL: it has never been trained on any of it.",
             fontsize=10, color=PUR, fontweight="bold", va="top")
    import textwrap
    rest = ("It was trained on a corpus assembled for this project - wikipedia, code, chat, chess, arithmetic, reasoning and "
            "simple stories - which contains no Victorian novels and nothing like PG19's prose. At the latest measurement its "
            f"own held-out sat at {own:.2f} BPB against {'%.2f' % (pg['bpb'] if pg else float('nan'))} here, and much of that "
            "gap is subject matter rather than capability. The published models were trained on long-form books and read 80x "
            "more data, and they evaluate at a context of 8,192 where this model was built with 4,096. " + dotted.replace("\n", " ")
            + ("" if not (e9 and eal) else
               " enwik9 is in distribution: Wikipedia from the 2006 dump the model trains on, on pages it has not read. "
               + (f"Its dotted line continues the rate enwik9 improved at between its {E['n']} measurements, alpha {eal:.3f}."
                  if E.get("how") == "measured" else
                  f"Its dotted line carries the rate this model's Wikipedia held-out fell at over its last 1B characters, alpha {eal:.3f}, "
                  f"across to enwik9 - assumed until enwik9's own measurements are {OWN_SPAN - 1:.0%} apart in reading"
                  + (f" (its {E['n']} so far are {100 * (E['span'] - 1):.1f}% apart)." if E.get("n", 1) >= 2 else ".")))
            + "".join(f" The hollow green circle is {n} on enwik9, {b_:.2f} after about {s_ / 1e12:,.0f} TB"
                      + (f", scored on {w} at a time" if w else "") + "."
                      for n, s_, b_, w in (known if e9 else []))
            + (" The other models in the enwik9 table are not drawn: how much they read was never published."
               if e9 and known else ""))
    fig.text(0.045, y0 - 2 * dy, textwrap.fill(rest, 228),
             fontsize=9, color=MUT, va="top", linespacing=1.5)

    b = fig.add_subplot(gs[0, 1])
    got = [d for d in DOMS if d in T]
    got.sort(key=lambda d: T[d]["alpha"])
    ex = [T[d]["alpha"] for d in got]
    now = [rows[-1]["dom"].get(d, np.nan) / LN2 for d in got]
    y = np.arange(len(got))
    b.barh(y, ex, color=[VAL if e > al else WARM for e in ex], height=.68)
    b.axvline(al, color=DEAD, ls="--", lw=1.4)
    b.text(al + .004, len(got) - .42, f"aggregate {al:.3f}",
           fontsize=9.5 if wide else 10, color=DEAD, fontweight="bold")
    for i, (e, n) in enumerate(zip(ex, now)):
        b.text(e + .006, i, f"{n:.2f} BPB", fontsize=8.5 if wide else 9.5,
               color=MUT, va="center")
    b.set_yticks(y); b.set_yticklabels(got, fontsize=10 if wide else 11)
    b.set_xlim(0, max(.40, max(ex) * 1.25))
    if wide:
        b.set_xticks([0, .1, .2, .3, .4])
    b.tick_params(labelsize=9.5 if wide else 10.5)
    b.set_xlabel(r"power-law exponent $\alpha$" + ("\n" if wide else "   ")
                 + "(steeper = still learning fast)",
                 fontsize=10.5 if wide else 12, color=MUT)
    b.set_title("Which domains still have slope",
                fontsize=12.5 if wide else 14.5, loc="left",
                fontweight="bold", color=INK)
    b.grid(alpha=.35, color=GRID, axis="x"); b.set_axisbelow(True)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)

    for where in ("assets/scaling.png", "runs/scaling/scaling.png",
                  "presentation/scaling.png"):
        path = os.path.join(ROOT, where)
        if os.path.isdir(os.path.dirname(path)):
            plt.savefig(path, dpi=130, facecolor="white")
            print(f"  wrote {where}")


if __name__ == "__main__":
    main()
