#!/usr/bin/env python3
"""
enwik9 on the scaling chart: the measurements behind its squares, the rate
its dotted line is carried on at, and what the README says about them.

What these tests hold it to:
  every measurement is kept, older weights' only in the history, so the
  README's table stays about the model as it stands
  the chart draws one square per set of weights, oldest first
  with one square the line borrows the Wikipedia held-out's recent rate and
  says it is assumed; so it does with two too close together to give a rate
  - 0.4% apart in reading made alpha 1.5 - and says how far apart they are;
  with measurements far enough apart it is enwik9's own rate
  the chart draws either without writing anything into the repo, and the
  README describes whichever it drew
  a published enwik9 score goes on the chart only where its model's training
  data is known, at that much data - the axis runs out to it and the side
  panel gives up width for it
  everything PG19 is purple - this model's diamonds and the published
  models' circles, hollow because they are not this model - and the README
  says so
  the first square is always drawn, and a later one only once it stands
  clear of the last one drawn; the line runs on from the last one drawn, the
  rate still counts every measurement, and the README says which are drawn
"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
import measure_enwik9 as ME                                 # noqa: E402
import plot_scaling as PS                                   # noqa: E402
import refresh_readme as RR                                 # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def point(step, chars, bpb, pieces=2000, experts=300):
    return {"step": step, "chars": chars, "experts": experts, "bpb": bpb,
            "se": 0.006, "pieces": pieces}


tmp = tempfile.mkdtemp(prefix="minagi-enwik9-scaling-")
try:
    out = os.path.join(tmp, "runs", "enwik9_measured.json")
    hist = os.path.join(tmp, "runs", "scaling", "enwik9_history.jsonl")
    os.makedirs(os.path.dirname(out))
    with open(out, "w") as f:
        json.dump(point(7, 2.0e9, 1.35), f)

    ME.record(point(9, 2.3e9, 1.33), True, out=out, hist=hist)
    kept = [json.loads(line) for line in open(hist)]
    check("the first history starts with the measurement already there, then "
          "adds the new one, which the README now quotes",
          [d["step"] for d in kept] == [7, 9]
          and json.load(open(out))["step"] == 9)

    ME.record(point(3, 1.05e9, 1.45, experts=150), False, out=out, hist=hist)
    kept = [json.loads(line) for line in open(hist)]
    check("older weights go into the history only",
          [d["step"] for d in kept] == [7, 9, 3]
          and json.load(open(out))["step"] == 9)

    man = lambda step, chars, experts=300: {"step": step, "read_chars": chars,
                                            "n_experts": experts}
    check("a measurement of the same weights is found in either, on at least "
          "as many pieces, and nothing else is",
          ME.already(man(3, 1.05e9, 150), 2000, out=out, hist=hist) is not None
          and ME.already(man(9, 2.3e9), 2000, out=out, hist=hist) is not None
          and ME.already(man(9, 2.3e9), 4000, out=out, hist=hist) is None
          and ME.already(man(5, 1.5e9), 1000, out=out, hist=hist) is None)

    # the same weights measured twice, the second time on more pieces
    with open(hist, "a") as f:
        f.write(json.dumps(point(3, 1.05e9, 1.44, pieces=4000, experts=150)) + "\n")
    PS.E9_HIST, PS.E9_OUT = hist, out
    pts = PS.enwik9_points()
    check("the chart has one square per set of weights, oldest first, the one "
          "on more pieces where it was measured twice",
          [d["step"] for d in pts] == [3, 7, 9] and pts[0]["pieces"] == 4000,
          f"{[(d['step'], d['pieces']) for d in pts]}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# a run whose Wikipedia held-out falls at 0.20 for its first billion
# characters and at 0.08 for the last billion
D = np.logspace(7, math.log10(3e9), 2000)
wiki = np.where(D < 2e9, 3.0 * (D / 1e7) ** -0.20, 0.0)
w2e9 = 3.0 * (2e9 / 1e7) ** -0.20
wiki = np.where(D < 2e9, wiki, w2e9 * (D / 2e9) ** -0.08)
rows = [{"chars": d / 1e6, "val": 0.9 * w, "experts": 300, "corpus": 8000.0,
         "dom": {"wikipedia": w, "chess": 0.5 * w}} for d, w in zip(D, wiki)]
rows[-1]["corpus"] = 8000.0

one = PS.enwik9_rate([point(9, 2.9e9, 1.33)], D, rows, back=0.9e9)
check("with one square the line borrows the Wikipedia held-out's rate over "
      "the last stretch read, not the whole run's",
      one["how"] == "wikipedia" and abs(one["alpha"] - 0.08) < 0.005,
      f"{one['alpha']:.4f}")
two = PS.enwik9_rate([point(3, 1e9, 1.5), point(9, 2e9, 1.5 * 2 ** -0.1)],
                     D, rows)
check("with two far enough apart it is enwik9's own rate between them",
      two["how"] == "measured" and abs(two["alpha"] - 0.1) < 1e-9
      and two["n"] == 2)
# the run's first two: 2.320B and 2.329B characters, 0.6% apart in score
close = PS.enwik9_rate([point(7, 2.3196e9, 1.3277), point(9, 2.3290e9, 1.3196)],
                       D, rows, back=0.9e9)
check("...but two too close together give no rate of their own: the line "
      "borrows the Wikipedia held-out's, as with one",
      close["how"] == "wikipedia" and abs(close["alpha"] - 0.08) < 0.005
      and close["n"] == 2 and abs(close["span"] - 1.004) < 0.001
      and PS.fit(np.array([2.3196e9, 2.329e9]),
                 np.array([1.3277, 1.3196]))[1] > 1.4,
      f"{close['alpha']:.4f}, span {close['span']:.4f}")

known = RR.enwik9_known()
check("only the published scores whose training data is known are placed: "
      "Llama-3.2-1B, at 9T tokens of 4 bytes",
      [(k[0], k[1], round(k[2], 2)) for k in known]
      == [("Llama-3.2-1B", 3.6e13, 0.68)], f"{known}")

# the chart, drawn into a root with no assets/ - so nothing is written
A, al, r2 = PS.fit(D, np.array([r["val"] for r in rows]))
END = 8.0e9
labels, texts, captions = [], [], []
for name, e9, E in (("one", [point(9, 2.9e9, 1.33)], one),
                    ("two", [point(3, 1e9, 1.5), point(9, 2e9, 1.5 * 2 ** -0.1)], two),
                    ("close", [point(7, 2.3196e9, 1.3277),
                               point(9, 2.3290e9, 1.3196)], close)):
    T = {"AGGREGATE": {"A": A, "alpha": al, "r2": r2},
         "meta": {"band": [0.05, 0.2]},
         "wikipedia": {"alpha": 0.15}, "chess": {"alpha": 0.2},
         "enwik9": dict(E, end_D=END,
                        end_bpb=E["bpb"] * (END / E["D"]) ** -E["alpha"])}
    PS.ROOT, saved = tempfile.mkdtemp(prefix="minagi-chart-"), PS.ROOT
    try:
        PS.draw(D, np.array([r["val"] for r in rows]) / math.log(2), T, rows,
                None, [], e9, END)
        a = plt.gcf().axes[0]
        labels.append([t.get_text() for t in a.get_legend().get_texts()])
        texts.append(" ".join(t.get_text() for t in a.texts))
        captions.append(" ".join(t.get_text().replace("\n", " ")
                                 for t in plt.gcf().texts))
        wrote = os.listdir(PS.ROOT)
    finally:
        shutil.rmtree(PS.ROOT, ignore_errors=True)
        PS.ROOT = saved
        plt.close("all")
    check(f"the chart draws {name} square{'' if name == 'one' else 's'}, "
          "labelled, with the line's end, and writes nothing",
          "mini-AGI on enwik9" in texts[-1] and "at one corpus pass" in texts[-1]
          and not wrote)
# a fifth run of the chart: four measurements, three of them close together
four = [point(7, 2.3196e9, 1.3277), point(9, 2.3290e9, 1.3196),
        point(11, 2.4524e9, 1.3137), point(13, 3.6e9, 1.29)]
E4 = PS.enwik9_rate(four, D, rows, back=0.9e9)
spaced = {}
for name, pts, Ex in (("close", [point(7, 2.3196e9, 1.3277),
                                 point(9, 2.3290e9, 1.3196)], close),
                      ("four", four, E4)):
    T = {"AGGREGATE": {"A": A, "alpha": al, "r2": r2},
         "meta": {"band": [0.05, 0.2]},
         "wikipedia": {"alpha": 0.15}, "chess": {"alpha": 0.2},
         "enwik9": dict(Ex, end_D=END, end_bpb=None)}
    PS.ROOT, saved = tempfile.mkdtemp(prefix="minagi-chart-"), PS.ROOT
    try:
        PS.draw(D, np.array([r["val"] for r in rows]) / math.log(2), T, rows,
                None, [], pts, END)
        a = plt.gcf().axes[0]
        sq = [l for l in a.lines if l.get_marker() == "s"]
        spaced[name] = ([(float(l.get_xdata()[0]), float(l.get_ydata()[0]))
                         for l in sq], T["enwik9"],
                        " ".join(t.get_text() for t in a.texts))
    finally:
        shutil.rmtree(PS.ROOT, ignore_errors=True)
        PS.ROOT = saved
        plt.close("all")
drawn_c, Ec, txt_c = spaced["close"]
drawn_4, E4d, txt_4 = spaced["four"]
check("the first square is always drawn, and one too close to it is not",
      drawn_c == [(2.3196e9, 1.3277)] and "mini-AGI on enwik9\n1.328 BPB"
      in txt_c and "1.320" not in txt_c, f"{drawn_c}")
check("...a later one is, once it stands clear of the last one drawn - and "
      "only the first keeps a small value beside it",
      drawn_4 == [(2.3196e9, 1.3277), (3.6e9, 1.29)]
      and txt_4.count("1.328") == 1 and "1.314" not in txt_4, f"{drawn_4}")
check("...the line runs on from the last one drawn, at the rate every "
      "measurement gives, and the README is told which are drawn",
      Ec["D"] == 2.3196e9 and Ec["bpb"] == 1.3277 and Ec["since"] == 1
      and Ec["latest_bpb"] == 1.3196 and Ec["alpha"] == close["alpha"]
      and abs(Ec["end_bpb"] - 1.3277 * (END / 2.3196e9) ** -close["alpha"])
      < 1e-12 and E4d["since"] == 0 and E4d["drawn"] == 2)

check("...and its legend says which rate the line is carried on at, and "
      "the caption how far apart measurements too close to give one are",
      any("assumed" in t for t in labels[0])
      and any("own rate between its 2 measurements" in t for t in labels[1])
      and any("assumed" in t for t in labels[2])
      and "(its 2 so far are 0.4% apart)" in captions[2]
      and "so far are" not in captions[0])

# everything PG19 purple: this model's diamonds, the published circles
PS.ROOT, saved = tempfile.mkdtemp(prefix="minagi-chart-"), PS.ROOT
try:
    T = {"AGGREGATE": {"A": A, "alpha": al, "r2": r2},
         "meta": {"band": [0.05, 0.2]},
         "wikipedia": {"alpha": 0.15}, "chess": {"alpha": 0.2},
         "enwik9": dict(one, end_D=END,
                        end_bpb=one["bpb"] * (END / one["D"]) ** -one["alpha"])}
    pgs = [{"chars_read": 3.9e8, "bpb": 2.45}, {"chars_read": 1.07e9, "bpb": 2.09}]
    PS.draw(D, np.array([r["val"] for r in rows]) / math.log(2), T, rows,
            None, pgs, [point(9, 2.9e9, 1.33)], END)
    a = plt.gcf().axes[0]
    pur = matplotlib.colors.to_rgba(PS.PUR)
    at = {(float(l.get_xdata()[0]), float(l.get_ydata()[0])): l
          for l in a.lines if len(l.get_xdata()) == 1}
    published = [at.get((b_, y_)) for _, b_, y_ in PS.BYTE]
    diamonds = [at.get((q["chars_read"], q["bpb"])) for q in pgs]
    names = [t for t in a.texts if "MegaByte" in t.get_text()
             or "MambaByte" in t.get_text()]
    legend = {t.get_text(): h for t, h in zip(a.get_legend().get_texts(),
                                             a.get_legend().legend_handles)}
    caption = " ".join(t.get_text() for t in plt.gcf().texts)
finally:
    shutil.rmtree(PS.ROOT, ignore_errors=True)
    PS.ROOT = saved
    plt.close("all")
check("everything PG19 is purple: the published circles hollow, this "
      "model's diamonds filled, the names beside them, and the legend",
      all(l is not None and matplotlib.colors.to_rgba(l.get_mec()) == pur
          and matplotlib.colors.to_rgba(l.get_mfc()) == (1, 1, 1, 1)
          for l in published)
      and all(l is not None and matplotlib.colors.to_rgba(l.get_mfc()) == pur
              for l in diamonds)
      and len(names) == 2
      and all(matplotlib.colors.to_rgba(t.get_color()) == pur for t in names)
      and matplotlib.colors.to_rgba(
          legend["published byte-level models on PG19"].get_mec()) == pur
      and "Everything purple is bits-per-byte on the PG19 test split" in caption
      and "black" not in caption)

PS.ROOT, saved = tempfile.mkdtemp(prefix="minagi-chart-"), PS.ROOT
try:
    T = {"AGGREGATE": {"A": A, "alpha": al, "r2": r2},
         "meta": {"band": [0.05, 0.2]},
         "wikipedia": {"alpha": 0.15}, "chess": {"alpha": 0.2},
         "enwik9": dict(one, end_D=END,
                        end_bpb=one["bpb"] * (END / one["D"]) ** -one["alpha"])}
    PS.draw(D, np.array([r["val"] for r in rows]) / math.log(2), T, rows,
            None, [], [point(9, 2.9e9, 1.33)], END, known)
    main_, side = plt.gcf().axes[:2]
    lo, hi = main_.get_xlim()
    drawn = " ".join(t.get_text() for t in main_.texts)
    legend = [t.get_text() for t in main_.get_legend().get_texts()]
    w_main = main_.get_position().width
    w_side = side.get_position().width
finally:
    shutil.rmtree(PS.ROOT, ignore_errors=True)
    PS.ROOT = saved
    plt.close("all")
check("Llama-3.2-1B is drawn at its 36 TB, with the axis running past it, "
      "labelled with what it read against this model",
      hi >= 3.6e13 * 1.5 and "Llama-3.2-1B on enwik9  0.68" in drawn
      and "~36 TB read, 12,000x mini-AGI" in drawn
      and any("training data known" in t for t in legend), f"{hi:.2g}")
check("...and the side panel gives up width so the main one keeps its scale",
      w_main / w_side > 3, f"{w_main / w_side:.2f}")

# what the README says about it
latest = {"chars": 2320.0, "bpb": 0.86, "total": 8037.0,
          "dom": {"wikipedia": 0.97, "chess": 0.4}}
trend = {"AGGREGATE": {"A": A, "alpha": 0.203, "r2": 0.97},
         "wikipedia": {"alpha": 0.15}, "chess": {"alpha": 0.2}}
pg = {"bpb": 2.091, "books": 100, "bytes_scored": 41_289_001, "window": 4096,
      "own_bpb": 0.94}
readme = "a\n<!-- auto:scaling -->\nold\n<!-- /auto:scaling -->\nb\n"
timing = (1175.0, 24600.0, 2.2e6)
plain, k1 = RR.scaling(readme, latest, trend, pg, timing)
e = dict(one, end_D=END, end_bpb=one["bpb"] * (END / one["D"]) ** -one["alpha"])
said, k2 = RR.scaling(readme, latest, dict(trend, enwik9=e), pg, timing)
m = dict(two, end_D=END, end_bpb=two["bpb"] * (END / two["D"]) ** -two["alpha"])
said2, k3 = RR.scaling(readme, latest, dict(trend, enwik9=m), pg, timing)
check("without enwik9 the README says everything purple is PG19, and "
      "nothing of enwik9",
      k1 and "Everything purple on this chart is **bits-per-byte on the "
      "PG19 test split**" in plain and "enwik9" not in plain)
check("with it, the README says everything purple is PG19 and the squares "
      "enwik9, where the line ends, and that the rate is assumed",
      k2 and "Everything purple on this chart is" in said
      and "the hollow circles the published byte-level models in the PG19 "
      "table above" in said
      and "**enwik9**" in said and f"**{one['bpb']:.3f} BPB**" in said
      and f"**{e['end_bpb']:.2f}**" in said
      and "an assumption until enwik9's own measurements are 25% apart in "
      "what the model has read." in said
      and "![Data scaling on PG19 and enwik9]" in said)
c = dict(close, end_D=END,
         end_bpb=close["bpb"] * (END / close["D"]) ** -close["alpha"])
said_c, k4 = RR.scaling(readme, latest, dict(trend, enwik9=c), pg, timing)
check("...and with two too close together, says how far apart they are",
      k4 and "an assumption until enwik9's own measurements are 25% apart in "
      "what the model has read; its 2 so far are 0.4% apart." in said_c)
said_s, k5 = RR.scaling(readme, latest, dict(trend, enwik9=dict(c, **{
    k: Ec[k] for k in ("D", "bpb", "since", "latest_D", "latest_bpb",
                       "end_bpb")})), pg, timing)
check("...and says which square is drawn, and the latest measurement when "
      "it is not drawn yet",
      k5 and "It scores **1.328 BPB** there after 2.32B characters. A square "
      "is drawn only once it stands clear of the one before it, so the one "
      "measured since is not drawn yet; the latest is 1.320 after 2.33B "
      "characters. The dotted line" in said_s
      and "not drawn yet" not in said_c)
check("...and once enwik9 has its own rate, says that instead",
      k3 and "between its 2 measurements" in said2
      and "an assumption" not in said2)
check("...and names the published model the chart places, and why the rest "
      "are not there",
      "The hollow green circle far to the right is Llama-3.2-1B, the only "
      "model in that table whose training data is published: 0.68 after "
      "about 36 TB" in said
      and "scored on 2,048 tokens, about 8 KB, at a time" in said
      and "how much they read was never published" in said)

# PG19's best effort is the PG19 table's, not the chart's: the chart is
# frozen measurements only
check("the scaling paragraph says nothing of PG19's best effort, which the "
      "PG19 table carries",
      "best effort" not in plain and "best effort" not in said
      and "pgn" not in RR.scaling.__code__.co_varnames)

print("\n  all enwik9 scaling tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
