#!/usr/bin/env python3
"""
Held-out read as blocks spread over the whole set, and checks at one level.

What these tests hold it to:
  a block read alone scores exactly what reading its file from the top scores
  there - which is what makes choosing blocks a choice of text and nothing
  else; reading every block spread is reading the whole set from the top, and
  reading from the top is still there, unchanged
  held-out counts whole blocks only - every chunk full, a file's leftover
  never a block of its own - unless a subject has no whole block at all;
  the old blocks are still there, for measuring what that moved
  n spread blocks are n distinct blocks evenly through all of a subject's
  files, the same ones every time
  the session banner's lines read back as written, and an old check moves by
  its own subjects' share of the difference; the session that switched to
  whole blocks says how far the old blocks sat, and that is a second move
  the plots and the README leave out the larger evaluation sessions ended on
  before the change, move older checks onto the new level subject by subject
  and mark them, and keep everything after the change as written; a log that
  never changed is left as it was, closing evaluations aside, and one from
  before the rate controller keeps every entry; after the switch to whole
  blocks everything before it moves again, on top of the first move
  the README's table is held-out measured on weights/ as it is now when
  there is such a measurement, and the latest check when it is older; and a
  measurement of these weights is never taken twice - the evaluation a run
  stopped cleanly on is used, as is one the tool took before - unless it was
  taken before held-out counted whole blocks only
"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-heldout-")
here = os.getcwd()
try:
    from minagi.create import create                          # noqa: E402
    from minagi.ingest import as_stream                       # noqa: E402
    from minagi.stream import FileReader, FolderEvaluator     # noqa: E402
    from minagi import heldout                                # noqa: E402
    import train as T                                         # noqa: E402
    import plot_progress as PP                                # noqa: E402
    import refresh_readme as RR                               # noqa: E402

    os.chdir(tmp)              # the file-verdict cache lands here, not in runs/
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, _ = T.build_paged(wdir, "cpu", ram_capacity=2)

    # a held-out set of two subjects, files of awkward lengths: one too short
    # to read, one a character past a block - exactly one whole block - a long
    # one with a short tail, and one a block long: a character short of whole
    rng = np.random.default_rng(0)
    words = ["the", "cat", "sat", "on", "a", "mat", "and", "then", "ran",
             "def", "x", "=", "1", "return", "\n", "if", "else", "for"]

    def text(n):
        out = ""
        while len(out) < n:
            out += words[int(rng.integers(len(words)))] + " "
        return out[:n]

    held = os.path.join(tmp, "val")
    for sub, sizes in (("a", (700, 5, 129, 300, 128)), ("b", (1000,))):
        os.makedirs(os.path.join(held, sub))
        for j, n in enumerate(sizes):
            with open(os.path.join(held, sub, f"{sub}{j}.txt"), "w") as f:
                f.write(text(n))

    ev = FolderEvaluator(model, held, 32, 128, "cpu")
    old = FolderEvaluator(model, held, 32, 128, "cpu", whole=False)
    model.eval()

    # ------------------------------------------------- 1. a block is a block
    ref = {}                      # file -> the chunk losses of each block
    with torch.no_grad():
        for name, files in ev.groups.items():
            for f in files:
                data = as_stream(f)
                if len(data) < 8:
                    continue
                r = FileReader(model, data, name, 32, 128, "cpu")
                ls = []
                while not r.done():
                    l = r.step(learn=False)
                    if l is None:
                        break
                    ls.append(float(l))
                ref[f] = [ls[k:k + ev.per] for k in range(0, len(ls), ev.per)]
        worst, n_blocks = 0.0, 0
        for name in old.groups:
            for f, s in old.blocks(name):
                data = as_stream(f)
                got = ev.score_block(data[s:s + ev.block + 1], name)
                want = ref[f][s // ev.block]
                n_blocks += 1
                if len(got) != len(want):
                    worst = float("inf")
                    break
                worst = max([worst] + [abs(a - b) for a, b in zip(got, want)])
    check("a block read alone scores what reading its file from the top "
          "scores there", worst < 1e-5 and n_blocks == sum(len(v) for v in ref.values()),
          f"{n_blocks} blocks, largest difference {worst:.1e}")

    everything = sum(len(v) * ev.per for v in ref.values())
    top = ev.run(everything)
    spread = old.run(everything, spread=True)
    check("reading every block spread is reading the whole set from the top",
          all(abs(top[k] - spread[k]) < 1e-5 for k in ev.groups),
          f"a {spread['a']:.4f} / {top['a']:.4f}, b {spread['b']:.4f} / {top['b']:.4f}")

    # whole blocks only: a block is `per` full chunks, its characters and the
    # target of its last; what is left of a file after that is not a block
    size = {f: len(as_stream(f)) for fs in ev.groups.values() for f in fs}
    whole = {k: [(f, s) for f, s in old.blocks(k) if s + ev.block + 1 <= size[f]]
             for k in ev.groups}
    named = {os.path.basename(f) for k in ev.groups for f, _ in ev.blocks(k)}
    check("held-out counts whole blocks only: every one there is, and nothing "
          "else - not a file's leftover, nor a block a character short",
          all(ev.blocks(k) == whole[k] for k in ev.groups)
          and all(len(ref[f][s // ev.block]) == ev.per
                  for k in ev.groups for f, s in ev.blocks(k))
          and "a2.txt" in named and "a4.txt" not in named
          and sum(map(len, whole.values())) < n_blocks,
          f"{sum(map(len, whole.values()))} of {n_blocks} blocks")
    spread = ev.run(everything, spread=True)
    want = {k: np.mean([l for f, s in whole[k] for l in ref[f][s // ev.block]])
            for k in ev.groups}
    check("...and reading every one of them spread scores their chunks",
          all(abs(spread[k] - want[k]) < 1e-5 for k in ev.groups),
          f"a {spread['a']:.4f} / {want['a']:.4f}")
    short = os.path.join(tmp, "short")
    os.makedirs(os.path.join(short, "c"))
    for j, n in enumerate((60, 100)):
        with open(os.path.join(short, "c", f"c{j}.txt"), "w") as f:
            f.write(text(n))
    sev = FolderEvaluator(model, short, 32, 128, "cpu")
    pieces = FolderEvaluator(model, short, 32, 128, "cpu", whole=False)
    check("...unless a subject has no whole block at all: then it keeps its "
          "pieces",
          sev.blocks("c") == pieces.blocks("c") and len(sev.blocks("c")) == 2
          and "c" in sev.run(8, spread=True))
    flat = {name: [l for f in ev.groups[name] if f in ref for b in ref[f] for l in b]
            for name in ev.groups}
    first7 = ev.run(7)
    check("reading from the top is unchanged: the first chunks, in order",
          all(abs(first7[k] - np.mean(flat[k][:7])) < 1e-5 for k in ev.groups))

    # ----------------------------------------------------- 2. which blocks
    every = ev.blocks("a")
    pick = ev.spread("a", 3)
    n = len(every)
    check("n spread blocks are n distinct blocks evenly through all of them",
          len(set(pick)) == 3
          and [every.index(p) for p in pick] == [int((j + 0.5) * n / 3) for j in range(3)]
          and len({f for f, _ in pick}) > 1,
          f"blocks {[every.index(p) for p in pick]} of {n}")
    check("...every block when there are no more than asked for",
          ev.spread("a", n + 5) == every)
    again = FolderEvaluator(model, held, 32, 128, "cpu").spread("a", 3)
    one, two = ev.run(5, spread=True), ev.run(5, spread=True)
    check("...the same blocks every time, and the same score from them",
          again == pick and all(abs(one[k] - two[k]) < 1e-6 for k in one)
          and len(ev._texts("a", 3)) == 3 and len(ev._texts("b", 3)) == 3)

    # ------------------------------------------------- 3. the banner's lines
    sp, ck, tp = {"a": 0.55, "b": 0.75}, {"a": 0.56, "b": 0.77}, {"a": 0.58, "b": 0.79}
    txt = "# session started\n" + heldout.banner(sp, ck, tp) + "more\n"
    lv = heldout.levels(txt)
    at, delta = heldout.first_shift(txt)
    check("the session banner's lines read back as written",
          len(lv) == 1 and lv[0][1][heldout.SPREAD] == sp
          and lv[0][1][heldout.CHECKS] == ck and lv[0][1][heldout.TOP] == tp
          and at == txt.index("# held-out")
          and all(abs(delta[k] - (tp[k] - sp[k])) < 1e-9 for k in sp))
    v, d = heldout.moved(0.70, {"a": 0.60, "c": 0.90}, delta)
    check("...and an old check moves by its own subjects' share",
          abs(v - (0.70 - delta["a"])) < 1e-9 and abs(d["a"] - (0.60 - delta["a"])) < 1e-9
          and d["c"] == 0.90 and heldout.first_shift("# session started\n") is None)
    # the session that switched to whole blocks: the new level, and the old
    # blocks read beside it on the same weights
    sw, pt = {"a": 0.54, "b": 0.745}, {"a": 0.57, "b": 0.74}
    txt2 = txt + "# session started\n" + heldout.banner(sw, ck, tp, pt) + "more\n"
    at2, pd = heldout.part_shift(txt2)
    check("...the session that switched to whole blocks says how far the old "
          "blocks sat, and only that session",
          heldout.levels(txt2)[1][1][heldout.PARTS] == pt
          and at2 == txt2.index("# held-out", len(txt))
          and all(abs(pd[k] - (pt[k] - sw[k])) < 1e-9 for k in pt)
          and heldout.first_shift(txt2)[0] == at and heldout.part_shift(txt) is None)

    # -------------------------------- 4. the plots and the README read it
    T._PLOTS["on"] = False                 # no plots of the real log, please
    log = os.path.join(tmp, "runs", "samples.txt")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    plast = {"t": 1.0, "n": 20.0, "e": 0.01, "scale": 0.5}

    def write(chars, val, dom, plast=plast, level=None):
        T.write_samples(log, int(chars * 1000), chars * 1e6, 12, [], val, 0.03,
                        dom, 10, 0.5, 1e-4, 100e6, 4096, 4096, None, 1400,
                        1.0, 1.0, plast, 30.0, level=level)

    def session(lines=""):
        with open(log, "a") as f:
            f.write(f"\n\n{'#' * 78}\n# session started 2026-10-17 10:00  "
                    f"at step 1, 0.0M of 100M characters read (0.00%), 10 "
                    f"experts\n{lines}{'#' * 78}\n")

    session()
    write(1.0, 0.70, {"a": 0.60, "b": 0.80})
    write(2.0, 0.69, {"a": 0.59, "b": 0.79})
    write(2.1, 0.68, {"a": 0.585, "b": 0.775}, plast=None)    # it closed
    session(heldout.banner(sp, ck, tp))
    write(3.0, 0.66, {"a": 0.55, "b": 0.77}, level=-0.015)
    write(3.1, 0.655, {"a": 0.55, "b": 0.76}, plast=None)     # and again

    rows = PP.parse(log)
    m = 0.5 * (delta["a"] + delta["b"])
    check("the plots leave out the closing evaluation from before the change",
          [r["chars"] for r in rows] == [1.0, 2.0, 3.0, 3.1])
    check("...move older checks onto the new level, subject by subject",
          abs(rows[0]["val"] - (0.70 - m)) < 1e-6
          and abs(rows[0]["dom"]["a"] - (0.60 - delta["a"])) < 1e-6
          and abs(rows[1]["dom"]["b"] - (0.79 - delta["b"])) < 1e-6
          and rows[0].get("shifted") and rows[1].get("shifted"),
          f"0.7000 -> {rows[0]['val']:.4f}")
    check("...and keep everything after it as written, the closing one too",
          rows[2]["val"] == 0.66 and rows[3]["val"] == 0.655
          and not rows[2].get("shifted") and not rows[3].get("shifted")
          and rows[2]["evidence"] is not None)

    RR.ROOT = tmp
    got = RR.checks()
    check("the README reads the same rounds at the same level",
          [d["chars"] for _, _, d in got] == [1.0, 2.0, 3.0, 3.1]
          and abs(got[0][2]["val"] - rows[0]["val"]) < 1e-6
          and abs(got[0][2]["bpb"] - got[0][2]["val"] / math.log(2)) < 1e-9
          and got[0][2].get("shifted") and not got[2][2].get("shifted")
          and abs(got[1][2]["dom"]["b"] - rows[1]["dom"]["b"]) < 1e-6)

    # held-out counts whole blocks only from the next session: everything
    # before it moves again, by what that session measured
    session(heldout.banner(sw, ck, tp, pt))
    write(4.0, 0.64, {"a": 0.53, "b": 0.735}, level=-0.01)
    rows2 = PP.parse(log)
    mp = 0.5 * (pd["a"] + pd["b"])
    check("after the switch to whole blocks everything before it moves again, "
          "subject by subject, on top of the first move - closing one too",
          [r["chars"] for r in rows2] == [1.0, 2.0, 3.0, 3.1, 4.0]
          and all(abs(rows2[i]["val"] - (rows[i]["val"] - mp)) < 1e-6
                  for i in range(4))
          and all(abs(rows2[i]["dom"][k] - (rows[i]["dom"][k] - pd[k])) < 1e-6
                  for i in range(4) for k in "ab")
          and all(rows2[i].get("shifted") for i in range(4)),
          f"0.7000 -> {rows2[0]['val']:.4f}, 0.6600 -> {rows2[2]['val']:.4f}")
    check("...and keeps the session that switched as written",
          rows2[4]["val"] == 0.64 and rows2[4]["dom"] == {"a": 0.53, "b": 0.735}
          and not rows2[4].get("shifted"))
    got2 = RR.checks()
    check("...the README at the same level",
          [d["chars"] for _, _, d in got2] == [1.0, 2.0, 3.0, 3.1, 4.0]
          and all(abs(got2[i][2]["val"] - rows2[i]["val"]) < 1e-6
                  and all(abs(got2[i][2]["dom"][k] - rows2[i]["dom"][k]) < 1e-6
                          for k in "ab") for i in range(5))
          and all(abs(got2[i][2]["bpb"] - got2[i][2]["val"] / math.log(2)) < 1e-9
                  and got2[i][2].get("shifted") for i in range(4))
          and not got2[4][2].get("shifted"))

    # the README's table: the measurement on weights/ as it is now when there
    # is one, the latest check when it is older
    import json
    os.makedirs(os.path.join(tmp, "weights"), exist_ok=True)
    meas = {"step": 7, "chars": 3.2e6, "experts": 12, "val": 0.61, "se": 0.01,
            "per_subject": {"a": 0.5, "b": 0.72}, "blocks": {"a": 30, "b": 30},
            "block_chars": 4096, "all": False, "whole": True}
    seen = {}
    for step, d in ((7, meas), (8, meas),
                    ("parts", dict(meas, whole=None))):
        with open(os.path.join(tmp, "runs", "heldout_measured.json"), "w") as f:
            json.dump(d, f)
        with open(os.path.join(tmp, "weights", "manifest.json"), "w") as f:
            json.dump({"step": 7 if step == "parts" else step}, f)
        seen[step] = RR.measured(got[-1][2])
    m = seen[7]
    check("the README's table is the measurement on weights/ as it is now, "
          "and the latest check when the measurement is older, or counted "
          "the old blocks",
          m is not None and m["val"] == 0.61 and m["se"] == 0.01
          and m["dom"] == {"a": 0.5, "b": 0.72} and m["chars"] == 3.2
          and abs(m["bpb"] - 0.61 / math.log(2)) < 1e-12
          and m["train"] == got[-1][2]["train"] and seen[8] is None
          and seen["parts"] is None)

    # nothing measured twice: a measurement of these weights already taken,
    # or the evaluation a cleanly stopped run ended on, is used as it is
    import measure_heldout as M
    out = os.path.join(tmp, "runs", "none.json")
    man = {"step": 3100, "read_chars": 3.1e6, "n_experts": 10,
           "cfg": {"block": 4096}}
    for parts in (None, pt):
        os.remove(log)
        session()
        write(1.0, 0.70, {"a": 0.60, "b": 0.80})
        session(heldout.banner(sp, ck, tp, parts))
        write(3.0, 0.66, {"a": 0.55, "b": 0.77}, level=-0.015)
        write(3.1, 0.655, {"a": 0.55, "b": 0.76}, plast=None)     # it closed
        if parts is None:
            before = M.already(man, out=out, log=log)
    d = M.already(man, out=out, log=log)
    check("the evaluation a run stopped cleanly on is used, not measured "
          "again",
          d is not None and d["val"] == 0.655 and d["se"] == 0.03
          and d["per_subject"] == {"a": 0.55, "b": 0.76} and d["chars"] == 3.1e6
          and d.get("source") and set(d["blocks"]) == {"a", "b"}
          and d.get("whole") is True, f"{d}")
    check("...but not one from before held-out counted whole blocks only",
          before is None)
    other = dict(man, step=3101)
    write(3.2, 0.65, {"a": 0.55, "b": 0.75})                  # a check after it
    after_check = M.already(man, out=out, log=log)
    check("...but not when the weights have moved on, or a check came after it",
          M.already(other, out=out, log=log) is None and after_check is None)
    with open(out, "w") as f:
        json.dump(dict(d, all=False), f)
    used = M.already(man, out=out, log=log)
    with open(out, "w") as f:
        json.dump(dict(d, all=False, whole=None), f)
    check("...and a measurement this tool already took of the same weights is "
          "used too - for --all only one that read every block, and none that "
          "counted the old blocks",
          used is not None
          and M.already(man, want_all=True, out=out, log=log) is None
          and M.already(other, out=out, log=log) is None
          and M.already(man, out=out, log=log) is None)

    # a log that never changed: closing evaluations out, nothing moved; and a
    # log from before the controller keeps everything
    os.remove(log)
    write(1.0, 0.70, {"a": 0.60, "b": 0.80}, plast=None)      # no controller yet
    write(2.0, 0.69, {"a": 0.59, "b": 0.79})
    write(2.1, 0.68, {"a": 0.585, "b": 0.775}, plast=None)    # a closing one
    rows = PP.parse(log)
    got = RR.checks()
    check("a log that never changed: closing evaluations out, nothing moved, "
          "and entries from before the controller kept",
          [r["chars"] for r in rows] == [1.0, 2.0] == [d["chars"] for _, _, d in got]
          and rows[0]["val"] == 0.70 and not any(r.get("shifted") for r in rows))
finally:
    os.chdir(here)
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all held-out level tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
