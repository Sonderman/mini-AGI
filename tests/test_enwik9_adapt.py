#!/usr/bin/env python3
"""
enwik9 read while the model learns from it (tools/measure_enwik9_adapt.py).

What these tests hold it to:
  the weights it measures are a copy, deleted at the end, and the weights
  directory is unchanged, file by file - also when the measurement fails
  partway
  a scratch directory it did not make is never deleted, and one that holds
  the weights, or is held by them, is refused
  the frozen pass is tools/measure_enwik9.py's measurement, piece for piece
  each piece is scored before the model learns from it: the first scores
  what it scores frozen, and the ones after it do not
  it learns at the run's rate - the manifest's base rate times its
  controller's factor - and its controller goes on working as in training:
  the session's held-out evaluation, then a check whenever one is due, as
  the run's manifest says; the record says how the rate went
  weights measured already are not measured again, unless asked
  --natural reads whole pages as training reads a file: pages framed as
  training's are, the window and the steps training's reader takes, every
  chunk scored alone, in its window, and in its window before it is learned
  from - the first chunk of a page the same all three ways, a page that fits
  one window scoring what reading it from the top scores, and nothing of it
  reaching the pieces' results or the enwik9 measurement
  it reads every page of the slice, in order, unless --budget asks for an
  even spread of them; every book of a folder of books
"""
import json
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

import measure_enwik9_adapt as MA                           # noqa: E402
import train as T                                           # noqa: E402
from minagi.create import create                            # noqa: E402
from minagi.stream import FolderEvaluator                   # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-adapt-")
try:
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=12, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=128, max_steps=3,
           top_k=2, depth=1)
    # what a training run leaves: its base rate, its controller's state
    with open(os.path.join(wdir, "manifest.json")) as f:
        man = json.load(f)
    # ...and how it reads its checks: a held-out folder, and a check due
    # after every step, so the controller works within a test's few steps
    val = os.path.join(tmp, "val")
    for subj, text in (("prose", "the river city was founded in 1820. " * 30),
                       ("code", "def f(x):\n    return x + 1\n" * 40)):
        os.makedirs(os.path.join(val, subj))
        with open(os.path.join(val, subj, "a.txt"), "w") as f:
            f.write(text)
    man.update(lr=2e-3, plasticity={"scale": 0.5, "step": 900},
               checks={"every": 1e-9, "held_out": val, "check_chars": 64,
                       "eval_chars": 128})
    with open(os.path.join(wdir, "manifest.json"), "w") as f:
        json.dump(man, f)

    rng = np.random.default_rng(0)
    words = ["the", "river", "city", "was", "founded", "in", "of", "and",
             "[[", "]]", "==", "\n", "1820", "by", "a", "king"]
    sl = os.path.join(tmp, "e9", "wikipedia")
    os.makedirs(sl)
    with open(os.path.join(sl, "slice.txt"), "w") as f:
        f.write(" ".join(words[int(i)] for i in rng.integers(0, len(words),
                                                            1500)))

    out = os.path.join(tmp, "runs", "adapt.jsonl")
    scratch = os.path.join(tmp, "scratch")
    args = ["--weights", wdir, "--slice", os.path.dirname(sl), "--pieces",
            "10", "--piece", "64", "--scratch", scratch, "--out", out,
            "--device", "cpu"]
    before = MA.fingerprint(wdir)
    rec = MA.main(args)
    per = np.load(os.path.splitext(out)[0] + "_pieces.npz")

    check("the copy is deleted and the weights are unchanged, file by file",
          not os.path.exists(scratch) and MA.fingerprint(wdir) == before
          and rec["weights_unchanged"] is True)
    check("a line is kept for the measurement, with the per-piece losses",
          [json.loads(l)["pieces"] for l in open(out)] == [10]
          and len(per["frozen"]) == len(per["adapted"]) == 10)

    # the frozen pass is tools/measure_enwik9.py's, on the original weights
    m, _, _, _ = T.build_paged(wdir, "cpu", ram_capacity=4, read_only=True)
    ev = FolderEvaluator(m, os.path.dirname(sl), 64, 64, "cpu", whole=False)
    v = ev.run(10, spread=True)
    v.pop("stderr")
    (nats,) = v.values()
    check("the frozen pass is the enwik9 measurement, piece for piece",
          abs(rec["frozen_bpb"] - nats / math.log(2)) < 1e-5,
          f"{rec['frozen_bpb']:.5f} / {nats / math.log(2):.5f}")
    check("each piece is scored before learning from it: the first as frozen, "
          "the later ones not",
          abs(per["adapted"][0] - per["frozen"][0]) < 1e-5
          and np.abs(per["adapted"][1:] - per["frozen"][1:]).max() > 1e-4,
          f"first {per['adapted'][0] - per['frozen'][0]:+.1e}")
    check("it learns at the run's rate: the manifest's base times the "
          "controller's factor, a step a piece",
          rec["lr_base"] == 2e-3 and rec["factor_start"] == 0.5
          and abs(rec["lr"] - 1e-3) < 1e-12 and rec["steps"] == 10)
    check("...and its controller works as in training: checks as the run's "
          "manifest says, and the record says how the rate went",
          rec["checks"] >= 1 and rec["check_every"] == 1e-9
          and len(rec["check_readings"]) == rec["checks"]
          and abs(rec["lr_end"] - 2e-3 * rec["factor"]) < 1e-15,
          f"{rec['checks']} checks, x{rec['factor_start']} -> "
          f"x{rec['factor']:.4f}")
    check("...and the gain is frozen minus adapting, by quarter as well",
          abs(rec["gain_bpb"] - (rec["frozen_bpb"] - rec["adapted_bpb"])) < 1e-9
          and len(rec["gain_by_quarter"]) == 4)

    # the same weights again: nothing is measured, nothing is copied
    again = MA.main(args)
    check("weights measured already are not measured again",
          again["date"] == rec["date"] and not os.path.exists(scratch)
          and len(open(out).read().splitlines()) == 1)
    args = args + ["--force"]

    # it fails partway: the copy still goes, and the weights stay as they were
    real = MA.measure

    def broken(*a, **k):
        assert os.path.exists(os.path.join(scratch, MA.MARK))
        raise RuntimeError("the card fell over")
    MA.measure = broken
    try:
        MA.main(args)
        failed = False
    except RuntimeError:
        failed = True
    finally:
        MA.measure = real
    check("a measurement that fails partway still deletes its copy",
          failed and not os.path.exists(scratch)
          and MA.fingerprint(wdir) == before)

    # scratch directories it did not make, or that overlap the weights
    os.makedirs(scratch)
    with open(os.path.join(scratch, "keep.txt"), "w") as f:
        f.write("someone's")

    def refused(argv):
        try:
            MA.main(argv)
        except SystemExit as x:
            return str(x)
        return None
    said = refused(args)
    check("a scratch directory it did not make is refused, not deleted",
          said and "not made by this tool" in said
          and os.path.exists(os.path.join(scratch, "keep.txt")))
    shutil.rmtree(scratch)
    inside = args[:args.index("--scratch") + 1] + [os.path.join(wdir, "s")] \
        + args[args.index("--scratch") + 2:]
    check("...and so is one inside the weights",
          "must not hold one another" in (refused(inside) or ""))
    with open(os.path.join(tmp, "stale"), "w"):
        pass
    os.makedirs(scratch)
    with open(os.path.join(scratch, MA.MARK), "w") as f:
        f.write("a copy a killed run left\n")
    MA.main(args)
    check("a copy a killed run left behind is cleared and made again",
          not os.path.exists(scratch)
          and len(open(out).read().splitlines()) == 2)
    # the frozen pass is kept as tools/measure_enwik9.py's measurement
    import measure_enwik9 as ME
    e_out = os.path.join(tmp, "runs", "e9.json")
    e_hist = os.path.join(tmp, "runs", "scaling", "e9.jsonl")
    std = dict(rec, pieces=2000, piece_bytes=2048, step=77, chars=3.1e9,
               experts=12)
    kept = MA.record_frozen(std, True, out=e_out, hist=e_hist)
    again = MA.record_frozen(std, True, out=e_out, hist=e_hist)
    hist_n = len(open(e_hist).read().splitlines())
    man = {"step": 77, "read_chars": 3.1e9, "n_experts": 12}
    check("the frozen pass is kept as the enwik9 measurement of the same "
          "weights, once",
          kept is not None and again is None and hist_n == 1
          and json.load(open(e_out))["bpb"] == rec["frozen_bpb"]
          and ME.already(man, 2000, out=e_out, hist=e_hist) is not None)
    MA.record_frozen(dict(std, step=78), False, out=e_out, hist=e_hist)
    check("...in the history only, when the weights were not weights/",
          json.load(open(e_out))["step"] == 77
          and len(open(e_hist).read().splitlines()) == 2)
    # --natural: whole pages, read as training reads a file
    raw = b"tail of a page the slice cut</page>\n"
    sizes = (40, 90, 150, 300, 25, 200, 600, 70)
    for j, n in enumerate(sizes):
        body = " ".join(words[int(i)] for i in rng.integers(0, len(words), n))
        raw += (b"  <page>\n    <title>P%d</title>\n    <text>" % j
                + body.encode()[:n] + b"</text>\n  </page>\n")
    raw += b"  <page>\n    <title>cut</title>\n    <text>the slice ends"
    nat = os.path.join(tmp, "nat", "wikipedia")
    os.makedirs(nat)
    with open(os.path.join(nat, "slice.txt"), "wb") as f:
        f.write(raw)
    pages = MA.slice_pages(raw)
    check("pages are framed as training's files are, and the ones the slice "
          "cuts are left out",
          len(pages) == len(sizes)
          and all(raw[s_:e_].startswith(b"<page>")
                  and raw[s_:e_].endswith(b"</page>") for s_, e_ in pages))
    st = MA.steps_through(300, 32, 128)
    check("...read as training's reader steps through a file: chunks of up "
          "to 32 laid end to end, each on the last 128 characters",
          st[0] == (0, 0, 32) and st[4] == (32, 128, 160)
          and all(b == st[k - 1][2] for k, (_, b, _) in enumerate(st) if k)
          and st[-1][2] == 299 and all(q - p <= 32 for _, p, q in st)
          and all(w == max(0, q - 128) for w, _, q in st))
    nout = os.path.join(tmp, "runs", "natural.jsonl")
    nargs = ["--weights", wdir, "--slice", os.path.dirname(nat),
             "--natural", "--chunk", "32",
             "--context", "128", "--scratch", scratch, "--out", nout,
             "--device", "cpu"]
    pieces_before = open(out).read()
    nrec = MA.main(nargs)
    nper = np.load(os.path.splitext(nout)[0] + "_pieces.npz")
    check("...every page of the slice read when no budget is given - its "
          "budget the whole slice - and the copy deleted",
          nrec["pages"] == nrec["pages_in_slice"] == len(sizes)
          and nrec["budget"] == len(raw) and not os.path.exists(scratch)
          and nrec["weights_unchanged"] is True
          and MA.fingerprint(wdir) == before
          and nrec["steps"] == len(nper["bytes"]) == nrec["learned_steps"])
    first = np.array([k == 0 or nper["page"][k] != nper["page"][k - 1]
                      for k in range(len(nper["page"]))])
    check("...the first chunk of a page scores the same alone and in its "
          "window, and the very first the same learning too",
          np.allclose(nper["alone"][first], nper["frozen"][first], atol=1e-4)
          and abs(nper["adapted"][0] - nper["frozen"][0]) < 1e-4)
    # a page that fits one window: frozen in its window is reading it from
    # the top, chunk by chunk through the cache
    from minagi.stream import FileReader
    m2, _, _, _ = T.build_paged(wdir, "cpu", ram_capacity=4, read_only=True)
    m2.eval()
    k = int(np.argmax(np.diff(np.r_[0, np.cumsum(np.bincount(nper["page"]))])
                      * 0 + np.bincount(nper["page"]) * (np.array(
                          [e_ - s_ for s_, e_ in pages]) <= 128)))
    s_, e_ = pages[k]
    from minagi.ingest import as_stream
    page = as_stream(os.path.join(nat, "slice.txt"))[s_:e_]
    r = FileReader(m2, page, "wikipedia", 32, 128, "cpu")
    top = 0.0
    with torch.no_grad():
        while not r.done():
            n0 = r.pos
            l = r.step(learn=False)
            if l is None:
                break
            top += float(l) * (r.pos - n0)
    mine = nper["frozen"][nper["page"] == k].sum()
    check("...and a page that fits one window scores, frozen in its window, "
          "what reading it from the top scores",
          abs(top - mine) < 1e-3 and (e_ - s_) <= 128,
          f"{top:.4f} / {mine:.4f}")
    check("...context and learning each say what they took off, and nothing "
          "reaches the pieces' results",
          abs(nrec["context_gain_bpb"] - (nrec["alone_bpb"]
                                          - nrec["frozen_bpb"])) < 1e-9
          and abs(nrec["learning_gain_bpb"] - (nrec["frozen_bpb"]
                                               - nrec["adapted_bpb"])) < 1e-9
          and open(out).read() == pieces_before
          and len(nrec["learning_gain_by_quarter"]) == 4)
    again = MA.main(nargs)
    check("...and the same weights are not read naturally twice",
          again["date"] == nrec["date"]
          and len(open(nout).read().splitlines()) == 1)

    # books: a file with no <page> in it is one document whole, as PG19's are
    books = os.path.join(tmp, "books", "pg19")
    os.makedirs(books)
    for j, n in enumerate((300, 520, 180)):
        with open(os.path.join(books, f"book{j}.txt"), "w") as f:
            f.write(" ".join(words[int(i)] for i in rng.integers(0, len(words),
                                                                 n))[:n * 3])
    bout = os.path.join(tmp, "runs", "books.jsonl")
    brec = MA.main(["--weights", wdir, "--slice", os.path.dirname(books),
                    "--natural", "--chunk", "32",
                    "--context", "128", "--scratch", scratch, "--out", bout,
                    "--device", "cpu"])
    sizes = [os.path.getsize(os.path.join(books, f))
             for f in sorted(os.listdir(books))]
    pp = [("f", i * 10, i * 10 + 10) for i in range(10)]
    check("a budget of the whole slice spreads over every page, in order; a "
          "smaller one over an even spread of them",
          MA.spread_pages(pp, 100) == pp and MA.spread_pages(pp, 1000) == pp
          and MA.spread_pages(pp, 30) == [pp[1], pp[5], pp[8]])
    check("a book, with no <page> in it, is read as one document whole, and "
          "every book of the folder is read",
          brec["pages"] == 3 and brec["pages_in_slice"] == 3
          and brec["bytes"] == sum(n - 1 for n in sizes)
          and brec["slice"].endswith("books") and not os.path.exists(scratch),
          f"{brec['pages']} documents, {brec['bytes']} bytes")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all enwik9 adaptation tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
