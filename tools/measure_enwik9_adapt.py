#!/usr/bin/env python3
"""
This model on enwik9 while it learns from what it reads - test-time
adaptation - against the same pieces read with its weights frozen.

tools/measure_enwik9.py scores enwik9 the way published models are scored:
weights frozen, 2,000 pieces of 2,048 bytes spread over a 50 MB slice the
model never trained on, each predicted alone. This reads the same pieces, in
the same order, as one stream the model goes on learning from - the way it
learns from a conversation (minagi/live.py), at the rate training would take
its next step at - and scores each piece BEFORE it learns from it, so no byte
is scored by weights that have already seen it. That is how online
compressors such as NNCP and cmix score enwik9, and what the literature calls
dynamic evaluation (Krause et al., 2018): a valid compression figure, to be
compared with those, not with frozen models.

LEARNING IS THE ONLY DIFFERENCE. Each piece is still predicted alone, from an
empty attention cache and a fresh expert vote, exactly as the frozen
measurement predicts it, and each step learns from that piece alone. The
pieces sit 25 KB apart in the slice, so nothing but the weights carries from
one to the next. Both passes are made here, on the same weights, so every
piece's two scores pair up and the difference is known far more precisely
than either score.

THE WEIGHTS ARE A COPY. Learning changes them, so weights/ is copied to a
scratch directory, the copy is loaded and learns, and it is deleted at the
end - also when the measurement fails or is interrupted. Nothing opens
weights/ itself, and it is checked unchanged, file by file, before the copy
goes. A scratch directory is only ever deleted if this tool made it.

Writes a line to runs/enwik9_adapt.jsonl and the per-piece losses to
runs/enwik9_adapt_pieces.npz. The frozen pass is tools/measure_enwik9.py's
measurement, piece for piece, on weights identical to weights/, so it is kept
as that measurement too - the README's two mini-AGI rows are then the same
weights. On the card it needs the card to itself, so it will not run beside a
training run.

NOTHING IS MEASURED TWICE: weights already measured this way, on at least
as many pieces, are not measured again (--force does), so
./measure_benchmarks.sh can run it every time.

--natural READS AS THE MODEL READS. The pieces above are cut out of their
pages and each read cold. --natural reads whole pages instead - framed as
training's Wikipedia files are, one page a file, from `<page>` to `</page>` -
each from its top the way training's reader goes through a file: a step every
`training.chunk` characters, on the window of the last `context_now`
characters, a window never holding two pages. It reads EVERY page of the
slice, in the order the slice holds them, as one stream it goes on learning
from: each chunk is scored before the model learns from it, so no byte is
scored by weights that have seen it, and what it learns from one page
carries into every page after it - the model doing its best on the test by
learning from the test as it goes, as it would from anything it read.
--budget reads an even spread of pages holding about that many bytes
instead, for a quicker look. Every chunk is scored three ways on the same
text, weighted by its characters: alone and frozen, as the pieces are; in
its window, frozen; and in its window while the model learns, before it
learns from it. The first two differ by what the context gives, the last two
by what learning does. Its results go to runs/enwik9_natural.jsonl. A file
with no `<page>` in it is one document whole - a PG19 book - so the same
reading measures PG19's best effort, every book of the test split:

    python3 tools/measure_enwik9_adapt.py --natural --slice data/eval/pg19 \
        --out runs/scaling/pg19_natural.jsonl

    python3 tools/measure_enwik9_adapt.py
    python3 tools/measure_enwik9_adapt.py --pieces 500     # a quicker look
"""
import argparse
import json
import math
import os
import shutil
import signal
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import measure_enwik9 as ME                                 # noqa: E402
import measure_heldout as MH                                # noqa: E402

OUT = os.path.join(ROOT, "runs", "enwik9_adapt.jsonl")
NATURAL_OUT = os.path.join(ROOT, "runs", "enwik9_natural.jsonl")
SCRATCH = os.path.join(ROOT, "runs", "enwik9_adapt_weights")
MARK = ".enwik9-adapt-scratch"       # what makes a scratch directory ours


def fingerprint(path):
    """Every file under `path`, with its size and modification time."""
    out = {}
    for dp, _, fs in os.walk(path):
        for f in fs:
            full = os.path.join(dp, f)
            st = os.stat(full)
            out[os.path.relpath(full, path)] = (st.st_size, st.st_mtime_ns)
    return out


def _inside(a, b):
    """Whether path `a` is `b` or lies under it."""
    a, b = os.path.realpath(a), os.path.realpath(b)
    return a == b or a.startswith(b.rstrip(os.sep) + os.sep)


def clear_scratch(path):
    """Delete a scratch directory this tool made. One it did not make is
    left alone, and the measurement stops."""
    if not os.path.exists(path):
        return
    if not os.path.exists(os.path.join(path, MARK)):
        sys.exit(f"  {path} exists and was not made by this tool - not "
                 f"deleting it; choose another --scratch")
    shutil.rmtree(path)


def already(man, pieces, piece=ME.PIECE, out=OUT):
    """A measurement of the weights `man` describes, on at least `pieces`
    pieces of `piece` bytes, kept already - or None. Only one taken with the
    learning rate's controller working counts (its record says `checks`):
    one from before it ran here learned at a rate nothing moved."""
    try:
        with open(out) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if (MH.same_weights(d, man) and d.get("pieces", 0) >= pieces
                and d.get("piece_bytes") == piece and "checks" in d):
            return d
    return None


def slice_pages(raw):
    """Every whole page of the slice as (start, end) offsets, framed as
    training's Wikipedia files are - from `<page>` to `</page>`. The pages
    the slice cuts through at its two ends are left out."""
    import re
    ends = [m.end() for m in re.finditer(rb"</page>", raw)]
    out, k = [], 0
    for m in re.finditer(rb"<page>", raw):
        while k < len(ends) and ends[k] <= m.start():
            k += 1
        if k == len(ends):
            break
        out.append((m.start(), ends[k]))
    return out


def spread_pages(pages, budget):
    """Pages spread evenly over the slice, as many as hold about `budget`
    bytes between them - every kind of page in the share the slice holds it.
    `pages` are (file, start, end)."""
    mean = sum(e - s for _, s, e in pages) / max(len(pages), 1)
    n = max(1, min(len(pages), round(budget / max(mean, 1))))
    return [pages[int((j + 0.5) * len(pages) / n)] for j in range(n)]


def steps_through(n, chunk, context):
    """Every step a reader takes through a file of `n` characters from its
    top, as (window start, chunk start, chunk end): the chunk is what the
    step reads new, the window the last `context` characters ending with it
    - minagi/stream.py, FileReader.step."""
    out, pos = [], 0
    while True:
        k = min(chunk, n - pos - 1)
        if k < 2:
            break
        end = pos + k
        out.append((max(0, end - context), pos, end))
        pos = end
    return out


def already_natural(man, budget, out=NATURAL_OUT):
    """A --natural measurement of the weights `man` describes, on at least
    `budget` bytes of pages, with the controller working, kept already - or
    None."""
    try:
        with open(out) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if (d.get("regime") == "natural" and MH.same_weights(d, man)
                and d.get("budget", 0) >= budget and "checks" in d):
            return d
    return None


def _ratio_se(num, den):
    """The standard error of sum(num) / sum(den), page by page."""
    import numpy as np
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(den)
    if n < 2:
        return 0.0
    r = num.sum() / den.sum()
    return float(np.sqrt(n / (n - 1) * np.sum((num - r * den) ** 2))
                 / den.sum())


def measure_natural(a, scratch, size):
    """--natural on the copy in `scratch`: (record, per-chunk losses)."""
    import numpy as np
    import torch
    import train as T
    from minagi.config import load, get
    from minagi.ingest import as_stream
    from minagi.live import LiveLearner
    from minagi.precision import set_compute_dtype
    from minagi.stream import FileReader

    c = load()
    set_compute_dtype(str(get(c, "training.precision", "bf16")))
    device = torch.device(a.device or "cuda")
    model, cfg, pool, man = T.build_paged(
        scratch, device, ram_capacity=int(get(c, "pool.ram_cache", 96)))
    # training's own steps: its chunk, and the window it had grown to
    chunk = a.chunk or int(get(c, "training.chunk", 2048))
    context = a.context or int(man.get("context_now") or cfg.block)
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(a.slice)
                   for f in fs)
    ids_of, pages = {}, []
    for f in files:
        with open(f, "rb") as fh:
            raw = fh.read()
        ids_of[f] = as_stream(f)
        if len(ids_of[f]) != len(raw):
            sys.exit(f"  {f} does not read one byte per id")
        # an enwik9 slice is pages; a file with none in it, a book, is one
        # document whole
        pages += [(f, s, e) for s, e in (slice_pages(raw) or [(0, len(raw))])]
    if not pages:
        sys.exit(f"  no document in {a.slice}")
    picked = spread_pages(pages, a.budget)
    plan = []                        # (page, window start, chunk start, end)
    for i, (f, s, e) in enumerate(picked):
        plan += [(i, w, p, q) for w, p, q in steps_through(e - s, chunk,
                                                            context)]
    data = [ids_of[f][s:e] for f, s, e in picked]
    nb = np.array([q - p for _, _, p, q in plan], float)
    total = nb.sum()
    print(f"  {len(picked):,} of the slice's {len(pages):,} pages, "
          f"{total / 1e6:.2f}M characters in {len(plan):,} steps of up to "
          f"{chunk:,}, windows of {context:,}", flush=True)

    def nats(d, w, p, q):
        """Over the chunk p..q, predicted with the window w..p before it -
        none when w == p - as a reader measuring does."""
        r = FileReader(model, d[w:q + 1], "wikipedia", chunk, context,
                       device)
        model.eval()
        with torch.no_grad():
            left = p - w
            while left > 0:
                k = min(chunk, left)
                r._measure(k)
                left -= k
            return float(r._measure(q - p)) * (q - p)

    t0 = time.time()
    alone = np.array([nats(data[i], p, p, q) for i, w, p, q in plan])
    framed = np.array([nats(data[i], w, p, q) for i, w, p, q in plan])
    t_frozen = time.time() - t0
    ln2 = math.log(2)
    print(f"  frozen: {alone.sum() / total / ln2:.4f} bits/byte chunk by "
          f"chunk alone, {framed.sum() / total / ln2:.4f} in its window, in "
          f"{t_frozen:.0f} s", flush=True)

    learner = LiveLearner.for_run(model, weights_dir=scratch, manifest=man,
                                  lr=a.lr, chunk=chunk, context=context,
                                  save_every=0, config=c,
                                  held_out=a.held_out)
    base = learner.opt.param_groups[-1]["base_lr"]
    f0 = learner.factor()
    print(f"  learning at the run's rate: {base:g} x{f0:.4f} "
          f"= {base * f0:.3g}, a step every chunk, on its window", flush=True)
    adapted = np.zeros(len(plan))
    t1 = time.time()
    for j, (i, w, p, q) in enumerate(plan):
        adapted[j] = nats(data[i], w, p, q)      # scored first, then learned
        learner.step_on(data[i][w:q + 1].tolist(), note="enwik9", new=q - p)
        if (j + 1) % max(1, len(plan) // 20) == 0 or j + 1 == len(plan):
            b = nb[:j + 1].sum()
            print(f"    {j + 1:5,} steps: in its window frozen "
                  f"{framed[:j + 1].sum() / b / ln2:.4f}, learning "
                  f"{adapted[:j + 1].sum() / b / ln2:.4f} bits/byte "
                  f"({(time.time() - t1) / (j + 1):.2f} s a step, rate "
                  f"x{learner.factor():.4f} after "
                  f"{len(learner.checked)} checks)", flush=True)

    # page by page, for the errors: a page's chunks share its text
    pg = np.array([i for i, _, _, _ in plan])
    by = lambda v: np.bincount(pg, weights=v, minlength=len(picked))
    pb, pa, pf, pd = by(nb), by(alone), by(framed), by(adapted)
    keep = pb > 0
    quarters = []
    for part in np.array_split(np.arange(len(plan)), 4):
        quarters.append(float((framed[part] - adapted[part]).sum()
                              / nb[part].sum() / ln2))
    rec = {"regime": "natural",
           "slice": os.path.relpath(os.path.abspath(a.slice), ROOT),
           "step": man.get("step"),
           "chars": man.get("read_chars"),
           "experts": int(man.get("n_experts") or 0),
           "pages": len(picked), "pages_in_slice": len(pages),
           "budget": a.budget, "bytes": int(total), "steps": len(plan),
           "chunk": chunk, "context": context, "slice_bytes": size,
           "alone_bpb": float(alone.sum() / total / ln2),
           "frozen_bpb": float(framed.sum() / total / ln2),
           "adapted_bpb": float(adapted.sum() / total / ln2),
           # what the window before a chunk gives, and what learning does
           "context_gain_bpb": float((alone - framed).sum() / total / ln2),
           "context_gain_se": _ratio_se((pa - pf)[keep], pb[keep]) / ln2,
           "learning_gain_bpb": float((framed - adapted).sum() / total / ln2),
           "learning_gain_se": _ratio_se((pf - pd)[keep], pb[keep]) / ln2,
           "learning_gain_by_quarter": quarters,
           "lr_base": base, **rate_record(learner, base, f0),
           "learned_steps": learner.steps,
           "date": time.strftime("%Y-%m-%d %H:%M"),
           "seconds_frozen": round(t_frozen),
           "seconds_adapting": round(time.time() - t1)}
    return rec, {"alone": alone, "frozen": framed, "adapted": adapted,
                 "bytes": nb, "page": pg}


def rate_record(learner, base, f0):
    """How the rate went while it learned. `lr` is the rate it started at -
    where the run had left it - as every record has said; the controller
    then worked as it does in training, and the rest say what it did:
    `checks` held-out checks (None where it had no held-out to read), every
    `check_every` minutes, to `lr_end`, with each check's reading."""
    gov = learner.gov
    return {"factor_start": f0, "factor": learner.factor(),
            "lr": base * f0, "lr_end": base * learner.factor(),
            "checks": len(learner.checked) if gov is not None else None,
            "check_every": gov.every if gov is not None else None,
            "check_readings": [round(c["val"], 5) for c in learner.checked]}


def rate_said(rec):
    """The rate's course, in a line."""
    if rec.get("checks") is None:
        return (f"the rate stayed at the run's x{rec['factor_start']:.4f}: no "
                f"held-out folder for its controller to read")
    return (f"the rate's controller, as in training: x"
            f"{rec['factor_start']:.4f} -> x{rec['factor']:.4f} over "
            f"{rec['checks']} held-out checks, one every "
            f"{rec['check_every']:g} minutes")


def record_frozen(rec, live, out=ME.OUT, hist=ME.HIST):
    """Keep the frozen pass as tools/measure_enwik9.py's own measurement of
    the same weights - unless that tool measured them already."""
    d = {"step": rec["step"], "chars": rec["chars"],
         "experts": rec["experts"], "bpb": rec["frozen_bpb"],
         "se": rec["frozen_se"], "pieces": rec["pieces"],
         "piece_bytes": rec["piece_bytes"], "slice_bytes": rec["slice_bytes"],
         "weights": rec.get("weights", "weights"), "date": rec["date"],
         "seconds": rec.get("seconds_frozen"),
         "source": "the frozen pass of tools/measure_enwik9_adapt.py"}
    man = {"step": d["step"], "read_chars": d["chars"],
           "n_experts": d["experts"]}
    if ME.already(man, d["pieces"], out=out, hist=hist) is not None:
        return None
    ME.record(d, live, out=out, hist=hist)
    return d


def keep_frozen(a, rec):
    """On the standard pieces, the frozen pass is the enwik9 measurement of
    these weights: kept as it, unless tools/measure_enwik9.py has one."""
    if not (a.piece == ME.PIECE and rec.get("weights_unchanged")
            and os.path.realpath(a.slice) == os.path.realpath(ME.SLICE)):
        return
    live = (os.path.realpath(a.weights)
            == os.path.realpath(os.path.join(ROOT, "weights")))
    if record_frozen(rec, live):
        print("  the frozen pass is kept as the enwik9 measurement of these "
              "weights")


def measure(a, scratch, size):
    """Both passes on the copy in `scratch`: (record, per-piece losses)."""
    import numpy as np
    import torch
    import train as T
    from minagi.config import load, get
    from minagi.live import LiveLearner
    from minagi.precision import set_compute_dtype
    from minagi.stream import FolderEvaluator

    c = load()
    set_compute_dtype(str(get(c, "training.precision", "bf16")))
    device = torch.device(a.device or "cuda")
    # the copy is opened to be written: what it learns pages out into it
    model, cfg, pool, man = T.build_paged(
        scratch, device, ram_capacity=int(get(c, "pool.ram_cache", 96)))
    # the pieces tools/measure_enwik9.py scores, exactly: one piece a block,
    # every one predicted alone
    ev = FolderEvaluator(model, a.slice, a.piece, a.piece, device,
                         whole=False)
    (name,) = list(ev.groups)
    texts = ev._texts(name, a.pieces)
    n = len(texts)

    def score(t):
        model.eval()
        with torch.no_grad():
            got = ev.score_block(t, name)
        return float(np.mean(got))

    t0 = time.time()
    frozen = np.array([score(t) for t in texts])
    t_frozen = time.time() - t0
    print(f"  frozen: {frozen.mean() / math.log(2):.4f} bits/byte on {n:,} "
          f"pieces of {a.piece:,} bytes, in {t_frozen:.0f} s", flush=True)

    # Learning as training would at its next step: the run's base rate and
    # its controller's factor, the optimiser's moments as the run left them.
    # Nothing is ever saved: the copy is scrapped.
    learner = LiveLearner.for_run(model, weights_dir=scratch, manifest=man,
                                  lr=a.lr, chunk=a.piece, save_every=0,
                                  config=c, held_out=a.held_out)
    base = learner.opt.param_groups[-1]["base_lr"]
    f0 = learner.factor()
    print(f"  learning at the run's rate: {base:g} x{f0:.4f} "
          f"= {base * f0:.3g}, one step per piece", flush=True)
    adapted = np.zeros(n)
    t1 = time.time()
    for j, t in enumerate(texts):
        adapted[j] = score(t)          # scored first, then learned from
        learner.step_on(t.tolist(), note="enwik9")
        if (j + 1) % max(1, n // 20) == 0 or j + 1 == n:
            d = frozen[:j + 1] - adapted[:j + 1]
            print(f"    {j + 1:5,} pieces: frozen "
                  f"{frozen[:j + 1].mean() / math.log(2):.4f}, adapting "
                  f"{adapted[:j + 1].mean() / math.log(2):.4f} bits/byte - "
                  f"{d.mean() / math.log(2):+.4f} +/-"
                  f"{d.std(ddof=1) / math.sqrt(len(d)) / math.log(2) if len(d) > 1 else 0:.4f}"
                  f"  ({(time.time() - t1) / (j + 1):.2f} s a piece)",
                  flush=True)

    ln2 = math.log(2)
    d = frozen - adapted
    q = [float(x.mean() / ln2) for x in np.array_split(d, 4)]
    rec = {"step": man.get("step"), "chars": man.get("read_chars"),
           "experts": int(man.get("n_experts") or 0), "pieces": n,
           "piece_bytes": a.piece, "slice_bytes": size,
           "frozen_bpb": float(frozen.mean() / ln2),
           "frozen_se": float(frozen.std(ddof=1) / math.sqrt(n) / ln2),
           "adapted_bpb": float(adapted.mean() / ln2),
           "adapted_se": float(adapted.std(ddof=1) / math.sqrt(n) / ln2),
           # frozen minus adapting: what learning took off, piece by piece
           "gain_bpb": float(d.mean() / ln2),
           "gain_se": float(d.std(ddof=1) / math.sqrt(n) / ln2),
           "gain_by_quarter": q,
           "lr_base": base, **rate_record(learner, base, f0),
           "steps": learner.steps,
           "date": time.strftime("%Y-%m-%d %H:%M"),
           "seconds_frozen": round(t_frozen),
           "seconds_adapting": round(time.time() - t1)}
    return rec, {"frozen": frozen, "adapted": adapted}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--pieces", type=int, default=2000)
    ap.add_argument("--piece", type=int, default=ME.PIECE,
                    help=argparse.SUPPRESS)         # bytes; tests use less
    ap.add_argument("--slice", default=ME.SLICE, help=argparse.SUPPRESS)
    ap.add_argument("--scratch", default=SCRATCH,
                    help="where the copy of the weights learns; deleted at "
                         "the end")
    ap.add_argument("--out", default=OUT, help=argparse.SUPPRESS)
    ap.add_argument("--lr", type=float, default=None,
                    help="the base rate the controller scales; defaults to "
                         "the run's own")
    ap.add_argument("--device", default=None,
                    help="cuda unless said otherwise; the CPU is for tests")
    ap.add_argument("--held-out", default=None,
                    help="the held-out folder the learning rate's controller "
                         "reads, as in training; defaults to the one the run "
                         "used (the weights' `checks`), else data/val")
    ap.add_argument("--force", action="store_true",
                    help="measure even if these weights were measured "
                         "already")
    ap.add_argument("--natural", action="store_true",
                    help="whole pages, read as training reads a file: a "
                         "window of the last context_now characters, a step "
                         "every training.chunk")
    ap.add_argument("--budget", type=int, default=None,
                    help="--natural: read an even spread of pages holding "
                         "about this many bytes, for a quicker look, instead "
                         "of every page the slice holds"),
    ap.add_argument("--chunk", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--context", type=int, default=None,
                    help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.natural and a.out == OUT:
        a.out = NATURAL_OUT

    import torch
    on_card = (a.device or ("cuda" if torch.cuda.is_available() else "cpu")
               ) != "cpu"
    if a.device is None and not on_card:
        sys.exit("  no GPU - learning through enwik9 on the CPU would take "
                 "days")
    if on_card:
        pid = MH.training_run(a.weights)
        if pid:
            sys.exit(f"  a training run is going (pid {pid}) - stop it to "
                     f"measure: this needs the card to itself")
    if not os.path.exists(os.path.join(a.weights, "manifest.json")):
        sys.exit(f"  no model in {a.weights}")
    with open(os.path.join(a.weights, "manifest.json")) as f:
        man = json.load(f)
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(a.slice)
             for f in fs]
    if not files:
        sys.exit(f"  {a.slice} is missing")
    size = sum(os.path.getsize(f) for f in files)
    if a.budget is None:
        # --natural reads everything the slice holds: a budget of the whole
        # slice spreads over every page of it, in order
        a.budget = size if a.natural else a.pieces * a.piece
    if a.natural:
        have = None if a.force else already_natural(man, a.budget, a.out)
        if have is not None:
            print(f"  enwik9 read naturally: {have['adapted_bpb']:.4f} "
                  f"learning, {have['frozen_bpb']:.4f} frozen, at step "
                  f"{have['step']:,}, is already measured on these weights; "
                  f"not measuring again")
            return have
    have = (None if a.force or a.natural
            else already(man, a.pieces, a.piece, a.out))
    if have is not None:
        print(f"  enwik9 learning as it reads: {have['adapted_bpb']:.4f} "
              f"against {have['frozen_bpb']:.4f} frozen, at step "
              f"{have['step']:,}, is already measured on these weights; not "
              f"measuring again")
        keep_frozen(a, have)
        return have
    if any(n.endswith(".tmp") or ".tmp." in n for n in os.listdir(a.weights)):
        sys.exit(f"  {a.weights} has a save in progress - not copying it")
    if _inside(a.scratch, a.weights) or _inside(a.weights, a.scratch):
        sys.exit("  the scratch directory and the weights must not hold one "
                 "another")
    clear_scratch(a.scratch)
    need = sum(v[0] for v in fingerprint(a.weights).values())
    os.makedirs(os.path.dirname(os.path.abspath(a.scratch)), exist_ok=True)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.scratch))).free
    if free < need + 10e9:
        sys.exit(f"  {free / 1e9:.0f} GB free; the copy needs {need / 1e9:.1f} "
                 f"GB and 10 GB to spare")

    before = fingerprint(a.weights)
    t0 = time.time()
    # killed rather than interrupted, it still deletes the copy on the way out
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("  terminated"))
    try:
        # a full copy, not links: nothing written into it can reach weights/
        shutil.copytree(a.weights, a.scratch)
        with open(os.path.join(a.scratch, MARK), "w") as f:
            f.write(f"copy of {os.path.realpath(a.weights)}, made "
                    f"{time.strftime('%Y-%m-%d %H:%M')} by "
                    f"tools/measure_enwik9_adapt.py; deleted when it ends\n")
        print(f"  copied {a.weights} ({need / 1e9:.1f} GB) to {a.scratch} in "
              f"{time.time() - t0:.0f} s", flush=True)
        rec, per = (measure_natural if a.natural else measure)(
            a, a.scratch, size)
        rec["weights_unchanged"] = fingerprint(a.weights) == before
        rec["weights"] = os.path.relpath(os.path.abspath(a.weights), ROOT)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not a.natural:
            keep_frozen(a, rec)
        import numpy as np
        np.savez(os.path.splitext(a.out)[0] + "_pieces.npz", **per)
    finally:
        shutil.rmtree(a.scratch, ignore_errors=True)
    if a.natural:
        print(f"\n  enwik9 read as the model reads, {rec['pages']:,} whole "
              f"pages, {rec['bytes'] / 1e6:.2f}M characters, at step "
              f"{rec['step']:,}:")
        print(f"    frozen, each chunk alone      {rec['alone_bpb']:.4f}")
        print(f"    frozen, in its window         {rec['frozen_bpb']:.4f}   "
              f"context took {rec['context_gain_bpb']:+.4f} +/-"
              f"{rec['context_gain_se']:.4f} off")
        print(f"    in its window, learning       {rec['adapted_bpb']:.4f}   "
              f"learning took {rec['learning_gain_bpb']:+.4f} +/-"
              f"{rec['learning_gain_se']:.4f} off, by quarter "
              + ", ".join(f"{x:+.4f}" for x in
                          rec["learning_gain_by_quarter"]))
        print(f"  {rate_said(rec)}")
        print(f"  the copy is deleted; {a.weights} "
              + ("unchanged" if rec["weights_unchanged"] else
                 "CHANGED while this ran - look at what else was writing it")
              + f" -> {os.path.relpath(os.path.abspath(a.out), ROOT)}")
        return rec
    print(f"\n  enwik9, {rec['pieces']:,} pieces of {rec['piece_bytes']:,} "
          f"bytes, at step {rec['step']:,}:")
    print(f"    frozen          {rec['frozen_bpb']:.4f} +/-{rec['frozen_se']:.4f}"
          f" bits/byte")
    print(f"    adapting        {rec['adapted_bpb']:.4f} +/-"
          f"{rec['adapted_se']:.4f} bits/byte")
    print(f"    learning took   {rec['gain_bpb']:+.4f} +/-{rec['gain_se']:.4f}"
          f" bits/byte off, by quarter "
          + ", ".join(f"{x:+.4f}" for x in rec["gain_by_quarter"]))
    print(f"  {rate_said(rec)}")
    print(f"  the copy is deleted; {a.weights} "
          + ("unchanged" if rec["weights_unchanged"] else
             "CHANGED while this ran - look at what else was writing it")
          + f" -> {os.path.relpath(os.path.abspath(a.out), ROOT)}")
    return rec


if __name__ == "__main__":
    main()
