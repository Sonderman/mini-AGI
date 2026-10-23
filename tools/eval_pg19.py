#!/usr/bin/env python3
"""
Bits per byte on PG19, the way the byte-level papers report it.

    python3 tools/eval_pg19.py --books data/eval/pg19 --fraction 0.1

WHY THIS EXISTS. Held-out loss on this project's own eight-subject mixture
cannot be compared to anyone else's number - a different held-out set is a
different quantity, however the units are spelled. PG19 is what MambaByte,
MegaByte, PerceiverAR and the byte-level Transformer all publish on, so a
number here is the one thing that puts this model on their axis.

HOW IT IS SCORED. Consecutive bytes of a book, in windows of the model's
context, each window scored once and the losses pooled by byte - not averaged
per window, which would weight a short trailing window like a full one. The
first byte of a window has no context inside that window, so it is dropped
rather than scored; at a 4,096 window that is 0.02% of the bytes and it stops
the score being flattered by predictions nothing informed.

WHAT IS NOT MATCHED. The published numbers use a context of 8,192 and this
model was built with 4,096 - the rotary tables are sized at build time, so
matching it needs a different model, not a different flag. The context this
ran at is printed and belongs beside any number taken from here.

NOTHING IS MEASURED TWICE: weights the whole split was already scored on are
not scored again (--force does), so measure_benchmarks.sh can run it
every time.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.precision import amp, set_compute_dtype           # noqa: E402
from minagi.recur import load_any                             # noqa: E402
from minagi.tokenizer import ByteTokenizer                    # noqa: E402


@torch.no_grad()
def score_book(model, ids, window, device, stride=1):
    """
    Total nats and bytes scored for one book.

    `stride` scores every nth window rather than the first 1/n of the book.
    A prefix is the worst possible sample here: the opening bytes of a
    Gutenberg text are a title page, a licence header and a table of
    contents, which is nothing like the prose the rest of the book is, and
    scoring only those made the first trial read 2.55 BPB. Spreading the
    same number of windows across the whole book is unbiased for the number
    the full pass would give.
    """
    tot, n = 0.0, 0
    for i in range(0, len(ids) - 1, window * stride):
        chunk = ids[i:i + window + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor([chunk[1:]], device=device)
        with amp(device):
            logits, _ = model(x, collect=False), None
        if isinstance(logits, tuple):
            logits = logits[0]
        ce = torch.nn.functional.cross_entropy(
            logits[0].float(), y[0], reduction="sum")
        tot += float(ce)
        n += y.shape[1]
    return tot, n


def already(man, out):
    """The whole-split measurement of the weights `man` describes, kept in
    `out` already - or None."""
    try:
        d = json.load(open(out))
    except (OSError, ValueError):
        return None
    if (d.get("step") == man.get("step")
            and d.get("experts") == len(man.get("experts", []))
            and abs((d.get("chars_read") or 0)
                    - float(man.get("read_chars") or 0)) < 1e5
            and d.get("fraction", 1.0) >= 1.0 and d.get("books", 0) >= 100):
        return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="data/eval/pg19")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="share of each book to score, spread evenly across "
                         "it rather than taken from the front. 0.1 scores "
                         "every tenth window")
    ap.add_argument("--books-limit", type=int, default=0,
                    help="score only the first N books; 0 for all")
    ap.add_argument("--window", type=int, default=0,
                    help="bytes per window; 0 uses the model's context")
    ap.add_argument("--precision", default=None)
    ap.add_argument("--force", action="store_true",
                    help="measure even if these weights were measured "
                         "already")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = a.fraction >= 1.0 and not a.books_limit and not a.window
    if full and not a.force:
        have = already(json.load(open(os.path.join(a.weights,
                                                   "manifest.json"))),
                       os.path.join(root, "runs/scaling/pg19.json"))
        if have is not None:
            print(f"  PG19 {have['bpb']:.4f} BPB at step {have['step']:,} is "
                  f"already measured on these weights; not measuring again")
            return 0

    from minagi.config import get as _g, load as _lc
    set_compute_dtype(a.precision or _g(_lc(), "training.precision", "bf16"))
    dev = torch.device(a.device if torch.cuda.is_available()
                       or a.device == "cpu" else "cpu")
    model, _ = load_any(a.weights, dev, read_only=True)
    model.eval()
    window = a.window or model.cfg.block
    tok = ByteTokenizer()

    files = sorted(glob.glob(os.path.join(a.books, "*.txt")))
    if a.books_limit:
        files = files[:a.books_limit]
    if not files:
        raise SystemExit(f"no books in {a.books} - run: python3 -m corpora pg19")

    # Each window's forward admits the experts it asks for - the same rule
    # reading trains with.

    stride_note = ("" if a.fraction >= 1.0
                   else f", every {max(1, round(1.0/a.fraction))}th window")
    print(f"  {len(files)} books, window {window:,}, "
          f"fraction {a.fraction:g}{stride_note}")
    t0 = time.time()
    tot, n = 0.0, 0
    stride = max(1, round(1.0 / a.fraction)) if a.fraction < 1.0 else 1
    for i, f in enumerate(files):
        raw = open(f, "rb").read()
        ids = tok.encode(raw.decode("utf-8", "replace")).ids
        bt, bn = score_book(model, ids, window, dev, stride)
        tot += bt
        n += bn
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            el = time.time() - t0
            print(f"    {i+1}/{len(files)}  {n/1e6:.2f}M bytes  "
                  f"{tot/n/math.log(2):.4f} BPB  "
                  f"{n/max(el,1e-9):,.0f} byte/s", flush=True)

    nats = tot / n
    # Written out so the figure reads a measurement instead of carrying a
    # number somebody typed in, and so a later full-split run replaces it.
    # WHERE THE MODEL WAS: read from the weights that were scored, not from
    # the log's last line. A snapshot measured while the run reads on is
    # behind the log, and the point would be drawn where the model was not.
    man = json.load(open(os.path.join(a.weights, "manifest.json")))
    rec = {"bpb": nats / math.log(2), "nats": nats, "bytes_scored": n,
           "books": len(files), "window": window, "fraction": a.fraction,
           "chars_read": float(man.get("read_chars") or 0) or None,
           "step": man.get("step"), "experts": len(man.get("experts", [])),
           "weights": os.path.abspath(a.weights),
           "date": time.strftime("%Y-%m-%d")}
    try:
        sys.path.insert(0, os.path.join(root, "tools"))
        import plot_progress as _pp                          # noqa: E402
        rows = [r for r in _pp.parse(os.path.join(root, "runs/samples.txt"))
                if r.get("val")]
        if rows:
            # the model's own held-out at the logged round nearest to it
            at = rec["chars_read"] or rows[-1]["chars"] * 1e6
            near = min(rows, key=lambda r: abs(r["chars"] * 1e6 - at))
            rec["own_bpb"] = near["val"] / math.log(2)
            if not rec["chars_read"]:
                rec["chars_read"] = near["chars"] * 1e6
    except Exception:                                        # noqa: BLE001
        pass
    # EVERY MEASUREMENT IS KEPT. pg19.json is the latest, which the README
    # quotes; the history holds every one, which the figure draws. A history
    # that does not exist yet starts with what pg19.json already held, so
    # the first measurement is not lost to the second.
    out = os.path.join(root, "runs/scaling/pg19.json")
    hist = os.path.join(root, "runs/scaling/pg19_history.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(hist) and os.path.exists(out):
        try:
            with open(hist, "w") as f:
                f.write(json.dumps(json.load(open(out))) + "\n")
        except (OSError, ValueError):
            pass
    with open(hist, "a") as f:
        f.write(json.dumps(rec) + "\n")
    json.dump(rec, open(out, "w"), indent=1)
    print(f"\n  PG19 {os.path.basename(a.books)}: {nats:.4f} nats/byte = "
          f"**{nats/math.log(2):.4f} BPB**")
    print(f"  over {n:,} bytes of {len(files)} books, context {window:,}, "
          f"in {(time.time()-t0)/60:.1f} min")
    print(f"  recorded in runs/scaling/pg19.json and pg19_history.jsonl")
    print("\n  published, at context 8,192 unless noted:")
    for nm, ctx, byt, bpb in (("Transformer-320M", "1,024", "80B", 1.057),
                              ("PerceiverAR-248M", "8,192", "80B", 1.104),
                              ("MegaByte-1.02B", "8,192", "80B", 1.000),
                              ("MambaByte-972M", "8,192", "75B", 0.883)):
        print(f"    {nm:<18} ctx {ctx:>5}  {byt:>4} bytes  {bpb:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
