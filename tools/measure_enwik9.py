#!/usr/bin/env python3
"""
This model on enwik9, measured the way published models were, for the README.

enwik9 is the first billion bytes of English Wikipedia from March 2006, and
the text compressors and language models are most often compared on, in bits
per byte. This model's Wikipedia lane comes from the same dump, so it is
scored on a 50 MB slice of enwik9 kept out of training: none of the pages in
data/enwik9/wikipedia is in data/train. And it is scored the way Delétang et
al. (2023) scored Chinchilla 1B - the text cut into pieces of 2,048 bytes,
each predicted with no memory of the others - on `--pieces` of them spread
evenly over the slice, the same pieces every time.

Writes runs/enwik9_measured.json, which tools/refresh_readme.py builds the
README's enwik9 table from, and adds every measurement to
runs/scaling/enwik9_history.jsonl, which tools/plot_scaling.py draws. Like
tools/measure_heldout.py it needs the card and the weights to itself,
measures nothing twice - a measurement of the same weights is reused - and
publish.sh stops if it fails.

    python3 tools/measure_enwik9.py
    python3 tools/measure_enwik9.py --weights weights_backup   # an older point

Weights other than weights/ go into the history only: the README's table is
about the model as it stands.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import measure_heldout as MH                                # noqa: E402

OUT = os.path.join(ROOT, "runs", "enwik9_measured.json")
HIST = os.path.join(ROOT, "runs", "scaling", "enwik9_history.jsonl")
SLICE = os.path.join(ROOT, "data", "enwik9")      # its one subject: wikipedia
PIECE = 2048                                      # bytes, as Delétang et al.


def history(out=OUT, hist=HIST):
    """Every measurement kept, oldest first: the history, and the latest
    measurement of weights/ in case the history never saw it."""
    ds = []
    try:
        with open(hist) as f:
            ds = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        ds = []
    try:
        with open(out) as f:
            ds.append(json.load(f))
    except (OSError, ValueError):
        pass
    return ds


def already(man, pieces, out=OUT, hist=HIST):
    """A measurement of the weights `man` describes, on at least as many
    pieces, or None."""
    for d in reversed(history(out, hist)):
        if MH.same_weights(d, man) and d.get("pieces", 0) >= pieces:
            return d
    return None


def record(d, live, out=OUT, hist=HIST):
    """Keep measurement `d`: in the history always, and as the README's
    latest only when it was taken on weights/ (`live`). A history that does
    not exist yet starts with the measurement already there, so the first
    point is not lost to the second."""
    os.makedirs(os.path.dirname(hist), exist_ok=True)
    if not os.path.exists(hist) and os.path.exists(out):
        try:
            with open(out) as f:
                seed = json.load(f)
            with open(hist, "w") as f:
                f.write(json.dumps(seed) + "\n")
        except (OSError, ValueError):
            pass
    with open(hist, "a") as f:
        f.write(json.dumps(d) + "\n")
    if live:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w") as f:
            json.dump(d, f, indent=1)
        os.replace(out + ".tmp", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--pieces", type=int, default=2000)
    ap.add_argument("--device", default=None,
                    help="cuda unless said otherwise; the CPU is for tests")
    a = ap.parse_args()

    pid = MH.training_run(a.weights)
    if pid:
        sys.exit(f"  a training run is going (pid {pid}) - stop it to measure "
                 f"enwik9")
    if any(n.endswith(".tmp") or ".tmp." in n for n in os.listdir(a.weights)):
        sys.exit(f"  {a.weights} has a save in progress - not measuring")
    with open(os.path.join(a.weights, "manifest.json")) as f:
        man = json.load(f)
    have = already(man, a.pieces)
    if have is not None:
        print(f"  enwik9 {have['bpb']:.4f} bits/byte at step {have['step']:,} "
              f"is already measured on these weights; not measuring again")
        return

    import torch
    import train as T
    from minagi.config import load, get
    from minagi.ingest import as_stream
    from minagi.precision import set_compute_dtype
    from minagi.stream import FolderEvaluator

    if a.device is None and not torch.cuda.is_available():
        sys.exit("  no GPU - enwik9 on the CPU would take hours")
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(SLICE) for f in fs]
    size = sum(os.path.getsize(f) for f in files)
    if not files or sum(len(as_stream(f)) for f in files) != size:
        # a structural marker would be one id for several bytes, and bits per
        # id would no longer be bits per byte
        sys.exit(f"  {SLICE} is missing, or does not read one byte per id")
    set_compute_dtype(str(get(load(), "training.precision", "bf16")))
    device = torch.device(a.device or "cuda")
    t0 = time.time()
    model, cfg, pool, man = T.build_paged(a.weights, device, ram_capacity=int(
        get(load(), "pool.ram_cache", 96)), read_only=True)
    # context = one piece: every piece starts with an empty cache and a new
    # expert vote, and is scored on its own
    # the old block list, with the slice's last part-piece: the pieces the
    # published GPT-2 rows were scored on, exactly. That piece is never one
    # of them, so nothing about it changes the measurement.
    ev = FolderEvaluator(model, SLICE, PIECE, PIECE, device, whole=False)
    v = ev.run(a.pieces, spread=True)
    se = v.pop("stderr", 0.0)
    (nats,) = v.values()
    out = {"step": man.get("step"), "chars": man.get("read_chars"),
           "experts": int(man.get("n_experts") or 0),
           "bpb": nats / math.log(2), "se": se / math.log(2),
           "pieces": a.pieces, "piece_bytes": PIECE, "slice_bytes": size,
           "weights": os.path.relpath(os.path.abspath(a.weights), ROOT),
           "date": time.strftime("%Y-%m-%d %H:%M"),
           "seconds": round(time.time() - t0)}
    live = (os.path.realpath(a.weights)
            == os.path.realpath(os.path.join(ROOT, "weights")))
    record(out, live)
    print(f"  enwik9 {out['bpb']:.4f} +/-{out['se']:.4f} bits/byte at step "
          f"{out['step']:,}, {a.pieces:,} pieces of {PIECE:,} bytes, in "
          f"{out['seconds']} s -> "
          + (f"{os.path.relpath(OUT, ROOT)} and " if live else "")
          + os.path.relpath(HIST, ROOT))


if __name__ == "__main__":
    main()
