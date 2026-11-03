#!/usr/bin/env python3
"""
Turn a training log into structured history, so a run outlives its log file.

The trainer prints its progress and keeps nothing else. That is fragile in two
ways that have both already happened here: the next run redirects over the same
filename, and two processes writing one log produced a figure that looked like
a model bug and was not. A run that takes twelve hours should not be one shell
redirect away from being unrecoverable.

This parses a log into JSON - the loss curve, the validation curve with its
error bars, per-domain losses, and every moment the pool changed size or
declined to. Safe to run against a live log; it records whatever has been
written so far.

    python3 tools/run_history.py --log runs/full.log
"""

import argparse
import json
import os
import re
import sys

STEP = re.compile(
    r"^step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+lr\s+([\d.e+-]+)\s+"
    r"gn\s+([\d.]+)(?:\s+ctx\s+(\d+)k)?\s+([\d.]+)k char/s"
    r"(?:\s+vram\s+(\d+)MB)?")
VAL = re.compile(r"val\s+([\d.]+)\s*\+/-([\d.]+)\s+ppl\s+([\d.]+)\s+(.*)$")
DOM = re.compile(r"(\w+)\s+([\d.]+)")
PRUNE = re.compile(r"pruned\s+(\d+)\s*->\s*(\d+)\s+experts")
GREW = re.compile(r"GREW\s+\+(\d+)\s*->\s*(\d+)\s+experts")
HOLD = re.compile(r"\[pool\]\s+(.*)$")
SAVED = re.compile(r"improved - weights/ advanced to val\s+([\d.]+)")
START = re.compile(r"resuming from \S+ at step\s+(\d+)")
POOL0 = re.compile(r"auto-growth armed:\s+(\d+)\s+experts")


def parse(path):
    out = {"log": os.path.basename(path), "steps": [], "vals": [], "pool": [],
           "saves": [], "start_step": None, "pool_start": None, "total": None}
    last = 0
    for raw in open(path, errors="ignore"):
        line = raw.strip()
        if re.match(r"^W\d", line):
            continue
        m = START.search(line)
        if m:
            out["start_step"] = int(m.group(1))
            continue
        m = POOL0.search(line)
        if m:
            out["pool_start"] = int(m.group(1))
            continue
        m = STEP.match(line)
        if m:
            last = int(m.group(1))
            out["total"] = int(m.group(2))
            out["steps"].append({
                "step": last, "loss": float(m.group(3)),
                "lr": float(m.group(4)), "grad_norm": float(m.group(5)),
                "context_k": int(m.group(6)) if m.group(6) else None,
                "chars_per_s": float(m.group(7)) * 1000,
                "vram_mb": int(m.group(8)) if m.group(8) else None})
            continue
        m = VAL.search(line)
        if m:
            out["vals"].append({
                "step": last, "val": float(m.group(1)),
                "stderr": float(m.group(2)), "ppl": float(m.group(3)),
                "per_domain": {k: float(v) for k, v in DOM.findall(m.group(4))}})
            continue
        m = PRUNE.search(line)
        if m:
            out["pool"].append({"step": last, "event": "pruned",
                                "n": int(m.group(1)), "to": int(m.group(2))})
            continue
        m = GREW.search(line)
        if m:
            out["pool"].append({"step": last, "event": "grew",
                                "n": int(m.group(1)), "to": int(m.group(2))})
            continue
        m = HOLD.search(line)
        if m:
            out["pool"].append({"step": last, "event": "held",
                                "reason": m.group(1)})
            continue
        m = SAVED.search(line)
        if m:
            out["saves"].append({"step": last, "val": float(m.group(1))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/full.log")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    h = parse(a.log)
    out = a.out or os.path.join(
        "runs", "results",
        os.path.splitext(os.path.basename(a.log))[0] + "_history.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(h, f, indent=1)
    v = h["vals"]
    sizes = [p["to"] for p in h["pool"] if "to" in p]
    print(f"  {len(h['steps'])} loss points, {len(v)} validation points, "
          f"{len(h['pool'])} pool events, {len(h['saves'])} saves")
    if v:
        print(f"  val {v[0]['val']:.4f} -> {v[-1]['val']:.4f}  "
              f"(best {min(x['val'] for x in v):.4f})")
    if sizes:
        print(f"  pool {h['pool_start']} -> {sizes[-1]} experts")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
