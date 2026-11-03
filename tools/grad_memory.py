#!/usr/bin/env python3
"""
How long a trunk gradient stays useful, and how much domains disagree.

AdEMAMix adds a SLOW exponential average of the gradient on the argument that
old gradients remain informative for far longer than beta1=0.9 assumes. That
argument is testable rather than assumed: if the trunk gradient decorrelates
in fifty steps, a ten-thousand-step average is averaging noise; if it stays
correlated for thousands, the slow term is reconstructing the multi-domain
gradient this reader can never compute directly, because one KV cache is
1.74 GB and seven lanes will not fit.

Two things measured, on the trunk only - it carries 97.6% of the gradient
norm (tools/clip_probe.py):

  autocorrelation   cos(g_t, g_t+k) against lag k, WITHIN one subject
  cross-domain      cos(g_A, g_B) for every pair of subjects

    python3 tools/grad_memory.py --lags 1,2,4,8,16,32,64,128
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T                                    # noqa: E402
from minagi.ingest import as_stream, collect         # noqa: E402
from minagi.precision import amp, set_compute_dtype  # noqa: E402


def flat_grad(ps):
    return torch.cat([(p.grad.detach().float().reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel(), device=p.device))
                      for p in ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--lags", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()
    lags = [int(x) for x in a.lags.split(",")]

    dev = torch.device("cuda"); set_compute_dtype("bf16")
    model, cfg, pool, man = T.build_paged(a.weights, dev, read_only=True)
    T.weights_store.load(model, a.weights, device=dev)
    trunk, _ = T._split_trunk_pool(model)
    model.train()
    print(f"  {a.weights} step {man['step']:,}, trunk {sum(p.numel() for p in trunk)/1e6:.2f}M\n",
          flush=True)

    subs = ["stories", "code", "chess", "reasoning", "wikipedia", "chat",
            "arithmetic"]
    # Most subjects are many short files, so one file cannot reach the lags
    # that matter. Concatenate until the stream is long enough - which is what
    # the reader does anyway, document boundaries and all.
    need = a.chunk * (max(lags) + 4)
    stream = {}
    for s in subs:
        fs = sorted(f for f in collect(["data/train"]) if f"/{s}/" in f)
        buf = []
        tot = 0
        for f in fs:
            d = as_stream(f)
            buf.append(np.asarray(d, np.int64))
            tot += len(d)
            if tot >= need:
                break
        if buf:
            stream[s] = np.concatenate(buf)
    subs = [s for s in subs if s in stream and len(stream[s]) >= a.chunk + 2]
    print("  streams: " + "  ".join(f"{s} {len(stream[s])/1e6:.2f}M"
                                    for s in subs) + "\n", flush=True)

    def grad_at(sub, start):
        d = stream[sub]
        start = start % max(1, len(d) - a.chunk - 1)
        arr = np.asarray(d[start:start + a.chunk + 1], np.int64)
        if len(arr) < a.chunk + 1:
            return None
        x = torch.tensor(arr[:-1][None], device=dev)
        y = torch.tensor(arr[1:][None], device=dev)
        model.zero_grad(set_to_none=True)
        with amp(dev):
            _, loss = model(x, y)
        loss.backward()
        return flat_grad(trunk)

    cos = lambda u, v: float(torch.nn.functional.cosine_similarity(u[None], v[None]))

    # ---- 1. autocorrelation within a subject -----------------------------
    print("  === how long a trunk gradient stays useful (within one subject) ===")
    print(f"  {'lag (steps)':>12} {'cosine':>9}   {'~characters':>12}")
    rows = []
    for k in lags:
        vals = []
        for r in range(a.reps):
            s = subs[r % len(subs)]
            base = 10000 + r * 37 * a.chunk
            g0 = grad_at(s, base)
            g1 = grad_at(s, base + k * a.chunk)
            if g0 is not None and g1 is not None:
                vals.append(cos(g0, g1))
        if vals:
            m = float(np.mean(vals))
            rows.append((k, m))
            print(f"  {k:>12} {m:>9.4f}   {k*a.chunk:>12,}", flush=True)

    # ---- 2. do different subjects agree? ---------------------------------
    print("\n  === do subjects pull the trunk the same way? ===")
    gs = {}
    for s in subs:
        g = grad_at(s, 50000)
        if g is not None:
            gs[s] = g
    ks = list(gs)
    print("  " + " " * 12 + "".join(f"{s[:6]:>8}" for s in ks))
    off = []
    for i, s in enumerate(ks):
        line = f"  {s[:11]:<12}"
        for j, t in enumerate(ks):
            c = cos(gs[s], gs[t])
            line += f"{c:>8.3f}"
            if i < j:
                off.append(c)
        print(line, flush=True)
    if off:
        print(f"\n  cross-domain cosine: mean {np.mean(off):+.4f}  "
              f"min {np.min(off):+.4f}  max {np.max(off):+.4f}")
        mo = float(np.mean(off))
        if mo > 0.05:
            verdict = "POSITIVE - domains share a direction a slow EMA accumulates"
        elif mo > -0.05:
            verdict = "NEAR ZERO - a slow EMA builds the balanced multi-domain gradient"
        else:
            verdict = "NEGATIVE - domains fight; averaging cancels them"
        print(f"  {verdict}")
    if rows:
        good = [k for k, m in rows if m > 0.05]
        if good:
            b3 = 1 - 1.0 / max(good)
            print(f"\n  the gradient is still correlated at lag {max(good)}, "
                  f"so beta3 around {b3:.4f} is what the data supports")


if __name__ == "__main__":
    main()
