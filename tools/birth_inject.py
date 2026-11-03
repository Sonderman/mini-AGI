#!/usr/bin/env python3
"""
Put one cohort per birth scheme into a pool, all at the same moment.

The schemes then compete on identical text, identical steps and identical
routing pressure, so whatever separates them afterwards is the scheme. Running
one condition per training run would confound the answer with everything that
differs between runs.

    python3 tools/birth_inject.py --weights weights_birth --per 4

Writes birth_cohorts.json beside the weights, naming which expert indices
belong to which scheme.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T                                     # noqa: E402
from birth_probe import capture_states, swiglu, cosf  # noqa: E402
from minagi import store as weights_store             # noqa: E402
from dataclasses import asdict                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights_birth")
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--states", type=int, default=192)
    ap.add_argument("--birth-gate", type=float, default=0.001)
    a = ap.parse_args()

    dev = torch.device("cpu")
    H, _ = capture_states(a.weights, 96, "stories", a.states)

    model, cfg, pool, man = T.build_paged(a.weights, dev)
    T.weights_store.load(model, a.weights, device=dev)
    model.eval()
    n0 = pool.n_experts()
    step = int(man["step"])
    use = pool.use.detach().cpu().numpy()
    order = np.argsort(-use)
    parent, mate = int(order[0]), int(order[1])
    print(f"  {n0} experts at step {step:,}; parent {parent}, mate {mate}",
          flush=True)

    def fetch(i):
        e = pool.tiers.fetch(int(i))
        return (e["w1"].float(), e["w3"].float(), e["w2"].float())

    P, M = fetch(parent), fetch(mate)
    with torch.no_grad():
        target = float(swiglu(H, *P).norm())
    g = torch.Generator().manual_seed(1234)
    rng = np.random.default_rng(1234)
    pool_ids = [i for i in range(n0) if i not in (parent, mate)]

    def rescale(c):
        """Match the parent's output magnitude; the map is bilinear, so sqrt."""
        with torch.no_grad():
            f = target / max(float(swiglu(H, *c).norm()), 1e-9)
        r = math.sqrt(f)
        return (c[0] * r, c[1], c[2] * r)

    def clone_abs(i, src):
        return tuple(t + 0.02 * torch.randn(t.shape, generator=g) for t in P)

    def clone_2std(i, src):
        return tuple(t + 2.0 * float(t.std()) * torch.randn(t.shape, generator=g)
                     for t in P)

    def cross50(i, src):
        m = torch.rand(P[0].shape[0], generator=g) < 0.5
        return (torch.where(m[:, None], M[0], P[0]),
                torch.where(m[:, None], M[1], P[1]),
                torch.where(m[None, :], M[2], P[2]))

    def _recomb(k):
        pick = [fetch(int(x)) for x in rng.choice(pool_ids, k, replace=False)]
        who = torch.randint(0, k, (P[0].shape[0],), generator=g)
        w1 = torch.stack([pick[who[u]][0][u] for u in range(P[0].shape[0])])
        w3 = torch.stack([pick[who[u]][1][u] for u in range(P[0].shape[0])])
        w2 = torch.stack([pick[who[u]][2][:, u] for u in range(P[0].shape[0])], 1)
        return (w1, w3, w2)

    def recomb16(i, src):
        return _recomb(16)

    def recomb16_rs(i, src):
        return rescale(_recomb(16))

    def random_rs(i, src):
        c = (torch.randn(P[0].shape, generator=g) * 0.02,
             torch.randn(P[1].shape, generator=g) * 0.02,
             torch.randn(P[2].shape, generator=g) * 0.02)
        return rescale(c)

    schemes = [("clone_abs", clone_abs), ("clone_2std", clone_2std),
               ("cross50", cross50), ("recomb16", recomb16),
               ("recomb16_rs", recomb16_rs), ("random_rs", random_rs)]

    with torch.no_grad():
        y_par = swiglu(H, *P)
    cohorts, report = {}, []
    for name, fn in schemes:
        first = pool.n_experts()
        # measure what this scheme produced, before it is written
        c = fn(0, None)
        with torch.no_grad():
            y = swiglu(H, *c)
        report.append((name, cosf(y, y_par), float(y.norm()) / target))
        pool.add_experts(a.per, device=dev, step=step,
                         birth_gate=a.birth_gate, make=fn)
        cohorts[name] = list(range(first, pool.n_experts()))
        print(f"    {name:<13} -> experts {cohorts[name][0]}..{cohorts[name][-1]}",
              flush=True)

    print(f"\n  {'scheme':<14} {'redundancy':>11} {'scale':>8}")
    for name, red, sc in report:
        print(f"  {name:<14} {red:>+11.3f} {sc:>8.2f}")

    weights_store.save(model, a.weights, step=step, val=man.get("val"),
                       cfg=asdict(cfg), verbose=True,
                       extra={k: man[k] for k in ("read_chars", "read_nats",
                                                  "context_now") if k in man})
    json.dump({"born": step, "per": a.per, "cohorts": cohorts,
               "parent": parent, "mate": mate},
              open(os.path.join(a.weights, "birth_cohorts.json"), "w"), indent=1)
    print(f"\n  {pool.n_experts()} experts now; cohorts written to "
          f"{a.weights}/birth_cohorts.json")


if __name__ == "__main__":
    main()
