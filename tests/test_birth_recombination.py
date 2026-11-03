#!/usr/bin/env python3
"""
A newborn expert must be something the pool does not already compute.

The old default cloned the busiest expert and added noise, which measured 0.98
output cosine to its parent while two established experts sit at 0.04 - so it
duplicated rather than added, and 201 of 224 such experts were pruned. These
tests hold the replacement to the property that matters: every hidden unit is
a trained one, and the composition is not any single source.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.paged import PagedPool                       # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


D_MODEL, D_FF, N = 32, 64, 12
d = tempfile.mkdtemp()
try:
    # a pool of distinguishable experts: expert i is built from value i, so
    # any unit can be traced back to the expert it came from
    for i in range(N):
        np.savez(os.path.join(d, "e%05d.npz" % i),
                 w1=np.full((D_FF, D_MODEL), i + 1, np.float32),
                 w3=np.full((D_FF, D_MODEL), i + 1, np.float32),
                 w2=np.full((D_MODEL, D_FF), i + 1, np.float32))
    p = PagedPool(d, D_MODEL, D_FF, N, resident=4, device="cpu")
    p.add_experts(3, seed_from=0, step=100, recombine=8)
    check("the pool grew", p.n_experts() == N + 3, f"{N} -> {p.n_experts()}")

    kids = [p.tiers.fetch(i) for i in range(N, N + 3)]
    src = {float(i + 1) for i in range(N)}
    all_units = [set(np.unique(k["w1"].numpy())) for k in kids]
    check("every unit came from a real expert",
          all(u <= src for u in all_units))
    check("no child is a copy of one expert",
          all(len(u) > 1 for u in all_units),
          f"distinct sources per child: {[len(u) for u in all_units]}")
    check("the seed is represented", any(1.0 in u for u in all_units))
    check("children differ from each other",
          not np.array_equal(kids[0]["w1"].numpy(), kids[1]["w1"].numpy()))

    # w1[u], w3[u] and w2[:, u] must come from the SAME expert, or the unit is
    # not a unit any more and its learned function is destroyed
    aligned = True
    for k in kids:
        a = k["w1"].numpy()[:, 0]
        b = k["w3"].numpy()[:, 0]
        c = k["w2"].numpy()[0, :]
        aligned = aligned and np.array_equal(a, b) and np.array_equal(a, c)
    check("w1, w3 and w2 stay aligned per unit", aligned)

    # and the newborn must be gated below the bar prune reads as dead
    check("born quiet", float(p.gate.data[-1]) < 0.005,
          f"gate {float(p.gate.data[-1]):.4f}")
    check("born stamped with the step", float(p.born[-1]) == 100.0)

    # the old behaviour is still reachable, and still what it was
    p2 = PagedPool(d, D_MODEL, D_FF, N, resident=4, device="cpu")
    p2.add_experts(1, seed_from=3, step=1, recombine=0)
    kid = p2.tiers.fetch(N)["w1"].numpy()
    check("recombine=0 falls back to the clone",
          abs(float(np.mean(kid)) - 4.0) < 0.5,
          f"mean {float(np.mean(kid)):.2f} (parent is 4.0)")
finally:
    shutil.rmtree(d, ignore_errors=True)

print("\n  all birth tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
