#!/usr/bin/env python3
"""
The routers must be built at exactly the expert count the checkpoint holds.

pool_max does two jobs that look like one, and conflating them broke loading.
It sizes the per-token routers, which keep one row per EXPERT, so a saved
router only fits a model built at the same count - one row too many and
load_state_dict refuses the entire model, which is what happened:

    size mismatch for recur.0.mlp.router.weight: copying a param with shape
    torch.Size([157, 512]) from checkpoint, the shape in current model is
    torch.Size([158, 512])

And it had also been used as the growth CEILING, where equalling the current
size means room = n < max_experts is false forever. Fixing the ceiling by
widening pool_max broke the sizing; the ceiling now lives elsewhere.

The manifest's pool_max ratchets up and never comes down, so after a prune it
is larger than the pool. Sizing must follow n_experts, never that.

What these tests hold it to, loading a real weights directory with
train.py's build_paged:
  a manifest whose pool_max ratcheted above the pool, as after a prune,
  still loads, with one router row per expert
  so does one whose pool grew past an old, smaller pool_max
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as T                                           # noqa: E402
from minagi.create import create                            # noqa: E402
from minagi.pool import PooledMLP                           # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-router-width-")
try:
    w = os.path.join(tmp, "w")
    create(w, seed=0, verbose=False, experts=12, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=128, max_steps=3,
           top_k=2, depth=1)
    path = os.path.join(w, "manifest.json")
    with open(path) as f:
        man = json.load(f)
    n = int(man["n_experts"])
    for name, pool_max in (
            ("a pool_max ratcheted above the pool, as after a prune", n + 1),
            ("a pool grown past an old, smaller pool_max", n - 4)):
        man.setdefault("cfg", {})["pool_max"] = pool_max
        with open(path, "w") as f:
            json.dump(man, f)
        try:
            m, cfg, pool, _ = T.build_paged(w, "cpu", ram_capacity=4)
            rows = {int(s.router.weight.shape[0]) for s in m.modules()
                    if isinstance(s, PooledMLP)}
            good = rows == {n} and pool.n_experts() == n
            said = f"routers {sorted(rows)}, pool {pool.n_experts()}"
        except Exception as e:                      # noqa: BLE001
            good, said = False, f"{type(e).__name__}: {str(e)[:120]}"
        check(f"{name} still loads, one router row per expert", good, said)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all router-width tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
