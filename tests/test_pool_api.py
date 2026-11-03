#!/usr/bin/env python3
"""
The two pools answer the same calls.

`read` swaps a PagedPool in; `stream` keeps the SharedPool a fresh model is
born with. One AutoGrow and one training loop drive whichever is there, so
every call they make has to bind against both - and for a while one did not:
`prune(step, survival=...)` and `add_experts(..., birth_gate=...)` existed
only on PagedPool, so `stream` raised TypeError at its first prune. It was
never caught because nothing exercised the calls against both pools.

Binding is all this checks. Whether the two then DO the same thing is a
different question and deliberately not asserted: SharedPool prunes on the
gate because every expert there trains on every step, while PagedPool prunes
on residency because an expert off the card cannot train and its gate cannot
move. Same question, different evidence, both documented where they live.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.paged import PagedPool                          # noqa: E402
from minagi.pool import SharedPool                          # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


# Exactly what train.py and AutoGrow call on `pool`, whichever pool it is.
CALLS = [
    ("prune", (0,), {"survival": 3000}),
    ("prune", (0,), {"survival": 3000, "protect": 4}),
    ("add_experts", (1,), {"seed_from": 0, "step": 10, "birth_gate": 0.001}),
    ("n_experts", (), {}),
]

for pool in (SharedPool, PagedPool):
    for name, args, kw in CALLS:
        fn = getattr(pool, name, None)
        if fn is None:
            check(f"{pool.__name__}.{name} exists", False)
            continue
        try:
            inspect.signature(fn).bind(None, *args, **kw)
            check(f"{pool.__name__}.{name}({', '.join(kw)})", True)
        except TypeError as e:
            check(f"{pool.__name__}.{name}({', '.join(kw)})", False, str(e))

# and that it is not merely accepted but reaches the pool
p = SharedPool(32, 4, 64, 16)
before = p.n_experts()
p.add_experts(2, seed_from=0, step=5, birth_gate=0.001)
check("SharedPool grows and honours birth_gate",
      p.n_experts() == before + 2
      and abs(float(p.gate.data[-1]) - 0.001) < 1e-9,
      f"{before} -> {p.n_experts()}, gate {float(p.gate.data[-1]):.4f}")
check("SharedPool.prune accepts the window under either name",
      p.prune(0, survival=3000) == 0 and p.prune(0, min_age=3000) == 0)

print("\n  all pool-API tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
