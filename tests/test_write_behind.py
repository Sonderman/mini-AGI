#!/usr/bin/env python3
"""
Evicted experts are written on a thread of their own, and nothing can tell.

What these tests hold it to:
  an expert evicted and not yet on disk is served from memory - never from
  the older file - and counts as the RAM hit it is
  once the writer has landed it, the file holds exactly what was evicted
  two evictions of one expert land in the order they were made
  flush lands what the writer holds before anything newer in RAM, so the
  file ends as the newest
  a pruned expert's file is not put back by a write still queued for it
  a write that fails is raised, not lost
  RAM keeps moments as bfloat16, and the optimiser gets them back widened
"""
import os
import sys
import tempfile
import threading

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from minagi.paged import Tiers                              # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


DM, DF = 8, 4


def entry(v):
    """An expert whose every weight is v, with moments."""
    return {"w1": torch.full((DF, DM), v), "w3": torch.full((DF, DM), v),
            "w2": torch.full((DM, DF), v),
            "w1_m": torch.full((DF, DM), v, dtype=torch.bfloat16),
            "w1_v": torch.full((DF, DM), v, dtype=torch.bfloat16)}


def on_disk(t, i):
    return float(t._from_disk(i)["w1"][0, 0])


with tempfile.TemporaryDirectory() as d:
    t = Tiers(d, DM, DF, ram_capacity=1)
    for i in range(3):
        t._to_disk(i, entry(float(i)))          # every expert starts on disk
    # hold the writer back, so an eviction is caught before it lands
    gate = threading.Event()
    real = t._to_disk

    def slow(i, ent):
        gate.wait(10)
        real(i, ent)
    t._to_disk = slow

    t.fetch(0)
    t.put(0, entry(10.0), dirty=True)          # trained: 0 is now 10.0
    t.fetch(1)                                 # evicts 0 to the writer
    check("an evicted expert waits for the writer, and its file is still the "
          "old one", 0 in t.pending and on_disk(t, 0) == 0.0)
    hits = t.hits
    got = t.fetch(0)
    check("...a fetch meanwhile gets what was evicted, from memory, as a hit",
          float(got["w1"][0, 0]) == 10.0 and t.hits == hits + 1)
    gate.set()
    t.drain()
    check("once landed, the file holds exactly what was evicted",
          on_disk(t, 0) == 10.0 and not t.pending)

    # two evictions of one expert, the first held back: the second must win
    gate.clear()
    t.put(0, entry(20.0), dirty=True)
    t.fetch(1)                                 # evicts 0 (20.0)
    t.fetch(0)                                 # back from pending
    t.put(0, entry(30.0), dirty=True)
    t.fetch(2)                                 # evicts 0 again (30.0)
    gate.set()
    t.drain()
    check("two evictions of one expert land in the order they were made",
          on_disk(t, 0) == 30.0)

    # flush: something newer in RAM than what the writer holds
    gate.clear()
    t.fetch(0)
    t.put(0, entry(40.0), dirty=True)
    t.fetch(1)                                 # 40.0 to the writer
    t.fetch(0)
    t.put(0, entry(50.0), dirty=True)          # newer, dirty in RAM
    threading.Timer(0.3, gate.set).start()
    t.flush()
    check("flush lands the writer's first, so the newest is what stays",
          on_disk(t, 0) == 50.0 and not t.dirty and not t.pending)

    # a failing write is raised by the next drain, not swallowed
    def broken(i, ent):
        raise OSError("disk full")
    t._to_disk = broken
    t.fetch(0)
    t.put(0, entry(60.0), dirty=True)
    t.fetch(1)
    try:
        t.drain()
        raised = False
    except OSError:
        raised = True
    check("a write that fails is raised by the next drain", raised)
    t._to_disk = real

with tempfile.TemporaryDirectory() as d:
    # the moments: bfloat16 in RAM, as they are on disk
    t = Tiers(d, DM, DF, ram_capacity=4)
    e = entry(0.0)
    e["w1_m"] = torch.randn(DF, DM)            # fp32, as an optimiser holds them
    e["w1_v"] = torch.rand(DF, DM)
    t._to_disk(5, e)
    back = t.fetch(5)
    check("RAM keeps the moments as bfloat16, the width the file keeps",
          back["w1_m"].dtype == torch.bfloat16 and back["w1"].dtype == torch.float32)
    check("...exactly the values written",
          torch.equal(back["w1_m"], e["w1_m"].to(torch.bfloat16)))

# a pruned expert stays deleted
from minagi.paged import PagedPool                          # noqa: E402
d = tempfile.mkdtemp()
pool = PagedPool(d, DM, DF, n_experts=4, resident=2, ram_capacity=1)
for i in range(4):
    pool.tiers._to_disk(pool._f(i), entry(float(i)))
gate = threading.Event()
real = pool.tiers._to_disk


def slow(i, ent):
    gate.wait(10)
    real(i, ent)
pool.tiers._to_disk = slow
u = pool._f(3)
pool.tiers.fetch(u)
pool.tiers.put(u, entry(99.0), dirty=True)
pool.tiers.fetch(pool._f(0))                   # expert 3 goes to the writer
threading.Timer(0.3, gate.set).start()
pool._keep_only([0, 1, 2])
check("a pruned expert's file is not put back by a write queued for it",
      not os.path.exists(pool.tiers._file(u)))

print("\n  all write-behind tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
