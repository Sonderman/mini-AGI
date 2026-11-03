#!/usr/bin/env python3
"""
An expert cannot have been ignored for longer than it has existed.

`last_seen` answers two questions that are not the same question. As an
exploration key it wants a newborn at the FRONT of the queue, so it is
initialised to 0 - "wanted at segment zero", the oldest possible claim. As a
staleness key it is read as "how long since anything asked for it", and there
zero means the opposite: ignored for the entire run.

So every newborn arrived already scored as maximally stale. Measured on the
live pool at step 55,793: a brand new expert read 1.143 of the survival
window, ABOVE the 1.0 line prune deletes on, from the instant it existed. It
showed in the near-deletion panel as a tall bar that appeared at each birth
and vanished the moment the expert was first admitted - which is exactly what
it looked like, an intermittent phantom.

`young` kept it from actually being deleted, so this cost no experts. The fix
is the clamp, not a different initialiser: exploration still wants the 0.
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
from minagi.paged import PagedPool                            # noqa: E402
import plot_experts as PE                                     # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


D = "/tmp/minagi_newborn_test"
N, DM, DF = 6, 8, 16
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)
for i in range(N):
    np.savez(os.path.join(D, f"e{i:05d}.npz"),
             w1=np.zeros((DF, DM), np.float32), w3=np.zeros((DF, DM), np.float32),
             w2=np.zeros((DM, DF), np.float32))

# the live numbers this was found on
STEP, SEG, BORN = 55_793, 98_298, 55_700
pool = PagedPool(D, DM, DF, N, resident=2, ram_capacity=2, device="cpu")
pool.now, pool.trial, pool.segments = STEP, STEP - 6_965, SEG
pool.slots = [-1, -1]
pool.born.fill_(0.0)
pool.last_seen.fill_(float(SEG))
pool.born[5], pool.last_seen[5] = float(BORN), 0.0   # newborn, never yet admitted

d = pool.dying()
check("a newborn is not scored stale for time before it existed",
      float(d[5]) < 1.0, f"{float(d[5]):.4f} of the window, was 1.143")
check("...and specifically cannot exceed the delete line on day one",
      float(d[5]) <= (STEP - BORN) / max(STEP, 1) + 1e-6,
      "bounded by its own age as a share of the run")
check("a founder that was just used still reads as fresh",
      float(d[0]) < 1e-6, f"{float(d[0]):.4f}")

# ------------------------------------ the exploration key is NOT what changed
check("last_seen stays 0 for a newborn, so exploration still reaches it first",
      float(pool.last_seen[5]) == 0.0,
      "the clamp is in the reader; the initialiser carries the other meaning")

# ------------------------------------------- a genuinely idle OLD expert still counts
pool.born[4], pool.last_seen[4] = 0.0, 0.0          # present from the start, long gone
check("an expert old enough to have really been idle is still scored stale",
      float(pool.dying()[4]) > 1.0, f"{float(pool.dying()[4]):.4f}",)

# ------------------------------------------------------ and the panel agrees
now = {"gate": [0.1] * N, "segments": SEG, "step": STEP,
       "last_seen": [float(SEG)] * 4 + [0.0, 0.0],
       "born": [0.0] * 5 + [float(BORN)]}
C = {"survival_chars": 100_000_000, "dying_at": 0.65, "chunk": 2048,
     "top_k": 8, "resident": 32}
frac, young = PE.dying(now, C, mask_young=False)
check("the near-deletion panel draws the newborn below the line",
      frac[5] < 1.0, f"{frac[5]:.4f} - no more phantom bar above 'deleted'")
check("...while still drawing the genuinely idle one above it",
      frac[4] > 1.0, f"{frac[4]:.4f}")

shutil.rmtree(D, ignore_errors=True)
print("\n  all newborn-staleness tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
