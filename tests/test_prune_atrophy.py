#!/usr/bin/env python3
"""
Use it or lose it: an expert is deleted for not being CHOSEN. The gate is
not consulted at all, in either direction.

THE BUG THIS IS WRITTEN AGAINST. Pruning used to need a third condition
besides "old" and "gate below the dead line": `flat`, meaning the gate had not
risen since the last check. That condition was a trap. An expert that is not
resident does not train, so its gate CANNOT rise - so `flat` was automatically
true for exactly the experts being starved of the card, and the low gate had
the same cause. An expert failed two of the three tests through neglect alone,
and the third was only a clock.

Measured on this run's own history, during a pool collapse from 235 experts to
192 while held-out lost 0.13 nats: of the 68 experts deleted, 57 had been
admitted to the card within 8,600 steps of their deletion, and a QUARTER had
been routed to in the very interval they were killed in. Only 11 had genuinely
never been seen.

The rule now asks whether an expert was chosen, which is a thing neglect
cannot fake, and gives every newborn one absolute grace period first.
"""
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.paged import PagedPool                       # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


D = "/tmp/minagi_atrophy_test"
N, DM, DF, W = 12, 8, 16, 1000


def pool():
    shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
    for i in range(N):
        np.savez(os.path.join(D, f"e{i:05d}.npz"),
                 w1=np.full((DF, DM), i + 1, np.float32),
                 w3=np.full((DF, DM), i + 1, np.float32),
                 w2=np.full((DM, DF), i + 1, np.float32))
    p = PagedPool(D, DM, DF, N, resident=2, ram_capacity=2, device="cpu")
    p.slots = [-1, -1]                    # nothing resident, so nothing immune
    p.gate.data.copy_(torch.full((N,), 0.30))
    p.gate.data[:4] = 0.001               # four with a dead gate
    p.born.fill_(0.0)                     # all long past their grace
    p.segments = 10 ** 5                  # 1 segment per step at step 1e5
    p.last_seen.fill_(0.0)                # ...and none chosen for a long time
    return p


# ------------------------------- 1. a dead gate alone is not enough to die
p = pool()
p.last_seen.fill_(10 ** 5)                # every one of them was just chosen
gone = p.prune(step=10 ** 5, survival=W)
check("an expert that is still being chosen is never deleted", gone == 0,
      "4 sit below the dead line and all 4 survive")

# ---------------------------------- 2. being unchosen alone is not enough
p = pool()
p.gate.data.copy_(torch.full((N,), 0.30))     # every gate healthy
gone = p.prune(step=10 ** 5, survival=W)
check("a healthy gate does NOT save an expert that has gone quiet",
      gone == N - 1, f"every gate at 0.30, none chosen in 100,000 steps, {gone} of {N} "
      f"deleted - the collapse guard keeps the last one")

# ------------------------------------------- 3. both together, and only both
p = pool()
p.last_seen[:2] = 10 ** 5                 # two of the twelve chosen recently
gone = p.prune(step=10 ** 5, survival=W)
check("the unchosen are deleted and the gate plays no part", gone == N - 2,
      f"2 of {N} chosen recently, {gone} deleted, gates spanning 0.001 to 0.30")

# --------------------------------------- 4. the grace period is unconditional
p = pool()
p.born.fill_(10 ** 5 - 1)                 # every expert one step old
gone = p.prune(step=10 ** 5, survival=W)
check("nothing is judged inside its survival window", gone == 0,
      "dead gate, never chosen, one step old - still safe")

p = pool()
p.born.fill_(10 ** 5 - W - 1)             # one step past it
gone = p.prune(step=10 ** 5, survival=W)
check("...and is judged the moment it ends", gone == N - 1,
      f"one step past the window, {gone} deleted")

# ------------------------- 5. the window is read in steps, not in segments
# `last_seen` counts segments and `survival` is in steps, so the pool converts
# with its own cumulative ratio. Get this wrong and the window is off by the
# segments-per-step factor, which on the live run is 1.68.
p = pool()
p.segments = 2 * 10 ** 5                  # 2 segments per step
p.last_seen.fill_(2 * 10 ** 5 - 2 * (W - 10))   # chosen (W-10) STEPS ago
gone = p.prune(step=10 ** 5, survival=W)
check("staleness is measured in steps however fast segments run", gone == 0,
      "chosen 990 steps ago against a 1,000-step window")

p = pool()
p.segments = 2 * 10 ** 5
p.last_seen.fill_(2 * 10 ** 5 - 2 * (W + 10))   # chosen (W+10) steps ago
gone = p.prune(step=10 ** 5, survival=W)
check("...and one step over the line still counts as atrophied", gone == N - 1,
      "chosen 1,010 steps ago against a 1,000-step window")

# ------------- 5b. the window is TRAILING: age earns nothing permanent
# An expert that worked for a very long time and has since gone quiet is not
# protected by its history. Only by having been chosen recently.
p = pool()
p.born.fill_(0.0)                         # alive since step 0 - 100,000 steps
p.use.fill_(10 ** 9)                      # and enormously used, historically
p.last_seen.fill_(0.0)                    # but not chosen in a very long time
gone = p.prune(step=10 ** 5, survival=W)
check("a long-lived expert that has gone quiet still dies", gone == N - 1,
      "100,000 steps old, 1e9 cumulative uses, unchosen for the last 1,000")

p = pool()
p.born.fill_(0.0); p.use.fill_(10 ** 9)
p.last_seen.fill_(10 ** 5 - (W - 10))     # ...but chosen inside the window
gone = p.prune(step=10 ** 5, survival=W)
check("...and survives if it was chosen inside the window", gone == 0,
      "chosen 990 steps ago against a 1,000-step window")

# ------------------------------------ 6. a pool of zero cannot route at all
p = pool()
p.gate.data.fill_(0.0001)
gone = p.prune(step=10 ** 5, survival=W)
check("an override that would delete everyone still leaves one",
      p.n_experts() == 1 and gone == N - 1)

# ------------- 7. the replaced condition is gone from the decision entirely
import inspect                                            # noqa: E402
src = inspect.getsource(PagedPool.prune)
# only the DECISION, not the reindexing that follows it
decision = src.split('"""')[2].split("if len(keep)")[0]
check("the decision no longer reads gate_seen", "gate_seen" not in decision,
      "a starved expert's gate cannot rise, so that test only measured neglect")
check("...nor min_delta", "min_delta" not in src)
check("the decision reads last_seen", "last_seen" in decision,
      "being chosen is the thing neglect cannot fake")
check("...and no longer reads the gate at all",
      "gate" not in decision
      and "min_gate" not in inspect.signature(PagedPool.prune).parameters,
      "lowest-|gate| picked the experts that would go on to do NOTHING at 8% "
      "against a 63% base rate - it is anti-predictive, so it is gone")
# gate_seen is still carried through growth and reindexing, and is now dead
# state like `age`: written, never read. Left alone deliberately - removing it
# touches four buffer lists in train.py and recur.py for no behaviour change.

print("\n  all atrophy tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
