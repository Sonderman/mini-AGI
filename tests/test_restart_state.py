#!/usr/bin/env python3
"""
What a restart must carry, and what it silently dropped.

The symptom was specific and repeatable: stop the run, start it again, and
held-out jumps while experts die and are reborn in bulk. The cause was not the
weights - those round-trip fine - but the buffers the pool DECIDES with, which
are all `persistent=False` and so never appear in a state_dict.

Selection itself now reads only the router rows, which are saved with the
model, so a restart cannot change which experts a text asks for. What the
pool still decides with - the prune clock `last_seen`, which also orders
eviction from the card, and `gate_seen` - must round-trip through the
telemetry, and this checks that it does.

  age         incremented every step and read by nothing. Left alone, and
              recorded here so the next reader does not go looking.
"""
import os, shutil, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.paged import PagedPool

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


d = "/tmp/minagi_restart_test"
shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
N, DM, DF = 16, 8, 16
rng = np.random.default_rng(0)
for i in range(N):                     # every expert genuinely different
    np.savez(os.path.join(d, f"e{i:05d}.npz"),
             w1=rng.normal(0, 1, (DF, DM)).astype(np.float32),
             w3=rng.normal(0, 1, (DF, DM)).astype(np.float32),
             w2=rng.normal(0, 1, (DM, DF)).astype(np.float32))

live = PagedPool(d, DM, DF, N, resident=4, ram_capacity=4, device="cpu")
live.ever[:] = True
live.last_seen.copy_(torch.arange(N, dtype=torch.float))
for i in range(N):
    live.gate.data[i] = 0.1 + 0.05 * i
live.gate_seen = live.gate.data.abs().clone()
tel = live.telemetry()

# ------------------------------------------ 1. the prune clock round-trips
fresh = PagedPool(d, DM, DF, N, resident=4, ram_capacity=4, device="cpu")
fresh.load_telemetry(tel)
check("last_seen survives a restart - it is what prune and eviction read",
      torch.allclose(fresh.last_seen, live.last_seen),
      "zeroed, every expert would look unasked-for since the run began")

# ------------------------------------------ 3. gate_seen round-trips now
check("gate_seen survives a restart",
      torch.allclose(fresh.gate_seen, live.gate_seen, atol=1e-5),
      "so `flat` measures the change since the last prune, not since zero")

# --------------------------- 4. every buffer a decision reads is carried
carried = set(tel)
reads = {"gate", "gate_seen", "born", "use", "admits",
         "last_seen", "ever", "uid", "recent"}
missing = sorted(reads - carried)
check("every buffer prune() or admit() reads is carried across a restart",
      not missing, f"missing: {missing}" if missing else f"{len(reads)} of {len(reads)}")

# ------------------------------------------------- 5. age really is dead
import inspect                                              # noqa: E402
src = inspect.getsource(PagedPool)
uses = [l.strip() for l in src.split("\n")
        if ".age" in l and "min_age" not in l and "self.age =" not in l
        and "self.age +=" not in l]
check("age is written and never read, so losing it costs nothing",
      not uses, f"unexpected readers: {uses}" if uses else "confirmed dead")

print("\n  all restart-state tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
