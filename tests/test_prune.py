#!/usr/bin/env python3
"""
Pruning: one definition of a dead gate, stable identity, and moments that
survive the operation.

Historic note: saturation() used to call an expert idle below
0.15 of the median gate while prune deleted only below an absolute 0.005.
Measured on the live pool, 44 of 175 experts sat in the gap - counted as idle
capacity so growth was blocked, and immune to deletion so the block could
never clear. Test 2 is the one that stops that coming back.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.paged import PagedPool                       # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


DM, DF = 16, 32


def build(d, n):
    for i in range(n):
        np.savez(os.path.join(d, "e%05d.npz" % i),
                 w1=np.full((DF, DM), i + 1, np.float32),
                 w3=np.full((DF, DM), i + 1, np.float32),
                 w2=np.full((DM, DF), i + 1, np.float32))
    return PagedPool(d, DM, DF, n, resident=2, device="cpu")


d = tempfile.mkdtemp()
try:
    # -------------------------- 1. dying() is staleness, scaled by the window
    p = build(d, 10)
    p.gate.data.copy_(torch.full((10,), 0.30))
    p.born.fill_(0.0); p.now = 100000; p.trial = 1000
    p.segments = 100000                     # 1 segment per step
    p.last_seen.fill_(100000 - 500)         # half way to the line
    d1 = p.dying()
    check("an expert half way to the prune line reads 0.5",
          abs(float(d1[0]) - 0.5) < 1e-6, f"{float(d1[0]):.4f}")
    p.last_seen.fill_(100000 - 1000)        # exactly at it
    check("...and 1.0 at the line itself",
          abs(float(p.dying()[0]) - 1.0) < 1e-6, f"{float(p.dying()[0]):.4f}")
    p.gate.data.mul_(10.0)
    check("the gate does not enter into it at all",
          abs(float(p.dying()[0]) - 1.0) < 1e-6, "gate x10, dying unchanged")
    p.born.fill_(100000 - 10)               # newborn, inside its trial
    check("a newborn inside its trial is never dying",
          float(p.dying().max()) == 0.0, "born 10 steps ago against a 1,000 trial")

    # ------------------ 2. the brake and the pruner read the SAME staleness
    # They disagreed for a long time - the brake asked about gates, prune about
    # being chosen - and the gate half was measurably backwards. Now there is
    # one quantity and two thresholds on it: dying_at stops growth, the window
    # itself deletes.
    p = build(d, 20)
    p.gate.data.copy_(torch.full((20,), 0.30))
    p.born.fill_(0.0); p.slots = [-1, -1]
    p.now = 100000; p.trial = 1000; p.dying_at = 0.75
    p.segments = 100000
    p.last_seen.fill_(100000 - 800)         # 0.8 of the way: dying, not dead
    idle = p.saturation()["idle"]
    gone = p.prune(step=100000, survival=1000)
    check("past dying_at the brake counts them, and prune still keeps them",
          idle == 20 and gone == 0, f"idle {idle}, pruned {gone}")
    p.last_seen.fill_(100000 - 1200)        # past the window itself
    gone = p.prune(step=100000, survival=1000)
    check("...and past the window they go", gone == 19,
          f"{gone} of 20 deleted")

    # --------------------------- 3. a CHOSEN expert is protected, however low
    # This used to protect an expert whose gate was still RISING. That test was
    # a trap: an expert that is not resident cannot train, so its gate cannot
    # rise, and the protection was unavailable to exactly the experts being
    # starved of the card. Being chosen is the thing neglect cannot fake.
    p = build(d, 20)
    g = torch.full((20,), 0.30); g[:4] = 0.02
    p.gate.data.copy_(g); p.born.fill_(0.0); p.slots = [-1, -1]
    p.segments = 100000
    p.last_seen.fill_(0.0)
    p.last_seen[:2] = 99995            # these two were on the card just now
    gone = p.prune(step=100000, survival=10)
    check("an expert chosen recently survives, however low its gate",
          gone == 18 and p.n_experts() == 2,
          f"2 of 20 chosen recently, {gone} deleted, the 2 survivors at gate 0.02")

    # ------------------------------------ 3b. and the grace period is absolute
    p = build(d, 20)
    g = torch.full((20,), 0.30); g[:4] = 0.02
    p.gate.data.copy_(g); p.slots = [-1, -1]
    p.segments = 100000; p.last_seen.fill_(0.0)
    p.born.fill_(99995.0)              # every expert is newborn
    gone = p.prune(step=100000, survival=10)
    check("nothing is touched inside its survival window", gone == 0,
          "born 5 steps ago, dead gate, never chosen - still safe")

    # ------------------------------------------------ 4. identity is stable
    p = build(d, 10)
    p.gate.data.copy_(torch.full((10,), 0.30))
    p.born.fill_(0.0); p.slots = [-1, -1]
    p.segments = 100000
    p.last_seen.fill_(100000); p.last_seen[3] = 0.0   # only 3 has gone quiet
    before = {int(p.uid[i]): float(p.tiers.fetch(int(p.uid[i]))["w1"][0, 0])
              for i in range(10)}
    p.prune(step=100000, survival=10)
    uids = [int(u) for u in p.uid]
    check("the pruned uid is gone and no other moved",
          3 not in uids and uids == [0, 1, 2, 4, 5, 6, 7, 8, 9], f"{uids}")
    same = all(float(p.tiers.fetch(u)["w1"][0, 0]) == before[u] for u in uids)
    check("every survivor still holds its own weights", same)
    check("the deleted file is gone",
          not os.path.exists(p.tiers._file(3)))
    check("no survivor's file was renamed",
          all(os.path.exists(p.tiers._file(u)) for u in uids))

    # ------------------------------------ 5. moments survive a prune
    p = build(d, 10)
    p.gate.data.copy_(torch.full((10,), 0.30))
    opt = torch.optim.AdamW([{"params": [p.gate], "name": "pool", "lr": 1e-3}])
    p.attach_optimiser(opt)
    for _ in range(5):
        opt.zero_grad(); p.gate.sum().backward(); opt.step()
    m_before = opt.state[p.gate]["exp_avg"].clone()
    p.born.fill_(0.0); p.slots = [-1, -1]
    p.segments = 100000
    p.last_seen.fill_(100000); p.last_seen[2] = 0.0   # only 2 has gone quiet
    p.prune(step=100000, survival=10)
    st = opt.state.get(p.gate, {})
    keep = [i for i in range(10) if i != 2]
    check("Adam's moments survive a prune", bool(st),
          f"exp_avg len {len(st.get('exp_avg', []))} for {p.n_experts()} experts")
    check("...and are the right rows",
          bool(st) and torch.allclose(st["exp_avg"], m_before[keep]))

    # ------------------------------------ 6. moments survive a growth
    p = build(d, 10)
    opt = torch.optim.AdamW([{"params": [p.gate], "name": "pool", "lr": 1e-3}])
    p.attach_optimiser(opt)
    for _ in range(5):
        opt.zero_grad(); p.gate.sum().backward(); opt.step()
    m_before = opt.state[p.gate]["exp_avg"].clone()
    p.add_experts(3, step=1, recombine=4)
    st = opt.state.get(p.gate, {})
    check("Adam's moments survive a growth",
          bool(st) and st["exp_avg"].shape[0] == 13,
          f"{m_before.shape[0]} -> {st['exp_avg'].shape[0] if st else 0}")
    check("...the old rows unchanged, the new ones at zero",
          bool(st) and torch.allclose(st["exp_avg"][:10], m_before)
          and float(st["exp_avg"][10:].abs().sum()) == 0.0)

    # ------------------------------- 7. ids keep counting up past a prune
    p = build(d, 6)
    p.add_experts(2, step=1, recombine=4)
    p.gate.data.copy_(torch.full((8,), 0.30))
    p.born.fill_(0.0); p.slots = [-1, -1]
    p.segments = 100000
    p.last_seen.fill_(100000); p.last_seen[0] = 0.0   # only 0 has gone quiet
    p.prune(step=100000, survival=10)
    p.add_experts(1, step=2, recombine=4)
    check("a new expert never reuses a dead id",
          int(p.uid[-1]) == 8 and 0 not in [int(u) for u in p.uid],
          f"uids {[int(u) for u in p.uid]}")

    # ---------------------- 8. the gate cannot kill, and staleness cannot
    #                            empty the pool completely
    p = build(d, 5)
    p.gate.data.fill_(0.0001); p.born.fill_(0.0)      # a uniform gate collapse
    p.slots = [-1, -1]
    p.segments = 100000; p.last_seen.fill_(100000)    # but all still chosen
    gone = p.prune(step=100000, survival=10)
    check("a total gate collapse deletes nobody",
          gone == 0 and p.n_experts() == 5,
          "every gate at 1e-4 and not one goes - the gate is not read")

    p.last_seen.fill_(0.0)                            # now nothing wants any
    gone = p.prune(step=100000, survival=10)
    check("...but unchosen empties it down to the guard's last one",
          p.n_experts() == 1 and gone == 4)
finally:
    shutil.rmtree(d, ignore_errors=True)

print("\n  all prune tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
