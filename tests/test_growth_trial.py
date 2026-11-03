"""
A grown expert must be able to earn its place.

Before this, it could not. It was born at gate 0.0, its selection score was
scaled by that gate, so it was never chosen, so the gate never moved, so prune
deleted it. Measured over 15M characters of real reading: 28 experts grown, 28
pruned, zero survivors - every expert alive was an original.

Now selection is the router's alone and the gate only scales what an expert
contributes, so what gives a newborn its chance is its router row - which is
its parents'. The trial is what protects it from pruning meanwhile.
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.paged import PagedPool          # noqa: E402
from minagi.pool import AutoGrow             # noqa: E402

D_MODEL, D_FF, N = 32, 16, 8
ok = True


def _pool(tmp):
    rng = np.random.default_rng(0)
    for i in range(N):
        np.savez(os.path.join(tmp, f"e{i:05d}.npz"),
                 w1=(rng.standard_normal((D_FF, D_MODEL)) * .02).astype(np.float32),
                 w3=(rng.standard_normal((D_FF, D_MODEL)) * .02).astype(np.float32),
                 w2=(rng.standard_normal((D_MODEL, D_FF)) * .02).astype(np.float32))
    p = PagedPool(tmp, D_MODEL, D_FF, N, resident=4, ram_capacity=8)
    p.gate.data.copy_(torch.full((N,), 0.5))     # healthy incumbents
    p.ever[:] = True
    return p


class Site:
    """A router with a mild, ordinary opinion - not a flat one, or topk
    breaks ties by index and the test measures tie-breaking."""

    def __init__(self, n):
        rng = np.random.default_rng(1)
        self.router = torch.nn.Linear(D_MODEL, n, bias=False)
        with torch.no_grad():
            self.router.weight.copy_(torch.from_numpy(
                (rng.standard_normal((n, D_MODEL)) * 0.05).astype(np.float32)))
        self.depth_emb = torch.zeros(D_MODEL)
        self.top_k = 2


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    ok = ok and cond


if True:
    # ---- 1. protected by the trial, not by an unclearable gate -----------
    # Born ABOVE prune's min_gate an expert can never be removed for failing
    # to earn a gate. That is not a hypothetical: a 3-minute run with
    # birth_gate 0.01 against min_gate 0.005 kept 30 of 30 grown experts,
    # every one of them still sitting within a hair of where it was born, and
    # the pool grew 12 -> 44 with nothing able to stop it.
    p = _pool(tempfile.mkdtemp())
    p.add_experts(2, seed_from=0, step=100, birth_gate=0.001)
    born_gate = float(p.gate.data[N])
    check("a new expert is born below the growth brake's idle line 0.005",
          born_gate < 0.005, f"born at {born_gate}")

    # what protects it is its age, for exactly the length of the trial
    p.segments = 100 + 10                  # 1 segment per step
    gone = p.prune(step=100 + 10, survival=1600)
    check("prune spares an expert inside its survival window", gone == 0,
          f"{gone} deleted")

    # ...and once the trial is over, an expert that earned nothing goes.
    # Without this the pool has no upper bound at all.
    p.slots = [-1] * p.resident                # not resident, so evictable
    p.segments = 100 + 5000
    p.last_seen.fill_(100 + 5000)              # the originals are still wanted
    p.last_seen[N:] = 0.0                      # the two additions never were
    gone = p.prune(step=100 + 5000, survival=1600)
    check("prune removes an unchosen non-earner once its trial is over",
          gone == 2, f"{gone} deleted")

    # ---- 2. a newborn is asked for where its parents are -----------------
    # Its router row decides whether any text ever asks for it. A random
    # row is asked for by chance or not at all; the row of a recombined
    # expert is its parents' rows, weighted by how many hidden units each
    # gave - so it is asked for where the units it carries already work.
    # Which parent each unit came from is read off the weights themselves,
    # not from anything add_experts kept.
    torch.manual_seed(0)
    p = _pool(tempfile.mkdtemp())
    site = Site(p._n)
    p._sites = [site]
    rows_before = site.router.weight.detach().clone()
    parents = torch.stack([p.tiers.fetch(p._f(i))["w1"].float()
                           for i in range(N)])            # [N, D_FF, D]
    p.add_experts(1, seed_from=0, step=100, birth_gate=0.001)
    child = p.tiers.fetch(p._f(N))["w1"].float()           # [D_FF, D]
    src = torch.stack([(parents[:, u] - child[u]).abs().sum(-1).argmin()
                       for u in range(D_FF)])
    share = torch.bincount(src, minlength=N).float() / D_FF
    expected = share @ rows_before
    got = site.router.weight.detach()[N]
    check("a newborn's router row is its parents' rows, weighted by units",
          torch.allclose(got, expected, atol=1e-5),
          f"max diff {float((got - expected).abs().max()):.2e} over "
          f"{int((share > 0).sum())} parents")
    check("...and the routers grew by exactly one row",
          site.router.weight.shape[0] == N + 1)

    # ---- 3. an original is never given a trial it did not earn ----------
    p = _pool(tempfile.mkdtemp())
    p.trial, p.now = 1600, 50
    born = p.born[:p._n]
    check("originals (born == 0) are never treated as young",
          bool(((born > 0) & ((p.now - born) < p.trial)).sum() == 0))

# ---- 4. pruning an OLD expert must not be charged to the last batch ------
class FakePool:
    """Just enough surface for AutoGrow to make one decision."""

    def __init__(self, n, born):
        self._n = n
        self.born = torch.tensor(born, dtype=torch.float)
        self.use = torch.ones(n)
        self.grow_events = []
        self.added = 0

    def n_experts(self):
        return self._n

    def saturation(self):
        return {"experts": self._n, "idle": 0, "entropy_frac": 1.0}

    def disk_bytes(self, extra=0):
        return (self._n + extra) * 40e6

    def add_experts(self, k, seed_from=None, step=0, birth_gate=0.01):
        self._n += k
        self.added += k
        self.born = torch.cat([self.born, torch.full((k,), float(step))])


g = AutoGrow(grow_k=4, keep_ratio_min=0.35, max_disk_gb=0, mem_frac_max=1.0)
pool = FakePool(10, [0.0] * 10)
g.step(1.0, pool, 400)                       # grows: cohort = (400, 4)
check("first decision grows", pool.added == 4)

# now delete an ANCIENT original - nothing to do with the batch just added
keep = [i for i in range(pool._n) if i != 0]
pool.born = pool.born[keep]
pool._n -= 1

g.step(1.0, pool, 800)
check("pruning an old original does not stop growth",
      g.keep_ratio == 1.0 and pool.added == 8,
      f"keep_ratio {g.keep_ratio:.2f}, added {pool.added}")

# A batch that was added and then PRUNED cannot be seen from the present
# network - those experts no longer exist. That question is inherently about
# the past, and the growth rule no longer asks it. What stops runaway growth
# instead is visible right now: an expert is born at a gate below the idle
# threshold, so every addition counts as idle until it earns one, and the
# idle fraction climbs with each batch that does not.
pool.born = pool.born[pool.born != 800]
pool._n = int(pool.born.numel())
g.step(1.0, pool, 1200)
check("a vanished batch leaves nothing to judge",
      g.keep_ratio == 1.0, "it is not remembered, by design")

# ...and the dying brake is what holds the line instead.
#
# The mechanism CHANGED with the rule. It used to be immediate: an expert was
# born at a gate below the idle line, so every addition counted as idle from
# the moment it existed and the brake engaged on the very next decision. Under
# staleness a newborn is exempt for its whole trial - it has not failed to be
# chosen, it has not finished being offered - so the brake engages one trial
# later. Growth still self-limits; it just gets a fair number of tries first.
class IdlePool(FakePool):
    def __init__(self, n, stale):
        super().__init__(n, [0.0] * n)
        self.trial = 1600
        self.dying_at = 0.75
        self.stale = stale          # originals nothing asks for any more
        self.now = 0

    def dying(self):
        n = self._n
        d = torch.zeros(n)
        d[:min(self.stale, n)] = 0.9                    # well past dying_at
        born = self.born[:n]
        done = (born > 0) & ((self.now - born) >= self.trial)
        d[done] = 0.9               # an addition nothing asked for, trial over
        return d

    def saturation(self):
        d = self.dying()
        return {"experts": self._n,
                "idle": int((d >= self.dying_at).sum()),
                "entropy_frac": 1.0}

g3 = AutoGrow(grow_k=4, dying_frac_max=0.25, max_disk_gb=0, mem_frac_max=1.0)
p3 = IdlePool(157, 27)
rounds = 0
while p3.added < 400:
    before = p3.added
    p3.now = 400 * (rounds + 1)
    g3.step(1.0, p3, p3.now)
    if p3.added == before:
        break
    rounds += 1
check("growth self-limits once the additions' trials end unaddressed",
      0 < rounds < 40,
      f"{rounds} batches, then held at {p3.saturation()['idle']}/{p3._n} dying")
check("...and it took longer than the gate version, which stopped at once",
      rounds >= 4, f"{rounds} batches before the brake engaged")

# ---- 4b. the in-flight cap is what makes the other brakes reachable ------
# Neither usage brake can see an expert inside its trial: a newborn reads
# dying 0 by design, and keep_ratio judges only experts whose trial has ENDED
# and returns 1.0 when there is nothing to judge. So for one whole trial after
# growth starts, nothing on the usage side can refuse - measured against the
# real AutoGrow at a 500M trial and a 204,800 decision interval, that is 2,441
# decisions of blindness, and the disk ceiling was the only thing that ever
# said no.
class FlightPool(FakePool):
    def __init__(self, n, trial):
        super().__init__(n, [0.0] * n)
        self.trial = trial
        self.dying_at = 0.75

    def dying(self):
        return torch.zeros(self._n)              # nothing is ever stale here

    def saturation(self):
        return {"experts": self._n, "idle": 0, "entropy_frac": 1.0}


def run_to_refusal(cap, decisions=200):
    g = AutoGrow(grow_k=4, dying_frac_max=0.25, keep_ratio_min=0.35,
                 max_disk_gb=10.0, mem_frac_max=1.0, max_in_flight=cap)
    p = FlightPool(76, 97_656)
    added = 0
    for i in range(decisions):
        r = g.step(1.0, p, 100 * (i + 1))
        added += r["grew"]
        if r["grew"] == 0:
            return added, r["reason"]
    return added, ""


off, why_off = run_to_refusal(0)
cap, why_cap = run_to_refusal(16)
check("without the cap only the disk ceiling stops growth",
      off >= 100 and ("no room" in why_off),
      f"{off} experts added, stopped by: {why_off[:50]}")
check("with it, growth stops at the cap and says so",
      cap == 16 and "trial" in why_cap,
      f"{cap} added, then: {why_cap[:60]}")
check("...which is far sooner than the disk would have",
      cap < off / 10, f"{cap} against {off}")

# ---- 5. the disk ceiling holds -------------------------------------------
g2 = AutoGrow(grow_k=4, max_disk_gb=0.5, mem_frac_max=1.0)   # 0.5 GB = 12 experts
pool2 = FakePool(11, [0.0] * 11)
r = g2.step(1.0, pool2, 400)
check("growth stops at the disk ceiling", pool2.added == 0,
      r.get("reason", ""))

print("\n  " + ("all growth-trial tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
