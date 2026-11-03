#!/usr/bin/env python3
"""
What the capacity bound throws away is counted, and now reported.

The batched dispatch pads every expert to a fixed slot buffer, so an expert
wanted by more tokens than its cap loses the excess: those assignments are
discarded, the token goes without that expert's contribution, and the expert
gets no gradient from it. The bound is what makes the arrangement affordable -
without it the buffer scales with the worst imbalance rather than with the
work, measured at 1.89 GB against 0.13 GB of real arithmetic.

`self.dropped` counted every discard from the day the bound was added and
NOTHING EVER READ IT. So the one trade `pool.capacity_factor` makes had no
number attached: whether clipping costs anything, whether a larger factor
would buy anything, whether the concentration in the usage graph is partly
this bound starving the tail. All unanswerable, in a project that measures
everything else.

A count on its own would not have helped either - a cumulative figure over a
week of reading says nothing about now, and nothing about scale. So there is a
`routed` denominator beside it, the aggregate is a SHARE over the interval
since it was last read, and the progress line stays silent when it is zero.
"""
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train                                                 # noqa: E402
from minagi.recur import RecurCoder, RecurConfig             # noqa: E402
from minagi.paged import PagedPool                           # noqa: E402
from minagi.pool import PooledMLP                            # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


D = "/tmp/minagi_dropped_test"
N, DM, DF = 8, 32, 64
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)
rng = np.random.default_rng(0)
for i in range(N):
    np.savez(os.path.join(D, f"e{i:05d}.npz"),
             w1=(rng.standard_normal((DF, DM)) * 0.05).astype(np.float32),
             w3=(rng.standard_normal((DF, DM)) * 0.05).astype(np.float32),
             w2=(rng.standard_normal((DM, DF)) * 0.05).astype(np.float32))

torch.manual_seed(0)
cfg = RecurConfig(vocab_size=265, d_model=DM, n_head=2, d_ff=DF, block=256,
                  max_steps=2, n_prelude=1, n_recur=1, n_coda=0, use_pool=True,
                  pool_experts=N, pool_d_ff=DF, pool_top_k=2)
model = RecurCoder(cfg)
pool = PagedPool(D, DM, DF, N, resident=4, ram_capacity=4, device="cpu")
for m in model.modules():
    if isinstance(m, PooledMLP):
        m._pool[0] = pool
SITES = [m for m in model.modules() if isinstance(m, PooledMLP)]
X = torch.randint(0, 260, (1, 128))
TOK, K = X.shape[1], cfg.pool_top_k


def forward(cf):
    """One forward at this capacity_factor, WITHOUT consuming the counters."""
    for m in SITES:
        m.capacity_factor = cf
        m.dropped = m.routed = 0
    with torch.no_grad():
        model(X)


def run(cf):
    forward(cf)
    return model.pool_dropped()          # this CLEARS them


# ------------------------------------------ 1. the denominator is the work
share, dropped, routed = run(1.5)
# The recurrent block runs several times per character, and the same site is
# called once per pass, so the count is per CALL and not per forward. Asserted
# as the relationship rather than an exact product, because the number of
# passes is the halting rule's business and not this test's.
per_call = TOK * K * len(SITES)
check("routed counts every token-expert assignment, once per pass",
      routed > 0 and routed % per_call == 0,
      f"{routed} = {routed // per_call} pass(es) x {TOK} tokens x top_k {K}")
check("the share is dropped over routed",
      abs(share - dropped / routed) < 1e-12, f"{share:.4f}")

# --------------------------------- 2. it resets, so the figure is a RATE
s2, d2, r2 = model.pool_dropped()
check("reading the counters clears them", d2 == 0 and r2 == 0,
      "a cumulative count over a week of reading says nothing about now")

# ------------------------ 3. a tighter bound really does discard more
loose = run(2.0)[0]
tight = run(1.01)[0]
check("a tighter capacity_factor drops a larger share", tight > loose,
      f"1.01 -> {tight:.1%}, 2.0 -> {loose:.1%}")
none = run(0.0)[0]
check("capacity_factor 0 disables the bound entirely", none == 0.0,
      f"{none:.1%} dropped with the bound off")

# ------------------------------- 4. the progress line says it, or nothing
forward(1.01)                            # forward only: the note must read them
note = train._dropped_note(model)
check("the note reports the share and the raw counts",
      "capacity dropped" in note and "%" in note and "of" in note, note.strip())
forward(0.0)
check("...and says nothing at all when nothing was dropped",
      train._dropped_note(model) == "", "silence is the common case")


class NoPool:
    pass


check("a model with no pool is silent rather than an error",
      train._dropped_note(NoPool()) == "")

# ------------------------------------ 5. the counter is read by SOMETHING
import inspect                                              # noqa: E402
prog = inspect.getsource(train.cmd_read)
check("cmd_read's progress line calls it",
      "_dropped_note(model)" in prog,
      "this is the whole point: it was counted and read nowhere")

shutil.rmtree(D, ignore_errors=True)
print("\n  all dropped-report tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
