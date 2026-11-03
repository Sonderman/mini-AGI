#!/usr/bin/env python3
"""
The expert layer computes each slot on exactly its own characters, and that is
the same computation as the padded rectangle it replaced.

  a training forward and backward with capacity dropping assignments: the
  same loss, the same gradient for every parameter, to rounding - and the
  same drops, so the counters the pool reports agree
  held-out under no gradient, and a prompt long enough to skip the writing
  path's own shortcut: the same loss and logits
  MINAGI_DISPATCH=padded still selects the rectangle, and does differ from
  nothing but rounding

Two models from one weights directory, one per path, so neither sees what the
other's forwards did to the pool's statistics.
"""
import os
import shutil
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-exact-")
saved = os.environ.get("MINAGI_DISPATCH")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.pool import PooledMLP                         # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=6, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=256, max_steps=3, top_k=2, depth=1)
    text = (b"The Roman Empire was the post-Republican period of ancient Rome. "
            b"def merge(a, b):\n    return sorted(a + b)\n1. e4 e5 2. Nf3 Nc6 ") * 2
    x = torch.tensor([list(text[:200])])

    def model_for(mode):
        os.environ["MINAGI_DISPATCH"] = mode
        model, cfg, pool, man = T.build_paged(wdir, "cpu", ram_capacity=8)
        pool.balance = 0.0
        torch.manual_seed(0)
        for site in (m for m in model.modules() if isinstance(m, PooledMLP)):
            # a router with strong opinions and a tight capacity: the load
            # piles onto a few slots, so capacity has assignments to drop
            site.router.weight.data.normal_(0.0, 3.0)
            site.capacity_factor = 1.0
        return model, pool

    def train_step(mode):
        model, pool = model_for(mode)
        model.train()
        torch.manual_seed(1)
        _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
        return (float(loss), grads, sum(int(s.routed) for s in sites),
                sum(int(s.dropped) for s in sites))

    def held_out(mode):
        model, pool = model_for(mode)
        model.eval()
        with torch.no_grad():
            logits, loss = model(x[:, :-1], x[:, 1:])
        return float(loss), logits

    la, ga, ra, da = train_step("padded")
    lb, gb, rb, db = train_step("exact")
    rel = float((ga - gb).norm() / ga.norm())
    check("capacity dropped assignments in this test", da > 0, f"{da} of {ra}")
    check("a training step: the same loss", abs(la - lb) < 1e-5, f"{la:.6f} vs {lb:.6f}")
    check("...the same gradient for every parameter, to rounding", rel < 1e-5,
          f"relative difference {rel:.1e} over {ga.numel():,} values")
    check("...and the same drops", (ra, da) == (rb, db), f"routed {ra} / {rb}, dropped {da} / {db}")

    ha, lga = held_out("padded")
    hb, lgb = held_out("exact")
    check("held-out under no gradient: the same loss and logits",
          abs(ha - hb) < 1e-5 and float((lga - lgb).abs().max()) < 1e-4,
          f"{ha:.6f} vs {hb:.6f}, max |dlogit| {float((lga - lgb).abs().max()):.1e}")

    import minagi.pool as P                                   # noqa: E402
    calls = {"exact": 0}
    orig_cat = torch.cat

    def counting_cat(*a, **kw):
        calls["exact"] += 1
        return orig_cat(*a, **kw)
    model, pool = model_for("padded")
    P.torch.cat = counting_cat
    try:
        model.eval()
        with torch.no_grad():
            model(x[:, :-1])
        padded_cats = calls["exact"]
        calls["exact"] = 0
        os.environ["MINAGI_DISPATCH"] = "exact"
        with torch.no_grad():
            model(x[:, :-1])
        exact_cats = calls["exact"]
    finally:
        P.torch.cat = orig_cat
    check("MINAGI_DISPATCH=padded selects the rectangle, the default the exact path",
          exact_cats > padded_cats, f"pool-level cats: padded {padded_cats}, exact {exact_cats}")
finally:
    if saved is None:
        os.environ.pop("MINAGI_DISPATCH", None)
    else:
        os.environ["MINAGI_DISPATCH"] = saved
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all exact-dispatch tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
