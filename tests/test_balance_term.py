#!/usr/bin/env python3
"""
The balance term (pool.balance) and the clock it keeps alive.

Left alone the router keeps choosing the same experts, so a term in the loss
charges it, in every forward that trains, for the probability it puts on each
expert over the whole pool, in proportion to that expert's share of recent
admissions. What this asserts, on a paged model:

  only a forward that trains, with gradients, computes it, and the model's
  pool_balance() is zero after any other forward

  it is the router's mean probability over the whole pool, each expert's
  charged by its share of recent admissions, times the weight - recomputed
  here from the router itself

  only the router's rows learn from it, a step on it moves the router's
  probability from the busy experts to the idle ones, and with usage even it
  pushes nowhere

  `recent` is the running share of training forwards that admitted each
  expert, and it survives a restart

  whatever admits an expert resets its prune clock - being used is being alive
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


tmp = tempfile.mkdtemp(prefix="minagi-balance-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.paged import PagedPool                        # noqa: E402
    from minagi.pool import PooledMLP                         # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", read_only=True)
    site = [m for m in model.modules() if isinstance(m, PooledMLP)][0]
    TEXT = "add 2856 + 741 = 3597\nadd 12 + 30 = 42\nmul 7 * 6 = 42\n"
    X = torch.tensor([[b for b in TEXT.encode()][:cfg.block]])

    # the router's own choice
    model.eval()
    with torch.no_grad():
        model(X)
    ref = set(pool._admitted)

    # --- whatever admits an expert resets its prune clock ---------------------
    check("every expert a forward admits has its prune clock reset",
          all(float(pool.last_seen[e]) == pool.segments for e in ref))
    check("...and no other",
          all(float(pool.last_seen[e]) < pool.segments
              for e in range(pool.n_experts()) if e not in ref))

    # --- only a forward that trains computes the term -------------------------
    pool.balance = 0.5
    pool.recent.zero_()
    pool.recent[list(ref)] = 1.0                  # the router's own: busy
    with torch.no_grad():
        model(X)
    check("a forward that does not train computes no balance term",
          pool.balance_term() is None and float(model.pool_balance()) == 0.0)
    model.train()
    with torch.no_grad():
        model(X[:, :-1], X[:, 1:])
    check("...nor one in training mode without gradients",
          pool.balance_term() is None)

    # --- its value, recomputed from the router --------------------------------
    share = pool.usage_share().clone()
    recent_before = pool.recent.clone()
    inputs = []
    h = site.register_forward_pre_hook(
        lambda m, a: inputs.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    model.zero_grad(set_to_none=True)
    model(X[:, :-1], X[:, 1:])
    h.remove()
    adm = set(pool._admitted)
    term = model.pool_balance()
    E = pool.router_rows()
    x0 = inputs[0] + site.depth_emb.detach()
    with torch.no_grad():
        P = F.softmax(F.linear(x0, site.router.weight[:E]).float(), -1).mean(0)
        want = 0.5 * (E * (share * P).sum() - 1.0)
    check("it is the router's probability over the pool, charged by each "
          "expert's share of recent admissions",
          abs(float(term) - float(want)) < 1e-5,
          f"{float(term):.5f} vs {float(want):.5f}")

    # --- `recent` is a running share of training forwards ----------------------
    a = 1.0 / pool.usage_steps
    want_recent = recent_before * (1 - a)
    want_recent[list(adm)] += a
    check("recent: each admitted expert gains, every expert decays",
          torch.allclose(pool.recent, want_recent, atol=1e-6))
    fresh = PagedPool(pool.tiers.path, cfg.d_model, cfg.pool_d_ff,
                      pool.n_experts(), resident=4, device="cpu",
                      read_only=True)
    fresh.load_telemetry(pool.telemetry())
    check("recent is carried across a restart",
          torch.allclose(fresh.recent, pool.recent, atol=1e-5))

    # --- what it moves ----------------------------------------------------------
    term.backward()
    g = site.router.weight.grad
    others = [nm for nm, q in model.named_parameters()
              if q is not site.router.weight and q.grad is not None
              and float(q.grad.abs().sum()) > 0]
    check("only the router's rows learn from it",
          g is not None and float(g.abs().sum()) > 0 and not others,
          f"{others[:3]}")
    with torch.no_grad():
        W = site.router.weight
        p0 = F.softmax(F.linear(x0, W[:E]).float(), -1).mean(0)
        p1 = F.softmax(F.linear(x0, (W - 0.5 * g / g.abs().max())[:E]).float(),
                       -1).mean(0)
    busy = sorted(ref)
    idle = [e for e in range(E) if e not in ref]
    check("a step on it moves the router's probability from the busy experts "
          "to the idle ones",
          float(p1[busy].sum()) < float(p0[busy].sum())
          and float(p1[idle].sum()) > float(p0[idle].sum()),
          f"busy {float(p0[busy].sum()):.3f} -> {float(p1[busy].sum()):.3f}")

    pool.recent.fill_(0.3)                        # every expert as used
    model.zero_grad(set_to_none=True)
    model(X[:, :-1], X[:, 1:])
    model.pool_balance().backward()
    g = site.router.weight.grad
    check("with usage even it pushes nowhere",
          g is None or float(g.abs().max()) < 1e-6,
          f"largest {float(g.abs().max()) if g is not None else 0.0:.1e}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all balance-term tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
