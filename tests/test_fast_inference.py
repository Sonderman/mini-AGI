#!/usr/bin/env python3
"""
Forwards that compute no gradient - a reply being written, a held-out chunk -
take a shorter path through the expert layer, and it computes what the long
one does.

  the slots' weights in the compute dtype are cast once and kept
  (PagedPool.inference_weights) instead of at every row; a forward of a few
  characters runs only the experts it was routed to

  held-out forwards give the same loss to the bit, and a reply written one
  character at a time is the same reply, with logits equal to rounding

  the kept copy is never stale: not after an in-place change to a slot, not
  after the card loads other experts, not after a real optimiser step

  a forward that trains drops it, so training holds the memory it always did,
  and its own path is untouched

  the routing statistics that no longer stop every row to be read still read
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-fast-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.pool import PooledMLP                         # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", ram_capacity=3)
    pool.balance = 0.0
    torch.manual_seed(0)
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    for site in sites:
        site.router.weight.data.normal_(0.0, 1.0)     # a router with opinions

    def fast(on):
        for site in sites:
            site.fast_inference = on

    def ids(t):
        return torch.tensor([[b for b in t.encode()][:cfg.block]])

    TEXTS = ["1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1",
             "def merge(a, b):\n    return sorted(a + b)\n\nclass Node:\n",
             "The Roman Empire was the post-Republican period of ancient Rome"]

    @torch.no_grad()
    def held_out(text):
        model.eval()
        x = ids(text)
        logits, _ = model(x[:, :-1])
        return float(torch.nn.functional.cross_entropy(
            logits[0].float(), x[0, 1:]))

    @torch.no_grad()
    def reply(text, n=24):
        """Greedy, one character a forward, recording every forward's logits."""
        model.eval()
        caches = model.empty_caches()
        cur, off, out, seen = ids(text), 0, [], []
        for _ in range(n):
            logits, _ = model(cur, caches=caches, pos_offset=off)
            off += cur.shape[1]
            nxt = logits[:, -1].float().argmax(-1, keepdim=True)
            seen.append(logits[:, -1].float())
            out.append(int(nxt))
            cur = nxt
        return out, torch.cat(seen)

    # ---- the same numbers -----------------------------------------------------
    for t in TEXTS:
        fast(False)
        slow_l = held_out(t)
        fast(True)
        fast_l = held_out(t)
        check(f"held-out loss identical ({t[:12]!r})", slow_l == fast_l,
              f"{slow_l!r} vs {fast_l!r}")
    for t in TEXTS:
        fast(False)
        a, la = reply(t)
        fast(True)
        b, lb = reply(t)
        err = float((la - lb).abs().max())
        check(f"same reply, logits equal to rounding ({t[:12]!r})", a == b and err < 1e-4,
              f"max |diff| {err:.2e}")

    # ---- the kept copy is never stale -------------------------------------------
    fast(True)
    reply(TEXTS[0], n=4)
    w = pool.inference_weights(torch.float32)
    check("the copy is reused while nothing changes",
          pool.inference_weights(torch.float32)[0] is w[0])
    with torch.no_grad():
        pool.w1.add_(0.05)                  # an in-place change, as a step makes
    check("an in-place change to a slot is seen",
          torch.equal(pool.inference_weights(torch.float32)[0], pool.w1.detach()))
    fast(False)
    a, la = reply(TEXTS[1], n=12)
    fast(True)
    b, lb = reply(TEXTS[1], n=12)
    check("after it, fast and slow still agree", a == b and float((la - lb).abs().max()) < 1e-4)

    swaps = pool.swaps
    reply(TEXTS[2], n=8)                    # another text: the card loads others
    moved = pool.swaps > swaps
    check("the card loaded other experts", moved, f"{pool.swaps - swaps} rearrangements")
    check("and the copy matches the slots after the loads",
          all(torch.equal(c, t.detach()) for c, t in
              zip(pool.inference_weights(torch.float32), (pool.w1, pool.w3, pool.w2))))
    bf = pool.inference_weights(torch.bfloat16)
    check("a bf16 copy is the slots cast exactly as each call cast them",
          all(torch.equal(c, t.detach().to(torch.bfloat16)) for c, t in
              zip(bf, (pool.w1, pool.w3, pool.w2))))

    # ---- training: untouched, and it drops the copy ------------------------------
    trunk, pool_ps = T._split_trunk_pool(model)
    opt = torch.optim.AdamW([{"params": trunk, "lr": 1e-3}, {"params": pool_ps, "lr": 1e-2}])
    pool.attach_optimiser(opt)
    model.train()
    x = ids(TEXTS[0])
    logits, loss = model(x[:, :-1], x[:, 1:])
    check("a forward that trains drops the copy", pool._infer is None)
    loss = loss + model.pool_aux() * 0.01
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    for t in TEXTS:
        fast(False)
        slow_l = held_out(t)
        fast(True)
        fast_l = held_out(t)
        check(f"after a real step, held-out still identical ({t[:12]!r})", slow_l == fast_l)

    # ---- the statistics still read ------------------------------------------------
    sat = pool.saturation()
    check("pressure and want_k still read as numbers",
          isinstance(sat["pressure"], float) and isinstance(sat["want_k"], float)
          and 0.0 <= sat["pressure"] <= 1.0 and sat["want_k"] >= 1.0,
          f"pressure {sat['pressure']:.3f}, want_k {sat['want_k']:.2f}")
    share, d, r = model.pool_dropped()
    check("the dropped-assignment count still reads", isinstance(d, int) and r > 0,
          f"{d} of {r}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all fast-inference tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
