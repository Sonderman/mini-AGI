#!/usr/bin/env python3
"""
The temperature of expert selection (pool.select_temperature).

At 0 the card holds the experts the text's first row votes for most and each
character takes its own top 8 - the rule as it has run. Above 0 the experts
are DRAWN: at each row, 8 are drawn from the text's probabilities until the
card is full, and each character draws its own 8 from its own probabilities
over the card, in proportion to p^(1/T). The characters themselves are still
chosen greedily. What this asserts, on a paged model with 40 experts and a
32-slot card:

  a training window's card fills 8 a row until all 32 slots are taken

  the text's probabilities at a row are every character's full router
  distribution at that row, summed over the whole text read so far - a
  prompt and every character written after it - recomputed here from the
  router itself

  every character routes only among the experts on the card

  any expert can be drawn: at temperature 1 a run of fresh texts brings
  nearly the whole pool onto the card, while at a temperature close to 0 a
  row's draw is the top of the text's probabilities, the same every time

  a new text starts its tally empty, and a forward that trains has a finite
  loss and a gradient
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


tmp = tempfile.mkdtemp(prefix="minagi-temp-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.paged import PagedPool                        # noqa: E402
    from minagi.pool import PooledMLP, capture_routes         # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=40, resident=32, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=256, max_steps=6,
           top_k=8, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", ram_capacity=64,
                                          read_only=True)
    pool.balance = 0.0
    site = [m for m in model.modules() if isinstance(m, PooledMLP)][0]
    torch.manual_seed(0)
    with torch.no_grad():
        site.router.weight.normal_(0.0, 1.0)     # a router with opinions

    def ids(t):
        return torch.tensor([list(t.encode())])

    TEXT = "def merge(a, b):\n    return sorted(a + b)\n" * 3
    pool.select_temperature = 1.0

    # --- a training window fills the card 8 a row ----------------------------
    per_row = []
    real_admit = PagedPool.admit

    def spy(self, *a, **k):
        before = len(self._admitted)
        n = real_admit(self, *a, **k)
        per_row.append(len(self._admitted) - before)
        return n
    PagedPool.admit = spy
    model.train()
    x = ids(TEXT)
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    PagedPool.admit = real_admit
    check("a training window's card fills 8 a row", per_row == [8, 8, 8, 8],
          f"{per_row}")
    grads = [q.grad for q in model.parameters() if q.grad is not None]
    check("...and the forward has a finite loss and a gradient",
          bool(torch.isfinite(loss)) and grads
          and all(bool(torch.isfinite(g).all()) for g in grads),
          f"loss {float(loss):.3f}")
    model.zero_grad(set_to_none=True)

    # --- the text's probabilities, row by row, over the whole text -----------
    model.eval()
    calls, starts = [], []
    h = site.register_forward_pre_hook(
        lambda m, a: calls.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    real_begin = PagedPool.begin_forward

    def mark(self, *a, **k):
        starts.append(len(calls))
        return real_begin(self, *a, **k)
    PagedPool.begin_forward = mark
    prompt = ids("The Roman Empire was the post-Republican period of Rome")
    reply = ids(" and it ruled")
    caches = model.empty_caches()
    with torch.no_grad(), capture_routes() as got:
        model(prompt, caches=caches, pos_offset=0)
        n0 = prompt.shape[1]
        for t in range(reply.shape[1]):
            model(reply[:, t:t + 1], caches=caches, pos_offset=n0 + t)
    h.remove()
    PagedPool.begin_forward = real_begin
    want = {}
    with torch.no_grad():
        E = pool.router_rows()
        for f, s0 in enumerate(starts):
            s1 = starts[f + 1] if f + 1 < len(starts) else len(calls)
            for r, i in enumerate(range(s0, min(s1, s0 + 4))):
                q = F.softmax(F.linear(calls[i] + site.depth_emb,
                                       site.router.weight[:E]).float(), -1)
                want[r] = want.get(r, torch.zeros(E)) + q.sum(0)
    err = max(float((pool._rows[r] - want[r]).abs().max()) for r in want)
    check("each row's probabilities are every character's, summed over the "
          "prompt and the reply", sorted(pool._rows) == sorted(want)
          and err < 1e-4, f"rows {sorted(pool._rows)}, largest difference "
                          f"{err:.1e}")
    routed = set(int(e) for ids_, _ in got[-6:] for e in ids_.reshape(-1).tolist())
    check("every character routes only among the experts on the card",
          routed <= set(pool._admitted), f"{len(routed)} used")

    # --- any expert can be drawn; close to 0, the top -------------------------
    def cards(temp, n):
        pool.select_temperature = temp
        seen, firsts = set(), set()
        for _ in range(n):
            per = []

            def spy2(self, *a, **k):
                before = set(self._admitted)
                r = real_admit(self, *a, **k)
                per.append(tuple(sorted(set(self._admitted) - before)))
                return r
            PagedPool.admit = spy2
            with torch.no_grad():
                model(ids("add 4917 + 388 = 5305\n"))
            PagedPool.admit = real_admit
            seen |= set(pool._admitted)
            firsts.add(per[0])
        return seen, firsts
    seen_hot, _ = cards(1.0, 30)
    _, firsts_cold = cards(1e-4, 10)
    check("at temperature 1 fresh texts bring nearly the whole pool onto the "
          "card", len(seen_hot) >= 36, f"{len(seen_hot)} of 40")
    check("close to 0, a row's draw is the same every time", len(firsts_cold) == 1,
          f"{len(firsts_cold)} different first rows")
    t0 = pool._rows[0]
    top8 = set(torch.topk(t0, 8).indices.tolist())
    check("...and it is the top of the text's probabilities",
          set(next(iter(firsts_cold))) == top8)

    # --- a new text starts empty ----------------------------------------------
    with torch.no_grad():
        model(ids("x"))
    one = pool._rows[0]
    with torch.no_grad():
        model(ids("x"))
    check("a new text starts its tally empty", torch.allclose(pool._rows[0], one))

    # --- at 0, the rule as it has run ------------------------------------------
    pool.select_temperature = 0.0
    del per_row[:]
    PagedPool.admit = spy
    with torch.no_grad():
        model(ids(TEXT))
    PagedPool.admit = real_admit
    check("at 0 the whole card is admitted at the first row, as before",
          per_row[0] == 32, f"{per_row}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all selection-temperature tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
