#!/usr/bin/env python3
"""
Freeze and carry: a character that has halted stops being computed.

With cfg.halt_freeze a character's state stops changing at the pass where it
halts, its experts stop running, and the characters after it read that final
state at every deeper pass. What this asserts, on a small pooled model whose
halting head is made to disagree with itself, so characters halt at many
different passes:

  reading a text one character at a time through the attention cache - the
  way writing reads - gives the same predictions and the same halting passes
  as reading it in one forward. The keys and values a halted character leaves
  for the deeper passes are exactly what the one-forward read computes

  a character written one at a time runs only the passes it asked for

  at every pass only the characters still going reach the experts: the rows
  routed shrink as characters halt, in writing and in training alike

  training with freezing has a finite loss and a gradient, and a halted
  character's state is the one it halted with

  with freezing off nothing changes: every character runs every pass
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minagi.pool import PooledMLP, capture_routes            # noqa: E402
from minagi.recur import RecurCoder, RecurConfig             # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


torch.manual_seed(0)
cfg = RecurConfig(vocab_size=64, d_model=32, n_head=2, block=128, d_ff=64,
                  use_pool=True, pool_experts=8, pool_d_ff=32, pool_top_k=2,
                  pool_capacity_factor=0.0, n_prelude=1, n_recur=1, n_coda=0,
                  max_steps=6, min_steps=1, train_steps_mean=0.0,
                  bptt_window=6, halt_thresh=0.9, halt_freeze=True)
model = RecurCoder(cfg)
with torch.no_grad():
    # a halting head with opinions, so characters halt at different passes
    model.halt.weight.normal_(0.0, 0.25)
    model.halt.bias.fill_(0.3)
    for m in model.modules():
        if isinstance(m, PooledMLP):
            m.router.weight.normal_(0.0, 1.0)
model.eval()

ids = torch.randint(0, cfg.vocab_size, (1, 48))

# --- one forward over the whole text --------------------------------------
with torch.no_grad(), capture_routes() as got:
    whole, out = model(ids, collect=True)
steps = out["steps"][0].long().tolist()
rows = [ids_.shape[0] for ids_, _ in got]
check("characters halt at many different passes",
      len(set(steps)) >= 3 and min(steps) < cfg.max_steps,
      f"halting passes {sorted(set(steps))}")
want = [sum(1 for s in steps if s > n) for n in range(cfg.max_steps)]
want = [w for w in want if w]
check("at every pass only the characters still going reach the experts",
      rows == want, f"routed per pass {rows}, still going {want}")

# --- the same text one character at a time, as writing reads -------------
caches = model.empty_caches()
one, passes = [], []
with torch.no_grad():
    for t in range(ids.shape[1]):
        with capture_routes() as got1:
            lg, o = model(ids[:, t:t + 1], caches=caches, pos_offset=t,
                          collect=True)
        one.append(lg[0, -1])
        passes.append((int(o["steps"][0, -1]), len(got1)))
one = torch.stack(one)
err = float((one - whole[0]).abs().max())
check("one character at a time predicts what one forward predicts",
      err < 1e-4, f"largest difference {err:.1e}")
check("...halting at the same passes",
      [p for p, _ in passes] == steps)
check("a character read one at a time runs only the passes it asked for",
      all(p == n for p, n in passes),
      f"{sum(n for _, n in passes)} passes run for {len(passes)} characters, "
      f"against {len(passes) * cfg.max_steps} without freezing")

# --- training ---------------------------------------------------------------
model.train()
x, y = ids[:, :-1], ids[:, 1:]
state = {}


def keep_state(mod, inp, out_):
    state.setdefault("adapter_in", []).append(inp[0].detach())


h = model.adapter.register_forward_hook(keep_state)
with capture_routes() as got2:
    _, loss = model(x, y)
h.remove()
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
check("training with freezing has a finite loss and a gradient",
      bool(torch.isfinite(loss)) and len(grads) > 0
      and all(bool(torch.isfinite(g).all()) for g in grads),
      f"loss {float(loss):.3f}")
rows2 = [ids_.shape[0] for ids_, _ in got2]
check("...and routes only the characters still going",
      rows2[0] == x.shape[1] and rows2 == sorted(rows2, reverse=True)
      and rows2[-1] < rows2[0], f"routed per pass {rows2}")
# The state entering pass n is [h, x]. A character that halted by the end
# of pass n keeps the h it halted with from then on.
ins = state["adapter_in"]
d = cfg.d_model
with torch.no_grad():
    model.eval()
    _, o2 = model(x, collect=True)
    model.train()
st = o2["steps"][0].long()
moved = 0.0
for n in range(1, len(ins) - 1):
    done = st <= n
    if bool(done.any()):
        moved = max(moved, float((ins[n + 1][0, done, :d]
                                  - ins[n][0, done, :d]).abs().max()))
check("a halted character's state is the one it halted with", moved == 0.0,
      f"largest change after halting {moved:.1e}")

# --- freezing off -------------------------------------------------------------
cfg.halt_freeze = False
model.eval()
with torch.no_grad(), capture_routes() as got3:
    model(ids)
check("with freezing off every character runs every pass",
      [i.shape[0] for i, _ in got3] == [ids.shape[1]] * cfg.max_steps)

print("\n  all halt-freeze tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
