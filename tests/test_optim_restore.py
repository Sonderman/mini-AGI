#!/usr/bin/env python3
"""
Adam's moments must cross a restart. On a model that never stops training,
losing them is not a slow start - it is a step change in the model.

THE BUG THIS IS WRITTEN AGAINST. `cmd_read` built a fresh optimiser, attached
it to the pool, and SAVED it at every checkpoint - and never loaded it back.
optim.npz was written faithfully and read by nobody on that path. Every
restart discarded 108.9 million moment values, 99.9% of them non-zero, and
Adam re-estimated its second moments from a single gradient: a large, badly
scaled step in a direction inferred from one batch, decaying over a few
hundred steps. That is the held-out jump after every restart.

The tell was in the file itself: Adam's step counter read t = 2,381 at
training step 458,000. It counts optimiser steps since the last RESTART.

What is tested here is the property, not the call: take a model, train it,
save, rebuild from disk, and require that the optimiser comes back with the
same moments - and that the first step after the restart is the step it would
have taken had it never stopped.
"""
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi import store as weights_store                 # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 8, bias=False)
        self.b = torch.nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x)))


D = "/tmp/minagi_optim_restore"
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)
torch.manual_seed(0)

m = Tiny()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randn(32, 8)
y = torch.randn(32, 4)
for _ in range(50):                       # build up real moments
    opt.zero_grad(); torch.nn.functional.mse_loss(m(x), y).backward(); opt.step()

live = {n: opt.state[p]["exp_avg"].clone()
        for n, p in m.named_parameters() if "exp_avg" in opt.state[p]}
live_sq = {n: opt.state[p]["exp_avg_sq"].clone()
           for n, p in m.named_parameters() if "exp_avg_sq" in opt.state[p]}
check("the optimiser has real moments to lose",
      live and all(float(v.abs().sum()) > 0 for v in live.values()),
      f"{sum(v.numel() for v in live.values())} values")

# what the NEXT step would do if the run had never stopped
ref = Tiny(); ref.load_state_dict(m.state_dict())
ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3)
ref_opt.load_state_dict(opt.state_dict())
ref_opt.zero_grad(); torch.nn.functional.mse_loss(ref(x), y).backward(); ref_opt.step()
uninterrupted = {n: p.detach().clone() for n, p in ref.named_parameters()}

# ---- save, then rebuild the way a restart does -------------------------
weights_store._save_optim(opt, m, D) if hasattr(weights_store, "_save_optim") \
    else weights_store.save_optim(opt, m, D)

fresh = Tiny(); fresh.load_state_dict(m.state_dict())
fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
check("a fresh optimiser starts with NO moments",
      all("exp_avg" not in fresh_opt.state[p] for p in fresh.parameters()),
      "this is what every restart used to run with")

weights_store._load_optim(fresh_opt, fresh, D)
back = {n: fresh_opt.state[p].get("exp_avg")
        for n, p in fresh.named_parameters()}
worst = max(float((back[n].float() - live[n].float()).abs().max())
            for n in live)
check("moments come back across a save/load", worst < 2e-2,
      f"largest difference {worst:.2e} (bf16 packing)")

worst_sq = max(float((fresh_opt.state[p].get("exp_avg_sq").float()
                      - live_sq[n].float()).abs().max())
               for n, p in fresh.named_parameters() if n in live_sq)
check("...second moments too", worst_sq < 2e-2, f"largest {worst_sq:.2e}")

# ---- and the step it then takes is the uninterrupted one ---------------
fresh_opt.zero_grad(); torch.nn.functional.mse_loss(fresh(x), y).backward()
fresh_opt.step()
restored_gap = max(float((p - uninterrupted[n]).abs().max())
                   for n, p in fresh.named_parameters())

cold = Tiny(); cold.load_state_dict(m.state_dict())
cold_opt = torch.optim.AdamW(cold.parameters(), lr=1e-3)
cold_opt.zero_grad(); torch.nn.functional.mse_loss(cold(x), y).backward()
cold_opt.step()
cold_gap = max(float((p - uninterrupted[n]).abs().max())
               for n, p in cold.named_parameters())

check("the step after a restart matches the uninterrupted step",
      restored_gap < cold_gap / 10,
      f"restored off by {restored_gap:.2e}, a fresh optimiser by {cold_gap:.2e}"
      f"  ({cold_gap / max(restored_gap, 1e-12):.0f}x worse)")

# ---- the read path must actually call it ------------------------------
import inspect                                            # noqa: E402
import train                                              # noqa: E402
src = inspect.getsource(train.cmd_read)
check("cmd_read restores the moments it saves",
      "_load_optim" in src,
      "it wrote optim.npz at every checkpoint and never read it back")

print("\n  all optimiser-restore tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
