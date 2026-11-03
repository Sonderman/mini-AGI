"""
The recurrence depth can be sampled while training, from config.yaml.

Training does not halt: with targets present the model runs every one of
max_steps passes and takes the halting-weighted mixture, because the halting
head only learns through those weights. So adaptive compute saves nothing
during training - every character costs all six passes, and the recurrence is
24 of the 26 block applications.

Sampling the depth is the way to spend less. The setting is NOT the mean
depth: the draw is poisson(mean) + 1, clipped, so 2.0 averages 3.0 of 6.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.recur import RecurConfig, RecurCoder  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    ok = ok and cond


def depths(mean, n=4000, max_steps=6, min_steps=1):
    cfg = RecurConfig(vocab_size=265, d_model=32, n_head=2, d_ff=64, block=64,
                      n_prelude=1, n_recur=1, n_coda=1, n_layer=3,
                      max_steps=max_steps, min_steps=min_steps,
                      use_pool=False, train_steps_mean=mean)
    m = RecurCoder(cfg)
    m.train()
    return torch.tensor([float(m.sample_depth()) for _ in range(n)])


# --- zero keeps the old behaviour, which every checkpoint was trained under
d = depths(0.0, n=200)
check("0 always runs max_steps", bool((d == 6).all()), f"mean {d.mean():.2f}")

# --- eval never samples, whatever the setting -----------------------------
cfg = RecurConfig(vocab_size=265, d_model=32, n_head=2, d_ff=64, block=64,
                  n_prelude=1, n_recur=1, n_coda=1, n_layer=3,
                  use_pool=False, train_steps_mean=2.0)
m = RecurCoder(cfg)
m.eval()
check("inference always runs the full ceiling",
      all(m.sample_depth() == cfg.max_steps for _ in range(50)))

# --- above zero it samples, and stays inside the bounds -------------------
d = depths(2.0)
check("sampling stays within [min_steps, max_steps]",
      bool((d >= 1).all() and (d <= 6).all()),
      f"min {d.min():.0f} max {d.max():.0f}")
check("it really varies", len(d.unique()) > 3, f"{len(d.unique())} distinct")

# --- the setting is not the mean, and the docs say so ---------------------
check("2.0 averages about 3 of 6, not 2",
      2.8 < d.mean() < 3.2, f"mean {d.mean():.2f}")

# --- deeper settings cost more --------------------------------------------
a, b = depths(1.0).mean(), depths(3.0).mean()
check("a larger setting means more compute", a < b, f"{a:.2f} then {b:.2f}")

# --- min_steps is respected ------------------------------------------------
d = depths(2.0, n=1500, min_steps=3)
check("min_steps is a floor", bool((d >= 3).all()), f"min {d.min():.0f}")

# --- and it is reachable from config.yaml, not only from code -------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.config import load, get  # noqa: E402
check("config.yaml carries it",
      get(load(), "model.train_steps_mean", None) is not None)
src = open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "train.py")).read()
check("the read path takes it from the flag, not the checkpoint",
      "cfg.train_steps_mean = float(args.train_steps_mean)" in src)

print("\n  " + ("all train-depth tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
