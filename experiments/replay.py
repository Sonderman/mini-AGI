#!/usr/bin/env python3
"""
Does a replay buffer beat slowing the trunk down?

Where the run stands: streaming through four corpora in order, with nothing
repeated, the best setting found so far is a growing pool plus a trunk learning
at a fraction of the pool's rate - mean 3.370, spread 0.878. Mixed batches,
which are not streaming at all, reach mean 2.426 with spread 1.410. So the
stream is behind on mean and ahead on evenness.

Replay is the obvious next lever and the one that most resembles cheating: keep
a buffer of what has already been read and fold a little of it into each batch.
It is still a stream - nothing is stored twice, the buffer only remembers WHERE
it has been and re-draws from there - but it does revisit.

Two things worth separating, so this sweeps both:

  REPLAY FRACTION   how much of each batch comes from the past. At 0.0 this is
                    the pure stream; at 1.0 nothing new is ever read.

  TRUNK RATE        the knob that already works. If replay only reproduces what
                    slowing the trunk already did, the two will not stack, and
                    the pair at x1.0 will match the pair at x0.1.

The question is not whether replay helps. It is whether it helps for a reason
the trunk rate has not already covered.

    python3 replay_test.py --steps 200 --replays 0.0 0.1 0.25 0.5
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from experiments.corpus import Corpus
from experiments.schedule import lr_at
from minagi.recur import RecurConfig, RecurCoder
from minagi.pool import PooledMLP, AutoGrow

DOM = ["data_char", "data_math_char", "data_chat_char", "data_chess_char"]
SHORT = {"data_char": "code", "data_math_char": "math",
         "data_chat_char": "chat", "data_chess_char": "chess"}
DOMS = ["code", "math", "chat", "chess"]


@torch.no_grad()
def val(m, cs, batches=6):
    m.eval()
    out = {}
    for n, c in cs.items():
        keep = c.batch
        c.batch = 8
        L = []
        for _ in range(batches):
            x, y = c.batch_for("val", np.random.default_rng(7))
            L.append(float(m(x, y)[1]))
        c.batch = keep
        out[SHORT[n]] = float(np.mean(L))
    m.train()
    return out


def split_params(m):
    sites = [x for x in m.modules() if isinstance(x, PooledMLP)]
    pool_ids = {id(p) for e in m.pool.experts for p in e.parameters()}
    pool_ids |= {id(s.router.weight) for s in sites}
    pool_ids |= {id(s.depth_emb) for s in sites}
    pool_ids.add(id(m.pool.gate))
    trunk, pool = [], []
    for p in m.parameters():
        (pool if id(p) in pool_ids else trunk).append(p)
    return trunk, pool


def run(replay, mult, cs, steps, batch, block, seed, base_lr, device, grow):
    """One streaming pass, with `replay` of each batch drawn from what is past."""
    torch.manual_seed(seed)
    cfg = RecurConfig(vocab_size=265, n_head=4, d_model=128, block=block,
                      d_ff=256, n_layer=4, n_prelude=1, n_recur=2, n_coda=1,
                      max_steps=2, use_pool=True, pool_experts=32,
                      pool_d_ff=64, pool_top_k=4, pool_max=160)
    m = RecurCoder(cfg).to(device)
    trunk, pool = split_params(m)
    opt = torch.optim.AdamW(
        [{"params": trunk, "weight_decay": 0.1, "name": "trunk"},
         {"params": pool, "weight_decay": 0.1, "name": "pool"}],
        lr=base_lr, betas=(0.9, 0.95))
    grower = AutoGrow(grow_k=16, idle_frac_max=0.35,
                      max_experts=160) if grow else None
    rng = np.random.default_rng(seed)
    names = list(cs)
    per = max(1, steps // len(names))

    for step in range(steps):
        lr = lr_at(step, steps, base_lr, max(4, steps // 20))
        for g in opt.param_groups:
            g["lr"] = lr * (mult if g.get("name") == "trunk" else 1.0)
        opt.zero_grad(set_to_none=True)

        here = min(step // per, len(names) - 1)
        n_old = int(round(batch * replay)) if here > 0 else 0
        n_old = min(n_old, batch - 1)          # always read something new
        n_new = batch - n_old

        cur = cs[names[here]]
        cur.batch = n_new
        xs, ys = list(cur.batch_for("train", rng))
        xs, ys = [xs], [ys]

        if n_old:
            # draw the old share from the corpora already passed, evenly. The
            # buffer is the corpus itself plus the knowledge of how far the
            # stream has got - no sample is ever stored.
            past = names[:here]
            take = [n_old // len(past)] * len(past)
            for i in range(n_old - sum(take)):
                take[i % len(past)] += 1
            for nm, k in zip(past, take):
                if not k:
                    continue
                cs[nm].batch = k
                a, b = cs[nm].batch_for("train", rng)
                xs.append(a)
                ys.append(b)
        x, y = torch.cat(xs), torch.cat(ys)

        _, loss = m(x, y)
        loss = loss + cfg.pool_aux * m.pool_aux()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        if grower is not None and (step + 1) % max(1, steps // 8) == 0:
            v = val(m, cs, batches=2)
            rec = grower.step(np.mean(list(v.values())), m.pool, step)
            if rec["grew"]:
                known = {id(p) for g in opt.param_groups for p in g["params"]}
                fresh = [p for p in m.parameters() if id(p) not in known]
                if fresh:
                    opt.add_param_group({"params": fresh, "weight_decay": 0.1,
                                         "name": "pool"})
    return val(m, cs), m.pool.n_experts()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--replays", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--mults", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--out", default="runs/replay_test.json")
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.set_num_threads(6)
    cs = {d: Corpus(d, dev, a.block, a.batch) for d in DOM}
    print(f"streaming, pool growing. replay fraction x trunk rate, "
          f"{a.steps} steps each\n")
    print(f"{'replay':>7}{'trunk':>7}" + "".join(f"{d:>9}" for d in DOMS)
          + f"{'mean':>9}{'spread':>9}{'pool':>7}")
    t0 = time.time()
    out = {}
    for mult in a.mults:
        for r in a.replays:
            v, n = run(r, mult, cs, a.steps, a.batch, a.block, 0, a.lr, dev, True)
            vals = [v[d] for d in DOMS]
            key = f"r{r}_m{mult}"
            out[key] = {"replay": r, "mult": mult, "val": v, "pool": n,
                        "mean": float(np.mean(vals)),
                        "spread": float(max(vals) - min(vals))}
            print(f"{r:>7.2f}{mult:>7.2f}"
                  + "".join(f"{x:>9.3f}" for x in vals)
                  + f"{np.mean(vals):>9.3f}{max(vals)-min(vals):>9.3f}{n:>7}",
                  flush=True)
        print()

    BASE_MEAN, BASE_SPREAD = 3.370, 0.878
    best = min(out.values(), key=lambda d: d["mean"])
    print(f"best mean: replay {best['replay']:.2f} trunk x{best['mult']:g} -> "
          f"{best['mean']:.3f} (spread {best['spread']:.3f})")
    print(f"target to beat (no replay, trunk x0.1): mean {BASE_MEAN:.3f}, "
          f"spread {BASE_SPREAD:.3f}")
    if best["mean"] < BASE_MEAN:
        print(f"  -> improved by {BASE_MEAN - best['mean']:.3f} nats")
    else:
        print(f"  -> did NOT beat it ({best['mean'] - BASE_MEAN:+.3f})")

    # does replay stack with the trunk knob, or replace it?
    for mult in a.mults:
        row = [out[f"r{r}_m{mult}"] for r in a.replays if f"r{r}_m{mult}" in out]
        if len(row) > 1:
            gain = row[0]["mean"] - min(x["mean"] for x in row)
            print(f"  replay is worth {gain:+.3f} mean at trunk x{mult:g}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n{time.time()-t0:.0f}s, wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
