#!/usr/bin/env python3
"""
What gradient clipping is actually doing to this model.

Three questions, all answered from real training steps against the live
weights, read-only so nothing on disk is touched:

  1. the per-step gradient norm distribution - what fraction of steps a clip
     of 1.0 truncates, and what fraction a clip of 3.0 would
  2. how that norm splits between the trunk and the pool. The clip is global
     over all 606M parameters, so if the pool carries essentially all of it,
     the trunk's gradient scale is set by whatever the pool is doing
  3. the immediate change to the Adam update if the clip moved, computed from
     the CURRENT moments rather than inferred - which is the transient, the
     only part scale-invariance does not absorb

    python3 tools/clip_probe.py --steps 200
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T                                    # noqa: E402
from minagi.ingest import as_stream, collect         # noqa: E402
from minagi.precision import amp, set_compute_dtype  # noqa: E402


def gnorm(ps):
    t = [p.grad.detach().float().norm() ** 2 for p in ps if p.grad is not None]
    return float(torch.stack(t).sum().sqrt()) if t else 0.0


def adam_step(ps, opt, scale):
    """The update Adam WOULD make with the gradient scaled by `scale`."""
    out = []
    for p in ps:
        if p.grad is None:
            continue
        st = opt.state.get(p)
        if not st or "exp_avg" not in st:
            continue
        b1, b2 = 0.9, 0.95
        t = float(st["step"]) + 1 if "step" in st else 1.0
        g = p.grad.detach().float() * scale
        m = st["exp_avg"].float() * b1 + (1 - b1) * g
        v = st["exp_avg_sq"].float() * b2 + (1 - b2) * g * g
        out.append((m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + 1e-8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--clips", default="1.0,3.0")
    a = ap.parse_args()

    dev = torch.device("cuda")
    set_compute_dtype("bf16")
    model, cfg, pool, man = T.build_paged(a.weights, dev, read_only=True)
    T.weights_store.load(model, a.weights, device=dev)
    trunk, pool_ps = T._split_trunk_pool(model)
    opt = torch.optim.AdamW(
        [{"params": trunk, "name": "trunk", "weight_decay": 0.1, "lr": 3e-5},
         {"params": pool_ps, "name": "pool", "weight_decay": 0.1, "lr": 3e-4}],
        lr=3e-4, betas=(0.9, 0.95))
    T.weights_store.load(model, a.weights, opt=opt, device=dev)
    pool.attach_optimiser(opt)
    n_state = sum(1 for g in opt.param_groups for p in g["params"]
                  if opt.state.get(p, {}).get("exp_avg") is not None)
    print(f"  {a.weights} step {man['step']:,}, {pool.n_experts()} experts")
    print(f"  {len(trunk)} trunk tensors, {len(pool_ps)} pool tensors; "
          f"{n_state} carry Adam moments\n", flush=True)

    files = sorted(collect(["data/train"]))
    rng = np.random.default_rng(0)
    tot, tr, po = [], [], []
    cos_all, mag_all = [], []
    clips = [float(x) for x in a.clips.split(",")]
    model.train()
    done = 0
    while done < a.steps:
        f = files[rng.integers(len(files))]
        d = as_stream(f)
        if len(d) < a.chunk + 2:
            continue
        s = int(rng.integers(0, len(d) - a.chunk - 1))
        arr = np.asarray(d[s:s + a.chunk + 1], np.int64)
        x = torch.tensor(arr[:-1][None], device=dev)
        y = torch.tensor(arr[1:][None], device=dev)
        opt.zero_grad(set_to_none=True)
        with amp(dev):
            _, loss = model(x, y)
        loss.backward()
        g_all = gnorm(trunk + pool_ps)
        tot.append(g_all); tr.append(gnorm(trunk)); po.append(gnorm(pool_ps))
        if done % 20 == 0 and g_all > 0:
            ps = [p for p in (trunk + pool_ps) if p.grad is not None]
            us = [adam_step(ps, opt, min(1.0, c / g_all)) for c in clips]
            f0 = torch.cat([u.reshape(-1) for u in us[0]])
            f1 = torch.cat([u.reshape(-1) for u in us[1]])
            cos_all.append(float(torch.nn.functional.cosine_similarity(
                f0[None], f1[None])))
            mag_all.append(float(f1.norm() / f0.norm().clamp_min(1e-12)))
        # keep the run realistic: clip as the trainer does, and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done += 1
        if done % 50 == 0:
            print(f"    {done}/{a.steps}", flush=True)

    t = np.array(tot); a_tr = np.array(tr); a_po = np.array(po)
    print(f"\n  === 1. per-step gradient norm over {len(t)} real steps ===")
    print(f"    median {np.median(t):.3f}   mean {t.mean():.3f}   "
          f"p90 {np.percentile(t,90):.3f}   max {t.max():.3f}")
    for c in clips:
        print(f"    clipped at {c:>4}:  {100*(t>c).mean():>5.1f}% of steps"
              f"   median truncation {np.median(np.minimum(1, c/t)):.2f}x")
    print(f"\n  === 2. where the norm comes from ===")
    print(f"    trunk  median {np.median(a_tr):.3f}   "
          f"pool median {np.median(a_po):.3f}")
    sh = a_po**2 / np.clip(a_tr**2 + a_po**2, 1e-12, None)
    print(f"    the pool carries {100*np.median(sh):.1f}% of the squared norm")
    print(f"\n  === 3. the Adam update, clip {clips[0]} vs {clips[1]} ===")
    if cos_all:
        print(f"    cosine similarity  {np.mean(cos_all):.4f}  "
              f"(1.0 = identical direction)")
        print(f"    magnitude ratio    {np.mean(mag_all):.3f}x  "
              f"(1.0 = identical size)")


if __name__ == "__main__":
    main()
