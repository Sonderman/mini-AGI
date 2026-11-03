#!/usr/bin/env python3
"""
What to spend the spare VRAM on.

Batch-1 streaming uses about 3.1 GB of 8.4, so roughly two thirds of the card
sits idle. There are three ways to spend it and they are not equivalent:

    CONTEXT     how far attention reaches. Costs cache, which is linear and
                cheap - about 0.107 MB per character.
    WORKING SET how many experts run per character. Costs activations in the
                dispatch buffers, and compute in proportion.
    EXPERT SIZE how wide each expert is, or how deep. Costs parameters, and
                is the only axis that changes tensor shapes.

That last distinction decides how this is measured. Context and working set
can be changed on the weights that exist, so they are swept by continuing the
current model and asking which configuration learns fastest from where it is.
Expert size cannot: a wider or deeper expert is a different tensor, so those
configurations have to start from scratch, and comparing them against a model
with 1800 steps of training already in it would measure the head start rather
than the shape.

So there are two sweeps here, and they answer different questions:

    python3 -m experiments capacity --mode adapt    # context x top-k, resumed
    python3 -m experiments capacity --mode fresh    # expert shape, from zero

Judged on validation loss and on peak VRAM together. A configuration that is
marginally better for twice the memory is not better - the point of the
exercise is to fill the card, once, with whatever buys the most.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from minagi.recur import RecurConfig, RecurCoder
from minagi.pool import PooledMLP, Expert
from minagi.stream import StreamSet, Evaluator, detach_caches
from experiments.schedule import lr_at
from minagi import store as weights_store

MIX = [("data_char", 0.25), ("data_math_char", 0.15),
       ("data_chat_char", 0.25), ("data_chess_char", 0.35)]


def mb():
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def clean():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_from_weights(wdir, device, context, top_k):
    """The trained model, re-shaped along the axes that do not change tensors."""
    with open(os.path.join(wdir, "manifest.json")) as f:
        man = json.load(f)
    cfgd = dict(man["cfg"])
    cfgd.update(block=context, pool_top_k=top_k)
    cfg = RecurConfig(**{k: v for k, v in cfgd.items()
                         if k in RecurConfig.__dataclass_fields__})
    m = RecurCoder(cfg).to(device)
    want = int(man["n_experts"])
    if want != m.pool.n_experts():
        m.pool.add_experts(want - m.pool.n_experts(), device=device)
    weights_store.load(m, wdir, device=device)
    # top_k lives on the call sites, not in any tensor
    for s in m.modules():
        if isinstance(s, PooledMLP):
            s.top_k = top_k
    return m, cfg


def build_fresh(device, context, top_k, d_ff, depth, n_experts):
    """A new model whose experts have a different shape."""
    cfg = RecurConfig(vocab_size=265, d_model=512, n_head=8, d_ff=1408,
                      n_prelude=2, n_recur=3, n_coda=1, max_steps=6,
                      block=context, use_pool=True, pool_experts=n_experts,
                      pool_d_ff=d_ff, pool_top_k=top_k, pool_max=512)
    if depth > 1:
        install_deep_expert(depth)
    m = RecurCoder(cfg).to(device)
    if depth > 1:
        restore_expert()
    return m, cfg


_ORIG_EXPERT_FORWARD = Expert.forward
_ORIG_EXPERT_INIT = Expert.__init__


def install_deep_expert(depth):
    """
    Make every expert `depth` SwiGLU blocks instead of one.

    Patched rather than parameterised because the pool builds experts in
    several places and the sweep needs the change to apply to all of them at
    once. restore_expert() puts it back, so nothing leaks between runs.
    """
    def __init__(self, d_model, d_ff):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "w1": nn.Linear(d_model, d_ff, bias=False),
                "w3": nn.Linear(d_model, d_ff, bias=False),
                "w2": nn.Linear(d_ff, d_model, bias=False)}))

    def forward(self, x):
        import torch.nn.functional as F
        for b in self.blocks:
            x = x + b["w2"](F.silu(b["w1"](x)) * b["w3"](x))
        return x

    Expert.__init__ = __init__
    Expert.forward = forward


def restore_expert():
    Expert.__init__ = _ORIG_EXPERT_INIT
    Expert.forward = _ORIG_EXPERT_FORWARD


def train_and_score(m, cfg, device, steps, chunk, lr, trunk_mult, seed, tag,
                    eval_every=0, replay=0.0, eval_chunks=240, accum=1):
    """
    Run the model forward through the stream and report where it gets to.

    A trajectory is recorded, not just the endpoint, because the axes being
    swept change throughput by up to half. Comparing only final losses at equal
    STEPS would flatter whichever configuration is slowest per character, and
    the run this feeds into has a time budget rather than a step budget - so
    both readings are kept and the trade is visible.
    """
    sites = [x for x in m.modules() if isinstance(x, PooledMLP)]
    pool_ids = {id(p) for e in m.pool.experts for p in e.parameters()}
    pool_ids |= {id(s.router.weight) for s in sites}
    pool_ids |= {id(s.depth_emb) for s in sites}
    pool_ids.add(id(m.pool.gate))
    trunk = [p for p in m.parameters() if id(p) not in pool_ids]
    pool = [p for p in m.parameters() if id(p) in pool_ids]
    opt = torch.optim.AdamW(
        [{"params": trunk, "name": "trunk", "weight_decay": 0.1},
         {"params": pool, "name": "pool", "weight_decay": 0.1}],
        lr=lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    rng = np.random.default_rng(seed)
    streams = StreamSet(m, MIX, chunk, cfg.block, device, rng,
                        replay=replay)
    ev = Evaluator(m, [n for n, _ in MIX], chunk, cfg.block, device)
    m.train()
    t0 = time.time()
    seen = 0
    traj = []
    for step in range(steps):
        g_lr = lr_at(step, steps, lr, max(8, steps // 20))
        for g in opt.param_groups:
            g["lr"] = g_lr * (trunk_mult if g["name"] == "trunk" else 1.0)
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            loss, _ = streams.step(learn=True, aux_weight=cfg.pool_aux)
            (loss / accum).backward()
            detach_caches(streams.active.caches)
            seen += chunk
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if eval_every and (step + 1) % eval_every == 0:
            v_ = ev.run(max(24, eval_chunks // 4))
            se_ = v_.pop("stderr", 0.0)
            val_ = float(np.mean(list(v_.values())))
            traj.append({"step": step + 1, "chars": seen,
                         "seconds": round(time.time() - t0, 1),
                         "val": val_})
            print(f"      [{tag}] step {step+1:>5}  {seen/1e6:>4.1f}M chars  "
                  f"val {val_:.4f} +/-{se_:.4f}", flush=True)
            m.train()
    el = time.time() - t0
    v = ev.run(eval_chunks)
    se = v.pop("stderr", 0.0)
    val = float(np.mean(list(v.values())))
    return {"val": val, "stderr": se, "per_domain": v, "vram_mb": mb(),
            "chars_per_s": seen / max(el, 1e-9), "seconds": el,
            "params_m": m.n_params() / 1e6, "trajectory": traj,
            "replay": replay, "revisits": streams.replays,
            "windows": streams.switches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="adapt", choices=["adapt", "fresh"])
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--trunk-mult", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-mb", type=float, default=7000.0,
                    help="peak VRAM a configuration may use and still count")
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--topks", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--d-ffs", type=int, nargs="+", default=[64, 128, 192])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--experts", type=int, default=74)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-chunks", type=int, default=240,
                    help="chunks of held-out text per domain when scoring")
    ap.add_argument("--chars", type=int, default=0,
                    help="fixed character budget per configuration; steps are "
                         "derived so accumulation is compared on equal data")
    ap.add_argument("--accums", type=int, nargs="+", default=[1],
                    help="chunks accumulated per optimiser step")
    ap.add_argument("--accum", type=int, default=1,
                    help="chunks accumulated per optimiser step; costs no "
                         "memory and multiplies the characters each step "
                         "averages over")
    ap.add_argument("--replay", type=float, nargs="+", default=[0.0],
                    help="fractions of windows that revisit a position already "
                         "read; batch stays 1 either way")
    ap.add_argument("--eval-every", type=int, default=500,
                    help="record a validation point this often, so "
                         "configurations can be compared at equal wall-clock "
                         "as well as at equal steps")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = torch.device(a.device)
    out = a.out or f"runs/results/capacity_{a.mode}.json"
    rows = []

    if a.mode == "adapt":
        combos = list(itertools.product(a.contexts, a.topks, a.replay,
                                       a.accums))
        print(f"resuming {a.weights_dir} into {len(combos)} configurations, "
              f"{a.steps} steps each\n")
        print(f"{'context':>8}{'top-k':>6}{'accum':>7}{'replay':>7}"
              f"{'steps':>7}{'val':>9}{'+/-':>9}{'VRAM':>8}{'time':>9}")
        for ctx, k, rp, ac in combos:
            clean()
            torch.manual_seed(a.seed)
            try:
                m, cfg = build_from_weights(a.weights_dir, device, ctx, k)
                # a fixed CHARACTER budget, so accumulation is compared on the
                # same data rather than on the same number of updates
                steps = (a.chars // (a.chunk * ac)) if a.chars else a.steps
                r = train_and_score(m, cfg, device, steps, a.chunk, a.lr,
                                    a.trunk_mult, a.seed, f"{ctx}/{k}",
                                    eval_every=a.eval_every, replay=rp,
                                    eval_chunks=a.eval_chunks, accum=ac)
            except torch.OutOfMemoryError:
                print(f"{ctx:>8}{k:>6}{ac:>7}{rp:>7.2f}{'OOM':>16}")
                del m
                clean()
                continue
            r.update(context=ctx, top_k=k, accum=ac, steps=steps)
            rows.append(r)
            print(f"{ctx:>8}{k:>6}{ac:>7}{rp:>7.2f}{steps:>7}"
                  f"{r['val']:>9.4f} +/-{r['stderr']:.4f}"
                  f"{r['vram_mb']:>8.0f}{r['seconds']:>8.0f}s", flush=True)
            del m
            clean()
    else:
        combos = [(d, p) for d in a.d_ffs for p in a.depths]
        print(f"from scratch, {len(combos)} expert shapes, {a.steps} steps each")
        print(f"(a fresh model at 1800 steps of disadvantage is not being "
              f"compared to the trained one - only to the others here)\n")
        print(f"{'d_ff':>6}{'depth':>7}{'val':>9}{'VRAM MB':>9}"
              f"{'char/s':>9}{'params':>9}")
        for d_ff, depth in combos:
            clean()
            torch.manual_seed(a.seed)
            try:
                m, cfg = build_fresh(device, a.contexts[0], a.topks[0],
                                     d_ff, depth, a.experts)
                r = train_and_score(m, cfg, device, a.steps, a.chunk, a.lr * 10,
                                    a.trunk_mult, a.seed, f"{d_ff}/{depth}",
                                    eval_every=a.eval_every,
                                    eval_chunks=a.eval_chunks)
            except torch.OutOfMemoryError:
                print(f"{d_ff:>6}{depth:>7}{'OOM':>9}")
                restore_expert()
                clean()
                continue
            r.update(d_ff=d_ff, depth=depth, context=a.contexts[0],
                     top_k=a.topks[0])
            rows.append(r)
            print(f"{d_ff:>6}{depth:>7}{r['val']:>9.4f}{r['vram_mb']:>9.0f}"
                  f"{r['chars_per_s']:>9.0f}{r['params_m']:>8.1f}M", flush=True)
            del m
            clean()

    if not rows:
        print("\nnothing completed")
        return 1
    ok = [r for r in rows if r["vram_mb"] <= a.budget_mb]
    print(f"\n{len(ok)}/{len(rows)} configurations fit the "
          f"{a.budget_mb:.0f} MB budget")
    if ok:
        best = min(ok, key=lambda r: r["val"])
        base = rows[0]
        keys = ("context", "top_k") if a.mode == "adapt" else ("d_ff", "depth")
        desc = "  ".join(f"{k} {best[k]}" for k in keys)
        print(f"best within budget: {desc} -> val {best['val']:.4f} "
              f"at {best['vram_mb']:.0f} MB")
        print(f"  against the first row ({'  '.join(f'{k} {base[k]}' for k in keys)}"
              f"): {best['val'] - base['val']:+.4f} val, "
              f"{best['vram_mb'] - base['vram_mb']:+.0f} MB, "
              f"{100*(best['chars_per_s']/base['chars_per_s'] - 1):+.0f}% throughput")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mode": a.mode, "steps": a.steps, "chunk": a.chunk,
                   "lr": a.lr, "rows": rows}, f, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
