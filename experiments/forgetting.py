#!/usr/bin/env python3
"""
Does learning from a pure stream actually work?

Two ways of showing the same data to the same model, for the same number of
steps, on the same seed:

  STREAM   read one corpus at a time, in order, the way a person reads. Code
           for a while, then arithmetic, then dialogue, then chess.

  MIXED    every batch drawn from all four corpora at once - which is what the
           real training loop does.

Both see identical total tokens. The question is what each knows at the end.
Validation loss is measured per domain, so forgetting is visible directly:
the stream should be good at whatever it read last and worse at whatever it
read first.

    python3 stream_vs_batch.py --steps 240
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
DOMAINS = ["data_char", "data_math_char", "data_chat_char", "data_chess_char"]
SHORT = {"data_char": "code", "data_math_char": "math",
         "data_chat_char": "chat", "data_chess_char": "chess"}


def build(cfg, device, seed):
    torch.manual_seed(seed)
    m = RecurCoder(cfg).to(device)
    m._grower = None
    if cfg.use_pool:
        from minagi.pool import AutoGrow
        m._grower = AutoGrow(grow_k=cfg.grow_k, idle_frac_max=0.35,
                             max_experts=cfg.pool_max)
    opt = torch.optim.AdamW(
        [{"params": [p for p in m.parameters() if p.dim() >= 2],
          "weight_decay": 0.1},
         {"params": [p for p in m.parameters() if p.dim() < 2],
          "weight_decay": 0.0}],
        lr=3e-3, betas=(0.9, 0.95))
    return m, opt


@torch.no_grad()
def per_domain_val(model, corpora, rng, batches=6):
    out = {}
    model.eval()
    for name, c in corpora.items():
        losses = []
        for _ in range(batches):
            x, y = c.batch_for("val", rng)
            _, l = model(x, y)
            losses.append(float(l))
        out[SHORT[name]] = float(np.mean(losses))
    model.train()
    return out


def train(mode, cfg, corpora, device, steps, batch, seed, log):
    m, opt = build(cfg, device, seed)
    rng = np.random.default_rng(seed)
    names = list(corpora)
    per_phase = max(1, steps // len(names))
    history = []
    for step in range(steps):
        lr = lr_at(step, steps, 3e-3, max(4, steps // 20))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        if mode == "stream":
            # one corpus at a time, in order - a pure stream
            c = corpora[names[min(step // per_phase, len(names) - 1)]]
            c.batch = batch
            x, y = c.batch_for("train", rng)
        else:
            # every batch contains all four - this is replay
            xs, ys = [], []
            share = [0.25, 0.15, 0.25, 0.35]
            used = 0
            for j, n in enumerate(names):
                k = batch - used if j == len(names) - 1 else \
                    max(1, int(round(batch * share[j])))
                k = max(0, min(k, batch - used))
                if not k:
                    continue
                corpora[n].batch = k
                a, b = corpora[n].batch_for("train", rng)
                xs.append(a); ys.append(b); used += k
            x, y = torch.cat(xs), torch.cat(ys)
        _, loss = m(x, y)
        if cfg.use_pool:
            loss = loss + cfg.pool_aux * m.pool_aux()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, steps // 8) == 0:
            v = per_domain_val(m, corpora, np.random.default_rng(7))
            phase = SHORT[names[min(step // per_phase, len(names) - 1)]] \
                if mode == "stream" else "all"
            extra = ""
            if m._grower is not None:
                rec = m._grower.step(np.mean(list(v.values())), m.pool, step)
                if rec["grew"]:
                    known = {id(p) for g in opt.param_groups for p in g["params"]}
                    fresh = [p for p in m.parameters() if id(p) not in known]
                    mats = [p for p in fresh if p.dim() >= 2]
                    vecs = [p for p in fresh if p.dim() < 2]
                    if mats:
                        opt.add_param_group({"params": mats, "weight_decay": 0.1})
                    if vecs:
                        opt.add_param_group({"params": vecs, "weight_decay": 0.0})
                extra = f"  pool {m.pool.n_experts()}" + \
                        (f" (+{rec['grew']})" if rec["grew"] else "")
            history.append({"step": step + 1, "reading": phase, **v})
            log(f"  {mode:<6} step {step+1:>4} reading {phase:<5} " +
                "  ".join(f"{k} {v[k]:.3f}" for k in v) + extra)
    return m, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/stream_vs_batch.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(6)
    def mkcfg(pool):
        c = RecurConfig(vocab_size=265, n_head=4, d_model=128, block=args.block,
                        d_ff=256, n_layer=4, n_prelude=1, n_recur=2, n_coda=1,
                        max_steps=2, use_pool=pool, pool_experts=16,
                        pool_d_ff=64, pool_top_k=4, pool_max=128)
        c.grow_k = 16
        return c

    corpora = {d: Corpus(d, device, args.block, args.batch) for d in DOMAINS}
    print(f"same seed, same {args.steps} steps, same data - only the "
          f"architecture and the order change\n")

    def log(s):
        print(s, flush=True)

    t0 = time.time()
    runs = {}
    for pool in (False, True):
        for mode in ("stream", "mixed"):
            tag = ("pooled" if pool else "dense") + "/" + mode
            print(f"--- {tag} ---")
            _, h = train(mode, mkcfg(pool), corpora, device, args.steps,
                         args.batch, args.seed, log)
            runs[tag] = h
            print()

    doms = [d for d in runs["dense/stream"][-1] if d not in ("step", "reading")]
    print(f"{'':<16}" + "".join(f"{d:>9}" for d in doms)
          + f"{'mean':>9}{'spread':>9}")
    for tag, h in runs.items():
        f = h[-1]
        vals = [f[d] for d in doms]
        print(f"{tag:<16}" + "".join(f"{v:>9.3f}" for v in vals)
              + f"{np.mean(vals):>9.3f}{max(vals)-min(vals):>9.3f}")

    ds = runs["dense/stream"][-1]; ps = runs["pooled/stream"][-1]
    sd = max(ds[d] for d in doms) - min(ds[d] for d in doms)
    sp = max(ps[d] for d in doms) - min(ps[d] for d in doms)
    print(f"\nstreaming only - does the pool reduce forgetting?")
    print(f"  spread  dense {sd:.3f}  ->  pooled {sp:.3f}   "
          f"({100*(sp-sd)/sd:+.1f}%)")
    first = doms[0]
    print(f"  {first} (read first, most forgotten): "
          f"dense {ds[first]:.3f} -> pooled {ps[first]:.3f}   "
          f"({100*(ps[first]-ds[first])/ds[first]:+.1f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"runs": runs, "steps": args.steps, "block": args.block},
              open(args.out, "w"), indent=1)
    print(f"\n{time.time()-t0:.0f}s, wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
