#!/usr/bin/env python3
"""
Is it the ARCHITECTURE, or would any model of this size do?

The run itself shows eight domains improving together under batch-1 single-pass
streaming. That is suggestive, but on its own it does not say the routed pool
is responsible - an over-parameterised dense model fed the same stream might do
just as well, and then the interesting claim would be about streaming, not
about experts.

So this trains two models that differ in ONE thing:

  DENSE   one feed-forward in the recurrent block. Every parameter gets a
          gradient from every character.

  POOLED  the same parameter budget split into experts, top-k routed. A
          character updates only the experts it selects; the rest are
          untouched by construction.

Matched on total parameters, same seed, same stream, same optimiser, same
number of batch-1 steps.

The curriculum is the hard version of the question - strictly SEQUENTIAL, with
no revisiting:

    chess ... then code ... then stories ... then arithmetic

and every domain is scored after every phase. Catastrophic forgetting, if it
happens, is the first domain's curve bending upward as soon as reading moves
on. Retention is that curve staying flat while the others come down.

    python3 tools/cl_control.py --steps-per-domain 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.recur import RecurConfig, RecurCoder            # noqa: E402
from minagi.stream import FileReader, FolderEvaluator       # noqa: E402
from minagi.ingest import collect, as_stream                # noqa: E402
from minagi.precision import set_compute_dtype              # noqa: E402

DOMAINS = ["chess", "code", "stories", "arithmetic"]


def _takes_two(cls):
    import inspect
    try:
        ps = list(inspect.signature(cls.__init__).parameters)[1:]
        return len(ps) >= 2
    except (TypeError, ValueError):
        return False


def _widen(old, d_model, d_ff):
    """Rebuild whatever MLP the block uses at a different inner width."""
    import torch.nn as nn
    lin = [m for m in old.modules() if isinstance(m, nn.Linear)]
    if not lin:
        raise SystemExit("  cannot widen this MLP: no Linear inside")
    import copy
    new = copy.deepcopy(old)
    for name, m in list(new.named_modules()):
        if not isinstance(m, nn.Linear):
            continue
        parent = new if "." not in name else dict(new.named_modules())[
            name.rsplit(".", 1)[0]]
        attr = name.rsplit(".", 1)[-1]
        if m.in_features == d_model:                 # up-projection
            setattr(parent, attr, nn.Linear(d_model, d_ff, bias=m.bias is not None))
        elif m.out_features == d_model:              # down-projection
            setattr(parent, attr, nn.Linear(d_ff, d_model, bias=m.bias is not None))
    return new


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(kind, base, experts, pool_ff, top_k, dense_ff, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = RecurConfig(**base)
    if kind == "pooled":
        cfg.use_pool = True
        cfg.pool_experts = experts
        cfg.pool_max = experts
        cfg.pool_d_ff = pool_ff
        cfg.pool_top_k = top_k
    else:
        cfg.use_pool = False
    m = RecurCoder(cfg).to(device)
    if kind == "dense":
        # Only the RECURRENT block carries the pool in the pooled model, so
        # only its feed-forward may be widened here. Setting cfg.d_ff widened
        # the two prelude blocks as well and made the "matched" dense model
        # 134% too big - which would have handed it the comparison.
        import torch.nn as nn
        from minagi.recur import Block
        for blk in m.recur:
            old = blk.mlp
            dm = cfg.d_model
            blk.mlp = type(old)(dm, dense_ff) if _takes_two(type(old)) \
                else _widen(old, dm, dense_ff)
            blk.mlp = blk.mlp.to(device)
    m._grower = None                     # fixed capacity: no growth, no pruning
    return m, cfg


def stream_for(domain, need, root="data/train"):
    files = collect([os.path.join(root, domain)])
    parts, total = [], 0
    for p in files:
        try:
            d = as_stream(p)
        except OSError:
            continue
        if len(d) < 8:
            continue
        parts.append(d)
        total += len(d)
        if total >= need:
            break
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-per-domain", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=75)
    ap.add_argument("--eval-chunks", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--pool-ff", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--trunk-lr-mult", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--held-out", default="data/val")
    ap.add_argument("--out", default="runs/cl/control.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_compute_dtype(a.precision)

    base = dict(vocab_size=265, d_model=a.d_model, n_head=4, block=a.context,
                n_prelude=2, n_recur=1, n_coda=0, max_steps=a.max_steps,
                d_ff=a.d_model * 3, tie_embeddings=True)

    # match the parameter budget: the pool's experts against one wide dense FF
    pool_params = a.experts * 3 * a.pool_ff * a.d_model
    dense_ff = max(1, int(round(pool_params / (3 * a.d_model))))
    mp, cfgp = build("pooled", base, a.experts, a.pool_ff, a.top_k,
                     dense_ff, a.seed, device)
    md, cfgd = build("dense", base, a.experts, a.pool_ff, a.top_k,
                     dense_ff, a.seed, device)
    print(f"  pooled {n_params(mp)/1e6:7.3f}M params   "
          f"{a.experts} experts x {a.pool_ff} wide, top-{a.top_k}")
    print(f"  dense  {n_params(md)/1e6:7.3f}M params   one FF {dense_ff} wide")
    print(f"  mismatch {abs(n_params(mp)-n_params(md))/n_params(mp)*100:.2f}%")
    print(f"  curriculum (no revisiting): {' -> '.join(DOMAINS)}")
    print(f"  {a.steps_per_domain} batch-1 steps per domain, "
          f"chunk {a.chunk}, context {a.context}\n")

    need = (a.steps_per_domain + 4) * a.chunk
    data = {d: stream_for(d, need) for d in DOMAINS}
    results = {}
    for kind, model in (("pooled", mp), ("dense", md)):
        ev = FolderEvaluator(model, a.held_out, a.chunk, a.context, device,
                             per_domain=True)
        for g in list(ev.groups):
            if g not in DOMAINS:
                del ev.groups[g]
        trunk = [p for n, p in model.named_parameters()
                 if not n.startswith("pool.") and ".mlp." not in n]
        rest = [p for n, p in model.named_parameters()
                if n.startswith("pool.") or ".mlp." in n]
        opt = torch.optim.AdamW(
            [{"params": trunk, "lr": a.lr * a.trunk_lr_mult, "weight_decay": 0.1},
             {"params": rest, "lr": a.lr, "weight_decay": 0.1}],
            lr=a.lr, betas=(0.9, 0.95))
        rows, done, t0 = [], 0, time.time()

        def score(phase, dom):
            d = ev.run(a.eval_chunks)
            d.pop("stderr", None)
            rows.append({"steps": done, "phase": phase, "reading": dom,
                         **{k: float(v) for k, v in d.items()}})
            return d

        d0 = score(-1, "none")
        print(f"  [{kind}] before any training: "
              + "  ".join(f"{k} {v:.3f}" for k, v in sorted(d0.items())))
        for pi, dom in enumerate(DOMAINS):
            reader = FileReader(model, data[dom], dom, a.chunk, a.context,
                                device)
            inphase = 0
            while inphase < a.steps_per_domain:
                if reader.done():
                    reader = FileReader(model, data[dom], dom, a.chunk,
                                        a.context, device)
                opt.zero_grad(set_to_none=True)
                loss = reader.step(learn=True, aux_weight=cfgp.pool_aux
                                   if kind == "pooled" else 0.0)
                if loss is None:
                    reader = FileReader(model, data[dom], dom, a.chunk,
                                        a.context, device)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                opt.step()
                inphase += 1
                done += 1
                if inphase % a.eval_every == 0:
                    score(pi, dom)
            d = rows[-1]
            print(f"  [{kind}] after {dom:<11} ({done:>4} steps) "
                  + "  ".join(f"{k} {d[k]:.3f}" for k in DOMAINS)
                  + f"   [{time.time()-t0:.0f}s]")
        results[kind] = {"params": n_params(model), "rows": rows}

    out = {"domains": DOMAINS, "steps_per_domain": a.steps_per_domain,
           "chunk": a.chunk, "context": a.context, "seed": a.seed,
           "experts": a.experts, "top_k": a.top_k, "pool_ff": a.pool_ff,
           "dense_ff": dense_ff, "lr": a.lr,
           "trunk_lr_mult": a.trunk_lr_mult, "runs": results}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  -> {a.out}")

    # forgetting = how much the FIRST domain decayed once reading moved on
    for kind in ("pooled", "dense"):
        r = results[kind]["rows"]
        first = DOMAINS[0]
        best = min(x[first] for x in r if x["reading"] == first)
        end = r[-1][first]
        print(f"  {kind:>6}: {first} best-while-reading {best:.4f} "
              f"-> final {end:.4f}   forgetting {end-best:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
