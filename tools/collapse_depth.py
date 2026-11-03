#!/usr/bin/env python3
"""
Turn the 4-blocks-per-pass model into a 1-block-per-row model, keeping the
weights.

WHAT CHANGES. n_recur 3 -> 1 and n_coda 1 -> 0, so the four pooled blocks that
run per pass become one block that runs per row, and max_steps rises from 6 to
24 so the ceiling of 24 block applications is unchanged. Halting then speaks
after every row instead of every fourth, which is the whole point: today the
shallowest thing the model can ask for is four rows.

WHAT DOES NOT CHANGE. The pool - 550M of experts - is untouched, and so are
the prelude, the adapter, the head, the norms and the halting head. Of 561M
parameters, 4.2M change shape. That is why this is a conversion and not a
retrain.

WHICH BLOCK BECOMES THE ONE BLOCK is not obvious and is not guessed at. The
single block plays two roles that were separate: it feeds the next row (what
recur did) and its output is read by ln_f and the head (what coda did). So
every candidate is built and scored on held-out, and the one that starts
lowest wins.

    python3 tools/collapse_depth.py --measure          # score the candidates
    python3 tools/collapse_depth.py --pick mean_recur --out weights_collapsed
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.recur import RecurConfig, RecurCoder      # noqa: E402
from minagi.paged import PagedPool                    # noqa: E402
from minagi.stream import FolderEvaluator             # noqa: E402
from minagi.precision import set_compute_dtype        # noqa: E402

BLOCK_KEYS = ("attn.qkv.weight", "attn.proj.weight", "ln1.weight", "ln2.weight")


def candidates(core, routers):
    """Every way of making one pooled block out of the four there are."""
    have = [f"recur.{i}" for i in range(3)] + ["coda.0"]
    out = {}
    for src in have:
        out[src.replace(".", "")] = [src]
    out["mean_recur"] = [f"recur.{i}" for i in range(3)]
    out["mean_all"] = have
    return out


def build_state(core, routers, srcs):
    """The new state dict: everything carried over, plus the collapsed block."""
    sd = {}
    for k in core.files:
        if k.startswith(("recur.", "coda.")):
            continue
        sd[k] = torch.from_numpy(core[k])
    for suffix in BLOCK_KEYS:
        vs = [core[f"{s}.{suffix}"] for s in srcs if f"{s}.{suffix}" in core.files]
        sd[f"recur.0.{suffix}"] = torch.from_numpy(np.mean(vs, axis=0))
    for suffix in ("mlp.router.weight", "mlp.depth_emb"):
        vs = [routers[f"{s}.{suffix}"] for s in srcs
              if f"{s}.{suffix}" in routers.files]
        sd[f"recur.0.{suffix}"] = torch.from_numpy(np.mean(vs, axis=0))
    return sd


def new_cfg(man, max_steps, min_steps):
    d = dict(man["cfg"])
    d.update(n_recur=1, n_coda=0, n_layer=d["n_prelude"] + 1,
             max_steps=max_steps, min_steps=min_steps)
    return RecurConfig(**{k: v for k, v in d.items()
                          if k in RecurConfig.__dataclass_fields__})


def load_new(wdir, experts_dir, srcs, man, max_steps, min_steps, device,
             resident=None):
    core = np.load(os.path.join(wdir, "core.npz"))
    routers = np.load(os.path.join(wdir, "routers.npz"))
    cfg = new_cfg(man, max_steps, min_steps)
    cfg.pool_max = int(man["n_experts"])
    model = RecurCoder(cfg).to(device)
    pool = PagedPool(experts_dir, cfg.d_model, cfg.pool_d_ff,
                     int(man["n_experts"]),
                     resident=resident or man["cfg"]["pool_resident"],
                     ram_capacity=128, device=device).to(device)
    model.pool = pool
    for m in model.modules():
        if hasattr(m, "_pool"):
            m._pool = [pool]
    pool.attach_sites(model)
    pool.gate.data.copy_(torch.from_numpy(routers["pool.gate"]).to(device))
    pool.load_telemetry(man.get("telemetry"))
    sd = build_state(core, routers, srcs)
    missing, unexpected = model.load_state_dict(
        {k: v.to(device) for k, v in sd.items()}, strict=False)
    keep = [k for k in missing if not k.startswith(("pool.", "w1", "w2", "w3"))]
    return model, cfg, pool, keep, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--experts", default=None,
                    help="expert directory to read; defaults to <weights>/experts")
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--held-out", default="data/val")
    ap.add_argument("--eval-chunks", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--pick", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    set_compute_dtype("bf16" if dev.type == "cuda" else "fp32")
    man = json.load(open(os.path.join(a.weights, "manifest.json")))
    core = np.load(os.path.join(a.weights, "core.npz"))
    routers = np.load(os.path.join(a.weights, "routers.npz"))
    experts = a.experts or os.path.join(a.weights, "experts")
    cands = candidates(core, routers)

    if a.measure:
        print(f"  scoring {len(cands)} ways to collapse 4 pooled blocks into 1")
        print(f"  ceiling stays at 24 rows; halting now speaks after each one\n")
        rows = []
        for name, srcs in cands.items():
            model, cfg, pool, missing, unexpected = load_new(
                a.weights, experts, srcs, man, a.max_steps, a.min_steps, dev)
            model.eval()
            ev = FolderEvaluator(model, a.held_out, 512, cfg.block, dev)
            v = ev.run(a.eval_chunks)
            se = v.pop("stderr", 0.0)
            s = float(np.mean(list(v.values())))
            rows.append((name, s, se))
            print(f"    {name:<12} held-out {s:.4f} +/-{se:.4f}"
                  f"   from {', '.join(srcs)}", flush=True)
            del model, pool
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        rows.sort(key=lambda r: r[1])
        print(f"\n  best: {rows[0][0]} at {rows[0][1]:.4f}")
        print(f"  for scale, the model before the change was at "
              f"{man.get('val') or float('nan'):.4f}")
        return 0

    if not a.pick or not a.out:
        raise SystemExit("give --measure, or --pick NAME --out DIR")
    srcs = cands[a.pick]
    os.makedirs(a.out, exist_ok=True)
    # the experts are the bulk and do not change; copy rather than link,
    # because a paged run writes them back on eviction and a link would
    # corrupt the directory this was made from
    dst = os.path.join(a.out, "experts")
    if not os.path.exists(dst):
        print(f"  copying {int(man['n_experts'])} experts...", flush=True)
        shutil.copytree(experts, dst)
    sd = build_state(core, routers, srcs)
    cfg = new_cfg(man, a.max_steps, a.min_steps)
    np.savez(os.path.join(a.out, "core.npz"),
             **{k: v.numpy() for k, v in sd.items()
                if not k.startswith("recur.0.mlp.")})
    np.savez(os.path.join(a.out, "routers.npz"),
             **{"recur.0.mlp.router.weight":
                sd["recur.0.mlp.router.weight"].numpy(),
                "recur.0.mlp.depth_emb": sd["recur.0.mlp.depth_emb"].numpy(),
                "pool.gate": routers["pool.gate"]})
    m2 = dict(man)
    from dataclasses import asdict
    m2["cfg"] = asdict(cfg)
    m2["cfg"]["pool_resident"] = man["cfg"]["pool_resident"]
    m2["cfg"]["pool_max"] = int(man["n_experts"])
    m2["core_tensors"] = sorted(k for k in sd if not k.startswith("recur.0.mlp."))
    m2["router_tensors"] = ["recur.0.mlp.router.weight", "recur.0.mlp.depth_emb",
                            "pool.gate"]
    m2.pop("val", None)          # the old score describes a different model
    json.dump(m2, open(os.path.join(a.out, "manifest.json"), "w"))
    print(f"  wrote {a.out} from {', '.join(srcs)}")
    print(f"  Adam's moments are NOT carried over: the block they described "
          f"no longer exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
