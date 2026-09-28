#!/usr/bin/env python3
"""
Does reading one subject damage what the model knows about the others?

This is the claim the project makes, measured on the model the project
actually trained, rather than on a toy. The procedure is the harshest
realistic version of the question:

  1. score every held-out domain
  2. train on ONE domain, batch 1, the way the real loop does
  3. score every domain again, at intervals, as the single-domain
     exposure gets longer and longer

If the model forgets, the seven domains it is NOT reading get worse while the
one it IS reading gets better, and the damage grows with exposure. If the
architecture does what is claimed, the seven stay flat.

Two things make the comparison trustworthy:

  PAIRED   FolderEvaluator reads the same files from position zero every
           call, so the same characters are scored every time. A difference
           is the model changing, not a different sample of text. Nothing
           here has to average away sampling noise, because there is none.

  NO GROWTH  the pool is not allowed to add or prune experts during the
           probe. A pool that grew mid-probe would confound "did not forget"
           with "bought new capacity".

It never writes to the directory it is given unless told to, and the caller
is expected to hand it a COPY: paging an expert in marks it dirty, so a
training probe against a real run's weights would rewrite them.

    python3 replication/forgetting_probe.py --weights /path/to/copy \
        --domain chess --steps 256 --at 0,16,32,64,128,256
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.stream import FileReader, FolderEvaluator      # noqa: E402
from minagi.ingest import collect, as_stream               # noqa: E402
from minagi.pool import PooledMLP                          # noqa: E402
from minagi.precision import set_compute_dtype             # noqa: E402
import train as T                                          # noqa: E402


READ_ROOTS = ["data/train"]                 # set from --read-root in main()


def build_stream(domain, need, root=None):
    """
    Enough consecutive text from one subject to cover the whole visit.

    Looked up in --read-root first and then in the corpus, because a probe
    that introduces a NEW domain reads it from somewhere the training run
    does not - while the recovery phase reads the OLD domains, which are in
    the corpus where they have always been. One lookup order serves both.
    """
    roots = [root] if root else READ_ROOTS
    files = []
    for r in roots:
        files = collect([os.path.join(r, domain)])
        if files:
            break
    if not files:
        raise SystemExit(f"  no files for {domain!r} under "
                         + " or ".join(roots))
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
    ap.add_argument("--weights", required=True, help="a COPY of a weights dir")
    ap.add_argument("--domain", default="chess", help="the one subject to read")
    ap.add_argument("--read-root", default="data/train",
                    help="where the READ subject's files live. Separate from "
                         "the corpus on purpose: a probe that introduces a "
                         "NEW domain must not put it under data/train, or the "
                         "training run would start reading it too.")
    ap.add_argument("--rotate", default="",
                    help="comma-separated subjects to ROTATE through, the way "
                         "the training loop does. The control for --domain: "
                         "same steps, same data, interleaved instead of massed.")
    ap.add_argument("--visit", type=int, default=16,
                    help="steps per subject before moving on, with --rotate")
    ap.add_argument("--recover", default="",
                    help="after the massed phase, ROTATE through these "
                         "subjects and watch what comes back. This is the "
                         "test that separates displaced knowledge from "
                         "destroyed knowledge: graceful forgetting returns "
                         "far faster than it was first learned; catastrophic "
                         "forgetting has to be learned again from scratch.")
    ap.add_argument("--recover-steps", type=int, default=112)
    ap.add_argument("--recover-at", default="16,48,112")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--at", default="0,16,32,64,128,256",
                    help="cumulative step counts to score at")
    ap.add_argument("--eval-chunks", type=int, default=24)
    ap.add_argument("--held-out", default="data/val")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cold-optim", action="store_true",
                    help="build the optimiser from scratch. OFF by default, "
                         "and the default is the point: the moments in "
                         "optim.npz are part of the model's state. Discarding "
                         "them makes Adam re-estimate its second moments from "
                         "a single gradient - a large, badly scaled step in a "
                         "direction inferred from one batch - and that shows "
                         "up as a held-out jump that decays over a few hundred "
                         "steps. train.py documents the same thing as the "
                         "restart spike. Measured here, a cold optimiser "
                         "degraded domains the probe was actively READING, "
                         "which is not forgetting by any definition.")
    ap.add_argument("--warmup", type=int, default=0,
                    help="Linear LR ramp over the first N steps. A cold Adam "
                         "carries no second-moment estimate, so its opening "
                         "updates are near full step size in every direction "
                         "at once - that damages good weights, and the damage "
                         "then reads as forgetting. The interleaved control is "
                         "what catches it: that run reads EVERY subject, so "
                         "anything it degrades is damage and not forgetting. "
                         "Set the rate and this ramp until the control is "
                         "flat, and only then believe the massed arms.")
    ap.add_argument("--trunk-lr-mult", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--out", default="runs/cl/probe.json")
    ap.add_argument("--no-swap", action="store_true",
                    help="WORST CASE. Freeze the experts: the first window "
                         "admits its set, and that set serves every later "
                         "forward, held-out included, so the same experts "
                         "serve every subject. This removes the one thing "
                         "that makes the pool a pool - that different text "
                         "pulls in "
                         "different experts - and leaves a fixed network of "
                         "the same size, trained on the same stream. It is "
                         "the closest thing here to an ordinary transformer "
                         "read end to end instead of in shuffled batches.")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    marks = sorted({int(x) for x in a.at.split(",") if x.strip()})
    a.steps = max(a.steps, max(marks))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_compute_dtype(a.precision)

    if a.read_root and a.read_root not in READ_ROOTS:
        READ_ROOTS.insert(0, a.read_root)
    model, cfg, pool, man = T.build_paged(a.weights, device,
                                          resident=32, ram_capacity=96)
    model._grower = None                       # the pool may not change size
    ctx = int(man.get("context_now") or cfg.block)
    n_exp = pool._n
    print(f"  {a.weights}: step {man.get('step'):,}  {n_exp} experts  "
          f"context {ctx:,}")

    ev = FolderEvaluator(model, a.held_out, a.chunk, ctx, device,
                         per_domain=True)
    doms = sorted(ev.groups)
    print(f"  scoring {len(doms)} held-out domains: {', '.join(doms)}")

    trunk, pool_ps = T._split_trunk_pool(model)
    tg = {"params": trunk, "name": "trunk", "weight_decay": a.wd,
          "lr": a.lr * a.trunk_lr_mult}
    pg = {"params": pool_ps, "name": "pool", "weight_decay": a.wd, "lr": a.lr}
    opt = torch.optim.AdamW([tg, pg], lr=a.lr, betas=(0.9, 0.95))
    # The pool steps every expert on the card with that expert's own moments,
    # which it can only do if it knows the optimiser. Unattached, an expert
    # paged in was stepped with whatever its slot held for the one before.
    pool.attach_optimiser(opt)
    if not a.cold_optim:
        from minagi.store import _load_optim
        try:
            _load_optim(opt, model, a.weights)
            n_st = sum(1 for g in opt.param_groups for p in g["params"]
                       if opt.state.get(p, {}).get("exp_avg") is not None)
            print(f"  optimiser state restored for {n_st} tensors "
                  f"(no cold start)")
        except Exception as e:                       # noqa: BLE001
            print(f"  could not restore optimiser state ({e}); running cold")
    else:
        print("  COLD optimiser - moments discarded")
    print(f"  trunk at {a.trunk_lr_mult:g}x the pool rate"
          + ("   EXPERTS FROZEN after the first window" if a.no_swap else ""))

    lanes = [x.strip() for x in a.rotate.split(",") if x.strip()] or [a.domain]
    per_lane = (a.steps // max(1, len(lanes))) + a.visit * 3 + 8
    data = {d: build_stream(d, per_lane * a.chunk) for d in lanes}
    if len(lanes) > 1:
        print(f"  ROTATING {' -> '.join(lanes)}, {a.visit} steps each, "
              f"{a.steps} steps total at batch 1")
    else:
        print(f"  reading {len(data[lanes[0]]):,} characters of {lanes[0]}, "
              f"{a.steps} steps at batch 1")
    print()
    lane_i = [0]
    # One reader per lane, kept alive. Rebuilding it on every visit restarted
    # that lane at position zero, so a rotating run re-read the same 32K
    # characters of each subject on every pass - and the model overfitted to
    # that slice, which showed up as the domains it was READING getting worse.
    # The training loop opens each visit at a fresh offset; this continues
    # where the lane left off, which has the same effect and is cheaper.
    _readers = {d: FileReader(model, data[d], d, a.chunk, ctx, device)
                for d in lanes}

    def fresh(i):
        d = lanes[i % len(lanes)]
        r = _readers[d]
        if r.done():                     # exhausted: only then start it over
            r = _readers[d] = FileReader(model, data[d], d, a.chunk, ctx,
                                         device)
        return r

    reader = fresh(0)

    # Where does the damage actually land? The trunk is 1.9% of the
    # parameters and is active for every character; the experts are the other
    # 98% and only move when they are routed to. Recording how far each has
    # travelled from its starting point turns "the forgetting lives in the
    # trunk" from a claim into a measurement.
    trunk0 = {id(p_): p_.detach().float().clone() for p_ in trunk}
    # NOT the expert slabs. `pool.experts.*` is slot-indexed and admission
    # overwrites a slot the moment a different expert pages into it, so the
    # tensor at the end of a run belongs to a different expert than the one
    # snapshotted at the start - the difference would measure paging, not
    # learning. Only the per-expert bookkeeping that stays put is comparable:
    # the gate and the router rows.
    _slab = lambda n: ".experts." in n
    _name = {id(q): n for n, q in model.named_parameters()}
    pool0 = {id(p_): p_.detach().float().clone() for p_ in pool_ps
             if not _slab(_name.get(id(p_), ""))}

    def drift():
        def rel(ps, ref):
            num = den = 0.0
            for p_ in ps:
                b = ref.get(id(p_))
                if b is None:
                    continue
                num += float((p_.detach().float() - b).pow(2).sum())
                den += float(b.pow(2).sum())
            return (num ** 0.5) / max(den ** 0.5, 1e-12)
        return rel(trunk, trunk0), rel(pool_ps, pool0)

    rows, touched, t0 = [], set(), time.time()
    done = 0
    for mark in marks:
        while done < mark:
            if len(lanes) > 1 and done and done % a.visit == 0 \
                    and reader.name != lanes[(done // a.visit) % len(lanes)]:
                lane_i[0] = (done // a.visit) % len(lanes)
                reader = fresh(lane_i[0])
            if reader.done():
                _readers.pop(reader.name, None)
                reader = fresh(lane_i[0])
            nxt = reader.peek()
            if a.warmup:
                w = min(1.0, (done + 1) / float(a.warmup))
                for g in opt.param_groups:
                    g["lr"] = a.lr * w * (a.trunk_lr_mult
                                          if g.get("name") == "trunk" else 1.0)
            opt.zero_grad(set_to_none=True)
            loss = reader.step(learn=True, aux_weight=cfg.pool_aux)
            if loss is None:
                reader = fresh(lane_i[0])
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            # the window's forward admitted its experts; count them now
            touched.update(int(s) for s in getattr(pool, "slots", []) if s >= 0)
            if a.no_swap and done == 0 and hasattr(pool, "begin_text"):
                # FROZEN from here on: no forward starts afresh and nothing
                # more is admitted, so every later forward routes among the
                # experts this first window admitted
                pool.begin_text = lambda *a, **k: None
                pool.begin_forward = lambda *a, **k: None
                pool.admitting = lambda: False
            done += 1
        d = ev.run(a.eval_chunks)
        d.pop("stderr", None)
        dt, dp = drift()
        row = {"steps": done, "chars": done * a.chunk,
               "experts_touched": len(touched), "experts_total": n_exp,
               "trunk_drift": dt, "router_drift": dp,
               **{k: float(v) for k, v in d.items()}}
        rows.append(row)
        held = [v for k, v in d.items() if k not in lanes] or [float("nan")]
        print(f"  {done:>4} steps ({done*a.chunk/1000:>5.0f}K chars)  "
              f"read {np.mean([d[x] for x in lanes if x in d]):.4f}   "
              f"others mean {np.mean(held):.4f}   "
              f"experts touched {len(touched)}/{n_exp}   "
              f"drift trunk {dt*100:.2f}% router {dp*100:.2f}%   "
              f"[{time.time()-t0:.0f}s]")

    # ---------------------------------------------- phase 2: does it come back?
    rec_rows = []
    if a.recover:
        rl = [x.strip() for x in a.recover.split(",") if x.strip()]
        rmarks = sorted({int(x) for x in a.recover_at.split(",") if x.strip()})
        a.recover_steps = max(a.recover_steps, max(rmarks))
        rdata = {d: build_stream(
            d, (a.recover_steps // max(1, len(rl)) + a.visit + 4) * a.chunk)
            for d in rl}
        print(f"\n  RECOVERY: rotating {' -> '.join(rl)}, "
              f"{a.visit} steps each, {a.recover_steps} steps\n")
        rdone, ri = 0, 0
        _rr = {d: FileReader(model, rdata[d], d, a.chunk, ctx, device)
               for d in rl}
        rreader = _rr[rl[0]]
        for mk in rmarks:
            while rdone < mk:
                nl = (rdone // a.visit) % len(rl)
                if nl != ri or rreader.done():
                    ri = nl
                    if _rr[rl[ri]].done():
                        _rr[rl[ri]] = FileReader(model, rdata[rl[ri]], rl[ri],
                                                 a.chunk, ctx, device)
                    rreader = _rr[rl[ri]]
                nxt = rreader.peek()
                opt.zero_grad(set_to_none=True)
                loss = rreader.step(learn=True, aux_weight=cfg.pool_aux)
                if loss is None:
                    _rr[rl[ri]] = rreader = FileReader(
                        model, rdata[rl[ri]], rl[ri], a.chunk, ctx, device)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                opt.step()
                rdone += 1
            d = ev.run(a.eval_chunks)
            d.pop("stderr", None)
            rec_rows.append({"steps": rdone, "chars": rdone * a.chunk,
                             **{k: float(v) for k, v in d.items()}})
            dmg = [k for k in doms if k not in lanes]
            print(f"  +{rdone:>4} recovery steps ({rdone*a.chunk/1000:>5.0f}K) "
                  f"  damaged domains mean "
                  f"{np.mean([d[k] for k in dmg]):.4f}   [{time.time()-t0:.0f}s]")

    base = rows[0]
    out = {"tag": a.tag or a.domain, "domain": a.domain,
           "recovery": rec_rows, "recover_lanes": a.recover,
           "lanes": lanes, "visit": a.visit, "rotating": len(lanes) > 1,
           "weights": a.weights, "step": man.get("step"),
           "experts_total": n_exp, "eval_chunks": a.eval_chunks,
           "chunk": a.chunk, "context": ctx, "lr": a.lr,
           "trunk_lr_mult": a.trunk_lr_mult, "optim": "adamw",
           "warmup": a.warmup, "cold_optim": bool(a.cold_optim),
           "no_swap": bool(a.no_swap),
           "domains": doms, "baseline": {k: base[k] for k in doms},
           "rows": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  -> {a.out}")

    last = rows[-1]
    worst = max(((k, last[k] - base[k]) for k in doms),
                key=lambda kv: kv[1])
    for x in lanes:
        print(f"  read {x}: {base[x]:.4f} -> {last[x]:.4f} "
              f"({last[x]-base[x]:+.4f})")
    oth = [k for k in doms if k not in lanes]
    if oth:
        print(f"  the {len(oth)} NOT read: mean "
              f"{np.mean([last[k]-base[k] for k in oth]):+.4f}")
    print("  every domain, change from baseline:")
    for k in doms:
        print(f"    {k:<14} {base[k]:.4f} -> {last[k]:.4f}  {last[k]-base[k]:+.4f}"
              + ("   (read)" if k in lanes else ""))
    print(f"  worst other domain: {worst[0]} {worst[1]:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
