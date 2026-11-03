#!/usr/bin/env python3
"""
How a newborn expert should be built.

An expert only earns a gate by contributing something the pool does not
already produce. So the question for any birth scheme is not "is this a valid
expert" but "is its OUTPUT novel, on the states the model actually routes on,
while still being worth listening to".

Measured here, on real hidden states captured from the running model:

  redundancy  cosine between the candidate's output and its parent's
  novelty     1 - the largest cosine against ANY existing expert
  scale       output magnitude relative to the parent's

Schemes compared: the clone-plus-noise the pool uses now, that noise made
relative to the parent's own scale, crossover by hidden unit, and a fresh
random expert.

    python3 tools/birth_probe.py --states 192
"""
import argparse
import json
import os
import sys

import math

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T                                     # noqa: E402
from minagi.ingest import as_stream, collect          # noqa: E402
from minagi.pool import PooledMLP                     # noqa: E402
from minagi.precision import set_compute_dtype        # noqa: E402


def cosf(x, y):
    return float(F.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)))


def swiglu(h, w1, w3, w2):
    return F.silu(h @ w1.T) * (h @ w3.T) @ w2.T


def capture_states(weights, chars, subject, n_keep):
    """The states the pool is actually asked to transform."""
    dev = torch.device("cpu")
    set_compute_dtype("fp32")
    model, cfg, pool, man = T.build_paged(weights, dev, read_only=True)
    T.weights_store.load(model, weights, device=dev)
    model.eval()
    got = []
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    h = sites[0].register_forward_pre_hook(
        lambda m, i: got.append(i[0].detach().reshape(-1, i[0].shape[-1])[:64].clone()))
    files = [f for f in collect(["data/train"]) if f"/{subject}/" in f]
    ids = torch.tensor(np.asarray(as_stream(sorted(files)[0])[:chars],
                                  np.int64)[None], device=dev)
    with torch.no_grad():
        model(ids)
    h.remove()
    H = torch.cat(got)[:n_keep].float()
    return H, man


def load_expert(d, i):
    z = np.load(os.path.join(d, "e%05d.npz" % i))
    return (torch.tensor(z["w1"]).float(), torch.tensor(z["w3"]).float(),
            torch.tensor(z["w2"]).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--states", type=int, default=192)
    ap.add_argument("--chars", type=int, default=96)
    ap.add_argument("--subject", default="stories")
    ap.add_argument("--pool-sample", type=int, default=48,
                    help="existing experts to measure novelty against")
    a = ap.parse_args()

    H, man = capture_states(a.weights, a.chars, a.subject, a.states)
    n = int(man["n_experts"])
    born = np.array(man["telemetry"]["born"], float)
    use = np.array(man["telemetry"]["use"], float)
    d = os.path.join(a.weights, "experts")
    parent_i = int(np.argmax(use))          # exactly what add_experts seeds from
    print(f"  {H.shape[0]} real states, {n} experts, "
          f"parent = the busiest, expert {parent_i}", flush=True)

    P = load_expert(d, parent_i)
    with torch.no_grad():
        y_par = swiglu(H, *P)
    par_scale = float(y_par.norm())

    # a sample of the pool, to measure novelty against
    rng = np.random.default_rng(0)
    others = sorted(rng.choice([i for i in range(n) if i != parent_i],
                               min(a.pool_sample, n - 1), replace=False))
    with torch.no_grad():
        Y = [swiglu(H, *load_expert(d, i)) for i in others]
    # the benchmark: how different are two EXISTING experts from each other?
    pair = [cosf(Y[i], Y[j]) for i in range(len(Y)) for j in range(i + 1, len(Y))]
    print(f"  novelty measured against {len(Y)} existing experts")
    print(f"  BENCHMARK - two existing experts, output cosine: "
          f"median {np.median(pair):+.3f}, p90 {np.percentile(pair, 90):+.3f}")
    print(f"  a newborn wants to look like a NEW member of that population\n",
          flush=True)

    def cos(x, y):
        return cosf(x, y)

    def report(name, cand):
        with torch.no_grad():
            y = swiglu(H, *cand)
        red = cos(y, y_par)
        best = max(cos(y, o) for o in Y + [y_par])
        wc = cos(cand[0], P[0])
        print(f"  {name:<34} {wc:+7.3f} {red:+11.3f} {1-best:10.3f} "
              f"{float(y.norm())/par_scale:8.2f}")

    print(f"  {'scheme':<34} {'w-cos':>7} {'redundancy':>11} "
          f"{'novelty':>10} {'scale':>8}")
    g = torch.Generator().manual_seed(0)

    def noisy(eps_abs=None, eps_rel=None):
        out = []
        for t in P:
            e = eps_abs if eps_abs is not None else eps_rel * float(t.std())
            out.append(t + e * torch.randn(t.shape, generator=g))
        return tuple(out)

    report("clone + 0.02 absolute  (CURRENT)", noisy(eps_abs=0.02))
    for r in (0.5, 1.0, 2.0, 4.0):
        report(f"clone + {r:g}x its own std", noisy(eps_rel=r))

    # crossover: a hidden unit is (w1[i], w3[i], w2[:,i]) and units are
    # interchangeable, so taking whole units from either parent keeps every
    # unit's learned function intact while building a new mixture
    mate_i = int(np.argsort(-use)[1])
    M = load_expert(d, mate_i)
    for frac in (0.5, 0.25):
        mask = (torch.rand(P[0].shape[0], generator=g) < frac)
        w1 = torch.where(mask[:, None], M[0], P[0])
        w3 = torch.where(mask[:, None], M[1], P[1])
        w2 = torch.where(mask[None, :], M[2], P[2])
        report(f"crossover, {frac:.0%} of units from expert {mate_i}",
               (w1, w3, w2))

    # recombination across the whole population: every hidden unit is taken
    # from some expert, so every unit is a trained feature and the mixture is
    # one nothing in the pool computes
    for k in (4, 16):
        pick = rng.choice(others, k, replace=False)
        srcs = [load_expert(d, int(i)) for i in pick]
        who = torch.randint(0, k, (P[0].shape[0],), generator=g)
        w1 = torch.stack([srcs[who[u]][0][u] for u in range(P[0].shape[0])])
        w3 = torch.stack([srcs[who[u]][1][u] for u in range(P[0].shape[0])])
        w2 = torch.stack([srcs[who[u]][2][:, u] for u in range(P[0].shape[0])], 1)
        report(f"units drawn from {k} random experts", (w1, w3, w2))

    def rescaled(cand):
        with torch.no_grad():
            y = swiglu(H, *cand)
        f = par_scale / max(float(y.norm()), 1e-9)
        # the output is bilinear in (w1|w3) and w2, so sqrt on the way in
        return (cand[0] * math.sqrt(f), cand[1], cand[2] * math.sqrt(f))

    fresh = (torch.randn(P[0].shape, generator=g) * 0.02,
             torch.randn(P[1].shape, generator=g) * 0.02,
             torch.randn(P[2].shape, generator=g) * 0.02)
    report("fresh random (0.02, as add_experts)", fresh)
    report("fresh random, rescaled to parent's output", rescaled(fresh))
    print(f"\n  parent output norm {par_scale:.2f}   "
          f"redundancy 1.0 = identical to the parent, 0 = unrelated")


if __name__ == "__main__":
    main()
