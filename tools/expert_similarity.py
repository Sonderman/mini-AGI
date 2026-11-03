#!/usr/bin/env python3
"""
How different are the experts from each other, and is the disk earning its keep?

Two experts can be identical functions and still look orthogonal in weight
space: a SwiGLU expert is unchanged if you permute its 2,048 hidden units
(the rows of w1 and w3 and the columns of w2 together). So this measures two
things that answer different questions.

  RAW      cosine between the weights as stored. Sensitive to that permutation,
           which makes it a LINEAGE test: experts are born from existing ones,
           and a child starts as a copy in the parent's ordering. High raw
           similarity means shared ancestry that never diverged.

  SUBSPACE cosine between w1.T @ w1, which is invariant to any reordering of
           the hidden units. It asks what part of the input space the expert
           responds to at all, regardless of how it numbers its own neurons.
           High subspace similarity with low raw similarity means two experts
           arrived at the same job independently.

  KEYS     cosine between the routing keys. What the router THINKS each expert
           is for, which is what actually decides who gets the token.

Read-only, CPU only, and it never imports torch - it is meant to run while
training is using the card.

    python3 tools/expert_similarity.py --weights weights --out runs
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def load_grams(files, coords, verbose=True):
    """One pass over the experts: a raw fingerprint and an input-space Gram."""
    n = len(files)
    raw = np.zeros((n, len(coords)), dtype=np.float32)
    gram = np.zeros((n, 512 * 512), dtype=np.float32)
    scale = np.zeros((n, 3), dtype=np.float32)
    for i, f in enumerate(files):
        with np.load(f) as d:
            w1, w3, w2 = d["w1"], d["w3"], d["w2"]
        flat = np.concatenate([w1.ravel(), w3.ravel(), w2.ravel()])
        raw[i] = flat[coords]
        scale[i] = (np.linalg.norm(w1), np.linalg.norm(w3), np.linalg.norm(w2))
        g = (w1.T @ w1) + (w3.T @ w3)          # [512, 512], permutation-free
        gram[i] = g.ravel()
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n} experts read", flush=True)
    return raw, gram, scale


def cosines(X, center=False):
    """Pairwise cosine similarity of the rows, with the diagonal removed.

    `center` subtracts the mean row first, and for the Gram features it is not
    optional. w1.T @ w1 is positive semi-definite, so every expert's Gram sits
    in the same orthant and they share a large common component - the average
    expert. Uncentred, that alone puts the median pair at +0.78 and makes the
    pool look like two experts in a trenchcoat. What the question is actually
    about is how each expert differs FROM the average one, which is what is
    left after the common part is removed.
    """
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, np.nan)
    return S


def effective_count(S):
    """How many genuinely independent experts the pool behaves like.

    The participation ratio of the eigenvalues of the similarity matrix:
    n if every expert is orthogonal to every other, 1 if they are all the
    same expert. Reported against the n that is actually on disk.
    """
    A = np.nan_to_num(S, nan=1.0)
    A = (A + A.T) / 2
    ev = np.linalg.eigvalsh(A)
    ev = np.clip(ev, 0, None)
    if ev.sum() <= 0:
        return float("nan")
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--coords", type=int, default=200_000,
                    help="weights sampled per expert for the raw fingerprint")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.weights, "experts", "*.npz")))
    if not files:
        print(f"  no experts under {a.weights}/experts")
        return 1
    man = json.load(open(os.path.join(a.weights, "manifest.json")))
    tel = man.get("telemetry", {})
    n = len(files)
    print(f"  {n} experts, {man['step']:,} steps, "
          f"{man.get('read_chars', 0) / 1e6:.0f}M characters read")

    rng = np.random.default_rng(0)
    total = 3 * 512 * 2048
    coords = np.sort(rng.choice(total, size=min(a.coords, total), replace=False))
    print(f"  fingerprinting on {len(coords):,} of {total:,} weights each")
    raw, gram, scale = load_grams(files, coords)

    S_raw = cosines(raw, center=True)
    S_sub = cosines(gram, center=True)
    S_sub_uncentred = cosines(gram)
    # a compact embedding so plotting never re-reads 5 GB of experts
    G = gram - gram.mean(axis=0, keepdims=True)
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    U, sv, _ = np.linalg.svd(G @ G.T)
    emb = U[:, :64] * np.sqrt(np.clip(sv[:64], 0, None))
    keys = np.load(os.path.join(a.weights, "routers.npz"))
    kname = [k for k in keys.files if k.endswith('mlp.router.weight')]
    S_key = cosines(keys[kname[0]].astype(np.float32), center=True) if kname else None

    np.savez_compressed(os.path.join(a.out, "expert_similarity.npz"),
                        raw=S_raw, subspace=S_sub,
                        keys=S_key if S_key is not None else np.zeros(1),
                        subspace_uncentred=S_sub_uncentred, emb=emb, spectrum=sv,
                        gate=np.asarray(tel.get("gate", []), dtype=np.float32),
                        use=np.asarray(tel.get("use", []), dtype=np.float64),
                        born=np.asarray(tel.get("born", []), dtype=np.float64),
                        scale=scale)
    print(f"  wrote {a.out}/expert_similarity.npz")

    for name, S in (("raw (lineage)", S_raw),
                    ("subspace (function), centred", S_sub),
                    ("subspace, UNCENTRED for comparison", S_sub_uncentred),
                    ("routing keys", S_key)):
        if S is None:
            continue
        off = S[~np.isnan(S)]
        nn = np.nanmax(S, axis=1)
        print(f"\n  {name}")
        print(f"    pairwise: median {np.median(off):+.3f}  "
              f"p99 {np.percentile(off, 99):+.3f}  max {off.max():+.3f}")
        print(f"    nearest neighbour: median {np.median(nn):+.3f}  "
              f"max {nn.max():+.3f}")
        for thr in (0.9, 0.8, 0.6, 0.4):
            pairs = int(((S > thr) & ~np.isnan(S)).sum() // 2)
            have = int((nn > thr).sum())
            print(f"    > {thr:.1f}: {pairs:>6,} pairs, {have:>3} experts "
                  f"({have / n:.0%}) have a twin this close")
        eff = effective_count(S)
        print(f"    effective independent experts: {eff:.0f} of {n} "
              f"({eff / n:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
