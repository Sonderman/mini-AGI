#!/usr/bin/env python3
"""
Exact-match arithmetic evaluation, broken out by task and digit count.

Unlike `func` in bench.py, this is a metric that can actually reach high values
at this model size - arithmetic is limited by data format and volume, not by
capacity. Accuracy by digit count is the interesting axis: models typically
hold up to some width and then fall off a cliff, and the cliff location is the
number worth tracking.

    python3 -m benchmarks arithmetic --ckpt weights --data data_math_char
"""

import os
import re
import sys
import json
import math
import random
import argparse

import numpy as np
import torch

import corpora.arithmetic as math_data
from minagi.tokenizer import load_tokenizer


@torch.no_grad()
def batch_complete(model, tok, prompts, max_new=24, stop="\n", temperature=0.0):
    """
    Greedy completion, batched by prompt length.

    Grouping equal-length prompts lets whole batches share one KV cache pass;
    left-padding would corrupt the positions, so grouping is the honest way to
    get the speedup.
    """
    device = next(model.parameters()).device
    stop_id = tok.encode(stop).ids[0] if stop else None
    out = [None] * len(prompts)
    encoded = [(i, tok.encode(p).ids) for i, p in enumerate(prompts)]
    groups = {}
    for i, ids in encoded:
        groups.setdefault(len(ids), []).append((i, ids))

    model.eval()
    for L, items in groups.items():
        for s in range(0, len(items), 128):
            chunk = items[s:s + 128]
            idx = torch.tensor([ids for _, ids in chunk], device=device)
            # A recurrent model has one cache slot per block-APPLICATION, not
            # per block: 2 prelude + 4 passes x (3 recur + 1 coda) = 18, while
            # cfg.n_layer is 6. Allocating n_layer slots indexes off the end.
            caches = (model.empty_caches() if hasattr(model, "empty_caches")
                      else [{"k": None, "v": None}
                            for _ in range(model.cfg.n_layer)])
            cur, offset = idx, 0
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            collected = [[] for _ in chunk]
            for _ in range(max_new):
                logits, _ = model(cur, caches=caches, pos_offset=offset)
                offset += cur.shape[1]
                nxt = logits[:, -1, :].argmax(-1, keepdim=True)
                for r in range(len(chunk)):
                    if not done[r]:
                        t = int(nxt[r, 0])
                        if stop_id is not None and t == stop_id:
                            done[r] = True
                        else:
                            collected[r].append(t)
                if bool(done.all()):
                    break
                cur = nxt
            for (i, _), toks in zip(chunk, collected):
                out[i] = tok.decode(toks)
    return out


def make_problems(n, tasks, seed=0, char=True, max_digits=8):
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        task = rng.choice(tasks)
        fn, cap, _ = math_data.TASKS[task]
        d = rng.randint(1, min(cap, max_digits))
        line = fn(rng, d)
        prompt, _, answer = line.rpartition("=")
        items.append({"task": task, "digits": d,
                      "prompt": prompt + "=", "truth": answer.strip()})
    return items


def evaluate(model, tok, n=400, tasks=None, seed=0, char=True, max_digits=8,
             max_new=110):
    tasks = tasks or list(math_data.TASKS)
    items = make_problems(n, tasks, seed, char, max_digits)
    preds = batch_complete(model, tok, [it["prompt"] for it in items], max_new)
    by_task, by_digits = {}, {}
    correct = 0
    for it, pred in zip(items, preds):
        # The model may write a <think> scratchpad before answering. The answer
        # is what follows it; comparing the whole string scored correct
        # reasoning as failure and made mod/sum look far worse than they are.
        if "</think>" in pred:
            pred = pred.split("</think>", 1)[1]
        elif "<think>" in pred:
            pred = ""          # ran out of budget mid-scratchpad
        ok = pred.strip() == it["truth"]
        correct += ok
        t = by_task.setdefault(it["task"], [0, 0])
        t[0] += ok
        t[1] += 1
        d = by_digits.setdefault(it["digits"], [0, 0])
        d[0] += ok
        d[1] += 1
        it["pred"] = pred.strip()
        it["ok"] = bool(ok)
    return {
        "overall": correct / max(len(items), 1),
        "n": len(items),
        "by_task": {k: round(v[0] / v[1], 4) for k, v in sorted(by_task.items())},
        "by_digits": {str(k): round(v[0] / v[1], 4)
                      for k, v in sorted(by_digits.items())},
        "samples": items[:12],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/val/arithmetic")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-digits", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=110,
                    help="must fit a <think> block AND the answer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from benchmarks.code import load_any
    device = torch.device(args.device)
    model, ckpt = load_any(args.ckpt, device)
    from minagi.tokenizer import ByteTokenizer
    tok = ByteTokenizer()      # the model is byte level; no corpus needed
    res = evaluate(model, tok, args.n, char=(meta.get("tokenizer") == "byte"),
                   max_digits=args.max_digits, max_new=args.max_new)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"overall exact match: {res['overall']:.3f}  (n={res['n']})")
    print("\nby task:")
    for k, v in res["by_task"].items():
        print(f"  {k:<8} {v:.3f}")
    print("\nby digit count:")
    for k, v in res["by_digits"].items():
        print(f"  {k:>2} digits  {v:.3f}")
    print("\nsamples:")
    for s in res["samples"][:8]:
        mark = "OK " if s["ok"] else "BAD"
        print(f"  {mark} {s['prompt']} {s['pred']!r} (truth {s['truth']!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
