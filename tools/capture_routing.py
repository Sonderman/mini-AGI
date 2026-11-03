#!/usr/bin/env python3
"""
Record one real selection, in full, so it can be explained.

Not a simulation. This loads the trained model, reads real text from the
corpus, and writes down every quantity the selection rule actually computes:

  the request over the WHOLE pool at the first pass: every character ranks
  every expert with the router and asks for its top_k, and each request
  carries the probability the router gave it

  which experts the forward admitted - the most requested, up to the card's
  slots - and which of them had to be loaded, against what was on the card
  from the text before

  the slot table - which expert sits in which VRAM slot, which is the part
  that makes addressing confusing

  per character, as the text continues one character at a time, the
  gate-weighted weight each slot received, which expert each slot held and
  how many the character loaded - every character is a forward of its own
  and routes among what its whole text so far has voted for

    python3 tools/capture_routing.py --out runs/routing_capture.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T                                    # noqa: E402
from minagi.ingest import as_stream, collect         # noqa: E402
from minagi.precision import amp, set_compute_dtype  # noqa: E402
from minagi.pool import PooledMLP                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--subject", default="stories")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--warm", type=int, default=3,
                    help="texts read first, so the card holds what reading "
                         "left there and the loads counted are real ones")
    ap.add_argument("--chars", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/routing_capture.npz")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    set_compute_dtype("fp32" if dev.type == "cpu" else "bf16")
    # read_only: paging an expert in marks it dirty whether or not anything
    # changed it, and the training run may own this directory
    model, cfg, pool, man = T.build_paged(a.weights, dev, read_only=True)
    model.eval()
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    print(f"  {pool.n_experts()} experts, {pool.resident} slots, "
          f"{len(sites)} call site(s), top_k {sites[0].top_k}", flush=True)

    files = [f for f in collect(["data/train"]) if f"/{a.subject}/" in f]
    data = as_stream(sorted(files)[0])
    ids = torch.tensor(np.asarray(data[:a.chunk * (a.warm + 1) + a.chars + 1],
                                  np.int64)[None], device=dev)

    # Read a few texts first. The card starts empty, so the first text loads
    # every slot and shows nothing about how little a later text moves.
    for w in range(a.warm):
        with torch.no_grad(), amp(dev):
            model(ids[:, w * a.chunk:(w + 1) * a.chunk])
    base = a.warm * a.chunk
    before_slots = list(pool.slots)
    loads_before = pool.loads

    # --- the selection, with its parts pulled out ---------------------------
    text = ids[:, base:base + a.chunk]
    first = []
    h = sites[0].register_forward_pre_hook(
        lambda m, i: first.append(i[0].detach().reshape(-1, i[0].shape[-1])))
    for s_ in sites:
        s_.record_weights = True
    caches = model.empty_caches()
    with torch.no_grad(), amp(dev):
        model(text, caches=caches, pos_offset=0)           # a new text
    h.remove()
    admitted = sorted(pool._admitted)
    after_slots = list(pool.slots)
    loads = pool.loads - loads_before
    with torch.no_grad():
        s0 = sites[0]
        E = pool.router_rows()
        q = F.softmax(F.linear(first[0].float() + s0.depth_emb.float(),
                               s0.router.weight[:E].float()), -1)
        top = torch.topk(q, s0.top_k, dim=-1)
        request = torch.zeros(E, device=q.device)
        request.index_add_(0, top.indices.reshape(-1), top.values.reshape(-1))

    # --- then the text continues, one character at a time --------------------
    per_slot = np.zeros((a.chars, len(sites), pool.resident), np.float32)
    slots_t = np.zeros((a.chars, pool.resident), np.int64)
    loads_t = np.zeros(a.chars, np.int64)
    off = text.shape[1]
    for t in range(a.chars):
        cur = ids[:, base + a.chunk + t:base + a.chunk + t + 1]
        l0 = pool.loads
        with torch.no_grad(), amp(dev):
            model(cur, caches=caches, pos_offset=off)     # its own admission
        off += 1
        slots_t[t] = pool.slots
        loads_t[t] = pool.loads - l0
        for si, s in enumerate(sites):
            lw = getattr(s, "last_weight", None)
            if lw is not None:
                v = lw.detach().float().cpu().numpy()
                per_slot[t, si, :min(len(v), pool.resident)] = v[:pool.resident]

    np.savez(
        a.out,
        n_experts=pool._n, resident=pool.resident, top_k=sites[0].top_k,
        n_sites=len(sites), max_steps=cfg.max_steps,
        gate=pool.gate.detach().abs().float().cpu().numpy(),
        request=request.cpu().numpy(), admitted=np.array(admitted),
        before_slots=np.array(before_slots), after_slots=np.array(after_slots),
        loads=loads, per_slot=per_slot, slots_t=slots_t, loads_t=loads_t,
        text=np.frombuffer(
            bytes(int(v) for v in data[base + a.chunk:base + a.chunk + a.chars]
                  if v < 256), dtype=np.uint8),
        step=man["step"],
    )
    print(f"  captured: the first pass asked for {int((request > 0).sum())} "
          f"experts, the forward admitted {len(admitted)}, {loads} had to be "
          f"loaded", flush=True)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
