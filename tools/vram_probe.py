#!/usr/bin/env python3
"""
Where the VRAM actually goes, at a given window.

Reading the code accounts for about 2.9 GB of the 6.36 GB an 8,192-character
window allocates. This measures the rest instead of arguing about it: load,
forward, backward, each phase reported against what the arithmetic predicts.

    python3 tools/vram_probe.py --weights weights --context 4096
    python3 tools/vram_probe.py --weights weights --context 8192 --sweep

Needs the card to itself. It will not run while training is using it.
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

import train as T                                           # noqa: E402
from minagi.precision import amp, set_compute_dtype        # noqa: E402


def gb(x):
    return x / 1e9


def probe(model, cfg, ctx, chunk, aux=0.0):
    """One learning step at `ctx`, reporting memory at each phase."""
    dev = next(model.parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    data = np.random.randint(0, 260, ctx + 1).astype(np.int64)
    x = torch.from_numpy(data[:-1]).to(dev).unsqueeze(0)
    y = torch.from_numpy(data[1:]).to(dev).unsqueeze(0)

    model.train()
    with amp(dev):
        _, loss = model(x, y, caches=None, pos_offset=0)
        if aux and getattr(model, "pool", None) is not None:
            loss = loss + aux * model.pool_aux()
    fwd = torch.cuda.memory_allocated()
    fwd_peak = torch.cuda.max_memory_allocated()

    loss.backward()
    bwd_peak = torch.cuda.max_memory_allocated()
    # The grads stay. A training step holds an fp32 gradient for every
    # parameter it touches - 109 M of them here, 0.44 GB - and zeroing before
    # reading the peak reported a step nobody runs. Leaving them out is what
    # made this probe recommend a 5,120 window that died at 5,116.
    grads = sum(p.grad.numel() * p.grad.element_size()
                for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return {"grads": gb(grads),
            "weights_and_moments": gb(base),
            "after_forward": gb(fwd - base),
            "forward_peak": gb(fwd_peak),
            "backward_peak": gb(bwd_peak),
            "reserved": gb(torch.cuda.memory_reserved())}


def predicted(ctx, depth, n_exp, d=512, V=265):
    """What reading the code says it should be, per phase."""
    per_pass = (1024 + 512 * 6 + 1536 + n_exp) * 2 + V * 2 + V * 4
    return {"activations": gb(ctx * per_pass * depth)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--precision", default=None,
                    help="default: whatever the checkpoint's config says")
    ap.add_argument("--sweep", action="store_true",
                    help="try doubling windows until one fails")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("  no CUDA; this measures VRAM and needs a card")
        return 1
    dev = torch.device("cuda")
    model, cfg, pool, man = T.build_paged(a.weights, dev, read_only=True)
    # build_paged does not set this; train.py does it from config at startup.
    # Without it the probe measures an fp32 forward the run never performs.
    set_compute_dtype(a.precision or getattr(cfg, "precision", "bf16"))
    depth = getattr(cfg, "train_steps_mean", None) or cfg.max_steps
    n_exp = pool.n_experts()
    print(f"  {a.weights}: {n_exp} experts, depth {depth} of {cfg.max_steps}, "
          f"bptt_window {cfg.bptt_window}")
    if cfg.bptt_window >= depth:
        print(f"  NOTE bptt_window {cfg.bptt_window} >= sampled depth {depth}: "
              f"the depth truncation never fires, so every pass keeps its "
              f"activations")
    from minagi.precision import compute_dtype
    print(f"  compute dtype: {compute_dtype()}")
    print(f"  card: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB\n")

    # read_only builds no optimiser, so its state is not on the card. The run
    # carries it at every step, so stand it up as ballast or the ceiling this
    # prints is one the run cannot reach.
    trunk = sum(p.numel() for n, p in model.named_parameters()
                if "expert" not in n and "pool" not in n)
    pooled = sum(p.numel() for p in model.parameters()) - trunk
    moments = trunk * 2 * 4 + pooled * 2 * 2
    ballast = torch.empty(int(moments), dtype=torch.uint8, device=dev)
    print(f"  ballast for Adam's moments: {moments/1e9:.2f} GB "
          f"(trunk fp32, pool bf16)\n")

    windows = ([a.context] if not a.sweep else
               [a.context * (2 ** i) for i in range(5)])
    print(f"  {'window':>8}{'weights':>10}{'fwd':>9}{'fwd peak':>10}"
          f"{'bwd peak':>10}{'grads':>9}{'predicted act':>15}")
    for ctx in windows:
        try:
            r = probe(model, cfg, ctx, a.chunk, aux=getattr(cfg, "pool_aux", 0.0))
            p = predicted(ctx, depth, n_exp)
            print(f"  {ctx:>8}{r['weights_and_moments']:>9.2f}G"
                  f"{r['after_forward']:>8.2f}G{r['forward_peak']:>9.2f}G"
                  f"{r['backward_peak']:>9.2f}G{r['grads']:>8.2f}G"
                  f"{p['activations']:>14.2f}G")
        except torch.OutOfMemoryError:
            print(f"  {ctx:>8}   OOM - this is the ceiling")
            torch.cuda.empty_cache()
            break
    print("\n  'fwd' is what the forward RETAINS for the backward. If that is"
          "\n  far above 'predicted act', something is holding activations the"
          "\n  arithmetic does not know about.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
