#!/usr/bin/env python3
"""
What this machine does with mini-AGI's arithmetic, measured on the spot.

Written for the AMD question - an RX 6800 XT reading at a fifth of an RTX
3070's speed, and out of memory minutes in - but it runs on any card, and on
the CPU. It needs the card to itself: it refuses to start while something else
holds most of it, so it cannot disturb a run.

    python3 tools/device_check.py               # everything below
    python3 tools/device_check.py --profile     # plus where a reading step's time goes
    python3 tools/device_check.py --skip-step   # attention and matmuls only

  device       the build, the card and its architecture, and the switches
               that decide which kernels exist
  attention    for bf16 and fp32: whether a fused kernel computes each case
               this model needs - causal over the reading window, and a chunk
               after a cache - and what the fused, blocked and math paths cost
               in time and memory at that window
  matmuls      the model's own shapes - the expert layer's batched multiplies
               and the trunk's - in fp32, bf16 and fp16, in TFLOP/s
  reading      one learning step of a model of the real size (random weights,
               in a temporary directory, deleted after): its forward over the
               whole window and its backward, in bf16 and in fp32, as
               characters per second and peak memory

Nothing here touches weights/. Reading speed is the step's chunk of new
characters divided by the step's time, as train.py counts it; the optimiser
step is left out (it is the same in every precision).
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402

from minagi import model as M                               # noqa: E402  (sets the AOTriton default)
from minagi.precision import amp, set_compute_dtype        # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev, reps=5, warm=2):
    """Median seconds of fn() over `reps`, after `warm` untimed calls."""
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def peak_of(fn, dev):
    """Peak memory fn() allocates beyond what was already there, in GB."""
    if dev.type != "cuda":
        return None
    sync(dev)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    fn()
    sync(dev)
    return (torch.cuda.max_memory_allocated(dev) - base) / 1e9


def fmt_gb(x):
    return "   -   " if x is None else f"{x:6.2f} GB"


# ---- device ------------------------------------------------------------------

def describe(dev):
    print("DEVICE")
    print(f"  torch {torch.__version__}"
          + (f", HIP {torch.version.hip}" if torch.version.hip else "")
          + (f", CUDA {torch.version.cuda}" if torch.version.cuda else ""))
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        arch = getattr(p, "gcnArchName", "") or f"sm_{p.major}{p.minor}"
        free, total = torch.cuda.mem_get_info(dev)
        print(f"  {p.name}, {arch}, {total / 2**30:.1f} GiB ({free / 2**30:.1f} free)")
    else:
        from minagi.precision import cpu_bf16_native
        print(f"  CPU, {torch.get_num_threads()} threads, "
              f"bf16 in hardware: {'yes' if cpu_bf16_native() else 'no'}")
    for k in ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MINAGI_ATTENTION",
              "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        if os.environ.get(k) is not None:
            print(f"  {k}={os.environ[k]}")
    print()


# ---- attention -----------------------------------------------------------------

def backends(dev, dt, T, S):
    """Which of PyTorch's fused backends run this shape at all, alone."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right
    q = torch.randn(1, 8, T, 64, device=dev, dtype=dt)
    kv = torch.randn(1, 8, S, 64, device=dev, dtype=dt)
    got = []
    for name, b in (("flash", SDPBackend.FLASH_ATTENTION),
                    ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with sdpa_kernel([b]):
                if S == T:
                    F.scaled_dot_product_attention(q, kv, kv, is_causal=True)
                else:
                    F.scaled_dot_product_attention(q, kv, kv,
                                                   attn_mask=causal_lower_right(T, S))
            got.append(name)
        except Exception:                                  # noqa: BLE001
            pass
    return ", ".join(got) or "none"


def attention(dev, window, dtypes):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    print(f"ATTENTION  (8 heads of 64; reading: {window} queries over {window} keys, "
          f"causal; a cached chunk: {window // 2} over {window})")
    print(f"  {'':6} {'case':14} {'kernels':18} {'fused?':7} "
          f"{'path':8} {'fwd+bwd':>9} {'memory':>10}")
    for name in dtypes:
        dt = DTYPES[name]
        for case, T in (("reading", window), ("cached chunk", window // 2)):
            S = window
            masked = T != S
            fused = M.fused_attention_available(dev, dt, masked=masked)
            kern = backends(dev, dt, T, S)
            q = torch.randn(1, 8, T, 64, device=dev, dtype=dt, requires_grad=True)
            k = torch.randn(1, 8, S, 64, device=dev, dtype=dt, requires_grad=True)
            v = torch.randn(1, 8, S, 64, device=dev, dtype=dt, requires_grad=True)
            g = torch.randn(1, 8, T, 64, device=dev, dtype=dt)

            def run(path):
                if path == "fused":
                    if masked:
                        from torch.nn.attention.bias import causal_lower_right
                        y = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=causal_lower_right(T, S))
                    else:
                        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                elif path == "blocked":
                    y = M.blocked_attention(q, k, v, S - T)
                else:
                    qi = torch.arange(S - T, S, device=dev).unsqueeze(1)
                    ki = torch.arange(S, device=dev).unsqueeze(0)
                    with sdpa_kernel([SDPBackend.MATH]):
                        y = F.scaled_dot_product_attention(q, k, v, attn_mask=ki <= qi)
                y.backward(g)
                q.grad = k.grad = v.grad = None

            first = True
            for path in (["fused"] if fused else []) + ["blocked", "math"]:
                try:
                    t = timed(lambda: run(path), dev, reps=3, warm=1)
                    mem = peak_of(lambda: run(path), dev)
                    res = f"{t * 1e3:7.1f}ms {fmt_gb(mem):>10}"
                except torch.cuda.OutOfMemoryError:
                    res = f"{'out of memory':>20}"
                    torch.cuda.empty_cache()
                except Exception as e:                      # noqa: BLE001
                    res = f"  failed: {type(e).__name__}"
                lead = (f"  {name:6} {case:14} {kern:18} {'yes' if fused else 'no':7}"
                        if first else f"  {'':6} {'':14} {'':18} {'':7}")
                print(f"{lead} {path:8} {res}", flush=True)
                first = False
    print("  the model uses the fused path where 'fused?' says yes, the blocked one where it "
          "says no;\n  math is what PyTorch falls back to on its own, shown for comparison")
    print()


# ---- matmuls ---------------------------------------------------------------------

def matmuls(dev, window, dtypes):
    per = max(1, window * 8 // 32)        # characters per expert: top-8 of 32 slots
    shapes = [("expert up", (32, per, 512), (32, 512, 2048)),
              ("expert down", (32, per, 2048), (32, 2048, 512)),
              ("qkv", (window, 512), (512, 1536)),
              ("adapter", (window, 1024), (1024, 512))]
    print(f"MATMULS  (the model's shapes at a {window}-character window), TFLOP/s")
    print(f"  {'':12} {'':32} " + " ".join(f"{n:>7}" for n in dtypes))
    for label, a, b in shapes:
        row = []
        for name in dtypes:
            dt = DTYPES[name]
            try:
                A = torch.randn(*a, device=dev, dtype=dt)
                B = torch.randn(*b, device=dev, dtype=dt)
                fn = (lambda: torch.bmm(A, B)) if len(a) == 3 else (lambda: A @ B)
                t = timed(fn, dev, reps=7, warm=3)
                flops = 2 * a[-2] * a[-1] * b[-1] * (a[0] if len(a) == 3 else 1)
                row.append(f"{flops / t / 1e12:7.2f}")
            except Exception:                              # noqa: BLE001
                row.append(f"{'-':>7}")
        dims = (f"{a[0]} x " if len(a) == 3 else "") + \
            " x ".join(str(d) for d in a[-2:]) + " @ " + " x ".join(str(d) for d in b[-2:])
        print(f"  {label:12} {dims:32} " + " ".join(row), flush=True)
    print()


# ---- one reading step -------------------------------------------------------------

def reading(dev, window, chunk, precisions, steps, profile):
    import train as T
    from minagi.config import load, get
    from minagi.create import create

    c = load()
    tmp = tempfile.mkdtemp(prefix="minagi-device-check-")
    print(f"READING  (one learning step over a {window}-character window, {chunk} of them "
          f"new; a real-size model with random weights)")
    try:
        wdir = os.path.join(tmp, "w")
        create(wdir, seed=0, verbose=False, experts=40, block=max(window, 512))
        torch.manual_seed(0)
        data = torch.randint(0, 256, (window + 1,))
        x = data[:-1].unsqueeze(0).to(dev)
        y = data[1:].unsqueeze(0).to(dev)
        for name in precisions:
            set_compute_dtype(name)
            model, cfg, pool, _ = T.build_paged(wdir, dev, ram_capacity=40)
            # the depth policy reading runs under, from config.yaml, as train.py sets it
            cfg.train_steps_mean = float(get(c, "model.train_steps_mean", 0.0) or 0.0)
            cfg.min_steps = max(1, min(int(get(c, "model.min_steps", 1) or 1), cfg.max_steps))
            if get(c, "model.bptt_window", None):
                cfg.bptt_window = int(get(c, "model.bptt_window"))
            if get(c, "model.halt_thresh", None):
                cfg.halt_thresh = float(get(c, "model.halt_thresh"))
            cfg.halt_freeze = bool(get(c, "model.halt_freeze", False))
            model.train()

            def step(i):
                torch.manual_seed(1000 + i)          # the same depths in every precision
                model.zero_grad(set_to_none=True)
                with amp(dev):
                    _, loss = model(x, y)
                loss.backward()
                return model.last_steps

            try:
                step(0)                              # loads the experts, warms the kernels
                sync(dev)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                t0, depth = time.perf_counter(), []
                for i in range(steps):
                    depth.append(step(i + 1))
                sync(dev)
                dt_ = (time.perf_counter() - t0) / steps
                mem = (torch.cuda.max_memory_allocated(dev) / 1e9
                       if dev.type == "cuda" else None)
                from minagi.precision import autocast_on
                print(f"  {name:5} {chunk / dt_:8.1f} char/s   {dt_:6.2f} s a step   "
                      f"peak {fmt_gb(mem)}   mean depth {sum(depth) / len(depth):.1f} rows"
                      + ("" if autocast_on(dev) or name == "fp32"
                         else "   (computed in fp32: no bf16 arithmetic here)"),
                      flush=True)
                if profile:
                    from torch.profiler import ProfilerActivity, profile as prof_
                    acts = [ProfilerActivity.CPU] + (
                        [ProfilerActivity.CUDA] if dev.type == "cuda" else [])
                    with prof_(activities=acts) as pr:
                        step(steps + 1)
                        sync(dev)
                    key = "self_cuda_time_total" if dev.type == "cuda" else "self_cpu_time_total"
                    try:
                        table = pr.key_averages().table(sort_by=key, row_limit=25)
                    except Exception:                       # noqa: BLE001
                        table = pr.key_averages().table(
                            sort_by=key.replace("cuda", "device"), row_limit=25)
                    print(f"\n  where one {name} step's time goes, by op:\n")
                    print("\n".join("    " + ln for ln in table.splitlines()))
                    print()
            except torch.cuda.OutOfMemoryError:
                print(f"  {name:5} out of memory")
            del model, pool
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window", type=int, default=4096,
                    help="characters a reading step forwards (model.context_end)")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="of which new, per step (training.chunk)")
    ap.add_argument("--steps", type=int, default=3, help="timed reading steps per precision")
    ap.add_argument("--precisions", default="bf16,fp32")
    ap.add_argument("--profile", action="store_true",
                    help="also break one reading step down by op")
    ap.add_argument("--skip-step", action="store_true", help="no reading step")
    ap.add_argument("--force", action="store_true",
                    help="run even though the card is busy (it will compete with what is there)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        if free < 0.6 * total and not a.force:
            raise SystemExit(f"the card has {free / 2**30:.1f} of {total / 2**30:.1f} GiB free - "
                             f"something else is using it. Run this on a free card "
                             f"(or --force).")
    describe(dev)
    names = ["bf16", "fp32", "fp16"] if dev.type == "cuda" else ["bf16", "fp32"]
    attention(dev, a.window, [n for n in names if n != "fp16"])
    matmuls(dev, a.window, names)
    if not a.skip_step:
        reading(dev, a.window, a.chunk, a.precisions.split(","), a.steps, a.profile)


if __name__ == "__main__":
    main()
