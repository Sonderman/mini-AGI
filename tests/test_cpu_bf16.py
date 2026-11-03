#!/usr/bin/env python3
"""
bfloat16 on the CPU, where the CPU multiplies it in hardware.

  autocast runs on the CPU only for bf16 and only where the CPU has it
  natively (AVX-512 BF16 or AMX); everywhere else the CPU path is exactly the
  fp32 path it always was

  where it runs, a forward agrees with the fp32 one to bf16's rounding, the
  expert dispatch computes in bf16 from a copy cast once, and a training step
  steps fp32 weights with fp32 gradients

  the GPU's rule: autocast whenever the dtype is not fp32 - except bf16 on a
  GPU with no bf16 arithmetic and the memory for fp32 (RDNA2 with 16 GB, an
  NVIDIA card before Ampere with 16 GB), which computes in fp32. Checked on
  described GPUs, so the test never touches a real one
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-cpubf16-")
saved = os.environ.get("MINAGI_CPU_BF16")
try:
    from minagi import precision as P                          # noqa: E402
    from minagi.create import create                          # noqa: E402
    from minagi.pool import PooledMLP                         # noqa: E402
    import train as T                                         # noqa: E402

    P.set_compute_dtype("bf16")
    native = P.cpu_bf16_native() if saved is None else None
    os.environ["MINAGI_CPU_BF16"] = "0"
    check("not native: no autocast on the CPU", not P.autocast_on("cpu"))
    check("...and the dispatch keeps the residual's dtype",
          P.dispatch_dtype(torch.device("cpu"), torch.float32) is torch.float32)
    from types import SimpleNamespace

    def gpu(arch="", major=8, minor=6, gib=8, amd=False):
        props = SimpleNamespace(gcnArchName=arch, major=major, minor=minor,
                                total_memory=gib * 2**30)
        return lambda device=None: (props, amd)

    real_info, saved_gpu = P._gpu_info, os.environ.pop("MINAGI_GPU_BF16", None)
    try:
        P._gpu_info = gpu(major=8, minor=6, gib=8)                    # RTX 3070
        check("bf16 where the GPU has it (an RTX 3070): autocast", P.autocast_on("cuda")
              and P.describe("cuda") == "bf16")
        P._gpu_info = gpu("gfx1030", gib=16, amd=True)                # RX 6800 XT
        check("RDNA2 with 16 GB: fp32, and the run says why", not P.autocast_on("cuda")
              and P.dispatch_dtype(torch.device("cuda"), torch.float32) is torch.float32,
              P.describe("cuda"))
        P._gpu_info = gpu("gfx1031", gib=12, amd=True)                # RX 6700 XT
        check("RDNA2 with 12 GB: bf16, which fits", P.autocast_on("cuda"), P.describe("cuda"))
        P._gpu_info = gpu("gfx1100", gib=24, amd=True)                # RX 7900 XTX
        check("RDNA3: bf16", P.autocast_on("cuda"))
        P._gpu_info = gpu("gfx942:sramecc+:xnack-", gib=192, amd=True)  # MI300X
        check("CDNA3: bf16", P.autocast_on("cuda"))
        P._gpu_info = gpu(major=7, minor=5, gib=16)                   # T4
        check("NVIDIA before Ampere with 16 GB: fp32", not P.autocast_on("cuda"))
        P._gpu_info = gpu("gfx1030", gib=16, amd=True)
        os.environ["MINAGI_GPU_BF16"] = "1"
        check("MINAGI_GPU_BF16=1 overrides it", P.autocast_on("cuda"))
        os.environ.pop("MINAGI_GPU_BF16")
        P.set_compute_dtype("fp32")
        check("fp32 configured: no autocast anywhere",
              not P.autocast_on("cuda") and not P.autocast_on("cpu"))
        P.set_compute_dtype("fp16")
        check("fp16 configured: the GPU rule leaves it alone", P.autocast_on("cuda"))
    finally:
        P._gpu_info = real_info
        if saved_gpu is not None:
            os.environ["MINAGI_GPU_BF16"] = saved_gpu
    os.environ["MINAGI_CPU_BF16"] = "1"
    check("fp16 configured: never on the CPU", not P.autocast_on("cpu"))
    P.set_compute_dtype("bf16")
    check("native: autocast on the CPU", P.autocast_on("cpu"))

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", ram_capacity=3)
    pool.balance = 0.0
    torch.manual_seed(0)
    for site in (m for m in model.modules() if isinstance(m, PooledMLP)):
        site.router.weight.data.normal_(0.0, 1.0)
    x = torch.tensor([list(b"def merge(a, b):\n    return sorted(a + b)\n\nclass Node:\n    pass\n")])

    @torch.no_grad()
    def forward():
        model.eval()
        with P.amp(x.device):
            logits, _ = model(x)
        return logits.float()

    os.environ["MINAGI_CPU_BF16"] = "0"
    ref = forward()
    os.environ["MINAGI_CPU_BF16"] = "1"
    got = forward()
    err = float((got - ref).abs().max())
    agree = float((got.argmax(-1) == ref.argmax(-1)).float().mean())
    check("bf16 on the CPU agrees with fp32 to bf16's rounding", err < 0.1 and agree > 0.9,
          f"max |dlogit| {err:.3f}, argmax agreement {agree:.0%}")
    check("the expert dispatch computed in bf16 from the kept copy",
          pool._infer is not None and pool._infer[0][0] is torch.bfloat16)

    @torch.no_grad()
    def cached(split=29):
        """The measuring path and the writing path: the text in two chunks,
        the second read after a cache (attention under the lower-right causal
        bias), then two characters one forward each."""
        model.eval()
        caches = model.empty_caches()
        xs, ys = x[:, :-1], x[:, 1:]
        losses, last = [], None
        with P.amp(x.device):
            for a, b in ((0, split), (split, xs.shape[1] - 2)):
                _, l = model(xs[:, a:b], ys[:, a:b], caches=caches, pos_offset=a)
                losses.append(float(l))
            for p in range(xs.shape[1] - 2, xs.shape[1]):
                last, _ = model(xs[:, p:p + 1], caches=caches, pos_offset=p)
        return losses, last.float()

    try:
        got_l, got_last = cached()
        failed = None
    except Exception as e:                                    # noqa: BLE001
        failed = f"{type(e).__name__}: {e}"
    check("a chunk after a cache, and a character after that, run under bf16",
          failed is None, failed or "")
    if failed is None:
        os.environ["MINAGI_CPU_BF16"] = "0"
        ref_l, ref_last = cached()
        os.environ["MINAGI_CPU_BF16"] = "1"
        dl = max(abs(a - b) for a, b in zip(got_l, ref_l))
        check("...and agree with fp32 to bf16's rounding",
              dl < 0.05 and float((got_last - ref_last).abs().max()) < 0.1,
              f"max |dloss| {dl:.4f}")

    trunk, pool_ps = T._split_trunk_pool(model)
    opt = torch.optim.AdamW([{"params": trunk, "lr": 1e-3}, {"params": pool_ps, "lr": 1e-2}])
    pool.attach_optimiser(opt)
    model.train()
    with P.amp(x.device):
        logits, loss = model(x[:, :-1], x[:, 1:])
    loss = loss + model.pool_aux() * 0.01
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("a training step under CPU bf16 has finite loss and fp32 gradients",
          torch.isfinite(loss).item() and grads and all(g.dtype is torch.float32 for g in grads),
          f"loss {float(loss):.3f}")
    before = pool.w1.detach().clone()
    opt.step()
    check("...and steps the fp32 weights", pool.w1.dtype is torch.float32
          and not torch.equal(before, pool.w1.detach()))
    if native is not None:
        print(f"  (this CPU {'has' if native else 'has no'} native bf16)")
finally:
    if saved is None:
        os.environ.pop("MINAGI_CPU_BF16", None)
    else:
        os.environ["MINAGI_CPU_BF16"] = saved
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all CPU bf16 tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
