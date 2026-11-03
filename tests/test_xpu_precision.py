#!/usr/bin/env python3
"""
Intel XPUs (Arc) compute in the configured precision, as CUDA and ROCm GPUs do.

  an XPU takes the GPU's rule: autocast whenever the dtype is not fp32, with
  bf16 taken as native there, so the configured bf16 applies instead of fp32

  nothing about an XPU is asked of the CUDA queries: they describe whatever
  other GPU the machine holds, so a CUDA card beside an Arc must not decide
  its precision - not even with MINAGI_GPU_BF16=0, which only overrides the
  native-bf16 rule, so the XPU keeps bf16 rather than falling back to fp32 by
  the CUDA card's memory

  CUDA, ROCm and the CPU keep their rules

  on a machine with an XPU, a tiny bf16 forward, backward and update runs on
  it; elsewhere that part is skipped
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi import precision as P                              # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


saved_env = {k: os.environ.pop(k, None)
             for k in ("MINAGI_GPU_BF16", "MINAGI_CPU_BF16")}
saved_dtype = P.compute_dtype()
real = (torch.cuda.current_device, torch.cuda.get_device_properties,
        P._gpu_info)
asked = []


def refuse(*a, **k):
    asked.append(a)
    raise AssertionError("a CUDA query for an XPU")


try:
    # ---- bf16 on an XPU, the CUDA queries out of reach -------------------
    torch.cuda.current_device = refuse
    torch.cuda.get_device_properties = refuse
    P._GPU_INFO.clear()
    P.set_compute_dtype("bf16")
    forms = ("xpu", "xpu:1", torch.device("xpu:1"))
    check("bf16 is native on an XPU, in every form a device is named",
          all(P.gpu_bf16_native(d) for d in forms))
    check("...so forwards there autocast to bf16",
          all(P.autocast_on(d) for d in forms))
    check("...and nothing falls back to fp32 there",
          not any(P.gpu_fp32_instead(d) for d in forms))
    check("the run's first line says bf16",
          all(P.describe(d) == "bf16" for d in forms),
          str([P.describe(d) for d in forms]))
    check("the expert dispatch computes in bf16",
          P.dispatch_dtype(torch.device("xpu:0"), torch.float32)
          is torch.bfloat16)
    # what amp asks torch.autocast for (a real one turns itself off with a
    # warning on a machine without an XPU)
    real_autocast, calls = torch.autocast, []
    torch.autocast = lambda *a, **k: calls.append((a, k))
    try:
        P.amp(torch.device("xpu:0"))
    finally:
        torch.autocast = real_autocast
    check("the autocast region is the XPU's, in bf16",
          calls == [(("xpu",), {"dtype": torch.bfloat16, "enabled": True})],
          str(calls))
    check("no CUDA query was made for any of it", not asked,
          f"{len(asked)} made")

    # ---- a CUDA card beside it decides nothing ---------------------------
    # a pre-Ampere card with the memory for fp32: what would send a CUDA
    # forward to fp32 must not reach the XPU
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda i: SimpleNamespace(
        major=7, minor=5, gcnArchName="", total_memory=24 * 2**30)
    P._GPU_INFO.clear()
    for force, native in (("0", False), ("false", False), ("1", True)):
        os.environ["MINAGI_GPU_BF16"] = force
        check(f"MINAGI_GPU_BF16={force}: the native rule says {native}, "
              f"and the XPU still computes in bf16",
              P.gpu_bf16_native("xpu:0") is native
              and P._gpu_info(torch.device("xpu:0")) is None
              and P.autocast_on("xpu:0") and not P.gpu_fp32_instead("xpu:0")
              and P.describe("xpu:0") == "bf16")
    os.environ.pop("MINAGI_GPU_BF16")
    check("...while that card itself still computes in fp32",
          not P.autocast_on("cuda:0") and P.gpu_fp32_instead("cuda:0"))
    torch.cuda.current_device, torch.cuda.get_device_properties = real[:2]
    P._GPU_INFO.clear()

    # ---- the configured dtype decides ------------------------------------
    for name, dt, on in (("fp32", torch.float32, False),
                         ("bf16", torch.bfloat16, True),
                         ("fp16", torch.float16, True)):
        P.set_compute_dtype(name)
        check(f"{name} on an XPU: autocast {'on' if on else 'off'}",
              P.autocast_on("xpu:0") is on and P.describe("xpu:0") == name
              and P.dispatch_dtype("xpu:0", torch.float64)
              is (dt if on else torch.float64))

    # ---- the other devices keep their rules ------------------------------
    for force in ("0", "1"):
        os.environ["MINAGI_CPU_BF16"] = force
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            P.set_compute_dtype(dt)
            want = force == "1" and dt is torch.bfloat16
            check(f"CPU, MINAGI_CPU_BF16={force}, {dt}: autocast {want}",
                  P.autocast_on("cpu") is want)
    os.environ.pop("MINAGI_CPU_BF16")
    P.set_compute_dtype("bf16")
    gpus = ((7, "", False, False), (8, "", False, True),
            (0, "gfx1030", True, False), (0, "gfx1100", True, True),
            (0, "gfx90a", True, True))
    for major, arch, amd, native in gpus:
        for gib in (8, 24):
            props = SimpleNamespace(major=major, minor=0, gcnArchName=arch,
                                    total_memory=gib * 2**30)
            P._gpu_info = lambda device=None, _p=props, _a=amd: (_p, _a)
            want = native or gib < P.FP32_MIN_GIB
            check(f"{'AMD ' + arch if amd else f'NVIDIA sm_{major}0'}, "
                  f"{gib} GB: autocast {want}",
                  P.gpu_bf16_native("cuda") is native
                  and P.autocast_on("cuda") is want and P.autocast_on() is want)
    P._gpu_info = real[2]

    # ---- a real XPU, where there is one -----------------------------------
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        P.set_compute_dtype("bf16")
        dev = torch.device("xpu:0")
        lin = torch.nn.Linear(16, 8, device=dev)
        before = lin.weight.detach().clone()
        opt = torch.optim.SGD(lin.parameters(), lr=0.01)
        with P.amp(dev):
            out = lin(torch.ones(4, 16, device=dev))
            loss = out.float().square().mean()
        loss.backward()
        opt.step()
        xpu.synchronize(dev)
        check(f"on {xpu.get_device_name(dev)}: the forward is bf16, the "
              f"weights and gradients fp32 and finite, and a step moves them",
              out.dtype is torch.bfloat16 and bool(torch.isfinite(loss))
              and all(q.dtype is torch.float32 and q.grad.dtype is torch.float32
                      and bool(torch.isfinite(q.grad).all())
                      for q in lin.parameters())
              and not torch.equal(before, lin.weight))
    else:
        print("  SKIP  no XPU here: the forward on a real one is not run")
finally:
    torch.cuda.current_device, torch.cuda.get_device_properties = real[:2]
    P._gpu_info = real[2]
    P._GPU_INFO.clear()
    P.set_compute_dtype(saved_dtype)
    for k, v in saved_env.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v

print("\n  " + ("all XPU precision tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
