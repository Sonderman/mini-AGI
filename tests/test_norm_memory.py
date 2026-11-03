#!/usr/bin/env python3
"""RMSNorm must not retain an fp32 copy of its input.

The norm runs 3 times per recurrent pass, so at depth 14 that is 42 calls a
window and an fp32 copy of the activation at each one. Keeping the square in
the input dtype - while still ACCUMULATING the mean in fp32, which is the part
that matters numerically - was worth 352 MB at a 4,096 window.

Two ways to give it back silently, both of which this guards:
  x.pow(2)  - autocast keeps pow on its fp32 list, so it upcasts anyway
  * self.weight without a cast - bf16 * fp32 promotes the result to fp32
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                # noqa: E402

from minagi.model import RMSNorm                            # noqa: E402

ok = True


def check(name, cond, note=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + note if note else ''}")


def saved_fp32_bytes(mod, x):
    """Bytes of fp32 the graph holds that are not the input or the weight."""
    skip = {x.untyped_storage().data_ptr(),
            mod.weight.untyped_storage().data_ptr()}
    out = mod(x)
    seen, total, stack, visited, keep = set(), 0, [out.grad_fn], set(), []
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in visited:
            continue
        visited.add(id(fn))
        keep.append(fn)                    # ids get recycled if these are freed
        for attr in dir(fn):
            if not attr.startswith("_saved_"):
                continue
            try:
                v = getattr(fn, attr)
            except Exception:
                continue
            for t in (v if isinstance(v, (tuple, list)) else [v]):
                if not torch.is_tensor(t) or t.dtype is not torch.float32:
                    continue
                ptr = t.untyped_storage().data_ptr()
                if ptr in seen or ptr in skip:
                    continue
                seen.add(ptr)
                total += t.untyped_storage().nbytes()
        for nxt, _ in fn.next_functions:
            stack.append(nxt)
    return out, total


if not torch.cuda.is_available():
    print("  SKIP  needs a card: autocast policy is what is being tested")
    sys.exit(0)

dev = torch.device("cuda")
T_, D = 1024, 512
norm = RMSNorm(D).to(dev)
x = torch.randn(1, T_, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
act = T_ * D * 4                                      # one fp32 copy of it

with torch.autocast("cuda", dtype=torch.bfloat16):
    out, fp32 = saved_fp32_bytes(norm, x)

check("output keeps the input dtype", out.dtype is torch.bfloat16,
      f"got {out.dtype}")
check("no fp32 copy of the activation is retained", fp32 < act // 4,
      f"{fp32/1e6:.2f} MB held; one copy would be {act/1e6:.2f} MB")

# the reduction must still be fp32, or this trades memory for a worse rsqrt
big = torch.full((1, 8, D), 300.0, device=dev, dtype=torch.bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = norm(big)
rms = y.float().pow(2).mean(-1).sqrt().mean().item()
check("still normalises where a bf16 accumulator would overflow",
      abs(rms - 1.0) < 0.02, f"rms {rms:.4f}")

with torch.autocast("cuda", dtype=torch.bfloat16):
    ref = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps)
    want = (x.float() * ref * norm.weight.float()).to(torch.bfloat16)
    rel = ((out.float() - want.float()).abs().mean()
           / want.float().abs().mean()).item()
check("agrees with the fp32 formulation to within a bf16 ulp", rel < 8e-3,
      f"relative {rel:.2e}")

print("\n  OK" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
