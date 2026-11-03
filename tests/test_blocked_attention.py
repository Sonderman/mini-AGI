#!/usr/bin/env python3
"""
Attention where the device has no fused kernel (AMD RDNA2, for one) is
computed in blocks, and it is the same attention.

  outputs and gradients match PyTorch's own attention with the
  bottom-right-aligned causal mask: with no cache, with a cache, for a single
  query, and for lengths that are not a multiple of the block

  under autograd it keeps for the backward pass what a fused kernel keeps -
  its inputs, its output and one normaliser per query - where the math path
  keeps every score matrix

  a fused kernel is trusted only once it has computed this attention: one
  that is missing, aligns the mask to the wrong corner, loses a gradient or
  returns NaN is refused, and an honest one passes in fp32 and in bf16

  the whole model reads and trains through it: forced on, held-out loss and
  gradients are those of the default path
"""
import os
import shutil
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


from torch.nn.attention import SDPBackend, sdpa_kernel     # noqa: E402
from minagi.model import (blocked_attention, fused_attention_available,  # noqa: E402
                          _probe_fused)


def reference(q, k, v, P):
    T, kv = q.shape[2], k.shape[2]
    qi = torch.arange(T).unsqueeze(1) + P
    ki = torch.arange(kv).unsqueeze(0)
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=(ki <= qi))


torch.manual_seed(0)
for P, T in ((0, 37), (50, 37), (100, 1), (16, 64)):
    q = torch.randn(2, 4, T, 16, requires_grad=True)
    k = torch.randn(2, 4, P + T, 16, requires_grad=True)
    v = torch.randn(2, 4, P + T, 16, requires_grad=True)
    w = torch.randn(2, 4, T, 16)
    ref = reference(q, k, v, P)
    (ref * w).sum().backward()
    gref = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    got = blocked_attention(q, k, v, P, block=8)
    (got * w).sum().backward()
    err = float((got - ref).abs().max())
    gerr = max(float((t.grad - g).abs().max()) for t, g in zip((q, k, v), gref))
    check(f"same attention and gradients, cache {P}, queries {T}", err < 1e-5 and gerr < 1e-5,
          f"max |d out| {err:.1e}, |d grad| {gerr:.1e}")

# ---- the lower-right causal bias is the offset mask ------------------------------
from torch.nn.attention.bias import causal_lower_right                  # noqa: E402
for P, T in ((0, 9), (7, 9), (40, 1), (100, 37)):
    q = torch.randn(1, 2, T, 16)
    k = torch.randn(1, 2, P + T, 16)
    v = torch.randn(1, 2, P + T, 16)
    a = reference(q, k, v, P)
    b = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(T, P + T))
    check(f"lower-right causal bias == the offset mask, cache {P}, queries {T}",
          torch.equal(a, b) or float((a - b).abs().max()) < 1e-6,
          f"max |d| {float((a - b).abs().max()):.1e}")

# ---- what the backward pass keeps -----------------------------------------------
def saved_bytes(fn, T=512, S=1024):
    """Memory the backward pass holds on to: each storage once, however many
    saved tensors refer to it - every block keeps a reference to the same k
    and v, which is no copy of them."""
    seen = {}

    def pack(t):
        st = t.untyped_storage()
        seen[st.data_ptr()] = st.nbytes()
        return t
    q = torch.randn(1, 8, T, 64, requires_grad=True)
    k = torch.randn(1, 8, S, 64, requires_grad=True)
    v = torch.randn(1, 8, S, 64, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(q, k, v)
    return sum(seen.values())


# What the blocked path keeps is its inputs, its output and one normaliser per
# query - linear in the lengths. The math path keeps the score matrices, which
# grow with their product: doubling both lengths must about double the one and
# about quadruple the other.
def kept(T, S):
    io = (T + 2 * S + T) * 8 * 64 * 4 + T * 8 * 4    # q, k, v, output, normalisers
    b = saved_bytes(lambda q, k, v: blocked_attention(q, k, v, S - T, block=128), T, S)
    m = saved_bytes(lambda q, k, v: reference(q, k, v, S - T), T, S)
    return io, b, m


io1, b1, m1 = kept(512, 1024)
io2, b2, m2 = kept(1024, 2048)
check("blocked keeps its inputs, output and a normaliser per query, and no more",
      b1 <= io1 * 1.05 and b2 <= io2 * 1.05,
      f"{b1 / 2**20:.1f} MB of {io1 / 2**20:.1f}, {b2 / 2**20:.1f} MB of {io2 / 2**20:.1f}")
check("...which grows with the lengths, where the math path grows with their product",
      1.8 < b2 / b1 < 2.2 and m2 / m1 > 3.0 and m1 > 3 * b1,
      f"doubling both: blocked x{b2 / b1:.2f}, math x{m2 / m1:.2f} "
      f"({m2 / 2**20:.0f} MB against {b2 / 2**20:.0f})")

# ---- a fused kernel is trusted only if it computes this attention -------------
cpu = torch.device("cpu")


def honest(q, k, v, attn_mask=None, is_causal=False, corner="bottom-right"):
    """Attention by its definition. The mask the probe asks for - is_causal,
    or a lower-right causal bias - is built here from the shapes."""
    s = (q.float() @ k.float().transpose(-2, -1)) * q.shape[-1] ** -0.5
    T, S = s.shape[-2:]
    keep = torch.ones(T, S, dtype=torch.bool).tril(S - T if corner == "bottom-right" else 0)
    return (s.masked_fill(~keep, float("-inf")).softmax(-1) @ v.float()).to(q.dtype)


def top_left(q, k, v, attn_mask=None, is_causal=False):
    """The classic mistake with a cache: is_causal's triangle, from the top-left."""
    return honest(q, k, v, corner="top-left")


class _LoseKeyGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k):
        return k.clone()

    @staticmethod
    def backward(ctx, g):
        return torch.zeros_like(g)


def lost_gradient(q, k, v, attn_mask=None, is_causal=False):
    return honest(q, _LoseKeyGrad.apply(k), v, attn_mask=attn_mask, is_causal=is_causal)


def nan_out(q, k, v, attn_mask=None, is_causal=False):
    return honest(q, k, v, attn_mask=attn_mask, is_causal=is_causal) * float("nan")


def missing(*a, **kw):
    raise RuntimeError("No available kernel. Aborting execution.")


for dt in (torch.float32, torch.bfloat16):
    for masked in (False, True):
        check(f"an honest kernel passes ({str(dt)[6:]}, {'masked' if masked else 'causal'})",
              _probe_fused(cpu, dt, masked, sdpa=honest))
check("a kernel with the mask in the wrong corner is refused",
      not _probe_fused(cpu, torch.bfloat16, True, sdpa=top_left))
check("a kernel that loses a gradient is refused",
      not _probe_fused(cpu, torch.bfloat16, False, sdpa=lost_gradient)
      and not _probe_fused(cpu, torch.bfloat16, True, sdpa=lost_gradient))
check("a kernel that returns NaN is refused",
      not _probe_fused(cpu, torch.bfloat16, True, sdpa=nan_out))
check("a missing kernel is refused", not _probe_fused(cpu, torch.bfloat16, True, sdpa=missing))
with torch.no_grad():
    check("the probe runs from inside a no-grad forward", _probe_fused(cpu, torch.float32, True,
                                                                        sdpa=honest))
with torch.inference_mode():
    check("and from inside inference mode", _probe_fused(cpu, torch.float32, True, sdpa=honest))
print("  (this CPU's own fused kernels: "
      + ", ".join(f"{'masked' if m else 'causal'} {str(dt)[6:]} "
                  f"{'yes' if _probe_fused(cpu, dt, m) else 'no'}"
                  for dt in (torch.float32, torch.bfloat16) for m in (False, True)) + ")")

# ---- the whole model through it ------------------------------------------------
tmp = tempfile.mkdtemp(prefix="minagi-blocked-")
saved = os.environ.get("MINAGI_ATTENTION")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.pool import PooledMLP                         # noqa: E402
    import train as T_                                        # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T_.build_paged(wdir, "cpu", ram_capacity=3)
    pool.balance = 0.0
    torch.manual_seed(0)
    for site in (m for m in model.modules() if isinstance(m, PooledMLP)):
        site.router.weight.data.normal_(0.0, 1.0)
    x = torch.tensor([list(b"The Roman Empire was the post-Republican period of ancient Rome, "
                            b"and it lasted for centuries.")])

    def held_out():
        model.eval()
        with torch.no_grad():
            logits, _ = model(x[:, :-1])
        return float(F.cross_entropy(logits[0].float(), x[0, 1:]))

    def grads():
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1)
        _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        return float(loss), torch.cat([p.grad.flatten() for p in model.parameters()
                                       if p.grad is not None])

    os.environ["MINAGI_ATTENTION"] = "fused"
    a = held_out()
    la, ga = grads()
    os.environ["MINAGI_ATTENTION"] = "blocked"
    b = held_out()
    lb, gb = grads()
    check("held-out loss through blocked attention", abs(a - b) < 1e-5, f"{a:.6f} vs {b:.6f}")
    rel = float((ga - gb).norm() / ga.norm())
    check("a training step's loss and gradients through it", abs(la - lb) < 1e-5 and rel < 1e-4,
          f"loss {la:.6f} vs {lb:.6f}, relative gradient difference {rel:.1e}")
finally:
    if saved is None:
        os.environ.pop("MINAGI_ATTENTION", None)
    else:
        os.environ["MINAGI_ATTENTION"] = saved
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all blocked-attention tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
