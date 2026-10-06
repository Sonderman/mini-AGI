"""The transformer itself.

Decoder-only on the current small-model recipe: RMSNorm, rotary position
embeddings, SwiGLU feed-forward, flash attention through
scaled_dot_product_attention, tied input/output embeddings, no biases.

Positions are rotary and carry no learned parameters, which is why the context
window extends by continued training rather than by re-initialising anything -
see stream.ramp_context.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from torch.utils.checkpoint import checkpoint

from dataclasses import dataclass


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    block: int = 512
    d_ff: int = 1408          # ~8/3 * d_model, rounded to a multiple of 64
    rope_theta: float = 10000.0
    tie_embeddings: bool = True


# ----------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalise in the input dtype, reduce in fp32. Three details, each
        # of which exists to keep an fp32 copy of the activation out of the
        # backward graph - that copy is what dominates activation memory at
        # this depth:
        #
        #   mean(dtype=fp32)  accumulates the reduction in fp32 without
        #                     upcasting x itself, so accuracy is kept where
        #                     the reciprocal-sqrt needs it and nowhere else.
        #   x * x, not pow()  autocast keeps pow on its fp32 list, so pow()
        #                     upcasts and its backward retains the copy. mul
        #                     is not on that list.
        #   weight.to(dtype)  bf16 * fp32 promotes the product back to fp32,
        #                     so the cast is explicit.
        #
        # The cost is one bf16 rounding on the scale multiply, well under the
        # held-out noise floor.
        scale = torch.rsqrt((x * x).mean(-1, keepdim=True,
                                         dtype=torch.float32) + self.eps)
        return x * scale.to(x.dtype) * self.weight.to(x.dtype)


def build_rope(block, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                           / head_dim))
    pos = torch.arange(block, device=device).float()
    freqs = torch.outer(pos, inv)                      # [T, hd/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: [B, H, T, hd]; rotate pairs (even, odd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# ----------------------------------------------------------------------------
# attention where no fused kernel exists
# ----------------------------------------------------------------------------
#
# scaled_dot_product_attention picks a fused kernel (flash or memory-efficient)
# when the device has one. Where it has none it falls back to a math path that
# materialises every score matrix and, under autograd, KEEPS it for the
# backward pass: once per block application, 26 per chunk at this model's
# depth, which on an RX 6800 XT was most of its 16 GB and an out-of-memory
# error two minutes in.
#
# AMD's consumer GPUs get fused kernels from AOTriton, in PyTorch ROCm builds
# recent enough to carry them (2.9 does for RDNA2, 2.5 does not), behind a
# switch PyTorch still calls experimental. It is turned on here, before
# anything reaches attention, unless the environment already sets it. A fused
# kernel is used only after its output and gradients have matched the blocked
# path below on a small problem; where there is none - an older build, another
# device - the blocked path computes the same attention with no score matrix
# kept.

if torch.version.hip:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

_FUSED = {}


def fused_attention_available(device, dtype, masked=False):
    """
    Whether `device` has a fused attention kernel for `dtype` that computes
    this model's attention: causal over a chunk with no cache, or, with
    `masked`, causal aligned to the bottom-right, as a chunk after a cache
    needs. Probed once each. MINAGI_ATTENTION=blocked or =fused overrides the
    probe.
    """
    mode = os.environ.get("MINAGI_ATTENTION", "auto").strip().lower()
    if mode == "blocked":
        return False
    if mode == "fused":
        return True
    key = (device.type, device.index, dtype, masked)
    if key not in _FUSED:
        _FUSED[key] = _probe_fused(device, dtype, masked)
        if not _FUSED[key]:
            # once, so a run that is slower than it should be says why
            print(f"  attention: no fused {str(dtype).replace('torch.', '')} kernel for "
                  f"{'a chunk after a cache' if masked else 'a window'} on {device} that "
                  f"passes the check - computing it in blocks", flush=True)
    return _FUSED[key]


def _probe_fused(device, dtype, masked, sdpa=None):
    """
    The fused kernels alone on a small causal problem, forward and backward,
    against the blocked path in fp32. A missing kernel raises; one that is
    present and wrong - the mask aligned to the wrong corner, a broken
    backward - disagrees by far more than rounding does (under 1% in bf16).
    """
    import warnings
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sdpa = sdpa or F.scaled_dot_product_attention
    P, T = (96, 160) if masked else (0, 256)
    try:
        # PyTorch explains every backend it skips, a warning each; the one
        # line fused_attention_available prints says what matters
        with warnings.catch_warnings(), torch.inference_mode(False), \
                torch.enable_grad(), torch.autocast(device.type, enabled=False):
            warnings.simplefilter("ignore")
            g = torch.Generator().manual_seed(0)
            q, k, v, w = (torch.randn(1, 2, n, 64, generator=g).to(device=device, dtype=dtype)
                          for n in (T, P + T, P + T, T))
            x = [t.requires_grad_() for t in (q, k, v)]
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                if masked:
                    y = sdpa(*x, attn_mask=causal_lower_right(T, P + T))
                else:
                    y = sdpa(*x, is_causal=True)
                got = (y,) + torch.autograd.grad((y * w).sum(), x)
            r = [t.detach().float().requires_grad_() for t in x]
            y = blocked_attention(*r, P)
            ref = (y,) + torch.autograd.grad((y * w.float()).sum(), r)
        return all(float((a.detach().float() - b.detach()).norm() / b.detach().norm()) < 0.05
                   for a, b in zip(got, ref))
    except Exception:                                      # noqa: BLE001
        return False


def _attend_block(qb, k, v, start, scale):
    """Queries at absolute positions start..start+len-1 against keys 0..end-1:
    the scores of one block, masked causally, softmax in fp32."""
    end = start + qb.shape[2]
    with torch.autocast(qb.device.type, enabled=False):
        s = torch.matmul(qb.float(), k[:, :, :end].float().transpose(-2, -1)) * scale
        qi = torch.arange(start, end, device=qb.device).unsqueeze(1)
        ki = torch.arange(end, device=qb.device).unsqueeze(0)
        s = s.masked_fill(ki > qi, float("-inf"))
        y = torch.matmul(s.softmax(-1), v[:, :, :end].float())
    return y.to(v.dtype)


def blocked_attention(q, k, v, P, block=512):
    """
    Causal attention for T queries at absolute positions P..P+T-1 against
    P+T keys, without a fused kernel and without keeping a score matrix.

    Queries go through in blocks; a block sees only the keys up to its own
    last position, so the masked-out part of the matrix is not computed at
    all; scores and softmax are fp32, as a fused kernel accumulates them. Under
    autograd each block is recomputed in the backward pass instead of stored,
    so what training keeps is q, k and v - the same as a fused kernel keeps.
    """
    scale = q.shape[-1] ** -0.5
    grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad
                                        or v.requires_grad)
    out = []
    for s in range(0, q.shape[2], block):
        qb = q[:, :, s:s + block]
        if grad:
            out.append(checkpoint(_attend_block, qb, k, v, P + s, scale,
                                  use_reentrant=False))
        else:
            out.append(_attend_block(qb, k, v, P + s, scale))
    return out[0] if len(out) == 1 else torch.cat(out, dim=2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # Masking with a cache is not the same problem as masking without one.
        # `is_causal` aligns its triangle to the TOP-LEFT of the score matrix,
        # which is correct only when the queries and the keys are the same
        # positions. With P cached positions the T queries sit at absolute
        # P..P+T-1 against P+T keys, so the triangle has to be offset by P -
        # otherwise query i sees keys 0..i instead of 0..P+i and every chunk
        # after the first attends to the wrong window.
        #
        # kv_len == T means there is no cache and the fast path is correct.
        # With a cache, causal_lower_right says the same thing as the offset
        # mask (ki <= qi) without building one: PyTorch hands it to the flash
        # kernel, which skips the blocks wholly in the future, or to the
        # memory-efficient kernel's own causal variant, and materialises the
        # mask only where neither exists - the CPU, which gets what it had.
        # Under autocast q and k leave apply_rope in fp32 - the rotary tables
        # are fp32 - while v is in the compute dtype. Plain SDPA casts all
        # three itself, being on autocast's list; causal_lower_right is
        # dispatched in Python before autocast sees the call, and refuses
        # mixed dtypes. So they are cast here, to what autocast casts them
        # to - which is also the dtype the fused-kernel check below must ask
        # about, since it is the one the kernel runs in.
        if q.dtype != v.dtype:
            q, k = q.to(v.dtype), k.to(v.dtype)

        # Without a fused kernel for the case at hand, blocked_attention.
        kv_len = k.shape[2]
        if T == 1:
            # one query, at the newest position: it sees every key, so no mask
            # is built - and none is needed for a fast kernel to take it. Its
            # score matrix is a single row, so even the math path keeps nothing
            y = F.scaled_dot_product_attention(q, k, v)
        elif kv_len == T:
            if fused_attention_available(q.device, q.dtype):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                y = blocked_attention(q, k, v, 0)
        elif fused_attention_available(q.device, q.dtype, masked=True):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_lower_right(T, kv_len))
        else:
            y = blocked_attention(q, k, v, kv_len - T)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def kv(self, x, cos, sin):
        """These positions' keys and values, as forward() would cache them,
        with nothing else computed."""
        B, T, C = x.shape
        k, v = F.linear(x, self.qkv.weight[C:]).split(C, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return apply_rope(k, cos, sin), v

    @staticmethod
    def extend(cache, k, v):
        """Append keys and values to a cache, as forward() does."""
        if cache.get("k") is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # value
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, active=None):
        """`active` marks the positions still being computed; an expert pool
        runs its experts for those alone. Attention still reads every
        position, because the rest are still what later positions see."""
        x = x + self.attn(self.ln1(x), cos, sin, cache)
        if active is not None and getattr(self.mlp, "takes_active", False):
            return x + self.mlp(self.ln2(x), active)
        return x + self.mlp(self.ln2(x))
