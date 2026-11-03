#!/usr/bin/env python3
"""
The largest attention logit a check reports (minagi/model.py, watch_logits).

What these tests hold it to:
  largest_logit is the largest q.k / sqrt(d) over the keys each query may
  see - with and without cached positions in front - read a few queries at
  a time
  inside watch_logits an attention reports exactly that for what it
  computed, the model's output does not change, and nothing is kept after
  training's check reads its blocks inside it, and the sample log carries
  the number where tools/export_history.py picks it up
"""
import inspect
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

from minagi.model import (Attention, Config, apply_rope, build_rope,  # noqa: E402
                          largest_logit, watch_logits)
import train as T                                             # noqa: E402
import export_history as EH                                   # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def brute(q, k, offset):
    s = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(q.shape[3])
    i = torch.arange(q.shape[2])[:, None] + offset
    j = torch.arange(k.shape[2])[None, :]
    return float(s.masked_fill(j > i, float("-inf")).max())


torch.manual_seed(0)
# -------------------------------------------------- 1. largest_logit alone
q, k = torch.randn(1, 3, 37, 16), torch.randn(1, 3, 37, 16)
got, want = float(largest_logit(q, k, 0, rows=8)), brute(q, k, 0)
check("the largest score over the keys each query may see",
      math.isclose(got, want, rel_tol=1e-5), f"{got:.5f} against {want:.5f}")
# the last key is what the first query would score highest, and it lies in
# that query's future
k[:, :, -1] = 50.0 * q[:, :, 0]
unmasked = float((q @ k.transpose(-1, -2)).max()) / 4.0
got = float(largest_logit(q, k, 0, rows=8))
check("...a key in a query's future does not count",
      math.isclose(got, brute(q, k, 0), rel_tol=1e-5) and got < unmasked,
      f"{got:.2f}, and {unmasked:.2f} had it counted")
qc, kc = torch.randn(1, 3, 5, 16), torch.randn(1, 3, 40, 16)
check("...and with cached positions in front, as a stream reads",
      math.isclose(float(largest_logit(qc, kc, 35, rows=2)),
                   brute(qc, kc, 35), rel_tol=1e-5))
qs, ks = torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
check("...whatever the block size, including blocks that end mid-way",
      all(math.isclose(float(largest_logit(qs, ks, 0, rows=r)),
                       brute(qs, ks, 0), rel_tol=1e-5) for r in (1, 2, 4, 9, 16))
      and all(math.isclose(float(largest_logit(qs[:, :, :3], ks, 6, rows=r)),
                           brute(qs[:, :, :3], ks, 6), rel_tol=1e-5)
              for r in (1, 2, 4)))

# ------------------------------------------- 2. inside an attention module
cfg = Config(vocab_size=64, n_layer=1, n_head=4, d_model=32, block=24)
att = Attention(cfg)
cos, sin = build_rope(cfg.block, att.head_dim, cfg.rope_theta, "cpu")
x = torch.randn(1, cfg.block, cfg.d_model)
with torch.no_grad():
    qq, kk, _ = att.qkv(x).split(cfg.d_model, dim=2)
    qq = apply_rope(qq.view(1, cfg.block, 4, -1).transpose(1, 2), cos, sin)
    kk = apply_rope(kk.view(1, cfg.block, 4, -1).transpose(1, 2), cos, sin)
    want = brute(qq, kk, 0)
    plain = att(x, cos, sin)
    with watch_logits(att) as seen:
        watched = att(x, cos, sin)
check("inside the watch an attention reports the largest logit it computed",
      list(seen) == [""] and math.isclose(seen[""], want, rel_tol=1e-5),
      f"{seen} against {want:.5f}")
check("...its output does not change",
      torch.equal(plain, watched))
check("...and nothing is kept outside the watch",
      att.watching is False and att.largest is None)

# --------------------------------------------- 3. the check and its record
src = inspect.getsource(T.cmd_read)
check("training's check reads its blocks inside the watch",
      "with watch_logits(model) as logits:\n"
      "                            d, v_, se_ = gov.reading()" in src
      and "logits=logits" in src)
tmp = tempfile.mkdtemp(prefix="minagi-logit-")
try:
    log = os.path.join(tmp, "samples.txt")
    T.write_samples(log, 1000, 2_000_000, 12.0, [], val=0.6, se=0.01,
                    per_domain={"a": 0.5, "b": 0.6, "c": 0.7, "d": 0.4,
                                "e": 0.8},
                    experts=12, train=0.55, lr=1e-5,
                    logits={"prelude.0.attn": 9.125,
                            "recur.0.attn": 18.4321})
    text = open(log).read()
    line = next(ln for ln in text.splitlines()
                if ln.startswith("attention logit"))
    rows = EH.parse(log)
    check("the sample log says it before held-out, each attention's beside it",
          line.startswith("attention logit 18.43 at most")
          and "prelude.0 9.12" in line and "recur.0 18.43" in line
          and text.index("attention logit") < text.index("held-out loss"),
          line)
    check("...and the history export reads it into the check's record",
          len(rows) == 1 and rows[0].get("logit") == 18.43
          and rows[0].get("val") == 0.6 and rows[0].get("dom_a") == 0.5,
          f"{rows}")
finally:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all attention-logit tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
