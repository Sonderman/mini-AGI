#!/usr/bin/env python3
"""
The gradient must reach every character the model attends over.

THE PROPERTY. A continual learner is taught by what it reads. The reader used
to consume `chunk` characters at a time against a DETACHED key-value cache:
attention reached back over everything read so far, but the gradient stopped
at the chunk boundary. The model could attend over thirty-two thousand
characters and be corrected by five hundred and twelve of them. That is not a
long context, it is context the model can never learn from.

The reader now re-forwards the whole window ending at the cursor, scores only
the new `chunk` characters, and backpropagates through all of it.

The loss spans the whole window. Scoring only the new chunk was tried and
rejected: it needs a "score from here" index threaded through the model's
forward, which is a subtlety that hides in a signature. Scoring everything also
re-learns the last window at every step, which for a model that never stops
training is rehearsal. The cost, accepted knowingly, is that a character
contributes to about R/S consecutive updates and the newest chunk is a small
share of each one.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.recur import RecurCoder, RecurConfig                 # noqa: E402
from minagi.stream import FileReader                             # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


torch.manual_seed(0)
cfg = RecurConfig(vocab_size=265, d_model=32, n_head=2, d_ff=64, block=64,
                  max_steps=2, n_prelude=1, n_recur=1, n_coda=0, use_pool=False)
model = RecurCoder(cfg)

# --------------------------------- 1. the loss covers the whole window
x = torch.randint(0, 265, (1, 40))
y = torch.randint(0, 265, (1, 40))
_, whole = model(x, y)
_, shorter = model(x[:, 32:], y[:, 32:])
check("the loss is computed over every position given to it",
      abs(float(whole) - float(shorter)) > 1e-6,
      f"40 positions {float(whole):.4f}, last 8 alone {float(shorter):.4f}")
import inspect                                              # noqa: E402
check("the model takes no 'score from here' index",
      "loss_from" not in inspect.signature(model.forward).parameters,
      "one objective, no hidden index in the signature")

# ------------------------- 2. THE REACH: gradient flows to the prefix that
# is never itself scored. This is the whole point of the change.
#
# Measured at the EMBEDDING OUTPUT, per position. Reading it off the embedding
# WEIGHT proves nothing: the embeddings are tied to the output head, so every
# vocabulary row receives gradient from the scored positions whatever the
# reach is. The first version of this test did that and passed vacuously.
captured = {}


def _hook(mod, inp, out):
    out.retain_grad()
    captured["emb"] = out
    return None                    # a hook that RETURNS replaces the output


hx = model.tok_emb.register_forward_hook(_hook)
model.zero_grad()
_, loss = model(x, y)
loss.backward()
hx.remove()
per_pos = captured["emb"].grad.abs().sum(-1)[0]
check("every position in the window receives gradient",
      bool((per_pos > 0).all()),
      f"{int((per_pos > 0).sum())} of {per_pos.numel()}, "
      f"smallest {float(per_pos.min()):.2e}")
check("...including the very first character of the window",
      float(per_pos[0]) > 0, f"position 0 gradient {float(per_pos[0]):.2e}")

# --------------------------- 3. LEARNING keeps no cache; MEASURING does
data = np.random.randint(0, 265, 200).astype(np.uint8)
r = FileReader(model, data, "t", chunk=8, context=32, device="cpu")
r.step(learn=True)
check("learning carries no key-value cache", r.caches is None,
      "a detached cache is exactly what cut the gradient")
# Measuring is not learning: under no_grad the gradient reaches nowhere, so a
# cached chunk sees the same attention context and computes the same loss for
# a fraction of the work. Missing this made the held-out evaluation take 66
# minutes at a 16,384 window, at every sample.
rm = FileReader(model, data, "t", chunk=8, context=32, device="cpu")
with torch.no_grad():
    rm.step(learn=False)
check("measuring uses a cache, and does not re-forward the window",
      rm.caches is not None, "same number, 1/(context/chunk) of the work")

# ------------------- 4. the window slides one chunk at a time
r = FileReader(model, data, "t", chunk=8, context=32, device="cpu")
spans = []
for _ in range(6):
    end = r.pos + min(r.chunk, len(r.data) - r.pos - 1)
    start = max(0, end - r.context)
    spans.append((start, r.pos, end))     # window start, first scored, end
    if r.step(learn=True) is None:
        break
advanced = [e - p for _, p, e in spans]
check("each step advances by exactly one chunk", all(s == 8 for s in advanced),
      str(advanced))
check("...and the cursor tiles the text without overlap or gap",
      all(spans[i][2] == spans[i + 1][1] for i in range(len(spans) - 1)),
      " ".join(f"{p}-{e}" for _, p, e in spans))
reach = [p - s for s, p, _ in spans]
check("the context behind each new chunk grows to the window and holds",
      reach[0] == 0 and reach[-1] == r.context - 8 and reach == sorted(reach),
      f"prefix lengths {reach} against a {r.context}-character window")

print("\n  all window-gradient tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
