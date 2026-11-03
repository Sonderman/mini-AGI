#!/usr/bin/env python3
"""
A read may never run past the rotary tables.

THE BUG. Both readers decided to start a fresh window with `seen >= context`,
then forwarded the next chunk at `pos_offset = seen`. That test covers the
characters already read and not the ones about to be, so the last window of a
pass ran from `seen` to `seen + chunk` - past the ceiling the rotary tables
were built to.

It survived a long time because it is invisible whenever the chunk divides the
context. At 32,768 and 512 the reset lands exactly on the ceiling and the
arithmetic is the same either way; the same is true of 1,024, 2,048 and 4,096.
The first chunk that is not a power of two - 1,536 - read to 33,792 against
tables built to 32,768 and stopped the run.

So the check is parameterised over chunks that divide the context and chunks
that do not, because only the second kind can fail.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.recur import RecurCoder, RecurConfig               # noqa: E402
from minagi.stream import FileReader, Reader                   # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


torch.manual_seed(0)
CTX = 64                                     # the rotary tables end here
cfg = RecurConfig(vocab_size=265, d_model=32, n_head=2, d_ff=64, block=CTX,
                  max_steps=2, n_prelude=1, n_recur=1, n_coda=0, use_pool=False)
model = RecurCoder(cfg)
dev = torch.device("cpu")
data = np.random.randint(0, 260, 4096).astype(np.uint16)

check("the ceiling is the rotary table length", model.rope_cos.shape[0] == CTX,
      f"tables {model.rope_cos.shape[0]}")

DIVIDES = [8, 16, 32, 64]
DOES_NOT = [7, 10, 24, 48, 50]

for label, chunks in (("divides the context", DIVIDES),
                      ("does NOT divide it", DOES_NOT)):
    for ch in chunks:
        # enough steps to wrap the window several times over
        steps = (CTX // ch + 2) * 3

        err = None
        try:
            r = FileReader(model, data, "t", ch, CTX, dev)
            for _ in range(steps):
                if r.done():
                    break
                r.step(learn=False)          # the cached, measuring path
        except Exception as e:               # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        check(f"measuring: chunk {ch:>3} ({label})", err is None, err or "")

        err = None
        try:
            rr = Reader(model, data, "t", chunk=ch, context=CTX, device=dev)
            for _ in range(steps):
                with torch.no_grad():
                    rr.step(learn=False)
        except Exception as e:               # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        check(f"reading:   chunk {ch:>3} ({label})", err is None, err or "")

# The loop above only proves nothing raised. A reader that never reset would
# also not raise - it would quietly read one endless window - so record the
# offset the forward is actually GIVEN, which is the number the ceiling
# applies to. Reading `seen` from outside sees the value before the reset.
CH = 24
r = FileReader(model, data, "t", CH, CTX, dev)
real = model.forward
starts = []


def spy(idx, targets=None, caches=None, pos_offset=0, **kw):
    starts.append((pos_offset, idx.shape[1]))
    return real(idx, targets, caches=caches, pos_offset=pos_offset, **kw)


model.forward = spy
for _ in range(9):
    if r.done():
        break
    r.step(learn=False)
model.forward = real

check("no read ever ends past the ceiling",
      all(o + n <= CTX for o, n in starts),
      f"worst {max((o + n) for o, n in starts)} of {CTX}; "
      f"starts {[o for o, _ in starts]}")
check("the window still resets rather than growing forever",
      starts.count((0, CH)) >= 2,
      f"{starts.count((0, CH))} resets in {len(starts)} reads")

print("\n  OK" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
