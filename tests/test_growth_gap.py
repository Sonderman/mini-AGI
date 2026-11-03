#!/usr/bin/env python3
"""
Growth is gated on the generalisation gap, not on parameters per character:
train.py's _may_grow, which cmd_read asks before it lets AutoGrow grow.

The ceiling it replaced was a proxy and a poor one. Across 117 checkpoints of
the live run its correlation with the gap was +0.20 and not monotonic - the
gap at density 20-40 was 0.066, LOWER than the 0.111 at 10-14 - and it had
been set to 8.0, an order of magnitude below the 96 where memorising was
actually observed (train 1.63, held-out 2.75). It held growth off for 14M
characters on evidence that did not exist.

What these tests hold it to:
  a run that generalises may grow, and one that has begun memorising may
  not - whatever its size, which is no longer asked at all
  every gap the live run produced permits growth
  a gap not yet measured does not block it, and max_gap 0 turns the test off
  cmd_read asks it, with the configured threshold
"""
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as T                                           # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


MAX_GAP = 0.4                    # growth.max_gap
g = T._may_grow
check("a small gap permits growth", g(0.069, MAX_GAP))
check("the measured memorising case is refused", not g(1.12, MAX_GAP),
      "train 1.63, held-out 2.75")
check("...and so is a gap at the threshold", not g(MAX_GAP, MAX_GAP))
check("every gap observed in 117 checkpoints permits growth",
      all(g(x, MAX_GAP) for x in (0.058, 0.069, 0.111, 0.150)),
      "0.058 .. 0.150, all under 0.4")
check("an unknown gap does not block growth", g(None, MAX_GAP))
check("max_gap 0 disables the gap test", g(5.0, 0))

src = inspect.getsource(T.cmd_read)
check("cmd_read asks it, with the configured threshold",
      "_may_grow(last_gap, args.max_gap)" in src)

print("\n  all growth-gap tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
