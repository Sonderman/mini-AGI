#!/usr/bin/env python3
"""
The window grows only while the model is still gaining deep inside it, and
--context is a ceiling that binds on resume: train.py's _window_gain and
_window_bounds, the rules cmd_read reads its window by.

What these tests hold it to:
  the gain is the window's first half of chunk positions against its last
  quarter: positive while the model predicts better deeper in, so the window
  grows; flat or worse deeper in, it does not; with fewer than eight losses
  at either end there is no evidence at all
  a lower --context brings a resumed window down - two runs configured at
  16,384 and then 8,192 both read 32,768 from the manifest and died with
  byte-identical out-of-memory errors, because the setting was silently
  ignored - while the rotary tables still cap it from above, a higher
  --context does not inflate a smaller saved window, a fresh model starts at
  context_start, and the floor holds
  cmd_read reads its window through both
"""
import collections
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as T                                           # noqa: E402
from minagi.config import get, load                         # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


CHUNK = 512
GAIN_MIN = float(get(load(), "model.context_gain_min", 0.01))


def grows(pattern, ctx=2048, rounds=40):
    """The gain for losses following `pattern(position, positions)`, and
    whether cmd_read would grow the window on it."""
    by = collections.defaultdict(list)
    n = max(2, ctx // CHUNK)
    for _ in range(rounds):
        for p in range(n):
            by[p].append(pattern(p, n))
    g = T._window_gain(by, ctx, CHUNK)
    return g, g is not None and g > GAIN_MIN


g, up = grows(lambda p, n: 2.0 - 0.5 * (p / n))
check("still gaining deeper in: the window grows", up, f"gain {g:+.4f}")
g, up = grows(lambda p, n: 2.0)
check("...the far end carried for nothing: it does not",
      not up and abs(g) < 1e-12, f"gain {g:+.4f}")
g, up = grows(lambda p, n: 2.0 + 0.5 * (p / n))
check("...worse deeper in: it does not", not up and g < 0, f"gain {g:+.4f}")
few = collections.defaultdict(list)
few[0] = [1.0] * 3
check("too little evidence is no evidence",
      T._window_gain(few, 2048, CHUNK) is None)
halves = collections.defaultdict(list)
for p in range(4):                       # a 2,048 window is 4 chunks of 512
    halves[p] = [float(p)] * 8
check("...early is the first half of the window's chunks, late its last "
      "quarter", T._window_gain(halves, 2048, CHUNK) == 0.5 - 3.0)

for name, got, want in [
    ("a lower --context brings a resumed window down",
     T._window_bounds(32768, 1024, 32768, 8192), (8192, 8192)),
    ("...the rotary tables still cap it from above",
     T._window_bounds(32768, 1024, 16384, 65536), (16384, 16384)),
    ("...a higher --context does not inflate a smaller saved window",
     T._window_bounds(4096, 1024, 32768, 32768), (4096, 32768)),
    ("...a fresh model with no saved window starts at context_start",
     T._window_bounds(None, 1024, 32768, 32768), (1024, 32768)),
    ("...and the floor still holds",
     T._window_bounds(8, 1024, 32768, 32768), (64, 32768)),
]:
    check(name, got == want, f"{got}")

src = inspect.getsource(T.cmd_read)
check("cmd_read reads its window through both",
      "_window_bounds(" in src and "_window_gain(" in src)

print("\n  all context-growth tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
