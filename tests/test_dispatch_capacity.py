#!/usr/bin/env python3
"""
The dispatch buffer must scale with the WORK, not with the imbalance.

THE BUG. The pool runs every expert as three batched matmuls by padding each
one to a fixed slot buffer of size `cap`, so the hidden tensor is
[n_experts, cap, d_ff]. `cap` was whatever the busiest expert happened to
receive. That is only half of "Switch-Transformer style": Switch bounds the
buffer with a capacity FACTOR and drops what overflows, and the bound is the
whole reason the arrangement is affordable.

Measured when this bit: a 4,096-character window, top_k 8, 32 experts present.
32,768 assignments, so a balanced buffer is 1,024 per expert and the hidden
tensor is 0.13 GB. One expert held 44% of the assignments, cap became 14,417,
and the same arithmetic asked for 1.89 GB - fourteen times the memory. The run
died before it took a step, inside the held-out evaluation.

Nothing about the imbalance was new. What was new was a longer window, which
multiplied the assignments by eight and turned a wasteful allocation into an
impossible one.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def buffer_gb(n, cap, d_ff=2048, bytes_per=2):
    return n * cap * d_ff * bytes_per / 1e9


def cap_after(counts, factor):
    """What the dispatch will allocate, given per-expert counts."""
    total, n = sum(counts), len(counts)
    unbounded = max(counts)
    if not factor:
        return unbounded
    return min(unbounded, max(1, math.ceil(factor * total / n)))


# --------------------------------- 1. the unbounded case, from the real OOM
T, k, n = 4096, 8, 32
total = T * k
skew = [int(total * 0.44)] + [(total - int(total * 0.44)) // (n - 1)] * (n - 1)
un = cap_after(skew, 0)
bd = cap_after(skew, 1.5)
check("unbounded, the busiest expert sets the buffer for all of them",
      un == max(skew), f"cap {un:,} from {total:,} assignments across {n}")
check("that is the allocation that killed the run",
      abs(buffer_gb(n, un) - 1.89) < 0.15, f"{buffer_gb(n, un):.2f} GB")
check("bounded at 1.5x fair share it fits",
      buffer_gb(n, bd) < 0.25,
      f"cap {bd:,} -> {buffer_gb(n, bd):.2f} GB "
      f"({buffer_gb(n, un) / buffer_gb(n, bd):.1f}x less)")

# ------------------------- 2. the bound is on SHARE, so it survives a bigger
# window. This is the property that makes the window size safe to raise.
for T2 in (512, 4096, 16384):
    tot = T2 * k
    sk = [int(tot * 0.44)] + [(tot - int(tot * 0.44)) // (n - 1)] * (n - 1)
    waste = n * cap_after(sk, 1.5) / tot
    check(f"at a {T2:>5}-character window the buffer stays near the work",
          waste <= 1.6, f"{waste:.2f}x the assignments actually made")

# ---------------------------------- 3. balanced routing is never penalised
bal = [total // n] * n
check("perfectly balanced routing is not clipped at all",
      cap_after(bal, 1.5) == max(bal),
      f"cap {cap_after(bal, 1.5):,} = the fair share, nothing dropped")

# ------------------------------- 4. and the code actually applies a bound
import inspect                                              # noqa: E402
from minagi.pool import PooledMLP                           # noqa: E402
src = inspect.getsource(PooledMLP._route)            # the dispatch itself
check("the dispatch consults a capacity factor",
      "capacity_factor" in src, "not just counts.max()")
check("...and drops what overflows rather than growing the buffer",
      "dropped" in src)
check("the default is a real bound, not zero",
      inspect.signature(PooledMLP.__init__).parameters
      ["capacity_factor"].default >= 1.0,
      f"default {inspect.signature(PooledMLP.__init__).parameters['capacity_factor'].default}")

print("\n  all dispatch-capacity tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
