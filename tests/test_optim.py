#!/usr/bin/env python3
"""
The signal meter.

Half the energy of each gradient is a fixed direction and half is noise, so
the meter should read about 0.5 on one stream and about nothing on the other.
The scaling is the whole test: with sigma per coordinate the noise carries
sigma^2 * D, so sigma = 1/sqrt(D) puts as much energy in the noise as in the
unit signal. An earlier version used sigma = 0.5 - sixteen times the signal -
and then expected a high reading. The meter was right and the test was wrong.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.optim import GradSNR                       # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


N = 400
torch.manual_seed(1)
D = 64
d = torch.randn(D); d /= d.norm()

SIG = 1.0 / D ** 0.5
consistent = [d + SIG * torch.randn(D) for _ in range(N)]
# the same energy, no consistent direction
noise = [(2 ** 0.5 / D ** 0.5) * torch.randn(D) for _ in range(N)]

m = GradSNR(beta=0.9)
p = torch.zeros(64, requires_grad=True)
for g in consistent[:80]:
    p.grad = g
    r_sig = m.observe([p])
m2 = GradSNR(beta=0.9)
for g in noise[:80]:
    p.grad = g
    r_noise = m2.observe([p])

check("the meter reads about the true signal share", 0.3 < r_sig < 0.7,
      f"{r_sig:.3f}, expected ~0.5")
check("the meter reads near zero on pure noise", r_noise < 0.1,
      f"{r_noise:.3f}")
check("and it separates them clearly", r_sig > 4 * r_noise,
      f"{r_sig:.3f} vs {r_noise:.3f}")
check("it returns nothing until it has evidence",
      GradSNR().ratio() is None)
check("nor on a parameter that never got a gradient",
      GradSNR().observe([torch.zeros(4, requires_grad=True)]) is None)

# The live crash: the halting head takes no part in a step whose sampled
# depth is one pass, so its gradient is None for that step alone - and the
# meter, built for 8,135,169 values, was handed 8,134,656 and stopped the run.
body = torch.zeros(64, requires_grad=True)
head = torch.zeros(513, requires_grad=True)            # Linear(512, 1)
m3 = GradSNR(beta=0.9)
failed = None
try:
    for i, g in enumerate(consistent[:80]):
        body.grad = g
        head.grad = None if i % 7 == 3 else torch.zeros(513)
        r_mixed = m3.observe([body, head])
except RuntimeError as e:
    failed = str(e)
check("a parameter with no gradient for a step counts as zero, and the meter carries on",
      failed is None and r_mixed is not None and 0.3 < r_mixed < 0.7,
      failed or f"{r_mixed:.3f}")
m4 = GradSNR(beta=0.9)
for g in consistent[:20]:
    body.grad = g
    m4.observe([body])
wider = torch.zeros(80, requires_grad=True)
wider.grad = torch.randn(80)
try:
    m4.observe([wider])
    restarted = m4.n == 1 and m4.m.numel() == 80
except RuntimeError:
    restarted = False
check("a parameter set that changes shape restarts the meter instead of stopping the run",
      restarted)

print("\n  all optimiser tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
