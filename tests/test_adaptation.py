"""
The adaptation trace: what was just said is suppressed, and it fades.

Greedy decoding re-enters the same state forever, and the guard that used to
hold it off did not work: rep_penalty marks a character the first time it
appears and never unmarks it, so after 140 characters 93% of the alphabet is
penalised equally and the rule carries no signal. Measured over six prompts,
mean repeated 8-grams: greedy alone 73.5%, with rep_penalty 59.0%, with
adaptation 11.6%.

The trace is short on purpose - a half-life of 5.4 characters - so ordinary
text never feels it and a loop, which cannot let it decay, is crushed.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.decode import pick_next  # noqa: E402

V = 265
A, B, C = 65, 66, 67
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    ok = ok and cond


def pick(logits, prev, **kw):
    return int(pick_next(logits, prev, **kw)[0, 0])


# a character just used is suppressed
lg = torch.zeros(1, V)
lg[0, A] = 1.0
check("without adaptation, it repeats", pick(lg, torch.tensor([[A]])) == A)
check("with adaptation, it does not",
      pick(lg, torch.tensor([[A]]), adapt_strength=2.5) != A)

# ...and the suppression fades, or ordinary text could not be written
far = torch.tensor([[A] + [B] * 30])
check("a character used 30 back is free again",
      pick(lg, far, adapt_strength=2.5) == A,
      f"trace {0.88**30:.4f}")

# a loop cannot let the trace decay, so it is crushed
loop = torch.tensor([[A, B] * 12])
lgl = torch.zeros(1, V)
lgl[0, A] = 5.0          # the model badly wants to continue the loop
lgl[0, C] = 0.0
check("a tight loop is broken even against a 5-logit preference",
      pick(lgl, loop, adapt_strength=2.5) != A)

# the whole point is that it stays deterministic
outs = {pick(lgl, loop, adapt_strength=2.5) for _ in range(8)}
check("still deterministic", len(outs) == 1, f"{outs}")

# off by default, so nothing that does not ask for it changes
check("adapt_strength 0 is exactly the old behaviour",
      pick(lg, torch.tensor([[A]]), adapt_strength=0.0) == A)

# the window must not change the answer for histories shorter than it
short = torch.tensor([[A, B, C]])
check("window longer than the history is harmless",
      pick(lg, short, adapt_strength=2.5, adapt_window=64)
      == pick(lg, short, adapt_strength=2.5, adapt_window=3))

# a batch must not leak between rows
lgb = torch.zeros(2, V)
lgb[:, A] = 1.0
prevb = torch.tensor([[A, A, A], [C, C, C]])
outb = pick_next(lgb, prevb, adapt_strength=2.5)
check("rows of a batch keep their own trace",
      int(outb[0, 0]) != A and int(outb[1, 0]) == A,
      f"{int(outb[0,0])}, {int(outb[1,0])}")

# the decay actually governs the memory
lg2 = torch.zeros(1, V)
lg2[0, A] = 1.0
mid = torch.tensor([[A] + [B] * 10])
check("a shorter half-life forgets sooner",
      pick(lg2, mid, adapt_strength=2.5, adapt_decay=0.5) == A
      and pick(lg2, mid, adapt_strength=2.5, adapt_decay=0.99) != A)

print("\n  " + ("all adaptation tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
