"""
Every growth condition reads the network as it stands, not a running total.

USED asks what fraction of gates are far below the median. EARNING asks
whether the experts that recently finished their trial cleared the bar. Both
are computed from `gate` and `born` at the moment of asking, so a restart, a
gap in the log, or a checkpoint written by an older version changes nothing.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.pool import AutoGrow  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    ok = ok and cond


class P:
    """Just enough pool for _earning to read.

    `dying` is staleness as a share of the prune window, which is what
    "earning its place" now means: 0 is something asked for it just now, 1 is
    the line prune deletes on. It used to be the gate, and the gate turned out
    to be near-uninformative about whether an expert would be asked for again.
    """

    def __init__(self, born, dying, trial=1600, dying_at=0.75):
        self.born = torch.tensor(born, dtype=torch.float)
        self._dying = torch.tensor(dying, dtype=torch.float)
        self.trial = trial
        self.dying_at = dying_at

    def dying(self):
        return self._dying


g = AutoGrow(grow_k=4, recent_mult=4.0)

# a pool of originals has not failed at anything
p = P([0.0] * 8, [0.0] * 8)
check("a pool of originals is not judged", g._earning(p, 50_000) == 1.0)

# experts still inside their trial are not judged either
p = P([0.0] * 4 + [49_000] * 4, [0.0] * 4 + [0.9] * 4)
check("an expert still on trial is not judged",
      g._earning(p, 50_000) == 1.0, "age 1000 < trial 1600")

# once past the trial, being asked for is the verdict
p = P([0.0] * 4 + [45_000] * 4, [0.0] * 4 + [0.1, 0.2, 0.9, 0.95])
check("half the recent additions earning reads 0.5",
      g._earning(p, 50_000) == 0.5, "age 5000, past trial")

p = P([0.0] * 4 + [45_000] * 4, [0.0] * 4 + [0.9] * 4)
check("none earning reads 0.0", g._earning(p, 50_000) == 0.0)

# and an addition too old to be recent drops out of the question
p = P([0.0] * 4 + [100.0] * 4, [0.0] * 4 + [0.9] * 4)
check("an ancient addition is no longer recent",
      g._earning(p, 50_000) == 1.0, "age 49900 > 4 x 1600")

# nothing is carried between calls
p = P([0.0] * 4 + [45_000] * 4, [0.0] * 4 + [0.1, 0.2, 0.9, 0.95])
a = g._earning(p, 50_000)
b = AutoGrow(grow_k=4, recent_mult=4.0)._earning(p, 50_000)
check("a fresh grower reads the same answer", a == b,
      "no state survives between decisions")

print("\n  " + ("all instantaneous-growth tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
