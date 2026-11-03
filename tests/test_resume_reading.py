#!/usr/bin/env python3
"""
A restart must continue reading, not replay the same tape.

THE BUG. Each subject lane picks a random file and a random offset inside it,
from a generator seeded `shuffle_seed + lane_index`. Both parts are fixed, and
the file list is sorted, so the generator was rebuilt IDENTICALLY at every
start. Every session read the same files at the same offsets in the same
order, beginning again from the front of that sequence.

Measured over 43 real sessions: the median read 3.2M characters and the
longest 66M, while the counter reported 500M. The unique text seen was about
the length of the longest session, because every short session was re-reading
the opening of the same sequence.

WHAT IT LOOKED LIKE. A restart spike nobody could place. At the last one,
train loss fell 0.675 -> 0.446 across the seam while held-out rose 0.925 ->
1.105 and the gap went +0.250 -> +0.660. The learning rate was identical on
both sides, nothing was pruned and nothing was added, which is exactly what
made it read as lost state rather than as repeated data. It was neither: the
model was being handed text it had already memorised and pushed to overfit it
further.

THE FIX. Mix the characters-already-read into the seed. Nothing new is
persisted - the count is in the manifest - and a run stays reproducible from
any given checkpoint.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train                                                # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


FILES = [f"data/train/{sub}/{i:05d}.txt"
         for sub in ("alpha", "beta", "gamma") for i in range(500)]
ROOTS = ["data/train"]


def draw(resume, n=12, seed=0):
    """The files and offsets a fresh process would visit, in order."""
    lanes = train._lanes(FILES, seed, ROOTS, resume=resume)
    out = []
    for L in lanes:
        out.append([(int(L.rng.integers(0, len(L.paths))),
                     int(L.rng.integers(0, 10 ** 6))) for _ in range(n)])
    return out


lanes = train._lanes(FILES, 0, ROOTS, resume=0)
check("one lane per top-level subject", len(lanes) == 3,
      f"{[L.name for L in lanes]}")

# ---------------------------------- 1. the bug itself, stated as a property
a = draw(500_000_000)
b = draw(500_000_000)
check("the same checkpoint reads the same thing", a == b,
      "reproducible, so a run can be repeated from a checkpoint")

c = draw(503_200_000)            # one median session later
check("a LATER checkpoint reads something else", a != c,
      "this is the whole fix: 3.2M characters on, the tape is different")

# ------------------------------ 2. and it is different in every lane, not
#                                   just the one whose index happens to shift
same = [x == y for x, y in zip(a, c)]
check("every lane moves on, not only the first",
      not any(same), f"lanes still identical: {sum(same)} of {len(same)}")

# --------------- 3. the failure mode: seeding without the read count at all
#     `seed + j` is what the code used to do. Two sessions collide completely.
import numpy as np                                          # noqa: E402


def old_draw(n=12, seed=0):
    return [[(int(r.integers(0, 1500)), int(r.integers(0, 10 ** 6)))
             for _ in range(n)]
            for r in (np.random.default_rng(seed + j) for j in range(3))]


check("the old seeding really did replay identically",
      old_draw() == old_draw(),
      "seed + lane_index alone, rebuilt at every start")

# -------------- 4. overlap between consecutive sessions should be near zero
#     Draw a long session from each of two checkpoints and count how many
#     (file, offset) visits they share. Under the old rule this was 100%.
A = set(map(tuple, draw(500_000_000, n=400)[0]))
C = set(map(tuple, draw(503_200_000, n=400)[0]))
overlap = len(A & C) / max(len(A), 1)
check("consecutive sessions barely overlap", overlap < 0.02,
      f"{overlap:.1%} of visits shared, against 100% before")

# ------------- 5. the read count actually reaches _lanes from the manifest
import inspect                                              # noqa: E402
src = inspect.getsource(train.cmd_read)
check("cmd_read passes the characters already read",
      "_lanes(" in src and "resume=" in src and "read_chars" in src,
      "the count is already in the manifest; nothing new is persisted")

# ------------- 6. a fresh model (nothing read) is still perfectly ordinary
f = draw(0)
check("a run from zero is well defined", len(f) == 3 and f != a,
      "resume 0 is just another point on the tape")

print("\n  all resume-reading tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
