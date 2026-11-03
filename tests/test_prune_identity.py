"""
Pruning must not lose or swap anyone's weights.

It used to hold that by RENUMBERING: files were renamed so the directory
stayed contiguous, which meant one deletion could rename every other file and
silently invalidated every record written by position. Now an expert's file is
named by a uid it keeps for life, so a prune renames nothing at all. The
property under test is the same one - your weights are still yours - and it is
now checked against the name the expert was born with.
"""
import os, shutil, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.paged import PagedPool

d = "/tmp/minagi_identity_test"
shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
N, DM, DF = 12, 8, 16
for i in range(N):                      # expert i is filled with the value i
    np.savez(os.path.join(d, f"e{i:05d}.npz"),
             w1=np.full((DF, DM), i, np.float32),
             w3=np.full((DF, DM), i, np.float32),
             w2=np.full((DM, DF), i, np.float32))
p = PagedPool(d, DM, DF, N, resident=2, ram_capacity=2, device="cpu")
p.begin_text()                          # a text that asks for 0 and 1 only:
mass = torch.zeros(N)                   # they are admitted, so resident,
mass[[0, 1]] = 1.0                      # and never pruned
p.admit(mass)

drop = [3, 4, 7]                        # make exactly these prunable
for i in range(N):
    p.born[i] = 0.0 if i in drop else 10**9
    p.gate.data[i] = 0.0 if i in drop else 1.0
    p.last_seen[i] = 0.0
p.segments = 10**6                      # 1 segment per step -> window == survival
gone = p.prune(step=10**6, survival=1)
kept = [i for i in range(N) if i not in drop]
print(f"  pruned {gone}, expecting {len(drop)}")
assert gone == len(drop), f"pruned {gone}, wanted {len(drop)}"

files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
assert len(files) == len(kept), f"{len(files)} files, wanted {len(kept)}"
# the survivors keep the names they were born with; the directory has holes
# where the pruned ones were, and that is the point
assert files == [f"e{i:05d}.npz" for i in kept], f"names moved: {files}"
print(f"  {len(kept)} files kept their own names, holes at {drop}")

# position -> uid is what the pool now uses to find a file
assert [int(u) for u in p.uid] == kept, f"uid map wrong: {[int(u) for u in p.uid]}"
for new, old in enumerate(kept):
    v = float(np.load(os.path.join(d, f"e{old:05d}.npz"))["w1"][0, 0])
    assert v == old, f"file e{old:05d} holds expert {v:.0f}"
    assert p._f(new) == old, f"position {new} maps to uid {p._f(new)}, wanted {old}"
    w = float(p.tiers.fetch(p._f(new))["w1"][0, 0])
    assert w == old, f"position {new} fetched expert {w:.0f}, wanted {old}"
print(f"  every survivor still reachable and still holding its own weights")

# and a new expert takes the next id, never a dead one
p.add_experts(1, step=2, recombine=4)
assert int(p.uid[-1]) == N, f"new uid {int(p.uid[-1])}, wanted {N}"
assert set(int(u) for u in p.uid) & set(drop) == set(), "a dead id was reused"
print(f"  a new expert took id {N}, no dead id reused")
print("  PASS")
