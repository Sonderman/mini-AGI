#!/usr/bin/env python3
"""
A pool that has pruned must be able to save and load itself.

The bug: _save_paged built each expert's filename from its POSITION, which
was the same as its name until pruning stopped renumbering. After that it
skipped every position whose file had a different name, so a pool of 137
experts wrote n_experts 128 while saving 137 router rows - and the directory
could not be loaded back at all:

    size mismatch for pool.gate: copying a param with shape [137]
    from checkpoint, the shape in current model is [128]

The invariant these tests hold: n_experts == len(uid) == router rows == files.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from minagi.paged import PagedPool                       # noqa: E402
from minagi import store as weights_store                # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


DM, DF, N = 16, 32, 12
d = tempfile.mkdtemp()
ed = os.path.join(d, "experts")
os.makedirs(ed, exist_ok=True)
try:
    for i in range(N):
        np.savez(os.path.join(ed, "e%05d.npz" % i),
                 w1=np.full((DF, DM), i + 1, np.float32),
                 w3=np.full((DF, DM), i + 1, np.float32),
                 w2=np.full((DM, DF), i + 1, np.float32))

    class Tiny(torch.nn.Module):
        def __init__(self, pool):
            super().__init__()
            self.pool = pool
            self.head = torch.nn.Linear(DM, 4, bias=False)

    p = PagedPool(ed, DM, DF, N, resident=2, device="cpu")
    p.gate.data.copy_(torch.full((N,), 0.30))
    p.born.fill_(0.0); p.slots = [-1, -1]
    p.segments = 10**6
    p.last_seen.fill_(10**6)                   # everyone chosen just now...
    p.last_seen[[2, 5, 9]] = 0.0               # ...except three, which will go
    gone = p.prune(step=10**6, survival=10)
    p.add_experts(2, step=2, recombine=4)      # ids continue past the count
    uid = [int(u) for u in p.uid]
    check("the pool pruned then grew", gone == 3 and p.n_experts() == 11,
          f"{gone} gone, {p.n_experts()} experts")
    check("ids are no longer positions",
          any(u != i for i, u in enumerate(uid)), f"uid {uid}")

    model = Tiny(p)
    weights_store.save(model, d, step=7, val=1.0, cfg={"pool_max": 64})

    man = json.load(open(os.path.join(d, "manifest.json")))
    r = np.load(os.path.join(d, "routers.npz"))
    files = sorted(int(f[1:6]) for f in os.listdir(ed) if f.endswith(".npz"))
    check("manifest n_experts matches the pool",
          man["n_experts"] == p.n_experts(),
          f"manifest {man['n_experts']}, pool {p.n_experts()}")
    check("...and the router rows it just wrote",
          r["pool.gate"].shape[0] == man["n_experts"],
          f"gate rows {r['pool.gate'].shape[0]}")
    check("...and the files on disk", files == sorted(uid),
          f"{len(files)} files")
    check("every expert is listed under its own id",
          sorted(int(e["id"]) for e in man["experts"]) == sorted(uid))
    check("each entry names the file that exists",
          all(os.path.exists(os.path.join(ed, e["file"]))
              for e in man["experts"]))
    check("the ids round-trip through telemetry",
          [int(u) for u in man["telemetry"]["uid"]] == uid)
    check("next_uid survives", man["telemetry"]["next_uid"] == p.next_uid,
          f"{man['telemetry']['next_uid']}")

    # the gate recorded per entry must be that expert's, not the one whose
    # position happens to share its number
    g = {int(e["id"]): e["gate"] for e in man["experts"]}
    live = {int(p.uid[i]): float(p.gate.data[i]) for i in range(p.n_experts())}
    check("each entry carries its own gate",
          all(abs(g[k] - live[k]) < 1e-6 for k in live))

    # and the whole point: a fresh pool sized from the manifest loads it
    p2 = PagedPool(ed, DM, DF, man["n_experts"], resident=2, device="cpu")
    p2.load_telemetry(man["telemetry"])
    # p2 IS the pool, so the keys lose their "pool." prefix
    sd = {"gate": torch.tensor(r["pool.gate"])}
    missing, unexpected = p2.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k == "gate"]
    check("a pool sized from the manifest accepts the saved tensors", not bad)
    check("and finds each expert's real file",
          all(os.path.exists(p2.tiers._file(p2._f(i)))
              for i in range(p2.n_experts())))
finally:
    shutil.rmtree(d, ignore_errors=True)

print("\n  all save round-trip tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
