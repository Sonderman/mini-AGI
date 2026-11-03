#!/usr/bin/env python3
"""
The parameter count is counted from the weights, and everything that shows
it says the same thing.

  count_params adds up exactly what the files hold: the core and router
  bundles element by element, and every expert the manifest names

  the table `python3 -m minagi.store` prints carries those numbers

  the scaling chart reads its sizes from the same count

  the README's parameter region is regenerated from it, and the region the
  published README carries is there to be found
"""
import os
import re
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-params-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.store import count_params, param_table        # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=12, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=128, max_steps=3,
           top_k=2, depth=1)
    p = count_params(wdir)

    def held(name):
        with np.load(os.path.join(wdir, name)) as z:
            return sum(z[k].size for k in z.files)
    files = sorted(f for f in os.listdir(os.path.join(wdir, "experts")))
    on_disk = 0
    for f in files:
        with np.load(os.path.join(wdir, "experts", f)) as z:
            on_disk += sum(z[k].size for k in ("w1", "w3", "w2"))
    check("the core and the routers are what their bundles hold",
          p["core"] == held("core.npz") and p["routers"] == held("routers.npz"),
          f"{p['core']:,} and {p['routers']:,}")
    check("the experts are every expert's weights on disk",
          p["experts"] == on_disk and p["n_experts"] == len(files) == 12,
          f"{p['n_experts']} x {p['per_expert']:,}")
    check("the total is their sum, and the resident part is the core, the "
          "routers and `resident` experts",
          p["total"] == p["core"] + p["routers"] + p["experts"]
          and p["resident_params"] == p["core"] + p["routers"]
          + 4 * p["per_expert"])

    table = "\n".join(param_table(p))
    check("the printed table carries the total",
          f"{p['total'] / 1e6:.2f}M" in table and "12 x" in table, table.splitlines()[-1])

    import plot_scaling as PS                                 # noqa: E402
    size = PS.model_size(wdir)
    check("the scaling chart reads the same count",
          size and size["n_total"] == p["total"]
          and size["dense_equiv"] == p["dense_equiv"] and size["experts"] == 12)

    import refresh_readme as RR                               # noqa: E402
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    check("the README has a parameter region to refresh",
          "<!-- auto:params -->" in text and "<!-- /auto:params -->" in text)
    root = RR.ROOT
    RR.ROOT = tmp                    # count the small model, not the run's
    os.rename(wdir, os.path.join(tmp, "weights"))
    try:
        new, done = RR.parameters(text)
    finally:
        RR.ROOT = root
    region = re.search(r"<!-- auto:params -->(.*?)<!-- /auto:params -->",
                       new, re.S).group(1)
    check("...and refreshing it writes the count in",
          done and table in region and f"of the {p['total'] / 1e6:.0f}M" in region)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all parameter-count tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
