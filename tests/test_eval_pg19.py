#!/usr/bin/env python3
"""
PG19 as tools/eval_pg19.py measures it, and the script that runs the long
measurements.

What these tests hold it to:
  weights the whole split was already scored on are not scored again: a
  record of the same step, pool and characters read, over every book, is
  found; one of part of the split, or of other weights, is not
  the long measurements run from measure_benchmarks.sh - enwik9 frozen,
  with dynamic evaluation and best effort, PG19 frozen and best effort -
  then the chart and the README; it refuses while a run is training; and
  release.sh runs none of them
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import eval_pg19 as EP                                      # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-pg19-")
try:
    path = os.path.join(tmp, "pg19.json")
    man = {"step": 100, "read_chars": 2.5e9, "experts": [{}] * 397}
    rec = {"step": 100, "chars_read": 2.5e9, "experts": 397, "fraction": 1.0,
           "books": 100, "bpb": 2.0}
    seen = {}
    for name, d in (("same", rec), ("part", dict(rec, fraction=0.1)),
                    ("few books", dict(rec, books=10)),
                    ("other weights", dict(rec, step=99))):
        with open(path, "w") as f:
            json.dump(d, f)
        seen[name] = EP.already(man, path)
    os.remove(path)
    check("the whole split on these weights is not scored again",
          seen["same"] is not None and EP.already(man, path) is None)
    check("...but part of the split, or other weights, is",
          seen["part"] is None and seen["few books"] is None
          and seen["other weights"] is None)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

script = open(os.path.join(ROOT, "measure_benchmarks.sh")).read()
release = open(os.path.join(ROOT, "release.sh")).read()
check("measure_benchmarks.sh measures enwik9 three ways and PG19 two, then "
      "the chart and the README",
      "python3 tools/measure_enwik9_adapt.py\n" in script
      and "python3 tools/measure_enwik9_adapt.py --natural\n" in script
      and "python3 tools/measure_enwik9.py" in script
      and "python3 tools/eval_pg19.py" in script
      and "--natural --slice data/eval/pg19" in script
      and script.index("tools/eval_pg19.py") < script.index("plot_scaling.py")
      < script.index("refresh_readme.py"))
check("...refuses while a run is training",
      "measure_heldout.py --check" in script)
check("...is a valid shell script, and release.sh runs none of it",
      subprocess.run(["bash", "-n", os.path.join(ROOT, "measure_benchmarks.sh")]
                     ).returncode == 0
      and not re.search(r"python3 [^\n]*(measure_enwik9|eval_pg19)", release))

print("\n  all PG19 and benchmark-script tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
