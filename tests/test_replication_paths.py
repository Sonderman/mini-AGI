#!/usr/bin/env python3
"""
A published result names no place on the machine that made it.

  replication/results/*.json are tracked, so they ship with the repository:
  no string in them is an absolute path - a home folder, a scratch directory,
  a drive letter

  the forgetting probe records the weights directory it was given relative to
  the repository when it is inside it, else by its name alone, never as an
  absolute path; the step and pool size beside it say which weights they were
"""
import ast
import glob
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "replication"))
import forgetting_probe as FP                                  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


ABS = re.compile(r"^(/|~|[A-Za-z]:[\\/]|\\\\)")


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from strings(k)
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


# ---- what is published ------------------------------------------------------
files = sorted(glob.glob(os.path.join(ROOT, "replication", "results", "*.json")))
check("there are results to look at", files, f"{len(files)} files")
for p in files:
    found = [s for s in strings(json.load(open(p, encoding="utf-8")))
             if ABS.match(s)]
    check(f"{os.path.relpath(p, ROOT)} names no absolute path", not found,
          str(found[:2]))

# ---- what the probe will write ----------------------------------------------
inside = os.path.join(ROOT, "weights")
check("a directory inside the repository: relative to it",
      FP.where(inside) == "weights", FP.where(inside))
check("...and with a trailing separator",
      FP.where(inside + os.sep) == "weights", FP.where(inside + os.sep))
outside = os.path.join(tempfile.gettempdir(), "scratch-dir", "w")
check("a directory outside it: its name alone",
      FP.where(outside) == "w", FP.where(outside))
check("...relative paths are resolved first, and never come back absolute",
      not os.path.isabs(FP.where(".")) and not os.path.isabs(FP.where("..")))

tree = ast.parse(open(FP.__file__, encoding="utf-8").read())
recorded = [v for n in ast.walk(tree) if isinstance(n, ast.Dict)
            for k, v in zip(n.keys, n.values)
            if isinstance(k, ast.Constant) and k.value == "weights"]
check("the result's \"weights\" is what where() makes of --weights",
      recorded and all(isinstance(v, ast.Call) and getattr(v.func, "id", "") == "where"
                       for v in recorded),
      str([ast.unparse(v) for v in recorded]))

print("\n  " + ("all replication path tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
