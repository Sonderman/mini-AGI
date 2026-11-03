#!/usr/bin/env python3
"""
measure_benchmarks.sh ships with everything it needs, so anyone with the
weights can repeat the README's benchmark tables.

What these tests hold it to:
  every script measure_benchmarks.sh runs, and every module of this repo
  those import - followed all the way down - is tracked by git, so a push
  carries it; the script itself is tracked too
  train.py ships whole, notifier and all: minagi/notify.py, which train.py
  imports, is tracked with it
  so is tools/measure_enwik9_gpt2.py, which measured the enwik9 table's
  GPT-2 rows, with everything it imports
  the set is not trivially satisfied: it reaches the six tools the script
  runs, train.py and the model package
"""
import ast
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def module_file(name):
    """The file in this repo a module name resolves to, or None: a tool,
    imported the way the tools import each other, train.py, or a module of
    minagi/ or corpora/."""
    parts = name.split(".")
    if len(parts) == 1 and os.path.exists(os.path.join(ROOT, "tools",
                                                       name + ".py")):
        return f"tools/{name}.py"
    if name == "train":
        return "train.py"
    if parts[0] in ("minagi", "corpora"):
        for cand in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
            if os.path.exists(os.path.join(ROOT, cand)):
                return cand
        return f"{parts[0]}/__init__.py"
    return None


def source(p):
    """A file of this repo, as it is."""
    return open(os.path.join(ROOT, p), encoding="utf-8").read()


def closure(start):
    """Every file of this repo `start` reaches through its imports."""
    seen, todo = set(), list(start)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        tree = ast.parse(source(p))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                names = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            else:
                continue
            for m in names:
                f = module_file(m)
                if f and f not in seen:
                    todo.append(f)
    return seen


script = open(os.path.join(ROOT, "measure_benchmarks.sh"),
              encoding="utf-8").read()
runs = sorted(set(re.findall(r"python3 (tools/\w+\.py)", script)))
# not run by the script, but what measured the GPT-2 rows it compares with
GPT2 = "tools/measure_enwik9_gpt2.py"
need = closure(runs + [GPT2])

# what a push carries
tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True).stdout.splitlines())

check("train.py ships whole, and the notifier train.py imports is tracked "
      "with it",
      "from minagi import notify" in source("train.py")
      and "minagi/notify.py" in need and "minagi/notify.py" in tracked)
missing = sorted(p for p in need | {"measure_benchmarks.sh"}
                 if p not in tracked)
check("everything measure_benchmarks.sh runs and imports is tracked",
      not missing, f"missing: {missing}" if missing else
      f"{len(need)} files")
check("...and that is more than the script: the tools it runs, train.py and "
      "the model package",
      set(runs) >= {"tools/measure_heldout.py", "tools/measure_enwik9.py",
                    "tools/measure_enwik9_adapt.py", "tools/eval_pg19.py",
                    "tools/plot_scaling.py", "tools/refresh_readme.py"}
      and {"train.py", "minagi/live.py", "minagi/store.py",
           "tools/plot_progress.py", GPT2} <= need, f"{runs}")

print("\n  all benchmark-publishing tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
