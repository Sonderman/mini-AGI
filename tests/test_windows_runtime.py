#!/usr/bin/env python3
"""
What a run on Windows needs from the files it writes and the children it starts.

  the logs are UTF-8 whatever the locale says: a sample holds whatever bytes
  the model wrote, and Windows opens text in cp1252 by default, where most of
  them cannot be written - one such sample stopped a run mid-training. The
  session banner and the sample log replace what cannot be encoded rather
  than raise

  a save replaces each file atomically by renaming over it, and on Windows a
  rename fails while any reader holds the old file - the plotters read
  manifest.json. There it is tried again with a doubling wait for 6.35 s and
  then raised; a different error is raised at once; elsewhere one attempt is
  all there is, as before. Every rename a save makes goes through this

  (the plotters are started without a shell: tests/test_plot_hook.py)
"""
import ast
import builtins
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T                                              # noqa: E402
from minagi import store                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


# ---- the sample log under a cp1252 locale ----------------------------------
real_open = builtins.open


def cp1252_open(path, mode="r", *a, **k):
    """open as Windows does: text in cp1252 unless told otherwise."""
    if "b" not in mode:
        k.setdefault("encoding", "cp1252")
    return real_open(path, mode, *a, **k)


tmp = tempfile.mkdtemp(prefix="minagi-windows-")
log = os.path.join(tmp, "samples.txt")
saved_plots = dict(T._PLOTS)
T._PLOTS["on"] = False                    # no plotters for a test's log
T.open = cp1252_open                      # train.py's open, as on Windows
try:
    T.write_samples(log, 1, 10, 1,
                    [("text", "prompt", [("raw", "\u6f22\U0001f600 \ufffd\ud800")])],
                    val=1.0, level=0.2)
    wrote = True
except UnicodeEncodeError as e:
    wrote = str(e)
finally:
    del T.open
    T._PLOTS.clear()
    T._PLOTS.update(saved_plots)
check("a sample of characters cp1252 does not have is written",
      wrote is True, "" if wrote is True else wrote)
if wrote is True:
    text = real_open(log, encoding="utf-8").read()
    check("...as UTF-8, with what cannot be encoded at all replaced",
          "\u6f22\U0001f600 \ufffd?" in text)
    check("...and the rest of the entry is there", "level +0.2000" in text)

# every file train.py appends to says its encoding
tree = ast.parse(real_open(os.path.join(ROOT, "train.py"), encoding="utf-8").read())
writers = [n for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
           and n.func.id == "open" and len(n.args) > 1
           and isinstance(n.args[1], ast.Constant) and n.args[1].value == "a"]
named = {n.lineno: {k.arg: ast.literal_eval(k.value) for k in n.keywords
                    if k.arg in ("encoding", "errors")} for n in writers}
check(f"all {len(writers)} files train.py appends to are opened as UTF-8",
      writers and all(v.get("encoding") == "utf-8" for v in named.values()),
      str(named))

# ---- the rename a save makes ------------------------------------------------
real_replace, real_sleep = os.replace, store.time.sleep


class Script:
    """os.replace that raises what it is given, in turn, then succeeds."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), []

    def __call__(self, src, dst):
        self.calls.append((src, dst))
        if self.errors:
            e = self.errors.pop(0)
            if e is not None:
                raise e


waits = []
store.time.sleep = waits.append
try:
    os.replace = s = Script(PermissionError(), PermissionError(), None)
    store._replace("a.tmp", "a", attempts=8)
    check("a reader that lets go: tried again until the rename goes through",
          s.calls == [("a.tmp", "a")] * 3 and waits == [0.05, 0.1],
          f"{len(s.calls)} calls, waits {waits}")

    waits.clear()
    held = PermissionError("still open")
    os.replace = s = Script(*[held] * 9)
    try:
        store._replace("a.tmp", "a", attempts=8)
        raised = None
    except PermissionError as e:
        raised = e
    check("a reader that does not: 8 attempts, 6.35 s of waiting, then its "
          "own error", raised is held and len(s.calls) == 8
          and abs(sum(waits) - 6.35) < 1e-9, f"{len(s.calls)} calls")

    waits.clear()
    os.replace = s = Script(FileNotFoundError())
    try:
        store._replace("a.tmp", "a", attempts=8)
        raised = None
    except FileNotFoundError as e:
        raised = e
    check("any other error is raised at once",
          raised is not None and len(s.calls) == 1 and not waits)

    waits.clear()
    os.replace = s = Script(PermissionError())
    try:
        store._replace("a.tmp", "a", attempts=1)
        raised = None
    except PermissionError as e:
        raised = e
    check("one attempt raises the first error, with no wait",
          raised is not None and len(s.calls) == 1 and not waits)
finally:
    os.replace, store.time.sleep = real_replace, real_sleep

import inspect                                                 # noqa: E402
want = 8 if os.name == "nt" else 1
check(f"on this system ({os.name}) a save makes {want} attempt(s)",
      inspect.signature(store._replace).parameters["attempts"].default == want)

src = real_open(os.path.join(ROOT, "minagi", "store.py"), encoding="utf-8").read()
mod = ast.parse(src)
fns = {n.name: n for n in mod.body if isinstance(n, ast.FunctionDef)}


def calls(fn):
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            out.append(f.id if isinstance(f, ast.Name)
                       else ast.unparse(f) if isinstance(f, ast.Attribute) else "")
    return out


check("every rename a save makes goes through it",
      all("_replace" in calls(fns[f]) and "os.replace" not in calls(fns[f])
          for f in ("save", "_save_paged", "_savez")))

# the real thing, end to end, on this system
d = tempfile.mkdtemp(prefix="minagi-replace-")
real_open(os.path.join(d, "x.tmp"), "w").write("new")
real_open(os.path.join(d, "x"), "w").write("old")
store._replace(os.path.join(d, "x.tmp"), os.path.join(d, "x"))
check("a real rename still replaces the file",
      real_open(os.path.join(d, "x")).read() == "new"
      and not os.path.exists(os.path.join(d, "x.tmp")))

print("\n  " + ("all Windows runtime tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
