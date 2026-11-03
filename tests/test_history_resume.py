#!/usr/bin/env python3
"""
A resume supersedes whatever the abandoned branch wrote.

runs/expert_history.jsonl is append-only and its character counter is not
monotonic across a restart: reading on from a checkpoint leaves rows
describing a model that no longer exists, sitting after the point being
resumed from. Every reader then has to cope with it, and they did not -
the dashboard drew a seam at the join, and window_use reached back to the
wrong row and silently widened its window to the whole run.

So the run drops them at startup, once, when it knows where it is resuming.
The alternative was cleaning the file by hand after every abandoned
experiment, which is what happened twice before this existed.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train                                                  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def write(path, chars):
    with open(path, "w") as f:
        for c in chars:
            f.write(json.dumps({"chars": c, "step": int(c // 2048)}) + "\n")


def read(path):
    return [json.loads(l)["chars"] for l in open(path)]


fd, p = tempfile.mkstemp(suffix=".jsonl")
os.close(fd)
try:
    # a run that reached 520M, then resumed from a 500M checkpoint
    write(p, [100e6, 300e6, 500e6, 510e6, 520e6])
    train._truncate_history(p, 500e6)
    check("rows past the resume point are dropped",
          read(p) == [100e6, 300e6, 500e6], f"{[c/1e6 for c in read(p)]}")

    # the row AT the resume point stays - it is the checkpoint's own row
    write(p, [100e6, 500e6])
    train._truncate_history(p, 500e6)
    check("the row at the resume point is kept", read(p) == [100e6, 500e6])

    # nothing to do on a clean log, and the file is not rewritten
    write(p, [100e6, 200e6])
    before = os.path.getmtime(p)
    train._truncate_history(p, 500e6)
    check("a clean log is left alone", read(p) == [100e6, 200e6]
          and os.path.getmtime(p) == before)

    # a fresh run has read nothing and must not wipe the file it inherits
    write(p, [100e6, 200e6])
    train._truncate_history(p, 0)
    check("a run resuming from zero does not truncate",
          read(p) == [100e6, 200e6], "0 means 'no checkpoint', not 'drop all'")

    # and it never raises, whatever it is handed
    train._truncate_history(p + ".missing", 100e6)
    with open(p, "w") as f:
        f.write("not json\n" + json.dumps({"chars": 9e8}) + "\n")
    train._truncate_history(p, 100e6)
    check("unparseable lines are kept rather than dropped",
          any("not json" in l for l in open(p)),
          "a reader can skip them; this must not delete data it cannot read")
finally:
    os.unlink(p)

print("\n  all history-resume tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
