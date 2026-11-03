#!/usr/bin/env python3
"""
Every cadence is configured in CHARACTERS, not in chunks or steps.

A step is `--chunk` characters, so anything counted in steps silently changes
meaning the moment the chunk changes. growth.every was 400 steps: 204,800
characters at chunk 512, and 1.6 million at chunk 4,096, without a line of the
config moving. Characters are what the model reads; steps are an
implementation detail of how often we stop to update.
"""
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


c = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))

# ------------------------------- 1. no cadence is named in chunks or steps
RETIRED = [("model", "context_grow_every"), ("model", "context_every"),
           ("pool", "segment_chunks"), ("pool", "dwell"), ("growth", "every"),
           ("prune", "survival"), ("data", "min_visit_chunks")]
for sec, key in RETIRED:
    check(f"{sec}.{key} is gone", key not in (c.get(sec) or {}),
          "it counted chunks or steps")

PRESENT = [("model", "context_grow_every_chars"), ("model", "context_every_chars"),
           ("growth", "every_chars"), ("prune", "survival_chars"),
           ("data", "passage")]
for sec, key in PRESENT:
    v = (c.get(sec) or {}).get(key)
    check(f"{sec}.{key} is present and in characters", isinstance(v, int) and v > 0,
          f"{v:,}" if isinstance(v, int) else str(v))

# ------------------- 2. the conversion happens once, where chunk is known
src = open(os.path.join(ROOT, "train.py")).read()
check("train.py converts characters to steps in one place",
      src.count("def _in_steps(") == 1)
for name in ("grow_every_steps", "survival_steps", "ctx_every_steps",
             "ctx_grow_every_steps"):
    check(f"{name} is derived from a character count",
          f"{name} = _in_steps(" in src)

# --------------- 3. nothing compares a raw character count against a step
for bad in ("step % args.grow_every", "step % args.context_every",
            "step % args.context_grow_every", "pool.trial = args.prune_survival"):
    check(f"no raw comparison: {bad}", bad not in src,
          "that would read a character count as a step count")

# ------------------------------ 4. the held-out size is characters too
from minagi.governor import steps_of                       # noqa: E402
check("the held-out size is a character count",
      "eval_steps = steps_of(args.eval_chunks, args.chunk)" in src
      and steps_of(30_720, 512) == 60 and steps_of(122_880, 2_048) == 60,
      "60 chunks is 30,720 characters at 512 and 122,880 at 2,048")

# ------------------------- 5. the numbers mean what they meant at chunk 512
#
# These pin the one-time conversion from chunks to characters, so a setting
# cannot silently change meaning when the chunk changes. They are NOT a pin on
# the value itself - a cadence that has since been deliberately retuned is
# listed below with `want=None`, and only its units are checked. Pinning a
# tuning knob forever turns a units test into a nuisance.
for sec, key, want in [("growth", "every_chars", None),   # migrated 400 x 512,
                                                          # retuned since
                       ("model", "context_every_chars", 128 * 512)]:
    got = c[sec][key]
    if want is None:
        # Retuned since the migration, so the value is the author's business.
        # What still matters is that it is a CHARACTER count and not a step
        # count left behind in a field whose name says characters - the exact
        # mistake this file exists to catch. The pre-migration spellings were
        # counts of chunks, so they were tens or hundreds; any character count
        # is at least a chunk, which is thousands.
        check(f"{sec}.{key} is a character count, not a step count",
              isinstance(got, int) and got >= 1_000,
              f"{got:,} characters = {got / 2048:.0f} steps at chunk 2,048")
    else:
        check(f"{sec}.{key} preserves the old cadence at chunk 512",
              got == want, f"{got:,} = {want // 512} x 512")

# ---- 6. a retired spelling must FAIL, not be silently reinterpreted ----
# --eval-chunks 60 meant 60 chunks. Kept as an alias it would now mean 60
# CHARACTERS - one chunk of held-out per domain instead of sixty. An alias that
# keeps the name and changes the unit is worse than no alias, which is the
# whole reason this file exists. Checked against the parser, not the source.
import subprocess                                           # noqa: E402
for retired, correct in [("--eval-chunks", "--eval-chars"),
                         ("--sample-eval-chunks", "--sample-eval-chars"),
                         # retired with the working set, no replacement: the
                         # text's vote admits them
                         ("--segment-chunks", "nothing"),
                         ("--segment-chars", "nothing"),
                         ("--dwell", "nothing"), ("--dwell-chars", "nothing"),
                         ("--margin", "nothing"), ("--explore", "nothing")]:
    r = subprocess.run([sys.executable, "train.py", "read", "data/train",
                        retired, "1"], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    out = (r.stdout + r.stderr).lower()
    refused = r.returncode != 0 and ("unrecognized" in out or "is gone" in out)
    check(f"{retired} is refused, pointing at {correct}", refused,
          (r.stdout + r.stderr).strip().splitlines()[-1][:64] if refused
          else "ACCEPTED - it would be read in the new unit")


# ---- 7. nothing in cmd_read is read before it is assigned --------------
# `pool.trial = survival_steps` sat four hundred lines ABOVE the conversion
# that produced survival_steps; `pool.now` before that; `best_val` read the
# manifest four hundred lines above build_paged. Three ordering bugs of the
# same shape, each costing a minute of model loading before Python noticed.
#
# This checked a hand-written list of names and so missed the third one. It
# now checks every local in the function, which is the only version that can
# catch the fourth. Loads inside a nested def are skipped - those run later,
# so textual order says nothing about them.
import ast as _ast                                          # noqa: E402

_fn = [n for n in _ast.walk(_ast.parse(src))
       if isinstance(n, _ast.FunctionDef) and n.name == "cmd_read"][0]
_params = {a.arg for a in _fn.args.args}


def _named(node, want):
    """(name, runs_later) for every Load/Store under node."""
    out = []

    def walk(n, deferred):
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                          _ast.Lambda)):
            for c in _ast.iter_child_nodes(n):
                walk(c, True)
            return
        if isinstance(n, _ast.Name) and isinstance(n.ctx, want):
            out.append((n.id, deferred))
        for c in _ast.iter_child_nodes(n):
            walk(c, deferred)

    walk(node, False)
    return out


_store, _load = {}, {}
for _i, _st in enumerate(_fn.body):
    for _nm, _ in _named(_st, _ast.Store):
        _store.setdefault(_nm, _i)
    for _nm, _deferred in _named(_st, _ast.Load):
        if not _deferred:
            _load.setdefault(_nm, _i)

_bad = [(nm, _load[nm], si) for nm, si in _store.items()
        if nm not in _params and _load.get(nm) is not None
        and _load[nm] < si]
check(f"every local in cmd_read is assigned before it is read "
      f"({len(_store)} locals)", not _bad,
      "; ".join(f"{nm} read at line {_fn.body[li].lineno}, assigned at "
                f"{_fn.body[si].lineno}" for nm, li, si in _bad))

print("\n  all unit tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
