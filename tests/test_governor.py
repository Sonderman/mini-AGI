#!/usr/bin/env python3
"""
The learning rate's controller, run the same way by everything that learns
(minagi/governor.py): training, a conversation, and the benchmarks' learning.

What these tests hold it to:
  a session starts by reading the larger evaluation and the checks' blocks,
  which set the level; a check reads the checks' blocks at that level, the
  controller observes their mean and every expert's own rate relaxes
  a check the controller would not count - too few steps since the last
  reading it took - is not read, and changes nothing in it, while the
  experts' rates still relax, as training's round relaxes them
  checks come every `every` minutes of learning
  training takes its start, its check and its response to the check through
  the governor and nowhere else, and records how it checks in the manifest
  a learner outside training does the same: the session's evaluation when it
  is set up, a check when one is due after a step, the model left in the
  mode it was in - on a real model, with a real held-out folder
  with no held-out folder the controller cannot read, the learner says so,
  and the rate stays where the run left it
"""
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as T                                           # noqa: E402
from minagi.create import create                            # noqa: E402
from minagi.governor import DEFAULTS, Governor, steps_of    # noqa: E402
from minagi.live import LiveLearner                         # noqa: E402
from minagi.plasticity import Plasticity                    # noqa: E402
from minagi.tokenizer import ByteTokenizer                  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


class Ev:
    """An evaluator that says what it was asked to read."""
    per = 2

    def __init__(self):
        self.asked = []

    def run(self, steps, spread=False):
        self.asked.append(steps)
        base = 1.0 if steps >= 8 else 1.1           # the checks read high
        return {"a": base, "b": base + 0.2, "stderr": 0.01}


plast, ev = Plasticity(), Ev()
g = Governor(plast, ev, chunk=16, every=10, check_chars=64, eval_chars=256)
v, se, c = g.start()
check("a session starts with the larger evaluation, then the checks' "
      "blocks, which set the level",
      ev.asked == [16, 4] and se == 0.01 and set(g.level) == {"a", "b"}
      and all(abs(x + 0.1) < 1e-12 for x in g.level.values()),
      f"{ev.asked} {g.level}")
d, mean, se_ = g.reading()
check("...a check reads the checks' blocks at that level",
      ev.asked[-1] == 4 and abs(d["a"] - 1.0) < 1e-12
      and abs(d["b"] - 1.2) < 1e-12 and abs(mean - 1.1) < 1e-12
      and se_ == 0.01)
seen = []
plast.observe = lambda v_, s_: seen.append((v_, s_)) or "moved"
note = g.govern(mean, se_)
check("...and the controller observes the mean",
      seen == [(mean, se_)] and note == "moved" and g.checks == 1)

# the controller's own counting rule decides what is skipped
p2, ev2 = Plasticity(), Ev()
g2 = Governor(p2, ev2, chunk=16, every=10, check_chars=64, eval_chars=256)
p2.step, p2.seen_at = 100, 90                     # 10 steps since a reading
before = json.dumps(p2.state(), sort_keys=True)
got = g2.check()
check("a check the controller would not count is not read and changes "
      "nothing in it",
      got is None and ev2.asked == []
      and json.dumps(p2.state(), sort_keys=True) == before)
p2.step = 90 + Plasticity.MIN_STEPS
got = g2.check()
check("...and one it would count is read and governs",
      got is not None and ev2.asked == [4] and p2.seen_at == p2.step)
g3 = Governor(Plasticity(), Ev(), chunk=16, every=10)
g3.last = time.time() - 9.9 * 60
due_early = g3.due()
g3.last = time.time() - 10.1 * 60
check("checks come every `every` minutes of learning",
      not due_early and g3.due() and steps_of(60 * 512, 2048) == 15
      and DEFAULTS == {"every": 10.0, "held_out": "data/val",
                       "check_chars": 60 * 512, "eval_chars": 240 * 512})

src = inspect.getsource(T.cmd_read)
check("training takes its start, its check and the response to it through "
      "the governor, and nowhere else",
      "gov.start()" in src and "gov.reading()" in src
      and "moved_lr = gov.govern(v_, se_)" in src
      and "plast.observe(" not in src
      and "gov.plast = plast" in src)
check("...and records how it checks in the manifest, at both saves",
      len(re.findall(r'"checks": checks_kept,', src)) == 2
      and '"every": float(args.sample_every)' in src)

# a learner outside training, on a real model and a real held-out folder
tmp = tempfile.mkdtemp(prefix="minagi-governor-")
try:
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=8, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=64, max_steps=2,
           top_k=2, depth=1)
    val = os.path.join(tmp, "val")
    for subj, text in (("prose", "the cat sat on the mat. " * 40),
                       ("code", "def f(x):\n    return x + 1\n" * 40)):
        os.makedirs(os.path.join(val, subj))
        with open(os.path.join(val, subj, "a.txt"), "w") as f:
            f.write(text)
    model, cfg, pool5, man = T.build_paged(wdir, "cpu", ram_capacity=4)
    man = dict(man, plasticity={"scale": 0.5, "step": 1000},
               checks={"every": 1e-9, "held_out": val, "check_chars": 64,
                       "eval_chars": 128})
    conf = {"training": {"lr": 1e-3, "trunk_lr_mult": 0.1, "chunk": 16}}
    model.eval()
    ln = LiveLearner.for_run(model, manifest=man, save_every=0, config=conf)
    check("a learner outside training reads the session's evaluation when it "
          "is set up, as the run's manifest says",
          ln.gov is not None and ln.gov.level != {}
          and ln.gov.every == 1e-9 and ln.gov.check_steps == 4
          and not model.training)
    ln.feed("the dog ran in the park. " * 60, ByteTokenizer())
    gaps = [b["step"] - a["step"] for a, b in zip(ln.checked,
                                                  ln.checked[1:])]
    check("...takes a check when one is due after a step - each one the "
          "controller would count - and governs the rate with it",
          len(ln.checked) >= 2 and ln.plast.seen_at is not None
          and all(gp >= Plasticity.MIN_STEPS for gp in gaps),
          f"{len(ln.log)} steps, checks at "
          f"{[c['step'] for c in ln.checked]}")
    check("...and leaves the model in the mode it was in",
          not model.training)
    gone = LiveLearner.for_run(model, manifest=dict(man, checks={
        "every": 10, "held_out": os.path.join(tmp, "nowhere")}),
        save_every=0, config=conf)
    f0 = gone.factor()
    gone.feed("the dog ran in the park. " * 60, ByteTokenizer())
    check("with no held-out folder the controller cannot read, and the rate "
          "stays where the run left it",
          gone.gov is None and gone.checked == [] and gone.factor() == f0)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all governor tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
