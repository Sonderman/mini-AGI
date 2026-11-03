#!/usr/bin/env python3
"""
Fixed weights for the subjects' shares - the override of the automatic
schedule (tests/test_lane_schedule.py has the schedule).

What these tests hold it to:
  with nothing set there are no fixed weights, and the schedule decides
  "subject=weight,..." and a mapping read alike; unlisted subjects weigh 1
  a name that is not a subject, a weight outside (0, 1] and a pair that is
  not subject=weight stop the run before it reads, saying which
  fixed weights become the shares, and the shares never move
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as T                                           # noqa: E402
from minagi.lanes import LaneSchedule                       # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


NAMES = ["arithmetic", "chat", "self-knowledge", "wikipedia"]

check("with nothing set there are no fixed weights",
      T._lane_weights("", NAMES) == {} and T._lane_weights(None, NAMES) == {}
      and T._lane_weights({}, NAMES) == {})
w = T._lane_weights("self-knowledge=0.25, arithmetic=0.5", NAMES)
check("pairs read as weights",
      w == {"self-knowledge": 0.25, "arithmetic": 0.5}, w)
check("...and a mapping, as config.yaml gives it",
      T._lane_weights({"self-knowledge": 0.25}, NAMES) == {"self-knowledge": 0.25})


def refused(spec):
    try:
        T._lane_weights(spec, NAMES)
    except SystemExit as x:
        return str(x)
    return None


check("a name that is not a subject stops the run, naming the subjects",
      "not a subject" in (refused("stories=0.5") or "")
      and "wikipedia" in (refused("stories=0.5") or ""))
check("...and so does a weight outside (0, 1]",
      "above 0 and at most 1" in (refused("chat=0") or "")
      and "above 0 and at most 1" in (refused("chat=1.5") or ""))
check("...and a pair that is not subject=weight",
      "not subject=weight" in (refused("chat") or ""))

s = LaneSchedule(NAMES, {n: 1e6 for n in NAMES}, ["chat", "wikipedia"],
                 mode="fixed", weights=w)
check("fixed weights become the shares - unlisted subjects at 1",
      abs(s.shares["self-knowledge"] - 0.25 / 2.75) < 1e-12
      and abs(s.shares["chat"] - 1 / 2.75) < 1e-12, s.shares)
s.observe(1e8, {"chat": 0.5, "wikipedia": 0.9})
check("...and they never move", s.step() == "" and not s.fits
      and abs(s.shares["arithmetic"] - 0.5 / 2.75) < 1e-12)
picks = [s.pick() for _ in range(275)]
check("...and visits follow them",
      picks.count("self-knowledge") == 25 and picks.count("chat") == 100,
      {n: picks.count(n) for n in NAMES})

print("\n  all lane-weight tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
