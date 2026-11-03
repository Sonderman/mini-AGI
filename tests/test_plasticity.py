#!/usr/bin/env python3
"""
The learning-rate controller, against the numbers this project produces.

It eases the rate at every evaluation unless held-out is clearly improving,
by a fraction of a percent a step, and doubles it only for a confirmed regime
change. What these tests hold it to:

  the shape of a nudge: worse evidence eases harder, never more than
  NUDGE_DOWN; up is slower than down; the response is continuous; and no
  gains could move it more than 2% at once
  a flat run is eased down visibly but gently, a slope it can see is not
  throttled like a plateau, a decline that stops is noticed within the fast
  fit's window, and a negligible but significant slope buys nothing
  a reading that does not hold, an evaluation with nothing read before it,
  and the step a restart makes are not evidence - and a restart does not
  erase a learning curve's flattening either
  the floor holds, and a rate below a hundredth prints as itself; state
  survives a restart; a state with no fits is seeded
  from the sample log, the full evaluation at a session's end left out
  replayed through the real log: every spike it had is dropped, no move is
  larger than 2%, the rate holds while the run improves fast and comes down
  across the plateau - by no more than 14% in any six hours
"""
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minagi.plasticity import Plasticity          # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


SIG = 0.0012          # one check's spread on the live run
SE = 0.0285           # the standard error the evaluator reports


def run(series, scale=0.3, p=None, se=SE):
    """Feed held-out readings; return the controller and every rate seen."""
    p = p or Plasticity()
    if p.scale == 1.0:
        p.scale = scale
    rates = [p.scale]
    for v in series:
        p.observe(v, se)
        rates.append(p.scale)
    return p, rates


def noisy(n, slope=0.0, base=0.63, seed=1, sig=SIG):
    r = random.Random(seed)
    return [base + slope * i + r.gauss(0, sig) for i in range(n)]


def fits(p):
    return (dict(p.S), dict(p.F), p.i)


# ---------------------------------------------------------------- 1. warmup
p = Plasticity()
f0 = p.factor()
for _ in range(Plasticity.WARMUP):
    p.tick()
check("warms up from below and reaches full rate",
      f0 < 0.05 and abs(p.factor() - 1.0) < 1e-9, f"start {f0:.3f}")

# ------------------------------------------- 2. a drift, never a staircase
rng = random.Random(11)
p = Plasticity()
seen = []
for i in range(80):
    p.observe(1.40 + rng.gauss(0, 0.040), 0.040)
    seen.append(p.scale)
moved = sum(1 for a, b in zip(seen[20:], seen[21:]) if a != b)
check("the rate changes at nearly every evaluation",
      moved >= 55, f"{moved} of 59 evaluations moved it")

# ------------------------------------------------- 3. the shape of a nudge
n = Plasticity.nudge
check("getting worse eases harder than merely not improving",
      n(-3.0) < n(0.0) < 1.0, f"x{n(-3.0):.5f} vs x{n(0.0):.5f} per evaluation")
check("...the response is continuous, not three steps",
      len({round(n(t), 6) for t in (-3, -1, 0, 1, 2, 3)}) == 6)
check("...in proportion to how bad it is, and never by more than NUDGE_DOWN",
      n(-12.0) < n(-4.0) < n(-2.0) < n(-1.0) < 1.0
      and n(-50.0) >= math.exp(-Plasticity.NUDGE_DOWN) - 1e-12,
      f"-1 -> x{n(-1.0):.5f}   -4 -> x{n(-4.0):.5f}   -12 -> x{n(-12.0):.5f}")
old = type("Old", (Plasticity,), {"NUDGE": 0.005, "NUDGE_DOWN": 0.025})
check("...and whatever the gains, no single nudge moves it more than 2%",
      old.nudge(-12.0) == 1 - Plasticity.MAX_MOVE and old.nudge(-1.0) > 1 - Plasticity.MAX_MOVE,
      f"the 19 September gains at a verdict of -12: "
      f"x{math.exp(0.025 * math.tanh(-14.2 / Plasticity.T_W_DOWN)):.4f}, held to x{old.nudge(-12.0):.2f}")
check("...it comes down faster than it goes up, and up is bounded by NUDGE",
      Plasticity.NUDGE_DOWN > Plasticity.NUDGE
      and abs(n(50.0) - math.exp(Plasticity.NUDGE)) < 1e-12,
      f"most up x{n(50.0):.5f}, most down x{n(-50.0):.5f}")

# ------------------------------------------------------- 4. a flat run isn't
# SIGMA is the scatter between checkpoints of an earlier era, the model
# genuinely moving rather than the measurement. The controller is scale-free,
# so the same holds at today's 0.0012.
SIGMA = 0.040
rng = random.Random(7)
p = Plasticity()
for i in range(256):
    p.observe(1.40 + rng.gauss(0, SIGMA), SIGMA)      # dead flat
six = p.scale ** (36 / 256)
check("a flat run eases the rate down, visibly but gently",
      0.85 < six < 0.99, f"x{p.scale:.3f} after 256 evaluations: x{six:.3f} per 36, "
      f"six hours of ten-minute rounds")

# ------------------------------------ 5. a slowly-improving run keeps its rate
# T_MID sits above the middle on purpose, so even a real slope decays a
# little. What must not happen is the slope going unrecognised: the median over
# twenty runs, since one run's noise moves the answer by a factor of two.
def after_192(slope, seed):
    r = random.Random(seed)
    p = Plasticity()
    for i in range(192):
        p.observe(1.60 + slope * i + r.gauss(0, SIGMA), SIGMA)
    return p.scale, p._trend()[0]


slow = sorted(after_192(-0.00144, s) for s in range(20))
flat = sorted(after_192(0.0, s)[0] for s in range(20))
improving, t_end = slow[10]
check("a slope the controller can see is not throttled like a plateau",
      improving > flat[10] and t_end > 2.0,
      f"improving x{improving:.3f} (t {t_end:+.2f}) against flat x{flat[10]:.3f}")
each_slow = (improving ** (1 / 192) - 1) * 100
each_flat = (flat[10] ** (1 / 192) - 1) * 100
check("...and is eased at under a third of a plateau's pace",
      abs(each_slow) < abs(each_flat) / 3,
      f"{each_slow:+.3f}% an evaluation against {each_flat:+.3f}%")

# ------------------------- 6. improvement that STOPS is noticed while recent
rng = random.Random(7)
p = Plasticity()
for i in range(40):                       # a genuine decline, at the real rate
    p.observe(1.20 - 0.002 * i + rng.gauss(0, 0.02), 0.10)
peak = p.scale
rose = []
for i in range(30):                       # ...which then stops dead
    p.observe(1.20 - 0.002 * 39 + rng.gauss(0, 0.02), 0.10)
    rose.append(p.scale)
first_fall = next((i for i, s in enumerate(rose) if s < peak), None)
check("a decline that stops is noticed within the fast fit's window",
      first_fall is not None and first_fall <= 15 and rose[-1] < peak,
      f"turned down {first_fall} evaluations after the loss went flat, "
      f"x{peak:.3f} -> x{rose[-1]:.3f}")

# -------------------- 7. significant but negligible must not buy a raise
p = Plasticity()
rng = random.Random(11)
for i in range(200):
    p.observe(1.20 - 0.00005 * i + rng.gauss(0, 0.02), 0.10)
t_tiny, _, e_tiny = p._verdict()
check("a real but negligible slope does not buy a higher rate",
      t_tiny < Plasticity.T_MID and p.scale < 1.0,
      f"t {t_tiny:+.2f}, effect {e_tiny:+.4f} per evaluation, rate x{p.scale:.3f}")

# ------------------------------- 8. the effect size does not grow with n
p = Plasticity()
rng = random.Random(13)
got = {}
for i in range(400):
    p.observe(1.20 - 0.0002 * i + rng.gauss(0, 0.02), 0.10)
    if i in (100, 399):
        got[i] = p._fit(p.S)
t_grew = got[399][0] / max(got[100][0], 1e-9)
e_grew = got[399][2] / max(got[100][2], 1e-9)
check("the effect size is scale-free where t is not",
      e_grew < 1.5 and t_grew > e_grew,
      f"same slope, 300 more evaluations: t x{t_grew:.2f}, effect x{e_grew:.2f}")

# ------------------------------------------------- 9. the cap, in a real run
p, rates = run(noisy(100) + noisy(100, +0.0004, base=0.63, seed=2))
steps = [b / a for a, b in zip(rates, rates[1:])]
hard = math.exp(-Plasticity.NUDGE_DOWN)
check("a run getting worse fast is eased hardest, and never faster than NUDGE_DOWN",
      min(steps) >= hard - 1e-12 and sum(1 for x in steps if x < hard ** 0.9) >= 20,
      f"steepest x{min(steps):.5f}, {sum(1 for x in steps if x < hard ** 0.9)} "
      f"evaluations within a tenth of x{hard:.5f}")

# ------------------------------------------------------------ 10. new material
p = Plasticity()
p.scale = 0.2
for v in [1.20, 1.19, 1.19, 1.18]:
    p.observe(v, 0.098)
before = p.scale
p.observe(2.44, 0.098)                           # the collapse, as it happened
mid = p.scale
p.observe(2.28, 0.098)                           # still up there -> confirmed
check("a confirmed regime change doubles the rate, held back until confirmed",
      mid == before and abs(p.scale / before - Plasticity.UP) < 1e-12
      and p.events[-1]["kind"] == "regime",
      f"x{before:.2f} -> x{mid:.2f} -> x{p.scale:.2f}")

# ------------------------------------ 11. a reading that does not hold
base = noisy(160)
a, ra = run(base)
for name, bad in (("a broken evaluation", 0.95), ("a spike of 10 sd, far under the regime bar", 0.63 + 10 * SIG),
                  ("a dip of 10 sd", 0.63 - 10 * SIG)):
    b, rb = run(base[:80] + [bad] + base[80:])
    check(f"{name} changes nothing at all",
          ra[-1] == rb[-1] and fits(a) == fits(b),
          f"x{ra[-1]:.4f} without it, x{rb[-1]:.4f} with it")
step = base[:80] + [x + 12 * SIG for x in base[80:]]
p, rates = run(step)
q, _ = run(base[:80])
check("...but a level that really moves is held one evaluation, then both count",
      rates[81] == rates[80] and p.i == 160 and rates[82] != rates[81],
      f"held at x{rates[81]:.4f}, then x{rates[82]:.4f}; all 160 readings in the fits")

# -------------------------------------------------------------- 12. no reading
p = Plasticity()
p.scale = 0.3
for v in noisy(60):
    for _ in range(400):
        p.tick()
    p.observe(v, SE)
snap = (p.scale, fits(p))
for _ in range(30):
    p.observe(0.70, SE)                       # nothing read since
check("an evaluation with no reading since the last is not evidence",
      (p.scale, fits(p)) == snap)


# ------------------------------------------------------------------ 13. restart
# What a restart does to the verdict, the number the rate moves on: a median
# over the checks after it, then over many runs.
def verdicts(p, series):
    out = []
    for v in series:
        p.observe(v, SE)
        out.append(p.last_t)
    return out


def median(v):
    return sorted(v)[len(v) // 2]


# A run improving clearly, then a step of +0.006 - a setting changed. Carried
# as a trend it reads as deterioration; across a restart it is nothing at all.
def after_step(seed, restart, size):
    p, _ = run(noisy(200, -0.00005, seed=seed))
    if restart:
        p = Plasticity.restore(p.state())
    tail = noisy(30, -0.00005, base=0.63 - 0.00005 * 200 + size, seed=seed + 1000)
    return median(verdicts(p, tail)[Plasticity.RELEVEL:20])


v = {(r, z): median([after_step(s, r, z) for s in range(40)])
     for r in (False, True) for z in (0.0, 0.006)}
check("a step at a restart is not read as harm",
      v[False, 0.006] < v[False, 0.0] - 5 and abs(v[True, 0.006] - v[True, 0.0]) < 0.5,
      f"the verdict after a +0.006 step: {v[False, 0.006]:+.1f} carried as a trend, "
      f"{v[False, 0.0]:+.1f} without it; {v[True, 0.006]:+.1f} across a restart, "
      f"{v[True, 0.0]:+.1f} without it")


# A restart with no step should change nothing. The step it measures carries
# the noise of a median of RELEVEL readings, a run restarted often carries that
# noise into its fits again and again, and a step read off the slow fit's line
# would take a flattening curve's lag behind it for a step every time.
def curve(n, seed):
    r = random.Random(seed)
    return [0.62 + 0.04 * math.exp(-i / 150) + r.gauss(0, SIG) for i in range(n)]


def through(series, every):
    p = Plasticity()
    p.scale = 0.3
    out = []
    for k, x in enumerate(series):
        if every and k and k % every == 0:
            p = Plasticity.restore(p.state())
        p.observe(x, SE)
        out.append(p.last_t)
    return out


shift = median([median([a - b for a, b in zip(through(curve(400, s), 25)[100:],
                                              through(curve(400, s), 0)[100:])])
                for s in range(30)])
check("...a restart with no step leaves the verdict where it was, however often it comes",
      abs(shift) < 0.25,
      f"a flattening run restarted every 25 checks reads {shift:+.2f} from the same run never restarted")


# A run that turned worse in the hour before it was stopped - the usual reason
# to stop one. That is not the restart's step, and must still count after it:
# the slow fit's line lags the turn, so a step read off it would erase it.
def turned(seed, restart):
    r = random.Random(seed)
    top = 0.63 - 0.00003 * 150
    a = [0.63 - 0.00003 * i + r.gauss(0, SIG) for i in range(150)]
    a += [top + 0.004 * (k + 1) / 8 + r.gauss(0, SIG) for k in range(8)]
    p, _ = run(a)
    if restart:
        p = Plasticity.restore(p.state())
    after = [top + 0.004 + r.gauss(0, SIG) for _ in range(40)]
    return median(verdicts(p, after)[Plasticity.RELEVEL:25])


kept = median([turned(s, False) for s in range(30)])
rest = median([turned(s, True) for s in range(30)])
check("...and a run that turned worse before it was stopped still reads as worse",
      rest < 0 and abs(rest - kept) < 1.5,
      f"the verdict after it: {rest:+.2f} across a restart, {kept:+.2f} with none")

p, _ = run(noisy(100))
q = Plasticity.restore(p.state())
level = q._line(q.S, q.i)[0]
run([0.63 + 15 * SIG] + [0.63] * Plasticity.RELEVEL, p=q)
check("...a spike as the first reading after a restart does not set the level",
      abs(q._line(q.S, q.i)[0] - 0.63) < 3 * SIG,
      f"line at {level:.4f} before, {q._line(q.S, q.i)[0]:.4f} after "
      f"a {0.63 + 15 * SIG:.4f} among 0.6300s")

p, _ = run(noisy(100))
q = Plasticity.restore(p.state())
before = q.scale
run([1.0] * 5, p=q)
check("...and a step the size of a regime change is left to the regime rule",
      abs(q.scale / before - Plasticity.UP) < 0.05
      and q.events[-1]["kind"] == "regime", f"x{before:.4f} -> x{q.scale:.4f}")

# -------------------------------------------------------------------- 14. floor
p, rates = run(noisy(6000))
check("the rate never goes below the floor of x0.001, and a long flat run reaches it",
      Plasticity.FLOOR == 0.001 and min(rates) >= Plasticity.FLOOR
      and abs(rates[-1] - Plasticity.FLOOR) < 1e-12, f"x{rates[-1]:.5f} after 6000 flat checks")
below = next((k for k, r in enumerate(rates) if r < 0.01), None)
check("...passing x0.01 on the way, where the floor once was",
      below is not None and below < len(rates) - 1, f"below x0.01 after {below} checks")
check("a scale below a hundredth is printed as itself, to three figures",
      [Plasticity.shown(v) for v in (0.5, 0.0122, 0.00951, 0.001)]
      == ["0.500", "0.0122", "0.00951", "0.00100"],
      " ".join(Plasticity.shown(v) for v in (0.5, 0.0122, 0.00951, 0.001)))

# ------------------------------------------------------------ 15. round-trips
p = Plasticity()
for v in noisy(150):
    for _ in range(400):
        p.tick()
    p.observe(v, SE)
q = Plasticity.restore(p.state())
check("state survives a restart",
      q.scale == p.scale and q.prev == p.prev and fits(q) == fits(p)
      and q.step == p.step
      and q.seen_at == p.seen_at and list(q.se_hist) == list(p.se_hist),
      "rate, every running sum, last reading, step, standard errors")

# levels from the controller that kept them; the sample log has the readings
old = {"scale": 0.0601, "step": 857744, "prev": 0.6301, "levels": [0.628] * 256,
       "cool": 35, "seen_at": 857363, "events": [{"at": 1, "kind": "plateau"}]}
q = Plasticity.restore(old)
had = q.has_fits()
log = ("\n".join(f"step 1\ntrain loss 0.5   lr 2.00e-05   evidence t +0.00 over 256"
                 f" (effect +0.0000)   rate x0.060\nheld-out loss {v:.4f} +/-0.0285 nats"
                 for v in noisy(150, seed=4))
       + "\nstep 2\ntrain loss 0.5   lr 2.00e-05\nheld-out loss 0.6150 +/-0.0140 nats"
       + "\n\n# session started 2026-10-10 18:10  at step 1, 1745.8M of 7,880M\n"
       + "\n".join(f"step 3\ntrain loss 0.5   lr 2.00e-05   evidence t +0.00 over 256"
                   f" (effect +0.0000)   rate x0.060\nheld-out loss {v:.4f} +/-0.0285 nats"
                   for v in noisy(40, seed=5)))
got = Plasticity.log_readings(log)
check("the sample log gives every round's reading and each session start, not the full evaluation",
      len(got) == 191 and got[150] is None and 0.6150 not in got,
      f"{sum(g is not None for g in got)} readings, {got.count(None)} session start")
q.seed(got)
check("...and a state with no fits keeps its rate and step and is seeded from it",
      not had and q.has_fits() and q.scale == 0.0601 and q.step == 857744
      and len(q.events) == 1 and q._resumed and q._fit(q.S)[1] > Plasticity.MIN_EFF,
      f"{q._fit(q.S)[1]:.0f} effective readings, rate x{q.scale:.4f}")


# ------------------------------------------------- 16. the real log, replayed
def real_log(path="runs/samples.txt"):
    """[(step, chars, val, se) | None at a session start], rounds only."""
    seq, step, chars, train = [], None, None, ""
    for ln in open(path, errors="ignore"):
        if ln.startswith("# session started"):
            seq.append(None)
        elif ln.startswith("step "):
            m = re.match(r"step ([\d,]+)\s+([\d.]+)M of", ln)
            if m:
                step, chars = int(m.group(1).replace(",", "")), float(m.group(2))
        elif ln.startswith("train loss"):
            train = ln
        elif ln.startswith("held-out loss"):
            m = re.match(r"held-out loss ([\d.]+)(?: \+/-([\d.]+))?", ln)
            if "evidence" in train and chars is not None:
                seq.append((step, chars, float(m.group(1)), float(m.group(2) or 0)))
            train = ""
    return seq


try:
    seq = real_log()
except Exception as e:                                   # no log available
    seq = []
    print(f"  (skipping real-log test: {e})")

if sum(1 for x in seq if x) > 2000:
    p = Plasticity()
    p.scale = 0.543                          # where the run stood at 500M
    begun, seen, notes = False, [], []
    for x in seq:
        if x is None:
            if begun:
                p = Plasticity.restore(p.state())
            continue
        step, chars, val, se = x
        if chars < 500:
            continue
        begun = True
        p.step = step                        # the log's own optimiser steps
        before = p.scale
        note = p.observe(val, se)
        if note:
            notes += [(chars, n) for n in note.split("; ")]
        seen.append((chars, before, p.scale))

    dropped = {round(float(re.match(r"held-out ([\d.]+)", n).group(1)), 4)
               for c, n in notes if "did not hold" in n}
    check("replayed from 500M, the broken evaluation and the plateau's spikes never count",
          {1.126, 0.6434, 0.6459, 0.6510} <= dropped,
          f"{len(dropped)} readings dropped, among them 1.1260, 0.6434, 0.6459 and 0.6510")
    moves = [b / a for c, a, b in seen]
    check("...no move is larger than 2%",
          min(moves) >= 1 - Plasticity.MAX_MOVE - 1e-12
          and max(moves) <= 1 + Plasticity.MAX_MOVE + 1e-12,
          f"steepest x{min(moves):.4f} down, x{max(moves):.4f} up")

    def at(c):
        return [s for ch, a, s in seen if ch <= c][-1]

    check("...the rate holds while the run improves fast",
          0.5 < at(1000) / at(600) < 2.0, f"x{at(600):.3f} at 600M, x{at(1000):.3f} at 1,000M")
    check("...comes down across the plateau after 1,342M",
          at(1756) < at(1342) * 0.6, f"x{at(1342):.3f} at 1,342M, x{at(1756):.3f} at the end")
    six = min(b[2] / a[1] for a, b in zip(seen, seen[36:]))
    check("...and gently: no 36 checks - six hours - take more than 14% off it",
          six >= 0.86,
          f"the most any 36 checks took off was {1 - six:.1%}")

# --------------------------------------------- 17. its only inputs, and why
import inspect                                            # noqa: E402
sig = list(inspect.signature(Plasticity.observe).parameters)
check("the controller's only inputs are a score and its error",
      sig == ["self", "val", "se"], f"observe{tuple(sig)}")

print("\n  all plasticity tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
