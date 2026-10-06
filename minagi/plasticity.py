"""
The learning rate, governed by held-out loss rather than by a horizon.

A cosine schedule asserts that the run ends. For a model that reads
continually that assertion is false, and it fails in a specific way: when the
regime changes - a new corpus, a change of shape - a count of characters
already read says the model is finished when it has in fact just started
again. 

TWO RULES, AND THEY POINT IN OPPOSITE DIRECTIONS. That symmetry is the whole
point; a cosine can only ever go down.

  MOVING    the rate is nudged at EVERY evaluation by an amount that varies
            smoothly with the evidence, rather than stepping every eighth one.
            The evidence is an exponentially weighted least-squares fit of
            held-out against evaluation index - no window, so nothing ever
            falls off an edge and produces a jump in the signal that becomes a
            jump in the rate.

            THE EVIDENCE IS AN EFFECT SIZE, NOT A t-STATISTIC. A t-statistic
            is the wrong variable here: se(slope) shrinks as the fit
            accumulates weight, so given enough observations ANY downward drift
            becomes significant. It measures how long the controller has been
            watching, not how much has improved, and pins the rate at the
            ceiling.

            So the deciding number is
                e = -slope / sigma          improvement per evaluation, in
                                            units of the residual scatter
            which is scale-free and does NOT grow as evidence accumulates. The
            significance test is kept as a CAP rather than as the signal, since
            acting on a slope that could be noise is still wrong, so the rate
            moves on min(t, EFFECT * e).

            A second fit at LAM_FAST caps it again. The slow fit remembers ~66
            evaluations, so when improvement stops it goes on reporting the old
            decline for tens of evaluations. The fast fit notices in about 12,
            and min() takes whichever is less impressed.

            WHICH WAY IT MOVES (Plasticity.move). Up while held-out is still
            improving - slowly, in proportion to the evidence, from one
            standard error of improvement to T_MID, and faster above it. It
            holds when held-out is flat. It comes down only when held-out is
            measurably getting WORSE (below -T_HARM), and only once that has
            been seen twice running; a back-off then pauses every increase for
            COOL evaluations. So the rate keeps probing upward until a higher
            one does harm. Each direction asks for evidence that has held: a
            rise is sized by the weakest of the last RISE_RUN verdicts, and
            harm needs the slow fit AND the fast one to say so - the verdict
            that sizes rises is pessimistic on purpose, and on a flat run its
            pessimism alone would read as harm. It used to ease down whenever improvement was merely
            slow, and on a run whose progress depends on its rate that loop
            only closes one way: a lower rate learns more slowly, slower
            learning reads as weaker evidence, weaker evidence lowers the rate.
            The real run followed it from x0.38 to x0.05 while still improving.

  REGIME    held-out jumped by several standard errors and stayed there ->
            step the rate back up. New material, or a change of shape. It has
            to persist to count, so a single noisy evaluation cannot trigger
            it. This is the same detector `tools/plot_progress.py` uses to
            decide where to fit its trend.

TWO THINGS ARE NOT EVIDENCE, and both once drove the rate down by themselves:

  A JUMP THAT DOES NOT HOLD. A suspected regime change is held back from the
            fits until the next evaluation confirms it. Unconfirmed, it was a
            bad measurement and never counts. Once, a single broken evaluation
            (1.126 among 0.73s) sat in the slow fit for dozens of evaluations
            and cut the rate from x0.38 to x0.21.
  NO READING. An evaluation with fewer than MIN_STEPS optimiser steps since the
            last one measures the same model again. The x axis counts
            evaluations, so thirty back-to-back rounds with nothing read read
            as thirty rounds of no progress, and each eased the rate.

Nothing here is a hyperparameter the user has to set. Both thresholds are read
off the measured standard error of the evaluation itself, so the controller
tightens automatically as the evaluation gets less noisy.
"""

import math
from collections import deque


def _sums():
    return dict(w=0.0, w2=0.0, x=0.0, y=0.0, xx=0.0, xy=0.0, yy=0.0)


class Plasticity:
    FLOOR = 0.05        # the rate is never allowed to reach zero
    CEIL = 1.0
    LAM = 0.97          # decay per evaluation; n_eff -> (1+L)/(1-L) = 65.7
    LAM_FAST = 0.85     # the second, shorter fit; n_eff -> 12.3. Not lower:
                        # below MIN_EFF the fast fit never engages at all and
                        # the cap silently stops existing.
    EFFECT = 45.0       # converts e into the same units as t, so the two can
                        # be compared by min(). Calibrated so that typical
                        # real-run evidence sits just BELOW T_MID, making
                        # gentle decay the resting posture.
    MIN_EFF = 12.0      # effective observations before it will act at all
    NUDGE = 0.005       # log-scale gain per evaluation, going UP
    NUDGE_DOWN = 0.025  # ...and coming down. Deliberately larger - see below.
    T_W_DOWN = 6.0      # the width on the way down, over the range the verdict
                        # actually reaches (it runs to about -12)
    T_MID = 2.2         # the verdict at which it neither eases up nor down.
                        # DELIBERATELY ABOVE THE MIDDLE, for two reasons.
                        #
                        # The errors are not symmetric. Overshooting the rate
                        # costs a fraction of a nat and tens of millions of
                        # characters to repair; undershooting only costs time.
                        # So slow-but-real progress reads as slight decay, and
                        # only clearly better progress buys more rate.
                        #
                        # And `e` has a bias to correct. It is slope over
                        # RESIDUAL SCATTER, and that scatter is the model's own
                        # checkpoint-to-checkpoint wobble rather than the
                        # evaluation's error. The wobble falls over a run as
                        # the pool stops churning, while the evaluation error
                        # barely moves - so a run that merely gets QUIETER
                        # starts scoring as a run that is improving. Sitting
                        # the neutral point high is what absorbs that.
                        #
                        # This does not disable raising: real evidence still
                        # clears it, a few evaluations in ten rather than a
                        # third of them.
    T_W = 0.75          # how sharply it responds ABOVE that
    JUMP_SE = 4.0       # standard errors that count as a regime change
    HOLD_SE = 2.0       # ...and it has to still be up here next time
    FLOOR_JUMP = 0.25   # a jump this large is a regime change whatever the noise
    UP = 2.0            # what a confirmed regime change restores
    WARMUP = 100        # optimiser steps, in case the moments are not restored
    PROBE = 0.004       # log-scale rise per evaluation while held-out is still
                        # improving: about x1.8 a day at a round every ten
                        # minutes, slow enough that harm shows before it
                        # compounds
    T_DEAD = 1.0        # ...and only past one standard error of improvement;
                        # below it the rate holds. Noise alone then creeps the
                        # rate up a few percent a day, which harm stops.
    T_HARM = 2.0        # a verdict below -T_HARM is held-out getting worse
    HARM_RUN = 2        # ...and it has to say so twice running to act on: the
                        # real run's dips below it last one or two rounds
    COOL = 12           # evaluations without a rise after a back-off
    RISE_RUN = 4        # a rise is sized by the weakest of the last few
                        # verdicts: the fast fit's reading swings by about
                        # +-3.8 between evaluations, so one good one is noise,
                        # and four in a row is a trend
    MIN_STEPS = 32      # optimiser steps since the last evaluation for a new
                        # one to count as evidence (a round reads ~400)

    def __init__(self, scale=1.0, best=None):
        self.scale = float(scale)
        # Two exponentially weighted least-squares fits of held-out against
        # evaluation index, kept as running sums. There is no window: every
        # past reading still counts, weighted by LAM ** age, so nothing ever
        # falls off an edge. A hard window of length N steps whenever its
        # oldest point drops out, which is a jump in the SIGNAL that then
        # becomes a jump in the rate - the staircase this replaced.
        #
        # The slow fit is the evidence. The fast one exists only to notice
        # sooner when improvement has stopped, and can only ever lower the
        # verdict - see observe().
        self.S = _sums()
        self.F = _sums()
        self.i = 0.0                   # evaluation counter, the x axis
        self.se_hist = deque(maxlen=64)
        self.prev = None
        self.jump_from = None          # a candidate regime change, unconfirmed
        self.events = []
        self.step = 0
        self.last_t = 0.0
        self.last_e = 0.0
        self.harm = 0                  # harm verdicts in a row
        self.recent = deque(maxlen=self.RISE_RUN)   # the last verdicts
        self.cool = 0                  # evaluations left without a rise
        self.seen_at = None            # self.step at the last evaluation taken

    # ---------------------------------------------------------------- lr
    def factor(self):
        """What to multiply every group's base rate by, right now."""
        w = min(1.0, (self.step + 1) / self.WARMUP) if self.step < self.WARMUP \
            else 1.0
        return self.scale * w

    def tick(self):
        self.step += 1

    # ------------------------------------------------------------ evidence
    def _se(self):
        if not self.se_hist:
            return 0.0
        s = sorted(self.se_hist)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def _accumulate(self, y):
        """Add one reading to BOTH fits. They share the x axis."""
        self.i += 1.0
        x = self.i
        for S, lam in ((self.S, self.LAM), (self.F, self.LAM_FAST)):
            for k in ("w", "x", "y", "xx", "xy", "yy"):
                S[k] *= lam
            S["w2"] *= lam * lam
            S["w"] += 1.0
            S["w2"] += 1.0
            S["x"] += x
            S["y"] += y
            S["xx"] += x * x
            S["xy"] += x * y
            S["yy"] += y * y

    @staticmethod
    def _fit(S):
        """
        (t, n_eff, e) for one set of running sums.

        t is the slope over its own standard error - the significance. e is
        the slope over the RESIDUAL SCATTER - the effect size, improvement per
        evaluation in units of the noise it has to be seen through. Both are
        signed so that positive means held-out is falling.

        The difference between them is the whole point of this file. t carries
        a sqrt(sxx) that grows until the window fills and then stays large, so
        t rewards patience rather than progress. e carries no such factor.
        """
        if S["w"] <= 2:
            return 0.0, 0.0, 0.0
        n_eff = S["w"] ** 2 / max(S["w2"], 1e-12)
        sxx = S["xx"] - S["x"] ** 2 / S["w"]
        sxy = S["xy"] - S["x"] * S["y"] / S["w"]
        syy = S["yy"] - S["y"] ** 2 / S["w"]
        if sxx <= 0 or n_eff <= 2:
            return 0.0, n_eff, 0.0
        slope = sxy / sxx
        resid = max(syy - slope * sxy, 0.0)
        sigma = math.sqrt(resid / (n_eff - 2)) if n_eff > 2 else 0.0
        if sigma <= 0:
            # A perfectly straight run has no residual, so neither statistic
            # is defined. Report nothing rather than infinity; real held-out
            # is never this clean, and a run that is has no noise to see
            # through and needs no help from here.
            return 0.0, n_eff, 0.0
        t = -slope / (sigma / math.sqrt(sxx))
        e = -slope / sigma
        return max(-50.0, min(50.0, t)), n_eff, e

    def _trend(self):
        """(t, effective observations) for the slow fit. Kept for callers."""
        t, n_eff, _ = self._fit(self.S)
        return t, n_eff

    def _verdict(self):
        """
        The number the rate actually moves on, and the two it is built from.

        min() of three readings, because each can only ever say "less than you
        think" and none may say "more":

          t_slow          is it distinguishable from noise at all - the
                          significance guard, taken from the better-powered
                          fit because that is what it is for
          EFFECT * e_slow is it big enough to be worth a higher rate
          EFFECT * e_fast ...and is it still happening now

        Only the SLOW t appears. t depends on how much window it was measured
        over - se(slope) shrinks as sqrt(sxx) - so a deliberately short fit has
        a mechanically small t and would veto everything regardless of what the
        loss is doing. The fast fit contributes its EFFECT SIZE instead, which
        does not know how long the window was.
        """
        t_s, n_s, e_s = self._fit(self.S)
        _, n_f, e_f = self._fit(self.F)
        t = min(t_s, self.EFFECT * e_s)
        if n_f >= self.MIN_EFF:
            t = min(t, self.EFFECT * e_f)
        return max(-50.0, min(50.0, t)), n_s, e_s

    @classmethod
    def move(cls, t):
        """
        The log-change one verdict asks for, before observe()'s two guards
        (harm has to repeat; a back-off pauses rises).

        Up while held-out improves: nothing below T_DEAD, rising linearly to
        PROBE at T_MID, then up to PROBE + NUDGE where improvement is
        provable. Nothing while it is flat. Down, by up to NUDGE_DOWN and in
        proportion to how bad it is, below -T_HARM. Down is a correction and
        up is a probe, so down is the faster of the two.
        """
        if t >= cls.T_MID:
            return cls.PROBE + cls.NUDGE * math.tanh((t - cls.T_MID) / cls.T_W)
        if t > cls.T_DEAD:
            return cls.PROBE * (t - cls.T_DEAD) / (cls.T_MID - cls.T_DEAD)
        if t >= -cls.T_HARM:
            return 0.0
        return cls.NUDGE_DOWN * math.tanh((t - cls.T_MID) / cls.T_W_DOWN)

    def _worse(self):
        """
        Whether held-out is measurably getting worse: the slow fit says so with
        significance AND the fast one says it is still happening. The verdict
        that sizes rises is the minimum of three readings, pessimistic on
        purpose, and on a flat run that pessimism alone reads as harm about as
        often as noise dips - so harm asks both fits instead.
        """
        t_s, _, _ = self._fit(self.S)
        _, n_f, e_f = self._fit(self.F)
        return t_s < -self.T_HARM and (n_f < self.MIN_EFF
                                       or self.EFFECT * e_f < -self.T_HARM)

    def observe(self, val, se=None):
        """
        One held-out evaluation. Returns a note if the rate moved notably.

        The rate moves by exp(move(t)) - a drift rather than a staircase, so
        nothing the rate does is ever a shock to the run - unless this
        evaluation is not evidence: too little read since the last one, or a
        jump still waiting for the next evaluation to confirm it.
        """
        if val is None or not math.isfinite(float(val)):
            return None
        val = float(val)
        # NO READING, NO NEWS. Only counted where the caller counts steps
        # (tick()); a controller that is never ticked takes every evaluation.
        if (self.step > 0 and self.seen_at is not None
                and self.step - self.seen_at < self.MIN_STEPS):
            return None
        if se is not None and math.isfinite(float(se)) and float(se) > 0:
            self.se_hist.append(float(se))
        s = self._se()
        note = None

        # ---- REGIME: a jump, confirmed on the following evaluation --------
        if self.jump_from is not None:
            if val > self.jump_from + self.HOLD_SE * max(s, 1e-9):
                before = self.scale
                self.scale = min(self.CEIL, self.scale * self.UP)
                note = (f"held-out moved to {val:.4f} from {self.jump_from:.4f} "
                        f"and stayed - new regime, rate {before:.3f} -> "
                        f"{self.scale:.3f}")
                self.events.append({"at": self.step, "kind": "regime",
                                    "val": val, "scale": self.scale})
                self.S = _sums()
                self.F = _sums()
                self.i = 0.0
                self.harm = self.cool = 0
                self.recent.clear()
            # unconfirmed, it was a bad measurement: it was never added to the
            # fits, and the run goes on from the evaluation before it
            self.jump_from = None
        elif self.prev is not None:
            bar = max(self.JUMP_SE * s, self.FLOOR_JUMP)
            if val - self.prev > bar:
                self.jump_from = self.prev      # confirm or discard next time
                self.seen_at = self.step
                return None                     # ...and until then, not evidence

        self.seen_at = self.step
        self.prev = val
        self._accumulate(val)
        t, n_eff, e = self._verdict()
        self.last_t = t
        self.last_e = e
        self.recent.append(t)

        # ---- the move, sized by the evidence: see move() ----------------------
        if note is None and n_eff >= self.MIN_EFF:
            before = self.scale
            g = self.move(t)
            if g > 0:
                # up only on evidence that has held: sized by the weakest of
                # the last RISE_RUN verdicts, nothing until there are that many
                g = (self.move(min(self.recent))
                     if len(self.recent) == self.RISE_RUN else 0.0)
            if g < 0 and not self._worse():
                g = 0.0          # the pessimistic verdict alone is not harm
            if g < 0:
                self.harm += 1
                if self.harm < self.HARM_RUN:
                    g = 0.0                     # harm has to repeat to count
                else:
                    self.cool = self.COOL
            else:
                self.harm = 0
                if self.cool > 0:               # after a back-off, no rise yet
                    self.cool -= 1
                    g = 0.0
            self.scale = max(self.FLOOR, min(self.CEIL, self.scale * math.exp(g)))
            # One line per evaluation would be noise. Record only when the
            # rate has drifted a full 2% since the last thing recorded.
            last = self.events[-1]["scale"] if self.events else 1.0
            if abs(math.log(self.scale / max(last, 1e-9))) > 0.02:
                kind = ("improving" if t > self.T_MID else
                        "deteriorating" if t < -self.T_HARM else "settling")
                note = (f"{kind}: t={t:+.2f} (effect {e:+.3f} per evaluation) "
                        f"over {n_eff:.0f} effective evaluations, rate "
                        f"{before:.3f} -> {self.scale:.3f}")
                self.events.append({"at": self.step, "kind": kind,
                                    "val": val, "scale": self.scale,
                                    "t": round(t, 2)})
        return note

    # ------------------------------------------------------------- restart
    def state(self):
        t, n_eff, e = self._verdict()
        return {"scale": self.scale, "prev": self.prev,
                "S": dict(self.S), "F": dict(self.F), "i": self.i,
                "se": list(self.se_hist), "step": self.step,
                "harm": self.harm, "cool": self.cool, "seen_at": self.seen_at,
                "recent": list(self.recent),
                "jump_from": self.jump_from,
                "n": round(n_eff, 1), "t": round(t, 3), "e": round(e, 4),
                "events": self.events[-40:]}

    @classmethod
    def restore(cls, d):
        p = cls()
        if not d:
            return p
        p.scale = float(d.get("scale", 1.0))
        p.prev = d.get("prev")
        if isinstance(d.get("S"), dict):
            p.S.update({k: float(v) for k, v in d["S"].items() if k in p.S})
            p.i = float(d.get("i", 0.0) or 0.0)
            if isinstance(d.get("F"), dict):
                p.F.update({k: float(v) for k, v in d["F"].items()
                            if k in p.F})
            # A checkpoint that carries no F leaves the fast fit empty, and
            # that is correct. Seeding it from the slow fit would copy the slow
            # fit's n_eff with it, so the "fast" reading would just be the slow
            # one again until it decayed and the cap would never bind. Empty
            # costs nothing: MIN_EFF gates only this cap, not the controller,
            # so the slow fit keeps steering throughout.
        elif d.get("hist"):
            # an older checkpoint kept a plain window; replay it so the fit
            # starts from the evidence that was already gathered
            for v in d["hist"]:
                p._accumulate(float(v))
        # The warmup is there in case the moments were lost. They ARE restored
        # here, so re-running it on every resume only puts a notch in the rate
        # that nothing asked for.
        p.step = int(d.get("step", 0) or 0)
        p.harm = int(d.get("harm", 0) or 0)
        for v in d.get("recent") or []:
            p.recent.append(float(v))
        p.cool = int(d.get("cool", 0) or 0)
        p.seen_at = d.get("seen_at")
        p.jump_from = d.get("jump_from")
        for v in d.get("se") or []:
            p.se_hist.append(float(v))
        p.events = list(d.get("events") or [])
        return p

    def describe(self):
        s = self._se()
        head = (f"learning rate is governed by held-out, not by a horizon - up "
                f"while it improves, down only when it worsens: "
                f"rate x{self.scale:.3f}, floor x{self.FLOOR}")
        return head + (f", noise {s:.4f}" if s else "")
