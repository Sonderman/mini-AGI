"""
The learning rate's controller, as every stream that learns runs it.

THE CONTROLLER IS PART OF THE MODEL. minagi/plasticity.py sets how fast the
model adapts - down on a plateau, back up when the material changes - and it
does that from held-out readings. Training, a conversation, and a test set
read while learning (the benchmarks' dynamic evaluation and best effort) all
continue the one stream the model learns from, so all of them run the
controller the same way, through this:

  the session's start  the larger evaluation and the checks' own blocks are
                       read side by side; every check reports what it reads
                       plus the difference between the two, the level
                       (minagi/heldout.py)
  a check              the checks' blocks, read at that level, every `every`
                       minutes of learning: the controller observes the mean
                       (Plasticity.observe)

Training takes its check in its sample round (train.py, cmd_read), a learner
outside training when it is due (minagi/live.py), and both go through
reading() and govern() below - the one code path, so a change to either
reaches both.

WHAT THE CHECK READS is set by the run and written into its manifest
(`checks`): how often, which held-out folder, how many characters of each
subject. A manifest written before that was recorded is read with DEFAULTS,
which are this run's own settings - the command in the README.
"""

import os
import time

import numpy as np

# The run's own settings, for weights that do not record them: a check every
# 10 minutes (--sample-every), on data/val (--held-out), reading 60 x 512
# characters of each subject (--sample-eval-chars) at the level of a session's
# 240 x 512 (--eval-chars). train.py's defaults for the last two are these.
DEFAULTS = {"every": 10.0, "held_out": "data/val",
            "check_chars": 60 * 512, "eval_chars": 240 * 512}


def steps_of(chars, chunk):
    """Characters of each subject, as the evaluator counts them: chunks."""
    return max(1, int(chars) // max(1, int(chunk)))


class Governor:
    """The held-out check and what it does to the rate, for one session of a
    model that learns. `plast` is its controller, `ev` a FolderEvaluator on
    the held-out folder."""

    def __init__(self, plast, ev, chunk, every=DEFAULTS["every"],
                 check_chars=DEFAULTS["check_chars"],
                 eval_chars=DEFAULTS["eval_chars"]):
        self.plast, self.ev = plast, ev
        self.every = float(every or 0)
        self.check_steps = steps_of(check_chars, chunk)
        self.eval_steps = steps_of(eval_chars, chunk)
        self.level = {}
        self.last = time.time()
        self.checks = 0

    @classmethod
    def for_learner(cls, model, plast, chunk, device, settings=None,
                    root=None):
        """A governor as the run that wrote `settings` (a manifest's
        `checks`, or None for DEFAULTS) ran its checks, or None when its
        held-out folder is not there to read - then the rate stays where the
        run left it, and says so."""
        s = dict(DEFAULTS)
        s.update({k: v for k, v in (settings or {}).items() if v is not None})
        path = s["held_out"]
        if root and not os.path.isabs(path):
            path = os.path.join(root, path)
        if not (s["every"] and os.path.isdir(path)):
            return None
        from .stream import FolderEvaluator
        # training's chunk, within the window the model was built with - a
        # chunk longer than that could not be read at all
        chunk = min(int(chunk), int(model.cfg.block))
        ev = FolderEvaluator(model, path, chunk, model.cfg.block, device)
        return cls(plast, ev, chunk,
                   every=s["every"], check_chars=s["check_chars"],
                   eval_chars=s["eval_chars"])

    def start(self):
        """The session's start: the larger evaluation and the checks' blocks,
        side by side, which set the level. Returns both, standard error
        removed from the first and returned beside it."""
        v = self.ev.run(self.eval_steps, spread=True)
        se = v.pop("stderr", 0.0)
        c = self.ev.run(self.check_steps, spread=True)
        c.pop("stderr", None)
        self.level = {k: v[k] - c[k] for k in v if k in c}
        self.last = time.time()
        return v, se, c

    def due(self, now=None):
        """Whether `every` minutes of learning have passed since the last
        check, or the start."""
        return bool(self.every) and ((now or time.time()) - self.last
                                     >= self.every * 60)

    def reading(self):
        """One check: the checks' blocks, at the start's level. Returns the
        subjects' readings, their mean - what the controller sees - and its
        standard error."""
        d = self.ev.run(self.check_steps, spread=True)
        se = d.pop("stderr", None)
        d = {k: x + self.level.get(k, 0.0) for k, x in d.items()}
        return d, float(np.mean(list(d.values()))), se

    def govern(self, v, se):
        """What a reading does to the rate: the controller observes it.
        Returns the controller's note, if the rate moved notably."""
        moved = self.plast.observe(v, se)
        self.checks += 1
        return moved

    def check(self):
        """A check outside training: read, govern, and start the clock again.

        One the controller would not count - fewer optimiser steps since the
        last reading it took than MIN_STEPS, Plasticity.observe's own rule -
        changes nothing in it, so the blocks are not read and the clock
        starts again.
        A quiet conversation spends nothing on checks. Returns (readings,
        mean, standard error, note), or None when skipped."""
        p = self.plast
        if (p.step > 0 and p.seen_at is not None
                and p.step - p.seen_at < p.MIN_STEPS):
            self.last = time.time()
            return None
        d, v, se = self.reading()
        moved = self.govern(v, se)
        self.last = time.time()
        return d, v, se, moved
