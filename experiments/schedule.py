"""Learning-rate schedule.

Cosine decay to a floor, with a linear warmup. One copy, shared by every
experiment, because a schedule that differs between two arms is not a
comparison.
"""

import math


def lr_at(step, total, base, warmup, floor_frac=0.1):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))
