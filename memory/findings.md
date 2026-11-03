# Findings

**Status (2026-10-28).** Moved here from `FINDINGS.md` at the repository root. Written 2026-09-02..03
(last changed 2026-09-03) on a model of 64-128 experts of 98,304 parameters, before almost everything in
history.md. Kept as found. Section by section:
- 1 superseded. Pruning and the growth brake stopped reading the gate on 09-13 ("the gate was pointed the
  wrong way"); an expert is dying when nothing has admitted it for long (staleness since last
  admission; `prune.survival_chars`); selection has been one rule since 09-27 (decisions.md).
  The lesson stands: a signal the objective shapes cannot judge what it shapes.
- 2 holds: stderr with every loss; ~0.03 between separate runs. Held-out has been rebuilt since
  (corrections.md).
- 3 holds: readers start a new window rather than slide. 4 holds: accumulation stays at 1.
- 5 holds: trunk at x0.1 of the experts' rate (`training.trunk_lr_mult`); the replication (experiments.md
  E10) confirmed it on the trained model.
- 6 fixed on 2026-09-02; cached chunks use the lower-right causal bias.
- 7: growth is five brakes now (ROOM, USED, EARNING, FITS, HONEST; config.yaml `growth:`), every 2M
  characters, a newborn at a small gate (`growth.birth_gate` 0.001) recombined from its parents.
- 8 and 9 hold as history.
`python3 -m experiments <name>` runs one of `experiments/` (tracked since 2026-10-31; last changed 2026-09-21).


Results that changed what the project does, with the numbers behind them.
Several are failures; those are the useful ones, and they are kept in the shape
they were found rather than tidied into the conclusion they led to.

Reproduce any of them with `python3 -m experiments <name>`.

---

## 1. Read the gate, not the routing traffic

**The mistake.** Three separate mechanisms — the idle brake, the pruner, and the
growth trigger — all asked "is this expert being used?" and answered it by
looking at how much routing traffic it received. All three were therefore
reading the same number, and that number is manufactured by the objective:

```
load-balancing auxiliary loss pushes routing toward uniform, by design
  -> routing entropy sits at 0.995 of maximum
  -> every expert receives roughly its 1/n share, live or dead
  -> idle count reads 0/n forever, and no `use` threshold can separate them
```

The pruner's condition `use/total < 1e-4` could never fire: an even share of 112
experts is 0.0089, two orders of magnitude above the threshold. So pruning never
ran, which meant the "back off if previous additions were pruned" brake had no
data, which meant it read 1.00 forever too. Three brakes, none of them capable of
engaging.

The pool ran 64 → 80 → 96 → 112 → 128 unopposed and died of CUDA OOM. At the
point it crashed, 42 of the 128 experts had never left a gate of zero.

**The fix.** The gate is under no load-balancing pressure. It is trained only by
whether the expert helps, and a new expert starts at exactly zero, so a gate
still at zero after a fair number of steps is the model saying it found no use
for that expert. Two guards prevent pruning something that is still learning:
age in *training steps*, and gate velocity — a gate that is tiny but rising is
left alone.

**A related bug.** `age` was incremented once per call site, so with 26 block
applications and 8 micro-batches it advanced ~200 per training step, while being
compared against a step count. A `born` buffer now records the actual step.

**Result.** Over one 1800-step run the pool went 86 → 83 → 80 → 77 → 75 → 74 in
decreasing increments and then stopped on its own, ending with **zero dead
gates** and a median |gate| of 1.141. The size was not chosen; the run arrived
at it.

---

## 2. The evaluator was measuring its own sampling noise

For most of this project validation was 6 chunks per domain — **12,288
characters, 0.14% of the held-out text**. Repeated draws on *unchanged* weights:

| chunks | characters | sd across draws |
|---|---|---|
| 6 | 12,288 | **0.0275** |
| 24 | 49,152 | 0.0170 |
| 96 | 196,608 | 0.0093 |
| 240 | 491,520 | 0.0064 |

A whole sweep of configurations was compared on differences of 0.0008 to
0.0114. Every one was inside the noise. The working-set result — reported with
explicit confidence — was **3% of one standard deviation**.

Worse, it invented a result that shaped weeks of work. Streaming was said to be
losing to batch training, 0.6864 against 0.6524, and a character-budget
explanation was built on top of that. Scored honestly, streaming *improves* on
what batch training left by 0.028 (2.0 sigma). The gap was the instrument.

**The fix is not a bigger number.** It is that `Evaluator` now returns
`stderr` alongside the loss, so a difference cannot be quoted without its error
bar being visible at the same moment. The sample is the WINDOW, not the chunk:
chunks inside a window are correlated, and using per-chunk spread overstates
the error about two and a half times. Calibrated against repeated draws.

Section 9 of this file already documented six benchmark instruments that
measured themselves. That section was written, and then the same mistake was
made again on the validation loss.

**There is a second variance underneath this one.** The figures above measure
how much the SCORE moves when the weights are fixed. They say nothing about how
much the WEIGHTS move when the recipe is fixed. Running one configuration twice
- same seed, same budget, same data - gave 0.6594 and 0.6735, a gap of 0.0141.
The expert dispatch and the scatter-add that follows it are non-deterministic
on CUDA, so two runs of the same recipe genuinely diverge.

Repeating every configuration across seeds would price this properly and is not
affordable here. So it is priced by judgement instead: **treat roughly 0.03 as
the threshold for a real difference**, not the +/- printed beside a single
score. Anything smaller is a run, not a result. Several conclusions in this file
would not survive that bar, including the apparent harm from accumulation at 64
in section 4.

---

## 3. Sliding a KV cache breaks rotary positions

A reader that ran past its own context slid the window: trim the cache, hold
the position offset at its last value. That is wrong, and quietly.

The keys left in the cache carry the rotary encoding of the positions they were
**computed** at. Every new query claims a position near the end of the window.
The relative distances RoPE encodes stop matching, and stay mismatched for the
rest of the read.

```
same weights, held-out loss
  reading within the window     0.68
  reading past it               2.60
```

Training never saw it, because `StreamSet` opened a fresh window every
`context` characters from the outside. It appeared the moment the evaluation
was made longer — so the attempt to *fix* the noise problem above would have
produced garbage and blamed the model.

Readers are self-limiting now: a full window starts a new one, which is what
the training loop was already doing externally.

---

## 4. Gradient accumulation buys nothing

Proposed to fix the gradient noise that was supposedly making streaming lose.
At a fixed budget of 3.0M characters, so only the averaging differs:

| accum | optimiser steps | characters per update | val |
|---|---|---|---|
| 1 | 5,859 | 512 | **0.6594 ± 0.0093** |
| 8 | 732 | 4,096 | 0.6615 ± 0.0090 |
| 64 | 91 | 32,768 | 0.6701 ± 0.0091 |

accum 1 against 8 is 0.23 sigma — nothing. Against 64 it is 1.2 sigma, and
confounded: at 64 the model gets only 91 updates, so "worse gradients" and "too
few updates" cannot be separated.

Kept as a flag defaulting to 1. The rationale for it was reasoning from a gap
that finding 2 shows never existed.

---

## 5. The forgetting lives in the trunk

Four corpora read one after another, nothing repeated. Whatever is read first is
worst at the end; the spread between best and worst domain is the forgetting.

| | code | math | chat | chess | mean | spread |
|---|---|---|---|---|---|---|
| dense, streaming | 5.595 | 4.602 | 5.754 | 1.716 | 4.417 | 4.038 |
| pooled, streaming | 4.970 | 4.459 | 5.201 | 1.666 | 4.074 | **3.535** |
| dense, mixed batches | 2.749 | 2.459 | 2.954 | 1.544 | **2.426** | 1.410 |
| pooled, mixed batches | 2.796 | 2.745 | 3.031 | 1.633 | 2.551 | 1.398 |

The expert pool is worth **−12.5% spread** on its own. Real, and nowhere near
enough.

**Freezing one part at a time** localises it: freezing the shared trunk removes
**99.7%** of the forgetting (+0.007 against +2.069). Freezing the routers or the
experts changes nothing measurable. Every domain writes through the trunk, so
every new subject overwrites the last one there.

**Slowing the trunk** rather than freezing it:

| trunk LR (× the pool's) | mean | spread |
|---|---|---|
| 1.0 | 3.519 | 2.331 |
| 0.3 | 3.475 | 1.307 |
| **0.1** | **3.370** | **0.878** |
| 0.03 | 4.004 | 0.723 |
| 0.0 (frozen) | 4.922 | 0.559 |

Frozen is the best forgetting and the worst learning — chess notation never
lands at all. ×0.1 improves both mean and spread together.

The prediction that this would only help streaming was wrong: on **mixed**
batches, ×0.3 improved mean 2.472 → 2.380 *and* spread 1.352 → 1.005. Even
inside one mixed batch the domains pull the trunk in different directions every
step, and letting the experts do the fast adapting reduces that conflict.

---

## 6. Attention with a KV cache was masked wrongly

`is_causal` aligns its triangle to the **top-left** of the score matrix, which is
correct only when the queries and the keys are the same tokens. With P cached
positions, the queries sit at absolute `P..P+T-1` against `P+T` keys, so a
top-left triangle lets query *i* see keys `0..i` instead of `0..P+i`.

Every chunk after the first attended to the wrong window entirely.

```
                chunked loss    max |logit diff| vs one full pass
before             4.0326                     3.485e+01
after              1.2022                     1.767e-01
```

The single-token decode path hid this for months of work, because a 1×N row
needs no mask at all — it only appears once more than one position is fed at a
time, which is exactly what prompt prefill does. Prompts served through the UI
were being read against a mis-aligned window.

Fixed with an explicit bottom-right-aligned mask. Chunked reading now matches a
full pass exactly.

---

## 7. Two triggers for growth that looked right and were not

**Loss plateau.** Deferred growth indefinitely: the learning rate anneals on a
cosine schedule, so loss keeps improving whether or not the model has room.
Measured mid-run with a completely saturated pool, growth still could not fire.

**Routing pressure.** `want_k ≈ 50 of 64` on every domain — which looks like the
pool is too small to express what the router wants. It is the aux loss again
(see finding 1). A measurement taken *after* normalisation also reported
"102.8% of probability mass", which should have been the tell.

What survived is not a trigger at all: add on speculation, and let pruning
arbitrate.

---

## 8. Silent data bugs that invalidated earlier results

Kept because each produced confident, plausible, wrong numbers.

**The mixture was a fiction.** With a micro-batch of 1 and four corpora,
`int(round(1 * 0.25))` is 0 for every source but the last, which received the
whole batch. The model trained on **chess only** — code loss 8.16 against 5.58
for random weights — while validation reported 0.6354 through the same broken
function. Every measurement taken before this was found describes a chess model.

**Expert files were permuted.** `sorted()` on checkpoint keys is lexicographic,
so expert 10 was written to `e00002.npy`. Silent, 1.4e-02 output mismatch.

**Resuming a grown pool discarded it.** The model is built at `cfg.pool_experts`;
`load_state_dict(strict=False)` then quietly ignores every expert past that size.
A 128-expert pool loaded as 64 with no error. There is now a guard that refuses
loudly.

**Zeroing a router row does not unroute an expert.** A zero logit is *middling*,
not excluded. Growth was perturbing the model's output by 7.4e-02 at the moment
it was supposed to change nothing. Fixed with additive banks and zero-init output
gates: drift is now exactly 0.000e+00.

---

## 9. Benchmark instruments that measured themselves

* dedented method prompts — 10/40 false failures
* truncation scored as a syntax error — 11/40
* unseeded generation — parse 0.470 vs 0.550 on *identical* checkpoints
* a line-only repetition metric blind to intra-line loops — scored 0.00 where
  the real figure was 0.654
* a token budget not scaled for a character model — the answer could not fit
