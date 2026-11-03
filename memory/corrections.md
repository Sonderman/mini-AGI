# corrections

In-flight corrections of the training run: what was measured wrong while it ran, how it was fixed, and how
the history was carried across the fix. The run is never started over, so a measurement found wrong is
fixed mid-run and its past is moved onto the corrected level. Until 2026-10-28 this lived in the README
(collapsible "Training in-flight corrections" section, anchor `#in-flight-corrections`, created on
2026-10-18 at the author's request); the README now states only today's method and links here. A third
correction gets its own numbered item below, not text in the README.

## Policy (the author, 2026-10-17 and 10-18)

- Measure both ways on the same weights. The first session after a fix reads held-out the old way as
  well as the new, at its start, and writes both into its banner in `runs/samples.txt` (the
  `# held-out at the start, ...` lines).
- Everything before moves by the difference, subject by subject. An earlier check is drawn moved by what
  each of its subjects differed by at the switch; its overall figure by the mean of those over the subjects
  the check itself has (`minagi/heldout.py` `moved`). The training graph, the scaling chart and its fit,
  the best round in the README, the README's figures and the lane-schedule fits all use moved values, so
  curves run on across a fix without a step.
- A correction never rewrites the log: `runs/samples.txt` keeps the checks as measured. In it, a check's held-out
  line ends with `level`: the difference its session added to what the check's own blocks read.
- No vertical lines, stars or footnotes on graphs. History drawn as logged, with a starred line where the
  reading changed, was tried; the author rejected it as clutter that breaks the
  curves and the fit.
- It assumes the difference held: one difference, measured at the fix, moves everything before it as if
  the two ways of reading stayed that far apart all along. Nothing can check that for the far past, which is
  one reason the raw log is kept.
- Any state fitted to the old level is re-seeded or moved at the switch: the best held-out the run
  compares new rounds against, the lane-schedule fits.

## 1. Checks read only the top of each subject (fixed 2026-10-17, at 2,290.1M characters)

A check read the first 30,720 characters of each subject's
files: the same 15 chunks every time, the first documents only. The top of a subject is unlike the rest, and
not in the same direction for every subject. Overall the checks read 0.011 high.

| subject | first 30,720 characters | 30 blocks spread over all of it | earlier checks moved by |
|---|---|---|---|
| arithmetic | 0.617 | 0.610 | -0.007 |
| chat | 0.582 | 0.517 | -0.065 |
| chat_hermes | 0.858 | 0.743 | -0.115 |
| chess | 0.382 | 0.384 | +0.002 |
| code | 0.503 | 0.514 | +0.010 |
| reasoning | 0.527 | 0.562 | +0.035 |
| stories | 0.391 | 0.486 | +0.094 |
| wikipedia | 1.011 | 0.972 | -0.039 |
| **all 8** | **0.609** | **0.598** | **-0.011** |

The larger evaluation a session ended on before then read the first 122,880 characters of each subject: a
third measurement matching neither, so the graphs leave those entries out instead of moving them (an entry
before the switch without the controller's `evidence t` line is one of those; `heldout.is_check`). Before
the fix a 0.5982 "dip" on the graph was such a session-end evaluation appended with a test arm's log.

The banner that set it (session started 2026-10-17 11:14, step 1,118,555, 330 experts):

```
# held-out at the start, spread over the whole set: 0.5984   arithmetic 0.6101   chat 0.5172   chat_hermes 0.7427   chess 0.3837   code 0.5138   reasoning 0.5621   stories 0.4856   wikipedia 0.9719
# held-out at the start, the checks' blocks: 0.6305   arithmetic 0.6121   chat 0.5203   chat_hermes 1.0242   chess 0.3863   code 0.4631   reasoning 0.5640   stories 0.4934   wikipedia 0.9806
# held-out at the start, from the top as checks read before: 0.6091   arithmetic 0.6168   chat 0.5824   chat_hermes 0.8582   chess 0.3822   code 0.5034   reasoning 0.5274   stories 0.3914   wikipedia 1.0107
```

`first_shift` takes the first banner with both a spread and a top line; delta = top - spread per subject
(the table's last column is minus that). Later sessions kept writing all three lines until the second fix;
only the first is used.

What came with it: every session starts with a 30-block evaluation spread over the whole set; each check
reads 8 blocks of each subject; the governor (`minagi/governor.py` `start`, `reading`) reports each check
at the start evaluation's level: what the check reads plus `level` = start evaluation - checks' blocks,
per subject. The checks' blocks sit a little above or below the whole set by an amount that stays put for
as long as the blocks do, so a check moves with the model at the level of an evaluation four times its size.

## 2. A file's leftover counted as a block (fixed 2026-10-18, at 2,386.0M characters)

Blocks are cut from each file 4,096 characters at a time; what was left after a file's last
whole block used to be a block of its own, however short. A chunk's loss is a mean over its characters and
a subject's a mean over its chunks, so a leftover of a few dozen characters, predicted from almost no
context, counted as much as a full chunk of 2,048. Reading from the top had seldom met one (the shortest
chunk a check read was 1,006 characters, the same one every time), but spread blocks did: chunks of 73, 68
and 32 characters were three of the fourteen in chat_hermes's check. That check read ~1.0 nats where the
session's 30-block evaluation read 0.74, and its noise went from 0.002 a check to 0.025. That is the jagged
stretch of chat_hermes's curve between the two fixes; an eighth of it (up to ~0.008 check to check) reached
the overall figure. The curve also looked to fall faster there than it did: the 30-block evaluation, the same
blocks every time, went only 0.743 -> 0.734 over that stretch. Found because the author noticed chat_hermes's
curve looked wrong.

Now only whole blocks are read (`FolderEvaluator(..., whole=True)`, `minagi/stream.py` "WHOLE BLOCKS
ONLY"), so every chunk is 2,048 characters. Left out: 16% of stories' text, 13% of chat_hermes's, 6% of
reasoning's (one of its 400 files is too short to hold a block at all), under 2% of each other subject's. A
subject with no whole block anywhere would keep its leftovers; none is like that.

| subject | with leftovers | whole blocks only | earlier checks moved by | short blocks among the 30 | text left out |
|---|---|---|---|---|---|
| arithmetic | 0.612 | 0.612 | 0.000 | 0 | 1.4% |
| chat | 0.516 | 0.506 | -0.010 | 0 | 0.8% |
| chat_hermes | 0.734 | 0.820 | +0.086 | 5 | 12.8% |
| chess | 0.381 | 0.386 | +0.005 | 1 | 0.5% |
| code | 0.510 | 0.512 | +0.003 | 1 | 1.8% |
| reasoning | 0.555 | 0.560 | +0.005 | 2 | 5.9% |
| stories | 0.483 | 0.464 | -0.019 | 5 | 15.9% |
| wikipedia | 0.969 | 0.929 | -0.040 | 0 | 0.1% |
| **all 8** | **0.595** | **0.599** | **+0.004** | 14 of 240 | 4.2% |

Most of that movement is not the short blocks. Leaving them out changes the list the 30 are spread over, so
most of the 30 change: wikipedia had no short block and moved 0.040 when 12 of its 30 picks moved to
neighbouring blocks. A subject's figure is a sample of 30 blocks, good to a few hundredths; the overall
figure is good to the +/- printed beside it, while the curves, which read the same blocks throughout, track
how the model moves on those.

The banner that set it (session started 2026-10-18 20:17, step 1,165,384, 378 experts):

```
# held-out at the start, spread over the whole set: 0.5987   arithmetic 0.6116   chat 0.5062   chat_hermes 0.8201   chess 0.3859   code 0.5123   reasoning 0.5602   stories 0.4638   wikipedia 0.9291
# held-out at the start, the checks' blocks: 0.6084   arithmetic 0.6303   chat 0.5166   chat_hermes 0.7957   chess 0.3866   code 0.5686   reasoning 0.5824   stories 0.4837   wikipedia 0.9036
# held-out at the start, from the top as checks read before: 0.6061   arithmetic 0.6178   chat 0.5814   chat_hermes 0.8475   chess 0.3819   code 0.5013   reasoning 0.5203   stories 0.3907   wikipedia 1.0082
# held-out at the start, with each file's leftover part-block, as read before: 0.5949   arithmetic 0.6115   chat 0.5164   chat_hermes 0.7338   chess 0.3808   code 0.5097   reasoning 0.5551   stories 0.4826   wikipedia 0.9690
```

`part_shift` takes the first banner with a spread and a parts line; delta = parts - spread. It applies to
everything logged before that banner, on top of correction 1 for entries before 2026-10-17. At that
session `train.py` read the old blocks once (only when the log has no parts line yet), moved `best_val`
by mean(parts) - spread, and dropped the lane-schedule fits so they were refitted from the moved log.

enwik9 and PG19 are untouched by this: each of the 2,000 enwik9 pieces is a full 2,048 bytes (its slice's
leftover was never one of them) and PG19 is scored as total bits over total bytes, so a short piece counts
for only what it holds.

## Where it is implemented

- `minagi/heldout.py`: banner labels (`SPREAD`, `CHECKS`, `TOP`, `PARTS`), `banner`, `levels`,
  `first_shift`, `part_shift`, `moved`, `is_check`, `parse_log` (every entry with held-out on one level,
  marked `shifted`; session-end evaluations from before correction 1 left out). Pure Python, no torch.
- `train.py` session start (~line 920-960): the start evaluation, the checks' blocks, the old ways once,
  `best_val` moved; lane fits dropped and re-bootstrapped from `parse_log` (~line 1118-1135).
- `minagi/governor.py`: `start` sets `level`, `reading` adds it to every check.
- Readers of the moved history: `tools/plot_progress.py` (training graph), `tools/plot_scaling.py` (chart
  and fit), `tools/refresh_readme.py` (best round, held-out figures; its run-blocks caption links here),
  `tools/measure_heldout.py`, `minagi/lanes.py` (`bootstrap`).
- Test: `tests/test_heldout_level.py`.

## Not corrections, but they look like gaps in the log

- Sessions start and stop (88 by 2026-10-28); each start re-measures and re-levels, which can put a small
  step between the last check of one session and the first of the next. A restart is a pause, never drawn as
  a marker. The learning-rate controller carries its fits across a restart by the step the
  restart made (`minagi/plasticity.py` `_relevel`).
- Optimizer switch to Muon at 3,194.4M chars (2026-10-27) and the newborn-rate removal:
  changes to the model, not to measurement; nothing moved.
- Corpus size in the header changed 7,880M -> 8,037M chars after the binary-detection fix (issue #30,
  2026-10-12); the percentage read is against the size at the time.

## How to do the next one

1. Keep the old evaluation callable (a flag on `FolderEvaluator` or a separate reader).
2. Add a banner label to `minagi/heldout.py`, write it from the first session after the fix only (detect
   that the log has no such line yet), on the same weights, at the start.
3. Add a `*_shift` function; apply it in `parse_log` and in `tools/refresh_readme.py`, for entries before
   that banner, on top of the earlier moves.
4. Move or re-seed fitted state at that session: `best_val`, lane fits, anything else read off the old level.
5. Extend `tests/test_heldout_level.py`; run an end-to-end tiny CPU session twice (the first writes the
   line and re-seeds; the second does not).
6. Add item 3 here with the per-subject table; in the README change only the sentence stating today's
   method. Tell the author when to restart the run.
