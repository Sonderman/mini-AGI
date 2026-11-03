# conventions

Rules for working on this project, each with the reason it exists. Most were set by the author after
something went wrong; the date says when. An agent picking the project up should treat these as binding
unless the author changes them.

## Ownership and authority

- The author owns direction and decisions; agents implement, measure and report. When the author asks
  only for an explanation, a review or a brainstorm, or says not to change or implement anything yet, do
  exactly that.
- Code: agents write nearly all of it (since 2026-09-27), and the author commits all of it (since
  2026-10-31, Commits), so the working tree holds work waiting for review: an agent's from an earlier
  session (understand it, finish it, hand it over) and the author's hand edits - `README.md` (prose and the
  author's own lines), `config.yaml` values (e.g. `pool.balance`, `pool.capacity_factor`), and now and then
  a test, a table header or a file deletion. Read `git diff` before changing a file, and never revert or
  rework the author's hunks.
- Reverted means decided (2026-10-23): if files an agent edited are found back at an earlier state, the author put
  them there on purpose. Never re-apply reverted work. Read an ambiguous follow-up, such as a bare request to try
  again, as a question, not permission. Incident: on 2026-10-23 an agent re-applied reverted work and had to undo
  it.
- Decisions recorded in decisions.md are settled; re-propose only with new evidence, and say what is new.

## The GPU and the weights

- The GPU belongs to the author. Before any GPU work check whether the run is live (`nvidia-smi`; scan
  `/proc/*/cmdline` for `train.py` with `read`). If it is, ask or work on CPU (`CUDA_VISIBLE_DEVICES=""`,
  `nice -n 19`). The 8 GB card holds one job.
- Never run anything with gradients against `weights/` directly. Paged experts are written back on
  eviction, so even a run without `--save` modifies the model; `weights/` is months of training and cannot
  be rebuilt. Work on a copy: `cp -r weights <scratch>/w` (full copy) or
  `cp -al` (hard links are safe: every save writes a `.tmp` and renames, breaking the link), and point
  `--weights-dir` at it. Verify `weights/` unchanged afterwards (file count, sizes, step, a fingerprint).
- A tool that only reads `weights/` while the run may own it opens it with `read_only=True`
  (`train.py build_paged`, `minagi/recur.py load_any`): paging an expert in marks it dirty, so a plain read
  writes expert files back under the run. Every measuring and inspection tool does, and the inspection tools
  default to `weights/` (both since 2026-10-31); a new tool passes it. It stops write-backs only: a prune
  deletes expert files whatever it says, so a tool never prunes or grows the pool.
- `serve.py` learns from the conversation and writes `weights/` by default. With the run training, give it a
  copy (`--weights`) or `--no-learn` (read-only), and `--device cpu` unless the author has freed the card.
- `weights/` is the run's latest state, saved at every check and every 5 minutes, not its best: nothing falls
  back to an earlier state by itself (`--revert-factor` belongs to `train.py stream`, not `read`), so a
  damaged `weights/` comes back only from the author's backups.
- Measurements that learn (dynamic evaluation, best efforts, A/B arms) always run on a scrapped copy that
  is deleted after scoring; nothing learned from a benchmark reaches `weights/`.
- Disk space is limited: a saved arm rewrites ~9 GB of expert files, and seven arms once filled the disk.
  Check `df` before each arm (>= 20 GB free) and delete each copy as soon as it is scored.
- RAM is shared with the run (its RSS is ~12 GB at a 288-expert RAM tier). Give every isolated run a
  memory watchdog (SIGINT at a soft RSS limit) so the OOM killer never fires.
- Weight backups (`weights_*_bkp/`) are the author's; never delete or modify them.

## Running things

- No blocking waits (2026-09-05): never sit in `until ! pgrep ...; do sleep; done`. Run long jobs in the
  background, give an ETA, report as results land. From the author's side a blocking wait looks like a hang.
- `pkill -f`/`pgrep -f` patterns that match the agent's own shell kill it (happened many times): kill by
  verified PID; anchor patterns (`^python3 train.py`) - under py-spy the profiler matches first.
- A running bash script is replaced by rename (new inode), never edited in place.
- Isolated `train.py read` runs: pass `--history`, `--sample-log`, `--loss-log` into scratch,
  `--no-plots`, a copy of the weights, and copy `runs/corpus_index.json` into a scratch tree (else a
  ~3 minute rescan). An arm without `--save` leaves the trunk unsaved, so scoring it measures the start.
- Tests are scripts: `python3 tests/test_x.py` prints PASS/FAIL lines and exits non-zero on failure. Run
  them with `CUDA_VISIBLE_DEVICES=""` while the run trains.
- Downloads (datasets, models) need the author's go-ahead in the session; the corpus builders download
  only when run.

## Commits

- Agents never stage, commit or push (2026-10-31): the author examines every change and makes every commit.
  (In the workshop agents committed freely, 2026-09-27 .. 10-31.) Leave finished work in the working tree
  and hand it over: what changed and why, how it was tested, which hunks are the agent's where a file also
  holds the author's edits, and a proposed commit for each group of changes - subject
  `area: lowercase description` (pool:, train:, readme:, corpora:, tools:, release:, plots:, held-out:,
  model:, optim:, ...), an explanatory body (what was wrong, what changed, measured result), and a
  Co-authored-by line for each contributor whose work it carries. Branch `main` (the workshop's was `master`).
- What the running program writes - `runs/samples.txt`, `weights/manifest.json`, the generated figures
  (`assets/*.png`, `assets/layers.svg`), the README's `auto:` regions - is proposed apart from code, as a
  deliberate `runs:` commit after `./release.sh` has refreshed it.
- With a change that matters to someone picking the project up, a dated entry in `memory/history.md`, committed
  with it. `memory/changes.jsonl` is the workshop's log, frozen (memory/README.md).

## The README

- The README working copy is the author's (2026-10-21): it differs from HEAD by the author's uncommitted edits plus
  regenerated auto regions, which exist nowhere else. Before changing it: copy it to scratch, read it in
  full, check its length, write atomically. `open("README.md","w").write(...open("README.md").read())`
  truncated it to 0 bytes once (restored then from the copy publish.sh had written): git has only what was
  committed, so the scratch copy is the only record of the rest.
- Numbers in the README are generated, never typed (`tools/refresh_readme.py` auto regions; `./release.sh`
  and `./measure_benchmarks.sh` regenerate them). A stale typed number looks maintained and nobody re-checks
  it.
- The README is for people: plain statements, no slogans, no changelog in figures, no "restart" on graphs
  (a resume is a pause), references say only how each was used. History, corrections and
  agent-facing detail live here in memory/.
- The README links only to files in the repository or to what the program creates as it runs (`runs/`,
  `data/`, `weights/`); `./release.sh` checks.
- Labels and wording are plain and exact: not "then/now", no rhetorical questions in titles ("Catastrophic,
  or nothing at all?" was rejected), numbers instead of words like "net improvement".

## Code and comments

- Comments describe the mechanism, not how it was found (2026-09-19). config.yaml comments say
  only what each knob does - no history, no comparisons. Removing code needs no explanation of
  why it went.
- A comments-only or explain-only request changes no code; an agent that changed `__all__` during one had
  to revert it.
- A bug fix comes with a test that fails when the fix is reverted (verify it by reverting, as for
  tests/test_pool_api.py and tests/test_corpora_contract.py); tests that only test themselves were found in
  a walkthrough of all 60 tests (2026-10-21) and rewritten against the extracted functions.

## Repo text is training data

- `tools/rebuild_self_knowledge.py` copies every `.py`/`.md`/`.yaml` of the repo into
  `data/train/self-knowledge` as "part of what I am made of" (except `data/`, `runs/`, `weights*`,
  `memory/`, and the files the self-description is generated from). That lane is read like any subject.
  It walks the folder, not git: an untracked file inside the repository's folder is copied too.
- So: never put model output or realistic model-like text in repo files (tests use placeholders such as
  "GENERATED TEXT"); never leave stale claims about the model in docs (the model learned "up to 6 passes"
  from a quoted sample, 2026-10-08). `memory/` is excluded precisely because history is full of numbers
  that are no longer true.
- After editing `corpora/self_knowledge.yaml` run `tests/test_self_knowledge.py`; rebuild the lane only
  with no training running.

## Measuring and reporting

- Held-out differences need their noise: the evaluator returns stderr; run-to-run recipe variance ~0.014
  on CUDA; "~0.03 is the threshold for a real difference" for separate runs; paired A/B arms
  (tools/ab_pool.py, identical text) resolve ~+/-0.0005 held-out and +/-0.00003 paired training loss.
- A 12-30M-character test sees level shifts, not changes of the power-law exponent (late alpha ~0.1 moves
  held-out ~0.0002 per 12M chars).
- Selection A/Bs always report "experts used at least once" (of the pool) beside experts carrying 99% of
  picks and the top-share (2026-10-02).
- A measurement fix adjusts history (2026-10-17/18): measure old and new methods side by side on the same
  weights at the switch, move earlier values by the per-subject difference in reporting/plotting code only,
  never rewrite the raw log, no markers or footnotes on graphs; re-seed any state fitted to the old level.
  Details and both corrections so far: corrections.md.
- Comparison tables carry a training-cost column (2026-10-21): what each model's makers paid; when
  unpublished, 6ND on the hardware of the time at 40% utilisation, at a big cloud's on-demand price then,
  linked to a dated price source, marked (est.); the cheaper clouds' price may go in the folded notes as a
  lower bound. Check an estimate's GPU-hours against a published run (Pythia-1B: 4,830 A100-hours for 300B
  tokens). mini-AGI at $0.081/h (cheapest uninterruptible RTX 3070 rental), hours as of the measured weights
  (`tools/refresh_readme.py` `ran_until`), not to date.
- The model's static scores are context, not the verdict: its claim is continual learning (adaptation).
  Continual-learning benchmarks are planned; suggest a fine-tuned standard transformer baseline then.
- Held-out and enwik9/PG19 numbers are the model's own PonderNet objective (CE mixture + 0.01 KL halting
  term), so they are upper bounds on code length.

## The model as it runs

- The learning-rate controller is part of the model (2026-10-23): every path that learns (training, chat,
  dynamic evaluation, best efforts) goes through `minagi/governor.py` and behaves exactly as in training.
  Never add a learning path that bypasses it.
- Precision: fp16 never - the recurrent state reaches ~9,920 at arithmetic delimiters and RMSNorm squares it
  (98.5M). A GPU without bf16 arithmetic computes fp32 when it has >= 14 GiB, else emulated bf16; decide
  per device with `tools/device_check.py`.
- The notifier, `minagi/notify.py` (since 2026-09-11), sends one line of numbers to Telegram's Bot API when
  held-out reaches a new best. It is inert unless both MINAGI_TELEGRAM_TOKEN and MINAGI_TELEGRAM_CHAT are set,
  in the environment or in a gitignored `.env` in the project root: no network call, no log line. Every failure
  is swallowed, so it cannot stop a run. It ships with the code, but the README and config.yaml do not mention
  it (the author's choice); if a user asks for notifications, help them set it up: `python3 -m minagi.notify`
  checks the token and finds the chat id. Removing it is deleting the file and the five marked lines in
  train.py.
- Agents never read `.env` or any other credential file; credentials are never committed or published.

## Deliverables

- Deliverables are files in the working directory, never hosted pages (2026-09-02).
- The author's own material is not part of the project: it is not tracked, and neither the README, the
  code nor this folder mentions it (the code since 2026-10-31).

## Publishing

- One public repository (since 2026-10-31): what is tracked is what a push carries, and `memory/` is
  tracked, so it follows memory/README.md, "Public". Until then `publish.sh` copied a filtered subset of
  the workshop here (decisions.md, Publishing).
- Before a push, `./release.sh` refreshes what the README reports: held-out measured on the weights as they
  are (so it refuses while the run trains), the scaling chart, the architecture figure, the auto regions.
  Then it checks that every tracked `.py` compiles, that no `.env` is tracked and that the README's links
  lead somewhere; `./release.sh --check` runs the checks alone, at any time. It never commits or pushes.
- `publish_weights.sh` pushes a weights snapshot to Hugging Face (`Volotat/mini-AGI-undertrained`) from a
  clone of it beside the repository's folder (or wherever `DEST` says), with the run stopped; README.md then
  carries the snapshot's date, to commit. Undertrained snapshots may be deleted after the first pass.
- `./measure_benchmarks.sh` (about a day of the card) re-measures the enwik9/PG19 rows and refreshes the
  README; `./release.sh` uses its last results.
