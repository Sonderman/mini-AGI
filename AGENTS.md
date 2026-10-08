# AGENTS.md — mini-AGI, Ali's local working copy

Scope: local checkout of **volotat/mini-AGI**, kept rebased onto upstream — upstream advanced while
this copy was being set up, and the local branch was **rebased onto `5c57a38` on 2026-10-08** (faster expert
loading in `minagi/paged.py`; README/assets). The 2026-10-07 rebase onto `e6cfda8` had brought reading-speed
work, AMD support + a robust LR controller, the `PYTORCH_ALLOC_CONF` rename fix and `pool.balance` retuned
3.5e-4 -> 2.5e-4; `runs/samples.txt` is kept as ours - upstream appends their own history. Seven local
commits ride on top: `serve` guard + UI defaults, this file, `start.bat`, the self-knowledge passage, the
samples history. The model is a
continually-learning byte-level LM that assembles its own architecture, trains on a single ≥8 GB GPU,
keeps its weights as files on disk and pages them onto the card. This copy is set up and verified to
run on this machine (Windows + RTX 5060 8 GB). It is NOT portable: `.venv/`, `data/` and `weights/`
are machine-local state. `minagi/`, `corpora/`, `train.py`, `serve.py` are upstream: touch them only
for real upstream work, and re-check the local patch after every pull. The 2026-10-05 upstream
commits (re-verified on this machine: existing weights load and train) added `pool.balance` (replaces
`explore_bias`), `pool.select_temperature` and `tools/device_check.py`.

Maintainer: Ali. Remotes: **`origin` = Ali's fork**, `github.com/Sonderman/mini-AGI` (the push
target); **`upstream` = volotat/mini-AGI**. Push to origin only when Ali asks; never push upstream.
Commits only after Ali asks.

## Environment (verified 2026-10-02)

- GPU: RTX 5060 8 GB (sm_120), driver 616.92 / CUDA UMD 13.4. Desktop baseline holds ~2.2 GB of VRAM;
  a training run peaked at ~6.3 GB used in total — a browser left open can push a run into OOM.
- Python: **use `.venv/Scripts/python.exe`** (CPython 3.13.15, uv-managed). Bare `python` is a
  different install and will not have torch.
- torch is **2.9.1+cu128** — deliberately NOT the README's reference `torch 2.6.0+cu124`: the 5060 is
  sm_120 and 2.6/cu124 has no kernels for it. Do not swap torch for a build without cu128+.
- Rebuild the venv (rarely needed; `uv` is available on this machine):
  ```
  uv venv --python 3.13 .venv
  uv pip install -p .venv/Scripts/python.exe "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
  uv pip install -p .venv/Scripts/python.exe numpy pyyaml matplotlib flask chess zstandard datasets scipy
  ```
- Local ignores (`.venv/`, `weights/`) live in `.git/info/exclude`; the tracked `.gitignore` is upstream's.

## Run + verify (commands executed on this machine, expected results inline)

Run from the repo root — the venv python opens `data/` and `weights/` relative to it.

- Build the corpus (already done; lanes on disk are skipped on re-run):
  ```
  .venv/Scripts/python.exe -m corpora all
  ```
  → built `data/train` + `data/val`; the reader sees **86,122 files / 2,646.8M characters** (86,121 + the self-knowledge prime file), `data/` ≈ 2.8 GB.
  `--limit N` for a small trial, `--full` for entire datasets (tens of GB, hours), `--only LANE…` to
  rebuild one lane. The build ends with a spurious `!! self-knowledge produced nothing` (see Warts);
  exit code is still 0.
- Train — the real run, meant to be left alone for days:
  ```
  .venv/Scripts/python.exe train.py read data/train --save --held-out data/val --sample-every 10
  ```
  Verified runs: `--minutes 3` (held-out 3.3342 → 3.0248), `--minutes 10` (step 519 → 1,434), two
  `--minutes 30` sessions (step 1,435 → 4,274 → 6,987), a `--minutes 50` session (step 6,988 →
  11,678 at 9.61M characters) and a `--minutes 240` session (step 11,679 → 31,589 at 40.78M
  characters; held-out 1.6137 → 1.2245; pool 76 → 97). After the 2026-10-05 upstream rebase (new
  `pool.balance` term): a `--minutes 30` session took step 31,590 → 33,716 (4.36M characters;
  held-out 1.2245 → 1.1779; pool 97 → 99), and a `--minutes 120` session took step 33,717 → 42,255
  (17.49M characters; held-out 1.1779 → 1.1339; pool 99 → 108). Its first launch was cut within
  minutes by an interruption; the background job was re-launched on the next Hermes start and ran
  checkpoints are atomic, nothing was lost. A `--minutes 240` session (step 42,256 → 49,411; 14.64M characters; held-out 1.1339 → 1.1260; pool 108 → 115) was cut by a power failure minutes from its end — the last checkpoint held everything; the follow-up `--minutes 30` (49,411 → 51,391; 4.06M characters; pool 115 → 117; held-out 1.1240 → 1.1426, inside the ±0.02 stderr) ran on fresh Adam moments (Warts #4) and rebuilt `optim.npz`. A `--minutes 45` session (51,391 → 54,121; 5.59M characters; held-out 1.1426 →
1.0829 — Δ/SE ≈ 2.0, the largest single drop of the recent runs; pool 117 → 120)
ran on those restored moments. A `--minutes 120` session (54,121 → 58,692; 9.36M characters; held-out 1.0829 →
1.0411; pool 120 → 125) followed: the evening launch died before reading anything and the
host re-started it from the same checkpoint the next morning; its last hour read at ~450 char/s
against ~2,600 before it - external load suspected, not the model. A `--minutes 180` session on the rebased code (58,692 ->
  71,593; 26.42M characters; held-out 1.0411 -> 1.0107; pool 125 -> 138) held **2.7-2.9k char/s**
  across the whole session (no slowdown - upstream's reading-speed commit; `balance 0.00025`) and
  its final samples answer `What are you?` with the primed self-description - the first meaningful
  chat answer. An overnight `--minutes 901` session (71,593 -> 134,197; 128.21M characters; held-out 1.0107 ->
  **0.7942** (-0.2165, ~15 sigma); pool 138 -> 202) held ~2.5-2.8k char/s through the night; its
  greedy samples got rougher despite the loss drop (generation quality is not monotone in loss). A `--minutes 60` session on the new `paged.py` code (134,197 ->
  137,681; 7.14M characters; 0.7939 -> 0.7850; pool 202 -> 205) held ~2.3-2.4k char/s and its samples
  already echo the edited count-free self-knowledge text. A `--minutes 240` session (137,681 -> 151,656; 28.62M characters; 0.7850 ->
  0.7792; pool 205 -> 220) hit a plateau: 4 hours bought only -0.0058 and the train/held-out gap
  closed to ~+0.002; fresh data (data_stage + big chess) is the next lever. Chess samples run
  12-15 legal moves now. Without `--save` it is a dry read and `weights/` is untouched.
  `weights/` is untouched.
- Serve the web UI:
  ```
  .venv/Scripts/python.exe serve.py --port 8080        # → http://127.0.0.1:8080   (HTTP 200 verified)
  ```
  **Learning is ON by default**: chats feed the learner and every 8 steps are saved into `weights/`
  (a streamed reply runs at ~18 char/s; the local reply cap is 2048 chars). Pass `--no-learn` to serve read-only.
  Or double-click **`start.bat`** — checks the venv, refuses to start a second instance, opens the
  browser when the model is ready; extra arguments are forwarded (`start.bat --no-learn`).
- Inspect the model (verified): `.venv/Scripts/python.exe -m minagi.store weights`
  → e.g. `step 151,656`, 220 experts, 224 files, 5993.7 MB (195 carrying + 25 gated; 700.4M params).
  `train.py read --help` lists every knob.

## File map

| Path | What it is | Who changes it |
|---|---|---|
| `minagi/`, `train.py`, `serve.py`, `corpora/`, `replication/`, `tools/`, `config.yaml`, `README.md` | upstream code and docs; README is authoritative on model design and flags | upstream; `serve.py` carries one local patch (below) |
| `.venv/` | this machine's Python + packages (torch 2.9.1+cu128) | rebuild with uv (recipe above) |
| `start.bat` | local launcher for the web UI: venv check, duplicate-server guard, opens the browser when the model is ready; forwards extra args to serve.py | Ali/Hermes; keep in sync with serve.py flags |
| `weights/` | THE model — best state so far (step 151,656, 220 experts, ≈5.99 GB). Written atomically; grows/prunes experts itself | written by `read --save` and by `serve` in learning mode; never hand-edit |
| `data/train`, `data/val` | the corpus the reader opens; subjects = top-level dirs (arithmetic, chat, chess, code, reasoning, stories, wikipedia, self-knowledge (added 2026-10-07; serve's prime source); `chat/hermes` is the OpenHermes part of the chat lane) | `corpora all` / `corpora expand` |
| `data_*_char/` | intermediate uint16 `.bin` sources for `expand` (code, arithmetic, chat, chess; chess rebuilt 2026-10-05: 4M games → 1.95B tokens) | the generators; keep — `expand --only` rebuilds `data/` from them without re-downloading |
| `runs/` | logs, sample history (`samples.txt` is the tracked one), dashboard/progress PNGs, `corpus_index.json` cache | training writes; nothing else there is committed |
| `data_stage/` | staged extra corpus, NOT yet wired into `data/` (staged 2026-10-05 while a long run was reading): supersets of wikipedia/stories/chat/reasoning, a new pg19 book lane (28,602 books) and a `turkce` lane (wiki/web/web2/cosmos/hukuk/chat, ≈17 GB) | Ali/Hermes; wiring (swap + expand) only while nothing reads `data/` |

## Invariants (do not break these)

1. **No push, and no commits without Ali's say-so.** `origin` is Ali's fork (`Sonderman/mini-AGI`) —
   the only push target; `upstream` points at volotat's repo, never push there.
2. **`weights/` is the model.** The first run creates it from `config.yaml`; every later run resumes
   from it (weights + Adam moments + step count) and advances it at every checkpoint — `read` mode writes the current state
   unconditionally (that is how reading progress carries forward; run-final saves set
   `val: null`), while the improve-only save and the divergence-revert live in `stream` mode
   (train.py:308). Deleting it deletes the model.
3. **Do not run two GPU jobs at once** on the 8 GB card (train + serve + another train). Serve falls
   back to CPU when CUDA is full; a training run will just OOM.
4. **Never train while a corpus build is mid-flight.** `corpora expand --only <lane>` deletes that
   lane's text shards first and rewrites them; a reader holding the old file list dies with
   `FileNotFoundError` (observed). Build first, then train.
5. **torch stays cu128+** (sm_120). The README's cu124 or a CPU wheel breaks the GPU path.
6. **The `model:` / `pool.experts` blocks of `config.yaml` are bound into `weights/manifest.json`.**
   Editing them after the weights exist changes nothing (shapes would no longer match); runtime knobs
   (`pool.resident`, `pool.ram_cache`, `pool.capacity_factor`, `training.*`, …) still apply per run.
7. **The reader rotates between top-level dirs of `data/train`.** To teach it your own material, add a
   folder there (or point `read` at any path) — the alphabet is the 256 byte values, nothing to prepare.
8. **`serve.py` learns by default** and writes `weights/` during chats; `--no-learn` serves read-only.

## Local patch (vs upstream)

1. `serve.py`, the pool print in `main()`: upstream calls `pool.vram_params()` unconditionally, but a
   FRESH (non-paged) weights directory loads as a `SharedPool` which has no such method — `serve.py`
   crashed with `AttributeError: 'SharedPool' object has no attribute 'vram_params'` on a fresh model
   (reproduced). Patch: use `vram_params()` when present, else `n_params()`. Re-check after upstream
   updates.
2. `serve.py`, web-UI defaults (Ali's call, 2026-10-08): the embedded UI sends `max_new: 2048`
   (was 400) and the backend default is 2048; `--prime-chars` defaults to **0** (was 1024) - no
   corpus prime / system-prompt-style context unless asked (`--prime-chars N` re-enables).
   Re-check after upstream updates.

## Known upstream warts (cosmetic — do not chase)

1. `corpora all` ends with `!! self-knowledge produced nothing` and a retry hint: stale label — the
   self-knowledge content actually lands in `data/train/chat` (`expand --only chat`); the suggested
   retry just rebuilds the chat text shards.
2. `serve.py`'s prime is OFF by default since 2026-10-08 (Local patch #2). `data/train/self-knowledge/self-01.txt`
   (2 turns; volatile counts stripped 2026-10-08) still exists as the 8th training subject and the model has learned it - its
   samples carry the self-description. With priming on and the dir missing, serve prints
   `no corpus to prime from`; `corpora` never writes the dir itself - `build_self_knowledge` just
   regenerates chat. It is hand-curated and therefore force-added to git
   (tracked despite the `data/` ignore).
3. Benign warnings on this platform: `expandable_segments not supported on this platform` and a
   requires_grad scalar warning in the reader. The deprecated `PYTORCH_CUDA_ALLOC_CONF` name was
   fixed upstream on 2026-10-07 (`d196eb2`/`e6cfda8`).
4. A hard power loss can leave `weights/optim.npz` truncated (`File is not a zip file`) while
   everything else survives: the next `--save` run notes `optimiser moments not restored`,
   starts with fresh Adam moments, and rewrites the file at its first checkpoint — weights
   are unaffected. Observed 2026-10-06.

## State left behind (2026-10-08)

`weights/` = final state after the training runs (**step 151,656, 220 experts - 195 carrying
+ 25 gated, 5,993.7 MB, 700.4M params**; held-out **0.7792** nats after the 4-hour session -
0.7850 before it; best subject chess 0.544, worst wikipedia 1.169). The 4-hour session hit a
plateau (-0.0058, train/held-out gap ~+0.002) - data_stage wiring + the rebuilt chess are the
next lever. Context 3,570 of 4,096; `optim.npz` healthy. `data/` complete but still the SAMPLED
corpus, read 310.6M of 2,646.8M (11.73%); the extended material waits in `data_stage/` (≈37 GB;
wiring in progress 2026-10-08), and `data_chess_char/` has been rebuilt (4M games, 1.95B tokens,
5.5 GB, not yet expanded into `data/`). `data_*` intermediates kept; the web UI runs via `start.bat`
(or scratch/serve_supervisor.py, which restarts it on exit and logs exit codes); the real run is
simply the train command above, left alone for days.
