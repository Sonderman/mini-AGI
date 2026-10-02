# AGENTS.md — mini-AGI, Ali's local working copy

Scope: local checkout of **volotat/mini-AGI** (HEAD `7361e7a "training process update"`, cloned
2026-10-02) — a continually-learning byte-level language model that assembles its own architecture,
trains on a single ≥8 GB GPU, keeps its weights as files on disk and pages them onto the card. This
copy is set up and verified to run on this machine (Windows + RTX 5060 8 GB). It is NOT portable:
`.venv/`, `data/` and `weights/` are machine-local state. `minagi/`, `corpora/`, `train.py`, `serve.py`
are upstream: touch them only for real upstream work, and re-check the local patch after every pull.

Maintainer: Ali. The clone kept its `origin` remote, but this copy is local-only: **never push**.
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
  → built `data/train` + `data/val`; the reader sees **86,121 files / 2,646.8M characters**, `data/` ≈ 2.8 GB.
  `--limit N` for a small trial, `--full` for entire datasets (tens of GB, hours), `--only LANE…` to
  rebuild one lane. The build ends with a spurious `!! self-knowledge produced nothing` (see Warts);
  exit code is still 0.
- Train — the real run, meant to be left alone for days:
  ```
  .venv/Scripts/python.exe train.py read data/train --save --held-out data/val --sample-every 10
  ```
  Verified short form `… train.py read --save --minutes 3`: exit 0, **~3.2k char/s**, checkpoint written
  at step 519, held-out 3.3342 → 3.0248. Without `--save` it is a dry read and `weights/` is untouched.
- Serve the web UI:
  ```
  .venv/Scripts/python.exe serve.py --port 8080        # → http://127.0.0.1:8080   (HTTP 200 verified)
  ```
  **Learning is ON by default**: chats feed the learner and every 8 steps are saved into `weights/`
  (a streamed reply runs at ~18 char/s). Pass `--no-learn` to serve read-only.
- Inspect the model (verified): `.venv/Scripts/python.exe -m minagi.store weights`
  → e.g. `step 519`, 65 experts, 69 files, 2091.9 MB. `train.py read --help` lists every knob.

## File map

| Path | What it is | Who changes it |
|---|---|---|
| `minagi/`, `train.py`, `serve.py`, `corpora/`, `replication/`, `tools/`, `config.yaml`, `README.md` | upstream code and docs; README is authoritative on model design and flags | upstream; `serve.py` carries one local patch (below) |
| `.venv/` | this machine's Python + packages (torch 2.9.1+cu128) | rebuild with uv (recipe above) |
| `weights/` | THE model — best state so far (step 519, 65 experts, ~2.1 GB). Written atomically; grows/prunes experts itself | written by `read --save` and by `serve` in learning mode; never hand-edit |
| `data/train`, `data/val` | the corpus the reader opens; subjects = top-level dirs (arithmetic, chat, chess, code, reasoning, stories, wikipedia; `chat/hermes` is the OpenHermes part of the chat lane) | `corpora all` / `corpora expand` |
| `data_*_char/` | intermediate uint16 `.bin` sources for `expand` (code, arithmetic, chat, chess) | the generators; keep — `expand --only` rebuilds `data/` from them without re-downloading |
| `runs/` | logs, sample history (`samples.txt` is the one tracked file) and the `corpus_index.json` file cache | training writes; nothing else there is committed |

## Invariants (do not break these)

1. **No push, and no commits without Ali saying so.** `origin` exists only because this was cloned.
2. **`weights/` is the model.** The first run creates it from `config.yaml`; every later run resumes
   from it (weights + Adam moments + step count) and advances it only when it improves on what is
   there. Deleting it deletes the model.
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

## Known upstream warts (cosmetic — do not chase)

1. `corpora all` ends with `!! self-knowledge produced nothing` and a retry hint: stale label — the
   self-knowledge content actually lands in `data/train/chat` (`expand --only chat`); the suggested
   retry just rebuilds the chat text shards.
2. `serve.py` prints `no corpus to prime from`: `load_prime()` expects `data/train/self-knowledge/`
   files that the current corpora never writes. Harmless; `--prime-chars 0` silences it.
3. Benign warnings on this platform: `expandable_segments not supported on this platform`,
   `PYTORCH_CUDA_ALLOC_CONF is deprecated` (upstream sets the old name), and a requires_grad scalar
   warning in `minagi/recur.py`.

## State left behind (2026-10-02)

`weights/` = best state after the verification reads (step 519, 65 experts, 2091.9 MB, pool grew 64→65);
`data/` complete; `data_*` intermediates kept; no background processes left running. The real run is
simply the train command above, left alone for days.
