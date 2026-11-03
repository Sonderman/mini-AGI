# memory

The project's memory, kept for agents: what was built, measured, decided, reversed and corrected, and the
rules work follows here. It preserves history. It is written to be read by an agent picking the project
up at any point, not by a person reading about the model (that is the repository's README), so it is
dense, cross-referenced and not tidied for reading.

Started 2026-10-28 from: the commits of the workshop (2026-09-02 onward; see Scope), the records of the
agent sessions that did the work (2026-09-18 onward), the run's own log (`runs/samples.txt`), the first run's
archive (`runs/archive/run-1`), and three earlier documents moved here (findings.md, architecture.md,
adaptation.md).

## Read in this order

1. `state.md` - where everything stands: the run, the configuration, the lanes, benchmarks, open work and
   known faults, the working tree, the machine. Stale by design; refresh it first.
2. `glossary.md` - the project's words and units, and how to read a check in `runs/samples.txt`.
3. `conventions.md` - how work is done here and why: ownership, the GPU and the weights, commits, the
   README, measuring, publishing. Binding unless the author changes them.
4. `decisions.md` - settled, reversed, rejected and deferred choices, with reasons.
5. `corrections.md` - in-flight corrections of the training run's measurements, and how history was
   carried across them. Read before interpreting any curve or old number.
6. `timeline.md` - where on the run's clock each change landed, and every session's date, characters,
   experts, held-out and rate. Read with corrections.md before reading a curve.
7. `experiments.md` - every test, probe and A/B with its numbers and verdict (E-numbers).
8. `history.md` - dated, oldest first; what changed and why.
9. `changes.jsonl` - the workshop's commits, one JSON line each (date, area, subject, body, files with lines
   added and removed), oldest first, 2026-09-02 to 2026-10-31; grep it by date or by words of a subject for
   a commit's full message. Frozen: later commits are in this repository's git log.
10. `findings.md`, `architecture.md`, `adaptation.md` - documents from early September, each with a status
   header saying what still holds.

## Scope

- Names: "the author" is the project's owner, who decides; "agent" is an AI coding session working on it;
  "the run" is the one continuous training run, begun 2026-09-16; "chars" are characters read (bytes, and the
  tokenizer's markers), the run's clock (3,321M = 3.321B). glossary.md has the rest.
- Until 2026-10-31 the project was developed in a private repository, "the workshop", and `publish.sh`
  copied a filtered subset of it into this public one. Since then this repository is the only one: the
  workshop's files moved here, its git history did not (its commits are changes.jsonl, without hashes).
  Paths here are this repository's. Results under `runs/` other than `runs/samples.txt` are not tracked,
  so a `runs/` path names a file on the machine that made it.
- It is excluded from the self-knowledge lane (`tools/rebuild_self_knowledge.py`): history is full of
  numbers and designs that are no longer true, and the model must not learn them as facts about itself.
- Numbers are as measured at the time; a later entry supersedes an earlier one. Dates are absolute.

## Public

Everything here is published. So it holds nothing personal and nothing that could harm or embarrass the
author:
- no personal information: no names beyond "the author", no contact details, no accounts other than the
  project's public ones, nothing about the author's life, plans, money or other work;
- no paths outside the repository (nothing that says where files live on anyone's machine) - paths here
  are relative to the repository root;
- no credentials, keys or tokens, and nothing read from `.env`, which agents never open;
- none of the author's own material that the repository does not include, nor its kinds or the folders
  it is kept in;
- none of the author's wording, quoted or close to it: only the intent behind a decision, and its reason;
- no workshop commit hashes: its history is not public, so they lead nowhere. Name the change and its
  date. A hash of this repository's own commits may follow the name once the commit is on `main` (since
  2026-10-31): those resolve on GitHub, while a branch's may be squashed away.
changes.jsonl was generated under the same rules: commits mostly of the author's own material, or about it,
were left out; those files and the sentences that name that material were dropped from the others; names and
machine paths were scrubbed; a commit is a date and a subject, and a hash in a message became the subject of
the commit it names; dates carry no time of day.

## Keeping it

- With a change: a dated entry in history.md for anything someone picking the project up should know; the
  author commits it with the change. changes.jsonl is not extended; git has the commits.
- A new test or A/B gets the next E-number in experiments.md: setup, numbers with their noise, verdict.
- A decision or a reversal goes in decisions.md with its reason; a measurement fix in corrections.md as the
  next numbered item; a new working rule in conventions.md with its reason and date.
- Refresh state.md when the run's state, configuration or the next steps change. A change that reaches the run
  gets a row in timeline.md at the first session that started after it, and the session table its new rows
  (the commands are in its header). A new word gets a line in glossary.md.
- Follow "Public" above for every line. Never paste model output here as if it were documentation.
- Write in the same terse style; date every change and name it (a hash of `main` may follow, never
  instead); prefer adding to rewriting.
