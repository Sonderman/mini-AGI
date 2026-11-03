#!/usr/bin/env python3
"""
Every corpus the model reads must have a sample prompt, in its own format.

THE BUG THIS IS WRITTEN AGAINST. For most of this run the sample set was code,
arithmetic, chat and chess - four prompts against seven corpora. `stories`,
`reasoning` and `wikipedia` were never generated from. That matters because the
domains differ enormously: measured at 271M characters, stories reads 1.24 bits
per character and wikipedia 2.29. Coherence appears in the easiest domain
first, so judging the model on the chat prompt alone understates it, and the
one domain where something readable would show up was the one never sampled.

A prompt must also LOOK like its corpus. The reasoning traces open the answer
with <think>, wikipedia is raw MediaWiki markup with [[links]] and ==headings==,
stories are plain simple-vocabulary narrative with no markers at all. A prompt
in the wrong shape puts the model off distribution before it writes a
character, and the sample then says nothing about the model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train                                              # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prompts = dict(train.SAMPLE_PROMPTS)

# ---------------------------------------------- 1. every corpus is covered
for split in ("train", "val"):
    d = os.path.join(ROOT, "data", split)
    if not os.path.isdir(d):
        print(f"  [skip] data/{split} not present")
        continue
    corpora = sorted(x for x in os.listdir(d)
                     if os.path.isdir(os.path.join(d, x)))
    missing = sorted(set(corpora) - set(prompts))
    check(f"every corpus in data/{split} has a sample prompt",
          not missing, f"missing: {missing}" if missing else
          f"{len(corpora)} corpora, all covered")

# ------------------------------------- 2. names are unique and non-empty
names = [n for n, _ in train.SAMPLE_PROMPTS]
check("prompt names are unique", len(names) == len(set(names)), str(names))
check("no prompt is empty", all(p.strip() for _, p in train.SAMPLE_PROMPTS))

# --------------------------- 3. each prompt is shaped like its own corpus
# Not a guess: these markers were read out of data/val/<corpus>/ directly.
SHAPE = {
    "chat":        lambda p: p.startswith("<user>") and p.rstrip().endswith("<bot>"),
    "self-knowledge": lambda p: (p.startswith("<user>")
                                 and p.rstrip().endswith("<bot>")),
    "chat_hermes": lambda p: p.startswith("<user>") and p.rstrip().endswith("<bot>"),
    "reasoning":   lambda p: p.startswith("<user>") and "<think>" in p,
    "chess":       lambda p: p.startswith("<g>"),
    "wikipedia":   lambda p: "[[" in p or "==" in p,
    "stories":     lambda p: "<" not in p and "[[" not in p,
    "code":        lambda p: "def " in p or "import " in p,
    "arithmetic":  lambda p: any(c in p for c in "+-*/") and "=" in p,
}
for name, prompt in train.SAMPLE_PROMPTS:
    if name not in SHAPE:
        continue
    check(f"the {name} prompt is in its corpus's format",
          SHAPE[name](prompt), repr(prompt[:52]))

# ------------------- 4. a prompt must not be LIFTED from the held-out set
# Only prompts long enough to carry content are testable this way. A short one
# is mostly format, and format is supposed to be shared - test 3 above demands
# it. Measured: "Once upon a time, there was a" appears in 20 of 20 story files
# because that is how TinyStories opens, and "1. e4 e5" is the commonest
# opening in chess. Flagging those would be penalising a prompt for being
# correctly shaped. What is worth catching is a whole prompt copied verbatim,
# where the continuation would measure recall rather than generation.
LONG = 55
for name, prompt in train.SAMPLE_PROMPTS:
    d = os.path.join(ROOT, "data", "val", name)
    if not os.path.isdir(d) or len(prompt.strip()) < LONG:
        continue
    probe = prompt.strip()
    hit = next((f for f in sorted(os.listdir(d))[:20]
                if os.path.isfile(os.path.join(d, f))
                and probe in open(os.path.join(d, f), errors="ignore").read()),
               None)
    check(f"the {name} prompt is not copied verbatim out of data/val",
          hit is None, "" if hit is None else f"found in {hit}")

print("\n  all sample-coverage tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
