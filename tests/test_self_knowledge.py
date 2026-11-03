#!/usr/bin/env python3
"""
What the model is told about itself, and what it is not.

  every wording of every answer fills from the model as it is - no unknown
  placeholder, no brace left over - and every topic has more than one
  wording, so a fact is not one string
  a question is written the ways people type it: "What are you?" and
  "what are you" both get an answer
  each topic gets an even share of the lane, and every answer written is one
  of that topic's wordings
  the README's samples - the model's own output - are cut from the copy the
  model reads as itself; the rest of the README stays
  the files the self-description is generated from are not copied at all
"""
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


from corpora.chat import _self_facts, _load_self_knowledge_variants  # noqa: E402
import rebuild_self_knowledge as R                                    # noqa: E402

F = _self_facts()
items = _load_self_knowledge_variants()
bad = []
for ask, says in items:
    for s in says:
        try:
            t = s.format(**F)
            if re.search(r"[{}]", t):
                bad.append(f"{ask[0]}: brace left in '{t[:40]}'")
        except (KeyError, ValueError, IndexError) as e:
            bad.append(f"{ask[0]}: {e!r}")
check("every wording fills from the model as it is", items and not bad,
      f"{len(items)} topics, {sum(len(s) for _, s in items)} wordings" +
      (f"; {bad[:3]}" if bad else ""))
single = [ask[0] for ask, says in items if len(set(says)) < 2]
check("every topic has at least two different wordings", not single, ", ".join(single[:5]))
check("questions are plain: no commas, no stray spaces",
      all("," not in q and q == q.strip() for ask, _ in items for q in ask))

rng = random.Random(0)
forms = {R.surface("what are you", rng) for _ in range(200)}
check("a question is written four ways, \"What are you?\" among them",
      forms == {"what are you", "what are you?", "What are you?", "What are you"},
      str(sorted(forms)))
check("...and a question mark is never doubled",
      all(not R.surface("what are you?", rng).endswith("??") for _ in range(50)))

tmp = tempfile.mkdtemp(prefix="minagi-self-")
try:
    out = os.path.join(tmp, "lane")
    n, written, topics, questions, wordings = R.write_corpus(out, 2_000_000, True)
    text = "".join(open(os.path.join(out, f)).read() for f in sorted(os.listdir(out)))
    filled = {s.format(**F) for _, says in items for s in says}
    answers = re.findall(r"<bot>\n(.*?)\n</bot>", text, re.S)
    check("every answer written is one of the filled wordings",
          answers and all(a in filled for a in answers), f"{len(answers)} answers")
    per_topic = Counter()
    lookup = {s.format(**F): i for i, (_, says) in enumerate(items) for s in says}
    for a in answers:
        per_topic[lookup[a]] += 1
    # drawn at random, so even means within chance of an equal share - not
    # the seven-to-one a topic asked seven ways used to get
    lo, hi = min(per_topic.values()), max(per_topic.values())
    check("each topic gets an even share of the lane",
          len(per_topic) == len(items) and hi <= 2 * lo, f"{lo} to {hi} answers per topic")

    # ---- the source copy: the README without its samples ------------------
    # The fixtures hold placeholder text, not a real sample: this file is
    # itself copied into the lane as source, and a quoted answer here would
    # put back exactly what the test checks is kept out.
    src = os.path.join(tmp, "repo")
    os.makedirs(os.path.join(src, "corpora"))
    readme = ("# Title\n\n<!-- auto:run-blocks -->\n<details>\n"
              "<summary><b>Graph of the whole run so far</b></summary>\n\n"
              "![training progress](assets/training_progress.png)\n\n</details>\n\n"
              "<details>\n<summary><b>Current quality of samples the model generates"
              "</b></summary>\n\n```\n--- lane ---\nprompt: 'PROMPT'\n"
              "[raw]  repeated 8-grams 0%\nGENERATED TEXT ONE\n"
              "[adapted]  repeated 8-grams 0%\nGENERATED TEXT TWO\n```\n\n"
              "</details>\n<!-- /auto:run-blocks -->\n\nProse that stays.\n")
    open(os.path.join(src, "README.md"), "w").write(readme)
    open(os.path.join(src, "corpora", "self_knowledge.yaml"), "w").write(
        "- ask: [QUESTION]\n  say: TEMPLATE {placeholder}\n")
    open(os.path.join(src, "notes.md"), "w").write(
        "Before.\n\n```\n--- lane ---\nprompt: 'PROMPT'\n[raw]  repeated 8-grams 0%\n"
        "GENERATED TEXT THREE\n\nwith a blank line in it\n[adapted]  repeated 8-grams 0%\n"
        "GENERATED TEXT FOUR\n```\n\nAfter.\n\n```\nprint('code stays')\n```\n")
    os.makedirs(os.path.join(src, "memory"))
    open(os.path.join(src, "memory", "history.md"), "w").write(
        "A NUMBER THAT WAS TRUE ONCE.\n")
    copy = os.path.join(tmp, "copy")
    os.makedirs(copy)
    sn, sw = R.write_source(copy, src, True)
    got = {open(os.path.join(copy, f)).read().split("\n")[1]: open(os.path.join(copy, f)).read()
           for f in os.listdir(copy)}
    r = next((t for h, t in got.items() if h.endswith(": README.md")), "")
    check("the README's samples are cut from the copy the model reads",
          r and "GENERATED TEXT" not in r and "[raw]" not in r
          and "Current quality" not in r)
    check("...and the rest of the README stays",
          "Graph of the whole run so far" in r and "Prose that stays." in r)
    notes = next((t for h, t in got.items() if h.endswith(": notes.md")), "")
    check("a sample log quoted in a code block anywhere else is cut too",
          "GENERATED TEXT" not in notes and "blank line" not in notes
          and "Before." in notes and "After." in notes and "code stays" in notes)
    check("the self-description's own sources are not copied",
          not any(h.endswith("self_knowledge.yaml") for h in got), str(sorted(got)))
    check("...nor memory/, the project's history: what it says was true once",
          not any(": memory/" in h for h in got)
          and "TRUE ONCE" not in "".join(got.values()), str(sorted(got)))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all self-knowledge tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
