#!/usr/bin/env python3
"""
The README's enwik9 table, and the measurement behind its mini-AGI row.

What these tests hold it to:
  the rows run best first, the best score - and only it - is in bold, every
  published row links its paper, the notes trace every published score to
  the figure its paper prints, and say how many pieces this model was
  scored on and how far off that can be
  if this model is the best, its score is the bold one
  a measurement of other weights leaves the table exactly as it was
  tools/measure_enwik9.py reuses a measurement of the same weights on at
  least as many pieces, and measures again otherwise
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import refresh_readme as RR                                 # noqa: E402
import measure_enwik9 as ME                                 # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


m = {"bpb": 1.31500061795693, "se": 0.0086, "pieces": 1000, "slice_bytes": 50_000_000,
     "date": "2026-10-17 16:20", "chars": 2.31e9}
t = RR.enwik9_table(m, 1_084_285_783, 610)
rows = [l for l in t.splitlines() if l.startswith("| ") and "---" not in l][1:]
check("the rows run best first, and only the best score is in bold",
      [r.split("|")[1].strip() for r in rows]
      == ["Llama-3.2-1B", "Chinchilla 1B",
          "[GPT-2 XL](https://huggingface.co/openai-community/gpt2-xl)",
          "[GPT-2 Large](https://huggingface.co/openai-community/gpt2-large)",
          "[GPT-2 Medium](https://huggingface.co/openai-community/gpt2-medium)",
          "mini-AGI (frozen)",
          "[GPT-2 Small](https://huggingface.co/openai-community/gpt2)"]
      and sum(r.count("**") for r in rows) == 4 and "**0.68**" in rows[0],
      f"{rows}")
check("...every published row links its paper, and the measured ones say so",
      "(https://arxiv.org/abs/2601.02875)" in rows[0]
      and "(https://arxiv.org/abs/2309.10668)" in rows[1]
      and "| 1.56B | Feb 2019 | 40 GB, passes not published | ~$43k | measured | 1.07 |" in rows[2]
      and "| 0.77B | Feb 2019 | 40 GB, passes not published | ~$21k (est.) | measured | 1.12 |" in rows[3]
      and "| 0.35B | Feb 2019 | 40 GB, passes not published | ~$10k (est.) | measured | 1.17 |" in rows[4]
      and "| 1.08B | 17 Oct 2026 | 2.3 GB | **~$49** | measured | 1.32 |" in rows[5]
      and "| 0.12B | Feb 2019 | 40 GB, passes not published | ~$3k (est.) | measured | 1.33 |" in rows[6])
check("...the score column says lower is better, the cheapest training is in "
      "bold, and a best-effort estimate says it is one",
      "| model | params | date | trained on | training cost ↓ | scored in | "
      "enwik9 bits/byte ↓ |" in t and "| ~$0.9M |" in rows[0]
      and "| ~$10k (est.) |" in rows[4]
      and sum(r.count("**~$") for r in rows) == 1
      and "best-effort estimates" in t)
check("...and this model's hours are priced at the rental the note names, on "
      "the page it links",
      "610 hours on one RTX 3070 Laptop GPU at $0.081 an hour" in t
      and "(https://computeprices.com/providers/vast/gpus/rtx3070)" in t
      and "Vast.ai's" in t)
check("...and what nobody published is estimated on a stated assumption, "
      "checked against a model that did publish its hours",
      "| ~1.4T tokens (~5.6 TB, est.) | ~$68k (est.) |" in rows[1]
      and "assumes it read as much as Chinchilla 70B did" in t
      and "Table 5" in t and "about 22,500 for 1.4T" in t)
check("...and every note is folded away under the table",
      "<details>" in t and "</details>" in t
      and t.index("<details>") > t.index(rows[-1])
      and t.index("**Date**") > t.index("<details>")
      and t.index("± 0.009") < t.index("</details>"))
check("...and says when each model came out, this one by the day it was "
      "measured",
      [r.split("|")[3].strip() for r in rows]
      == ["Sep 2024", "Mar 2022", "Feb 2019", "Feb 2019", "Feb 2019",
          "17 Oct 2026", "Feb 2019"]
      and "**Date** is when each model came out" in t)
check("...and the models measured here say so, and how, with each one's error",
      "GPT-2 XL, GPT-2 Large, GPT-2 Medium and GPT-2 Small are measured here "
      "the same way as mini-AGI - the GPT-2 paper reports enwik8, not enwik9 "
      "- on 2,000 pieces of the same slice" in t
      and "on the same 2,000 pieces" in RR.enwik9_table(dict(m, pieces=2000),
                                                        1_084_285_783, 610)
      and "± 0.005 for GPT-2 XL, 0.005 for GPT-2 Large, 0.005 for GPT-2 "
      "Medium, 0.006 for GPT-2 Small." in t)
saved = RR.ENWIK9_MEASURED
RR.ENWIK9_MEASURED = [dict(e, se=0.005) for e in saved]
t3 = RR.enwik9_table(m, 1_084_285_783, 610)
RR.ENWIK9_MEASURED = saved[:1]
t4 = RR.enwik9_table(m, 1_084_285_783, 610)
RR.ENWIK9_MEASURED = saved
check("...one error for all when they agree, and one model alone as one",
      "bytes, ± 0.005 for each." in t3
      and "- GPT-2 XL is measured here the same way as mini-AGI" in t4
      and "bytes, ± 0.005." in t4)
check("...and the note says how many pieces, and how far off that can be",
      "1,000 pieces" in t and "50 MB slice" in t and "± 0.009" in t)
check("...and traces every published score to the figure its paper prints",
      "Table 2, row \"Llama-3.2-1B\"" in t and "**0.085**" in t
      and "0.085 × 8 = 0.68" in t and "Table 1, row \"Chinchilla 1B\"" in t
      and "**11.3%**" in t and "0.113 × 8 = 0.90" in t)

t2 = RR.enwik9_table(dict(m, bpb=0.5), 1_084_285_783, 610)
rows2 = [l for l in t2.splitlines() if l.startswith("| ")][1:]
check("if this model is the best, its score is the bold one",
      rows2[0].startswith("| mini-AGI") and "**0.50**" in rows2[0]
      and sum(r.count("**0.") for r in rows2) == 1)

tmp = tempfile.mkdtemp(prefix="minagi-enwik9-")
try:
    os.makedirs(os.path.join(tmp, "runs"))
    os.makedirs(os.path.join(tmp, "weights"))
    with open(os.path.join(tmp, "weights", "manifest.json"), "w") as f:
        json.dump({"step": 8, "read_chars": 3.1e6, "n_experts": 10}, f)
    with open(os.path.join(tmp, "runs", "enwik9_measured.json"), "w") as f:
        json.dump(dict(m, step=7, chars=3.1e6, experts=10), f)
    readme = "a\n<!-- auto:enwik9 -->\nold table\n<!-- /auto:enwik9 -->\nb\n"
    RR.ROOT = tmp
    out, k = RR.enwik9(readme)
    check("a measurement of other weights leaves the table exactly as it was",
          out == readme and k)

    man = {"step": 7, "read_chars": 3.1e6, "n_experts": 10}
    path = os.path.join(tmp, "runs", "enwik9_measured.json")
    h = os.path.join(tmp, "runs", "scaling", "enwik9_history.jsonl")
    check("the measuring tool reuses a measurement of the same weights on at "
          "least as many pieces, and measures again otherwise",
          ME.already(man, 1000, out=path, hist=h) is not None
          and ME.already(man, 500, out=path, hist=h) is not None
          and ME.already(man, 2000, out=path, hist=h) is None
          and ME.already(dict(man, step=8), 1000, out=path, hist=h) is None)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# learning while it reads: its own row, marked, and the mark explained
# straight under the table
dyn = {"pieces": 2000, "piece_bytes": 2048, "date": "2026-10-19 12:37",
       "chars": 2.458e9, "frozen_bpb": 1.30914, "frozen_se": 0.0058,
       "adapted_bpb": 1.25146, "adapted_se": 0.0065, "gain_bpb": 0.05769,
       "gain_se": 0.00148, "gain_by_quarter": [0.042, 0.060, 0.061, 0.068],
       "lr": 8.3585e-06, "weights_unchanged": True}
td = RR.enwik9_table(dict(m, bpb=1.30914), 1_257_000_000, 633, dyn)
rows_d = [l for l in td.splitlines() if l.startswith("| ") and "---" not in l][1:]
names_d = [r.split("|")[1].strip() for r in rows_d]
under = td[td.index(rows_d[-1]) + len(rows_d[-1]):td.index("<details>")]
check("learning while it reads gets its own row, marked, in its place by "
      "score, and the frozen row stays",
      names_d[5:7] == ["mini-AGI (dynamic evaluation)\\*",
                       "mini-AGI (frozen)"]
      and "| 1.26B | 19 Oct 2026 | 2.5 GB | **~$51** | measured | 1.25 |"
      in rows_d[5] and rows_d[6].endswith("| 1.31 |"), f"{names_d}")
check("...the mark is explained straight under the table: learning while it "
      "reads, scored before learning, what the same weights score frozen, "
      "that the learned weights are thrown away, and that for a continual "
      "learning model it is simply how it runs",
      under.strip().startswith("\\* **Learning while it reads**")
      and "scored first and then learned from" in under
      and "Read frozen, the same weights score 1.31" in under
      and "thrown away after the measurement" in under
      and "nothing learned from enwik9 ever reaches the weights" in under
      and "it is a continual learning model" in under
      and "(https://arxiv.org/abs/1709.07432)" in under
      and "closer to online compressors" in under, under)
check("...and the notes say how it was measured, and by how much learning "
      "helped",
      "one optimiser step on it at 8.4×10<sup>-6</sup>" in td
      and "the same weights score 1.309, and learning took 0.058 ± 0.0015 "
      "off - 0.042 over the first quarter" in td
      and "so it was still gaining at the end." in td)
check("...and without one the table is as it was",
      "dynamic evaluation" not in RR.enwik9_table(m, 1_084_285_783, 610))

# which measurement the row comes from
tmp2 = tempfile.mkdtemp(prefix="minagi-enwik9-dyn-")
try:
    os.makedirs(os.path.join(tmp2, "runs"))
    saved, RR.ROOT = RR.ROOT, tmp2
    none = RR.enwik9_dynamic()
    with open(os.path.join(tmp2, "runs", "enwik9_adapt.jsonl"), "w") as f:
        for d in (dict(dyn, date="a"), dict(dyn, date="b", piece_bytes=64),
                  dict(dyn, date="c", pieces=500),
                  dict(dyn, date="d", weights_unchanged=False)):
            f.write(json.dumps(d) + "\n")
    got = RR.enwik9_dynamic()
    RR.ROOT = saved
finally:
    shutil.rmtree(tmp2, ignore_errors=True)
check("the row is the latest measurement on the standard pieces with the "
      "weights left unchanged - none, none",
      none is None and got is not None and got["date"] == "a")

# the best effort: whole pages in its own window, learning as it reads
nat = {"regime": "natural", "pages": 1429, "budget": 4096000,
       "bytes": 3969348, "steps": 2862, "chunk": 2048, "context": 4096,
       "date": "2026-10-19 22:35", "chars": 2.5014e9, "alone_bpb": 1.27271,
       "frozen_bpb": 1.21434, "adapted_bpb": 1.19025,
       "weights_unchanged": True}
tn = RR.enwik9_table(dict(m, bpb=1.30914), 1_257_000_000, 633, dyn, nat)
rows_n = [l for l in tn.splitlines() if l.startswith("| ") and "---" not in l][1:]
names_n = [r.split("|")[1].strip() for r in rows_n]
under_n = tn[tn.index(rows_n[-1]) + len(rows_n[-1]):tn.index("<details>")]
check("the best effort gets a row of its own, marked twice, in its place by "
      "score beside the other two",
      names_n[4:8] == [
          "[GPT-2 Medium](https://huggingface.co/openai-community/gpt2-medium)",
          "mini-AGI (best effort)\\*\\*",
          "mini-AGI (dynamic evaluation)\\*", "mini-AGI (frozen)"]
      and rows_n[5].endswith("| measured | 1.19 |"), f"{names_n}")
check("...the second mark explained under the first: the model used exactly "
      "as in training and conversation, in its native window, on a copy "
      "thrown away, and not on the same footing as the rest",
      under_n.index("\\*\\* **Best effort**")
      > under_n.index("\\* **Learning while it reads**")
      and "exactly as it is used in training and in conversation" in under_n
      and "native 4,096-byte window, sliding 2,048 characters at a time"
      in under_n
      and "thrown away afterwards" in under_n
      and "not on the same footing" in under_n, under_n)
check("...and the notes say how it read, and what the same pages score "
      "frozen",
      "1,429 of the slice's pages, spread evenly over it" in tn
      and "the same pages score 1.214, and chunk by chunk alone 1.273" in tn)
check("...and without one, no second mark",
      "Best effort" not in td and "best effort" not in td)
ta = RR.enwik9_table(dict(m, bpb=1.30914), 1_257_000_000, 633, dyn,
                     dict(nat, pages=17426, pages_in_slice=17426))
check("a best effort over every page of the slice says so, and that what it "
      "learns from one page carries into the rest",
      "every one of the slice's 17,426 pages, in the order the slice holds "
      "them" in ta
      and "so what it learns from one page carries into every page after it"
      in ta
      and "every page of the slice, whole, in its native 4,096-byte window" in ta
      and "spread evenly over it" not in ta)

tmp3 = tempfile.mkdtemp(prefix="minagi-enwik9-natural-")
try:
    os.makedirs(os.path.join(tmp3, "runs"))
    saved, RR.ROOT = RR.ROOT, tmp3
    none_n = RR.enwik9_natural()
    with open(os.path.join(tmp3, "runs", "enwik9_natural.jsonl"), "w") as f:
        for d in (dict(nat, date="a"), dict(nat, date="b", budget=100000),
                  dict(nat, date="c", weights_unchanged=False)):
            f.write(json.dumps(d) + "\n")
    got_n = RR.enwik9_natural()
    RR.ROOT = saved
finally:
    shutil.rmtree(tmp3, ignore_errors=True)
check("the best effort is the latest whole-pages measurement of the "
      "standard size with the weights left unchanged - none, none",
      none_n is None and got_n is not None and got_n["date"] == "a")

print("\n  all enwik9 table tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
