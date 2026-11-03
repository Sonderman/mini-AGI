#!/usr/bin/env python3
"""
The README's PG19 table: the published byte-level models beside this model.

What these tests hold it to:
  the rows run best first; only the best score and the cheapest training are
  in bold; every published row links its paper and says how much of PG19 it
  was trained on, roughly what that cost and the context it was scored in;
  the published scores are the ones the scaling chart draws, MambaByte by its
  best model
  a published cost is an estimate of what its authors paid and says so, and
  the notes show how it was reached - the 6ND rule, A100s at 40%, and AWS's
  on-demand A100 price when the papers came out, linked as it was then - and
  that smaller clouds charged about a quarter as much
  a cost has two significant figures
  a mini-AGI row is of the weights it was measured on - their date, their
  reading, their size worked out from the pool they had, and the hours they
  had trained for by then - and says it was trained on no PG19
  those hours are the run's, counted up to where it first reached the step
  measured; up to the end they are every session's
  under the table it says the published models trained on PG19 and this
  one never did; the notes are folded away and trace every published score
  to the table its paper prints it in, and say what the 972M scores with a
  sliding window and what the smaller MambaByte scores
  the best effort, once measured, gets its own marked row, the mark is
  explained under the table, and without it there is no mark; read over
  every book, it says so
  if this model is the best, its score is the bold one
  a measurement of part of the split, or none, leaves the table as it was;
  one of the whole split fills it, and the README has the place for it
"""
import inspect
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import plot_scaling as PS                                   # noqa: E402
import refresh_readme as RR                                 # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def body(t):
    return [l for l in t.splitlines() if l.startswith("| ") and "---" not in l][1:]


pg = {"bpb": 2.090971662451195, "bytes_scored": 41_289_001, "books": 100,
      "window": 4096, "fraction": 1.0, "chars_read": 1_057_445_888.0,
      "step": 516_696, "experts": 156, "date": "2026-10-03",
      "own_bpb": 0.94}
p = {"total": 1_257_329_038, "per_expert": 3_145_728, "routers": 204_173,
     "n_experts": 397}
size = lambda n: RR.size_at(p, n)                           # noqa: E731
hours = {516_696: 312.85, 1_259_747: 654.45}.get
t = RR.pg19_table(pg, None, size, hours)
rows = body(t)
names = [r.split("|")[1].strip() for r in rows]
check("the rows run best first, and only the best score and the cheapest "
      "training are in bold",
      names == ["MambaByte-972M", "MegaByte-758M+262M", "Transformer-320M",
                "PerceiverAR-248M", "mini-AGI (frozen)"]
      and sum(r.count("**") for r in rows) == 4 and "**0.883**" in rows[0]
      and "| **~$25** |" in rows[4], f"{names}")
check("...every published row links its paper, and says how much PG19 it "
      "was trained on, roughly what that cost and the context it was scored "
      "in",
      "| 0.97B | Jan 2024 | 75 GB of PG19 | ~$4k (est.) | 8,192 | [Wang et "
      "al., *MambaByte*, Jan 2024](https://arxiv.org/abs/2401.13660) |"
      in rows[0]
      and "| 1.02B | May 2023 | 80 GB of PG19 | ~$1.6k (est.) | 8,192 | [Yu et "
      "al., *MEGABYTE*, May 2023](https://arxiv.org/abs/2305.07185) | 1.000 |"
      in rows[1]
      and "| 0.32B | May 2023 | 80 GB of PG19 | ~$1.6k (est.) | 1,024 |"
      in rows[2] and rows[2].endswith("| 1.057 |")
      and "| 0.25B | May 2023 | 80 GB of PG19 | ~$1.6k (est.) | 8,192 |"
      in rows[3] and rows[3].endswith("| 1.104 |")
      and "| model | params | date | trained on | training cost ↓ | context, "
      "bytes | scored in | PG19 bits/byte ↓ |" in t)
check("...and the published scores are the ones the scaling chart draws",
      sorted((e["bytes"], e["bpb"]) for e in RR.PG19)
      == sorted((b, v) for _, b, v in PS.BYTE))
mb, ma = RR.PG19[1], RR.PG19[0]
check("a published cost is 6 x the parameters a byte runs through x the "
      "bytes, on A100s at 40%, at AWS's on-demand price then",
      abs(RR.a100_hours(mb) - 6 * (262e6 + 758e6 / 8) * 80e9
          / (312e12 * 0.4) / 3600) < 1e-6
      and round(RR.a100_hours(mb)) == 381 and round(RR.a100_hours(ma)) == 974
      and ma["per_byte"] == 972e6
      and RR.AWS_A100 == 32.77 / 8
      and round(RR.a100_hours(mb) * RR.AWS_A100) == 1561
      and round(RR.a100_hours(ma) * RR.AWS_A100) == 3988
      and all(e["per_byte"] == mb["per_byte"] for e in RR.PG19[1:]))
check("...and the notes show how, linked to the price as it was then, and "
      "say that smaller clouds charged less",
      "best-effort estimates of what their authors paid" in t
      and "the 6ND rule" in t and "on A100s at 40% utilisation" in t
      and "at what AWS charged on demand for an A100 when the papers came "
      "out: $32.77 an hour for a machine with eight, $4.10 each, the same on "
      "its pricing page in [May 2023](https://web.archive.org/web/"
      "20230518023128/https://aws.amazon.com/ec2/instance-types/p4/)" in t
      and "and [April 2024](https://web.archive.org/web/20240402175315/" in t
      and "[Lambda](https://web.archive.org/web/20230512094749/https://"
      "lambdalabs.com/service/gpu-cloud) for $1.10 in May 2023, about a "
      "quarter as much - so the real cost may have been lower" in t
      and "1.7×10<sup>20</sup> FLOPs, about 380 A100-hours." in t
      and "75 billion bytes when it scored this, of the 150 billion it went "
      "on to - 380,000 steps of 48 × 8,192" in t
      and "4.4×10<sup>20</sup> FLOPs, about 970 A100-hours." in t)
check("...a cost has two significant figures",
      [RR.dollars(x) for x in (1561, 3988, 419.3, 53.2, 9800, 2.5e6)]
      == ["~$1.6k", "~$4k", "~$420", "~$53", "~$9.8k", "~$2.5M"])
check("this model's row is of the weights it was measured on - their date, "
      "reading, size and cost - and says it was trained on no PG19",
      rows[4] == "| mini-AGI (frozen) | 0.50B | 3 Oct 2026 | 1.1 GB, none of "
      "it PG19 | **~$25** | 4,096 | measured | 2.091 |"
      and "at $0.081 an hour, priced as in the enwik9 table above - the "
      "cheapest rental of that card, which no big cloud offers: 313 hours "
      "for the frozen row." in t, rows[4])
check("...its size worked out from the pool it had: each expert since taken "
      "off with its row of the routers",
      RR.size_at(p, 397) == p["total"]
      and abs(RR.size_at(p, 156) - (p["total"] - 241 * (3_145_728
                                                         + 204_173 / 397)))
      < 1 and RR.size_at({"total": 18_400, "per_expert": 1000,
                          "routers": 400, "n_experts": 10}, 4) == 12_160)
under = t[t.index(rows[-1]) + len(rows[-1]):t.index("<details>")]
check("under the table: the published models trained on PG19's own books, "
      "this one never did",
      "trained on PG19's own training books, 75 to 80 GB of them" in under
      and "mini-AGI has never been trained on PG19" in under
      and "\\*" not in under)
check("...and the notes are folded away, and trace every published score to "
      "the table its paper prints it in",
      t.index("<details>") > t.index(rows[-1])
      and t.index("**Date**") > t.index("<details>")
      and "1.057 for the Transformer, 1.104 for PerceiverAR and 1.000 for "
      "MegaByte, in the PG-19 column of their Table 2" in t
      and "0.883 for MambaByte-972M in their Table 5" in t
      and "10.1 GB in all (their Table 1)" in t
      and "without a sliding window" not in t)
check("...say what the 972M scores with a sliding window, and what the "
      "smaller MambaByte scores",
      "0.863 for the same model scored with a sliding window; the table "
      "shows the first" in t
      and "MambaByte is shown by its best model only" in t
      and "MambaByte-353M" in t and "scores 0.930 after 30 billion bytes "
      "(their Table 2)" in t)
check("...and how this model's score was measured",
      "mini-AGI's row is measured here:" in t
      and "over every book of the test split - 100 books, 41,289,001 bytes - "
      "in consecutive windows of 4,096 bytes" in t
      and "with the weights frozen" in t and "total bits ÷ total bytes" in t)

# the best effort: whole books in its own window, learning as it reads
nat = {"regime": "natural", "pages": 10, "budget": 4_096_000,
       "bytes": 4_123_456, "chunk": 2048, "context": 4096, "experts": 397,
       "step": 1_259_747, "date": "2026-10-22 01:00", "chars": 2.58e9,
       "adapted_bpb": 1.9012, "frozen_bpb": 1.9534, "alone_bpb": 2.0456,
       "lr": 8.1e-6, "weights_unchanged": True}
tn = RR.pg19_table(pg, nat, size, hours)
rows_n = body(tn)
names_n = [r.split("|")[1].strip() for r in rows_n]
under_n = tn[tn.index(rows_n[-1]) + len(rows_n[-1]):tn.index("<details>")]
check("the best effort gets a marked row of its own, in its place by score, "
      "of the weights it was measured on",
      names_n[4:] == ["mini-AGI (best effort)\\*", "mini-AGI (frozen)"]
      and rows_n[4] == "| mini-AGI (best effort)\\* | 1.26B | 22 Oct 2026 | "
      "2.6 GB, none of it PG19 | ~$53 | 4,096 | measured | 1.901 |"
      and "313 hours for the frozen row and 654 hours for the best effort."
      in tn, f"{rows_n}")
check("...the mark explained under the table: the model used exactly as in "
      "training and conversation, in its native window, scored before it "
      "learns, on a copy thrown away, on how many books",
      "\\* **Best effort**" in under_n
      and "exactly as it is used in training and in conversation" in under_n
      and "native 4,096-byte window sliding 2,048 characters at a time"
      in under_n
      and "scored first and then learned from" in under_n
      and "thrown away afterwards" in under_n
      and "It reads 10 whole books spread over the split, where the frozen "
      "row reads all 100." in under_n, under_n)
check("...and the notes say how it learned, and what the same books score "
      "frozen",
      "mini-AGI's two rows are measured here:" in tn
      and "learning at 8.1×10<sup>-6</sup>" in tn
      and "the same books score 1.953, and chunk by chunk alone 2.046" in tn)
check("...and without one, no mark",
      "Best effort" not in t and "best effort" not in t)
ta = RR.pg19_table(pg, dict(nat, pages=100, pages_in_slice=100), size, hours)
check("a best effort over every book says so: the frozen row's books, one "
      "after another, what it learns from one carried into the next",
      "It reads all 100 books of the split, the frozen row's books, one "
      "after another, so what it learns from one book carries into every "
      "book after it." in ta
      and "--natural on all 100 books of the split, in order," in ta
      and "spread over the split" not in ta)

tb = RR.pg19_table(dict(pg, bpb=0.8), None, size, hours)
rows_b = body(tb)
check("if this model is the best, its score is the bold one",
      rows_b[0].startswith("| mini-AGI (frozen)") and "**0.800**" in rows_b[0]
      and sum(r.count("**0.") for r in rows_b) == 1)

# the hours a measurement's weights had trained for, from the run's log
LOG = """# session started 2026-10-01 00:00  at step 0, 0.0M of 100M characters read (0.00%), 10 experts
step 100   1.0M of 100M characters (1.00%)   30 min   10 experts
step 200   2.0M of 100M characters (2.00%)   60 min   10 experts
# session started 2026-10-02 00:00  at step 200, 2.0M of 100M characters read (2.00%), 10 experts
step 300   3.0M of 100M characters (3.00%)   20 min   10 experts
step 400   4.0M of 100M characters (4.00%)   40 min   10 experts
# session started 2026-10-03 00:00  at step 400, 4.0M of 100M characters read (4.00%), 10 experts
step 500   5.0M of 100M characters (5.00%)   15 min   10 experts
"""
tmp = tempfile.mkdtemp(prefix="minagi-pg19-table-")
try:
    os.makedirs(os.path.join(tmp, "runs", "scaling"))
    with open(os.path.join(tmp, "runs", "samples.txt"), "w") as f:
        f.write(LOG)
    w = os.path.join(tmp, "weights")
    os.makedirs(w)
    with open(os.path.join(w, "manifest.json"), "w") as f:
        json.dump({"step": 9, "experts": [{"params": 1000}] * 10}, f)
    np.savez(os.path.join(w, "core.npz"),
             **{"tok_emb.weight": np.zeros((265, 2), np.float32)})
    np.savez(os.path.join(w, "routers.npz"), r=np.zeros((10, 3), np.float32))
    readme = "a\n<!-- auto:pg19 -->\nold table\n<!-- /auto:pg19 -->\nb\n"
    saved, RR.ROOT = RR.ROOT, tmp
    ran = [RR.ran_until(s) for s in (200, 400, 450, 10 ** 9)]
    total = RR.running()[2]
    none, k1 = RR.pg19(readme)
    path = os.path.join(tmp, "runs", "scaling", "pg19.json")
    with open(path, "w") as f:
        json.dump(dict(pg, fraction=0.1, experts=4, step=400), f)
    part, k2 = RR.pg19(readme)
    with open(path, "w") as f:
        json.dump(dict(pg, experts=4, step=400), f)
    whole, k3 = RR.pg19(readme)
    RR.ROOT = saved
finally:
    shutil.rmtree(tmp, ignore_errors=True)
check("a measurement is costed at the hours the run had trained for when it "
      "first reached that step; to the end, every session's",
      ran == [3600, 6000, 6900, 6900] and total == 6900, f"{ran} {total}")
check("no measurement, or one of part of the split, leaves the table as it "
      "was",
      none == readme and k1 and part == readme and k2)
check("...one of the whole split fills it, its size from the weights' own "
      "count at the pool it was measured with, its cost from the log",
      k3 and "old table" not in whole
      and "| mini-AGI (frozen) | 0.00B | 3 Oct 2026 | 1.1 GB, none of it "
      "PG19 | **~$0** |" in whole
      and whole.startswith("a\n<!-- auto:pg19 -->\n| model |"))

text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
at = [text.find(s) for s in ("### enwik9 against published models",
                             "### PG19 against published models",
                             "<!-- auto:pg19 -->", "<!-- /auto:pg19 -->",
                             "### Data Scaling")]
check("the README has its place, between the enwik9 table and the scaling "
      "chart, and the refresh fills it",
      all(a >= 0 for a in at) and at == sorted(at)
      and "pg19(out)" in inspect.getsource(RR.main), f"{at}")

print("\n  all PG19 table tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
