#!/usr/bin/env python3
"""
ARC as this model reads it, for the next stage's four ARC lanes.

A GRID IS ROWS OF DIGITS, one character a cell, its size beside it:

    input 1 (3x4):
    0070
    0700
    7000

Byte-level, a digit is already one symbol, so a cell costs one character and
keeps its column. ARC-AGI-3's screens have sixteen colors, written 0-9 then
a-f.

A TASK IS ONE CONVERSATION: the demonstrations and the first test input in
the request, the output grid as the reply, any further test input in a turn
of its own. BARC's tasks come with the program that made them, so each also
gives a record whose reply is that program.

NOTHING FROM AN EVALUATION SET IS TRAINED ON. ARC-AGI-1's 400 and ARC-AGI-2's
120 public evaluation tasks are copied to data/eval/arc-agi-1 and
data/eval/arc-agi-2 for the benchmark, and every task bound for training is
checked against them twice:
  - by name: any eight-hex-digit task id in its name. ARC-AGI-2's 1,000
    training tasks are 391 of ARC-AGI-1's training tasks, 376 of its 400
    evaluation tasks and 233 new ones, and some collections name their
    variants after the task they vary (synth_riddles' archive of 250,000
    riddles made from evaluation task 009d5c81).
  - by content: any pair that matches an evaluation pair under one of the
    eight rotations and reflections with its colors relabelled - which is
    what an augmented copy is. Pairs whose input has under 12 cells match
    too much by chance, and are not compared.

AUGMENTATION. The hand-made tasks are few - 1,000 in ARC-AGI-2's training
set, 400 in ARC-AGI-1's, 160 in ConceptARC - so each is also written turned,
mirrored and recolored (1-9 permuted, 0 the background kept), with its
demonstrations reordered: --arc-augment copies, 64 by default. A task's
copies share its held-out side, so no copy of a held-out task is trained on.

Driven by `python3 -m corpora agent`, which downloads the sources; the
ARC-AGI-3 games are played by corpora/arc3_play.py.
"""

import gzip
import hashlib
import io
import json
import os
import random
import re
import tarfile
import zipfile
from collections import Counter
from itertools import chain

DIGITS = "0123456789abcdef"


# ------------------------------------------------------------------- grids
def grid(x, colors=10, most=30):
    """A grid as a list of rows of ints, or None if it is not one."""
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except ValueError:
            return None
    if hasattr(x, "tolist"):
        x = x.tolist()
    if not isinstance(x, (list, tuple)) or not 0 < len(x) <= most:
        return None
    out = []
    for r in x:
        if hasattr(r, "tolist"):
            r = r.tolist()
        if not isinstance(r, (list, tuple)) or not 0 < len(r) <= most:
            return None
        row = []
        for c in r:
            if isinstance(c, bool):
                return None
            try:
                v = int(c)
            except (TypeError, ValueError):
                return None
            if v != c or not 0 <= v < colors:
                return None
            row.append(v)
        out.append(row)
    return out if len({len(r) for r in out}) == 1 else None


def rows(g):
    return "\n".join("".join(DIGITS[c] for c in r) for r in g)


def dims(g):
    return f"{len(g)}x{len(g[0])}"


def pairs(items):
    """[(input, output)] from {"input", "output"} dicts or [input, output]
    lists; None if any of them is broken."""
    out = []
    for p in items if isinstance(items, (list, tuple)) else []:
        if isinstance(p, dict):
            a, b = p.get("input"), p.get("output")
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            a, b = p
        else:
            return None
        a, b = grid(a), grid(b)
        if a is None or b is None:
            return None
        out.append((a, b))
    return out


def task(obj):
    """{"train": [(in, out)], "test": [(in, out)]} from an ARC-format
    dict, or None - a test without its output teaches nothing."""
    if not isinstance(obj, dict):
        return None
    tr, te = pairs(obj.get("train")), pairs(obj.get("test"))
    return {"train": tr, "test": te} if tr and te else None


def chunks(pool, rng, lo=3, hi=5):
    """A pool of pairs from one rule - RE-ARC's thousand a task, BARC's
    examples - as tasks of lo to hi demonstrations and a test."""
    out, i = [], 0
    while len(pool) - i >= 3:
        n = min(rng.randint(lo, hi) + 1, len(pool) - i)
        part = pool[i:i + n]
        out.append({"train": part[:-1], "test": part[-1:]})
        i += n
    return out


# --------------------------------------------------------------- records
ASK = ["Each input grid below becomes its output grid by one rule. Find the "
       "rule and give the output for the test input.",
       "These input and output grids share one rule. Apply it to the test "
       "input.",
       "Work out what turns every input into its output, then do the same to "
       "the test input.",
       "Solve this ARC puzzle: the examples show a transformation; produce "
       "the output grid for the test input.",
       "Look at how each grid changes, then make the same change to the last "
       "one."]
ABOUT = ("A grid is rows of digits, one digit a cell and each digit a color; "
         "sizes are rows x columns.")
WRITE = ["Write a Python function main(input_grid) that turns each input grid "
         "into its output grid. Grids are numpy arrays of color numbers.",
         "Each input below becomes its output by one rule. Write it as a "
         "Python program: main(input_grid) returns the output grid."]


def _shown(examples):
    out = []
    for i, (a, b) in enumerate(examples, 1):
        out += [f"input {i} ({dims(a)}):", rows(a),
                f"output {i} ({dims(b)}):", rows(b), ""]
    return out


def solve_record(t, rng):
    head = [rng.choice(ASK), ABOUT, ""] + _shown(t["train"])
    many, turns = len(t["test"]) > 1, []
    for j, (a, b) in enumerate(t["test"]):
        q = f"test input{f' {j + 1}' if many else ''} ({dims(a)}):\n{rows(a)}"
        turns.append(("user", "\n".join(head) + q if j == 0 else q))
        turns.append(("bot", [("text", f"output ({dims(b)}):\n{rows(b)}")]))
    return {"tools": [], "turns": turns}


def program_record(examples, source, rng):
    head = [rng.choice(WRITE), ABOUT, ""] + _shown(examples)
    return {"tools": [], "turns": [
        ("user", "\n".join(head).rstrip()),
        ("bot", [("text", "```python\n" + source.strip() + "\n```")])]}


# ---------------------------------------------------------- augmentation
def _d8(g, k):
    """One of the square's eight symmetries, by the bits of k: transpose,
    then mirror left-right, then flip top-bottom."""
    if k & 4:
        g = [list(r) for r in zip(*g)]
    if k & 1:
        g = [r[::-1] for r in g]
    if k & 2:
        g = g[::-1]
    return g


def variant(t, k, seed):
    """Copy k of a task: 0 is the task as given; the others apply one of the
    eight symmetries to every grid alike, recolor 1-9 alike from the eighth
    copy on, and reorder the demonstrations."""
    if k == 0:
        return t
    rng = random.Random(f"{seed}:{k}")
    perm = list(range(10))
    if k >= 8:
        rest = list(range(1, 10))
        rng.shuffle(rest)
        perm = [0] + rest

    def f(g):
        return [[perm[c] for c in r] for r in _d8(g, k % 8)]

    train = [(f(a), f(b)) for a, b in t["train"]]
    rng.shuffle(train)
    return {"train": train, "test": [(f(a), f(b)) for a, b in t["test"]]}


# -------------------------------------------------------- the evaluation
def _cells(g):
    return len(g) * len(g[0])


def _shape(a, b):
    """What a symmetry or a recoloring cannot change: each grid's sides, and
    how many cells each color fills. A cheap test before the real one."""
    return (tuple(sorted((len(a), len(a[0])))), tuple(sorted((len(b), len(b[0])))),
            tuple(sorted(Counter(chain.from_iterable(a)).values())),
            tuple(sorted(Counter(chain.from_iterable(b)).values())))


def pair_sig(a, b):
    """A pair's identity under the eight symmetries and any recoloring: each
    symmetry's grids with colors renamed in order of first appearance, the
    least of the eight."""
    best = None
    for k in range(8):
        name = {}
        s = "|".join("/".join("".join(chr(65 + name.setdefault(c, len(name)))
                                      for c in r) for r in _d8(g, k))
                     for g in (a, b))
        if best is None or s < best:
            best = s
    return hashlib.blake2b(best.encode(), digest_size=8).digest()


_ID = re.compile(r"[0-9a-f]{8}")


class Guard:
    """The evaluation tasks, and whether a task is one of them."""

    SMALL = 12

    def __init__(self, tasks):
        self.ids = {i.lower() for i in tasks}
        self.shapes, self.sigs = set(), set()
        for t in tasks.values():
            for a, b in t["train"] + t["test"]:
                if _cells(a) >= self.SMALL:
                    self.shapes.add(_shape(a, b))
                    self.sigs.add(pair_sig(a, b))

    def named(self, name):
        return any(h in self.ids for h in _ID.findall(str(name).lower()))

    def bad(self, t, name=""):
        if self.named(name):
            return True
        return any(_cells(a) >= self.SMALL and _shape(a, b) in self.shapes
                   and pair_sig(a, b) in self.sigs
                   for a, b in t["train"] + t["test"])


def task_key(t):
    return hashlib.blake2b(json.dumps([t["train"], t["test"]]).encode(),
                           digest_size=8).digest()


def records(t, name, guard, copies, per_mille, programs=None, side=None):
    """(episode, held out?, identity) for a task and its copies, or one
    ("evaluation overlap", None, None) if the guard refuses it. Identity and
    side follow the task's content, so the same task met in two sources is
    the same records, on the same side - unless `side` names what decides it:
    the tasks cut from one pool (a BARC problem, a RE-ARC rule) share its
    side, so none is held out while another of its rule is trained on."""
    if guard.bad(t, name):
        yield "evaluation overlap", None, None
        return
    key = task_key(t)
    by = key if side is None else hashlib.blake2b(str(side).encode(),
                                                  digest_size=8).digest()
    held = int.from_bytes(by[:4], "big") % 1000 < per_mille
    for k in range(max(1, copies)):
        rng = random.Random(key.hex() + str(k))
        ident = hashlib.blake2b(key + k.to_bytes(4, "big"), digest_size=8).digest()
        yield solve_record(variant(t, k, key.hex()), rng), held, ident
    if programs:
        examples, source = programs
        ident = hashlib.blake2b(key + b"program", digest_size=8).digest()
        yield program_record(examples, source, random.Random(key.hex())), \
            held, ident


# ----------------------------------------------------------------- reading
def _parse(name, f):
    """(name, task or pool) for every task one file holds. JSON may be a
    task, a {name: task} map or a list of tasks; a list of bare pairs is a
    pool (RE-ARC's form); JSON lines hold one of those a line; Parquet rows
    are dicts like the JSON ones."""
    low = name.lower()
    if low.endswith(".gz") and not low.endswith(".tar.gz"):
        with gzip.open(f) as g:
            yield from _parse(name[:-3], g)
        return
    if low.endswith(".parquet"):
        import pyarrow.parquet as pq
        for i, row in enumerate(chain.from_iterable(
                b.to_pylist() for b in pq.ParquetFile(f).iter_batches(256))):
            yield from _value(row, f"{name}/{i}")
        return
    if low.endswith((".jsonl", ".ndjson")):
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    yield from _value(json.loads(line), f"{name}/{i}")
                except ValueError:
                    continue
        return
    if low.endswith(".json"):
        try:
            yield from _value(json.load(f), name)
        except (ValueError, UnicodeDecodeError):
            return


def _value(obj, name):
    t = task(obj)
    if t:
        yield name, t
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            t = task(v) if isinstance(v, dict) else None
            if t:
                yield f"{name}/{k}", t
        return
    if isinstance(obj, list) and obj:
        if all(isinstance(x, dict) and "input" in x and "output" in x
               for x in obj[:8]):
            p = pairs(obj)
            if p:
                yield name, p
            return
        for i, v in enumerate(obj):
            t = task(v)
            if t:
                yield f"{name}/{i}", t


def _kaggle_pair(names):
    """Kaggle's split form, a challenges file and a solutions file side by
    side: {challenge: solution} by file name."""
    out = {}
    for n in names:
        if "challenges" in os.path.basename(n):
            s = n.replace("challenges", "solutions")
            if s in names:
                out[n] = s
    return out


def _merge(ch, so):
    """Kaggle's challenges ({id: task with test inputs}) and solutions ({id:
    [output, ...]}) as whole tasks."""
    for k, t in (ch or {}).items():
        outs = (so or {}).get(k)
        if isinstance(t, dict) and isinstance(outs, list):
            tests = [dict(x, output=o) for x, o in zip(t.get("test", []), outs)
                     if isinstance(x, dict)]
            got = task({"train": t.get("train"), "test": tests})
            if got:
                yield k, got


def walk(path, skip=lambda name: False):
    """(name, task or pool) for every ARC task in a folder or archive, in a
    fixed order. `skip` passes over a file or archive by its name before it
    is opened."""
    if os.path.isdir(path):
        names = []
        for dp, ds, fs in os.walk(path):
            ds.sort()
            names += [os.path.join(dp, f) for f in sorted(fs)]
        paired = _kaggle_pair(set(names))
        for n in names:
            if skip(n) or n in paired.values():
                continue
            if n in paired:
                with open(n) as a, open(paired[n]) as b:
                    yield from _merge(json.load(a), json.load(b))
            else:
                yield from walk(n, skip)
        return
    low = path.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            infos = sorted((i for i in z.infolist() if not i.is_dir()),
                           key=lambda i: i.filename)
            members = {i.filename for i in infos}
            paired = _kaggle_pair(members)
            for i in infos:
                n = f"{path}/{i.filename}"
                if skip(n) or i.filename in paired.values():
                    continue
                if i.filename in paired:
                    yield from _merge(json.loads(z.read(i)),
                                      json.loads(z.read(paired[i.filename])))
                    continue
                with z.open(i) as f:
                    yield from _parse(n, f)
    elif low.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as tf:
            for m in tf:
                n = f"{path}/{m.name}"
                if m.isfile() and not skip(n):
                    yield from _parse(n, tf.extractfile(m))
    else:
        with open(path, "rb") as f:
            yield from _parse(path, f)


def evaluation_sets(arc1_zip, arc2_zip):
    """{"arc-agi-1": {id: task}, "arc-agi-2": {id: task}}: the public
    evaluation tasks, from the two repositories' archives."""
    out = {}
    for name, path in (("arc-agi-1", arc1_zip), ("arc-agi-2", arc2_zip)):
        got = {}
        for n, t in walk(path):
            if "/data/evaluation/" in n and isinstance(t, dict):
                got[os.path.splitext(os.path.basename(n))[0]] = t
        out[name] = got
    return out


def split_tasks(path, split="training"):
    """(id, task) for a repository archive's data/<split> tasks."""
    for n, t in walk(path):
        if f"/data/{split}/" in n and isinstance(t, dict):
            yield os.path.splitext(os.path.basename(n))[0], t


# The parts of neoneye's arc-dataset-collection read here, and how many copies
# of each task. ARC-AGI-1, ARC-AGI-2, RE-ARC and ARC-Heavy are read from their
# own sources instead; arc-dataset-bad is counter-examples, puzzles that break
# ARC's rules on purpose.
COMMUNITY = {"ConceptARC": 64, "Mini-ARC": 8, "1D-ARC": 1, "IPARC": 1,
             "Sequence_ARC": 1, "arc-dataset-diva": 1, "arc-dataset-tama": 1,
             "arc_1d": 1, "dbigham": 1, "nosound": 1, "synth_riddles": 1,
             "ARC_synthetic_extend": 1}
COMMUNITY_UNLICENSED = {"PQA": 1, "Sort-of-ARC": 1, "arc-community": 1}
_VERSION = re.compile(r"_v(\d+)$")


def community(zip_path, which, skip=lambda name: False):
    """(name, task, copies) for the chosen datasets of the collection, the
    newest version of each task only (a task fixed upstream is kept as
    name_v2, name_v3). `skip` passes over a file or an archive by its name
    unopened: synth_riddles' 250,000 riddles of an evaluation task are
    never read only to be refused one by one."""
    for ds, copies in which.items():
        prefix = f"dataset/{ds}/"
        newest = {}
        with zipfile.ZipFile(zip_path) as z:
            names = sorted(n for n in z.namelist()
                           if n.split("/", 1)[-1].startswith(prefix)
                           and not n.endswith("/") and not skip(n))
        for n in names:
            stem, ext = os.path.splitext(n)
            if ext.lower() in (".zip", ".gz", ".tar", ".tgz"):
                continue                          # read from the archive below
            m = _VERSION.search(stem)
            base = _VERSION.sub("", stem)
            v = int(m.group(1)) if m else 1
            if base not in newest or newest[base][0] < v:
                newest[base] = (v, n)
        keep = {n for _, n in newest.values()}
        archives = [n for n in names
                    if os.path.splitext(n)[1].lower() in (".zip", ".gz", ".tar",
                                                          ".tgz")]
        with zipfile.ZipFile(zip_path) as z:
            for n in names:
                if n in keep:
                    with z.open(n) as f:
                        for name, t in _parse(n, f):
                            yield f"{ds}/{name}", t, copies
            for n in archives:
                # an archive inside the archive: out to a temporary file,
                # since tar and gzip want to seek
                import tempfile
                with z.open(n) as src, tempfile.NamedTemporaryFile(
                        suffix=os.path.basename(n)) as tmp:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        tmp.write(b)
                    tmp.flush()
                    for name, t in walk(tmp.name):
                        yield f"{ds}/{os.path.basename(n)}/{name}", t, copies


# --------------------------------------------------------------- ARC-AGI-3
ACT3 = {"name": "act",
        "description": "Take one action in the game. ACTION1 to ACTION4 usually "
                       "move up, down, left and right; ACTION5 interacts; "
                       "ACTION6 clicks the cell at column x, row y; ACTION7 "
                       "undoes the last action.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string",
                       "enum": [f"ACTION{i}" for i in range(1, 8)]},
            "x": {"type": "integer", "description": "column, 0-63, for ACTION6"},
            "y": {"type": "integer", "description": "row, 0-63, for ACTION6"}},
            "required": ["action"]}}


def screen(g):
    g = grid(g, colors=16, most=64)
    return None if g is None else f"screen ({dims(g)}):\n{rows(g)}"


def game_record(rec):
    """One episode the player wrote, as tool use: the request is the game's
    first screen, each call one action, each result the screen after it."""
    first = screen(rec.get("start"))
    if first is None or not rec.get("steps"):
        return None
    total, solved = rec.get("win_levels") or "?", rec.get("kind") == "solution"
    ask = (f"A game you have not seen before. Its screen is a grid of cells, "
           f"each one of 16 colors written 0-9 and a-f. Levels completed: "
           f"{rec.get('level', 0)} of {total}. "
           + ("Work out how it is played and complete the next level."
              if solved else "Try actions and watch what they change."))
    steps = []
    for s in rec["steps"]:
        args = {"action": s.get("action")}
        if s.get("x") is not None:
            args.update(x=s["x"], y=s["y"])
        shown = screen(s.get("frame"))
        if shown is None:
            return None
        steps += [("call", "act", args),
                  ("result", f"levels completed: {s.get('levels', 0)} of "
                             f"{total}; {s.get('state', '')}\n{shown}")]
    last = rec["steps"][-1].get("state")
    steps.append(("text", f"Level {rec.get('level', 0) + 1} is complete."
                  if solved else "The game is over." if last == "GAME_OVER"
                  else f"That was {len(rec['steps'])} actions."))
    return {"tools": [ACT3], "turns": [("user", f"{ask}\n\n{first}"),
                                       ("bot", steps)]}


# -------------------------------------------------------- Nemotron's text
_ROW = r"[0-9](?:[ \t]?[0-9])*[ \t]*"
_TEXT_PAIR = re.compile(rf"Input:\s*\n((?:{_ROW}\n)+)\s*Output:\s*\n"
                        rf"((?:{_ROW}(?:\n|$))+)")


def text_pairs(text):
    """The demonstration pairs a solving prompt writes as rows of digits,
    with or without a space between them ("0070" and "0 0 7 0")."""
    out = []
    for m in _TEXT_PAIR.finditer(text or ""):
        a, b = (grid([[int(c) for c in re.findall(r"[0-9]", r)]
                      for r in part.split("\n") if r.strip()])
                for part in (m.group(1), m.group(2)))
        if a and b:
            out.append((a, b))
    return out
