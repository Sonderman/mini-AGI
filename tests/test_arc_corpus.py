#!/usr/bin/env python3
"""
The ARC lanes, built with every download stubbed out.

  a grid is rows of digits with its size; a task is the demonstrations and
  the test in the request and the output grid as the reply, a second test
  in a turn of its own
  the eight symmetries are eight different grids, and a pair's signature
  does not change under any of them or under recoloring
  a task's copies keep 0 the background, recolor every grid alike, and are
  the same pairs up to symmetry and color; all of them sit on one side
  the guard refuses a task holding an evaluation pair turned and recolored,
  and one named after an evaluation task; it lets a different task through,
  and never compares pairs too small to mean anything
  JSON tasks, {name: task} maps, pools of pairs, JSON lines, Kaggle's
  challenges-and-solutions split, zip and tar archives are all read
  from the community collection: the chosen datasets only, the newest
  version of a task, archives inside it read, an archive named after an
  evaluation task never opened
  a played ARC-AGI-3 episode is tool use whose results are 16-color screens
  the player's search finds the shortest win of a small game, and clicks
  only on the things on its screen
  an offline build of the ARC sources writes the arc and arc-synthetic
  lanes, copies the evaluation sets to data/eval untouched, drops a training
  task that is an evaluation task by name or in disguise, and lets
  ARC-AGI-1's in only when told to give that benchmark up
"""
import contextlib
import io
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    extra = extra if (not cond or len(extra) < 100) else ""
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


from corpora import agent as A                                # noqa: E402
from corpora import arc as R                                  # noqa: E402
from corpora import arc3_play as P                            # noqa: E402
from minagi.tokenizer import ByteTokenizer, SPECIAL_ID        # noqa: E402

TOK = ByteTokenizer()


def balanced(text):
    ids = TOK.encode(text).ids
    n = {m: ids.count(SPECIAL_ID[m]) for m in ("<user>", "</user>", "<bot>",
                                               "</bot>")}
    return len(set(n.values())) == 1 and n["<user>"] > 0


def rand_grid(rng, h, w, colors=5):
    return [[rng.randrange(colors) for _ in range(w)] for _ in range(h)]


rng = random.Random(1)
A1 = rand_grid(rng, 4, 5)
B1 = [r[::-1] for r in A1]                      # the rule: mirror
T = {"train": [(A1, B1), (rand_grid(rng, 3, 4), None)], "test": []}
T["train"][1] = (T["train"][1][0], [r[::-1] for r in T["train"][1][0]])
TEST_IN = rand_grid(rng, 5, 3)
T["test"] = [(TEST_IN, [r[::-1] for r in TEST_IN])]

# ------------------------------------------------------------------- grids
check("a grid is checked: ragged, out of range, boolean and oversize grids "
      "are refused",
      R.grid([[1, 2], [3]]) is None and R.grid([[10]]) is None
      and R.grid([[True]]) is None and R.grid([[0] * 31]) is None
      and R.grid("[[1,2],[3,4]]") == [[1, 2], [3, 4]]
      and R.grid([[15]], colors=16) == [[15]])
rec = R.solve_record(T, random.Random(0))
text = A.render(rec)
check("a task is one conversation: demonstrations, the test, the output",
      "input 1 (4x5):\n" + R.rows(A1) in text
      and "output 2 (3x4):" in text
      and f"test input (5x3):\n{R.rows(TEST_IN)}\n</user>" in text
      and f"<bot>\noutput (5x3):\n{R.rows(T['test'][0][1])}\n</bot>" in text
      and balanced(text), text[:300])
two = {"train": T["train"], "test": T["test"] * 2}
text = A.render(R.solve_record(two, random.Random(0)))
check("a second test input asks again in a turn of its own",
      text.count("<user>") == 2 and "test input 2 (5x3):" in text
      and text.count("output (5x3):") == 2)

# ------------------------------------------------------------ augmentation
seen = {json.dumps(R._d8(A1, k)) for k in range(8)}
check("the eight symmetries of an uneven grid are eight different grids",
      len(seen) == 8)
sig = R.pair_sig(A1, B1)
perm = [0, 3, 1, 4, 2, 9, 8, 7, 6, 5]
check("a pair's signature is the same under every symmetry and recoloring",
      all(R.pair_sig([[perm[c] for c in r] for r in R._d8(A1, k)],
                     [[perm[c] for c in r] for r in R._d8(B1, k)]) == sig
          for k in range(8))
      and R.pair_sig(A1, A1) != sig)
v = R.variant(T, 13, "seed")
zeros = sum(c == 0 for g in (A1, B1) for r in g for c in r)
vz = [p for p in v["train"] if sorted(map(len, p[0])) == sorted(map(len, R._d8(A1, 13 % 8)))]
check("a copy keeps every pair up to symmetry and color, 0 kept the "
      "background, the test turned like the rest",
      sorted(R.pair_sig(a, b) for a, b in v["train"])
      == sorted(R.pair_sig(a, b) for a, b in T["train"])
      and R.pair_sig(*v["test"][0]) == R.pair_sig(*T["test"][0])
      and any(sum(c == 0 for g in p for r in g for c in r) == zeros for p in vz)
      and R.variant(T, 0, "seed") is T)

# ------------------------------------------------------------------- guard
EV = {"00576224": {"train": [(rand_grid(rng, 6, 6), rand_grid(rng, 6, 6))],
                   "test": [(rand_grid(rng, 5, 7), rand_grid(rng, 7, 5))]}}
G = R.Guard(EV)
ea, eb = EV["00576224"]["test"][0]
disguised = {"train": [(A1, B1),
                       ([[perm[c] for c in r] for r in R._d8(ea, 6)],
                        [[perm[c] for c in r] for r in R._d8(eb, 6)])],
             "test": T["test"]}
tiny = {"00576224": {"train": [([[1, 2]], [[2, 1]])], "test": [([[3]], [[4]])]}}
check("the guard refuses an evaluation pair turned and recolored, and a "
      "task named after an evaluation task; it passes a different task",
      G.bad(disguised) and G.bad(T, "synth/e_00576224_10x10/7")
      and not G.bad(T, "arc2/aabbccdd") and not R.Guard(tiny).sigs)

recs = list(R.records(T, "x/t1", G, 16, 1000))
check("a task's copies: as many as asked, distinct, all on one side",
      len(recs) == 16 and len({r[2] for r in recs}) == 16
      and len({r[1] for r in recs}) == 1)
again = list(R.records(T, "y/other-name", G, 16, 10))
check("the same task from another source is the same records, same side",
      [r[2] for r in again] == [r[2] for r in recs]
      and list(R.records(disguised, "z", G, 4, 10))
      == [("evaluation overlap", None, None)])
pool = [(rand_grid(rng, 3, 3), rand_grid(rng, 3, 3)) for _ in range(23)]
cut = R.chunks(pool, random.Random(0))
sides = {r[1] for i, c in enumerate(cut)
         for r in R.records(c, f"p/{i}", G, 1, 500, side="p")}
check("the tasks cut from one pool all take the pool's side", len(sides) == 1)
check("a pool of pairs becomes tasks of 3 to 5 demonstrations and a test",
      all(3 <= len(t["train"]) + 1 and len(t["train"]) <= 5
          and len(t["test"]) == 1 for t in cut)
      and sum(len(t["train"]) + 1 for t in cut) >= 21)

spaced = ("Train Example 1:\n\nInput:\n" + R.rows(A1).replace("", " ").strip()
          .replace(" \n ", "\n") + "\n\nOutput:\n" + R.rows(B1) + "\n\nTest:")
check("a solving prompt's grids are read, with or without spaces",
      R.text_pairs(spaced) == [(A1, B1)]
      and R.text_pairs("Input:\n12\n34\n\nOutput:\n5\n") == [([[1, 2], [3, 4]],
                                                                 [[5]])])

# ------------------------------------------------------------------ files
J = json.dumps


def task_json(t):
    return {"train": [{"input": a, "output": b} for a, b in t["train"]],
            "test": [{"input": a, "output": b} for a, b in t["test"]]}


tmp = tempfile.mkdtemp(prefix="arc-corpus-")
try:
    d = os.path.join(tmp, "walk")
    os.makedirs(os.path.join(d, "sub"))
    json.dump(task_json(T), open(os.path.join(d, "one.json"), "w"))
    json.dump({"k1": task_json(T), "k2": task_json(two)},
              open(os.path.join(d, "map.json"), "w"))
    json.dump([{"input": a, "output": b} for a, b in pool],
              open(os.path.join(d, "pool.json"), "w"))
    with open(os.path.join(d, "lines.jsonl"), "w") as f:
        f.write(J(task_json(T)) + "\n" + J(task_json(two)) + "\n")
    ch = {"kid": {"train": task_json(T)["train"],
                  "test": [{"input": TEST_IN}]}}
    with zipfile.ZipFile(os.path.join(d, "sub", "kaggle.zip"), "w") as z:
        z.writestr("x_challenges.json", J(ch))
        z.writestr("x_solutions.json", J({"kid": [T["test"][0][1]]}))
    with tarfile.open(os.path.join(d, "sub", "packed.tar.gz"), "w:gz") as tf:
        b = J(task_json(T)).encode()
        info = tarfile.TarInfo("inner/t.json")
        info.size = len(b)
        tf.addfile(info, io.BytesIO(b))
    got = list(R.walk(d))
    names = [n.replace(d, "") for n, _ in got]
    check("JSON tasks, maps, pools, JSON lines, Kaggle's split form, zip and "
          "tar are all read",
          "/one.json" in names and "/map.json/k2" in names
          and any(isinstance(t, list) and len(t) == 23 for _, t in got)
          and "/lines.jsonl/1" in names and "kid" in names
          and "/sub/packed.tar.gz/inner/t.json" in names, str(names))
    check("a file is passed over by its name before it is opened",
          not any("packed" in n for n, _ in R.walk(d, skip=lambda n:
                                                        "packed" in n)))

    # the community collection: chosen datasets, newest versions, archives.
    # Its tasks are its own - a copy of one ARC-AGI-2 also has would be
    # dropped as a duplicate in the build below.
    def fresh(seed):
        r = random.Random(seed)
        ins = [rand_grid(r, 4, 4) for _ in range(3)]
        return {"train": [(g, g[::-1]) for g in ins[:2]],
                "test": [(ins[2], ins[2][::-1])]}

    C1, C2, C3 = fresh(11), fresh(12), fresh(13)
    coll = os.path.join(tmp, "neoneye.zip")
    with zipfile.ZipFile(coll, "w") as z:
        top = "arc-dataset-collection-main/dataset/"
        z.writestr(top + "ConceptARC/data/Copy1.json", J(task_json(C1)))
        z.writestr(top + "ConceptARC/data/Fix1.json", J(task_json(T)))
        z.writestr(top + "ConceptARC/data/Fix1_v2.json", J(task_json(C2)))
        z.writestr(top + "arc-dataset-bad/data/b.json", J(task_json(T)))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for nm, t in (("r1.json", C3), ("r2.json", fresh(14))):
                b = J(task_json(t)).encode()
                info = tarfile.TarInfo(nm)
                info.size = len(b)
                tf.addfile(info, io.BytesIO(b))
        z.writestr(top + "synth_riddles/all_depth_2_10k.tar.gz", buf.getvalue())
        z.writestr(top + "synth_riddles/e_00576224_10x10_250k.tar.gz",
                   buf.getvalue())
    got = list(R.community(coll, R.COMMUNITY, skip=G.named))
    names = [n for n, _, _ in got]
    check("the collection: chosen datasets, the newest version of each task, "
          "archives inside it read, one named for an evaluation task skipped",
          any(n.endswith("Fix1_v2.json") for n in names)
          and not any(n.endswith("Fix1.json") for n in names)
          and not any("arc-dataset-bad" in n for n in names)
          and sum("all_depth_2" in n for n in names) == 2
          and not any("e_00576224" in n for n in names)
          and dict((n.split("/")[0], c) for n, _, c in got)["ConceptARC"] == 64,
          str(names))

    # ARC-AGI-3: a played episode as tool use
    frame = [[(x + y) % 16 for x in range(8)] for y in range(6)]
    rec = {"game": "ls20-1", "kind": "solution", "level": 0, "win_levels": 3,
           "start": frame,
           "steps": [{"action": "ACTION6", "x": 2, "y": 1, "frame": frame,
                      "state": "NOT_FINISHED", "levels": 0},
                     {"action": "ACTION1", "x": None, "y": None, "frame": frame,
                      "state": "NOT_FINISHED", "levels": 1}]}
    text = A.render(R.game_record(rec))
    calls = [json.loads(c) for c in
             __import__("re").findall(r"<call>(.*?)</call>", text)]
    check("an ARC-AGI-3 episode: actions as calls, 16-color screens as "
          "results, the level said complete",
          calls == [{"name": "act", "arguments": {"action": "ACTION6", "x": 2,
                                                  "y": 1}},
                    {"name": "act", "arguments": {"action": "ACTION1"}}]
          and "screen (6x8):\n0123456789abcdef"[:20] in text
          and "levels completed: 1 of 3" in text
          and "Level 1 is complete." in text and balanced(text), text[-300:])

    # the player's search on a game small enough to know: a corridor of
    # five cells, ACTION4 steps right, ACTION3 left, the end wins the level
    class Act:
        def __init__(self, n):
            self.name, self.value = n, int(n[6:])

    class Obs:
        def __init__(self, pos, lev):
            self.frame = [[[1 if x == pos else 0 for x in range(5)]]]
            self.levels_completed, self.win_levels = lev, 1
            self.state = type("S", (), {"name": "WIN" if lev else
                                        "NOT_FINISHED"})()
            self.available_actions = [3, 4, 6]

    class Corridor:
        def reset(self):
            self.pos = 0
            return Obs(0, 0)

        def step(self, act, data=None):
            if act.name == "ACTION4":
                self.pos = min(4, self.pos + 1)
            elif act.name == "ACTION3":
                self.pos = max(0, self.pos - 1)
            return Obs(self.pos, int(self.pos == 4))

    P.GameAction = type("GA", (), {"__class_getitem__":
                                   classmethod(lambda c, n: Act(n))})
    game = P.Game.__new__(P.Game)
    game.id, game.used, game.env = "corridor", 0, Corridor()
    path = P.solve(game, [], 10_000, 10)
    check("the search finds the shortest win of a small game",
          path == [(4, None, None)] * 4, str(path))
    check("clicks go to the things on the screen, not to all its cells",
          [m for m in P.moves(Obs(2, 0)) if m[0] == 6] == [(6, 2, 0)]
          and (3, None, None) in P.moves(Obs(2, 0)))

    # ------------------------------------------------- a build, offline
    raw, out = os.path.join(tmp, "raw"), os.path.join(tmp, "data")
    other = {"train": [(rand_grid(rng, 4, 4), rand_grid(rng, 4, 4))],
             "test": [(rand_grid(rng, 3, 3), rand_grid(rng, 3, 3))]}

    def repo_zip(path, top, training, evaluation):
        with zipfile.ZipFile(path, "w") as z:
            for k, t in training.items():
                z.writestr(f"{top}/data/training/{k}.json", J(task_json(t)))
            for k, t in evaluation.items():
                z.writestr(f"{top}/data/evaluation/{k}.json", J(task_json(t)))

    zips = os.path.join(tmp, "zips")
    os.makedirs(zips)
    # ARC-AGI-2's training set holds an ARC-AGI-1 evaluation task (by id) and
    # one in disguise (a turned, recolored evaluation pair)
    repo_zip(os.path.join(zips, "arc2.zip"), "ARC-AGI-2-main",
             {"aaaa0001": T, "00576224": other, "bbbb0002": disguised},
             {"eeee0001": fresh(31)})
    repo_zip(os.path.join(zips, "arc1.zip"), "ARC-AGI-master",
             {"aaaa0001": T, "cccc0003": two}, EV)
    with zipfile.ZipFile(os.path.join(zips, "re_arc.zip"), "w") as z:
        z.writestr("re_arc/tasks/aaaa0001.json",
                   J([{"input": a, "output": b} for a, b in pool]))
    shutil.copy(coll, os.path.join(zips, "neoneye.zip"))

    def fake_url(url, path):
        name = {"ARC-AGI-2": "arc2.zip", "ARC-AGI/": "arc1.zip",
                "re_arc": "re_arc.zip", "dataset-collection": "neoneye.zip"}
        src = next(v for k, v in name.items() if k in url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy(os.path.join(zips, src), path)
        return path

    def fake_hf(name, repo, raw_dir, patterns):
        where = os.path.join(raw_dir, name, repo.split("/")[-1])
        os.makedirs(where, exist_ok=True)
        if name == "barc":
            ex = [[a, b] for a, b in pool[:7]]
            with open(os.path.join(where, "data.jsonl"), "w") as f:
                f.write(J({"source": "def main(input_grid):\n    return "
                           "input_grid\n", "examples": ex,
                           "seeds": ["x.py"]}) + "\n")
        elif name == "nemotron-rl":
            for split, t in (("train", fresh(21)), ("validation", fresh(22))):
                p = os.path.join(where, "data", "transductive", f"{split}.jsonl")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    f.write(J({"train": J(task_json(t)["train"]),
                               "test_input": J(t["test"][0][0]),
                               "expected_output": J(t["test"][0][1]),
                               "problem_id": "dddd0004" + split,
                               "task_id": "x"}) + "\n")
        elif name == "nemotron-sft":
            p = os.path.join(where, "data", "small", "data.jsonl")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            rows = R.rows

            def spaced_rows(g):
                return "\n".join(" ".join(str(c) for c in r) for r in g)

            with open(p, "w") as f:
                for pid, (a, b) in (("ffff0006", (A1, B1)),
                                    ("gggg0007", (ea, eb))):
                    f.write(J({"problem_id": pid, "tools": [], "messages": [
                        {"role": "system", "content": "Solve ARC puzzles."},
                        {"role": "user", "content": f"Train Example 1:\n\n"
                         f"Input:\n{spaced_rows(a)}\n\nOutput:\n"
                         f"{spaced_rows(b)}\n\nTest Input:\n{rows(a)}\n"},
                        {"role": "assistant", "content": "The answer.",
                         "reasoning_content": "Mirror it."}]}) + "\n")
        return where

    A.fetch_url, A.fetch_hf = fake_url, fake_hf
    A.shutil.disk_usage = lambda p: shutil._ntuple_diskusage(10**13, 0, 10**13)
    A._GUARD.clear()
    # nothing held out by hash here: with a handful of tasks, a source whose
    # every task drew the held-out side would have nothing to train on (one
    # task's copies all taking one side is checked above, through records)
    sys.argv = ["agent", "--out", out, "--raw", raw, "--yes", "--arc-augment",
                "16", "--val-per-mille", "0", "--only", "arc2", "arc1",
                "arc-community", "rearc", "barc", "nemotron-rl", "nemotron-sft"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = A.main()
    log = buf.getvalue()
    check("an offline build of the ARC sources succeeds", rc == 0, log[-800:])
    check("the evaluation sets are copied to data/eval as they were",
          sorted(os.listdir(os.path.join(out, "eval", "arc-agi-1")))
          == ["00576224.json"]
          and json.load(open(os.path.join(out, "eval", "arc-agi-1",
                                          "00576224.json")))
          == task_json(EV["00576224"])
          and os.listdir(os.path.join(out, "eval", "arc-agi-2"))
          == ["eeee0001.json"])
    state = json.load(open(os.path.join(raw, "manifest.json")))["sources"]
    check("ARC-AGI-2's copy of an ARC-AGI-1 evaluation task, and the one in "
          "disguise, are kept out",
          state["arc2"]["skipped"].get("evaluation overlap") == 2,
          J(state["arc2"]))
    check("the ARC-AGI-1 evaluation copy is the benchmark's: not trained on",
          "TRAINED ON" not in open(os.path.join(raw, "manifest.json")).read())
    check("ARC-AGI-1's task already in ARC-AGI-2 is dropped as a duplicate",
          state["arc1"]["duplicates"] == 16 and
          state["arc1"]["train_records"] + state["arc1"]["val_records"] == 16,
          J(state["arc1"]))
    check("Nemotron-SFT's trace of a disguised evaluation puzzle is kept out",
          state["nemotron-sft"]["skipped"].get("evaluation overlap") == 1
          and state["nemotron-sft"]["train_records"]
          + state["nemotron-sft"]["val_records"] == 1, J(state["nemotron-sft"]))
    lanes = sorted(os.listdir(os.path.join(out, "train")))
    check("the arc, arc-synthetic and arc-traces lanes are written",
          lanes == ["arc", "arc-synthetic", "arc-traces"], str(lanes))
    texts = []
    for dp, _, fs in os.walk(os.path.join(out, "train", "arc-synthetic")):
        texts += [open(os.path.join(dp, f)).read() for f in fs]
    for dp, _, fs in os.walk(os.path.join(out, "val", "arc-synthetic")):
        texts += [open(os.path.join(dp, f)).read() for f in fs]
    every = "".join(texts)
    check("BARC gives a program record as well as the tasks",
          "```python\ndef main(input_grid):" in every
          and "test input (" in every)
    check("Nemotron-RL's validation split is held out, its training split "
          "trained on",
          state["nemotron-rl"]["val_records"] == 1
          and state["nemotron-rl"]["train_records"] == 1,
          J(state["nemotron-rl"]))
    check("the community's ConceptARC task is written in all its copies",
          state["arc-community"]["train_records"] >= 16,
          J(state["arc-community"]))

    # giving up ARC-AGI-1's evaluation as a benchmark lets its tasks through,
    # and the manifest says so; ARC-AGI-2's stays out
    A._GUARD.clear()
    sys.argv = ["agent", "--out", out, "--raw", raw, "--yes", "--force",
                "--only", "arc2", "--arc-augment", "16", "--val-per-mille", "0",
                "--arc1-eval-for-training"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = A.main()
    man = json.load(open(os.path.join(raw, "manifest.json")))
    check("--arc1-eval-for-training lets ARC-AGI-1's evaluation tasks in and "
          "marks that benchmark as trained on",
          rc == 0 and not man["sources"]["arc2"]["skipped"]
          and any("TRAINED ON" in k for k in man["kept_for_benchmarks"]),
          J(man["sources"]["arc2"]))
    A._GUARD.clear()

    # ARC-AGI-3, from what the player wrote: a finished run is not played
    # again, its episodes become the arc-games lane, and the games it held
    # out are listed for the benchmark
    played = os.path.join(raw, "arc3")
    os.makedirs(played, exist_ok=True)
    with open(os.path.join(played, "episodes.jsonl"), "w") as f:
        f.write(J(rec) + "\n" + J(dict(rec, kind="walk", steps=rec["steps"][:1]))
                + "\n")
    open(os.path.join(played, "episodes.jsonl.done"), "w").close()
    json.dump({"held_out": ["vc33-9"], "played": ["ls20-1"]},
              open(os.path.join(played, "held_out.json"), "w"))
    sys.argv = ["agent", "--out", out, "--raw", raw, "--yes", "--only", "arc3",
                "--arc3-python", sys.executable, "--val-per-mille", "0"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = A.main()
    state = json.load(open(os.path.join(raw, "manifest.json")))["sources"]
    check("ARC-AGI-3's episodes become the arc-games lane, and its held-out "
          "games are listed in data/eval",
          rc == 0 and state["arc3"]["train_records"] == 2
          and json.load(open(os.path.join(out, "eval", "arc-agi-3",
                                          "held_out_games.json")))["held_out"]
          == ["vc33-9"], buf.getvalue()[-400:])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all ARC corpus tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
