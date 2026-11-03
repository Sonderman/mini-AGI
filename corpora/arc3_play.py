#!/usr/bin/env python3
"""
Play ARC-AGI-3's public games by search, and write down what happened.

ARC-AGI-3 publishes games, not transcripts: nobody has released recordings
of the public games being played, so this makes them. It runs under the
ARC-AGI toolkit (`pip install arc-agi`, which needs Python 3.12 or newer),
apart from the rest of the corpus build, and writes one JSON line per
episode for corpora/arc.py to turn into tool use.

Two kinds of episode:
  - solution: a breadth-first search from the start of a level, over the
    game's actions, finds the fewest actions that complete it, and the
    episode is that path, screen by screen. The search goes on to the next
    level from there, until the game is won or the search's budget runs
    out.
  - walk: random actions from the start of each level reached, so the
    model also sees what actions do where they do not win.

ACTION6 clicks a cell, and 4,096 cells are too many branches to search.
Clicks go to one cell of each connected patch of a non-background color -
the things on the screen - at most 24, largest first.

The search restores a state by replaying its actions from a fresh reset,
which costs a game step each. --steps is the budget per game, in steps;
at the toolkit's ~2,000 steps a second, the default is about four minutes.

Every --hold-every'th game, in order of id, is not played at all: those
are listed in --held, for measuring the model on games it never saw.

    python3.12 corpora/arc3_play.py --out episodes.jsonl \
        --environments environment_files --held held.json
"""

import argparse
import collections
import hashlib
import json
import random
import sys
import time

import numpy as np

TARGETS = 24
GameAction = None        # arcengine's, set by main(); the search needs only names


def screen(obs):
    """The observation's last frame as a list of rows, or None."""
    f = getattr(obs, "frame", None)
    if f is None:
        return None
    a = np.asarray(f)
    if a.ndim == 3:
        a = a[-1]
    return a.astype(int).tolist() if a.ndim == 2 else None


def state(obs):
    s = getattr(obs, "state", None)
    return getattr(s, "name", str(s))


def levels(obs):
    return int(getattr(obs, "levels_completed", 0) or 0)


def available(obs):
    out = []
    for a in getattr(obs, "available_actions", None) or []:
        v = getattr(a, "value", a)
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            name = str(getattr(a, "name", a))
            if name.startswith("ACTION"):
                out.append(int(name[6:]))
    return sorted(set(out))


def targets(g):
    """One cell of each connected patch of a color other than the commonest
    (the background), largest patches first."""
    h, w = len(g), len(g[0])
    back = collections.Counter(c for r in g for c in r).most_common(1)[0][0]
    seen, patches = set(), []
    for y in range(h):
        for x in range(w):
            if (y, x) in seen or g[y][x] == back:
                continue
            color, todo, cells = g[y][x], [(y, x)], []
            seen.add((y, x))
            while todo:
                cy, cx = todo.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1),
                               (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen \
                            and g[ny][nx] == color:
                        seen.add((ny, nx))
                        todo.append((ny, nx))
            cells.sort()
            patches.append((len(cells), cells[len(cells) // 2]))
    patches.sort(key=lambda p: -p[0])
    return [(x, y) for _, (y, x) in patches[:TARGETS]]


def moves(obs):
    """The actions worth trying from this observation: every simple one on
    offer but undo, and clicks on the screen's patches."""
    g = screen(obs)
    out = []
    for a in available(obs):
        if a == 6:
            if g:
                out += [(6, x, y) for x, y in targets(g)]
        elif a in (1, 2, 3, 4, 5):
            out.append((a, None, None))
    return out


class Game:
    """One game, stepped by action triples, counting every step taken."""

    def __init__(self, arc, game_id):
        self.arc, self.id, self.used = arc, game_id, 0
        self.env = arc.make(game_id)
        if self.env is None:
            raise RuntimeError(f"could not load {game_id}")

    def reset(self):
        self.used += 1
        return self.env.reset()

    def step(self, move):
        self.used += 1
        a, x, y = move
        act = GameAction[f"ACTION{a}"]
        if a == 6:
            return self.env.step(act, data={"x": x, "y": y})
        return self.env.step(act)

    def replay(self, path):
        obs = self.reset()
        for m in path:
            if obs is None or state(obs) in ("GAME_OVER", "WIN"):
                return None
            obs = self.step(m)
        return obs


def key(obs):
    return hashlib.blake2b(json.dumps([screen(obs), levels(obs), state(obs)])
                           .encode(), digest_size=12).digest()


def solve(game, prefix, budget, depth):
    """The fewest moves after `prefix` that complete one more level, or
    None, searching breadth-first until `budget` steps are spent."""
    start = game.replay(prefix)
    if start is None:
        return None
    goal, seen, frontier = levels(start) + 1, {key(start)}, [([], start)]
    stop = game.used + budget
    for _ in range(depth):
        nxt = []
        for path, obs in frontier:
            for m in moves(obs):
                if game.used >= stop:
                    return None
                here = game.replay(prefix + path)
                if here is None:
                    break
                got = game.step(m)
                if got is None:
                    continue
                if levels(got) >= goal or state(got) == "WIN":
                    return path + [m]
                k = key(got)
                if k in seen or state(got) == "GAME_OVER":
                    continue
                seen.add(k)
                nxt.append((path + [m], got))
        if not nxt:
            return None
        frontier = nxt
    return None


def record(game, prefix, path, kind, level, total):
    """The episode: the screen at the start of the level, then each move with
    the screen after it."""
    obs = game.replay(prefix)
    if obs is None or screen(obs) is None:
        return None
    rec = {"game": game.id, "kind": kind, "level": level, "win_levels": total,
           "start": screen(obs), "steps": []}
    for m in path:
        obs = game.step(m)
        if obs is None:
            break
        rec["steps"].append({"action": f"ACTION{m[0]}", "x": m[1], "y": m[2],
                             "frame": screen(obs), "state": state(obs),
                             "levels": levels(obs)})
        if state(obs) in ("GAME_OVER", "WIN"):
            break
    return rec if rec["steps"] else None


def walk(game, prefix, length, rng):
    obs = game.replay(prefix)
    path = []
    for _ in range(length):
        if obs is None or state(obs) in ("GAME_OVER", "WIN"):
            break
        options = moves(obs)
        if not options:
            break
        m = rng.choice(options)
        path.append(m)
        obs = game.step(m)
    return path


def play(arc, game_id, a, out, rng):
    game = Game(arc, game_id)
    first = game.reset()
    total = int(getattr(first, "win_levels", 0) or 0)
    prefix, wrote, t0 = [], 0, time.time()
    while game.used < a.steps:
        here = game.replay(prefix)
        level = levels(here) if here is not None else 0
        for _ in range(a.walks):
            path = walk(game, prefix, a.walk_length, rng)
            rec = record(game, prefix, path, "walk", level, total)
            if rec:
                out.write(json.dumps(rec) + "\n")
                wrote += 1
        path = solve(game, prefix, a.steps - game.used, a.depth)
        if not path:
            break
        rec = record(game, prefix, path, "solution", level, total)
        if rec:
            out.write(json.dumps(rec) + "\n")
            wrote += 1
        prefix += path
        out.flush()
        if rec and rec["steps"][-1]["state"] == "WIN":
            break
    print(f"  {game_id}: {wrote} episodes, {len(prefix)} moves of solution, "
          f"{game.used:,} steps, {time.time() - t0:,.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--held", required=True,
                    help="where to list the games held out")
    ap.add_argument("--environments", default="environment_files")
    ap.add_argument("--steps", type=int, default=500_000,
                    help="game steps per game, search and walks together")
    ap.add_argument("--depth", type=int, default=60,
                    help="longest solution searched for, in moves")
    ap.add_argument("--walks", type=int, default=8,
                    help="random walks from the start of each level reached")
    ap.add_argument("--walk-length", type=int, default=40)
    ap.add_argument("--hold-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if sys.version_info < (3, 12):
        sys.exit("the ARC-AGI toolkit needs Python 3.12 or newer")
    global GameAction
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    # NORMAL loads the local games and the ARC API's, and caches a remote
    # game's source in the environments folder the first time it is made;
    # ARC_API_KEY, if set, unlocks the public games an anonymous key cannot
    arc = Arcade(operation_mode=OperationMode.NORMAL,
                 environments_dir=a.environments)
    games = sorted({g.game_id for g in arc.get_environments()})
    held = games[a.hold_every - 1::a.hold_every] if a.hold_every else []
    with open(a.held, "w") as f:
        json.dump({"held_out": held, "played": [g for g in games
                                                if g not in held]}, f, indent=1)
    print(f"  {len(games)} games: playing {len(games) - len(held)}, holding out "
          f"{len(held)}", flush=True)
    rng = random.Random(a.seed)
    with open(a.out, "w") as out:
        for g in games:
            if g in held:
                continue
            try:
                play(arc, g, a, out, rng)
            except Exception as e:                # one game never stops the rest
                print(f"  {g}: failed - {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
