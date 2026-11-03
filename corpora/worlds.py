#!/usr/bin/env python3
"""
Text games played to the end by their own solvers, written as tool use.

The game is a tool here: `act` takes one command and returns what the game
says back. The request is the game's opening - the task and the first room -
and the bot's turn is the solver's commands, each followed by the game's
reply, until the game reports the task done. Only games that end won are
kept, so every transcript is a path that works.

TextWorld makes the games (Côté et al. 2018): 1 to 8 rooms, a few objects
per room, a quest of 1 to 8 steps, one seed each, compiled by the Inform 7
tools the package ships. The commands are the game's own walkthrough
(policy_commands). A compiled game is deleted once played: it is a few
hundred kilobytes, its transcript a few.

ALFWorld's household tasks (Shridhar et al. 2021) are played by its
hand-written expert, one command at a time. Its training games are trained
on, its valid_seen games are held out, and valid_unseen is left untouched
for a benchmark.

Both need their packages - `pip install alfworld` brings TextWorld with it -
and are skipped, with that said, when they are missing. Driven by
`python3 -m corpora agent`.
"""

import glob
import json
import os
import random
import re
import subprocess

ACT = {"name": "act",
       "description": "Type one command into the game, and read what it "
                      "says back.",
       "parameters": {"type": "object",
                      "properties": {"command": {
                          "type": "string",
                          "description": "e.g. go north, open door, take key "
                                         "from table"}},
                      "required": ["command"]}}
ACT_HOUSE = {**ACT, "parameters": {
    "type": "object",
    "properties": {"command": {
        "type": "string",
        "description": "e.g. go to cabinet 1, take mug 1 from countertop 1, "
                       "clean mug 1 with sinkbasin 1"}},
    "required": ["command"]}}
STEPS = 50                    # ALFWorld's own limit for its expert's episodes


def clean(text):
    """A game's text without its banner art, its prompt, or its offer to
    restart once the game is over."""
    keep = []
    for line in str(text).splitlines():
        bare = [c for c in line if not c.isspace()]
        if len(bare) >= 5 and sum(c.isalnum() for c in bare) < 0.2 * len(bare):
            continue                                    # ASCII art
        keep.append(line.rstrip())
    out = "\n".join(keep)
    out = re.sub(r"Would you like to RESTART.*", "", out, flags=re.S)
    out = re.sub(r"^\s*>\s*$", "", out, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def transcript(intro, steps, tool):
    body = []
    for cmd, said in steps:
        body += [("call", "act", {"command": cmd}),
                 ("result", said or "(nothing happens)")]
    body.append(("text", "Done: the game reports the task complete."))
    return {"tools": [tool], "turns": [("user", intro), ("bot", body)]}


# --------------------------------------------------------------- TextWorld
def _textworld_game(job):
    """Make one game, play its walkthrough, delete it: (intro, steps) for a
    won game, None for any other outcome - a seed whose quest cannot be
    built is simply passed over."""
    seed, scratch = job
    import textworld
    rng = random.Random(seed)
    opts = textworld.GameOptions()
    opts.seeds = seed
    opts.nb_rooms = rng.randint(1, 8)
    opts.nb_objects = rng.randint(2, 4 + 2 * opts.nb_rooms)
    opts.quest_length = rng.randint(1, min(8, 1 + opts.nb_objects))
    opts.path = os.path.join(scratch, f"tw-{seed}.z8")
    opts.force_recompile = True
    env, won, intro, steps = None, False, "", []
    try:
        path, _ = textworld.make(opts)
        env = textworld.start(path, textworld.EnvInfos(
            feedback=True, policy_commands=True, won=True))
        state = env.reset()
        intro = clean(state["feedback"])
        for cmd in list(state["policy_commands"] or []):
            state, _, done = env.step(cmd)
            steps.append((cmd, clean(state["feedback"])))
            if done:
                break
        won = bool(state["won"])
    except Exception:
        return None
    finally:
        if env is not None:
            env.close()
        for f in glob.glob(os.path.join(scratch, f"tw-{seed}.*")):
            try:
                os.remove(f)
            except OSError:
                pass
    return (intro, steps) if won and intro and steps else None


def textworld(n, workers, scratch, held_every=100):
    """(episode, held out?) for games seeded 0 to n-1, made `workers` at a
    time; every hundredth seed is held out."""
    import textworld as _tw                                # noqa: F401
    from multiprocessing import Pool
    os.makedirs(scratch, exist_ok=True)
    jobs = [(seed, scratch) for seed in range(n)]
    with Pool(max(1, workers)) as pool:
        for seed, got in zip(range(n), pool.imap(_textworld_game, jobs,
                                                 chunksize=4)):
            if got:
                yield transcript(got[0], got[1], ACT), seed % held_every == 0


# ---------------------------------------------------------------- ALFWorld
def _alfworld_games(root):
    """The split's solvable text games, as ALFWorld's own loader finds them:
    a folder with traj_data.json and game.tw-pddl, not a movable or sliced
    variant, marked solvable."""
    out = []
    for dp, _, fs in os.walk(root):
        if "traj_data.json" not in fs or "game.tw-pddl" not in fs:
            continue
        if "movable" in dp or "Sliced" in dp:
            continue
        path = os.path.join(dp, "game.tw-pddl")
        try:
            with open(path) as f:
                if not json.load(f).get("solvable"):
                    continue
        except (OSError, ValueError):
            continue
        out.append(path)
    return sorted(out)


def _alfworld_play(game):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredExpert, AlfredInfos)
    infos = textworld.EnvInfos(won=True, admissible_commands=True,
                               extras=["gamefile", "expert_plan"])
    env_id = textworld.gym.register_games(
        [game], infos, batch_size=1, asynchronous=False,
        max_episode_steps=STEPS,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos,
                  AlfredExpert("handcoded")])
    env = textworld.gym.make(env_id)
    try:
        obs, info = env.reset()
        intro = clean(obs[0].replace("-= Welcome to TextWorld, ALFRED! =-", ""))
        steps = []
        for _ in range(STEPS):
            plan = info["extra.expert_plan"][0]
            if not plan:
                break
            obs, _, dones, info = env.step([plan[0]])
            steps.append((plan[0], clean(obs[0])))
            if info["won"][0] or dones[0]:
                break
        won = bool(info["won"][0])
    except Exception:
        return None
    finally:
        env.close()
    return (intro, steps) if won and steps else None


def alfworld(raw, limit=0):
    """(episode, held out?) for ALFWorld's training games, then its
    valid_seen games. The data comes down with ALFWorld's own
    alfworld-download, into `raw`: about 320 MB, 178 of it an object
    detector the text games never use."""
    import alfworld as _alf                                # noqa: F401
    data = os.path.abspath(raw)
    os.environ["ALFWORLD_DATA"] = data
    if not os.path.isdir(os.path.join(data, "json_2.1.1", "train")):
        print(f"  fetching ALFWorld's games into {data}", flush=True)
        subprocess.run(["alfworld-download", "--data-dir", data], check=True)
    for split, held in (("train", False), ("valid_seen", True)):
        games = _alfworld_games(os.path.join(data, "json_2.1.1", split))
        if limit:
            games = games[:limit if not held else max(1, limit // 10)]
        print(f"  ALFWorld {split}: {len(games):,} games", flush=True)
        for game in games:
            got = _alfworld_play(game)
            if got:
                yield transcript(got[0], got[1], ACT_HOUSE), held
