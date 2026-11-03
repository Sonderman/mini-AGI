"""
Replace what the model has been told about itself.

Two jobs. It strips the self-description baked into the chat corpus by an
older build - which states a pool of 74 experts, a top-16 router and a 4,096
character window, none of which is true any more - and it writes a fresh one
from corpora/self_knowledge.yaml, where every number is filled from the model
that exists.

Repetition is the point here and worth being honest about. A fixed
self-description is something the model should be able to recite, so the
corpus repeats each answer across many phrasings and orderings. That is
memorisation by design, and it is why `--target` is a dial: it decides how
much of the model's training is spent reciting.

What gets repeated is the FACT, not one string. An answer may come in several
wordings (a list under `say`), and every question is written in a few surface
forms - as written, capitalised, with a question mark. One wording repeated
221 times taught the model a sentence it then gave to questions it had not
been asked, "What are you?" among them: the yaml asked only "what are you",
and the one place the capitalised question had an answer was a quoted sample
of the model's own.

Nothing the model wrote goes back in. The source and documentation it reads
as itself (write_source) have the README's samples cut out, and the files the
self-description is generated FROM are not copied at all - their templates
and the stale fallback would contradict the filled answers.

    python3 tools/rebuild_self_knowledge.py --dry-run
    python3 tools/rebuild_self_knowledge.py --apply

Run it with no training in progress: it deletes and rewrites files under
data/train/chat, and a reader that opens one mid-rewrite will fail.
"""

import argparse
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MARK = re.compile(rb"<user>.*?</bot>", re.S)


def stale_questions():
    """
    Every question the old in-code self-description answered.

    Matching on the name alone is not enough: most of those answers never say
    "mini-AGI" - they begin "I hold a pool of 74 experts" - so a strip keyed
    on the name leaves the majority of the wrong text in place. The questions
    are the reliable handle, and they are still in the source.
    """
    qs = set()
    try:
        from corpora.chat import SELF_FROM_CODE
        for ask, _ in SELF_FROM_CODE:
            qs.update(q.strip().lower() for q in ask)
    except Exception:
        pass
    try:
        from corpora.chat import _load_self_knowledge
        for ask, _ in _load_self_knowledge():
            qs.update(q.strip().lower() for q in ask)
    except Exception:
        pass
    return qs


QS = stale_questions()
# claims the old build stated as fact and that are no longer true
STALE = (b"mini-AGI", b"I am mini", b"pool of 74 experts", b"the top 16 run",
         b"last 4096 characters", b"236 megabytes", b"room for 512")


def is_self(ex):
    """Is this exchange the model talking about itself?"""
    if any(s in ex for s in STALE):
        return True
    m = re.match(rb"<user>\s*(.*?)\s*</user>", ex, re.S)
    return bool(m) and m.group(1).decode("utf-8", "replace").strip().lower() in QS


def training_is_running():
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/cmdline", "rb") as f:
                a = f.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        # the process itself, not a shell whose command line happens to
        # quote the training command. Its arguments arrive NUL-separated: a
        # test for " read" never matched, and this never saw a run at all
        if ("python" in os.path.basename(a[0])
                and any(x.endswith("train.py") for x in a) and "read" in a):
            return p
    return None


def strip(chat_dir, apply):
    removed = kept = files = 0
    for r, _, fs in os.walk(chat_dir):
        if "self-knowledge" in r:
            continue
        for f in fs:
            p = os.path.join(r, f)
            b = open(p, "rb").read()
            out, cut = bytearray(), 0
            last = 0
            for m in MARK.finditer(b):
                if is_self(m.group(0)):
                    out += b[last:m.start()]
                    last = m.end()
                    cut += len(m.group(0))
            out += b[last:]
            removed += cut
            kept += len(out)
            files += 1
            if apply:
                if out.strip():
                    open(p, "wb").write(bytes(out))
                else:
                    os.remove(p)
    return files, removed, kept


def surface(q, rng):
    """
    One question as a person might type it.

    The yaml writes questions lowercase and unpunctuated; people do not, and
    to a model that reads characters "What are you?" is a different string
    from "what are you". Every form gets its share, so each has an answer.
    """
    base = q.strip().rstrip("?").rstrip()
    cap = base[:1].upper() + base[1:]
    return rng.choice((base, base + "?", cap + "?", cap))


def write_corpus(out_dir, target, apply, seed=0):
    from corpora.chat import _self_facts, _load_self_knowledge_variants
    F, items = _self_facts(), _load_self_knowledge_variants()
    if not items:
        raise SystemExit("  corpora/self_knowledge.yaml is empty")
    # every wording of an answer filled at once, so a placeholder that does
    # not exist stops the build here rather than in the middle of writing
    topics = [(list(ask), [s.format(**F) for s in says]) for ask, says in items]
    rng = random.Random(seed)
    if apply:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
    written = n = 0
    PER = 16_000
    while written < target:
        buf = []
        while sum(len(x) for x in buf) < PER:
            # a TOPIC first, then one of its questions and one of its
            # wordings. Drawing a question first gave a topic asked seven
            # ways seven times the share of one asked once, and its one
            # sentence was the string repeated most.
            ask, answers = topics[rng.randrange(len(topics))]
            buf.append(f"<user>\n{surface(rng.choice(ask), rng)}\n</user>\n"
                       f"<bot>\n{rng.choice(answers)}\n</bot>\n")
        text = "".join(buf)
        if apply:
            with open(os.path.join(out_dir, f"self-{n:05d}.txt"), "w") as f:
                f.write(text)
        written += len(text)
        n += 1
    questions = sum(len(ask) for ask, _ in topics)
    wordings = sum(len(says) for _, says in topics)
    return n, written, len(topics), questions, wordings


# THE MODEL'S OWN OUTPUT. tools/refresh_readme.py copies a round of samples
# from runs/samples.txt into the README, and runs/ is skipped below precisely
# so that nothing the model wrote is read back as fact - but the README was
# copied whole and brought them back. The only answer in this lane to the
# capitalised "What are you?" was a sample of the model's own, which it then
# kept giving: "up to 6" passes, a number no current answer contains.
SAMPLES = re.compile(r"<details>\s*<summary>[^<]*<b>[^<]*samples[^<]*</b>"
                     r"</summary>.*?</details>\s*", re.S | re.I)
# ...and a code block quoting the sample log anywhere else - a prompt line and
# a raw or adapted reading. The whole block goes: a reading's text has no end
# marker of its own, and can hold blank lines, so only the fence says where
# the model's words stop.
FENCE = re.compile(r"^```[^\n]*\n.*?^```[^\n]*(?:\n|\Z)", re.S | re.M)
SAMPLE_LOG = re.compile(r"^prompt: .*?^\[(?:raw|adapted)\]", re.S | re.M)

# What the self-description is generated FROM: its templates, the stale
# in-code fallback, and this tool, whose comments quote what is no longer
# true. Copied as "part of what I am made of" they say "{n_experts}" and
# "74 experts" where the filled answers say how many there are.
NOT_SOURCE = {"corpora/self_knowledge.yaml", "corpora/chat.py",
              "tools/rebuild_self_knowledge.py"}


def without_model_output(body):
    body = SAMPLES.sub("", body)
    return FENCE.sub(lambda m: "" if SAMPLE_LOG.search(m.group(0)) else m.group(0),
                     body)


def write_source(out_dir, root, apply):
    """
    The model's own source and documentation, as text it reads.

    A model that can recite its architecture from self_knowledge.yaml knows
    what it was TOLD. This is the thing itself: the code that defines the
    expert pool, the file that decides when one is deleted, the comment
    explaining why, and every piece of prose written about the project. It is the only part of the corpus that changes when the
    model changes, and it is generated rather than written so it cannot drift
    from what is actually running - the description this tool replaced still
    claimed 74 experts and a top-16 router long after both were false.

    Each file is framed so it reads as the model's own, not as some anonymous
    Python. The framing matters: unframed source would teach it to write code,
    which the code lane already does, rather than to know its own.
    """
    # Walk the repository rather than a list of directories. The old version
    # listed six folders one level deep and so never saw a folder it did not
    # name, or anything nested.
    #
    # SKIP is what keeps this from eating itself. `data` holds the corpus,
    # including this lane's own output, so including it would fold the
    # training data into the training data and grow without bound. `runs` is
    # logs and images, `weights*` is the model, `.git` is history. `memory` is
    # history too, kept for agents: every number and design the project has
    # left behind, which read as the model's own would be stale claims about
    # itself.
    SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules",
            "data", "runs", "memory", ".pytest_cache", ".idea", ".vscode"}
    CODE, DOCS = (".py",), (".md", ".yaml", ".yml")
    want = []
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in sorted(dirs)
                   if d not in SKIP and not d.startswith(".")
                   and not d.startswith("weights")]
        for name in sorted(files):
            if name.startswith("."):
                continue
            if not name.endswith(CODE + DOCS):
                continue
            rel = os.path.relpath(os.path.join(dp, name), root)
            if rel.replace(os.sep, "/") in NOT_SOURCE:
                continue
            want.append(rel)
    want = sorted(set(want))
    written = n = 0
    for rel in want:
        try:
            body = open(os.path.join(root, rel), errors="ignore").read()
        except OSError:
            continue
        body = without_model_output(body)
        if not body.strip():
            continue
        kind = "documentation" if rel.endswith((".md", ".yaml")) else "source"
        text = (f"<user>\nshow me your {kind}: {rel}\n</user>\n<bot>\n"
                f"This is {rel}, part of what I am made of.\n\n"
                f"{body}\n</bot>\n")
        if apply:
            with open(os.path.join(out_dir, f"auto-src-{n:04d}.txt"), "w") as f:
                f.write(text)
        written += len(text)
        n += 1
    return n, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", default="data/train/chat")
    ap.add_argument("--out", default="data/train/self-knowledge",
                    help="its own LANE now, not a folder inside chat: the "
                         "reader picks a lane at random, so buried under chat "
                         "this was 0.25%% of reading and the model had seen "
                         "2%% of it once after 284M characters")
    ap.add_argument("--source", action="store_true", default=True,
                    help="also write the model's own code and docs")
    ap.add_argument("--no-source", dest="source", action="store_false")
    ap.add_argument("--target", type=float, default=5.0,
                    help="million characters of self-description to write; "
                         "the old build had 5.5")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    apply_it = a.apply and not a.dry_run

    pid = training_is_running()
    if apply_it and pid:
        print(f"  training is running (pid {pid}). It reads the files this "
              f"rewrites,\n  so stop it first.")
        return 1

    out = a.out
    files, removed, kept = strip(a.chat, apply_it)
    print(f"  stale self-description: {removed/1e6:.2f}M characters across "
          f"{files:,} files{'' if apply_it else '  (nothing written)'}")
    n, w, topics, phr, wordings = write_corpus(out, a.target * 1e6, apply_it)
    print(f"  fresh self-description: {w/1e6:.2f}M characters in {n:,} files, "
          f"from {topics} topics, {phr} questions (each in 4 surface forms) "
          f"and {wordings} wordings of the answers")
    if a.source:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sn, sw = write_source(out, root, apply_it)
        print(f"  own source and docs   : {sw/1e6:.2f}M characters in "
              f"{sn:,} files")
        w += sw
    print(f"  self-knowledge lane    : {w/1e6:.2f}M characters -> {out}")

    # HOW OFTEN THE MODEL WILL RE-READ THIS. The reader picks a lane at random,
    # so the lane's share is 1/(number of lanes) - it does not depend on how
    # big the lane is. A small lane is therefore read MORE times, not less, and
    # --target is the only thing deciding whether the model learns this or
    # memorises it. At 12.5% of reading, 5M characters is re-read every 40M and
    # 33M every 264M.
    train_root = os.path.dirname(os.path.abspath(out))
    try:
        lanes = [x for x in os.listdir(train_root)
                 if os.path.isdir(os.path.join(train_root, x))]
    except OSError:
        lanes = []
    if lanes and w:
        share = 1.0 / len(lanes)
        print(f"  {len(lanes)} lanes, so this is {100*share:.1f}% of reading: "
              f"one pass every {w/share/1e6:.0f}M characters read")
        existing = 0
        for root_, _, fs in os.walk(out):
            for f_ in fs:
                try: existing += os.path.getsize(os.path.join(root_, f_))
                except OSError: pass
        if existing and not apply_it and abs(existing - w) / max(existing, 1) > 0.2:
            print(f"  NOTE the lane currently holds {existing/1e6:.1f}M and "
                  f"--apply REPLACES it with {w/1e6:.1f}M\n"
                  f"       (--target {existing/1e6:.0f} keeps it as it is)")
    if not apply_it:
        print("\n  re-run with --apply to make the change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
