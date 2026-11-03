#!/usr/bin/env python3
"""
Every generator writes where `corpora expand` looks, in the alphabet the
model reads.

Two ways this broke at once, and neither showed up until someone outside
tried it. chat.py defaulted to --out data_chat while expand reads
data_chat_char, so the bytes landed where nothing would find them. And
chat.py and code.py still carried --char from when there was a BPE path:
without the flag they fell through to a tokenizer at data/tokenizer.json
that does not exist, or fitted a BPE nobody wanted. `corpora all` passes no
flags, so its chat and code lanes were broken from the day it was written.

Nothing here builds a corpus - these are seconds-cheap contract checks on the
argument parsers, which is exactly the layer that drifted.
"""
import io
import contextlib
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corpora import expand                                   # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


# corpora/__main__.py target -> module, for the four that generate .bin
GEN = {"code": "corpora.code", "arithmetic": "corpora.arithmetic",
       "chat": "corpora.chat", "chess": "corpora.chess_games"}
# what expand reads: name -> source directory
EXPECT = {name: src for name, src, _ext in expand.SOURCES}


def defaults(mod):
    """The parser's defaults, without running the generator."""
    m = importlib.import_module(mod)
    import argparse
    got = {}
    real = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        raise SystemExit(self)               # hand the parser back

    argparse.ArgumentParser.parse_args = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            m.main()
    except SystemExit as e:
        p = e.code
        if isinstance(p, argparse.ArgumentParser):
            got = {a.dest: a.default for a in p._actions}
    except BaseException:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real
    return got


for target, mod in GEN.items():
    d = defaults(mod)
    if not d:
        check(f"{target}: parser inspected", False, "could not read defaults")
        continue
    want = EXPECT.get(target)
    check(f"{target}: --out default is what expand reads",
          d.get("out") == want, f"{d.get('out')!r} vs expand's {want!r}")
    check(f"{target}: no --char (the BPE path is gone)", "char" not in d)
    check(f"{target}: no --tokenizer pointing at a missing file",
          "tokenizer" not in d)

# and the dispatcher lists every generator expand knows about
import corpora.__main__ as cm                                # noqa: E402
for name in EXPECT:
    check(f"`corpora {name}` is a real target", name in cm.BUILDERS)

print("\n  all corpora contract tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
