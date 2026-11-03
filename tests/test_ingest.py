#!/usr/bin/env python3
"""
Which files are read as text, and the cache that remembers the answer.

The rule (GitHub issue #30):
  a character cut by the 4,096-byte sample no longer makes a text file look
  binary - at every byte of every width of character
  everything else that made a file binary still does: a character broken
  anywhere in the sample, a file that really ends inside one, a NUL, a binary
  type, an empty file, mostly unprintable text
  nothing the old rule accepted is refused, and nothing is newly accepted but
  a file the old rule refused only for a cut character - checked against the
  old rule itself on thousands of random files

The cache:
  an unchanged file is not opened again; one rewritten under the same name is
  checked afresh; a cache from the old rule is not trusted; files that are
  gone are forgotten; overlapping paths list a file once; what is skipped is
  said with the reason, and quiet says nothing
"""
import contextlib
import io
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minagi.ingest as ingest                          # noqa: E402
from minagi.ingest import collect, looks_like_text, why_not_text  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def old_rule(path, probe=4096):
    """looks_like_text as it was before the fix, frozen for comparison."""
    if os.path.splitext(path)[1].lower() in ingest.SKIP_EXT:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(probe)
    except OSError:
        return False
    if not head or b"\x00" in head:
        return False
    try:
        s = head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return sum(1 for c in s if c.isprintable() or c in "\n\r\t") / len(s) > 0.9


tmp = tempfile.TemporaryDirectory()
T = tmp.name


def put(name, data):
    p = os.path.join(T, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p


# ---------------------------------------------------------- 1. the issue itself
p = put("issue.txt", b"a" * 4095 + "●\n".encode())
check("the issue's file - 4,095 bytes, then a character the sample cuts - is text",
      looks_like_text(p) and not old_rule(p))

cut = []
for ch in ("é", "€", "●", "中", "𝄞"):
    b = ch.encode()
    for k in range(1, len(b)):           # the sample keeps k bytes of it
        q = put(f"cut-{len(b)}-{k}.txt", b"x" * (4096 - k) + b + "more text\n".encode() * 50)
        cut.append(looks_like_text(q))
check("...and so is every width of character, cut after every one of its bytes",
      all(cut), f"{sum(cut)} of {len(cut)}")

# ---------------------------------------------------- 2. what still is not text
cases = {
    "broken.txt": (b"a" * 100 + b"\xff" + b"a" * 5000, "not UTF-8"),
    "short-ends-mid.txt": (b"hello " + "€".encode()[:2], "not UTF-8"),
    "exact-ends-mid.txt": (b"a" * 4094 + "€".encode()[:2], "not UTF-8"),
    "nul.txt": (b"abc\x00def", "with a NUL byte"),
    "picture.png": (b"plain text in a png", "of a binary type"),
    "empty.txt": (b"", "empty"),
    "controls.txt": (bytes(range(1, 9)) * 200, "mostly unprintable"),
}
got = {n: why_not_text(put(n, d)) for n, (d, _) in cases.items()}
check("a broken character, a file that ends inside one, NUL, a binary type, "
      "empty and unprintable are still not text, each for its reason",
      all(got[n] == w for n, (_, w) in cases.items()),
      ", ".join(f"{n}: {got[n]}" for n in got if got[n] != cases[n][1]))
check("...a file that really ends mid-character at exactly 4,096 bytes is not "
      "taken for a cut one", got["exact-ends-mid.txt"] == "not UTF-8")
check("...a NUL past the sample does not count, as before",
      looks_like_text(put("late-nul.txt", b"a" * 5000 + b"\x00")))
check("...and a path that cannot be opened is unreadable",
      why_not_text(os.path.join(T, "no-such-file.txt")) == "unreadable")

# ---------------------------------------- 3. against the old rule, at random
rng = random.Random(30)
alphabet = (["a", "b", " ", "\n", "e", "t"] * 8 + ["é", "€", "●", "中", "𝄞", "ж"]
            + ["\t", "\x01", "\x1b"])
newly, regressed, wrong = 0, [], []
for i in range(3000):
    n = rng.choice([10, 500, 4090, 4094, 4095, 4096, 4097, 5000, 9000])
    s = "".join(rng.choice(alphabet) for _ in range(n)).encode()[:n + rng.randint(0, 3)]
    if rng.random() < 0.1:               # a stray byte somewhere
        k = rng.randrange(len(s) + 1)
        s = s[:k] + bytes([rng.choice([0x00, 0x80, 0xbf, 0xc3, 0xff])]) + s[k:]
    q = put(f"fuzz/{i}.txt", s)
    old, new = old_rule(q), looks_like_text(q)
    if old and not new:
        regressed.append(q)
    if new and not old:
        newly += 1
        head = s[:4096]
        try:                             # it must be exactly the cut case
            head.decode("utf-8")
            wrong.append(q)
        except UnicodeDecodeError as e:
            if not (len(s) > 4096 and e.start >= 4093 and e.reason ==
                    "unexpected end of data"):
                wrong.append(q)
check("nothing the old rule accepted is refused", not regressed,
      f"{len(regressed)} of 3,000 random files")
check("...and the only files newly accepted are ones the sample cut a character in",
      not wrong and newly > 0, f"{newly} newly accepted, {len(wrong)} for any other reason")

# --------------------------------------------------------------- 4. the cache
A = os.path.join(T, "corpus")
put("corpus/x.txt", b"text\n" * 10)
put("corpus/y.png", b"text, but a png")
put("corpus/z.txt", b"has a \x00 in it")
put("corpus/sub/w.txt", b"more text\n" * 10)
C = os.path.join(T, "cache.json")
calls = []
real = ingest.why_not_text


def counted(path, probe=4096):
    calls.append(path)
    return real(path, probe)


ingest.why_not_text = counted
first = collect([A], cache=C, quiet=True)
n_first = len(calls)
calls.clear()
second = collect([A], cache=C, quiet=True)
check("the first survey opens every file, a second opens none",
      n_first == 4 and not calls and first == second
      and first == [os.path.join(A, "sub/w.txt"), os.path.join(A, "x.txt")],
      f"{n_first} then {len(calls)} opened")

calls.clear()
zp = os.path.join(A, "z.txt")
with open(zp, "wb") as f:
    f.write(b"no longer has one\n")
os.utime(zp, ns=(10**18, 10**18))       # a different time, whatever the clock
third = collect([A], cache=C, quiet=True)
check("a file rewritten under the same name is checked again, and now read",
      calls == [zp] and zp in third, f"opened {len(calls)}")

with open(C, "w") as f:                 # a cache in the old layout
    json.dump({"entries": {"anything": [os.path.join(A, "x.txt")]}}, f)
calls.clear()
fourth = collect([A], cache=C, quiet=True)
check("a cache written under the old rule is not trusted - every file is checked",
      len(calls) == 4 and fourth == third, f"opened {len(calls)}")

os.remove(os.path.join(A, "sub/w.txt"))
collect([A], cache=C, quiet=True)
held = json.load(open(C))["files"]
check("a file that is gone is forgotten by the cache",
      os.path.join(A, "sub/w.txt") not in held and os.path.join(A, "x.txt") in held)

put("corpus/sub/w.txt", b"more text\n" * 10)
both = collect([A, os.path.join(A, "sub")], cache=C, quiet=True)
check("overlapping paths list a file once", len(both) == len(set(both)) == 3)

ingest.why_not_text = real
out = io.StringIO()
with contextlib.redirect_stdout(out):
    collect([A], cache=C)
said = out.getvalue()
check("what is skipped is said, with the reason",
      "skipped 1 of 4 files" in said and "of a binary type" in said, said.strip())
out = io.StringIO()
with contextlib.redirect_stdout(out):
    collect([A], cache=C, quiet=True)
check("...and quiet says nothing", out.getvalue() == "")
out = io.StringIO()
with contextlib.redirect_stdout(out):
    collect([os.path.join(A, "sub")], cache=C)
check("...nor does a survey that skipped nothing", out.getvalue() == "")

check("no cache at all still works",
      collect([A], cache=None, quiet=True) == fourth)

tmp.cleanup()
print("\n  all ingest tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
