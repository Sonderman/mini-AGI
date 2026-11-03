#!/usr/bin/env python3
"""
The enwik9 slice the README's enwik9 table is scored on (corpora/enwik9.py).

What these tests hold it to:
  it cuts exactly the bytes asked for out of the zip, without unpacking it,
  and writes them only if their sha256 is the one expected; another cut, or
  another file, is refused and nothing is written
  the slice is the last 50 MB of enwik9, bytes 950,000,000 to 1,000,000,000,
  and the slice on disk, where there is one, is the one the code expects
  `python3 -m corpora enwik9` reaches it, and it writes where the measuring
  tools read
"""
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import corpora.__main__ as CM                               # noqa: E402
import corpora.enwik9 as CE                                 # noqa: E402
import measure_enwik9 as ME                                 # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-enwik9-slice-")
try:
    body = bytes(range(256)) * 400                    # 102,400 bytes
    zpath = os.path.join(tmp, "enwik9.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("enwik9", body)
    want = body[90_000:100_000]
    sha = hashlib.sha256(want).hexdigest()
    out = os.path.join(tmp, "wikipedia", "test.txt")
    got = CE.cut(zpath, out, 90_000, 100_000, sha)
    check("it cuts exactly the bytes asked for and writes them when their "
          "sha256 is the one expected",
          got == sha and open(out, "rb").read() == want)
    os.remove(out)
    got2 = CE.cut(zpath, out, 90_001, 100_001, sha)
    check("...another cut is refused, and nothing is written",
          got2 != sha and not os.path.exists(out)
          and not os.path.exists(out + ".tmp"))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

check("the slice is the last 50 MB of enwik9",
      (CE.START, CE.END) == (950_000_000, 1_000_000_000)
      and CE.URL.endswith("/dc/enwik9.zip"))
on_disk = os.path.join(ME.SLICE, "wikipedia", "test.txt")
if os.path.exists(on_disk):
    with open(on_disk, "rb") as f:
        disk_sha = hashlib.sha256(f.read()).hexdigest()
    check("...and the slice on disk is the one the code expects",
          disk_sha == CE.SHA256, disk_sha[:16])
check("`python3 -m corpora enwik9` reaches it, and it writes where the "
      "measuring tools read",
      CM.BUILDERS.get("enwik9") == "corpora.enwik9"
      and os.path.normpath(os.path.join(ROOT, "data/enwik9/wikipedia"))
      .startswith(os.path.normpath(ME.SLICE)))

print("\n  all enwik9 slice tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
