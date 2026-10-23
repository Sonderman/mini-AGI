#!/usr/bin/env python3
"""
The enwik9 slice this model is scored on, and nothing else.

    python3 -m corpora enwik9 --out data/enwik9/wikipedia

enwik9 is the first billion bytes of English Wikipedia from March 2006 (Matt
Mahoney's Large Text Compression Benchmark), the text compressors and
language models are most often compared on. This model's Wikipedia comes from
the same 2006 dump, so it is not scored on all of it: it is scored on the last
50 MB, bytes 950,000,000 to 1,000,000,000, none of whose pages it trained on -
held out the way enwik8's standard test split is its last 5 MB.
tools/measure_enwik9.py and tools/measure_enwik9_adapt.py read it from
data/enwik9/wikipedia/test.txt.

THE SLICE IS CHECKED. It is written only if its sha256 is that of the slice
every number in the README was measured on, so another download, or another
cut, cannot pass for it.

ONLY THE ZIP IS DOWNLOADED, about 320 MB. The slice is read straight out of
it, without unpacking the whole billion bytes, and the zip is deleted after
unless --keep-zip says otherwise. --zip uses one already on disk.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

URL = "https://mattmahoney.net/dc/enwik9.zip"
START, END = 950_000_000, 1_000_000_000
SHA256 = "fa616cef732f5622ff8bd6706fec08638491beb6b7934786bcf95bf69c8a86cc"
CHUNK = 16 << 20


def download(url, dest):
    """`url` to `dest`, through a .part file, so a broken download never
    looks like a finished one."""
    req = urllib.request.Request(url, headers={"User-Agent": "mini-AGI/1.0"})
    part = dest + ".part"
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        size = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            f.write(b)
            got += len(b)
            if size:
                print(f"\r    {got / 1e6:,.0f} of {size / 1e6:,.0f} MB",
                      end="", flush=True)
    print()
    os.replace(part, dest)


def cut(zpath, out, start=START, end=END, sha256=SHA256, member="enwik9"):
    """Bytes start..end of `member` in the zip at `zpath`, written to `out` if
    their sha256 is `sha256`. Returns the sha256 found; writes nothing when it
    is not the one expected."""
    h = hashlib.sha256()
    parts = []
    with zipfile.ZipFile(zpath) as z, z.open(member) as f:
        left = start
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                raise ValueError(f"{member} ends before byte {start:,}")
            left -= len(b)
        left = end - start
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                raise ValueError(f"{member} ends before byte {end:,}")
            h.update(b)
            parts.append(b)
            left -= len(b)
    got = h.hexdigest()
    if got != sha256:
        return got
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        for b in parts:
            f.write(b)
    os.replace(out + ".tmp", out)
    return got


def main():
    ap = argparse.ArgumentParser(prog="python3 -m corpora enwik9")
    ap.add_argument("--out", default="data/enwik9/wikipedia",
                    help="the folder test.txt is written to")
    ap.add_argument("--zip", default=None,
                    help="an enwik9.zip already on disk, instead of "
                         "downloading one")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the downloaded zip")
    a = ap.parse_args()

    dest = os.path.join(a.out, "test.txt")
    if os.path.exists(dest):
        with open(dest, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == SHA256:
                print(f"  {dest} is already the slice")
                return 0
    zpath = a.zip or os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                  "enwik9.zip")
    fetched = False
    if not os.path.exists(zpath):
        print(f"  downloading {URL} -> {zpath}")
        download(URL, zpath)
        fetched = True
    print(f"  bytes {START:,} to {END:,} of enwik9 -> {dest}")
    got = cut(zpath, dest)
    if fetched and not a.keep_zip:
        os.remove(zpath)
    if got != SHA256:
        print(f"  the slice's sha256 is {got}, not {SHA256} - not the enwik9 "
              f"the README was measured on; nothing written", file=sys.stderr)
        return 1
    print(f"  {END - START:,} bytes, sha256 checked")
    print("  score it:  ./measure_benchmarks.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
