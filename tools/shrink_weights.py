#!/usr/bin/env python3
"""
Narrow Adam's moments to bfloat16 in a weights directory, in place.

Two thirds of every expert file is Adam state, not weights. The moments are
tiny and span an enormous range - m median 3.5e-09, v median 2.0e-15 - so what
they need is exponent, which bfloat16 keeps in full, and not mantissa.
Measured on the real pool against a typical Adam update of 1.8e-05:

    m bfloat16    0.10% of its own magnitude    0.000x one update
    v bfloat16    0.10% of its own magnitude    0.000x one update

The weights themselves stay fp32 and are not touched: bfloat16 costs 0.80x an
update there, and float16's resolution at a typical weight is comparable to
an update, so half of what an expert learns would round away - and an expert
is rewritten a median of 406 times.

Nothing here is required. New writes are already bf16, so a directory converts
itself as experts are paged out. This just does it now, for the space.

    python3 tools/shrink_weights.py weights
    python3 tools/shrink_weights.py weights --dry-run
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.precision import is_moment, pack_bf16, unpack_bf16  # noqa: E402


def convert(path, dry=False):
    """Rewrite one .npz with its moments in bf16. Returns (before, after)."""
    before = os.path.getsize(path)
    z = np.load(path)
    if all(z[k].dtype == np.int16 for k in z.files if is_moment(k)) \
            and any(is_moment(k) for k in z.files):
        return before, before                       # already converted
    out = {}
    for k in z.files:
        a = z[k]
        out[k] = pack_bf16(a) if (is_moment(k) and a.dtype != np.int16) else a
    if dry:
        import io
        b = io.BytesIO()
        np.savez(b, **out)
        return before, len(b.getvalue())
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)                            # atomic
    return before, os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default="weights")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be saved and change nothing")
    a = ap.parse_args()

    targets = []
    opt = os.path.join(a.weights, "optim.npz")
    if os.path.exists(opt):
        targets.append(opt)
    exp = os.path.join(a.weights, "experts")
    if os.path.isdir(exp):
        targets += [os.path.join(exp, f) for f in sorted(os.listdir(exp))
                    if f.endswith(".npz")]
    if not targets:
        raise SystemExit(f"nothing to convert in {a.weights}")

    # Verify on one file before touching any: unpacking must return what was
    # packed, to within what bf16 can represent.
    z = np.load(targets[-1])
    for k in z.files:
        if is_moment(k) and z[k].dtype != np.int16:
            orig = z[k].astype(np.float64)
            back = unpack_bf16(pack_bf16(z[k])).numpy().astype(np.float64)
            nz = orig != 0
            err = np.abs(back[nz] - orig[nz]) / np.abs(orig[nz])
            print(f"  round-trip check on {os.path.basename(targets[-1])}:{k}"
                  f"  median {np.median(err):.2e}  worst {err.max():.2e}")
            if err.max() > 0.02:
                raise SystemExit("  round-trip error too large - not converting")
            break

    tot_b = tot_a = 0
    for i, t in enumerate(targets):
        b, aft = convert(t, a.dry_run)
        tot_b += b
        tot_a += aft
        if (i + 1) % 25 == 0 or i + 1 == len(targets):
            print(f"  {i+1}/{len(targets)}  {tot_b/1e9:.2f} GB -> "
                  f"{tot_a/1e9:.2f} GB", end="\r", flush=True)
    print()
    saved = tot_b - tot_a
    print(f"  {'would save' if a.dry_run else 'saved'} "
          f"{saved/1e9:.2f} GB of {tot_b/1e9:.2f} GB "
          f"({100*saved/max(tot_b,1):.0f}%)")


if __name__ == "__main__":
    main()
