#!/usr/bin/env python3
"""
The RAM tier reads expert files without np.load, and reads exactly what
np.load reads.

  every array of a freshly created weights directory: the same bytes, dtype
  and shape - fp32 weights and the int16 that carries bf16 moments alike
  an expert fetched through the tier is the same tensors np.load gives
  names the file does not have are left out, as `k in z.files` left them
  a Fortran-ordered array, a 0-d one and an empty one come back as they went in
  a compressed archive, which this cannot read in place, goes through np.load
  a member shorter than its own header says is refused, not half-read
"""
import io
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


from minagi.paged import Tiers, read_npz                       # noqa: E402
from minagi.precision import as_bf16                           # noqa: E402


def same(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes(order="A") == b.tobytes(order="A")
            and a.flags.f_contiguous == b.flags.f_contiguous)


tmp = tempfile.mkdtemp(prefix="minagi-npz-")
try:
    from minagi.create import create                          # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=6, resident=2, d_ff=32, d_model=32,
           trunk_d_ff=32, n_head=2, block=64, max_steps=3, top_k=2, depth=1)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(wdir) for f in fs
                   if f.endswith(".npz"))
    bad, n = [], 0
    for p in files:
        f = os.path.relpath(p, wdir)
        with np.load(p) as z:
            got = read_npz(p, z.files)
            for k in z.files:
                n += 1
                if not same(got[k], z[k]):
                    bad.append(f"{f}:{k}")
    check("every array of a created weights directory reads as np.load reads it",
          n > 0 and not bad, f"{n} arrays in {len(files)} files" + (f", differ: {bad}" if bad else ""))

    # an expert with stepped moments, written the way the tier writes them
    edir = os.path.join(wdir, "experts")
    t = Tiers(edir, 32, 32, ram_capacity=4)
    ent = t.fetch(0)
    g = torch.Generator().manual_seed(1)
    stepped = {k: torch.randn(ent[k].shape, generator=g) for k in ("w1", "w3", "w2")}
    for k in ("w1", "w3", "w2"):
        stepped[k + "_m"] = torch.randn(ent[k].shape, generator=g) * 1e-3
        stepped[k + "_v"] = torch.rand(ent[k].shape, generator=g) * 1e-6
    t._to_disk(0, stepped)
    with np.load(t._file(0)) as z:
        want = {k: torch.from_numpy(z[k]) for k in ("w1", "w3", "w2")}
        want.update({k: as_bf16(z[k]) for k in z.files if k.endswith(("_m", "_v"))})
        packed = all(z[k].dtype == np.int16 for k in z.files if k.endswith(("_m", "_v")))
    got = Tiers(edir, 32, 32, ram_capacity=4)._from_disk(0)
    check("an expert read through the tier: the tensors np.load gives, moments as "
          "the bf16 they are stored as",
          packed and got.keys() == want.keys()
          and all(torch.equal(got[k], want[k]) and got[k].dtype == want[k].dtype for k in want))

    arrays = {"c": np.arange(12, dtype=np.float32).reshape(3, 4),
              "f": np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
              "s": np.array(3.5, dtype=np.float32),
              "e": np.zeros((0, 7), dtype=np.int16),
              "u": np.arange(5, dtype=np.uint8)}
    p = os.path.join(tmp, "a.npz")
    np.savez(p, **arrays)
    got = read_npz(p, ["c", "f", "s", "e", "u", "missing"])
    check("a name the file lacks is left out", "missing" not in got and len(got) == 5)
    check("Fortran order, 0-d and empty arrays come back as they went in",
          all(same(got[k], v) for k, v in arrays.items()))

    p = os.path.join(tmp, "z.npz")
    np.savez_compressed(p, **arrays)
    got = read_npz(p, list(arrays))
    check("a compressed archive goes through np.load and reads the same",
          all(same(got[k], v) for k, v in arrays.items()))

    # a member whose header promises 100 floats and holds 50
    hdr = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        hdr, {"descr": "<f4", "fortran_order": False, "shape": (100,)})
    p = os.path.join(tmp, "short.npz")
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("x.npy", hdr.getvalue() + np.zeros(50, np.float32).tobytes())
    try:
        read_npz(p, ["x"])
        refused = False
    except ValueError:
        refused = True
    check("a member shorter than its own header is refused", refused)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all npz reader tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
