"""
Adam's moments store as bfloat16; the weights must not.

Two thirds of an expert file is Adam state. The moments are tiny and span an
enormous range - m median 3.5e-09, v median 2.0e-15 - so what they need is
exponent, which bf16 keeps in full. The weights are the opposite case: bf16
costs 0.80x an Adam update there, which is most of what the expert just
learned, and an expert is rewritten a median of 406 times.
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.paged import Tiers                                   # noqa: E402
from minagi.precision import is_moment, pack_bf16, unpack_bf16   # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    ok = ok and cond


check("is_moment knows both naming schemes",
      is_moment("w1_m") and is_moment("a|v") and not is_moment("w1")
      and not is_moment("a|t"))

# moments survive the round trip to within bf16's floor
rng = np.random.default_rng(0)
for name, scale in (("m", 3.5e-9), ("v", 2.0e-15)):
    x = torch.from_numpy((rng.standard_normal(20000) * scale).astype(np.float32))
    r = unpack_bf16(pack_bf16(x))
    nz = x != 0
    err = ((r[nz] - x[nz]).abs() / x[nz].abs()).median().item()
    check(f"{name} survives bf16 (median rel err)", err < 5e-3, f"{err:.2e}")

# ...and a value far below fp16's smallest subnormal is not lost
tiny = torch.tensor([2.0e-15], dtype=torch.float32)
check("bf16 keeps a value fp16 would flush to zero",
      float(unpack_bf16(pack_bf16(tiny))) > 0
      and float(tiny.to(torch.float16)) == 0.0)

# the write path: weights fp32 on disk, moments int16-packed bf16
with tempfile.TemporaryDirectory() as d:
    t = Tiers(d, 8, 4)
    ent = {"w1": torch.randn(4, 8), "w3": torch.randn(4, 8),
           "w2": torch.randn(8, 4), "w1_m": torch.randn(4, 8) * 1e-9,
           "w1_v": torch.rand(4, 8) * 1e-15}
    t._to_disk(0, ent)
    z = np.load(t._file(0))
    check("weights stay float32 on disk",
          all(z[k].dtype == np.float32 for k in ("w1", "w3", "w2")),
          str({k: str(z[k].dtype) for k in ("w1", "w3", "w2")}))
    check("moments are packed to 16 bits",
          all(z[k].dtype == np.int16 for k in ("w1_m", "w1_v")))

    back = t._from_disk(0)
    check("weights come back BIT-IDENTICAL",
          all(torch.equal(back[k], ent[k]) for k in ("w1", "w3", "w2")))
    # RAM keeps the moments at the width the file does, so the tier holds
    # half as many again in the same memory and half the bytes go to the
    # card, where the optimiser widens them
    check("moments come back as bfloat16, exactly the values written",
          all(back[k].dtype == torch.bfloat16
              and torch.equal(back[k].float(), unpack_bf16(pack_bf16(ent[k])))
              for k in ("w1_m", "w1_v")))

    # an older all-fp32 file must still load - directories predate this change
    np.savez(t._file(1), **{k: v.numpy() for k, v in ent.items()})
    old = t._from_disk(1)
    check("a pre-existing fp32 file still loads",
          all(torch.equal(old[k], ent[k]) for k in ("w1", "w3", "w2")))
    check("...its fp32 moments narrowed as its next write would narrow them",
          all(old[k].dtype == torch.bfloat16
              and torch.equal(old[k].float(), unpack_bf16(pack_bf16(ent[k])))
              for k in ("w1_m", "w1_v")))


    # ---- the disk ceiling must be measured against these very files -----
    # growth.max_disk_gb is enforced against PagedPool.disk_bytes(), which
    # assumed fp32 moments - twelve bytes a parameter. This file is the reason
    # that was wrong: the moments are bf16, so an expert is EIGHT bytes a
    # parameter. The estimate ran 1.52x over and the ceiling bit a third early.
    # Measured live: growth refused at 261 experts on a 10 GB budget because
    # (261 + 4) x 37.75 MB came to 10.003 GB, while the directory held 6.48 GB.
    from minagi.paged import PagedPool                        # noqa: E402
    d2 = tempfile.mkdtemp()
    # big enough that npz's per-array headers are noise: at DM 8 / DF 16 an
    # expert is 384 parameters and nine zip members of header dwarf the data.
    DM, DF, N2 = 64, 256, 2
    for i in range(N2):
        e = {"w1": torch.randn(DF, DM), "w3": torch.randn(DF, DM),
             "w2": torch.randn(DM, DF)}
        for nm in ("w1", "w3", "w2"):
            e[nm + "_m"] = torch.randn_like(e[nm])
            e[nm + "_v"] = torch.rand_like(e[nm])
        Tiers(d2, DM, DF, ram_capacity=4)._to_disk(i, e)
    pool = PagedPool(d2, DM, DF, N2, resident=1, ram_capacity=2, device="cpu")
    real = sum(os.path.getsize(os.path.join(d2, f))
               for f in os.listdir(d2) if f.endswith(".npz")) / N2
    est = pool.disk_bytes(0) / N2
    check("the disk estimate matches what an expert file weighs",
          0.9 <= est / real <= 1.4,
          f"estimate {est/1e3:.1f} kB, file {real/1e3:.1f} kB ({est/real:.2f}x)")
    check("...and never under-counts by more than a tenth",
          est >= real * 0.9,
          "a ceiling that under-counts would overshoot growth.max_disk_gb; "
          "which side it errs on flips with size - zip headers at small "
          "d_ff, compression at real ones")
    per_param = pool.disk_bytes(0) / (N2 * 3 * DM * DF)
    check("it is 8 bytes a parameter: fp32 weight + bf16 m + bf16 v",
          abs(per_param - 8.0) < 1e-9, f"{per_param:.2f} bytes/param")

print("\n  " + ("all moment-storage tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
