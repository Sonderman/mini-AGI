"""
Put a weights directory back into agreement with itself.

The expert files and the tensors that index them can drift apart: pruning
deletes and renumbers files on disk immediately, while the manifest, the
routers and the gate are only written at a checkpoint. A run that prunes and
then dies leaves a directory whose index names experts that are no longer
there, and the next load fails on the first missing file.

This keeps only the experts that still exist, and subsets EVERY per-expert
tensor by the same surviving ids, so an expert stays paired with the router
row and gate that were trained for it. Then it renumbers the files to be
contiguous again.

    python3 tools/repair_pool.py weights            # report only
    python3 tools/repair_pool.py weights --apply    # rewrite

It never invents an expert. Anything it cannot keep, it drops.
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np


def survivors(wdir, n_declared):
    d = os.path.join(wdir, "experts")
    have = set()
    for f in os.listdir(d):
        m = re.search(r"(\d+)", f)
        if m and f.endswith(".npz"):
            have.add(int(m.group(1)))
    return sorted(i for i in have if i < n_declared), sorted(have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default="weights")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    w = a.weights

    man = json.load(open(os.path.join(w, "manifest.json")))
    # This tool predates stable expert ids and assumes file number == router
    # row. Under uids they are different things: an expert at position 7 may
    # live in e00193.npz, so every id past the row count looks like a file the
    # index does not describe, and every position whose uid differs looks
    # gone. Applied, it would drop live experts and pair the survivors with
    # the wrong router rows. Refuse instead - a loud stop beats quiet damage.
    _tel = man.get("telemetry") or {}
    if _tel.get("uid") and any(int(u) != i for i, u in enumerate(_tel["uid"])):
        raise SystemExit(
            f"{w} names its expert files by stable id, and this tool assumes "
            f"they are positions. It would report live experts as missing and "
            f"subset the per-expert tensors against the wrong rows.\n\n"
            f"If the directory really is inconsistent, the authority is "
            f"telemetry.uid: n_experts must equal len(uid), every uid must "
            f"have a file, and the router tensors must have len(uid) rows. "
            f"Check those three and fix the one that disagrees.")
    # The index space is however many rows the per-expert tensors carry, not
    # what the manifest claims. When a run dies between changing the pool and
    # writing the index, the manifest is the stale one of the two - the
    # routers and the gate are what an expert id actually means.
    # only the tensors that really carry one row per expert - a router
    # weight is [experts, d_model] and the gate is [experts], while
    # depth_emb is [d_model] and would otherwise be mistaken for a huge pool
    rows = 0
    with np.load(os.path.join(w, "routers.npz")) as z:
        for k in z.files:
            t = z[k]
            if k.endswith("router.weight") and t.ndim == 2:
                rows = max(rows, int(t.shape[0]))
            elif k == "pool.gate" and t.ndim == 1:
                rows = max(rows, int(t.shape[0]))
    n = max(int(man["n_experts"]), rows)
    if rows != int(man["n_experts"]):
        print(f"  manifest says {man['n_experts']} experts but the routers "
              f"carry {rows} rows; trusting the routers")
    keep, all_ids = survivors(w, n)
    beyond = [i for i in all_ids if i >= n]

    print(f"  index space       {n} experts")
    print(f"  files on disk     {len(all_ids)}  "
          f"(ids {all_ids[0]}..{all_ids[-1]})")
    print(f"  usable            {len(keep)}  "
          f"(a file exists and the index has a row for it)")
    if beyond:
        print(f"  past the index    {len(beyond)} files the manifest does not "
              f"describe: {beyond[:8]}{'...' if len(beyond) > 8 else ''}")
    missing = [i for i in range(n) if i not in set(keep)]
    print(f"  gone              {len(missing)}: "
          f"{missing[:12]}{'...' if len(missing) > 12 else ''}")
    if not missing and not beyond:
        print("  nothing to repair")
        return 0
    if not a.apply:
        print("\n  re-run with --apply to keep the usable experts and drop "
              "the rest")
        return 0

    idx = np.array(keep)
    # every tensor with one row per expert, subset the same way
    for f in ("routers.npz", "optim.npz"):
        p = os.path.join(w, f)
        with np.load(p) as z:
            out, touched = {}, 0
            for k in z.files:
                t = z[k]
                if t.ndim >= 1 and t.shape[0] == n:
                    out[k] = t[idx]
                    touched += 1
                else:
                    out[k] = t            # already subset, or not per-expert
        # write beside it and replace, so a failure cannot leave a half-written
        # archive where the original was. The temp name has to end in .npz or
        # savez appends one and the replace has nothing to move.
        tmp_p = p[:-4] + ".repair.npz"
        np.savez(tmp_p, **out)
        os.replace(tmp_p, p)
        print(f"  {f}: {touched} per-expert tensors subset to {len(keep)}"
              if touched else f"  {f}: already at {len(keep)}, left alone")

    # renumber the files, low to high, into a scratch name first so a partial
    # run cannot overwrite a file it has not moved yet
    ed = os.path.join(w, "experts")
    tmp = os.path.join(ed, "_repair")
    os.makedirs(tmp, exist_ok=True)
    for new, old in enumerate(keep):
        shutil.move(os.path.join(ed, f"e{old:05d}.npz"),
                    os.path.join(tmp, f"e{new:05d}.npz"))
    for f in os.listdir(ed):
        if f.endswith(".npz"):
            os.remove(os.path.join(ed, f))
    for f in os.listdir(tmp):
        shutil.move(os.path.join(tmp, f), os.path.join(ed, f))
    os.rmdir(tmp)
    print(f"  experts/: renumbered to 0..{len(keep) - 1}")

    byid = {int(e["id"]): e for e in man["experts"]}
    ents = []
    for new, old in enumerate(keep):
        e = dict(byid.get(old) or {"params": 0, "bytes": 0, "moments": True,
                                   "gate": 1.0})
        e["id"] = new
        e["file"] = f"e{new:05d}.npz"
        e["bytes"] = os.path.getsize(os.path.join(w, "experts", e["file"]))
        ents.append(e)
    man["experts"] = ents
    man["n_experts"] = len(keep)
    # Renumbering the files invalidates every stable id, so the identity map
    # has to be reset to match - otherwise each uid names a file that now
    # belongs to a different expert, which is worse than the drift this tool
    # exists to fix. arange is exactly what a directory written before ids
    # existed looks like, so it loads cleanly.
    tel = man.get("telemetry")
    if isinstance(tel, dict) and "uid" in tel:
        tel["uid"] = list(range(len(keep)))
        tel["next_uid"] = len(keep)
        print(f"  telemetry: stable ids RESET to 0..{len(keep) - 1} - the "
              f"files were renumbered, so the old ids no longer name them. "
              f"Anything recorded against an old id (expert_history.jsonl) "
              f"refers to the pool as it was.")
    man["cfg"]["pool_max"] = max(int(man["cfg"].get("pool_max") or 0), len(keep))
    for k in ("pool_ever", "pool_since"):
        v = man["cfg"].get(k)
        if v and len(v) >= n:
            man["cfg"][k] = [v[i] for i in keep]
    json.dump(man, open(os.path.join(w, "manifest.json"), "w"))
    print(f"  manifest: {len(keep)} experts, index rebuilt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
