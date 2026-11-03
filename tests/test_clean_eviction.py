#!/usr/bin/env python3
"""
An expert that nothing trained is not written back.

Reading text after text, and writing replies, moves experts on and off the
card. None of them changes while it is read or written, so each is still
exactly its copy in RAM or on disk, and parking it must copy nothing back. An expert the optimiser has stepped while it was on the card is
the opposite case: its copy is stale, and it must be written.

What this asserts, on a writable paged model with a RAM cache smaller than
the card, so that anything parked as dirty reaches the disk:

  reading text after text with no training marks nothing dirty and writes no
  expert file, though experts come and go

  after an optimiser step, an expert leaving the card is written back, and
  its copy then holds the trained weights

  once a flush has written the card, reading moves experts again without a
  single further write
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-clean-")
try:
    from minagi.create import create                          # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", ram_capacity=2)
    model.eval()

    def ids(t):
        return torch.tensor([[b for b in t.encode()][:cfg.block]])

    def read(text):
        with torch.no_grad():
            model(ids(text))

    def mtimes():
        d = pool.tiers.path
        return {f: os.path.getmtime(os.path.join(d, f))
                for f in os.listdir(d) if f.endswith(".npz")}

    TEXTS = ["1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1",
             "def merge(a, b):\n    return sorted(a + b)\n\nclass Node:\n",
             "The Roman Empire was the post-Republican period of ancient Rome",
             "add 4917 + 388 = 5305\nadd 72459 + 1 = 72460\nsub 33 - 3 = 30"]

    # --- reading, with nothing trained --------------------------------------
    files = mtimes()
    loads = pool.loads
    for _ in range(3):
        for t in TEXTS:
            read(t)
        with torch.no_grad():
            model.generate(ids(TEXTS[0]), 16)
    check("experts came and went while reading and writing",
          pool.loads - loads > pool.resident,
          f"{pool.loads - loads} loaded")
    check("nothing untrained was marked dirty", not pool.tiers.dirty,
          f"{sorted(pool.tiers.dirty)}")
    check("...and no expert file was written",
          pool.tiers.writebacks == 0 and mtimes() == files,
          f"{pool.tiers.writebacks} written")

    # --- one optimiser step, then the trained experts leave -----------------
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    pool.attach_optimiser(opt)
    model.train()
    x = ids(TEXTS[0])
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.eval()
    trained = {e: pool.w1.data[s].detach().clone()
               for s, e in enumerate(pool.slots) if e >= 0}
    for t in TEXTS[1:]:
        read(t)
    left = [e for e in trained if e not in pool.slots]
    held = {e: pool.tiers.fetch(pool._f(e))["w1"] for e in left}
    check("trained experts left the card", len(left) > 0, f"{left}")
    check("...and each was written back with its trained weights",
          all(torch.allclose(held[e], trained[e]) for e in left)
          and pool.tiers.writebacks > 0,
          f"{pool.tiers.writebacks} written")

    # --- after a flush, experts move without being written again ------------
    pool.flush()
    written = pool.tiers.writebacks
    loads = pool.loads
    for _ in range(2):
        for t in TEXTS:
            read(t)
    check("after a flush, experts move again", pool.loads > loads,
          f"{pool.loads - loads} loaded")
    check("...without a single further write",
          pool.tiers.writebacks == written and not pool.tiers.dirty,
          f"{pool.tiers.writebacks - written} written")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all clean-eviction tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
