#!/usr/bin/env python3
"""
Every expert is stepped with its own moments.

Adam's two moments belong to the expert, not to the slot of the card it sits
in. An expert comes to the card with its weights only; its moments come just
before an optimiser step, for the experts on the card then. What this asserts,
on a writable paged model whose RAM cache is smaller than the card, so experts
go to disk and come back:

  at every optimiser step, every expert on the card holds its own moments -
  the ones its last step left it, or the ones in its file if nothing here has
  stepped it yet - whatever happened on the card in between: replies written
  one character at a time, held-out forwards, experts parked on disk and read
  back

  writing a reply moves no moments: the optimiser's state is untouched while
  experts come and go

  THE RESTART. The trainer's held-out check puts experts on the card before
  the optimiser exists, and the restore that follows writes slot-indexed
  moments from optim.npz, which belong to whichever experts sat in the slots
  when it was saved. The first step after it still steps every expert on the
  card with its own moments.

  the check is not blind: a slot handed another expert's moments is caught

The record of what each expert's moments should be is kept here, from the
optimiser's state after every step and from the files, not read back from the
pool that is being tested.
"""
import os
import shutil
import sys
import tempfile
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


NAMES = ("w1", "w3", "w2")


class Ledger:
    """
    What every expert's moments should be, and a check of every slot against
    it before every step. Registered after the pool's own hooks, so it sees
    the slots exactly as the step will.
    """

    def __init__(self, pool, opt, start):
        self.pool, self.opt = pool, opt
        self.truth = dict(start)             # file id -> {w1_m: ..., ...}
        self.checked = self.wrong = 0
        self.last_wrong = []
        opt.register_step_pre_hook(self.before)
        opt.register_step_post_hook(self.after)

    def _slot(self, s):
        out = {}
        for p, nm in zip((self.pool.w1, self.pool.w3, self.pool.w2), NAMES):
            o = self.opt.state.get(p)
            if not (o and "exp_avg" in o):
                return None
            out[nm + "_m"] = o["exp_avg"][s].detach().clone()
            out[nm + "_v"] = o["exp_avg_sq"][s].detach().clone()
        return out

    def before(self, *_):
        self.last_wrong = []
        for s, e in enumerate(self.pool.slots):
            if e < 0:
                continue
            got = self._slot(s)
            if got is None:                  # the first step creates them
                return
            want = self.truth[self.pool._f(e)]
            self.checked += 1
            # parked on disk the moments are bf16, so a round trip rounds them
            if not all(torch.allclose(got[k], want[k], rtol=1e-2, atol=1e-12)
                       for k in got):
                self.wrong += 1
                self.last_wrong.append(e)

    def after(self, *_):
        for s, e in enumerate(self.pool.slots):
            if e >= 0:
                self.truth[self.pool._f(e)] = self._slot(s)


def from_files(pool):
    """Every expert's moments as its file holds them - zeros if it has none."""
    out = {}
    for e in range(pool.n_experts()):
        f = pool._f(e)
        z = np.load(pool.tiers._file(f))
        ent = {}
        for nm in NAMES:
            shape = z[nm].shape
            for k in (nm + "_m", nm + "_v"):
                ent[k] = (unpack_bf16(z[k]).float() if k in z.files
                          else torch.zeros(shape))
        out[f] = ent
    return out


def moment_state(pool, opt):
    return [opt.state[p][k].clone() for p in (pool.w1, pool.w3, pool.w2)
            for k in ("exp_avg", "exp_avg_sq")]


tmp = tempfile.mkdtemp(prefix="minagi-moments-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi import store as weights_store                 # noqa: E402
    from minagi.pool import PooledMLP                         # noqa: E402
    from minagi.precision import unpack_bf16                  # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", ram_capacity=2)
    pool.balance = 0.0
    # a router with opinions, as a trained one has, so that each character
    # asks for experts of its own and a reply keeps changing the card
    torch.manual_seed(0)
    for site in (m for m in model.modules() if isinstance(m, PooledMLP)):
        site.router.weight.data.normal_(0.0, 1.0)

    def ids(t):
        return torch.tensor([[b for b in t.encode()][:cfg.block]])

    TEXTS = ["1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1",
             "def merge(a, b):\n    return sorted(a + b)\n\nclass Node:\n",
             "The Roman Empire was the post-Republican period of ancient Rome",
             "add 4917 + 388 = 5305\nadd 72459 + 1 = 72460\nsub 33 - 3 = 30",
             "Once upon a time, there was a little boy named Tom. One day he"]

    def train_step(m, o, text):
        m.train()
        x = ids(text)
        _, loss = m(x[:, :-1], x[:, 1:])
        loss.backward()
        o.step()
        o.zero_grad(set_to_none=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pool.attach_optimiser(opt)
    ledger = Ledger(pool, opt, from_files(pool))

    # which experts went to disk, and which of those were read back from it
    written, reread = set(), set()
    to_disk, from_disk = pool.tiers._to_disk, pool.tiers._from_disk

    def spy_to(i, ent):
        written.add(i)
        return to_disk(i, ent)

    def spy_from(i):
        if i in written:
            reread.add(i)
        return from_disk(i)
    pool.tiers._to_disk, pool.tiers._from_disk = spy_to, spy_from

    # --- training, writing and held-out, interleaved ------------------------
    untouched, wrote = True, 0
    for r in range(6):
        for j in range(3):
            train_step(model, opt, TEXTS[(r + j) % len(TEXTS)])
        model.eval()
        before = moment_state(pool, opt)
        with torch.no_grad():
            model(ids(TEXTS[(r + 3) % len(TEXTS)]))            # held-out
            for t in TEXTS:                                    # replies
                loads = pool.loads
                model.generate(ids(t[:12]), 24, temperature=1.0)
                wrote += pool.loads - loads
        untouched &= all(torch.equal(a, b)
                         for a, b in zip(before, moment_state(pool, opt)))
    check("experts came and went while replies were written",
          wrote > 10 * pool.resident, f"{wrote} loaded")
    check("...and not one moment moved while they did", untouched)
    check("experts were parked on disk and read back from it",
          len(reread) > 0, f"{len(written)} written, {len(reread)} read back")
    check("at every step, every expert on the card held its own moments",
          ledger.checked > 0 and ledger.wrong == 0,
          f"{ledger.checked} checked, {ledger.wrong} wrong")

    # --- the check is not blind ---------------------------------------------
    # right after a step every expert on the card holds its own moments; hand
    # one slot another expert's, and the ledger has to say so
    train_step(model, opt, TEXTS[0])
    occupied = [s for s, e in enumerate(pool.slots) if e >= 0]
    a, b = next((a, b) for a in occupied for b in occupied if a != b
                and not torch.equal(opt.state[pool.w1]["exp_avg"][a],
                                    opt.state[pool.w1]["exp_avg"][b]))
    keep = {(p, k): opt.state[p][k][a].clone()
            for p in (pool.w1, pool.w3, pool.w2)
            for k in ("exp_avg", "exp_avg_sq")}
    for p, k in keep:
        opt.state[p][k][a].copy_(opt.state[p][k][b])
    ledger.before()
    caught = pool.slots[a] in ledger.last_wrong
    for (p, k), v in keep.items():
        opt.state[p][k][a].copy_(v)
    ledger.before()
    check("a slot holding another expert's moments is caught",
          caught and not ledger.last_wrong,
          f"expert {pool.slots[a]} given expert {pool.slots[b]}'s")

    # --- the restart ----------------------------------------------------------
    weights_store.save(model, wdir, step=1, opt=opt, cfg=asdict(cfg))
    saved_slots = list(pool.slots)

    model2, cfg2, pool2, _ = T.build_paged(wdir, "cpu", ram_capacity=2)
    pool2.balance = 0.0
    own = from_files(pool2)
    # what the trainer does: a held-out check before the optimiser exists...
    x = ids(TEXTS[2])
    model2.eval()
    with torch.no_grad():
        model2(x[:, :-1])
    # ...then the optimiser, and the restore of what the checkpoint saved
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    pool2.attach_optimiser(opt2)
    weights_store._load_optim(opt2, model2, wdir)
    foreign = [s for s, e in enumerate(pool2.slots)
               if e >= 0 and not all(
                   torch.allclose(opt2.state[p][k][s],
                                  own[pool2._f(e)][nm + suf],
                                  rtol=1e-2, atol=1e-12)
                   for p, nm in zip((pool2.w1, pool2.w3, pool2.w2), NAMES)
                   for k, suf in (("exp_avg", "_m"), ("exp_avg_sq", "_v")))]
    check("after the restore, slots hold moments that are not their "
          "expert's", len(foreign) > 0,
          f"{len(foreign)} of {sum(e >= 0 for e in pool2.slots)} slots; "
          f"saved with {saved_slots}, now holding {pool2.slots}")
    # the same text trains first, so every expert the check put on the card
    # stays there for the step - the ones that were never paged in again
    held = list(pool2.slots)
    ledger2 = Ledger(pool2, opt2, own)
    train_step(model2, opt2, TEXTS[2])
    check("...and training kept them on the card", pool2.slots == held,
          f"{held} -> {pool2.slots}")
    check("the first step after a restart steps every expert with its own",
          ledger2.checked > 0 and ledger2.wrong == 0,
          f"{ledger2.checked} checked, {ledger2.wrong} wrong")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all own-moment tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
