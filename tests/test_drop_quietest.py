#!/usr/bin/env python3
"""
Dropping the quietest experts (train.py read --drop-quietest).

What these tests hold it to:
  it removes the n smallest gates not on the card, with their files and
  every row they had, and leaves an expert inside its trial alone however
  quiet; it goes ahead only when the number is typed back, and that is asked
  before anything loads
"""
import contextlib
import inspect
import io
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


tmp = tempfile.mkdtemp(prefix="minagi-drop-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.pool import PooledMLP                         # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "a")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pa, _ = T.build_paged(wdir, "cpu", ram_capacity=2)
    pa.balance = 0.0
    torch.manual_seed(0)
    for site in (m for m in model.modules() if isinstance(m, PooledMLP)):
        site.router.weight.data.normal_(0.0, 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pa.attach_optimiser(opt)
    # a training step first, so the card holds the experts the text admitted
    text = "def merge(a, b):\n    return sorted(a + b)\n\nclass Node:\n    pass\n"
    x = torch.tensor([[c for c in text.encode()][:cfg.block]])
    model.train()
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    opt.step()

    # ---------------------------------------------- 1. dropping the quietest
    n = pa.n_experts()
    g = pa.gate.data
    g[:] = torch.linspace(0.5, 0.01, n)      # the last experts are the quietest
    on = set(i for i in pa.slots if i >= 0)
    quiet = [i for i in sorted(range(n), key=lambda i: float(g[i].abs()))
             if i not in on][:2]
    quiet_uids = [int(pa.uid[i]) for i in quiet]
    files = [pa.tiers._file(u) for u in quiet_uids]
    out = pa.drop_quietest(2)
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    check("the quietest experts not on the card are dropped, with their files",
          on and sorted(u for u, _ in out) == sorted(quiet_uids)
          and pa.n_experts() == n - 2 and not any(os.path.exists(f) for f in files),
          f"dropped {out}, {len(on)} on the card")
    check("...and every row they had goes with them",
          all(s.router.weight.shape[0] == n - 2 for s in sites)
          and pa.gate.numel() == n - 2)

    # a newborn has the smallest gate of all and is still left alone
    n = pa.n_experts()
    pa.born[:] = 0.0
    on = set(i for i in pa.slots if i >= 0)
    baby = next(i for i in range(n) if i not in on)
    pa.born[baby] = 9_000.0
    pa.gate.data[baby] = 1e-4
    baby_uid = int(pa.uid[baby])
    out = pa.drop_quietest(1, step=10_000, survival=5_000)
    check("an expert inside its trial is never dropped, however quiet",
          baby_uid in [int(u) for u in pa.uid] and baby_uid not in [u for u, _ in out],
          f"dropped {out}, the newborn kept")

    # ------------------------------------------ 2. asked before dropping
    def answer(text):
        return lambda prompt: text

    def nobody(prompt):
        raise EOFError

    told = io.StringIO()
    with contextlib.redirect_stdout(told):
        yes = [T.confirm_drop(50, wdir, ask=answer(a)) for a in ("50", " 50\n")]
        no = [T.confirm_drop(50, wdir, ask=answer(a)) for a in ("y", "yes", "", "5")]
        no.append(T.confirm_drop(50, wdir, ask=nobody))
    check("dropping goes ahead only when the number is typed back",
          all(yes) and not any(no), f"{yes} {no}")
    check("...the question names how many experts there are, and a no says so",
          "of the 12 experts" in told.getvalue()
          and told.getvalue().count("kept: nothing will be dropped") == 5)
    src = inspect.getsource(T.cmd_read)
    check("...and it is asked before anything loads",
          0 <= src.index("confirm_drop(") < src.index("build_paged("))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all drop-quietest tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
