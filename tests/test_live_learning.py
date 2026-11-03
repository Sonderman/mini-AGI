"""
Learning from a conversation as it happens.

The rules here are the ones the measurements forced, and each test is one of
them. Held-out on data/val, error bar +/-0.06:

    54 steps, its own output only               +0.5855
    54 steps, the whole window every chunk      +0.6507
     6 steps, one per finished exchange         +0.1018
     5 steps, real documents of the same size   +0.0197   (control)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.live import LiveLearner, exchange_text  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    ok = ok and cond


class Cfg:
    block = 256
    vocab_size = 265


class Tiny(nn.Module):
    """Small enough to train in a test, real enough to have a loss."""

    def __init__(self):
        super().__init__()
        self.cfg = Cfg()
        self.emb = nn.Embedding(Cfg.vocab_size, 16)
        self.head = nn.Linear(16, Cfg.vocab_size)

    def forward(self, x, y=None, **kw):
        h = self.emb(x)
        logits = self.head(h)
        if y is None:
            return logits, None
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, Cfg.vocab_size), y.reshape(-1))
        return logits, loss


class Tok:
    class R:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, s):
        return Tok.R([min(ord(c), 264) for c in s])


tok = Tok()

# --- the exchange is marked up like the chat corpus ----------------------
t = exchange_text("what are you?", "a model.")
check("an exchange carries both halves",
      "<user>" in t and "what are you?" in t
      and "<bot>" in t and "a model." in t)
check("an empty exchange is nothing to read", exchange_text("", "") == "")

# --- the stream steps every `chunk` characters, not every message --------
m = Tiny()
before = m.head.weight.detach().clone()
lr = LiveLearner(m, lr=1e-2, save_every=0, chunk=32, context=128)
recs = lr.feed("x" * 100, tok)
check("a long message takes several steps", len(recs) == 3, f"{len(recs)} steps")
check("the weights moved", not torch.equal(before, m.head.weight))
check("and the remainder is held, not dropped", lr.pending == 4, f"{lr.pending}")

# --- THE PROPERTY THAT MAKES IT ONE STREAM ------------------------------
# A message boundary is not a boundary the model sees. Two half-chunks in
# separate calls must step exactly as one whole chunk would - otherwise the
# turn structure is silently part of the learning rule.
m2 = Tiny()
a = LiveLearner(m2, lr=1e-2, save_every=0, chunk=32, context=128)
check("a message shorter than a chunk takes no step", a.feed("y" * 20, tok) == [])
check("...and the next one completes it", len(a.feed("z" * 20, tok)) == 1,
      "20 + 20 characters is one step, wherever the turn boundary fell")
b = LiveLearner(Tiny(), lr=1e-2, save_every=0, chunk=32, context=128)
one = b.feed("y" * 20 + "z" * 20, tok)
check("...exactly as the same text in a single message would",
      len(one) == 1 and one[0]["chars"] == a.log[-1]["chars"],
      f"split {a.log[-1]['chars']} chars, whole {one[0]['chars']}")

# --- the window is bounded, and it is what the step sees ----------------
c = LiveLearner(Tiny(), lr=1e-2, save_every=0, chunk=32, context=128)
for _ in range(20):
    c.feed("q" * 32, tok)
check("the buffer never grows past the context window",
      len(c.buf) <= c.context, f"{len(c.buf)} of {c.context}")
check("and a step reads the whole window",
      c.log[-1]["chars"] == c.context, f"{c.log[-1]['chars']}")

# --- the cursor only moves forward --------------------------------------
d = LiveLearner(Tiny(), lr=1e-2, save_every=0, chunk=32, context=128)
d.feed("m" * 64, tok)
n1 = d.steps
d.feed("m" * 64, tok)
check("re-feeding does not re-read: every step is new text",
      d.steps == n1 + 2, f"{d.steps} steps for 128 characters at chunk 32")

# --- gradients must not be left lying around ----------------------------
check("no gradient survives the step",
      all(p.grad is None or float(p.grad.abs().sum()) == 0
          for p in m.parameters()))

# --- eval mode is restored, or generation silently changes behaviour ----
m.eval()
lr.feed("w" * 64, tok)
check("eval mode is restored after learning", not m.training)

# --- saving is on a counter, so a crash costs at most that many ---------
m3 = Tiny()
lr2 = LiveLearner(m3, lr=1e-2, save_every=3, weights_dir=None,
                  chunk=32, context=128)
lr2.feed("a" * 64, tok)
check("not due to save before the counter", not lr2.due_to_save())
lr2.feed("b" * 32, tok)
check("due to save on the counter",
      lr2.unsaved == 3 and not lr2.due_to_save(),
      "no weights_dir, so nothing to save to")
lr2.weights_dir = "/tmp"
check("due to save once there is somewhere to save", lr2.due_to_save())

# --- the log is what makes it auditable ---------------------------------
check("every step is logged", len(lr2.log) == 3
      and all("loss" in r and "chars" in r for r in lr2.log))

# --- learn() stays as a thin wrapper for callers that want one record ---
e = LiveLearner(Tiny(), lr=1e-2, save_every=0, chunk=32, context=128)
check("learn() returns the last step, or None if none was taken",
      e.learn("s" * 10, tok) is None and e.learn("t" * 40, tok) is not None)

# --- Adam's moments carry on from the weights directory -------------------
# The learner used to build a fresh optimiser: its first step started every
# tensor from zero moments, and its first save wrote that over the trainer's
# optim.npz. It resumes from them now, as the trainer does, and every expert
# on the card is stepped with its own.
import json                                               # noqa: E402
import shutil                                             # noqa: E402
import tempfile                                           # noqa: E402
from dataclasses import asdict                            # noqa: E402

from minagi import store as weights_store                 # noqa: E402
from minagi.create import create                          # noqa: E402
from minagi.precision import unpack_bf16                  # noqa: E402
from minagi.tokenizer import ByteTokenizer                # noqa: E402
import train as T                                         # noqa: E402

tmp = tempfile.mkdtemp(prefix="minagi-live-")
try:
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=12, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=128, max_steps=3,
           top_k=2, depth=1)
    pm, pcfg, ppool, _ = T.build_paged(wdir, "cpu", ram_capacity=4)
    popt = torch.optim.AdamW(pm.parameters(), lr=1e-3)
    ppool.attach_optimiser(popt)
    pm.train()
    text = b"def merge(a, b):\n    return sorted(a + b)\n" * 2
    for _ in range(3):                  # the trainer's history, then a save
        x = torch.tensor([list(text)])
        _, loss = pm(x[:, :-1], x[:, 1:])
        loss.backward()
        popt.step()
        popt.zero_grad(set_to_none=True)
    weights_store.save(pm, wdir, step=3, opt=popt, cfg=asdict(pcfg))
    trunk_name, trunk_p = next((n, p) for n, p in pm.named_parameters()
                               if not n.startswith("pool."))
    saved = popt.state[trunk_p]["exp_avg"].clone()

    m4, _, pool4, man4 = T.build_paged(wdir, "cpu", ram_capacity=4)
    ptok = ByteTokenizer()
    with torch.no_grad():               # serve.py's prime, before the learner
        m4(torch.tensor([list(b"class Node:\n    pass\n")]))
    live = LiveLearner(m4, lr=1e-3, save_every=0, chunk=32, context=128,
                       weights_dir=wdir, manifest=man4)
    p4 = dict(m4.named_parameters())[trunk_name]
    st = live.opt.state.get(p4, {})
    check("the learner starts from the trunk's saved moments, not zeros",
          "exp_avg" in st and torch.allclose(st["exp_avg"].float(),
                                             saved.float(), atol=1e-2)
          and float(saved.abs().sum()) > 0)
    check("...and from the saved step count",
          "step" in st and float(st["step"]) == 3.0,
          f"step {float(st['step']) if 'step' in st else None}")

    seen = []

    def own_at_step(*_):
        for s, e in enumerate(pool4.slots):
            if e < 0:
                continue
            z = np.load(pool4.tiers._file(pool4._f(e)))
            want = (unpack_bf16(z["w1_m"]).float() if "w1_m" in z.files
                    else torch.zeros_like(pool4.w1.data[s]))
            seen.append(torch.equal(live.opt.state[pool4.w1]["exp_avg"][s],
                                    want))
    live.opt.register_step_pre_hook(own_at_step)
    recs = live.feed("x = merge([1, 3], [2])\n" * 2, ptok)
    check("its first step steps every expert on the card with its own moments",
          len(recs) >= 1 and seen and all(seen),
          f"{sum(seen)} of {len(seen)} slots")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --- THE RATE IS THE RUN'S ------------------------------------------------
# A conversation, or a test read while learning, carries the training stream
# on, so it learns at the rate training would step at next: the base rate
# times the controller's factor, restored with the weights. A flat 3e-4 was
# ~35x the rate training had come down to.
f = LiveLearner(Tiny(), lr=1e-2, save_every=0, chunk=32, context=128,
                manifest={"plasticity": {"scale": 0.1, "step": 5000}})
f.feed("r" * 32, tok)
check("the stream learns at the base rate times the controller's factor",
      abs(f.log[-1]["lr"] - 1e-3) < 1e-12 and f.log[-1]["scale"] == 0.1
      and abs(f.opt.param_groups[0]["lr"] - 1e-2 * 0.2 * 0.1) < 1e-12,
      f"pool {f.opt.param_groups[-1]['lr']:.2e}, trunk "
      f"{f.opt.param_groups[0]['lr']:.2e}")
check("...and its steps count on the controller's clock",
      f.plast.step == 5001)
g = LiveLearner(Tiny(), lr=1e-2, save_every=0, chunk=32, context=128)
g.feed("r" * 32, tok)
check("...a model with no controller yet warms up, as training starts one",
      abs(g.log[-1]["scale"] - 1 / 100) < 1e-12)

tmp = tempfile.mkdtemp(prefix="minagi-live-rate-")
try:
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=12, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=128, max_steps=3,
           top_k=2, depth=1)
    pm, pcfg, ppool, _ = T.build_paged(wdir, "cpu", ram_capacity=4)
    popt = torch.optim.AdamW(pm.parameters(), lr=1e-3)
    ppool.attach_optimiser(popt)
    # what a training run leaves in the manifest besides the weights
    ran = {"read_chars": 1000, "read_nats": 900.0, "context_now": 128,
           "plasticity": {"scale": 0.25, "step": 500},
           "lanes": {"mode": "auto", "shares": {"a": 1.0}}, "lr": 2e-3}
    weights_store.save(pm, wdir, step=40, opt=popt, cfg=asdict(pcfg),
                       extra=ran)
    m5, _, _, man5 = T.build_paged(wdir, "cpu", ram_capacity=4)
    conf = {"training": {"lr": 3e-4, "trunk_lr_mult": 0.1, "chunk": 32}}
    # checks=False: the controller's checks are tests/test_governor.py's
    h = LiveLearner.for_run(m5, weights_dir=wdir, manifest=man5,
                            save_every=0, config=conf, checks=False)
    check("set up as training is: the run's own base rate, from the manifest",
          h.opt.param_groups[-1]["base_lr"] == 2e-3 and h.chunk == 32
          and LiveLearner.for_run(m5, manifest=man5, lr=5e-4, config=conf,
                                  checks=False).opt.param_groups[-1][
                                      "base_lr"] == 5e-4)
    tr, pp = T._split_trunk_pool(m5)
    check("...each parameter in the group training puts it in: router rows "
          "and depth embedding with the pool, not the trunk",
          {id(q) for q in h.opt.param_groups[0]["params"]} == {id(q) for q in tr}
          and {id(q) for q in h.opt.param_groups[-1]["params"]}
          == {id(q) for q in pp}
          and any("router" in nm for nm, q in m5.named_parameters()
                  if id(q) in {id(x) for x in h.opt.param_groups[-1]["params"]}))
    h.feed("def f(x):\n    return x\n" * 3, ByteTokenizer())
    check("...stepping at the run's base rate times the controller's factor",
          abs(h.log[-1]["lr"] - 2e-3 * 0.25) < 1e-12, f"{len(h.log)} steps")
    h.save()
    with open(os.path.join(wdir, "manifest.json")) as fh:
        back = json.load(fh)
    k = len(h.log)
    check("a save keeps what the run left - the controller, the reading "
          "shares, the base rate - and carries the stream on",
          back.get("lanes") == ran["lanes"] and back.get("lr") == 2e-3
          and back.get("context_now") == 128
          and back["plasticity"]["scale"] == 0.25
          and back["plasticity"]["step"] == 500 + k
          and back["step"] == 40 + k and back["read_chars"] == 1000 + 32 * k
          and back["read_nats"] > 900.0,
          f"{sorted(back)}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# train.py writes the base rate the controller scales, so chat can find it
import inspect                                            # noqa: E402
import re                                                 # noqa: E402
check("the training run records its base rate in the manifest, at both saves",
      len(re.findall(r'"lanes": sched\.state\(\),(?:\s*#[^\n]*)*\s*'
                     r'"lr": args\.lr,', inspect.getsource(T.cmd_read))) == 2)

print("\n  " + ("all live-learning tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
