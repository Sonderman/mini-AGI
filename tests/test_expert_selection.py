#!/usr/bin/env python3
"""
The selection rule, and what it promises.

The text chooses. Every forward's first pass adds its characters' requests to
the vote of its text - every character ranks the WHOLE pool with the router
and asks for its top_k, each request carrying the probability the router gave
it - and a forward routes among the `resident` experts its text has voted for
most, the text being everything read since position 0. A training window is a
text of its own, so its set is the most requested of its first pass; a reply
is voted for by its prompt and by every character of it so far. A character
whose request was not admitted takes its best admitted expert instead.

What this asserts:

  a text is a blank slate - the same text admits the same experts whatever
  was read before it, because the ranking never looks at the card

  the admitted set is exactly the most-requested experts of the first pass,
  recomputed here from the router itself

  no character routes to an expert its text did not admit, and nothing a
  forward used leaves the card before its backward

  a forward that continues a text through the attention cache begins its own
  admission, decided by its text's vote, and nothing read before the text
  counts; only a forward from position 0 moves the prune clock

  every character of a reply is a forward of its own, and routes among the
  experts its whole text so far has voted for, recomputed here from the
  router over the prompt and every character written

  reading the same text again loads nothing

  the language-model gradient reaches only the router rows of admitted
  experts - an expert off the card is not trained by a text that did not
  admit it
"""
import os
import shutil
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-sel-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.pool import PooledMLP, capture_routes         # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", read_only=True)
    model.eval()
    site = [m for m in model.modules() if isinstance(m, PooledMLP)][0]

    def ids(t):
        return torch.tensor([[b for b in t.encode()][:cfg.block]])

    def read(text):
        with torch.no_grad():
            model(ids(text))

    def admitted():
        return tuple(sorted(pool._admitted))

    BEFORE = ["1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1",
              "def merge(a, b):\n    return sorted(a + b)\n\nclass Node:\n",
              "The Roman Empire was the post-Republican period of ancient Rome",
              "add 4917 + 388 = 5305\nadd 72459 + 1 = 72460\nsub 33 - 3 = 30"]
    PROMPT = "add 2856 + 741 = 3597\nadd 12 + 30 = 42\nmul 7 * 6 = 42\n"

    # --- a text is a blank slate ----------------------------------------
    seen = set()
    for b in BEFORE:
        read(b)
        read(PROMPT)
        seen.add(admitted())
    check("one admitted set, whatever was read before it",
          len(seen) == 1, f"{len(seen)} distinct over {len(BEFORE)} histories")
    check("a text uses at most `resident` experts",
          all(len(s) <= pool.resident for s in seen),
          f"{len(next(iter(seen)))} of {pool.resident}")

    # --- the admitted set is the first pass's most requested ------------
    first = []
    h = site.register_forward_pre_hook(
        lambda m, a: first.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    read(PROMPT)
    h.remove()
    with torch.no_grad():
        q = F.softmax(F.linear(first[0] + site.depth_emb,
                               site.router.weight[:pool.router_rows()]), -1)
        top = torch.topk(q, site.top_k, dim=-1)
        mass = torch.zeros(pool.router_rows())
        mass.index_add_(0, top.indices.reshape(-1), top.values.reshape(-1))
    asked = int((mass > 0).sum())
    want = tuple(sorted(torch.topk(mass, min(pool.resident, asked)).indices.tolist()))
    check("the first pass asks for more experts than the card holds",
          asked > pool.resident, f"{asked} asked for, {pool.resident} slots")
    check("the admitted set is exactly the first pass's most requested",
          admitted() == want, f"{admitted()} vs {want}")

    # --- routing stays inside the admitted set ---------------------------
    with torch.no_grad(), capture_routes() as got:
        model(ids(PROMPT))
    used = set(int(e) for ids_, _ in got for e in ids_.reshape(-1).tolist())
    check("no character routes to an expert its text did not admit",
          used <= set(pool._admitted), f"{sorted(used)} vs {admitted()}")
    check("everything the forward admitted is still on the card",
          set(pool._admitted) <= set(pool.slots))

    # --- a forward that continues a text --------------------------------
    starts = []
    real_begin = pool.begin_forward
    pool.begin_forward = lambda *a, **k: (starts.append(1),
                                          real_begin(*a, **k))[1]

    def continued(history):
        read(history)
        caches = model.empty_caches()
        with torch.no_grad():
            model(ids(PROMPT), caches=caches, pos_offset=0)
            n = ids(PROMPT).shape[1]
            del starts[:]
            segs = pool.segments
            model(ids("x"), caches=caches, pos_offset=n)
        return admitted(), len(starts), pool.segments - segs

    outs = [continued(b) for b in BEFORE]
    check("a forward that continues a text starts its own admission",
          all(o[1] == 1 for o in outs), f"{[o[1] for o in outs]}")
    check("...which does not depend on what was read before the text",
          len({o[0] for o in outs}) == 1,
          f"{len({o[0] for o in outs})} distinct over {len(BEFORE)} histories")
    check("...and does not move the prune clock",
          all(o[2] == 0 for o in outs), f"{[o[2] for o in outs]}")
    segs = pool.segments
    read(BEFORE[0])
    check("a forward at position 0 starts a new text",
          pool.segments == segs + 1, f"texts begun {segs} -> {pool.segments}")

    del starts[:]
    with torch.no_grad():
        model.generate(ids(PROMPT), 12)
    check("every character of a reply is a forward of its own",
          len(starts) == 1 + 11,
          f"{len(starts)} forwards began for a 12-character reply")
    pool.begin_forward = real_begin

    # --- a reply is admitted by its text's vote --------------------------
    # the first pass of every forward since the text began, recomputed from
    # the router: a prompt, a continuation of ten characters, then one
    # character at a time
    def first_pass_mass(rows):
        with torch.no_grad():
            q = F.softmax(F.linear(rows + site.depth_emb,
                                   site.router.weight[:pool.router_rows()]), -1)
            top = torch.topk(q, site.top_k, dim=-1)
            m = torch.zeros(pool.router_rows())
            m.index_add_(0, top.indices.reshape(-1), top.values.reshape(-1))
        return m

    calls, firsts = [], []
    h = site.register_forward_pre_hook(
        lambda m, a: calls.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    pool.begin_forward = lambda *a, **k: (firsts.append(len(calls)),
                                          real_begin(*a, **k))[1]
    text = ids(PROMPT + BEFORE[1])
    n0 = ids(PROMPT).shape[1]
    caches = model.empty_caches()
    agree, pieces = [], [(0, n0), (n0, n0 + 10)] + [
        (t, t + 1) for t in range(n0 + 10, text.shape[1])]
    with torch.no_grad():
        for a, b in pieces:
            model(text[:, a:b], caches=caches, pos_offset=a)
            vote = sum(first_pass_mass(calls[i]) for i in firsts)
            k = min(pool.resident, int((vote > 0).sum()))
            want = tuple(sorted(torch.topk(vote, k).indices.tolist()))
            agree.append(admitted() == want)
    h.remove()
    pool.begin_forward = real_begin
    check("every forward of a text routes among what the whole text voted for",
          all(agree), f"{sum(agree)} of {len(agree)} forwards")

    # --- reading the same text again loads nothing -----------------------
    read(BEFORE[1])
    loads = pool.loads
    read(BEFORE[1])
    check("the same text twice in a row loads nothing the second time",
          pool.loads == loads, f"{pool.loads - loads} loaded")

    # --- gradient reaches only admitted experts' router rows -------------
    model.train()
    model.zero_grad(set_to_none=True)
    x = ids(PROMPT)
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    g = site.router.weight.grad
    rows_with_grad = set(int(i) for i in (g.abs().sum(-1) > 0).nonzero().flatten())
    check("only admitted experts' router rows receive gradient",
          rows_with_grad <= set(pool._admitted),
          f"{sorted(rows_with_grad)} vs {admitted()}")
    check("...and they do receive it", len(rows_with_grad) > 0,
          f"{len(rows_with_grad)} rows")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all expert-selection tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
