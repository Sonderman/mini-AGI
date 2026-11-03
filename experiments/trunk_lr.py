#!/usr/bin/env python3
"""
The trunk is where the forgetting lives. So how slowly should it learn?

Freezing the shared trunk removed 99.7% of forgetting but cost the ability to
pick up a genuinely new domain - chess notation never landed. Freezing is the
extreme end of a knob, not the only setting of it.

This sweeps the trunk's learning rate as a fraction of the pool's. At 1.0 both
learn equally (the failing baseline); at 0.0 the trunk is frozen. Somewhere
between is the point where the model keeps what it read and can still take on
something new.

The pool grows throughout, since growth was worth 12.5% on its own and the two
mechanisms address different halves of the problem.
"""
import argparse, json, sys, numpy as np, torch
from experiments.corpus import Corpus
from experiments.schedule import lr_at
from minagi.recur import RecurConfig, RecurCoder
from minagi.pool import PooledMLP, AutoGrow

DOM = ["data_char", "data_math_char", "data_chat_char", "data_chess_char"]
SHORT = {"data_char":"code","data_math_char":"math",
         "data_chat_char":"chat","data_chess_char":"chess"}

@torch.no_grad()
def val(m, cs, batches=6):
    m.eval(); out={}
    for n,c in cs.items():
        L=[]
        for _ in range(batches):
            x,y=c.batch_for("val", np.random.default_rng(7)); L.append(float(m(x,y)[1]))
        out[SHORT[n]]=float(np.mean(L))
    m.train(); return out

def split_params(m):
    """trunk = everything the pool does not own."""
    sites = [x for x in m.modules() if isinstance(x, PooledMLP)]
    pool_ids = {id(p) for e in m.pool.experts for p in e.parameters()}
    pool_ids |= {id(s.router.weight) for s in sites}
    pool_ids |= {id(s.depth_emb) for s in sites}
    pool_ids.add(id(m.pool.gate))
    trunk, pool = [], []
    for p in m.parameters():
        (pool if id(p) in pool_ids else trunk).append(p)
    return trunk, pool

def run(mult, cs, steps, batch, block, seed, base_lr, device, grow, log,
        mode="stream"):
    torch.manual_seed(seed)
    cfg = RecurConfig(vocab_size=265, n_head=4, d_model=128, block=block,
                      d_ff=256, n_layer=4, n_prelude=1, n_recur=2, n_coda=1,
                      max_steps=2, use_pool=True, pool_experts=32,
                      pool_d_ff=64, pool_top_k=4, pool_max=160)
    m = RecurCoder(cfg).to(device)
    trunk, pool = split_params(m)
    opt = torch.optim.AdamW(
        [{"params": trunk, "weight_decay": 0.1, "name": "trunk"},
         {"params": pool,  "weight_decay": 0.1, "name": "pool"}],
        lr=base_lr, betas=(0.9,0.95))
    grower = AutoGrow(grow_k=16, idle_frac_max=0.35,
                      max_experts=160) if grow else None
    rng = np.random.default_rng(seed)
    names=list(cs); per=max(1, steps//len(names))
    for step in range(steps):
        lr = lr_at(step, steps, base_lr, max(4, steps//20))
        for g in opt.param_groups:
            g["lr"] = lr * (mult if g.get("name")=="trunk" else 1.0)
        opt.zero_grad(set_to_none=True)
        if mode == "stream":
            c = cs[names[min(step//per, len(names)-1)]]; c.batch=batch
            x,y = c.batch_for("train", rng)
        else:
            # every batch holds all four corpora - batch >= n_sources here, so
            # the integer split is safe
            xs, ys, used = [], [], 0
            share = [0.25, 0.15, 0.25, 0.35]
            for j, n in enumerate(names):
                k = batch - used if j == len(names)-1 else max(1, int(round(batch*share[j])))
                k = max(0, min(k, batch - used))
                if not k: continue
                cs[n].batch = k
                a, b = cs[n].batch_for("train", rng)
                xs.append(a); ys.append(b); used += k
            x, y = torch.cat(xs), torch.cat(ys)
        _, loss = m(x,y)
        loss = loss + cfg.pool_aux * m.pool_aux()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if grower is not None and (step+1) % max(1, steps//8) == 0:
            v = val(m, cs)
            rec = grower.step(np.mean(list(v.values())), m.pool, step)
            if rec["grew"]:
                known = {id(p) for g in opt.param_groups for p in g["params"]}
                fresh = [p for p in m.parameters() if id(p) not in known]
                if fresh:
                    opt.add_param_group({"params": fresh, "weight_decay": 0.1,
                                         "name": "pool"})
    v = val(m, cs)
    return v, m.pool.n_experts()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=200); ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--block",type=int,default=256); ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--mults",type=float,nargs="+",default=[1.0,0.3,0.1,0.03,0.0])
    ap.add_argument("--mode",default="stream",choices=["stream","mixed"])
    a=ap.parse_args()
    dev=torch.device(a.device); torch.set_num_threads(6)
    cs={d:Corpus(d,dev,a.block,a.batch) for d in DOM}
    doms=["code","math","chat","chess"]
    print(f"{a.mode} batches, pool growing, trunk learning rate swept\n")
    print(f"{'trunk lr':>9}" + "".join(f"{d:>9}" for d in doms)
          + f"{'mean':>9}{'spread':>9}{'pool':>7}")
    out={}
    for mult in a.mults:
        v, n = run(mult, cs, a.steps, a.batch, a.block, 0, a.lr, dev, True,
                   print, a.mode)
        vals=[v[d] for d in doms]
        out[str(mult)]={"val":v,"pool":n,
                        "mean":float(np.mean(vals)),
                        "spread":float(max(vals)-min(vals))}
        print(f"{mult:>9.2f}" + "".join(f"{x:>9.3f}" for x in vals)
              + f"{np.mean(vals):>9.3f}{max(vals)-min(vals):>9.3f}{n:>7}", flush=True)
    best = min(out.items(), key=lambda kv: kv[1]["mean"])
    print(f"\nbest mean loss at trunk lr x{best[0]}: {best[1]['mean']:.3f} "
          f"(spread {best[1]['spread']:.3f})")
    print(f"mixed-batch reference from the earlier run: mean 2.426, spread 1.410")
    json.dump(out, open(f"runs/trunk_lr_{a.mode}.json","w"), indent=1)

if __name__=="__main__": sys.exit(main())
