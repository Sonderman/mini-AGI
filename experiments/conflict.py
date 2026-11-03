#!/usr/bin/env python3
"""
Does splitting on gradient conflict stop the forgetting that plateau-growth did not?

Same stream, same seed, same steps. Four conditions:
  dense        no pool at all
  plateau      pool that grows when the loss flattens
  conflict     pool that SPLITS an expert when its gradient reverses
  conflict+lr  the same, with a lower learning rate
"""
import argparse, json, sys, numpy as np, torch
from experiments.corpus import Corpus
from experiments.schedule import lr_at
from minagi.recur import RecurConfig, RecurCoder
from minagi.pool import PooledMLP


class ConflictGrow:
    """
    Grow where an expert is being torn, not where the loss happens to be flat.

    A loss plateau is a global signal: it says the model is stuck but not which
    part is short of capacity. Growing on it alone adds experts the router has
    no reason to send anything to, so the new data keeps landing on the same
    experts and keeps overwriting them.

    The local signal is conflict. For each expert, keep a running average of the
    gradient direction it has been receiving. When a new batch pushes it the
    OTHER way - negative cosine against that average - the incoming data is
    trying to undo what the expert already knows. An expert that is both busy
    and being pulled in two directions is the exact place capacity is missing.

    On detection the expert is SPLIT, not merely supplemented: the new expert
    starts as a copy, and the router is nudged so the traffic that caused the
    conflict prefers the copy. The original keeps serving what it already
    served, which is the part that stops the forgetting.
    """

    def __init__(self, pool, conflict_thresh=-0.05, min_use=0.005,
                 patience=2, cooldown=3, max_split=8, ema=0.9):
        self.pool = pool
        self.conflict_thresh = conflict_thresh
        self.min_use = min_use
        self.patience, self.cooldown = patience, cooldown
        self.max_split, self.ema = max_split, ema
        self.dir = {}          # expert -> running gradient direction
        self.strikes = {}      # expert -> consecutive conflicting batches
        self.cool = 0
        self.events = []

    @torch.no_grad()
    def observe(self):
        """Call after backward(), before the optimiser step."""
        p = self.pool
        n = p.n_experts()
        use = p.use / p.use.sum().clamp_min(1)
        conflicted = []
        for i in range(n):
            g = p.experts[i].w2.weight.grad
            if g is None:
                continue
            v = g.reshape(-1)
            nv = v.norm()
            if nv < 1e-12:
                continue
            v = v / nv
            prev = self.dir.get(i)
            if prev is not None:
                cos = float(torch.dot(prev, v))
                if cos < self.conflict_thresh and float(use[i]) > self.min_use:
                    self.strikes[i] = self.strikes.get(i, 0) + 1
                    if self.strikes[i] >= self.patience:
                        conflicted.append((i, cos, float(use[i])))
                else:
                    self.strikes[i] = 0
                self.dir[i] = (self.ema * prev + (1 - self.ema) * v)
                self.dir[i] = self.dir[i] / self.dir[i].norm().clamp_min(1e-9)
            else:
                self.dir[i] = v.clone()
        return conflicted

    @torch.no_grad()
    def maybe_split(self, sites, hidden_mean, step):
        """Split the most conflicted experts and steer the new traffic to the copies."""
        if self.cool > 0:
            self.cool -= 1
            return []
        conflicted = self.observe()
        if not conflicted:
            return []
        conflicted.sort(key=lambda t: t[1])          # most negative first
        chosen = conflicted[:self.max_split]
        p = self.pool
        made = []
        for src, cos, u in chosen:
            before = p.n_experts()
            if before >= p.max_experts:
                break
            p.add_experts(1, seed_from=src)
            new = p.n_experts() - 1
            # the copy is a working expert immediately, not a zero-gated blank:
            # it has to take the conflicting traffic NOW or the split does
            # nothing to stop the overwrite
            p.gate.data[new] = float(p.gate.data[src])
            for site in sites:
                w = site.router.weight.data
                w[new] = w[src].clone()
                if hidden_mean is not None:
                    # tilt the copy toward whatever is currently being read, so
                    # the new domain prefers it and the original is left alone
                    hm = hidden_mean / hidden_mean.norm().clamp_min(1e-9)
                    w[new] += 0.5 * hm.to(w.dtype)
                    w[src] -= 0.1 * hm.to(w.dtype)
            self.dir.pop(src, None)
            self.strikes[src] = 0
            made.append({"step": step, "from": src, "to": new,
                         "cos": round(cos, 4), "use": round(u, 4)})
        if made:
            self.cool = self.cooldown
            self.events.extend(made)
        return made


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

def run(kind, cs, steps, batch, block, seed, lr0, device, log):
    torch.manual_seed(seed)
    pool = kind != "dense"
    cfg = RecurConfig(vocab_size=265, n_head=4, d_model=128, block=block,
                      d_ff=256, n_layer=4, n_prelude=1, n_recur=2, n_coda=1,
                      max_steps=2, use_pool=pool, pool_experts=16,
                      pool_d_ff=64, pool_top_k=4, pool_max=160)
    m = RecurCoder(cfg).to(device)
    opt = torch.optim.AdamW(
        [{"params":[p for p in m.parameters() if p.dim()>=2],"weight_decay":0.1},
         {"params":[p for p in m.parameters() if p.dim()<2],"weight_decay":0.0}],
        lr=lr0, betas=(0.9,0.95))
    grower = None
    if kind == "plateau":
        from minagi.pool import AutoGrow
        grower = AutoGrow(grow_k=16,
                          idle_frac_max=0.35, max_experts=160)
    splitter = None
    if kind.startswith("conflict"):
        splitter = ConflictGrow(m.pool, conflict_thresh=-0.02, min_use=0.002,
                                max_split=4)
    sites = [x for x in m.modules() if isinstance(x, PooledMLP)]
    rng = np.random.default_rng(seed)
    names = list(cs); per = max(1, steps//len(names))
    hidden = {}
    if sites:
        def cap(mod, inp, out): hidden["h"] = inp[0].reshape(-1, inp[0].shape[-1]).mean(0).detach()
        sites[0].register_forward_hook(cap)
    hist=[]
    for step in range(steps):
        lr = lr_at(step, steps, lr0, max(4, steps//20))
        for g in opt.param_groups: g["lr"]=lr
        opt.zero_grad(set_to_none=True)
        c = cs[names[min(step//per, len(names)-1)]]; c.batch = batch
        x,y = c.batch_for("train", rng)
        _, loss = m(x,y)
        if pool: loss = loss + cfg.pool_aux * m.pool_aux()
        loss.backward()
        if splitter is not None:
            made = splitter.maybe_split(sites, hidden.get("h"), step)
            if made:
                known = {id(p) for g in opt.param_groups for p in g["params"]}
                fresh = [p for p in m.parameters() if id(p) not in known]
                mats=[p for p in fresh if p.dim()>=2]; vecs=[p for p in fresh if p.dim()<2]
                if mats: opt.add_param_group({"params":mats,"weight_decay":0.1})
                if vecs: opt.add_param_group({"params":vecs,"weight_decay":0.0})
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (step+1) % max(1, steps//8) == 0:
            v = val(m, cs)
            if grower is not None:
                rec = grower.step(np.mean(list(v.values())), m.pool, step)
                if rec["grew"]:
                    known = {id(p) for g in opt.param_groups for p in g["params"]}
                    fresh = [p for p in m.parameters() if id(p) not in known]
                    mats=[p for p in fresh if p.dim()>=2]; vecs=[p for p in fresh if p.dim()<2]
                    if mats: opt.add_param_group({"params":mats,"weight_decay":0.1})
                    if vecs: opt.add_param_group({"params":vecs,"weight_decay":0.0})
            npool = m.pool.n_experts() if pool else 0
            hist.append({"step":step+1, **v, "pool":npool})
            log(f"  {kind:<12} {step+1:>4} reading {SHORT[names[min(step//per,len(names)-1)]]:<5} "
                + "  ".join(f"{k} {v[k]:.3f}" for k in v)
                + (f"  pool {npool}" if pool else ""))
    ev = len(splitter.events) if splitter else 0
    return hist, ev

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=240); ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--block",type=int,default=256); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--device",default="cpu")
    a=ap.parse_args()
    dev=torch.device(a.device); torch.set_num_threads(6)
    cs={d:Corpus(d,dev,a.block,a.batch) for d in DOM}
    log=lambda s: print(s, flush=True)
    out={}
    for kind, lr in (("dense",3e-3), ("plateau",3e-3), ("conflict",3e-3),
                     ("conflict+lr",8e-4)):
        print(f"--- {kind} (lr {lr:g}) ---")
        h, ev = run(kind, cs, a.steps, a.batch, a.block, a.seed, lr, dev, log)
        out[kind]={"hist":h,"splits":ev,"lr":lr}; print()
    doms=["code","math","chat","chess"]
    print(f"{'':<14}"+"".join(f"{d:>8}" for d in doms)+f"{'mean':>8}{'spread':>8}{'pool':>7}{'splits':>8}")
    for k,r in out.items():
        f=r["hist"][-1]; vals=[f[d] for d in doms]
        print(f"{k:<14}"+"".join(f"{v:>8.3f}" for v in vals)
              +f"{np.mean(vals):>8.3f}{max(vals)-min(vals):>8.3f}{f['pool']:>7}{r['splits']:>8}")
    json.dump(out, open("runs/conflict_test.json","w"), indent=1)
    print("\nwrote runs/conflict_test.json")

if __name__=="__main__": sys.exit(main())
