#!/usr/bin/env python3
"""
Where does the forgetting actually happen?

Growing the pool did not stop it and splitting on conflict did not stop it.
Both assume the damage is in the experts. This checks that assumption by
learning domain 1, then freezing one part at a time while learning domains
2-4, and asking which freeze rescues domain 1.

  freeze nothing   the baseline failure
  freeze core      embeddings, attention, norms - everything outside the pool
  freeze routers   the thing that decides WHICH experts run
  freeze experts   the weights themselves
  freeze all-but-new  experts learned so far are frozen; only new ones train

Whichever freeze recovers domain 1 is where the forgetting lives.
"""
import argparse, json, sys, numpy as np, torch
from experiments.corpus import Corpus
from experiments.schedule import lr_at
from minagi.recur import RecurConfig, RecurCoder
from minagi.pool import PooledMLP

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

def apply_freeze(m, mode, n_frozen_experts):
    for p in m.parameters(): p.requires_grad = True
    if mode == "none": return
    sites = [x for x in m.modules() if isinstance(x, PooledMLP)]
    router_ids = {id(s.router.weight) for s in sites} | \
                 {id(s.depth_emb) for s in sites}
    expert_ids = {id(p) for e in m.pool.experts for p in e.parameters()}
    for n,p in m.named_parameters():
        if mode=="core" and id(p) not in router_ids and id(p) not in expert_ids \
           and "pool.gate" not in n:
            p.requires_grad = False
        if mode=="routers" and id(p) in router_ids:
            p.requires_grad = False
        if mode=="experts" and id(p) in expert_ids:
            p.requires_grad = False
    if mode=="old-experts":
        for i,e in enumerate(m.pool.experts):
            if i < n_frozen_experts:
                for p in e.parameters(): p.requires_grad = False

def run(mode, cs, steps, batch, block, seed, device, log):
    torch.manual_seed(seed)
    cfg = RecurConfig(vocab_size=265, n_head=4, d_model=128, block=block,
                      d_ff=256, n_layer=4, n_prelude=1, n_recur=2, n_coda=1,
                      max_steps=2, use_pool=True, pool_experts=32,
                      pool_d_ff=64, pool_top_k=4, pool_max=160)
    m = RecurCoder(cfg).to(device)
    opt = torch.optim.AdamW([p for p in m.parameters()], lr=3e-3, betas=(0.9,0.95))
    rng = np.random.default_rng(seed)
    names=list(cs); per=max(1, steps//len(names))
    after_phase1 = None
    for step in range(steps):
        phase = min(step//per, len(names)-1)
        if phase == 1 and after_phase1 is None:
            after_phase1 = val(m, cs)
            apply_freeze(m, mode, cfg.pool_experts)
            n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in m.parameters())
            log(f"    [{mode}] after code: code {after_phase1['code']:.3f} | "
                f"now training {100*n_tr/n_all:.0f}% of parameters")
        lr = lr_at(step, steps, 3e-3, max(4, steps//20))
        for g in opt.param_groups: g["lr"]=lr
        opt.zero_grad(set_to_none=True)
        c = cs[names[phase]]; c.batch=batch
        x,y = c.batch_for("train", rng)
        _, loss = m(x,y)
        loss = loss + cfg.pool_aux * m.pool_aux()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step()
    final = val(m, cs)
    return after_phase1, final

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=200); ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--block",type=int,default=256); ap.add_argument("--device",default="cpu")
    a=ap.parse_args()
    dev=torch.device(a.device); torch.set_num_threads(6)
    cs={d:Corpus(d,dev,a.block,a.batch) for d in DOM}
    log=lambda s: print(s, flush=True)
    rows={}
    for mode in ("none","core","routers","experts","old-experts"):
        print(f"--- freeze {mode} ---")
        before, after = run(mode, cs, a.steps, a.batch, a.block, 0, dev, log)
        kept = 100*(1 - (after["code"]-before["code"])/max(before["code"],1e-9))
        rows[mode]={"before":before,"after":after,"kept":kept}
        print(f"    code {before['code']:.3f} -> {after['code']:.3f}   "
              f"final: " + "  ".join(f"{k} {after[k]:.3f}" for k in after))
    print(f"\n{'freeze':<14}{'code after phase 1':>20}{'code at the end':>18}{'damage':>10}")
    for k,r in rows.items():
        d = r["after"]["code"] - r["before"]["code"]
        print(f"{k:<14}{r['before']['code']:>20.3f}{r['after']['code']:>18.3f}"
              f"{d:>+10.3f}")
    json.dump(rows, open("runs/where_forgetting.json","w"), indent=1)
    print("\nthe freeze with the smallest damage is where the forgetting lives.")

if __name__=="__main__": sys.exit(main())
