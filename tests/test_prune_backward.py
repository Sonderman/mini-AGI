"""Forward, backward, prune, forward, backward - the crash the user hit."""
import os, shutil, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.recur import RecurConfig, RecurCoder
from minagi.paged import PagedPool
from minagi.pool import PooledMLP

d = "/tmp/minagi_prune_test"
shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
N, DM, DF = 20, 32, 48
for i in range(N):
    np.savez(os.path.join(d, f"e{i:05d}.npz"),
             w1=(np.random.randn(DF, DM) * .02).astype(np.float32),
             w3=(np.random.randn(DF, DM) * .02).astype(np.float32),
             w2=(np.random.randn(DM, DF) * .02).astype(np.float32))
cfg = RecurConfig(vocab_size=265, d_model=DM, n_head=4, d_ff=64, block=128,
                  use_pool=True, pool_experts=N, pool_d_ff=DF, pool_top_k=4,
                  pool_max=N, n_prelude=1, n_recur=1, n_coda=1, max_steps=2)
m = RecurCoder(cfg)
pool = PagedPool(d, DM, DF, N, resident=8, ram_capacity=8, device="cpu")
for mod in m.modules():
    if isinstance(mod, PooledMLP):
        mod._pool[0] = pool
m.pool = pool; pool.attach_sites(m); m.train()

x = torch.randint(0, 265, (1, 32)); y = torch.randint(0, 265, (1, 32))
def stepper(tag):
    _, loss = m(x, y)
    (loss + 0.01 * m.pool_aux()).backward()
    print(f"  {tag}: ok, loss {float(loss):.3f}, experts {pool.n_experts()}")

stepper("before prune")
# force two prunable experts: not resident, past their grace, silent, unchosen
free = [i for i in range(pool.n_experts()) if i not in pool.slots][:2]
pool.born[free] = 0.0
pool.gate.data[free] = 0.0
pool.last_seen[free] = 0.0
pool.segments = 10**6
gone = pool.prune(step=10**6, survival=1)
print(f"  pruned {gone} -> {pool.n_experts()} experts")
stepper("after prune")
print("  PASS")
