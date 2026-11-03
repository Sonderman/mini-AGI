"""
The model reading, one character at a time, with the experts it picks.

Three things move together:

  the text          what has been read, with the current character marked
  the trace         the last few hundred characters of routing, depth by
                    depth, scrolling under a playhead
  the column        every expert running for the CURRENT character - all
                    top_k of them at each of the 24 pooled depths, so what
                    persists from one character to the next is visible as
                    colour staying put rather than being averaged away

Colour is an expert's identity and is stable for the whole animation, so the
same colour anywhere really is the same expert. The file is read chunk after
chunk behind its KV cache, and every chunk's forward admits the experts its
characters ask for most; the frame says how many each chunk had to load.

    python3 tools/animate_routing.py --chars 512 --chunk 128
    -> runs/routing.mp4

Runs on CPU, so it can be made while a training run has the GPU.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minagi.pool import capture_routes                       # noqa: E402
from minagi.ingest import collect, as_stream                 # noqa: E402
from minagi.tokenizer import ByteTokenizer                   # noqa: E402
from minagi.precision import set_compute_dtype               # noqa: E402
import train as T                                            # noqa: E402


def gather(model, pool, data, chunk, n_chunks, device, learn, opt, aux):
    """
    Read exactly as the trainer reads.

    That means three things this used to get wrong. The KV cache carries from
    one chunk to the next, so a character is computed with the context behind
    it rather than as if the file had just started - attention drives the
    hidden state and the hidden state drives the router, so a missing cache
    changes which experts get picked. Every chunk is a forward of its own and
    is admitted what its text so far has voted for most, as every training
    window is. And with
    `learn`
    the gradient step happens between chunks, so what the picture shows is
    routing while the model is changing, not routing frozen at an instant.
    """
    from minagi.stream import detach_caches
    picks, best, swaps, losses = [], [], [], []
    caches = model.empty_caches()
    seen = 0
    pos = 0
    for c in range(n_chunks):
        a = data[pos:pos + chunk + 1].astype(np.int64)
        if len(a) < 3:
            break
        x = torch.from_numpy(a[:-1]).to(device).unsqueeze(0)
        y = torch.from_numpy(a[1:]).to(device).unsqueeze(0)
        loads_before = pool.loads
        if learn:
            opt.zero_grad(set_to_none=True)
        with torch.enable_grad() if learn else torch.no_grad():
            with capture_routes() as got:
                logits, loss = model(x, y, caches=caches, pos_offset=seen)
            if learn:
                (loss + aux * model.pool_aux()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        detach_caches(caches)
        moved = pool.loads - loads_before
        swaps.append((c * chunk, moved,
                      sorted(int(e) for e in pool.slots if e >= 0)))
        losses.append(float(loss))
        cur = np.stack([g[0].numpy() for g in got])           # [D, T, k]
        wgt = np.stack([g[1].numpy() for g in got])
        order = np.argsort(-wgt, axis=-1)                     # heaviest first
        cur = np.take_along_axis(cur, order, axis=-1)
        picks.append(cur)
        best.append(cur[:, :, 0])
        pos += x.shape[1]
        seen += x.shape[1]
        print(f"  chunk {c + 1}/{n_chunks}  (+{moved} loaded)  "
              f"loss {losses[-1]:.4f}", flush=True)
    return (np.concatenate(picks, axis=1), np.concatenate(best, axis=1),
            swaps, losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--text", default=None)
    ap.add_argument("--chunk", type=int, default=512,
                    help="characters per forward; each chunk admits the "
                         "experts it asks for")
    ap.add_argument("--chars", type=int, default=2048)
    ap.add_argument("--learn", action="store_true",
                    help="take the real gradient step between chunks, so the "
                         "routing shown is routing while the model changes")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--trail", type=int, default=192,
                    help="characters of trace visible behind the playhead")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="fp32",
                    help="bf16 matches training, but autocast is CUDA-only, "
                         "so a CPU render is fp32 whatever this says")
    ap.add_argument("--out", default="runs/routing.mp4")
    a = ap.parse_args()

    device = torch.device(a.device)
    set_compute_dtype(a.precision)
    # read_only: the training run may own this directory, and paging an
    # expert in marks it dirty whether or not anything changed it. --learn's
    # steps change the model in memory and are never written back.
    model, cfg, pool, man = T.build_paged(a.weights, device, None, 96,
                                          read_only=True)
    model.train()                    # what the trainer runs; there is no
    tok = ByteTokenizer()            # dropout, so this only matters for BPTT
    opt = None
    if a.learn:
        trunk, pool_ps = T._split_trunk_pool(model)
        opt = torch.optim.AdamW(
            [{"params": trunk, "name": "trunk", "lr": a.lr * 0.2},
             {"params": pool_ps, "name": "pool", "lr": a.lr}],
            lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
        pool.attach_optimiser(opt)

    path = a.text
    if path is None:
        files = collect(["data/val"]) or collect(["data/train"])
        path = files[len(files) // 2]
    data = as_stream(path)
    n_chunks = max(1, a.chars // a.chunk)
    print(f"reading {a.chars} characters of {os.path.relpath(path)}", flush=True)
    picks, best, swaps, losses = gather(model, pool, data, a.chunk, n_chunks,
                                        device, a.learn, opt, cfg.pool_aux)

    D, Tn, K = picks.shape
    text = tok.decode(list(data[:Tn].astype(int)))
    used = sorted(set(int(v) for v in picks.ravel()))
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    remap = {e: i for i, e in enumerate(used)}
    cmap = ListedColormap([base[i % 20] for i in range(len(used))])
    cmap.set_bad("#f2f2f2")           # nothing read yet - not expert zero
    trace = np.vectorize(remap.get)(best)
    grid = np.vectorize(remap.get)(picks)                    # [D, T, K]
    swap_at = {s[0]: s[1] for s in swaps}

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 5],
                          hspace=0.05, wspace=0.10)
    ax_txt = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_now = fig.add_subplot(gs[1, 1])

    ax_txt.axis("off")
    t_prev = ax_txt.text(0.0, 0.30, "", family="monospace", fontsize=10.5,
                         color="#7f8c8d", va="center", ha="left",
                         transform=ax_txt.transAxes)
    t_head = ax_txt.text(0.0, -0.10, "", family="monospace", fontsize=11.5,
                         color="#c0392b", va="center", ha="left",
                         transform=ax_txt.transAxes)

    im = ax_map.imshow(np.zeros((D, a.trail)), aspect="auto", cmap=cmap,
                       vmin=0, vmax=max(len(used) - 1, 1), origin="lower",
                       interpolation="nearest")
    ax_map.set_ylabel("depth (pooled block application)")
    ax_map.set_xlabel(f"the last {a.trail} characters read")
    ax_map.set_xticks([])
    play = ax_map.axvline(a.trail - 0.5, color="k", lw=2.0)

    now = ax_now.imshow(np.zeros((D, K)), aspect="auto", cmap=cmap,
                        vmin=0, vmax=max(len(used) - 1, 1), origin="lower",
                        interpolation="nearest")
    ax_now.set_title(f"this character\nthe {K} of {pool.n_resident()} "
                     f"resident chosen at EACH depth\n(heaviest at left)",
                     fontsize=8.5)
    ax_now.set_yticks([])
    ax_now.set_xticks([])

    title = fig.suptitle("", fontsize=12)
    # say on the picture what produced it, so it cannot be mistaken for
    # something it is not
    provenance = (
        f"weights at step {man.get('step', '?'):,} · {pool.n_experts()} "
        f"experts, {pool.n_resident()} resident · top-{K} per call site · "
        f"KV cache carried · chunk {a.chunk} · "
        + ("gradient step between chunks"
           if a.learn else "weights frozen (no gradient step)")
        + f" · {a.precision if device.type == 'cuda' else 'fp32 (CPU)'}")

    def frame(t):
        lo = max(0, t - a.trail + 1)
        win = trace[:, lo:t + 1].astype(float)
        if win.shape[1] < a.trail:
            # blank, not colour: before the trail fills there is genuinely
            # nothing there, and padding with an index would draw an expert
            # that never ran
            win = np.pad(win, ((0, 0), (a.trail - win.shape[1], 0)),
                         constant_values=np.nan)
        im.set_data(np.ma.masked_invalid(win))
        now.set_data(grid[:, t, :])

        s = max(0, t - 60)
        t_prev.set_text(repr(text[s:t])[1:-1][-110:])
        t_head.set_text(f"next: {text[t:t + 1]!r}")

        moved = swap_at.get((t // a.chunk) * a.chunk)
        boundary = (t % a.chunk) == 0
        note = (f"   —   {moved} experts loaded for this chunk"
                if boundary and moved else "")
        title.set_text(f"character {t + 1} of {Tn}   "
                       f"chunk {t // a.chunk + 1}{note}\n{provenance}")
        for spine in ax_now.spines.values():
            spine.set_edgecolor("#c0392b" if boundary else "#cccccc")
            spine.set_linewidth(2.5 if boundary else 0.8)
        return im, now, play, t_prev, t_head, title

    print(f"rendering {Tn} frames", flush=True)
    anim = animation.FuncAnimation(fig, frame, frames=Tn, interval=1000 / a.fps,
                                   blit=False)
    anim.save(a.out, writer=animation.FFMpegWriter(
        fps=a.fps, bitrate=2400,
        metadata={"title": "mini-AGI expert routing"}))
    print(f"wrote {a.out}  ({Tn} frames, {Tn / a.fps:.0f}s, "
          f"{len(used)} distinct experts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
