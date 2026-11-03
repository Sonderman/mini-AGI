"""
One number per capability, and how each has moved since last time.

What belongs here is what this model can actually be measured on. It is a
character model, so the standard is BITS PER CHARACTER on held-out text -
the enwik8 and text8 metric, comparable with published character models. The
benchmarks small instruction-tuned models are scored on (MMLU, HellaSwag,
ARC, GSM8K) are not included, and that is deliberate: they need knowledge and
instruction-following far beyond what this has read, so they would return
chance and carry no signal about whether training is working.

Every result is appended to runs/benchmarks.jsonl, and each run prints the
change since the one before it, so the table reads as progress rather than a
snapshot.

    python3 -m benchmarks report
    python3 -m benchmarks report --tasks          # + arithmetic, chess, code
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = "runs/benchmarks.jsonl"


def bits_per_char(model, cfg, folder, chunk, device, chunks):
    """Held-out loss per domain, read the way training reads."""
    from minagi.stream import FolderEvaluator
    ctx = int(getattr(model, "_ctx_now", 0) or cfg.block)
    ev = FolderEvaluator(model, folder, chunk, ctx, device)
    d = ev.run(chunks)
    d.pop("stderr", None)
    return {k: v / math.log(2) for k, v in d.items()}


def word_completion(model, tok, folder, device, n=60, ctx=512):
    """
    Give the model a passage with the last word removed, and see whether it
    writes that word. The LAMBADA idea, on our own held-out text: it needs
    the context, not just the local spelling.
    """
    from minagi.ingest import collect, as_stream
    import re
    files = collect([folder])
    rng = np.random.default_rng(0)
    hit = tried = 0
    prefix = 0.0            # partial credit, so the metric says something
                            # before whole words are ever right
    for f in files[: max(1, n // 6)]:
        data = as_stream(f)
        for _ in range(6):
            if tried >= n or len(data) < ctx + 64:
                break
            off = int(rng.integers(0, len(data) - ctx - 64))
            text = tok.decode([int(v) for v in data[off:off + ctx]])
            m = list(re.finditer(r"\b[A-Za-z]{4,}\b", text))
            if len(m) < 4:
                continue
            last = m[-1]
            prompt, want = text[:last.start()], last.group(0)
            ids = torch.tensor([list(tok.encode(prompt).ids)], device=device)
            with torch.no_grad():
                out = model.generate(ids, len(want) + 2, temperature=0.0)
            got = tok.decode([int(v) for v in out[0, ids.shape[1]:]])
            tried += 1
            hit += got.strip().startswith(want)
            g = got.strip()
            k = 0
            while k < len(want) and k < len(g) and g[k] == want[k]:
                k += 1
            prefix += k / len(want)
    return {"word_completion": hit / max(tried, 1),
            "word_prefix": prefix / max(tried, 1), "_n": tried}


def show_history(path):
    """Every result so far. Reads the log; measures nothing."""
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    if not rows:
        print(f"  nothing measured yet - run `python3 -m benchmarks report`")
        return 0
    print(f"  {len(rows)} measurements from {path}\n")
    print(f"  {'when':<17}{'step':>9}{'Mchar':>8}{'ctx':>8}"
          f"{'experts':>9}{'bpc':>8}{'wordpfx':>9}")
    print("  " + "-" * 67)
    for r in rows:
        when = time.strftime("%m-%d %H:%M", time.localtime(r.get("t", 0)))
        print(f"  {when:<17}{(r.get('step') or 0):>9,}"
              f"{(r.get('chars') or 0)/1e6:>8.1f}{(r.get('context') or 0):>8,}"
              f"{(r.get('experts') or 0):>9}"
              f"{(r.get('bpc') or float('nan')):>8.4f}"
              f"{(r.get('word_prefix') or 0):>8.3f}")
    if len(rows) > 1:
        d = (rows[-1].get("bpc") or 0) - (rows[0].get("bpc") or 0)
        print(f"\n  bits per character over the whole record: {d:+.4f}")
    return 0


def previous(path):
    if not os.path.exists(path):
        return None
    prev = None
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    prev = json.loads(ln)
                except json.JSONDecodeError:
                    pass
    return prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights")
    ap.add_argument("--held-out", default="data/val")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=60,
                    help="held-out chunks per domain; more is steadier")
    ap.add_argument("--words", type=int, default=60)
    ap.add_argument("--tasks", action="store_true",
                    help="also run arithmetic, chess and code (slow)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--history", action="store_true",
                    help="print what has already been measured and exit; "
                         "runs nothing and needs no GPU")
    a = ap.parse_args()

    if a.history:
        return show_history(a.log)

    from minagi.precision import set_compute_dtype
    from minagi.tokenizer import ByteTokenizer
    from benchmarks.code import load_any
    from minagi.config import load as load_cfg, get as cfg_get

    conf = load_cfg()
    set_compute_dtype(cfg_get(conf, "training.precision", "bf16"))
    device = torch.device(a.device)
    model, ck = load_any(a.ckpt, device)
    cfg = model.cfg
    tok = ByteTokenizer()

    man = {}
    mf = os.path.join(a.ckpt, "manifest.json")
    if os.path.isdir(a.ckpt) and os.path.exists(mf):
        man = json.load(open(mf))
    model._ctx_now = man.get("context_now") or cfg.block

    print(f"  {a.ckpt}: step {man.get('step', '?'):,} · "
          f"{(man.get('read_chars') or 0)/1e6:.1f}M characters read · "
          f"{man.get('n_experts', '?')} experts · "
          f"context {model._ctx_now:,}")

    res = {"t": time.time(), "step": man.get("step"),
           "chars": man.get("read_chars"), "experts": man.get("n_experts"),
           "context": model._ctx_now}

    bpc = bits_per_char(model, cfg, a.held_out, a.chunk, device, a.chunks)
    res.update({f"bpc_{k}": v for k, v in bpc.items()})
    res["bpc"] = float(np.mean(list(bpc.values())))

    wc = word_completion(model, tok, a.held_out + "/chat", device, a.words)
    res["word_completion"] = wc["word_completion"]
    res["word_prefix"] = wc["word_prefix"]

    if a.tasks:
        try:
            from benchmarks.arithmetic import evaluate as arith_eval
            res["arithmetic"] = arith_eval(model, tok, device)
        except Exception as e:
            print(f"  (arithmetic skipped: {e})")

    prev = previous(a.log)
    print(f"\n  {'metric':<26} {'value':>10} {'change':>10}")
    print("  " + "-" * 48)

    def row(name, key, val, lower_better=True, fmt="{:.4f}"):
        was = (prev or {}).get(key)
        if was is None:
            ch = "     -"
        else:
            d = val - was
            good = (d < 0) if lower_better else (d > 0)
            ch = f"{d:+.4f} {'better' if good else 'worse'}" if abs(d) > 1e-9 \
                else "  same"
        print(f"  {name:<26} {fmt.format(val):>10} {ch:>10}")

    row("bits per character", "bpc", res["bpc"])
    for k in sorted(bpc):
        row(f"  {k}", f"bpc_{k}", bpc[k])
    row("word completion (exact)", "word_completion",
        res["word_completion"], lower_better=False)
    row("word completion (prefix)", "word_prefix", res["word_prefix"],
        lower_better=False)
    if "arithmetic" in res:
        row("arithmetic exact match", "arithmetic", res["arithmetic"],
            lower_better=False)

    print(f"\n  perplexity per character: "
          f"{2 ** res['bpc']:.2f}   ({wc['_n']} word-completion trials)")

    if not a.no_log:
        os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
        with open(a.log, "a") as f:
            f.write(json.dumps(res, separators=(",", ":")) + "\n")
        print(f"  appended to {a.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
