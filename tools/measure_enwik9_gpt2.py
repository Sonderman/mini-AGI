#!/usr/bin/env python3
"""
GPT-2, all four sizes, on the enwik9 slice this model is scored on, measured
the same way, for the README's enwik9 table - the GPT-2 paper reports enwik8.

The pieces are tools/measure_enwik9.py's: 2,048-byte pieces of the held-out
enwik9 slice, the same `--pieces` of them, each predicted with no memory of
the others. A piece is cut back to whole UTF-8 characters at its two ends
(at most three bytes each side), tokenized, and preceded by <|endoftext|> so
its first token is predicted from nothing, as the first byte of each piece
is for this model. Bits per byte = the total -log2 probability of the pieces' tokens /
their total bytes - the papers' compression rate times 8, without running an
arithmetic coder. Full precision (fp32).

The models are openai-community/gpt2-xl (6.4 GB), gpt2-large (3.2 GB),
gpt2-medium (1.5 GB) and gpt2 (0.5 GB, the small one) on Hugging Face, MIT
licence; download a model's config, tokenizer files and model.safetensors
and pass the folder they are in:

    python3 tools/measure_enwik9_gpt2.py MODEL_DIR [--pieces 2000]

GPT-2 XL needs about 7 GB of the card in fp32, so not beside a training run.
"""
import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def whole_chars(b):
    """The piece cut back to whole UTF-8 characters at both ends: (text,
    bytes it covers, whether anything inside it was not UTF-8)."""
    i = 0
    while i < len(b) and i < 3 and (b[i] & 0xC0) == 0x80:
        i += 1                                   # a character's tail
    b = b[i:]
    for cut in range(4):                         # a character's head at the end
        try:
            return b[:len(b) - cut].decode("utf-8"), len(b) - cut, False
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace"), len(b), True


def score(model, ids, ctx, device):
    """Nats over ids[1:], each predicted from what comes before it: one window
    when the ids fit the context; otherwise the first window scores
    everything in it, and each next one - the last `ctx` ids ending half a
    context further on - scores only the ids not yet scored, each with at
    least half a context before it. Returns (nats, ids scored)."""
    import torch
    nats, scored, done = 0.0, 0, 1               # done: the next id to score
    while done < len(ids):
        end = min(len(ids), ctx if done == 1 else done + ctx // 2)
        start = max(0, end - ctx)
        x = torch.tensor([ids[start:end]], device=device)
        logp = torch.log_softmax(model(x, use_cache=False).logits[0].float(), -1)
        tgt = torch.tensor(ids[done:end], device=device)
        # logits at window position j predict ids[start + j + 1]
        rows = logp[done - start - 1:end - start - 1]
        nats -= float(rows.gather(1, tgt[:, None]).sum())
        scored += end - done
        done = end
    return nats, scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--pieces", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None,
                    help="what to call the model in the result; by default "
                         "its Hugging Face name, from the folder")
    a = ap.parse_args()

    import numpy as np
    import torch
    sys.path.insert(0, REPO)
    from minagi.stream import FolderEvaluator
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # the same pieces tools/measure_enwik9.py reads for this model
    ev = FolderEvaluator(None, os.path.join(REPO, "data", "enwik9"), 2048,
                         2048, "cpu", whole=False)
    pieces = [bytes(t[:2048].astype(np.uint8)) for t in ev._texts("wikipedia",
                                                                a.pieces)]
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32, use_safetensors=True).to(a.device)
    model.eval()
    eos, ctx = tok.eos_token_id, model.config.n_positions
    bits, nbytes, per, longest, over, bad = 0.0, 0, [], 0, 0, 0
    with torch.no_grad():
        for k, p in enumerate(pieces):
            text, n, broken = whole_chars(p)
            bad += broken
            ids = [eos] + tok(text, add_special_tokens=False)["input_ids"]
            longest = max(longest, len(ids))
            nats, _ = score(model, ids, ctx, a.device)
            over += len(ids) > ctx
            b = nats / math.log(2)
            bits += b
            nbytes += n
            per.append(b / n)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1} pieces, {bits / nbytes:.4f} bits/byte so far, "
                      f"{time.time() - t0:.0f} s", flush=True)
    parts = os.path.normpath(a.model).split(os.sep)
    hub = [x for x in parts if x.startswith("models--")]
    name = a.name or (hub[-1].split("--")[-1] if hub else parts[-1])
    out = {"model": name, "params": sum(q.numel() for q in model.parameters()),
           "dtype": "float32", "pieces": len(pieces), "bytes": nbytes,
           "bpb": bits / nbytes,
           "se": float(np.std(per, ddof=1) / math.sqrt(len(per))),
           "longest_tokens": longest, "pieces_over_context": over,
           "pieces_not_utf8": bad, "date": time.strftime("%Y-%m-%d %H:%M"),
           "seconds": round(time.time() - t0)}
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
