#!/usr/bin/env python3
"""
How often is the model's next chess move legal?

Full games answer a different question: one illegal move ends a game, so a
game-level score conflates per-move accuracy with game length. This measures
the per-move rate directly, on real positions taken from held-out games, which
is the number that decides whether a clean game is possible at all.

If p is the chance a single move is legal, a game where the model plays m moves
finishes cleanly with probability p**m. For a 40-move game that needs p > 0.983
to be more likely than not.

    python3 tools/chess_legality.py --weights weights --positions 80
"""
import argparse
import os
import random
import re
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.code import load_any                   # noqa: E402
from minagi.precision import set_compute_dtype         # noqa: E402
from minagi.tokenizer import ByteTokenizer             # noqa: E402

MOVE = re.compile(r"^(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|"
                  r"[a-h]x?[a-h][1-8](?:=[QRBN])?|[a-h][1-8](?:=[QRBN])?|"
                  r"O-O(?:-O)?)[+#]?$")


def games(folder, limit=400):
    """Real held-out games, as (header, list-of-san)."""
    out = []
    for fn in sorted(os.listdir(folder)):
        with open(os.path.join(folder, fn), errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("<g>"):
                    continue
                body = line[3:].replace("</g>", "").strip()
                bits = body.split()
                if len(bits) < 6:
                    continue
                head = " ".join(bits[:2])            # elo, result
                sans = [b for b in bits[2:] if not b.endswith(".")]
                if len(sans) >= 12:
                    out.append((head, sans))
                if len(out) >= limit:
                    return out
    return out


def prompt_for(head, sans, n, style):
    body = ""
    for i, san in enumerate(sans[:n]):
        body += (f"{i//2 + 1}. {san} " if i % 2 == 0 else f"{san} ")
    if n % 2 == 0:
        body += f"{n//2 + 1}."
    if style == "real":
        return f"<g>{head} " + body
    return f"<g {head}>" + " " + body     # the format the benchmark used to use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--data", default="data/val/chess")
    ap.add_argument("--positions", type=int, default=80)
    ap.add_argument("--max-new", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--styles", nargs="+", default=["real"],
                    choices=["real", "old"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device(a.device)
    set_compute_dtype("fp32" if dev.type == "cpu" else "bf16")
    model, ck = load_any(a.weights, dev)       # read_only by default
    model.eval()
    tok = ByteTokenizer()
    print(f"  step {ck.get('step'):,}", flush=True)

    gs = games(a.data)
    rng = random.Random(a.seed)
    picks = []
    for _ in range(a.positions):
        head, sans = gs[rng.randrange(len(gs))]
        n = rng.randrange(4, min(len(sans) - 1, 60))
        picks.append((head, sans, n))
    print(f"  {len(gs)} held-out games, {len(picks)} positions", flush=True)

    for style in a.styles:
        legal = parsed = shaped = 0
        bad = []
        for i, (head, sans, n) in enumerate(picks):
            board = chess.Board()
            ok = True
            for san in sans[:n]:
                try:
                    board.push_san(san)
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            p = prompt_for(head, sans, n, style)
            ids = tok.encode(p).ids[-model.cfg.block:]
            idx = torch.tensor([ids], device=dev)
            with torch.no_grad():
                out = model.generate(idx, a.max_new, temperature=0.0)
            txt = tok.decode(out[0].tolist())[len(p):]
            first = txt.strip().split()[0].rstrip(".,") if txt.strip().split() else ""
            if MOVE.match(first):
                shaped += 1
            try:
                board.parse_san(first)
                legal += 1
                parsed += 1
            except Exception:
                if len(bad) < 8:
                    bad.append(first or "<empty>")
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(picks)}  legal {legal}/{i+1}",
                      flush=True)
        n_try = len(picks)
        p_legal = legal / max(n_try, 1)
        print(f"\n  [{style}] legal {legal}/{n_try} = {p_legal:.3f}   "
              f"well-shaped SAN {shaped}/{n_try} = {shaped/max(n_try,1):.3f}")
        print(f"        rejected examples: {bad}")
        for m in (10, 20, 40):
            print(f"        P(clean game of {m} model moves) = "
                  f"{p_legal**m:.2e}")
        need = 0.5 ** (1 / 40)
        print(f"        for a 40-move game to be clean more often than not, "
              f"p must exceed {need:.4f}")


if __name__ == "__main__":
    main()
