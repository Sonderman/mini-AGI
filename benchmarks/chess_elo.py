#!/usr/bin/env python3
"""
Play the model against fixed opponents and estimate its rating.

RULES
An illegal move loses the game immediately. That is the honest scoring: a model
that cannot keep track of the position is not playing chess, however plausible
its moves look in isolation. Illegal-move rate is reported separately, because
early on it is the number that matters most.

THE OPPONENTS
There is no calibrated engine on this machine, so strength is anchored against
three opponents that can be described exactly:

    random    picks uniformly among legal moves
    greedy    one ply, takes the highest-value capture, else a random move
    minimax   two ply with alpha-beta, material plus piece-square tables

Their true ratings are approximate. The Elo numbers below are therefore
estimates against a described opponent, not FIDE ratings - the score against
each is the primary result and the Elo conversion is a convenience.

    python3 -m benchmarks chess --ckpt weights --games 40
"""

import argparse
import math
import random
import re
import sys

import chess
import torch

PIECE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
         chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# a coarse piece-square table: centre control for knights, advancement for pawns
PST_PAWN = [0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0]
PST_KNIGHT = [-50, -40, -30, -30, -30, -30, -40, -50,
              -40, -20, 0, 5, 5, 0, -20, -40,
              -30, 5, 10, 15, 15, 10, 5, -30,
              -30, 0, 15, 20, 20, 15, 0, -30,
              -30, 5, 15, 20, 20, 15, 5, -30,
              -30, 0, 10, 15, 15, 10, 0, -30,
              -40, -20, 0, 0, 0, 0, -20, -40,
              -50, -40, -30, -30, -30, -30, -40, -50]

APPROX_ELO = {"random": 250, "greedy": 800, "minimax2": 1400}


def evaluate(board):
    if board.is_checkmate():
        return -99999 if board.turn else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    s = 0
    for sq, pc in board.piece_map().items():
        v = PIECE[pc.piece_type]
        idx = sq if pc.color == chess.WHITE else chess.square_mirror(sq)
        if pc.piece_type == chess.PAWN:
            v += PST_PAWN[idx]
        elif pc.piece_type == chess.KNIGHT:
            v += PST_KNIGHT[idx]
        s += v if pc.color == chess.WHITE else -v
    return s


def minimax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    if board.turn == chess.WHITE:
        best = -1e9
        for m in board.legal_moves:
            board.push(m)
            best = max(best, minimax(board, depth - 1, alpha, beta))
            board.pop()
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = 1e9
    for m in board.legal_moves:
        board.push(m)
        best = min(best, minimax(board, depth - 1, alpha, beta))
        board.pop()
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def bot_move(board, kind, rng):
    moves = list(board.legal_moves)
    if not moves:
        return None
    if kind == "random":
        return rng.choice(moves)
    if kind == "greedy":
        caps = [(PIECE[board.piece_at(m.to_square).piece_type], m)
                for m in moves if board.piece_at(m.to_square)]
        if caps:
            best = max(c[0] for c in caps)
            return rng.choice([m for v, m in caps if v == best])
        return rng.choice(moves)
    best_m, best_v = None, (-1e9 if board.turn == chess.WHITE else 1e9)
    rng.shuffle(moves)
    for m in moves:
        board.push(m)
        v = minimax(board, 1, -1e9, 1e9)
        board.pop()
        if (board.turn == chess.WHITE and v > best_v) or \
           (board.turn == chess.BLACK and v < best_v):
            best_v, best_m = v, m
    return best_m or rng.choice(moves)


class ModelPlayer:
    """Feeds the game so far back in the training format and reads a move out."""

    def __init__(self, model, tok, elo=2200, max_new=10):
        self.model, self.tok, self.elo, self.max_new = model, tok, elo, max_new
        self.device = next(model.parameters()).device

    def move_text(self, header, moves_san, board):
        body = ""
        for i, san in enumerate(moves_san):
            if i % 2 == 0:
                body += f"{i//2 + 1}. {san} "
            else:
                body += f"{san} "
        if len(moves_san) % 2 == 0:
            body += f"{len(moves_san)//2 + 1}."
        return header + " " + body

    @torch.no_grad()
    def pick(self, board, moves_san, result_hint):
        # The training format is `<g>ELO RESULT 1. e4 e5 ...`, where <g> is a
        # single marker token. This used to build `<g ELO RESULT>`, which
        # tokenises as twelve raw bytes the model has never seen - so every
        # game was played from a prompt outside the training distribution.
        header = f"<g>{self.elo} {result_hint}"
        prompt = self.move_text(header, moves_san, board)
        ids = self.tok.encode(prompt).ids[-self.model.cfg.block:]
        idx = torch.tensor([ids], device=self.device)
        out = self.model.generate(idx, self.max_new, temperature=0.0)
        text = self.tok.decode(out[0].tolist())[len(prompt):]
        san = text.strip().split()[0] if text.strip().split() else ""
        san = san.rstrip(".,")
        if not san:
            return None, ""
        try:
            return board.parse_san(san), san
        except Exception:
            return None, san


def play(model_player, opponent, model_white, rng, max_plies=200):
    board = chess.Board()
    hint = "1-0" if model_white else "0-1"
    sans = []
    plies = 0
    while not board.is_game_over() and plies < max_plies:
        model_turn = (board.turn == chess.WHITE) == model_white
        if model_turn:
            mv, raw = model_player.pick(board, sans, hint)
            if mv is None:
                return ("illegal", raw, plies)
            sans.append(board.san(mv))
            board.push(mv)
        else:
            mv = bot_move(board, opponent, rng)
            if mv is None:
                break
            sans.append(board.san(mv))
            board.push(mv)
        plies += 1
    if board.is_game_over():
        r = board.result()
        if r == "1-0":
            return ("win" if model_white else "loss", "", plies)
        if r == "0-1":
            return ("loss" if model_white else "win", "", plies)
        return ("draw", "", plies)
    return ("draw", "", plies)


def elo_from_score(score, opponent_elo):
    if score <= 0.001:
        return opponent_elo - 800
    if score >= 0.999:
        return opponent_elo + 800
    return opponent_elo - 400 * math.log10(1 / score - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights",
                    help="the weights directory, or a .pt checkpoint")
    ap.add_argument("--data", default="data/val/chess")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--elo-prompt", type=int, default=2200)
    ap.add_argument("--opponents", nargs="+",
                    default=["random", "greedy", "minimax2"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from benchmarks.code import load_any
    from minagi.tokenizer import load_tokenizer
    device = torch.device(args.device)
    model, ck = load_any(args.ckpt, device)
    from minagi.tokenizer import ByteTokenizer
    tok = ByteTokenizer()      # the model is byte level; no corpus needed
    player = ModelPlayer(model, tok, args.elo_prompt)
    print(f"checkpoint step {ck.get('step')}, val {ck.get('val', float('nan')):.4f}")
    print(f"prompting the model as a {args.elo_prompt}-rated player\n")

    overall_illegal = overall_plies = 0
    results = {}
    for opp in args.opponents:
        rng = random.Random(args.seed)
        tally = {"win": 0, "loss": 0, "draw": 0, "illegal": 0}
        plies = []
        bad_examples = []
        for g in range(args.games):
            outcome, raw, n = play(player, opp, g % 2 == 0, rng)
            plies.append(n)
            if outcome == "illegal":
                tally["illegal"] += 1
                tally["loss"] += 1          # illegal move loses the game
                if len(bad_examples) < 3:
                    bad_examples.append((n, raw))
            else:
                tally[outcome] += 1
            print(f"\r  {opp}: {g+1}/{args.games}", end="", file=sys.stderr)
        score = (tally["win"] + 0.5 * tally["draw"]) / args.games
        est = elo_from_score(score, APPROX_ELO.get(opp, 1000))
        results[opp] = {"score": score, "elo": est, **tally,
                        "mean_plies": sum(plies) / len(plies)}
        overall_illegal += tally["illegal"]
        overall_plies += sum(plies)
        print(f"\r  vs {opp:<9} score {score:.3f}  "
              f"W{tally['win']} D{tally['draw']} L{tally['loss']}  "
              f"illegal {tally['illegal']}/{args.games}  "
              f"~{est:.0f} Elo   (mean {results[opp]['mean_plies']:.0f} plies)")
        if bad_examples:
            print(f"      first illegal outputs: "
                  f"{[b[1] for b in bad_examples]}")

    legal_games = sum(args.games - r["illegal"] for r in results.values())
    total = args.games * len(results)
    print(f"\ngames finished without an illegal move: {legal_games}/{total} "
          f"({100*legal_games/total:.0f}%)")
    ests = [r["elo"] for r in results.values()]
    print(f"rating estimate: {min(ests):.0f} - {max(ests):.0f} Elo "
          f"(mean {sum(ests)/len(ests):.0f})")
    print("\nThese are estimates against described opponents, not FIDE ratings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
