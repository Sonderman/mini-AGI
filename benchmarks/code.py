#!/usr/bin/env python3
"""
Benchmarks for a very small Python code model.

The point of this file is to make training decisions falsifiable. A benchmark
that reads 0% at every checkpoint carries no information, so HumanEval-style
functional correctness is included but is NOT the steering signal at this
scale - it is there to stay honest about the ceiling.

The signal that actually moves at 10-100M params, in order of usefulness:

  top1        next-token accuracy on held-out Python (moves every few hundred steps)
  parse       generated function bodies that ast.parse accepts
  balance     brackets/quotes closed
  defined     fraction of referenced names that were actually bound
  nonempty    bodies that are more than `pass` or a bare string
  func        tiny executable tasks (expected ~0 for a long time; kept anyway)

Usage:
    python3 -m benchmarks code --ckpt weights --data data_char
    python3 bench.py --ckpt a.pt --data data --json > a.json   # to diff runs
"""

import os
import re
import ast
import sys
import json
import time
import math
import types
import builtins
import argparse
import textwrap
import subprocess

import numpy as np
import torch

from minagi.recur import load_model
from minagi.tokenizer import load_tokenizer


def held_out_bytes(data_dir, limit=None):
    """
    The held-out text of a domain, as bytes.

    Takes either a folder of text files - which is what the corpora are now -
    or one of the old .bin directories, so a benchmark pointed at either
    still works.
    """
    import numpy as _np
    if os.path.isdir(data_dir) and not os.path.exists(
            os.path.join(data_dir, "val.bin")):
        out = bytearray()
        for root, _, files in os.walk(data_dir):
            for f in sorted(files):
                with open(os.path.join(root, f), "rb") as fh:
                    out += fh.read()
                if limit and len(out) >= limit:
                    return bytes(out[:limit])
        return bytes(out)
    val = _np.memmap(os.path.join(data_dir, "val.bin"),
                     dtype=_np.uint16, mode="r")
    n = limit or len(val)
    return bytes(int(c) for c in val[:n] if c < 256)


# load_any lives in minagi/recur.py: it is how anything loads this model,
# not a benchmark concern, and serve.py needs it without importing this
# package. Re-exported so `from benchmarks.code import load_any` keeps
# working for the probes that already do it.
from minagi.recur import load_any                       # noqa: F401


BUILTINS = set(dir(builtins)) | {"self", "cls", "__name__", "__file__", "_"}

# Tiny executable tasks. Deliberately trivial - if a model this size ever moves
# this number off zero, that is real news rather than noise.
FUNCTIONAL = [
    ("def add(a, b):\n    return ", "assert add(2, 3) == 5"),
    ("def is_even(n):\n    return ", "assert is_even(4) and not is_even(3)"),
    ("def double(xs):\n    return [", "assert double([1, 2]) == [2, 4]"),
    ("def longest(words):\n    return max(", "assert longest(['a','bbb']) == 'bbb'"),
    ("def total(xs):\n    return ", "assert total([1, 2, 3]) == 6"),
    ("def first_or_none(xs):\n    return xs[0] if ", "assert first_or_none([]) is None"),
    ("def reverse(s):\n    return ", "assert reverse('abc') == 'cba'"),
    ("def count_true(flags):\n    return ", "assert count_true([True, False, True]) == 2"),
    ("def clamp(v, lo, hi):\n    return ", "assert clamp(5, 0, 3) == 3"),
    ("def unique(xs):\n    return ", "assert sorted(unique([1,1,2])) == [1,2]"),
]


def cut_function(text):
    """Keep the def line plus its indented body; drop whatever follows."""
    lines = text.split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line[0].isspace():
            break
        out.append(line)
    return "\n".join(out).rstrip()


def analyse(src):
    """Structural checks on one generated function."""
    res = {"parse": 0, "trunc": 0, "valid": 0, "repeat": 0.0, "balance": 0,
           "nonempty": 0, "names_used": 0, "names_defined": 0}
    # bracket/quote balance is checkable even when the parse fails
    try:
        list(__import__("tokenize").generate_tokens(
            iter(src.splitlines(True)).__next__))
        res["balance"] = 1
    except Exception:
        res["balance"] = 0
    kind = classify(src)
    res["parse"] = 1 if kind == "parse" else 0
    res["trunc"] = 1 if kind == "trunc" else 0
    res["valid"] = 1 if kind in ("parse", "trunc") else 0
    res["repeat"] = repetition(src)
    if kind != "parse":
        return res
    tree = ast.parse(src)

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is not None:
        body = [s for s in fn.body
                if not isinstance(s, ast.Pass)
                and not (isinstance(s, ast.Expr)
                         and isinstance(s.value, ast.Constant))]
        res["nonempty"] = 1 if body else 0

    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    used = [n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    res["names_used"] = len(used)
    res["names_defined"] = sum(1 for u in used if u in bound or u in BUILTINS)
    return res


def run_functional(model, tok, device, n_tokens=64, temp=0.2, timeout=5):
    """Execute generated solutions in a subprocess. Never in this process."""
    passed = 0
    for prompt, check in FUNCTIONAL:
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        out = model.generate(ids, n_tokens, temperature=temp, top_k=40)
        src = cut_function(tok.decode(out[0].tolist()))
        program = src + "\n" + check + "\nprint('PASS')\n"
        try:
            r = subprocess.run([sys.executable, "-c", program],
                               capture_output=True, timeout=timeout,
                               env={"PATH": "/usr/bin:/bin"})
            if b"PASS" in r.stdout:
                passed += 1
        except (subprocess.TimeoutExpired, OSError):
            pass
    return passed / len(FUNCTIONAL)


@torch.no_grad()
def teacher_forced(model, data_dir, device, batches=40, batch=16,
                   tokens_per_batch=16384):
    raw = held_out_bytes(data_dir, 4_000_000)
    val = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    block = model.cfg.block
    # `batch` was chosen when the context was 1024. At 4096 the same batch is
    # four times the activations and will not fit an 8GB card, so hold the
    # TOKEN count fixed instead - that is what the memory scales with, and it
    # keeps the number the benchmark averages over the same across contexts.
    batch = max(1, min(batch, tokens_per_batch // max(block, 1)))
    rng = np.random.default_rng(1234)          # fixed: runs stay comparable
    losses, correct, total = [], 0, 0
    ac = torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"))
    model.eval()
    for _ in range(batches):
        ix = rng.integers(0, len(val) - block - 1, size=batch)
        x = torch.from_numpy(np.stack([val[i:i + block] for i in ix]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val[i + 1:i + block + 1] for i in ix]).astype(np.int64)).to(device)
        with ac:
            logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return float(np.mean(losses)), correct / total


def held_out_prompts(data_dir, tok, n, max_tokens=400_000):
    """
    Function signatures from the held-out split, kept syntactically valid.

    A method signature lifted out of its class and dedented to column 0 is a
    trap: the model correctly continues at method-body depth and the result
    cannot parse. That scored 10/40 generations as syntax errors when the
    model had done nothing wrong. Indented defs therefore keep their
    indentation and get a synthetic class header.
    """
    text = held_out_bytes(data_dir, max_tokens).decode("utf-8", "replace")
    seen, prompts = set(), []
    for m in re.finditer(r"^([ \t]*)def ([a-zA-Z_]\w*)\(([^\n]*)\):[ \t]*$",
                         text, re.M):
        indent, name = m.group(1), m.group(2)
        line = m.group(0)
        if name.startswith("_") or name in seen or len(line.strip()) > 110:
            continue
        if "\t" in indent or (len(indent) % 4):
            continue
        seen.add(name)
        prompts.append(f"class _C:\n{line}\n" if indent else line + "\n")
        if len(prompts) >= n:
            break
    return prompts


# Failures that mean "ran out of budget mid-construct", not "wrote bad Python".
TRUNCATION = (
    "unexpected eof", "were never closed", "unterminated triple-quoted",
    "unterminated string", "expected an indented block",
)


def classify(src_text):
    """-> 'parse' | 'trunc' | 'error'."""
    try:
        ast.parse(src_text)
        return "parse"
    except SyntaxError as e:
        msg = (e.msg or "").lower()
        if any(t in msg for t in TRUNCATION):
            return "trunc"
        lines = src_text.split("\n")
        last = max((i for i, l in enumerate(lines, 1) if l.strip()), default=1)
        return "trunc" if (e.lineno or 0) >= last else "error"
    except (ValueError, MemoryError, RecursionError):
        return "error"


def repetition(src_text, n=12):
    """
    Repetition as the max of a line-level and a character-n-gram measure.

    The line-level version alone is blind in exactly the case that matters. A
    greedy loop that never emits a newline - "a list of all the elements in a
    list of all the elements in ..." - is one single line, so duplicate-line
    rate reads 0.00 for the worst degeneration the model produces. The n-gram
    term catches it.
    """
    lines = [l.strip() for l in src_text.split("\n")[1:] if l.strip()]
    line_rep = 1.0 - len(set(lines)) / len(lines) if len(lines) >= 2 else 0.0

    body = src_text[src_text.find("\n") + 1:]
    grams = [body[i:i + n] for i in range(len(body) - n + 1)]
    gram_rep = 1.0 - len(set(grams)) / len(grams) if len(grams) >= 2 else 0.0
    return round(max(line_rep, gram_rep), 4)


def run_bench(model, tok, data_dir, device, n_prompts=120, n_tokens=192,
              temp=0.0, do_exec=True, verbose=True, seed=1234,
              rep_penalty=1.0, no_repeat_ngram=0):
    """
    Seeded generation, deliberately.

    The teacher-forced metrics use a fixed rng and were reproducible; the
    generation metrics did not, and the same checkpoint at the same settings
    scored parse 0.470 and 0.550 on two runs. At n=100 the standard error is
    about 0.05, so that spread is noise - and it is the same size as the
    differences between models being compared. Seeding makes a comparison a
    comparison rather than a coin flip. Raise n_prompts to shrink the interval.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    # A budget of N tokens is N CHARACTERS at character level but roughly 3.3x
    # that many at BPE level. Using the same number for both truncates the
    # character model mid-function and reports it as failure.
    if getattr(tok, "size", None) == 256 and n_tokens < 400:
        n_tokens = int(n_tokens * 3.3)
    t0 = time.time()
    out = {}
    loss, top1 = teacher_forced(model, data_dir, device)
    out["val_loss"] = round(loss, 4)
    out["ppl"] = round(math.exp(min(loss, 20)), 2)
    out["top1"] = round(top1, 4)

    prompts = held_out_prompts(data_dir, tok, n_prompts)
    agg = {"parse": 0, "trunc": 0, "valid": 0, "repeat": 0.0,
           "balance": 0, "nonempty": 0}
    used = defined = 0
    for i, prompt in enumerate(prompts):
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        gen = model.generate(ids, n_tokens, temperature=temp,
                             top_k=(40 if temp > 0 else 0),
                             rep_penalty=rep_penalty,
                             no_repeat_ngram=no_repeat_ngram)
        src = cut_function(tok.decode(gen[0].tolist()))
        r = analyse(src)
        for k in agg:
            agg[k] += r[k]
        used += r["names_used"]
        defined += r["names_defined"]
        if verbose and i % 20 == 0:
            print(f"\r  generating {i}/{len(prompts)}", end="", file=sys.stderr)
    if verbose:
        print("\r" + " " * 30 + "\r", end="", file=sys.stderr)
    n = max(len(prompts), 1)
    out["n_prompts"] = n
    for k, v in agg.items():
        out[k] = round(v / n, 4)
    out["defined"] = round(defined / used, 4) if used else 0.0
    out["func"] = round(run_functional(model, tok, device), 4) if do_exec else None
    # 95% interval on a proportion, so two runs can be told apart honestly
    out["ci95"] = round(1.96 * math.sqrt(0.25 / max(n, 1)), 4)
    out["seconds"] = round(time.time() - t0, 1)
    return out


HEADER = ["val_loss", "ppl", "top1", "valid", "parse", "trunc", "repeat",
          "nonempty", "defined", "func"]


def format_row(res, label=""):
    cells = []
    for k in HEADER:
        v = res.get(k)
        cells.append("  n/a" if v is None else
                     (f"{v:7.4f}" if k in ("val_loss",) else
                      f"{v:7.2f}" if k == "ppl" else f"{v:7.3f}"))
    return f"{label:<14}" + " ".join(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/val/code")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-prompts", type=int, default=120)
    ap.add_argument("--n-tokens", type=int, default=192)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--no-exec", dest="exec", action="store_false", default=True)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ckpt = load_any(args.ckpt, device)
    from minagi.tokenizer import ByteTokenizer
    tok = ByteTokenizer()      # the model is byte level; no corpus needed
    res = run_bench(model, tok, args.data, device, args.n_prompts,
                    args.n_tokens, args.temp, args.exec, verbose=not args.json,
                    seed=args.seed, rep_penalty=args.rep_penalty,
                    no_repeat_ngram=args.no_repeat_ngram)
    res["step"] = ckpt.get("step")
    res["params"] = model.n_params()
    res["ckpt"] = args.ckpt
    if args.json:
        print(json.dumps(res))
    else:
        print(f"{'':<14}" + " ".join(f"{h:>7}" for h in HEADER))
        print(format_row(res, args.label or os.path.basename(args.ckpt)))
        print(f"{'':<14}(generation metrics +/- {res['ci95']:.3f} at 95%, "
              f"n={res['n_prompts']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
