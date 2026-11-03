#!/usr/bin/env python3
"""
Create a fresh weights directory from config.yaml.

A thin command line over minagi.create, which is where the work lives now -
`train.py read` calls it itself when pointed at a directory that is not there,
so this script is a convenience for making a model with settings other than
the ones in the file, not a step anybody has to run.

    python3 tools/init_weights.py --experts 32 --d-ff 768
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from minagi.config import load, get
    from minagi.create import create

    c = load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=get(c, "data.weights", "weights"))
    # Every shape argument defaults to None so that unset means "whatever
    # config.yaml says" rather than a second copy of the defaults that can
    # drift away from it.
    for name, help_ in (("experts", "how many exist at the start"),
                        ("resident", "how many are on the card at once"),
                        ("d-ff", "hidden units inside one expert"),
                        ("depth", "SwiGLU blocks stacked inside one expert"),
                        ("d-model", "width of the trunk"),
                        ("trunk-d-ff", "the trunk's own feed-forward"),
                        ("n-head", "attention heads"),
                        ("block", "the context ceiling"),
                        ("max-steps", "most rows a character may take"),
                        ("top-k", "experts one character routes through")):
        ap.add_argument(f"--{name}", type=int, default=None, help=help_)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    try:
        create(a.out, seed=a.seed, force=a.force, verbose=True,
               experts=a.experts, resident=a.resident, d_ff=a.d_ff,
               depth=a.depth, d_model=a.d_model, trunk_d_ff=a.trunk_d_ff,
               n_head=a.n_head, block=a.block, max_steps=a.max_steps,
               top_k=a.top_k)
    except FileExistsError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
