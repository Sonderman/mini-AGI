"""Run an experiment: python3 -m experiments <name> [args]

    forgetting   streaming vs mixed batches, with and without the expert pool
    trunk_lr     how slowly the shared trunk should learn
    freeze       which part of the model is doing the forgetting
    conflict     whether gradient conflict is a usable growth signal
    replay       whether replaying past text beats slowing the trunk
    capacity     what to spend the spare VRAM on: context, working set, expert shape

Each prints a table and writes JSON under runs/.
"""
import sys

RUNS = ["forgetting", "trunk_lr", "freeze", "conflict", "replay", "capacity"]

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in RUNS:
        print(f"usage: python3 -m experiments <{'|'.join(RUNS)}> [options]",
              file=sys.stderr)
        return 2
    name = sys.argv.pop(1)
    import importlib
    return importlib.import_module(f"experiments.{name}").main()

if __name__ == "__main__":
    sys.exit(main())
