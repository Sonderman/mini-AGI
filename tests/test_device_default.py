#!/usr/bin/env python3
"""
Without --device, train.py takes the GPU the machine has: CUDA (or ROCm) where
there is one, then Apple Silicon's (MPS), then the CPU. A machine with only
MPS read on the CPU unless told otherwise.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T                                              # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


class FakeMPS:
    def __init__(self, on):
        self.on = on

    def is_available(self):
        return self.on


real_cuda, real_mps = torch.cuda.is_available, getattr(torch.backends, "mps", None)
real_parse, real_argv = argparse.ArgumentParser.parse_args, sys.argv


def default_device():
    """What main() would read --device as, with nothing given."""
    got = {}

    def grab(self, *a, **k):
        got["args"] = real_parse(self, *a, **k)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = grab
    sys.argv = ["train.py", "read", "nowhere"]
    try:
        T.main()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
        sys.argv = real_argv
    return got["args"].device


try:
    for cuda, mps, want in ((True, True, "cuda"), (True, False, "cuda"),
                            (False, True, "mps"), (False, False, "cpu")):
        torch.cuda.is_available = lambda _c=cuda: _c
        torch.backends.mps = FakeMPS(mps)
        got = T._auto_device()
        check(f"cuda {cuda}, mps {mps}: {want}", got == want, got)
        got = default_device()
        check("...and that is what --device reads with nothing given",
              got == want, got)
    torch.cuda.is_available = lambda: False
    del torch.backends.mps
    check("a torch without an MPS backend at all: cpu",
          T._auto_device() == "cpu")
finally:
    torch.cuda.is_available = real_cuda
    if real_mps is not None:
        torch.backends.mps = real_mps

print("\n  " + ("all device-default tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
