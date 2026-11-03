"""
Run a benchmark.

    python3 -m benchmarks report        every metric, and how each has moved
    python3 -m benchmarks arithmetic    exact-match arithmetic by digit count
    python3 -m benchmarks chess         rating against fixed opponents
    python3 -m benchmarks code          parse rate and functional correctness
"""
import sys

SUITES = {"report": "benchmarks.report",
          "code": "benchmarks.code",
          "arithmetic": "benchmarks.arithmetic",
          "chess": "benchmarks.chess_elo"}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SUITES:
        print(f"usage: python3 -m benchmarks <{'|'.join(SUITES)}> [options]",
              file=sys.stderr)
        return 2
    name = sys.argv.pop(1)
    import importlib
    return importlib.import_module(SUITES[name]).main()

if __name__ == "__main__":
    sys.exit(main())
