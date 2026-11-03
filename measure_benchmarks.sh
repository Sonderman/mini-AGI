#!/usr/bin/env bash
#
# The long measurements behind the README's benchmarks, run apart from
# ./release.sh because together they take about a day of the card.
#
#   enwik9, frozen and with dynamic evaluation, on the 2,000 pieces of
#     2,048 bytes ........................ tools/measure_enwik9_adapt.py    ~35 min
#   enwik9, best effort: every page of the 50 MB held-out slice, in the
#     model's native window, learning as it reads
#                                          tools/measure_enwik9_adapt.py
#                                            --natural                     ~13 h
#   enwik9, frozen, for the table ......... tools/measure_enwik9.py
#     (the first one's frozen pass is that measurement, so it is reused)
#   PG19, frozen: the whole test split .... tools/eval_pg19.py               ~1 h
#   PG19, best effort: all 100 books of the test split, in the model's
#     native window, learning as it reads  tools/measure_enwik9_adapt.py
#                                            --natural --slice data/eval/pg19  ~10 h
#
# The best efforts read everything they are scored on, as one stream the
# model goes on learning from: each chunk is scored first and then learned
# from, so no byte is scored by weights that have seen it, and what the
# model learns carries into everything it reads after.
#
# Every measurement that learns runs the learning rate's controller as
# training does, because it is part of the model (minagi/governor.py): it
# starts by reading the held-out set, data/val, as a training session does,
# then takes training's held-out check every ten minutes of learning - about
# half a minute each - and the controller sets the rate from it. Without
# data/val the rate stays where the run left it, and the measurement says so.
#
# Every measurement that learns does it on a copy of the weights, thrown away
# afterwards; weights/ is never written. Weights already measured are not
# measured again, so a second run on the same weights costs nothing. A
# measurement that fails is reported and the rest still run.
#
# Then it redraws the scaling chart and refreshes the README from what it
# measured. ./release.sh refreshes the README too, but from the last of these
# measurements, so run this first when the benchmark numbers should move.
#
#     ./measure_benchmarks.sh
#
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRC"

# the measurements need the card to itself
if ! busy=$(python3 tools/measure_heldout.py --check); then
    echo "$busy."
    echo "These measurements need the GPU to themselves. Stop the run, then run"
    echo "./measure_benchmarks.sh again. Nothing was measured."
    exit 1
fi

failed=()
step() {
    local what="$1"; shift
    echo
    echo "== $what"
    if ! "$@"; then
        echo "  WARN    $what failed - its README figures keep their older,"
        echo "          dated measurement."
        failed+=("$what")
    fi
}

step "enwik9: frozen, and with dynamic evaluation, on the 2,048-byte pieces" \
    python3 tools/measure_enwik9_adapt.py
step "enwik9: best effort, every page of the slice in the native window" \
    python3 tools/measure_enwik9_adapt.py --natural
step "enwik9: frozen, for the table" \
    python3 tools/measure_enwik9.py
step "PG19: frozen, the whole test split" \
    python3 tools/eval_pg19.py
step "PG19: best effort, all 100 books in the native window" \
    python3 tools/measure_enwik9_adapt.py --natural --slice data/eval/pg19 \
        --out runs/scaling/pg19_natural.jsonl
step "the scaling chart" \
    python3 tools/plot_scaling.py
step "the README" \
    python3 tools/refresh_readme.py

echo
if [ "${#failed[@]}" -eq 0 ]; then
    echo "all measured; the README and the scaling chart are up to date"
else
    echo "${#failed[@]} step(s) failed:"
    for f in "${failed[@]}"; do echo "  - $f"; done
    exit 1
fi
