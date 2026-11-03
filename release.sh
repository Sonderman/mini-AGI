#!/usr/bin/env bash
#
# Bring what the README reports up to date, and check the tree, before a push.
#
# The README's figures, the scaling chart and the architecture figure are
# outputs of the run, not prose, and a stale number is worse than none: it
# looks maintained and nobody re-checks it. This rebuilds them and checks what
# a push would carry. It commits nothing and pushes nothing: read `git diff`,
# then commit and push yourself.
#
#     ./release.sh              # refresh, then check
#     ./release.sh --check      # the checks alone, nothing regenerated
#
# Held-out is measured on the model as it is (tools/measure_heldout.py), and
# that needs the GPU and weights that are not changing on disk, so the refresh
# refuses while a run is training. enwik9 and PG19 are measured by
# ./measure_benchmarks.sh, which takes about a day of the card; the README
# uses what it last measured.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CHECK_ONLY=0
for a in "$@"; do
    case "$a" in
        --check)   CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

say() { printf '%s\n' "$*"; }

# --- refresh -----------------------------------------------------------------
if [ "$CHECK_ONLY" = 0 ]; then
    if ! busy=$(python3 tools/measure_heldout.py --check); then
        say "$busy."
        say "Held-out can't be measured for the README while it trains: the"
        say "measurement needs the GPU, and the weights keep changing on disk."
        say "Stop the run and run this again, or run ./release.sh --check."
        exit 1
    fi
    say "refreshing what the README reports"
    # first, so that if it fails nothing has been regenerated: the README's
    # table is built from this measurement
    if ! python3 tools/measure_heldout.py; then
        say "Held-out could not be measured, so the README's figures would not"
        say "be faithful. Stopping here; nothing was regenerated."
        exit 1
    fi
    if python3 tools/plot_scaling.py >/dev/null 2>&1; then
        say "  ok      assets/scaling.png redrawn"
    else
        say "  WARN    tools/plot_scaling.py failed - assets/scaling.png is"
        say "          whatever it was. Run it by hand to see why."
    fi
    # the architecture figure names the pool's size, which moves as it grows
    if python3 tools/draw_layers.py >/dev/null 2>&1; then
        say "  ok      assets/layers.svg redrawn"
    else
        say "  WARN    tools/draw_layers.py failed - assets/layers.svg is"
        say "          whatever it was. Run it by hand to see why."
    fi
    if ! python3 tools/refresh_readme.py; then
        say "  WARN    tools/refresh_readme.py reported a problem - the README"
        say "          may carry stale figures."
    fi
    say ""
fi

# --- checks ------------------------------------------------------------------
say "checks"
fail=0

# tracked files only, and only those still on disk: a deletion not yet
# committed is not something a push would carry as a file
bad=$(git ls-files -z -- '*.py' \
      | while IFS= read -r -d '' f; do [ -f "$f" ] && printf '%s\0' "$f"; done \
      | xargs -0 -r -n 50 python3 -m py_compile 2>&1 || true)
if [ -n "$bad" ]; then
    say "  FAIL    not every tracked .py compiles:"
    sed 's/^/            /' <<< "$bad"
    fail=1
else
    say "  ok      every tracked .py compiles"
fi

# credentials live in .env, which is gitignored; one that is tracked anyway
# would be pushed
if git ls-files | grep -Eq '(^|/)\.env(\..*)?$'; then
    say "  FAIL    a .env file is tracked:"
    git ls-files | grep -E '(^|/)\.env(\..*)?$' | sed 's/^/            /'
    fail=1
else
    say "  ok      no .env is tracked"
fi

# The README's links must lead somewhere: a relative link names a file or a
# directory of the repository, or one the program creates as it runs.
missing=$(python3 - <<'PY'
import os, re, subprocess
tracked = set(subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True).stdout.split())
dirs = {p.split("/")[0] for p in tracked if "/" in p}
runtime = {"runs", "data", "weights"}
text = open("README.md", encoding="utf-8").read()
for m in re.finditer(r"\]\(([^)\s]+)\)", text):
    link = m.group(1).split("#")[0]
    if not link or re.match(r"^[a-z]+:", link):
        continue
    path = link.rstrip("/")
    top = path.split("/")[0]
    if path in tracked or top in runtime or (path in dirs) or any(
            t.startswith(path + "/") for t in tracked):
        continue
    print(link)
PY
)
if [ -n "$missing" ]; then
    say "  WARN    the README links to paths the repository does not have:"
    sed 's/^/            /' <<< "$missing"
else
    say "  ok      every README link leads to something in the repository"
fi

say ""
say "Nothing was committed or pushed. Review and push yourself:"
say "    git status"
say "    git diff"
exit $fail
