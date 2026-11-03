#!/usr/bin/env bash
#
# Publish the weights to Hugging Face as the undertrained snapshot.
#
# The run keeps training in weights/ here. The Hugging Face repository is a
# git clone of its own: mini-AGI-undertrained beside this repository, or
# wherever DEST says. This script mirrors weights/ into it and pushes. Stop
# the run first - the script refuses otherwise.
#
#   1. checks: no train.py or serve.py running (both write weights/), no
#      half-written file, every expert the manifest lists is present and
#      nothing else is, room on the disk for git-lfs's own copy
#   2. mirrors weights/ into DEST/weights with hard links: instant, no extra
#      disk, and an expert deleted here disappears there. Every save writes a
#      new file and renames it into place, so a run restarted after this step
#      never touches the files being published
#   3. writes the date and how much the model has read into the model card
#   4. in DEST: git add -A, commit, push - git-lfs shows the upload progress
#   5. only once the push has succeeded: the date goes into README.md here,
#      between the auto:weights-date markers, and `git lfs prune` drops the
#      previous snapshot from DEST's local store (~10 GB each)
#
# A failed push can simply be run again: nothing new gets committed, and the
# commit already made is pushed.
#
#     ./publish_weights.sh             # publish
#     ./publish_weights.sh --dry-run   # the checks, and what it would do
#
# Log in once with `hf auth login`; the push uses that login. Afterwards
# README.md here has the new date: commit it and push.
#
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$(dirname "$SRC")/mini-AGI-undertrained}"
W="$SRC/weights"
README="$SRC/README.md"
CARD="$DEST/README.md"
DRY=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

say()  { printf '%s\n' "$*"; }
stop() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
gb()   { awk -v k="$1" 'BEGIN { printf "%.1f", k / 1048576 }'; }
count() { sed '/^$/d' | wc -l; }
# what a running trainer leaves around stops a publish; a dry run, which is
# meant to be safe at any time, only reports it
problem() { [ "$DRY" = 1 ] || stop "$*"; say "  WARN    $*"; }

# put FILE NAME TEXT - replace what is between <!-- auto:NAME --> and
# <!-- /auto:NAME -->. A marker pair on one line takes TEXT inline; a pair on
# lines of their own keeps TEXT on a line between them.
put() {
    python3 - "$@" <<'PY'
import os, sys
path, name, text = sys.argv[1:4]
a, b = f"<!-- auto:{name} -->", f"<!-- /auto:{name} -->"
s = open(path, encoding="utf-8").read()
i, j = s.find(a), s.find(b)
if s.count(a) != 1 or s.count(b) != 1 or j < i:
    sys.exit(f"STOP: {path} needs exactly one {a} ... {b} pair")
inner = s[i + len(a):j]
s = s[:i + len(a)] + (f"\n{text}\n" if "\n" in inner else text) + s[j:]
with open(path + ".tmp", "w", encoding="utf-8") as f:
    f.write(s)
os.replace(path + ".tmp", path)
PY
}

# --- 1. checks ----------------------------------------------------------------
say "checks"

# Anything that writes weights/: the trainer, and serve.py, which learns from
# conversations and saves by default. Matched on each process's own
# arguments - /proc/*/cmdline holds them NUL-separated - so a shell that only
# quotes the command line does not count.
writers=""
for p in /proc/[0-9]*; do
    # a process can end while this loop looks at it
    c=$( { tr '\0' ' ' < "$p/cmdline"; } 2>/dev/null ) || continue
    case "$c" in
        python*[\ /]train.py\ read\ *|*/python*[\ /]train.py\ read\ *) ;;
        python*[\ /]serve.py*|*/python*[\ /]serve.py*) ;;
        *) continue ;;
    esac
    writers+=$'\n'"            ${p#/proc/}  ${c:0:100}"
done
if [ -n "$writers" ]; then
    problem "stop these first - they write weights/:$writers"
else
    say "  ok      nothing is training or serving"
fi

[ -f "$W/manifest.json" ] || stop "no $W/manifest.json"
half=$(find "$W" -name '*.tmp*' -print -quit)
if [ -n "$half" ]; then
    problem "a half-written file is in weights/: ${half#"$SRC"/} - a save is
            under way, or one was interrupted (then start the run and stop
            it cleanly)"
fi
if ! odd=$(python3 - "$W" <<'PY'
import json, os, sys
w = sys.argv[1]
m = json.load(open(os.path.join(w, "manifest.json")))
listed = {e["file"] for e in m["experts"]}
there = {f for f in os.listdir(os.path.join(w, "experts")) if ".tmp" not in f}
missing = sorted(listed - there) + [f for f in ("core.npz", "routers.npz", "optim.npz")
                                    if not os.path.exists(os.path.join(w, f))]
extra = sorted(there - listed)
if missing or extra:
    print(f"weights/ does not match its manifest - missing {missing[:5]}, "
          f"not in it {extra[:5]}")
    sys.exit(1)
PY
); then
    problem "$odd"
else
    say "  ok      weights/ is whole: every file its manifest lists, and nothing else"
fi

[ -d "$DEST/.git" ] \
    || stop "$DEST is not a git clone of the Hugging Face repository; set DEST to yours"
git -C "$DEST" remote get-url origin 2>/dev/null | grep -q 'huggingface\.co/' \
    || stop "the origin of $DEST is not on huggingface.co"
[ "$(realpath -m "$DEST/weights")" != "$(realpath "$W")" ] \
    || stop "$DEST/weights is weights/ itself - mirroring would delete it"
[ "$(stat -c %d "$W")" = "$(stat -c %d "$DEST")" ] \
    || stop "weights/ and $DEST are on different disks; hard links need the same one"
say "  ok      $DEST is the Hugging Face clone, on the same disk"

# The push authenticates with the login `hf auth login` stored. A browser
# login never hands its token to git, so it is passed to git for the push
# only, by a credential helper that asks huggingface_hub for it - nothing is
# written to git's config and the token is never printed.
HF_GIT_USER=$(python3 -c 'from huggingface_hub import get_token, whoami
t = get_token()
print(whoami(t)["name"] if t else "")' 2>/dev/null) || HF_GIT_USER=""
[ -n "$HF_GIT_USER" ] \
    || stop "no working Hugging Face login: run hf auth login (or check the network)"
export HF_GIT_USER
say "  ok      logged in to Hugging Face as $HF_GIT_USER"

grep -q '<!-- auto:weights-date -->' "$README" \
    || stop "README.md has no <!-- auto:weights-date --> marker for the date"
grep -q '<!-- auto:snapshot -->' "$CARD" 2>/dev/null \
    || stop "$CARD has no <!-- auto:snapshot --> marker"
say "  ok      date markers in README.md and the model card"

# git-lfs keeps its own copy of every file it commits, in DEST/.git/lfs, and
# the run needs room to keep saving
need=$(du -sk "$W" | cut -f1)
free=$(df -Pk "$DEST" | awk 'NR == 2 { print $4 }')
spare=$((5 * 1024 * 1024))
[ "$free" -gt $((need + spare)) ] \
    || stop "git-lfs needs $(gb "$need") GB for its copy and 5 GB should stay free; $(gb "$free") GB free"
say "  ok      $(gb "$need") GB to publish, $(gb "$free") GB free"

# --- what this snapshot is ------------------------------------------------------
DATE=$(LC_ALL=C date +'%-d %b %Y')
{ read -r LINE; read -r MSG; } < <(python3 - "$W" "$SRC/runs/samples.txt" "$DATE" <<'PY'
import json, re, sys
w, log, date = sys.argv[1:]
m = json.load(open(w + "/manifest.json"))
chars, n = m["read_chars"], m["n_experts"]
part = ""
try:
    # the corpus size, from the newest "...M of 7,880M characters" in the log
    with open(log, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200_000))
        total = re.findall(r"M of ([\d,]+)M characters", f.read().decode("utf-8", "ignore"))
    if total:
        frac = chars / (float(total[-1].replace(",", "")) * 1e6)
        if frac < 1:
            part = f" ({frac:.0%} of the first pass)"
except OSError:
    pass
read = f"{chars / 1e6:,.0f}M characters read{part}"
print(f"**Snapshot:** {date} · {read} · {n} experts")
print(f"snapshot {date}: {read}, {n} experts")
PY
)

old=$(ls "$DEST/weights/experts" 2>/dev/null | sort || true)
new=$(ls "$W/experts" | sort)
n_add=$(comm -13 <(printf '%s\n' "$old") <(printf '%s\n' "$new") | count)
n_del=$(comm -23 <(printf '%s\n' "$old") <(printf '%s\n' "$new") | count)
n_keep=$(comm -12 <(printf '%s\n' "$old") <(printf '%s\n' "$new") | count)

if [ "$DRY" = 1 ]; then
    say ""
    say "would publish: $MSG"
    say "  experts     $n_add new, $n_del removed, $n_keep updated"
    say "  model card  ${LINE//\*\*/}"
    say "  README.md   date -> $DATE, once the push has succeeded"
    say ""
    say "dry run - nothing changed"
    exit 0
fi

# --- 2. mirror ----------------------------------------------------------------
say ""
say "mirroring weights/ into $DEST/weights"
rm -rf -- "$DEST/weights"
cp -al -- "$W" "$DEST/weights"
say "  ok      $n_add experts new, $n_del removed, $n_keep updated (hard links - nothing copied)"

# --- 3. model card ------------------------------------------------------------
put "$CARD" snapshot "$LINE"
say "  ok      model card: ${LINE//\*\*/}"

# --- 4. commit and push -------------------------------------------------------
cd "$DEST"
say ""
say "committing - git-lfs hashes all $(gb "$need") GB first, which takes a few minutes"
git add -A
if git diff --cached --quiet; then
    say "  nothing new since the last commit"
else
    git commit -q -m "$MSG"
    say "  ok      $(git log -1 --format='%h %s')"
fi
say ""
say "pushing to $(git remote get-url origin)"
read -r -d '' HF_HELPER <<'SH' || true
!f() { test "$1" = get || return 0; cat >/dev/null; echo "username=$HF_GIT_USER"; printf 'password=%s\n' "$(python3 -c 'from huggingface_hub import get_token; print(get_token() or "")')"; }; f
SH
# never a password prompt: if the login is refused, the push fails and says so
env -u GIT_ASKPASS -u SSH_ASKPASS GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= -c credential.helper="$HF_HELPER" push

# --- 5. after a successful push -----------------------------------------------
say ""
put "$README" weights-date "$DATE"
say "  ok      README.md here now says the snapshot is from $DATE"
git lfs prune --recent
say "  ok      older snapshots pruned from $DEST/.git/lfs"
say ""
say "published. README.md has the new date: commit it and push."
