#!/usr/bin/env bash
# Slim clone of the assembly techstack for one area (AT 164).
#
#   bash slim-clone.sh <area> [folder] [--url <repo url>] [--ref <branch>]
#
# Needs only git and git-lfs. Partial clone (no file contents up front), sparse checkout of what
# every area needs plus the folders of <area>; LFS files come only for those folders.
# The areas and their folders are defined in tools/clone/areas.tsv, which the script reads from
# the clone itself (slim-clone.ps1 reads the same file).
set -euo pipefail

URL="https://gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/assembly-techstack.git"
REF=""
AREA=""
DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        --ref) REF="$2"; shift 2 ;;
        -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) if [ -z "$AREA" ]; then AREA="$1"; elif [ -z "$DIR" ]; then DIR="$1"; else echo "Too many arguments: $1" >&2; exit 2; fi; shift ;;
    esac
done
if [ -z "$AREA" ]; then
    echo "Usage: bash slim-clone.sh <area> [folder] [--url <repo url>] [--ref <branch>]" >&2
    echo "Run it with a name that is not an area (for example 'list') to see the valid areas." >&2
    exit 2
fi
DIR="${DIR:-assembly-techstack}"
if [ -e "$DIR" ]; then
    echo "'$DIR' exists already; pick another folder or remove it." >&2
    exit 1
fi

export GIT_LFS_SKIP_SMUDGE=1  # the clone downloads no LFS files; the checkout below does, for the area only
START=$(pwd)
branch=()
[ -n "$REF" ] && branch=(--branch "$REF")
git -c core.longpaths=true clone --filter=blob:none --no-checkout ${branch[@]+"${branch[@]}"} "$URL" "$DIR"
cd "$DIR"
git config core.longpaths true

# Only this one file is downloaded before the area is known. A branch cut before the table existed
# does not have it: take the one on main.
table=$(git show HEAD:tools/clone/areas.tsv 2>/dev/null || git show origin/main:tools/clone/areas.tsv 2>/dev/null) || {
    cd "$START"
    rm -rf "$DIR"
    echo "Neither '${REF:-the default branch}' nor main has tools/clone/areas.tsv yet. Merge main into your branch, or clone without --ref." >&2
    exit 1
}
areas=$(printf '%s\n' "$table" | awk -F'\t' '!/^#/ && NF && $1 != "area" && $1 != "*" {print "  " $1 "  -  " $2}')
row() {  # row <area> <column number>
    printf '%s\n' "$table" | awk -F'\t' -v a="$1" -v c="$2" '!/^#/ && $1 == a {print $c; exit}'
}
if ! printf '%s\n' "$areas" | awk '{print $1}' | grep -qx -- "$AREA"; then
    cd "$START"
    rm -rf "$DIR"
    echo "Unknown area '$AREA'. Valid areas:" >&2
    printf '%s\n' "$areas" >&2
    exit 1
fi

sparse="$(row '*' 3) $(row "$AREA" 3)"
tests=$(row "$AREA" 4)

# Checkout downloads the LFS files of the sparse folders only (the LFS filter runs per file, in one batch);
# a later `git pull` does the same, so there is no separate `git lfs pull`.
unset GIT_LFS_SKIP_SMUDGE
git lfs install --local
# shellcheck disable=SC2086
git sparse-checkout set --cone $sparse
git checkout

echo
echo "Area '$AREA' is in $(pwd)."
echo "Next: uv sync && uv run pre-commit install"
echo "Fast tests of your area:"
echo "  uv run pytest -q -m \"not slow and not geom and not gpu\" $tests"
