#!/usr/bin/env bash
# overlays/FCGF/build.sh — produce build/FCGF/ as a sibling
# symlink-mirror with overlays applied.
#
# See overlays/Pointcept/build.sh for the full algorithm comments;
# this file is the FCGF mirror of the same logic.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SUB="FCGF"
SRC="$REPO_ROOT/examples/$SUB"
DST="$REPO_ROOT/build/$SUB"

if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC does not exist. Run `git submodule update --init` first." >&2
    exit 2
fi

expected_sha="$(awk '/^[a-f0-9]/{print $1; exit}' "$HERE/PINNED_COMMIT.txt")"
actual_sha="$(git -C "$SRC" rev-parse HEAD)"
if [ "$expected_sha" != "$actual_sha" ]; then
    echo "ERROR: $SRC is at $actual_sha, overlay pinned at $expected_sha." >&2
    exit 2
fi

rm -rf "$DST"
mkdir -p "$DST"

(cd "$DST" && cp -rs "$SRC"/. .)
rm -f "$DST/.git"

if [ -d "$HERE/add" ]; then
    while IFS= read -r f; do
        rel="${f#$HERE/add/}"
        target="$DST/$rel"
        mkdir -p "$(dirname "$target")"
        rm -f "$target"
        cp "$f" "$target"
    done < <(find "$HERE/add" -type f -not -path '*/__pycache__/*' -not -name '.gitignore')
fi

for p in "$HERE"/patch/*.patch; do
    [ -f "$p" ] || continue
    paths="$(awk '/^diff --git/{print $4}' "$p" | sed 's|^b/||')"
    for path in $paths; do
        target="$DST/$path"
        if [ -L "$target" ]; then
            tmp="$(mktemp)"
            cat "$target" > "$tmp"
            chmod --reference="$target" "$tmp" 2>/dev/null || true
            rm -f "$target"
            mv "$tmp" "$target"
        elif [ ! -e "$target" ]; then
            mkdir -p "$(dirname "$target")"
        fi
    done
    (cd "$REPO_ROOT" && git apply --directory="build/$SUB" "$p")
done

find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Built $DST (pinned $expected_sha)"
