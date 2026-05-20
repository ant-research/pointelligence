#!/usr/bin/env bash
# overlays/Pointcept/build.sh — produce build/Pointcept/ as a sibling
# symlink-mirror with overlays applied.
#
# Sibling-build re-architecture (2026-05-14, see overlays/REARCHITECTURE.md):
# Instead of copying overlay content into the submodule working tree
# (the old apply.sh model), build.sh produces an out-of-tree directory
# at <repo>/build/Pointcept/ that:
#   - symlinks every unchanged upstream file to examples/Pointcept/<path>
#   - holds a REAL copy from overlays/Pointcept/add/<path> where add/ has it
#   - holds a REAL copy of upstream with each patch applied where patch/ touches it
#
# Net effect: examples/Pointcept/ stays 100% pristine; PYTHONPATH points
# at build/Pointcept/ at runtime; all overlay behavior is preserved.
#
# Idempotent: re-running wipes + rebuilds build/Pointcept/ in ~1s.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SUB="Pointcept"
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
    echo "       Either checkout the pinned SHA or update PINNED_COMMIT.txt." >&2
    exit 2
fi

# --- 1. Wipe + symlink-mirror examples/$SUB into build/$SUB --------------

rm -rf "$DST"
mkdir -p "$DST"

# cp -rs: recursively create symlinks for files (dirs become real dirs).
# Symlink targets are absolute paths into $SRC. The mirror is bound to
# this checkout's absolute paths — re-run build.sh after any relocate.
(cd "$DST" && cp -rs "$SRC"/. .)

# Remove the symlinked .git (it's a file pointing into the parent repo's
# .git/modules/Pointcept; not relevant for runtime imports and could
# confuse pkgutil walks).
rm -f "$DST/.git"

# --- 2. Override with files from overlays/$SUB/add/ ----------------------

if [ -d "$HERE/add" ]; then
    while IFS= read -r f; do
        rel="${f#$HERE/add/}"
        target="$DST/$rel"
        mkdir -p "$(dirname "$target")"
        rm -f "$target"   # remove the symlink (or stale real file)
        cp "$f" "$target"
    done < <(find "$HERE/add" -type f -not -path '*/__pycache__/*' -not -name '.gitignore')
fi

# --- 3. Materialize patched files + apply each patch ---------------------
# For each file a patch will touch, we must replace the symlink with a
# real copy of the upstream content BEFORE applying — otherwise patch
# would modify the symlink target (i.e., the pristine submodule file).

for p in "$HERE"/patch/*.patch; do
    [ -f "$p" ] || continue

    # Extract target paths from the patch (strip "b/" prefix from
    # "diff --git a/X b/Y" lines).
    paths="$(awk '/^diff --git/{print $4}' "$p" | sed 's|^b/||')"

    for path in $paths; do
        target="$DST/$path"
        if [ -L "$target" ]; then
            # Materialize: read the symlink target's content into a real file.
            tmp="$(mktemp)"
            cat "$target" > "$tmp"
            chmod --reference="$target" "$tmp" 2>/dev/null || true
            rm -f "$target"
            mv "$tmp" "$target"
        elif [ ! -e "$target" ]; then
            # Patch creates a new file — git apply will handle it.
            mkdir -p "$(dirname "$target")"
        fi
    done

    # Apply the patch to $DST. --directory rewrites a/<path> → $DST/<path>.
    # Use relative path: git apply rejects absolute --directory as
    # "invalid path".
    (cd "$REPO_ROOT" && git apply --directory="build/$SUB" "$p")
done

# --- 4. Drop any __pycache__ that crept in --------------------------------

find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Built $DST (pinned $expected_sha)"
