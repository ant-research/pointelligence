#!/usr/bin/env bash
# overlays/Pointcept/apply.sh — copy add/ + apply patch/ on top of upstream-pristine Pointcept.
#
# Usage:
#   bash overlays/Pointcept/apply.sh [--force] [SUBMODULE_PATH]
#
# Default SUBMODULE_PATH is examples/Pointcept (relative to repo root). The
# submodule must already be checked out at the SHA recorded in
# PINNED_COMMIT.txt — run `git submodule update --init --recursive` first.
#
# --force: reset the submodule to the pinned commit before applying, discarding
#          any local modifications (e.g. from a previous overlay that wasn't
#          cleaned up). Without --force, a dirty submodule causes an error.
#
# Idempotent: if every patch already applies in reverse (i.e. overlay was
# already applied), the script reports that and exits 0 without re-touching
# anything. If only some patches are reversed, exits non-zero so the user
# can investigate (likely a partial-apply).
set -euo pipefail

FORCE=false
if [ "${1:-}" = "--force" ]; then
  FORCE=true
  shift
fi

SUBMODULE="${1:-examples/Pointcept}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$SUBMODULE/.git" ] && [ ! -f "$SUBMODULE/.git" ]; then
  echo "ERROR: $SUBMODULE is not an initialized git submodule" >&2
  echo "       Run: git submodule update --init --recursive" >&2
  exit 2
fi

expected_sha="$(awk '/^[a-f0-9]/{print $1; exit}' "$HERE/PINNED_COMMIT.txt")"
actual_sha="$(git -C "$SUBMODULE" rev-parse HEAD)"

# Idempotent skip: if every patch is already applied (reverse-check passes
# for ALL patches), exit cleanly with a message.
already_applied=true
for p in "$HERE"/patch/*.patch; do
  [ -f "$p" ] || continue
  if ! git -C "$SUBMODULE" apply --reverse --check "$p" >/dev/null 2>&1; then
    already_applied=false
    break
  fi
done
if [ "$already_applied" = true ] && ls "$HERE"/patch/*.patch >/dev/null 2>&1; then
  echo "Overlay already applied to $SUBMODULE — nothing to do."
  exit 0
fi

# Refuse to apply if submodule is at the wrong upstream pin.
if [ "$expected_sha" != "$actual_sha" ]; then
  echo "ERROR: $SUBMODULE is at $actual_sha, overlay was pinned to $expected_sha" >&2
  echo "       Either:" >&2
  echo "         (a) git submodule update --init --recursive   (lands the pinned SHA)" >&2
  echo "         (b) edit overlays/Pointcept/PINNED_COMMIT.txt and rebuild patches" >&2
  exit 2
fi

# If --force, reset the submodule to pristine state before applying.
# This handles the common case where a previous overlay left the submodule dirty.
if [ "$FORCE" = true ]; then
  git -C "$SUBMODULE" checkout -f HEAD 2>/dev/null || true
  git -C "$SUBMODULE" clean -fd 2>/dev/null || true
fi

# If the submodule is dirty (and not --force), warn and exit.
# .gitmodules often has ignore=dirty, so git submodule update won't fix this.
dirty_files="$(git -C "$SUBMODULE" diff --name-only HEAD 2>/dev/null || true)"
if [ -n "$dirty_files" ] && [ "$FORCE" = false ]; then
  echo "WARNING: $SUBMODULE has local modifications (leftover from a previous overlay?)." >&2
  echo "         Re-run with --force to reset and re-apply." >&2
  echo "         Dirty files: $dirty_files" >&2
  exit 1
fi
if [ -n "$dirty_files" ] && [ "$FORCE" = true ]; then
  echo "Resetting dirty submodule $SUBMODULE to pristine state..."
  for f in $dirty_files; do
    git -C "$SUBMODULE" show "HEAD:$f" > "$SUBMODULE/$f" 2>/dev/null || true
  done
  git -C "$SUBMODULE" clean -fd 2>/dev/null || true
fi

# Dry-run: apply patches sequentially to validate the full chain.
# Later patches may depend on earlier ones (e.g. 0006 modifies train.py
# after 0003 has already changed it), so we must validate each patch
# against the accumulated state, not the pristine tree.
ALL_OK=true
for p in "$HERE"/patch/*.patch; do
  [ -f "$p" ] || continue
  if ! git -C "$SUBMODULE" apply --check "$p" 2>/dev/null; then
    ALL_OK=false
    echo "ERROR: $(basename "$p") does not apply cleanly (after previous patches)" >&2
    break
  fi
  # Apply the patch to accumulate state for the next dry-run check.
  git -C "$SUBMODULE" apply "$p" 2>/dev/null || { ALL_OK=false; echo "ERROR: $(basename "$p") failed during sequential dry-run" >&2; break; }
done
# Roll back all patches applied during dry-run.
git -C "$SUBMODULE" checkout -f HEAD 2>/dev/null || true
git -C "$SUBMODULE" clean -fd 2>/dev/null || true
if [ "$ALL_OK" = false ]; then
  exit 1
fi

# Copy add/ tree.
if [ -d "$HERE/add" ]; then
  cp -r "$HERE/add/." "$SUBMODULE/"
fi

# Apply patches in lexical order.
for p in "$HERE"/patch/*.patch; do
  [ -f "$p" ] || continue
  git -C "$SUBMODULE" apply "$p"
done

# Clean up any __pycache__ directories that may have been copied from add/.
find "$SUBMODULE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Overlay applied to $SUBMODULE (pinned $expected_sha)"