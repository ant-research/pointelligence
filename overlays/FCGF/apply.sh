#!/usr/bin/env bash
# overlays/FCGF/apply.sh — copy add/ + apply patch/ on top of upstream-pristine Pointcept.
#
# Usage:
#   bash overlays/FCGF/apply.sh [SUBMODULE_PATH]
#
# Default SUBMODULE_PATH is examples/FCGF (relative to repo root). The
# submodule must already be checked out at the SHA recorded in
# PINNED_COMMIT.txt — run `git submodule update --init --recursive` first.
#
# Idempotent: if every patch already applies in reverse (i.e. overlay was
# already applied), the script reports that and exits 0 without re-touching
# anything. If only some patches are reversed, exits non-zero so the user
# can investigate (likely a partial-apply).
set -euo pipefail

SUBMODULE="${1:-examples/FCGF}"
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
  echo "         (b) edit overlays/FCGF/PINNED_COMMIT.txt and rebuild patches" >&2
  exit 2
fi

# Dry-run all patches before any side-effect.
for p in "$HERE"/patch/*.patch; do
  [ -f "$p" ] || continue
  if ! git -C "$SUBMODULE" apply --check "$p"; then
    echo "ERROR: $p does not apply cleanly to $SUBMODULE at $actual_sha" >&2
    exit 1
  fi
done

# Copy add/ tree.
if [ -d "$HERE/add" ]; then
  cp -r "$HERE/add/." "$SUBMODULE/"
fi

# Apply patches in lexical order.
for p in "$HERE"/patch/*.patch; do
  [ -f "$p" ] || continue
  git -C "$SUBMODULE" apply "$p"
done

# Post-patch fixups: changes that require editing already-patched files.
# These cannot be expressed as simple git-format patches over the pristine
# upstream because they layer on top of 0001's modifications.

# benchmark_3dmatch.py: PyTorch 2.6 changed torch.load default to weights_only=True.
# Also pass voxel_size=config.voxel_size so Model() uses training-time grid scale.
BENCH="$SUBMODULE/scripts/benchmark_3dmatch.py"
if [ -f "$BENCH" ]; then
  python3 -c "
path = '$BENCH'
c = open(path).read()
c = c.replace('checkpoint = torch.load(args.model)', 'checkpoint = torch.load(args.model, weights_only=False)', 1)
c = c.replace('        conv1_kernel_size=config.conv1_kernel_size,\n        D=3)', '        conv1_kernel_size=config.conv1_kernel_size,\n        D=3,\n        voxel_size=config.voxel_size)', 1)
open(path, 'w').write(c)
print('Post-patch: fixed benchmark_3dmatch.py torch.load + voxel_size')
"
fi

# Clean up any __pycache__ directories that may have been copied from add/.
find "$SUBMODULE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Overlay applied to $SUBMODULE (pinned $expected_sha)"
