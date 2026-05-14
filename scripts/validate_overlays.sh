#!/usr/bin/env bash
# scripts/validate_overlays.sh — validate overlay integrity across worktrees.
#
# Checks:
#   1. Each overlay's patches apply cleanly (dry-run) on a fresh submodule checkout
#   2. No __pycache__ directories in add/
#   3. No file in add/ collides with a file that a patch also modifies
#      (this is an ERROR — see overlays/README.md "Pitfall #1")
#   4. PINNED_COMMIT.txt matches across all worktrees for the same submodule
#   5. apply.sh blob equals main's apply.sh blob — research branches must
#      not diverge their apply.sh from main; if you genuinely need to
#      change apply.sh, do it via a fix/<topic> PR onto main, not a
#      research-branch edit (see overlays/README.md "Pitfall #2"). This
#      check is bypassed on branches whose name starts with `fix/` or `main`.
#
# Usage:
#   bash scripts/validate_overlays.sh [WORKTREE_ROOT]
#
# Default WORKTREE_ROOT is the repo root (detected from this script's location).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Detect repo root. Two layouts:
#   - Multi-worktree: $REPO_ROOT/worktrees/{main,threads/*}/ (developer machine).
#   - Single-clone:   $REPO_ROOT/ directly contains overlays/ + scripts/ (CI runner, fresh clone).
# Walk up from script location. Prefer a parent containing worktrees/;
# fall back to the nearest ancestor that contains both overlays/ and
# scripts/ (single-clone case). Never escape the actual repo to `/`.
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"   # parent of scripts/ = repo root in single-clone layout
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ]; do
    if [ -d "$REPO_ROOT/worktrees" ]; then
        break
    fi
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
done
if [ "$REPO_ROOT" = "/" ]; then
    # No worktrees/ found anywhere above — assume single-clone layout, use the
    # script's own repo root (one dir up from scripts/).
    REPO_ROOT="$DEFAULT_REPO_ROOT"
fi
REPO_ROOT="${1:-$REPO_ROOT}"

PASS=0
FAIL=0
WARN=0

pass() { PASS=$((PASS + 1)); echo "  [PASS] $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  [FAIL] $1"; }
warn() { WARN=$((WARN + 1)); echo "  [WARN] $1"; }

# Discover worktrees with overlays.
# Multi-worktree layout (developer machines): repo_root/worktrees/{main,threads/*}/
# Single-clone layout (CI runners, fresh clones): repo_root/ directly contains overlays/
WORKTREES=()
for wt_dir in "$REPO_ROOT"/worktrees/main/ "$REPO_ROOT"/worktrees/threads/*/; do
    [ -d "$wt_dir" ] || continue
    [ -d "$wt_dir/overlays" ] || continue
    WORKTREES+=("$wt_dir")
done

# Fallback: if no multi-worktree layout, check $REPO_ROOT itself (single-clone case).
if [ ${#WORKTREES[@]} -eq 0 ] && [ -d "$REPO_ROOT/overlays" ]; then
    WORKTREES+=("$REPO_ROOT/")
fi

if [ ${#WORKTREES[@]} -eq 0 ]; then
    echo "No worktrees with overlays found (looked at $REPO_ROOT/worktrees/* and $REPO_ROOT)."
    exit 0
fi

echo "=== Overlay Validation ==="
echo "Worktrees to check: ${#WORKTREES[@]}"
echo ""

# --- Check 4: PINNED_COMMIT consistency ---
echo "--- PINNED_COMMIT.txt consistency ---"
declare -A PINNED_COMMITS
for wt in "${WORKTREES[@]}"; do
    branch="$(git -C "$wt" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    for pin_file in "$wt"/overlays/*/PINNED_COMMIT.txt; do
        [ -f "$pin_file" ] || continue
        submod="$(basename "$(dirname "$pin_file")")"
        sha="$(awk '/^[a-f0-9]/{print $1; exit}' "$pin_file")"
        key="${submod}:${sha}"
        PINNED_COMMITS["$key"]+=" $branch"
    done
done

# Check if any submodule has multiple different pins
declare -A SUBMOD_PINS
for key in "${!PINNED_COMMITS[@]}"; do
    submod="${key%%:*}"
    sha="${key#*:}"
    SUBMOD_PINS["$submod"]+="$sha "
done

for submod in "${!SUBMOD_PINS[@]}"; do
    shas="$(echo "${SUBMOD_PINS[$submod]}" | tr ' ' '\n' | sort -u | tr '\n' ' ')"
    unique_count="$(echo "$shas" | tr ' ' '\n' | grep -c . || true)"
    if [ "$unique_count" -gt 1 ]; then
        fail "$submod has multiple pinned commits: $shas"
    else
        pass "$submod has consistent pinned commit across all worktrees"
    fi
done
echo ""

# Resolve the effective branch name. In CI the checkout is typically detached
# HEAD, so `git rev-parse --abbrev-ref HEAD` returns the literal "HEAD". We
# need the real branch the PR is targeting, otherwise Check 5 fails open.
# Order: CI env vars → git name-rev → fallback "unknown".
resolve_branch() {
    local wt="$1"
    local b=""
    # GitHub Actions PR
    [ -n "${GITHUB_HEAD_REF:-}" ] && b="$GITHUB_HEAD_REF"
    # GitHub Actions push
    [ -z "$b" ] && [ -n "${GITHUB_REF_NAME:-}" ] && b="$GITHUB_REF_NAME"
    # GitLab CI
    [ -z "$b" ] && [ -n "${CI_COMMIT_BRANCH:-}" ] && b="$CI_COMMIT_BRANCH"
    [ -z "$b" ] && [ -n "${CI_MERGE_REQUEST_SOURCE_BRANCH_NAME:-}" ] && b="$CI_MERGE_REQUEST_SOURCE_BRANCH_NAME"
    # Fall back to git
    [ -z "$b" ] && b="$(git -C "$wt" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
    # If we got literal HEAD (detached), try name-rev as best-effort
    if [ "$b" = "HEAD" ] || [ -z "$b" ]; then
        b="$(git -C "$wt" name-rev --name-only HEAD 2>/dev/null | sed -e 's|^remotes/origin/||' -e 's/~.*//' -e 's/\^.*//' || echo '')"
    fi
    [ -z "$b" ] && b="unknown"
    echo "$b"
}

# --- Per-worktree checks ---
for wt in "${WORKTREES[@]}"; do
    branch="$(resolve_branch "$wt")"
    echo "--- $branch ---"

    for overlay_dir in "$wt"/overlays/*/; do
        [ -d "$overlay_dir" ] || continue
        submod="$(basename "$overlay_dir")"
        echo "  Overlay: $submod"

        # --- Check 2: No __pycache__ in add/ ---
        pycache_count="$(find "$overlay_dir/add/" -name "__pycache__" -type d 2>/dev/null | wc -l || echo 0)"
        if [ "$pycache_count" -gt 0 ]; then
            fail "$submod/add/ contains __pycache__ directories ($pycache_count found)"
        else
            pass "$submod/add/ has no __pycache__"
        fi

        # --- Check 3: No add/ file collides with patched files ---
        if [ -d "$overlay_dir/add" ] && ls "$overlay_dir"/patch/*.patch >/dev/null 2>&1; then
            # Extract patched-file paths from `--- a/foo` (git default) or `--- foo`
            # (`--no-prefix` patches). Handle both via awk. /dev/null fallback in case
            # the patch starts with something else.
            patched_files="$(awk '/^--- / {sub(/^--- /, ""); sub(/^a\//, ""); if ($0 != "/dev/null") print}' "$overlay_dir"/patch/*.patch | sort -u)"
            collision=0
            while IFS= read -r patched; do
                [ -z "$patched" ] && continue
                if [ -f "$overlay_dir/add/$patched" ]; then
                    fail "$submod: add/$patched also modified by a patch (collision — use a patch, not add/; see overlays/README.md Pitfall #1)"
                    collision=$((collision + 1))
                fi
            done <<< "$patched_files"
            if [ "$collision" -eq 0 ]; then
                pass "$submod: no add/ vs patch/ file collisions"
            fi
        fi

        # --- Check 5: apply.sh blob equals main's apply.sh blob ---
        # On research branches, apply.sh is frozen. To modify apply.sh, open
        # a fix/<topic> branch off main, land via PR, let it propagate.
        # Policy: fail closed. fix/* and main are the ONLY exempt branch
        # patterns. HEAD/unknown (CI detached, ambiguous) are NOT exempt —
        # the check still runs; in those cases the resolve_branch helper
        # above tries CI env vars first, so HEAD typically only appears when
        # we genuinely couldn't determine the branch.
        case "$branch" in
            fix/*|main)
                pass "$submod: apply.sh check skipped on branch '$branch' (allowed to modify)"
                ;;
            *)
                if [ -f "$overlay_dir/apply.sh" ]; then
                    # Get main's apply.sh blob via git rev-parse (works without a main worktree present).
                    main_apply_blob="$(git -C "$wt" rev-parse "origin/main:overlays/$submod/apply.sh" 2>/dev/null || echo '')"
                    local_apply_blob="$(git -C "$wt" hash-object "$overlay_dir/apply.sh" 2>/dev/null || echo '')"
                    if [ -z "$main_apply_blob" ]; then
                        # Fail-closed: a validator that depends on origin/main must require it.
                        # CI runners often do shallow fetch (fetch-depth: 1) and need an explicit
                        # `git fetch origin main` step before running this validator.
                        fail "$submod: cannot resolve origin/main:overlays/$submod/apply.sh — run 'git fetch origin main' first (CI runners typically need this; see overlays/README.md Pitfall #2)"
                    elif [ "$main_apply_blob" = "$local_apply_blob" ]; then
                        pass "$submod: apply.sh matches main blob ($main_apply_blob)"
                    else
                        fail "$submod: apply.sh diverges from main on branch '$branch' (local=$local_apply_blob, main=$main_apply_blob) — see overlays/README.md Pitfall #2"
                    fi
                fi
                ;;
        esac

        # --- Check 1: Patches apply cleanly (dry-run) ---
        submod_path="${wt}examples/${submod}"
        if [ ! -d "$submod_path/.git" ] && [ ! -f "$submod_path/.git" ]; then
            warn "$submod submodule not initialized in $branch, skipping dry-run"
            continue
        fi

        # Save current state
        was_dirty="$(git -C "$submod_path" status --porcelain | head -1)"

        if [ -n "$was_dirty" ]; then
            # Submodule has uncommitted changes (overlay already applied).
            # Try reverse-check — must iterate in REVERSE lexical order because
            # if patch N depends on patch N-1's context, reverse-applying patch
            # N-1 first would leave patch N's context broken. Reverse-apply
            # patch N first, then N-1, etc.
            all_reversed=true
            # Use a null-delimited collection to be safe against filenames with spaces.
            patch_list_rev=()
            while IFS= read -r -d '' p; do
                patch_list_rev+=("$p")
            done < <(find "$overlay_dir/patch" -maxdepth 1 -name '*.patch' -print0 2>/dev/null | sort -rz)
            for p in "${patch_list_rev[@]}"; do
                [ -f "$p" ] || continue
                if ! git -C "$submod_path" apply --reverse --check "$p" >/dev/null 2>&1; then
                    all_reversed=false
                    break
                fi
            done
            if [ "$all_reversed" = true ]; then
                pass "$submod: all patches currently applied (reverse-check OK)"
            else
                warn "$submod: patches partially applied, cannot verify dry-run"
            fi
        else
            # Submodule is clean — try dry-run
            all_ok=true
            for p in "$overlay_dir"/patch/*.patch; do
                [ -f "$p" ] || continue
                if ! git -C "$submod_path" apply --check "$p" >/dev/null 2>&1; then
                    fail "$submod: $(basename "$p") does not apply cleanly"
                    all_ok=false
                fi
            done
            if [ "$all_ok" = true ]; then
                pass "$submod: all patches apply cleanly (dry-run)"
            fi
        fi
    done
    echo ""
done

# --- Summary ---
echo "=== Summary ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Warnings: $WARN"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "VALIDATION FAILED — $FAIL check(s) did not pass."
    exit 1
else
    echo "All checks passed."
    exit 0
fi