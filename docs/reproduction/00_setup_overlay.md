# 00 — Submodule overlay setup

> **What this is.** The two reproduction examples — FCGF and Pointcept —
> live in `examples/FCGF` and `examples/Pointcept` as git submodules that
> track upstream **bit-for-bit** (no overlay commits). The PointCNN++
> code that adapts each example to use this repo's kernels lives entirely
> in `pointelligence/overlays/`, applied at runtime via a small script.
>
> This is the only setup step in the reproduction guide that's specific
> to this repo's PointCNN++ overlay. The rest of the reproduction guide
> (`01_environment.md` onward) treats the post-overlay submodule trees
> as canonical FCGF / Pointcept checkouts.

## Why an overlay (not a fork)

A fork of FCGF / Pointcept on a `pointcnnpp-version` branch couples
the PointCNN++ adapter with the upstream tree, which causes three
problems:

1. Any upstream sync becomes a manual merge.
2. The fork's `README.md` is replaced with PointCNN++-specific
   instructions, masking the upstream project's documentation.
3. Pointcept upstream is actively developed, so a fork branch
   long-running divergence.

The overlay model: this repo keeps the overlay (configs, model
files, modified-file patches, an apply script) in `overlays/<sub>/`.
The submodules check out upstream-pristine, and `apply.sh` installs
the overlay onto the upstream tree at the user's checkout.

## One-time setup

```bash
# 1. Clone with submodules.
git clone --recursive https://github.com/ant-research/pointelligence.git
cd pointelligence

# 2. Verify the submodules landed at the pinned upstream commits.
git -C examples/Pointcept rev-parse HEAD       # -> 96e109d... (Pointcept v1.6.0)
git -C examples/FCGF      rev-parse HEAD       # -> 0612340... (chrischoy/master)

# 3. Install pointelligence in editable mode so `internals.*` and
#    `extensions.sparse_engines_cuda` are importable from the overlay.
pip install -e .

# 4. Apply the PointCNN++ overlays.
bash overlays/Pointcept/apply.sh examples/Pointcept
bash overlays/FCGF/apply.sh examples/FCGF
```

After step 4, the submodule working trees contain upstream files **plus**
the PointCNN++ overlay (new configs, MSC_pointcnnpp model package, the
ResUNetBN2C backbone for FCGF, etc.). At this point continue with
`01_environment.md`.

## What `apply.sh` does

Each `overlays/<sub>/apply.sh` runs three phases:

1. **Pin check.** Reads the upstream SHA from
   `overlays/<sub>/PINNED_COMMIT.txt` and refuses to apply if the
   submodule isn't checked out at that exact commit. This prevents
   silent drift if a future Pointcept release advances `main`.
2. **Idempotent skip.** If every patch already applies in reverse
   (i.e. the overlay is already on this checkout), the script
   reports "already applied" and exits 0 without touching anything.
3. **Dry-run + apply.** Every patch under `patch/*.patch` is run
   through `git apply --check` first; only on a clean dry-run does
   the script then `cp -r add/.` and `git apply` for real.

The overlay layout per submodule:

```
overlays/<sub>/
├── PINNED_COMMIT.txt    # upstream SHA + tag/branch comment
├── add/                 # files copied verbatim onto the upstream tree
│   └── <same paths as the submodule's working tree>
├── patch/
│   ├── 0001-pointcnnpp-overlay.patch          # the bulk modifications
│   └── 000N-<topic>.patch                      # individually-tracked
│                                               # fixes (e.g. yapf
│                                               # fallback for Pointcept,
│                                               # scipy compat for FCGF)
└── apply.sh
```

## When the upstream submodule is updated

Two parts to update:

1. **Bump the submodule pin** in `.gitmodules` and `git submodule update --remote`.
2. **Re-cook the overlay** against the new upstream:
   - `cd examples/<sub> && git remote add upstream <upstream-url> && git fetch upstream`
   - Find the new diff: `git diff <new-pin>..origin/pointcnnpp-version -- ...`
   - Re-extract `add/` and re-generate `patch/0001-pointcnnpp-overlay.patch`.
   - Update `overlays/<sub>/PINNED_COMMIT.txt` with the new SHA.
3. Verify: `git -C examples/<sub> reset --hard <new-pin>` then
   `bash overlays/<sub>/apply.sh examples/<sub>` → should land cleanly.

If a patch *would* still apply against the new upstream (no upstream
file motion in the patched files), step 2 is just bumping the pin and
the SHA. If it doesn't, fix the rejected hunks in the patch by hand.

## When upstream merges one of our patches

Track the patches as separate files inside `patch/` so individual ones
can drop out cleanly:

- `0001-pointcnnpp-overlay.patch` is the long-tail PointCNN++ overlay
  that we expect to maintain forever.
- `0002-config-yapf-fallback.patch` (Pointcept only) is a generic
  resilience fix that we'd like upstream to merge. When/if Pointcept
  merges it, delete this patch and bump the pin.
- `0002-eval-scipy-workers-compat.patch` (FCGF only) is a scipy>=1.6
  compat shim. FCGF upstream is dormant so this likely stays
  indefinitely.

## Common failures

- **`apply.sh` reports "$SUBMODULE is at <X>, overlay was pinned to <Y>"**:
  the submodule pointer was bumped without re-cooking the overlay.
  Either roll the submodule back (`git -C examples/<sub> reset --hard <Y>`)
  or follow the "When the upstream submodule is updated" steps above.
- **`patch/000X.patch does not apply cleanly`**: someone modified the
  upstream submodule tree manually after a previous `apply.sh` run
  (or an upstream change conflicts with the patch). Reset the
  submodule (`git -C examples/<sub> reset --hard HEAD`) and retry.
- **`Overlay already applied` but you wanted a fresh apply**: that's
  the idempotent skip path. To re-apply from scratch, reset the
  submodule first: `git -C examples/<sub> reset --hard <pinned-sha>
  && git -C examples/<sub> clean -fd && bash overlays/<sub>/apply.sh
  examples/<sub>`.

## Why the upstream READMEs aren't touched

Upstream READMEs are the source of truth for the upstream projects —
especially Pointcept, which is actively developed. PointCNN++-specific
docs live in this guide instead, so that a clean clone of either
submodule still reads as the canonical upstream project.

After `apply.sh` runs, `examples/Pointcept/README.md` and
`examples/FCGF/README.md` remain upstream-pristine. The overlay
deliberately leaves those files alone.
