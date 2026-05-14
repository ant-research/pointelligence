# `overlays/` — submodule overlay system

Pointelligence vendors third-party research repos as git submodules under
`examples/` (e.g. `examples/Pointcept`, `examples/FCGF`). Each submodule is
pinned to a specific upstream commit. When we need to add files to or patch
those submodules — without forking them and carrying the divergence — we
use the per-submodule overlay convention here.

This directory is the **canonical** mechanism. Do not introduce parallel
overlay schemes (e.g. `pointcept_overlay/` flat layouts that older feature
branches used).

## Per-submodule layout

Each submodule overlay lives at `overlays/<SubmoduleName>/`:

```
overlays/<SubmoduleName>/
├── PINNED_COMMIT.txt    # upstream commit SHA the overlay targets
├── add/                 # new files to add over the submodule (cp -r)
├── patch/               # *.patch files to apply (lexical order)
└── apply.sh             # idempotent applier
```

Currently populated:

- `overlays/Pointcept/` — overlays for `examples/Pointcept` submodule.
- `overlays/FCGF/` — overlays for `examples/FCGF` submodule.

## How to apply

After `git submodule update --init --recursive`:

```bash
bash overlays/Pointcept/apply.sh
bash overlays/FCGF/apply.sh
```

Each `apply.sh`:

1. Verifies the submodule is at the SHA recorded in `PINNED_COMMIT.txt`
   (refuses to apply if drift detected — bump `PINNED_COMMIT.txt` and
   regenerate patches first).
2. Dry-runs every patch via `git apply --check` before any side-effect.
3. **Idempotent skip**: if every patch already applies in reverse (i.e.
   the overlay has already been applied), exits 0 without re-touching.
4. Copies `add/` over the submodule.
5. Applies `patch/*.patch` in lexical order.

## How to add a new patch

From a worktree with the submodule cleanly at the pinned commit:

```bash
# Make your edit to the submodule, test it, then:
cd examples/Pointcept
git diff > /tmp/myfix.patch                # or: git format-patch HEAD~1 -1 --stdout
cd ../..
mv /tmp/myfix.patch overlays/Pointcept/patch/0099-my-fix.patch
# Reset the submodule to clean state:
git -C examples/Pointcept reset --hard
# Re-apply via apply.sh to verify:
bash overlays/Pointcept/apply.sh
```

Patches are applied in **lexical order** of filename — prefix with a
zero-padded numeric (`0001-`, `0002-`, ...) to control ordering when
order matters.

## How to add files via `add/`

For files that don't conflict with anything upstream (new configs, new
modules), put them under `overlays/<SubmoduleName>/add/<relative path>`.
At apply time they're copied over the submodule with `cp -r`.

Example: `overlays/Pointcept/add/configs/nuscenes/semseg-pointcnnpp-base.py`
gets copied to `examples/Pointcept/configs/nuscenes/semseg-pointcnnpp-base.py`.

## How to bump the pin

When upstream lands changes you want to track:

1. Update the submodule pin: `git -C examples/Pointcept fetch && git -C examples/Pointcept checkout <new-sha>`.
2. Update `overlays/<SubmoduleName>/PINNED_COMMIT.txt` to match.
3. Regenerate any `patch/*.patch` files that need updating against the new SHA.
4. Run `bash overlays/<SubmoduleName>/apply.sh` to verify clean apply.
5. Run `scripts/mirror_submodules.sh` to refresh the cluster's OSS mirror
   (per CLAUDE.md "Submodule pin bump → mirror refresh required" rule;
   `cluster_setup.sh` silently swallows submodule init failures, so
   mirror staleness causes confusing cluster-side fails).
6. Commit the bump as a single PR (don't bundle pin bumps with feature work).

## When NOT to use overlays

- For NEW features that don't depend on the submodule's internal source —
  put them in pointelligence proper (`sparse_engines/`, `layers/`,
  `models/`, etc.). Overlays are only for files that need to live INSIDE
  the submodule's tree to be picked up at runtime.
- For PNT-specific architecture configs that only live on a specific
  research branch — those go in the branch's own overlay extension, not
  in main's `overlays/`.

## Branch convention

`main`'s `overlays/Pointcept/` + `overlays/FCGF/` are the **shared baseline**
that every code branch (conv_extreme, attn_extreme, architectures,
ptv3_and_sota) inherits via branch-from-main. Branches may EXTEND with
their own patches in their own subtrees of `overlays/`, but should not
DELETE or REPLACE main's overlays without a `fix/<topic>` PR onto main.

**Concretely, on a research branch (e.g. `threads/architectures`) you MAY:**

- Add new files under `overlays/<SubmoduleName>/add/<rel-path>/` as long as
  they don't collide with files modified by any `patch/`.
- Add new patches under `overlays/<SubmoduleName>/patch/0099-<my-topic>.patch`
  (use a numeric prefix above existing patches so lexical apply-order is preserved).
- Update `overlays/<SubmoduleName>/PINNED_COMMIT.txt` IF you simultaneously
  regenerate every patch against the new pin and re-run `apply.sh` to verify.

**You MUST NOT, on a research branch:**

- Modify `overlays/<SubmoduleName>/apply.sh` (this file is shared infra;
  changes go through a `fix/<topic>` PR onto `main`).
- Modify existing patches `0001-…` through `0005-…` (same reason).
- Add an `add/<path>` file when any patch already modifies the same `<path>`
  (this triggers the collision check in `scripts/validate_overlays.sh` —
  it is an ERROR, not a warning).

## Common pitfalls (case studies — DO NOT repeat)

These are real anti-patterns that have shipped to research branches and
caused cascade-failure debugging cycles. Each is explicitly blocked by
`scripts/validate_overlays.sh` and/or the branch convention above.

### Pitfall #1 — Double-modify a file via both `add/` and `patch/`

**Symptom**: `apply.sh` runs the `cp -r add/` step first, then `git apply
<patch>` second. If `<patch>` modifies the same file that `add/` just
copied in, the patch's context lines (which were generated against the
*upstream* file) won't match the *overridden* file, and the patch fails
with confusing whitespace / line-number errors.

**Concrete example (architectures branch, 2026-05-10)**: an agent added
`overlays/Pointcept/add/pointcept/engines/train.py` as a defensive AMP-fix
override, not realizing that `overlays/Pointcept/patch/0003-engine-infra.patch`
already modified the same file with an equivalent guard. The `cp -r add/`
put the override in place; the subsequent `git apply 0003-engine-infra.patch`
failed because the override didn't have the patch's context lines. The
agent then patched `apply.sh` to add `--ignore-space-change` + verbose-debug,
then added a *second* `cp` at the end of `apply.sh` to re-copy the override
*after* patches — both workarounds layered on top of the original mistake.
Net: 9 fix commits, multiple cluster-job submission failures, days lost.

**Why it's wrong**: `add/` and `patch/` are mutually exclusive channels
for modifying any given file. The validator enforces this — see
`scripts/validate_overlays.sh` Check 3.

**Correct fix**: pick ONE channel. If the modification is a small diff
against upstream, put it in a `patch/`. If you're adding a new file (no
upstream counterpart), put it in `add/`. Never both.

### Pitfall #2 — Modify `apply.sh` to work around something

**Symptom**: `apply.sh` on a research branch diverges from `main`'s
`apply.sh` blob — typically additions like `--ignore-space-change`,
verbose-debug, or "second-copy after patches" hooks. These are almost
always workarounds for an actual problem elsewhere (often Pitfall #1).

**Why it's wrong**: `apply.sh` is shared infrastructure. Modifying it
on a research branch means (a) the cluster runs a different `apply.sh`
on different branches — different overlay-apply semantics is a
significant cross-branch consistency hazard, and (b) when the underlying
problem is fixed, the research-branch `apply.sh` workarounds become
dead code that future agents don't dare remove for fear of regression.

**Correct fix**: revert `apply.sh` to match `main`'s blob (use
`git show origin/main:overlays/<Submod>/apply.sh > overlays/<Submod>/apply.sh`).
If `apply.sh` itself genuinely needs changing, open a `fix/<topic>` PR
onto `main` — the new behavior propagates to every research branch on
its next sync from `main`.

### Pitfall #3 — "Defensive double override" of the same modification

**Symptom**: the same intended modification appears in multiple
locations — e.g., a patch in `patch/0003-…` AND a file in `add/`, OR a
patch in `patch/0003-…` AND a runtime check in cluster setup code.

**Why it's wrong**: two sources-of-truth for the same modification
never stay in sync. When the modification needs to change (e.g., the
underlying bug evolves), one location gets updated and the other
silently drifts. Bugs compound. Pitfall #1 is one specific instance of
this; the general rule extends to any duplicated guard.

**Correct fix**: choose ONE channel, document why, delete the other.
If you're tempted to "keep both as defensive belt-and-suspenders", the
right move is instead to make the validator (`validate_overlays.sh`)
explicitly check the property you'd want the defensive copy to enforce.

## How to verify your overlay change

Before pushing any branch that touches `overlays/`:

```bash
bash scripts/validate_overlays.sh
```

The validator checks (per `scripts/validate_overlays.sh`):

1. Patches apply cleanly (dry-run) on a fresh submodule checkout.
2. No `__pycache__` directories in `add/`.
3. **No file collisions between `add/` and `patch/` (this is an ERROR,
   not a warning — see Pitfall #1).**
4. `PINNED_COMMIT.txt` matches across all worktrees for the same submodule.
5. **`apply.sh` blob equals `main`'s `apply.sh` blob** (on research
   branches; see Pitfall #2). If you're on a `fix/<topic>` branch
   actually modifying `apply.sh`, the check skips itself.

Pass = green to push. Fail = fix the underlying issue, not the validator.
