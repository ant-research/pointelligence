# `overlays/` — submodule overlay system

Pointelligence vendors third-party research repos as git submodules under
`examples/` (e.g. `examples/Pointcept`, `examples/FCGF`). Each submodule is
pinned to a specific upstream commit. When we need to add files to or patch
those submodules — without forking them and carrying the divergence — we
use the per-submodule overlay convention here.

This directory is the **canonical** mechanism. Do not introduce parallel
overlay schemes (e.g. `pointcept_overlay/` flat layouts that older feature
branches used).

## Principles

These five rules are the contract. Everything below — layout, workflows,
pitfalls — derives from them.

1. **Single source of truth per file.** A given path under `examples/<sub>/`
   is modified either by an entry in `overlays/<sub>/add/` OR by a hunk in
   one of `overlays/<sub>/patch/*.patch`. Never both. (See Pitfall #1.)

2. **No rogue content in the submodule working tree.** Every non-pristine
   file in `examples/<sub>/` must be registered in `overlays/<sub>/add/`
   (or produced by a patch). Under the sibling-build model (2026-05-14
   onward), `examples/<sub>/` stays pristine and `build.sh` produces
   output at `build/<sub>/` — `build/<sub>/` is fully regenerable from
   the source overlay tree, so there is no data-loss class to guard
   against. Pitfall #4 is historical, retained as a case study from
   the pre-migration `apply.sh` model.

3. **Canonical infrastructure scripts.** The blobs of `build.sh`,
   `scripts/validate_overlays.sh`, and `scripts/install_safety_hooks.sh`
   must equal `main`'s on every branch. Modifications route through
   `fix/<topic>` PRs onto `main`, never research-branch edits.
   (See Pitfall #2. Pre-2026-05-14 this rule covered `apply.sh` and
   `safe_overlay_apply.sh`, both deleted with the sibling-build migration.)

4. **Decomposed patches.** Each patch covers one orthogonal concern.
   Numeric lexical prefix `00NN-` controls apply order. When a patch
   grows beyond one concern, split it.

5. **Refactors finish on every branch in the same cycle.** Adding,
   removing, splitting, or merging patches must propagate to every
   active branch simultaneously. A partial refactor leaves the
   un-propagated branches with a stale + new patch set that conflicts
   at apply time. Use the bug-fix-propagation pattern (CLAUDE.md
   "Bug fixes — land at upstream ancestor"). (See Pitfall #5.)

## Quick reference — common workflows

| Goal | Command / steps |
|---|---|
| Fresh clone setup | `git submodule update --init --recursive && bash scripts/install_safety_hooks.sh && bash overlays/Pointcept/build.sh && bash overlays/FCGF/build.sh` |
| Build overlays | `bash overlays/<sub>/build.sh` — produces `build/<sub>/` |
| Add a new file to the submodule | place at `overlays/<sub>/add/<rel-path>/`, commit on the relevant branch |
| Modify an upstream file | generate patch (see "How to add a new patch"), commit on the relevant branch |
| Bump pin | see "How to bump the pin"; mirror submodules per CLAUDE.md |
| Validate before pushing | `bash scripts/validate_overlays.sh` (whole repo) — also auto-runs in the pre-commit hook scoped to the current worktree |
| Propagate a fix across branches | follow CLAUDE.md "Bug fixes — land at upstream ancestor, propagate downstream" |

## Per-submodule layout

Each submodule overlay lives at `overlays/<SubmoduleName>/`:

```
overlays/<SubmoduleName>/
├── PINNED_COMMIT.txt    # upstream commit SHA the overlay targets
├── add/                 # new files to add over the submodule (cp -r)
├── patch/               # *.patch files to apply (lexical order)
└── build.sh             # produces build/<SubmoduleName>/ as sibling
```

Currently populated:

- `overlays/Pointcept/` — overlays for `examples/Pointcept` submodule.
- `overlays/FCGF/` — overlays for `examples/FCGF` submodule.

## Architecture: sibling-build model (2026-05-14 onward)

As of the 2026-05-14 re-architecture, overlays produce output at a
sibling path `build/<sub>/` instead of writing into the submodule.
`examples/<sub>/` stays 100% pristine. See `REARCHITECTURE.md` for the
design rationale and migration plan.

```
overlays/<sub>/{add,patch}/   ← SOURCE: unchanged convention
        │
        │  bash overlays/<sub>/build.sh
        ▼
build/<sub>/                  ← OUTPUT: symlink mirror of examples/<sub>/
                                with overlay add/ and patch/ applied.
                                Gitignored; rebuilt by build.sh.

examples/<sub>/               ← stays 100% PRISTINE
```

`build.sh` is the canonical entry point. The pre-2026-05-14 `apply.sh`
model (which wrote overlay output INTO the submodule) was deleted on
2026-05-14 once the sibling-build path was verified in production.

## How to build

After `git submodule update --init --recursive`:

```bash
bash overlays/Pointcept/build.sh
bash overlays/FCGF/build.sh
```

Each `build.sh`:

1. Verifies the submodule is at the SHA recorded in `PINNED_COMMIT.txt`.
2. Wipes + recreates `build/<sub>/` (idempotent).
3. Symlink-mirrors `examples/<sub>/` into `build/<sub>/` (~400 symlinks).
4. Overrides symlinks with real files from `overlays/<sub>/add/`.
5. Materializes (symlink → real-copy) every file a patch will touch.
6. Applies `patch/*.patch` in lexical order via `git apply --directory=build/<sub>`.
7. Removes any `__pycache__` that crept in.

Runtime: use `PYTHONPATH=<repo>/build/<sub>` (or `cd build/<sub>` then
imports resolve naturally). Downstream cluster-bootstrap scripts (where
applicable) typically export `OVERLAY_ROOT_POINTCEPT` /
`OVERLAY_ROOT_FCGF` env vars pointing at `build/<sub>/` and include
them in `PYTHONPATH`; job configs reference these env vars or the
`build/<sub>/` path directly.

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
# Re-build via build.sh to verify:
bash overlays/Pointcept/build.sh
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
4. Run `bash overlays/<SubmoduleName>/build.sh` to verify clean build.
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
  regenerate every patch against the new pin and re-run `build.sh` to verify.

**You MUST NOT, on a research branch:**

- Modify `overlays/<SubmoduleName>/build.sh` (this file is shared infra;
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

### Pitfall #2 — Modify infrastructure scripts to work around something

**Scope**: `build.sh`, `scripts/validate_overlays.sh`,
`scripts/install_safety_hooks.sh`. All three are shared infrastructure
governed by Principle #3. (Pre-2026-05-14 this list also included
`apply.sh` and `scripts/safe_overlay_apply.sh`; both deleted with the
sibling-build migration.)

**Symptom**: any of these scripts on a research branch diverges from
`main`'s blob — typically additions like `--ignore-space-change`,
verbose-debug, "second-copy after patches" hooks, or local-only
exemptions. These are almost always workarounds for an actual problem
elsewhere (often Pitfall #1).

**Why it's wrong**: shared infrastructure must behave identically
across branches. Per-branch divergence means (a) the cluster runs
different scripts on different branches — different overlay-apply
semantics is a significant cross-branch consistency hazard, (b) when
the underlying problem is fixed, the research-branch workarounds become
dead code that future agents don't dare remove for fear of regression,
and (c) pre-commit hooks calling the local script may pass on one
branch and fail on another for unrelated reasons.

**Correct fix**: revert the script to match `main`'s blob (use
`git show origin/main:<path> > <path>`). If the script itself
genuinely needs changing, open a `fix/<topic>` PR onto `main` — the
new behavior then propagates to every research branch via the standard
bug-fix propagation pattern (CLAUDE.md "Bug fixes — land at upstream
ancestor"). `scripts/validate_overlays.sh` Check 5 enforces this for
`build.sh` automatically; the other two scripts are governed by
the same rule but checked manually for now.

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

### Pitfall #4 — Data loss from `apply.sh --force` / `git clean -fd` (historical, resolved 2026-05-14)

> **Status**: Resolved by the sibling-build migration. `apply.sh` and
> `safe_overlay_apply.sh` were deleted on 2026-05-14; `build.sh` writes
> to `build/<sub>/` (gitignored, regenerable) and never touches
> `examples/<sub>/`. Retained below as a case study of what the
> pre-migration model could destroy.


**Symptom**: a session of local development leaves untracked files in
the submodule working tree (new configs, prototype model packages, etc.)
that were never registered in `overlays/<sub>/add/`. The next time
anyone runs `apply.sh --force` — or any agent runs `git clean -fdx`
inside the submodule for diagnosis — those files are silently destroyed.
There is no warning, no confirmation, no audit trail beyond `git reflog`
(which doesn't help for untracked content).

**Concrete example (autoresearch worktree, 2026-05-14)**: while
diagnosing a separate overlay bug, an agent ran
`git -C examples/Pointcept clean -fdx` to reset to pristine. The
submodule had ~10 untracked PNT config files
(`configs/scannet/semseg-pnt-v0-paper-mini.py`, `pnt_v0.py`, etc.) and
an entire `pointcept/models/point_native_transformer/` package. All
were wiped. Recovery was only possible because the
`worktrees/threads/architectures/` worktree happened to hold separate
copies — had only the autoresearch worktree existed, the work would
have been irretrievably lost.

**Why it's wrong**: untracked content in the submodule has the same
status as untracked content in any working tree — it's ephemeral
unless someone version-controls it. The submodule's own git tree is
pinned-and-read-only from the project's perspective, so even adding
untracked files there doesn't help. They have to live in
`overlays/<sub>/add/<rel-path>/` to survive.

**Resolution (2026-05-14)**: the sibling-build migration eliminated
this failure mode entirely. `build.sh` produces output at
`build/<sub>/` and never touches `examples/<sub>/`, so there is no
in-submodule destructive operation to guard against. `apply.sh` and
`safe_overlay_apply.sh` were deleted in the same migration.

### Pitfall #5 — Stale patches after a partial refactor

**Symptom**: a multi-file refactor (e.g., decomposing a monolithic
patch into N smaller patches) is completed on `main` and most thread
branches, but at least one branch is missed. The missed branch ends
up holding BOTH the old monolithic patch AND the new decomposed set;
the two overlap and `apply.sh` fails with confusing "patch does not
apply cleanly after previous patches" errors.

**Concrete example (autoresearch branch, 2026-05-14)**: the
2026-05-09 refactor that decomposed `0001-pointcnnpp-overlay.patch`
(76 KB, 29 files) into the canonical 5 patches
(`0001-core-infra` through `0005-loss-enhancements`) shipped to
`main`, `threads/architectures`, `threads/attn_extreme`,
`threads/conv_extreme`, `threads/ptv3_and_sota`, and
`threads/sparse_sdpa`. The autoresearch branch was not included —
likely because nobody runs cluster jobs from autoresearch's overlay
(`deploy.py` prefers the target branch's worktree per the
`scripts/deploy.py:267-275` overlay-source-selection logic), so the
breakage went unnoticed for 5 days. The autoresearch overlay
therefore retained 7 patches (5 canonical + the 2 stale monolithic
ones), `apply.sh` failed with conflict errors, and local development
on autoresearch silently used a half-applied submodule state. Fixed
2026-05-14 by deleting the 2 stale patches on autoresearch.

**Why it's wrong**: branch propagation is not automatic. Refactors
of the overlay system must follow the same propagation rule as
bug fixes: land on the highest common ancestor (`main`), then
cherry-pick to every active downstream branch in the SAME session.
Skipping a branch is silent — there's no signal until someone tries
to apply the overlay there, which may be days or weeks later.

**Correct fix**: after any refactor that adds, removes, splits, or
merges patches, run `bash scripts/validate_overlays.sh` (whole repo,
no `--only`) to verify every active branch is consistent. The
validator's Check 1 (dry-run apply) catches stale-patch conflicts.
For propagation, use the CLAUDE.md "Bug fixes — land at upstream
ancestor, propagate downstream" pattern: open a `fix/<topic>` PR
onto `main`, cherry-pick to every active branch independently, do
not wait for the `main` PR to merge before propagating.

## How to verify your overlay change

Before pushing any branch that touches `overlays/`:

```bash
bash scripts/validate_overlays.sh                       # whole-repo audit
bash scripts/validate_overlays.sh --only "$(pwd)"       # current worktree only
```

The validator checks (per `scripts/validate_overlays.sh`):

1. Patches apply cleanly (dry-run) on a fresh submodule checkout.
2. No `__pycache__` directories in `add/`.
3. **No file collisions between `add/` and `patch/` (this is an ERROR,
   not a warning — see Pitfall #1).**
4. `PINNED_COMMIT.txt` matches across all worktrees for the same submodule.
5. **`build.sh` blob equals `main`'s `build.sh` blob** (on research
   branches; see Pitfall #2). If you're on a `fix/<topic>` branch
   actually modifying `build.sh`, the check skips itself.
6. **`SAFETY_CHECKS.txt` runtime assertions** — for each overlay
   shipping `overlays/<sub>/SAFETY_CHECKS.txt`, runs `build.sh`
   end-to-end and asserts each declared `(path, regex)` pair is
   present in the built tree. Catches the failure mode where a
   patch and an `add/` override silently conflict and stop the
   build before downstream patches apply (the 2026-05-15 zero_coef
   incident — a `0003` train.py hunk silently collided with the
   `add/` override that carried the actual PR #34 logic, and
   Check 1's per-patch dry-run did not detect it because each
   patch passes against pristine upstream in isolation).
   Downstream consumers (e.g. cluster-runtime bootstrap scripts)
   can read the same file to keep local + runtime checks in
   lockstep.

Pass = green to push. Fail = fix the underlying issue, not the validator.

### Pre-commit hook

`scripts/install_safety_hooks.sh` installs a local `pre-commit` hook
(once per clone — `.git/hooks` is shared across all worktrees of one
clone). When a commit touches anything under `overlays/`, the hook
runs `bash scripts/validate_overlays.sh --only "$TOPLEVEL"` against
the worktree being committed to. If the validation fails, the commit
is aborted. Pre-existing failures on sibling worktrees do not block
unrelated commits (that's what the `--only` scoping is for).

Run `bash scripts/install_safety_hooks.sh` after every fresh clone.
It also reifies the AI-attribution-stripping `prepare-commit-msg`
hook (CLAUDE.md "Commit messages — no AI attribution").

### Layered guarantee

Three layers prevent the failure classes catalogued above:

| Failure class | Prevention layer |
|---|---|
| Data loss from `git clean -fd` (Pitfall #4) | Resolved by 2026-05-14 sibling-build migration — `build.sh` writes only to `build/<sub>/` (gitignored, regenerable), never modifies the submodule |
| Broken patch stack committed (Pitfall #5) | pre-commit hook runs `validate_overlays.sh --only` on every `overlays/`-touching commit |
| Stylistic drift across branches (Pitfall #2) | `validate_overlays.sh` Check 5 enforces `build.sh` blob equality; the other shared scripts are governed by the same rule (Principle #3) and audited manually for now |
