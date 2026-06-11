"""Test-only dispatch override for the MVMR / VVOR Python wrappers.

Production callers use the threshold-based dispatch in
``sparse_engines.mvmr_triton._try_grouped_dispatch`` (and the VVOR
analog). For tests we need to:

  - force-grouped: bypass the avg-triplets-per-k threshold so the
    grouped path runs on tiny inputs (correctness probes, smoke tests),
  - force-per-triplet: skip the grouped path entirely so we can
    benchmark the legacy kernel head-to-head, or verify that fallbacks
    haven't bit-rotted.

Implementation: a module-level ``_state`` dict guarded by a
``contextlib.contextmanager`` so test scopes are explicit and the
default ("auto") restores cleanly even if a test raises.

Wrappers consult ``current_mode()``; the threshold and sort checks are
short-circuited per the returned mode. ``"auto"`` preserves the
original threshold-based behaviour.

Not thread-safe — tests using these context managers must run serially
(pytest's default).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Literal


# ── Dispatch path (grouped vs per-triplet) ──
#   "auto"           — production threshold-based dispatch (default)
#   "force_fsg"  — grouped Triton at ANY G (native G>1, one group per
#                      program or BSG>1 multi-group blocks); still falls
#                      back if `a_idx`/`o_idx` not sorted
#                      (correctness > speed)
#   "force_pt" — skip grouped, always use legacy kernel
#   "force_fsg_wmma_vvor" — vvor-only: route through the hand-CUDA
#                      WMMA-direct grouped kernel.
#                      mvmr stays on whatever Triton routing applies.
#                      Same preconditions as the WMMA wrapper:
#                      G == 1, dtype in {fp16, bf16, fp32→scalar-FMA
#                      fallback}, M % 16 == 0, C % 16 == 0.
#   "force_fsg_cutlass_mvmr" — mvmr-only: route the mvmr functional
#                      op (forward AND the grad_b backward, which is a
#                      second mvmr call with a transposed weight) through
#                      the Tier-2 CUTLASS path.
#                      fp16 OR bf16 (via the
#                      SM80_16x8x16_F32BF16BF16F32_TN atom) / G==1 /
#                      sorted-a_idx only; fp32 (and mixed-dtype pairs)
#                      fall back to the existing Triton-grouped path —
#                      no SM80 fp32-input tensor-core atom of this shape.
#                      vvor stays on whatever Triton routing applies.
#   "force_fsg_cutlass_mvmr_vvor" — combined mode:
#                      route mvmr fwd+grad_b → CUTLASS mvmr AND
#                      vvor grad_a → CUTLASS vvor *simultaneously* in the
#                      same forward/backward. The single-mode modes are
#                      mutually exclusive: under
#                      "force_fsg_cutlass_mvmr" vvor's grad_a falls
#                      back to Triton, and "force_fsg_cutlass_vvor"
#                      leaves mvmr on Triton — so having CUTLASS mvmr
#                      fwd+grad_b AND CUTLASS vvor grad_a active together
#                      is inexpressible without this. Composes the two
#                      component routings under one label; each keeps its
#                      own dtype gating (both mvmr AND vvor CUTLASS
#                      accept fp16 OR bf16; fp32 / mixed-dtype pairs
#                      fall back — mvmr to Triton-grouped, vvor to
#                      scalar-FMA).
#   "force_fsg_fused" — route PointConv3d's mvmr+autograd
#                      through the single `FusedPointConv3d` Function
#                      (mvmr_cutlass.py). Collapses the 3 @triton_op/
#                      autograd-graph boundaries + 2 seg_offs builds + the
#                      duplicate Python .contiguous() into one Function;
#                      forward S2 is zero-copy (no-op-collapse view), the
#                      frozen CUTLASS mvmr/vvor full kernels are reused
#                      as-is. grad_b retains its single existing host
#                      transpose-repack. fp16/
#                      G==1/sorted/tile-multiple — same hard preconditions
#                      as the underlying _cutlass entry points (which
#                      raise on violation). All other modes/paths are
#                      byte-unchanged when this mode is not set (the
#                      routing site short-circuits only on this string).

# v1.2.0 PT/FSG/TIG taxonomy: canonical dispatch strings are
# generation-prefixed. The pre-rename strings are NOT aliased (clean
# break) — they raise with the replacement named.
_LEGACY_RENAMES = {
    "force_per_triplet": "force_pt",
    "force_grouped": "force_fsg",
    "force_grouped_wmma_vvor": "force_fsg_wmma_vvor",
    "force_grouped_wmma_coop_vvor": "force_fsg_wmma_coop_vvor",
    "force_grouped_cutlass_vvor": "force_fsg_cutlass_vvor",
    "force_grouped_cutlass_mvmr": "force_fsg_cutlass_mvmr",
    "force_grouped_cutlass_mvmr_vvor": "force_fsg_cutlass_mvmr_vvor",
    "force_fused_conv": "force_fsg_fused",
    "force_smig": "force_tig",
}

DispatchMode = Literal[
    "auto", "force_fsg", "force_pt",
    "force_fsg_wmma_vvor", "force_fsg_wmma_coop_vvor",
    "force_fsg_cutlass_vvor", "force_fsg_cutlass_mvmr",
    "force_fsg_cutlass_mvmr_vvor", "force_fsg_fused",
    "force_tig", "force_tig",
]

# ── tl.dot input precision for the grouped path ──
#   "default" → "tf32" on Ampere+ (the silent default — fast but ~1e-3 rel
#               err per multiply on fp32 inputs)
#   "ieee"    → IEEE single-precision; ~1e-7 rel err but no tensor-core mma
#               at fp32 inputs (fp16/bf16 still use tensor cores)
PrecisionMode = Literal["default", "ieee"]


_state: dict = {"mode": "auto", "precision": "default"}


def current_mode() -> DispatchMode:
    """Return the current dispatch mode. Wrappers consult this to
    short-circuit the grouped-vs-per-triplet decision."""
    return _state["mode"]


def current_precision() -> str:
    """Return the tl.dot ``input_precision`` string for the grouped path.
    Used as the ``INPUT_PRECISION`` constexpr passed to the kernel."""
    p = _state["precision"]
    return "ieee" if p == "ieee" else "tf32"


@contextmanager
def dispatch_mode(mode: DispatchMode):
    """Context manager scoping a non-default dispatch mode.

    Example:
        with dispatch_mode("force_pt"):
            out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(...)
        # back to "auto" outside the block
    """
    if mode not in (
        "auto", "force_fsg", "force_pt",
        "force_fsg_wmma_vvor", "force_fsg_wmma_coop_vvor",
        "force_fsg_cutlass_vvor", "force_fsg_cutlass_mvmr",
        "force_fsg_cutlass_mvmr_vvor", "force_fsg_fused",
        "force_tig", "force_tig",
    ):
        if mode in _LEGACY_RENAMES:
            raise ValueError(
                f"dispatch mode {mode!r} was renamed to "
                f"{_LEGACY_RENAMES[mode]!r} (v1.2.0 PT/FSG/TIG taxonomy)")
        raise ValueError(f"unknown dispatch mode: {mode!r}")
    prev = _state["mode"]
    _state["mode"] = mode
    try:
        yield
    finally:
        _state["mode"] = prev


@contextmanager
def precision_mode(mode: PrecisionMode):
    """Context manager scoping the tl.dot precision for the grouped path.

    Example:
        with precision_mode("ieee"):
            out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(...)
        # back to default ("tf32") outside the block

    No effect on the per-triplet kernel (which doesn't use tl.dot).
    No effect at fp16/bf16 inputs (those always use mma at their dtype's
    native precision; ``input_precision="ieee"`` is only meaningful at
    fp32 inputs).
    """
    if mode not in ("default", "ieee"):
        raise ValueError(f"unknown precision mode: {mode!r}")
    prev = _state["precision"]
    _state["precision"] = mode
    try:
        yield
    finally:
        _state["precision"] = prev
