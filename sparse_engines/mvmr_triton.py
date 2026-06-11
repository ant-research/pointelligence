"""MVMR: Matrix-Vector Multiplication and Reduction.

The core forward-pass sparse engine for point convolution. Given triplets
(i, j, k), computes output[i] += weight[k] @ input[j] via a Triton kernel
with autotuned block sizes. See PointCNN++ (arXiv:2511.23227) Section 3.4.

Two compute paths:
- Per-triplet (legacy, pre-tensor-core): each triplet's MV product handled
  one-at-a-time. Used as a fallback when the kernel-offset distribution
  is too sparse for tensor-core batching, or when a_idx is not sorted.
- Grouped (default when applicable): triplets sorted by kernel offset are
  batched into L_CHUNK-row tl.dot GEMMs, which engage sm_80+ tensor cores.
  Threshold for routing here: ``T / K >= _GROUPED_MIN_TRIPLETS_PER_K``.
"""

import torch
from torch import Tensor

import triton
from torch.library import triton_op, wrap_triton

from .mvmr_triton_kernel import (
    sparse_matrix_vector_multiplication_reduction_kernel,
    sparse_matrix_vector_multiplication_reduction_grouped_kernel,
)
from ._seg_offs import kernel_offset_segments, total_chunks_for_lchunks
from ._dispatch_override import current_mode, current_precision

# L_CHUNK options that the grouped kernel's autotune palette covers.
# See mvmr_triton_kernel._GROUPED_AUTOTUNE_CONFIGS. Must match.
_L_CHUNK_OPTIONS = (16, 32, 64, 128, 256)


# Channel threshold: grouped path wins above the tensor-core mma break-even.
# Empirical (PointConv3d engine bench at PTv3 stage shapes,
# RTX 5880 Ada, May 2026, fwd+bwd ms vs per_triplet):
#   - C = 32:   per_triplet wins by ~30%  (tiny C → tl.dot setup cost > tensor-core gain)
#   - C = 64:   per_triplet wins by ~10%
#   - C = 128:  grouped wins by 12-21% (fp16/bf16); fp32 within 2% (tied)
#   - C = 256:  grouped wins by 27% (fp16) / 9% (fp32)
#   - C = 512:  grouped wins by ~10% (fwd+bwd) — tensor-core mma fully amortised
# Production threshold: grouped at C ≥ 128. fp32-C=128 is a wash but other
# dtypes win meaningfully there. Picking dtype-specific thresholds adds
# complexity for no measurable improvement at the wash point.
_GROUPED_MIN_C = 128

# Minimum triplets-per-kernel-offset (sm_80+ tensor-core mma needs ≥ 16
# rows). Below this the grouped kernel is also wrong (M_inner < 16 in
# tl.dot would lose tensor cores anyway).
_GROUPED_MIN_TRIPLETS_PER_K = 16


def _try_grouped_dispatch(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor,
    o: Tensor, o_idx: Tensor, T: int, G: int, M: int, C: int,
) -> bool:
    """Attempt the grouped-kernel path. Returns True on success.

    Preconditions (in "auto" mode):
      - ``a_idx`` is sorted ascending (corresponds to ``sort_by="k"``)
      - C ≥ _GROUPED_MIN_C (tensor-core advantage outweighs segment-
        dispatch overhead)
      - average triplets-per-kernel-offset ≥ _GROUPED_MIN_TRIPLETS_PER_K
        (so tl.dot's M_inner = L_CHUNK ≥ 16 → tensor cores fire)
      - G == 1 (grouped's tl.dot path requires G=1; G > 1 falls back)

    The test-only ``dispatch_mode`` override can force the grouped path
    on (still requiring the hard preconditions G==1 and sorted a_idx)
    or force it off entirely (regardless of threshold).
    """
    mode = current_mode()
    if mode == "force_per_triplet":
        return False
    K_offsets = a.shape[0]
    # Cheap precondition first — G is a Python int, no GPU work.
    if G != 1:
        return False
    # Threshold checks before the sortedness check so auto-mode below-threshold
    # calls don't pay a CPU↔GPU sync just to fall through. force_grouped
    # bypasses these and runs the sortedness check.
    if mode == "auto":
        if C < _GROUPED_MIN_C:
            return False
        if T < _GROUPED_MIN_TRIPLETS_PER_K * K_offsets:
            return False
    # Sortedness check (correctness precondition for grouped path) — has a
    # `.item()` sync, so deferred until we've decided we want grouped.
    if not bool((a_idx[1:] >= a_idx[:-1]).all().item()):
        return False
    seg_offs = kernel_offset_segments(a_idx, K_offsets)
    # Compute total_chunks for each L_CHUNK in the autotune palette so
    # the grid lambda can size the launch correctly per autotune trial.
    # One batched .item() sync covers all options. Kernel walks seg_offs
    # itself so we don't need to materialise a chunk_seg_offs tensor.
    total_chunks_options = total_chunks_for_lchunks(seg_offs, _L_CHUNK_OPTIONS)
    total_by_lc = dict(zip(_L_CHUNK_OPTIONS, total_chunks_options))
    if max(total_chunks_options) == 0:
        return False

    grid = lambda META: (
        total_by_lc[META["L_CHUNK"]]
        * triton.cdiv(G, META["BLOCK_SIZE_G"])
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(C, META["BLOCK_SIZE_C"]),
    )

    wrap_triton(sparse_matrix_vector_multiplication_reduction_grouped_kernel)[grid](
        a, b, b_idx, o, o_idx,
        seg_offs,
        K_offsets, G, M, C,
        INPUT_PRECISION=current_precision(),
    )
    return True


@triton_op(
    "sparse_engines::sparse_matrix_vector_multiplication_reduction", mutates_args={}
)
def sparse_matrix_vector_multiplication_reduction(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n_o: int
) -> Tensor:
    # When the dispatch override is "force_grouped_cutlass_mvmr", route
    # this functional op to the Tier-2 CUTLASS implicit-GEMM + S-mode
    # IndexedGather + atomicAdd scatter mvmr. Because grad_b is computed
    # by a *second* call to this same functional op (with a transposed
    # weight `a.transpose(2,3)` — see `_backward_…` below), routing here
    # closes BOTH the forward and the grad_b backward. fp32 / mixed-dtype
    # pairs fall through to the existing Triton-grouped path. The CUTLASS
    # wrapper enforces the remaining hard preconditions (G == 1, sorted
    # a_idx, M/C tile multiples) and raises on violation. The autograd
    # hooks registered on this @triton_op below still apply, so the
    # backward of a CUTLASS-routed forward also flows back through here.
    # The combined "force_grouped_cutlass_mvmr_vvor" mode ALSO routes
    # mvmr (fwd+grad_b) here — it composes this mvmr routing with the
    # vvor grad_a CUTLASS routing in vvor_triton.py so both kernels are
    # active in the same forward/backward.
    #    # Require BOTH operands the SAME tensor-core dtype, not just `a`.
    # The CUTLASS mvmr kernel needs matched operands —
    # `mvmr_cutlass.py` raises on any mismatched or non-{fp16,bf16} pair.
    # An `a.dtype == float16`-only guard is safe for standard AMP (the
    # weight Parameter stays fp32 → a=fp32 → falls through), but a MIXED
    # (a=fp16 weight, b=fp32 input) pair would pass it and crash the kernel.
    # Gating on both operands sharing a CUTLASS-supported dtype mirrors the
    # kernel's real precondition and routes any other pair to the
    # Triton-grouped path below. No silent cast (preserves autocast numerics).
    # fp16 and bf16 both engage CUTLASS; fp32 still falls through (no SM80
    # fp32-input tensor-core atom).
    if current_mode() in (
        "force_grouped_cutlass_mvmr", "force_grouped_cutlass_mvmr_vvor",
    ) and a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16):
        from .mvmr_cutlass import (
            sparse_matrix_vector_multiplication_reduction_cutlass,
        )
        # Build seg_offs ONCE here at the autograd.Function boundary and
        # share it with the grad_b second mvmr call (it depends only on
        # a_idx + K_offsets — the fixed triplet structure, invariant
        # across fwd and grad_b). _setup_context recomputes the identical
        # buffer for backward (it cannot read this local), so the share is
        # "computed once per side, not re-staged from scratch inside each
        # _cutlass call"; the decisive win is the eliminated redundant
        # Python .contiguous() + the single seg_offs build per side
        # instead of per _cutlass entry.
        seg_offs = kernel_offset_segments(a_idx, a.shape[0]).to(torch.int64)
        out = sparse_matrix_vector_multiplication_reduction_cutlass(
            a, a_idx, b, b_idx, o_idx, n_o, seg_offs=seg_offs,
        )
        return out.to(a.dtype) if out.dtype != a.dtype else out

    a = a.contiguous()
    b = b.contiguous()

    T, G, M, C = a_idx.numel(), a.shape[1], a.shape[3], a.shape[2]
    input_dtype = a.dtype
    # Accumulate in fp32 for numerical stability, cast back after
    o = torch.zeros((n_o, G, M), dtype=torch.float32, device=a.device)

    if not _try_grouped_dispatch(a, a_idx, b, b_idx, o, o_idx, T, G, M, C):
        grid = lambda META: (
            triton.cdiv(T, META["L"])
            * triton.cdiv(G, META["BLOCK_SIZE_G"])
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(C, META["BLOCK_SIZE_C"]),
        )
        wrap_triton(sparse_matrix_vector_multiplication_reduction_kernel)[grid](
            a, a_idx, b, b_idx, o, o_idx, T, G, M, C
        )

    return o.to(input_dtype) if input_dtype != torch.float32 else o


def _backward_sparse_matrix_vector_multiplication_reduction(ctx, grad):
    a, a_idx, b, b_idx, o_idx = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        from .vvor_triton import sparse_vector_vector_outer_product_reduction

        grad_a = sparse_vector_vector_outer_product_reduction(
            b,
            b_idx,
            grad,
            o_idx,
            a_idx,
            a.shape[0] if isinstance(a, torch.Tensor) else ctx.a_shape_0,
        )
    if ctx.needs_input_grad[2]:
        b_shape_0 = b.shape[0] if isinstance(b, torch.Tensor) else ctx.b_shape_0
        seg_offs = getattr(ctx, "mvmr_seg_offs", None)
        if seg_offs is not None:
            # The grad_b second mvmr call is the transposed-weight path.
            # Route it directly to the CUTLASS impl with the
            # forward-shared seg_offs (a_idx is the same fixed triplet
            # structure → seg_offs is identical to the fwd's), so it is
            # staged ONCE per step (fwd builds it; setup_context saved an
            # identical copy here) rather than rebuilt inside this second
            # _cutlass entry. Skipping the inner `.contiguous()` (inside
            # _cutlass) also removes the redundant full materialization
            # of the non-contiguous a.transpose(2,3) on top of the host
            # fn's own repack. grad casts back to a.dtype to match the
            # functional op's short-circuit (.to(a.dtype)) behaviour.
            from .mvmr_cutlass import (
                sparse_matrix_vector_multiplication_reduction_cutlass,
            )
            grad_b = sparse_matrix_vector_multiplication_reduction_cutlass(
                a.transpose(2, 3), a_idx, grad, o_idx, b_idx, b_shape_0,
                seg_offs=seg_offs,
            )
            if grad_b.dtype != a.dtype:
                grad_b = grad_b.to(a.dtype)
        else:
            grad_b = sparse_matrix_vector_multiplication_reduction(
                a.transpose(2, 3),
                a_idx,
                grad,
                o_idx,
                b_idx,
                b_shape_0,
            )
    return grad_a, None, grad_b, None, None, None


def _setup_context_sparse_matrix_vector_multiplication_reduction(ctx, inputs, output):
    a, a_idx, b, b_idx, o_idx, n = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[2]:
        saved_a = a
    ctx.save_for_backward(saved_a, a_idx, saved_b, b_idx, o_idx)
    ctx.a_shape_0 = a.shape[0]
    ctx.b_shape_0 = b.shape[0]
    # When the grad_b second mvmr call will route to CUTLASS (same gate
    # as the forward short-circuit above: cutlass-mvmr mode + fp16), build
    # seg_offs ONCE here and hand it to _backward_ so it is NOT rebuilt
    # inside the grad_b _cutlass entry. a_idx + K_offsets (a.shape[0]) are
    # the fixed triplet structure — invariant across fwd and the
    # transposed grad_b call — so this is bit-identical to the forward's
    # seg_offs (staged once per step, shared via ctx).
    ctx.mvmr_seg_offs = None
    if (
        ctx.needs_input_grad[2]
        and current_mode() in (
            "force_grouped_cutlass_mvmr", "force_grouped_cutlass_mvmr_vvor",
        )
        and a.dtype == torch.float16
    ):
        ctx.mvmr_seg_offs = kernel_offset_segments(
            a_idx, a.shape[0]
        ).to(torch.int64)


sparse_matrix_vector_multiplication_reduction.register_autograd(
    _backward_sparse_matrix_vector_multiplication_reduction,
    setup_context=_setup_context_sparse_matrix_vector_multiplication_reduction,
)
