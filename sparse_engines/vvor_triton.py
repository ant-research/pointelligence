"""VVOR: Vector-Vector Outer product and Reduction.

Computes weight gradients in the backward pass of point convolution.
Given triplets (i, j, k), accumulates grad_weight[k] += input[j] (x) grad_output[i].
Paired with MVMR for the forward pass.

Two compute paths:
- Per-triplet (legacy): each triplet's outer product accumulated one at a time.
  K_inner=1 prevents tensor-core utilisation at any input dtype.
- Grouped (default when applicable): triplets with sort_by="k" already group
  consecutive entries by `o_idx` (which IS the kernel offset in VVOR's call
  pattern). We batch L_CHUNK triplets and compute the per-kernel-offset
  weight-grad tile via a single `(M, L_CHUNK) @ (L_CHUNK, C) → (M, C)`
  tensor-core GEMM — fully eliminating the K=1 issue.
"""

import torch
from torch import Tensor
from typing import Optional

import triton
from torch.library import triton_op, wrap_triton

from .vvor_triton_kernel import (
    sparse_vector_vector_outer_product_reduction_kernel,
    sparse_vector_vector_outer_product_reduction_grouped_kernel,
)
from ._seg_offs import (is_sorted_cached, kernel_offset_segments,
                         kernel_offset_segments_cached)
from ._dispatch_override import current_mode, resolve_input_precision


_GROUPED_MIN_TRIPLETS_PER_K = 16
# Channel/M threshold (matches mvmr_triton._GROUPED_MIN_C). Grouped VVOR's
# tl.dot has shape `(M, L) @ (L, C) → (M, C)`. We require both M and C
# ≥ 128 — i.e. the matmul tile must be deep on both axes for the tensor-
# core throughput to overcome the per-program setup cost. Empirical
# crossover from the PointConv3d engine bench PTv3 stage profile
# matches MVMR's: grouped beats per_triplet starting at C=128 (fp16/bf16,
# tied at fp32) and dominates at C ≥ 256.
#
# Lowered 128 → 64, in lockstep with
# mvmr_triton._GROUPED_MIN_C (see the rationale there). This is the grad_a
# (dW) leg of the same eager convs; at the generative deconv cells the
# min(M,C)=64 stages misrouted to per_triplet and cost up to 1.9x f+b vs
# force_fsg on Hopper (generative-shapes operator bench, sm_89 + sm_90).
_GROUPED_MIN_DIM = 64


def _try_grouped_dispatch(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor,
    o: Tensor, o_idx: Tensor, T: int, G: int, M: int, C: int, n_o: int,
    seg_offs: "Optional[Tensor]" = None,
) -> bool:
    """Attempt the grouped-kernel path. Returns True on success.

    Preconditions (in "auto" mode):
      - ``o_idx`` is sorted ascending (in VVOR-as-bwd-dW with sort_by="k",
        ``o_idx`` is the kernel offset → already sorted)
      - average triplets-per-kernel-offset ≥ _GROUPED_MIN_TRIPLETS_PER_K
      - G == 1 (production path; G > 1 falls back)

    ``seg_offs`` — torch.compile contract, mirrors the mvmr wrapper.
    When passed it ASSERTS ``o_idx`` is sorted ascending and skips the
    ``is_sorted_cached`` ``.item()`` + ``kernel_offset_segments_cached``
    ``.data_ptr()`` memo, so the ``@triton_op`` body is sync-free and
    traces under ``torch.compile``. This op is reached as mvmr's backward
    grad_a (``a_idx`` k-sorted → passed here as ``o_idx``), so a single
    forward-built seg_offs closes the {mvmr, vvor} dual sync-free.
    ``None`` (default) keeps the exact pre-contract behavior.
    """
    mode = current_mode()
    if mode == "force_pt":
        return False
    K_offsets = n_o
    # Threshold checks before sortedness so auto-mode below-threshold calls
    # don't pay a CPU↔GPU sync just to fall through.
    # The G==1 gate is lifted for force modes (kernel is G-generic, one
    # group per program at BLOCK_SIZE_G=1); auto keeps G==1 until the
    # routing decision lands.
    if mode == "auto":
        if G != 1:
            return False
        if min(M, C) < _GROUPED_MIN_DIM:
            return False
        if T < _GROUPED_MIN_TRIPLETS_PER_K * K_offsets:
            return False
    if seg_offs is None:
        # Sortedness check (correctness precondition) — has a `.item()` sync.
        if not is_sorted_cached(o_idx):
            return False
        seg_offs = kernel_offset_segments_cached(o_idx, K_offsets)
    # else: caller-asserted sorted o_idx + precomputed seg_offs (compile path).
    # Grid: sync-free UPPER BOUND on the chunk count (G>1
    # squeeze — see the mvmr wrapper's comment). Programs past the true
    # chunk count exit early in the kernel.
    grid = lambda META: (
        ((T + META["L_CHUNK"] - 1) // META["L_CHUNK"] + K_offsets)
        * triton.cdiv(G, META["BLOCK_SIZE_G"])
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(C, META["BLOCK_SIZE_C"]),
    )

    wrap_triton(sparse_vector_vector_outer_product_reduction_grouped_kernel)[grid](
        a, a_idx, b, b_idx, o,
        seg_offs,
        K_offsets, G, M, C,
        INPUT_PRECISION=resolve_input_precision(a.dtype),
    )
    return True


@triton_op(
    "sparse_engines::sparse_vector_vector_outer_product_reduction", mutates_args={}
)
def sparse_vector_vector_outer_product_reduction(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n_o: int,
    seg_offs: "Optional[Tensor]" = None,
) -> Tensor:
    # Gate 1: when the dispatch override is
    # "force_fsg_wmma_vvor", bypass the Triton path entirely and route
    # to the hand-CUDA WMMA-direct grouped kernel. Same autograd hooks
    # apply (registered on this @triton_op below), so the backward path
    # of an mvmr-fwd invocation still flows through here correctly.
    if current_mode() == "force_fsg_wmma_vvor":
        from .vvor_grouped_wmma import (
            sparse_vector_vector_outer_product_reduction_grouped_wmma,
        )
        return sparse_vector_vector_outer_product_reduction_grouped_wmma(
            a, a_idx, b, b_idx, o_idx, n_o,
        )
    if current_mode() == "force_fsg_wmma_coop_vvor":
        from .vvor_grouped_wmma_coop import (
            sparse_vector_vector_outer_product_reduction_grouped_wmma_coop,
        )
        return sparse_vector_vector_outer_product_reduction_grouped_wmma_coop(
            a, a_idx, b, b_idx, o_idx, n_o,
        )
    if current_mode() in (
        "force_fsg_cutlass_vvor", "force_fsg_cutlass_mvmr_vvor",
    ):
        # Route to the Tier-2 CUTLASS
        # implicit-GEMM + K-mode IndexedGather vvor. fp16 OR bf16 (the
        # bf16 atom was added later); fp32 falls back to scalar-FMA exactly
        # like the WMMA paths (no SM80 fp32-input tensor-core atom).
        # The combined "force_fsg_cutlass_mvmr_vvor" mode ALSO
        # routes vvor here. mvmr's backward computes grad_a via a vvor
        # call (_backward_…sparse_matrix_vector_multiplication_reduction),
        # so under the combined mode grad_a hits CUTLASS vvor here while
        # mvmr fwd+grad_b hit CUTLASS mvmr in mvmr_triton.py.
        #
        # The fallback must fire when the operands are not a
        # matched CUTLASS-supported pair, not only when `a` is fp32. Under
        # `torch.autocast(fp16)` the conv weight Parameter stays fp32 at the
        # dispatch boundary, so grad_a arrives as a=fp16 (autocast-produced
        # grad) but b=fp32. The old `a.dtype == float32`-only guard let that
        # MIXED (a=fp16, b=fp32) pair through to the CUTLASS kernel, which
        # raised and crashed every AMP step using this mode. The CUTLASS
        # kernel now supports fp16 AND bf16 (matched operands), so the
        # CUTLASS route is taken iff both operands share a dtype in
        # {fp16, bf16}; any other pair (mixed, or fp32) takes the scalar-FMA
        # fallback. No silent cast (that would change the user's autocast
        # numerics).
        #
        # The scalar fallback (`vvor_grouped_cuda`) itself requires a and b
        # to share a dtype (its CUDA kernel raises "expected Half but found
        # Float" on a mixed pair). So normalize by promoting the narrower
        # operand UP to the other's dtype before the fallback — lossless
        # (a widening cast) and consistent with the kernel's fp32 accumulate;
        # never a narrowing cast that would drop autocast-preserved precision.
        _cutlass_ok = (
            a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16)
        )
        if not _cutlass_ok:
            from .vvor_grouped_cuda import (
                sparse_vector_vector_outer_product_reduction_grouped_cuda
                as _fallback_scalar_fma,
            )
            if a.dtype != b.dtype:
                promoted = torch.promote_types(a.dtype, b.dtype)
                a = a.to(promoted)
                b = b.to(promoted)
            return _fallback_scalar_fma(a, a_idx, b, b_idx, o_idx, n_o)
        from .vvor_cutlass import (
            sparse_vector_vector_outer_product_reduction_grouped_cutlass,
        )
        return sparse_vector_vector_outer_product_reduction_grouped_cutlass(
            a, a_idx, b, b_idx, o_idx, n_o,
        )

    # Mirrors mvmr_triton.py: the grouped kernel feeds
    # operands natively into tl.dot — a MIXED dtype pair (AMP boundary)
    # would CompilationError. Promote the narrower operand UP, lossless.
    if a.dtype != b.dtype:
        promoted = torch.promote_types(a.dtype, b.dtype)
        a = a.to(promoted)
        b = b.to(promoted)

    a = a.contiguous()
    b = b.contiguous()

    T, G, M, C = a_idx.numel(), a.shape[1], a.shape[2], b.shape[2]
    input_dtype = a.dtype
    # Accumulate in fp32 for numerical stability, cast back after
    o = torch.zeros((n_o, G, M, C), dtype=torch.float32, device=a.device)

    if not _try_grouped_dispatch(a, a_idx, b, b_idx, o, o_idx, T, G, M, C, n_o,
                                 seg_offs=seg_offs):
        if current_mode() == "auto":  # PT advisory (once/shape)
            from ._dispatch_override import warn_pt_fallback
            warn_pt_fallback("vvor", "below the grouped floors or unsorted",
                             K=n_o, G=G, C=C, M=M)
        grid = lambda META: (
            triton.cdiv(T, META["L"])
            * triton.cdiv(G, META["BLOCK_SIZE_G"])
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(C, META["BLOCK_SIZE_C"]),
        )
        wrap_triton(sparse_vector_vector_outer_product_reduction_kernel)[grid](
            a, a_idx, b, b_idx, o, o_idx, T, G, M, C
        )

    return o.to(input_dtype) if input_dtype != torch.float32 else o


@sparse_vector_vector_outer_product_reduction.register_fake
def _sparse_vector_vector_outer_product_reduction_fake(
    a, a_idx, b, b_idx, o_idx, n_o, seg_offs=None,
):
    # Output [n_o, G, M, C] is a pure function of input shapes + n_o:
    # G = a.shape[1], M = a.shape[2], C = b.shape[2] (see the body's
    # `T, G, M, C = ...` + `torch.zeros((n_o, G, M, C))`), cast from the
    # fp32 accumulate back to the promoted operand dtype. No value-
    # dependent extent → data-independent meta, traceable as a leaf.
    out_dtype = torch.promote_types(a.dtype, b.dtype)
    return a.new_empty((n_o, a.shape[1], a.shape[2], b.shape[2]), dtype=out_dtype)


def _backward_sparse_vector_vector_outer_product_reduction(ctx, grad):
    from .mvmr_triton import sparse_matrix_vector_multiplication_reduction

    a, a_idx, b, b_idx, o_idx = ctx.saved_tensors
    grad_a, grad_b = None, None
    # torch.compile contract: vvor is a valid standalone forward op whose
    # backward is mvmr (we don't call it that way today, but it is correct
    # computation). The forward-supplied seg_offs is built from the sorted
    # o_idx over K=n_o bins; both backward mvmr sub-calls pass o_idx as mvmr's
    # a_idx (sort key) with K=grad.shape[0]=n_o — the SAME structure — so one
    # forward build keeps a standalone vvor fwd+bwd sync-free under compile.
    # None on the eager path → the mvmr sub-calls run their own runtime check.
    fwd_seg_offs = getattr(ctx, "compile_seg_offs", None)
    if ctx.needs_input_grad[0]:
        grad_a = sparse_matrix_vector_multiplication_reduction(
            grad.transpose(2, 3),
            o_idx,
            b,
            b_idx,
            a_idx,
            a.shape[0] if isinstance(a, torch.Tensor) else ctx.a_shape_0,
            fwd_seg_offs,
        )
    if ctx.needs_input_grad[2]:
        grad_b = sparse_matrix_vector_multiplication_reduction(
            grad,
            o_idx,
            a,
            a_idx,
            b_idx,
            b.shape[0] if isinstance(b, torch.Tensor) else ctx.b_shape_0,
            fwd_seg_offs,
        )
    return grad_a, None, grad_b, None, None, None, None


def _setup_context_sparse_vector_vector_outer_product_reduction(ctx, inputs, output):
    a, a_idx, b, b_idx, o_idx, n, seg_offs = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[2]:
        saved_a = a
    ctx.save_for_backward(saved_a, a_idx, saved_b, b_idx, o_idx)
    ctx.a_shape_0 = a.shape[0]
    ctx.b_shape_0 = b.shape[0]
    # torch.compile contract: propagate the forward seg_offs (built from
    # the sorted o_idx over K=n_o bins) so a standalone vvor's backward mvmr
    # calls stay sync-free. None on the eager path → unchanged behavior.
    ctx.compile_seg_offs = seg_offs


sparse_vector_vector_outer_product_reduction.register_autograd(
    _backward_sparse_vector_vector_outer_product_reduction,
    setup_context=_setup_context_sparse_vector_vector_outer_product_reduction,
)
