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

import triton
from torch.library import triton_op, wrap_triton

from .vvor_triton_kernel import (
    sparse_vector_vector_outer_product_reduction_kernel,
    sparse_vector_vector_outer_product_reduction_grouped_kernel,
)
from ._seg_offs import kernel_offset_segments, total_chunks_for_lchunks
from ._dispatch_override import current_mode, current_precision


_GROUPED_MIN_TRIPLETS_PER_K = 16
# L_CHUNK options that the grouped kernel's autotune palette covers.
# Must match vvor_triton_kernel._GROUPED_AUTOTUNE_CONFIGS.
_L_CHUNK_OPTIONS = (16, 32, 64, 128, 256)

# Channel/M threshold (matches mvmr_triton._GROUPED_MIN_C). Grouped VVOR's
# tl.dot has shape `(M, L) @ (L, C) → (M, C)`. We require both M and C
# ≥ 128 — i.e. the matmul tile must be deep on both axes for the tensor-
# core throughput to overcome the per-program setup cost. Empirical
# crossover from the PointConv3d engine bench PTv3 stage profile
# matches MVMR's: grouped beats per_triplet starting at C=128 (fp16/bf16,
# tied at fp32) and dominates at C ≥ 256.
_GROUPED_MIN_DIM = 128


def _try_grouped_dispatch(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor,
    o: Tensor, o_idx: Tensor, T: int, G: int, M: int, C: int, n_o: int,
) -> bool:
    """Attempt the grouped-kernel path. Returns True on success.

    Preconditions (in "auto" mode):
      - ``o_idx`` is sorted ascending (in VVOR-as-bwd-dW with sort_by="k",
        ``o_idx`` is the kernel offset → already sorted)
      - average triplets-per-kernel-offset ≥ _GROUPED_MIN_TRIPLETS_PER_K
      - G == 1 (production path; G > 1 falls back)
    """
    mode = current_mode()
    if mode == "force_per_triplet":
        return False
    K_offsets = n_o
    # Cheap precondition first — no GPU work.
    if G != 1:
        return False
    # Threshold checks before sortedness so auto-mode below-threshold calls
    # don't pay a CPU↔GPU sync just to fall through.
    if mode == "auto":
        if min(M, C) < _GROUPED_MIN_DIM:
            return False
        if T < _GROUPED_MIN_TRIPLETS_PER_K * K_offsets:
            return False
    # Sortedness check (correctness precondition) — has a `.item()` sync.
    if not bool((o_idx[1:] >= o_idx[:-1]).all().item()):
        return False
    seg_offs = kernel_offset_segments(o_idx, K_offsets)
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

    wrap_triton(sparse_vector_vector_outer_product_reduction_grouped_kernel)[grid](
        a, a_idx, b, b_idx, o,
        seg_offs,
        K_offsets, G, M, C,
        INPUT_PRECISION=current_precision(),
    )
    return True


@triton_op(
    "sparse_engines::sparse_vector_vector_outer_product_reduction", mutates_args={}
)
def sparse_vector_vector_outer_product_reduction(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n_o: int
) -> Tensor:
    # When the dispatch override is "force_grouped_wmma_vvor", bypass the
    # Triton path entirely and route to the hand-CUDA WMMA-direct grouped
    # kernel. Same autograd hooks apply (registered on this @triton_op
    # below), so the backward path of an mvmr-fwd invocation still flows
    # through here correctly.
    if current_mode() == "force_grouped_wmma_vvor":
        from .vvor_grouped_wmma import (
            sparse_vector_vector_outer_product_reduction_grouped_wmma,
        )
        return sparse_vector_vector_outer_product_reduction_grouped_wmma(
            a, a_idx, b, b_idx, o_idx, n_o,
        )
    if current_mode() == "force_grouped_wmma_coop_vvor":
        from .vvor_grouped_wmma_coop import (
            sparse_vector_vector_outer_product_reduction_grouped_wmma_coop,
        )
        return sparse_vector_vector_outer_product_reduction_grouped_wmma_coop(
            a, a_idx, b, b_idx, o_idx, n_o,
        )
    if current_mode() in (
        "force_grouped_cutlass_vvor", "force_grouped_cutlass_mvmr_vvor",
    ):
        # Route to the Tier-2 CUTLASS implicit-GEMM + K-mode IndexedGather
        # vvor. fp16 OR bf16; fp32 falls back to scalar-FMA exactly like
        # the WMMA paths (no SM80 fp32-input tensor-core atom).
        # The combined "force_grouped_cutlass_mvmr_vvor" mode ALSO routes
        # vvor here. mvmr's backward computes grad_a via a vvor call
        # (_backward_…sparse_matrix_vector_multiplication_reduction), so
        # under the combined mode grad_a hits CUTLASS vvor here while mvmr
        # fwd+grad_b hit CUTLASS mvmr in mvmr_triton.py.
        #        # The fallback must fire when the operands are not a matched
        # CUTLASS-supported pair, not only when `a` is fp32. Under
        # `torch.autocast(fp16)` the conv weight Parameter stays fp32 at the
        # dispatch boundary, so grad_a arrives as a=fp16 (autocast-produced
        # grad) but b=fp32. An `a.dtype == float32`-only guard would let that
        # MIXED (a=fp16, b=fp32) pair through to the CUTLASS kernel, which
        # raises. The CUTLASS kernel supports fp16 AND bf16 (matched
        # operands), so the CUTLASS route is taken iff both operands share a
        # dtype in {fp16, bf16}; any other pair (mixed, or fp32) takes the
        # scalar-FMA fallback. No silent cast (that would change the user's
        # autocast numerics).
        #        # The scalar fallback (`vvor_grouped_cuda`) itself requires a and b
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

    a = a.contiguous()
    b = b.contiguous()

    T, G, M, C = a_idx.numel(), a.shape[1], a.shape[2], b.shape[2]
    input_dtype = a.dtype
    # Accumulate in fp32 for numerical stability, cast back after
    o = torch.zeros((n_o, G, M, C), dtype=torch.float32, device=a.device)

    if not _try_grouped_dispatch(a, a_idx, b, b_idx, o, o_idx, T, G, M, C, n_o):
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


def _backward_sparse_vector_vector_outer_product_reduction(ctx, grad):
    from .mvmr_triton import sparse_matrix_vector_multiplication_reduction

    a, a_idx, b, b_idx, o_idx = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = sparse_matrix_vector_multiplication_reduction(
            grad.transpose(2, 3),
            o_idx,
            b,
            b_idx,
            a_idx,
            a.shape[0] if isinstance(a, torch.Tensor) else ctx.a_shape_0,
        )
    if ctx.needs_input_grad[2]:
        grad_b = sparse_matrix_vector_multiplication_reduction(
            grad,
            o_idx,
            a,
            a_idx,
            b_idx,
            b.shape[0] if isinstance(b, torch.Tensor) else ctx.b_shape_0,
        )
    return grad_a, None, grad_b, None, None, None


def _setup_context_sparse_vector_vector_outer_product_reduction(ctx, inputs, output):
    a, a_idx, b, b_idx, o_idx, n = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[2]:
        saved_a = a
    ctx.save_for_backward(saved_a, a_idx, saved_b, b_idx, o_idx)
    ctx.a_shape_0 = a.shape[0]
    ctx.b_shape_0 = b.shape[0]


sparse_vector_vector_outer_product_reduction.register_autograd(
    _backward_sparse_vector_vector_outer_product_reduction,
    setup_context=_setup_context_sparse_vector_vector_outer_product_reduction,
)
