"""Grouped VVOR via CUDA WMMA-direct kernel.

Same algorithm and calling convention as
``sparse_vector_vector_outer_product_reduction_grouped_cuda``
(sparse_engines.vvor_grouped_cuda) but the inner loop uses
``wmma::mma_sync`` on m16n16k16 fp16/bf16 tiles instead of scalar
``__fmaf_rn``. Targets the finding that the vvor 3.5x regression vs
Triton-grouped is compute-bound (scalar FMA vs tensor-core mma) rather
than memory-bound (a composite-sort A/B experiment refuted memory-axis
closure).

Preconditions (in addition to the grouped-CUDA wrapper's):
  - dtype in {fp16, bf16}; fp32 falls back to ``sparse_vvor_grouped_cuda``
  - M % 16 == 0 and C % 16 == 0 (m16n16k16 WMMA atom)
"""

import torch
from torch import Tensor

from ._seg_offs import kernel_offset_segments
import sparse_engines_cuda._C  # ensure TORCH_LIBRARY static initializers run

# Fall-back path for fp32 inputs (WMMA atom is m16n16k16 fp16/bf16 only).
from .vvor_grouped_cuda import (
    sparse_vector_vector_outer_product_reduction_grouped_cuda as _fallback_scalar_fma,
)


def sparse_vector_vector_outer_product_reduction_grouped_wmma(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n_o: int
) -> Tensor:
    """Grouped VVOR via CUDA WMMA-direct kernel with weight reuse + tensor-core mma.

    Calling convention matches sparse_vector_vector_outer_product_reduction:
      - a = grad_output (n_o_points, G, M)
      - a_idx = output-point indices (T,)
      - b = input (N_b, G, C)
      - b_idx = input-point indices (T,)
      - o_idx = kernel-offset indices (T,) — sorted ascending
      - n_o = number of kernel offsets (K_offsets)

    Preconditions:
      - o_idx sorted ascending (sort_by="k")
      - dtype in {fp16, bf16, fp32}; fp32 routes to scalar-FMA grouped path
      - per-group M % 16 == 0 and C % 16 == 0 for the WMMA path

    G >= 1 supported natively: the frozen kernel's warp grid is
    K * G * (M/16) * (C/16) and all loads/stores use G-strided pointer
    math — no per-group loop or repack needed. The old wrapper-level
    ``G == 1`` ValueError was stricter than the kernel. M / C here are
    the PER-GROUP channel counts (a is (N, G, M), b is (N, G, C)), so
    the %16 tile gates below are per-group gates.
    """
    if a.dtype == torch.float32:
        # fp32 not supported by m16n16k16 WMMA atom; dispatch to the
        # scalar-FMA grouped path (which handles all three precisions).
        return _fallback_scalar_fma(a, a_idx, b, b_idx, o_idx, n_o)

    a = a.contiguous()
    b = b.contiguous()

    T = o_idx.numel()
    G = a.shape[1]
    M = a.shape[2]
    C = b.shape[2]
    K_offsets = n_o
    input_dtype = a.dtype

    if M % 16 != 0 or C % 16 != 0:
        raise ValueError(
            "WMMA-direct vvor requires per-group M and C divisible by 16; "
            f"got per-group M={M}, C={C} (G={G})"
        )
    if not bool((o_idx[1:] >= o_idx[:-1]).all().item()):
        raise ValueError("o_idx must be sorted ascending for grouped path")

    seg_offs = kernel_offset_segments(o_idx, K_offsets)

    # Pass int64 indices directly. The kernel accepts int64_t* natively,
    # avoiding the per-call int64→int32 cast overhead that profiling
    # identified as the dominant T-scaling wrapper overhead.
    o = torch.ops.sparse_engines_cuda.sparse_vvor_grouped_wmma(
        a, a_idx, b, b_idx, o_idx, seg_offs, n_o
    )

    return o if o.dtype == input_dtype else o.to(input_dtype)
