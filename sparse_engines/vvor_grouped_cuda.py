"""Grouped VVOR via CUDA C++ kernel with weight reuse.

Dispatch path for the fused grouped kernel (sparse_vvor_grouped_mma)
that processes triplets grouped by kernel offset with weight reuse.
No global-memory materialization of gathered data (no im2col).
"""

import torch
from torch import Tensor

from ._seg_offs import kernel_offset_segments
import sparse_engines_cuda._C  # ensure TORCH_LIBRARY static initializers run


def sparse_vector_vector_outer_product_reduction_grouped_cuda(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n_o: int
) -> Tensor:
    """Grouped VVOR via CUDA kernel with weight reuse.

    Computes grad_weight[k] += grad_output[i] (outer) input[j] for triplets
    grouped by kernel offset. No global-memory materialization.

    Calling convention matches sparse_vector_vector_outer_product_reduction:
      - a = grad_output (n_o_points, G, M)
      - a_idx = output-point indices (T,)
      - b = input (N_b, G, C)
      - b_idx = input-point indices (T,)
      - o_idx = kernel-offset indices (T,) — sorted ascending
      - n_o = number of kernel offsets (K_offsets)

    Preconditions:
      - o_idx sorted ascending (sort_by="k")

    G >= 1 supported natively: the frozen kernel decodes (seg_k, g, mt,
    cw) from its warp grid and indexes grad_output (N, G, M) / input
    (N, G, C) / grad_weight (K, G, M, C) with G-strided pointer math —
    no per-group loop or repack needed. The old wrapper-level ``G == 1``
    ValueError was stricter than the kernel.
    """
    a = a.contiguous()
    b = b.contiguous()

    T = o_idx.numel()
    G = a.shape[1]
    M = a.shape[2]
    C = b.shape[2]
    K_offsets = n_o
    input_dtype = a.dtype

    if not bool((o_idx[1:] >= o_idx[:-1]).all().item()):
        raise ValueError("o_idx must be sorted ascending for grouped path")

    seg_offs = kernel_offset_segments(o_idx, K_offsets)

    # CUDA kernel expects int32 index tensors, int64 seg_offs
    a_idx_i32 = a_idx.to(torch.int32)
    b_idx_i32 = b_idx.to(torch.int32)
    o_idx_i32 = o_idx.to(torch.int32)

    o = torch.ops.sparse_engines_cuda.sparse_vvor_grouped_mma(
        a, a_idx_i32, b, b_idx_i32, o_idx_i32, seg_offs, n_o
    )

    return o.to(input_dtype) if input_dtype != torch.float32 else o