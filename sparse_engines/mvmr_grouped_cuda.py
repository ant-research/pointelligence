"""Grouped MVMR/VVOR via CUDA C++ kernel with weight reuse.

Dispatch path for the fused grouped kernel (sparse_mvmr_grouped_mma)
that processes triplets grouped by kernel offset with weight reuse.
No global-memory materialization of gathered data (no im2col).
"""

import torch
from torch import Tensor

from ._seg_offs import kernel_offset_segments
import sparse_engines_cuda._C  # ensure TORCH_LIBRARY static initializers run


def sparse_matrix_vector_multiplication_reduction_grouped_cuda(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n_o: int
) -> Tensor:
    """Grouped MVMR via CUDA kernel with weight reuse.

    Preconditions (same as Triton grouped path):
      - a_idx sorted ascending (sort_by="k")
      - G == 1
      - C >= 128 (weight reuse advantage threshold)
    """
    a = a.contiguous()
    b = b.contiguous()

    T = a_idx.numel()
    G = a.shape[1]
    M = a.shape[3]
    C = a.shape[2]
    K_offsets = a.shape[0]
    input_dtype = a.dtype

    if G != 1:
        raise ValueError("Grouped CUDA kernel requires G == 1")
    if not bool((a_idx[1:] >= a_idx[:-1]).all().item()):
        raise ValueError("a_idx must be sorted ascending for grouped path")

    seg_offs = kernel_offset_segments(a_idx, K_offsets)

    # CUDA kernel expects int32 index tensors, int64 seg_offs
    a_idx_i32 = a_idx.to(torch.int32)
    b_idx_i32 = b_idx.to(torch.int32)
    o_idx_i32 = o_idx.to(torch.int32)

    o = torch.ops.sparse_engines_cuda.sparse_mvmr_grouped_mma(
        a, a_idx_i32, b, b_idx_i32, o_idx_i32, seg_offs, n_o
    )

    return o.to(input_dtype) if input_dtype != torch.float32 else o