import torch
from torch import Tensor

import triton
from torch.library import triton_op, wrap_triton

from .indexed_distance_triton_kernel import (
    indexed_distance_kernel_euclidean,
    indexed_distance_kernel_chebyshev,
)

try:
    from sparse_engines_cuda.ops import indexed_distance as indexed_distance_cuda
    CUDA_EXT_AVAILABLE = True
except ImportError:
    CUDA_EXT_AVAILABLE = False
    indexed_distance_cuda = None


@triton_op(
    "sparse_engines::indexed_distance",
    mutates_args={},
)
def indexed_distance(a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, distance_type: str = "ball") -> Tensor:
    a = a.contiguous()
    b = b.contiguous()

    if CUDA_EXT_AVAILABLE:
        distance_type_int = 0 if distance_type == "ball" else 1
        return indexed_distance_cuda(a, a_idx, b, b_idx, distance_type_int)
    
    n = a_idx.numel()
    distances = torch.empty((n,), dtype=a.dtype, device=a.device)
    
    def grid_fn(META):
        if META is None:
            return (triton.cdiv(n, 128),)
        block_size = META.get("BLOCK_SIZE", 128)
        return (triton.cdiv(n, block_size),)
    
    # Select specialized kernel based on distance_type to avoid runtime branches
    if distance_type == "ball":
        kernel = indexed_distance_kernel_euclidean
    else:  # distance_type == "cube"
        kernel = indexed_distance_kernel_chebyshev
    
    wrap_triton(kernel)[grid_fn](
        a, a_idx, b, b_idx, distances, n
    )
    return distances
