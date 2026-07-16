"""Fused grid-index-to-1D-hash Triton kernel.

Replaces the per-element arithmetic chain in reduce_indices_to_1d:
  (inds - min) * stride → sum(dim=-1)
with a single kernel launch. Reads min/stride from GPU tensor pointers
to avoid GPU→CPU sync.
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 4}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 4}, num_stages=1),
    ],
    # key must be n-independent: point-cloud augmentation produces a fresh
    # `n` every iter, so keying on it re-autotunes every iter (~900 do_bench
    # trials each, 256 MB L2 flush apiece). This is a memory-bound
    # elementwise hash whose optimal (BLOCK_SIZE, num_warps) is effectively
    # n-independent, so a single tuned config (key=[]) is correct and matches
    # the convention in sparse_engines/indexed_distance{,_mask}_triton_kernel.py.
    key=[],
)
@triton.jit
def grid_hash_kernel_4d(
    inds,          # (N, 4) int32 input grid indices, contiguous
    out,           # (N,) output 1D hashes
    inds_min_ptr,  # (4,) per-column minimums
    stride_ptr,    # (4,) per-column strides
    n,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    # Load min and stride constants (broadcast to all threads)
    min0 = tl.load(inds_min_ptr + 0)
    min1 = tl.load(inds_min_ptr + 1)
    min2 = tl.load(inds_min_ptr + 2)
    min3 = tl.load(inds_min_ptr + 3)
    s0 = tl.load(stride_ptr + 0)
    s1 = tl.load(stride_ptr + 1)
    s2 = tl.load(stride_ptr + 2)
    s3 = tl.load(stride_ptr + 3)

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    base = offsets * 4
    c0 = tl.load(inds + base + 0, mask=mask, other=0)
    c1 = tl.load(inds + base + 1, mask=mask, other=0)
    c2 = tl.load(inds + base + 2, mask=mask, other=0)
    c3 = tl.load(inds + base + 3, mask=mask, other=0)

    h = (c0 - min0) * s0 + (c1 - min1) * s1 + (c2 - min2) * s2 + (c3 - min3) * s3
    tl.store(out + offsets, h, mask=mask)
