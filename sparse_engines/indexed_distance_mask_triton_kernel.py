"""Fused indexed-distance-mask Triton kernel.

Single kernel that computes pairwise distance and outputs a boolean mask
(dist <= radius) directly, avoiding the intermediate float distances tensor
and the separate comparison kernel.
"""

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32, "num_warps": 1}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64, "num_warps": 1}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 1}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 1}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 32, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 32, "num_warps": 4}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64, "num_warps": 4}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 4}, num_stages=1),
    ],
    key=[],
)
@triton.jit
def indexed_distance_mask_kernel(
    a,
    a_idx,
    b,
    b_idx,
    out_mask,  # (n,) bool output
    radius,    # float: search radius
    n,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Compute Euclidean distance and output bool mask (dist <= radius) in one pass."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_n = offsets < n
    a_offsets = tl.load(a_idx + offsets, mask=mask_n, other=0)
    b_offsets = tl.load(b_idx + offsets, mask=mask_n, other=0)

    ax = tl.load(a + a_offsets * 3 + 0, mask=mask_n)
    ay = tl.load(a + a_offsets * 3 + 1, mask=mask_n)
    az = tl.load(a + a_offsets * 3 + 2, mask=mask_n)

    bx = tl.load(b + b_offsets * 3 + 0, mask=mask_n)
    by = tl.load(b + b_offsets * 3 + 1, mask=mask_n)
    bz = tl.load(b + b_offsets * 3 + 2, mask=mask_n)

    dx = ax - bx
    dy = ay - by
    dz = az - bz

    dist = tl.sqrt(dx * dx + dy * dy + dz * dz)
    within = dist <= radius
    tl.store(out_mask + offsets, within, mask=mask_n)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32, "num_warps": 1}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64, "num_warps": 1}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 1}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 1}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 32, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 2}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 32, "num_warps": 4}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64, "num_warps": 4}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 4}, num_stages=1),
    ],
    key=[],
)
@triton.jit
def indexed_distance_mask_and_dist_kernel(
    a,
    a_idx,
    b,
    b_idx,
    out_mask,   # (n,) bool output
    out_dist,   # (n,) float output (distances, only valid where mask=True)
    radius,     # float: search radius
    n,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Compute distance, output bool mask AND distances in one pass."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_n = offsets < n
    a_offsets = tl.load(a_idx + offsets, mask=mask_n, other=0)
    b_offsets = tl.load(b_idx + offsets, mask=mask_n, other=0)

    ax = tl.load(a + a_offsets * 3 + 0, mask=mask_n)
    ay = tl.load(a + a_offsets * 3 + 1, mask=mask_n)
    az = tl.load(a + a_offsets * 3 + 2, mask=mask_n)

    bx = tl.load(b + b_offsets * 3 + 0, mask=mask_n)
    by = tl.load(b + b_offsets * 3 + 1, mask=mask_n)
    bz = tl.load(b + b_offsets * 3 + 2, mask=mask_n)

    dx = ax - bx
    dy = ay - by
    dz = az - bz

    dist = tl.sqrt(dx * dx + dy * dy + dz * dz)
    within = dist <= radius
    tl.store(out_mask + offsets, within, mask=mask_n)
    tl.store(out_dist + offsets, dist, mask=mask_n & within)


# --- Chebyshev (L-inf) variants ---

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 32, "num_warps": 1}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 64, "num_warps": 1}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 128, "num_warps": 1}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 256, "num_warps": 1}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 32, "num_warps": 2}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 64, "num_warps": 2}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 128, "num_warps": 2}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 256, "num_warps": 2}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 32, "num_warps": 4}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 64, "num_warps": 4}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 128, "num_warps": 4}, num_stages=1),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=[])
@triton.jit
def indexed_distance_mask_kernel_chebyshev(
    a,
    a_idx,
    b,
    b_idx,
    out_mask,
    radius,
    n,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Compute Chebyshev distance and output bool mask in one pass."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_n = offsets < n
    a_offsets = tl.load(a_idx + offsets, mask=mask_n, other=0)
    b_offsets = tl.load(b_idx + offsets, mask=mask_n, other=0)

    ax = tl.load(a + a_offsets * 3 + 0, mask=mask_n)
    ay = tl.load(a + a_offsets * 3 + 1, mask=mask_n)
    az = tl.load(a + a_offsets * 3 + 2, mask=mask_n)

    bx = tl.load(b + b_offsets * 3 + 0, mask=mask_n)
    by = tl.load(b + b_offsets * 3 + 1, mask=mask_n)
    bz = tl.load(b + b_offsets * 3 + 2, mask=mask_n)

    dist = tl.maximum(tl.maximum(tl.abs(ax - bx), tl.abs(ay - by)), tl.abs(az - bz))
    within = dist <= radius
    tl.store(out_mask + offsets, within, mask=mask_n)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=[])
@triton.jit
def indexed_distance_mask_and_dist_kernel_chebyshev(
    a,
    a_idx,
    b,
    b_idx,
    out_mask,
    out_dist,
    radius,
    n,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Compute Chebyshev distance, output bool mask AND distances in one pass."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_n = offsets < n
    a_offsets = tl.load(a_idx + offsets, mask=mask_n, other=0)
    b_offsets = tl.load(b_idx + offsets, mask=mask_n, other=0)

    ax = tl.load(a + a_offsets * 3 + 0, mask=mask_n)
    ay = tl.load(a + a_offsets * 3 + 1, mask=mask_n)
    az = tl.load(a + a_offsets * 3 + 2, mask=mask_n)

    bx = tl.load(b + b_offsets * 3 + 0, mask=mask_n)
    by = tl.load(b + b_offsets * 3 + 1, mask=mask_n)
    bz = tl.load(b + b_offsets * 3 + 2, mask=mask_n)

    dist = tl.maximum(tl.maximum(tl.abs(ax - bx), tl.abs(ay - by)), tl.abs(az - bz))
    within = dist <= radius
    tl.store(out_mask + offsets, within, mask=mask_n)
    tl.store(out_dist + offsets, dist, mask=mask_n & within)
