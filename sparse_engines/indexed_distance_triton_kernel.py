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
    key=["n"],
)
@triton.jit
def indexed_distance_kernel_euclidean(
    a,
    a_idx,
    b,
    b_idx,
    o,
    n,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
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
    tl.store(o + offsets, dist, mask=mask_n)


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
    key=["n"],
)
@triton.jit
def indexed_distance_kernel_chebyshev(
    a,
    a_idx,
    b,
    b_idx,
    o,
    n,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
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

    abs_dx = tl.abs(dx)
    abs_dy = tl.abs(dy)
    abs_dz = tl.abs(dz)
    max_xy = tl.maximum(abs_dx, abs_dy)
    dist = tl.maximum(max_xy, abs_dz)
    tl.store(o + offsets, dist, mask=mask_n)
