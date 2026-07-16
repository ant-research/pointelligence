"""Triton selector for one center-nearest representative per sorted grid cell."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(left, right):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SEGMENTS": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SEGMENTS": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SEGMENTS": 256}, num_warps=8, num_stages=1),
    ],
    # Point counts change every iteration. This selector is memory-bound and
    # uses the same configuration across shapes, so tune once per process.
    key=[],
)
@triton.jit
def center_nearest_segment_kernel(
    points,
    grid_inds,
    sorter,
    segment_starts,
    segment_lengths,
    selected,
    num_segments,
    grid_size,
    grid_dims: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
):
    segment = tl.program_id(axis=0) * BLOCK_SEGMENTS + tl.arange(0, BLOCK_SEGMENTS)
    segment_mask = segment < num_segments
    start = tl.load(segment_starts + segment, mask=segment_mask, other=0)
    length = tl.load(segment_lengths + segment, mask=segment_mask, other=0)

    first_index = tl.load(sorter + start, mask=segment_mask, other=0)
    grid_base = first_index * grid_dims
    gx = tl.load(grid_inds + grid_base, mask=segment_mask, other=0)
    gy = tl.load(grid_inds + grid_base + 1, mask=segment_mask, other=0)
    gz = tl.load(grid_inds + grid_base + 2, mask=segment_mask, other=0)
    cx = _mul_rn(gx.to(tl.float32) + 0.5, grid_size)
    cy = _mul_rn(gy.to(tl.float32) + 0.5, grid_size)
    cz = _mul_rn(gz.to(tl.float32) + 0.5, grid_size)

    best_distance = tl.full([BLOCK_SEGMENTS], float("inf"), tl.float32)
    best_index = first_index
    max_length = tl.max(tl.where(segment_mask, length, 0), axis=0)

    for offset in range(max_length):
        active = segment_mask & (offset < length)
        point_index = tl.load(sorter + start + offset, mask=active, other=0)
        point_base = point_index * 3
        px = tl.load(points + point_base, mask=active, other=0.0).to(tl.float32)
        py = tl.load(points + point_base + 1, mask=active, other=0.0).to(tl.float32)
        pz = tl.load(points + point_base + 2, mask=active, other=0.0).to(tl.float32)
        dx = _sub_rn(cx, px)
        dy = _sub_rn(cy, py)
        dz = _sub_rn(cz, pz)
        dx2 = _mul_rn(dx, dx)
        dy2 = _mul_rn(dy, dy)
        dz2 = _mul_rn(dz, dz)
        distance = _add_rn(_add_rn(dx2, dy2), dz2)
        better = active & (
            (distance < best_distance)
            | ((distance == best_distance) & (point_index < best_index))
        )
        best_distance = tl.where(better, distance, best_distance)
        best_index = tl.where(better, point_index, best_index)

    tl.store(selected + segment, best_index, mask=segment_mask)


def center_nearest_segment_indices(
    points: torch.Tensor,
    grid_inds: torch.Tensor,
    sorter: torch.Tensor,
    segment_lengths: torch.Tensor,
    grid_size: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the nearest observed point in every sorted grid-cell segment."""
    if not points.is_cuda:
        raise ValueError("the Triton center-nearest selector requires CUDA tensors")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
    if grid_inds.ndim != 2 or grid_inds.shape[1] not in (3, 4):
        raise ValueError(
            f"grid_inds must have shape [N, 3] or [N, 4], got {tuple(grid_inds.shape)}"
        )

    points = points.contiguous()
    grid_inds = grid_inds.contiguous()
    sorter = sorter.contiguous()
    segment_lengths = segment_lengths.contiguous()
    segment_ends = torch.cumsum(segment_lengths, dim=0)
    segment_starts = segment_ends - segment_lengths
    if out is None:
        out = torch.empty_like(segment_lengths, dtype=sorter.dtype)
    elif out.shape != segment_lengths.shape or out.dtype != sorter.dtype:
        raise ValueError("out must match segment_lengths shape and sorter dtype")

    num_segments = segment_lengths.numel()
    grid = lambda meta: (triton.cdiv(num_segments, meta["BLOCK_SEGMENTS"]),)
    center_nearest_segment_kernel[grid](
        points,
        grid_inds,
        sorter,
        segment_starts,
        segment_lengths,
        out,
        num_segments,
        float(grid_size),
        grid_dims=grid_inds.shape[1],
    )
    return out
