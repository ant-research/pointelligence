"""Triton kernels for the strided-grid radius search inner loop.

Two-pass count+compact (mirrors `brute_force_radius_triton_kernel.py`),
specialized for the strided grid-downsample algorithm: each query has
``num_offsets`` candidate-cell ranges (each a [start, end) into the
sorted-input table), and we filter by Euclidean distance ≤ radius.

The fused kernels avoid materializing the O(total-candidates) flat
``(q_idx, p_idx)`` array that the PyTorch-vectorized path produces, which
is the main bottleneck at K=1 small-R (PT-v3 stem regime — many small
cells, lots of indexing overhead per cell).

Distance-filter semantics match ``indexed_distance_mask_triton_kernel``:
``sqrt(dx² + dy² + dz²) <= radius`` (NOT ``dist_sq <= radius_sq``) so
boundary membership remains stable at fp32
boundary points.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _distance_3d(qx, qy, qz, px, py, pz, DISTANCE_TYPE: tl.constexpr):
    dx = qx - px
    dy = qy - py
    dz = qz - pz
    if DISTANCE_TYPE == 0:
        return tl.sqrt(dx * dx + dy * dy + dz * dz)
    return tl.maximum(tl.maximum(tl.abs(dx), tl.abs(dy)), tl.abs(dz))


@triton.jit
def _offset_vector_radius_count_kernel(
    queries, points, sort_perm, start_idx, end_idx, cell_counts,
    radius, num_queries, num_offsets: tl.constexpr,
    DISTANCE_TYPE: tl.constexpr, BLOCK_O: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Count all candidate cells for one query in a two-dimensional tile."""
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return
    off = tl.arange(0, BLOCK_O)
    lane = tl.arange(0, BLOCK_P)
    off_mask = off < num_offsets
    base = q_idx * num_offsets
    p_base = tl.load(start_idx + base + off, mask=off_mask, other=0).to(tl.int64)
    p_end = tl.load(end_idx + base + off, mask=off_mask, other=0).to(tl.int64)
    counts = tl.zeros((BLOCK_O,), dtype=tl.int32)
    qx = tl.load(queries + q_idx * 3 + 0)
    qy = tl.load(queries + q_idx * 3 + 1)
    qz = tl.load(queries + q_idx * 3 + 2)
    while tl.sum(((p_base < p_end) & off_mask).to(tl.int32), axis=0) > 0:
        p_positions = p_base[:, None] + lane[None, :].to(tl.int64)
        p_mask = off_mask[:, None] & (p_positions < p_end[:, None])
        p_indices = tl.load(
            sort_perm + p_positions, mask=p_mask, other=0).to(tl.int64)
        px = tl.load(points + p_indices * 3 + 0, mask=p_mask, other=0.0)
        py = tl.load(points + p_indices * 3 + 1, mask=p_mask, other=0.0)
        pz = tl.load(points + p_indices * 3 + 2, mask=p_mask, other=0.0)
        dist = _distance_3d(
            qx, qy, qz, px, py, pz, DISTANCE_TYPE=DISTANCE_TYPE)
        counts += tl.sum(((dist <= radius) & p_mask).to(tl.int32), axis=1)
        p_base += BLOCK_P
    tl.store(cell_counts + base + off, counts, mask=off_mask)


@triton.jit
def _triplet_kernel_index(
    qx, qy, qz, px, py, pz, kernel_grid_size,
    KERNEL_SIZE: tl.constexpr,
):
    """Match voxelize_3d's round-to-even, clamp, and row-major tap index."""
    half = KERNEL_SIZE // 2
    dx = libdevice.rint(libdevice.div_rn(
        px.to(tl.float32) - qx.to(tl.float32),
        kernel_grid_size,
    )).to(tl.int32)
    dy = libdevice.rint(libdevice.div_rn(
        py.to(tl.float32) - qy.to(tl.float32),
        kernel_grid_size,
    )).to(tl.int32)
    dz = libdevice.rint(libdevice.div_rn(
        pz.to(tl.float32) - qz.to(tl.float32),
        kernel_grid_size,
    )).to(tl.int32)
    kx = tl.minimum(tl.maximum(dx + half, 0), KERNEL_SIZE - 1)
    ky = tl.minimum(tl.maximum(dy + half, 0), KERNEL_SIZE - 1)
    kz = tl.minimum(tl.maximum(dz + half, 0), KERNEL_SIZE - 1)
    return (kx * KERNEL_SIZE + ky) * KERNEL_SIZE + kz


@triton.jit
def _offset_vector_triplet_count_kernel(
    queries, points, sort_perm, start_idx, end_idx, cell_counts, tap_counts,
    radius, kernel_grid_size, num_queries, num_offsets: tl.constexpr,
    DISTANCE_TYPE: tl.constexpr, KERNEL_SIZE: tl.constexpr,
    TAP_STRIPES: tl.constexpr, BLOCK_O: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Count per query-cell edges and final kernel-tap segments in one pass."""
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return
    off = tl.arange(0, BLOCK_O)
    lane = tl.arange(0, BLOCK_P)
    off_mask = off < num_offsets
    base = q_idx * num_offsets
    p_base = tl.load(start_idx + base + off, mask=off_mask, other=0).to(tl.int64)
    p_end = tl.load(end_idx + base + off, mask=off_mask, other=0).to(tl.int64)
    counts = tl.zeros((BLOCK_O,), dtype=tl.int32)
    qx = tl.load(queries + q_idx * 3 + 0)
    qy = tl.load(queries + q_idx * 3 + 1)
    qz = tl.load(queries + q_idx * 3 + 2)
    while tl.sum(((p_base < p_end) & off_mask).to(tl.int32), axis=0) > 0:
        p_positions = p_base[:, None] + lane[None, :].to(tl.int64)
        p_mask = off_mask[:, None] & (p_positions < p_end[:, None])
        p_indices = tl.load(
            sort_perm + p_positions, mask=p_mask, other=0).to(tl.int64)
        px = tl.load(points + p_indices * 3 + 0, mask=p_mask, other=0.0)
        py = tl.load(points + p_indices * 3 + 1, mask=p_mask, other=0.0)
        pz = tl.load(points + p_indices * 3 + 2, mask=p_mask, other=0.0)
        dist = _distance_3d(
            qx, qy, qz, px, py, pz, DISTANCE_TYPE=DISTANCE_TYPE)
        within = (dist <= radius) & p_mask
        counts += tl.sum(within.to(tl.int32), axis=1)
        tap = _triplet_kernel_index(
            qx, qy, qz, px, py, pz, kernel_grid_size,
            KERNEL_SIZE=KERNEL_SIZE)
        stripe = q_idx % TAP_STRIPES
        tl.atomic_add(
            tap_counts + stripe * (KERNEL_SIZE ** 3) + tap,
            1,
            mask=within,
        )
        p_base += BLOCK_P
    tl.store(cell_counts + base + off, counts, mask=off_mask)


@triton.jit
def _offset_vector_radius_reduce_kernel(
    cell_offsets, num_neighbors, num_queries,
    num_offsets: tl.constexpr, BLOCK_O: tl.constexpr,
):
    """Convert per-cell counts to per-query relative offsets in place."""
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return
    off = tl.arange(0, BLOCK_O)
    mask = off < num_offsets
    base = q_idx * num_offsets
    counts = tl.load(cell_offsets + base + off, mask=mask, other=0).to(tl.int32)
    inclusive = tl.cumsum(counts, axis=0)
    tl.store(cell_offsets + base + off, inclusive - counts, mask=mask)
    tl.store(num_neighbors + q_idx, tl.sum(counts, axis=0))


@triton.jit
def _offset_vector_radius_compact_kernel(
    queries, points, sort_perm, start_idx, end_idx, cell_offsets,
    query_offsets, out_neighbors, out_distances,
    radius, num_queries, num_offsets: tl.constexpr,
    RETURN_DIST: tl.constexpr, DISTANCE_TYPE: tl.constexpr,
    BLOCK_O: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """Fill all cells for one query from a two-dimensional candidate tile."""
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return
    off = tl.arange(0, BLOCK_O)
    lane = tl.arange(0, BLOCK_P)
    off_mask = off < num_offsets
    base = q_idx * num_offsets
    p_base = tl.load(start_idx + base + off, mask=off_mask, other=0).to(tl.int64)
    p_end = tl.load(end_idx + base + off, mask=off_mask, other=0).to(tl.int64)
    cell_base = tl.load(
        cell_offsets + base + off, mask=off_mask, other=0).to(tl.int64)
    query_base = tl.load(query_offsets + q_idx).to(tl.int64)
    written = tl.zeros((BLOCK_O,), dtype=tl.int32)
    qx = tl.load(queries + q_idx * 3 + 0)
    qy = tl.load(queries + q_idx * 3 + 1)
    qz = tl.load(queries + q_idx * 3 + 2)
    while tl.sum(((p_base < p_end) & off_mask).to(tl.int32), axis=0) > 0:
        p_positions = p_base[:, None] + lane[None, :].to(tl.int64)
        p_mask = off_mask[:, None] & (p_positions < p_end[:, None])
        p_indices = tl.load(
            sort_perm + p_positions, mask=p_mask, other=0).to(tl.int64)
        px = tl.load(points + p_indices * 3 + 0, mask=p_mask, other=0.0)
        py = tl.load(points + p_indices * 3 + 1, mask=p_mask, other=0.0)
        pz = tl.load(points + p_indices * 3 + 2, mask=p_mask, other=0.0)
        dist = _distance_3d(
            qx, qy, qz, px, py, pz, DISTANCE_TYPE=DISTANCE_TYPE)
        within = (dist <= radius) & p_mask
        within_i32 = within.to(tl.int32)
        tile_cumsum = tl.cumsum(within_i32, axis=1)
        write_pos = (
            query_base + cell_base[:, None] + written[:, None]
            + tile_cumsum.to(tl.int64) - 1)
        tl.store(out_neighbors + write_pos, p_indices, mask=within)
        if RETURN_DIST:
            tl.store(out_distances + write_pos, dist, mask=within)
        written += tl.sum(within_i32, axis=1)
        p_base += BLOCK_P


@triton.jit
def _offset_vector_triplet_compact_kernel(
    queries, points, sort_perm, start_idx, end_idx, tap_cursors,
    out_i, out_j, radius, kernel_grid_size, num_queries,
    num_offsets: tl.constexpr, DISTANCE_TYPE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr, TAP_STRIPES: tl.constexpr,
    BLOCK_O: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """Write accepted edges directly into their final kernel-tap segments."""
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return
    off = tl.arange(0, BLOCK_O)
    lane = tl.arange(0, BLOCK_P)
    off_mask = off < num_offsets
    base = q_idx * num_offsets
    p_base = tl.load(start_idx + base + off, mask=off_mask, other=0).to(tl.int64)
    p_end = tl.load(end_idx + base + off, mask=off_mask, other=0).to(tl.int64)
    qx = tl.load(queries + q_idx * 3 + 0)
    qy = tl.load(queries + q_idx * 3 + 1)
    qz = tl.load(queries + q_idx * 3 + 2)
    while tl.sum(((p_base < p_end) & off_mask).to(tl.int32), axis=0) > 0:
        p_positions = p_base[:, None] + lane[None, :].to(tl.int64)
        p_mask = off_mask[:, None] & (p_positions < p_end[:, None])
        p_indices = tl.load(
            sort_perm + p_positions, mask=p_mask, other=0).to(tl.int64)
        px = tl.load(points + p_indices * 3 + 0, mask=p_mask, other=0.0)
        py = tl.load(points + p_indices * 3 + 1, mask=p_mask, other=0.0)
        pz = tl.load(points + p_indices * 3 + 2, mask=p_mask, other=0.0)
        dist = _distance_3d(
            qx, qy, qz, px, py, pz, DISTANCE_TYPE=DISTANCE_TYPE)
        within = (dist <= radius) & p_mask
        tap = _triplet_kernel_index(
            qx, qy, qz, px, py, pz, kernel_grid_size,
            KERNEL_SIZE=KERNEL_SIZE)
        stripe = q_idx % TAP_STRIPES
        cursor = stripe * (KERNEL_SIZE ** 3) + tap
        write_pos = tl.atomic_add(tap_cursors + cursor, 1, mask=within)
        tl.store(out_i + write_pos, q_idx, mask=within)
        tl.store(out_j + write_pos, p_indices, mask=within)
        p_base += BLOCK_P
