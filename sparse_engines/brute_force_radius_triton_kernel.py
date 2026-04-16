"""Tiled brute-force radius search Triton kernels (PyKeOPS-style).

Two-pass approach that never materializes the full Q×P distance matrix:
  Pass 1 (count): for each query, count points within radius
  Pass 2 (compact): recompute distances, write matching pairs to output

Each query is one Triton program (1 thread-block per query). The program
iterates over points in tiles of BLOCK_P, computing distances in registers.
"""

import triton
import triton.language as tl


@triton.jit
def _tiled_radius_count_kernel(
    queries,            # (Q, 3) float32
    points,             # (P, 3) float32
    q_batch_starts,     # (Q,) int64 — start of point range for each query's batch
    q_batch_ends,       # (Q,) int64 — end of point range for each query's batch
    num_neighbors,      # (Q,) int32 output — neighbor count per query
    radius_sq,          # float32 — radius * radius
    num_queries,        # int
    BLOCK_P: tl.constexpr,
):
    """Pass 1: count neighbors per query using tiled distance computation."""
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return

    # Load query coordinates
    qx = tl.load(queries + q_idx * 3 + 0)
    qy = tl.load(queries + q_idx * 3 + 1)
    qz = tl.load(queries + q_idx * 3 + 2)

    # Batch boundaries for this query
    p_start = tl.load(q_batch_starts + q_idx)
    p_end = tl.load(q_batch_ends + q_idx)

    count = tl.zeros([], dtype=tl.int32)

    # Tile over points in this query's batch
    p_off = p_start
    while p_off < p_end:
        p_indices = p_off + tl.arange(0, BLOCK_P)
        p_mask = p_indices < p_end

        # Load point coordinates
        px = tl.load(points + p_indices * 3 + 0, mask=p_mask, other=0.0)
        py = tl.load(points + p_indices * 3 + 1, mask=p_mask, other=0.0)
        pz = tl.load(points + p_indices * 3 + 2, mask=p_mask, other=0.0)

        # Compute squared distances
        dx = qx - px
        dy = qy - py
        dz = qz - pz
        dist_sq = dx * dx + dy * dy + dz * dz

        # Count points within radius
        within = (dist_sq <= radius_sq) & p_mask
        count += tl.sum(within.to(tl.int32))

        p_off += BLOCK_P

    tl.store(num_neighbors + q_idx, count)


@triton.jit
def _tiled_radius_compact_kernel(
    queries,            # (Q, 3) float32
    points,             # (P, 3) float32
    q_batch_starts,     # (Q,) int64
    q_batch_ends,       # (Q,) int64
    write_offsets,      # (Q,) int64 — exclusive cumsum of num_neighbors
    out_neighbors,      # (N_total,) int32 output
    out_distances,      # (N_total,) float32 output (or dummy if not needed)
    radius_sq,          # float32
    num_queries,        # int
    RETURN_DIST: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Pass 2: recompute distances and write matching pairs to output.

    Uses cumsum-based compaction within each BLOCK_P tile to avoid
    serial loops.
    """
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return

    qx = tl.load(queries + q_idx * 3 + 0)
    qy = tl.load(queries + q_idx * 3 + 1)
    qz = tl.load(queries + q_idx * 3 + 2)

    p_start = tl.load(q_batch_starts + q_idx)
    p_end = tl.load(q_batch_ends + q_idx)
    base_offset = tl.load(write_offsets + q_idx)

    written = 0

    p_off = p_start
    while p_off < p_end:
        p_indices = p_off + tl.arange(0, BLOCK_P)
        p_mask = p_indices < p_end

        px = tl.load(points + p_indices * 3 + 0, mask=p_mask, other=0.0)
        py = tl.load(points + p_indices * 3 + 1, mask=p_mask, other=0.0)
        pz = tl.load(points + p_indices * 3 + 2, mask=p_mask, other=0.0)

        dx = qx - px
        dy = qy - py
        dz = qz - pz
        dist_sq = dx * dx + dy * dy + dz * dz

        within = (dist_sq <= radius_sq) & p_mask
        within_i32 = within.to(tl.int32)

        # Prefix sum for compaction: compute write position for each match
        # cumsum gives [1-based positions], subtract 1 for 0-based, add written count
        tile_cumsum = tl.cumsum(within_i32, axis=0)  # [0..BLOCK_P], each is count of matches up to that index
        tile_count = tl.sum(within_i32)

        # Write position for each element: base_offset + written + (cumsum - 1)
        write_pos = base_offset + written + tile_cumsum.to(tl.int64) - 1
        write_mask = within

        tl.store(out_neighbors + write_pos, p_indices.to(tl.int32), mask=write_mask)
        if RETURN_DIST:
            dist = tl.sqrt(dist_sq)
            tl.store(out_distances + write_pos, dist, mask=write_mask)

        written += tile_count.to(tl.int32)
        p_off += BLOCK_P


@triton.jit
def _tiled_radius_count_kernel_chebyshev(
    queries, points, q_batch_starts, q_batch_ends,
    num_neighbors, radius, num_queries,
    BLOCK_P: tl.constexpr,
):
    """Pass 1 count — Chebyshev distance."""
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return

    qx = tl.load(queries + q_idx * 3 + 0)
    qy = tl.load(queries + q_idx * 3 + 1)
    qz = tl.load(queries + q_idx * 3 + 2)

    p_start = tl.load(q_batch_starts + q_idx)
    p_end = tl.load(q_batch_ends + q_idx)

    count = tl.zeros([], dtype=tl.int32)

    p_off = p_start
    while p_off < p_end:
        p_indices = p_off + tl.arange(0, BLOCK_P)
        p_mask = p_indices < p_end

        px = tl.load(points + p_indices * 3 + 0, mask=p_mask, other=0.0)
        py = tl.load(points + p_indices * 3 + 1, mask=p_mask, other=0.0)
        pz = tl.load(points + p_indices * 3 + 2, mask=p_mask, other=0.0)

        dist = tl.maximum(tl.maximum(tl.abs(qx - px), tl.abs(qy - py)), tl.abs(qz - pz))
        within = (dist <= radius) & p_mask
        count += tl.sum(within.to(tl.int32))

        p_off += BLOCK_P

    tl.store(num_neighbors + q_idx, count)


@triton.jit
def _tiled_radius_compact_kernel_chebyshev(
    queries, points, q_batch_starts, q_batch_ends,
    write_offsets, out_neighbors, out_distances,
    radius, num_queries,
    RETURN_DIST: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Pass 2 compact — Chebyshev distance."""
    q_idx = tl.program_id(0)
    if q_idx >= num_queries:
        return

    qx = tl.load(queries + q_idx * 3 + 0)
    qy = tl.load(queries + q_idx * 3 + 1)
    qz = tl.load(queries + q_idx * 3 + 2)

    p_start = tl.load(q_batch_starts + q_idx)
    p_end = tl.load(q_batch_ends + q_idx)
    base_offset = tl.load(write_offsets + q_idx)

    written = 0

    p_off = p_start
    while p_off < p_end:
        p_indices = p_off + tl.arange(0, BLOCK_P)
        p_mask = p_indices < p_end

        px = tl.load(points + p_indices * 3 + 0, mask=p_mask, other=0.0)
        py = tl.load(points + p_indices * 3 + 1, mask=p_mask, other=0.0)
        pz = tl.load(points + p_indices * 3 + 2, mask=p_mask, other=0.0)

        dist = tl.maximum(tl.maximum(tl.abs(qx - px), tl.abs(qy - py)), tl.abs(qz - pz))
        within = (dist <= radius) & p_mask
        within_i32 = within.to(tl.int32)

        tile_cumsum = tl.cumsum(within_i32, axis=0)
        tile_count = tl.sum(within_i32)

        write_pos = base_offset + written + tile_cumsum.to(tl.int64) - 1
        write_mask = within

        tl.store(out_neighbors + write_pos, p_indices.to(tl.int32), mask=write_mask)
        if RETURN_DIST:
            tl.store(out_distances + write_pos, dist, mask=write_mask)

        written += tile_count.to(tl.int32)
        p_off += BLOCK_P
