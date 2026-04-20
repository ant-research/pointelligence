"""GPU-accelerated radius-based neighborhood search.

Finds all points within a fixed radius of each query point, producing
(i, j) pairs that represent the sparse neighborhood graph. Supports
batched point clouds via per-sample confinement.
"""

import torch

from .constants import Constants
from .grid_lookup import build_lookup_struct
from .grid_lookup import compute_grid_indices
from .grid_lookup import query_lookup_struct
from .indexing import arange_cached
from .indexing import cumsum_exclusive
from .indexing import arrange_indices
from .indexing import repeat_interleave_indices

from sparse_engines.indexed_distance_mask_triton_kernel import (
    indexed_distance_mask_kernel,
    indexed_distance_mask_and_dist_kernel,
    indexed_distance_mask_kernel_chebyshev,
    indexed_distance_mask_and_dist_kernel_chebyshev,
)
from sparse_engines.brute_force_radius_triton_kernel import (
    _tiled_radius_count_kernel,
    _tiled_radius_compact_kernel,
    _tiled_radius_count_kernel_chebyshev,
    _tiled_radius_compact_kernel_chebyshev,
)
import triton


def clamp_by_radius(
    queries,
    q_inds,
    points,
    p_inds,
    radius,
    return_distances=False,
    dtype_num_neighbors=torch.int64,
    distance_type="ball",
):
    """Filter candidate (query, point) pairs to those within a given radius.

    Takes flat arrays of candidate pair indices produced by grid lookup and
    computes exact distances, keeping only pairs where dist <= radius.

    Args:
        queries: (Q, 3) query point coordinates.
        q_inds: (C,) query index for each candidate pair.
        points: (P, 3) reference point coordinates.
        p_inds: (C,) point index for each candidate pair.
        radius: scalar distance threshold.
        return_distances: if True, also return per-neighbor distances.
        dtype_num_neighbors: dtype for the per-query neighbor count tensor.
        distance_type: "ball" (Euclidean) or "chebyshev" (L-inf).

    Returns:
        neighbors: (N,) point indices of confirmed neighbors.
        num_neighbors: (Q,) neighbor count per query.
        distances (optional): (N,) distances for confirmed neighbors.
    """
    n = q_inds.numel()
    device = queries.device

    # Select fused kernel pair based on distance type
    if distance_type == "ball":
        mask_kernel = indexed_distance_mask_kernel
        mask_and_dist_kernel = indexed_distance_mask_and_dist_kernel
    else:
        mask_kernel = indexed_distance_mask_kernel_chebyshev
        mask_and_dist_kernel = indexed_distance_mask_and_dist_kernel_chebyshev

    # Fused kernel: compute distance + mask in one pass, saving the
    # intermediate (C,) float32 distances tensor and a separate comparison.
    mask = torch.empty((n,), dtype=torch.bool, device=device)
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
    if return_distances:
        distances_all = torch.empty((n,), dtype=queries.dtype, device=device)
        mask_and_dist_kernel[grid](
            queries, q_inds, points, p_inds, mask, distances_all, radius, n)
        neighbors_indices = torch.nonzero(mask).squeeze(-1)
        del mask
        distances = distances_all[neighbors_indices]
        del distances_all
    else:
        mask_kernel[grid](
            queries, q_inds, points, p_inds, mask, radius, n)
        neighbors_indices = torch.nonzero(mask).squeeze(-1)
        del mask
        distances = None

    neighbors = p_inds[neighbors_indices]
    del p_inds

    q_inds_masked = q_inds[neighbors_indices]
    del q_inds, neighbors_indices

    num_queries = queries.shape[0]
    num_neighbors = torch.zeros((num_queries,), dtype=torch.int32, device=points.device)
    num_neighbors.index_add_(
        dim=0,
        index=q_inds_masked,
        source=Constants.get_one(points.device, torch.int32).expand(
            q_inds_masked.numel()
        ),
    )
    del q_inds_masked
    num_neighbors = num_neighbors.to(dtype_num_neighbors)

    if return_distances:
        return neighbors, num_neighbors, distances
    else:
        return neighbors, num_neighbors


def radius_search_lookup(
    points,
    queries,
    radius,
    sample_inds=None,
    query_sample_inds=None,
    return_distances=False,
    dtype_num_neighbors=torch.int64,
    distance_type="ball",
    max_candidate_mem_bytes=2 * 1024**3,
):
    """Grid-accelerated fixed-radius neighbor search.

    Uses a shifted spatial hash grid to find candidate pairs, then filters
    by exact distance. The algorithm:
      1. Hash both points and queries into a uniform grid (cell size = 2*radius).
         The smaller set gets 8 half-cell shifts to guarantee coverage across
         cell boundaries — any true neighbor pair shares a cell in >= 1 shift.
      2. Build a lookup structure (sorted unique hashes) on whichever set
         produces fewer grid entries; the other set queries into it.
      3. Expand per-cell matches into flat (q_ind, p_ind) candidate arrays.
      4. Compute exact distances via Triton kernel and keep pairs <= radius.

    Args:
        points: (P, 3) reference point coordinates.
        queries: (Q, 3) query point coordinates.
        radius: search radius.
        sample_inds: (P,) int batch index per point, or None for single-batch.
        query_sample_inds: (Q,) int batch index per query, or None.
        return_distances: if True, also return per-neighbor distances.
        dtype_num_neighbors: dtype for per-query neighbor counts.
        distance_type: "ball" (Euclidean) or "chebyshev" (L-inf).

    Returns:
        Same as clamp_by_radius: (neighbors, num_neighbors[, distances]).
    """
    if sample_inds is not None:
        assert query_sample_inds is not None
    else:
        assert query_sample_inds is None

    device = points.device
    num_points, num_queries = points.shape[0], queries.shape[0]

    shift_on_points = num_points < num_queries
    
    p_grid_inds = compute_grid_indices(points, 2 * radius, sample_inds, with_shifts=shift_on_points)
    q_grid_inds = compute_grid_indices(
        queries, 2 * radius, query_sample_inds, with_shifts=not shift_on_points
    )
    num_shifts = 8

    build_on_points = p_grid_inds.shape[0] < q_grid_inds.shape[0]
    p_mask, q_mask = None, None
    if build_on_points:
        lookup_struct, p_lookup_inds = build_lookup_struct(p_grid_inds)
        del p_grid_inds
        q_lookup_inds, q_mask = query_lookup_struct(lookup_struct, q_grid_inds)
        del q_grid_inds
    else:
        lookup_struct, q_lookup_inds = build_lookup_struct(q_grid_inds)
        del q_grid_inds
        p_lookup_inds, p_mask = query_lookup_struct(lookup_struct, p_grid_inds)
        del p_grid_inds 
    lookup_struct_size = lookup_struct.size()
    del lookup_struct

    p_lookup_inds_int64 = p_lookup_inds.to(torch.int64)
    del p_lookup_inds
    p_point_inds, p_grid_sizes, p_grid_splits = arrange_indices(
        p_lookup_inds_int64,
        lookup_struct_size,
        num_shifts=num_shifts if shift_on_points else 1,
        mask=p_mask,
    )
    del p_mask, p_lookup_inds_int64

    q_lookup_inds_int64 = q_lookup_inds.to(torch.int64)
    del q_lookup_inds
    q_repeat_num = p_grid_sizes[q_lookup_inds_int64]
    del p_grid_sizes
    if q_mask is not None:
        q_repeat_num = q_repeat_num.mul_(q_mask)
        del q_mask
    q_repeat_num_cumsum, q_repeat_num_sum = cumsum_exclusive(
        q_repeat_num, return_sum=True
    )
    del q_repeat_num

    # ── Candidate expansion + radius filtering ──
    # Each candidate pair costs ~29 bytes across all temporary tensors
    # (indices_for_repeat int64 + q_inds int32 + p_inds_offset int64 +
    #  arange int64 + mask bool = 8+4+8+8+1 = 29 bytes).
    # If the total fits within the memory budget, use the fast monolithic path.
    # Otherwise, process queries in adaptive chunks to cap peak memory.
    _BYTES_PER_CANDIDATE = 29
    max_candidates = max_candidate_mem_bytes // _BYTES_PER_CANDIDATE

    if q_repeat_num_sum <= max_candidates:
        # ── Fast path: all candidates fit in memory budget ──
        indices_for_repeat = repeat_interleave_indices(
            repeats_cumsum=q_repeat_num_cumsum,
            output_size=q_repeat_num_sum,
            may_contain_zero_repeats=True,
        )
        q_inds = arange_cached(num_queries, device=device, dtype=torch.int32)
        if not shift_on_points:
            q_inds = torch.reshape(q_inds.unsqueeze(dim=-1).expand(-1, num_shifts), (-1,))
        q_inds = q_inds[indices_for_repeat]

        p_inds_offset = p_grid_splits[q_lookup_inds_int64]
        del p_grid_splits, q_lookup_inds_int64
        p_inds_offset -= q_repeat_num_cumsum
        del q_repeat_num_cumsum
        p_inds_offset = p_inds_offset[indices_for_repeat]
        del indices_for_repeat

        p_inds = arange_cached(q_inds.numel(), device=device, dtype=torch.int64)
        p_inds = p_point_inds[p_inds_offset.to(torch.int64).add_(p_inds)]
        del p_inds_offset
        del p_point_inds

        results = clamp_by_radius(
            queries, q_inds, points, p_inds,
            radius, return_distances, dtype_num_neighbors, distance_type,
        )
        del q_inds, p_inds
        return results

    # ── Chunked path: split queries to stay within memory budget ──
    effective_q = q_repeat_num_cumsum.shape[0]

    # Find chunk boundaries where cumulative candidates cross budget thresholds
    thresholds = torch.arange(
        max_candidates, q_repeat_num_sum + max_candidates,
        max_candidates, device=device, dtype=torch.int64,
    )
    split_indices = torch.searchsorted(q_repeat_num_cumsum, thresholds).clamp(max=effective_q)
    split_indices = torch.cat([
        torch.zeros(1, device=device, dtype=split_indices.dtype),
        split_indices,
    ]).unique()

    q_inds_base = arange_cached(num_queries, device=device, dtype=torch.int32)
    if not shift_on_points:
        q_inds_base = torch.reshape(
            q_inds_base.unsqueeze(dim=-1).expand(-1, num_shifts), (-1,)
        )

    num_neighbors_accum = torch.zeros(num_queries, dtype=dtype_num_neighbors, device=device)
    all_neighbors = []
    all_distances = []

    for ci in range(len(split_indices) - 1):
        start = split_indices[ci].item()
        end = split_indices[ci + 1].item()
        if start == end:
            continue

        base_offset = q_repeat_num_cumsum[start]
        chunk_total = (
            q_repeat_num_cumsum[end] if end < effective_q else q_repeat_num_sum
        ) - base_offset
        if chunk_total == 0:
            continue

        chunk_cumsum = q_repeat_num_cumsum[start:end] - base_offset

        indices_for_repeat = repeat_interleave_indices(
            repeats_cumsum=chunk_cumsum,
            output_size=chunk_total,
            may_contain_zero_repeats=True,
        )

        chunk_q_inds = q_inds_base[start:end][indices_for_repeat]

        chunk_p_offset = p_grid_splits[q_lookup_inds_int64[start:end]]
        chunk_p_offset = (chunk_p_offset - chunk_cumsum)[indices_for_repeat]
        del indices_for_repeat

        local_arange = torch.arange(chunk_total, device=device, dtype=torch.int64)
        chunk_p_inds = p_point_inds[chunk_p_offset.to(torch.int64).add_(local_arange)]
        del chunk_p_offset, local_arange

        result = clamp_by_radius(
            queries, chunk_q_inds, points, chunk_p_inds,
            radius, return_distances, dtype_num_neighbors, distance_type,
        )
        del chunk_q_inds, chunk_p_inds

        all_neighbors.append(result[0])
        num_neighbors_accum.add_(result[1])
        if return_distances:
            all_distances.append(result[2])

    del q_repeat_num_cumsum, q_lookup_inds_int64, p_grid_splits, p_point_inds

    if all_neighbors:
        neighbors = torch.cat(all_neighbors)
    else:
        neighbors = torch.empty(0, dtype=torch.int32, device=device)

    if return_distances:
        distances = (
            torch.cat(all_distances) if all_distances
            else torch.empty(0, dtype=queries.dtype, device=device)
        )
        return neighbors, num_neighbors_accum, distances
    return neighbors, num_neighbors_accum


def radius_search_brute_force(points, queries, radius, return_distances=False):
    """O(P*Q) brute-force radius search via full distance matrix.

    Reference implementation for correctness testing. Computes
    ``torch.cdist(queries, points)`` and filters by radius in-place
    to limit peak memory.

    Args:
        points: (P, 3) reference points.
        queries: (Q, 3) query points.
        radius: search radius.
        return_distances: if True, also return per-neighbor distances.

    Returns:
        neighbors: (N,) point indices of confirmed neighbors.
        num_neighbors: (Q,) neighbor count per query.
        distances (optional): (N,) distances for confirmed neighbors.
    """
    distance_matrix = torch.cdist(queries, points, p=2.0)

    # in-place manipulation to save memory usage: distance_matrix = min(distance_matrix-radius, 0)
    distance_matrix = distance_matrix.sub_(radius).clamp_(max=0.0)

    distance_matrix = distance_matrix.reshape(-1)
    neighbors_indices = torch.nonzero(distance_matrix).squeeze(-1)
    # recover the distances from the in-place manipulated distance_matrix
    distances = (
        distance_matrix[neighbors_indices].add_(radius) if return_distances else None
    )

    # now, release the big matrix
    del distance_matrix

    q_inds_masked = neighbors_indices.div(points.shape[0], rounding_mode="floor")
    num_neighbors = torch.zeros(
        (queries.shape[0],), dtype=torch.int32, device=points.device
    )
    num_neighbors.index_add_(
        dim=0,
        index=q_inds_masked,
        source=Constants.get_one(points.device, torch.int32).expand(
            q_inds_masked.numel()
        ),
    )
    del q_inds_masked
    num_neighbors = num_neighbors.to(torch.int64)

    neighbors = neighbors_indices.remainder_(points.shape[0])

    if return_distances:
        return neighbors, num_neighbors, distances
    else:
        return neighbors, num_neighbors


def _compute_batch_ranges(sample_inds, query_sample_inds, num_points, num_queries, device):
    """Convert per-point sample_inds to per-query (p_start, p_end) ranges."""
    if sample_inds is None:
        q_batch_starts = torch.zeros(num_queries, dtype=torch.int64, device=device)
        q_batch_ends = torch.full((num_queries,), num_points, dtype=torch.int64, device=device)
        return q_batch_starts, q_batch_ends

    p_sizes = torch.bincount(sample_inds)
    p_offsets = torch.zeros(p_sizes.numel() + 1, dtype=torch.int64, device=device)
    p_offsets[1:] = torch.cumsum(p_sizes, 0)

    q_batch_starts = p_offsets[query_sample_inds.long()]
    q_batch_ends = p_offsets[query_sample_inds.long() + 1]
    return q_batch_starts, q_batch_ends


def radius_search_tiled(
    points,
    queries,
    radius,
    sample_inds=None,
    query_sample_inds=None,
    return_distances=False,
    dtype_num_neighbors=torch.int64,
    distance_type="ball",
):
    """Tiled brute-force radius search (PyKeOPS-style).

    Never materializes the full Q×P distance matrix. Computes distances
    in tiles of BLOCK_P points per query, streaming results to output.
    Two passes: count neighbors, then compact.

    Same signature as ``radius_search_lookup`` — drop-in replacement.
    Faster than grid-accelerated search when radius/grid_size is large
    (> ~5-7x), where the grid provides little pruning benefit.

    Args:
        points: (P, 3) reference point coordinates.
        queries: (Q, 3) query point coordinates.
        radius: search radius.
        sample_inds: (P,) int batch index per point, or None for single-batch.
        query_sample_inds: (Q,) int batch index per query, or None.
        return_distances: if True, also return per-neighbor distances.
        dtype_num_neighbors: dtype for per-query neighbor counts.
        distance_type: "ball" (Euclidean) or "chebyshev" (L-inf).

    Returns:
        neighbors: (N,) point indices of confirmed neighbors.
        num_neighbors: (Q,) neighbor count per query.
        distances (optional): (N,) distances for confirmed neighbors.
    """
    if sample_inds is not None:
        assert query_sample_inds is not None
    else:
        assert query_sample_inds is None

    device = points.device
    num_points, num_queries = points.shape[0], queries.shape[0]
    BLOCK_P = 256

    q_batch_starts, q_batch_ends = _compute_batch_ranges(
        sample_inds, query_sample_inds, num_points, num_queries, device)

    # Select kernels based on distance type
    if distance_type == "ball":
        count_kernel = _tiled_radius_count_kernel
        compact_kernel = _tiled_radius_compact_kernel
        radius_param = radius * radius  # squared for Euclidean
    else:
        count_kernel = _tiled_radius_count_kernel_chebyshev
        compact_kernel = _tiled_radius_compact_kernel_chebyshev
        radius_param = radius

    # Pass 1: count
    num_neighbors = torch.zeros(num_queries, dtype=torch.int32, device=device)
    grid = (num_queries,)
    count_kernel[grid](
        queries, points, q_batch_starts, q_batch_ends,
        num_neighbors, radius_param, num_queries,
        BLOCK_P=BLOCK_P,
    )

    # Compute write offsets
    num_neighbors_out = num_neighbors.to(dtype_num_neighbors)
    offsets, total = cumsum_exclusive(num_neighbors_out, return_sum=True)

    if total == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        if return_distances:
            return empty, num_neighbors_out, torch.empty(0, dtype=queries.dtype, device=device)
        return empty, num_neighbors_out

    # Pass 2: compact
    out_neighbors = torch.empty(total, dtype=torch.int32, device=device)
    out_distances = torch.empty(total, dtype=queries.dtype, device=device) if return_distances else torch.empty(0, device=device)
    compact_kernel[grid](
        queries, points, q_batch_starts, q_batch_ends,
        offsets, out_neighbors, out_distances,
        radius_param, num_queries, return_distances,
        BLOCK_P=BLOCK_P,
    )

    if return_distances:
        return out_neighbors, num_neighbors_out, out_distances
    return out_neighbors, num_neighbors_out


def radius_search(
    points,
    query_points,
    radius,
    sample_inds=None,
    query_sample_inds=None,
    return_distances=False,
    point_num_max=1000000,
    sample_sizes=None,
    query_sample_sizes=None,
    distance_type="ball",
    tiled_batch_threshold=20000,
):
    """Top-level entry point for fixed-radius neighbor search.

    Adaptively selects between two backends:
      - **Grid-accelerated** (``radius_search_lookup``): best when per-batch
        point counts are large and radius is small relative to the scene.
      - **Tiled brute-force** (``radius_search_tiled``): best when per-batch
        point counts are small (≤ ``tiled_batch_threshold``) or radius is
        large relative to point spacing. Never materializes the full distance
        matrix, scaling as O(Q × P_batch) regardless of radius.

    The heuristic: if the max per-batch point count (across both points and
    queries) is ≤ ``tiled_batch_threshold``, use tiled; otherwise use grid.

    For large batched clouds (> point_num_max), splits by sample boundaries
    and runs the chosen backend per chunk to bound memory.

    Args:
        points: (P, 3) reference point coordinates.
        query_points: (Q, 3) query point coordinates.
        radius: search radius.
        sample_inds: (P,) batch index per point, or None for single-batch.
        query_sample_inds: (Q,) batch index per query, or None.
        return_distances: if True, also return per-neighbor distances.
        point_num_max: threshold above which to split into chunks.
        sample_sizes: (B,) pre-computed per-sample point counts (avoids bincount).
        query_sample_sizes: (B,) pre-computed per-sample query counts.
        distance_type: "ball" (Euclidean) or "chebyshev" (L-inf).
        tiled_batch_threshold: max per-batch size to prefer tiled over grid.
            Set to 0 to always use grid, or a large value to always use tiled.

    Returns:
        neighbors: (N,) point indices of confirmed neighbors.
        num_neighbors: (Q,) neighbor count per query.
        distances (optional): (N,) distances for confirmed neighbors.
    """
    point_num = max(points.shape[0], query_points.shape[0])

    # Choose backend based on per-batch size
    use_tiled = False
    if sample_inds is not None and tiled_batch_threshold > 0:
        _sample_sizes = (
            sample_sizes if sample_sizes is not None else torch.bincount(sample_inds)
        )
        _query_sample_sizes = (
            query_sample_sizes
            if query_sample_sizes is not None
            else torch.bincount(query_sample_inds)
        )
        max_batch = max(
            _sample_sizes.max().item(),
            _query_sample_sizes.max().item(),
        )
        use_tiled = max_batch <= tiled_batch_threshold
    elif sample_inds is None:
        use_tiled = point_num <= tiled_batch_threshold

    if use_tiled:
        return radius_search_tiled(
            points=points,
            queries=query_points,
            radius=radius,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
            return_distances=return_distances,
            dtype_num_neighbors=torch.int64,
            distance_type=distance_type,
        )

    if point_num <= point_num_max or sample_inds is None:
        return radius_search_lookup(
            points=points,
            queries=query_points,
            radius=radius,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
            return_distances=return_distances,
            distance_type=distance_type,
        )

    # split inputs, run search, then merge results
    num_splits = (point_num + point_num_max - 1) // point_num_max
    sample_sizes = (
        sample_sizes if sample_sizes is not None else torch.bincount(sample_inds)
    )
    query_sample_sizes = (
        query_sample_sizes
        if query_sample_sizes is not None
        else torch.bincount(query_sample_inds)
    )
    sample_num = sample_sizes.numel()
    assert sample_num == query_sample_sizes.numel()
    step = max(1, sample_num // num_splits)

    neighbors_list = list()
    num_neighbors_list = list()
    distances_list = list()

    points_start = 0
    query_points_start = 0
    sample_start = 0
    while sample_start < sample_num:
        sample_end = min(sample_num, sample_start + step)
        points_end = points_start + torch.sum(sample_sizes[sample_start:sample_end])
        query_points_end = query_points_start + torch.sum(
            query_sample_sizes[sample_start:sample_end]
        )
        points_split = points[points_start:points_end]
        query_points_split = query_points[query_points_start:query_points_end]
        sample_inds_split = sample_inds[points_start:points_end]
        query_sample_inds_split = query_sample_inds[query_points_start:query_points_end]
        result = radius_search_lookup(
            points=points_split,
            queries=query_points_split,
            radius=radius,
            sample_inds=sample_inds_split,
            query_sample_inds=query_sample_inds_split,
            return_distances=return_distances,
            dtype_num_neighbors=torch.int32,
            distance_type=distance_type,
        )
        neighbors_list.append(result[0] + points_start)
        num_neighbors_list.append(result[1])
        if return_distances:
            distances_list.append(result[2])

        points_start = points_end
        query_points_start = query_points_end
        sample_start = sample_end

    neighbors = torch.cat(neighbors_list, dim=-1)
    num_neighbors = torch.cat(num_neighbors_list, dim=-1).to(torch.int64)
    if return_distances:
        return neighbors, num_neighbors, torch.cat(distances_list, dim=-1)
    else:
        return neighbors, num_neighbors


def segment_sort(input, indices_for_repeat, distances=None, max_distance=None):
    """Sort elements within each segment defined by ``indices_for_repeat``.

    Each segment corresponds to one query's neighbor set. Sorts by distance
    if provided, otherwise by random jitter (for stochastic clipping).

    Args:
        input: (N,) values to reorder (typically neighbor indices).
        indices_for_repeat: (N,) segment id for each element.
        distances: (N,) optional sort key within each segment.
        max_distance: pre-computed max(distances) to avoid a reduction.

    Returns:
        (N,) reordered ``input`` with elements sorted within segments.
    """
    if distances is None:
        segment_distances = torch.rand_like(
            indices_for_repeat, dtype=torch.float32
        ).add_(indices_for_repeat)
    else:
        max_distance = torch.max(distances) if max_distance is None else max_distance
        segment_distances = indices_for_repeat * (
            max_distance * 1.1
        )  # *1.1 to make a gap between segments
        segment_distances = segment_distances.add_(distances)

    return input[torch.argsort(segment_distances)]


def clip_neighbors(
    neighbors,
    num_neighbors,
    neighbor_clip,
    distances=None,
    max_distance=None,
    randomize=True,
):
    """Cap each query's neighbor count to ``neighbor_clip``.

    When distances are provided, keeps the closest neighbors. Otherwise,
    if ``randomize=True``, shuffles before clipping for stochastic sampling.

    Args:
        neighbors: (N,) flat neighbor indices from radius search.
        num_neighbors: (Q,) per-query neighbor counts.
        neighbor_clip: maximum neighbors to keep per query.
        distances: (N,) optional distances for distance-based selection.
        max_distance: pre-computed max distance (avoids reduction).
        randomize: if True and no distances, randomly shuffle before clipping.

    Returns:
        (neighbors_clipped, num_neighbors_clipped,
         num_neighbors_clipped_cumsum, indices_for_repeat)
    """
    randomize = (
        False if distances is not None else randomize
    )  # disable randomize when distance is provided
    num_neighbors_cumsum = cumsum_exclusive(num_neighbors)
    indices_for_repeat = repeat_interleave_indices(
        repeats_cumsum=num_neighbors_cumsum,
        output_size=neighbors.numel(),
        may_contain_zero_repeats=False,
    )
    if neighbor_clip >= torch.max(num_neighbors):
        return neighbors, num_neighbors, num_neighbors_cumsum, indices_for_repeat

    if randomize or distances is not None:
        neighbors = segment_sort(neighbors, indices_for_repeat, distances, max_distance)
    num_neighbors_clipped = torch.clamp(num_neighbors, max=neighbor_clip)
    num_neighbors_clipped_cumsum, num_neighbors_clipped_sum = cumsum_exclusive(
        num_neighbors_clipped, return_sum=True
    )
    indices_for_repeat = repeat_interleave_indices(
        repeats_cumsum=num_neighbors_clipped_cumsum,
        output_size=num_neighbors_clipped_sum,
        may_contain_zero_repeats=False,
    )

    num_clipped_cumsum = num_neighbors_cumsum.sub_(num_neighbors_clipped_cumsum)
    neighbors_clipped_indices = num_clipped_cumsum[indices_for_repeat]
    neighbors_clipped_indices += arange_cached(
        num_neighbors_clipped_sum, device=neighbors.device
    )
    neighbors_clipped = neighbors[neighbors_clipped_indices]

    return (
        neighbors_clipped,
        num_neighbors_clipped,
        num_neighbors_clipped_cumsum,
        indices_for_repeat,
    )


def nearest_neighbors(neighbors, num_neighbors, distances):
    """Extract the single closest neighbor for each query.

    Finds the minimum distance within each query's neighbor set and returns
    the corresponding point index. Handles ties by keeping one arbitrarily.

    Args:
        neighbors: (N,) flat neighbor indices.
        num_neighbors: (Q,) per-query neighbor counts.
        distances: (N,) per-neighbor distances.

    Returns:
        (Q,) point index of the nearest neighbor for each query.
    """
    indices_for_repeat = repeat_interleave_indices(
        repeats=num_neighbors,
        output_size=neighbors.numel(),
        may_contain_zero_repeats=False,
    )
    distances_min = torch.segment_reduce(distances, reduce="min", lengths=num_neighbors)
    distances_min_repeated = distances_min[indices_for_repeat]
    nearest_mask = distances_min_repeated.sub_(distances) == 0
    nearest_indices = torch.nonzero(nearest_mask).squeeze(dim=-1)
    if nearest_indices.numel() == num_neighbors.numel():
        return neighbors[nearest_indices]

    # in case there are multiple min distances in a neighborhood
    nearest_one_neighbors = torch.empty_like(num_neighbors)
    return nearest_one_neighbors.index_copy_(
        0, indices_for_repeat[nearest_indices], neighbors[nearest_indices]
    )
