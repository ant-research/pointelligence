"""GPU-accelerated radius-based neighborhood search.

Finds all points within a fixed radius of each query point, producing
(i, j) pairs that represent the sparse neighborhood graph. Supports
batched point clouds via per-sample confinement.
"""

import math
import torch

from .constants import Constants
from .indexing import cumsum_exclusive
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


@torch.no_grad()
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


@torch.no_grad()
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


@torch.no_grad()
def radius_search_tiled(
    points,
    queries,
    radius,
    sample_inds=None,
    query_sample_inds=None,
    return_distances=False,
    dtype_num_neighbors=torch.int64,
    distance_type="ball",
    block_p: int = 256,
):
    """Tiled brute-force radius search (PyKeOPS-style).

    Never materializes the full Q×P distance matrix. Computes distances
    in tiles of BLOCK_P points per query, streaming results to output.
    Two passes: count neighbors, then compact.

    Shares the production radius-search output contract.
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
        block_p: point tile size per query program.

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
    BLOCK_P = int(block_p)

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
    grid_size=None,
    tiled_radius_multiplier_threshold=4.5,
    backend="auto",
    fixed_grid_radius_multiplier_threshold=2.75,
):
    """Top-level entry point for exact fixed-radius neighbor search.

    ``auto`` always selects the compact, sorted eight-cell implementation.
    It has no data-dependent host dispatch, candidate materialization, or
    query chunk/concatenate path. The sorted 27-cell implementation remains an
    explicit alternative for benchmarking and hardware-specific studies.

    The legacy tuning arguments are retained for source compatibility but no
    longer affect automatic dispatch. ``grid_size`` is validated when given.
    """
    aliases = {
        "sorted_grid8": "sorted8_materialized",
        "fixed_grid": "sorted27_materialized",
    }
    selected = aliases.get(backend, backend)
    valid = {
        "auto",
        "sorted8_materialized",
        "sorted27_materialized",
        "tiled",
    }
    if selected not in valid:
        raise ValueError(
            "backend must be auto, sorted8_materialized, "
            "sorted27_materialized, or tiled; "
            f"got {backend!r}")
    if grid_size is not None and float(grid_size) <= 0:
        raise ValueError(f"grid_size must be positive; got {grid_size}")

    if selected == "auto":
        selected = "sorted8_materialized"
    if selected == "sorted8_materialized":
        return radius_search_sorted_grid8(
            points, query_points, radius, sample_inds, query_sample_inds,
            return_distances, torch.int64, distance_type)
    if selected == "sorted27_materialized":
        return radius_search_fixed_grid(
            points, query_points, radius, sample_inds, query_sample_inds,
            return_distances, torch.int64, distance_type)
    return radius_search_tiled(
        points, query_points, radius, sample_inds, query_sample_inds,
        return_distances, torch.int64, distance_type, block_p=4096)


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
    neighbors_clipped_indices += torch.arange(
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


@torch.no_grad()
def radius_search_strided_grid(
    points,
    queries,
    radius,
    cell_size,
    sample_inds=None,
    query_sample_inds=None,
    return_distances=False,
    dtype_num_neighbors=torch.int64,
    triton_fused=True,
    distance_type="ball",
    cell_cover_mode="symmetric",
    kernel_size=None,
    kernel_grid_size=None,
    tap_stripes=1,
):
    """Strided grid-downsample radius search via bounded (2K+1)^3-cell scan.

    Sorted-cell radius search for arbitrary cell sizes. It is efficient when
    ``cell_size`` (typically the strided downsample stride in physical
    units) is not vastly smaller than ``radius``. Bins inputs to a
    coord-grid of side ``cell_size``; for each query, scans the
    ``(2K+1)^3`` cells where ``K = ceil(radius / cell_size)``, then
    filters by exact Euclidean distance ≤ radius.

    Exact distance filtering preserves the same unique ``(query, point)`` set
    as an independent tiled reference; neighbor ordering is unspecified.

    Args:
        points: (P, 3) reference point coordinates.
        queries: (Q, 3) query point coordinates.
        radius: search radius (physical units).
        cell_size: cell side length for the grid (physical units; usually
            the strided downsample stride). When ``cell_size >= 2 * radius``
            (K = 1, 27-cell window), this hits the minimum-cost regime
            described in the plan §2.1.
        sample_inds: (P,) batch index per point, or None for single-batch.
        query_sample_inds: (Q,) batch index per query, or None.
        return_distances: if True, also return per-neighbor distances.
        dtype_num_neighbors: dtype for per-query neighbor counts.
        triton_fused: when True (default), use the fused count+compact
            Triton kernel for the inner candidate-cell scan. Avoids
            materializing the O(total-candidates) flat ``(q_idx, p_idx)``
            array. Falls back to the PyTorch-vectorized + ``clamp_by_radius``
            path when False — used by the parity test fallback.
        cell_cover_mode: ``"symmetric"`` scans `(2K+1)^3` cells around the
            query cell. ``"exact8"`` requires `cell_size == 2 * radius` and
            scans the exact eight cells intersected by the query radius box.
        kernel_size: optional odd cubic kernel size. When supplied together
            with ``kernel_grid_size``, emit accepted pairs directly in
            kernel-tap-major order and return segment offsets instead of the
            query-major neighbor vector. This specialized contract is used by
            point convolution and leaves the historical API unchanged when
            omitted.
        kernel_grid_size: physical spacing used to quantize point-minus-query
            offsets into kernel taps. Must be supplied with ``kernel_size``.

    Returns:
        Normally returns ``(neighbors, num_neighbors[, distances])`` with
        neighbors sorted by query. In prepared mode returns
        ``(i, j, seg_offs, num_neighbors)``: explicit query and point indices
        grouped by kernel tap, plus the ``kernel_size**3 + 1`` tap offsets.
    """
    assert points.dim() == 2 and points.shape[1] == 3
    assert queries.dim() == 2 and queries.shape[1] == 3
    assert points.dtype == queries.dtype, (points.dtype, queries.dtype)
    assert points.device == queries.device
    if distance_type not in ("ball", "chebyshev"):
        raise ValueError(
            "distance_type must be 'ball' or 'chebyshev'; "
            f"got {distance_type!r}"
        )
    prepare_segments = kernel_size is not None or kernel_grid_size is not None
    if prepare_segments:
        if kernel_size is None or kernel_grid_size is None:
            raise ValueError(
                "kernel_size and kernel_grid_size must be supplied together")
        if return_distances:
            raise ValueError(
                "prepared kernel segments do not support return_distances")
        if not triton_fused:
            raise ValueError(
                "prepared kernel segments require triton_fused=True")
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer; got {kernel_size}")
        if float(kernel_grid_size) <= 0:
            raise ValueError(
                "kernel_grid_size must be positive; "
                f"got {kernel_grid_size}")
        tap_stripes = int(tap_stripes)
        if (
            tap_stripes <= 0
            or tap_stripes > 256
            or tap_stripes & (tap_stripes - 1)
        ):
            raise ValueError(
                "tap_stripes must be a power of two in [1, 256]; "
                f"got {tap_stripes}")
    if cell_cover_mode not in ("symmetric", "exact8"):
        raise ValueError(
            "cell_cover_mode must be 'symmetric' or 'exact8'; "
            f"got {cell_cover_mode!r}")
    if cell_cover_mode == "exact8" and not math.isclose(
        float(cell_size), 2.0 * float(radius), rel_tol=1e-7, abs_tol=0.0
    ):
        raise ValueError("exact8 requires cell_size == 2 * radius")
    if sample_inds is not None:
        assert query_sample_inds is not None
        # The strided algorithm folds the batch dim into the cell hash via a
        # high-order stride term (one cell-extent per batch). This avoids
        # cross-sample candidate matches without per-sample loops.
    else:
        assert query_sample_inds is None

    device = points.device
    num_points, num_queries = points.shape[0], queries.shape[0]

    if num_points == 0 or num_queries == 0:
        empty_n = torch.empty(0, dtype=torch.int32, device=device)
        empty_c = torch.zeros(num_queries, dtype=dtype_num_neighbors, device=device)
        if prepare_segments:
            seg_offs = torch.zeros(
                kernel_size ** 3 + 1, dtype=torch.int64, device=device)
            return empty_n, empty_n.clone(), seg_offs, empty_c
        if return_distances:
            return empty_n, empty_c, torch.empty(0, dtype=points.dtype, device=device)
        return empty_n, empty_c

    # Materialize non-contiguous inputs at most once; count and fill reuse them.
    points_c = points.contiguous()
    queries_c = queries.contiguous()

    K = int(math.ceil(float(radius) / float(cell_size)))
    if cell_cover_mode == "exact8":
        num_offsets = 8
    else:
        side = 2 * K + 1
        num_offsets = side * side * side  # (2K+1)^3

    # ── Step 1: integer cell coords for both clouds, packed for hash ──
    # ``floor`` rather than ``trunc`` so negative coords land in cells with
    # the expected monotone hash ordering.
    inv_cs = 1.0 / float(cell_size)
    cell_in = torch.floor(points_c * inv_cs).to(torch.int64)      # [P, 3]
    if cell_cover_mode == "exact8":
        cell_out = torch.floor((queries_c - float(radius)) * inv_cs).to(
            torch.int64)  # [Q, 3], lower corner of exact radius box
    else:
        cell_out = torch.floor(queries_c * inv_cs).to(torch.int64)  # [Q, 3]

    # Per-axis padding by K so neighbor cells of edge outputs stay in-range.
    in_min = cell_in.amin(dim=0)                                   # [3]
    out_min = cell_out.amin(dim=0)                                 # [3]
    in_max = cell_in.amax(dim=0)
    out_max = cell_out.amax(dim=0)
    base = torch.minimum(in_min, out_min) - K                      # [3]
    top = torch.maximum(in_max, out_max) + K + 1                   # [3]
    extent = (top - base).clamp_min(1)                             # [3], CUDA

    # 1D row-major hash: x * (Y * Z) + y * Z + z. Folded batch (if any) goes
    # into a high-order term so cross-sample lookups can never match.
    # Keep these as device scalars. Converting ``extent`` with ``tolist()``
    # serialized the host with CUDA even though every consumer is a tensor op.
    stride_x = extent[1] * extent[2]
    stride_y = extent[2]
    cell_in_shifted = cell_in - base                               # [P, 3]
    cell_out_shifted = cell_out - base                             # [Q, 3]

    if sample_inds is not None:
        # Batch hash bump = sample_id * total_grid_volume.
        grid_volume = extent.prod()
        batch_bump_in = sample_inds.to(torch.int64) * grid_volume
        batch_bump_q = query_sample_inds.to(torch.int64) * grid_volume
    else:
        batch_bump_in = None
        batch_bump_q = None

    hash_in = (
        cell_in_shifted[:, 0] * stride_x
        + cell_in_shifted[:, 1] * stride_y
        + cell_in_shifted[:, 2]
    )
    hash_out = (
        cell_out_shifted[:, 0] * stride_x
        + cell_out_shifted[:, 1] * stride_y
        + cell_out_shifted[:, 2]
    )
    if batch_bump_in is not None:
        hash_in = hash_in + batch_bump_in
        hash_out = hash_out + batch_bump_q

    # ── Step 2: sort inputs by hash for binary-search candidate lookup ──
    sort_perm = torch.argsort(hash_in)
    sorted_hash = hash_in[sort_perm]

    # ── Step 3: reuse the cached Cartesian offset cube ──
    lower, upper = (0, 1) if cell_cover_mode == "exact8" else (-K, K)
    offsets = Constants.get_3d_offset_cube(device, lower, upper)
    offset_hash = (
        offsets[:, 0] * stride_x + offsets[:, 1] * stride_y + offsets[:, 2])

    # ── Step 4: candidate cell ranges via binary search ──
    # candidate_hash[q, o] = hash_out[q] + offset_hash[o]
    flat_hash = (hash_out.unsqueeze(1) + offset_hash.unsqueeze(0)).reshape(-1)
    use_int32_ranges = num_points <= torch.iinfo(torch.int32).max
    start_idx = torch.searchsorted(
        sorted_hash, flat_hash, right=False, out_int32=use_int32_ranges)
    # Reuse candidate-key storage for the exclusive upper-bound lookup.
    flat_hash.add_(1)
    end_idx = torch.searchsorted(
        sorted_hash, flat_hash, right=False, out_int32=use_int32_ranges)
    del flat_hash

    # ── Offset-vector Triton path (default): one program per query scans all
    # candidate cells in a two-dimensional offset-by-point tile. A small
    # per-query reduction converts cell counts into disjoint output spans.
    if triton_fused:
        from sparse_engines.strided_grid_radius_triton_kernel import (
            _offset_vector_radius_compact_kernel,
            _offset_vector_radius_count_kernel,
            _offset_vector_radius_reduce_kernel,
            _offset_vector_triplet_compact_kernel,
            _offset_vector_triplet_count_kernel,
        )
        count_dtype = (
            torch.int64 if dtype_num_neighbors == torch.int64 else torch.int32)
        cell_offsets = torch.empty(
            num_queries * num_offsets, dtype=torch.int32, device=device)
        num_neighbors = torch.zeros(
            num_queries, dtype=count_dtype, device=device)
        block_o = 1 << (num_offsets - 1).bit_length()
        block_p = 32 if block_o <= 8 else 16
        grid = (num_queries,)
        distance_code = 0 if distance_type == "ball" else 1
        sort_perm_c = sort_perm.contiguous()
        start_idx_c = start_idx.contiguous()
        end_idx_c = end_idx.contiguous()
        if prepare_segments:
            num_kernel_offsets = kernel_size ** 3
            tap_counts = torch.zeros(
                (tap_stripes, num_kernel_offsets),
                dtype=torch.int32,
                device=device,
            )
            _offset_vector_triplet_count_kernel[grid](
                queries_c,
                points_c,
                sort_perm_c,
                start_idx_c,
                end_idx_c,
                cell_offsets,
                tap_counts,
                float(radius),
                float(kernel_grid_size),
                num_queries,
                num_offsets,
                DISTANCE_TYPE=distance_code,
                KERNEL_SIZE=kernel_size,
                TAP_STRIPES=tap_stripes,
                BLOCK_O=block_o,
                BLOCK_P=block_p,
            )
        else:
            _offset_vector_radius_count_kernel[grid](
                queries_c,
                points_c,
                sort_perm_c,
                start_idx_c,
                end_idx_c,
                cell_offsets,
                float(radius),
                num_queries,
                num_offsets,
                DISTANCE_TYPE=distance_code,
                BLOCK_O=block_o,
                BLOCK_P=block_p,
            )
        _offset_vector_radius_reduce_kernel[grid](
            cell_offsets, num_neighbors, num_queries, num_offsets,
            BLOCK_O=block_o)
        num_neighbors_64 = (
            num_neighbors if count_dtype == torch.int64
            else num_neighbors.to(torch.int64))
        if prepare_segments:
            seg_offs = torch.zeros(
                num_kernel_offsets + 1, dtype=torch.int64, device=device)
            tap_totals = tap_counts.sum(dim=0, dtype=torch.int64)
            seg_offs[1:] = torch.cumsum(tap_totals, dim=0, dtype=torch.int64)
            total = int(seg_offs[-1].item())
            counts_out = num_neighbors.to(dtype_num_neighbors)
            if total == 0:
                empty_n = torch.empty(0, dtype=torch.int32, device=device)
                return empty_n, empty_n.clone(), seg_offs, counts_out
            if total > torch.iinfo(torch.int32).max:
                raise RuntimeError(
                    "prepared kernel-segment output currently requires fewer "
                    f"than 2^31 edges; got {total}")
            if tap_stripes == 1:
                tap_cursors = seg_offs[:-1].to(torch.int32)
            else:
                stripe_prefix = (
                    torch.cumsum(tap_counts, dim=0, dtype=torch.int64)
                    - tap_counts
                )
                tap_cursors = (
                    stripe_prefix + seg_offs[:-1].unsqueeze(0)
                ).to(torch.int32).contiguous().view(-1)
            out_i = torch.empty(total, dtype=torch.int32, device=device)
            out_j = torch.empty(total, dtype=torch.int32, device=device)
            _offset_vector_triplet_compact_kernel[grid](
                queries_c,
                points_c,
                sort_perm_c,
                start_idx_c,
                end_idx_c,
                tap_cursors,
                out_i,
                out_j,
                float(radius),
                float(kernel_grid_size),
                num_queries,
                num_offsets,
                DISTANCE_TYPE=distance_code,
                KERNEL_SIZE=kernel_size,
                TAP_STRIPES=tap_stripes,
                BLOCK_O=block_o,
                BLOCK_P=block_p,
            )
            return out_i, out_j, seg_offs, counts_out
        write_offsets, total_t = cumsum_exclusive(
            num_neighbors_64, return_sum=True)
        total = int(total_t.item())
        # The sole D2H sync in this path obtains the exact accepted-edge count;
        # avoiding it would require candidate-sized output over-allocation.
        if total == 0:
            empty_n = torch.empty(0, dtype=torch.int32, device=device)
            empty_c = num_neighbors.to(dtype_num_neighbors)
            if return_distances:
                return empty_n, empty_c, torch.empty(
                    0, dtype=points.dtype, device=device)
            return empty_n, empty_c
        out_neighbors = torch.empty(total, dtype=torch.int64, device=device)
        out_distances = (
            torch.empty(total, dtype=points.dtype, device=device)
            if return_distances
            else torch.empty(0, dtype=points.dtype, device=device))
        _offset_vector_radius_compact_kernel[grid](
            queries_c,
            points_c,
            sort_perm_c,
            start_idx_c,
            end_idx_c,
            cell_offsets,
            write_offsets.contiguous(),
            out_neighbors,
            out_distances,
            float(radius),
            num_queries,
            num_offsets,
            RETURN_DIST=return_distances,
            DISTANCE_TYPE=distance_code,
            BLOCK_O=block_o,
            BLOCK_P=block_p,
        )
        counts_out = num_neighbors.to(dtype_num_neighbors)
        if return_distances:
            return out_neighbors, counts_out, out_distances
        return out_neighbors, counts_out

    counts = end_idx - start_idx                                        # [Q * num_offsets]
    total = int(counts.sum().item())
    # This fallback materializes candidate pairs, so their exact allocation
    # size must cross to the host once before constructing the arrays.

    if total == 0:
        empty_n = torch.empty(0, dtype=torch.int32, device=device)
        empty_c = torch.zeros(num_queries, dtype=dtype_num_neighbors, device=device)
        if return_distances:
            return empty_n, empty_c, torch.empty(0, dtype=points.dtype, device=device)
        return empty_n, empty_c

    # ── Step 5: flatten the per-cell ranges into a candidate (q_ind, p_ind) array ──
    counts_cumsum, _ = cumsum_exclusive(counts.to(torch.int64), return_sum=True)
    meta_indices = repeat_interleave_indices(
        repeats_cumsum=counts_cumsum,
        output_size=total,
        may_contain_zero_repeats=True,
    )                                                                   # [total]

    # Position of each candidate WITHIN its (q, offset) cell range.
    cell_arange = torch.arange(total, device=device, dtype=torch.int64)
    inner_offset = cell_arange - counts_cumsum[meta_indices]            # [total]

    candidate_p_in_sorted = start_idx[meta_indices] + inner_offset      # [total]
    candidate_q = meta_indices // num_offsets                            # [total]
    candidate_p = sort_perm[candidate_p_in_sorted]                       # [total]

    # ── Step 6: exact distance filter.
    # Reusing ``clamp_by_radius`` preserves the established arithmetic: same
    # fp32 rounding for sqrt(dx²+dy²+dz²) and the ``dist <= radius`` compare.
    # A standalone ``dist² <= radius²`` or PyTorch-side sqrt differs by 1-2
    # ULPs at boundary points, breaking set-equality parity.
    # Pass query indices as int32 and point indices as int64.
    return clamp_by_radius(
        queries,
        candidate_q.to(torch.int32),
        points,
        candidate_p,                       # already int64 from sort_perm
        radius,
        return_distances=return_distances,
        dtype_num_neighbors=dtype_num_neighbors,
        distance_type=distance_type,
    )


def radius_search_sorted_grid8(
    points,
    queries,
    radius,
    sample_inds=None,
    query_sample_inds=None,
    return_distances=False,
    dtype_num_neighbors=torch.int64,
    distance_type="ball",
):
    """Exact eight-cell sorted search with one index entry per point.

    Cells have side ``2 * radius``. The radius box intersects exactly two
    half-open cells per axis, so their Cartesian product is a complete
    duplicate-free eight-cell candidate cover.
    """
    if float(radius) <= 0:
        raise ValueError(f"radius must be positive; got {radius}")
    return radius_search_strided_grid(
        points=points,
        queries=queries,
        radius=radius,
        cell_size=2.0 * float(radius),
        sample_inds=sample_inds,
        query_sample_inds=query_sample_inds,
        return_distances=return_distances,
        dtype_num_neighbors=dtype_num_neighbors,
        triton_fused=True,
        distance_type=distance_type,
        cell_cover_mode="exact8",
    )


def radius_search_sorted_grid8_segments(
    points,
    queries,
    radius,
    kernel_size,
    kernel_grid_size,
    sample_inds=None,
    query_sample_inds=None,
    dtype_num_neighbors=torch.int64,
    distance_type="ball",
    tap_stripes=32,
):
    """Exact eight-cell search with direct kernel-tap-major output.

    Returns explicit ``(i, j)`` pairs grouped by their quantized convolution
    tap, the tap-segment offsets, and per-query neighbor counts. This avoids
    materializing ``k`` and avoids the global post-search sort and gathers.
    Pair ordering within a tap is unspecified. Thirty-two independent cursor
    stripes reduce atomic contention without data-dependent dispatch.
    """
    if float(radius) <= 0:
        raise ValueError(f"radius must be positive; got {radius}")
    return radius_search_strided_grid(
        points=points,
        queries=queries,
        radius=radius,
        cell_size=2.0 * float(radius),
        sample_inds=sample_inds,
        query_sample_inds=query_sample_inds,
        dtype_num_neighbors=dtype_num_neighbors,
        triton_fused=True,
        distance_type=distance_type,
        cell_cover_mode="exact8",
        kernel_size=kernel_size,
        kernel_grid_size=kernel_grid_size,
        tap_stripes=tap_stripes,
    )


def radius_search_fixed_grid(
    points,
    queries,
    radius,
    sample_inds=None,
    query_sample_inds=None,
    return_distances=False,
    dtype_num_neighbors=torch.int64,
    distance_type="ball",
):
    """Search one radius-sized grid and its 27 adjacent cell offsets.

    Every point is stored in one cell. A true radius neighbor differs from
    its query by at most one cell per axis, so the 27-cell candidate scan is
    complete. The exact distance predicate then determines membership.

    Neighbor order within each query is unspecified. The duplicate-free
    ``(query, point)`` set is the API invariant.
    """
    if float(radius) <= 0:
        raise ValueError(f"radius must be positive; got {radius}")
    return radius_search_strided_grid(
        points=points,
        queries=queries,
        radius=radius,
        cell_size=radius,
        sample_inds=sample_inds,
        query_sample_inds=query_sample_inds,
        return_distances=return_distances,
        dtype_num_neighbors=dtype_num_neighbors,
        triton_fused=True,
        distance_type=distance_type,
    )
