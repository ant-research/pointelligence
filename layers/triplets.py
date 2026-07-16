"""Convolution triplet (i, j, k) construction.

Builds the sparse connectivity that maps each (query i, neighbor j) pair
to a kernel weight index k via local voxelization. The triplets define
the data-weight routing for PointConv3d.
"""

from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable, Optional

import math
import weakref
import torch

from torch import Tensor
from torch.nn.common_types import _size_3_t
from torch.nn.modules.utils import _triple

from internals.neighbors import (
    radius_search,
    radius_search_sorted_grid8_segments,
    radius_search_strided_grid,
)
from internals.indexing import cumsum_exclusive, repeat_interleave_indices
from internals.triplet_cache import _active_cache, triplet_key
from internals.grid_indexing import compute_grid_indices, reduce_indices_to_1d
from internals.grid_sample import grid_sample_filter

from .metadata import MetaData
from .downsample import downsample
from .contract import TripletContract


@dataclass(frozen=True)
class FullCoverStridedRulebook:
    """Rulebook for overlapping strided point convolution with full coverage."""

    points: Tensor
    sample_inds: Tensor
    sample_sizes: Tensor
    center_source_indices: Tensor
    initial_center_source_indices: Tensor
    additional_center_source_indices: Tensor
    point_to_initial_center: Tensor
    i: Tensor
    j: Tensor
    k: Tensor
    num_neighbors: Optional[Tensor]
    i_upsample: Tensor
    j_upsample: Tensor
    k_upsample: Tensor
    radius: float
    radius_scaler: float
    initial_center_counts: Tensor
    additional_center_counts: Tensor
    selector_round_count: int
    coverage_per_input: Tensor


def full_cover_radius_scaler(stride: float, radius_margin: float = 1e-2) -> float:
    if stride <= 0:
        raise ValueError(f"stride must be positive; got {stride}")
    if radius_margin < 0:
        raise ValueError(f"radius_margin must be non-negative; got {radius_margin}")
    return (math.sqrt(3.0) / 2.0) * float(stride) * (1.0 + float(radius_margin))


def minimum_full_cover_kernel_size(radius_scaler: float) -> int:
    return 2 * int(math.ceil(float(radius_scaler))) + 1


def _strict_voxelize_offsets(
    offsets: Tensor,
    *,
    grid_size: float,
    kernel_size: _size_3_t,
    context: str,
) -> Tuple[Tensor, Tensor]:
    sizes = _triple(kernel_size)
    if not (sizes[0] == sizes[1] == sizes[2]):
        raise ValueError(f"{context} requires a cubic kernel; got {sizes}")
    K = int(sizes[0])
    if K <= 0 or K % 2 != 1:
        raise ValueError(f"{context} requires a positive odd kernel; got {K}")
    half = K // 2
    d = torch.round(offsets / float(grid_size)).to(torch.int64)
    if d.numel() > 0:
        max_abs = int(d.abs().max().item())
        if max_abs > half:
            required = 2 * max_abs + 1
            raise ValueError(
                f"{context} kernel_size={K} is insufficient for observed "
                f"offset {max_abs} grid steps; need at least K{required}. "
                "Full-cover strided convolution never clamps or aliases "
                "out-of-range offsets."
            )
    shifted = d + half
    multipliers = torch.tensor([K * K, K, 1], device=offsets.device,
                               dtype=torch.int64)
    k = torch.sum(shifted * multipliers, dim=1)
    return k, d


def _center_nearest_sources(
    points: Tensor,
    sample_inds: Tensor,
    cell_size: float,
) -> Tuple[Tensor, Tensor]:
    grid_inds = compute_grid_indices(
        points, cell_size, sample_inds, dtype=torch.int64
    )
    keys, _, _, _ = reduce_indices_to_1d(grid_inds)
    xyz_grid = grid_inds[:, :3]
    centers = (xyz_grid.to(points.dtype) + 0.5) * float(cell_size)
    dist2 = torch.sum((points - centers).square(), dim=1)
    src = torch.arange(points.shape[0], device=points.device, dtype=torch.int64)

    order = torch.argsort(src, stable=True)
    order = order[torch.argsort(dist2[order], stable=True)]
    order = order[torch.argsort(keys[order], stable=True)]

    sorted_keys = keys[order]
    _, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    first = torch.cumsum(counts, dim=0) - counts
    source_indices = order[first].to(torch.long)

    rank_order = torch.argsort(
        sample_inds[source_indices].to(torch.int64) * (points.shape[0] + 1)
        + source_indices,
        stable=True,
    )
    source_indices = source_indices[rank_order]

    source_rank = torch.empty(points.shape[0], device=points.device,
                              dtype=torch.long)
    source_rank[source_indices] = torch.arange(
        source_indices.numel(), device=points.device, dtype=torch.long)
    lookup_key = keys[source_indices]
    lookup_order = torch.argsort(lookup_key)
    pos = torch.searchsorted(lookup_key[lookup_order], keys)
    point_to_initial = source_rank[source_indices[lookup_order[pos]]]
    return source_indices, point_to_initial


def _rank_priorities(gain: Tensor, dist2: Tensor, source: Tensor) -> Tensor:
    order = torch.argsort(source, stable=True)
    order = order[torch.argsort(-dist2[order], stable=True)]
    order = order[torch.argsort(-gain[order], stable=True)]
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=order.device,
                               dtype=order.dtype)
    return rank


def _select_residual_full_cover_centers(
    points: Tensor,
    sample_inds: Tensor,
    c0_points: Tensor,
    point_to_initial_center: Tensor,
    uncovered: Tensor,
    radius: float,
    *,
    backend: str,
) -> Tuple[Tensor, int]:
    if not bool(uncovered.any().item()):
        return torch.empty(0, device=points.device, dtype=torch.long), 0

    u_source = torch.nonzero(uncovered).squeeze(1).to(torch.long)
    u_points = points[u_source]
    u_sample = sample_inds[u_source]
    if u_source.numel() <= 2048:
        diff = u_points[:, None, :] - u_points[None, :, :]
        same_sample = u_sample[:, None] == u_sample[None, :]
        within = same_sample & (torch.sqrt(diff.square().sum(dim=-1)) <= radius)
        q_index, p_index = torch.nonzero(within, as_tuple=True)
        q_index = q_index.to(torch.long)
        p_index = p_index.to(torch.long)
    else:
        uu_neighbors, uu_counts = radius_search(
            points=u_points,
            query_points=u_points,
            radius=radius,
            sample_inds=u_sample,
            query_sample_inds=u_sample,
            backend=backend,
        )
        q_offsets, total_edges = cumsum_exclusive(uu_counts, return_sum=True)
        q_index = repeat_interleave_indices(
            repeats_cumsum=q_offsets,
            output_size=total_edges,
            may_contain_zero_repeats=False,
        ).to(torch.long)
        p_index = uu_neighbors.to(torch.long)
    c0_for_u = point_to_initial_center[u_source].to(torch.long)
    dist2_to_c0 = torch.sum((u_points - c0_points[c0_for_u]).square(), dim=1)

    active = torch.ones(u_source.numel(), device=points.device, dtype=torch.bool)
    selected_chunks = []
    rounds = 0
    while bool(active.any().item()):
        active_edge = active[q_index] & active[p_index]
        active_gain = torch.zeros(u_source.numel(), device=points.device,
                                  dtype=torch.long)
        if bool(active_edge.any().item()):
            active_gain.index_add_(
                0,
                q_index[active_edge],
                torch.ones(int(active_edge.sum().item()), device=points.device,
                           dtype=torch.long),
            )
        rank = _rank_priorities(active_gain, dist2_to_c0, u_source)
        large = torch.full_like(rank, rank.numel() + 1)
        neighbor_rank = torch.where(active[p_index], rank[p_index], large[p_index])
        min_neighbor_rank = torch.full_like(rank, rank.numel() + 1)
        min_neighbor_rank.scatter_reduce_(
            0, q_index, neighbor_rank, reduce="amin", include_self=True)
        selected = active & (rank == min_neighbor_rank)
        selected_local = torch.nonzero(selected).squeeze(1)
        if selected_local.numel() == 0:
            raise RuntimeError("full-cover R-net selector made no progress")
        selected_chunks.append(u_source[selected_local])
        deactivate_edge = selected[q_index] & active[p_index]
        if bool(deactivate_edge.any().item()):
            active[p_index[deactivate_edge]] = False
        rounds += 1

    added = torch.cat(selected_chunks) if selected_chunks else torch.empty(
        0, device=points.device, dtype=torch.long)
    order = torch.argsort(sample_inds[added].to(torch.int64) * (points.shape[0] + 1)
                          + added, stable=True)
    return added[order], rounds


@torch.compiler.disable
def build_full_cover_strided_rulebook(
    points: Tensor,
    sample_inds: Tensor,
    sample_sizes: Optional[Tensor],
    *,
    stride: float,
    input_grid_size: float,
    kernel_size: _size_3_t,
    radius_margin: float = 1e-2,
    radius_backend: str = "auto",
    return_num_neighbors: bool = True,
) -> FullCoverStridedRulebook:
    """Build a full-cover overlapping strided point-convolution rulebook.

    Centers are observed input points. Initial centers are nearest input points
    to occupied stride-cell centers; residual uncovered points are covered by a
    deterministic maximal R-net over the uncovered set.
    """
    with torch.no_grad():
        if points.dim() != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3); got {tuple(points.shape)}")
        if sample_inds is None:
            sample_inds = torch.zeros(points.shape[0], device=points.device,
                                      dtype=torch.long)
        sample_inds = sample_inds.to(torch.long)
        if sample_sizes is None:
            sample_sizes = torch.bincount(sample_inds)
        if points.numel() == 0:
            empty = torch.empty(0, device=points.device, dtype=torch.long)
            empty_counts = torch.zeros(0, device=points.device, dtype=torch.long)
            return FullCoverStridedRulebook(
                points=points,
                sample_inds=sample_inds,
                sample_sizes=sample_sizes,
                center_source_indices=empty,
                initial_center_source_indices=empty,
                additional_center_source_indices=empty,
                point_to_initial_center=empty,
                i=empty,
                j=empty,
                k=empty,
                num_neighbors=empty_counts if return_num_neighbors else None,
                i_upsample=empty,
                j_upsample=empty,
                k_upsample=empty,
                radius=0.0,
                radius_scaler=0.0,
                initial_center_counts=torch.zeros_like(sample_sizes),
                additional_center_counts=torch.zeros_like(sample_sizes),
                selector_round_count=0,
                coverage_per_input=empty_counts,
            )

        cell_size = float(stride) * float(input_grid_size)
        radius_scaler = full_cover_radius_scaler(stride, radius_margin)
        radius = float(input_grid_size) * radius_scaler
        required_kernel = minimum_full_cover_kernel_size(radius_scaler)
        ks = _triple(kernel_size)
        K = int(ks[0])
        if not (ks[0] == ks[1] == ks[2]):
            raise ValueError(f"full-cover strided conv requires a cubic kernel; got {ks}")
        if K < required_kernel:
            raise ValueError(
                f"full-cover stride={stride} at grid_size={input_grid_size} "
                f"needs kernel_size >= {required_kernel}; got {kernel_size}")

        c0_source, point_to_c0 = _center_nearest_sources(
            points, sample_inds, cell_size)
        c0_points = points[c0_source]
        c0_sample = sample_inds[c0_source]
        c0_neighbors, c0_counts = radius_search(
            points=points,
            query_points=c0_points,
            radius=radius,
            sample_inds=sample_inds,
            query_sample_inds=c0_sample,
            backend=radius_backend,
        )
        c0_coverage = torch.zeros(points.shape[0], device=points.device,
                                  dtype=torch.long)
        if c0_neighbors.numel() > 0:
            c0_coverage.index_add_(
                0, c0_neighbors.to(torch.long),
                torch.ones(c0_neighbors.numel(), device=points.device,
                           dtype=torch.long),
            )
        uncovered = c0_coverage == 0
        c1_source, rounds = _select_residual_full_cover_centers(
            points, sample_inds, c0_points, point_to_c0, uncovered, radius,
            backend=radius_backend)
        if c1_source.numel() == 0:
            center_source = c0_source
            point_to_c0_final = point_to_c0.to(torch.long)
        elif sample_sizes.numel() == 1:
            center_source = torch.cat([c0_source, c1_source])
            point_to_c0_final = point_to_c0.to(torch.long)
        else:
            combined_source = torch.cat([c0_source, c1_source])
            center_group = torch.cat([
                torch.zeros(c0_source.numel(), device=points.device, dtype=torch.long),
                torch.ones(c1_source.numel(), device=points.device, dtype=torch.long),
            ])
            center_order = torch.argsort(
                sample_inds[combined_source].to(torch.int64) * (2 * (points.shape[0] + 1))
                + center_group * (points.shape[0] + 1)
                + combined_source,
                stable=True,
            )
            center_source = combined_source[center_order]
            source_to_center = torch.empty(points.shape[0], device=points.device,
                                           dtype=torch.long)
            source_to_center[center_source] = torch.arange(
                center_source.numel(), device=points.device, dtype=torch.long)
            point_to_c0_final = source_to_center[c0_source[point_to_c0.long()]]

        center_points = points[center_source].contiguous()
        center_sample = sample_inds[center_source].contiguous()
        center_sizes = torch.bincount(center_sample, minlength=sample_sizes.numel())

        c0_offsets, c0_total = cumsum_exclusive(c0_counts, return_sum=True)
        i_c0_local = repeat_interleave_indices(
            repeats_cumsum=c0_offsets,
            output_size=c0_total,
            may_contain_zero_repeats=False,
        ).to(torch.long)
        i_parts = [i_c0_local]
        j_parts = [c0_neighbors.to(torch.long)]
        num_neighbors_parts = [c0_counts]

        if c1_source.numel() > 0:
            c1_points = points[c1_source]
            c1_sample = sample_inds[c1_source]
            if c1_source.numel() <= 16:
                diff = c1_points[:, None, :] - points[None, :, :]
                same_sample = c1_sample[:, None] == sample_inds[None, :]
                within = same_sample & (torch.sqrt(diff.square().sum(dim=-1)) <= radius)
                i_c1_local, c1_neighbors = torch.nonzero(within, as_tuple=True)
                i_c1_local = i_c1_local.to(torch.long)
                c1_neighbors = c1_neighbors.to(torch.long)
                c1_counts = within.sum(dim=1).to(c0_counts.dtype)
            else:
                c1_neighbors, c1_counts = radius_search(
                    points=points,
                    query_points=c1_points,
                    radius=radius,
                    sample_inds=sample_inds,
                    query_sample_inds=c1_sample,
                    backend=radius_backend,
                )
                c1_offsets, c1_total = cumsum_exclusive(c1_counts, return_sum=True)
                i_c1_local = repeat_interleave_indices(
                    repeats_cumsum=c1_offsets,
                    output_size=c1_total,
                    may_contain_zero_repeats=False,
                ).to(torch.long)
            if sample_sizes.numel() == 1:
                i_parts.append(c0_source.numel() + i_c1_local)
                num_neighbors_parts.append(c1_counts)
            else:
                i_parts[0] = source_to_center[c0_source[i_c0_local]]
                i_parts.append(source_to_center[c1_source[i_c1_local]])
                num_neighbors = torch.zeros(
                    center_points.shape[0], device=points.device, dtype=c0_counts.dtype)
                num_neighbors[source_to_center[c0_source]] = c0_counts
                num_neighbors[source_to_center[c1_source]] = c1_counts
                num_neighbors_parts = []
            j_parts.append(c1_neighbors.to(torch.long))

        num_neighbors = (torch.cat(num_neighbors_parts) if num_neighbors_parts
                         else num_neighbors)
        i = torch.cat(i_parts)
        j = torch.cat(j_parts)
        offsets = points[j] - center_points[i]
        k, d = _strict_voxelize_offsets(
            offsets, grid_size=input_grid_size, kernel_size=kernel_size,
            context="full-cover patchify")
        k, order = torch.sort(k)
        i = i[order]
        j = j[order]
        d = d[order]

        coverage = torch.zeros(points.shape[0], device=points.device,
                               dtype=torch.long)
        coverage.index_add_(
            0, j,
            torch.ones(j.numel(), device=points.device, dtype=torch.long))
        if bool((coverage == 0).any().item()):
            raise RuntimeError(
                "full-cover rulebook construction failed: some input points "
                "have zero incident edges")

        k_up, _ = _strict_voxelize_offsets(
            -d.to(points.dtype) * float(input_grid_size),
            grid_size=input_grid_size,
            kernel_size=kernel_size,
            context="full-cover unpatchify")
        i_up = j
        j_up = i
        k_up, up_order = torch.sort(k_up)
        i_up = i_up[up_order]
        j_up = j_up[up_order]

        idx_dtype = torch.int32 if max(points.shape[0], center_points.shape[0]) <= 2147483647 else torch.int64
        initial_counts = torch.bincount(
            sample_inds[c0_source], minlength=sample_sizes.numel())
        additional_counts = torch.bincount(
            sample_inds[c1_source], minlength=sample_sizes.numel())

        return FullCoverStridedRulebook(
            points=center_points,
            sample_inds=center_sample,
            sample_sizes=center_sizes,
            center_source_indices=center_source.to(idx_dtype),
            initial_center_source_indices=c0_source.to(idx_dtype),
            additional_center_source_indices=c1_source.to(idx_dtype),
            point_to_initial_center=point_to_c0_final.to(idx_dtype),
            i=i.to(idx_dtype),
            j=j.to(idx_dtype),
            k=k.to(idx_dtype),
            num_neighbors=num_neighbors if return_num_neighbors else None,
            i_upsample=i_up.to(idx_dtype),
            j_upsample=j_up.to(idx_dtype),
            k_upsample=k_up.to(idx_dtype),
            radius=radius,
            radius_scaler=radius_scaler,
            initial_center_counts=initial_counts,
            additional_center_counts=additional_counts,
            selector_round_count=rounds,
            coverage_per_input=coverage.to(idx_dtype),
        )


def _build_triplets_from_neighbor_pairs(
    points: Tensor,
    query_points: Tensor,
    neighbor_indices: Tensor,
    num_neighbors: Tensor,
    kernel_indexer: Callable,
    neighbor_radius: float,
    radius_scaler: float,
    sort_by: str,
    return_num_neighbors: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Shared post-radius-search triplet construction.

    Both ``build_triplets`` (which calls ``radius_search``) and
    ``build_triplets_strided_grid`` (which calls
    ``radius_search_strided_grid``) hand off to this helper for the
    cumsum + repeat_interleave + offset + kernel_indexer + sort +
    dtype-norm steps. Behavior matches the original inlined logic in
    ``build_triplets`` line-for-line.

    Internally wraps in ``torch.no_grad()`` — caller does NOT need
    to be inside a no_grad block.
    """
    with torch.no_grad():
        assert torch.min(num_neighbors) > 0, (
            "Neighborhood search failed for some points, consider increase the neighbor_radius. "
            "It is likely that this happens in an *upsample* phase, "
            "where the *query_points* are not a subset of the *points*."
        )

        num_neighbors_cumsum, repeats_sum = cumsum_exclusive(
            num_neighbors, return_sum=True
        )
        center_indices = repeat_interleave_indices(
            repeats_cumsum=num_neighbors_cumsum, output_size=repeats_sum
        )

        offsets = (points[neighbor_indices] - query_points[center_indices])

        i = center_indices
        j = neighbor_indices
        k = kernel_indexer(points=offsets, grid_size=neighbor_radius / radius_scaler)

        if "i" == sort_by:
            pass
        elif "j" == sort_by:
            j, sorter = torch.sort(j)
            i = i[sorter]
            k = k[sorter]
        elif "k" == sort_by:
            k, sorter = torch.sort(k)
            i = i[sorter]
            j = j[sorter]
        else:
            assert (
                False
            ), f'Unknown sort_by argument "{sort_by}", it should be i, j, or k!'

        # Normalize all triplet indices to a consistent dtype.
        # The upstream code paths produce mixed dtypes (i: int64 from cumsum,
        # j: int32 from radius_search, k: int64 from torch.sum promotion).
        # Triton's autotuner keys on pointer dtypes, so mixed dtypes cause
        # redundant cache entries and mid-training autotune stalls.
        # Use int32 when safe, int64 otherwise. We check point counts
        # (already CPU ints) as upper bounds instead of calling .max().item()
        # which would force GPU→CPU syncs.
        _INT32_MAX = 2147483647
        max_possible = max(query_points.shape[0], points.shape[0])
        idx_dtype = torch.int32 if max_possible <= _INT32_MAX else torch.int64
        i = i.to(idx_dtype)
        j = j.to(idx_dtype)
        k = k.to(idx_dtype)

        if return_num_neighbors:
            return i, j, k, num_neighbors
        else:
            return i, j, k, None


def voxelize_3d(
    kernel_size: _size_3_t,
    points: Tensor,
    grid_size: Optional[Tensor] = None,
) -> Tensor:
    with torch.no_grad():
        sizes = _triple(kernel_size)
        ks = sizes[0]

        kernel_offset = (points / grid_size).round().int()
        k_l = torch.clamp(kernel_offset + (ks // 2), min=0, max=ks - 1)
        multipliers = torch.tensor([ks * ks, ks, 1], device=points.device, dtype=torch.int32)
        indices = torch.sum(k_l * multipliers, dim=1)

    return indices


def should_use_direct_segmented_triplets(
    kernel_size: _size_3_t,
) -> bool:
    """Return the shape-static production policy for direct tap emission.

    Direct tap emission wins across the measured production K1/K3/K5/K7/K15
    shapes after striped count and compact passes. Keeping this decision
    kernel-shape-only avoids data-dependent dispatch and per-iteration
    synchronization; custom indexers and non-ball searches retain the generic
    builder.
    """
    sizes = _triple(kernel_size)
    return (
        sizes[0] == sizes[1] == sizes[2]
        and int(sizes[0]) >= 1
        and int(sizes[0]) % 2 == 1
    )


@torch.compiler.disable
def build_triplets_segmented(
    points: Tensor,
    sample_inds: Tensor,
    sample_sizes: Tensor,
    neighbor_radius: float,
    kernel_size: _size_3_t,
    query_points: Optional[Tensor] = None,
    query_sample_inds: Optional[Tensor] = None,
    query_sample_sizes: Optional[Tensor] = None,
    return_num_neighbors: bool = False,
    radius_scaler: Optional[float] = None,
    tap_stripes: int = 32,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Build k-major convolution triplets directly during radius search.

    The sorted-eight search counts accepted edges per kernel tap, then writes
    i and j into their final tap segments. It therefore avoids the query-major
    i reconstruction, per-edge offset tensor, voxelization, global k sort, and
    post-sort gathers used by the generic builder.

    k is still materialized from the already-known segment lengths for
    compatibility with convolution engines that consume it. TIG consumes
    seg_offs directly and does not re-derive the same boundaries. The static
    32-stripe default reduces tap-counter contention without per-batch tuning.
    """
    sizes = _triple(kernel_size)
    if not (sizes[0] == sizes[1] == sizes[2]):
        raise ValueError(
            f"segmented triplet construction requires a cubic kernel; got {sizes}")
    kernel_size_scalar = int(sizes[0])
    if kernel_size_scalar <= 0 or kernel_size_scalar % 2 == 0:
        raise ValueError(
            "segmented triplet construction requires a positive odd kernel; "
            f"got {kernel_size_scalar}")
    if radius_scaler is None or float(radius_scaler) <= 0:
        raise ValueError(
            "segmented triplet construction requires a positive radius_scaler")
    kernel_grid_size = float(neighbor_radius) / float(radius_scaler)

    with torch.no_grad():
        if (
            query_points is None
            or query_sample_inds is None
            or query_sample_sizes is None
        ):
            assert query_points is None
            assert query_sample_inds is None
            assert query_sample_sizes is None
            query_points = points
            query_sample_inds = sample_inds
            query_sample_sizes = sample_sizes

        cache = _active_cache()
        key = None
        if cache is not None:
            indexer = partial(voxelize_3d, kernel_size=sizes)
            base_key = triplet_key(
                points, query_points, neighbor_radius, radius_scaler,
                "k", return_num_neighbors, sample_inds,
                query_sample_inds, indexer,
            )
            if base_key is not None:
                key = base_key + (
                    "direct_kernel_segments_v2", int(tap_stripes))
                hit = cache.get(key)
                if hit is not None:
                    cached_out, pref = hit
                    if pref() is not None:
                        return cached_out

        i, j, seg_offs, num_neighbors = radius_search_sorted_grid8_segments(
            points=points,
            queries=query_points,
            radius=neighbor_radius,
            kernel_size=kernel_size_scalar,
            kernel_grid_size=kernel_grid_size,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
            dtype_num_neighbors=torch.int64,
            distance_type="ball",
            tap_stripes=tap_stripes,
        )
        if num_neighbors.numel() and bool((num_neighbors == 0).any().item()):
            raise AssertionError(
                "Neighborhood search failed for some points, consider increase "
                "the neighbor_radius. It is likely that this happens in an "
                "*upsample* phase, where the query_points are not a subset of "
                "the points.")

        tap_ids = torch.arange(
            kernel_size_scalar ** 3, dtype=torch.int32, device=points.device)
        k = torch.repeat_interleave(
            tap_ids, seg_offs.diff(), output_size=i.shape[0])
        num_neighbors_out = num_neighbors if return_num_neighbors else None
        out = (i, j, k, seg_offs, num_neighbors_out)

        if cache is not None and key is not None:
            cache[key] = (out, weakref.ref(points))
        return out


@torch.compiler.disable
def build_triplets(
    points: Tensor,
    sample_inds: Tensor,
    sample_sizes: Tensor,
    neighbor_radius: float,
    kernel_indexer: Callable,
    query_points: Optional[Tensor] = None,
    query_sample_inds: Optional[Tensor] = None,
    query_sample_sizes: Optional[Tensor] = None,
    sort_by: str = "k",
    return_num_neighbors=False,
    radius_scaler: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
    with torch.no_grad():
        # They have to be all None, or all not None
        if (
            query_points is None
            or query_sample_inds is None
            or query_sample_sizes is None
        ):
            assert query_points is None
            assert query_sample_inds is None
            assert query_sample_sizes is None
            query_points = points
            query_sample_inds = sample_inds
            query_sample_sizes = sample_sizes

        cache = _active_cache()
        key = None
        if cache is not None:
            key = triplet_key(
                points, query_points, neighbor_radius, radius_scaler,
                sort_by, return_num_neighbors, sample_inds,
                query_sample_inds, kernel_indexer,
            )
            if key is not None:           # None => uncacheable indexer; just build
                hit = cache.get(key)
                if hit is not None:
                    cached_out, pref = hit
                    if pref() is not None:   # keyed tensor alive => addr not recycled
                        return cached_out
                    # stale (data_ptr recycled): fall through, rebuild, overwrite

        grid_size = neighbor_radius / radius_scaler if radius_scaler is not None else None
        neighbor_indices, num_neighbors = radius_search(
            points=points,
            query_points=query_points,
            radius=neighbor_radius,
            sample_inds=sample_inds,
            sample_sizes=sample_sizes,
            query_sample_inds=query_sample_inds,
            query_sample_sizes=query_sample_sizes,
            grid_size=grid_size,
        )

    out = _build_triplets_from_neighbor_pairs(
        points=points,
        query_points=query_points,
        neighbor_indices=neighbor_indices,
        num_neighbors=num_neighbors,
        kernel_indexer=kernel_indexer,
        neighbor_radius=neighbor_radius,
        radius_scaler=radius_scaler,
        sort_by=sort_by,
        return_num_neighbors=return_num_neighbors,
    )
    if cache is not None and key is not None:
        # Weakref the primary keyed tensor `points`; on a hit a dead weakref is
        # treated as a miss, closing the freed-then-recycled-address window.
        # We guard ONLY `points`, not the other data_ptr-keyed tensors
        # (sample_inds / query / query_sample_inds): every builder here stores
        # those on the SAME MetaData (or its parent chain) as `points`, so they
        # share a lifetime and are co-pinned for the whole forward scope — a live
        # `points` implies its companions are live. A public-API caller pairing a
        # persistent `points` with a transient, recycled `sample_inds` of
        # different contents would be the only way to defeat this; such a caller
        # should weakref-guard all keyed tensors.
        cache[key] = (out, weakref.ref(points))
    return out


def radius_scaler_for_kernel_size(kernel_size: _size_3_t, receptive_field_scaler: float = 1.0, distance_type: str = "ball"):
    """Compute the radius_scaler that determines the neighborhood search radius.

    PointCNN++ decouples kernel resolution from the receptive field's physical
    span. kernel_size controls the fineness of the weight grid (e.g., 3^3 or
    5^3 kernel cells), while receptive_field_scaler controls how much physical
    space the search sphere covers, as a volume multiplier of the kernel_size^3
    cube. This decoupling allows using a fine kernel (large kernel_size) over
    a small region, or a coarse kernel over a large region, independently.

    The returned radius_scaler is used as:
        neighbor_radius = grid_size * radius_scaler

    And inside build_triplets, grid_size is recovered as:
        grid_size = neighbor_radius / radius_scaler

    so that voxelize_3d always discretizes at the original grid_size scale,
    regardless of receptive_field_scaler.

    Args:
        kernel_size: Number of kernel cells per dimension (e.g., 3 -> 3^3 = 27 weights)
        receptive_field_scaler: Volume multiplier. 1.0 means the search sphere has
            the same volume as the kernel_size^3 cube. >1.0 searches wider (more
            neighbors per kernel cell). Default: 1.0
        distance_type: "ball" (spherical search) or "cube" (cubic search)

    Returns:
        radius_scaler: Multiply by grid_size to get the physical search radius.
            For kernel_size=3, ball mode, rfs=1.0: radius_scaler ≈ 1.86
    """
    kernel_size = _triple(kernel_size)
    kernel_size_max = max(kernel_size)

    if distance_type == "cube":
        cube_edge_length = kernel_size_max * math.pow(receptive_field_scaler, 1 / 3)
        radius_scaler = cube_edge_length / 2.0
    else:
        volume = math.pow(kernel_size_max, 3) * receptive_field_scaler
        radius_scaler = math.pow(3 * volume / (4 * math.pi), 1 / 3)

    return radius_scaler


def _no_overlap_check(
    kernel_size: _size_3_t,
    receptive_field_scaler: float,
    distance_type: str,
    stride: float,
) -> float:
    """Enforce the no-overlap invariant and return the computed radius_scaler.

    Raises ``ValueError`` if ``2 * radius_scaler > stride`` — the
    ball-disjoint contract required by ``conv_with_stride_disjoint`` and
    ``handle_stride_disjoint_and_build_triplets``.
    """
    radius_scaler = radius_scaler_for_kernel_size(
        kernel_size, receptive_field_scaler, distance_type
    )
    if 2 * radius_scaler > stride:
        raise ValueError(
            f"conv_with_stride_disjoint no-overlap violated: "
            f"2 * radius_scaler ({2 * radius_scaler:.4f}) > stride ({stride:.4f}) "
            f"with kernel_size={kernel_size}, "
            f"receptive_field_scaler={receptive_field_scaler}, "
            f"distance_type={distance_type!r}. "
            f"Reduce receptive_field_scaler, increase stride, or use "
            f"handle_stride_and_build_triplets / conv_with_stride for "
            f"overlap-allowed strided conv."
        )
    return radius_scaler


@torch.compiler.disable
def build_triplets_strided_grid(
    points: Tensor,
    sample_inds: Tensor,
    sample_sizes: Tensor,
    neighbor_radius: float,
    kernel_indexer: Callable,
    cell_size: float,
    query_points: Optional[Tensor] = None,
    query_sample_inds: Optional[Tensor] = None,
    query_sample_sizes: Optional[Tensor] = None,
    sort_by: str = "k",
    return_num_neighbors: bool = False,
    radius_scaler: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Strided grid-downsample variant of build_triplets.

    Identical shape contract as ``build_triplets``, but the neighbor
    query is ``radius_search_strided_grid`` (Triton-fused inner-loop,
    3.58-9.11x faster at H200 ball-disjoint regimes).

    Required: ``cell_size`` MUST be set (typically
    ``stride * grid_size_in``). Caller is responsible for ensuring
    no-overlap (2 * radius_scaler * grid_size_in <= cell_size); this
    function does NOT re-validate (the wrapper
    ``handle_stride_disjoint_and_build_triplets`` and the public
    ``conv_with_stride_disjoint`` enforce the contract).
    """
    with torch.no_grad():
        if (
            query_points is None
            or query_sample_inds is None
            or query_sample_sizes is None
        ):
            assert query_points is None
            assert query_sample_inds is None
            assert query_sample_sizes is None
            query_points = points
            query_sample_inds = sample_inds
            query_sample_sizes = sample_sizes

        neighbor_indices, num_neighbors = radius_search_strided_grid(
            points=points,
            queries=query_points,
            radius=neighbor_radius,
            cell_size=cell_size,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
        )

    return _build_triplets_from_neighbor_pairs(
        points=points,
        query_points=query_points,
        neighbor_indices=neighbor_indices,
        num_neighbors=num_neighbors,
        kernel_indexer=kernel_indexer,
        neighbor_radius=neighbor_radius,
        radius_scaler=radius_scaler,
        sort_by=sort_by,
        return_num_neighbors=return_num_neighbors,
    )


@torch.compiler.disable
def handle_stride_and_build_triplets(
    m: MetaData,
    stride: float,
    kernel_size: _size_3_t = _triple(3),
    receptive_field_scaler: float = 1.0,
    sort_by: str = "k",
    return_num_neighbors=False,
    distance_type: str = "ball",
) -> MetaData:
    with torch.no_grad():
        if stride != 1:
            if (m.parent is not None and
                m.parent.points.shape == m.points.shape and
                m.parent.grid_size == m.grid_size and
                torch.equal(m.parent.points, m.points) and
                torch.equal(m.parent.sample_inds, m.sample_inds)):
                parent_meta = m.parent
            else:
                parent_meta = MetaData(
                    points=m.points,
                    sample_inds=m.sample_inds,
                    sample_sizes=m.sample_sizes,
                    grid_size=m.grid_size,
                    parent=m.parent,
                )
                m.parent = parent_meta

            points_ = parent_meta.points
            sample_inds_ = parent_meta.sample_inds
            sample_sizes_ = parent_meta.sample_sizes
            grid_size_ = parent_meta.grid_size

            radius_scaler = radius_scaler_for_kernel_size(kernel_size, receptive_field_scaler, distance_type)
            neighbor_radius = grid_size_ * radius_scaler

            m.points, m.sample_inds, m.grid_size, m.downsample_indices = downsample(
                m.points, m.sample_inds, m.grid_size, stride
            )
            m.sample_sizes = torch.bincount(m.sample_inds)

            if (
                sort_by == "k"
                and distance_type == "ball"
                and should_use_direct_segmented_triplets(kernel_size)
            ):
                (m.i, m.j, m.k, m.seg_offs,
                 m.num_neighbors) = build_triplets_segmented(
                    points=points_,
                    sample_inds=sample_inds_,
                    sample_sizes=sample_sizes_,
                    neighbor_radius=neighbor_radius,
                    kernel_size=kernel_size,
                    query_points=m.points,
                    query_sample_inds=m.sample_inds,
                    query_sample_sizes=m.sample_sizes,
                    return_num_neighbors=return_num_neighbors,
                    radius_scaler=radius_scaler,
                )
            else:
                m.i, m.j, m.k, m.num_neighbors = build_triplets(
                    points=points_,
                    sample_inds=sample_inds_,
                    sample_sizes=sample_sizes_,
                    neighbor_radius=neighbor_radius,
                    kernel_indexer=partial(
                        voxelize_3d, kernel_size=kernel_size),
                    query_points=m.points,
                    query_sample_inds=m.sample_inds,
                    query_sample_sizes=m.sample_sizes,
                    sort_by=sort_by,
                    return_num_neighbors=return_num_neighbors,
                    radius_scaler=radius_scaler,
                )
                if sort_by == "k":
                    from sparse_engines._seg_offs import (
                        kernel_offset_segments)
                    m.seg_offs = kernel_offset_segments(
                        m.k, math.prod(_triple(kernel_size)))
                else:
                    m.seg_offs = None

            parent_meta.i_upsample = m.j
            parent_meta.j_upsample = m.i
            parent_meta.k_upsample = m.k
            parent_meta.seg_offs_upsample = m.seg_offs

        if m.empty_triplets():
            radius_scaler = radius_scaler_for_kernel_size(kernel_size, receptive_field_scaler, distance_type)
            neighbor_radius = m.grid_size * radius_scaler
            if (
                sort_by == "k"
                and distance_type == "ball"
                and should_use_direct_segmented_triplets(kernel_size)
            ):
                (m.i, m.j, m.k, m.seg_offs,
                 m.num_neighbors) = build_triplets_segmented(
                    points=m.points,
                    sample_inds=m.sample_inds,
                    sample_sizes=m.sample_sizes,
                    neighbor_radius=neighbor_radius,
                    kernel_size=kernel_size,
                    return_num_neighbors=return_num_neighbors,
                    radius_scaler=radius_scaler,
                )
            else:
                m.i, m.j, m.k, m.num_neighbors = build_triplets(
                    points=m.points,
                    sample_inds=m.sample_inds,
                    sample_sizes=m.sample_sizes,
                    neighbor_radius=neighbor_radius,
                    kernel_indexer=partial(
                        voxelize_3d, kernel_size=kernel_size),
                    sort_by=sort_by,
                    return_num_neighbors=return_num_neighbors,
                    radius_scaler=radius_scaler,
                )
                if sort_by == "k":
                    from sparse_engines._seg_offs import (
                        kernel_offset_segments)
                    m.seg_offs = kernel_offset_segments(
                        m.k, math.prod(_triple(kernel_size)))
                else:
                    m.seg_offs = None

    # Submanifold / strided (overlapping) rulebook: variable per-tap neighbour
    # counts (T != n, T != n_in) ⇒ neither exact cover, not uniform. The
    # k_sorted flag is HONEST about the requested sort: conv builders pass the
    # default sort_by="k" (→ submanifold(), k_sorted=True), but max_pool3d calls
    # this with sort_by="i" — its triplets are i-sorted, so stamping
    # k_sorted=True would be a dishonest contract (a conv consuming it would
    # silently route the assume-k-sorted TIG path on i-sorted data). max_pool3d
    # feeds indexed_segment_reduce, never a conv (contract.py), so this is
    # latent today; deriving k_sorted from sort_by keeps the contract truthful
    # if that ever changes. Structural constant — no host reduction.
    m.contract = TripletContract(k_sorted=(sort_by == "k"))
    return m


@torch.compiler.disable
def handle_stride_disjoint_and_build_triplets(
    m: MetaData,
    stride: float,
    kernel_size: _size_3_t = _triple(3),
    sort_by: str = "k",
) -> MetaData:
    """DISJOINT strided conv as a true GRID PARTITION (no radius search).

    Each input point is assigned to exactly ONE output cell by integer
    voxelization at ``cell_size = stride * grid_size`` — a tiling that is
    disjoint *and* covering (every point lands in exactly one cell; no point is
    orphaned). The output token sits at the **cell (voxel) center**
    (``grid_sample_filter(reduction="center")``), and the kernel-bucket ``k`` is
    the point's **cell-grid-relative sub-voxel slot**
    ``floor((point - cell_origin)/cell_size * K) in [0, K)`` per axis (cubic
    kernel ``K = kernel_size``), flattened to ``[0, K**3)``. The slot is bounded
    by construction (no clamp / no aliasing) and decoupled from the token
    coordinate: the slot answers "which sub-position within the patch", the cell
    center answers "where is the patch".

    This is the point analogue of a ViT ``Conv2d(kernel=P, stride=P)`` patchify:
    a partition into non-overlapping cubic patches with the weight indexed by
    the sub-position. The cached ``(i, j, k)_upsample`` edges are the exact
    inverse map, so an ``Upsample(straight_recover=True, recompute_k=False)``
    recovers EVERY raw point from its cell-center token (the unpatchify) with
    full coverage — see ``layers/upsample.py``.

    Supersedes the prior ball-radius-search disjoint conv, which only gathered
    the inscribed sphere of the patch and orphaned ~25-50% of points at the
    corners. ``receptive_field_scaler`` / ``distance_type`` no longer apply (a
    partition has no search radius) and were removed from the signature.
    """
    if stride <= 1:
        raise ValueError(
            f"handle_stride_disjoint_and_build_triplets requires stride > 1 "
            f"(downsample case); got stride={stride}."
        )
    ks = _triple(kernel_size)
    if not (ks[0] == ks[1] == ks[2]):
        raise ValueError(
            f"partition disjoint conv requires a cubic kernel; got kernel_size={ks}")
    K = int(ks[0])

    with torch.no_grad():
        if (m.parent is not None and
            m.parent.points.shape == m.points.shape and
            m.parent.grid_size == m.grid_size and
            torch.equal(m.parent.points, m.points) and
            torch.equal(m.parent.sample_inds, m.sample_inds)):
            parent_meta = m.parent
        else:
            parent_meta = MetaData(
                points=m.points,
                sample_inds=m.sample_inds,
                sample_sizes=m.sample_sizes,
                grid_size=m.grid_size,
                parent=m.parent,
            )
            m.parent = parent_meta

        points_ = parent_meta.points
        sample_inds_ = parent_meta.sample_inds
        grid_size_ = parent_meta.grid_size
        cell_size = stride * grid_size_

        # Grid partition: each point -> its cell; token coord = cell CENTER.
        # `mapping` (lookup_inds) is per-point -> token row, aligned with the
        # returned cell-center coords (100% coverage by construction).
        points_c, sample_inds_c, ds_indices, mapping = grid_sample_filter(
            points=points_,
            grid_size=cell_size,
            sample_inds=sample_inds_,
            reduction="center",
            return_mapping=True,
        )
        m.points = points_c.contiguous()
        m.sample_inds = sample_inds_c
        m.grid_size = cell_size
        m.sample_sizes = torch.bincount(sample_inds_c)
        m.downsample_indices = ds_indices

        # Cell-grid-relative sub-voxel slot, bounded [0, K) per axis. cell_vox via
        # floor(pt/cell_size) matches grid_sample_filter's cell assignment
        # (compute_grid_indices).
        cell_vox = torch.div(points_, cell_size, rounding_mode="floor")
        frac = points_ / cell_size - cell_vox                 # [0, 1) per axis
        sub = torch.floor(frac * K).to(torch.long).clamp_(0, K - 1)
        kk = (sub[:, 0] * K + sub[:, 1]) * K + sub[:, 2]      # [0, K**3)

        n_pts = points_.shape[0]
        idx = torch.arange(n_pts, device=points_.device)
        if sort_by == "k":  # keep the k-ascending contract (cached k_upsample reuse)
            order = torch.argsort(kk)
            idx, kk = idx[order], kk[order]
        m.i = mapping.to(torch.long)[idx]   # output token per edge (one edge / point)
        m.j = idx                           # input point per edge
        m.k = kk
        from sparse_engines._seg_offs import kernel_offset_segments
        m.seg_offs = kernel_offset_segments(m.k, K ** 3)
        m.num_neighbors = None

        parent_meta.i_upsample = m.j
        parent_meta.j_upsample = m.i
        parent_meta.k_upsample = m.k
        parent_meta.seg_offs_upsample = m.seg_offs

    # Disjoint strided rulebook is the one builder that can be fan-out-1: each
    # input point contributes to AT MOST one output cell, so when every input
    # row appears in exactly one triplet (T == n_in) the partition-stem
    # exact_cover_in contract holds (routes FI1 grad_input + the dense-GEMM
    # partition engine). Prove it ONCE here with the same memoized bincount the
    # old forward used — inside this @compiler.disable region the .item() is
    # free and never traced (vs. per-forward in _conv_forward, which broke
    # compile). Gate on the free host checks first (T == n_in and T != n),
    # exactly mirroring the retired conv.py auto-detect. exact_cover_out stays
    # False (deconv/strided is never fan-in-1 here); not uniform (radius search).
    _eci = False
    if m.k is not None and m.k.numel() > 0:
        n_out = m.points.shape[0]
        n_in = points_.shape[0]
        if m.k.numel() == n_in and m.k.numel() != n_out:
            from sparse_engines._seg_offs import exact_cover_cached
            _eci = exact_cover_cached(m.j, n_in)
    m.contract = TripletContract(k_sorted=True, exact_cover_out=False,
                                 exact_cover_in=_eci, uniform_seg_len=None)
    return m
