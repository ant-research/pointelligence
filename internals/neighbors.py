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

from sparse_engines.indexed_distance_triton import indexed_distance


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
    distances_all = indexed_distance(queries, q_inds, points, p_inds, distance_type=distance_type)
    neighbors_mask = distances_all <= radius
    neighbors_indices = torch.nonzero(neighbors_mask).squeeze(-1)
    del neighbors_mask

    distances = distances_all[neighbors_indices] if return_distances else None
    del distances_all

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
):
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

    p_inds = arange_cached(q_inds.numel(), device=device, dtype=torch.int32)
    p_inds = p_point_inds[p_inds_offset.add_(p_inds)]
    del p_inds_offset
    del p_point_inds

    results = clamp_by_radius(
        queries,
        q_inds,
        points,
        p_inds,
        radius,
        return_distances,
        dtype_num_neighbors,
        distance_type,
    )
    del q_inds, p_inds

    return results


def radius_search_brute_force(points, queries, radius, return_distances=False):
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
):
    point_num = max(points.shape[0], query_points.shape[0])

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
