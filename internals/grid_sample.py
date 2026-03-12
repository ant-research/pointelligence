import torch
from .indexing import arange_cached
from .grid_lookup import build_lookup_struct
from .grid_lookup import compute_grid_indices


def grid_sample_filter(
    points,
    grid_size,
    sample_inds=None,
    reduction="center_nearest",
    return_mapping=False,
):
    reduction_list = ["center", "center_nearest", "random", "mean"]
    assert reduction in reduction_list, "reduction should be in one of %s!" % (
        " ".join(reduction_list)
    )

    grid_inds = compute_grid_indices(points, grid_size, sample_inds, with_shifts=False)
    require_sorter_and_unique_counts = "center_nearest" == reduction
    x = build_lookup_struct(
        grid_inds,
        return_unique_inds=True,
        return_sorter=require_sorter_and_unique_counts,
        return_unique_counts=require_sorter_and_unique_counts,
    )
    lookup_struct, lookup_inds, indices = x[0], x[1], x[2]
    num_points_filtered = lookup_struct.size()
    del lookup_struct

    if "center_nearest" == reduction:
        grid_inds_pure = grid_inds if sample_inds is None else grid_inds[:, :3]
        grid_centers = (grid_inds_pure + 0.5).mul_(grid_size)
        distances = torch.sum(grid_centers.sub_(points).square_(), dim=-1)

        sorter, grid_sizes = x[3], x[4]
        distances_min = torch.segment_reduce(
            distances[sorter], reduce="min", lengths=grid_sizes
        )
        distances_min_repeated = distances_min[lookup_inds]

        nearest_mask = distances_min_repeated.sub_(distances) == 0
        nearest_indices = torch.nonzero(nearest_mask).squeeze(dim=-1)

        if not return_mapping and nearest_indices.numel() == indices.numel():
            indices = nearest_indices
        else:
            # in case there are multiple min distances in a neighborhood
            # in case return_mapping need make sampled points order match lookup_inds
            inds_arange = arange_cached(lookup_inds.shape[0], device=points.device)
            indices.index_copy_(
                0, lookup_inds[nearest_indices], inds_arange[nearest_indices]
            )

    # use indices to get sample_inds_filtered for return
    sample_inds_filtered = None
    if sample_inds is not None:
        sample_inds_filtered = sample_inds[indices]

    if "center" == reduction:
        # use grid centers as points_filtered to return
        grid_inds_filtered = grid_inds[indices]
        points_filtered = (grid_inds_filtered + 0.5).mul_(grid_size)
    elif "random" == reduction or "center_nearest" == reduction:
        # use indices to get points_filtered for return
        points_filtered = points[indices]
    else:  # reduction == 'mean':
        # compute points_filtered by average of all points in the grid
        points_filtered = torch.empty(
            (num_points_filtered, 3), dtype=points.dtype, device=points.device
        )
        points_filtered.index_reduce_(
            dim=0, index=lookup_inds, source=points, reduce="mean", include_self=False
        )

    # use lookup_inds(pts in which grid) for mapping
    mapping = lookup_inds if return_mapping else None

    return points_filtered, sample_inds_filtered, indices, mapping
