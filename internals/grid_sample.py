"""Grid-based point cloud downsampling with configurable reduction modes.

Voxelizes a point cloud and selects one representative point per voxel.
Supports center_nearest (closest to voxel center), random, mean, and
center reductions.
"""

import torch
from .grid_indexing import build_sorted_grid_segments
from .grid_indexing import compute_grid_indices
from .grid_sample_triton_kernel import center_nearest_segment_indices


def _center_nearest_indices_torch(
    points, grid_inds, grid_size, sorter, grid_sizes, lookup_inds
):
    """Reference selector with deterministic lowest-source-index tie breaking."""
    grid_inds_pure = grid_inds[:, :3]
    grid_centers = (grid_inds_pure + 0.5).mul_(grid_size)
    distances = torch.sum(grid_centers.sub_(points).square_(), dim=-1)
    distances_min = torch.segment_reduce(
        distances[sorter], reduce="min", lengths=grid_sizes
    )
    point_indices = torch.arange(points.shape[0], device=points.device)
    candidates = torch.where(
        distances == distances_min[lookup_inds],
        point_indices,
        torch.full_like(point_indices, points.shape[0]),
    )
    selected = torch.full(
        (grid_sizes.numel(),),
        points.shape[0],
        dtype=point_indices.dtype,
        device=points.device,
    )
    selected.scatter_reduce_(
        0, lookup_inds, candidates, reduce="amin", include_self=True
    )
    return selected


def grid_sample_filter(
    points,
    grid_size,
    sample_inds=None,
    reduction="center_nearest",
    return_mapping=False,
    center_nearest_impl="auto",
):
    reduction_list = ["center", "center_nearest", "random", "mean"]
    assert reduction in reduction_list, "reduction should be in one of %s!" % (
        " ".join(reduction_list)
    )
    if center_nearest_impl not in ("auto", "torch", "triton"):
        raise ValueError("center_nearest_impl must be one of: auto, torch, triton")
    selector_impl = center_nearest_impl
    if selector_impl == "auto":
        selector_impl = "triton" if points.is_cuda else "torch"
    if selector_impl == "triton" and not points.is_cuda:
        raise ValueError("center_nearest_impl='triton' requires CUDA tensors")

    grid_inds = compute_grid_indices(points, grid_size, sample_inds)
    is_center_nearest = reduction == "center_nearest"
    require_inverse = (
        return_mapping
        or reduction == "mean"
        or (is_center_nearest and selector_impl == "torch")
    )
    sorter, grid_sizes, lookup_inds = build_sorted_grid_segments(
        grid_inds,
        return_inverse=require_inverse,
    )
    num_points_filtered = grid_sizes.numel()
    indices = None

    if is_center_nearest:
        if selector_impl == "triton":
            indices = center_nearest_segment_indices(
                points,
                grid_inds,
                sorter,
                grid_sizes,
                grid_size,
            )
        else:
            indices = _center_nearest_indices_torch(
                points, grid_inds, grid_size, sorter, grid_sizes, lookup_inds
            )
    else:
        segment_ends = torch.cumsum(grid_sizes, dim=0).sub_(1)
        indices = sorter[segment_ends]

    # use indices to get sample_inds_filtered for return
    sample_inds_filtered = None
    if sample_inds is not None:
        sample_inds_filtered = sample_inds[indices]

    if "center" == reduction:
        # use grid centers as points_filtered to return
        # ``sample_inds`` is appended to the lookup key for batch isolation,
        # but it is not a spatial coordinate.  Keep the public point shape
        # [num_voxels, 3] for batched and unbatched inputs alike.
        grid_inds_pure = grid_inds if sample_inds is None else grid_inds[:, :3]
        grid_inds_filtered = grid_inds_pure[indices]
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
