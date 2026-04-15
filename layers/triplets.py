from functools import partial
from typing import Tuple, Callable, Optional

import math
import torch

from torch import Tensor
from torch.nn.common_types import _size_3_t
from torch.nn.modules.utils import _triple

from internals.neighbors import radius_search
from internals.indexing import cumsum_exclusive, repeat_interleave_indices

from .metadata import MetaData
from .downsample import downsample

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
    radius_scaler: float=None,
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

        neighbor_indices, num_neighbors = radius_search(
            points=points,
            query_points=query_points,
            radius=neighbor_radius,
            sample_inds=sample_inds,
            sample_sizes=sample_sizes,
            query_sample_inds=query_sample_inds,
            query_sample_sizes=query_sample_sizes,
        )

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

        # Calculate actual offsets (not normalized)
        # The voxelize_3d function will handle normalization internally
        offsets = (points[neighbor_indices] - query_points[center_indices])

        i = center_indices
        j = neighbor_indices
        k = kernel_indexer(points=offsets, grid_size=neighbor_radius / radius_scaler)

        if "i" == sort_by:
            # The indices from radius_search are already sorted by i,
            # so the sorting could be skipped here.
            pass
            # i, sorter = torch.sort(i)
            # j = j[sorter]
            # k = k[sorter]
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


def radius_scaler_for_kernel_size(kernel_size: _size_3_t, receptive_field_scaler: float = 1.0, distance_type: str = "ball"):
    kernel_size = _triple(kernel_size)
    kernel_size_max = max(kernel_size)

    if distance_type == "cube":
        cube_edge_length = kernel_size_max * math.pow(receptive_field_scaler, 1 / 3)
        radius_scaler = cube_edge_length / 2.0
    else:
        volume = math.pow(kernel_size_max, 3) * receptive_field_scaler
        radius_scaler = math.pow(3 * volume / (4 * math.pi), 1 / 3)
        
    return radius_scaler


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

            # Use PointOps style mapping: pass grid_size directly to build_triplets
            m.i, m.j, m.k, m.num_neighbors = build_triplets(
                points=points_,
                sample_inds=sample_inds_,
                sample_sizes=sample_sizes_,
                neighbor_radius=neighbor_radius,
                kernel_indexer=partial(voxelize_3d, kernel_size=kernel_size),
                query_points=m.points,
                query_sample_inds=m.sample_inds,
                query_sample_sizes=m.sample_sizes,
                sort_by=sort_by,
                return_num_neighbors=return_num_neighbors,
                radius_scaler=radius_scaler,
            )

            parent_meta.i_upsample = m.j
            parent_meta.j_upsample = m.i
            parent_meta.k_upsample = m.k

        if m.empty_triplets():
            radius_scaler = radius_scaler_for_kernel_size(kernel_size, receptive_field_scaler, distance_type)
            neighbor_radius = m.grid_size * radius_scaler
            # Use PointOps style mapping: pass radius_scaler to build_triplets
            m.i, m.j, m.k, m.num_neighbors = build_triplets(
                points=m.points,
                sample_inds=m.sample_inds,
                sample_sizes=m.sample_sizes,
                neighbor_radius=neighbor_radius,
                kernel_indexer=partial(voxelize_3d, kernel_size=kernel_size),
                sort_by=sort_by,
                return_num_neighbors=return_num_neighbors,
                radius_scaler=radius_scaler,
            )

    return m
