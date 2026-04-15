"""Voxel-based point cloud downsampling.

Reduces point density by selecting representative points per voxel
using grid_sample_filter with center_nearest mode, which preserves
thin structures better than random selection.
"""

import torch
from torch import Tensor

from internals.grid_sample import grid_sample_filter


@torch.compiler.disable
def downsample(
    points: Tensor,
    sample_inds: Tensor,
    grid_size: float,
    stride: float,
):
    """
    Downsample point cloud by voxelizing and selecting representative points.
    
    Args:
        points: Point coordinates [N, 3]
        sample_inds: Sample indices for each point
        grid_size: Current grid size
        stride: Downsample stride factor
        
    Returns:
        points_, sample_inds_, grid_size, indices
    """
    with torch.no_grad():
        grid_size = grid_size * stride
        points_, sample_inds_, indices, _ = grid_sample_filter(
            points=points,
            grid_size=grid_size,
            sample_inds=sample_inds,
            reduction="center_nearest",
            return_mapping=True,
        )
        return points_, sample_inds_, grid_size, indices
