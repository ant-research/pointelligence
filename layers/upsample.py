import torch
import warnings
from torch import Tensor
from typing import Tuple
from functools import partial

from layers.metadata import MetaData
from layers.conv import PointConv3d
from layers.triplets import build_triplets, voxelize_3d, radius_scaler_for_kernel_size


class Upsample(torch.nn.Module):
    """
    Upsample layer: each high-res output point gathers features from nearby
    low-res input points via learned convolution weights.

        output[i_high] += weight[k] @ input[j_low]

    The neighborhood search is direct: high-res points (query) search among
    low-res points (source). The search radius uses grid_size_low as its base
    to guarantee every high-res point finds at least one low-res neighbor.

    Minimum radius requirement:
        Low-res points are spaced at grid_size_low (one per voxel from
        center_nearest downsampling). The worst case is a high-res point at
        a voxel corner, at distance sqrt(3)/2 * grid_size_low from the
        nearest low-res point. The radius must exceed this:

            radius_min = grid_size_low * sqrt(3) / 2  (~0.87 * grid_size_low)
            radius      = grid_size_low * radius_scaler  (~1.86 * grid_size_low for ks=3)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3)
        bias: Whether to use bias in convolution
        receptive_field_scaler: Volume multiplier for the search sphere (default: 1.0)
        distance_type: Distance metric, "ball" or "cube" (default: "ball")
        straight_recover: If True, reuse cached triplets from the corresponding
            downsampling step (faster, but may be inaccurate if kernel_size,
            receptive_field_scaler, or distance_type differ). Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        receptive_field_scaler: float = 1.0,
        distance_type: str = "ball",
        straight_recover: bool = False,
    ):
        super().__init__()
        self.conv = PointConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )
        self.receptive_field_scaler = receptive_field_scaler
        self.distance_type = distance_type
        self.kernel_size = kernel_size
        self.straight_recover = straight_recover
    
    def forward(
        self,
        x_low: Tensor,
        m_low: MetaData,
    ) -> Tuple[Tensor, MetaData]:

        if m_low.parent is None:
            raise ValueError("Parent information is required for upsampling. "
                           "Make sure to use conv_with_stride with stride > 1 to save parent info.")

        points_high = m_low.parent.points
        sample_inds_high = m_low.parent.sample_inds
        sample_sizes_high = m_low.parent.sample_sizes
        grid_size_high = m_low.parent.grid_size

        use_cached = self.straight_recover and (
            m_low.parent.i_upsample is not None and 
            m_low.parent.j_upsample is not None and 
            m_low.parent.k_upsample is not None
        )

        if use_cached:
            # Fast path: reuse triplets cached during downsampling
            i_high = m_low.parent.i_upsample
            j_low = m_low.parent.j_upsample
            k = m_low.parent.k_upsample
        else:
            # Direct search: high-res queries among low-res sources
            grid_size_low = m_low.grid_size
            radius_scaler = radius_scaler_for_kernel_size(
                self.kernel_size,
                self.receptive_field_scaler,
                self.distance_type
            )
            neighbor_radius = grid_size_low * radius_scaler

            import math
            radius_min = grid_size_low * math.sqrt(3) / 2
            if neighbor_radius < radius_min:
                warnings.warn(
                    f"Upsample search radius ({neighbor_radius:.4f}) is smaller than "
                    f"the minimum required ({radius_min:.4f} = grid_size_low * sqrt(3)/2). "
                    f"Some high-res points may receive zero output. "
                    f"Increase receptive_field_scaler (currently {self.receptive_field_scaler})."
                )

            i_high, j_low, k, _ = build_triplets(
                points=m_low.points,
                sample_inds=m_low.sample_inds,
                sample_sizes=m_low.sample_sizes,
                neighbor_radius=neighbor_radius,
                kernel_indexer=partial(voxelize_3d, kernel_size=self.kernel_size),
                query_points=points_high,
                query_sample_inds=sample_inds_high,
                query_sample_sizes=sample_sizes_high,
                sort_by="k",
                return_num_neighbors=False,
                radius_scaler=radius_scaler,
            )

        x_high = self.conv(x_low, i_high, j_low, k, points_high.shape[0])

        m_high = MetaData(
            points=points_high,
            sample_inds=sample_inds_high,
            sample_sizes=sample_sizes_high,
            grid_size=grid_size_high,
            i=i_high,
            j=j_low,
            k=k,
            parent=m_low.parent.parent if m_low.parent is not None else None,
        )
        
        return x_high, m_high
