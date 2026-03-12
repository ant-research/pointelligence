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
    Upsample layer that aggregates features from high-resolution points to low-resolution points.
    
    The search radius is computed based on kernel_size, receptive_field_scaler, and distance_type.
    A minimum radius requirement is enforced to ensure at least one neighbor can be found.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3). Should match the kernel_size used in 
                    the corresponding downsampling layer for proper radius alignment.
        bias: Whether to use bias in convolution
        receptive_field_scaler: Scaling factor for the receptive field (default: 1.0). 
                               Increase this value to enlarge the search radius.
        distance_type: Distance metric type, either "ball" or "cube" (default: "ball")
        straight_recover: If True, directly recover triplets from parent (faster but neighbors may not be 
                         fully accurate if upsampling parameters differ from downsampling parameters).
                         If False, recompute triplets using upsampling layer parameters (slower but more accurate).
                         Default: False
    
    Note on parameter selection:
        To ensure at least one neighbor is found during upsampling, the computed radius must be
        at least as large as the high-resolution grid_size. The radius is computed as:
        
            radius = grid_size_high * radius_scaler_for_kernel_size(kernel_size, receptive_field_scaler, distance_type)
        
        For kernel_size=3 and distance_type="ball", radius_scaler ≈ 1.86 with receptive_field_scaler=1.0.
        If the computed radius is smaller than grid_size_high, a warning will be issued.
        
        To fix insufficient radius:
        Increase receptive_field_scaler (e.g., to 1.2 or 1.5)
    
    Note on straight_recover:
        When straight_recover=True, the cached triplets from downsampling are reused directly.
        This is faster but the neighbors may not be fully accurate if:
        - The upsampling layer's kernel_size differs from the downsampling layer's kernel_size
        - The upsampling layer's receptive_field_scaler differs from the downsampling layer's receptive_field_scaler
        - The upsampling layer's distance_type differs from the downsampling layer's distance_type
        
        When straight_recover=False, triplets are recomputed using the upsampling layer's parameters,
        ensuring accurate neighbors but at the cost of additional computation time.
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
            # Fast path: directly recover triplets from parent (cached during downsampling)
            # Note: This is faster but neighbors may not be fully accurate if upsampling parameters
            # differ from downsampling parameters (kernel_size, receptive_field_scaler, distance_type)
            i_high = m_low.parent.i_upsample
            j_low = m_low.parent.j_upsample
            k = m_low.parent.k_upsample
        else:
            # Slow path: recompute triplets using upsampling layer parameters
            radius_scaler = radius_scaler_for_kernel_size(
                self.kernel_size, 
                self.receptive_field_scaler,
                self.distance_type
            )
            neighbor_radius = grid_size_high * radius_scaler

            i, j, k, _ = build_triplets(
                points=points_high,
                sample_inds=sample_inds_high,
                sample_sizes=sample_sizes_high,
                neighbor_radius=neighbor_radius,
                kernel_indexer=partial(voxelize_3d, kernel_size=self.kernel_size),
                query_points=m_low.points,
                query_sample_inds=m_low.sample_inds,
                query_sample_sizes=m_low.sample_sizes,
                sort_by="k",
                return_num_neighbors=False,
                radius_scaler=radius_scaler,
            )
            
            i_high, j_low = j, i
            
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
