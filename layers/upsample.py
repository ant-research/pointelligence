import torch
import warnings
from torch import Tensor
from typing import Tuple
from functools import partial

from layers.metadata import MetaData
from layers.conv import PointConv3d
from layers.triplets import build_triplets, voxelize_3d, radius_scaler_for_kernel_size
from layers.contract import TripletContract


def recompute_cached_upsample_k(m_low: "MetaData", kernel_size: int) -> Tensor:
    """Kernel-bucket indices ``k`` for an upsample that REUSES the cached
    ``(i_upsample, j_upsample)`` downsample edges at an arbitrary ``kernel_size``.

    The neighbour graph ``(i, j)`` is kernel-size-INDEPENDENT — it is the
    expensive radius search, cached once by the stride conv. Only the per-edge
    bucket ``k`` depends on ``kernel_size``, and it is a cheap quantization of
    the source-minus-query offset. So a larger-kernel upsample (e.g. 5³) can
    take the cached fast path even though the stride conv cached ``k`` at its own
    (smaller, 3³) kernel size — we just re-voxelize, no search.

    Quantization is at the FINE grid (``m_low.parent.grid_size``): the cached
    upsample offsets span ``±1.86·grid_fine`` (the stride conv's search radius is
    built on the fine grid — see ``handle_stride_and_build_triplets``), so
    fine-grid quantization spreads them across all ``kernel_size`` buckets per
    axis. Quantizing at the COARSE grid would collapse a 5³ kernel to its centre
    3³ (offsets reach only ``±0.93`` coarse-grid units). See ``docs/ADVANCED.md``
    "Separable neighbour graph and kernel bucketing".
    """
    i_high = m_low.parent.i_upsample          # fine query indices
    j_low = m_low.parent.j_upsample           # coarse source indices
    with torch.no_grad():
        # offset = source(coarse) − query(fine), matching build_triplets'
        # ``points[neighbor] − query_points[center]`` convention.
        offset = m_low.points[j_low] - m_low.parent.points[i_high]
        return voxelize_3d(kernel_size, offset, grid_size=m_low.parent.grid_size)


class Upsample(torch.nn.Module):
    """
    Upsample layer: each high-res output point gathers features from nearby
    low-res input points via learned convolution weights.

        output[i_high] += weight[k] @ input[j_low]

    The neighborhood search is direct: high-res points (query) search among
    low-res points (source). The search radius uses grid_size_low as its base
    to guarantee every high-res point finds at least one low-res neighbor.

    Minimum radius requirement:
        Low-res points come from center_nearest downsampling: one point per
        voxel, chosen as the point closest to the voxel center. Crucially,
        center_nearest does NOT place the representative AT the center — it
        picks from existing points, which may all be clustered at one corner.

        The worst case is an isolated voxel (no adjacent occupied voxels)
        where the center_nearest representative is at one corner and a
        high-res query point is at the opposite corner. The maximum
        within-voxel distance is the full voxel diagonal:

            radius_min = grid_size_low * sqrt(3)  (~1.73 * grid_size_low)
            radius      = grid_size_low * radius_scaler  (~1.86 * grid_size_low for ks=3)

        This leaves only ~7% margin. For kernel_size=3 with default
        receptive_field_scaler=1.0, the guarantee holds but is tight.
        It breaks when receptive_field_scaler < ~0.81.

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
        recompute_k: bool = False,
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
        # recompute_k: on the cached fast path, REUSE the cached (i, j) neighbour
        # edges but RE-VOXELIZE the kernel bucket k for THIS conv's kernel_size.
        # The neighbour graph (i, j) is the expensive radius search and is
        # kernel-size-INDEPENDENT; the bucket k is a cheap quantization of the
        # source-minus-query offset that depends only on kernel_size + grid_size.
        # So a conv whose kernel_size differs from the stride conv's (e.g. a 5^3
        # upsample reusing a 3^3 downsample's cached edges) can take the fast
        # path WITHOUT a fresh search — only the k buckets are recomputed, which
        # is bit-identical to what the direct-search path would produce on those
        # edges. See docs/ADVANCED.md "Separable neighbour graph and kernel
        # bucketing".
        self.recompute_k = recompute_k

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
            (self.recompute_k or m_low.parent.k_upsample is not None)
        )

        if use_cached:
            # Fast path: reuse the (i, j) neighbour edges cached during downsampling.
            i_high = m_low.parent.i_upsample
            j_low = m_low.parent.j_upsample
            if self.recompute_k:
                # recompute_k re-buckets k at a kernel_size != the stride conv's
                # (see recompute_cached_upsample_k). Re-bucketing leaves the cached
                # edges out of k-order, so RE-SORT the triplets by the NEW k
                # ascending: the grouped conv paths — in particular the fused-CUTLASS
                # VVOR backward, which builds per-kernel-offset segments via
                # searchsorted — REQUIRE k sorted ascending (the sort_by="k"
                # contract). Without this the fused path (in_channels >= 512) raises
                # "o_idx must be sorted ascending"; the Triton path silently tolerates
                # unsorted k. Sorting keeps the triplets valid for ALL conv paths.
                k = recompute_cached_upsample_k(m_low, self.kernel_size)
                k, order = torch.sort(k)          # match build_triplets' sort_by="k"
                i_high, j_low = i_high[order], j_low[order]
            else:
                # Cached k_upsample is the downsample's m.k — already k-sorted
                # (handle_stride_and_build_triplets uses sort_by="k").
                k = m_low.parent.k_upsample
        else:
            # Direct search below also returns k-sorted triplets (sort_by="k").
            # Direct search: high-res queries among low-res sources
            grid_size_low = m_low.grid_size
            radius_scaler = radius_scaler_for_kernel_size(
                self.kernel_size,
                self.receptive_field_scaler,
                self.distance_type
            )
            neighbor_radius = grid_size_low * radius_scaler

            import math
            # The true worst case is the full voxel diagonal (center_nearest
            # can place the representative at any corner, not near the center).
            radius_min = grid_size_low * math.sqrt(3)
            if neighbor_radius < radius_min:
                warnings.warn(
                    f"Upsample search radius ({neighbor_radius:.4f}) is smaller than "
                    f"the worst-case voxel diagonal ({radius_min:.4f} = grid_size_low * sqrt(3)). "
                    f"Some high-res points in isolated voxels may find zero neighbors. "
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

        # : build_triplets uses sort_by="k" (line above) and the upsample
        # rulebook is submanifold-shaped (T != n_high) ⇒ the submanifold
        # contract holds; pass it so the conv traces under torch.compile.
        x_high = self.conv(x_low, i_high, j_low, k, points_high.shape[0],
                            contract=TripletContract.submanifold())

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
