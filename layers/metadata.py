"""MetaData dataclass for point cloud spatial bookkeeping.

Carries points, sample indices/sizes, grid_size, and convolution triplets
(i, j, k) through the network. Acts as the "metadata" that guides how
features flow through convolution, pooling, and normalization layers.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Callable
from functools import partial

import torch
from torch import Tensor
from torch.nn.modules.utils import _triple

from .contract import TripletContract


@dataclass(kw_only=True)
class MetaData:
    points: Tensor
    sample_inds: Tensor
    sample_sizes: Tensor
    grid_size: float
    i: Tensor = None
    j: Tensor = None
    k: Tensor = None
    seg_offs: Tensor = None
    num_neighbors: Tensor = None
    downsample_indices: Tensor = None

    # Structural contract for the (i, j, k) triplets — produced by the builder
    # (which knows these construction-time facts), consumed by _conv_forward
    # WITHOUT re-derivation, so the forward traces under torch.compile. None
    # until triplets are built; nulled in dirty_triplets alongside i/j/k.
    contract: Optional[TripletContract] = None

    i_upsample: Tensor = None
    j_upsample: Tensor = None
    k_upsample: Tensor = None
    seg_offs_upsample: Tensor = None

    parent: Optional['MetaData'] = None

    kernel_size: Optional[Tuple[int, int, int]] = None
    receptive_field_scaler: float = 1.0
    sort_by: str = "k"
    return_num_neighbors: bool = False
    query_points: Optional[Tensor] = None
    query_sample_inds: Optional[Tensor] = None
    query_sample_sizes: Optional[Tensor] = None
    distance_type: str = "ball"
    auto_build_triplets: bool = True

    def __post_init__(self):
        if self.auto_build_triplets and self.kernel_size is not None and self.empty_triplets():
            self._build_triplets()

    # private method to build triplets
    def _build_triplets(self):
        self.build_triplets()

    # users can call this method to build their owntriplets
    def build_triplets(
        self,
        kernel_size: Optional[Tuple[int, int, int]] = None,
        neighbor_radius: Optional[float] = None,
        kernel_indexer: Optional[Callable] = None,
        **kwargs
    ):
        from .triplets import (
            build_triplets,
            build_triplets_segmented,
            radius_scaler_for_kernel_size,
            should_use_direct_segmented_triplets,
            voxelize_3d,
        )

        kernel_size = kernel_size or self.kernel_size
        if kernel_size is None:
            raise ValueError("kernel_size must be provided either in __init__ or build_triplets()")

        radius_scaler = radius_scaler_for_kernel_size(
            kernel_size,
            kwargs.get('receptive_field_scaler', self.receptive_field_scaler),
            kwargs.get('distance_type', self.distance_type)
        )
        if neighbor_radius is None:
            neighbor_radius = self.grid_size * radius_scaler

        use_segmented = (
            kernel_indexer is None
            and kwargs.get('sort_by', self.sort_by) == "k"
            and kwargs.get('distance_type', self.distance_type) == "ball"
            and should_use_direct_segmented_triplets(kernel_size)
        )
        if kernel_indexer is None:
            kernel_indexer = partial(voxelize_3d, kernel_size=kernel_size)

        build_params = {
            'points': self.points,
            'sample_inds': self.sample_inds,
            'sample_sizes': self.sample_sizes,
            'neighbor_radius': neighbor_radius,
            'kernel_indexer': kernel_indexer,
            'query_points': kwargs.get('query_points', self.query_points),
            'query_sample_inds': kwargs.get('query_sample_inds', self.query_sample_inds),
            'query_sample_sizes': kwargs.get('query_sample_sizes', self.query_sample_sizes),
            'sort_by': kwargs.get('sort_by', self.sort_by),
            'return_num_neighbors': kwargs.get('return_num_neighbors', self.return_num_neighbors),
            'radius_scaler': radius_scaler,
        }

        if use_segmented:
            segmented_params = dict(build_params)
            segmented_params.pop('kernel_indexer')
            segmented_params.pop('sort_by')
            i, j, k, seg_offs, num_neighbors = build_triplets_segmented(
                kernel_size=kernel_size, **segmented_params)
        else:
            i, j, k, num_neighbors = build_triplets(**build_params)
            if kwargs.get('sort_by', self.sort_by) == "k":
                from sparse_engines._seg_offs import kernel_offset_segments
                seg_offs = kernel_offset_segments(
                    k, math.prod(_triple(kernel_size)))
            else:
                seg_offs = None

        self.i = i
        self.j = j
        self.k = k
        self.seg_offs = seg_offs
        self.num_neighbors = num_neighbors
        # Radius-search submanifold rulebook: k-sorted (sort_by="k"), variable
        # per-tap neighbor counts (T != n, never uniform / exact-cover). No
        # host reduction — the contract is a structural constant here.
        self.contract = TripletContract(
            k_sorted=(kwargs.get("sort_by", self.sort_by) == "k"))

    def dirty_triplets(self):
        self.i = None
        self.j = None
        self.k = None
        self.seg_offs = None
        self.contract = None

    def empty_triplets(self):
        if self.i is None or self.j is None or self.k is None:
            assert self.i is None
            assert self.j is None
            assert self.k is None
            return True

        return False

    def num_points(self):
        return self.points.shape[0]
