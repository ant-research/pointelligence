from dataclasses import dataclass
from typing import Optional, Tuple, Callable
from functools import partial

import torch
from torch import Tensor


@dataclass(kw_only=True)
class MetaData:
    points: Tensor
    sample_inds: Tensor
    sample_sizes: Tensor
    grid_size: float
    i: Tensor = None
    j: Tensor = None
    k: Tensor = None
    num_neighbors: Tensor = None
    downsample_indices: Tensor = None

    i_upsample: Tensor = None
    j_upsample: Tensor = None
    k_upsample: Tensor = None

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
        from .triplets import build_triplets, radius_scaler_for_kernel_size, voxelize_3d

        kernel_size = kernel_size or self.kernel_size
        if kernel_size is None:
            raise ValueError("kernel_size must be provided either in __init__ or build_triplets()")

        if neighbor_radius is None:
            radius_scaler = radius_scaler_for_kernel_size(
                kernel_size,
                kwargs.get('receptive_field_scaler', self.receptive_field_scaler),
                kwargs.get('distance_type', self.distance_type)
            )
            neighbor_radius = self.grid_size * radius_scaler

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
        
        i, j, k, num_neighbors = build_triplets(**build_params)
        
        self.i = i
        self.j = j
        self.k = k
        self.num_neighbors = num_neighbors

    def dirty_triplets(self):
        self.i = None
        self.j = None
        self.k = None

    def empty_triplets(self):
        if self.i is None or self.j is None or self.k is None:
            assert self.i is None
            assert self.j is None
            assert self.k is None
            return True

        return False

    def num_points(self):
        return self.points.shape[0]
