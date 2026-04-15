"""Global and local pooling for ragged point cloud features.

Provides GlobalPool (per-sample aggregation) and max_pool3d (local
neighborhood max-pooling with strided downsampling).
"""

from typing import Tuple

import torch

from torch import Tensor

from sparse_engines.ops import large_segment_reduce
from sparse_engines.ops import indexed_segment_reduce

from .metadata import MetaData
from .triplets import handle_stride_and_build_triplets


class GlobalPool(torch.nn.Module):
    def __init__(self, method="mean"):
        super(GlobalPool, self).__init__()
        assert method in ["max", "min", "mean"]
        self.method = method

    def forward(self, x: Tensor, sample_sizes: Tensor) -> Tensor:
        return large_segment_reduce(x, reduce=self.method, lengths=sample_sizes)

    def extra_repr(self) -> str:
        return "method: {method}".format(**self.__dict__)


def max_pool3d(
    x: Tensor, m: MetaData, kernel_size: int, stride: float
) -> Tuple[Tensor, MetaData]:
    m = handle_stride_and_build_triplets(
        m, stride, kernel_size, sort_by="i", return_num_neighbors=True
    )

    # for example
    # num_neighbors: 3     1 4       2
    # i(out):        0 0 0 1 2 2 2 2 3 3
    # j(in):         7 9 1 6 1 7 0 4 5 3

    out = indexed_segment_reduce(x, reduce="max", indices=m.j, lengths=m.num_neighbors)

    return out, m
