"""TwoPhaseOp adapter for the point-conv operator. build_indices runs the
triplet/downsample build (the @compiler.disable geometry work); apply runs the
break-free PointConv3d compute over the prebuilt (i,j,k,contract).

ConvOp mirrors conv_with_stride's internal build call byte-for-byte: it passes
the operator's expanded ``kernel_size_3`` (NOT the scalar ``kernel_size``) and
``receptive_field_scaler`` positionally, with ``distance_type`` as keyword and
``sort_by``/``return_num_neighbors`` left at their defaults -- so the hoisted
build produces bit-identical (i, j, k, contract) and the scheduler path is
torch.equal to the serial conv_with_stride.
"""
from dataclasses import dataclass

import torch

from layers.triplets import handle_stride_and_build_triplets
from layers.metadata import MetaData


@dataclass
class ConvBundle:
    meta: MetaData          # carries i, j, k, contract, num_points()
    next_geom: MetaData     # advanced (downsampled) geometry for the next op


class ConvOp:
    separable = True

    def __init__(self, conv_op, stride: float, receptive_field_scaler: float = 1.0,
                 distance_type: str = "ball"):
        self.conv_op = conv_op
        self.stride = float(stride)
        self.rf = float(receptive_field_scaler)
        self.distance_type = distance_type

    def build_indices(self, geom: MetaData) -> ConvBundle:
        m = handle_stride_and_build_triplets(
            geom, self.stride, self.conv_op.kernel_size_3, self.rf,
            distance_type=self.distance_type)
        return ConvBundle(meta=m, next_geom=m)

    def apply(self, x: torch.Tensor, b: ConvBundle) -> torch.Tensor:
        m = b.meta
        return self.conv_op(
            x, m.i, m.j, m.k, m.num_points(), contract=m.contract,
            seg_offs=m.seg_offs)
