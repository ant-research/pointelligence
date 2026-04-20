"""Point convolution layer (PointConv3d).

Implements convolution on native 3D points using the (i, j, k) triplet
representation and the MVMR sparse engine. This is the core operator
introduced in PointCNN++ (arXiv:2511.23227).
"""

import math
from typing import Optional, Callable, Tuple

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.common_types import _size_3_t
from torch.nn.parameter import Parameter


from torch.nn import Module
from torch.nn.modules.utils import _triple

from sparse_engines.ops import sparse_matrix_vector_multiplication_reduction

from .metadata import MetaData
from .triplets import handle_stride_and_build_triplets


class GeneralConv(Module):
    __constants__ = [
        "groups",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    in_channels: int
    out_channels: int
    kernel_size: int
    groups: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        bias: bool,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.weight = Parameter(
            torch.empty(
                (in_channels, out_channels // groups, kernel_size),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * kernel_size
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)

    def _conv_forward(  # type: ignore[empty-body]
        self,
        input: Tensor,
        i: Tensor,
        j: Tensor,
        k: Tensor,
        n: int,
        weight: Tensor,
        bias: Optional[Tensor],
    ) -> Tensor:
        # Reshape and permute weight to fit the memory layout requirement of MVMR.
        # This overhead could be saved if the weights are initialized to the desired layout.
        # TODO: save this overhead.
        w = weight.reshape(
            self.groups,
            self.in_channels // self.groups,
            self.out_channels // self.groups,
            self.kernel_size,
        ).permute(3, 0, 1, 2)

        # Reshape input from 2D (N, C_in) to 3D (N, G, C_in/G).
        # The group dimension must match the weight's so that VVOR (backward)
        # produces grad_weight with shape (K, G, Ci/G, Co/G) matching the
        # weight after reshape+permute. Using (N, 1, C_in) breaks grouped
        # convolutions because VVOR would output (K, 1, C_in, Co/G) instead.
        if input.dim() == 2:
            input_3d = input.view(input.shape[0], self.groups, -1).contiguous()
        else:
            input_3d = input.contiguous()

        output = sparse_matrix_vector_multiplication_reduction(w, k, input_3d, j, i, n)

        # Reshape output from 3D (n_o, 1, out_channels) back to 2D (n_o, out_channels)
        output = output.reshape(-1, self.out_channels)

        if bias is not None:
            output = output + bias

        return output


class PointConv3d(GeneralConv):
    __doc__ = r"""Convolution on Native Points as introduced in PointCNN++ (https://www.arxiv.org/abs/2511.23227)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.kernel_size_3 = _triple(kernel_size)
        super().__init__(
            in_channels,
            out_channels,
            math.prod(self.kernel_size_3),
            groups,
            bias,
            **factory_kwargs,
        )

    def forward(self, input: Tensor, i: Tensor, j: Tensor, k: Tensor, n: int) -> Tensor:
        return self._conv_forward(input, i, j, k, n, self.weight, self.bias)


def conv_with_stride(
    conv_op: Callable, x: Tensor, m: MetaData, stride: float, receptive_field_scaler=1.0
) -> Tuple[Tensor, MetaData]:
    m = handle_stride_and_build_triplets(
        m, stride, conv_op.kernel_size_3, receptive_field_scaler
    )
    x = conv_op(x, m.i, m.j, m.k, m.num_points())
    return x, m
