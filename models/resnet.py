"""ResNet architectures (18/34/50/101/152) for native point convolution.

Standard ResNet building blocks (BasicBlock, Bottleneck) adapted to use
PointConv3d instead of Conv2d, with ragged normalization and strided
downsampling via voxel grid filtering.
"""

from typing import Any, Callable, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from functools import partial

from sparse_engines.ops import large_segment_reduce
from layers import (
    max_pool3d,
    voxelize_3d,
    build_triplets,
    conv_with_stride,
    radius_scaler_for_kernel_size,
)
from layers import (
    MetaData,
    PointConv3d,
    GlobalPool,
    MultiSequential,
    RaggedNorm,
    RaggedLayerNorm,
)

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3x3(in_planes: int, out_planes: int, groups: int = 1) -> PointConv3d:
    """Point convolution with 3x3x3 kernels"""
    return PointConv3d(in_planes, out_planes, kernel_size=3, groups=groups, bias=False)


def conv1x1x1(in_planes: int, out_planes: int) -> nn.Linear:
    """1x1x1 convolution"""
    return nn.Linear(in_planes, out_planes, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(RaggedLayerNorm, reduce_fn=large_segment_reduce)
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: Tensor,
        m: MetaData,
    ) -> Tuple[Tensor, MetaData]:
        identity = x

        out, m = conv_with_stride(self.conv1, x, m, self.stride)
        out = self.bn1(out, m.sample_sizes)
        out = self.relu(out)

        if self.stride != 1:
            neighbor_radius = m.grid_size * radius_scaler_for_kernel_size(kernel_size=3)
            m.i, m.j, m.k, _ = build_triplets(
                points=m.points,
                sample_inds=m.sample_inds,
                sample_sizes=m.sample_sizes,
                neighbor_radius=neighbor_radius,
                kernel_indexer=partial(voxelize_3d, kernel_size=3),
                radius_scaler=radius_scaler_for_kernel_size(kernel_size=3),
                return_num_neighbors=False,
            )

        out = self.conv2(out, m.i, m.j, m.k, m.num_points())
        out = self.bn2(out, m.sample_sizes)

        if self.downsample is not None:
            x_downsample = x if self.stride == 1 else x[m.downsample_indices]
            # identity = self.downsample(x_downsample)
            for module in self.downsample:
                if isinstance(module, RaggedNorm):
                    x_downsample = module(x_downsample, lengths=m.sample_sizes)
                else:
                    x_downsample = module(x_downsample)
            identity = x_downsample

        out += identity
        out = self.relu(out)

        return out, m


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(RaggedLayerNorm, reduce_fn=large_segment_reduce)
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3(width, width, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: Tensor,
        m: MetaData,
    ) -> Tuple[Tensor, MetaData]:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, m.sample_sizes)
        out = self.relu(out)

        out, m = conv_with_stride(self.conv2, out, m, self.stride)

        if self.stride != 1:
            m.dirty_triplets()

        out = self.bn2(out, m.sample_sizes)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, m.sample_sizes)

        if self.downsample is not None:
            x_downsample = x if self.stride == 1 else x[m.downsample_indices]
            # identity = self.downsample(x_downsample)
            for module in self.downsample:
                if isinstance(module, RaggedNorm):
                    x_downsample = module(x_downsample, lengths=m.sample_sizes)
                else:
                    x_downsample = module(x_downsample)
            identity = x_downsample

        out += identity
        out = self.relu(out)

        return out, m


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(RaggedLayerNorm, reduce_fn=large_segment_reduce)
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, bias=False)
        self.conv1 = PointConv3d(in_channels, self.inplanes, kernel_size=7, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # We will use a max_pool3d function for this
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = GlobalPool("mean")

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, PointConv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                )
            )

        return MultiSequential(*layers)

    def _forward_impl(self, x: Tensor, m: MetaData) -> Tensor:
        # See note [TorchScript super()]
        
        x, m = conv_with_stride(self.conv1, x, m, 2)
        m.dirty_triplets()

        x = self.bn1(x, m.sample_sizes)
        x = self.relu(x)

        # x = self.maxpool(x)
        x, m = max_pool3d(x, m, kernel_size=3, stride=2)
        m.dirty_triplets()

        x, m = self.layer1(x, m)
        x, m = self.layer2(x, m)
        x, m = self.layer3(x, m)
        x, m = self.layer4(x, m)

        x = self.avgpool(x, m.sample_sizes)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(
        self,
        x: Tensor,
        points: Tensor,
        sample_sizes: Tensor,
        grid_size: float,
    ) -> Tensor:
        sample_inds = torch.repeat_interleave(
            torch.arange(0, sample_sizes.numel(), device=sample_sizes.device),
            sample_sizes,
        )
        m = MetaData(
            points=points,
            sample_inds=sample_inds,
            sample_sizes=sample_sizes,
            grid_size=grid_size,
        )

        return self._forward_impl(x, m)


def _resnet(
    block: type[Union[BasicBlock, Bottleneck]], layers: list[int], **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    return model


def resnet18(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnext101_64x4d(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)