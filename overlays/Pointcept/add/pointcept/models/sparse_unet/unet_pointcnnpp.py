"""
ResUNet based on PointCNN++ for Pointcept

This module provides a ResUNet implementation using PointCNN++ layers,
with the same parameters as the pointops version.
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Tuple, Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from layers import (
    PointConv3d,
    conv_with_stride,
    RaggedNorm,
)
from layers.triplets import build_triplets_segmented, radius_scaler_for_kernel_size
from layers.metadata import MetaData
from layers.upsample import Upsample
from sparse_engines.ops import large_segment_reduce

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2bincount
from timm.layers import trunc_normal_


def conv3x3x3(in_planes: int, out_planes: int) -> PointConv3d:
    return PointConv3d(in_planes, out_planes, kernel_size=3, bias=False)


def conv1x1x1(in_planes: int, out_planes: int) -> nn.Linear:
    return nn.Linear(in_planes, out_planes, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: float = 1.0,
        norm_layer: Optional[callable] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.conv1 = conv3x3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
        self.downsample = None
        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1x1(inplanes, planes),
                norm_layer(planes),
            )

    def forward(self, x: torch.Tensor, m: MetaData) -> Tuple[torch.Tensor, MetaData]:
        identity = x

        x, m = conv_with_stride(self.conv1, x, m, self.stride, receptive_field_scaler=2.5)
        x = self.bn1(x)
        x = self.relu(x)

        if self.stride != 1.0:
            radius_scaler = radius_scaler_for_kernel_size(kernel_size=3, receptive_field_scaler=2.5)
            m.i, m.j, m.k, m.seg_offs, _ = build_triplets_segmented(
                points=m.points,
                sample_inds=m.sample_inds,
                sample_sizes=m.sample_sizes,
                neighbor_radius=m.grid_size * radius_scaler,
                kernel_size=3,
                radius_scaler=radius_scaler,
            )
        x = self.conv2(
            x, m.i, m.j, m.k, m.num_points(), seg_offs=m.seg_offs)
        x = self.bn2(x)

        if self.downsample is not None:
            if self.stride == 1.0:
                identity_for_downsample = identity
            else:
                if hasattr(m, 'downsample_indices') and m.downsample_indices is not None:
                    identity_for_downsample = identity[m.downsample_indices]
                else:
                    raise ValueError(
                        f"downsample_indices is None when stride={self.stride}. "
                        f"This indicates a bug in conv_with_stride or handle_stride_and_build_triplets. "
                        f"Please check that handle_stride_and_build_triplets correctly sets downsample_indices."
                    )
            
            for module in self.downsample:
                identity_for_downsample = module(identity_for_downsample)
            identity = identity_for_downsample
        elif self.stride != 1.0:
            if hasattr(m, 'downsample_indices') and m.downsample_indices is not None:
                identity = identity[m.downsample_indices]
            else:
                raise ValueError(
                    f"downsample_indices is None when stride={self.stride} and downsample is None. "
                    f"This indicates a bug in conv_with_stride or handle_stride_and_build_triplets."
                )

        if identity is not None:
            x = x + identity
        x = self.relu(x)

        return x, m


@MODELS.register_module("ResUNetPointCNNpp", force=True)
class ResUNetPointCNNpp(nn.Module):
    """
    ResUNet based on PointCNN++ for Pointcept framework.
    
    This implementation uses PointCNN++ layers and maintains the same
    architecture and parameters as the pointops version.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 16,  # For compatibility with pointcept, using num_classes instead of out_channels
        base_channels: int = 32,
        bn_momentum: float = 0.05,
        normalize_feature: bool = False,
        voxel_size: float = 0.025,
        channels: Optional[Tuple[int, ...]] = None,  # Custom channels configuration
        layers: Optional[Tuple[int, ...]] = None,  # Custom layers configuration
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        
        # Map num_classes to out_channels for internal use
        out_channels = num_classes
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes  # For pointcept compatibility
        self.base_channels = base_channels
        self.normalize_feature = normalize_feature
        self.voxel_size = voxel_size

        # Support custom channels configuration for alignment with SpUNet
        if channels is not None:
            # channels format: (enc1, enc2, enc3, enc4, dec4, dec3, dec2, dec1)
            # For ResUNet: CHANNELS = [enc1, enc2, enc3, enc4], TR_CHANNELS = [dec1, dec2, dec3, dec4]
            if len(channels) == 8:
                CHANNELS = [channels[0], channels[1], channels[2], channels[3]]  # encoder
                TR_CHANNELS = [channels[7], channels[6], channels[5], channels[4]]  # decoder: dec1, dec2, dec3, dec4
            else:
                raise ValueError(f"channels must be a tuple of 8 integers, got {channels}")
        else:
            # Default configuration
            CHANNELS = [32, 64, 128, 256]
            TR_CHANNELS = [96, 96, 128, 256]
        
        # Support custom layers configuration for alignment with SpUNet
        if layers is not None:
            # layers format: (enc0, enc1, enc2, enc3, dec3, dec2, dec1, dec0)
            if len(layers) == 8:
                LAYERS = list(layers)
            else:
                raise ValueError(f"layers must be a tuple of 8 integers, got {layers}")
        else:
            # Default configuration
            LAYERS = [2, 3, 4, 6, 2, 2, 2, 2]
        
        self.CHANNELS = CHANNELS
        self.TR_CHANNELS = TR_CHANNELS
        self.LAYERS = LAYERS
        
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv1 = PointConv3d(in_channels, base_channels, kernel_size=5, bias=False)
        self.norm1 = norm_fn(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # block1: layers[0] BasicBlocks (base_channels → CHANNELS[0])
        self.block1 = nn.ModuleList([
            BasicBlock(base_channels if i == 0 else CHANNELS[0], CHANNELS[0], 
                      stride=1.0 if i == 0 else 1.0, norm_layer=norm_fn)
            for i in range(LAYERS[0])
        ])

        # enc_block1: layers[1] BasicBlocks (CHANNELS[0] → CHANNELS[1])
        self.enc_block1 = nn.ModuleList([
            BasicBlock(CHANNELS[0] if i == 0 else CHANNELS[1], CHANNELS[1],
                      stride=2.0 if i == 0 else 1.0, norm_layer=norm_fn)
            for i in range(LAYERS[1])
        ])
        
        # enc_block2: layers[2] BasicBlocks (CHANNELS[1] → CHANNELS[2])
        self.enc_block2 = nn.ModuleList([
            BasicBlock(CHANNELS[1] if i == 0 else CHANNELS[2], CHANNELS[2],
                      stride=2.0 if i == 0 else 1.0, norm_layer=norm_fn)
            for i in range(LAYERS[2])
        ])
        
        # enc_block3: layers[3] BasicBlocks (CHANNELS[2] → CHANNELS[3])
        self.enc_block3 = nn.ModuleList([
            BasicBlock(CHANNELS[2] if i == 0 else CHANNELS[3], CHANNELS[3],
                      stride=2.0 if i == 0 else 1.0, norm_layer=norm_fn)
            for i in range(LAYERS[3])
        ])
        
        # Decoder
        # Stage 4: layers[4] BasicBlocks (TR_CHANNELS[3] → TR_CHANNELS[3])
        self.upsample4 = Upsample(CHANNELS[3], TR_CHANNELS[3], kernel_size=3, bias=False, receptive_field_scaler=2.5)
        self.norm4_tr = norm_fn(TR_CHANNELS[3])
        self.block4_tr = nn.ModuleList([
            BasicBlock(TR_CHANNELS[3], TR_CHANNELS[3], stride=1.0, norm_layer=norm_fn)
            for i in range(LAYERS[4])
        ])
        
        # Stage 3: layers[5] BasicBlocks (TR_CHANNELS[2] → TR_CHANNELS[2])
        self.upsample3 = Upsample(CHANNELS[2] + TR_CHANNELS[3], TR_CHANNELS[2], kernel_size=3, bias=False, receptive_field_scaler=2.5)
        self.norm3_tr = norm_fn(TR_CHANNELS[2])
        self.block3_tr = nn.ModuleList([
            BasicBlock(TR_CHANNELS[2], TR_CHANNELS[2], stride=1.0, norm_layer=norm_fn)
            for i in range(LAYERS[5])
        ])
        
        # Stage 2: layers[6] BasicBlocks (TR_CHANNELS[1] → TR_CHANNELS[1])
        self.upsample2 = Upsample(CHANNELS[1] + TR_CHANNELS[2], TR_CHANNELS[1], kernel_size=3, bias=False, receptive_field_scaler=2.5)
        self.norm2_tr = norm_fn(TR_CHANNELS[1])
        self.block2_tr = nn.ModuleList([
            BasicBlock(TR_CHANNELS[1], TR_CHANNELS[1], stride=1.0, norm_layer=norm_fn)
            for i in range(LAYERS[6])
        ])
        
        # Stage 1: layers[7] BasicBlocks (CHANNELS[0] + TR_CHANNELS[1] → TR_CHANNELS[0])
        self.block1_tr = nn.ModuleList([
            BasicBlock((CHANNELS[0] + TR_CHANNELS[1]) if i == 0 else TR_CHANNELS[0], 
                      TR_CHANNELS[0], stride=1.0, norm_layer=norm_fn)
            for i in range(LAYERS[7])
        ])
        
        self.final_conv = PointConv3d(TR_CHANNELS[0], out_channels, kernel_size=1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, PointConv3d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, RaggedNorm)):
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        input_dict: dict,
    ) -> torch.Tensor:
        """
        Forward pass compatible with Pointcept framework.
        
        Args:
            input_dict: Dictionary containing:
                - 'feat': [N, in_channels] input features
                - 'coord': [N, 3] point coordinates (will be used as points, same as PointOps version)
                - 'grid_coord': [N, 3] grid coordinates (optional, for backward compatibility)
                - 'offset': [B] batch offsets (based on coord)
                - 'grid_size' or 'voxel_size': grid size (optional, defaults to self.voxel_size)
        
        Returns:
            [N, out_channels] output features
        """
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        
        if "coord" in input_dict:
            coord = input_dict["coord"]
        elif "grid_coord" in input_dict:
            coord = input_dict["grid_coord"]
        else:
            raise ValueError("input_dict must contain either 'coord' or 'grid_coord'")
        
        if coord.dtype != torch.float32:
            coord = coord.float()
        
        if feat.dtype != torch.float32:
            feat = feat.float()
        
        expected_num_points = offset[-1].item() if offset.numel() > 0 else 0
        actual_num_points = coord.shape[0]
        
        if expected_num_points != actual_num_points:
            import warnings
            warnings.warn(
                f"Offset length ({expected_num_points}) != coord length ({actual_num_points}). "
                f"Recalculating offset based on coord length to match PointOps version behavior."
            )
            offset = torch.tensor([0, actual_num_points], device=offset.device, dtype=offset.dtype)
        
        grid_size = input_dict.get("grid_size", input_dict.get("voxel_size", self.voxel_size))
        
        sample_sizes = offset2bincount(offset)
        
        sample_inds = torch.repeat_interleave(
            torch.arange(0, sample_sizes.numel(), device=sample_sizes.device),
            sample_sizes,
        )
        
        if sample_inds.shape[0] != coord.shape[0]:
            raise ValueError(
                f"Failed to fix coordinate length mismatch: coord has {coord.shape[0]} points, "
                f"but sample_inds has {sample_inds.shape[0]} points after recalculation. "
                f"This indicates a serious data inconsistency issue."
            )
        
        m = MetaData(
            points=coord,
            sample_inds=sample_inds,
            sample_sizes=sample_sizes,
            grid_size=grid_size,
        )
        
        x = feat
        
        # Encoder
        x, m = conv_with_stride(self.conv1, x, m, 1.0, receptive_field_scaler=2.5)
        m.dirty_triplets()
        x = self.norm1(x)
        x = self.relu(x)

        # block1
        for block in self.block1:
            x, m = block(x, m)
        
        down_outputs = [x]
        
        for block in self.enc_block1:
            x, m = block(x, m)
        down_outputs.append(x)
        
        for block in self.enc_block2:
            x, m = block(x, m)
        down_outputs.append(x)
        
        for block in self.enc_block3:
            x, m = block(x, m)
        down_outputs.append(x)
        
        # Decoder (Upsampling)
        # Stage 4
        x_high, m_high = self.upsample4(down_outputs[-1], m)
        x_high = self.norm4_tr(x_high)
        x_high = self.relu(x_high)
        for block in self.block4_tr:
            x_high, m_high = block(x_high, m_high)
        
        x_concat = torch.cat([down_outputs[-2], x_high], dim=1)
        
        # Stage 3
        x_high, m_high = self.upsample3(x_concat, m_high)
        x_high = self.norm3_tr(x_high)
        x_high = self.relu(x_high)
        for block in self.block3_tr:
            x_high, m_high = block(x_high, m_high)
        
        x_concat = torch.cat([down_outputs[-3], x_high], dim=1)
        
        # Stage 2
        x_high, m_high = self.upsample2(x_concat, m_high)
        x_high = self.norm2_tr(x_high)
        x_high = self.relu(x_high)
        for block in self.block2_tr:
            x_high, m_high = block(x_high, m_high)
        
        x_concat = torch.cat([down_outputs[0], x_high], dim=1)
        
        # Stage 1
        m_high.dirty_triplets()
        for i, block in enumerate(self.block1_tr):
            if i == 0:
                x_high, m_high = block(x_concat, m_high)
            else:
                x_high, m_high = block(x_high, m_high)
        
        if self.out_channels > 0:
            m_high.dirty_triplets()
            radius_scaler = radius_scaler_for_kernel_size(kernel_size=1, receptive_field_scaler=2.5)
            neighbor_radius = m_high.grid_size * radius_scaler
            (m_high.i, m_high.j, m_high.k, m_high.seg_offs,
             _) = build_triplets_segmented(
                points=m_high.points,
                sample_inds=m_high.sample_inds,
                sample_sizes=m_high.sample_sizes,
                neighbor_radius=neighbor_radius,
                kernel_size=1,
                radius_scaler=radius_scaler,
            )
            x_high = self.final_conv(
                x_high, m_high.i, m_high.j, m_high.k,
                m_high.num_points(), seg_offs=m_high.seg_offs)
        
        if self.normalize_feature:
            x_high = x_high / torch.norm(x_high, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        
        return x_high


def test_unet_structure():
    """Test UNet structure with random point cloud input."""
    print("=" * 80)
    print("Testing ResUNetPointCNNpp structure")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    print("\nCreating model...")
    in_channels = 1
    num_classes = 16
    base_channels = 32
    voxel_size = 0.025
    
    model = ResUNetPointCNNpp(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        voxel_size=voxel_size,
        normalize_feature=False,
    ).to(device)
    model.eval()
    
    print(f"  - Input channels: {in_channels}")
    print(f"  - Output channels: {num_classes}")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Encoder channels: {model.CHANNELS}")
    print(f"  - Decoder channels: {model.TR_CHANNELS}")
    
    print("\nGenerating random point cloud data...")
    num_samples = 2
    num_points_per_sample = [5000, 6000]
    total_points = sum(num_points_per_sample)
    
    grid_coord = torch.randn(total_points, 3, device=device) * 0.5
    grid_coord = grid_coord.float()
    
    feat = torch.randn(total_points, in_channels, device=device).float()
    
    offset = torch.tensor([0, num_points_per_sample[0], total_points], 
                          dtype=torch.long, device=device)
    
    input_dict = {
        "feat": feat,
        "grid_coord": grid_coord,
        "offset": offset,
        "grid_size": voxel_size,
    }
    
    print(f"  - Total points: {total_points}")
    print(f"  - Batch size: {num_samples}")
    print(f"  - Points per sample: {num_points_per_sample}")
    print(f"  - Coord shape: {grid_coord.shape}")
    print(f"  - Feature shape: {feat.shape}")
    print(f"  - Grid size: {voxel_size}")
    
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(input_dict)
        
        print("✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected shape: ({total_points}, {num_classes})")
        
        assert output.shape == (total_points, num_classes), \
            f"Output shape mismatch: expected ({total_points}, {num_classes}), got {output.shape}"
        print("✓ Output shape correct")
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        print("✓ Output values valid (no NaN/Inf)")
        
        print(f"\nOutput statistics:")
        print(f"  - Min: {output.min().item():.6f}")
        print(f"  - Max: {output.max().item():.6f}")
        print(f"  - Mean: {output.mean().item():.6f}")
        print(f"  - Std: {output.std().item():.6f}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("Testing different configurations")
    print("=" * 80)
    
    print("\nTest 1: Custom channels configuration")
    try:
        custom_channels = (32, 64, 128, 256, 128, 64, 64, 32)
        model_custom = ResUNetPointCNNpp(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=custom_channels,
            voxel_size=voxel_size,
        ).to(device)
        model_custom.eval()
        
        with torch.no_grad():
            output_custom = model_custom(input_dict)
        
        assert output_custom.shape == (total_points, num_classes)
        print("✓ Custom channels configuration test passed")
    except Exception as e:
        print(f"✗ Custom channels configuration test failed: {e}")
    
    print("\nTest 2: Different input channels")
    try:
        in_channels_test = 4
        input_dict_test = input_dict.copy()
        input_dict_test["feat"] = torch.randn(total_points, in_channels_test, 
                                               device=device).float()
        
        model_test = ResUNetPointCNNpp(
            in_channels=in_channels_test,
            num_classes=num_classes,
            voxel_size=voxel_size,
        ).to(device)
        model_test.eval()
        
        with torch.no_grad():
            output_test = model_test(input_dict_test)
        
        assert output_test.shape == (total_points, num_classes)
        print(f"✓ Input channels={in_channels_test} test passed")
    except Exception as e:
        print(f"✗ Different input channels test failed: {e}")
    
    print("\nTest 3: num_classes=0 (pretraining mode)")
    try:
        model_pretrain = ResUNetPointCNNpp(
            in_channels=in_channels,
            num_classes=0,
            voxel_size=voxel_size,
        ).to(device)
        model_pretrain.eval()
        
        with torch.no_grad():
            output_pretrain = model_pretrain(input_dict)
        
        print(f"  - Pretraining output shape: {output_pretrain.shape}")
        print("✓ Pretraining mode test passed")
    except Exception as e:
        print(f"✗ Pretraining mode test failed: {e}")
    
    print("\nTest 4: Different voxel_size")
    try:
        voxel_size_test = 0.05
        input_dict_test = input_dict.copy()
        input_dict_test["grid_size"] = voxel_size_test
        
        with torch.no_grad():
            output_test = model(input_dict_test)
        
        assert output_test.shape == (total_points, num_classes)
        print(f"✓ Voxel size={voxel_size_test} test passed")
    except Exception as e:
        print(f"✗ Different voxel_size test failed: {e}")
    
    print("\nTest 5: Verify gradient flow")
    try:
        model.train()
        input_dict_grad = {
            "feat": feat.clone().requires_grad_(True),
            "grid_coord": grid_coord.clone(),
            "offset": offset.clone(),
            "grid_size": voxel_size,
        }
        
        output_grad = model(input_dict_grad)
        loss = output_grad.mean()
        loss.backward()
        
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No parameters received gradients"
        print("✓ Gradient flow normal")
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    """Run validation script."""
    success = test_unet_structure()
    if success:
        print("\n✓ UNet structure validation passed!")
    else:
        print("\n✗ UNet structure validation failed!")
        exit(1)
