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
from sparse_engines._dispatch_override import current_mode

from .metadata import MetaData
from .triplets import (
    handle_stride_and_build_triplets,
    radius_scaler_for_kernel_size,
)


# Perf-optimal auto-router measured boundary. The fused CUTLASS
# PointConv3d path is net-positive vs the Triton-grouped path ONLY in the
# measured large-C regime; at smaller C it is CATASTROPHICALLY slower.
# Measured enc4 fp16 H200 (dispatcher-reverified, fused forward+backward
# ÷ grouped forward+backward, median-of-3):
#   enc1 C=64  → 24.7×   enc2 C=128 → 7.4×   enc3 C=256 → 2.1×  (SLOWER)
#   enc4 C=512 → 0.715×  (net-positive — the ONLY validated win)
# So under the production "auto" mode, auto-engage the fused path ONLY
# when C (= weight.shape[2], the conv in/G channel) is at/above the
# measured-net-positive boundary; default to the Triton path
# (zero-regression by construction) for every unmeasured-or-net-negative
# shape. 512 is the conservative measured cut (net-positive at 512,
# net-NEGATIVE 2.1× at 256 → the boundary is in (256, 512]; pick the
# validated point). Widen ONLY with additional measured net-positive
# cells — never speculatively (the zero-regression guarantee is the
# allowlist being a strict subset of the measured-net-positive set).
# The explicit `force_fused_conv` override (benches/tests) is unaffected.
_FUSED_AUTOROUTER_MIN_C = 512


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

        # Weight layout: native (K, G, in/G, out/G). Eliminates the
        # per-fwd `weight.reshape(...).permute(3,0,1,2)` +
        # implicit `.contiguous()` inside mvmr (which allocated ~28 MB at enc4
        # per forward call, ~390 μs at sm_89). Backward-compat for legacy
        # (in, out/G, K) checkpoints is handled by _load_from_state_dict below.
        self.weight = Parameter(
            torch.empty(
                (kernel_size, groups, in_channels // groups, out_channels // groups),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Preserve the legacy kaiming_uniform init magnitude (fan_in
        # = (out/G) * K with a=sqrt(5) → uniform(-1/sqrt(k), 1/sqrt(k)))
        # for backward-compat with checkpoints trained against the legacy
        # (in, out/G, K) Parameter layout. Computed manually because the
        # new (K, G, in/G, out/G) layout gives a different default fan_in
        # under PyTorch's _calculate_fan_in_and_fan_out heuristic.
        fan_in = (self.out_channels // self.groups) * self.kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None and fan_in != 0:
            init.uniform_(self.bias, -bound, bound)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Backward-compat: detect legacy (in, out/G, K) weight
        # checkpoints and reshape+permute to the new (K, G, in/G, out/G)
        # layout before assigning. Silent — load proceeds with the converted
        # tensor, no warning (most legacy checkpoints match the old shape
        # exactly).
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            w = state_dict[weight_key]
            legacy_shape = (
                self.in_channels,
                self.out_channels // self.groups,
                self.kernel_size,
            )
            if tuple(w.shape) == legacy_shape:
                state_dict[weight_key] = (
                    w.reshape(
                        self.groups,
                        self.in_channels // self.groups,
                        self.out_channels // self.groups,
                        self.kernel_size,
                    )
                    .permute(3, 0, 1, 2)
                    .contiguous()
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

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
        # weight is already in (K, G, in/G, out/G) layout —
        # the historical per-fwd `weight.reshape(...).permute(3,0,1,2)` +
        # implicit `.contiguous()` inside mvmr (~390 μs at sm_89 enc4) is
        # dropped. mvmr's internal `.contiguous()` is now a no-op.

        # Reshape input from 2D (N, C_in) to 3D (N, G, C_in/G).
        # The group dimension must match the weight's so that VVOR (backward)
        # produces grad_weight with shape (K, G, Ci/G, Co/G) matching the
        # native weight layout. Using (N, 1, C_in) breaks grouped
        # convolutions because VVOR would output (K, 1, C_in, Co/G) instead.
        if input.dim() == 2:
            input_3d = input.view(input.shape[0], self.groups, -1).contiguous()
        else:
            input_3d = input.contiguous()

        # Under "force_fused_conv" route the whole
        # mvmr+autograd through the single `FusedPointConv3d` Function
        # (collapses the 3 @triton_op/autograd-graph boundaries + 2
        # seg_offs builds + the duplicate .contiguous() into one Function;
        # forward S2 zero-copy via the no-op-collapse view; the CUTLASS
        # mvmr/vvor full kernels reused as-is; grad_b keeps its single
        # host-repack residual). Intercepted HERE — above
        # the @triton_op — so FusedPointConv3d's autograd is the sole
        # autograd boundary (not nested inside the op's registered
        # autograd). fp16-only; non-fp16 falls through to the unchanged
        # eager op. Every other mode/path is byte-unchanged: this branch
        # fires ONLY on the "force_fused_conv" string.
        #
        # Small-C crash-avoidance fallback. The fused path's CUTLASS mvmr
        # full kernel hard-requires the weight's M and C to be tile
        # multiples: `sparse_mvmr_cutlass_sm{80,90}_full` TORCH_CHECKs
        # `M_full % TileM == 0` and `C_full % TileK == 0` (TileM=64,
        # TileK=32 — `MvmrConfig`/`Sm90MvmrConfig` in
        # extensions/sparse_engines_cuda/csrc/cuda/sparse_mvmr_cutlass_sm80.cu
        # lines 526-531, identical in sm90.cu:199-204; the pinned Python
        # mirrors are `M_TILE`/`C_TILE` in sparse_engines/mvmr_cutlass.py).
        # `weight` is (K, G=1, C, M) here, so the kernel's M_full is
        # weight.shape[3] and C_full is weight.shape[2]. At enc0-class
        # depth (C=32, M not a 64-multiple) the kernel raises a C++
        # RuntimeError, which crashes the bench grid before any verdict
        # JSON. So decide BY SHAPE *before* dispatch (not a fragile
        # try/except around the kernel): when the tile constraints are
        # unmet, fall through to EXACTLY the non-fused composition the
        # `else` below takes when the mode is not "force_fused_conv" —
        # already correct & parity-tested. The decision is on the forward
        # weight shape, invariant across fwd/bwd, so the whole step is
        # owned by one path (no fwd/bwd mix). Constraint-MET shapes still
        # take the fused path byte-unchanged. This is the crash-avoidance
        # fallback only, NOT the perf-optimal "engage only where
        # net-positive" router.
        from sparse_engines.mvmr_cutlass import M_TILE, C_TILE

        _fused_tiles_ok = (
            weight.shape[3] % M_TILE == 0 and weight.shape[2] % C_TILE == 0
        )
        # Perf-optimal auto-router. `_fused_safe` = the fused
        # CUTLASS path's hard preconditions (fp16 + tile-multiple M/C —
        # the small-C crash-avoidance guard, unchanged). The
        # routing DECISION on top of it:
        #   • "force_fused_conv" (benches/tests): engage whenever
        #     `_fused_safe` — byte-unchanged explicit override.
        #   • "auto" (production default): engage ONLY in the measured-
        #     net-positive regime (C >= _FUSED_AUTOROUTER_MIN_C); every
        #     other shape → the Triton path (zero-regression).
        #   • any other mode: → the eager op (those force_* modes are
        #     dispatched inside `sparse_matrix_vector_multiplication_
        #     reduction`); unchanged.
        _mode = current_mode()
        _fused_safe = (weight.dtype == torch.float16) and _fused_tiles_ok
        _use_fused = _fused_safe and (
            _mode == "force_fused_conv"
            or (
                _mode == "auto"
                and weight.shape[2] >= _FUSED_AUTOROUTER_MIN_C
            )
        )
        if _use_fused:
            from sparse_engines.mvmr_cutlass import FusedPointConv3d

            output = FusedPointConv3d.apply(weight, k, input_3d, j, i, n)
        else:
            output = sparse_matrix_vector_multiplication_reduction(weight, k, input_3d, j, i, n)

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
    conv_op: Callable,
    x: Tensor,
    m: MetaData,
    stride: float,
    receptive_field_scaler=1.0,
    distance_type: str = "ball",
) -> Tuple[Tensor, MetaData]:
    """Standard strided convolution with radius-search-based triplet building.

    This is the general-purpose strided conv that works for all stride/kernel
    combinations.

    Args:
        conv_op: PointConv3d instance (uses .kernel_size_3)
        x: Input features tensor (N, C_in)
        m: Input metadata with points, sample_inds, grid_size
        stride: Voxel ratio for downsampling (>1 for downsample)
        receptive_field_scaler: Volume multiplier for search radius (default: 1.0)
        distance_type: "ball" (spherical) or "cube" (cubic) search

    Returns:
        Tuple of (output_features, output_metadata)
    """
    m = handle_stride_and_build_triplets(
        m, stride, conv_op.kernel_size_3, receptive_field_scaler,
        distance_type=distance_type,
    )
    x = conv_op(x, m.i, m.j, m.k, m.num_points())
    return x, m

