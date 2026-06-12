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
from sparse_engines._dispatch_override import current_mode, current_precision

from .metadata import MetaData
from .triplets import (
    handle_stride_and_build_triplets,
    radius_scaler_for_kernel_size,
)
from .generative import CoordinateGenerator, GeneratedSites, KernelStampGenerator


# The production "auto" default is **TIG** at every (shape, G, dtype):
# the decision tables (sm_89 + H200, real ScanNet cells) have TIG best
# f+b at 25-26/30 cells on both arches (exceptions are near-ties at
# half precision, <=1.02x; a few fp32 small-channel cells favor the
# grouped path by up to 1.4x
# force_fsg_fused tie), and TIG strictly beats the legacy auto routing
# (Triton-grouped@C>=128 / fused@C>=512 / per-triplet) at every
# measured cell including all fp32 cells. The fused-at-C>=512
# auto-engagement is RETIRED with its constant (`force_fsg_fused`
# remains as the explicit override; its half-precision wins are all
# <=1.02x over TIG — not worth a second default engine). FSG stays
# fully reachable via the force_fsg* modes (rollback + ablation, one
# release minimum).


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
        *,
        tig_hint: Optional[dict] = None,
    ) -> Tensor:
        # `tig_hint` is a caller-asserted structure contract (e.g.
        # GenerativePointConv3d's stamp rulebook): k-sortedness by
        # construction (skips the is_sorted D2H sync) and optionally
        # `uniform_seg_len` (closed-form TIG index — no searchsorted, no
        # cumsum). Default None — every existing caller byte-unchanged.
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

        # Under "force_fsg_fused" route the whole
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
        # fires ONLY on the "force_fsg_fused" string.
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
        # `else` below takes when the mode is not "force_fsg_fused" —
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
        # Auto-router (supersedes the fused-at-C>=512 rule): **TIG is the
        # production default at every (shape, G, dtype)** — the decision
        # tables (sm_89 + H200, real ScanNet cells) have TIG best f+b at
        # 25-26/30 cells on BOTH arches (exceptions are near-ties at
        # half precision, <=1.02x; a few fp32 small-channel cells favor
        # the grouped path by up to 1.4x; fused
        # tie), and TIG strictly beats the legacy auto (grouped@C>=128 /
        # fused@C>=512 / per-triplet) at every measured cell including
        # all fp32 cells. Routing:
        #   • "force_fsg_fused" (benches/tests): fused whenever
        #     `_fused_safe` — explicit override, unchanged.
        #   • "auto" (production default): TIG when the triplets are
        #     k-sorted (the build_triplets contract; memoized check) —
        #     all dtypes, all G. Unsorted (non-production callers) falls
        #     to the eager op below, which re-checks and lands on the
        #     per-triplet path.
        #   • "force_tig": TIG unconditionally (assume_sorted contract).
        #   • any other mode: → the eager op (those force_* modes are
        #     dispatched inside `sparse_matrix_vector_multiplication_
        #     reduction`); unchanged.
        _mode = current_mode()
        # FusedPointConv3d is G-complete via fold-G-into-K (single launch
        # per leg at any G). The only extra gate at G>1 is fold legality
        # (int32 index ranges); unmet folds fall through to the eager
        # composition below — never raise. The per-group tile check
        # above already evaluates Mg/Cg (the weight layout is
        # (K, G, Cg, Mg)).
        from sparse_engines.mvmr_cutlass import fused_fold_legal

        _fused_safe = (
            (weight.dtype == torch.float16)
            and _fused_tiles_ok
            and fused_fold_legal(self.groups, input_3d.shape[0], n,
                                 k.numel())
        )
        _use_fused = _fused_safe and _mode == "force_fsg_fused"
        # TIG precondition (the submanifold gate is RETIRED): grad_input
        # is sized by the index's n_in, so generative / strided convs
        # (N_in != N_out: partition stem, fan-in-1 deconv,
        # GenerativePointConv3d) route TIG too. The decision tables
        # (sm_89, real ScanNet, all dtypes) have generative TIG best f+b
        # at EVERY generative cell — stem 0.57-0.71x the per-triplet
        # path, deconv 0.55-0.74x FSG at half precision (fp32 marginal
        # at two cells, still >= the legacy route). Remaining gate, auto
        # only: k-sorted triplets (the build_triplets contract, memoized
        # check — force_tig keeps its explicit assume_sorted contract).
        if _mode == "auto":
            if tig_hint is not None:  # sorted by caller contract
                _use_tig = True
            else:
                from sparse_engines._seg_offs import is_sorted_cached
                _use_tig = is_sorted_cached(k)
                if not _use_tig:
                    # Unsorted triplets forfeit TIG AND the grouped path
                    # — the eager op will land on the per-triplet path.
                    from sparse_engines._dispatch_override import (
                        warn_pt_fallback)
                    warn_pt_fallback(
                        "PointConv3d", "triplets are not k-sorted",
                        K=weight.shape[0], C=self.in_channels,
                        M=self.out_channels, G=self.groups)
        else:
            _use_tig = _mode == "force_tig"
        if _use_fused:
            from sparse_engines.mvmr_cutlass import FusedPointConv3d

            output = FusedPointConv3d.apply(weight, k, input_3d, j, i, n)
        elif _use_tig:
            # The TIG engine (flat orientation — the k-sorted triplets
            # ARE the index, per-call index cost is one searchsorted;
            # the hybrid mode needs the level-0 transform and is
            # reserved for cached-topology use). One autograd node,
            # 3 kernel launches f+b, zero weight staging.
            # Also the automatic production default (sortedness verified
            # above via the memoized check, so assume_sorted holds).
            from sparse_engines.tig import TigIndex, tig_mvmr

            _eco = (tig_hint or {}).get("exact_cover_out", False)
            if not _eco and tig_hint is None and k.numel() == n:
                from sparse_engines._seg_offs import exact_cover_cached
                _eco = exact_cover_cached(i, n)
            # The input-side twin: fan-out-1 (every input row in exactly
            # one triplet — the disjoint-partition stem class). T == N_in
            # is the free host gate (submanifold and deconv shapes fail
            # it); the same memoized bincount proof on j then sets
            # exact_cover_in (FI1 grad_input where that wins) and, at
            # large K, routes the dense-GEMM partition engine (measured
            # 3-8x at the K>=512 patchify-stem shapes on both Ada and
            # Hopper; the dense z is ~(1/occupancy)*input bytes —
            # hard-capped at 256 MB).
            _n_in_rows = input_3d.shape[0]
            _eci = (tig_hint or {}).get("exact_cover_in", False)
            if (not _eci and tig_hint is None
                    and k.numel() == _n_in_rows and k.numel() != n):
                from sparse_engines._seg_offs import exact_cover_cached
                _eci = exact_cover_cached(j, _n_in_rows)
            _K = weight.shape[0]
            if (_eci and self.groups == 1 and _K > 64
                    and n * _K * self.in_channels * input.element_size()
                    <= 256 * 2 ** 20):
                from sparse_engines.partition_gemm import (
                    partition_dense_mvmr)

                output = partition_dense_mvmr(
                    weight, input_3d.view(_n_in_rows, -1), i, j, k, n,
                ).view(-1, self.groups, self.out_channels // self.groups)
            else:
                idx = TigIndex(i, j, k, n, _K,
                                build_hybrid=False, assume_sorted=True,
                                n_in=_n_in_rows,
                                uniform_seg_len=(tig_hint or {}).get(
                                    "uniform_seg_len"),
                                exact_cover_out=_eco,
                                exact_cover_in=_eci)
                output = tig_mvmr(
                    weight, input_3d.view(_n_in_rows, -1), idx,
                    input_precision=current_precision(),
                ).view(-1, self.groups,
                       self.out_channels // self.groups)
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

    def forward(self, input: Tensor, i: Tensor, j: Tensor, k: Tensor, n: int,
                tig_hint: Optional[dict] = None) -> Tensor:
        # Forward the caller's structure contract (e.g. exact_cover_out
        # for an exact 2**3 octant deconv) to the dispatcher so `auto`
        # reaches the single-pass store + native-fp16 grad paths. None
        # (the default) preserves prior behavior bit-for-bit.
        return self._conv_forward(input, i, j, k, n, self.weight, self.bias,
                                  tig_hint=tig_hint)


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


class GenerativePointConv3d(GeneralConv):
    __doc__ = r"""Generative-expansion point convolution.

    Unlike :class:`PointConv3d` (submanifold: output sites ≡ input
    sites), this operator *invents* a denser output point set Y from the
    input set X and convolves X → Y. The output coordinate set is
    produced by a swappable
    :class:`~layers.generative.CoordinateGenerator` (default:
    :class:`~layers.generative.KernelStampGenerator`, the deterministic
    kernel-tap stamping rule).

    ``forward(input, m)`` returns ``(features, m_out)`` — the new
    :class:`MetaData` carries the generated points and links back to the
    input as its ``parent``. A precomputed
    :class:`~layers.generative.GeneratedSites` may be passed as ``sites``
    to reuse a rulebook across sibling generative convs (an explicit
    rulebook-reuse handle).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t = 3,
        expansion: float = 2.0,
        generator: Optional[CoordinateGenerator] = None,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.kernel_size_3 = _triple(kernel_size)
        self.expansion = expansion
        if generator is None:
            generator = KernelStampGenerator(self.kernel_size_3, expansion)
        # The conv weight's tap count K is whatever the generator's
        # stencil produces — for the default KernelStampGenerator that is
        # prod(kernel_size); a custom-stencil generator may differ, and
        # then the `kernel_size` argument is unused.
        super().__init__(
            in_channels,
            out_channels,
            generator.kernel_taps,
            groups,
            bias,
            **factory_kwargs,
        )
        self.generator = generator

    def forward(
        self,
        input: Tensor,
        m: MetaData,
        sites: Optional[GeneratedSites] = None,
    ) -> Tuple[Tensor, MetaData]:
        if sites is None:
            sites = self.generator(m)
        sites.validate()
        # The stamp rulebook is k-sorted by construction with uniform
        # per-tap segments — hand the contract to the dispatch so the
        # TIG index path goes closed-form (no sync/searchsorted/cumsum).
        # Generation-from-one (dedup merged nothing) additionally
        # engages the FI1 plain-store forward — the subdivision-upsample
        # mode.
        hint = None
        if sites.uniform_seg_len is not None or sites.exact_cover_out:
            hint = dict(uniform_seg_len=sites.uniform_seg_len,
                        exact_cover_out=sites.exact_cover_out)
        x = self._conv_forward(
            input, sites.i, sites.j, sites.k, sites.n_out, self.weight,
            self.bias, tig_hint=hint,
        )
        return x, sites.to_metadata(parent=m)
