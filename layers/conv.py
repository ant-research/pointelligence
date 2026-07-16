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
from sparse_engines._dispatch_override import current_mode, resolve_input_precision

from .metadata import MetaData
from .contract import TripletContract
from .triplets import (
    build_full_cover_strided_rulebook,
    handle_stride_and_build_triplets,
    handle_stride_disjoint_and_build_triplets,
    radius_scaler_for_kernel_size,
)
from .generative import CoordinateGenerator, GeneratedSites, KernelStampGenerator


# torch.compile contract — debug re-validation. When POINTELLIGENCE_DEBUG_CONTRACTS=1,
# _conv_forward re-proves the builder-asserted TripletContract against the
# actual triplet data (eager only — the .item()s here are exactly what we
# moved OUT of the production path). Lets CI run the parity battery and assert
# builder-asserted == data-true, catching any future builder that ships a wrong
# flag (the one eager-numeric-drift surface). Default off; never runs under
# torch.compile or in production.
import os as _os
_POINTELLIGENCE_DEBUG_CONTRACTS = _os.environ.get("POINTELLIGENCE_DEBUG_CONTRACTS", "0") == "1"


def _assert_contract_matches_data(contract, i, j, k, n, n_in):
    """Eager-only proof that a TripletContract matches its triplet data."""
    from sparse_engines._seg_offs import is_sorted_cached, exact_cover_cached
    if contract.k_sorted:
        assert is_sorted_cached(k), (
            "TripletContract.k_sorted=True but k is NOT sorted ascending")
    if contract.exact_cover_out:
        assert k.numel() == n and exact_cover_cached(i, n), (
            "TripletContract.exact_cover_out=True but i is not a permutation "
            f"of [0, n={n}) (T={k.numel()})")
    if contract.exact_cover_in:
        assert k.numel() == n_in and exact_cover_cached(j, n_in), (
            "TripletContract.exact_cover_in=True but j is not a permutation "
            f"of [0, n_in={n_in}) (T={k.numel()})")
    if contract.uniform_seg_len is not None:
        K = int(k.max().item()) + 1 if k.numel() else 0
        assert contract.uniform_seg_len * K == k.numel(), (
            f"TripletContract.uniform_seg_len={contract.uniform_seg_len} * "
            f"K={K} != T={k.numel()}")


# The production "auto" default is the v1.4 best-known route:
# eligible exact submanifold fp16 k=3/G=1/Cin=Cout convs use the fused
# gather-sum operator where the release decision table shows a stable win,
# while ineligible shapes and C512 training fall back to TIG. The older
# Triton-grouped/FSG engines remain reachable via force_fsg* modes for
# rollback and ablation.


def _auto_fused_gather_sum_width(C: int, grad_enabled: bool) -> bool:
    """v1.4 auto policy for an already structurally eligible fused point conv.

    C512 fwd+bwd is slower than TIG on the release grids, but forward-only is
    near-tie/slightly favorable with the fused rulebook/cache route. Treat
    grad-enabled as the train path; no-grad is the validation/inference path.
    """
    return C <= 256 or not grad_enabled


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

        # weight layout: native (K, G, in/G, out/G).
        # Eliminates the per-fwd `weight.reshape(...).permute(3,0,1,2)` +
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
        # tensor, no warning (most legacy checkpoints are pre-Lane-Q and
        # match the old shape exactly).
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
        contract: "Optional[TripletContract]" = None,
        seg_offs: Optional[Tensor] = None,
    ) -> Tensor:
        # torch.compile contract: `contract` is the triplet index's
        # structural facts (k-sortedness, exact-cover, uniform segments),
        # produced ONCE by the builder that made (i, j, k) and carried as DATA.
        # The forward CONSUMES it without re-derivation, so it has no host sync
        # and traces under torch.compile(fullgraph=True). This REPLACES the old
        # per-forward `is_sorted_cached(k).item()` + `exact_cover_cached(i/j)`
        # bincount auto-detection (those D2H syncs were the Dynamo graph break).
        #
        # k-sortedness is an UNCONDITIONAL INVARIANT of the conv path: every
        # builder reaching here emits sort_by="k" (the only sort_by="i"
        # builder, max_pool3d, feeds indexed_segment_reduce, never conv). So
        # `contract=None` defaults to the submanifold contract (k-sorted, no
        # cover). BACK-COMPAT DROP: an external caller that hand-builds
        # genuinely-UNSORTED triplets on `auto` must now pass
        # `TripletContract(k_sorted=False)` to opt into the eager per-triplet
        # path — a bare unsorted MetaData is otherwise treated as sorted
        # (force_pt mode remains the escape hatch). The env flag
        # POINTELLIGENCE_DEBUG_CONTRACTS=1 re-validates the contract against the data
        # (eager only) so CI proves builder-asserted == data-true.
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
        # forward S2 zero-copy via the no-op-collapse view; frozen CUTLASS
        # mvmr/vvor full kernels reused as-is; grad_b keeps its single
        # host-repack = the named residual). Intercepted HERE — above
        # the @triton_op — so FusedPointConv3d's autograd is the sole
        # autograd boundary (not nested inside the op's registered
        # autograd). fp16-only; non-fp16 falls through to the unchanged
        # eager op. Every other mode/path is byte-unchanged: this branch
        # fires ONLY on the "force_fsg_fused" string.
        # Small-C crash-avoidance fallback (verdict-cell
        # unblock + graduation-gate auto-router seed). The fused
        # path's CUTLASS mvmr full kernel hard-requires the weight's M and
        # C to be tile multiples: `sparse_mvmr_cutlass_sm{80,90}_full`
        # TORCH_CHECKs `M_full % TileM == 0` and `C_full % TileK == 0`
        # (TileM=64, TileK=32 — `MvmrConfig`/`Sm90MvmrConfig` in
        # extensions/sparse_engines_cuda/csrc/cuda/sparse_mvmr_cutlass_sm80.cu
        # lines 526-531, identical in sm90.cu:199-204; the pinned Python
        # mirrors are `M_TILE`/`C_TILE` in sparse_engines/mvmr_cutlass.py).
        # `weight` is (K, G=1, C, M) here, so the kernel's M_full is
        # weight.shape[3] and C_full is weight.shape[2]. At enc0-class
        # depth (C=32, M not a 64-multiple) the kernel raises a C++
        # RuntimeError; T3 Finding 2 showed that crashes the Phase-B bench
        # grid before any verdict JSON. So decide BY SHAPE *before*
        # dispatch (not a fragile try/except around the kernel): when the
        # tile constraints are unmet, fall through to EXACTLY the
        # non-fused composition the `else` below takes when the mode is
        # not "force_fsg_fused" — already correct & parity-tested. The
        # decision is on the forward weight shape, invariant across
        # fwd/bwd, so the whole step is owned by one path (no fwd/bwd
        # mix). Constraint-MET shapes still take the fused path
        # byte-unchanged. This is the crash-avoidance fallback only, NOT
        # the perf-optimal "engage only where net-positive" router (a
        # graduation refinement, out of T3.5 scope).
        from sparse_engines.mvmr_cutlass import M_TILE, C_TILE

        _fused_tiles_ok = (
            weight.shape[3] % M_TILE == 0 and weight.shape[2] % C_TILE == 0
        )
        # The auto-router (supersedes the
        # fused-at-C>=512 rule): **TIG is the production default at
        # every (shape, G, dtype)** — the decision tables
        # have TIG
        # best f+b at 25-26/30 cells on BOTH arches (exceptions are
        # near-ties at half precision, <=1.02x; a few fp32 small-channel
        # cells favor the grouped path by up to 1.4x), and TIG strictly
        # beats the LEGACY auto
        # (grouped@C>=128 / fused@C>=512 / PT) at every measured cell
        # including all fp32 cells. Routing:
        #   • "force_fsg_fused" (benches/tests): fused whenever
        #     `_fused_safe` — explicit override, unchanged.
        #   • "auto" (production default): TIG when the triplets are
        #     k-sorted (the build_triplets contract; memoized check) —
        #     all dtypes, all G. Unsorted (non-production callers) falls
        #     to the eager op below, which re-checks and lands on PT.
        #   • "force_tig": TIG unconditionally (assume_sorted contract).
        #   • any other mode: → the eager op (those force_* modes are
        #     dispatched inside `sparse_matrix_vector_multiplication_
        #     reduction`); unchanged.
        _mode = current_mode()
        # FusedPointConv3d is G-complete via fold-G-into-K
        # (single launch per leg at any G). The only extra gate at G>1
        # is fold legality (int32 index ranges); unmet folds fall
        # through to the eager composition below — never raise. The
        # per-group tile check above already evaluates Mg/Cg (the
        # weight layout is (K, G, Cg, Mg)).
        from sparse_engines.mvmr_cutlass import fused_fold_legal

        _fused_safe = (
            (weight.dtype == torch.float16)
            and _fused_tiles_ok
            and fused_fold_legal(self.groups, input_3d.shape[0], n,
                                 k.numel())
        )
        _use_fused = _fused_safe and _mode == "force_fsg_fused"
        # TIG precondition (the submanifold gate is RETIRED):
        # grad_input is sized by the index's n_in, so generative /
        # strided convs (N_in != N_out: partition stem, fan-in-1 deconv,
        # GenerativePointConv3d) route TIG too. On measured grids (sm_89,
        # real ScanNet, all dtypes): generative TIG is best f+b at EVERY
        # generative cell — stem 0.57-0.71x PT, deconv 0.55-0.74x FSG at
        # half precision (fp32 marginal at two cells, still >= the legacy
        # route). Remaining gate, auto only: k-sorted triplets (the
        # build_triplets contract, memoized check — force_tig keeps its
        # explicit assume_sorted contract).
        # Resolve the triplet structural contract ONCE — no host sync.
        # None defaults to the submanifold contract (k-sorted, no cover): the
        # conv-path invariant. The structural flags below are READ from it,
        # never re-derived from the tensors (that re-derivation was the old
        # is_sorted_cached/exact_cover_cached .item() graph break).
        if contract is None:
            contract = TripletContract.submanifold()
        # Optional eager-only re-validation (POINTELLIGENCE_DEBUG_CONTRACTS=1): prove the
        # builder-asserted flags equal the data-true values. Skipped under
        # torch.compile (the .item()s would break the trace) and off by
        # default, so production never pays the sync.
        if (_POINTELLIGENCE_DEBUG_CONTRACTS and k.numel() > 0
                and not torch.compiler.is_compiling()):
            _assert_contract_matches_data(contract, i, j, k, n,
                                          input_3d.shape[0])

        if _mode == "auto":
            # k-sortedness is the conv-path invariant; trust the contract.
            # Unsorted (external opt-out) forfeits TIG → the eager op lands on
            # PT, exactly as before, just decided from data not a runtime sync.
            _use_tig = contract.k_sorted
            if not _use_tig:
                from sparse_engines._dispatch_override import warn_pt_fallback
                warn_pt_fallback(
                    "PointConv3d", "triplets are not k-sorted (contract)",
                    K=weight.shape[0], C=self.in_channels,
                    M=self.out_channels, G=self.groups)
        else:
            _use_tig = _mode == "force_tig"
        # v1.4 fused gather-sum route: within-stage submanifold convs only
        # (k=3 / G=1 / M=C / fp16 / N_in==N_out). ``force_fused_gather_sum`` keeps
        # its diagnostic meaning and fuses every structurally eligible width;
        # ``auto`` uses the best-known release route and falls back to TIG for
        # C512 fwd+bwd. Other backbone convs (stem, 1x1, strided/generative)
        # are ineligible and use TIG where the contract allows it.
        _fb_structural = (
            weight.shape[0] == 27
            and self.groups == 1
            and self.in_channels == self.out_channels
            and weight.dtype == torch.float16
            and input_3d.shape[0] == n
            and contract.k_sorted
        )
        _use_fused_gather_sum = (
            (_mode == "force_fused_gather_sum" and _fb_structural)
            or (_mode == "auto" and _fb_structural
                and _auto_fused_gather_sum_width(self.in_channels,
                                           torch.is_grad_enabled()))
        )
        if _use_fused:
            from sparse_engines.mvmr_cutlass import fused_pointconv3d

            output = fused_pointconv3d(weight, k, input_3d, j, i, n)
        elif _use_fused_gather_sum:
            from sparse_engines.fused_point_conv import (
                fused_gather_sum_conv3d)

            output = fused_gather_sum_conv3d(
                input_3d.view(input_3d.shape[0], -1),
                weight.view(weight.shape[0], weight.shape[2], weight.shape[3]),
                i, j, k, n,
            ).view(-1, self.groups, self.out_channels // self.groups)
        elif _use_tig or _use_fused_gather_sum:
            # The TIG engine (flat orientation — the k-sorted
            # triplets ARE the index, per-call index cost is one
            # searchsorted; the hybrid mode needs the level-0 transform
            # and is reserved for cached-topology use). One autograd
            # node, 3 kernel launches f+b, zero weight staging.
            # Also the "auto" production default (k-sortedness is the
            # contract invariant, so assume_sorted holds).
            from sparse_engines.tig import TigIndex, tig_mvmr

            # exactly-once-scatter fast paths (FI1 forward / FI1 grad_input /
            # the dense-GEMM partition engine at large K). The flags are
            # carried on the contract — PROVEN once at build time by the
            # builder's bincount (in its @compiler.disable region), never
            # re-derived here. exact_cover_out: the 2**3 octant deconv head
            # (fan-in-1). exact_cover_in: the disjoint-partition stem
            # (fan-out-1); dense z is ~(1/occupancy)*input bytes, hard-capped
            # at 256 MB.
            _eco = contract.exact_cover_out
            _eci = contract.exact_cover_in
            _n_in_rows = input_3d.shape[0]
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
                if seg_offs is not None:
                    idx = TigIndex.from_flat(
                        i,
                        j,
                        seg_offs,
                        n_out=n,
                        n_in=_n_in_rows,
                        num_kernel_offsets=_K,
                        exact_cover_out=_eco,
                        exact_cover_in=_eci,
                        uniform_seg_len=(
                            contract.uniform_seg_len
                            if contract.uniform_seg_len is not None else -1),
                    )
                else:
                    idx = TigIndex(
                        i, j, k, n, _K,
                        build_hybrid=False, assume_sorted=True,
                        n_in=_n_in_rows,
                        uniform_seg_len=contract.uniform_seg_len,
                        exact_cover_out=_eco,
                        exact_cover_in=_eci)
                output = tig_mvmr(
                    weight, input_3d.view(_n_in_rows, -1), idx,
                    input_precision=resolve_input_precision(input_3d.dtype),
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

    def forward(
        self,
        input: Tensor,
        i: Tensor,
        j: Tensor,
        k: Tensor,
        n: int,
        contract: "Optional[TripletContract]" = None,
        seg_offs: Optional[Tensor] = None,
    ) -> Tensor:
        # Forward the triplet's structural contract (k-sorted / exact-cover
        # / uniform segments — produced by the builder) so `auto` reaches the
        # FI1 forward + native-fp16 grad_input fast paths WITHOUT a per-forward
        # host sync. None defaults to the submanifold contract (the conv-path
        # invariant); the forward then traces under torch.compile.
        return self._conv_forward(
            input, i, j, k, n, self.weight, self.bias,
            contract=contract, seg_offs=seg_offs)


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
    combinations. For the special case where input neighborhoods are guaranteed
    to be disjoint (no-overlap regime), use ``conv_pooling_disjoint`` for
    significantly faster performance (3.58-9.11x speedup on triplet building).

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
    x = conv_op(
        x, m.i, m.j, m.k, m.num_points(), contract=m.contract,
        seg_offs=m.seg_offs)
    return x, m


def conv_with_stride_full_cover(
    conv_op: Callable,
    x: Tensor,
    m: MetaData,
    stride: float,
    radius_margin: float = 1e-2,
    radius_backend: str = "auto",
) -> Tuple[Tensor, MetaData]:
    """Overlapping strided point convolution with guaranteed input coverage.

    Output centers are observed input points. The initial center set picks the
    nearest input point to each occupied stride-cell center, then a deterministic
    residual R-net adds observed points until every input point lies within the
    strided radius of at least one center. The cached reverse rulebook is the
    exact transpose graph over the same edges.
    """
    if stride <= 1:
        raise ValueError(
            f"conv_with_stride_full_cover requires stride > 1; got {stride}.")
    with torch.no_grad():
        if (m.parent is not None and
            m.parent.points.shape == m.points.shape and
            m.parent.grid_size == m.grid_size and
            torch.equal(m.parent.points, m.points) and
            torch.equal(m.parent.sample_inds, m.sample_inds)):
            parent_meta = m.parent
        else:
            parent_meta = MetaData(
                points=m.points,
                sample_inds=m.sample_inds,
                sample_sizes=m.sample_sizes,
                grid_size=m.grid_size,
                parent=m.parent,
            )
            m.parent = parent_meta

        rb = build_full_cover_strided_rulebook(
            points=parent_meta.points,
            sample_inds=parent_meta.sample_inds,
            sample_sizes=parent_meta.sample_sizes,
            stride=stride,
            input_grid_size=parent_meta.grid_size,
            kernel_size=conv_op.kernel_size_3,
            radius_margin=radius_margin,
            radius_backend=radius_backend,
            return_num_neighbors=True,
        )

        m.points = rb.points
        m.sample_inds = rb.sample_inds
        m.sample_sizes = rb.sample_sizes
        m.grid_size = float(stride) * parent_meta.grid_size
        m.downsample_indices = rb.center_source_indices
        m.i = rb.i
        m.j = rb.j
        m.k = rb.k
        from sparse_engines._seg_offs import kernel_offset_segments
        num_kernel_offsets = math.prod(conv_op.kernel_size_3)
        m.seg_offs = kernel_offset_segments(m.k, num_kernel_offsets)
        m.num_neighbors = rb.num_neighbors
        m.contract = TripletContract(k_sorted=True)

        parent_meta.i_upsample = rb.i_upsample
        parent_meta.j_upsample = rb.j_upsample
        parent_meta.k_upsample = rb.k_upsample
        parent_meta.seg_offs_upsample = kernel_offset_segments(
            rb.k_upsample, num_kernel_offsets)
        parent_meta.full_cover_point_to_initial_center = rb.point_to_initial_center
        parent_meta.full_cover_initial_center_indices = rb.initial_center_source_indices
        parent_meta.full_cover_additional_center_indices = rb.additional_center_source_indices
        parent_meta.full_cover_coverage_per_input = rb.coverage_per_input
        parent_meta.full_cover_telemetry = {
            "radius": rb.radius,
            "radius_scaler": rb.radius_scaler,
            "initial_center_count": int(rb.initial_center_source_indices.numel()),
            "additional_center_count": int(rb.additional_center_source_indices.numel()),
            "selector_round_count": rb.selector_round_count,
            "edge_count": int(rb.k.numel()),
        }

    x = conv_op(
        x, m.i, m.j, m.k, m.num_points(), contract=m.contract,
        seg_offs=m.seg_offs)
    return x, m



def conv_with_stride_disjoint(
    conv_op: Callable,
    x: Tensor,
    m: MetaData,
    stride: float,
) -> Tuple[Tensor, MetaData]:
    """Strided conv as a true GRID PARTITION — the patchify operator.

    Each input point is assigned to exactly one output cell (voxel) at
    ``cell_size = stride * grid_size``; the output token sits at the cell
    CENTER and the conv weight is indexed by the point's cell-grid-relative
    sub-voxel slot (``[0, K**3)``, ``K = conv_op`` kernel size). Disjoint AND
    covering: every input point contributes to exactly one token (no
    orphaning), unlike a radius gather. This is the point analogue of a ViT
    ``Conv2d(kernel=P, stride=P)`` patchify.

    The partition's cached upsample edges make
    ``Upsample(straight_recover=True, recompute_k=False)`` the exact inverse
    (the unpatchify): every raw point is recovered from its cell-center token,
    weighted by its sub-voxel slot. Together they are a lossless
    patchify/unpatchify pair.

    Args:
        conv_op: PointConv3d instance (its ``kernel_size_3`` sets the K**3 slots).
        x: Input features (N_in, C_in).
        m: Input metadata (points at grid_size).
        stride: Voxel ratio for downsampling (>1); cell_size = stride * grid_size.

    Returns:
        (output_features at N_tokens, output_metadata at cell centers).

    Raises:
        ValueError: If stride <= 1 or the kernel is non-cubic.

    See Also:
        handle_stride_disjoint_and_build_triplets: the partition triplet builder.
        conv_with_stride: general overlapping strided conv (radius search).

    NOTE: supersedes the prior ball-radius-search disjoint conv (a fast
    no-overlap *gather* that matched ``conv_with_stride`` but only covered the
    inscribed sphere of the cell, orphaning ~25-50% of points at the corners).
    ``receptive_field_scaler`` / ``distance_type`` were removed (a partition has
    no search radius).
    """
    m = handle_stride_disjoint_and_build_triplets(m, stride, conv_op.kernel_size_3)
    x = conv_op(
        x, m.i, m.j, m.k, m.num_points(), contract=m.contract,
        seg_offs=m.seg_offs)
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
    to reuse a rulebook across sibling generative convs.
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
        # per-tap segments (and, for generation-from-one, fan-in-1 exact
        # cover) — the generator already knows this, so it ships the same
        # typed TripletContract every conv path uses (no parallel hint dict).
        # The TIG index path then goes closed-form (no sync/searchsorted/cumsum)
        # and the forward traces under torch.compile.
        x = self._conv_forward(
            input, sites.i, sites.j, sites.k, sites.n_out, self.weight,
            self.bias, contract=sites.to_contract(),
        )
        return x, sites.to_metadata(parent=m)
