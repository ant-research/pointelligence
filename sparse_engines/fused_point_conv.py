"""Fused point convolution — the v1.4.0 point-native within-stage 3x3x3
(K=27) engine. Its gather-sum schedule pre-sums each bucket of inputs IN
REGISTER and spends ONE matmul on the sum in all three training passes
(forward, grad_input, grad_weight).

The radius-neighborhood a point-native conv produces is
NON-INJECTIVE: for a given (output point i, tap k) several input points
may land in the same bucket. The dense/exact submanifold engines assume
at most one input per bucket; the general rulebook handles the extra
inputs as separate triplets, so a bucket of multiplicity m costs m
matmul rows. The fused idea collapses that: because the conv contracts a
bucket's inputs against the SAME weight ``W[k]``,

    out[i] += (Σ_{j in bucket(i,k)} feat[j]) @ W[k]

the inputs can be summed first and the bucket then costs ONE matmul row
regardless of multiplicity — num_buckets matmul rows instead of
num_triplets. "Fused" refers to this per-bucket fusion of gather + sum before
the channel matmul; it does not mean the forward, grad_input, and grad_weight
passes are collapsed into a single kernel. All three passes reuse the same
gather-sum schedule:

- **forward** (``_seg_masked_kernel``): output-stationary masked iGEMM
  over the level-0 compacted rows (one row per occupied output point),
  with a MAXLVL-deep in-register slot loop summing each bucket's inputs
  and a direct ``tl.store`` (one writer per output row). Buckets with
  multiplicity above MAXLVL spill into a small flat residual rulebook
  run on top.
- **grad_input**: the SAME forward kernel on the TRANSPOSED rulebook —
  swap input<->output (bucket key becomes (j, k)) and transpose the
  weight ``W[k]^T`` — so ``grad_in[j] += (Σ_i grad_out[i]) @ W[k]^T``
  reuses the gather-sum verbatim, no separate kernel.
- **grad_weight** (``_seg_fused_wgrad_kernel``): bucket-major VVOR reduction —
  one outer product ``(Σ_j feat[j])^T (grad_out[i])`` per bucket instead of
  one outer product per triplet. This is the v1.4 weight-gradient win: the
  gather-sum idea reduces VVOR multiplicity before the deterministic segment
  reduction inherited from v1.3. The cross-program reduction
  over the split-K chunks is selected by ``REDUCE_MODE`` (host-summed
  partials / in-kernel atomic into one fp32 buffer / a no-split-K direct
  store), so the narrow-C atomic vs wide-C direct routing is decided by
  the autotuner per (arch, C, dtype), not hardcoded.

Tile + reduce configs are chosen by a Triton ``@triton.autotune`` keyed
ONLY on ``(C, M, dtype_key)`` — all three are constant per stage. The key
NEVER contains N or the triplet/bucket count (a per-iteration-varying
length would re-trigger autotune every step). ``dtype_key`` splits the
cache per numeric path (fp16 / bf16 / tf32 / fp32-ieee) so the tf32 vs
ieee precision choice for fp32 inputs caches independently even though
both carry fp32 tensors. ``INPUT_PRECISION`` follows the dtype: ``"tf32"``
for fp32 inputs (tensor cores, the fast path), ``"ieee"`` for fp16/bf16
and for an explicit exact-fp32 request.

Scope: G==1 within-stage point conv with the submanifold contract
(N_in == N_out), kernel_size=3 (K=27), fp16 / bf16 / fp32. The autotune
key is bucketed on channel sizes only — never on N or the triplet count.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from .tig import TigIndex, tig_forward, tig_grad_weight
from ._seg_offs import kernel_offset_segments

__all__ = [
    "FusedGatherSumRulebook",
    "fused_forward",
    "fused_grad_input",
    "fused_grad_weight",
    "FusedPointConv3d",
    "fused_gather_sum_conv3d",
]

K = 27  # kernel volume for kernel_size=3
# MAXLVL = in-register slot depth: how many inputs per bucket the fused
# kernel sums directly. Buckets deeper than this spill to a per-bucket
# overflow drain (gated, in-kernel) and a tiny flat residual. These are
# rulebook params (the index layout depends on them), not tile knobs — they
# are NOT autotuned. They are the per-rulebook DEFAULTS used when the caller
# gives no width; the production entry point overrides them per width via
# ``_maxlvl_for_width`` (see below). Env overrides expose them for re-tuning.
_MAXLVL_FWD = int(os.environ.get("PNT_MAXLVL_FWD", "4"))
_MAXLVL_GI = int(os.environ.get("PNT_MAXLVL_GI", "4"))
_MAXLVL_WGRAD = int(os.environ.get("PNT_MAXLVL_WGRAD", "3"))


def _maxlvl_for_width(C: int) -> int:
    """Forward / grad_input slot depth as a function of channel width.

    The radius-neighborhood is multiplicity-1-dominated (≈80% of nonempty buckets
    carry a single input at rfs*-aligned real ScanNet), so deep
    in-register slot loops mostly load masked-empty lanes. A corrected
    local Ada sweep over C={32,64,128,256,512} found depth 2 best for
    fwd+grad_input+grad_weight on every width. Anything past the chosen
    depth is drained by the gated overflow CSR + flat residual, so parity
    is unchanged. wgrad keeps its own ``_MAXLVL_WGRAD`` because the
    bucket-major reduction has a different optimum."""
    override = os.environ.get("PNT_MAXLVL_FWD")
    if override is not None:
        ml = int(override)
        if ml <= 0:
            raise ValueError("PNT_MAXLVL_FWD must be a positive integer")
        return ml
    return 2


def _fold_for_width(C: int) -> bool:
    """Whether to fold the overflow residual into the main kernels.

    Folding deletes three launch-bound residual passes, but it adds a gated
    overflow drain to the main fwd/gi/gw kernels. Real ScanNet release-batch
    A/B keeps it for narrow widths (C64/C128) and leaves C256+ on the explicit
    residual path, where folding regresses. Env override is for retuning / arch
    confirmation only: ``PNT_FOLD_OVERFLOW=1`` forces on, ``0`` forces off,
    unset = width rule.
    """
    override = os.environ.get("PNT_FOLD_OVERFLOW")
    if override is not None:
        return override not in ("0", "false", "False", "off", "OFF")
    return C <= 128

# Row tile for the forward / grad_input masked kernel. The host builds the
# per-tap occupancy ``tilemask`` and launch grid for one active B1 per call, so
# the autotune pruner keeps only configs matching the width policy below.
_SEG_B1_ENV = os.environ.get("PNT_SEG_B1")
_SEG_MAXNREG_ENV = os.environ.get("PNT_SEG_MAXNREG")
_SEG_ROW_ORDER_ENV = os.environ.get("PNT_SEG_ROW_ORDER", "gray").lower()


def _seg_force_env(name: str, allowed: tuple[int, ...]) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = int(raw)
    if value not in allowed:
        opts = ", ".join(str(v) for v in allowed)
        raise ValueError(f"{name} must be one of {opts}")
    return value


_SEG_BM_FORCE = _seg_force_env("PNT_SEG_BM", (32, 64, 128, 256))
_SEG_BC_FORCE = _seg_force_env("PNT_SEG_BC", (32, 64, 128))
_SEG_WARPS_FORCE = _seg_force_env("PNT_SEG_WARPS", (4, 8))
_SEG_STAGES_FORCE = _seg_force_env("PNT_SEG_STAGES", (2, 3))


def _seg_b1_for_width(C: int, dtype_key: int | None = None) -> int:
    if _SEG_B1_ENV is not None:
        b1 = int(_SEG_B1_ENV)
        if b1 not in (32, 64, 128, 256):
            raise ValueError("PNT_SEG_B1 must be one of 32, 64, 128, 256")
        return b1
    # Real ScanNet forward sweeps split by arch: Ada fp16 benefits from a
    # smaller C64 row tile and a wider C256 row tile, but C512 is row-starved
    # enough that the prior B1=64 tile wins. H200 regresses on the Ada fp16
    # changes and keeps the prior policy. Wide tf32 still prefers B1=128 on
    # both arches. Keep bf16 and exact-fp32 on the prior route until they have
    # their own confirmation.
    is_ada = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8
    if is_ada and dtype_key == 0 and C == 64:
        return 32
    if is_ada and dtype_key == 0 and C == 256:
        return 128
    if C >= 256 and dtype_key == 2:
        return 128
    return 64

# dtype_key encoding for the autotune key (keeps each numeric path's tuned
# config in its own cache slot — notably fp32-ieee vs tf32, which share the
# fp32 tensor dtype but differ in INPUT_PRECISION).
_DTYPE_KEY = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}


def _dtype_key(dtype: torch.dtype, fp32_ieee: bool) -> int:
    """Map (input dtype, exact-fp32 override) to the constexpr dtype_key.
    fp32 splits into tf32 (2, the fast tensor-core default) and fp32-ieee
    (3, the exact override)."""
    if dtype == torch.float32:
        return 3 if fp32_ieee else 2
    return _DTYPE_KEY[dtype]


def _input_precision(dtype: torch.dtype, fp32_ieee: bool) -> str:
    """INPUT_PRECISION for ``tl.dot``: tf32 for fp32 inputs (tensor cores,
    the 1.2-1.7x win), ieee for fp16/bf16 (already native-precision mma) and
    for an explicit exact-fp32 request."""
    if dtype == torch.float32:
        return "ieee" if fp32_ieee else "tf32"
    return "ieee"


# ── autotune config grids ─────────────────────────────────────────────────────


def _seg_autotune_configs():
    """fwd / grad_input masked-iGEMM tile candidates. Anchored on the per-width
    sweep optimum (num_stages=2, num_warps=8, the M-tile growing with C and a
    64-wide contraction tile) plus neighbours (BM in {32,64,128,256}, BC in {32,64,128},
    warps in {4,8}, stages in {2,3}) so a different arch (H200) can pick its own
    optimum. B1 is dtype/width-routed; the pruner keeps only active configs.

    Each tile is emitted twice — uncapped AND ``maxnreg``-capped — so a
    register-pressure-vs-occupancy trade can be made per (arch, width). The
    uncapped variant compiles to ~180-240 regs/thread at the wide tiles, which
    on Ada limits the kernel to ~1 block/SM; a 128-reg cap raises achievable
    occupancy on a high-block-count stage (measured: real-ScanNet C256 fp16
    forward ~4-5% faster at fixed tile and on the autotuned path). The capped
    variant only helps where the stage is occupancy-limited with enough rows to
    fill the freed blocks — see the gate in ``_seg_prune`` (C512 stays uncapped).
    A real cap via ``PNT_SEG_MAXNREG`` overrides the pair with that single
    value."""
    cfgs = []
    if _SEG_MAXNREG_ENV is not None:
        maxnreg = int(_SEG_MAXNREG_ENV)
        if maxnreg <= 0:
            raise ValueError("PNT_SEG_MAXNREG must be a positive integer")
        maxnreg_opts = (maxnreg,)
    else:
        maxnreg_opts = (None, 128)
    for mnr in maxnreg_opts:
        for B1 in (32, 64, 128, 256):
            for BM in (32, 64, 128, 256):
                for BC in (32, 64, 128):
                    for nw in (4, 8):
                        for ns in (2, 3):
                            cfgs.append(triton.Config(
                                dict(B1=B1, BM=BM, BC=BC), num_warps=nw,
                                num_stages=ns, maxnreg=mnr))
    return cfgs


def _seg_prune(configs, nargs, **kwargs):
    """Drop seg tiles that over-tile a narrow stage (BM/BC past max(32, C/M))."""
    C = nargs["C"]; M = nargs["M"]; active_b1 = nargs["B1_ACTIVE"]
    keep = []
    for c in configs:
        kw = c.kwargs
        if kw["B1"] != active_b1:
            continue
        # maxnreg-capped variants trade register pressure for occupancy. The
        # win is real ONLY where the stage is occupancy-limited AND has enough
        # row-tiles to fill the extra resident blocks: on Ada fp16, C256
        # (fine 0.16 grid -> many rows, wide tile -> ~180 regs/thread capping it
        # to ~1 block/SM) gets ~5% from a 128-reg cap. C512 is the opposite —
        # gather-bound and block-starved (coarse 0.32 grid -> few rows), so a cap
        # only shrinks the tile / spills and regresses. The capped/uncapped pair
        # also sits within do_bench noise, so letting the autotuner choose flips
        # run-to-run; instead apply the cap DETERMINISTICALLY: keep ONLY the
        # capped variant where it is validated to win, ONLY the uncapped variant
        # elsewhere. (H200 occupancy at C512 is untested — revisit with a cluster
        # run; env PNT_SEG_MAXNREG forces a single cap at any width, bypassing
        # this gate.)
        if _SEG_MAXNREG_ENV is None:
            is_ada = (torch.cuda.is_available()
                      and torch.cuda.get_device_capability()[0] == 8)
            cap_wins = is_ada and nargs["dtype_key"] == 0 and C == 256
            if cap_wins and c.maxnreg is None:
                continue          # force the cap at the validated stage
            if (not cap_wins) and c.maxnreg is not None:
                continue          # no capped variant anywhere else
        if _SEG_BM_FORCE is not None and kw["BM"] != _SEG_BM_FORCE:
            continue
        if _SEG_BC_FORCE is not None and kw["BC"] != _SEG_BC_FORCE:
            continue
        if _SEG_WARPS_FORCE is not None and c.num_warps != _SEG_WARPS_FORCE:
            continue
        if _SEG_STAGES_FORCE is not None and c.num_stages != _SEG_STAGES_FORCE:
            continue
        if kw["BM"] > max(32, M):
            continue
        if kw["BC"] > max(32, C):
            continue
        keep.append(c)
    return keep or [configs[0]]


def _wgrad_splitk_cap(C: int) -> int:
    """Max split-K the wgrad partials buffer is sized for at this width. Wide
    stages keep it small (each split writes a full [C, M] grad tile, so
    over-splitting inflates grad-write traffic AND the partials buffer is
    C^2-sized); narrow stages allow a deeper split for SM occupancy."""
    return 4 if C >= 256 else 8


def _wgrad_autotune_configs():
    """grad_weight candidates spanning {REDUCE_MODE, SPLITK, tile}. REDUCE_MODE
    0=host-summed partials, 1=in-kernel atomic into one fp32 buffer, 2=direct
    no-split-K single store. The atomic-narrow / direct-wide routing is left to
    the autotuner (it benchmarks all three per (C, dtype) and the per-width
    pruner drops the over-split / over-tiled ones)."""
    cfgs = []
    # split-K modes (0 partials, 1 atomic): SPLITK in {1,2,4,8}.
    for mode in (0, 1):
        for sk in (1, 2, 4, 8):
            for BL in (32, 64, 128):
                for BC in (32, 64, 128):
                    for BM in (32, 64):
                        for nw in (4, 8):
                            cfgs.append(triton.Config(
                                dict(REDUCE_MODE=mode, SPLITK=sk, BL=BL, BC=BC, BM=BM),
                                num_warps=nw, num_stages=2))
    # direct mode (2): single program per (tap, c-tile, m-tile), SPLITK fixed 1.
    for BL in (32, 64, 128):
        for BC in (32, 64, 128):
            for BM in (32, 64):
                for nw in (4, 8):
                    cfgs.append(triton.Config(
                        dict(REDUCE_MODE=2, SPLITK=1, BL=BL, BC=BC, BM=BM),
                        num_warps=nw, num_stages=2))
    return cfgs


def _wgrad_prune(configs, nargs, **kwargs):
    """Per-width wgrad pruning: keep SPLITK <= the width's buffer cap, and drop
    tiles wider than the stage. Keeps the benchmarked set per (C, dtype) small
    and bounds the partials buffer the host pre-allocates."""
    C = nargs["C"]; M = nargs["M"]
    cap = _wgrad_splitk_cap(C)
    keep = []
    for c in configs:
        kw = c.kwargs
        if kw["SPLITK"] > cap:
            continue
        if kw["BC"] > max(32, C):
            continue
        if kw["BM"] > max(32, M):
            continue
        keep.append(c)
    return keep or [configs[0]]


# ── kernels ──────────────────────────────────────────────────────────────────


@triton.autotune(configs=_seg_autotune_configs(), key=["C", "M", "dtype_key", "B1_ACTIVE"],
                 prune_configs_by={"early_config_prune": _seg_prune})
@triton.jit
def _seg_masked_kernel(nbr_multi, ov_off, ov_cnt, ov_j, tile_ovmax,
                       rows_sorted, b, a, out, tilemask,
                       NROWS, M, C, G, dtype_key,
                       B1_ACTIVE: tl.constexpr,
                       INPUT_PRECISION: tl.constexpr,
                       KV: tl.constexpr, MAXLVL: tl.constexpr, FOLD: tl.constexpr,
                       B1: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    """Output-stationary fused gather-sum forward over the level-0
    compacted rows. Each program owns a B1-row tile of occupied output
    points; for every tap present in the tile (``tilemask`` skips empty
    taps) it sums up to MAXLVL bucket inputs in-register (``xsum``) then
    issues ONE ``tl.dot`` of the summed inputs against ``W[tap]``. The
    weight is the contiguous (K, G, C, M) forward layout. Each output row
    is written by exactly one program per (group, M-tile), so the store is
    plain (no atomic). ``dtype_key`` is an autotune-key constexpr only
    (splits the config cache per numeric path).

    When ``FOLD`` is set the rare deep-multiplicity tail (bucket inputs
    beyond MAXLVL) is drained IN this kernel via a per-(row-tile, tap)
    runtime-bounded loop instead of a separate flat residual launch. The
    drain is keyed by a per-(row,tap) CSR (``ov_off`` / ``ov_cnt`` into
    ``ov_j``) and gated by ``tile_ovmax[pid_n, tap]`` — the whole drain
    block (incl. its B1-wide CSR gathers) is hoisted under ``ovmax > 0`` so
    the overwhelmingly common no-overflow tiles pay nothing. The host
    clusters overflow-bearing rows into a contiguous row-tile suffix so
    clean tiles get ``ovmax == 0``."""
    num_pid_m = tl.cdiv(M, BM)
    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid_g = (pid // num_pid_m) % G
    pid_n = pid // (num_pid_m * G)
    r_offsets = pid_n * B1 + tl.arange(0, B1)
    r_mask = r_offsets < NROWS
    rows = tl.load(rows_sorted + r_offsets, mask=r_mask, other=0).to(tl.int64)
    tm = tl.load(tilemask + pid_n)
    m_offsets = pid_m * BM + tl.arange(0, BM)
    m_mask = m_offsets < M
    acc = tl.zeros((B1, BM), dtype=tl.float32)
    for v in range(0, KV):
        if ((tm >> v) & 1) == 1:
            base = (rows * KV + v) * MAXLVL
            ovmax = 0
            if FOLD:
                ovmax = tl.load(tile_ovmax + pid_n * KV + v)
            for c0 in range(0, C, BC):
                c_offsets = c0 + tl.arange(0, BC)
                c_mask = c_offsets < C
                xsum = tl.zeros((B1, BC), dtype=tl.float32)
                for L in range(0, MAXLVL):
                    jv = tl.load(nbr_multi + base + L, mask=r_mask, other=-1)
                    jm = jv >= 0
                    jv64 = jv.to(tl.int64)
                    xL = tl.load(b + jv64[:, None] * (G * C) + pid_g * C + c_offsets[None, :],
                                 mask=jm[:, None] & c_mask[None, :], other=0.0)
                    xsum += xL.to(tl.float32)
                # variable overflow drain — gated per-(tile,tap): the whole
                # block (incl. the B1-wide ov_cnt/ov_off gathers) is skipped
                # when no row in this tile overflows on this tap.
                if FOLD:
                    if ovmax > 0:
                        ovc = tl.load(ov_cnt + rows * KV + v, mask=r_mask, other=0)
                        ovo = tl.load(ov_off + rows * KV + v, mask=r_mask, other=0).to(tl.int64)
                        for p in range(0, ovmax):
                            pj = tl.load(ov_j + ovo + p, mask=r_mask & (p < ovc), other=-1)
                            pjm = pj >= 0
                            pj64 = pj.to(tl.int64)
                            xp = tl.load(b + pj64[:, None] * (G * C) + pid_g * C + c_offsets[None, :],
                                         mask=pjm[:, None] & c_mask[None, :], other=0.0)
                            xsum += xp.to(tl.float32)
                w = tl.load(a + v * (G * C * M) + pid_g * (C * M)
                            + c_offsets[:, None] * M + m_offsets[None, :],
                            mask=c_mask[:, None] & m_mask[None, :], other=0.0)
                acc = tl.dot(xsum.to(w.dtype), w, acc, input_precision=INPUT_PRECISION)
    out_off = out + rows[:, None] * (G * M) + pid_g * M + m_offsets[None, :]
    tl.store(out_off, acc, mask=r_mask[:, None] & m_mask[None, :])


@triton.autotune(configs=_wgrad_autotune_configs(), key=["C", "M", "dtype_key"],
                 prune_configs_by={"early_config_prune": _wgrad_prune},
                 reset_to_zero=["gw_buf"])
@triton.jit
def _seg_fused_wgrad_kernel(
    x, go, bucket_inputs, bucket_i, ov_off, ov_cnt, ov_j, tap_ovmax,
    gw_buf, seg_offs,
    M, C, G, KTOTAL, dtype_key,
    INPUT_PRECISION: tl.constexpr, MAXLVL: tl.constexpr, FOLD: tl.constexpr,
    REDUCE_MODE: tl.constexpr, SPLITK: tl.constexpr,
    BL: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
):
    """Bucket-major fused gather-sum weight-grad. The grad of ``W[k]`` is
    ``Σ_buckets (Σ_j feat[j])^T @ grad_out[i]`` summed over the buckets of
    tap k. Each program contracts a chunk of one tap's bucket-major segment
    in-register — for every BL block of buckets it sums up to MAXLVL inputs
    per bucket (``xsum``) and accumulates the outer product against the
    bucket's grad-out row.

    The cross-program reduction over the split-K chunks is the
    ``REDUCE_MODE`` constexpr (chosen by the autotuner per (C, dtype)):

      0  PARTIALS — each (split, tile) writes its OWN slot of
         ``gw_buf[SPLITK, K, G, C, M]``; a host-side ``.sum(0)`` reduces.
      1  ATOMIC   — split-K, but every (split, tile) atomic_add's into ONE
         ``gw_buf[1, K, G, C, M]`` fp32 buffer; the host just casts slot 0.
      2  DIRECT   — SPLITK==1: one program contracts the tap's WHOLE segment
         and plain-stores slot 0 (one writer per element, no atomics).

    ``gw_buf`` is always the same [SPLITK_BUF, K, G, C, M] tensor (cap-sized
    on the host); modes 1/2 use only slot 0. ``dtype_key`` is an
    autotune-key constexpr only."""
    num_pid_c = tl.cdiv(C, BC); num_pid_m = tl.cdiv(M, BM)
    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m; pid = pid // num_pid_m
    pid_c = pid % num_pid_c; pid = pid // num_pid_c
    pid_g = pid % G; pid = pid // G
    pid_sk = pid % SPLITK; k_offset = pid // SPLITK
    seg_start = tl.load(seg_offs + k_offset).to(tl.int32)
    seg_end = tl.load(seg_offs + k_offset + 1).to(tl.int32)
    seglen = seg_end - seg_start
    sk_start = seg_start + (seglen * pid_sk) // SPLITK
    sk_end = seg_start + (seglen * (pid_sk + 1)) // SPLITK
    c_offsets = pid_c * BC + tl.arange(0, BC); c_mask = c_offsets < C
    m_offsets = pid_m * BM + tl.arange(0, BM); m_mask = m_offsets < M
    # per-tap overflow bound — ONE scalar load; the BL-wide CSR gathers in
    # the drain are hoisted under ovmax>0 so taps with no overflow bucket
    # pay nothing (the common case).
    ovmax = 0
    if FOLD:
        ovmax = tl.load(tap_ovmax + k_offset)
    acc = tl.zeros((BC, BM), dtype=tl.float32)
    for l0 in range(sk_start, sk_end, BL):
        l_offsets = l0 + tl.arange(0, BL)
        l_mask = l_offsets < sk_end
        oi = tl.load(bucket_i + l_offsets, mask=l_mask, other=0).to(tl.int64)
        xsum = tl.zeros((BL, BC), dtype=tl.float32)
        base = l_offsets * MAXLVL
        for L in range(0, MAXLVL):
            bj = tl.load(bucket_inputs + base + L, mask=l_mask, other=-1)
            jm = bj >= 0
            bj64 = bj.to(tl.int64)
            xb = tl.load(x + bj64[:, None] * (G * C) + pid_g * C + c_offsets[None, :],
                         mask=jm[:, None] & c_mask[None, :], other=0.0)
            xsum += xb.to(tl.float32)
        # folded overflow drain — gated per-tap; the whole block (incl. the
        # BL-wide ov_cnt/ov_off gathers) is skipped when no bucket in this
        # tap overflows. ov_* are keyed by the post-ord_k bucket position.
        if FOLD:
            if ovmax > 0:
                ovc = tl.load(ov_cnt + l_offsets, mask=l_mask, other=0)
                ovo = tl.load(ov_off + l_offsets, mask=l_mask, other=0).to(tl.int64)
                for p in range(0, ovmax):
                    pj = tl.load(ov_j + ovo + p, mask=l_mask & (p < ovc), other=-1)
                    pjm = pj >= 0
                    pj64 = pj.to(tl.int64)
                    xp = tl.load(x + pj64[:, None] * (G * C) + pid_g * C + c_offsets[None, :],
                                 mask=pjm[:, None] & c_mask[None, :], other=0.0)
                    xsum += xp.to(tl.float32)
        gb = tl.load(go + oi[:, None] * (G * M) + pid_g * M + m_offsets[None, :],
                     mask=l_mask[:, None] & m_mask[None, :], other=0.0)
        acc += tl.dot(tl.trans(xsum.to(gb.dtype)), gb, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
    if REDUCE_MODE == 0:
        slot = pid_sk
    else:
        slot = 0
    out_base = (slot.to(tl.int64) * (KTOTAL * G * C * M)
                + k_offset.to(tl.int64) * (G * C * M) + pid_g * (C * M))
    ptr = gw_buf + out_base + c_offsets[:, None] * M + m_offsets[None, :]
    tile_mask = c_mask[:, None] & m_mask[None, :]
    if REDUCE_MODE == 1:
        tl.atomic_add(ptr, acc, mask=tile_mask)
    else:
        tl.store(ptr, acc, mask=tile_mask)


# ── rulebook builder ──────────────────────────────────────────────────────────


# Hard cap on the per-tile/per-bucket overflow drain length (ranks beyond
# MAXLVL). Real ScanNet deep-multiplicity tails carry <=4 extra inputs at
# MAXLVL=4; anything past this cap (effectively never) spills to a tiny flat
# residual so the drain stays bounded.
_OVERFLOW_HARD_CAP = 16


def _ik_sort(i, j, k):
    """Shared (i, k)-bucket sort prefix for ``_build_seg`` and ``_build_wgrad``
    — both bucket by the same key ``i*K + k``, so the argsort + segment/rank
    decomposition is paid once when forward + grad_weight share the same
    triplets (the eager forward stashes it for the lazy wgrad build)."""
    dev = i.device
    T = i.numel()
    key = i.long() * K + k.long()
    order = torch.argsort(key, stable=True)
    ks_ = key[order]
    first = torch.ones(T, dtype=torch.bool, device=dev)
    first[1:] = ks_[1:] != ks_[:-1]
    start_pos = torch.nonzero(first, as_tuple=True)[0]
    seg_id = torch.cumsum(first.long(), 0) - 1
    rank = torch.arange(T, device=dev) - start_pos[seg_id]
    return dict(order=order, ks=ks_, first=first, seg_id=seg_id,
                start_pos=start_pos, rank=rank, j_sorted=j[order])


def _build_seg(i, j, k, N, maxlvl, fold=True, pre=None):
    """Bucket the triplets by (output point i, tap k) and pack up to
    ``maxlvl`` inputs per bucket into ``nbr_multi[N, K, maxlvl]`` (-1
    sentinel for empty slots). ``pre`` reuses a precomputed ``_ik_sort``.

    The deep-multiplicity tail (rank >= maxlvl) is handled one of two ways:

    - ``fold=True`` (default): the overflow inputs (up to ``maxlvl +
      _OVERFLOW_HARD_CAP``) are packed into a per-(output,tap) CSR
      (``ov_off`` / ``ov_cnt`` into ``ov_j``) drained INSIDE the main
      kernel, deleting the separate launch-bound flat residual pass. Only
      ranks beyond the cap (aimed empty) fall to a tiny residual.
    - ``fold=False``: all rank >= maxlvl inputs go to a flat residual
      ``TigIndex`` run as a per-triplet pass on top (legacy two-pass).

    Returns (nbr_multi, effective_maxlvl, ov_off, ov_cnt, ov_j,
    residual_index). Powers the forward (i=output) and, on the transposed
    triplets, grad_input (i becomes the input side)."""
    dev = i.device
    T = i.numel()
    if pre is None:
        pre = _ik_sort(i, j, k)
    order = pre["order"]; ks_ = pre["ks"]; first = pre["first"]
    seg_id = pre["seg_id"]; rank = pre["rank"]; j_sorted = pre["j_sorted"]
    maxmult = int(rank.max()) + 1
    ml = maxmult if maxlvl is None else min(maxlvl, maxmult)
    keep = rank < ml
    nm = torch.full((N * K, ml), -1, device=dev, dtype=torch.int32)
    nm[ks_[keep], rank[keep]] = j_sorted[keep].int()
    ov_off = torch.zeros(N * K, device=dev, dtype=torch.int32)
    ov_cnt = torch.zeros(N * K, device=dev, dtype=torch.int32)
    ov_j = torch.zeros(0, device=dev, dtype=torch.int32)
    if fold:
        cap = ml + _OVERFLOW_HARD_CAP
        ov_band = (rank >= ml) & (rank < cap)
        if ov_band.any():
            ob_key = ks_[ov_band]
            ob_rank = rank[ov_band]
            ob_j = j_sorted[ov_band].int()
            # bucket-major, rank-minor -> each bucket's overflow is contiguous.
            sec = torch.argsort(ob_key * cap + ob_rank, stable=True)
            ob_key = ob_key[sec]; ob_j = ob_j[sec]
            ov_j = ob_j.contiguous()
            cnt = torch.bincount(ob_key, minlength=N * K).int()
            ov_cnt = cnt
            ov_off[1:] = torch.cumsum(cnt, 0)[:-1].int()
        res = rank >= cap
    else:
        res = ~keep
    res_idx = None
    if res.any():
        ri, rj, rk = i[order][res], j[order][res], k[order][res]
        s = torch.argsort(rk, stable=True)
        res_idx = TigIndex(ri[s].int(), rj[s].int(), rk[s].int(), N, K,
                           build_hybrid=False, assume_sorted=True)
    return (nm.view(N, K, ml).contiguous(), ml,
            ov_off.contiguous(), ov_cnt.contiguous(), ov_j, res_idx)


def _overflow_clustered_rows(idx, ov_cnt_NK, b1):
    """Re-sort the hybrid level-0 rows so overflow-bearing rows cluster into
    a contiguous SUFFIX (their own row-tiles), leaving the clean-row tiles
    with ``tile_ovmax == 0`` so the folded drain is fully skipped there.

    Primary key = has-any-overflow (clean rows first), secondary = the
    incoming gray-code order (preserved among clean rows for the level-0
    locality the fused kernel relies on). Returns a folded-private
    (rows_sorted, tilemask, tile_ovmax). The level-0 math is order-invariant
    (per-row gather-sum), so reordering only changes which program owns a
    row, never the result."""
    dev = idx.rows_sorted.device
    rows = idx.rows_sorted.long()
    mbits = idx._mbits_sorted
    has_ov = (ov_cnt_NK.view(-1, K)[rows].sum(1) > 0).long()
    perm = torch.argsort(has_ov, stable=True)
    rows_s = rows[perm].int()
    mbits_s = mbits[perm]
    nrows = rows_s.numel()
    num_pid_n = -(-nrows // b1)
    pad = num_pid_n * b1 - nrows
    mb = mbits_s
    if pad:
        mb = torch.cat([mb, torch.zeros(pad, dtype=mb.dtype, device=dev)])
    mb = mb.view(num_pid_n, b1)
    tm = torch.zeros(num_pid_n, dtype=torch.int64, device=dev)
    for v in range(K):
        tm |= ((mb >> v) & 1).amax(1) << v
    tm = tm.int()
    cnt_rows = ov_cnt_NK.view(-1, K)[rows_s.long()]
    if pad:
        cnt_rows = torch.cat([cnt_rows,
                              torch.zeros(pad, K, dtype=cnt_rows.dtype, device=dev)], 0)
    tom = cnt_rows.view(num_pid_n, b1, K).amax(1).contiguous().int()
    return rows_s.contiguous(), tm, tom


def _maybe_reorder_hybrid_rows(idx):
    """Default-off probe for row locality vs tap-mask clustering.

    ``TigIndex`` builds ``rows_sorted`` in gray-code mask order. That reduces
    per-tile tap divergence, but it can separate spatially adjacent rows and
    hurt feature-load locality. ``PNT_SEG_ROW_ORDER=row`` restores row-id
    order for fused-conv probes without changing the default route.
    """
    if _SEG_ROW_ORDER_ENV in ("gray", ""):
        return
    if _SEG_ROW_ORDER_ENV != "row":
        raise ValueError("PNT_SEG_ROW_ORDER must be 'gray' or 'row'")
    perm = torch.argsort(idx.rows_sorted.long(), stable=True)
    idx.rows_sorted = idx.rows_sorted[perm].contiguous()
    idx._mbits_sorted = idx._mbits_sorted[perm].contiguous()
    idx._tilemask_cache = {}


def _build_wgrad(i, j, k, N, maxlvl, fold=True, pre=None):
    """Bucket-major rulebook for the weight-grad: pack each (i, k) bucket's
    inputs into ``bucket_inputs[num_buckets, maxlvl]`` ordered by tap, with
    ``bucket_i`` the output point each bucket scatters to and ``seg_offs``
    the per-tap segment boundaries (the split-K kernel walks one tap's
    segment).

    The deep-multiplicity tail (rank >= maxlvl) is handled like ``_build_seg``:

    - ``fold=True`` (default): the overflow inputs (up to ``maxlvl +
      _OVERFLOW_HARD_CAP``) are packed into a per-bucket CSR (``ov_off`` /
      ``ov_cnt`` into ``ov_j``, keyed by the post-ord_k bucket position the
      kernel iterates) drained INSIDE the wgrad kernel, gated by a per-tap
      ``tap_ovmax`` so taps with no overflow bucket skip the drain.
    - ``fold=False``: rank >= maxlvl spills to a flat residual ``TigIndex``.

    Returns (bucket_inputs, bucket_i, seg_offs, num_buckets, ov_off, ov_cnt,
    ov_j, tap_ovmax, residual_index)."""
    dev = i.device
    T = i.numel()
    if pre is None:
        pre = _ik_sort(i, j, k)
    order = pre["order"]; ks_ = pre["ks"]; first = pre["first"]
    seg_id = pre["seg_id"]; rank = pre["rank"]; j_sorted = pre["j_sorted"]
    bucket_keys = ks_[first]
    nb = bucket_keys.numel()
    keep = rank < maxlvl
    bucket_inputs = torch.full((nb, maxlvl), -1, device=dev, dtype=torch.int32)
    bucket_inputs[seg_id[keep], rank[keep]] = j_sorted[keep].int()
    bucket_i = bucket_keys // K
    bucket_kk = bucket_keys % K
    # overflow band per bucket: maxlvl <= rank < maxlvl + cap. CSR keyed by
    # the ORIGINAL bucket id (seg_id), reordered to ord_k below.
    ov_cnt = torch.zeros(nb, device=dev, dtype=torch.int32)
    ov_off = torch.zeros(nb, device=dev, dtype=torch.int32)
    ov_j = torch.zeros(0, device=dev, dtype=torch.int32)
    cap = maxlvl + _OVERFLOW_HARD_CAP
    if fold:
        ov_band = (rank >= maxlvl) & (rank < cap)
        if ov_band.any():
            ob_bucket = seg_id[ov_band]
            ob_rank = rank[ov_band]
            ob_j = j_sorted[ov_band].int()
            sec = torch.argsort(ob_bucket.long() * cap + ob_rank, stable=True)
            ob_bucket = ob_bucket[sec]; ob_j = ob_j[sec]
            ov_j = ob_j.contiguous()
            cnt = torch.bincount(ob_bucket, minlength=nb).int()
            ov_cnt = cnt
            ov_off[1:] = torch.cumsum(cnt, 0)[:-1].int()
    # reorder buckets (+ CSR) to be tap-major so seg_offs gives contiguous
    # per-tap segments (the split-K kernel indexes one tap's segment at a time).
    ord_k = torch.argsort(bucket_kk, stable=True)
    bucket_inputs = bucket_inputs[ord_k].contiguous()
    bucket_i_ks = bucket_i[ord_k].int().contiguous()
    ov_cnt = ov_cnt[ord_k].contiguous()
    # ov_off rows move to position ord_k, but each value is an ABSOLUTE
    # offset into ov_j (which stays in original-bucket order) — so the row
    # permutation is correct without touching the offsets themselves.
    ov_off = ov_off[ord_k].contiguous()
    bucket_kk_s = bucket_kk[ord_k].long()
    seg_offs = kernel_offset_segments(bucket_kk_s, K)
    # per-tap overflow bound: max ov_cnt over each tap's (contiguous) buckets.
    tap_ovmax = torch.zeros(K, device=dev, dtype=torch.int32)
    if fold and int(ov_cnt.max()) > 0:
        tap_ovmax.scatter_reduce_(0, bucket_kk_s, ov_cnt, reduce="amax",
                                  include_self=True)
    if fold:
        res = rank >= cap
    else:
        res = ~keep
    res_idx = None
    if res.any():
        ri, rj, rk = i[order][res], j[order][res], k[order][res]
        s = torch.argsort(rk, stable=True)
        res_idx = TigIndex(ri[s].int(), rj[s].int(), rk[s].int(), N, K,
                           build_hybrid=False, assume_sorted=True)
    return (bucket_inputs, bucket_i_ks, seg_offs, nb,
            ov_off, ov_cnt, ov_j, tap_ovmax, res_idx)


def _rows_from_nbr(nbr_NKml: Tensor):
    """Derive (rows_sorted, mbits_sorted) — the gray-code-ordered non-empty
    output rows + their per-row tap bitmask — directly from the seg builder's
    ``nbr_multi`` (slot-0 occupancy == bucket non-empty, identical to the
    TigIndex ``nbr0 >= 0`` test). Replaces a full ``build_hybrid`` TigIndex:
    same ``rows_sorted`` / ``tilemask`` the masked kernel reads, without the
    unused ``nbr0`` / residual / T-sized argsort the fused op never touches."""
    dev = nbr_NKml.device
    mask = nbr_NKml[:, :, 0] >= 0                       # (N, K) bucket occupancy
    rows_any = mask.any(1).nonzero(as_tuple=True)[0]
    mbits = (mask[rows_any].long() << torch.arange(K, device=dev)).sum(1)
    gray = mbits ^ (mbits >> 1)
    perm = torch.argsort(gray)
    return rows_any[perm].int(), mbits[perm]


class _HybRows:
    """Minimal hybrid-row carrier the fused masked kernel needs: gray-sorted
    non-empty rows + their tap bitmask, with ``tilemask`` recomputed on demand.
    Stands in for a full TigIndex (the fused passes only ever read
    ``rows_sorted`` / ``_mbits_sorted`` / ``tilemask`` — never ``nbr0`` /
    ``res_*`` / the flat arrays)."""

    def __init__(self, rows_sorted: Tensor, mbits_sorted: Tensor):
        self.rows_sorted = rows_sorted
        self._mbits_sorted = mbits_sorted
        self._tilemask_cache: dict = {}

    def tilemask(self, b1: int) -> Tensor:
        tm = self._tilemask_cache.get(b1)
        if tm is None:
            mb = self._mbits_sorted
            pad = (-mb.numel()) % b1
            if pad:
                mb = torch.cat([mb, torch.zeros(pad, dtype=mb.dtype,
                                                device=mb.device)])
            mb = mb.view(-1, b1)
            tm = torch.zeros(mb.size(0), dtype=torch.int64, device=mb.device)
            for v in range(K):
                tm |= ((mb >> v) & 1).amax(1) << v
            tm = tm.to(torch.int32)
            self._tilemask_cache[b1] = tm
        return tm


class FusedGatherSumRulebook:
    """All index structures the three fused passes need, built from a triplet
    rulebook ``(i, j, k)`` + output count ``N``.

    Forward and grad_weight share the (output point i, tap k) bucketing;
    grad_input reuses the forward gather-sum on the TRANSPOSED triplets
    (output<->input swapped, so the bucket key is (j, k)). The forward
    structures are built eagerly; the grad_input (transposed seg) and
    grad_weight structures are built lazily on the first backward, so
    inference (``no_grad``) never pays for them. ``rows_sorted`` / ``tilemask``
    are derived from the seg ``nbr_multi`` rather than a separate full
    TigIndex. ``maxlvl_fwd`` / ``maxlvl_gi`` set the forward / grad_input slot
    depth; ``maxlvl_wgrad`` the weight-grad bucket depth (None = full
    multiplicity, no residual)."""

    def __init__(self, i: Tensor, j: Tensor, k: Tensor, N: int,
                 maxlvl_fwd: int = _MAXLVL_FWD,
                 maxlvl_gi: int = _MAXLVL_GI,
                 maxlvl_wgrad: int = _MAXLVL_WGRAD,
                 fold: bool = True):
        self.N = int(N)
        self.fold = fold
        i = i.long()
        j = j.long()
        k = k.long()
        # Kept for the lazy backward build (grad_input transposed seg + wgrad).
        self._i, self._j, self._k = i, j, k
        self._maxlvl_gi = maxlvl_gi
        self._maxlvl_wgrad = maxlvl_wgrad
        # forward: bucket by (output i, tap k). rows_sorted / tilemask are
        # derived from the seg nbr_multi (no separate full TigIndex). The
        # (i, k) sort is stashed — grad_weight buckets by the same key, so the
        # lazy backward reuses it instead of re-sorting.
        self._pre_fwd = _ik_sort(i, j, k)
        (self.fwd_nbr, self.fwd_maxlvl, self.fwd_ov_off, self.fwd_ov_cnt,
         self.fwd_ov_j, self.fwd_res) = _build_seg(i, j, k, N, maxlvl_fwd,
                                                   fold=fold, pre=self._pre_fwd)
        self.idx_hyb = _HybRows(*_rows_from_nbr(self.fwd_nbr))
        _maybe_reorder_hybrid_rows(self.idx_hyb)
        # folded forward / grad_input use overflow-clustered, folded-private
        # rows (clean tiles get ovmax==0 -> drain skipped). The layout depends
        # on B1, so build it lazily for the active width policy.
        self._fwd_fold_layouts = {}
        self._gi_fold_layouts = {}
        self._bwd_built = False

    def _ensure_backward(self):
        """Build the grad_input transposed seg + the grad_weight structures on
        the first backward. A no-op once built; never reached under no_grad."""
        if self._bwd_built:
            return
        i, j, k, N, fold = self._i, self._j, self._k, self.N, self.fold
        # grad_weight shares the forward (i, k) bucketing — reuse the sort.
        self.wgrad = _build_wgrad(i, j, k, N, self._maxlvl_wgrad, fold=fold,
                                  pre=self._pre_fwd)
        self.wgrad_maxlvl = self._maxlvl_wgrad
        # grad_input: transposed rulebook (bucket key (j, k), gather i).
        (self.gi_nbr, self.gi_maxlvl, self.gi_ov_off, self.gi_ov_cnt,
         self.gi_ov_j, self.gi_res) = _build_seg(j, i, k, N, self._maxlvl_gi, fold=fold)
        self.idxT_hyb = _HybRows(*_rows_from_nbr(self.gi_nbr))
        _maybe_reorder_hybrid_rows(self.idxT_hyb)
        self._bwd_built = True

    def _fwd_layout(self, b1: int):
        if self.fold:
            if b1 not in self._fwd_fold_layouts:
                self._fwd_fold_layouts[b1] = _overflow_clustered_rows(
                    self.idx_hyb, self.fwd_ov_cnt, b1)
            return self._fwd_fold_layouts[b1]
        return (self.idx_hyb.rows_sorted,
                self.idx_hyb.tilemask(b1),
                self.fwd_ov_cnt)

    def _gi_layout(self, b1: int):
        if self.fold:
            if b1 not in self._gi_fold_layouts:
                self._gi_fold_layouts[b1] = _overflow_clustered_rows(
                    self.idxT_hyb, self.gi_ov_cnt, b1)
            return self._gi_fold_layouts[b1]
        return (self.idxT_hyb.rows_sorted,
                self.idxT_hyb.tilemask(b1),
                self.gi_ov_cnt)


# ── ops ───────────────────────────────────────────────────────────────────────


def _seg_forward(weight, feat, rows_sorted, tilemask, nbr_multi, maxlvl, N, C,
                 dtype_key, input_precision, b1,
                 ov_off, ov_cnt, ov_j, tile_ovmax, fold):
    """Launch the autotuned fused masked forward (level-0 compacted rows)
    for the weight ``(K, C, M=C)`` contiguous layout. Returns (N, M) in
    feat.dtype. The grid is a function of the autotuned ``BM`` and the
    width-routed active ``B1``.

    When ``fold`` is set, ``rows_sorted`` / ``tilemask`` are the
    overflow-clustered folded-private layout and the kernel drains the
    deep-multiplicity tail in-place via the ``ov_*`` CSR (no separate flat
    residual launch). When not set, the legacy two-pass path passes the
    plain hybrid rows and empty overflow buffers, and the caller adds the
    flat residual on top."""
    M = C
    G = 1
    nrows = rows_sorted.numel()
    out = torch.zeros(N, G * M, device=feat.device, dtype=feat.dtype)
    grid = lambda META: (-(-nrows // META["B1"]) * G * triton.cdiv(M, META["BM"]),)
    _seg_masked_kernel[grid](
        nbr_multi.view(-1), ov_off, ov_cnt, ov_j, tile_ovmax,
        rows_sorted, feat, weight, out, tilemask,
        nrows, M, C, G, dtype_key, b1, INPUT_PRECISION=input_precision,
        KV=K, MAXLVL=maxlvl, FOLD=fold)
    return out


def fused_forward(weight: Tensor, feat: Tensor,
                  rulebook: FusedGatherSumRulebook,
                  fp32_ieee: bool = False) -> Tensor:
    """Forward ``out[i] += (Σ_{j in bucket(i,k)} feat[j]) @ W[k]``.

    weight: (K, C, M) with M == C (within-stage). feat: (N, C). Returns
    (N, M) in feat.dtype. Sums each (output, tap) bucket in-register and
    spends one matmul per bucket; the deep-multiplicity tail is a flat
    residual added on top. ``fp32_ieee`` forces exact fp32 (no tf32) for
    fp32 inputs."""
    if weight.dim() == 4:
        weight = weight.view(weight.shape[0], weight.shape[2], weight.shape[3])
    C = feat.size(1)
    dk = _dtype_key(feat.dtype, fp32_ieee)
    ip = _input_precision(feat.dtype, fp32_ieee)
    b1 = _seg_b1_for_width(C, dk)
    rows, tm, tom = rulebook._fwd_layout(b1)
    if rulebook.fold:
        tom = tom.view(-1)
    out = _seg_forward(weight, feat, rows, tm,
                       rulebook.fwd_nbr, rulebook.fwd_maxlvl, rulebook.N, C,
                       dk, ip, b1, rulebook.fwd_ov_off, rulebook.fwd_ov_cnt,
                       rulebook.fwd_ov_j, tom, rulebook.fold)
    if rulebook.fwd_res is not None:
        out = out + tig_forward(weight, feat, rulebook.fwd_res,
                                mode="flat", input_precision=ip).to(out.dtype)
    return out


def fused_grad_input(weight: Tensor, grad_out: Tensor,
                     rulebook: FusedGatherSumRulebook,
                     fp32_ieee: bool = False) -> Tensor:
    """grad_input ``grad_in[j] += (Σ_{i: (i,j,k)} grad_out[i]) @ W[k]^T``.

    Reuses the forward gather-sum on the transposed rulebook: swap
    input<->output (bucket key (j, k), gather i) and transpose the weight
    per tap. weight: (K, C, M) as stored; grad_out: (N, M). Returns (N, C)
    in grad_out.dtype."""
    rulebook._ensure_backward()
    if weight.dim() == 4:
        weight = weight.view(weight.shape[0], weight.shape[2], weight.shape[3])
    wT = weight.transpose(1, 2).contiguous()
    M = grad_out.size(1)
    dk = _dtype_key(grad_out.dtype, fp32_ieee)
    ip = _input_precision(grad_out.dtype, fp32_ieee)
    b1 = _seg_b1_for_width(M, dk)
    rows, tm, tom = rulebook._gi_layout(b1)
    if rulebook.fold:
        tom = tom.view(-1)
    out = _seg_forward(wT, grad_out, rows, tm,
                       rulebook.gi_nbr, rulebook.gi_maxlvl, rulebook.N, M,
                       dk, ip, b1, rulebook.gi_ov_off, rulebook.gi_ov_cnt,
                       rulebook.gi_ov_j, tom, rulebook.fold)
    if rulebook.gi_res is not None:
        out = out + tig_forward(wT, grad_out, rulebook.gi_res,
                                mode="flat", input_precision=ip).to(out.dtype)
    return out


def fused_grad_weight(feat: Tensor, grad_out: Tensor,
                      rulebook: FusedGatherSumRulebook, weight_shape,
                      fp32_ieee: bool = False) -> Tensor:
    """grad_weight ``grad_W[k] += Σ_buckets (Σ_j feat[j])^T @ grad_out[i]``.

    Bucket-major reduction; the cross-split reduce (host-summed partials /
    in-kernel atomic / direct no-split-K store) is selected by the
    autotuner per (C, dtype). weight_shape: (K, C, M). Returns grad_W of
    that shape in feat.dtype. The deep-multiplicity tail is a per-triplet
    residual added on top."""
    rulebook._ensure_backward()
    (bucket_inputs, bucket_i, seg_offs, nb,
     ov_off, ov_cnt, ov_j, tap_ovmax, res_idx) = rulebook.wgrad
    C = feat.size(1)
    M = grad_out.size(1)
    G = 1
    dk = _dtype_key(feat.dtype, fp32_ieee)
    ip = _input_precision(feat.dtype, fp32_ieee)
    # The partials reduce needs up to SPLITK slots; the buffer is cap-sized
    # to the width's pruned SPLITK ceiling (atomic/direct use only slot 0).
    # The autotuned config (<= cap) decides the actual reduction below.
    splitk_buf = _wgrad_splitk_cap(C)
    gw_buf = torch.zeros(splitk_buf, K, G, C, M, device=feat.device,
                         dtype=torch.float32)
    # grid spans (tap, group, c-tile, m-tile, split). Direct mode pins
    # SPLITK=1, so this single META-driven grid covers all three reduces.
    grid = lambda META: (
        K * G * triton.cdiv(C, META["BC"]) * triton.cdiv(M, META["BM"]) * META["SPLITK"],)
    _seg_fused_wgrad_kernel[grid](
        feat, grad_out, bucket_inputs.view(-1), bucket_i,
        ov_off, ov_cnt, ov_j, tap_ovmax, gw_buf, seg_offs,
        M, C, G, K, dk, INPUT_PRECISION=ip, MAXLVL=rulebook.wgrad_maxlvl,
        FOLD=rulebook.fold)
    # the chosen config tells the host how to reduce: partials sum the used
    # slots, atomic/direct already accumulated into slot 0.
    best = _seg_fused_wgrad_kernel.best_config
    if best.kwargs["REDUCE_MODE"] == 0:
        sk = best.kwargs["SPLITK"]
        gw = gw_buf[:sk].sum(0).to(feat.dtype).view(K, C, M)
    else:
        gw = gw_buf[0].to(feat.dtype).view(K, C, M)
    if res_idx is not None:
        gw = gw + tig_grad_weight(feat, grad_out, res_idx, (K, C, M),
                                  input_precision=ip)
    if len(weight_shape) == 4:
        return gw.view(weight_shape)
    return gw.view(K, C, M)


# ── autograd glue ─────────────────────────────────────────────────────────────


class FusedPointConv3d(torch.autograd.Function):
    """Autograd wiring of the three fused passes. Forward stores feat,
    weight and the prebuilt rulebook; backward dispatches grad_input and
    grad_weight to the matching fused kernels (the transposed gather-sum
    and the bucket-major reduction). ``fp32_ieee`` (default False) forces
    exact fp32 instead of the tf32 tensor-core path."""

    @staticmethod
    def forward(ctx, feat, weight, rulebook, fp32_ieee=False):
        out = fused_forward(weight, feat, rulebook, fp32_ieee=fp32_ieee)
        ctx.save_for_backward(feat, weight)
        ctx.rulebook = rulebook
        ctx.fp32_ieee = fp32_ieee
        return out

    @staticmethod
    def backward(ctx, grad_out):
        feat, weight = ctx.saved_tensors
        rulebook = ctx.rulebook
        fp32_ieee = ctx.fp32_ieee
        grad_out = grad_out.contiguous()
        grad_feat = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_feat = fused_grad_input(weight, grad_out, rulebook,
                                         fp32_ieee=fp32_ieee)
        if ctx.needs_input_grad[1]:
            grad_weight = fused_grad_weight(feat, grad_out, rulebook,
                                            tuple(weight.shape),
                                            fp32_ieee=fp32_ieee)
        # feat, weight, rulebook, fp32_ieee
        return grad_feat, grad_weight, None, None


def fused_gather_sum_conv3d(feat: Tensor, weight: Tensor,
                                  triplets_i: Tensor, triplets_j: Tensor,
                                  triplets_k: Tensor, N: int,
                                  fp32_ieee: bool = False) -> Tensor:
    """Thin entry point: build (or reuse a cached) fused rulebook from triplets
    and run the autograd Function. weight: (K, C, M) with M == C (within-stage,
    kernel_size=3, K=27). feat: (N, C). triplets ``(i, j, k)`` are the
    radius-neighborhood rulebook (output i, input j, tap k). ``fp32_ieee`` forces
    exact fp32 (no tf32). Returns (N, M).

    A backbone reuses the SAME (i, j, k) across all sibling convs at a stage
    (e.g. a ResNet block's conv1 + conv2 share the metadata's triplets), so the
    rulebook — a pure function of (i, j, k, N) — is cached ON the ``i`` tensor
    and reused by sibling convolutions that share the same triplets. The cache
    lives exactly as long as the triplet tensor (freed with it at the end of the
    forward pass), so it holds no memory across iterations. Without this,
    force_fused_gather_sum rebuilds the heavy rulebook per conv and the operator win
    is lost to redundant build cost."""
    C = feat.size(1)
    cached = getattr(triplets_i, "_pnt_fused_rulebook", None)
    if (cached is not None and cached[0] is triplets_j
            and cached[1] is triplets_k and cached[3] == C
            and cached[2].N == int(N)):
        rulebook = cached[2]
    else:
        ml = _maxlvl_for_width(C)
        fold = _fold_for_width(C)
        rulebook = FusedGatherSumRulebook(triplets_i, triplets_j, triplets_k, N,
                                     maxlvl_fwd=ml, maxlvl_gi=ml, fold=fold)
        try:  # plain index tensors accept attributes; tracing/compile may not
            triplets_i._pnt_fused_rulebook = (triplets_j, triplets_k, rulebook, C)
        except (AttributeError, RuntimeError):
            pass
    return FusedPointConv3d.apply(feat, weight, rulebook, fp32_ieee)
