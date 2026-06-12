"""TIG — Triplet Implicit GEMM: the third execution generation for
MVMR/VVOR (PT = per-triplet v1.0; FSG = full-segment grouped v1.1;
TIG = triplet implicit GEMM v1.2) for the GENERAL PointConv3d mvmr.

Computes ``out[i] += feat[j] @ W[k]`` over a triplet rulebook
``(i, j, k)`` with variable fan-in (the general, point-native operator —
float coords never enter here; the neighbor set is exactly what
``build_triplets`` produced). Two execution structures, sharing one
epilogue (fp32 global accumulation buffer + final cast):

- **flat**: one kernel over the k-sorted triplets (weight-stationary;
  the structure the production grouped kernel uses) with fp16/bf16
  operands fed straight to ``tl.dot`` (fp32 accumulator), a register
  C-loop (no per-C-chunk atomics), and a single fp32 ``atomic_add``
  per (chunk, M-tile).
- **hybrid**: level-0 sub-rulebook (first neighbor per (i, tap) — a
  binary-mask problem) as an output-stationary masked iGEMM over
  gray-sorted compacted rows with tile-level tap skipping; the ragged
  residual (fan-in ≥ 2) goes through the flat kernel on top.

Memory contract: no T×C / im2col materialization; index structures are
O(T + N·27 bytes) like the existing rulebook; the only transient is the
fp32 accumulation buffer (N×M×4B, freed after the cast).

Config selection is bucketed on channel sizes only — NEVER on N or T.
Point-cloud N/T vary every iteration; any N/T-keyed config selection
(or autotune key) would re-trigger tuning on every fresh point cloud.

Earlier revisions were forward-only G==1. As of v1.2.0, ALL kernels
(flat, hybrid masked, wgrad, fused backward) carry the group axis natively
(block-diagonal, grid-axis G + per-group pointer offsets), and the
small-per-group cells route group-PACKED kernels (GP adjacent groups
per block-diagonal dot tile — dense x/out accesses, fewer dots vs
per-group tiles). fwd/grad_input pack GP=4 at Cg==Mg==8 (G%4==0),
GP=2 at Cg==Mg==8 (G even) and Cg==Mg==16 (G even); wgrad packs GP=2
at Cg==Mg==8 only (the wider shapes measured no wgrad win).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from ._seg_offs import kernel_offset_segments

__all__ = ["TigIndex", "tig_forward", "tig_grad_input",
           "tig_grad_weight", "tig_backward_fused", "tig_mvmr"]


# ── kernels ──────────────────────────────────────────────────────────────────


@triton.jit
def _tig_flat_kernel(
    a, b, b_idx, o_idx, o32, seg_offs,
    K_offsets, M, C, G,
    stride_wk, stride_wg, stride_wc, stride_wm,
    INPUT_PRECISION: tl.constexpr,
    L: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
):
    """Weight-stationary flat pass over k-sorted triplets. M and C are
    PER-GROUP widths; features/outputs are (N, G*C)/(N, G*M) flat; the
    grid carries a group axis (block-diagonal math, v1.2.0 G>1 support).

    Computes ``o32[o_idx[t]] += b[b_idx[t]] @ W[k(t)]`` where the logical
    weight ``W[k]`` is (C, M) addressed through explicit strides — so the
    SAME kernel serves the forward (gather=j, scatter=i, W as stored) and
    grad_input (gather=i, scatter=j, W transposed via swapped strides,
    zero-copy)."""
    num_pid_m = tl.cdiv(M, BM)
    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid_g = (pid // num_pid_m) % G
    pid_chunk = pid // (num_pid_m * G)

    # map pid_chunk -> (k segment, chunk within segment); K_offsets is
    # small (27 / 125), the scan is cheap and avoids materialising a
    # per-chunk offset table (pattern proven in the grouped kernel).
    k_offset = 0
    seg_start_k = tl.load(seg_offs + 0).to(tl.int32)
    seg_end_k = tl.load(seg_offs + 1).to(tl.int32)
    chunk_idx_within_k = 0
    running = 0
    for k_iter in tl.range(0, K_offsets):
        seg_s = tl.load(seg_offs + k_iter).to(tl.int32)
        seg_e = tl.load(seg_offs + k_iter + 1).to(tl.int32)
        chunks_here = (seg_e - seg_s + L - 1) // L
        if running <= pid_chunk and pid_chunk < running + chunks_here:
            k_offset = k_iter
            seg_start_k = seg_s
            seg_end_k = seg_e
            chunk_idx_within_k = pid_chunk - running
        running += chunks_here

    # upper-bound grid: programs past the true chunk total exit (lets the
    # host size the grid as cdiv(T, L) + K with NO device->host sync)
    if pid_chunk >= running:
        return

    l_offsets = seg_start_k + chunk_idx_within_k * L + tl.arange(0, L)
    l_mask = l_offsets < seg_end_k

    m_offsets = pid_m * BM + tl.arange(0, BM)
    m_mask = m_offsets < M

    bj = tl.load(b_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)
    oi = tl.load(o_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)

    acc = tl.zeros((L, BM), dtype=tl.float32)
    for c0 in range(0, C, BC):
        c_offsets = c0 + tl.arange(0, BC)
        c_mask = c_offsets < C
        x = tl.load(b + bj[:, None] * (G * C) + pid_g * C
                    + c_offsets[None, :],
                    mask=l_mask[:, None] & c_mask[None, :], other=0.0)
        w = tl.load(a + k_offset.to(tl.int64) * stride_wk
                    + pid_g * stride_wg
                    + c_offsets[:, None] * stride_wc
                    + m_offsets[None, :] * stride_wm,
                    mask=c_mask[:, None] & m_mask[None, :], other=0.0)
        acc = tl.dot(x, w, acc, input_precision=INPUT_PRECISION)

    tl.atomic_add(o32 + oi[:, None] * (G * M) + pid_g * M
                  + m_offsets[None, :], acc,
                  mask=l_mask[:, None] & m_mask[None, :])


@triton.jit
def _tig_flat_bisect_kernel(
    a, b, b_idx, o_idx, o32, seg_offs, cum_chunks,
    K_offsets, M, C, G,
    stride_wk, stride_wg, stride_wc, stride_wm,
    INPUT_PRECISION: tl.constexpr,
    L: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
):
    """``_tig_flat_kernel`` with the pid_chunk -> k map done by
    BINARY SEARCH over a precomputed cumulative-chunk table instead of the
    linear K-segment scan. At large tap counts (K = 8^3 = 512, a
    generative-stem shape) the linear scan runs 512 iterations in EVERY
    program and dominates the forward (~1 ms on Ada (sm_89) at N=165k);
    bisection is ~9 loads. Routed only at
    K > _BISECT_MIN_K — the production K=27 path keeps the scan kernel
    byte-unchanged. ``cum_chunks`` is (K+1,) int32, cum_chunks[k] = total
    chunks of segments < k (per this L); built once per (orientation, L)
    and cached on the TigIndex."""
    num_pid_m = tl.cdiv(M, BM)
    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid_g = (pid // num_pid_m) % G
    pid_chunk = pid // (num_pid_m * G)

    total = tl.load(cum_chunks + K_offsets).to(tl.int32)
    if pid_chunk >= total:
        return

    # largest k with cum_chunks[k] <= pid_chunk (duplicates from empty
    # segments converge past the empties — chunk ranges there are empty)
    lo = 0
    hi = K_offsets
    while hi - lo > 1:
        mid = (lo + hi) // 2
        cm = tl.load(cum_chunks + mid).to(tl.int32)
        if cm <= pid_chunk:
            lo = mid
        else:
            hi = mid
    k_offset = lo
    chunk_idx_within_k = pid_chunk - tl.load(cum_chunks + lo).to(tl.int32)
    seg_start_k = tl.load(seg_offs + lo).to(tl.int32)
    seg_end_k = tl.load(seg_offs + lo + 1).to(tl.int32)

    l_offsets = seg_start_k + chunk_idx_within_k * L + tl.arange(0, L)
    l_mask = l_offsets < seg_end_k

    m_offsets = pid_m * BM + tl.arange(0, BM)
    m_mask = m_offsets < M

    bj = tl.load(b_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)
    oi = tl.load(o_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)

    acc = tl.zeros((L, BM), dtype=tl.float32)
    for c0 in range(0, C, BC):
        c_offsets = c0 + tl.arange(0, BC)
        c_mask = c_offsets < C
        x = tl.load(b + bj[:, None] * (G * C) + pid_g * C
                    + c_offsets[None, :],
                    mask=l_mask[:, None] & c_mask[None, :], other=0.0)
        w = tl.load(a + k_offset.to(tl.int64) * stride_wk
                    + pid_g * stride_wg
                    + c_offsets[:, None] * stride_wc
                    + m_offsets[None, :] * stride_wm,
                    mask=c_mask[:, None] & m_mask[None, :], other=0.0)
        acc = tl.dot(x, w, acc, input_precision=INPUT_PRECISION)

    tl.atomic_add(o32 + oi[:, None] * (G * M) + pid_g * M
                  + m_offsets[None, :], acc,
                  mask=l_mask[:, None] & m_mask[None, :])


@triton.jit
def _tig_flat_fi1_kernel(
    a, b, b_idx, o_idx, o, seg_offs, cum_chunks,
    K_offsets, M, C, G,
    stride_wk, stride_wg, stride_wc, stride_wm,
    INPUT_PRECISION: tl.constexpr,
    L: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
):
    """The fan-in-1 (exactly-once scatter) flat kernel — plain
    ``tl.store`` into a NATIVE-dtype uninitialized output instead of
    fp32-zeros + atomicAdd + cast (3 output passes -> 1). Legal ONLY when
    every output row receives exactly one triplet (the disjoint builders'
    contract: deconv forward, partition grad_input) — routed via the
    TigIndex ``exact_cover_*`` caller flags, never inferred. Chunk map by
    bisection (any K)."""
    num_pid_m = tl.cdiv(M, BM)
    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid_g = (pid // num_pid_m) % G
    pid_chunk = pid // (num_pid_m * G)

    total = tl.load(cum_chunks + K_offsets).to(tl.int32)
    if pid_chunk >= total:
        return
    lo = 0
    hi = K_offsets
    while hi - lo > 1:
        mid = (lo + hi) // 2
        cm = tl.load(cum_chunks + mid).to(tl.int32)
        if cm <= pid_chunk:
            lo = mid
        else:
            hi = mid
    k_offset = lo
    chunk_idx_within_k = pid_chunk - tl.load(cum_chunks + lo).to(tl.int32)
    seg_start_k = tl.load(seg_offs + lo).to(tl.int32)
    seg_end_k = tl.load(seg_offs + lo + 1).to(tl.int32)

    l_offsets = seg_start_k + chunk_idx_within_k * L + tl.arange(0, L)
    l_mask = l_offsets < seg_end_k

    m_offsets = pid_m * BM + tl.arange(0, BM)
    m_mask = m_offsets < M

    bj = tl.load(b_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)
    oi = tl.load(o_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)

    acc = tl.zeros((L, BM), dtype=tl.float32)
    for c0 in range(0, C, BC):
        c_offsets = c0 + tl.arange(0, BC)
        c_mask = c_offsets < C
        x = tl.load(b + bj[:, None] * (G * C) + pid_g * C
                    + c_offsets[None, :],
                    mask=l_mask[:, None] & c_mask[None, :], other=0.0)
        w = tl.load(a + k_offset.to(tl.int64) * stride_wk
                    + pid_g * stride_wg
                    + c_offsets[:, None] * stride_wc
                    + m_offsets[None, :] * stride_wm,
                    mask=c_mask[:, None] & m_mask[None, :], other=0.0)
        acc = tl.dot(x, w, acc, input_precision=INPUT_PRECISION)

    tl.store(o + oi[:, None] * (G * M) + pid_g * M + m_offsets[None, :],
             acc.to(o.dtype.element_ty),
             mask=l_mask[:, None] & m_mask[None, :])


@triton.jit
def _tig_flat_packed_kernel(
    a, b, b_idx, o_idx, o32, seg_offs,
    K_offsets, G,
    stride_wk, stride_wg, stride_wc, stride_wm,
    INPUT_PRECISION: tl.constexpr,
    CG: tl.constexpr, MG: tl.constexpr, GP: tl.constexpr,
    L: tl.constexpr,
):
    """Group-packed flat pass for tiny per-group channels (v1.2.0
    squeeze): each program covers GP adjacent groups in ONE
    (L, GP*CG) x (GP*CG, GP*MG) tl.dot with a block-diagonal weight-tile
    mask (zeros off-diagonal). vs the per-group 16-wide tile at CG=8 this
    makes the x/out accesses fully dense and halves the dot count (flop
    waste 4x -> 2x): real c64 G=8 fp16 fwd 0.93 -> 0.61 ms. Requires
    GP*CG and GP*MG to be tl.dot-legal powers of two; full per-group C/M
    in one tile (no C loop), so it serves CG, MG <= 16 only. The same
    stride-swap trick as ``_tig_flat_kernel`` serves grad_input."""
    BC: tl.constexpr = GP * CG
    BM: tl.constexpr = GP * MG
    num_pg = G // GP
    pid = tl.program_id(axis=0)
    pid_g = pid % num_pg
    pid_chunk = pid // num_pg

    k_offset = 0
    seg_start_k = tl.load(seg_offs + 0).to(tl.int32)
    seg_end_k = tl.load(seg_offs + 1).to(tl.int32)
    chunk_idx_within_k = 0
    running = 0
    for k_iter in tl.range(0, K_offsets):
        seg_s = tl.load(seg_offs + k_iter).to(tl.int32)
        seg_e = tl.load(seg_offs + k_iter + 1).to(tl.int32)
        chunks_here = (seg_e - seg_s + L - 1) // L
        if running <= pid_chunk and pid_chunk < running + chunks_here:
            k_offset = k_iter
            seg_start_k = seg_s
            seg_end_k = seg_e
            chunk_idx_within_k = pid_chunk - running
        running += chunks_here
    if pid_chunk >= running:
        return

    l_offsets = seg_start_k + chunk_idx_within_k * L + tl.arange(0, L)
    l_mask = l_offsets < seg_end_k

    bj = tl.load(b_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)
    oi = tl.load(o_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)

    c_off = tl.arange(0, BC)
    m_off = tl.arange(0, BM)
    pg_c = c_off // CG
    pg_m = m_off // MG
    blk = pg_c[:, None] == pg_m[None, :]

    x = tl.load(b + bj[:, None] * (G * CG) + pid_g * (GP * CG)
                + c_off[None, :], mask=l_mask[:, None], other=0.0)
    w = tl.load(a + k_offset.to(tl.int64) * stride_wk
                + (pid_g * GP + pg_c)[:, None] * stride_wg
                + (c_off - pg_c * CG)[:, None] * stride_wc
                + (m_off - pg_m * MG)[None, :] * stride_wm,
                mask=blk, other=0.0)
    acc = tl.dot(x, w, input_precision=INPUT_PRECISION,
                 out_dtype=tl.float32)
    tl.atomic_add(o32 + oi[:, None] * (G * MG) + pid_g * (GP * MG)
                  + m_off[None, :], acc, mask=l_mask[:, None])


@triton.jit
def _tig_masked_kernel(
    nbr0, rows_sorted, b, a, o32, tilemask,
    NROWS, M, C, G,
    INPUT_PRECISION: tl.constexpr,
    KV: tl.constexpr, B1: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
):
    """Output-stationary masked pass over the injective level-0 rulebook.
    M and C are PER-GROUP widths; the grid carries a group axis exactly
    like the flat kernel (block-diagonal math, v1.2.0 G>1 support).
    Weight is the contiguous (K, G, C, M) forward layout."""
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
            jv = tl.load(nbr0 + rows * KV + v, mask=r_mask, other=-1)
            jm = jv >= 0
            jv64 = jv.to(tl.int64)
            for c0 in range(0, C, BC):
                c_offsets = c0 + tl.arange(0, BC)
                c_mask = c_offsets < C
                x = tl.load(b + jv64[:, None] * (G * C) + pid_g * C
                            + c_offsets[None, :],
                            mask=jm[:, None] & c_mask[None, :], other=0.0)
                w = tl.load(a + v * (G * C * M) + pid_g * (C * M)
                            + c_offsets[:, None] * M
                            + m_offsets[None, :],
                            mask=c_mask[:, None] & m_mask[None, :], other=0.0)
                acc = tl.dot(x, w, acc, input_precision=INPUT_PRECISION)

    # level-0 rows are written by exactly one program per (group, M-tile),
    # but the flat residual pass accumulates on top -> atomic for
    # composability.
    tl.atomic_add(o32 + rows[:, None] * (G * M) + pid_g * M
                  + m_offsets[None, :], acc,
                  mask=r_mask[:, None] & m_mask[None, :])


@triton.jit
def _tig_wgrad_kernel(
    x, go, b_idx, o_idx, gw32, seg_offs,
    K_offsets, M, C, G,
    INPUT_PRECISION: tl.constexpr,
    L: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
):
    """Implicit wgrad with chunk-level Split-K (M/C per-group; grid
    carries the group axis):
    ``gw32[k] += x[b_idx]^T @ go[o_idx]`` per L-row chunk of each k
    segment (atomic accumulation across chunks — the long-reduction /
    tiny-output shape where Split-K is mandatory)."""
    num_pid_c = tl.cdiv(C, BC)
    num_pid_m = tl.cdiv(M, BM)
    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid = pid // num_pid_m
    pid_c = pid % num_pid_c
    pid = pid // num_pid_c
    pid_g = pid % G
    pid_chunk = pid // G

    k_offset = 0
    seg_start_k = tl.load(seg_offs + 0).to(tl.int32)
    seg_end_k = tl.load(seg_offs + 1).to(tl.int32)
    chunk_idx_within_k = 0
    running = 0
    for k_iter in tl.range(0, K_offsets):
        seg_s = tl.load(seg_offs + k_iter).to(tl.int32)
        seg_e = tl.load(seg_offs + k_iter + 1).to(tl.int32)
        chunks_here = (seg_e - seg_s + L - 1) // L
        if running <= pid_chunk and pid_chunk < running + chunks_here:
            k_offset = k_iter
            seg_start_k = seg_s
            seg_end_k = seg_e
            chunk_idx_within_k = pid_chunk - running
        running += chunks_here

    # upper-bound grid: programs past the true chunk total exit (lets the
    # host size the grid as cdiv(T, L) + K with NO device->host sync)
    if pid_chunk >= running:
        return

    l_offsets = seg_start_k + chunk_idx_within_k * L + tl.arange(0, L)
    l_mask = l_offsets < seg_end_k

    c_offsets = pid_c * BC + tl.arange(0, BC)
    m_offsets = pid_m * BM + tl.arange(0, BM)
    c_mask = c_offsets < C
    m_mask = m_offsets < M

    bj = tl.load(b_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)
    oi = tl.load(o_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)

    xb = tl.load(x + bj[:, None] * (G * C) + pid_g * C
                 + c_offsets[None, :],
                 mask=l_mask[:, None] & c_mask[None, :], other=0.0)
    gb = tl.load(go + oi[:, None] * (G * M) + pid_g * M
                 + m_offsets[None, :],
                 mask=l_mask[:, None] & m_mask[None, :], other=0.0)
    acc = tl.dot(tl.trans(xb), gb, input_precision=INPUT_PRECISION,
                 out_dtype=tl.float32)

    tl.atomic_add(gw32 + k_offset.to(tl.int64) * (G * C * M)
                  + pid_g * (C * M)
                  + c_offsets[:, None] * M + m_offsets[None, :], acc,
                  mask=c_mask[:, None] & m_mask[None, :])


@triton.jit
def _tig_wgrad_bisect_kernel(
    x, go, b_idx, o_idx, gw32, seg_offs, cum_chunks,
    K_offsets, M, C, G,
    INPUT_PRECISION: tl.constexpr,
    L: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
):
    """``_tig_wgrad_kernel`` with the binary-search chunk map
    (see ``_tig_flat_bisect_kernel``) — routed at K > _BISECT_MIN_K."""
    num_pid_c = tl.cdiv(C, BC)
    num_pid_m = tl.cdiv(M, BM)
    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid = pid // num_pid_m
    pid_c = pid % num_pid_c
    pid = pid // num_pid_c
    pid_g = pid % G
    pid_chunk = pid // G

    total = tl.load(cum_chunks + K_offsets).to(tl.int32)
    if pid_chunk >= total:
        return
    lo = 0
    hi = K_offsets
    while hi - lo > 1:
        mid = (lo + hi) // 2
        cm = tl.load(cum_chunks + mid).to(tl.int32)
        if cm <= pid_chunk:
            lo = mid
        else:
            hi = mid
    k_offset = lo
    chunk_idx_within_k = pid_chunk - tl.load(cum_chunks + lo).to(tl.int32)
    seg_start_k = tl.load(seg_offs + lo).to(tl.int32)
    seg_end_k = tl.load(seg_offs + lo + 1).to(tl.int32)

    l_offsets = seg_start_k + chunk_idx_within_k * L + tl.arange(0, L)
    l_mask = l_offsets < seg_end_k

    c_offsets = pid_c * BC + tl.arange(0, BC)
    m_offsets = pid_m * BM + tl.arange(0, BM)
    c_mask = c_offsets < C
    m_mask = m_offsets < M

    bj = tl.load(b_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)
    oi = tl.load(o_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)

    xb = tl.load(x + bj[:, None] * (G * C) + pid_g * C
                 + c_offsets[None, :],
                 mask=l_mask[:, None] & c_mask[None, :], other=0.0)
    gb = tl.load(go + oi[:, None] * (G * M) + pid_g * M
                 + m_offsets[None, :],
                 mask=l_mask[:, None] & m_mask[None, :], other=0.0)
    acc = tl.dot(tl.trans(xb), gb, input_precision=INPUT_PRECISION,
                 out_dtype=tl.float32)

    tl.atomic_add(gw32 + k_offset.to(tl.int64) * (G * C * M)
                  + pid_g * (C * M)
                  + c_offsets[:, None] * M + m_offsets[None, :], acc,
                  mask=c_mask[:, None] & m_mask[None, :])


@triton.jit
def _tig_wgrad_packed_kernel(
    x, go, b_idx, o_idx, gw32, seg_offs,
    K_offsets, G,
    INPUT_PRECISION: tl.constexpr,
    CG: tl.constexpr, MG: tl.constexpr, GP: tl.constexpr,
    L: tl.constexpr,
):
    """Group-packed implicit wgrad (v1.2.0 squeeze, same packing as
    ``_tig_flat_packed_kernel``): x/go loads dense across GP adjacent
    groups, one (GP*CG, GP*MG) trans(x)@go dot per (chunk, group-pair);
    the off-diagonal (cross-group) quadrants of the dot are garbage and
    simply never stored (block-diagonal atomic mask). Real c64 G=8 fp16
    wgrad 0.38 -> 0.15 ms."""
    num_pg = G // GP
    pid = tl.program_id(axis=0)
    pid_g = pid % num_pg
    pid_chunk = pid // num_pg

    k_offset = 0
    seg_start_k = tl.load(seg_offs + 0).to(tl.int32)
    seg_end_k = tl.load(seg_offs + 1).to(tl.int32)
    chunk_idx_within_k = 0
    running = 0
    for k_iter in tl.range(0, K_offsets):
        seg_s = tl.load(seg_offs + k_iter).to(tl.int32)
        seg_e = tl.load(seg_offs + k_iter + 1).to(tl.int32)
        chunks_here = (seg_e - seg_s + L - 1) // L
        if running <= pid_chunk and pid_chunk < running + chunks_here:
            k_offset = k_iter
            seg_start_k = seg_s
            seg_end_k = seg_e
            chunk_idx_within_k = pid_chunk - running
        running += chunks_here
    if pid_chunk >= running:
        return

    l_offsets = seg_start_k + chunk_idx_within_k * L + tl.arange(0, L)
    l_mask = l_offsets < seg_end_k

    bj = tl.load(b_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)
    oi = tl.load(o_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)

    c_off = tl.arange(0, GP * CG)
    m_off = tl.arange(0, GP * MG)
    pg_c = c_off // CG
    pg_m = m_off // MG
    blk = pg_c[:, None] == pg_m[None, :]

    xb = tl.load(x + bj[:, None] * (G * CG) + pid_g * (GP * CG)
                 + c_off[None, :], mask=l_mask[:, None], other=0.0)
    gb = tl.load(go + oi[:, None] * (G * MG) + pid_g * (GP * MG)
                 + m_off[None, :], mask=l_mask[:, None], other=0.0)
    acc = tl.dot(tl.trans(xb), gb, input_precision=INPUT_PRECISION,
                 out_dtype=tl.float32)
    tl.atomic_add(gw32 + k_offset.to(tl.int64) * (G * CG * MG)
                  + (pid_g * GP + pg_c)[:, None] * (CG * MG)
                  + (c_off - pg_c * CG)[:, None] * MG
                  + (m_off - pg_m * MG)[None, :], acc, mask=blk)


@triton.jit
def _tig_bwd_fused_kernel(
    a, x, go, b_idx, o_idx, gx32, gw32, seg_offs,
    K_offsets, M, C, G,
    INPUT_PRECISION: tl.constexpr,
    L: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
):
    """One-pass backward: per (chunk, group, c-tile, m-tile) program,
    load go/x/W once and emit BOTH grads (M/C per-group; grid carries
    the group axis):
        gw[k, g, c, m] += x[j]^T @ go[i]        (Split-K over chunks)
        gx[j, g, c]    += go[i] @ W[k, g]^T     (partial over the m-tile)
    gx atomic traffic scales with num_m_tiles — net-positive only at
    small M (the config table routes; measured, not assumed)."""
    num_pid_c = tl.cdiv(C, BC)
    num_pid_m = tl.cdiv(M, BM)
    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid = pid // num_pid_m
    pid_c = pid % num_pid_c
    pid = pid // num_pid_c
    pid_g = pid % G
    pid_chunk = pid // G

    k_offset = 0
    seg_start_k = tl.load(seg_offs + 0).to(tl.int32)
    seg_end_k = tl.load(seg_offs + 1).to(tl.int32)
    chunk_idx_within_k = 0
    running = 0
    for k_iter in tl.range(0, K_offsets):
        seg_s = tl.load(seg_offs + k_iter).to(tl.int32)
        seg_e = tl.load(seg_offs + k_iter + 1).to(tl.int32)
        chunks_here = (seg_e - seg_s + L - 1) // L
        if running <= pid_chunk and pid_chunk < running + chunks_here:
            k_offset = k_iter
            seg_start_k = seg_s
            seg_end_k = seg_e
            chunk_idx_within_k = pid_chunk - running
        running += chunks_here

    # upper-bound grid: programs past the true chunk total exit (lets the
    # host size the grid as cdiv(T, L) + K with NO device->host sync)
    if pid_chunk >= running:
        return

    l_offsets = seg_start_k + chunk_idx_within_k * L + tl.arange(0, L)
    l_mask = l_offsets < seg_end_k

    c_offsets = pid_c * BC + tl.arange(0, BC)
    m_offsets = pid_m * BM + tl.arange(0, BM)
    c_mask = c_offsets < C
    m_mask = m_offsets < M

    bj = tl.load(b_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)
    oi = tl.load(o_idx + l_offsets, mask=l_mask, other=0).to(tl.int64)

    xb = tl.load(x + bj[:, None] * (G * C) + pid_g * C
                 + c_offsets[None, :],
                 mask=l_mask[:, None] & c_mask[None, :], other=0.0)
    gb = tl.load(go + oi[:, None] * (G * M) + pid_g * M
                 + m_offsets[None, :],
                 mask=l_mask[:, None] & m_mask[None, :], other=0.0)
    wb = tl.load(a + k_offset.to(tl.int64) * (G * C * M)
                 + pid_g * (C * M)
                 + c_offsets[:, None] * M + m_offsets[None, :],
                 mask=c_mask[:, None] & m_mask[None, :], other=0.0)

    gw_part = tl.dot(tl.trans(xb), gb, input_precision=INPUT_PRECISION,
                     out_dtype=tl.float32)
    tl.atomic_add(gw32 + k_offset.to(tl.int64) * (G * C * M)
                  + pid_g * (C * M)
                  + c_offsets[:, None] * M + m_offsets[None, :], gw_part,
                  mask=c_mask[:, None] & m_mask[None, :])

    gx_part = tl.dot(gb, tl.trans(wb), input_precision=INPUT_PRECISION,
                     out_dtype=tl.float32)
    tl.atomic_add(gx32 + bj[:, None] * (G * C) + pid_g * C
                  + c_offsets[None, :], gx_part,
                  mask=l_mask[:, None] & c_mask[None, :])


# ── index structure ──────────────────────────────────────────────────────────


class TigIndex:
    """Reusable TIG rulebook built once per (triplets, K) — the
    warm-cache analog of FlexGEMM's neighbor_cache.

    Holds BOTH orientations: the flat k-sorted arrays (shared with the
    existing production path) and the hybrid split (level-0 ``nbr0``
    [N, K] int32 with -1 sentinel + gray-sorted compacted rows + ragged
    residual, k-sorted). Per-B1 tile masks are derived lazily and cached
    (the FlexGEMM ``valid_kernel`` pattern).
    """

    def __init__(self, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor,
                 n_out: int, num_kernel_offsets: int = 27,
                 build_hybrid: bool = True, assume_sorted: bool = False,
                 n_in: Optional[int] = None,
                 exact_cover_out: bool = False,
                 exact_cover_in: bool = False,
                 uniform_seg_len: Optional[int] = None):
        dev = i.device
        # caller-asserted exactly-once-scatter contracts (the
        # disjoint builders own them; NEVER inferred — verification would
        # cost a device sync). exact_cover_out: every output row receives
        # exactly one triplet (fan-in-1 deconv forward) -> the FI1 plain-
        # store forward. exact_cover_in: every input row appears in
        # exactly one triplet (partition fan-out-1) -> FI1 grad_input.
        self.exact_cover_out = bool(exact_cover_out)
        self.exact_cover_in = bool(exact_cover_in)
        self.N = int(n_out)
        # input-side row count, used to size
        # grad_input. Defaults to n_out — the submanifold case — so every
        # existing caller is byte-unchanged. For generative convs
        # (N_in != N_out: strided partition / fan-in-1 deconv) pass the
        # input point count explicitly.
        self.N_in = int(n_in) if n_in is not None else self.N
        self.K = int(num_kernel_offsets)
        self.T = int(i.numel())
        i = i.long()
        j = j.long()
        k = k.long()
        # flat orientation (k-sorted — production triplets already are;
        # re-sort defensively only if needed)
        if self.T and not assume_sorted and not bool(
                (k[1:] >= k[:-1]).all()):  # D2H sync — callers that
            # guarantee k-sorted triplets pass assume_sorted=True
            order = torch.argsort(k, stable=True)
            i, j, k = i[order], j[order], k[order]
        # kernels cast index loads to int64 — keep the caller's tensors
        # as-is (no .int() copy; T-length copies cost ~0.1 ms at enc0)
        self.flat_i = i
        self.flat_j = j
        # uniform segments (caller contract — e.g. the stamp
        # generator's k-sorted rulebook has EXACTLY uniform_seg_len
        # triplets per tap): seg_offs is closed-form, no searchsorted.
        self.uniform_seg_len = (int(uniform_seg_len)
                                if uniform_seg_len is not None else None)
        if self.uniform_seg_len is not None:
            assert self.uniform_seg_len * self.K == self.T, (
                f"uniform_seg_len {self.uniform_seg_len} * K {self.K} "
                f"!= T {self.T}")
            self.flat_seg = (torch.arange(self.K + 1, device=dev,
                                          dtype=torch.int64)
                             * self.uniform_seg_len)
        else:
            self.flat_seg = kernel_offset_segments(k, self.K)

        self._tilemask_cache: dict = {}
        self._chunks_cache: dict = {}
        self.has_hybrid = False
        if build_hybrid and self.T:
            K = self.K
            key = i * K + k
            order = torch.argsort(key, stable=True)
            ks = key[order]
            first = torch.ones(self.T, dtype=torch.bool, device=dev)
            first[1:] = ks[1:] != ks[:-1]
            f_idx = order[first]
            nbr0 = torch.full((self.N, K), -1, device=dev, dtype=torch.int32)
            nbr0.view(-1)[key[f_idx]] = j[f_idx].int()
            r_idx = order[~first]
            rk = k[r_idx]
            r_idx = r_idx[torch.argsort(rk, stable=True)]
            self.res_i = i[r_idx].int()
            self.res_j = j[r_idx].int()
            self.res_seg = kernel_offset_segments(k[r_idx], K)
            self.res_T = int(r_idx.numel())
            mask = nbr0 >= 0
            rows_any = mask.any(1).nonzero(as_tuple=True)[0]
            mbits = (mask[rows_any].long()
                     << torch.arange(K, device=dev)).sum(1)
            gray = mbits ^ (mbits >> 1)
            perm = torch.argsort(gray)
            self.nbr0 = nbr0
            self.rows_sorted = rows_any[perm].int()
            self._mbits_sorted = mbits[perm]
            self.has_hybrid = True

    def chunk_table(self, which: str, L: int) -> torch.Tensor:
        """(K+1,) int32 cumulative chunk table for the bisect
        kernel — cum[k] = sum over k' < k of ceil(seg_len_k' / L). Built
        device-side (no D2H sync) once per (orientation, L) and cached."""
        key = ("cumtab", which, L)
        t = self._chunks_cache.get(key)
        if t is None:
            if which == "flat" and self.uniform_seg_len is not None:
                # uniform-segment closed form — ceil(uniform/L) chunks per
                # segment, one arange, no cumsum chain.
                per = -(-self.uniform_seg_len // L)
                t = (torch.arange(self.K + 1, device=self.flat_seg.device,
                                  dtype=torch.int32) * per)
            else:
                seg = self.flat_seg if which == "flat" else self.res_seg
                lens = seg[1:] - seg[:-1]
                ch = torch.div(lens + (L - 1), L, rounding_mode="floor")
                t = torch.zeros(seg.numel(), dtype=torch.int32,
                                device=seg.device)
                t[1:] = torch.cumsum(ch, 0).to(torch.int32)
            self._chunks_cache[key] = t
        return t

    def chunks(self, which: str, L: int) -> int:
        """Host-side chunk count for the flat kernel grid, cached so the
        per-call D2H sync happens once per (orientation, L)."""
        key = (which, L)
        n = self._chunks_cache.get(key)
        if n is None:
            if which == "flat" and self.uniform_seg_len is not None:
                n = self.K * -(-self.uniform_seg_len // L)  # closed form
            else:
                seg = self.flat_seg if which == "flat" else self.res_seg
                n = _chunks(seg, L)
            self._chunks_cache[key] = n
        return n

    def tilemask(self, b1: int) -> torch.Tensor:
        tm = self._tilemask_cache.get(b1)
        if tm is None:
            mb = self._mbits_sorted
            pad = (-mb.numel()) % b1
            if pad:
                mb = torch.cat([mb, torch.zeros(pad, dtype=mb.dtype,
                                                device=mb.device)])
            mb = mb.view(-1, b1)
            tm = torch.zeros(mb.size(0), dtype=torch.int64, device=mb.device)
            for v in range(self.K):
                tm |= ((mb >> v) & 1).amax(1) << v
            tm = tm.to(torch.int32)
            self._tilemask_cache[b1] = tm
        return tm


# above this K the flat kernels' linear pid_chunk->k scan (K
# iterations in EVERY program) dominates and the bisect variant routes
# instead. 27 (3^3) and 125 (5^3) stay on the proven scan kernel; 512
# (8^3, a generative stem) bisects. Threshold is structural, not tuned.
_BISECT_MIN_K = 64

_FI1_CC_CACHE: dict = {}


def _fi1_wins_here() -> bool:
    """Measured verdict: the FI1 plain-store forward wins on Ada (sm_89)
    (1.6-2.7x fwd at the fan-in-1 deconv stages) but LOSES to the regular
    atomic path on Hopper (sm_90) (up to 2.3x slower fwd — fp32 atomics +
    bandwidth make the zeros/cast passes near-free). Honor the caller's
    exact_cover flags only where the data says FI1 is the right engine;
    per-arch, cached once per device."""
    dev = torch.cuda.current_device()
    hit = _FI1_CC_CACHE.get(dev)
    if hit is None:
        hit = torch.cuda.get_device_capability(dev) < (9, 0)
        _FI1_CC_CACHE[dev] = hit
    return hit

# ── config buckets (channel-keyed only — never N/T) ─────────────────────────


def _packed_ok(G: int, C: int, M: int) -> bool:
    """WGRAD-only packed route: GP=2 at exactly the Cg==Mg==8 shape
    (G even), measured on the real c64 G=8 fp16 cell (the cell where
    TIG previously lost to the per-triplet path). A follow-up probe
    re-measured the wider shapes for wgrad and found NO win
    (Cg=Mg=16 GP=2: 0.177 vs 0.180 ms wash; Cg=Mg=8 GP=4: no win vs
    the GP=2 default) — wgrad keeps this shape only. fwd/grad_input
    route more widely: see ``_packed_gp``."""
    return G % 2 == 0 and C == 8 and M == 8


def _packed_gp(G: int, C: int, M: int):
    """GP for the fwd/grad_input packed route (None = unpacked flat).
    Probe-verified on a real c64 ScanNet-scene fp16 protocol: packing
    wins at Cg==Mg==16 too — the earlier assumption that "Cg>=16 tiles
    are already dense (no packing win)" did not hold; the measured
    truth is GP=2 @ Cg=Mg=16 (c64 G=4 cell, correct to 1.07e-7 vs the
    fp64 oracle) beats the routed 16-wide default fwd 0.775->0.529,
    gi 0.766->0.555 ms, and GP=4 @ Cg=Mg=8 (c64 G=8 cell, 9.4e-8)
    beats the GP=2 default ~13% (fwd 0.660->0.570, gi 0.651->0.564 ms).
    GP*Cg=32-wide tiles stay tl.dot-legal; the dense x/out accesses +
    fewer dots win despite the block-diagonal flop waste. Cg<8 would
    need GP=8 (unmeasured); non-power-of-2 Cg is not tl.arange-legal."""
    if C == 8 and M == 8 and G % 2 == 0:
        return 4 if G % 4 == 0 else 2
    if C == 16 and M == 16 and G % 2 == 0:
        return 2
    return None


# packed-kernel launch configs, bucketed on (Cg, GP) — channel-keyed
# only, never N/T. (8, 2): original sweep on real c64 G=8 fp16
# (L=256/w=4 best of {64,128,256,512} x {2,4,8}). (16, 2) / (8, 4):
# follow-up probe winners — fwd L=256/w=8 at both shapes; gi
# L=128/w=4 at Cg=16, L=256/w=8 at Cg=8/GP=4.
_PACKED_FWD = {(8, 2): dict(L=256, num_warps=4),
               (8, 4): dict(L=256, num_warps=8),
               (16, 2): dict(L=256, num_warps=8)}
_PACKED_GI = {(8, 2): dict(L=256, num_warps=4),
              (8, 4): dict(L=256, num_warps=8),
              (16, 2): dict(L=128, num_warps=4)}
_PACKED_WG = dict(L=256, num_warps=4)


def _flat_cfg(C: int, M: int):
    if C <= 16 and M <= 16:
        # v1.2.0 (Cg<=16 grouped cells): 16-wide dot tiles — the
        # sm_8x tl.dot minimum — instead of the 32-wide floor, which
        # padded Cg=8 4x in BOTH dot dims (real c64 G=8 fp16 fwd:
        # 3.15 -> 0.93 ms). An fma (no-dot) micro-path measured WORSE
        # (1.37 ms same cell) — padded tensor-core dot still wins.
        # NOTE: at G even + square Cg==Mg in {8, 16} the packed kernels
        # supersede this branch for fwd/gi (GP=4/GP=2, see _packed_gp;
        # 0.93 -> 0.61 ms at the original c64 G=8 cell); this cfg
        # remains the path for odd G, non-square Cg!=Mg cells, and
        # explicit-override replays.
        return dict(L=128, BM=16, BC=16, num_warps=4)
    if M >= 256:
        return dict(L=64, BM=64, BC=32, num_warps=8)
    return dict(L=128, BM=32, BC=32, num_warps=8)


def _l0_cfg(C: int, M: int):
    if C <= 16 and M <= 16:
        return dict(B1=32, BM=16, BC=16, num_warps=4)
    if M >= 256:
        return dict(B1=64, BM=32, BC=32, num_warps=4)
    return dict(B1=32, BM=32, BC=32, num_warps=4)


def _chunks(seg_offs: torch.Tensor, L: int) -> int:
    lens = (seg_offs[1:] - seg_offs[:-1]).cpu().tolist()
    return sum((ln + L - 1) // L for ln in lens)


# ── public op ────────────────────────────────────────────────────────────────


def tig_forward(
    weight: torch.Tensor,
    feat: torch.Tensor,
    index: TigIndex,
    mode: str = "auto",
    input_precision: str = "tf32",
    flat_cfg: Optional[dict] = None,
    l0_cfg: Optional[dict] = None,
) -> torch.Tensor:
    """General mvmr forward: ``out[i] += feat[j] @ W[k]``.

    weight: (K, G, C, M) or (K, C, M) (=G==1); feat: (N, G*C) flat,
    fp16/bf16/fp32. mode: "flat" | "hybrid" | "auto" (auto: hybrid for
    small per-group M where the masked pass measured net-positive, else
    flat; both modes carry the group axis natively).
    Returns (N, G*M) flat in feat.dtype. fp32 inputs follow
    ``input_precision`` ("tf32" matches the production grouped path's
    default; "ieee" exact). G>1 is block-diagonal (v1.2.0).
    """
    if weight.dim() == 4:
        K, G, C, M = weight.shape
    else:
        K, C, M = weight.shape
        G = 1
    assert K == index.K and feat.size(1) == G * C
    # feat rows are the input side (gathered by j); for
    # generative indices this differs from the output count index.N.
    assert feat.size(0) == index.N_in, (
        f"feat rows {feat.size(0)} != index.N_in {index.N_in}")
    weight = weight.contiguous()
    feat = feat.contiguous()
    N = index.N

    if mode == "auto":
        mode = "hybrid" if (index.has_hybrid and M <= 64) else "flat"
    if mode == "hybrid" and not index.has_hybrid:
        mode = "flat"

    o32 = torch.zeros(N, G * M, device=feat.device, dtype=torch.float32)
    num_pid_m_of = lambda BM: triton.cdiv(M, BM)

    if mode == "flat" and index.exact_cover_out and _fi1_wins_here():
        # exactly-once scatter — single pass, native dtype,
        # no fp32 staging buffer, no atomics, no cast pass. Arch-gated
        # (sm_89 wins / sm_90 loses — see _fi1_wins_here).
        cfg = dict(_flat_cfg(C, M), **(flat_cfg or {}))
        w = cfg.pop("num_warps")
        out = torch.empty(N, G * M, device=feat.device, dtype=feat.dtype)
        if index.T:
            grid = ((triton.cdiv(index.T, cfg["L"]) + K) * G
                    * num_pid_m_of(cfg["BM"]),)
            _tig_flat_fi1_kernel[grid](
                weight, feat, index.flat_j, index.flat_i, out,
                index.flat_seg, index.chunk_table("flat", cfg["L"]),
                K, M, C, G, G * C * M, C * M, M, 1,
                INPUT_PRECISION=input_precision, num_warps=w, **cfg)
        else:
            out.zero_()
        return out

    if mode == "flat":
        gp = _packed_gp(G, C, M) if flat_cfg is None else None
        if gp is not None:
            if index.T:
                pcfg = _PACKED_FWD[(C, gp)]
                grid = ((triton.cdiv(index.T, pcfg["L"]) + K)
                        * (G // gp),)
                _tig_flat_packed_kernel[grid](
                    weight, feat, index.flat_j, index.flat_i, o32,
                    index.flat_seg, K, G, G * C * M, C * M, M, 1,
                    INPUT_PRECISION=input_precision, CG=C, MG=M, GP=gp,
                    **pcfg)
            return o32.to(feat.dtype)
        cfg = dict(_flat_cfg(C, M), **(flat_cfg or {}))
        w = cfg.pop("num_warps")
        grid = ((triton.cdiv(index.T, cfg["L"]) + K) * G
                * num_pid_m_of(cfg["BM"]),)
        if index.T:
            if K > _BISECT_MIN_K:  # large-K (e.g. 8^3 stem): bisect
                _tig_flat_bisect_kernel[grid](
                    weight, feat, index.flat_j, index.flat_i, o32,
                    index.flat_seg, index.chunk_table("flat", cfg["L"]),
                    K, M, C, G, G * C * M, C * M, M, 1,
                    INPUT_PRECISION=input_precision, num_warps=w, **cfg)
            else:
                _tig_flat_kernel[grid](
                    weight, feat, index.flat_j, index.flat_i, o32,
                    index.flat_seg,
                    K, M, C, G, G * C * M, C * M, M, 1,
                    INPUT_PRECISION=input_precision, num_warps=w, **cfg)
    elif mode == "hybrid":
        l0 = dict(_l0_cfg(C, M), **(l0_cfg or {}))
        w0 = l0.pop("num_warps")
        nrows = index.rows_sorted.numel()
        tm = index.tilemask(l0["B1"])
        grid0 = (triton.cdiv(nrows, l0["B1"]) * G * num_pid_m_of(l0["BM"]),)
        if nrows:
            _tig_masked_kernel[grid0](
                index.nbr0, index.rows_sorted, feat, weight, o32, tm,
                nrows, M, C, G, INPUT_PRECISION=input_precision,
                KV=K, num_warps=w0, **l0)
        if index.res_T:
            cfg = dict(_flat_cfg(C, M), **(flat_cfg or {}))
            w = cfg.pop("num_warps")
            grid = ((triton.cdiv(index.res_T, cfg["L"]) + K) * G
                    * num_pid_m_of(cfg["BM"]),)
            _tig_flat_kernel[grid](
                weight, feat, index.res_j, index.res_i, o32, index.res_seg,
                K, M, C, G, G * C * M, C * M, M, 1,
                INPUT_PRECISION=input_precision, num_warps=w, **cfg)
    else:
        raise ValueError(f"unknown TIG mode {mode!r}")

    return o32.to(feat.dtype)


def _gi_cfg(C_out: int, M_in: int):
    """grad_input configs — swept on real ScanNet stages; small-channel
    branch from the v1.2.0 grouped-cell tuning (c64 G=8 fp16 gi:
    3.10 -> 0.92 ms)."""
    if C_out <= 16 and M_in <= 16:
        return dict(L=128, BM=16, BC=16, num_warps=4)
    if C_out >= 64:
        return dict(L=64, BM=64, BC=64 if M_in >= 64 else 32, num_warps=8)
    return dict(L=128, BM=32, BC=32, num_warps=8)


def tig_grad_input(
    weight: torch.Tensor,
    grad_out: torch.Tensor,
    index: TigIndex,
    input_precision: str = "tf32",
    flat_cfg: Optional[dict] = None,
) -> torch.Tensor:
    """grad_feat[j] += grad_out[i] @ W[k]^T — the SAME flat kernel with
    gather/scatter roles swapped and the weight transposed via strides
    (zero-copy, no staging). Sized by ``index.N_in`` so generative
    convs (N_in != N_out) get a correctly-shaped grad; for submanifold
    indices N_in == N (unchanged behavior)."""
    if weight.dim() == 4:
        K, G, C, M = weight.shape
    else:
        K, C, M = weight.shape
        G = 1
    assert grad_out.size(0) == index.N, (
        f"grad_out rows {grad_out.size(0)} != index.N {index.N}")
    weight = weight.contiguous()
    grad_out = grad_out.contiguous()
    N = index.N_in
    if index.exact_cover_in and _fi1_wins_here():
        # each input row appears in exactly one triplet
        # (partition fan-out-1) — grad_input is a pure exactly-once
        # scatter: single pass, native dtype, no fp32 buffer/atomics/cast.
        cfg = dict(_gi_cfg(C, M), **(flat_cfg or {}))
        w = cfg.pop("num_warps")
        out = torch.empty(N, G * C, device=grad_out.device,
                          dtype=grad_out.dtype)
        if index.T:
            grid = ((triton.cdiv(index.T, cfg["L"]) + K) * G
                    * triton.cdiv(C, cfg["BM"]),)
            _tig_flat_fi1_kernel[grid](
                weight, grad_out, index.flat_i, index.flat_j, out,
                index.flat_seg, index.chunk_table("flat", cfg["L"]),
                K, C, M, G, G * C * M, C * M, 1, M,
                INPUT_PRECISION=input_precision, num_warps=w, **cfg)
        else:
            out.zero_()
        return out
    # roles per group: INW=M (gather grad_out rows by i), OUTW=C (scatter
    # by j); logical W'[k, g, m, c] = W[k, g, c, m] -> wc-stride 1, wm-stride M
    o32 = torch.zeros(N, G * C, device=grad_out.device, dtype=torch.float32)
    gp = _packed_gp(G, C, M) if flat_cfg is None else None
    if gp is not None:
        if index.T:
            pcfg = _PACKED_GI[(C, gp)]
            grid = ((triton.cdiv(index.T, pcfg["L"]) + K) * (G // gp),)
            _tig_flat_packed_kernel[grid](
                weight, grad_out, index.flat_i, index.flat_j, o32,
                index.flat_seg, K, G, G * C * M, C * M, 1, M,
                INPUT_PRECISION=input_precision, CG=M, MG=C, GP=gp,
                **pcfg)
        return o32.to(grad_out.dtype)
    cfg = dict(_gi_cfg(C, M), **(flat_cfg or {}))
    w = cfg.pop("num_warps")
    if index.T:
        grid = ((triton.cdiv(index.T, cfg["L"]) + K) * G
                * triton.cdiv(C, cfg["BM"]),)
        if K > _BISECT_MIN_K:  # large-K: bisect (see tig_forward)
            _tig_flat_bisect_kernel[grid](
                weight, grad_out, index.flat_i, index.flat_j, o32,
                index.flat_seg, index.chunk_table("flat", cfg["L"]),
                K, C, M, G, G * C * M, C * M, 1, M,
                INPUT_PRECISION=input_precision, num_warps=w, **cfg)
        else:
            _tig_flat_kernel[grid](
                weight, grad_out, index.flat_i, index.flat_j, o32,
                index.flat_seg,
                K, C, M, G, G * C * M, C * M, 1, M,
                INPUT_PRECISION=input_precision, num_warps=w, **cfg)
    return o32.to(grad_out.dtype)


def _wgrad_cfg(C: int, M: int, dtype: torch.dtype = torch.float16):
    """wgrad configs — swept on real ScanNet stages: long chunks win
    (Split-K granularity vs atomic traffic); fp32 operands double smem
    so the chunk halves. Small-channel branch from the v1.2.0
    grouped-cell tuning (c64 G=8 fp16 wgrad: 1.55 -> 0.39 ms; L=256
    measured over 512 at 16-wide tiles)."""
    if C <= 16 and M <= 16:
        L = 128 if dtype == torch.float32 else 256
        return dict(L=L, BM=16, BC=16, num_warps=4)
    L = 128 if dtype == torch.float32 else 512
    return dict(L=L, BM=64 if M >= 64 else 32, BC=64 if C >= 64 else 32,
                num_warps=8 if C >= 64 else 4)


def tig_grad_weight(
    feat: torch.Tensor,
    grad_out: torch.Tensor,
    index: TigIndex,
    weight_shape,
    input_precision: str = "tf32",
    wgrad_cfg: Optional[dict] = None,
) -> torch.Tensor:
    """grad_W[k] += feat[j]^T (outer) grad_out[i] — implicit wgrad,
    Split-K via per-chunk atomic accumulation into an fp32 buffer."""
    feat = feat.contiguous()
    grad_out = grad_out.contiguous()
    G = weight_shape[1] if len(weight_shape) == 4 else 1
    C = feat.size(1) // G
    M = grad_out.size(1) // G
    K = index.K
    gw32 = torch.zeros(K, G, C, M, device=feat.device, dtype=torch.float32)
    if wgrad_cfg is None and _packed_ok(G, C, M):
        if index.T:
            grid = ((triton.cdiv(index.T, _PACKED_WG["L"]) + K) * (G // 2),)
            _tig_wgrad_packed_kernel[grid](
                feat, grad_out, index.flat_j, index.flat_i, gw32,
                index.flat_seg, K, G, INPUT_PRECISION=input_precision,
                CG=C, MG=M, GP=2, **_PACKED_WG)
        gw = gw32.to(feat.dtype)
        return (gw.view(weight_shape) if len(weight_shape) == 4
                else gw.view(K, C, M))
    cfg = dict(_wgrad_cfg(C, M, feat.dtype), **(wgrad_cfg or {}))
    w = cfg.pop("num_warps")
    if index.T:
        grid = ((triton.cdiv(index.T, cfg["L"]) + K) * G
                * triton.cdiv(C, cfg["BC"]) * triton.cdiv(M, cfg["BM"]),)
        if K > _BISECT_MIN_K:  # large-K: bisect (see tig_forward)
            _tig_wgrad_bisect_kernel[grid](
                feat, grad_out, index.flat_j, index.flat_i, gw32,
                index.flat_seg, index.chunk_table("flat", cfg["L"]),
                K, M, C, G, INPUT_PRECISION=input_precision,
                num_warps=w, **cfg)
        else:
            _tig_wgrad_kernel[grid](
                feat, grad_out, index.flat_j, index.flat_i, gw32,
                index.flat_seg,
                K, M, C, G, INPUT_PRECISION=input_precision,
                num_warps=w, **cfg)
    gw = gw32.to(feat.dtype)
    return (gw.view(weight_shape) if len(weight_shape) == 4
            else gw.view(K, C, M))


def tig_backward_fused(
    weight: torch.Tensor,
    feat: torch.Tensor,
    grad_out: torch.Tensor,
    index: TigIndex,
    weight_shape,
    input_precision: str = "tf32",
    cfg: Optional[dict] = None,
):
    """One-pass backward (both grads). Net-positive at small M where the
    extra per-m-tile gx atomics don't dominate — routed by config.
    G>1 is block-diagonal (group axis in the grid, like the split path)."""
    if weight.dim() == 4:
        K, G, C, M = weight.shape
    else:
        K, C, M = weight.shape
        G = 1
    weight = weight.contiguous()
    feat = feat.contiguous()
    grad_out = grad_out.contiguous()
    c = dict(_wgrad_cfg(C, M, feat.dtype), **(cfg or {}))
    w = c.pop("num_warps")
    gx32 = torch.zeros(index.N_in, G * C, device=feat.device,
                       dtype=torch.float32)
    gw32 = torch.zeros(K, G, C, M, device=feat.device, dtype=torch.float32)
    if index.T:
        grid = ((triton.cdiv(index.T, c["L"]) + K) * G
                * triton.cdiv(C, c["BC"]) * triton.cdiv(M, c["BM"]),)
        _tig_bwd_fused_kernel[grid](
            weight, feat, grad_out, index.flat_j, index.flat_i, gx32, gw32,
            index.flat_seg, K, M, C, G,
            INPUT_PRECISION=input_precision, num_warps=w, **c)
    gw = gw32.to(feat.dtype)
    gw = (gw.view(weight_shape) if len(weight_shape) == 4
          else gw.view(K, C, M))
    return gw, gx32.to(feat.dtype)


class _TigMvmr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, feat, index, mode, input_precision):
        out = tig_forward(weight, feat, index, mode=mode,
                           input_precision=input_precision)
        ctx.save_for_backward(weight, feat)
        ctx.index = index
        ctx.precision = input_precision
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight, feat = ctx.saved_tensors
        index, prec = ctx.index, ctx.precision
        grad_w = grad_f = None
        # Measured verdict: split beats the fused one-pass backward at
        # EVERY real stage on sm_89 (the per-m-tile gx atomics dominate);
        # tig_backward_fused is retained for sm_90-class re-testing.
        if ctx.needs_input_grad[0]:
            grad_w = tig_grad_weight(feat, grad_out, index, weight.shape,
                                      input_precision=prec)
        if ctx.needs_input_grad[1]:
            grad_f = tig_grad_input(weight, grad_out, index,
                                     input_precision=prec)
        return grad_w, grad_f, None, None, None


def tig_mvmr(weight: torch.Tensor, feat: torch.Tensor, index: TigIndex,
              mode: str = "auto", input_precision: str = "tf32"):
    """Differentiable TIG mvmr: one autograd node for fwd + both grads
    (3 kernel launches total on the flat path, zero weight staging)."""
    return _TigMvmr.apply(weight, feat, index, mode, input_precision)
