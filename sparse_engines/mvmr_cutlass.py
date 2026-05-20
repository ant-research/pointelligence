"""Tier-2 CUTLASS skeleton mvmr (G22 plan §4 Task M1).

This is the *skeleton* path per the G22 plan
(autoresearch/threads/conv_extreme/3_plans/g22_cutlass_mvmr_closure_plan.md)
§4 Task-M1 GO/NO-GO — affine CUTLASS layouts only, no IndexedGather, no
scatter. Structurally mirrors vvor_cutlass.py's Task-1 surface; the only
semantic change is the contraction axis.

mvmr vs vvor (the one structural difference):
  - vvor (the template) contracts over seg_len (K). Both operands gathered.
  - mvmr contracts over CHANNELS (C). The weight W[k] is affine/dense
    (indexed by segment id k only — NOT gathered). input is gathered by
    b_idx. Output is scatter-accumulated by a_idx.

M1 exercises ONLY the contraction-axis-C GEMM core with a dense-write
epilogue. The Python wrapper PRE-GATHERS one tile's input rows into a
contiguous (S_TILE, C_seg) buffer and slices the affine (M_TILE, C_seg)
weight tile, then calls the single-tile op which performs the inner
(M_TILE, S_TILE, C_seg) GEMM via
`CollectiveMma<MainloopSm80CpAsyncUnpredicated>`. M2 replaces the
explicit input pre-gather with `make_gather_tensor`; M3 adds the outer
grid + the a_idx scatter-accumulate epilogue.

Public surface:
  mvmr_cutlass_sm80_single_tile_reference(...)
      A scalar-FMA fp32 reference using contiguous W_seg / B_seg — used
      for parity comparison against the CUTLASS path in the unit test.
  mvmr_cutlass_sm80_single_tile(...)
      The CUTLASS single-tile path. Caller pre-gathers + pads.
"""

import torch
from torch import Tensor

import sparse_engines_cuda._C  # noqa: F401 — load TORCH_LIBRARY init

from ._seg_offs import kernel_offset_segments


# Pinned to match Config::TileM / TileN / TileK in
# sparse_mvmr_cutlass_sm80.cuh. TileN plays the S_TILE (input-triplet
# row) role; TileK is the channel-contraction tile.
M_TILE = 64
S_TILE = 64
C_TILE = 32


def _pad_to_c_tile(x: Tensor, c_seg: int) -> tuple[Tensor, int]:
    """Pad x along its last dim (the C contraction axis) to a multiple
    of C_TILE with zeros."""
    pad = (-c_seg) % C_TILE
    if pad == 0:
        return x.contiguous(), c_seg
    padded = torch.nn.functional.pad(x, (0, pad))
    return padded.contiguous(), c_seg + pad


def stage_one_tile(
    weight: Tensor,
    input_b: Tensor,
    b_idx_seg: Tensor,
    m_start: int,
    c_start: int,
) -> tuple[Tensor, Tensor, int]:
    """Pre-gather a single (M_TILE, C_seg) / (S_TILE, C_seg) pair.

    weight:    (G=1, C_full, M_full) fp16 — the W[k] slice for one
               kernel-offset segment k. (Affine: indexed by k only;
               the caller already selected k. mvmr's authoritative
               weight layout is a[k] = (G, C, M); we squeeze G=1.)
    input_b:   (N_b, G=1, C_full) fp16
    b_idx_seg: (seg_len,) long input-row indices for this k-segment

    Returns:
      W_seg (M_TILE, C_seg_padded) fp16 row-major contig (C-contiguous),
      B_seg (S_TILE, C_seg_padded) fp16 row-major contig (C-contiguous),
      C_seg_padded (int).

    seg_len is clamped/padded to exactly S_TILE rows (M1 is a single
    fixed (M_TILE, S_TILE) tile, mirroring vvor Task-1's single-tile
    contract). The contraction axis C is what gets tiled by C_TILE.
    """
    assert weight.dim() == 3 and weight.size(0) == 1
    assert input_b.dim() == 3 and input_b.size(1) == 1
    assert weight.dtype == torch.float16
    assert input_b.dtype == torch.float16

    # Weight tile: W[k] is (G=1, C_full, M_full). Slice the (C_TILE-range,
    # M_TILE-range) face and lay it out (M_TILE, C_seg) C-contiguous so
    # the channel contraction axis is the inner contiguous dim (matches
    # the kernel's expected gmem layout — row-major (M, C), stride (C, 1)).
    w_2d = weight[0, c_start : c_start + C_TILE, m_start : m_start + M_TILE]
    # w_2d is (C_TILE, M_TILE); transpose to (M_TILE, C_TILE) C-contig.
    W_seg = w_2d.transpose(0, 1).contiguous()

    # Input tile: gather seg_len rows, slice the C-tile, → (S_TILE, C_TILE)
    # C-contiguous (the gathered triplet rows, padded/clamped to S_TILE).
    input_2d = input_b[:, 0, c_start : c_start + C_TILE]  # (N_b, C_TILE)
    B_gathered = torch.index_select(input_2d, 0, b_idx_seg.long())
    seg_len = int(b_idx_seg.numel())
    if seg_len < S_TILE:
        pad_rows = S_TILE - seg_len
        B_gathered = torch.nn.functional.pad(B_gathered, (0, 0, 0, pad_rows))
    elif seg_len > S_TILE:
        B_gathered = B_gathered[:S_TILE]
    B_seg = B_gathered.contiguous()

    C_seg = C_TILE
    W_seg_p, C_pad = _pad_to_c_tile(W_seg, C_seg)
    B_seg_p, _ = _pad_to_c_tile(B_seg, C_seg)
    return W_seg_p, B_seg_p, C_pad


def mvmr_cutlass_sm80_single_tile(
    W_seg_padded: Tensor,
    B_seg_padded: Tensor,
    C_seg_padded: int,
) -> Tensor:
    """Single-tile CUTLASS mvmr kernel.

    Returns the fp32 (M_TILE, S_TILE) output tile:
        O[m, s] = sum_c W_seg[m, c] * B_seg[s, c]
    Caller is responsible for pre-gather + pad alignment.
    """
    return torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_single_tile(
        W_seg_padded, B_seg_padded, C_seg_padded,
    )


def mvmr_cutlass_sm80_single_tile_reference(
    W_seg_padded: Tensor,
    B_seg_padded: Tensor,
) -> Tensor:
    """Scalar-FMA fp32 reference for parity:
        O[m, s] = sum_c W_seg[m, c] * B_seg[s, c].

    Computes in fp32 from fp16 inputs to match the CUTLASS path's fp32
    accumulator. Used only by the unit test.
    """
    W32 = W_seg_padded.float()                       # (M_TILE, C)
    B32 = B_seg_padded.float()                       # (S_TILE, C)
    # einsum mc, sc -> ms
    return W32 @ B32.transpose(0, 1)


# ─── Task M2 — kernel-side IndexedGather on the B (input) operand ─────────────
#
# M1 pre-gathers B Python-side (`stage_one_tile` does `input_b[b_idx]`).
# M2 moves that gather INSIDE the CUTLASS mainloop: the caller hands the
# kernel the raw `input_b` rows + the b_idx index buffer, and a composed
# `IndexedGather` custom-stride layout drives the S-axis gather inside the
# CollectiveMma cp.async loads. The affine W_seg tile is still produced
# Python-side (W is affine in mvmr — indexed by segment id k only).
#
# Unlike vvor Task 2 (which gathered along the contraction axis K and
# needed a transposing 2nd Config + ldmatrix-T), mvmr gathers along the
# S/triplet axis while the contraction axis C stays gmem-contiguous, so
# the M1 Config composes directly. See the .cu header for the full
# rationale.


def stage_w_tile(
    weight: Tensor,
    m_start: int,
    c_start: int,
) -> tuple[Tensor, int]:
    """Slice + pad the affine W[k] tile for the M2 gathered path.

    Identical W-side preparation to `stage_one_tile` (M1) — only B's
    gather is deferred to the kernel for M2, so this returns just the
    weight tile + padded C length.

    weight:  (G=1, C_full, M_full) fp16 — the W[k] slice for one
             kernel-offset segment k (caller already selected k).

    Returns:
      W_seg (M_TILE, C_seg_padded) fp16 row-major contig (C-contiguous),
      C_seg_padded (int).
    """
    assert weight.dim() == 3 and weight.size(0) == 1
    assert weight.dtype == torch.float16
    w_2d = weight[0, c_start : c_start + C_TILE, m_start : m_start + M_TILE]
    W_seg = w_2d.transpose(0, 1).contiguous()        # (M_TILE, C_TILE) C-contig
    W_seg_p, C_pad = _pad_to_c_tile(W_seg, C_TILE)
    return W_seg_p, C_pad


def clamp_b_idx_for_gather(b_idx_seg: Tensor) -> Tensor:
    """Clamp/pad b_idx to exactly S_TILE rows (int32 kernel ABI).

    seg_len < S_TILE  → pad with sentinel index 0 (the test mirrors this
                        so reference + kernel agree on the same padded-S
                        layout — the padded-slot product is non-zero in
                        general, applied symmetrically on both sides).
    seg_len > S_TILE  → truncate to the first S_TILE rows.
    """
    seg_len = int(b_idx_seg.numel())
    if seg_len < S_TILE:
        zero = torch.zeros(
            S_TILE - seg_len, dtype=b_idx_seg.dtype, device=b_idx_seg.device
        )
        b_idx_seg = torch.cat([b_idx_seg, zero], dim=0)
    elif seg_len > S_TILE:
        b_idx_seg = b_idx_seg[:S_TILE]
    return b_idx_seg.to(dtype=torch.int32, copy=False).contiguous()


def mvmr_cutlass_sm80_single_tile_gathered(
    W_seg_padded: Tensor,        # (M_TILE, C_seg_padded) fp16 contig (affine)
    input_b: Tensor,            # (N_b, 1, C_full) or (N_b, C_full) fp16 contig
    b_idx_seg: Tensor,          # (S_TILE,) int32 — gathered input-row indices
    c_start: int,
    C_seg_padded: int,
) -> Tensor:
    """Task-M2 entry: composed IndexedGather on B inside the CUTLASS mainloop.

    W is affine (pre-sliced + padded by `stage_w_tile`); B is read directly
    from `input_b` and gathered along the S axis by `b_idx_seg` inside the
    CollectiveMma cp.async loads. Returns the fp32 (M_TILE, S_TILE) tile:
        O[m, s] = sum_c W_seg[m, c] * input_b[b_idx_seg[s], c_start + c]

    Caller is responsible for padding C_seg up to a TileK multiple and
    clamping/padding b_idx_seg to exactly S_TILE rows
    (use `stage_w_tile` + `clamp_b_idx_for_gather`).
    """
    return torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_single_tile_gathered(
        W_seg_padded, input_b, b_idx_seg, int(c_start), int(C_seg_padded),
    )


def mvmr_cutlass_sm80_single_tile_gathered_reference(
    W_seg_padded: Tensor,
    input_b: Tensor,
    b_idx_seg: Tensor,
    c_start: int,
) -> Tensor:
    """Scalar reference for the Task-M2 entrypoint.

    Gathers B Python-side then does the same fp32 `W @ B_gathered^T` the
    M1 reference does. Mirrors the kernel's index/shape contract:
    b_idx_seg is int32 of length S_TILE; padded slots gather real input
    rows (handled identically on both sides). Used only by the unit test.
    """
    i2d = input_b.select(1, 0) if input_b.dim() == 3 else input_b
    C_pad = int(W_seg_padded.size(1))
    # Gather S_TILE rows, slice the padded C range → (S_TILE, C_pad).
    B_gathered = torch.index_select(
        i2d[:, c_start : c_start + C_pad], 0, b_idx_seg.long()
    ).float()
    W32 = W_seg_padded.float()                       # (M_TILE, C_pad)
    # O[m, s] = sum_c W[m, c] * B_gathered[s, c]
    return W32 @ B_gathered.transpose(0, 1)


# ─── Task M3 — full mvmr op (outer ragged-K-segment grid + scatter epilogue) ──


def sparse_matrix_vector_multiplication_reduction_cutlass(
    a: Tensor,        # (K_offsets, G=1, C, M) fp16 — affine weight W[k]
    a_idx: Tensor,    # (T,) int — kernel-offset segment id per triplet, sorted asc
    b: Tensor,        # (N_b, G=1, C) fp16 — input rows
    b_idx: Tensor,    # (T,) int — input-row idx into b
    o_idx: Tensor,    # (T,) int — output-row idx into o
    n_o: int,         # number of output points (o's leading dim)
    seg_offs: Tensor | None = None,  # G24-T3: pre-built (K+1,) int64 seg_offs
) -> Tensor:
    """Full mvmr forward via the Tier-2 CUTLASS path (G22 plan §4 Task M3).

    Drop-in replacement for the Triton-grouped
    ``sparse_matrix_vector_multiplication_reduction``: same call
    signature, same ``(n_o, G=1, M)`` fp32-accumulated output. One CUDA
    CTA per (M-tile, k-segment); the CTA loops over S-chunks of its
    segment, gathers the chunk's input rows on the S axis via a composed
    IndexedGather (clamped to the segment via a sentinel zero row), runs
    the (M_TILE, S_TILE, C) CUTLASS GEMM, and scatter-accumulates each
    result column into ``o[o_idx[t]]`` via atomicAdd with prev_out
    run-length coalescing.

    Preconditions (shared with the other grouped paths):
      - a_idx sorted ascending (sort_by="k")
      - G == 1
      - fp16 inputs (fp32 / bf16 not supported by this Tier-2 path)
      - M and C multiples of the kernel tile (TileM=64, TileK=32)

    G24-T3 (S1 + S4 elimination):
      - S1: the per-call ``a = a.contiguous()`` is removed. The C++ host
        fn already resolves any input striding via its own
        ``select(1,0).transpose(1,2).contiguous()`` repack (S2), so the
        Python pre-contiguous was pure redundancy. On the grad_b second
        call ``a`` arrives as ``W.transpose(2,3)`` (non-contiguous); the
        deleted ``.contiguous()`` was a *full transposed-weight
        materialization* on top of S2's copy — eliminating it removes
        one of the two per-grad_b full-tensor copies that T1's profile
        flagged as half the dominant ~170 µs DirectCopy.
      - S4: ``seg_offs`` may be passed in pre-built (computed once at the
        autograd.Function boundary and shared between the fwd and the
        grad_b second call — ``a_idx``/``K_offsets`` are the fixed
        triplet structure, invariant across both). When ``None`` it is
        built here as before (non-autograd / "auto"-mode callers).
    """
    b = b.contiguous()

    K_offsets = a.shape[0]
    G = a.shape[1]
    C = a.shape[2]
    M = a.shape[3]

    if G != 1:
        raise ValueError("CUTLASS full mvmr requires G == 1")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError(
            "CUTLASS full mvmr is fp16-only "
            f"(got a={a.dtype}, b={b.dtype})"
        )
    if M % M_TILE != 0 or C % C_TILE != 0:
        raise ValueError(
            f"CUTLASS full mvmr requires M % {M_TILE} == 0 and "
            f"C % {C_TILE} == 0; got M={M}, C={C}"
        )
    if not bool((a_idx[1:] >= a_idx[:-1]).all().item()):
        raise ValueError("a_idx must be sorted ascending for grouped path")

    if seg_offs is None:
        seg_offs = kernel_offset_segments(a_idx, int(K_offsets)).to(torch.int64)
    else:
        seg_offs = seg_offs.to(torch.int64)
    b_idx_i32 = b_idx.to(torch.int32)
    o_idx_i32 = o_idx.to(torch.int32)

    # Arch dispatch (mirrors vvor_cutlass.py): route Hopper (sm_90+)
    # hardware to the sm_90-targeted op (G22 plan §4 Task M6 P1). The two
    # ops are algorithmically identical (same frozen Sm80 cp.async-
    # Unpredicated + M2 S-gather + scatter-accumulate); the sm_90 symbol
    # exists so the H200 cell exercises the sm_90 SASS path. On sm_80/89
    # the sm_80 op stays the path of record.
    major = torch.cuda.get_device_capability(a.device)[0]
    if major >= 9:
        return torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm90_full(
            a, b_idx_i32, b, o_idx_i32, seg_offs, int(n_o),
        )

    return torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full(
        a, b_idx_i32, b, o_idx_i32, seg_offs, int(n_o),
    )


# ─── G26-T2(b) + G25-T2 — Python-only fused PointConv3d autograd.Function ─────
#
# Scope (cycle_7_g26_t2_rescope.md option (b); plan §2/§4 Phase-A T2):
# collapse the eager PointConv3d fwd+bwd composition — 3 @triton_op /
# autograd-graph op boundaries + 2 seg_offs builds + the duplicate Python
# `.contiguous()` calls — into ONE autograd.Function, reusing the frozen
# CUTLASS mvmr/vvor full kernels AS-IS (no .cu/.cuh edit). Wins, all
# Python/autograd-side:
#
#   (1) G25-T2: fwd + grad_b both feed the C++ `_full_prestaged` host
#       entry (G25-T1, commit 9de9051) with a caller-pre-staged
#       (n_o_k, M_full, C_full)-C-contiguous buffer, which SKIPS `_full`'s
#       unconditional internal `a.select(1,0).transpose(1,2).contiguous()`
#       repack (sm80.cu:543-545 / sm90.cu:216-218). Each side stages its
#       buffer EXACTLY ONCE:
#         fwd:    aT_fwd   = weight.select(1,0).transpose(1,2).contiguous()
#                           → (K, M_w, C_w) C-contig
#         grad_b: aT_gradb = weight.select(1,0).contiguous()
#                           → (K, C_w, M_w) C-contig (the `.transpose(1,2)`
#                             the fwd stage applies is DROPPED — G25-T1
#                             verified this is byte-equal to `_full`'s
#                             internal repack of `weight.transpose(2,3)`).
#       The kernel only ever reads a (K, M_full, C_full)-C-contig buffer;
#       fwd-vs-grad_b is purely which weight axes the caller maps. This
#       RETIRES the G26-T2(b) no-op-collapse 4-D-view trick (which fed the
#       frozen `_full` and relied on `host.contiguous() is host`) AND
#       eliminates the grad_b-S2 host-repack-inside-`_full` (T2(b)'s
#       named, un-subsumed residual): the single per-side copy now happens
#       at the controlled stage, with NO internal `_full` repack on top.
#   (2) One Function instead of 3 @triton_op/autograd-graph boundaries;
#       `seg_offs` built ONCE in forward, saved on ctx, reused in backward
#       (no rebuild); no per-call Python `.contiguous()` on the staged path.
#
# grad_a: frozen CUTLASS vvor full, reused exactly as the eager
#   `_backward_sparse_matrix_vector_multiplication_reduction` does
#   (UNCHANGED by G25-T2).
# grad_b: frozen CUTLASS mvmr full via `_full_prestaged` fed
#   `weight.select(1,0).contiguous()` + the saved `seg_offs` (the
#   transpose dropped vs fwd; see (1)). The grad_b-S2 host-repack is now
#   ELIMINATED (was the named, un-subsumed G25/T-S2 residual).


class FusedPointConv3d(torch.autograd.Function):
    """Fused PointConv3d fwd+bwd (G26-T2(b), Python-only).

    Collapses the eager mvmr-fwd / vvor-grad_a / mvmr-grad_b composition
    into one Function reusing the frozen CUTLASS full kernels. fp16 / G==1
    / sorted-a_idx / M,C tile-multiple — same hard preconditions as the
    underlying `sparse_matrix_vector_multiplication_reduction_cutlass` /
    `..._grouped_cutlass` entry points, which raise on violation.

    Inputs mirror `sparse_matrix_vector_multiplication_reduction`:
      weight (K, G=1, C, M) fp16, a_idx (T,) sorted asc, input_3d
      (N, G=1, C) fp16, b_idx (T,), o_idx (T,), n_o int.
    """

    @staticmethod
    def forward(ctx, weight, a_idx, input_3d, b_idx, o_idx, n_o):
        K_offsets = weight.shape[0]

        # ── G25-T2: single S2 stage, fed straight to `_full_prestaged`.
        # weight (K,1,C,M) → aT_fwd (K,M,C) C-contiguous — exactly the
        # (n_o_k, M_full, C_full) layout `_full_prestaged` consumes (it
        # skips `_full`'s unconditional internal
        # `select(1,0).transpose(1,2).contiguous()` repack by contract).
        # This stage is THE one weight copy for the fwd side; amortized
        # over fwd + grad_a. The G26-T2(b) no-op-collapse 4-D-view trick
        # that fed the frozen `_full` is RETIRED — `_prestaged` consumes
        # the staged buffer directly (no host-side repack on top of it).
        aT_fwd = weight.select(1, 0).transpose(1, 2).contiguous()

        # ── seg_offs built ONCE; reused by fwd host fn and grad_b. a_idx +
        # K_offsets are the fixed triplet structure, invariant fwd↔grad_b.
        seg_offs = kernel_offset_segments(a_idx, int(K_offsets)).to(torch.int64)

        b = input_3d.contiguous()
        b_idx_i32 = b_idx.to(torch.int32)
        o_idx_i32 = o_idx.to(torch.int32)

        major = torch.cuda.get_device_capability(weight.device)[0]
        if major >= 9:
            out = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm90_full_prestaged(
                aT_fwd, b_idx_i32, b, o_idx_i32, seg_offs, int(n_o),
            )
        else:
            out = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full_prestaged(
                aT_fwd, b_idx_i32, b, o_idx_i32, seg_offs, int(n_o),
            )
        out = out.to(weight.dtype) if out.dtype != weight.dtype else out

        ctx.save_for_backward(weight, a_idx, input_3d, b_idx, o_idx, seg_offs)
        ctx.n_o = int(n_o)
        ctx.K_offsets = int(K_offsets)
        ctx.b_shape_0 = input_3d.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight, a_idx, input_3d, b_idx, o_idx, seg_offs = ctx.saved_tensors
        grad_weight = grad_b = None

        grad_out = grad_out.contiguous()

        if ctx.needs_input_grad[0]:
            # grad_a (grad of weight) via the frozen CUTLASS vvor full
            # kernel — exactly the eager mvmr-backward's
            #   vvor(input_3d, b_idx, grad, o_idx, a_idx, K_offsets)
            # routing, reused as-is. Returns (K, 1, C, M) matching the
            # native weight layout.
            from .vvor_cutlass import (
                sparse_vector_vector_outer_product_reduction_grouped_cutlass,
            )
            grad_weight = (
                sparse_vector_vector_outer_product_reduction_grouped_cutlass(
                    input_3d, b_idx, grad_out, o_idx, a_idx, ctx.K_offsets,
                )
            )
            if grad_weight.dtype != weight.dtype:
                grad_weight = grad_weight.to(weight.dtype)

        if ctx.needs_input_grad[2]:
            # grad_b (grad of input) via the frozen CUTLASS mvmr full
            # kernel. G25-T2: route through `_full_prestaged` with a
            # caller-staged buffer instead of through `_full`
            # (`sparse_matrix_vector_multiplication_reduction_cutlass`)
            # fed `weight.transpose(2,3)`. The kernel only ever reads a
            # (n_o_k, M_full, C_full)-C-contiguous buffer; fwd-vs-grad_b
            # is purely which weight axes the caller maps onto M/C. For
            # grad_b that buffer is `weight.select(1,0).contiguous()`
            # ((K,1,C,M) → (K,C,M) C-contig) — the `.transpose(1,2)` the
            # fwd stage applies is DROPPED (G25-T1 verified this is
            # byte-equal to `_full`'s internal repack of
            # `weight.transpose(2,3)`). This ELIMINATES the grad_b-S2
            # host-repack-inside-`_full`: the single copy now happens at
            # this controlled stage, with NO internal `_full` repack on
            # top of it (vs T2(b): a strided transposed-weight view fed
            # to `_full`, whose unconditional internal
            # select/transpose/.contiguous() did the repack). seg_offs is
            # reused (not rebuilt); grad_a (vvor) path unchanged.
            aT_gradb = weight.select(1, 0).contiguous()
            b_idx_i32 = b_idx.to(torch.int32)
            o_idx_i32 = o_idx.to(torch.int32)
            major = torch.cuda.get_device_capability(weight.device)[0]
            if major >= 9:
                grad_b = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm90_full_prestaged(
                    aT_gradb, o_idx_i32, grad_out, b_idx_i32, seg_offs,
                    ctx.b_shape_0,
                )
            else:
                grad_b = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full_prestaged(
                    aT_gradb, o_idx_i32, grad_out, b_idx_i32, seg_offs,
                    ctx.b_shape_0,
                )
            if grad_b.dtype != weight.dtype:
                grad_b = grad_b.to(weight.dtype)

        return grad_weight, None, grad_b, None, None, None
