"""Tier-2 CUTLASS skeleton vvor.

This is the *skeleton* path — affine CUTLASS layouts only, no
IndexedGather. The Python wrapper PRE-GATHERS one (k, mt, ct) segment's
rows from grad_output / input into contiguous (M_TILE, K_seg) and
(N_TILE, K_seg) buffers, then calls the single-tile op which performs
the inner (M_TILE, N_TILE, K_seg) GEMM via
`CollectiveMma<MainloopSm80CpAsyncUnpredicated>`.

The gathered variant replaces the explicit pre-gather with
`make_gather_tensor` composed-layout inside cp.async loads. The full op
adds the outer (k, mt, ct) grid scheduler.

Public surface:
  vvor_cutlass_sm80_single_tile_reference(...)
      A scalar-FMA reference using contiguous A_seg / B_seg — used for
      parity comparison against the CUTLASS path in the unit test.
  vvor_cutlass_sm80_single_tile(...)
      The CUTLASS single-tile path.  Caller pre-gathers + pads.
"""

import torch
from torch import Tensor

import sparse_engines_cuda._C  # noqa: F401 — load TORCH_LIBRARY init

from ._seg_offs import kernel_offset_segments
from .mvmr_cutlass import (
    _INT32_LIM,
    _folded_row_idx,
    _folded_seg_offs,
)


# Escape hatch / test hook for the fold-G-into-K lever (mirrors
# mvmr_cutlass._FOLD_G_ENABLED): False forces the per-group loop
# fallback (the same path the int32-overflow guard takes).
_FOLD_G_ENABLED = True


# Pinned to match Config::TileM / TileN / TileK in
# sparse_vvor_cutlass_sm80.cuh. Keeping these in one place means the
# Python wrapper and test can stay in sync with kernel-side tile choices
# without re-grepping the .cuh.
M_TILE = 64
N_TILE = 64
K_TILE = 32


def _pad_to_k_tile(x: Tensor, k_seg: int) -> tuple[Tensor, int]:
    """Pad x along its last dim to a multiple of K_TILE with zeros."""
    pad = (-k_seg) % K_TILE
    if pad == 0:
        return x.contiguous(), k_seg
    padded = torch.nn.functional.pad(x, (0, pad))
    return padded.contiguous(), k_seg + pad


def stage_one_tile(
    grad_output: Tensor,
    input_b: Tensor,
    i_idx_seg: Tensor,
    j_idx_seg: Tensor,
    m_start: int,
    c_start: int,
) -> tuple[Tensor, Tensor, int]:
    """Pre-gather a single (M_TILE, K_seg) / (N_TILE, K_seg) pair.

    grad_output: (N_o_points, G=1, M_full) fp16
    input_b:     (N_b,       G=1, C_full) fp16
    i_idx_seg:   (seg_len,)  long output-row indices for this k-segment
    j_idx_seg:   (seg_len,)  long input-row indices for this k-segment

    Returns:
      A_seg (M_TILE, K_seg_padded) fp16 row-major contig (K-contiguous),
      B_seg (N_TILE, K_seg_padded) fp16 row-major contig,
      K_seg_padded (int).
    """
    assert grad_output.dim() == 3 and grad_output.size(1) == 1
    assert input_b.dim() == 3 and input_b.size(1) == 1
    assert grad_output.dtype == torch.float16
    assert input_b.dtype == torch.float16

    # Slice the M-tile / C-tile from the channel axis FIRST so we
    # gather only what the kernel needs (M_TILE / N_TILE columns).
    grad_out_2d = grad_output[:, 0, m_start : m_start + M_TILE]  # (N_o, M_TILE)
    input_2d    = input_b   [:, 0, c_start : c_start + N_TILE]   # (N_b, N_TILE)

    # Gather along the row (N) axis using segment indices.
    # `index_select` gives a contiguous (seg_len, M_TILE) tensor.
    A_gathered = torch.index_select(grad_out_2d, 0, i_idx_seg.long())
    B_gathered = torch.index_select(input_2d,   0, j_idx_seg.long())

    # Transpose to (M_TILE, K_seg) row-major so K is the contiguous dim
    # (matches the kernel's expected gmem layout — `make_tensor(...,
    # make_layout((M, K), (K, 1)))` row-major).
    A_seg = A_gathered.transpose(0, 1).contiguous()
    B_seg = B_gathered.transpose(0, 1).contiguous()

    K_seg = int(i_idx_seg.numel())
    A_seg_p, K_pad = _pad_to_k_tile(A_seg, K_seg)
    B_seg_p, _     = _pad_to_k_tile(B_seg, K_seg)
    return A_seg_p, B_seg_p, K_pad


def vvor_cutlass_sm80_single_tile(
    A_seg_padded: Tensor,
    B_seg_padded: Tensor,
    K_seg_padded: int,
) -> Tensor:
    """Single-tile CUTLASS vvor kernel.

    Returns the fp32 (M_TILE, N_TILE) grad_weight tile for the staged
    segment.  Caller is responsible for pre-gather + pad alignment.
    """
    return torch.ops.sparse_engines_cuda.sparse_vvor_cutlass_sm80_single_tile(
        A_seg_padded, B_seg_padded, K_seg_padded,
    )


def vvor_cutlass_sm80_single_tile_reference(
    A_seg_padded: Tensor,
    B_seg_padded: Tensor,
) -> Tensor:
    """Scalar fp32 reference for parity:  C[m, n] = sum_k A[m, k] * B[n, k].

    Computes in fp32 from fp16 inputs to match the CUTLASS path's fp32
    accumulator.  Used only by the unit test.
    """
    A32 = A_seg_padded.float()                       # (M_TILE, K)
    B32 = B_seg_padded.float()                       # (N_TILE, K)
    # einsum mk, nk -> mn
    return A32 @ B32.transpose(0, 1)


# ─── Kernel-side IndexedGather (K-mode) ───────────────────────────────────────


def pad_indices_for_gather(
    i_idx_seg: Tensor, j_idx_seg: Tensor, k_seg: int,
) -> tuple[Tensor, Tensor, int]:
    """Pad i_idx / j_idx with sentinel index 0 up to a multiple of K_TILE.

    Returns int32 buffers (kernel ABI). Padded slots use index 0 — the
    test mirrors this so reference + kernel agree on the same padded
    K layout (the padded contribution is non-zero in general, so it
    must be applied symmetrically on both sides).
    """
    assert i_idx_seg.numel() == k_seg and j_idx_seg.numel() == k_seg
    pad = (-k_seg) % K_TILE
    if pad > 0:
        zero = torch.zeros(pad, dtype=i_idx_seg.dtype, device=i_idx_seg.device)
        i_idx_seg = torch.cat([i_idx_seg, zero], dim=0)
        j_idx_seg = torch.cat([j_idx_seg, zero], dim=0)
    return (
        i_idx_seg.to(dtype=torch.int32, copy=False).contiguous(),
        j_idx_seg.to(dtype=torch.int32, copy=False).contiguous(),
        k_seg + pad,
    )


def vvor_cutlass_sm80_single_tile_gathered(
    grad_output: Tensor,         # (N_o, 1, M_full) or (N_o, M_full) fp16 contig
    input_b:     Tensor,         # (N_b, 1, C_full) or (N_b, C_full) fp16 contig
    i_idx_seg:   Tensor,         # (K_seg_padded,) int32 — output-row indices
    j_idx_seg:   Tensor,         # (K_seg_padded,) int32 — input-row indices
    m_start:     int,
    c_start:     int,
    k_seg_padded: int,
) -> Tensor:
    """Gathered entry: composed IndexedGather inside the CUTLASS mainloop.

    Reads grad_output / input directly (no Python pre-gather) and lets the
    `make_gather_tensor` composed layout drive K-mode gather inside the
    CollectiveMma's cp.async loads. Returns the fp32 (M_TILE, N_TILE) tile.

    Caller is responsible for padding i_idx / j_idx to k_seg_padded
    (use `pad_indices_for_gather`).
    """
    return torch.ops.sparse_engines_cuda.sparse_vvor_cutlass_sm80_single_tile_gathered(
        grad_output, input_b, i_idx_seg, j_idx_seg,
        int(m_start), int(c_start), int(k_seg_padded),
    )


def vvor_cutlass_sm80_single_tile_gathered_reference(
    grad_output: Tensor,
    input_b:     Tensor,
    i_idx_seg:   Tensor,
    j_idx_seg:   Tensor,
    m_start:     int,
    c_start:     int,
) -> Tensor:
    """Scalar reference for the gathered entrypoint.

    Computes (M_TILE, N_TILE) fp32 tile by gathering on the Python side
    and doing the matmul in fp32 (matches the CUTLASS fp32 accumulator).
    Used only by the unit test. Same shape/index assumptions as the
    kernel: i_idx_seg / j_idx_seg are int32 of length k_seg_padded;
    padded slots contribute (the test handles padding the same way
    the kernel does).
    """
    g2d = grad_output.select(1, 0) if grad_output.dim() == 3 else grad_output
    i2d = input_b   .select(1, 0) if input_b   .dim() == 3 else input_b
    # (K, M_TILE) and (K, N_TILE)
    A_gathered = torch.index_select(
        g2d[:, m_start : m_start + M_TILE], 0, i_idx_seg.long()
    )
    B_gathered = torch.index_select(
        i2d[:, c_start : c_start + N_TILE], 0, j_idx_seg.long()
    )
    # C[m, n] = sum_k A_gathered[k, m] * B_gathered[k, n]
    # = A_gathered.T @ B_gathered  →  (M_TILE, K) @ (K, N_TILE)
    return A_gathered.transpose(0, 1).float() @ B_gathered.float()


# ─── Full vvor backward (outer (k, mt, ct) grid scheduler) ────────────────────


def sparse_vector_vector_outer_product_reduction_grouped_cutlass(
    a: Tensor,        # (N_a, G=1, M) fp16 — grad_output rows
    a_idx: Tensor,    # (T,) int — output-row indices into a
    b: Tensor,        # (N_b, G=1, C) fp16 — input rows
    b_idx: Tensor,    # (T,) int — input-row indices into b
    o_idx: Tensor,    # (T,) int — kernel-offset index per triplet, sorted asc
    n_o: int,         # number of kernel offsets (K_offsets) = grad_weight k-dim
) -> Tensor:
    """Full vvor backward via the Tier-2 CUTLASS path.

    Drop-in replacement for
    ``sparse_vector_vector_outer_product_reduction_grouped_wmma_coop`` /
    ``..._grouped_cuda``: same call signature, same ``(n_o, G=1, M, C)``
    grad_weight output. One CUDA CTA per (k-segment, M-tile, C-tile); the
    kernel gathers triplet rows on the K axis via a composed IndexedGather
    layout and uses a K-axis-predicated ``CollectiveMma`` so arbitrary
    (incl. empty) segment lengths need no index padding.

    Preconditions (shared with the other grouped paths):
      - o_idx sorted ascending (sort_by="k")
      - G == 1
      - fp16 OR bf16 inputs, both operands the same dtype (the
        SM80_16x8x16_F32BF16BF16F32_TN atom path handles bf16). fp32 is
        NOT supported — there is no SM80 tensor-core atom of this shape for
        true fp32 (only TF32/BF16/FP16); fp32 conv stays on the Triton path.
      - M and C multiples of the kernel tile (TileM=TileN=64)
    """
    G = a.shape[1]
    M = a.shape[2]
    C = b.shape[2]

    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "CUTLASS full vvor is fp16/bf16-only, both operands same dtype "
            f"(got a={a.dtype}, b={b.dtype})"
        )
    # Tile gates apply PER GROUP — M / C here are the per-group channel
    # counts (grad_output is (N, G, Mg), input (N, G, Cg)).
    if M % M_TILE != 0 or C % N_TILE != 0:
        raise ValueError(
            f"CUTLASS full vvor requires per-group M % {M_TILE} == 0 and "
            f"per-group C % {N_TILE} == 0; got per-group M={M}, C={C} (G={G})"
        )
    if not bool((o_idx[1:] >= o_idx[:-1]).all().item()):
        raise ValueError("o_idx must be sorted ascending for grouped path")

    seg_offs = kernel_offset_segments(o_idx, int(n_o))
    seg_offs_i64 = seg_offs.to(torch.int64)

    # Arch dispatch: route Hopper (sm_90+) hardware to the sm_90-targeted
    # op. The two ops are algorithmically identical (same Sm80
    # cp.async-Unpredicated + sentinel-zero-row); the sm_90 symbol exists
    # so sm_90 hardware exercises the sm_90 SASS path. On sm_80/89 the
    # sm_80 op stays the path of record.
    major = torch.cuda.get_device_capability(a.device)[0]
    if major >= 9:
        op = torch.ops.sparse_engines_cuda.sparse_vvor_cutlass_sm90_full
    else:
        op = torch.ops.sparse_engines_cuda.sparse_vvor_cutlass_sm80_full

    if G == 1:
        a_idx_i32 = a_idx.to(torch.int32)
        b_idx_i32 = b_idx.to(torch.int32)
        a = a.contiguous()
        b = b.contiguous()
        return op(a, a_idx_i32, b, b_idx_i32, seg_offs_i64, int(n_o))

    # ── G > 1: fold G into the kernel-offset axis — ONE launch of the
    # frozen G==1 kernel (the per-group loop is retained in
    # `_vvor_groups_loop` below as the int32-overflow fallback).
    # Verified BITWISE-equal to the loop on real ScanNet scenes (the
    # vvor kernel is deterministic per segment — no atomics); c256/G=4
    # fp16 0.398 → 0.128 ms, c512 0.303 → 0.078 ms with cached indices.
    # Layout (mirrors mvmr_cutlass's fold; same cached index helpers):
    #   grad_out (N_a, G, Mg) → (G*N_a, 1, Mg)   (row g*N_a+i = group g's row i)
    #   input    (N_b, G, Cg) → (G*N_b, 1, Cg)
    #   a_idx' = concat_g(a_idx + g*N_a), b_idx' = concat_g(b_idx + g*N_b)
    #   seg_offs' = concat_g(g*T + seg_offs[:-1]) ++ [G*T]
    # Output (G*n_o, 1, Mg, Cg) → unfold → (n_o, G, Mg, Cg), identical
    # shape/dtype contract as the loop's cat.
    n_a = a.shape[0]
    n_b = b.shape[0]
    T = int(a_idx.numel())
    if (
        _FOLD_G_ENABLED
        and G * n_a < _INT32_LIM
        and G * n_b < _INT32_LIM
        and G * T < _INT32_LIM
    ):
        af = a.permute(1, 0, 2).reshape(G * n_a, 1, M).contiguous()
        bf = b.permute(1, 0, 2).reshape(G * n_b, 1, C).contiguous()
        a_fold = _folded_row_idx(a_idx, n_a, G)
        b_fold = _folded_row_idx(b_idx, n_b, G)
        seg_fold = _folded_seg_offs(seg_offs_i64, o_idx, T, G)
        out = op(af, a_fold, bf, b_fold, seg_fold, G * int(n_o))
        return out.view(G, int(n_o), M, C).permute(1, 0, 2, 3).contiguous()

    # Fold would overflow the int32 index ABI (or fold disabled) — take
    # the per-group loop.
    return _vvor_groups_loop(
        a, b, a_idx.to(torch.int32), b_idx.to(torch.int32), seg_offs_i64,
        int(n_o), G, op,
    )


def _vvor_groups_loop(
    a: Tensor,
    b: Tensor,
    a_idx_i32: Tensor,
    b_idx_i32: Tensor,
    seg_offs_i64: Tensor,
    n_o: int,
    G: int,
    op,
) -> Tensor:
    """G > 1 wrapper-level per-group loop.

    Pre-fold path of record, retained as the fallback when the
    fold-G-into-K lever would overflow the kernels' int32 index ABI
    (G*N or G*T ≥ 2^31), and as the reference arm for the fold parity
    tests. The frozen `_full` host fn hard-TORCH_CHECKs G == 1 on both
    operands, so groups are block-diagonal-decomposed into G
    independent G==1 calls: group g uses grad_output[:, g],
    input[:, g], writes grad_weight[:, g]. Index tensors + seg_offs are
    GROUP-INVARIANT — built once by the caller, shared by every
    per-group call.

    Repack accounting (honest count): ONE batched
    `permute(1,0,2).contiguous()` per operand — (N, G, Cg) →
    (G, N, Cg) — so each per-group slice is already (N, 1, Cg)-
    contiguous and the host fn's internal `.contiguous()` calls are
    no-ops. Same total bytes as G==1's two `.contiguous()` calls, one
    launch each instead of G. Structural handicap that remains: G
    kernel launches + the final cat — quantified in the
    grouped-operator benchmarks under benchmarks/operators/.

    NOTE (measured negative): mvmr_cutlass's side-stream overlap was
    tried here too and REVERTED — the per-group vvor kernels are ~0.1 ms
    (vs mvmr's ~0.9 ms), so the event/record/wait plumbing costs more
    than the overlap recovers (re-measured: c256/G=4 0.391 → 0.368 ms —
    within noise; c512/G=4 0.289 → 0.354 ms — clear loss). Plain
    sequential loop is the optimum at this size.
    """
    a_perm = a.permute(1, 0, 2).contiguous()
    b_perm = b.permute(1, 0, 2).contiguous()
    outs = [
        op(
            a_perm[g].unsqueeze(1), a_idx_i32, b_perm[g].unsqueeze(1),
            b_idx_i32, seg_offs_i64, int(n_o),
        )
        for g in range(G)
    ]
    return torch.cat(outs, dim=1)  # (n_o, G, M, C)
