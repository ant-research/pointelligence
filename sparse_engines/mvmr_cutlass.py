"""Tier-2 CUTLASS skeleton mvmr.

This is the *skeleton* path — affine CUTLASS layouts only, no
IndexedGather, no scatter. Structurally mirrors vvor_cutlass.py's
single-tile surface; the only semantic change is the contraction axis.

mvmr vs vvor (the one structural difference):
  - vvor (the template) contracts over seg_len (K). Both operands gathered.
  - mvmr contracts over CHANNELS (C). The weight W[k] is affine/dense
    (indexed by segment id k only — NOT gathered). input is gathered by
    b_idx. Output is scatter-accumulated by a_idx.

The skeleton exercises ONLY the contraction-axis-C GEMM core with a
dense-write epilogue. The Python wrapper PRE-GATHERS one tile's input
rows into a contiguous (S_TILE, C_seg) buffer and slices the affine
(M_TILE, C_seg) weight tile, then calls the single-tile op which
performs the inner (M_TILE, S_TILE, C_seg) GEMM via
`CollectiveMma<MainloopSm80CpAsyncUnpredicated>`. The gathered variant
replaces the explicit input pre-gather with `make_gather_tensor`; the
full op adds the outer grid + the a_idx scatter-accumulate epilogue.

Public surface:
  mvmr_cutlass_sm80_single_tile_reference(...)
      A scalar-FMA fp32 reference using contiguous W_seg / B_seg — used
      for parity comparison against the CUTLASS path in the unit test.
  mvmr_cutlass_sm80_single_tile(...)
      The CUTLASS single-tile path. Caller pre-gathers + pads.
"""

import weakref
from collections import OrderedDict

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


# ── G>1 per-group side streams ────────────────────────────────────────────────
# The frozen CUTLASS `_full` kernels parallelize over (M-tiles ×
# k-segments) only, so a per-group call at Cg=Mg=64 launches a mere
# K_offsets CTAs (27 for a 3³ kernel) — heavily under-occupied. The G
# independent group calls are therefore issued on dedicated side streams
# and re-joined to the caller's stream, overlapping the under-occupied
# kernels (measured 3.3× on c256/G=4 fp16).
# Streams are created lazily per device and reused across calls. Shared
# by vvor_cutlass.py (imported from here).
_GROUP_STREAMS: dict = {}


def _group_streams(device: torch.device, G: int) -> list:
    key = torch.cuda.current_device() if device.index is None else device.index
    streams = _GROUP_STREAMS.setdefault(key, [])
    while len(streams) < G:
        streams.append(torch.cuda.Stream(device=key))
    return streams[:G]


# ── G>1 fold-G-into-K_offsets (single-launch lever) ──────────────────────────
# The frozen CUTLASS `_full` host fns accept arbitrary K_offsets /
# seg_offs / n_o, so a G>1 block-diagonal call can be laid out as
# G*K_offsets independent kernel-offset segments in ONE launch instead of
# G per-group launches (G× more CTAs — directly fixes the
# under-occupancy the per-group loop only papered over with side
# streams). The replicated index buffers are GROUP-INVARIANT
# functions of the triplet structure:
#   row_idx'  = concat_g(row_idx + g*n_rows)            (T*G,)  int32
#   seg_offs' = concat_g(g*T + seg_offs[:-1]) ++ [G*T]  (G*K+1,) int64
# so they are cached across calls (training steps, and the fwd/grad_b
# pair which share the same a_idx/b_idx/o_idx structure) — same sharing
# precedent as the seg_offs ctx-share in mvmr_triton.py.
#
# Cache key per folded buffer:
#   (kind, src.data_ptr(), src._version, src.numel(), src.dtype,
#    fold params…, device)
# and each entry stores a `weakref` to the source tensor. A hit is valid
# only if (a) the key matches AND (b) the entry's original source tensor
# is STILL ALIVE with the same data_ptr/_version. `_version` invalidates
# on any IN-PLACE write to the source index tensor; the weakref-liveness
# check closes the recycled-data_ptr hole (a *different* tensor freshly
# allocated at a freed entry's data_ptr with coincidentally identical
# _version/numel/dtype can never alias — its entry's weakref is dead, so
# it misses and is rebuilt). Documented ASSUMPTION that remains: an
# in-place content change that bypasses the version counter (e.g. raw
# `.data` writes / `set_` games) is invisible to the cache — supported
# callers never do that to index tensors. The seg_offs fold is keyed on
# the SOURCE kernel-offset index tensor (mvmr a_idx / vvor o_idx, of
# which seg_offs is a pure function), never on the internally-rebuilt
# seg_offs temp whose allocator slot is recycled every call. Tensors
# created under torch.inference_mode() expose no version counter — those
# calls bypass the cache (folded per call; still correct).
_FOLD_IDX_CACHE: OrderedDict = OrderedDict()
_FOLD_IDX_CACHE_MAX = 32
_INT32_LIM = 2 ** 31
# Escape hatch / test hook: False forces the per-group loop fallback
# (the same path the int32-overflow guard takes).
_FOLD_G_ENABLED = True


def fused_fold_legal(G: int, n_b: int, n_o: int, T: int) -> bool:
    """conv.py routing guard for FusedPointConv3d at G>1: fold-G must be
    enabled and the folded index ranges must fit int32 (the kernels'
    index dtype). G==1 is always legal (no fold)."""
    if G == 1:
        return True
    return (
        _FOLD_G_ENABLED
        and G * n_b < _INT32_LIM
        and G * n_o < _INT32_LIM
        and G * T < _INT32_LIM
    )


def _fold_cache_key(kind: str, src: Tensor, params: tuple):
    try:
        ver = src._version
    except RuntimeError:        # inference tensor — no version counter
        return None
    return (kind, src.data_ptr(), ver, src.numel(), src.dtype) + params + (
        src.device.index,
    )


def _fold_cache_get(key):
    if key is None:
        return None
    entry = _FOLD_IDX_CACHE.get(key)
    if entry is None:
        return None
    src_ref, val = entry
    src = src_ref()
    # Liveness + identity re-check (see cache notes above): the original
    # source tensor must still be alive, unmutated, and at the same
    # storage — otherwise this key is a recycled-ptr imposter; drop it.
    if src is None or src._version != key[2] or src.data_ptr() != key[1]:
        del _FOLD_IDX_CACHE[key]
        return None
    _FOLD_IDX_CACHE.move_to_end(key)
    return val


def _fold_cache_put(key, src: Tensor, val) -> None:
    if key is None:
        return
    _FOLD_IDX_CACHE[key] = (weakref.ref(src), val)
    while len(_FOLD_IDX_CACHE) > _FOLD_IDX_CACHE_MAX:
        _FOLD_IDX_CACHE.popitem(last=False)


def _folded_row_idx(idx: Tensor, n_rows: int, G: int) -> Tensor:
    """concat_g(idx + g*n_rows) as int32, cached (see cache notes above)."""
    key = _fold_cache_key("row", idx, (int(n_rows), int(G)))
    hit = _fold_cache_get(key)
    if hit is not None:
        return hit
    ar = torch.arange(G, device=idx.device, dtype=torch.int64)
    folded = (
        (idx.to(torch.int64).unsqueeze(0) + (ar * int(n_rows)).unsqueeze(1))
        .reshape(-1).to(torch.int32)
    )
    _fold_cache_put(key, idx, folded)
    return folded


def _folded_seg_offs(seg_offs: Tensor, key_src: Tensor, T: int, G: int) -> Tensor:
    """concat_g(g*T + seg_offs[:-1]) ++ [G*T] as int64, cached.

    Keyed on `key_src` (the kernel-offset index tensor seg_offs was built
    from), NOT on seg_offs itself — seg_offs is usually an internal
    per-call temp whose allocator slot is recycled every call, which
    would make the data_ptr/_version/weakref staleness machinery thrash
    (see cache notes above).
    """
    K = int(seg_offs.numel()) - 1
    key = _fold_cache_key("seg", key_src, (K, int(T), int(G)))
    hit = _fold_cache_get(key)
    if hit is not None:
        return hit
    seg64 = seg_offs.to(torch.int64)
    ar = torch.arange(G, device=seg_offs.device, dtype=torch.int64)
    folded = torch.cat([
        (seg64[:-1].unsqueeze(0) + (ar * int(T)).unsqueeze(1)).reshape(-1),
        torch.full((1,), int(T) * G, device=seg_offs.device, dtype=torch.int64),
    ])
    _fold_cache_put(key, key_src, folded)
    return folded


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

    seg_len is clamped/padded to exactly S_TILE rows (the skeleton is a
    single fixed (M_TILE, S_TILE) tile, mirroring vvor's single-tile
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


# ─── Kernel-side IndexedGather on the B (input) operand ───────────────────────
## The skeleton pre-gathers B Python-side (`stage_one_tile` does
# `input_b[b_idx]`). This variant moves that gather INSIDE the CUTLASS
# mainloop: the caller hands the kernel the raw `input_b` rows + the
# b_idx index buffer, and a composed `IndexedGather` custom-stride layout
# drives the S-axis gather inside the CollectiveMma cp.async loads. The
# affine W_seg tile is still produced Python-side (W is affine in mvmr —
# indexed by segment id k only).
## Unlike vvor's gathered variant (which gathered along the contraction
# axis K and needed a transposing 2nd Config + ldmatrix-T), mvmr gathers
# along the S/triplet axis while the contraction axis C stays
# gmem-contiguous, so the skeleton Config composes directly. See the .cu
# header for the full rationale.


def stage_w_tile(
    weight: Tensor,
    m_start: int,
    c_start: int,
) -> tuple[Tensor, int]:
    """Slice + pad the affine W[k] tile for the gathered path.

    Identical W-side preparation to `stage_one_tile` — only B's gather is
    deferred to the kernel in the gathered path, so this returns just the
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
    """Gathered entry: composed IndexedGather on B inside the CUTLASS mainloop.

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
    """Scalar reference for the gathered entrypoint.

    Gathers B Python-side then does the same fp32 `W @ B_gathered^T` the
    skeleton reference does. Mirrors the kernel's index/shape contract:
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


# ─── Full mvmr op (outer ragged-K-segment grid + scatter epilogue) ────────────


def sparse_matrix_vector_multiplication_reduction_cutlass(
    a: Tensor,        # (K_offsets, G=1, C, M) fp16 — affine weight W[k]
    a_idx: Tensor,    # (T,) int — kernel-offset segment id per triplet, sorted asc
    b: Tensor,        # (N_b, G=1, C) fp16 — input rows
    b_idx: Tensor,    # (T,) int — input-row idx into b
    o_idx: Tensor,    # (T,) int — output-row idx into o
    n_o: int,         # number of output points (o's leading dim)
    seg_offs: Tensor | None = None,  # pre-built (K+1,) int64 seg_offs
) -> Tensor:
    """Full mvmr forward via the Tier-2 CUTLASS path.

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
      - fp16 OR bf16 inputs, both operands the same dtype (the
        SM80_16x8x16_F32BF16BF16F32_TN atom path handles bf16). fp32 NOT
        supported (no SM80 fp32-input TC atom of this shape); fp32 conv
        stays on the Triton path.
      - M and C multiples of the kernel tile (TileM=64, TileK=32)

    Striding + seg_offs handling:
      - The per-call ``a = a.contiguous()`` is omitted. The C++ host fn
        already resolves any input striding via its own
        ``select(1,0).transpose(1,2).contiguous()`` repack, so a Python
        pre-contiguous would be pure redundancy. On the grad_b second
        call ``a`` arrives as ``W.transpose(2,3)`` (non-contiguous);
        avoiding the ``.contiguous()`` here removes a full
        transposed-weight materialization on top of the host repack.
      - ``seg_offs`` may be passed in pre-built (computed once at the
        autograd.Function boundary and shared between the fwd and the
        grad_b second call — ``a_idx``/``K_offsets`` are the fixed
        triplet structure, invariant across both). When ``None`` it is
        built here as before (non-autograd / "auto"-mode callers).
    """
    K_offsets = a.shape[0]
    G = a.shape[1]
    C = a.shape[2]
    M = a.shape[3]

    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "CUTLASS full mvmr is fp16/bf16-only, both operands same dtype "
            f"(got a={a.dtype}, b={b.dtype})"
        )
    # Tile gates apply PER GROUP — C / M here are the per-group channel
    # counts (weight is (K, G, Cg, Mg), input (N, G, Cg)).
    if M % M_TILE != 0 or C % C_TILE != 0:
        raise ValueError(
            f"CUTLASS full mvmr requires per-group M % {M_TILE} == 0 and "
            f"per-group C % {C_TILE} == 0; got per-group M={M}, C={C} (G={G})"
        )
    if not bool((a_idx[1:] >= a_idx[:-1]).all().item()):
        raise ValueError("a_idx must be sorted ascending for grouped path")

    if seg_offs is None:
        seg_offs = kernel_offset_segments(a_idx, int(K_offsets)).to(torch.int64)
    else:
        seg_offs = seg_offs.to(torch.int64)

    # Arch dispatch (mirrors vvor_cutlass.py): route Hopper (sm_90+)
    # hardware to the sm_90-targeted op. The two ops are algorithmically
    # identical (same Sm80 cp.async-Unpredicated + S-gather +
    # scatter-accumulate); the sm_90 symbol exists so sm_90 hardware
    # exercises the sm_90 SASS path. On sm_80/89 the sm_80 op stays the
    # path of record.
    major = torch.cuda.get_device_capability(a.device)[0]
    if major >= 9:
        op = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm90_full
    else:
        op = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full

    if G == 1:
        b_idx_i32 = b_idx.to(torch.int32)
        o_idx_i32 = o_idx.to(torch.int32)
        b = b.contiguous()
        return op(a, b_idx_i32, b, o_idx_i32, seg_offs, int(n_o))

    # ── G > 1: fold G into K_offsets — ONE launch of the frozen G==1
    # kernel (the per-group side-stream loop is retained in
    # `_mvmr_groups_loop` below as the int32-overflow fallback).
    # Verified against the loop on real ScanNet scenes: parity
    # relF ≈ 2e-8 (atomicAdd-scatter reordering only); c256/G=4 fp16
    # 1.138 → 0.962 ms, c512 0.450 → 0.273 ms with cached indices.
    # Layout:
    #   weight (K, G, Cg, Mg) → (G*K, 1, Cg, Mg)  (segment g*K+k = group g's W[k])
    #   input  (N, G, Cg)     → (G*N, Cg)         (row g*N+i = group g's b[i])
    #   b_idx' = concat_g(b_idx + g*N), o_idx' = concat_g(o_idx + g*n_o)
    #   seg_offs' = concat_g(g*T + seg_offs[:-1]) ++ [G*T]
    # Folded indices are group-invariant → cached (_folded_row_idx /
    # _folded_seg_offs; see the fold-cache notes above _fold_cache_key).
    # Output (G*n_o, 1, M) → unfold → (n_o, G, M) fp32, identical shape/
    # dtype contract as the loop's cat.
    n_b = b.shape[0]
    T = int(b_idx.numel())
    if (
        _FOLD_G_ENABLED
        and G * n_b < _INT32_LIM
        and G * int(n_o) < _INT32_LIM
        and G * T < _INT32_LIM
    ):
        wf = a.permute(1, 0, 2, 3).reshape(G * K_offsets, 1, C, M).contiguous()
        ff = b.permute(1, 0, 2).reshape(G * n_b, C).contiguous()
        b_fold = _folded_row_idx(b_idx, n_b, G)
        o_fold = _folded_row_idx(o_idx, int(n_o), G)
        seg_fold = _folded_seg_offs(seg_offs, a_idx, T, G)
        out = op(wf, b_fold, ff, o_fold, seg_fold, G * int(n_o))
        return out.view(G, int(n_o), M).permute(1, 0, 2).contiguous()

    # Fold would overflow the int32 index ABI (or fold disabled) — take
    # the per-group loop.
    return _mvmr_groups_loop(
        a, b, b_idx.to(torch.int32), o_idx.to(torch.int32), seg_offs,
        int(n_o), G, major,
    )


def _mvmr_groups_loop(
    a: Tensor,
    b: Tensor,
    b_idx_i32: Tensor,
    o_idx_i32: Tensor,
    seg_offs: Tensor,
    n_o: int,
    G: int,
    major: int,
) -> Tensor:
    """G > 1 wrapper-level per-group loop.

    Pre-fold path of record, retained as the fallback when the
    fold-G-into-K_offsets lever would overflow the kernels' int32 index
    ABI (G*N or G*T ≥ 2^31), and as the reference arm for the fold
    parity tests. The frozen `_full` host fn hard-TORCH_CHECKs
    `a.size(1) == 1`, so groups are block-diagonal-decomposed into G
    independent G==1 calls: group g uses weight[:, g], input[:, g],
    writes out[:, g]. Index tensors + seg_offs are GROUP-INVARIANT —
    built once by the caller, shared by every per-group call (not
    rebuilt per group).

    Repack accounting (honest count):
      - input b: ONE batched repack `permute(1,0,2).contiguous()` —
        (N, G, Cg) → (G, N, Cg) — so each per-group slice b_perm[g]
        is already (N, 1, Cg)-contiguous and the host fn's internal
        `inb_2d.contiguous()` is a no-op. Same total bytes as G==1's
        single `.contiguous()`, one kernel launch instead of G.
      - weight a: ONE batched pre-stage `permute(1,0,3,2).contiguous()`
        — (K, G, Cg, Mg) → (G, K, Mg, Cg) — feeding the `_full_prestaged`
        entry, which SKIPS `_full`'s per-call internal
        `select(1,0).transpose(1,2).contiguous()` repack. Same total
        bytes as the G host-internal repacks it replaces, one launch
        instead of G.
      - the per-group launches go to dedicated side streams
        (`_group_streams`) — the under-occupied kernels (K_offsets CTAs
        each) overlap instead of serializing.
    Structural handicap that remains: G under-occupied kernel launches
    + the final cat — quantified in the grouped-operator benchmarks
    under benchmarks/operators/.
    """
    if major >= 9:
        op_pre = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm90_full_prestaged
    else:
        op_pre = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full_prestaged
    b_perm = b.permute(1, 0, 2).contiguous()
    aT = a.permute(1, 0, 3, 2).contiguous()   # (G, K, Mg, Cg)
    cur = torch.cuda.current_stream()
    streams = _group_streams(a.device, G)
    ev = torch.cuda.Event()
    ev.record(cur)
    outs: list = [None] * G
    for g in range(G):
        with torch.cuda.stream(streams[g]):
            streams[g].wait_event(ev)
            outs[g] = op_pre(
                aT[g], b_idx_i32, b_perm[g].unsqueeze(1), o_idx_i32,
                seg_offs, int(n_o),
            )
            outs[g].record_stream(cur)
    for s in streams:
        cur.wait_stream(s)
    return torch.cat(outs, dim=1)  # (n_o, G, M) fp32


# ─── Python-only fused PointConv3d autograd.Function ──────────────────────────
## Collapse the eager PointConv3d fwd+bwd composition — 3 @triton_op /
# autograd-graph op boundaries + 2 seg_offs builds + the duplicate Python
# `.contiguous()` calls — into ONE autograd.Function, reusing the
# CUTLASS mvmr/vvor full kernels AS-IS (no .cu/.cuh edit). Wins, all
# Python/autograd-side:
##   (1) fwd + grad_b both feed the C++ `_full_prestaged` host
#       entry with a caller-pre-staged
#       (n_o_k, M_full, C_full)-C-contiguous buffer, which SKIPS `_full`'s
#       unconditional internal `a.select(1,0).transpose(1,2).contiguous()`
#       repack. Each side stages its buffer EXACTLY ONCE:
#         fwd:    aT_fwd   = weight.select(1,0).transpose(1,2).contiguous()
#                           → (K, M_w, C_w) C-contig
#         grad_b: aT_gradb = weight.select(1,0).contiguous()
#                           → (K, C_w, M_w) C-contig (the `.transpose(1,2)`
#                             the fwd stage applies is DROPPED — this is
#                             byte-equal to `_full`'s internal repack of
#                             `weight.transpose(2,3)`).
#       The kernel only ever reads a (K, M_full, C_full)-C-contig buffer;
#       fwd-vs-grad_b is purely which weight axes the caller maps.
#       `_prestaged` consumes the staged buffer directly, so the per-side
#       copy happens at the controlled stage with NO internal `_full`
#       repack on top.
#   (2) One Function instead of 3 @triton_op/autograd-graph boundaries;
#       `seg_offs` built ONCE in forward, saved on ctx, reused in backward
#       (no rebuild); no per-call Python `.contiguous()` on the staged path.
## grad_a: CUTLASS vvor full, reused exactly as the eager
#   `_backward_sparse_matrix_vector_multiplication_reduction` does.
# grad_b: CUTLASS mvmr full via `_full_prestaged` fed
#   `weight.select(1,0).contiguous()` + the saved `seg_offs` (the
#   transpose dropped vs fwd; see (1)), so no host repack happens inside
#   `_full`.


class FusedPointConv3d(torch.autograd.Function):
    """Fused PointConv3d fwd+bwd (Python-only).

    Collapses the eager mvmr-fwd / vvor-grad_a / mvmr-grad_b composition
    into one Function reusing the frozen CUTLASS full kernels. fp16 /
    sorted-a_idx / per-group M,C tile-multiple — same hard preconditions
    as the underlying `sparse_matrix_vector_multiplication_reduction_cutlass`
    / `..._grouped_cutlass` entry points, which raise on violation.

    Inputs mirror `sparse_matrix_vector_multiplication_reduction`:
      weight (K, G, Cg, Mg) fp16, a_idx (T,) sorted asc, input_3d
      (N, G, Cg) fp16, b_idx (T,), o_idx (T,), n_o int.

    G > 1: fold-G-into-K_offsets, single launch per leg — the same
    mechanics as the fold wrappers above (weight → (G·K,1,Cg,Mg)
    g-major, group-offset row indices via `_folded_row_idx`, seg_offs'
    via `_folded_seg_offs` — both cached), so the fused mode keeps its
    one-autograd-node / 3-launches shape at any G. Before this, G>1
    entering here would `select(1, 0)` — group-0-only, a
    silent-wrong-result hole; conv.py additionally guards fold legality
    (int32 ranges) and falls through to the eager composition when
    unmet.
    """

    @staticmethod
    def forward(ctx, weight, a_idx, input_3d, b_idx, o_idx, n_o):
        K_offsets = weight.shape[0]
        G = weight.shape[1]
        N_b = input_3d.shape[0]
        T = a_idx.numel()

        if G > 1 and not (
            _FOLD_G_ENABLED
            and G * N_b < _INT32_LIM
            and G * int(n_o) < _INT32_LIM
            and G * T < _INT32_LIM
        ):
            raise ValueError(
                f"FusedPointConv3d G={G}: fold-G is disabled or exceeds "
                f"int32 index range (G*N={G * N_b}, G*n_o={G * int(n_o)}, "
                f"G*T={G * T}); route the eager composition instead"
            )

        # ── Single weight stage, fed straight to `_full_prestaged`.
        # weight (K,1,C,M) → aT_fwd (K,M,C) C-contiguous — exactly the
        # (n_o_k, M_full, C_full) layout `_full_prestaged` consumes (it
        # skips `_full`'s unconditional internal
        # `select(1,0).transpose(1,2).contiguous()` repack by contract).
        # This stage is THE one weight copy for the fwd side; amortized
        # over fwd + grad_a. `_prestaged` consumes the staged buffer
        # directly (no host-side repack on top of it).
        # G>1: g-major fold (G·K, Mg, Cg) matching the folded seg_offs'
        # segment order; still exactly one weight copy.
        if G == 1:
            aT_fwd = weight.select(1, 0).transpose(1, 2).contiguous()
        else:
            aT_fwd = weight.permute(1, 0, 3, 2).reshape(
                G * K_offsets, weight.shape[3], weight.shape[2]
            ).contiguous()

        # ── seg_offs built ONCE; reused by fwd host fn and grad_b. a_idx +
        # K_offsets are the fixed triplet structure, invariant fwd↔grad_b.
        seg_offs = kernel_offset_segments(a_idx, int(K_offsets)).to(torch.int64)

        if G == 1:
            b = input_3d.contiguous()
            b_idx_i32 = b_idx.to(torch.int32)
            o_idx_i32 = o_idx.to(torch.int32)
        else:
            b = input_3d.permute(1, 0, 2).reshape(
                G * N_b, 1, weight.shape[2]).contiguous()
            b_idx_i32 = _folded_row_idx(b_idx, N_b, G)
            o_idx_i32 = _folded_row_idx(o_idx, int(n_o), G)
            # NOTE: at G>1 the ctx-saved seg_offs is the FOLDED one
            # (length G*K+1) — grad_b reuses it directly.
            seg_offs = _folded_seg_offs(seg_offs, a_idx, T, G)

        n_o_launch = int(n_o) if G == 1 else G * int(n_o)
        major = torch.cuda.get_device_capability(weight.device)[0]
        if major >= 9:
            out = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm90_full_prestaged(
                aT_fwd, b_idx_i32, b, o_idx_i32, seg_offs, n_o_launch,
            )
        else:
            out = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full_prestaged(
                aT_fwd, b_idx_i32, b, o_idx_i32, seg_offs, n_o_launch,
            )
        out = out.to(weight.dtype) if out.dtype != weight.dtype else out
        if G > 1:
            # (G·n_o, 1, Mg) → (n_o, G, Mg), group-major channels —
            # matches the (N, G, Cg) input convention conv.py flattens.
            out = out.reshape(G, int(n_o), weight.shape[3]).permute(
                1, 0, 2).contiguous()

        ctx.save_for_backward(weight, a_idx, input_3d, b_idx, o_idx, seg_offs)
        ctx.n_o = int(n_o)
        ctx.K_offsets = int(K_offsets)
        ctx.b_shape_0 = input_3d.shape[0]
        ctx.G = G
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
            # grad_b (grad of input) via the CUTLASS mvmr full kernel,
            # routed through `_full_prestaged` with a caller-staged buffer
            # instead of through `_full`
            # (`sparse_matrix_vector_multiplication_reduction_cutlass`)
            # fed `weight.transpose(2,3)`. The kernel only ever reads a
            # (n_o_k, M_full, C_full)-C-contiguous buffer; fwd-vs-grad_b
            # is purely which weight axes the caller maps onto M/C. For
            # grad_b that buffer is `weight.select(1,0).contiguous()`
            # ((K,1,C,M) → (K,C,M) C-contig) — the `.transpose(1,2)` the
            # fwd stage applies is DROPPED (this is byte-equal to
            # `_full`'s internal repack of `weight.transpose(2,3)`). The
            # single copy happens at this controlled stage, with NO
            # internal `_full` repack on top of it. seg_offs is reused
            # (not rebuilt); grad_a (vvor) path unchanged.
            G = ctx.G
            if G == 1:
                aT_gradb = weight.select(1, 0).contiguous()
                b_idx_i32 = b_idx.to(torch.int32)
                o_idx_i32 = o_idx.to(torch.int32)
                grad_in = grad_out
                n_rows = ctx.b_shape_0
            else:
                # Fold-G mirror of the forward (ctx seg_offs is already
                # folded; row-index folds are cache hits from forward).
                Cg, Mg = weight.shape[2], weight.shape[3]
                aT_gradb = weight.permute(1, 0, 2, 3).reshape(
                    G * ctx.K_offsets, Cg, Mg).contiguous()
                b_idx_i32 = _folded_row_idx(b_idx, ctx.b_shape_0, G)
                o_idx_i32 = _folded_row_idx(o_idx, ctx.n_o, G)
                grad_in = grad_out.permute(1, 0, 2).reshape(
                    G * ctx.n_o, 1, Mg).contiguous()
                n_rows = G * ctx.b_shape_0
            major = torch.cuda.get_device_capability(weight.device)[0]
            if major >= 9:
                grad_b = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm90_full_prestaged(
                    aT_gradb, o_idx_i32, grad_in, b_idx_i32, seg_offs,
                    n_rows,
                )
            else:
                grad_b = torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full_prestaged(
                    aT_gradb, o_idx_i32, grad_in, b_idx_i32, seg_offs,
                    n_rows,
                )
            if grad_b.dtype != weight.dtype:
                grad_b = grad_b.to(weight.dtype)
            if G > 1:
                # (G·N, 1, Cg) → (N, G, Cg) matching input_3d.
                grad_b = grad_b.reshape(G, ctx.b_shape_0,
                                        weight.shape[2]).permute(
                    1, 0, 2).contiguous()

        return grad_weight, None, grad_b, None, None, None
