import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Matrix-Vector Multiplication-Reduction (MVMR) — forward conv kernel
#
# Per triplet (a_idx, b_idx, o_idx):
#   o[o_idx, m] += sum_c( a[a_idx, c, m] * b[b_idx, c] )
#
# G is the group dim (BLOCK_SIZE_G is constexpr; in this codebase G=1 always
# for PointConv3d k=3, but the kernel supports group conv too).
#
# Compute path: cast inputs to fp32 before the multiply so the running
# `block_o` accumulator is fp32. The previous "Native fp16" version summed
# in fp16, which loses ~9 bits across the 512-element C-reduction — a
# correctness liability we observed in the cycle-9 hybrid bench. Forcing
# fp32 accum costs some register pressure but on Ada the fp16/fp32 fma
# throughput is 1:1 outside tensor cores, so the per-element multiply rate
# is unchanged.
#
# tl.dot was tried but the per-triplet shape `(1, C) @ (C, M) → (1, M)`
# has LHS M_inner=1 which is below the tensor-core threshold (≥16 on
# sm_80+); it compiled to scalar fma with extra reshape overhead and
# regressed perf at small C. To actually leverage tensor cores we'd need
# to BATCH multiple triplets sharing the same a_offset into a `(L, C)` tile
# (a deeper restructuring; see the mvmr_dot variant — future work).
# ─────────────────────────────────────────────────────────────────────────────


_AUTOTUNE_CONFIGS = [
    # L: triplets per program. Larger L = more reuse of loaded a/b blocks
    # when consecutive triplets share the same a_offset or b_offset.
    # BLOCK_SIZE_M / BLOCK_SIZE_C: tile sizes for output / input channel dims.

    # ── L=32: high program parallelism (good for small T) ───────────────────
    triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 16}, num_warps=1),
    triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
    triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64}, num_warps=4),

    # ── L=64: balanced ──────────────────────────────────────────────────────
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 16}, num_warps=1),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 128}, num_warps=4),

    # ── L=128: original pivot + larger tiles for high-C stages ──────────────
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=1),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64}, num_warps=8),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 128}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),

    # ── L=256: more sequential per program — better reuse for large T ───────
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=4),
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["G", "M", "C"], reset_to_zero=["o"])
@triton.jit
def sparse_matrix_vector_multiplication_reduction_kernel(
    a,
    a_idx,
    b,
    b_idx,
    o,
    o_idx,
    T,
    G,
    M,
    C,
    L: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_c = tl.cdiv(C, BLOCK_SIZE_C)
    num_pid_g = tl.cdiv(G, BLOCK_SIZE_G)

    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid //= num_pid_m
    pid_c = pid % num_pid_c
    pid //= num_pid_c
    pid_g = pid % num_pid_g
    pid_t = pid // num_pid_g

    g_offsets = pid_g * BLOCK_SIZE_G + tl.arange(0, BLOCK_SIZE_G)
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    g_mask = g_offsets < G
    m_mask = m_offsets < M
    c_mask = c_offsets < C

    gm_mask = g_mask[:, None] & m_mask[None, :]
    gc_mask = g_mask[:, None] & c_mask[None, :]
    gcm_mask = g_mask[:, None, None] & c_mask[None, :, None] & m_mask[None, None, :]

    a_ptrs = a + (
        g_offsets[:, None, None] * (C * M)
        + c_offsets[None, :, None] * M
        + m_offsets[None, None, :]
    )
    b_ptrs = b + (g_offsets[:, None] * C + c_offsets[None, :])
    o_ptrs = o + (g_offsets[:, None] * M + m_offsets[None, :])

    t_offset = pid_t * L
    a_offset = tl.load(a_idx + t_offset)
    b_offset = tl.load(b_idx + t_offset)
    o_offset = tl.load(o_idx + t_offset)

    # Load with `other=0.0` so masked lanes don't contribute (Triton's
    # default for masked loads is undefined). Cast to fp32 BEFORE the
    # multiply so the running `block_o` is fp32 and the C-axis reduction
    # accumulates in fp32 — fixes the fp16-accum precision loss.
    block_a = tl.load(a_ptrs + a_offset * (G * C * M), mask=gcm_mask, other=0.0).to(tl.float32)
    block_b = tl.load(b_ptrs + b_offset * (G * C), mask=gc_mask, other=0.0).to(tl.float32)
    block_o = tl.sum(block_a * block_b[:, :, None], axis=1)

    for t in tl.range(1, min(L, T - t_offset)):
        a_offset_next = tl.load(a_idx + t_offset + t)
        b_offset_next = tl.load(b_idx + t_offset + t)
        o_offset_next = tl.load(o_idx + t_offset + t)

        if a_offset_next != a_offset:
            block_a = tl.load(a_ptrs + a_offset_next * (G * C * M), mask=gcm_mask, other=0.0).to(tl.float32)
            a_offset = a_offset_next

        if b_offset_next != b_offset:
            block_b = tl.load(b_ptrs + b_offset_next * (G * C), mask=gc_mask, other=0.0).to(tl.float32)
            b_offset = b_offset_next

        if o_offset_next != o_offset:
            tl.atomic_add(o_ptrs + o_offset * (G * M), block_o, mask=gm_mask)
            o_offset = o_offset_next
            block_o = tl.sum(block_a * block_b[:, :, None], axis=1)
        else:
            block_o += tl.sum(block_a * block_b[:, :, None], axis=1)

    tl.atomic_add(o_ptrs + o_offset * (G * M), block_o, mask=gm_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped (segment-based) variant for tensor-core utilisation.
#
# When triplets are sorted by kernel offset (`sort_by="k"` — the default),
# all triplets sharing an `a_idx` form a contiguous segment. Instead of
# processing them one-by-one and computing `(1, C) @ (C, M) → (1, M)` (which
# defeats sm_80+ tensor cores due to LHS_M=1), we batch L_CHUNK triplets at
# a time:
#
#   block_a       = weight[k_offset]                           # (C, M)  — once per program
#   block_b_chunk = gather(b, b_idx[seg_start : seg_start+L])  # (L, C)
#   out_chunk     = tl.dot(block_b_chunk, block_a)             # (L, M)  — tensor cores fire
#   atomic_add(o[o_idx[seg_start+row]], out_chunk[row, :])     for row in 0..L-1
#
# Grid parameter `pid_chunk` indexes a flat list of all chunks across all
# kernel offsets; the kernel locates its kernel offset via a linear search
# over `chunk_seg_offs` (whose length is K+1, typically 28 for k=3 conv).
# ─────────────────────────────────────────────────────────────────────────────


_GROUPED_AUTOTUNE_CONFIGS = [
    # L_CHUNK ∈ {16, 32, 64}. The kernel walks seg_offs on-the-fly to
    # locate its kernel offset (no precomputed chunk_seg_offs argument);
    # the launcher only needs the total chunk count per L_CHUNK to build
    # the grid. Larger L_CHUNK = bigger tl.dot M_inner = better tensor-
    # core utilisation and fewer programs per kernel offset.

    # ── L_CHUNK=32 (legacy default) ────────────────────────────────────────
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=4),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 256}, num_warps=8),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 256}, num_warps=8),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_C": 64},  num_warps=8),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_C": 128}, num_warps=8),
    triton.Config({"L_CHUNK": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_C": 256}, num_warps=8),

    # ── L_CHUNK=64 — better tensor-core utilisation at deep stages ─────────
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 256}, num_warps=8),
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_C": 128}, num_warps=8),

    # ── L_CHUNK=16 — better fit when avg seg length is short (small T/K) ──
    triton.Config({"L_CHUNK": 16, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32,  "BLOCK_SIZE_C": 32},  num_warps=2),
    triton.Config({"L_CHUNK": 16, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 16, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64},  num_warps=4),

    # ── L_CHUNK=128 — biggest weight-reuse-per-mma at deep stages ──────────
    # Output tile is (128, BLOCK_M); large but k_offset-tile reuse across
    # 128 triplets in a single mma keeps the SM busy.
    triton.Config({"L_CHUNK": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 128}, num_warps=8),
    triton.Config({"L_CHUNK": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),
    triton.Config({"L_CHUNK": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 256}, num_warps=8),

    # ── L_CHUNK=256 — biggest single-mma at deep stages, more weight reuse.
    # Output tile (256, BLOCK_M) at fp32 acc is (256×BM)×4 bytes; with
    # BM=32 that's 32 KB / 8 warps = 4 KB / warp, fits comfortably.
    # BM=64 is 64 KB / 8 warps = 8 KB / warp, still fits. BM=128 (128 KB)
    # likely register-spills; autotune will skip if so.
    triton.Config({"L_CHUNK": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32,  "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32,  "BLOCK_SIZE_C": 128}, num_warps=8),
    triton.Config({"L_CHUNK": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 64},  num_warps=8),
    triton.Config({"L_CHUNK": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 128}, num_warps=8),
]


@triton.autotune(configs=_GROUPED_AUTOTUNE_CONFIGS, key=["G", "M", "C"], reset_to_zero=["o"])
@triton.jit
def sparse_matrix_vector_multiplication_reduction_grouped_kernel(
    a,                # (K_offsets, G, C, M)
    b,                # (N_b, G, C)
    b_idx,            # (T,)
    o,                # (N_o, G, M)
    o_idx,            # (T,)
    seg_offs,         # (K_offsets + 1,) — segment starts in the triplet arrays
    K_offsets,
    G, M, C,
    L_CHUNK: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    num_pid_g = tl.cdiv(G, BLOCK_SIZE_G)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_c = tl.cdiv(C, BLOCK_SIZE_C)

    pid = tl.program_id(axis=0)
    pid_m = pid % num_pid_m
    pid //= num_pid_m
    pid_c = pid % num_pid_c
    pid //= num_pid_c
    pid_g = pid % num_pid_g
    pid_chunk = pid // num_pid_g

    # Locate kernel offset by walking seg_offs and accumulating
    # ceil(seg_len / L_CHUNK) per offset. K_offsets is small (27 for k=3,
    # 125 for k=5); the loop is a tight constexpr scan. The chunk_seg_offs
    # array isn't materialised — keeping L_CHUNK as a real autotune knob.
    # Cast seg_offs loads to int32: T ≤ 2³¹ in practice, and pid/running
    # are all int32, so this avoids loop-carried type mismatch.
    k_offset = 0
    seg_start_k = tl.load(seg_offs + 0).to(tl.int32)
    seg_end_k = tl.load(seg_offs + 1).to(tl.int32)
    chunk_idx_within_k = 0
    running = 0
    for k_iter in tl.range(0, K_offsets):
        seg_s = tl.load(seg_offs + k_iter).to(tl.int32)
        seg_e = tl.load(seg_offs + k_iter + 1).to(tl.int32)
        chunks_in_this_k = (seg_e - seg_s + L_CHUNK - 1) // L_CHUNK
        if running <= pid_chunk and pid_chunk < running + chunks_in_this_k:
            k_offset = k_iter
            seg_start_k = seg_s
            seg_end_k = seg_e
            chunk_idx_within_k = pid_chunk - running
        running += chunks_in_this_k

    seg_start = seg_start_k + chunk_idx_within_k * L_CHUNK
    seg_end_k = seg_end_k

    # Triplet positions this chunk processes. Tail handling: positions
    # past seg_end_k are masked off (zero contribution).
    l_offsets = seg_start + tl.arange(0, L_CHUNK)
    l_mask = l_offsets < seg_end_k

    g_offsets = pid_g * BLOCK_SIZE_G + tl.arange(0, BLOCK_SIZE_G)
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    g_mask = g_offsets < G
    m_mask = m_offsets < M
    c_mask = c_offsets < C
    gm_mask = g_mask[:, None] & m_mask[None, :]
    gc_mask = g_mask[:, None] & c_mask[None, :]
    gcm_mask = g_mask[:, None, None] & c_mask[None, :, None] & m_mask[None, None, :]

    # Load weight tile for this kernel offset — one global load per program.
    a_ptr = a + k_offset * (G * C * M) + (
        g_offsets[:, None, None] * (C * M)
        + c_offsets[None, :, None] * M
        + m_offsets[None, None, :]
    )
    block_a = tl.load(a_ptr, mask=gcm_mask, other=0.0).to(tl.float32)

    # Gather indices for this chunk.
    b_idx_chunk = tl.load(b_idx + l_offsets, mask=l_mask, other=0)
    o_idx_chunk = tl.load(o_idx + l_offsets, mask=l_mask, other=0)

    # Gather b features: (L_CHUNK, G, C_tile).
    b_ptrs = b + (
        b_idx_chunk[:, None, None] * (G * C)
        + g_offsets[None, :, None] * C
        + c_offsets[None, None, :]
    )
    b_load_mask = l_mask[:, None, None] & gc_mask[None, :, :]
    block_b = tl.load(b_ptrs, mask=b_load_mask, other=0.0).to(tl.float32)

    if BLOCK_SIZE_G == 1:
        # tl.dot path — squeeze G dim. (L_CHUNK, C) @ (C, M) → (L_CHUNK, M).
        # M_inner = L_CHUNK ≥ 16, K = BLOCK_SIZE_C ≥ 16 → tensor cores fire.
        b_2d = tl.reshape(block_b, (L_CHUNK, BLOCK_SIZE_C))
        a_2d = tl.reshape(block_a, (BLOCK_SIZE_C, BLOCK_SIZE_M))
        out_2d = tl.dot(b_2d, a_2d, out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION)
        out_3d = tl.reshape(out_2d, (L_CHUNK, BLOCK_SIZE_G, BLOCK_SIZE_M))
    else:
        # Group-conv fallback (G > 1 not in production today). Elementwise
        # multiply + reduce over C.
        out_3d = tl.sum(
            block_b[:, :, :, None] * block_a[None, :, :, :],
            axis=2,
        )

    # Scatter-add per-row to o[o_idx_chunk[row]].
    o_ptrs = o + (
        o_idx_chunk[:, None, None] * (G * M)
        + g_offsets[None, :, None] * M
        + m_offsets[None, None, :]
    )
    store_mask = l_mask[:, None, None] & gm_mask[None, :, :]
    tl.atomic_add(o_ptrs, out_3d, mask=store_mask)
