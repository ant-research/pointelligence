import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Vector-Vector Outer-Product Reduction (VVOR) — backward weight-grad
#
# Per triplet (a_idx, b_idx, o_idx):
#   o[o_idx, m, c] += a[a_idx, m] * b[b_idx, c]
#
# Used in PointConv3d backward to accumulate grad_weight at each kernel offset.
#
# fp16-precision concern: the previous implementation accumulated the running
# outer-product `block_o` in the inputs' native dtype, which means at fp16,
# 128 outer-products sum in fp16 → ~9 bits of magnitude lost per accumulation
# (matches a hybrid bench symptom). Fix: cast inputs to fp32 before
# multiply so the running accumulator and the per-triplet outer-product are
# both fp32.
#
# tl.dot opportunity: VVOR's per-triplet operation is a (M,)-by-(C,) outer
# product. Cast to (M, 1) @ (1, C) makes K=1 — too small for tensor cores
# (which need K ≥ 16 on Ampere/Ada). We don't switch to tl.dot for VVOR;
# the precision fix is the principal win here. A future-direction option
# is to BATCH L triplets sharing the same o_offset into a (M, L) @ (L, C)
# tensor-core-shaped GEMM, but that's a deeper restructuring.
# ─────────────────────────────────────────────────────────────────────────────


_AUTOTUNE_CONFIGS = [
    # VVOR: outer product → (G, M, C) per triplet. Register pressure is
    # higher than MVMR due to the 3D `block_o` accumulator. Smaller M/C
    # tiles often help fit in registers; more warps help hide latency.

    # ── L=32: high parallelism ──────────────────────────────────────────────
    triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 16}, num_warps=1),
    triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),

    # ── L=64: balanced ──────────────────────────────────────────────────────
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 16}, num_warps=1),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 16}, num_warps=2),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
    triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 64}, num_warps=4),

    # ── L=128: original pivot + larger tiles for high-C stages ──────────────
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=1),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 16}, num_warps=2),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 16}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 128}, num_warps=4),
    triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),

    # ── L=256: more sequential ──────────────────────────────────────────────
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 16}, num_warps=2),
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64}, num_warps=4),
    triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["G", "M", "C"], reset_to_zero=["o"])
@triton.jit
def sparse_vector_vector_outer_product_reduction_kernel(
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
    num_pid_g = tl.cdiv(G, BLOCK_SIZE_G)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_c = tl.cdiv(C, BLOCK_SIZE_C)

    pid = tl.program_id(axis=0)
    pid_c = pid % num_pid_c
    pid //= num_pid_c
    pid_m = pid % num_pid_m
    pid //= num_pid_m
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
    gmc_mask = g_mask[:, None, None] & m_mask[None, :, None] & c_mask[None, None, :]

    a_ptrs = a + (g_offsets[:, None] * M + m_offsets[None, :])
    b_ptrs = b + (g_offsets[:, None] * C + c_offsets[None, :])
    o_ptrs = o + (
        g_offsets[:, None, None] * (M * C)
        + m_offsets[None, :, None] * C
        + c_offsets[None, None, :]
    )

    t_offset = pid_t * L
    a_offset = tl.load(a_idx + t_offset)
    b_offset = tl.load(b_idx + t_offset)
    o_offset = tl.load(o_idx + t_offset)

    # Load with `other=0.0` to ensure masked lanes don't contribute.
    # Cast to fp32 so the per-triplet outer product AND the running
    # accumulator are both fp32 — fixes the fp16 precision loss across L=128
    # accumulations that the previous implementation suffered from.
    block_a = tl.load(a_ptrs + a_offset * (G * M), mask=gm_mask, other=0.0).to(tl.float32)
    block_b = tl.load(b_ptrs + b_offset * (G * C), mask=gc_mask, other=0.0).to(tl.float32)
    block_o = block_a[:, :, None] * block_b[:, None, :]
    for t in tl.range(1, min(L, T - t_offset)):
        a_offset_next = tl.load(a_idx + t_offset + t)
        b_offset_next = tl.load(b_idx + t_offset + t)
        o_offset_next = tl.load(o_idx + t_offset + t)

        if a_offset_next != a_offset:
            block_a = tl.load(a_ptrs + a_offset_next * (G * M), mask=gm_mask, other=0.0).to(tl.float32)
            a_offset = a_offset_next

        if b_offset_next != b_offset:
            block_b = tl.load(b_ptrs + b_offset_next * (G * C), mask=gc_mask, other=0.0).to(tl.float32)
            b_offset = b_offset_next

        if o_offset_next != o_offset:
            tl.atomic_add(o_ptrs + o_offset * (G * M * C), block_o, mask=gmc_mask)
            o_offset = o_offset_next
            block_o = block_a[:, :, None] * block_b[:, None, :]
        else:
            block_o += block_a[:, :, None] * block_b[:, None, :]
    tl.atomic_add(o_ptrs + o_offset * (G * M * C), block_o, mask=gmc_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped (segment-based) variant for tensor-core utilisation.
#
# In VVOR backward (weight grad) the output index `o_idx` is the kernel
# offset; with `sort_by="k"` the triplets sorted by `k` group consecutive
# triplets onto the same `o_offset`. We exploit that by batching L_CHUNK
# triplets into a tensor-core GEMM:
#
#   a_chunk = (L_CHUNK, M) gathered from a via a_idx[seg_start : seg_start+L]
#   b_chunk = (L_CHUNK, C) gathered from b via b_idx[seg_start : seg_start+L]
#   weight_grad_tile = a_chunk.T @ b_chunk     # (M, L_CHUNK) @ (L_CHUNK, C) → (M, C)
#                                              # K_inner = L_CHUNK ≥ 16 → tensor cores fire,
#                                              # FIXING the K=1 issue of the per-triplet path
#   atomic_add(o[k_offset], weight_grad_tile)
#
# Grid parameter `pid_chunk` indexes the flat list of all chunks across
# kernel offsets; the kernel locates its kernel offset (= the o_offset
# for this VVOR pass) via a constexpr scan over `chunk_seg_offs`.
# ─────────────────────────────────────────────────────────────────────────────


_GROUPED_AUTOTUNE_CONFIGS = [
    # L_CHUNK ∈ {16, 32, 64} — see mvmr_triton_kernel for rationale.
    # ── L_CHUNK=32 ──────────────────────────────────────────────────────────
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

    # ── L_CHUNK=64 — deeper K_inner for tensor-core mma ────────────────────
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 256}, num_warps=8),
    triton.Config({"L_CHUNK": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_C": 128}, num_warps=8),

    # ── L_CHUNK=16 — short avg seg length ──────────────────────────────────
    triton.Config({"L_CHUNK": 16, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32,  "BLOCK_SIZE_C": 32},  num_warps=2),
    triton.Config({"L_CHUNK": 16, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 64},  num_warps=4),

    # ── L_CHUNK=128 — biggest K_inner GEMM per program ─────────────────────
    triton.Config({"L_CHUNK": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),
    triton.Config({"L_CHUNK": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 256}, num_warps=8),

    # ── L_CHUNK=256 — VVOR's K_inner is L_CHUNK; bigger = better tensor-core
    # mma utilization (256×128×128 mma is 2× the work of 128×128×128).
    # Output tile is (BLOCK_M, BLOCK_C) — same as smaller L_CHUNK configs;
    # only K_inner grows, so register pressure on output is unchanged.
    triton.Config({"L_CHUNK": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_C": 64},  num_warps=4),
    triton.Config({"L_CHUNK": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8),
]


@triton.autotune(configs=_GROUPED_AUTOTUNE_CONFIGS, key=["G", "M", "C"], reset_to_zero=["o"])
@triton.jit
def sparse_vector_vector_outer_product_reduction_grouped_kernel(
    a,                # (N_a, G, M)
    a_idx,            # (T,)
    b,                # (N_b, G, C)
    b_idx,            # (T,)
    o,                # (N_o, G, M, C)  — N_o is the number of kernel offsets here
    seg_offs,         # (K_offsets + 1,)  — segment starts in the triplet arrays
                      #   (built from the o_idx that maps triplets → kernel offsets)
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
    pid_c = pid % num_pid_c
    pid //= num_pid_c
    pid_m = pid % num_pid_m
    pid //= num_pid_m
    pid_g = pid % num_pid_g
    pid_chunk = pid // num_pid_g

    # Locate kernel offset by walking seg_offs and accumulating
    # ceil(seg_len / L_CHUNK). See mvmr_triton_kernel for rationale.
    # Cast seg_offs loads to int32 to avoid loop-carried type mismatch.
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
    gmc_mask = g_mask[:, None, None] & m_mask[None, :, None] & c_mask[None, None, :]

    # Gather indices for this chunk.
    a_idx_chunk = tl.load(a_idx + l_offsets, mask=l_mask, other=0)
    b_idx_chunk = tl.load(b_idx + l_offsets, mask=l_mask, other=0)

    # Gather a (gradients): (L_CHUNK, G, M_tile).
    a_ptrs = a + (
        a_idx_chunk[:, None, None] * (G * M)
        + g_offsets[None, :, None] * M
        + m_offsets[None, None, :]
    )
    a_load_mask = l_mask[:, None, None] & gm_mask[None, :, :]
    block_a = tl.load(a_ptrs, mask=a_load_mask, other=0.0).to(tl.float32)

    # Gather b (features): (L_CHUNK, G, C_tile).
    b_ptrs = b + (
        b_idx_chunk[:, None, None] * (G * C)
        + g_offsets[None, :, None] * C
        + c_offsets[None, None, :]
    )
    b_load_mask = l_mask[:, None, None] & gc_mask[None, :, :]
    block_b = tl.load(b_ptrs, mask=b_load_mask, other=0.0).to(tl.float32)

    if BLOCK_SIZE_G == 1:
        # tl.dot path: a^T @ b for the L_CHUNK rows.
        # block_a: (L_CHUNK, 1, M_tile) → (L_CHUNK, M_tile).T = (M_tile, L_CHUNK)
        # block_b: (L_CHUNK, 1, C_tile) → (L_CHUNK, C_tile)
        # result:  (M_tile, L_CHUNK) @ (L_CHUNK, C_tile) → (M_tile, C_tile)
        a_2d = tl.reshape(block_a, (L_CHUNK, BLOCK_SIZE_M))
        b_2d = tl.reshape(block_b, (L_CHUNK, BLOCK_SIZE_C))
        # tl.trans for transpose; tl.dot needs (M_tile, K) @ (K, N).
        a_2d_T = tl.trans(a_2d)
        out_2d = tl.dot(a_2d_T, b_2d, out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION)   # (M_tile, C_tile)
        out_3d = tl.reshape(out_2d, (BLOCK_SIZE_G, BLOCK_SIZE_M, BLOCK_SIZE_C))
    else:
        # Group-conv fallback (G > 1 not in production today).
        # block_a: (L_CHUNK, G, M)        block_b: (L_CHUNK, G, C)
        # outer + reduce over L:
        out_3d = tl.sum(
            block_a[:, :, :, None] * block_b[:, :, None, :],
            axis=0,
        )

    # Atomic-add to the kernel-offset's weight-grad tile.
    o_ptr = o + k_offset * (G * M * C) + (
        g_offsets[:, None, None] * (M * C)
        + m_offsets[None, :, None] * C
        + c_offsets[None, None, :]
    )
    tl.atomic_add(o_ptr, out_3d, mask=gmc_mask)
