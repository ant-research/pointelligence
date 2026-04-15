import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # L: triplets per block. Larger L = more reuse of loaded a/b blocks when
        # consecutive triplets share the same a_offset or b_offset.
        # BLOCK_SIZE_M/C: tile size for output/input channel dimensions.
        # num_warps: thread-level parallelism within a block.

        # --- L=32: high parallelism, good for small T ---
        triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 16}, num_warps=1),
        triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
        triton.Config({"L": 32, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
        # --- L=64: balanced ---
        triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_C": 16}, num_warps=1),
        triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
        triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
        triton.Config({"L": 64, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 64}, num_warps=4),
        # --- L=128: original config + variants ---
        triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=1),
        triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
        triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 32}, num_warps=4),
        triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 64}, num_warps=4),
        triton.Config({"L": 128, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),
        # --- L=256: more sequential, better reuse for large T ---
        triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_C": 32}, num_warps=2),
        triton.Config({"L": 256, "BLOCK_SIZE_G": 1, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_C": 64}, num_warps=4),
    ],
    key=["T", "G", "M", "C"],
    reset_to_zero=["o"],
)
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
    num_pid_g = tl.cdiv(G, BLOCK_SIZE_G)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_c = tl.cdiv(C, BLOCK_SIZE_C)

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

    # Native fp16: load, multiply, and reduce all in native dtype.
    # Accumulator block_o stays in native dtype within each o_offset segment.
    # The fp32 output buffer + atomic_add provides final precision.
    block_a = tl.load(a_ptrs + a_offset * (G * C * M), mask=gcm_mask)
    block_b = tl.load(b_ptrs + b_offset * (G * C), mask=gc_mask)
    block_o = tl.sum(block_a * block_b[:, :, None], axis=1)
    for t in tl.range(1, min(L, T - t_offset)):
        a_offset_next = tl.load(a_idx + t_offset + t)
        b_offset_next = tl.load(b_idx + t_offset + t)
        o_offset_next = tl.load(o_idx + t_offset + t)

        if a_offset_next != a_offset:
            block_a = tl.load(a_ptrs + a_offset_next * (G * C * M), mask=gcm_mask)
            a_offset = a_offset_next

        if b_offset_next != b_offset:
            block_b = tl.load(b_ptrs + b_offset_next * (G * C), mask=gc_mask)
            b_offset = b_offset_next

        if o_offset_next != o_offset:
            tl.atomic_add(o_ptrs + o_offset * (G * M), block_o.to(tl.float32), mask=gm_mask)
            o_offset = o_offset_next
            block_o = tl.sum(block_a * block_b[:, :, None], axis=1)
        else:
            block_o += tl.sum(block_a * block_b[:, :, None], axis=1)
    tl.atomic_add(o_ptrs + o_offset * (G * M), block_o.to(tl.float32), mask=gm_mask)
