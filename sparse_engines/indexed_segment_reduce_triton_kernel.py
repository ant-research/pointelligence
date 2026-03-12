import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Forward Kernel
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"num_warps": 2}, num_stages=2),
        triton.Config({"num_warps": 4}, num_stages=2),
        triton.Config({"num_warps": 4}, num_stages=4),
        triton.Config({"num_warps": 8}, num_stages=2),
    ],
    key=["BLOCK_C", "BLOCK_K", "OP_TYPE"],
)
@triton.jit
def indexed_segment_reduce_fwd_kernel(
    x_ptr,
    indices_ptr,
    offsets_ptr,
    lengths_ptr,
    y_ptr,
    # arg_ptr removed (optimization)
    stride_x_t,
    stride_x_c,
    stride_y_k,
    stride_y_c,
    T,
    C,
    K,
    OP_TYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    # 1. Define bounds for the tile
    k_start = pid * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    # 2. Load Metadata
    curr_offsets = tl.load(offsets_ptr + offs_k, mask=mask_k, other=0).to(tl.int64)
    curr_lengths = tl.load(lengths_ptr + offs_k, mask=mask_k, other=0).to(tl.int64)
    max_len = tl.max(curr_lengths, axis=0)

    # 3. Setup Accumulators
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    if OP_TYPE == 0 or OP_TYPE == 1:  # Sum or Mean
        acc = tl.zeros([BLOCK_K, BLOCK_C], dtype=tl.float32)
    else:  # Max or Min
        neutral = float("-inf") if (OP_TYPE == 2) else float("inf")
        acc = tl.full([BLOCK_K, BLOCK_C], neutral, dtype=tl.float32)

    # 4. Reduction Loop
    for i in range(max_len):
        mask_k_active = (i < curr_lengths) & mask_k

        idx_lookup = curr_offsets + i
        idx_in_x = tl.load(indices_ptr + idx_lookup, mask=mask_k_active, other=0).to(
            tl.int64
        )

        mask_kc_active = mask_k_active[:, None] & mask_c[None, :]
        src_ptrs = (
            x_ptr + (idx_in_x[:, None] * stride_x_t) + (offs_c[None, :] * stride_x_c)
        )

        if OP_TYPE == 0 or OP_TYPE == 1:  # Sum / Mean
            val = tl.load(src_ptrs, mask=mask_kc_active, other=0.0)
            acc += val
        else:  # Max / Min
            neutral = float("-inf") if (OP_TYPE == 2) else float("inf")
            val = tl.load(src_ptrs, mask=mask_kc_active, other=neutral)
            if OP_TYPE == 2:  # Max
                acc = tl.maximum(acc, val)
            else:  # Min
                acc = tl.minimum(acc, val)

    # 5. Finalize and Store
    mask_kc = mask_k[:, None] & mask_c[None, :]

    if OP_TYPE == 1:  # Mean
        acc = acc / curr_lengths[:, None].to(tl.float32)

    dst_ptrs = y_ptr + (offs_k[:, None] * stride_y_k) + (offs_c[None, :] * stride_y_c)
    tl.store(dst_ptrs, acc, mask=mask_kc)


# -----------------------------------------------------------------------------
# Backward Kernel
# -----------------------------------------------------------------------------
@triton.jit
def indexed_segment_reduce_bwd_kernel(
    grad_y_ptr,
    indices_ptr,
    offsets_ptr,
    lengths_ptr,
    x_ptr,  # Added for recompute
    y_ptr,  # Added for recompute
    grad_x_ptr,
    stride_gy_k,
    stride_gy_c,
    stride_gx_t,
    stride_gx_c,
    stride_x_t,  # Added
    stride_x_c,  # Added
    stride_y_k,  # Added
    stride_y_c,  # Added
    K,
    C,
    OP_TYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    k_start = pid * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    # Metadata Load
    curr_offsets = tl.load(offsets_ptr + offs_k, mask=mask_k, other=0).to(tl.int64)
    curr_lengths = tl.load(lengths_ptr + offs_k, mask=mask_k, other=0).to(tl.int64)
    max_len = tl.max(curr_lengths, axis=0)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    mask_kc = mask_k[:, None] & mask_c[None, :]

    # Load Grad Y
    gy_ptrs = (
        grad_y_ptr + (offs_k[:, None] * stride_gy_k) + (offs_c[None, :] * stride_gy_c)
    )
    grad_val = tl.load(gy_ptrs, mask=mask_kc, other=0.0)

    # Normalize if Mean
    if OP_TYPE == 1:
        grad_val = grad_val / curr_lengths[:, None].to(tl.float32)

    # For Max/Min recompute: Load the forward output Y
    # NOTE: If OP_TYPE is 0/1, this branch is compiled out, so y_ptr can be null
    target_val = tl.zeros([BLOCK_K, BLOCK_C], dtype=tl.float32)
    if OP_TYPE == 2 or OP_TYPE == 3:
        y_loc_ptrs = (
            y_ptr + (offs_k[:, None] * stride_y_k) + (offs_c[None, :] * stride_y_c)
        )
        target_val = tl.load(y_loc_ptrs, mask=mask_kc, other=0.0)

    # Unified Loop for all ops
    for i in range(max_len):
        mask_k_active = (i < curr_lengths) & mask_k
        idx_lookup = curr_offsets + i
        idx_in_x = tl.load(indices_ptr + idx_lookup, mask=mask_k_active, other=0).to(
            tl.int64
        )

        mask_kc_active = mask_k_active[:, None] & mask_c[None, :]

        # Calculate destination pointer in Grad X
        target_gx_ptrs = (
            grad_x_ptr
            + (idx_in_x[:, None] * stride_gx_t)
            + (offs_c[None, :] * stride_gx_c)
        )

        # Logic Dispatch
        if OP_TYPE == 0 or OP_TYPE == 1:
            # Sum/Mean: Just scatter grad_val
            tl.atomic_add(target_gx_ptrs, grad_val, mask=mask_kc_active)
        else:
            # Max/Min: Check for value match
            neutral = float("-inf") if (OP_TYPE == 2) else float("inf")

            src_x_ptrs = (
                x_ptr
                + (idx_in_x[:, None] * stride_x_t)
                + (offs_c[None, :] * stride_x_c)
            )
            # Load original X
            val_x = tl.load(src_x_ptrs, mask=mask_kc_active, other=neutral)

            # Recompute check: Does this x match the segment reduction result?
            is_match = val_x == target_val

            # Only apply gradient where values match
            mask_final = mask_kc_active & is_match
            tl.atomic_add(target_gx_ptrs, grad_val, mask=mask_final)
