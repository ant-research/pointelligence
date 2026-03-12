import triton
import triton.language as tl


@triton.jit
def large_segment_reduce_kernel(
    X_ptr,  # [N, C]
    Out_ptr,  # [K, C]
    SegmentIds_ptr,  # [N]
    stride_xn,
    stride_xc,
    stride_outk,
    stride_outc,
    N,
    C,
    OP_TYPE: tl.constexpr,  # 0=SUM, 1=MEAN, 2=MAX, 3=MIN
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    # 1. Define Block Bounds
    start_n = pid_n * BLOCK_N
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # 2. Check Uniformity (Fast Path Detection)
    # Load the segment ID of the first and last element of this chunk.
    # Note: We must handle the case where start_n >= N safely.

    off_first = start_n
    id_first = tl.load(SegmentIds_ptr + off_first, mask=off_first < N, other=-1)

    # We load the ID of the last element of the chunk (or N-1 if chunk truncates)
    off_last = min(start_n + BLOCK_N - 1, N - 1)
    id_last = tl.load(SegmentIds_ptr + off_last, mask=off_last < N, other=-2)

    # Fast Path condition: Valid segment AND start ID equals end ID
    is_uniform = (id_first == id_last) and (id_first != -1)

    # --- FAST PATH: Entire chunk is one segment ---
    if is_uniform:
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        x_ptrs = X_ptr + (n_offs[:, None] * stride_xn) + (c_offs[None, :] * stride_xc)

        # Setup pad_val for the reduction
        pad_val = 0.0
        if OP_TYPE == 2:
            pad_val = float("-inf")
        elif OP_TYPE == 3:
            pad_val = float("inf")

        # Vectorized Load [BLOCK_N, BLOCK_C]
        val = tl.load(x_ptrs, mask=n_mask[:, None] & c_mask[None, :], other=pad_val)

        if OP_TYPE == 0 or OP_TYPE == 1:  # Sum / Mean(sum part)
            agg = tl.sum(val, axis=0)
            tl.atomic_add(
                Out_ptr + id_first * stride_outk + c_offs * stride_outc,
                agg,
                mask=c_mask,
            )

        elif OP_TYPE == 2:  # Max
            agg = tl.max(val, axis=0)
            tl.atomic_max(
                Out_ptr + id_first * stride_outk + c_offs * stride_outc,
                agg,
                mask=c_mask,
            )

        elif OP_TYPE == 3:  # Min
            agg = tl.min(val, axis=0)
            tl.atomic_min(
                Out_ptr + id_first * stride_outk + c_offs * stride_outc,
                agg,
                mask=c_mask,
            )

    # --- SLOW PATH: Segment boundaries inside chunk ---
    else:
        # We iterate sequentially row-by-row.

        # Initialize accumulator
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        if OP_TYPE == 2:
            acc = tl.full((BLOCK_C,), float("-inf"), dtype=tl.float32)
        elif OP_TYPE == 3:
            acc = tl.full((BLOCK_C,), float("inf"), dtype=tl.float32)

        curr_seg = id_first  # Start with the first ID found

        for i in range(0, BLOCK_N):
            curr_n = start_n + i
            if curr_n < N:
                # 1. Load Segment ID for this row
                row_seg = tl.load(SegmentIds_ptr + curr_n)

                # 2. Check for segment change
                if row_seg != curr_seg:
                    # Flush accumulator to old segment
                    if curr_seg != -1:
                        out_ptrs = (
                            Out_ptr + curr_seg * stride_outk + c_offs * stride_outc
                        )
                        if OP_TYPE == 0 or OP_TYPE == 1:
                            tl.atomic_add(out_ptrs, acc, mask=c_mask)
                        elif OP_TYPE == 2:
                            tl.atomic_max(out_ptrs, acc, mask=c_mask)
                        elif OP_TYPE == 3:
                            tl.atomic_min(out_ptrs, acc, mask=c_mask)

                    # Reset accumulator
                    curr_seg = row_seg
                    if OP_TYPE == 0 or OP_TYPE == 1:
                        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
                    elif OP_TYPE == 2:
                        acc = tl.full((BLOCK_C,), float("-inf"), dtype=tl.float32)
                    elif OP_TYPE == 3:
                        acc = tl.full((BLOCK_C,), float("inf"), dtype=tl.float32)

                # 3. Load Data Row
                row_val_ptr = X_ptr + (curr_n * stride_xn) + (c_offs * stride_xc)

                pad_val = 0.0
                if OP_TYPE == 2:
                    pad_val = float("-inf")
                elif OP_TYPE == 3:
                    pad_val = float("inf")

                row_val = tl.load(row_val_ptr, mask=c_mask, other=pad_val)

                # 4. Accumulate
                if OP_TYPE == 0 or OP_TYPE == 1:
                    acc += row_val
                elif OP_TYPE == 2:
                    acc = tl.maximum(acc, row_val)
                elif OP_TYPE == 3:
                    acc = tl.minimum(acc, row_val)

        # Final flush at end of chunk
        if curr_seg != -1:
            out_ptrs = Out_ptr + curr_seg * stride_outk + c_offs * stride_outc
            if OP_TYPE == 0 or OP_TYPE == 1:
                tl.atomic_add(out_ptrs, acc, mask=c_mask)
            elif OP_TYPE == 2:
                tl.atomic_max(out_ptrs, acc, mask=c_mask)
            elif OP_TYPE == 3:
                tl.atomic_min(out_ptrs, acc, mask=c_mask)


@triton.jit
def large_segment_sum_backward_kernel(
    GradOut_ptr,
    GradIn_ptr,
    SegmentIds_ptr,
    stride_gout_k,
    stride_gout_c,
    stride_gin_n,
    stride_gin_c,
    N,
    C,
    OP_TYPE: tl.constexpr,
    Lengths_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    k_ids = tl.load(SegmentIds_ptr + n_offs, mask=n_mask, other=0)
    gout_ptrs = (
        GradOut_ptr
        + (k_ids[:, None] * stride_gout_k)
        + (c_offs[None, :] * stride_gout_c)
    )
    grad_val = tl.load(gout_ptrs, mask=n_mask[:, None] & c_mask[None, :], other=0.0)
    if OP_TYPE == 1:
        lens = tl.load(Lengths_ptr + k_ids, mask=n_mask, other=1.0)
        lens = tl.where(lens > 0, lens, 1.0)
        grad_val = grad_val / lens[:, None]
    gin_ptrs = (
        GradIn_ptr + (n_offs[:, None] * stride_gin_n) + (c_offs[None, :] * stride_gin_c)
    )
    tl.store(gin_ptrs, grad_val, mask=n_mask[:, None] & c_mask[None, :])


@triton.jit
def large_segment_minmax_backward_kernel(
    GradOut_ptr,  # [K, C]
    X_ptr,  # [N, C]
    OutVal_ptr,  # [K, C] (The computed max/min values)
    SegmentIds_ptr,  # [N]
    GradIn_ptr,  # [N, C]
    stride_gout_k,
    stride_gout_c,
    stride_xn,
    stride_xc,
    stride_outk,
    stride_outc,
    stride_gin_n,
    stride_gin_c,
    N,
    C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # 1. Identify Segment
    seg_ids = tl.load(SegmentIds_ptr + n_offs, mask=n_mask, other=0)

    # 2. Load Input X
    x_ptrs = X_ptr + (n_offs[:, None] * stride_xn) + (c_offs[None, :] * stride_xc)
    x_val = tl.load(x_ptrs, mask=n_mask[:, None] & c_mask[None, :], other=0.0)

    # 3. Load Computed Max/Min Value for this segment
    out_ptrs = (
        OutVal_ptr + (seg_ids[:, None] * stride_outk) + (c_offs[None, :] * stride_outc)
    )
    out_val = tl.load(out_ptrs, mask=n_mask[:, None] & c_mask[None, :], other=0.0)

    # 4. Compare
    mask_match = x_val == out_val

    # 5. Load Gradient Out
    gout_ptrs = (
        GradOut_ptr
        + (seg_ids[:, None] * stride_gout_k)
        + (c_offs[None, :] * stride_gout_c)
    )
    grad_out = tl.load(gout_ptrs, mask=n_mask[:, None] & c_mask[None, :], other=0.0)

    # 6. Compute Gradient In
    grad_in = tl.where(mask_match, grad_out, 0.0)

    # 7. Store
    gin_ptrs = (
        GradIn_ptr + (n_offs[:, None] * stride_gin_n) + (c_offs[None, :] * stride_gin_c)
    )
    tl.store(gin_ptrs, grad_in, mask=n_mask[:, None] & c_mask[None, :])
