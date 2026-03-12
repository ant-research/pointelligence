import torch
import triton
from torch.library import triton_op, wrap_triton
from typing import Tuple, Optional

from .large_segment_reduce_triton_kernel import (
    large_segment_reduce_kernel,
    large_segment_sum_backward_kernel,
    large_segment_minmax_backward_kernel,
)


def _get_segment_ids(lengths, N):
    if lengths.numel() == 0:
        return torch.zeros((0,), dtype=torch.int32, device=lengths.device)

    # repeat_interleave is torch.compile compatible
    ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.int32),
        lengths,
    )
    return ids


def large_segment_reduce_fwd(x, lengths, reduce_op_str):
    N, C = x.shape
    K = lengths.shape[0]

    # 1. Prepare Inputs
    segment_ids = _get_segment_ids(lengths, N)

    # 2. Configs
    # BLOCK_N = 64 balances the atomic fast path (vectorized) with the slow path loop overhead.
    BLOCK_N = 64
    # BLOCK_C is capped to ensure we don't spill registers in the inner loop of the slow path.
    BLOCK_C = min(triton.next_power_of_2(int(C)), 64)

    op_map = {"sum": 0, "mean": 1, "max": 2, "min": 3}
    op_enum = op_map[reduce_op_str]

    output_dtype = torch.float32 if x.dtype == torch.float16 else x.dtype

    # 3. Allocation
    if reduce_op_str in ["sum", "mean"]:
        out = torch.zeros((K, C), device=x.device, dtype=output_dtype)
    elif reduce_op_str == "max":
        out = torch.full((K, C), float("-inf"), device=x.device, dtype=output_dtype)
    elif reduce_op_str == "min":
        out = torch.full((K, C), float("inf"), device=x.device, dtype=output_dtype)
    # 4. Kernel Launch
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C))

    wrap_triton(large_segment_reduce_kernel)[grid](
        x,
        out,
        segment_ids,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        N,
        C,
        OP_TYPE=op_enum,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    if reduce_op_str == "mean":
        safe_lens = lengths.clone()
        safe_lens[safe_lens == 0] = 1
        out = out / safe_lens.unsqueeze(1)
        if output_dtype == torch.float32 and x.dtype == torch.float16:
            fp16_max = 65504.
            out = torch.clamp(out, min=-fp16_max, max=fp16_max)

    return out, segment_ids


# -----------------------------------------------------------------------------
# Autograd & Registration
# -----------------------------------------------------------------------------


@triton_op("sparse_engines::large_segment_reduce", mutates_args={})
def large_segment_reduce_op(
    x: torch.Tensor, lengths: torch.Tensor, reduce: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    return large_segment_reduce_fwd(x, lengths, reduce)


def _backward_large_segment_reduce(ctx, grad_out, grad_seg_ids):
    # Retrieve saved tensors
    if ctx.reduce_op in ["max", "min"]:
        lengths, out_val, segment_ids, x_in = ctx.saved_tensors
    else:
        lengths, segment_ids = ctx.saved_tensors

    N = ctx.N
    C = ctx.C
    reduce_op = ctx.reduce_op

    grad_in = torch.zeros((N, C), device=grad_out.device, dtype=grad_out.dtype)

    # Cap BLOCK_C for backward to match forward consistency
    BLOCK_C = min(triton.next_power_of_2(int(C)), 64)

    if reduce_op in ["sum", "mean"]:
        BLOCK_N = 128
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C))
        op_enum = 1 if reduce_op == "mean" else 0

        wrap_triton(large_segment_sum_backward_kernel)[grid](
            grad_out,
            grad_in,
            segment_ids,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_in.stride(0),
            grad_in.stride(1),
            N,
            C,
            OP_TYPE=op_enum,
            Lengths_ptr=lengths,
            BLOCK_N=BLOCK_N,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

    elif reduce_op in ["max", "min"]:
        # Recompute gradients based on value matching
        BLOCK_N = 64
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C))

        wrap_triton(large_segment_minmax_backward_kernel)[grid](
            grad_out,
            x_in,
            out_val,
            segment_ids,
            grad_in,
            grad_out.stride(0),
            grad_out.stride(1),
            x_in.stride(0),
            x_in.stride(1),
            out_val.stride(0),
            out_val.stride(1),
            grad_in.stride(0),
            grad_in.stride(1),
            N,
            C,
            BLOCK_N=BLOCK_N,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

    return grad_in, None, None


def _setup_context_large_segment_reduce(ctx, inputs, output):
    x, lengths, reduce_op = inputs
    out, segment_ids = output

    ctx.N = x.shape[0]
    ctx.C = x.shape[1]
    ctx.reduce_op = reduce_op

    # Save optimized subset of tensors
    if reduce_op in ["max", "min"]:
        ctx.save_for_backward(lengths, out, segment_ids, x)
    else:
        ctx.save_for_backward(lengths, segment_ids)


large_segment_reduce_op.register_autograd(
    _backward_large_segment_reduce, setup_context=_setup_context_large_segment_reduce
)


def large_segment_reduce(x, reduce, *, lengths):
    """
    Public Interface.
    """
    out, _ = large_segment_reduce_op(x, lengths, reduce)
    if out.dtype == torch.float32 and x.dtype == torch.float16:
        fp16_max = 65504.
        out = torch.clamp(out, min=-fp16_max, max=fp16_max)
        out = out.to(x.dtype)
    return out
