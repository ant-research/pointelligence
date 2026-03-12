import torch
import triton
from torch.library import triton_op, wrap_triton

from .indexed_segment_reduce_triton_kernel import (
    indexed_segment_reduce_fwd_kernel,
    indexed_segment_reduce_bwd_kernel,
)


def get_block_k(C, reduce):
    target_threads = 256
    block_c = triton.next_power_of_2(C)

    # Reduce block size for MAX/MIN (Modes 2, 3)
    # Even without tracking indices, lower register pressure helps
    if reduce in [2, 3]:
        target_threads = 128

    block_k = max(1, target_threads // block_c)
    return min(block_k, 128)


@triton_op("sparse_engines::indexed_segment_reduce_op", mutates_args={})
def indexed_segment_reduce_op(
        x: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        op_enum: int,
) -> torch.Tensor: # Returns only y now

    T, C = x.shape
    K = lengths.shape[0]

    y = x.new_empty((K, C))
    # arg tensor allocation removed

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_K = get_block_k(C, op_enum)

    grid = (triton.cdiv(K, BLOCK_K),)

    wrap_triton(indexed_segment_reduce_fwd_kernel)[grid](
        x,
        indices,
        offsets,
        lengths,
        y,
        # arg removed
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        T,
        C,
        K,
        OP_TYPE=op_enum,
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
    )
    return y


def _setup_context_indexed_segment_reduce(ctx, inputs, output):
    x, indices, offsets, lengths, op_enum = inputs
    y = output

    # Context saving optimized:
    # Only save x and y for Max/Min (op_enum 2 or 3)
    # For Sum/Mean, we don't need x or y in backward.
    tensors_to_save = [indices, offsets, lengths]
    if op_enum in [2, 3]:
        tensors_to_save.append(x)
        tensors_to_save.append(y)

    ctx.save_for_backward(*tensors_to_save)
    ctx.op_enum = op_enum
    ctx.x_shape = x.shape
    ctx.x_dtype = x.dtype
    ctx.x_device = x.device
    ctx.x_strides = x.stride()
    ctx.y_strides = y.stride()


def _backward_indexed_segment_reduce(ctx, grad_y):
    saved = ctx.saved_tensors
    indices = saved[0]
    offsets = saved[1]
    lengths = saved[2]

    op_enum = ctx.op_enum

    # Recover x and y only if strictly needed
    x, y = None, None
    if op_enum in [2, 3]:
        x = saved[3]
        y = saved[4]

    grad_x = torch.zeros(ctx.x_shape, dtype=ctx.x_dtype, device=ctx.x_device)
    grad_y = grad_y.contiguous()
    K, C = grad_y.shape

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_K = get_block_k(C, op_enum)

    num_warps = 4 if (BLOCK_C * BLOCK_K) >= 128 else 2
    grid = (triton.cdiv(K, BLOCK_K),)

    # Handle strides for x and y even if they are None (pass 0/dummy)
    stride_x_t, stride_x_c = ctx.x_strides
    stride_y_k, stride_y_c = ctx.y_strides

    wrap_triton(indexed_segment_reduce_bwd_kernel)[grid](
        grad_y,
        indices,
        offsets,
        lengths,
        x if x is not None else indices, # pass valid pointer placeholder if unused
        y if y is not None else indices, # pass valid pointer placeholder if unused
        grad_x,
        grad_y.stride(0),
        grad_y.stride(1),
        grad_x.stride(0),
        grad_x.stride(1),
        stride_x_t,
        stride_x_c,
        stride_y_k,
        stride_y_c,
        K,
        C,
        OP_TYPE=op_enum,
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=1,
    )
    return grad_x, None, None, None, None


indexed_segment_reduce_op.register_autograd(
    _backward_indexed_segment_reduce,
    setup_context=_setup_context_indexed_segment_reduce,
)


def indexed_segment_reduce(
        x,
        reduce,
        indices,
        *,
        lengths,
):
    reduce_map = {"sum": 0, "mean": 1, "max": 2, "min": 3}
    if reduce not in reduce_map:
        raise ValueError(f"Mode must be one of {list(reduce_map.keys())}")

    x = x.contiguous()

    zero = torch.tensor([0], device=lengths.device, dtype=lengths.dtype)
    offsets = torch.cat([zero, torch.cumsum(lengths, 0)[:-1]]).contiguous()

    # Op now returns single tensor, unpacking removed
    y = torch.ops.sparse_engines.indexed_segment_reduce_op(
        x, indices, offsets, lengths, reduce_map[reduce]
    )
    return y