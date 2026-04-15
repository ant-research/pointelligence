import torch
from torch import Tensor

import triton
from torch.library import triton_op, wrap_triton

from .vvor_triton_kernel import sparse_vector_vector_outer_product_reduction_kernel


@triton_op(
    "sparse_engines::sparse_vector_vector_outer_product_reduction", mutates_args={}
)
def sparse_vector_vector_outer_product_reduction(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n_o: int
) -> Tensor:
    a = a.contiguous()
    b = b.contiguous()

    T, G, M, C = a_idx.numel(), a.shape[1], a.shape[2], b.shape[2]
    input_dtype = a.dtype
    # Accumulate in fp32 for numerical stability, cast back after
    o = torch.zeros((n_o, G, M, C), dtype=torch.float32, device=a.device)

    grid = lambda META: (
        triton.cdiv(T, META["L"])
        * triton.cdiv(G, META["BLOCK_SIZE_G"])
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(C, META["BLOCK_SIZE_C"]),
    )
    wrap_triton(sparse_vector_vector_outer_product_reduction_kernel)[grid](
        a, a_idx, b, b_idx, o, o_idx, T, G, M, C
    )

    return o.to(input_dtype) if input_dtype != torch.float32 else o


def _backward_sparse_vector_vector_outer_product_reduction(ctx, grad):
    from .mvmr_triton import sparse_matrix_vector_multiplication_reduction

    a, a_idx, b, b_idx, o_idx = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = sparse_matrix_vector_multiplication_reduction(
            grad.transpose(2, 3),
            o_idx,
            b,
            b_idx,
            a_idx,
            a.shape[0] if isinstance(a, torch.Tensor) else ctx.a_shape_0,
        )
    if ctx.needs_input_grad[2]:
        grad_b = sparse_matrix_vector_multiplication_reduction(
            grad,
            o_idx,
            a,
            a_idx,
            b_idx,
            b.shape[0] if isinstance(b, torch.Tensor) else ctx.b_shape_0,
        )
    return grad_a, None, grad_b, None, None, None


def _setup_context_sparse_vector_vector_outer_product_reduction(ctx, inputs, output):
    a, a_idx, b, b_idx, o_idx, n = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[2]:
        saved_a = a
    ctx.save_for_backward(saved_a, a_idx, saved_b, b_idx, o_idx)
    ctx.a_shape_0 = a.shape[0]
    ctx.b_shape_0 = b.shape[0]


sparse_vector_vector_outer_product_reduction.register_autograd(
    _backward_sparse_vector_vector_outer_product_reduction,
    setup_context=_setup_context_sparse_vector_vector_outer_product_reduction,
)
