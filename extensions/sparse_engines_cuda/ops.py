import torch
from torch import Tensor

__all__ = [
    "sparse_matrix_vector_multiplication_reduction",
    "sparse_vector_vector_outer_product_reduction",
    "indexed_distance",
    "bucket_arrange",
]

def indexed_distance(a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, distance_type: int = 0) -> Tensor:
    return torch.ops.sparse_engines_cuda.indexed_distance(a, a_idx, b, b_idx, distance_type)

def bucket_arrange(bucket_indices: Tensor, num_buckets: int) -> (Tensor, Tensor):
    return torch.ops.sparse_engines_cuda.bucket_arrange(bucket_indices, num_buckets)

def sparse_matrix_vector_multiplication_reduction(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n: int
) -> Tensor:
    return torch.ops.sparse_engines_cuda.sparse_matrix_vector_multiplication_reduction.default(
        a, a_idx, b, b_idx, o_idx, n
    )

@torch.library.register_fake(
    "sparse_engines_cuda::sparse_matrix_vector_multiplication_reduction"
)
def _(a, a_idx, b, b_idx, o_idx, n):
    torch._check(a.device == a_idx.device)
    torch._check(a.device == b.device)
    torch._check(a.device == b_idx.device)
    torch._check(a.device == o_idx.device)

    torch._check(a.dtype == torch.float32)
    torch._check(a.ndim == 4)

    torch._check(a_idx.dtype == torch.int32)
    torch._check(a_idx.ndim == 1)

    torch._check(b.dtype == torch.float32)
    torch._check(b.ndim == 3)
    torch._check(a.size(1) == b.size(1))
    torch._check(b.size(2) == b.size(2))

    torch._check(b_idx.dtype == a_idx.dtype)
    torch._check(b_idx.numel() == a_idx.numel())
    torch._check(b_idx.ndim == 1)

    torch._check(o_idx.dtype == a_idx.dtype)
    torch._check(o_idx.numel() == a_idx.numel())
    torch._check(o_idx.ndim == 1)

    return torch.empty((n, a.size(1), a.size(-1)), dtype=a.dtype, device=a.device)

def _backward_sparse_matrix_vector_multiplication_reduction(ctx, grad):
    a, a_idx, b, b_idx, o_idx = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.sparse_engines_cuda.sparse_vector_vector_outer_product_reduction.default(
            b,
            b_idx,
            grad,
            o_idx,
            a_idx,
            a.shape[0] if isinstance(a, torch.Tensor) else ctx.a_shape_0,
        )
    if ctx.needs_input_grad[2]:
        grad_b = torch.ops.sparse_engines_cuda.sparse_matrix_vector_multiplication_reduction.default(
            a.transpose(2, 3),
            a_idx,
            grad,
            o_idx,
            b_idx,
            b.shape[0] if isinstance(b, torch.Tensor) else ctx.b_shape_0,
        )
    return grad_a, None, grad_b, None, None, None

def _setup_context_sparse_matrix_vector_multiplication_reduction(ctx, inputs, output):
    a, a_idx, b, b_idx, o_idx, n = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[2]:
        saved_a = a
    ctx.save_for_backward(saved_a, a_idx, saved_b, b_idx, o_idx)
    ctx.a_shape_0 = a.shape[0]
    ctx.b_shape_0 = b.shape[0]

torch.library.register_autograd(
    "sparse_engines_cuda::sparse_matrix_vector_multiplication_reduction",
    _backward_sparse_matrix_vector_multiplication_reduction,
    setup_context=_setup_context_sparse_matrix_vector_multiplication_reduction,
)

def sparse_vector_vector_outer_product_reduction(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor, n: int
) -> Tensor:
    return torch.ops.sparse_engines_cuda.sparse_vector_vector_outer_product_reduction.default(
        a, a_idx, b, b_idx, o_idx, n
    )

@torch.library.register_fake(
    "sparse_engines_cuda::sparse_vector_vector_outer_product_reduction"
)
def _(a, a_idx, b, b_idx, o_idx, n):
    torch._check(a.device == a_idx.device)
    torch._check(a.device == b.device)
    torch._check(a.device == b_idx.device)
    torch._check(a.device == o_idx.device)

    torch._check(a.dtype == torch.float32)
    torch._check(a.ndim == 3)

    torch._check(a_idx.dtype == torch.int32)
    torch._check(a_idx.ndim == 1)

    torch._check(b.dtype == torch.float32)
    torch._check(b.ndim == 3)
    torch._check(a.size(1) == b.size(1))

    torch._check(b_idx.dtype == a_idx.dtype)
    torch._check(b_idx.numel() == a_idx.numel())
    torch._check(b_idx.ndim == 1)

    torch._check(o_idx.dtype == a_idx.dtype)
    torch._check(o_idx.numel() == a_idx.numel())
    torch._check(o_idx.ndim == 1)

    return torch.empty(
        (n, a.size(1), a.size(2), b.size(2)), dtype=a.dtype, device=a.device
    )

def _backward_sparse_vector_vector_outer_product_reduction(ctx, grad):
    a, a_idx, b, b_idx, o_idx = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.sparse_engines_cuda.sparse_matrix_vector_multiplication_reduction.default(
            grad.transpose(2, 3),
            o_idx,
            b,
            b_idx,
            a_idx,
            a.shape[0] if isinstance(a, torch.Tensor) else ctx.a_shape_0,
        )
    if ctx.needs_input_grad[2]:
        grad_b = torch.ops.sparse_engines_cuda.sparse_matrix_vector_multiplication_reduction.default(
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

torch.library.register_autograd(
    "sparse_engines_cuda::sparse_vector_vector_outer_product_reduction",
    _backward_sparse_vector_vector_outer_product_reduction,
    setup_context=_setup_context_sparse_vector_vector_outer_product_reduction,
)
