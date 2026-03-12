import torch
from torch import Tensor

__all__ = [
    "sparse_matrix_vector_multiplication_reduction",
    "sparse_vector_vector_outer_product_reduction",
    "sparse_scaled_dot_product_attention",
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


def sparse_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_idx: Tensor,
    k_idx: Tensor,
    k_cumsum: Tensor,
    scale: float,
) -> Tensor:
    o, m = torch.ops.sparse_engines_cuda.sparse_scaled_dot_product_attention.default(
        q, k, v, q_idx, k_idx, k_cumsum, scale
    )
    return o


@torch.library.register_fake("sparse_engines_cuda::sparse_scaled_dot_product_attention")
def _(q, k, v, q_idx, k_idx, k_cumsum, scale):
    torch._check(q.device == k.device)
    torch._check(q.device == v.device)
    torch._check(q.device == q_idx.device)
    torch._check(q.device == k_idx.device)
    torch._check(q.device == k_cumsum.device)

    torch._check(q.dtype == torch.float32)
    torch._check(k.dtype == q.dtype)
    torch._check(v.dtype == q.dtype)

    torch._check(q_idx.dtype == torch.int32)
    torch._check(k_idx.dtype == q_idx.dtype)
    torch._check(k_cumsum.dtype == q_idx.dtype)

    torch._check(q.ndim == 4)
    torch._check(k.ndim == 4)
    torch._check(v.ndim == 4)
    torch._check(q_idx.ndim == 1)
    torch._check(k_idx.ndim == 1)
    torch._check(k_cumsum.ndim == 1)

    torch._check(k.size(0) == q.size(0))
    torch._check(v.size(0) == q.size(0))
    torch._check(q_idx.size(0) == k_cumsum.size(0))
    torch._check(k.size(1) == q.size(1))
    torch._check(v.size(1) == q.size(1))
    torch._check(v.size(2) == k.size(2))
    torch._check(k.size(3) == q.size(3))

    o = torch.empty(
        (q.size(0), q.size(1), q.size(2), v.size(3)), dtype=v.dtype, device=q.device
    )
    m = torch.empty((q.size(0), q.size(1), q.size(2)), dtype=v.dtype, device=q.device)
    return o, m


def _backward_sparse_scaled_dot_product_attention(ctx, grad_o, grad_m):
    q, k, v, q_idx, k_idx, k_cumsum, o, m = ctx.saved_tensors
    dq, dk, dv = (
        torch.ops.sparse_engines_cuda.sparse_scaled_dot_product_attention_backward.default(
            q, k, v, q_idx, k_idx, k_cumsum, o, m, grad_o, ctx.scale
        )
    )
    return dq, dk, dv, None, None, None, None


def _setup_context_sparse_scaled_dot_product_attention(ctx, inputs, output):
    q, k, v, q_idx, k_idx, k_cumsum, scale = inputs
    o, m = output
    ctx.save_for_backward(q, k, v, q_idx, k_idx, k_cumsum, o, m)
    ctx.scale = scale


torch.library.register_autograd(
    "sparse_engines_cuda::sparse_scaled_dot_product_attention",
    _backward_sparse_scaled_dot_product_attention,
    setup_context=_setup_context_sparse_scaled_dot_product_attention,
)
