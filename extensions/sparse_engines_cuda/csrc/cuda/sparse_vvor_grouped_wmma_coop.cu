#include "sparse_vvor_grouped_wmma_coop.cuh"

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace sparse_engines_cuda {

template <typename T_native, typename T_c10, typename IdxT>
static void sparse_vvor_grouped_wmma_coop_impl_cuda(
    at::Tensor grad_out, at::Tensor a_idx,
    at::Tensor b, at::Tensor b_idx,
    at::Tensor o_idx, at::Tensor seg_offs,
    at::Tensor o, int32_t W
) {
    const auto K_offsets = o.size(0);
    const auto G = o.size(1);
    const auto M = o.size(2);
    const auto C = o.size(3);

    TORCH_CHECK(M % COOP_WMMA_M == 0,
        "sparse_vvor_grouped_wmma_coop: M must be divisible by 16; got M=", M);
    TORCH_CHECK(C % COOP_WMMA_N == 0,
        "sparse_vvor_grouped_wmma_coop: C must be divisible by 16; got C=", C);
    TORCH_CHECK(W >= 1 && W <= 64,
        "sparse_vvor_grouped_wmma_coop: W must be in [1, 64]; got W=", W);

    const int32_t Mt = M / COOP_WMMA_M;
    const int32_t Ct = C / COOP_WMMA_N;
    const int32_t total_tiles = K_offsets * G * Mt * Ct;
    const int32_t grid_size   = total_tiles * W;

    dim3 block(32, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_vvor_grouped_wmma_coop_kernel<T_native, IdxT><<<grid_size, block, 0, stream>>>(
        reinterpret_cast<const T_native*>(grad_out.data_ptr<T_c10>()),
        a_idx.data_ptr<IdxT>(),
        reinterpret_cast<const T_native*>(b.data_ptr<T_c10>()),
        b_idx.data_ptr<IdxT>(),
        o_idx.data_ptr<IdxT>(),
        seg_offs.data_ptr<int64_t>(),
        o.data_ptr<float>(),
        static_cast<int32_t>(K_offsets),
        static_cast<int32_t>(G),
        static_cast<int32_t>(M),
        static_cast<int32_t>(C),
        Mt, Ct, W
    );

    cudaError_t err{cudaGetLastError()};
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

#define SPARSE_VVOR_WMMA_COOP_DISPATCH_IDX(T_native, T_c10) do {             \
    const auto idx_dtype = a_idx.scalar_type();                              \
    TORCH_CHECK(idx_dtype == at::kInt || idx_dtype == at::kLong,             \
        "sparse_vvor_grouped_wmma_coop: a_idx dtype must be int32 or int64, got ", idx_dtype); \
    TORCH_CHECK(b_idx.scalar_type() == idx_dtype && o_idx.scalar_type() == idx_dtype, \
        "sparse_vvor_grouped_wmma_coop: a_idx, b_idx, o_idx must share dtype"); \
    if (idx_dtype == at::kInt) {                                              \
        sparse_vvor_grouped_wmma_coop_impl_cuda<T_native, T_c10, int32_t>(   \
            grad_out, a_idx, b, b_idx, o_idx, seg_offs, o, W);               \
    } else {                                                                  \
        sparse_vvor_grouped_wmma_coop_impl_cuda<T_native, T_c10, int64_t>(   \
            grad_out, a_idx, b, b_idx, o_idx, seg_offs, o, W);               \
    }                                                                         \
} while (0)

at::Tensor sparse_vvor_grouped_wmma_coop_cuda(
    at::Tensor grad_out, at::Tensor a_idx,
    at::Tensor b, at::Tensor b_idx,
    at::Tensor o_idx, at::Tensor seg_offs,
    int64_t n_k, int64_t w
) {
    grad_out = grad_out.contiguous();
    b = b.contiguous();

    const auto G = grad_out.size(1);
    const auto M = grad_out.size(2);
    const auto C = b.size(2);
    const auto options = at::TensorOptions().dtype(torch::kFloat32).device(grad_out.device());
    auto o = torch::zeros({n_k, G, M, C}, options);

    const int32_t W = static_cast<int32_t>(w);
    const auto dtype = grad_out.scalar_type();
    if (dtype == at::kHalf) {
        SPARSE_VVOR_WMMA_COOP_DISPATCH_IDX(__half, c10::Half);
    } else if (dtype == at::kBFloat16) {
        SPARSE_VVOR_WMMA_COOP_DISPATCH_IDX(__nv_bfloat16, c10::BFloat16);
    } else {
        TORCH_CHECK(false,
            "sparse_vvor_grouped_wmma_coop: only fp16 / bf16 inputs supported; "
            "dispatch fp32 to sparse_vvor_grouped_mma. got dtype=", dtype);
    }

    return o.to(dtype);
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
    m.impl("sparse_vvor_grouped_wmma_coop", &sparse_vvor_grouped_wmma_coop_cuda);
}

} // namespace sparse_engines_cuda
