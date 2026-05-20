#include "sparse_vvor_grouped_mma.cuh"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/types.h>

namespace sparse_engines_cuda {

template <typename T, int Tm>
void sparse_vvor_grouped_impl_cuda_ct(
    at::Tensor grad_out, at::Tensor a_idx,
    at::Tensor b, at::Tensor b_idx,
    at::Tensor o_idx, at::Tensor seg_offs,
    at::Tensor o
) {
    const auto K_offsets = o.size(0);
    const auto G = o.size(1);
    const auto M = o.size(2);
    const auto C = o.size(3);

    const int32_t Mt = (M + Tm - 1) / Tm;
    const int32_t Cw = (C + 32 - 1) / 32;
    const int32_t total_tiles = K_offsets * G * Mt * Cw;

    const int32_t warps_per_block = 32;
    dim3 block(32, warps_per_block);
    const int32_t grid_size = (total_tiles + warps_per_block - 1) / warps_per_block;

    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_vvor_grouped_kernel<T, Tm><<<grid_size, block, 0, stream>>>(
        grad_out.data_ptr<T>(),
        a_idx.data_ptr<int32_t>(),
        b.data_ptr<T>(),
        b_idx.data_ptr<int32_t>(),
        o_idx.data_ptr<int32_t>(),
        seg_offs.data_ptr<int64_t>(),
        o.data_ptr<float>(),
        K_offsets, G, M, C, Mt, Cw, warps_per_block
    );

    cudaError_t err{cudaGetLastError()};
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err))
}

at::Tensor sparse_vvor_grouped_mma_cuda(
    at::Tensor grad_out, at::Tensor a_idx,
    at::Tensor b, at::Tensor b_idx,
    at::Tensor o_idx, at::Tensor seg_offs, int64_t n_k
) {
    grad_out = grad_out.contiguous();
    b = b.contiguous();

    const auto G = grad_out.size(1);
    const auto M = grad_out.size(2);
    const auto C = b.size(2);
    const auto options = at::TensorOptions().dtype(torch::kFloat32).device(grad_out.device());
    auto o = torch::zeros({n_k, G, M, C}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad_out.scalar_type(), "sparse_vvor_grouped_mma", [&] {
            const int Tm = (M < 32) ? M : 32;
            if (Tm == 32) {
                sparse_vvor_grouped_impl_cuda_ct<scalar_t, 32>(grad_out, a_idx, b, b_idx, o_idx, seg_offs, o);
            } else if (Tm == 16) {
                sparse_vvor_grouped_impl_cuda_ct<scalar_t, 16>(grad_out, a_idx, b, b_idx, o_idx, seg_offs, o);
            } else if (Tm == 8) {
                sparse_vvor_grouped_impl_cuda_ct<scalar_t, 8>(grad_out, a_idx, b, b_idx, o_idx, seg_offs, o);
            } else if (Tm == 4) {
                sparse_vvor_grouped_impl_cuda_ct<scalar_t, 4>(grad_out, a_idx, b, b_idx, o_idx, seg_offs, o);
            } else {
                sparse_vvor_grouped_impl_cuda_ct<scalar_t, 1>(grad_out, a_idx, b, b_idx, o_idx, seg_offs, o);
            }
        }
    );

    auto input_dtype = grad_out.scalar_type();
    return input_dtype == at::kFloat ? o : o.to(input_dtype);
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
    m.impl("sparse_vvor_grouped_mma", &sparse_vvor_grouped_mma_cuda);
}

} // namespace sparse_engines_cuda