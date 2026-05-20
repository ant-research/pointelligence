#include "sparse_mvmr_grouped_mma.cuh"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/types.h>

namespace sparse_engines_cuda {

template <typename T, int Tc>
void sparse_mvmr_grouped_impl_cuda_ct(
    at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx,
    at::Tensor o_idx, at::Tensor seg_offs, at::Tensor o
) {
    const auto K_offsets = a.size(0);
    const auto G = a.size(1);
    const auto C = a.size(2);
    const auto M = a.size(3);

    const int32_t Ct = (C + Tc - 1) / Tc;
    const int32_t Mw = (M + 32 - 1) / 32;
    const int32_t total_tiles = K_offsets * G * Ct * Mw;

    // Use 32 warps per block (1024 threads) for good occupancy
    const int32_t warps_per_block = 32;
    dim3 block(32, warps_per_block);
    const int32_t grid_size = (total_tiles + warps_per_block - 1) / warps_per_block;

    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_mvmr_grouped_kernel<T, Tc><<<grid_size, block, 0, stream>>>(
        a.data_ptr<T>(),
        a_idx.data_ptr<int32_t>(),
        b.data_ptr<T>(),
        b_idx.data_ptr<int32_t>(),
        o_idx.data_ptr<int32_t>(),
        seg_offs.data_ptr<int64_t>(),
        o.data_ptr<float>(),
        K_offsets, G, C, M, Ct, Mw, warps_per_block
    );

    cudaError_t err{cudaGetLastError()};
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err))
}

at::Tensor sparse_mvmr_grouped_mma_cuda(
    at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx,
    at::Tensor o_idx, at::Tensor seg_offs, int64_t n_o
) {
    a = a.contiguous();
    b = b.contiguous();

    const auto C = a.size(2);
    const auto M = a.size(3);
    const auto G = a.size(1);
    const auto options = at::TensorOptions().dtype(torch::kFloat32).device(a.device());
    auto o = torch::zeros({n_o, G, M}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, a.scalar_type(), "sparse_mvmr_grouped_mma", [&] {
            const int Tc = (C < 32) ? C : 32;
            if (Tc == 32) {
                sparse_mvmr_grouped_impl_cuda_ct<scalar_t, 32>(a, a_idx, b, b_idx, o_idx, seg_offs, o);
            } else if (Tc == 16) {
                sparse_mvmr_grouped_impl_cuda_ct<scalar_t, 16>(a, a_idx, b, b_idx, o_idx, seg_offs, o);
            } else if (Tc == 8) {
                sparse_mvmr_grouped_impl_cuda_ct<scalar_t, 8>(a, a_idx, b, b_idx, o_idx, seg_offs, o);
            } else if (Tc == 4) {
                sparse_mvmr_grouped_impl_cuda_ct<scalar_t, 4>(a, a_idx, b, b_idx, o_idx, seg_offs, o);
            } else {
                sparse_mvmr_grouped_impl_cuda_ct<scalar_t, 1>(a, a_idx, b, b_idx, o_idx, seg_offs, o);
            }
        }
    );

    auto input_dtype = a.scalar_type();
    return input_dtype == at::kFloat ? o : o.to(input_dtype);
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
    m.impl("sparse_mvmr_grouped_mma", &sparse_mvmr_grouped_mma_cuda);
}

} // namespace sparse_engines_cuda