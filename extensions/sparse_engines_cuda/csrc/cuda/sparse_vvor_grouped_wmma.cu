#include "sparse_vvor_grouped_wmma.cuh"

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace sparse_engines_cuda {

template <typename T_native, typename T_c10, typename IdxT>
static void sparse_vvor_grouped_wmma_impl_cuda(
    at::Tensor grad_out, at::Tensor a_idx,
    at::Tensor b, at::Tensor b_idx,
    at::Tensor o_idx, at::Tensor seg_offs,
    at::Tensor o
) {
    const auto K_offsets = o.size(0);
    const auto G = o.size(1);
    const auto M = o.size(2);
    const auto C = o.size(3);

    TORCH_CHECK(M % WMMA_M == 0,
        "WMMA-direct vvor requires M divisible by 16; got M=", M);
    TORCH_CHECK(C % WMMA_N == 0,
        "WMMA-direct vvor requires C divisible by 16; got C=", C);

    const int32_t Mt = M / WMMA_M;
    const int32_t Ct = C / WMMA_N;
    const int32_t total_tiles = K_offsets * G * Mt * Ct;

    dim3 block(32, WMMA_WARPS_PER_BLOCK);
    const int32_t grid_size = (total_tiles + WMMA_WARPS_PER_BLOCK - 1) / WMMA_WARPS_PER_BLOCK;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Reinterpret c10::Half / c10::BFloat16 pointers as the native CUDA
    // types that nvcuda::wmma::fragment is specialized on. The two are
    // layout-compatible (both are 16-bit storage with the same bit
    // pattern) but not type-equivalent at the C++ level.
    sparse_vvor_grouped_wmma_kernel<T_native, IdxT><<<grid_size, block, 0, stream>>>(
        reinterpret_cast<const T_native*>(grad_out.data_ptr<T_c10>()),
        a_idx.data_ptr<IdxT>(),
        reinterpret_cast<const T_native*>(b.data_ptr<T_c10>()),
        b_idx.data_ptr<IdxT>(),
        o_idx.data_ptr<IdxT>(),
        seg_offs.data_ptr<int64_t>(),
        o.data_ptr<float>(),
        K_offsets, G, M, C, Mt, Ct
    );

    cudaError_t err{cudaGetLastError()};
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

// G11.5: dispatch on both feature dtype AND index dtype so the wrapper
// doesn't need to cast indices. build_triplets produces int32 indices for
// most production point counts (<= INT32_MAX); parity tests use int64;
// both are accepted natively.
#define SPARSE_VVOR_WMMA_DISPATCH_IDX(T_native, T_c10) do {                \
    const auto idx_dtype = a_idx.scalar_type();                            \
    TORCH_CHECK(idx_dtype == at::kInt || idx_dtype == at::kLong,           \
        "sparse_vvor_grouped_wmma: a_idx dtype must be int32 or int64, got ", idx_dtype); \
    TORCH_CHECK(b_idx.scalar_type() == idx_dtype && o_idx.scalar_type() == idx_dtype, \
        "sparse_vvor_grouped_wmma: a_idx, b_idx, o_idx must share dtype"); \
    if (idx_dtype == at::kInt) {                                            \
        sparse_vvor_grouped_wmma_impl_cuda<T_native, T_c10, int32_t>(       \
            grad_out, a_idx, b, b_idx, o_idx, seg_offs, o);                 \
    } else {                                                                \
        sparse_vvor_grouped_wmma_impl_cuda<T_native, T_c10, int64_t>(       \
            grad_out, a_idx, b, b_idx, o_idx, seg_offs, o);                 \
    }                                                                       \
} while (0)

at::Tensor sparse_vvor_grouped_wmma_cuda(
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

    const auto dtype = grad_out.scalar_type();
    if (dtype == at::kHalf) {
        SPARSE_VVOR_WMMA_DISPATCH_IDX(__half, c10::Half);
    } else if (dtype == at::kBFloat16) {
        SPARSE_VVOR_WMMA_DISPATCH_IDX(__nv_bfloat16, c10::BFloat16);
    } else {
        // fp32 / TF32 path not supported by m16n16k16 WMMA atom; caller is
        // expected to dispatch fp32 inputs to sparse_vvor_grouped_mma (the
        // scalar-FMA grouped path) instead. Documented in
        // sparse_engines/vvor_grouped_wmma.py wrapper.
        TORCH_CHECK(false,
            "sparse_vvor_grouped_wmma: only fp16 / bf16 inputs supported; ",
            "dispatch fp32 to sparse_vvor_grouped_mma. got dtype=", dtype);
    }

    return o.to(dtype);
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
    m.impl("sparse_vvor_grouped_wmma", &sparse_vvor_grouped_wmma_cuda);
}

} // namespace sparse_engines_cuda
