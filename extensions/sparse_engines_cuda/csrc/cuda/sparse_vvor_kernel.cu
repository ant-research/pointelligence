#include "sparse_vvor_kernel.cuh"

namespace sparse_engines_cuda {

// Explicit instantiations are in sparse_vvor_kernel_small.cu and sparse_vvor_kernel_large.cu

void sparse_vector_vector_outer_product_reduction_impl_cuda(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
) {
	if (a.size(2) > 1 && b.size(2) == 1) {
		sparse_vector_vector_outer_product_reduction_impl_cuda(b, b_idx, a, a_idx, o_idx, o);
		return;
	}

	const auto M = a.size(2);
	if (M <= 1) {
		sparse_vector_vector_outer_product_reduction_impl_cuda_ct<1>(a, a_idx, b, b_idx, o_idx, o);
	} else if (M <= 2) {
		sparse_vector_vector_outer_product_reduction_impl_cuda_ct<2>(a, a_idx, b, b_idx, o_idx, o);
	} else if (M <= 3) {
		sparse_vector_vector_outer_product_reduction_impl_cuda_ct<3>(a, a_idx, b, b_idx, o_idx, o);
	} else if (M <= 4) {
		sparse_vector_vector_outer_product_reduction_impl_cuda_ct<4>(a, a_idx, b, b_idx, o_idx, o);
	} else if (M <= 8) {
		sparse_vector_vector_outer_product_reduction_impl_cuda_ct<8>(a, a_idx, b, b_idx, o_idx, o);
	} else if (M <= 16) {
		sparse_vector_vector_outer_product_reduction_impl_cuda_ct<16>(a, a_idx, b, b_idx, o_idx, o);
	} else {
		sparse_vector_vector_outer_product_reduction_impl_cuda_ct<32>(a, a_idx, b, b_idx, o_idx, o);
	}
	cudaError_t err{cudaGetLastError()};
	TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err))
}

at::Tensor sparse_vector_vector_outer_product_reduction_cuda(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, int64_t n
) {
	sparse_vector_vector_outer_product_reduction_check(a, a_idx, b, b_idx, o_idx);

	a = a.contiguous();
	b = b.contiguous();

	const auto options = at::TensorOptions().dtype(torch::kFloat32).device(a.device());
	auto o = torch::zeros({n, a.size(1), a.size(2), b.size(2)}, options);

	sparse_vector_vector_outer_product_reduction_impl_cuda(a, a_idx, b, b_idx, o_idx, o);

	return o;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
	m.impl("sparse_vector_vector_outer_product_reduction", &sparse_vector_vector_outer_product_reduction_cuda);
}

} // namespace sparse_engines_cuda
