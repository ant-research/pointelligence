#include "sparse_mvmr_kernel.cuh"

namespace sparse_engines_cuda {

// Explicit instantiations are in sparse_mvmr_kernel_small.cu and sparse_mvmr_kernel_large.cu

void sparse_matrix_vector_multiplication_reduction_impl_cuda(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
) {
	const auto C = a.size(2);
	if (C == 1) {
		sparse_vector_vector_outer_product_reduction_impl_cuda(b, b_idx, a.squeeze(2), a_idx, o_idx, o);
		return;
	}

	if (C <= 1) {
		sparse_matrix_vector_multiplication_reduction_impl_cuda_ct<1>(a, a_idx, b, b_idx, o_idx, o);
	} else if (C <= 2) {
		sparse_matrix_vector_multiplication_reduction_impl_cuda_ct<2>(a, a_idx, b, b_idx, o_idx, o);
	} else if (C <= 3) {
		sparse_matrix_vector_multiplication_reduction_impl_cuda_ct<3>(a, a_idx, b, b_idx, o_idx, o);
	} else if (C <= 4) {
		sparse_matrix_vector_multiplication_reduction_impl_cuda_ct<4>(a, a_idx, b, b_idx, o_idx, o);
	} else if (C <= 8) {
		sparse_matrix_vector_multiplication_reduction_impl_cuda_ct<8>(a, a_idx, b, b_idx, o_idx, o);
	} else if (C <= 16) {
		sparse_matrix_vector_multiplication_reduction_impl_cuda_ct<16>(a, a_idx, b, b_idx, o_idx, o);
	} else {
		sparse_matrix_vector_multiplication_reduction_impl_cuda_ct<32>(a, a_idx, b, b_idx, o_idx, o);
	}
	cudaError_t err{cudaGetLastError()};
	TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err))
}

at::Tensor sparse_matrix_vector_multiplication_reduction_cuda(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, int64_t n
) {
	sparse_matrix_vector_multiplication_reduction_check(a, a_idx, b, b_idx, o_idx);

	a = a.contiguous();
	b = b.contiguous();

	const auto options = at::TensorOptions().dtype(torch::kFloat32).device(a.device());
	auto o = torch::zeros({n, a.size(1), a.size(-1)}, options);

	sparse_matrix_vector_multiplication_reduction_impl_cuda(a, a_idx, b, b_idx, o_idx, o);

	return o;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
	m.impl("sparse_matrix_vector_multiplication_reduction", &sparse_matrix_vector_multiplication_reduction_cuda);
}
} // namespace sparse_engines_cuda
