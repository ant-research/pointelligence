#include "sparse_vvor_kernel.cuh"

namespace sparse_engines_cuda {

template void sparse_vector_vector_outer_product_reduction_impl_cuda_ct<8>(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o);
template void sparse_vector_vector_outer_product_reduction_impl_cuda_ct<16>(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o);
template void sparse_vector_vector_outer_product_reduction_impl_cuda_ct<32>(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o);

} // namespace sparse_engines_cuda
