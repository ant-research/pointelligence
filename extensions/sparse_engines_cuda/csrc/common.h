#ifndef SPARSE_ENGINES_CUDA_CSRC_COMMON_H
#define SPARSE_ENGINES_CUDA_CSRC_COMMON_H

#include <torch/types.h>

namespace sparse_engines_cuda {

unsigned int next_highest_power_of_2(unsigned int v);

void sparse_matrix_vector_multiplication_reduction_check(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx);

void sparse_matrix_vector_multiplication_reduction_impl(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
);

void sparse_vector_vector_outer_product_reduction_check(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx);

void sparse_vector_vector_outer_product_reduction_impl(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
);

void bucket_arrange_check(at::Tensor bucket_indices);

void indexed_distance_check(at::Tensor a, at::Tensor idx_a, at::Tensor b, at::Tensor idx_b);

#ifndef WITHOUT_CUDA
void sparse_vector_vector_outer_product_reduction_impl_cuda(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
);

void sparse_matrix_vector_multiplication_reduction_impl_cuda(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
);
#endif

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_CSRC_COMMON_H