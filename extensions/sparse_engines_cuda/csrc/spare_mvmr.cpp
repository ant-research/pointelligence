#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>

#include <torch/library.h>

#include "common.h"

namespace sparse_engines_cuda {

void sparse_matrix_vector_multiplication_reduction_impl(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
) {
	const auto C = a.size(2);
	if (C == 1) {
		sparse_vector_vector_outer_product_reduction_impl(b, b_idx, a.squeeze(2), a_idx, o_idx, o);
		return;
	}

	const auto T = a_idx.size(0);
	const auto G = a.size(1);
	const auto M = a.size(3);

	const auto a_ptr = a.data_ptr<float>();
	const auto a_idx_ptr = a_idx.data_ptr<int32_t>();
	const auto b_ptr = b.data_ptr<float>();
	const auto b_idx_ptr = b_idx.data_ptr<int32_t>();
	const auto o_idx_ptr = o_idx.data_ptr<int32_t>();
	auto o_ptr = o.data_ptr<float>();

	int round = fegetround();
	fesetround(FE_TONEAREST);
	for (int t = 0; t < T; ++t) {
		const auto a_k = a_idx_ptr[t];
		const auto b_k = b_idx_ptr[t];
		const auto o_k = o_idx_ptr[t];

		for (int g = 0; g < G; ++g) {
			const auto a_offset = (a_k * G + g) * C * M;
			const auto b_offset = (b_k * G + g) * C;
			const auto o_offset = (o_k * G + g) * M;
			for (int m = 0; m < M; ++m) {
				float o = 0.0f;
				int Tc;
				if (C <= 8) {
					Tc = 8;
				} else if (C <= 16) {
					Tc = 16;
				} else {
					Tc = 32;
				}
				int L = C / Tc;
				float o_b = 0.0f;
				for (int l = 0; l < L; ++l) {
					o_b = 0.0f;
					for (int tc = 0; tc < Tc; ++tc) {
						int c = l * Tc + tc;
						o_b = fmaf(a_ptr[a_offset + c * M + m], b_ptr[b_offset + c], o_b);
					}
					o += o_b;
				}
				o_b = 0.0f;
				if (Tc * L < C) {
					for (int i = Tc * L; i < C; ++i) {
						o_b = fmaf(a_ptr[a_offset + i * M + m], b_ptr[b_offset + i], o_b);
					}
				}
				o += o_b;
				o_ptr[o_offset + m] += o;
			}
		}
	}
	fesetround(round);
}

void sparse_matrix_vector_multiplication_reduction_check(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx){
	// clang-format off
    TORCH_CHECK(a.device() == a_idx.device(), "a and a_idx must be on the same device.")
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device.")
    TORCH_CHECK(a.device() == b_idx.device(), "a and b_idx must be on the same device.")
    TORCH_CHECK(a.device() == o_idx.device(), "a and o_idx must be on the same device.")

    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must have dtype torch::kFloat32.")
    TORCH_CHECK(a.dim() == 4, "a must be a 4-D tensor.")

    TORCH_CHECK(a_idx.dtype() == torch::kInt32, "a_idx must have dtype torch::kInt32.")
    TORCH_CHECK(a_idx.dim() == 1, "a_idx must be a 1-D tensor.")

    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must have dtype torch::kFloat32.")
    TORCH_CHECK(b.dim() == 3, "b must be a 3-D tensor.")
    TORCH_CHECK(a.size(1) == b.size(1), "a.size(1) must be equal to b.size(1).")
    TORCH_CHECK(a.size(2) == b.size(2), "a.size(2) must be equal to b.size(2).")

    TORCH_CHECK(b_idx.dtype() == a_idx.dtype(), "b_idx must have same dtype with a_idx.")
    TORCH_CHECK(b_idx.numel() == a_idx.numel(), "b_idx must have same numel with a_idx.")
    TORCH_CHECK(b_idx.dim() == 1, "b_idx must be a 1-D tensor.")

    TORCH_CHECK(o_idx.dtype() == a_idx.dtype(), "o_idx must have same dtype with a_idx.")
    TORCH_CHECK(o_idx.numel() == a_idx.numel(), "o_idx must have same numel with a_idx.")
    TORCH_CHECK(o_idx.dim() == 1, "o_idx must be a 1-D tensor.")
	// clang-format on
}

at::Tensor
	sparse_matrix_vector_multiplication_reduction(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, int64_t n) {
	sparse_matrix_vector_multiplication_reduction_check(a, a_idx, b, b_idx, o_idx);

	a = a.contiguous();
	b = b.contiguous();

	const auto options = at::TensorOptions().dtype(torch::kFloat32).device(a.device());
	auto o = torch::zeros({n, a.size(1), a.size(-1)}, options);

	sparse_matrix_vector_multiplication_reduction_impl(a, a_idx, b, b_idx, o_idx, o);

	return o;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CPU, m) {
	m.impl("sparse_matrix_vector_multiplication_reduction", &sparse_matrix_vector_multiplication_reduction);
}

} // namespace sparse_engines_cuda
