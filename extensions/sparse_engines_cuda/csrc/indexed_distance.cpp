#include <torch/library.h>

#include "common.h"

namespace sparse_engines_cuda {

struct EuclideanDistance {
	static float compute(float x, float y, float z) {
		return std::sqrt(x * x + y * y + z * z);
	}
	
	static float load_diff(const float a, const float b) {
		return a - b;
	}
};

struct ChebyshevDistance {
	static float compute(float x, float y, float z) {
		union { float f; int32_t i; } ux, uy, uz;
		ux.f = x;
		uy.f = y;
		uz.f = z;
		ux.i &= 0x7FFFFFFF;
		uy.i &= 0x7FFFFFFF;
		uz.i &= 0x7FFFFFFF;
		const auto abs_x = ux.f;
		const auto abs_y = uy.f;
		const auto abs_z = uz.f;
		const auto max_xy = (abs_x > abs_y) ? abs_x : abs_y;
		return (max_xy > abs_z) ? max_xy : abs_z;
	}
	
	static float load_diff(const float a, const float b) {
		return a - b;
	}
};

template <typename DistanceMetric, typename T_a, typename T_b>
void indexed_distance_impl_typed(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor distance) {
	const int64_t num_pairs = a_idx.size(0);
	const auto a_ptr = a.data_ptr<float>();
	const auto b_ptr = b.data_ptr<float>();
	const auto idx_a_ptr = a_idx.data_ptr<T_a>();
	const auto idx_b_ptr = b_idx.data_ptr<T_b>();
	auto distance_ptr = distance.data_ptr<float>();

	for (int64_t i = 0; i < num_pairs; ++i) {
		const auto i_a = idx_a_ptr[i] * 3;
		const auto i_b = idx_b_ptr[i] * 3;
		const auto x = DistanceMetric::load_diff(a_ptr[i_a + 0], b_ptr[i_b + 0]);
		const auto y = DistanceMetric::load_diff(a_ptr[i_a + 1], b_ptr[i_b + 1]);
		const auto z = DistanceMetric::load_diff(a_ptr[i_a + 2], b_ptr[i_b + 2]);
		distance_ptr[i] = DistanceMetric::compute(x, y, z);
	}
}

template <typename DistanceMetric>
void indexed_distance_impl(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor distance) {
	if (torch::kInt64 == a_idx.dtype() and torch::kInt64 == b_idx.dtype()) {
		indexed_distance_impl_typed<DistanceMetric, int64_t, int64_t>(a, a_idx, b, b_idx, distance);
	} else if (torch::kInt64 == a_idx.dtype() and torch::kInt32 == b_idx.dtype()) {
		indexed_distance_impl_typed<DistanceMetric, int64_t, int32_t>(a, a_idx, b, b_idx, distance);
	} else if (torch::kInt32 == a_idx.dtype() and torch::kInt64 == b_idx.dtype()) {
		indexed_distance_impl_typed<DistanceMetric, int32_t, int64_t>(a, a_idx, b, b_idx, distance);
	} else { // if (torch::kInt32 == a_idx.dtype() and torch::kInt32 == b_idx.dtype()) {
		indexed_distance_impl_typed<DistanceMetric, int32_t, int32_t>(a, a_idx, b, b_idx, distance);
	}
}

void indexed_distance_check(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx) {
	// clang-format off
	TORCH_CHECK(
		a.device() == a_idx.device() && a.device() == b.device() && a.device() == b_idx.device(),
		"a, a_idx, b and b_idx must be on the same device."
	);

	TORCH_CHECK(a_idx.dtype() == torch::kInt64 || a_idx.dtype() == torch::kInt32, "a_idx must have dtype torch::kInt64 or torch::kInt32.");
	TORCH_CHECK(a_idx.dim() == 1, "a_idx must be a 1-D tensor.")

	TORCH_CHECK(b_idx.dtype() == torch::kInt64 || b_idx.dtype() == torch::kInt32, "b_idx must have dtype torch::kInt64 or torch::kInt32.");
	TORCH_CHECK(b_idx.dim() == 1, "b_idx must be a 1-D tensor.")

	TORCH_CHECK(a_idx.numel() == b_idx.numel(), "a_idx must be of the same size as b_idx.")

	TORCH_CHECK(a.dim() == 2, "a must be a 2-D tensor.")
	TORCH_CHECK(b.dim() == 2, "b must be a 2-D tensor.")

	TORCH_CHECK(a.dtype() == torch::kFloat32, "a must have dtype torch::kFloat32.")
	TORCH_CHECK(b.dtype() == torch::kFloat32, "b must have dtype torch::kFloat32.")

	TORCH_CHECK(a.size(1) == 3, "a must have size 3 at dimension 1.")
	TORCH_CHECK(b.size(1) == 3, "b must have size 3 at dimension 1.")
	// clang-format on
}

at::Tensor indexed_distance(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, int64_t distance_type) {
	indexed_distance_check(a, a_idx, b, b_idx);

	a = a.contiguous();
	b = b.contiguous();

	auto options = at::TensorOptions().dtype(a.dtype()).device(a.device());
	auto distance = torch::zeros({a_idx.size(0)}, options);

	if (distance_type == 0) {
		indexed_distance_impl<EuclideanDistance>(a, a_idx, b, b_idx, distance);
	} else {
		indexed_distance_impl<ChebyshevDistance>(a, a_idx, b, b_idx, distance);
	}

	return distance;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CPU, m) { m.impl("indexed_distance", &indexed_distance); }

} // namespace sparse_engines_cuda