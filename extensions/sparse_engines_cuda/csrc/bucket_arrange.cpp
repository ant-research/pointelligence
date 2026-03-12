#include <torch/library.h>

#include "common.h"

namespace sparse_engines_cuda {

template <typename T> void bucket_arrange_impl(at::Tensor bucket_indices, at::Tensor bucket_sizes, at::Tensor bucket_slots) {
	const auto bucket_indices_ptr = bucket_indices.data_ptr<T>();
	auto bucket_sizes_ptr = bucket_sizes.data_ptr<T>();
	auto bucket_slots_ptr = bucket_slots.data_ptr<T>();
	for (int i = 0; i < bucket_indices.size(0); ++i) {
		bucket_slots_ptr[i] = __atomic_fetch_add(bucket_sizes_ptr + bucket_indices_ptr[i], 1, __ATOMIC_RELAXED);
	}
}

void bucket_arrange_check(at::Tensor bucket_indices) {
	// clang-format off
    TORCH_CHECK(bucket_indices.dtype() == torch::kInt64 || bucket_indices.dtype() == torch::kInt32,
                "bucket_indices must have dtype torch::kInt64 or torch::kInt32");
    TORCH_CHECK(bucket_indices.dim() == 1, "bucket_indices must be 1-D tensor.")
	// clang-format on
}

std::tuple<at::Tensor, at::Tensor> bucket_arrange(
	at::Tensor bucket_indices, /* (N,) */
	int64_t num_buckets
) {
	bucket_arrange_check(bucket_indices);
	bucket_indices = bucket_indices.contiguous();

	auto options = at::TensorOptions().dtype(bucket_indices.dtype()).device(bucket_indices.device());
	auto bucket_sizes = torch::zeros({num_buckets}, options);
	auto bucket_slots = torch::empty(bucket_indices.sizes(), options);

	if (torch::kInt64 == bucket_indices.dtype()) {
		bucket_arrange_impl<int64_t>(bucket_indices, bucket_sizes, bucket_slots);
	} else {
		bucket_arrange_impl<int32_t>(bucket_indices, bucket_sizes, bucket_slots);
	}

	return {bucket_sizes, bucket_slots};
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CPU, m) { m.impl("bucket_arrange", &bucket_arrange); }

} // namespace sparse_engines_cuda