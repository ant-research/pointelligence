#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "common.cuh"

typedef unsigned long long int uint64;

namespace sparse_engines_cuda {

template <typename T>
__global__ void bucket_arrange_kernel(
	const T* __restrict__ bucket_indices_ptr, size_t num_indices, T* __restrict__ bucket_sizes_ptr, T* __restrict__ bucket_slots_ptr
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_indices) {
		bucket_slots_ptr[idx] = atomicAdd(bucket_sizes_ptr + bucket_indices_ptr[idx], T(1));
	}
}

void bucket_arrange_impl_cuda(
	at::Tensor bucket_indices, /* (N,) */
	int num_buckets,
	at::Tensor bucket_sizes, /* (num_buckets,) */
	at::Tensor bucket_slots  /* (N,) */
) {
	auto num_indices = bucket_indices.size(0);
	const int threads = 1024;
	const int blocks = (num_indices + threads - 1) / threads;
	if (torch::kInt64 == bucket_indices.dtype()) {
		// atomicAdd does not take signed int64...
		auto bucket_indices_ptr = (uint64*)bucket_indices.data_ptr<int64_t>();
		auto bucket_sizes_ptr = (uint64*)bucket_sizes.data_ptr<int64_t>();
		auto bucket_slots_ptr = (uint64*)bucket_slots.data_ptr<int64_t>();
		bucket_arrange_kernel<<<blocks, threads>>>(bucket_indices_ptr, num_indices, bucket_sizes_ptr, bucket_slots_ptr);
	} else {
		auto bucket_indices_ptr = (int32_t*)bucket_indices.data_ptr<int32_t>();
		auto bucket_sizes_ptr = (int32_t*)bucket_sizes.data_ptr<int32_t>();
		auto bucket_slots_ptr = (int32_t*)bucket_slots.data_ptr<int32_t>();
		bucket_arrange_kernel<<<blocks, threads>>>(bucket_indices_ptr, num_indices, bucket_sizes_ptr, bucket_slots_ptr);
	}

	return;
}

std::tuple<at::Tensor, at::Tensor> bucket_arrange_cuda(
	at::Tensor bucket_indices, /* (N,) */
	int64_t num_buckets
) {
	bucket_arrange_check(bucket_indices);
	bucket_indices = bucket_indices.contiguous();

	auto options = at::TensorOptions().dtype(bucket_indices.dtype()).device(bucket_indices.device());
	auto bucket_sizes = torch::zeros({num_buckets}, options);
	auto bucket_slots = torch::empty(bucket_indices.sizes(), options);

	bucket_arrange_impl_cuda(bucket_indices, num_buckets, bucket_sizes, bucket_slots);
	return {bucket_sizes, bucket_slots};
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) { m.impl("bucket_arrange", &bucket_arrange_cuda); }

} // namespace sparse_engines_cuda