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

struct EuclideanDistance {
	__device__ __forceinline__
	static float compute(float x, float y, float z) {
		return sqrt(x * x + y * y + z * z);
	}
	
	__device__ __forceinline__
	static float load_diff(const float* a, const float* b) {
		return *a - *b;
	}
};

struct ChebyshevDistance {
	__device__ __forceinline__
	static float compute(float x, float y, float z) {
		const auto abs_x = fabsf(x);
		const auto abs_y = fabsf(y);
		const auto abs_z = fabsf(z);
		return (abs_x > abs_y) 
			? ((abs_x > abs_z) ? abs_x : abs_z)
			: ((abs_y > abs_z) ? abs_y : abs_z);
	}
	
	__device__ __forceinline__
	static float load_diff(const float* a, const float* b) {
		return __ldg(a) - __ldg(b);
	}
};

template <typename DistanceMetric, typename T_a, typename T_b>
__global__ void indexed_distance_kernel(
	const float* __restrict__ a_ptr,
	const T_a* __restrict__ idx_a_ptr,
	const float* __restrict__ b_ptr,
	const T_b* __restrict__ idx_b_ptr,
	size_t num_pairs,
	float* __restrict__ distance_ptr
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_pairs) {
		const auto i_a = idx_a_ptr[idx] * 3;
		const auto i_b = idx_b_ptr[idx] * 3;
		const auto x = DistanceMetric::load_diff(&a_ptr[i_a + 0], &b_ptr[i_b + 0]);
		const auto y = DistanceMetric::load_diff(&a_ptr[i_a + 1], &b_ptr[i_b + 1]);
		const auto z = DistanceMetric::load_diff(&a_ptr[i_a + 2], &b_ptr[i_b + 2]);
		distance_ptr[idx] = DistanceMetric::compute(x, y, z);
	}
}

template <typename DistanceMetric, typename T_a, typename T_b>
void indexed_distance_cuda_impl_typed(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor distance) {
	auto num_pairs = a_idx.size(0);
	const int threads = 1024;
	const int blocks = (num_pairs + threads - 1) / threads;

	auto a_ptr = (float*)a.data_ptr<float>();
	auto b_ptr = (float*)b.data_ptr<float>();
	auto idx_a_ptr = (T_a*)a_idx.data_ptr<T_a>();
	auto idx_b_ptr = (T_b*)b_idx.data_ptr<T_b>();
	auto distance_ptr = (float*)distance.data_ptr<float>();

	indexed_distance_kernel<DistanceMetric, T_a, T_b><<<blocks, threads>>>(a_ptr, idx_a_ptr, b_ptr, idx_b_ptr, num_pairs, distance_ptr);
}

template <typename DistanceMetric>
void indexed_distance_impl_cuda(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor distance) {
	if (torch::kInt64 == a_idx.dtype() and torch::kInt64 == b_idx.dtype()) {
		indexed_distance_cuda_impl_typed<DistanceMetric, int64_t, int64_t>(a, a_idx, b, b_idx, distance);
	} else if (torch::kInt64 == a_idx.dtype() and torch::kInt32 == b_idx.dtype()) {
		indexed_distance_cuda_impl_typed<DistanceMetric, int64_t, int32_t>(a, a_idx, b, b_idx, distance);
	} else if (torch::kInt32 == a_idx.dtype() and torch::kInt64 == b_idx.dtype()) {
		indexed_distance_cuda_impl_typed<DistanceMetric, int32_t, int64_t>(a, a_idx, b, b_idx, distance);
	} else { // if (torch::kInt32 == a_idx.dtype() and torch::kInt32 == b_idx.dtype()) {
		indexed_distance_cuda_impl_typed<DistanceMetric, int32_t, int32_t>(a, a_idx, b, b_idx, distance);
	}
}

at::Tensor indexed_distance_cuda(at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, int64_t distance_type) {
	indexed_distance_check(a, a_idx, b, b_idx);

	a = a.contiguous();
	b = b.contiguous();

	auto options = at::TensorOptions().dtype(a.dtype()).device(a.device());
	auto distance = torch::zeros({a_idx.size(0)}, options);

	if (distance_type == 0) {
		indexed_distance_impl_cuda<EuclideanDistance>(a, a_idx, b, b_idx, distance);
	} else {
		indexed_distance_impl_cuda<ChebyshevDistance>(a, a_idx, b, b_idx, distance);
	}

	return distance;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) { m.impl("indexed_distance", &indexed_distance_cuda); }

} // namespace sparse_engines_cuda
