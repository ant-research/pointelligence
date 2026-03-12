#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "common.cuh"

namespace sparse_engines_cuda {

template <typename T_a, typename T_b, int warp_size, int Tm>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)

	sparse_vector_vector_outer_product_reduction_kernel(
		const T_a* __restrict__ a_ptr,
		const int32_t* __restrict__ a_idx_ptr,
		const T_b* __restrict__ b_ptr,
		const int32_t* __restrict__ b_idx_ptr,
		const int32_t* __restrict__ o_idx_ptr,
		const int T,
		const int G,
		const int M,
		const int C,
		const int N,
		const int L,
		float* __restrict__ o_ptr
	) {
	float o[Tm] = {0.0f};

	const auto Mt = (M + Tm - 1) / Tm;
	const auto Cw = (C + warp_size - 1) / warp_size;

	int n_g_mt_cw = blockIdx.x * blockDim.y + threadIdx.y;
	if (n_g_mt_cw >= N * G * Mt * Cw) {
		return;
	}

	const auto cw = n_g_mt_cw % Cw;
	n_g_mt_cw /= Cw;
	const auto mt = n_g_mt_cw % Mt;
	n_g_mt_cw /= Mt;
	const auto g = n_g_mt_cw % G;
	n_g_mt_cw /= G;
	const auto& n = n_g_mt_cw;

	const int& thread_id = threadIdx.x;

	const auto m = mt * Tm;
	const auto c = cw * warp_size + thread_id;
	const int l = min(L, T - (n * L));

	T_a a = 0.0f;
	T_b b = 0.0f;
	constexpr auto FULL_MASK = 0xffffffff;

	auto a_k_prev = std::numeric_limits<int32_t>::max();
	auto b_k_prev = std::numeric_limits<int32_t>::max();
	auto o_k_prev = std::numeric_limits<int32_t>::max();
	if (m + Tm <= M) {
		const auto MASK = __ballot_sync(FULL_MASK, c < C && thread_id < Tm);
		for (int i = 0; i < l; ++i) {
			const auto t = n * L + i;
			const auto a_k_i = (a_idx_ptr[t] * G + g) * M + m;
			const auto b_k_i = (b_idx_ptr[t] * G + g) * C;
			const auto o_k_i = ((o_idx_ptr[t] * G + g) * M + m) * C;
			;

			if (a_k_i != a_k_prev) {
				if (thread_id < Tm) {
					a = a_ptr[a_k_i + thread_id];
				}
				a_k_prev = a_k_i;
			}

			if (b_k_i != b_k_prev) {
				if (c < C) {
					b = b_ptr[b_k_i + c];
				}
				b_k_prev = b_k_i;
			}

			if (o_k_i != o_k_prev) {
				if (c < C && i != 0) {
#pragma unroll
					for (int tm = 0; tm < Tm; ++tm) {
						atomicAdd(o_ptr + tm * C + o_k_prev + c, o[tm]);
					}
				}

				o_k_prev = o_k_i;

				__syncwarp();
#pragma unroll
				for (int tm = 0; tm < Tm; ++tm) {
					o[tm] = __fmul_rn(__shfl_sync(MASK, a, tm), b);
				}
			} else {
				__syncwarp();
#pragma unroll
				for (int tm = 0; tm < Tm; ++tm) {
					o[tm] = __fmaf_rn(__shfl_sync(MASK, a, tm), b, o[tm]);
				}
			}
		}
		if (c < C) {
#pragma unroll
			for (int tm = 0; tm < Tm; ++tm) {
				atomicAdd(o_ptr + tm * C + o_k_prev + c, o[tm]);
			}
		}
	} else {
		const auto Tm_max = M - m;
		const auto MASK = __ballot_sync(FULL_MASK, c < C && thread_id < Tm_max);
		for (int i = 0; i < l; ++i) {
			const auto t = n * L + i;
			const auto a_k_i = (a_idx_ptr[t] * G + g) * M + m;
			const auto b_k_i = (b_idx_ptr[t] * G + g) * C;
			const auto o_k_i = ((o_idx_ptr[t] * G + g) * M + m) * C;
			;

			if (a_k_i != a_k_prev) {
				if (thread_id < Tm_max) {
					a = a_ptr[a_k_i + thread_id];
				}
				a_k_prev = a_k_i;
			}

			if (b_k_i != b_k_prev) {
				if (c < C) {
					b = b_ptr[b_k_i + c];
				}
				b_k_prev = b_k_i;
			}

			if (o_k_i != o_k_prev) {
				if (c < C && i != 0) {
					for (int tm = 0; tm < Tm_max; ++tm) {
						atomicAdd(o_ptr + tm * C + o_k_prev + c, o[tm]);
					}
				}

				o_k_prev = o_k_i;

				__syncwarp();
				for (int tm = 0; tm < Tm_max; ++tm) {
					o[tm] = __fmul_rn(__shfl_sync(MASK, a, tm), b);
				}
			} else {
				__syncwarp();
				for (int tm = 0; tm < Tm_max; ++tm) {
					o[tm] = __fmaf_rn(__shfl_sync(MASK, a, tm), b, o[tm]);
				}
			}
		}
		if (c < C) {
			for (int tm = 0; tm < Tm_max; ++tm) {
				atomicAdd(o_ptr + tm * C + o_k_prev + c, o[tm]);
			}
		}
	}
}

template <int Tm>
void sparse_vector_vector_outer_product_reduction_impl_cuda_ct(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
) {
	const auto T = a_idx.size(0);
	const auto G = a.size(1);
	const auto M = a.size(2);
	const auto C = b.size(2);

	const auto a_ptr = a.data_ptr<float>();
	const auto a_idx_ptr = a_idx.data_ptr<int32_t>();
	const auto b_ptr = b.data_ptr<float>();
	const auto b_idx_ptr = b_idx.data_ptr<int32_t>();
	const auto o_idx_ptr = o_idx.data_ptr<int32_t>();
	auto o_ptr = o.data_ptr<float>();

	const auto warp_size = 32;
	const auto warp_num = 32;
	dim3 block(warp_size, warp_num);

	const int K_upper = next_highest_power_of_2(T / a.size(0)) * 2;
	const auto L = min(max(K_upper, warp_size), warp_size * 4);
	const auto N = (T + L - 1) / L;
	const auto Cw = (C + warp_size - 1) / warp_size;

	const auto Mt = (M + Tm - 1) / Tm;
	const auto N_G_Mt_Cw = N * G * Mt * Cw;

	dim3 grid((N_G_Mt_Cw + warp_num - 1) / warp_num);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	sparse_vector_vector_outer_product_reduction_kernel<float, float, warp_size, Tm>
		<<<grid, block, 0, stream>>>(a_ptr, a_idx_ptr, b_ptr, b_idx_ptr, o_idx_ptr, T, G, M, C, N, L, o_ptr);

	cudaError_t err{cudaGetLastError()};
	TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err))
}

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
