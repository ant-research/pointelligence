#ifndef SPARSE_ENGINES_CUDA_SPARSE_MVMR_KERNEL_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_MVMR_KERNEL_CUH

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "common.cuh"

namespace sparse_engines_cuda {

template <typename T_a, typename T_b, int warp_size, int Tc>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)

	sparse_matrix_vector_multiplication_reduction_kernel(
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
	float a[Tc] = {0.0f};

	const auto Ct = (C + Tc - 1) / Tc;
	const auto Mw = (M + warp_size - 1) / warp_size;

	int n_g_ct_mw = blockIdx.x * blockDim.y + threadIdx.y;
	if (n_g_ct_mw >= N * G * Ct * Mw) {
		return;
	}

	const auto mw = n_g_ct_mw % Mw;
	n_g_ct_mw /= Mw;
	const auto ct = n_g_ct_mw % Ct;
	n_g_ct_mw /= Ct;
	const auto g = n_g_ct_mw % G;
	n_g_ct_mw /= G;
	const auto& n = n_g_ct_mw;

	const int& thread_id = threadIdx.x;

	const auto m = mw * warp_size + thread_id;
	const auto c = ct * Tc;
	const int l = min(L, T - (n * L));

	T_b b = 0.0f;
	float o;
	constexpr auto FULL_MASK = 0xffffffff;
	const auto m_lt_M_MASK = __ballot_sync(FULL_MASK, m < M);

	auto a_k_prev = std::numeric_limits<int32_t>::max();
	auto b_k_prev = std::numeric_limits<int32_t>::max();
	auto o_k_prev = std::numeric_limits<int32_t>::max();
	if (c + Tc <= C) {
		for (int i = 0; i < l; ++i) {
			const auto t = n * L + i;
			const auto a_k_i = ((a_idx_ptr[t] * G + g) * C + c) * M;
			const auto b_k_i = (b_idx_ptr[t] * G + g) * C + c;
			const auto o_k_i = (o_idx_ptr[t] * G + g) * M;

			if (a_k_i != a_k_prev) {
				if (m < M) {
#pragma unroll
					for (int tc = 0; tc < Tc; ++tc) {
						a[tc] = a_ptr[a_k_i + m + tc * M];
					}
				}
				a_k_prev = a_k_i;
			}

			if (b_k_i != b_k_prev) {
				if (thread_id < Tc) {
					b = b_ptr[b_k_i + thread_id];
				}
				b_k_prev = b_k_i;
			}

			if (o_k_i != o_k_prev) {
				if (m < M && i != 0) {
					atomicAdd(o_ptr + o_k_prev + m, o);
				}
				o_k_prev = o_k_i;
				o = 0.0f;
			}

			__syncwarp();
#pragma unroll
			for (int tc = 0; tc < Tc; ++tc) {
				o = __fmaf_rn(a[tc], __shfl_sync(m_lt_M_MASK, b, tc), o);
			}
		}
	} else {
		const auto Tc_max = C - c;
		for (int i = 0; i < l; ++i) {
			const auto t = n * L + i;
			const auto a_k_i = ((a_idx_ptr[t] * G + g) * C + c) * M;
			const auto b_k_i = (b_idx_ptr[t] * G + g) * C + c;
			const auto o_k_i = (o_idx_ptr[t] * G + g) * M;

			if (a_k_i != a_k_prev) {
				if (m < M) {
					for (int tc = 0; tc < Tc_max; ++tc) {
						a[tc] = a_ptr[a_k_i + m + tc * M];
					}
				}
				a_k_prev = a_k_i;
			}

			if (b_k_i != b_k_prev) {
				if (thread_id < Tc_max) {
					b = b_ptr[b_k_i + thread_id];
				}
				b_k_prev = b_k_i;
			}

			if (o_k_i != o_k_prev) {
				if (m < M && i != 0) {
					atomicAdd(o_ptr + o_k_prev + m, o);
				}
				o_k_prev = o_k_i;
				o = 0.0f;
			}

			__syncwarp();
			for (int tc = 0; tc < Tc_max; ++tc) {
				o = __fmaf_rn(a[tc], __shfl_sync(m_lt_M_MASK, b, tc), o);
			}
		}
	}
	if (m < M) {
		atomicAdd(o_ptr + o_k_prev + m, o);
	}
}

template <int Tc>
void sparse_matrix_vector_multiplication_reduction_impl_cuda_ct(
	at::Tensor a, at::Tensor a_idx, at::Tensor b, at::Tensor b_idx, at::Tensor o_idx, at::Tensor o
) {
	const auto T = a_idx.size(0);
	const auto G = a.size(1);
	const auto C = a.size(2);
	const auto M = a.size(3);

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
	const auto Mw = (M + warp_size - 1) / warp_size;

	const auto Ct = (C + Tc - 1) / Tc;
	const auto N_G_Ct_Mw = N * G * Ct * Mw;

	dim3 grid((N_G_Ct_Mw + warp_num - 1) / warp_num);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	sparse_matrix_vector_multiplication_reduction_kernel<float, float, warp_size, Tc>
		<<<grid, block, 0, stream>>>(a_ptr, a_idx_ptr, b_ptr, b_idx_ptr, o_idx_ptr, T, G, M, C, N, L, o_ptr);

	cudaError_t err{cudaGetLastError()};
	TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err))
}

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_MVMR_KERNEL_CUH
