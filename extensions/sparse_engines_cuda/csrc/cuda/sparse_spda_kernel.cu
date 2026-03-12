#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "common.cuh"

namespace sparse_engines_cuda {

__device__ __forceinline__

	void
	warp_reduce_sum(float& val) {
	constexpr auto FULL_MASK = 0xffffffff;
	val += __shfl_xor_sync(FULL_MASK, val, 16);
	val += __shfl_xor_sync(FULL_MASK, val, 8);
	val += __shfl_xor_sync(FULL_MASK, val, 4);
	val += __shfl_xor_sync(FULL_MASK, val, 2);
	val += __shfl_xor_sync(FULL_MASK, val, 1);
}

template <typename T, int warp_size, int e_warp_num, int ev_warp_num, int e_reg_num, int ev_reg_num>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)

	forward_kernel(
		const T* __restrict__ q_ptr,
		const T* __restrict__ k_ptr,
		const T* __restrict__ v_ptr,
		const int32_t* __restrict__ q_idx_ptr,
		const int32_t* __restrict__ k_idx_ptr,
		const int32_t* __restrict__ k_cumsum_ptr,
		const float scale,
		const int q_length,
		const int H,
		const int L,
		const int S,
		const int E,
		const int Ev,
		T* __restrict__ o_ptr,
		T* __restrict__ m_ptr
	) {
	int idx_q_h = blockIdx.x * blockDim.y + threadIdx.y;
	if (idx_q_h >= q_length * H) {
		return;
	}

	const auto idx_q = idx_q_h / H;
	const auto h = idx_q_h % H;

	const auto n_l = q_idx_ptr[idx_q];
	const auto n = n_l / L;
	const auto l = n_l % L;

	const auto l_offset = (n * H + h) * L + l;
	const auto q_offset = l_offset * E;

	const int& thread_id = threadIdx.x;

	T q_regs[e_reg_num];
#pragma unroll
	for (auto i = 0; i < e_warp_num; ++i) {
		q_regs[i] = q_ptr[q_offset + i * warp_size + thread_id];
	}
	if (e_reg_num > e_warp_num) {
		const auto e_offset = e_warp_num * warp_size + thread_id;
		if (e_offset < E) {
			q_regs[e_warp_num] = q_ptr[q_offset + e_offset];
		}
	}

	T o_regs[ev_reg_num] = {0.0f};

	float s_star = 0.0f;
	float m_star = std::numeric_limits<float>::lowest();

	const auto k_begin = idx_q > 0 ? k_cumsum_ptr[idx_q - 1] : 0;
	const auto k_end = k_cumsum_ptr[idx_q];
	for (auto idx_k = k_begin; idx_k < k_end; ++idx_k) {
		const auto n_s = k_idx_ptr[idx_k];
		const auto s = n_s % S;

		const auto s_offset = (n * H + h) * S + s;
		const auto k_offset = s_offset * E;
		const auto v_offset = s_offset * Ev;

		float dot = 0.0f;
#pragma unroll
		for (auto i = 0; i < e_warp_num; ++i) {
			const auto e_offset = i * warp_size + thread_id;
			dot = __fmaf_rn(q_regs[i], k_ptr[k_offset + e_offset], dot);
		}
		if (e_reg_num > e_warp_num) {
			const auto e_offset = e_warp_num * warp_size + thread_id;
			if (e_offset < E) {
				dot = __fmaf_rn(q_regs[e_warp_num], k_ptr[k_offset + e_offset], dot);
			}
		}
		warp_reduce_sum(dot);
		dot *= scale;

		const auto max_dot = fmaxf(m_star, dot);
		const auto p_star = __expf(m_star - max_dot);
		const auto p = __expf(dot - max_dot);

#pragma unroll
		for (auto i = 0; i < ev_warp_num; ++i) {
			const auto ev_offset = i * warp_size + thread_id;
			o_regs[i] = __fmaf_rn(o_regs[i], p_star, v_ptr[v_offset + ev_offset] * p);
		}
		if (ev_reg_num > ev_warp_num) {
			const auto ev_offset = ev_warp_num * warp_size + thread_id;
			if (ev_offset < Ev) {
				o_regs[ev_warp_num] = __fmaf_rn(o_regs[ev_warp_num], p_star, v_ptr[v_offset + ev_offset] * p);
			}
		}
		s_star = __fmaf_rn(s_star, p_star, p);
		m_star = max_dot;
	}

	if (s_star != 0.0f) {
		if (thread_id == 0) {
			m_ptr[l_offset] = m_star + __logf(s_star);
		}
		const auto o_offset = l_offset * Ev;
#pragma unroll
		for (auto i = 0; i < ev_warp_num; ++i) {
			const auto ev_offset = i * warp_size + thread_id;
			o_ptr[o_offset + ev_offset] = __fdiv_rn(o_regs[i], s_star);
		}
		if (ev_reg_num > ev_warp_num) {
			const auto ev_offset = ev_warp_num * warp_size + thread_id;
			if (ev_offset < Ev) {
				o_ptr[o_offset + ev_offset] = __fdiv_rn(o_regs[ev_warp_num], s_star);
			}
		}
	}
}

template <int warp_size, int e_warp_num, int ev_warp_num, int e_reg_num, int ev_reg_num>
void forward_impl_cuda(
	at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor q_idx, at::Tensor k_idx, at::Tensor k_cumsum, const double scale, at::Tensor o, at::Tensor m
) {
	const auto q_length = q_idx.size(0);

	const auto H = q.size(1);

	const auto L = q.size(2);
	const auto S = k.size(2);

	const auto E = q.size(3);
	const auto Ev = v.size(3);

	const auto q_ptr = q.data_ptr<float>();
	const auto k_ptr = k.data_ptr<float>();
	const auto v_ptr = v.data_ptr<float>();
	const auto q_idx_ptr = q_idx.data_ptr<int32_t>();
	const auto k_idx_ptr = k_idx.data_ptr<int32_t>();
	const auto k_cumsum_ptr = k_cumsum.data_ptr<int32_t>();
	auto o_ptr = o.data_ptr<float>();
	auto m_ptr = m.data_ptr<float>();

	const auto warp_num = 32;
	dim3 grid((q_length * H + warp_num - 1) / warp_num);
	dim3 block(warp_size, warp_num);

	const auto& kernel = &forward_kernel<float, warp_size, e_warp_num, ev_warp_num, e_reg_num, ev_reg_num>;
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	kernel<<<grid, block, 0, stream>>>(q_ptr, k_ptr, v_ptr, q_idx_ptr, k_idx_ptr, k_cumsum_ptr, scale, q_length, H, L, S, E, Ev, o_ptr, m_ptr);
	cudaError_t err{cudaGetLastError()};
	TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err))
}

template <int warp_size, int e_warp_num, int ev_warp_num>
void forward_impl_cuda(
	at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor q_idx, at::Tensor k_idx, at::Tensor k_cumsum, const double scale, at::Tensor o, at::Tensor m
) {
	const auto E = q.size(3);
	const auto Ev = v.size(3);

	const int e_leftover = E - e_warp_num * warp_size;
	const int ev_leftover = Ev - ev_warp_num * warp_size;

	if (e_leftover > 0 && ev_leftover > 0) {
		forward_impl_cuda<warp_size, e_warp_num, ev_warp_num, e_warp_num + 1, ev_warp_num + 1>(q, k, v, q_idx, k_idx, k_cumsum, scale, o, m);
	} else if (e_leftover == 0 && ev_leftover > 0) {
		forward_impl_cuda<warp_size, e_warp_num, ev_warp_num, std::max(e_warp_num, 1), ev_warp_num + 1>(
			q, k, v, q_idx, k_idx, k_cumsum, scale, o, m
		);
	} else if (e_leftover > 0 && ev_leftover == 0) {
		forward_impl_cuda<warp_size, e_warp_num, ev_warp_num, e_warp_num + 1, std::max(ev_warp_num, 1)>(
			q, k, v, q_idx, k_idx, k_cumsum, scale, o, m
		);
	} else if (e_leftover == 0 && ev_leftover == 0) {
		forward_impl_cuda<warp_size, e_warp_num, ev_warp_num, std::max(e_warp_num, 1), std::max(ev_warp_num, 1)>(
			q, k, v, q_idx, k_idx, k_cumsum, scale, o, m
		);
	}
}

template <typename T, int warp_size, int e_warp_num, int ev_warp_num, int e_reg_num, int ev_reg_num>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)

	backward_kernel(
		const T* __restrict__ q_ptr,
		const T* __restrict__ k_ptr,
		const T* __restrict__ v_ptr,
		const int32_t* __restrict__ q_idx_ptr,
		const int32_t* __restrict__ k_idx_ptr,
		const int32_t* __restrict__ k_cumsum_ptr,
		const T* __restrict__ o_ptr,
		const T* __restrict__ m_ptr,
		const T* __restrict__ do_ptr,
		const float scale,
		const int q_length,
		const int H,
		const int L,
		const int S,
		const int E,
		const int Ev,
		T* __restrict__ dq_ptr,
		T* __restrict__ dk_ptr,
		T* __restrict__ dv_ptr
	) {
	int idx_q_h = blockIdx.x * blockDim.y + threadIdx.y;
	if (idx_q_h >= q_length * H) {
		return;
	}

	const auto idx_q = idx_q_h / H;
	const auto h = idx_q_h % H;

	const auto n_l = q_idx_ptr[idx_q];
	const auto n = n_l / L;
	const auto l = n_l % L;

	const auto l_offset = (n * H + h) * L + l;
	const auto q_offset = l_offset * E;
	const auto o_offset = l_offset * Ev;

	// const int &warp_id = threadIdx.y;
	const int& thread_id = threadIdx.x;

	T q_regs[e_reg_num];
#pragma unroll
	for (auto i = 0; i < e_warp_num; ++i) {
		q_regs[i] = q_ptr[q_offset + i * warp_size + thread_id];
	}
	if (e_reg_num > e_warp_num) {
		const auto e_offset = e_warp_num * warp_size + thread_id;
		if (e_offset < E) {
			q_regs[e_warp_num] = q_ptr[q_offset + e_offset];
		}
	}

	T k_regs[e_reg_num];

	T dq_regs[e_reg_num] = {0.0f};

	T do_regs[ev_reg_num];

	float d = 0.0f;
#pragma unroll
	for (auto i = 0; i < ev_warp_num; ++i) {
		const auto ev_offset = i * warp_size + thread_id;
		do_regs[i] = do_ptr[o_offset + ev_offset];
		d = __fmaf_rn(do_regs[i], o_ptr[o_offset + ev_offset], d);
	}
	if (ev_reg_num > ev_warp_num) {
		const auto ev_offset = ev_warp_num * warp_size + thread_id;
		if (ev_offset < Ev) {
			do_regs[ev_warp_num] = do_ptr[o_offset + ev_offset];
			d = __fmaf_rn(do_regs[ev_warp_num], o_ptr[o_offset + ev_offset], d);
		}
	}
	warp_reduce_sum(d);

	const auto max_dot = m_ptr[l_offset];

	const auto k_begin = idx_q > 0 ? k_cumsum_ptr[idx_q - 1] : 0;
	const auto k_end = k_cumsum_ptr[idx_q];
	for (auto idx_k = k_begin; idx_k < k_end; ++idx_k) {
		const auto n_s = k_idx_ptr[idx_k];
		const auto s = n_s % S;

		const auto s_offset = (n * H + h) * S + s;
		const auto k_offset = s_offset * E;
		const auto v_offset = s_offset * Ev;

		float dot = 0.0f;
#pragma unroll
		for (auto i = 0; i < e_warp_num; ++i) {
			const auto e_offset = i * warp_size + thread_id;
			k_regs[i] = k_ptr[k_offset + e_offset];
			dot = __fmaf_rn(q_regs[i], k_regs[i], dot);
		}
		if (e_reg_num > e_warp_num) {
			const auto e_offset = e_warp_num * warp_size + thread_id;
			if (e_offset < E) {
				k_regs[e_warp_num] = k_ptr[k_offset + e_offset];
				dot = __fmaf_rn(q_regs[e_warp_num], k_regs[e_warp_num], dot);
			}
		}
		warp_reduce_sum(dot);
		dot *= scale;

		const auto p = __expf(dot - max_dot);

		float dp = 0.0f;
#pragma unroll
		for (auto i = 0; i < ev_warp_num; ++i) {
			const auto ev_offset = i * warp_size + thread_id;
			atomicAdd(dv_ptr + v_offset + ev_offset, p * do_regs[i]);
			dp = __fmaf_rn(do_regs[i], v_ptr[v_offset + ev_offset], dp);
		}
		if (ev_reg_num > ev_warp_num) {
			const auto ev_offset = ev_warp_num * warp_size + thread_id;
			if (ev_offset < Ev) {
				atomicAdd(dv_ptr + v_offset + ev_offset, p * do_regs[ev_warp_num]);
				dp = __fmaf_rn(do_regs[ev_warp_num], v_ptr[v_offset + ev_offset], dp);
			}
		}
		warp_reduce_sum(dp);

		const auto ds = p * (dp - d) * scale;
#pragma unroll
		for (auto i = 0; i < e_warp_num; ++i) {
			const auto e_offset = i * warp_size + thread_id;
			atomicAdd(dk_ptr + k_offset + e_offset, ds * q_regs[i]);
			dq_regs[i] = __fmaf_rn(ds, k_regs[i], dq_regs[i]);
		}
		if (e_reg_num > e_warp_num) {
			const auto e_offset = e_warp_num * warp_size + thread_id;
			if (e_offset < E) {
				atomicAdd(dk_ptr + k_offset + e_offset, ds * q_regs[e_warp_num]);
				dq_regs[e_warp_num] = __fmaf_rn(ds, k_regs[e_warp_num], dq_regs[e_warp_num]);
			}
		}
	}
#pragma unroll
	for (auto i = 0; i < e_warp_num; ++i) {
		const auto e_offset = i * warp_size + thread_id;
		dq_ptr[q_offset + e_offset] = dq_regs[i];
	}
	if (e_reg_num > e_warp_num) {
		const auto e_offset = e_warp_num * warp_size + thread_id;
		if (e_offset < E) {
			dq_ptr[q_offset + e_offset] = dq_regs[e_warp_num];
		}
	}
}

template <int warp_size, int e_warp_num, int ev_warp_num, int e_reg_num, int ev_reg_num>
void backward_impl_cuda(
	at::Tensor q,
	at::Tensor k,
	at::Tensor v,
	at::Tensor q_idx,
	at::Tensor k_idx,
	at::Tensor cumsum,
	at::Tensor o,
	at::Tensor m,
	at::Tensor d_o,
	const double scale,
	at::Tensor dq,
	at::Tensor dk,
	at::Tensor dv
) {
	const auto H = q.size(1);

	const auto L = q.size(2);
	const auto S = k.size(2);

	const auto E = q.size(3);
	const auto Ev = v.size(3);

	const auto q_ptr = q.data_ptr<float>();
	const auto k_ptr = k.data_ptr<float>();
	const auto v_ptr = v.data_ptr<float>();
	const auto q_idx_ptr = q_idx.data_ptr<int32_t>();
	const auto k_idx_ptr = k_idx.data_ptr<int32_t>();
	const auto cumsum_ptr = cumsum.data_ptr<int32_t>();
	const auto o_ptr = o.data_ptr<float>();
	const auto m_ptr = m.data_ptr<float>();
	const auto do_ptr = d_o.data_ptr<float>();
	auto dq_ptr = dq.data_ptr<float>();
	auto dk_ptr = dk.data_ptr<float>();
	auto dv_ptr = dv.data_ptr<float>();

	const auto warp_num = 32;
	dim3 block(warp_size, warp_num);
	const auto length = q_idx.size(0);
	dim3 grid((length * H + warp_num - 1) / warp_num);

	const auto& kernel = &backward_kernel<float, warp_size, e_warp_num, ev_warp_num, e_reg_num, ev_reg_num>;
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	kernel<<<grid, block, 0, stream>>>(
		q_ptr, k_ptr, v_ptr, q_idx_ptr, k_idx_ptr, cumsum_ptr, o_ptr, m_ptr, do_ptr, scale, length, H, L, S, E, Ev, dq_ptr, dk_ptr, dv_ptr
	);

	cudaError_t err{cudaGetLastError()};
	TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err))
}

template <int warp_size, int e_warp_num, int ev_warp_num>
void backward_impl_cuda(
	at::Tensor q,
	at::Tensor k,
	at::Tensor v,
	at::Tensor q_idx,
	at::Tensor k_idx,
	at::Tensor cumsum,
	at::Tensor o,
	at::Tensor m,
	at::Tensor d_o,
	const double scale,
	at::Tensor dq,
	at::Tensor dk,
	at::Tensor dv
) {
	const auto E = q.size(3);
	const auto Ev = v.size(3);

	const int e_leftover = E - e_warp_num * warp_size;
	const int ev_leftover = Ev - ev_warp_num * warp_size;

	if (e_leftover > 0 && ev_leftover > 0) {
		backward_impl_cuda<warp_size, e_warp_num, ev_warp_num, e_warp_num + 1, ev_warp_num + 1>(
			q, k, v, q_idx, k_idx, cumsum, o, m, d_o, scale, dq, dk, dv
		);
	} else if (e_leftover == 0 && ev_leftover > 0) {
		backward_impl_cuda<warp_size, e_warp_num, ev_warp_num, std::max(e_warp_num, 1), ev_warp_num + 1>(
			q, k, v, q_idx, k_idx, cumsum, o, m, d_o, scale, dq, dk, dv
		);
	} else if (e_leftover > 0 && ev_leftover == 0) {
		backward_impl_cuda<warp_size, e_warp_num, ev_warp_num, e_warp_num + 1, std::max(ev_warp_num, 1)>(
			q, k, v, q_idx, k_idx, cumsum, o, m, d_o, scale, dq, dk, dv
		);
	} else if (e_leftover == 0 && ev_leftover == 0) {
		backward_impl_cuda<warp_size, e_warp_num, ev_warp_num, std::max(e_warp_num, 1), std::max(ev_warp_num, 1)>(
			q, k, v, q_idx, k_idx, cumsum, o, m, d_o, scale, dq, dk, dv
		);
	}
}

void sparse_scaled_dot_product_attention_impl_cuda(
	at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor q_idx, at::Tensor k_idx, at::Tensor k_cumsum, const double scale, at::Tensor o, at::Tensor m
) {
	const auto E = q.size(3);
	const auto Ev = v.size(3);
	const auto warp_size = 32;
	const auto e_warp_num = E / warp_size;
	const auto ev_warp_num = Ev / warp_size;

#define CASE_E_Ev(i, j)                                                                   \
	case j:                                                                               \
		forward_impl_cuda<warp_size, i, j>(q, k, v, q_idx, k_idx, k_cumsum, scale, o, m); \
		break;
#define CASE_E(i)                                                                                      \
	case i:                                                                                            \
		switch (ev_warp_num) {                                                                         \
			CASE_E_Ev(i, 0) CASE_E_Ev(i, 1) CASE_E_Ev(i, 2) default                                    \
				: TORCH_CHECK(false, "Channel number Ev: " + std::to_string(Ev) + " is not supported") \
		}                                                                                              \
		break;

	switch (e_warp_num) {
		CASE_E(0)
		CASE_E(1)
		CASE_E(2)
		default: TORCH_CHECK(false, "Channel number E: " + std::to_string(E) + " is not supported")
	}
#undef CASE_E_Ev
#undef CASE_E
}

void sparse_scaled_dot_product_attention_backward_impl_cuda(
	at::Tensor q,
	at::Tensor k,
	at::Tensor v,
	at::Tensor q_idx,
	at::Tensor k_idx,
	at::Tensor cumsum,
	at::Tensor o,
	at::Tensor m,
	at::Tensor d_o,
	const double scale,
	at::Tensor dq,
	at::Tensor dk,
	at::Tensor dv
) {
	const auto E = q.size(3);
	const auto Ev = v.size(3);
	const auto warp_size = 32;
	const auto e_warp_num = E / warp_size;
	const auto ev_warp_num = Ev / warp_size;
#define CASE_E_Ev(i, j)                                                                                   \
	case j:                                                                                               \
		backward_impl_cuda<warp_size, i, j>(q, k, v, q_idx, k_idx, cumsum, o, m, d_o, scale, dq, dk, dv); \
		break;
#define CASE_E(i)                                                                                      \
	case i:                                                                                            \
		switch (ev_warp_num) {                                                                         \
			CASE_E_Ev(i, 0) CASE_E_Ev(i, 1) CASE_E_Ev(i, 2) default                                    \
				: TORCH_CHECK(false, "Channel number Ev: " + std::to_string(Ev) + " is not supported") \
		}                                                                                              \
		break;

	switch (e_warp_num) {
		CASE_E(0)
		CASE_E(1)
		CASE_E(2)
		default: TORCH_CHECK(false, "Channel number E: " + std::to_string(E) + " is not supported")
	}
#undef CASE_E_Ev
#undef CASE_E
}

std::tuple<at::Tensor, at::Tensor> sparse_scaled_dot_product_attention_cuda(
	at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor q_idx, at::Tensor k_idx, at::Tensor k_cumsum, const double scale
) {
	sparse_scaled_dot_product_attention_check(q, k, v, q_idx, k_idx, k_cumsum);

	const auto options = at::TensorOptions().dtype(v.dtype()).device(q.device());
	auto o = torch::zeros({q.size(0), q.size(1), q.size(2), v.size(3)}, options);
	auto m = torch::zeros({q.size(0), q.size(1), q.size(2)}, options);

	q = q.contiguous();
	k = k.contiguous();
	v = v.contiguous();

	sparse_scaled_dot_product_attention_impl_cuda(q, k, v, q_idx, k_idx, k_cumsum, scale, o, m);

	return {o, m};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_scaled_dot_product_attention_backward_cuda(
	at::Tensor q,
	at::Tensor k,
	at::Tensor v,
	at::Tensor q_idx,
	at::Tensor k_idx,
	at::Tensor k_cumsum,
	at::Tensor o,
	at::Tensor m,
	at::Tensor d_o,
	const double scale
) {
	sparse_scaled_dot_product_attention_backward_check(q, k, v, q_idx, k_idx, k_cumsum, o, m, d_o);

	const auto options = at::TensorOptions().dtype(v.dtype()).device(q.device());
	auto dq = torch::zeros({q.size(0), q.size(1), q.size(2), q.size(3)}, options);
	auto dk = torch::zeros({k.size(0), k.size(1), k.size(2), k.size(3)}, options);
	auto dv = torch::zeros({v.size(0), v.size(1), v.size(2), v.size(3)}, options);

	q = q.contiguous();
	k = k.contiguous();
	v = v.contiguous();
	o = o.contiguous();
	m = m.contiguous();
	d_o = d_o.contiguous();

	sparse_scaled_dot_product_attention_backward_impl_cuda(q, k, v, q_idx, k_idx, k_cumsum, o, m, d_o, scale, dq, dk, dv);

	return {dq, dk, dv};
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
	m.impl("sparse_scaled_dot_product_attention", &sparse_scaled_dot_product_attention_cuda);
	m.impl("sparse_scaled_dot_product_attention_backward", &sparse_scaled_dot_product_attention_backward_cuda);
}
} // namespace sparse_engines_cuda