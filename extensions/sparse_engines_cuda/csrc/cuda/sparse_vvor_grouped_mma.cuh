#ifndef SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_MMA_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_MMA_CUH

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.cuh"

// ─── Grouped VVOR kernel ───────────────────────────────────────────────────
//
// VVOR: grad_weight[k] += grad_output[i] (outer) input[j]
// Grouped by kernel offset (o_idx sorted ascending).
// Each warp handles one (segment_k, g, mt, cw) tile.
// Multiple warps per block improve occupancy.

namespace sparse_engines_cuda {

template <typename T, int Tm>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)
sparse_vvor_grouped_kernel(
    const T* __restrict__ grad_out_ptr,   // grad_output (n_o_points, G, M)
    const int32_t* __restrict__ a_idx,    // output-point indices (T,)
    const T* __restrict__ b_ptr,          // input (N_b, G, C)
    const int32_t* __restrict__ b_idx,    // input-point indices (T,)
    const int32_t* __restrict__ o_idx,    // kernel-offset indices (T,) — sorted
    const int64_t* __restrict__ seg_offs, // segment boundaries (K+1,)
    float* __restrict__ o_ptr,            // output grad_weight (K_offsets, G, M, C)
    const int32_t K_offsets,
    const int32_t G,
    const int32_t M,
    const int32_t C,
    const int32_t Mt,
    const int32_t Cw,
    const int32_t warps_per_block
) {
    const int32_t thread_id = threadIdx.x;
    const int32_t warp_id = threadIdx.y;
    const int32_t global_warp = blockIdx.x * warps_per_block + warp_id;

    const int32_t total_tiles = K_offsets * G * Mt * Cw;
    if (global_warp >= total_tiles) return;

    // Decode global warp index into (seg_k, g, mt, cw)
    const int32_t total_tiles_per_k = G * Mt * Cw;
    const int32_t seg_k = global_warp / total_tiles_per_k;
    const int32_t tile_in_k = global_warp % total_tiles_per_k;
    const auto cw = tile_in_k % Cw;
    const auto mt = (tile_in_k / Cw) % Mt;
    const auto g = tile_in_k / (Cw * Mt);

    const int32_t seg_start = static_cast<int32_t>(seg_offs[seg_k]);
    const int32_t seg_end = static_cast<int32_t>(seg_offs[seg_k + 1]);
    const int32_t seg_len = seg_end - seg_start;
    if (seg_len == 0) return;

    const auto m = mt * Tm;
    const auto c = cw * 32 + thread_id;

    // Weight grad output layout: (K_offsets, G, M, C)
    const int64_t weight_base = static_cast<int64_t>(seg_k) * G * M * C
                                + static_cast<int64_t>(g) * M * C;

    constexpr auto FULL_MASK = 0xffffffff;
    const auto valid_mask = __ballot_sync(FULL_MASK, c < C && thread_id < Tm);

    float o_acc[Tm] = {0.0f};
    int32_t prev_out = -1;
    float grad_out_reg[Tm] = {0.0f};

    for (int32_t t = 0; t < seg_len; ++t) {
        const int32_t global_t = seg_start + t;
        const int32_t out_idx = a_idx[global_t];
        const int32_t in_idx = b_idx[global_t];

        if (out_idx != prev_out) {
            if (m + Tm <= M) {
                if (thread_id < Tm) {
                    const auto m_idx = m + thread_id;
                    if (m_idx < M) {
                        grad_out_reg[thread_id] = to_float_src(
                            grad_out_ptr[(static_cast<int64_t>(out_idx) * G + g) * M + m_idx]);
                    } else {
                        grad_out_reg[thread_id] = 0.0f;
                    }
                }
            } else {
                const auto Tm_max = M - m;
                if (thread_id < Tm_max) {
                    grad_out_reg[thread_id] = to_float_src(
                        grad_out_ptr[(static_cast<int64_t>(out_idx) * G + g) * M + m + thread_id]);
                }
            }
            prev_out = out_idx;
        }

        float b_val = 0.0f;
        if (c < C) {
            b_val = to_float_src(b_ptr[(static_cast<int64_t>(in_idx) * G + g) * C + c]);
        }

        __syncwarp();

        if (c < C) {
            if (m + Tm <= M) {
#pragma unroll
                for (int tm = 0; tm < Tm; ++tm) {
                    o_acc[tm] = __fmaf_rn(
                        __shfl_sync(valid_mask, grad_out_reg[tm], tm),
                        b_val, o_acc[tm]
                    );
                }
            } else {
                const auto Tm_max = M - m;
                for (int tm = 0; tm < Tm_max; ++tm) {
                    o_acc[tm] = __fmaf_rn(
                        __shfl_sync(valid_mask, grad_out_reg[tm], tm),
                        b_val, o_acc[tm]
                    );
                }
            }
        }
    }

    // Write weight grad to global memory (K_offsets, G, M, C)
    if (c < C) {
        if (m + Tm <= M) {
#pragma unroll
            for (int tm = 0; tm < Tm; ++tm) {
                if (m + tm < M) {
                    o_ptr[weight_base + (m + tm) * C + c] = o_acc[tm];
                }
            }
        } else {
            const auto Tm_max = M - m;
            for (int tm = 0; tm < Tm_max; ++tm) {
                o_ptr[weight_base + (m + tm) * C + c] = o_acc[tm];
            }
        }
    }
}

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_MMA_CUH