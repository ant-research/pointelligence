#ifndef SPARSE_ENGINES_CUDA_SPARSE_MVMR_GROUPED_MMA_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_MVMR_GROUPED_MMA_CUH

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.cuh"

// ─── Grouped MVMR kernel with weight reuse ─────────────────────────────────
//
// Processes triplets grouped by kernel offset (a_idx sorted ascending).
// Weight[k] is loaded to registers once per warp and reused across
// all triplets in the segment. Input rows are gathered on-the-fly from
// global memory (no im2col / no global-memory materialization).
// Output accumulated in registers, flushed via atomicAdd.
//
// Each warp handles one (segment_k, g, ct, mw) tile.
// Multiple warps per block improve occupancy.

namespace sparse_engines_cuda {

template <typename T, int Tc>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)
sparse_mvmr_grouped_kernel(
    const T* __restrict__ a_ptr,       // weight (K_offsets, G, C, M)
    const int32_t* __restrict__ a_idx, // kernel-offset indices (T,) — sorted
    const T* __restrict__ b_ptr,       // input (N_b, G, C)
    const int32_t* __restrict__ b_idx, // input-point indices (T,)
    const int32_t* __restrict__ o_idx, // output-point indices (T,)
    const int64_t* __restrict__ seg_offs, // segment boundaries (K+1,)
    float* __restrict__ o_ptr,         // output (n_o, G, M)
    const int32_t K_offsets,
    const int32_t G,
    const int32_t C,
    const int32_t M,
    const int32_t Ct,                  // number of C-tiles
    const int32_t Mw,                  // number of M-warps
    const int32_t warps_per_block      // number of warps per block
) {
    const int32_t thread_id = threadIdx.x;
    const int32_t warp_id = threadIdx.y;  // warp index within block
    const int32_t global_warp = blockIdx.x * warps_per_block + warp_id;

    const int32_t total_tiles = K_offsets * G * Ct * Mw;
    if (global_warp >= total_tiles) return;

    // Decode global warp index into (seg_k, g, ct, mw)
    const int32_t total_tiles_per_k = G * Ct * Mw;
    const int32_t seg_k = global_warp / total_tiles_per_k;
    const int32_t tile_in_k = global_warp % total_tiles_per_k;
    const auto mw = tile_in_k % Mw;
    const auto ct = (tile_in_k / Mw) % Ct;
    const auto g = tile_in_k / (Mw * Ct);

    const int32_t seg_start = static_cast<int32_t>(seg_offs[seg_k]);
    const int32_t seg_end = static_cast<int32_t>(seg_offs[seg_k + 1]);
    const int32_t seg_len = seg_end - seg_start;
    if (seg_len == 0) return;

    const auto m = mw * 32 + thread_id;
    const auto c = ct * Tc;

    // Load weight tile
    const int64_t weight_base = static_cast<int64_t>(seg_k) * G * C * M
                                + static_cast<int64_t>(g) * C * M;

    float a_reg[Tc] = {0.0f};
    if (c + Tc <= C) {
        if (m < M) {
#pragma unroll
            for (int tc = 0; tc < Tc; ++tc) {
                a_reg[tc] = to_float_src(a_ptr[weight_base + (c + tc) * M + m]);
            }
        }
    } else {
        const auto Tc_max = C - c;
        if (m < M) {
            for (int tc = 0; tc < Tc_max; ++tc) {
                a_reg[tc] = to_float_src(a_ptr[weight_base + (c + tc) * M + m]);
            }
        }
    }

    constexpr auto FULL_MASK = 0xffffffff;
    const auto m_lt_M_MASK = __ballot_sync(FULL_MASK, m < M);

    int32_t prev_out = -1;
    float o_acc = 0.0f;

    for (int32_t t = 0; t < seg_len; ++t) {
        const int32_t global_t = seg_start + t;
        const int32_t out_idx = o_idx[global_t];
        const int32_t in_idx = b_idx[global_t];

        if (out_idx != prev_out && prev_out >= 0) {
            if (m < M) {
                atomicAdd(o_ptr + (prev_out * G + g) * M + m, o_acc);
            }
            o_acc = 0.0f;
        }

        const int64_t b_offset = (static_cast<int64_t>(in_idx) * G + g) * C + c;
        float b_val = 0.0f;
        if (c + Tc <= C) {
            if (thread_id < Tc) {
                b_val = to_float_src(b_ptr[b_offset + thread_id]);
            }
        } else {
            const auto Tc_max = C - c;
            if (thread_id < Tc_max) {
                b_val = to_float_src(b_ptr[b_offset + thread_id]);
            }
        }

        __syncwarp();

        if (c + Tc <= C) {
#pragma unroll
            for (int tc = 0; tc < Tc; ++tc) {
                o_acc = __fmaf_rn(a_reg[tc], __shfl_sync(m_lt_M_MASK, b_val, tc), o_acc);
            }
        } else {
            const auto Tc_max = C - c;
            for (int tc = 0; tc < Tc_max; ++tc) {
                o_acc = __fmaf_rn(a_reg[tc], __shfl_sync(m_lt_M_MASK, b_val, tc), o_acc);
            }
        }

        prev_out = out_idx;
    }

    if (m < M && prev_out >= 0) {
        atomicAdd(o_ptr + (prev_out * G + g) * M + m, o_acc);
    }
}

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_MVMR_GROUPED_MMA_CUH