#ifndef SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_WMMA_COOP_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_WMMA_COOP_CUH

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "common.cuh"

// ─── Cooperative-warp split-K WMMA vvor kernel (§1.9a, cycle-3) ─────────────
//
// VVOR: grad_weight[k][m][c] += grad_out[a_idx[t]][m] * b[b_idx[t]][c]
//       summed over triplets t in segment k.
//
// Design: split-K atomicAdd. Each (k, mt, ct) 16x16 grad_weight tile is
// processed by W independent 1-warp blocks. Block (tile_id*W + w_slice)
// handles chunk slice [w_slice*cps, (w_slice+1)*cps) of segment k, where
// cps = ceil(n_chunks / W). After accumulating into fp32 c_frag, the block
// stages to __shared__ float c_smem[256] then atomicAdds to pre-zeroed o_ptr.
//
// Parallelism gain at small-C stages (enc0: 108 tiles → 864 blocks at W=8).
// Tradeoff: W× more atomicAdds at segment end (256*W per tile).
//
// Pre-reg: autoresearch/threads/conv_extreme/0_expectations/cycle3_g11_7_three_lanes.md §1.9a

namespace sparse_engines_cuda {

using namespace nvcuda;

// Zero-initializers for native CUDA fp16/bf16 types (no int ctor).
__device__ __forceinline__ __half        coop_wmma_zero(__half)        { return __float2half(0.0f); }
__device__ __forceinline__ __nv_bfloat16 coop_wmma_zero(__nv_bfloat16) { return __float2bfloat16(0.0f); }

constexpr int COOP_WMMA_M     = 16;
constexpr int COOP_WMMA_N     = 16;
constexpr int COOP_WMMA_K     = 16;
constexpr int COOP_CHUNK_SIZE = COOP_WMMA_K;
constexpr int COOP_A_STRIDE   = COOP_WMMA_M + 8;  // 24 — avoids bank conflicts
constexpr int COOP_B_STRIDE   = COOP_WMMA_N + 8;  // 24

template <typename T, typename IdxT>
__global__ void __launch_bounds__(32)
sparse_vvor_grouped_wmma_coop_kernel(
    const T* __restrict__ grad_out_ptr,
    const IdxT* __restrict__ a_idx,
    const T* __restrict__ b_ptr,
    const IdxT* __restrict__ b_idx,
    const IdxT* __restrict__ /*o_idx*/,    // API symmetry; seg_offs encodes ranges
    const int64_t* __restrict__ seg_offs,
    float* __restrict__ o_ptr,             // pre-zeroed by launcher
    const int32_t K_offsets,
    const int32_t G,
    const int32_t M,
    const int32_t C,
    const int32_t Mt,
    const int32_t Ct,
    const int32_t W                        // T-axis slice count
) {
    const int32_t lane    = threadIdx.x;
    const int32_t tile_id = blockIdx.x / W;
    const int32_t w_slice = blockIdx.x % W;

    const int32_t total_tiles = K_offsets * G * Mt * Ct;
    if (tile_id >= total_tiles) return;

    // Decode tile_id → (seg_k, g, mt, ct).
    const int32_t per_k = G * Mt * Ct;
    const int32_t seg_k = tile_id / per_k;
    const int32_t in_k  = tile_id % per_k;
    const int32_t ct    = in_k % Ct;
    const int32_t mt    = (in_k / Ct) % Mt;
    const int32_t g     = in_k / (Ct * Mt);

    const int32_t m_block_start = mt * COOP_WMMA_M;
    const int32_t c_block_start = ct * COOP_WMMA_N;

    const int32_t seg_start = static_cast<int32_t>(seg_offs[seg_k]);
    const int32_t seg_end   = static_cast<int32_t>(seg_offs[seg_k + 1]);
    const int32_t seg_len   = seg_end - seg_start;

    if (seg_len == 0) return;  // o_ptr pre-zeroed; no atomicAdd needed.

    // Chunk slice in segment-relative triplet offsets.
    const int32_t n_chunks       = (seg_len + COOP_CHUNK_SIZE - 1) / COOP_CHUNK_SIZE;
    const int32_t cps            = (n_chunks + W - 1) / W;
    const int32_t my_chunk_start = w_slice * cps * COOP_CHUNK_SIZE;
    const int32_t my_chunk_end   = min((w_slice + 1) * cps * COOP_CHUNK_SIZE, seg_len);

    if (my_chunk_start >= seg_len) return;  // excess slice for n_chunks < W.

    __shared__ T     a_smem[COOP_WMMA_K * COOP_A_STRIDE];
    __shared__ T     b_smem[COOP_WMMA_K * COOP_B_STRIDE];
    __shared__ float c_smem[COOP_WMMA_M * COOP_WMMA_N];

    wmma::fragment<wmma::accumulator, COOP_WMMA_M, COOP_WMMA_N, COOP_WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int32_t chunk_start = my_chunk_start; chunk_start < my_chunk_end;
         chunk_start += COOP_CHUNK_SIZE) {
        const int32_t chunk_len = min(COOP_CHUNK_SIZE, my_chunk_end - chunk_start);

        // Load A tile: a_smem[t * COOP_A_STRIDE + m], 256 elements, 8 per lane.
        for (int i = 0; i < (COOP_WMMA_K * COOP_WMMA_M) / 32; ++i) {
            const int s = i * 32 + lane;
            const int t = s / COOP_WMMA_M;
            const int m = s % COOP_WMMA_M;
            T val;
            if (t < chunk_len) {
                const int64_t out_idx = static_cast<int64_t>(a_idx[seg_start + chunk_start + t]);
                val = grad_out_ptr[(out_idx * G + g) * M + m_block_start + m];
            } else {
                val = coop_wmma_zero(T{});
            }
            a_smem[t * COOP_A_STRIDE + m] = val;
        }

        // Load B tile: b_smem[t * COOP_B_STRIDE + c], 256 elements, 8 per lane.
        for (int i = 0; i < (COOP_WMMA_K * COOP_WMMA_N) / 32; ++i) {
            const int s = i * 32 + lane;
            const int t = s / COOP_WMMA_N;
            const int c = s % COOP_WMMA_N;
            T val;
            if (t < chunk_len) {
                const int64_t in_idx = static_cast<int64_t>(b_idx[seg_start + chunk_start + t]);
                val = b_ptr[(in_idx * G + g) * C + c_block_start + c];
            } else {
                val = coop_wmma_zero(T{});
            }
            b_smem[t * COOP_B_STRIDE + c] = val;
        }

        __syncwarp();

        wmma::fragment<wmma::matrix_a, COOP_WMMA_M, COOP_WMMA_N, COOP_WMMA_K,
                       T, wmma::col_major> a_frag;
        wmma::fragment<wmma::matrix_b, COOP_WMMA_M, COOP_WMMA_N, COOP_WMMA_K,
                       T, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, a_smem, COOP_A_STRIDE);
        wmma::load_matrix_sync(b_frag, b_smem, COOP_B_STRIDE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncwarp();
    }

    // Stage c_frag → c_smem (row-major, stride=COOP_WMMA_N=16).
    wmma::store_matrix_sync(c_smem, c_frag, COOP_WMMA_N, wmma::mem_row_major);
    __syncwarp();

    // AtomicAdd 256 partial-result floats to o_ptr. Each lane handles 8.
    const int64_t weight_base = static_cast<int64_t>(seg_k) * G * M * C
                              + static_cast<int64_t>(g) * M * C;
    for (int i = 0; i < 8; ++i) {
        const int s       = i * 32 + lane;
        const int m_local = s / COOP_WMMA_N;
        const int c_local = s % COOP_WMMA_N;
        const int64_t idx = weight_base
                          + static_cast<int64_t>(m_block_start + m_local) * C
                          + c_block_start + c_local;
        atomicAdd(o_ptr + idx, c_smem[s]);
    }
}

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_WMMA_COOP_CUH
