#ifndef SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_WMMA_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_WMMA_CUH

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "common.cuh"

// ─── Grouped VVOR kernel with WMMA inner loop (Tier-1.5 / cycle-3 §1) ─────
//
// VVOR: grad_weight[k][m][c] += grad_out[a_idx[t]][m] * b[b_idx[t]][c]
//       summed over triplets t in segment k (o_idx-sorted).
//
// Same algorithm as sparse_vvor_grouped_mma (the scalar-FMA path), but the
// inner loop is rewritten as a chunk-of-16-triplets WMMA m16n16k16 GEMM:
//
//   For chunk of 16 triplets in segment k:
//     A[m_idx=0..15, t=0..15] = grad_out[a_idx[chunk_start + t]][m_block_start + m_idx]
//     B[t=0..15, c_idx=0..15] = b[b_idx[chunk_start + t]][c_block_start + c_idx]
//     c_frag += A @ B   (16x16 fp32 accumulator via wmma::mma_sync)
//
// At segment end, store_matrix_sync the (16, 16) c_frag to grad_weight.
//
// Each warp owns one (k, mt, ct) 16x16 tile. Total warps = K * G * (M/16) * (C/16).
// fp16/bf16 inputs use WMMA atoms; fp32 inputs fall through to the scalar-FMA
// kernel from sparse_vvor_grouped_mma.cuh (WMMA's TF32 atom has m16n16k8, not
// m16n16k16, and we don't need fp32 perf for the cycle-3 target).
//
// Pre-reg: autoresearch/threads/conv_extreme/0_expectations/cycle3_wmma_direct_vvor.md

namespace sparse_engines_cuda {

using namespace nvcuda;

// Zero-initializers for the native half-precision types (no int ctor).
__device__ __forceinline__ __half        wmma_zero(__half /*tag*/)        { return __float2half(0.0f); }
__device__ __forceinline__ __nv_bfloat16 wmma_zero(__nv_bfloat16 /*tag*/) { return __float2bfloat16(0.0f); }

// WMMA tile dims. m16n16k16 is the canonical sm_70+ fp16/bf16 atom.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Chunk size = WMMA_K. Each chunk maps to one mma_sync.
constexpr int CHUNK_SIZE = WMMA_K;

// One warp per output tile; choose warps_per_block for occupancy. 8 warps =
// 256 threads/block, ~2 KB shared mem/warp for staging, fits well in 96 KB
// sm_89 SM with multiple resident blocks.
constexpr int WMMA_WARPS_PER_BLOCK = 8;

template <typename T, typename IdxT>
__global__ void __launch_bounds__(WMMA_WARPS_PER_BLOCK * 32)
sparse_vvor_grouped_wmma_kernel(
    const T* __restrict__ grad_out_ptr,   // grad_output (N_o_points, G, M)
    const IdxT* __restrict__ a_idx,       // output-point indices (T,) — int32 or int64
    const T* __restrict__ b_ptr,          // input (N_b, G, C)
    const IdxT* __restrict__ b_idx,       // input-point indices (T,)
    const IdxT* __restrict__ o_idx,       // kernel-offset indices (T,) — sorted ascending; unused inside the kernel because seg_offs encodes the partitioning, kept for API parity
    const int64_t* __restrict__ seg_offs, // segment boundaries (K+1,) — int64 always (small)
    float* __restrict__ o_ptr,            // grad_weight (K_offsets, G, M, C) — fp32
    const int32_t K_offsets,
    const int32_t G,
    const int32_t M,
    const int32_t C,
    const int32_t Mt,                     // M tiles = M / WMMA_M
    const int32_t Ct                      // C tiles = C / WMMA_N
) {
    const int32_t thread_id = threadIdx.x;            // 0..31 (lane)
    const int32_t warp_id   = threadIdx.y;            // 0..WMMA_WARPS_PER_BLOCK-1
    const int32_t global_warp = blockIdx.x * WMMA_WARPS_PER_BLOCK + warp_id;

    const int32_t total_tiles = K_offsets * G * Mt * Ct;
    if (global_warp >= total_tiles) return;

    // Decode global warp index into (seg_k, g, mt, ct).
    const int32_t total_tiles_per_k = G * Mt * Ct;
    const int32_t seg_k     = global_warp / total_tiles_per_k;
    const int32_t tile_in_k = global_warp % total_tiles_per_k;
    const int32_t ct        = tile_in_k % Ct;
    const int32_t mt        = (tile_in_k / Ct) % Mt;
    const int32_t g         = tile_in_k / (Ct * Mt);

    const int32_t m_block_start = mt * WMMA_M;
    const int32_t c_block_start = ct * WMMA_N;

    const int32_t seg_start = static_cast<int32_t>(seg_offs[seg_k]);
    const int32_t seg_end   = static_cast<int32_t>(seg_offs[seg_k + 1]);
    const int32_t seg_len   = seg_end - seg_start;

    // ── Shared-memory staging for A (grad_out tile) and B (b tile) ──
    // Pad one element per row to avoid bank conflicts on column loads.
    constexpr int A_STRIDE = WMMA_M + 8;   // 24 — avoids 32-bank conflicts on fp16 col-loads
    constexpr int B_STRIDE = WMMA_N + 8;
    __shared__ T a_smem[WMMA_WARPS_PER_BLOCK][WMMA_K * A_STRIDE];  // [t, m] layout
    __shared__ T b_smem[WMMA_WARPS_PER_BLOCK][WMMA_K * B_STRIDE];  // [t, c] layout

    T* a_smem_warp = a_smem[warp_id];
    T* b_smem_warp = b_smem[warp_id];

    // Initialize the 16x16 fp32 accumulator fragment to zero.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    if (seg_len == 0) {
        // Empty segment: write zeros and exit.
        const int64_t weight_base = static_cast<int64_t>(seg_k) * G * M * C
                                    + static_cast<int64_t>(g) * M * C;
        wmma::store_matrix_sync(
            o_ptr + weight_base + m_block_start * C + c_block_start,
            c_frag, C, wmma::mem_row_major);
        return;
    }

    // Walk segment in chunks of CHUNK_SIZE=16 triplets.
    for (int32_t chunk_start = 0; chunk_start < seg_len; chunk_start += CHUNK_SIZE) {
        const int32_t chunk_len = min(CHUNK_SIZE, seg_len - chunk_start);

        // Cooperative load: 32 threads stage 16x16=256 grad_out vals + 256 b vals.
        // Layout in shared mem:
        //   a_smem[t * A_STRIDE + m] = grad_out[a_idx[chunk + t]][m_block_start + m]
        //   b_smem[t * B_STRIDE + c] = b[b_idx[chunk + t]][c_block_start + c]
        // Each thread handles 8 (t, m) cells and 8 (t, c) cells (256 / 32 = 8 reads each).
        #pragma unroll
        for (int i = 0; i < (WMMA_K * WMMA_M) / 32; ++i) {
            const int linear = i * 32 + thread_id;       // 0..255
            const int t = linear / WMMA_M;               // 0..15
            const int m = linear % WMMA_M;               // 0..15
            T a_val;
            if (t < chunk_len) {
                const int64_t out_idx = a_idx[seg_start + chunk_start + t];
                a_val = grad_out_ptr[
                    (static_cast<int64_t>(out_idx) * G + g) * M + m_block_start + m
                ];
            } else {
                a_val = wmma_zero(T{});
            }
            a_smem_warp[t * A_STRIDE + m] = a_val;
        }
        #pragma unroll
        for (int i = 0; i < (WMMA_K * WMMA_N) / 32; ++i) {
            const int linear = i * 32 + thread_id;
            const int t = linear / WMMA_N;
            const int c = linear % WMMA_N;
            T b_val;
            if (t < chunk_len) {
                const int64_t in_idx = b_idx[seg_start + chunk_start + t];
                b_val = b_ptr[
                    (static_cast<int64_t>(in_idx) * G + g) * C + c_block_start + c
                ];
            } else {
                b_val = wmma_zero(T{});
            }
            b_smem_warp[t * B_STRIDE + c] = b_val;
        }

        __syncwarp();

        // Load WMMA fragments from shared memory.
        // A is (M=16, K=16) row-major: rows are m, columns are t.
        //   But we staged as [t, m] with stride A_STRIDE. To present (m, t)
        //   to wmma::load_matrix_sync as row_major, we want A[m][t] which
        //   means rows are m and columns are t. Our stage layout has t as
        //   the leading dim; we instead present it as col_major to flip.
        // Simpler: stage as [m, t] layout to load row_major naturally.
        //
        // Wait — let's revisit. mma_sync(C, A, B, C) computes C += A * B where:
        //   A: (M, K), row_major means consecutive K elements per row
        //   B: (K, N), row_major means consecutive N elements per row
        //   C: (M, N), accumulator
        // For our compute: C[m][c] += sum_t A[m][t] * B[t][c]
        //   So A is indexed [m][t], stored row-major → row m has 16 t-elements consecutive
        //   B is indexed [t][c], stored row-major → row t has 16 c-elements consecutive
        //
        // Our a_smem layout: a_smem[t * A_STRIDE + m] indexed [t][m].
        // To present as A[m][t] row-major, we want elements adjacent in t per row.
        // The transposition swaps the roles — load with col_major loads row m as
        // a column-major-strided sequence (m fixed, t varies → t * A_STRIDE strides).
        // load_matrix_sync(A_frag, ptr, ld=A_STRIDE, col_major) does exactly this:
        //   - Treats ptr as a column-major matrix with leading dim A_STRIDE
        //   - Per-row m-element is at ptr + t * A_STRIDE + m
        //   - mma_sync still treats A_frag as (M, K) logically — col_major just
        //     describes the layout in memory.
        //
        // So: load A as col_major from [t, m] layout; load B as row_major
        // from [t, c] layout (rows are t, columns are c — exactly B's natural layout).
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, a_smem_warp, A_STRIDE);
        wmma::load_matrix_sync(b_frag, b_smem_warp, B_STRIDE);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncwarp();
    }

    // Store the (16, 16) c_frag (fp32) to grad_weight[k][m_block..][c_block..].
    const int64_t weight_base = static_cast<int64_t>(seg_k) * G * M * C
                                + static_cast<int64_t>(g) * M * C;
    wmma::store_matrix_sync(
        o_ptr + weight_base + m_block_start * C + c_block_start,
        c_frag, C, wmma::mem_row_major);
}

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_VVOR_GROUPED_WMMA_CUH
