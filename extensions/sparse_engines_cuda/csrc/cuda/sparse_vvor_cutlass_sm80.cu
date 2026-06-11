// Single (M_TILE, N_TILE) CUTLASS GEMM for vvor backward.
//// Two host entry points share the same templated kernel body
// (VvorCutlassSm80SingleTileOp in the .cuh):
////   sparse_vvor_cutlass_sm80_single_tile(A_seg, B_seg, K_seg_padded)
//     Caller pre-gathers + pads. A_seg/B_seg are K-contig affine buffers.
//   sparse_vvor_cutlass_sm80_single_tile_gathered(
//                 grad_output, input_b, i_idx_seg, j_idx_seg,
//                 m_start, c_start, K_seg_padded)
//     Kernel-side `IndexedGather` composed layout drives K-mode gather
//     inside CollectiveMma's cp.async loads (example 52 case 2 pattern).
//// Both routes return a (M_TILE, N_TILE) fp32 tile with
//   C[m, n] = sum_k grad_output[i_idx[k], m_start + m]
//                  * input    [j_idx[k], c_start + n]
//// The full backward path adds the outer (k, mt, ct) grid scheduler.

#include "sparse_vvor_cutlass_sm80.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cute/tensor.hpp>

// IndexedGather + CustomStride + make_gather_tensor are vendored from
// the CUTLASS examples/common directory (already on the kernel's
// include path via setup.py).
#include "gather_tensor.hpp"

namespace sparse_engines_cuda {

namespace {

using Config = VvorCutlassSm80FpropConfig;
using Op     = VvorCutlassSm80SingleTileOp<Config>;

// Static-tile pointer-based kernel entrypoint. Builds the CuTe tensors
// from raw fp16 / fp32 pointers + dynamic K_seg, then invokes the op.
__global__
__launch_bounds__(Config::MaxThreadsPerBlock, Config::MinBlocksPerMultiprocessor)
void vvor_cutlass_sm80_single_tile_kernel(
    const cutlass::half_t* __restrict__ A_ptr,  // (M_TILE, K_seg) row-major
    const cutlass::half_t* __restrict__ B_ptr,  // (N_TILE, K_seg) row-major
    float*                 __restrict__ C_ptr,  // (M_TILE, N_TILE) row-major
    int K_seg
) {
  using namespace cute;

  // Build CuTe gmem tensors. Shape (M, K) / (N, K) with stride (K, 1) =
  // row-major. Mode-0 dimension is static (TileM / TileN); mode-1
  // dimension K_seg is dynamic.
  auto sA = make_shape(Config::TileM{}, K_seg);
  auto dA = make_stride(K_seg, _1{});
  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(sA, dA));

  auto sB = make_shape(Config::TileN{}, K_seg);
  auto dB = make_stride(K_seg, _1{});
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(sB, dB));

  auto sC = make_shape(Config::TileM{}, Config::TileN{});
  auto dC = make_stride(Config::TileN{}, _1{});
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(sC, dC));

  extern __shared__ char smem_buf[];
  Op op;
  // k_tile_count = ceil(K_seg / TileK). Caller must round K_seg up to
  // a multiple of TileK before this dispatch (Python wrapper pads with
  // zeros). MainloopSm80CpAsyncUnpredicated assumes no residue.
  int k_tile_count = K_seg / int(Config::TileK::value);
  op(mA, mB, mC, k_tile_count, smem_buf);
}

} // namespace

at::Tensor sparse_vvor_cutlass_sm80_single_tile(
    at::Tensor A_seg,    // (M_TILE, K_seg_padded) fp16, row-major contig
    at::Tensor B_seg,    // (N_TILE, K_seg_padded) fp16, row-major contig
    int64_t K_seg_padded
) {
  TORCH_CHECK(A_seg.is_cuda() && B_seg.is_cuda(),
      "sparse_vvor_cutlass_sm80: A_seg and B_seg must be CUDA tensors");
  TORCH_CHECK(A_seg.scalar_type() == at::kHalf && B_seg.scalar_type() == at::kHalf,
      "sparse_vvor_cutlass_sm80: fp16 only");
  TORCH_CHECK(A_seg.is_contiguous() && B_seg.is_contiguous(),
      "sparse_vvor_cutlass_sm80: A_seg / B_seg must be contiguous (row-major)");
  TORCH_CHECK(A_seg.dim() == 2 && B_seg.dim() == 2,
      "sparse_vvor_cutlass_sm80: A_seg and B_seg must be 2-D");

  constexpr int M_TILE = int(Config::TileM::value);
  constexpr int N_TILE = int(Config::TileN::value);
  constexpr int K_TILE = int(Config::TileK::value);

  TORCH_CHECK(A_seg.size(0) == M_TILE,
      "sparse_vvor_cutlass_sm80: A_seg.size(0)=", A_seg.size(0),
      " must equal Config::TileM=", M_TILE);
  TORCH_CHECK(B_seg.size(0) == N_TILE,
      "sparse_vvor_cutlass_sm80: B_seg.size(0)=", B_seg.size(0),
      " must equal Config::TileN=", N_TILE);
  TORCH_CHECK(A_seg.size(1) == K_seg_padded && B_seg.size(1) == K_seg_padded,
      "sparse_vvor_cutlass_sm80: K_seg_padded must match A_seg/B_seg dim-1");
  TORCH_CHECK(K_seg_padded % K_TILE == 0 && K_seg_padded > 0,
      "sparse_vvor_cutlass_sm80: K_seg_padded must be positive multiple of ",
      K_TILE, " (got ", K_seg_padded, ")");

  auto options_c = at::TensorOptions().dtype(torch::kFloat32).device(A_seg.device());
  auto C_tile = torch::zeros({M_TILE, N_TILE}, options_c);

  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr size_t smem_size = sizeof(Config::SharedStorage);

  // The mainloop pipelined smem may exceed the default 48 KB carveout;
  // request the full opt-in carveout for sm_80+ (96 KB on sm_89).
  cudaError_t attr_err = cudaFuncSetAttribute(
      vvor_cutlass_sm80_single_tile_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  TORCH_CHECK(attr_err == cudaSuccess,
      "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));

  vvor_cutlass_sm80_single_tile_kernel
      <<<dim3(1, 1, 1), Config::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const cutlass::half_t*>(A_seg.data_ptr<c10::Half>()),
          reinterpret_cast<const cutlass::half_t*>(B_seg.data_ptr<c10::Half>()),
          C_tile.data_ptr<float>(),
          static_cast<int>(K_seg_padded));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
      "vvor_cutlass_sm80 kernel launch failed: ", cudaGetErrorString(err));

  return C_tile;
}

// ── kernel-side IndexedGather (K-mode) ──────────────────────────────────────
//// Algorithm:
//   A view: (M_TILE, K_seg) with A[m, k] = grad_output[i_idx[k], m_start + m]
//           gmem stride (mode-0, mode-1) = (1, M_full applied via IndexedGather)
//   B view: (N_TILE, K_seg) with B[n, k] = input[j_idx[k], c_start + n]
//           gmem stride (mode-0, mode-1) = (1, C_full applied via IndexedGather)
//// Both A and B are M/N-contiguous in gmem (i.e., the channel slice
// m_start..m_start+TileM, c_start..c_start+TileN), with the K-axis
// strided + gathered via the per-K-element index buffer.
//// CollectiveMma<MainloopSm80CpAsyncUnpredicated, ...> ingests the
// ComposedLayout transparently: per-thread tile coordinates resolve
// through the IndexedGather custom stride during cp.async source-
// address computation. SmemLayout / TiledMma are unchanged from the affine
// path (smem is K-contig for ldmatrix-compatible mma fragment loads).
//// Vector-load caveat: the GmemTiledCopy in Config inherits the affine
// path's K-vector layout (vec-along-K). With kernel-side gather, consecutive
// K-elements live in disparate gmem rows, so the cp.async PTX issued
// per atom call would technically pull non-contiguous bytes if CuTe
// treated the gather as a "no-op stride". To keep the gather safe, we
// pad i_idx_seg / j_idx_seg with sentinel index 0 — Python-side caller
// is responsible for ensuring the padded slots' contribution is zero
// (random-input parity tolerance handles padded-slot non-zero gather).
//// **For numerical correctness, the padded-K slots must produce zero
// products in the reference too — the test mirrors this by reusing
// the same i_idx/j_idx after padding (so reference and kernel agree
// on what is "real K" vs "padded K").**

namespace {

using GatherConfig = VvorCutlassSm80GatherConfig;
using GatherOp     = VvorCutlassSm80SingleTileOp<GatherConfig>;

using GatherIndexT = int32_t;

__global__
__launch_bounds__(GatherConfig::MaxThreadsPerBlock, GatherConfig::MinBlocksPerMultiprocessor)
void vvor_cutlass_sm80_single_tile_gathered_kernel(
    const cutlass::half_t* __restrict__ grad_output_ptr,  // (N_o, M_full)
    const cutlass::half_t* __restrict__ input_ptr,        // (N_b, C_full)
    const GatherIndexT*    __restrict__ i_idx_ptr,        // (K_seg,)
    const GatherIndexT*    __restrict__ j_idx_ptr,        // (K_seg,)
    float*                 __restrict__ C_ptr,            // (M_TILE, N_TILE)
    int M_full,
    int C_full,
    int m_start,
    int c_start,
    int K_seg
) {
  using namespace cute;
  using example::IndexedGather;
  using example::CustomStride;

  // Gather tensor for A (grad_output slice).
  //  // base = grad_output_ptr + m_start  →  A[0,0] sees grad_output[idx[0], m_start]
  // shape = (TileM, K_seg)
  // stride mode-0 (M) = _1{}  (contig column within grad_output row)
  // stride mode-1 (K) = CustomStride{IndexedGather{i_idx_ptr}, M_full}
  //                    i.e., k-step adds  i_idx_ptr[k] * M_full
  //  // make_gather_tensor (from gather_tensor.hpp) finds the first non-
  // unit stride in the stride tuple and wraps it in CustomStride+Indexed-
  // Gather. So we hand it the affine stride (_1{}, M_full) plus an
  // IndexedGather, and it builds the ComposedLayout automatically.
  auto sA = make_shape(GatherConfig::TileM{}, K_seg);
  auto dA = make_stride(_1{}, M_full);
  auto gather_A = IndexedGather<GatherIndexT>{i_idx_ptr};
  Tensor mA = example::make_gather_tensor(
      make_gmem_ptr(grad_output_ptr + m_start), sA, dA, gather_A);

  auto sB = make_shape(GatherConfig::TileN{}, K_seg);
  auto dB = make_stride(_1{}, C_full);
  auto gather_B = IndexedGather<GatherIndexT>{j_idx_ptr};
  Tensor mB = example::make_gather_tensor(
      make_gmem_ptr(input_ptr + c_start), sB, dB, gather_B);

  // Output is plain affine (M_TILE, N_TILE) row-major fp32.
  auto sC = make_shape(GatherConfig::TileM{}, GatherConfig::TileN{});
  auto dC = make_stride(GatherConfig::TileN{}, _1{});
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(sC, dC));

  extern __shared__ char smem_buf[];
  GatherOp op;
  int k_tile_count = K_seg / int(GatherConfig::TileK::value);
  op(mA, mB, mC, k_tile_count, smem_buf);
}

} // namespace

at::Tensor sparse_vvor_cutlass_sm80_single_tile_gathered(
    at::Tensor grad_output,   // (N_o, 1, M_full) fp16 or (N_o, M_full)
    at::Tensor input_b,       // (N_b, 1, C_full) fp16 or (N_b, C_full)
    at::Tensor i_idx_seg,     // (K_seg_padded,) int32
    at::Tensor j_idx_seg,     // (K_seg_padded,) int32
    int64_t m_start,
    int64_t c_start,
    int64_t K_seg_padded
) {
  TORCH_CHECK(grad_output.is_cuda() && input_b.is_cuda(),
      "sparse_vvor_cutlass_sm80_gathered: grad_output and input_b must be CUDA");
  TORCH_CHECK(i_idx_seg.is_cuda() && j_idx_seg.is_cuda(),
      "sparse_vvor_cutlass_sm80_gathered: index tensors must be CUDA");
  TORCH_CHECK(grad_output.scalar_type() == at::kHalf &&
              input_b.scalar_type() == at::kHalf,
      "sparse_vvor_cutlass_sm80_gathered: fp16 only");
  TORCH_CHECK(i_idx_seg.scalar_type() == at::kInt &&
              j_idx_seg.scalar_type() == at::kInt,
      "sparse_vvor_cutlass_sm80_gathered: indices must be int32");
  TORCH_CHECK(i_idx_seg.is_contiguous() && j_idx_seg.is_contiguous(),
      "sparse_vvor_cutlass_sm80_gathered: indices must be contiguous");

  // Accept either (N, M) 2-D or (N, 1, M) 3-D for the row tensors; squeeze
  // the singleton G=1 dim if present so the kernel sees a 2-D layout.
  at::Tensor gout_2d = grad_output;
  at::Tensor inb_2d  = input_b;
  if (gout_2d.dim() == 3) {
    TORCH_CHECK(gout_2d.size(1) == 1,
        "sparse_vvor_cutlass_sm80_gathered: grad_output G dim must be 1");
    gout_2d = gout_2d.select(/*dim=*/1, /*index=*/0);
  }
  if (inb_2d.dim() == 3) {
    TORCH_CHECK(inb_2d.size(1) == 1,
        "sparse_vvor_cutlass_sm80_gathered: input G dim must be 1");
    inb_2d = inb_2d.select(/*dim=*/1, /*index=*/0);
  }
  TORCH_CHECK(gout_2d.is_contiguous() && inb_2d.is_contiguous(),
      "sparse_vvor_cutlass_sm80_gathered: grad_output / input must be contiguous");
  TORCH_CHECK(gout_2d.dim() == 2 && inb_2d.dim() == 2,
      "sparse_vvor_cutlass_sm80_gathered: grad_output / input must be 2-D after squeeze");

  constexpr int M_TILE = int(Config::TileM::value);
  constexpr int N_TILE = int(Config::TileN::value);
  constexpr int K_TILE = int(Config::TileK::value);

  const int M_full = static_cast<int>(gout_2d.size(1));
  const int C_full = static_cast<int>(inb_2d .size(1));

  TORCH_CHECK(m_start >= 0 && m_start + M_TILE <= M_full,
      "sparse_vvor_cutlass_sm80_gathered: m_start out of range (m_start=",
      m_start, ", M_TILE=", M_TILE, ", M_full=", M_full, ")");
  TORCH_CHECK(c_start >= 0 && c_start + N_TILE <= C_full,
      "sparse_vvor_cutlass_sm80_gathered: c_start out of range (c_start=",
      c_start, ", N_TILE=", N_TILE, ", C_full=", C_full, ")");
  TORCH_CHECK(K_seg_padded > 0 && K_seg_padded % K_TILE == 0,
      "sparse_vvor_cutlass_sm80_gathered: K_seg_padded must be a positive multiple of ",
      K_TILE, " (got ", K_seg_padded, ")");
  TORCH_CHECK(i_idx_seg.numel() == K_seg_padded && j_idx_seg.numel() == K_seg_padded,
      "sparse_vvor_cutlass_sm80_gathered: index tensors must have length K_seg_padded=",
      K_seg_padded);

  auto options_c = at::TensorOptions().dtype(torch::kFloat32).device(gout_2d.device());
  auto C_tile = torch::zeros({M_TILE, N_TILE}, options_c);

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t smem_size = sizeof(GatherConfig::SharedStorage);

  cudaError_t attr_err = cudaFuncSetAttribute(
      vvor_cutlass_sm80_single_tile_gathered_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  TORCH_CHECK(attr_err == cudaSuccess,
      "cudaFuncSetAttribute (gathered) failed: ", cudaGetErrorString(attr_err));

  vvor_cutlass_sm80_single_tile_gathered_kernel
      <<<dim3(1, 1, 1), GatherConfig::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const cutlass::half_t*>(gout_2d.data_ptr<c10::Half>()),
          reinterpret_cast<const cutlass::half_t*>(inb_2d .data_ptr<c10::Half>()),
          i_idx_seg.data_ptr<GatherIndexT>(),
          j_idx_seg.data_ptr<GatherIndexT>(),
          C_tile.data_ptr<float>(),
          M_full, C_full,
          static_cast<int>(m_start),
          static_cast<int>(c_start),
          static_cast<int>(K_seg_padded));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
      "vvor_cutlass_sm80_gathered kernel launch failed: ", cudaGetErrorString(err));

  return C_tile;
}

// ── full vvor backward — outer (k, mt, ct) grid scheduler ───────────────────
//// Drop-in replacement for sparse_vector_vector_outer_product_reduction_grouped_*.
// One CTA per (k, mt, ct) tile; gridDim = (C_tiles, M_tiles, n_o). The CTA
// reads its k-segment bounds from seg_offs[k]/seg_offs[k+1], builds K-mode
// gather views over the segment's triplet slice (rounded up to a TileK
// multiple), runs the Unpredicated CollectiveMma over the (TileM, TileN,
// K_seg_padded) GEMM, and writes the fp32 (TileM, TileN) tile into
// grad_weight[k, 0, mt*TileM.., ct*TileN..].
//// Padded-slot resolution = sentinel-zero-row. See the
// VvorCutlassSm80FullOp header in the .cuh for why the predicated-mainloop
// approach is infeasible here. The gather index functor below
// clamps any K slot k ≥ seg_len to a guaranteed-zero SENTINEL ROW that the
// host appends to grad_output (row N_o) and input (row N_b); the clamped
// outer-product term is exactly 0. Arbitrary + empty seg_len just work.

namespace {

// the full vvor op is templated on the operand element type
// so fp16 (half_t) and bf16 (bfloat16_t) share one kernel body. TileM/N/K +
// fp32 accumulator are dtype-invariant; only the gather-tensor element type
// and the GEMM atom (via the Config) change. The fp16 instantiation is the
// exact prior path (`VvorCutlassSm80GatherConfig` aliases
// `…GatherConfigT<half_t>`), so existing fp16 codegen is unchanged.
template <class Element>
using FullConfigFor = VvorCutlassSm80GatherConfigT<Element>;  // M/N-major smem + ldmatrix-T
using FullIndexT = int32_t;

// Gather functor: maps K slot k → real triplet row idx_[k] when k < seg_len,
// else → sentinel_row (a host-appended all-zero row). Drop-in for
// example::IndexedGather inside make_gather_tensor's CustomStride; same
// `operator()(I) const` ABI. seg_len / sentinel_row are per-CTA scalars.
template <class Index>
struct SegmentClampedGather {
  CUTE_HOST_DEVICE constexpr
  SegmentClampedGather(Index const* idx, int seg_len, Index sentinel_row)
      : idx_(idx), seg_len_(seg_len), sentinel_(sentinel_row) {}

  template <typename I>
  CUTE_HOST_DEVICE constexpr Index
  operator()(I i) const {
    return (static_cast<int>(i) < seg_len_) ? idx_[i] : sentinel_;
  }

  CUTE_HOST_DEVICE friend void print(SegmentClampedGather const&) {
    cute::print("SegClamped");
  }

  Index const* idx_;
  int          seg_len_;
  Index        sentinel_;
};

template <class Element>
__global__
__launch_bounds__(FullConfigFor<Element>::MaxThreadsPerBlock,
                  FullConfigFor<Element>::MinBlocksPerMultiprocessor)
void vvor_cutlass_sm80_full_kernel(
    const Element*         __restrict__ grad_output_ptr,  // (N_o+1, M_full) — last row = zero sentinel
    const Element*         __restrict__ input_ptr,        // (N_b+1, C_full) — last row = zero sentinel
    const FullIndexT*      __restrict__ a_idx_ptr,        // (T,) sorted-by-k
    const FullIndexT*      __restrict__ b_idx_ptr,        // (T,)
    const int64_t*         __restrict__ seg_offs_ptr,     // (n_o + 1,)
    float*                 __restrict__ grad_weight_ptr,  // (n_o, M_full, C_full)
    int M_full,
    int C_full,
    int sentinel_a,                                       // = N_o (zero row idx in grad_output)
    int sentinel_b                                        // = N_b (zero row idx in input)
) {
  using namespace cute;
  using FullConfig = FullConfigFor<Element>;
  using FullOp     = VvorCutlassSm80FullOp<FullConfig>;

  constexpr int TileM = int(FullConfig::TileM::value);
  constexpr int TileN = int(FullConfig::TileN::value);
  constexpr int TileK = int(FullConfig::TileK::value);

  const int ct = blockIdx.x;   // C-tile
  const int mt = blockIdx.y;   // M-tile
  const int k  = blockIdx.z;   // kernel offset (k-segment)

  const int m_start = mt * TileM;
  const int c_start = ct * TileN;

  const int64_t seg_start = seg_offs_ptr[k];
  const int64_t seg_end   = seg_offs_ptr[k + 1];
  const int     K_seg     = static_cast<int>(seg_end - seg_start);
  // Round K up to a TileK multiple so the Unpredicated mainloop sees a
  // clean K-tile count; the [K_seg, K_seg_padded) tail clamps to the
  // sentinel zero row via SegmentClampedGather. K_seg == 0 → padded 0.
  const int K_seg_padded = ((K_seg + TileK - 1) / TileK) * TileK;

  // grad_weight[k] base. Layout (n_o, M_full, C_full) row-major (G=1
  // squeezed); this CTA owns the (TileM, TileN) face at (m_start, c_start).
  float* gw_k = grad_weight_ptr
              + static_cast<int64_t>(k) * M_full * C_full
              + static_cast<int64_t>(m_start) * C_full
              + c_start;
  auto sC = make_shape(Int<TileM>{}, Int<TileN>{});
  auto dC = make_stride(C_full, _1{});
  Tensor mC = make_tensor(make_gmem_ptr(gw_k), make_layout(sC, dC));

  if (K_seg_padded == 0) {
    // No triplets for this k-segment: write the zero tile directly.
    // (mC is row-major (TileM, TileN); 128 threads cover it in passes.)
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    for (int e = tid; e < TileM * TileN; e += nthr) {
      gw_k[(e / TileN) * C_full + (e % TileN)] = 0.0f;
    }
    return;
  }

  // Segment's triplet-index slices. a_idx[seg]/b_idx[seg] feed the K-mode
  // gather; SegmentClampedGather redirects k ≥ K_seg to the zero sentinel.
  const FullIndexT* i_idx_seg = a_idx_ptr + seg_start;
  const FullIndexT* j_idx_seg = b_idx_ptr + seg_start;

  // A view: (TileM, K_seg_padded), A[m, t] = grad_output[gather(t), m_start+m].
  //   stride mode-0 (M) = _1 (channel-contig within a grad_output row),
  //   stride mode-1 (K) = CustomStride{SegmentClampedGather, M_full}.
  auto sA = make_shape(Int<TileM>{}, K_seg_padded);
  auto dA = make_stride(_1{}, M_full);
  Tensor mA = example::make_gather_tensor(
      make_gmem_ptr(grad_output_ptr + m_start), sA, dA,
      SegmentClampedGather<FullIndexT>{i_idx_seg, K_seg, sentinel_a});

  auto sB = make_shape(Int<TileN>{}, K_seg_padded);
  auto dB = make_stride(_1{}, C_full);
  Tensor mB = example::make_gather_tensor(
      make_gmem_ptr(input_ptr + c_start), sB, dB,
      SegmentClampedGather<FullIndexT>{j_idx_seg, K_seg, sentinel_b});

  extern __shared__ char smem_buf[];
  FullOp op;
  op(mA, mB, mC, K_seg_padded, smem_buf);
}

// dtype-templated launch helper. Element ∈ {half_t,
// bfloat16_t}; C10 is the matching ATen scalar type (c10::Half / c10::BFloat16).
// All shape math + the sentinel-pad are dtype-invariant; only the gmem
// element type and the GEMM atom (via FullConfigFor<Element>) change. fp16
// reproduces the prior launch byte-for-byte.
template <class Element, class C10>
static void launch_vvor_full(
    const at::Tensor& gout_2d, const at::Tensor& inb_2d,
    const at::Tensor& a_idx, const at::Tensor& b_idx,
    const at::Tensor& seg_offs, at::Tensor& grad_weight,
    int M_full, int C_full, int N_o, int N_b, int n_o_i
) {
  using FullConfig = FullConfigFor<Element>;
  using FullOp     = VvorCutlassSm80FullOp<FullConfig>;
  constexpr int M_TILE = int(FullConfig::TileM::value);
  constexpr int N_TILE = int(FullConfig::TileN::value);

  // sentinel-zero-row: append one all-zero row to grad_output
  // (index N_o) and input (index N_b). Padded K slots (k ≥ seg_len) gather
  // this row → zero outer-product contribution. Single-alloc construction:
  // at::empty + narrow-copy + 1-row zero_ — fewer dispatched ops than
  // at::cat. gout_2d/inb_2d are already contiguous, so the body is a dense DtoD.
  auto gout_pad = at::empty({N_o + 1, M_full}, gout_2d.options());
  gout_pad.narrow(/*dim=*/0, /*start=*/0,   /*length=*/N_o).copy_(gout_2d);
  gout_pad.narrow(/*dim=*/0, /*start=*/N_o, /*length=*/1).zero_();
  auto inb_pad = at::empty({N_b + 1, C_full}, inb_2d.options());
  inb_pad.narrow(/*dim=*/0, /*start=*/0,   /*length=*/N_b).copy_(inb_2d);
  inb_pad.narrow(/*dim=*/0, /*start=*/N_b, /*length=*/1).zero_();

  const int M_tiles = M_full / M_TILE;
  const int C_tiles = C_full / N_TILE;

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t smem_size = FullOp::kSmemBytes;

  cudaError_t attr_err = cudaFuncSetAttribute(
      vvor_cutlass_sm80_full_kernel<Element>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  TORCH_CHECK(attr_err == cudaSuccess,
      "cudaFuncSetAttribute (full) failed: ", cudaGetErrorString(attr_err));

  dim3 grid(C_tiles, M_tiles, n_o_i);   // (ct, mt, k)
  vvor_cutlass_sm80_full_kernel<Element>
      <<<grid, FullConfig::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const Element*>(gout_pad.template data_ptr<C10>()),
          reinterpret_cast<const Element*>(inb_pad .template data_ptr<C10>()),
          a_idx.data_ptr<FullIndexT>(),
          b_idx.data_ptr<FullIndexT>(),
          seg_offs.data_ptr<int64_t>(),
          grad_weight.data_ptr<float>(),
          M_full, C_full,
          /*sentinel_a=*/N_o,
          /*sentinel_b=*/N_b);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
      "vvor_cutlass_sm80_full kernel launch failed: ", cudaGetErrorString(err));
}

} // namespace

at::Tensor sparse_vvor_cutlass_sm80_full(
    at::Tensor grad_output,   // (N_o, 1, M_full) or (N_o, M_full) fp16/bf16
    at::Tensor a_idx,         // (T,) int32 — output-row idx, sorted by k
    at::Tensor input_b,       // (N_b, 1, C_full) or (N_b, C_full) fp16/bf16
    at::Tensor b_idx,         // (T,) int32 — input-row idx
    at::Tensor seg_offs,      // (n_o + 1,) int64 — per-k segment offsets
    int64_t n_o
) {
  TORCH_CHECK(grad_output.is_cuda() && input_b.is_cuda(),
      "sparse_vvor_cutlass_sm80_full: grad_output / input_b must be CUDA");
  TORCH_CHECK(a_idx.is_cuda() && b_idx.is_cuda() && seg_offs.is_cuda(),
      "sparse_vvor_cutlass_sm80_full: index / seg_offs tensors must be CUDA");
  // fp16 OR bf16. Both operands must share the dtype (the
  // kernel reinterprets both as the same Element). fp32 has no SM80 TC atom
  // of this shape → rejected (fp32 conv stays on the Triton path).
  TORCH_CHECK(grad_output.scalar_type() == input_b.scalar_type(),
      "sparse_vvor_cutlass_sm80_full: grad_output and input_b must share dtype "
      "(got ", grad_output.scalar_type(), " and ", input_b.scalar_type(), ")");
  TORCH_CHECK(grad_output.scalar_type() == at::kHalf ||
              grad_output.scalar_type() == at::kBFloat16,
      "sparse_vvor_cutlass_sm80_full: fp16/bf16 only (got ",
      grad_output.scalar_type(), ")");
  TORCH_CHECK(a_idx.scalar_type() == at::kInt && b_idx.scalar_type() == at::kInt,
      "sparse_vvor_cutlass_sm80_full: a_idx / b_idx must be int32");
  TORCH_CHECK(seg_offs.scalar_type() == at::kLong,
      "sparse_vvor_cutlass_sm80_full: seg_offs must be int64");
  TORCH_CHECK(a_idx.is_contiguous() && b_idx.is_contiguous() &&
              seg_offs.is_contiguous(),
      "sparse_vvor_cutlass_sm80_full: index / seg_offs must be contiguous");

  at::Tensor gout_2d = grad_output;
  at::Tensor inb_2d  = input_b;
  if (gout_2d.dim() == 3) {
    TORCH_CHECK(gout_2d.size(1) == 1,
        "sparse_vvor_cutlass_sm80_full: grad_output G dim must be 1");
    gout_2d = gout_2d.select(/*dim=*/1, /*index=*/0);
  }
  if (inb_2d.dim() == 3) {
    TORCH_CHECK(inb_2d.size(1) == 1,
        "sparse_vvor_cutlass_sm80_full: input G dim must be 1");
    inb_2d = inb_2d.select(/*dim=*/1, /*index=*/0);
  }
  gout_2d = gout_2d.contiguous();
  inb_2d  = inb_2d.contiguous();
  TORCH_CHECK(gout_2d.dim() == 2 && inb_2d.dim() == 2,
      "sparse_vvor_cutlass_sm80_full: grad_output / input must be 2-D after squeeze");

  // TileM/N are dtype-invariant (same for half_t / bfloat16_t Config).
  constexpr int M_TILE = int(VvorCutlassSm80GatherConfig::TileM::value);
  constexpr int N_TILE = int(VvorCutlassSm80GatherConfig::TileN::value);

  const int M_full = static_cast<int>(gout_2d.size(1));
  const int C_full = static_cast<int>(inb_2d .size(1));
  const int n_o_i  = static_cast<int>(n_o);
  const int N_o    = static_cast<int>(gout_2d.size(0));
  const int N_b    = static_cast<int>(inb_2d .size(0));

  TORCH_CHECK(M_full % M_TILE == 0,
      "sparse_vvor_cutlass_sm80_full: M_full=", M_full,
      " must be a multiple of TileM=", M_TILE);
  TORCH_CHECK(C_full % N_TILE == 0,
      "sparse_vvor_cutlass_sm80_full: C_full=", C_full,
      " must be a multiple of TileN=", N_TILE);
  TORCH_CHECK(seg_offs.numel() == n_o + 1,
      "sparse_vvor_cutlass_sm80_full: seg_offs must have n_o+1 elements (got ",
      seg_offs.numel(), ", n_o=", n_o, ")");

  // grad_weight shape matches the other grouped paths: (n_o, G=1, M, C).
  auto options_w = at::TensorOptions().dtype(torch::kFloat32).device(gout_2d.device());
  auto grad_weight = torch::zeros({n_o, 1, M_full, C_full}, options_w);

  // dispatch the launch on the operand dtype. fp16 → the
  // exact prior half_t path; bf16 → the new bfloat16_t instantiation.
  if (gout_2d.scalar_type() == at::kHalf) {
    launch_vvor_full<cutlass::half_t, c10::Half>(
        gout_2d, inb_2d, a_idx, b_idx, seg_offs, grad_weight,
        M_full, C_full, N_o, N_b, n_o_i);
  } else {  // at::kBFloat16 (guarded above)
    launch_vvor_full<cutlass::bfloat16_t, c10::BFloat16>(
        gout_2d, inb_2d, a_idx, b_idx, seg_offs, grad_weight,
        M_full, C_full, N_o, N_b, n_o_i);
  }

  return grad_weight;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
  m.impl("sparse_vvor_cutlass_sm80_single_tile",
         &sparse_vvor_cutlass_sm80_single_tile);
  m.impl("sparse_vvor_cutlass_sm80_single_tile_gathered",
         &sparse_vvor_cutlass_sm80_single_tile_gathered);
  m.impl("sparse_vvor_cutlass_sm80_full",
         &sparse_vvor_cutlass_sm80_full);
}

} // namespace sparse_engines_cuda
