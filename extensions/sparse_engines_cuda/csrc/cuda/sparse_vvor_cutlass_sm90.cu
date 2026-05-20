// Task 4 — Hopper sm_90 vvor backward entry point (cycle-4 §1.12 G14/G18).
//
// Build-only locally (dev box is sm_89; sm_90 runs on the H200 cluster
// cell). See sparse_vvor_cutlass_sm90.cuh for the full rationale on why
// this is the proven Sm80 `MainloopSm80CpAsyncUnpredicated` +
// `SegmentClampedGather` op cross-compiled for sm_90 (R2-compliant
// cp.async Hopper path) rather than a genuine WGMMA Sm90 collective.
//
// Host entry point `sparse_vvor_cutlass_sm90_full` mirrors
// `sparse_vvor_cutlass_sm80_full` exactly (same algorithm, same
// sentinel-zero-row at::cat, same (ct, mt, k) grid). It exists as a
// distinct symbol so the Python dispatcher can route sm_90 hardware
// explicitly (compute-capability query on the device) without disturbing
// the sm_80 path. On a Hopper device this launches a kernel whose SASS
// for sm_90 issues Ampere-class cp.async + mma.sync (m16n8k16 HMMA),
// which Hopper executes correctly.

#include "sparse_vvor_cutlass_sm90.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cute/tensor.hpp>

#include "gather_tensor.hpp"

namespace sparse_engines_cuda {

namespace {

using Sm90FullConfig = VvorCutlassSm90Config;                 // == Sm80GatherConfig
using Sm90FullOp     = VvorCutlassSm90FullOp<Sm90FullConfig>;  // == Sm80FullOp
using Sm90IndexT     = int32_t;

// Gather functor: identical semantics to the sm_80 SegmentClampedGather
// (k < seg_len → real triplet row; else → host-appended zero sentinel
// row). Re-declared in this TU so the sm_90 kernel's ComposedLayout is
// instantiated independently of the sm_80 TU (the sm_80 functor lives in
// an anonymous namespace in sparse_vvor_cutlass_sm80.cu and is not
// visible here). Same `operator()(I) const` ABI.
template <class Index>
struct Sm90SegmentClampedGather {
  CUTE_HOST_DEVICE constexpr
  Sm90SegmentClampedGather(Index const* idx, int seg_len, Index sentinel_row)
      : idx_(idx), seg_len_(seg_len), sentinel_(sentinel_row) {}

  template <typename I>
  CUTE_HOST_DEVICE constexpr Index
  operator()(I i) const {
    return (static_cast<int>(i) < seg_len_) ? idx_[i] : sentinel_;
  }

  CUTE_HOST_DEVICE friend void print(Sm90SegmentClampedGather const&) {
    cute::print("Sm90SegClamped");
  }

  Index const* idx_;
  int          seg_len_;
  Index        sentinel_;
};

__global__
__launch_bounds__(Sm90FullConfig::MaxThreadsPerBlock,
                  Sm90FullConfig::MinBlocksPerMultiprocessor)
void vvor_cutlass_sm90_full_kernel(
    const cutlass::half_t* __restrict__ grad_output_ptr,  // (N_o+1, M_full)
    const cutlass::half_t* __restrict__ input_ptr,        // (N_b+1, C_full)
    const Sm90IndexT*      __restrict__ a_idx_ptr,        // (T,) sorted-by-k
    const Sm90IndexT*      __restrict__ b_idx_ptr,        // (T,)
    const int64_t*         __restrict__ seg_offs_ptr,     // (n_o + 1,)
    float*                 __restrict__ grad_weight_ptr,  // (n_o, M_full, C_full)
    int M_full,
    int C_full,
    int sentinel_a,                                       // = N_o
    int sentinel_b                                        // = N_b
) {
  using namespace cute;

  constexpr int TileM = int(Sm90FullConfig::TileM::value);
  constexpr int TileN = int(Sm90FullConfig::TileN::value);
  constexpr int TileK = int(Sm90FullConfig::TileK::value);

  const int ct = blockIdx.x;
  const int mt = blockIdx.y;
  const int k  = blockIdx.z;

  const int m_start = mt * TileM;
  const int c_start = ct * TileN;

  const int64_t seg_start = seg_offs_ptr[k];
  const int64_t seg_end   = seg_offs_ptr[k + 1];
  const int     K_seg     = static_cast<int>(seg_end - seg_start);
  const int     K_seg_padded = ((K_seg + TileK - 1) / TileK) * TileK;

  float* gw_k = grad_weight_ptr
              + static_cast<int64_t>(k) * M_full * C_full
              + static_cast<int64_t>(m_start) * C_full
              + c_start;
  auto sC = make_shape(Int<TileM>{}, Int<TileN>{});
  auto dC = make_stride(C_full, _1{});
  Tensor mC = make_tensor(make_gmem_ptr(gw_k), make_layout(sC, dC));

  if (K_seg_padded == 0) {
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    for (int e = tid; e < TileM * TileN; e += nthr) {
      gw_k[(e / TileN) * C_full + (e % TileN)] = 0.0f;
    }
    return;
  }

  const Sm90IndexT* i_idx_seg = a_idx_ptr + seg_start;
  const Sm90IndexT* j_idx_seg = b_idx_ptr + seg_start;

  auto sA = make_shape(Int<TileM>{}, K_seg_padded);
  auto dA = make_stride(_1{}, M_full);
  Tensor mA = example::make_gather_tensor(
      make_gmem_ptr(grad_output_ptr + m_start), sA, dA,
      Sm90SegmentClampedGather<Sm90IndexT>{i_idx_seg, K_seg, sentinel_a});

  auto sB = make_shape(Int<TileN>{}, K_seg_padded);
  auto dB = make_stride(_1{}, C_full);
  Tensor mB = example::make_gather_tensor(
      make_gmem_ptr(input_ptr + c_start), sB, dB,
      Sm90SegmentClampedGather<Sm90IndexT>{j_idx_seg, K_seg, sentinel_b});

  extern __shared__ char smem_buf[];
  Sm90FullOp op;
  op(mA, mB, mC, K_seg_padded, smem_buf);
}

} // namespace

at::Tensor sparse_vvor_cutlass_sm90_full(
    at::Tensor grad_output,   // (N_o, 1, M_full) or (N_o, M_full) fp16
    at::Tensor a_idx,         // (T,) int32 — output-row idx, sorted by k
    at::Tensor input_b,       // (N_b, 1, C_full) or (N_b, C_full) fp16
    at::Tensor b_idx,         // (T,) int32 — input-row idx
    at::Tensor seg_offs,      // (n_o + 1,) int64 — per-k segment offsets
    int64_t n_o
) {
  TORCH_CHECK(grad_output.is_cuda() && input_b.is_cuda(),
      "sparse_vvor_cutlass_sm90_full: grad_output / input_b must be CUDA");
  TORCH_CHECK(a_idx.is_cuda() && b_idx.is_cuda() && seg_offs.is_cuda(),
      "sparse_vvor_cutlass_sm90_full: index / seg_offs tensors must be CUDA");
  TORCH_CHECK(grad_output.scalar_type() == at::kHalf &&
              input_b.scalar_type() == at::kHalf,
      "sparse_vvor_cutlass_sm90_full: fp16 only");
  TORCH_CHECK(a_idx.scalar_type() == at::kInt && b_idx.scalar_type() == at::kInt,
      "sparse_vvor_cutlass_sm90_full: a_idx / b_idx must be int32");
  TORCH_CHECK(seg_offs.scalar_type() == at::kLong,
      "sparse_vvor_cutlass_sm90_full: seg_offs must be int64");
  TORCH_CHECK(a_idx.is_contiguous() && b_idx.is_contiguous() &&
              seg_offs.is_contiguous(),
      "sparse_vvor_cutlass_sm90_full: index / seg_offs must be contiguous");

  at::Tensor gout_2d = grad_output;
  at::Tensor inb_2d  = input_b;
  if (gout_2d.dim() == 3) {
    TORCH_CHECK(gout_2d.size(1) == 1,
        "sparse_vvor_cutlass_sm90_full: grad_output G dim must be 1");
    gout_2d = gout_2d.select(/*dim=*/1, /*index=*/0);
  }
  if (inb_2d.dim() == 3) {
    TORCH_CHECK(inb_2d.size(1) == 1,
        "sparse_vvor_cutlass_sm90_full: input G dim must be 1");
    inb_2d = inb_2d.select(/*dim=*/1, /*index=*/0);
  }
  gout_2d = gout_2d.contiguous();
  inb_2d  = inb_2d.contiguous();
  TORCH_CHECK(gout_2d.dim() == 2 && inb_2d.dim() == 2,
      "sparse_vvor_cutlass_sm90_full: grad_output / input must be 2-D after squeeze");

  constexpr int M_TILE = int(Sm90FullConfig::TileM::value);
  constexpr int N_TILE = int(Sm90FullConfig::TileN::value);

  const int M_full = static_cast<int>(gout_2d.size(1));
  const int C_full = static_cast<int>(inb_2d .size(1));
  const int n_o_i  = static_cast<int>(n_o);
  const int N_o    = static_cast<int>(gout_2d.size(0));
  const int N_b    = static_cast<int>(inb_2d .size(0));

  // G17: single-alloc construction (see sm80 twin for rationale). Identical
  // contiguous (N+1, W) sentinel-row buffer, strictly fewer dispatched ops.
  auto gout_pad = at::empty({N_o + 1, M_full}, gout_2d.options());
  gout_pad.narrow(/*dim=*/0, /*start=*/0,   /*length=*/N_o).copy_(gout_2d);
  gout_pad.narrow(/*dim=*/0, /*start=*/N_o, /*length=*/1).zero_();
  auto inb_pad = at::empty({N_b + 1, C_full}, inb_2d.options());
  inb_pad.narrow(/*dim=*/0, /*start=*/0,   /*length=*/N_b).copy_(inb_2d);
  inb_pad.narrow(/*dim=*/0, /*start=*/N_b, /*length=*/1).zero_();

  TORCH_CHECK(M_full % M_TILE == 0,
      "sparse_vvor_cutlass_sm90_full: M_full=", M_full,
      " must be a multiple of TileM=", M_TILE);
  TORCH_CHECK(C_full % N_TILE == 0,
      "sparse_vvor_cutlass_sm90_full: C_full=", C_full,
      " must be a multiple of TileN=", N_TILE);
  TORCH_CHECK(seg_offs.numel() == n_o + 1,
      "sparse_vvor_cutlass_sm90_full: seg_offs must have n_o+1 elements (got ",
      seg_offs.numel(), ", n_o=", n_o, ")");

  const int M_tiles = M_full / M_TILE;
  const int C_tiles = C_full / N_TILE;

  auto options_w = at::TensorOptions().dtype(torch::kFloat32).device(gout_2d.device());
  auto grad_weight = torch::zeros({n_o, 1, M_full, C_full}, options_w);

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t smem_size = Sm90FullOp::kSmemBytes;

  cudaError_t attr_err = cudaFuncSetAttribute(
      vvor_cutlass_sm90_full_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  TORCH_CHECK(attr_err == cudaSuccess,
      "cudaFuncSetAttribute (sm90 full) failed: ", cudaGetErrorString(attr_err));

  dim3 grid(C_tiles, M_tiles, n_o_i);   // (ct, mt, k)
  vvor_cutlass_sm90_full_kernel
      <<<grid, Sm90FullConfig::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const cutlass::half_t*>(gout_pad.data_ptr<c10::Half>()),
          reinterpret_cast<const cutlass::half_t*>(inb_pad .data_ptr<c10::Half>()),
          a_idx.data_ptr<Sm90IndexT>(),
          b_idx.data_ptr<Sm90IndexT>(),
          seg_offs.data_ptr<int64_t>(),
          grad_weight.data_ptr<float>(),
          M_full, C_full,
          /*sentinel_a=*/N_o,
          /*sentinel_b=*/N_b);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
      "vvor_cutlass_sm90_full kernel launch failed: ", cudaGetErrorString(err));

  return grad_weight;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
  m.impl("sparse_vvor_cutlass_sm90_full",
         &sparse_vvor_cutlass_sm90_full);
}

} // namespace sparse_engines_cuda
