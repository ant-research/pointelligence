// Hopper sm_90 mvmr full entry point.
//// Build-only locally (dev box is sm_89; sm_90 runs on the H200 cluster
// cell). See sparse_mvmr_cutlass_sm90.cuh for the full rationale on why
// this is the proven frozen Sm80 `MainloopSm80CpAsyncUnpredicated` +
// S-axis `make_gather_tensor` + scatter-accumulate op cross-compiled
// for sm_90 (cp.async Hopper path) rather than a genuine
// WGMMA Sm90 collective. This is the mvmr twin of
// sparse_vvor_cutlass_sm90.cu — same pattern, 1:1 surface.
//// Host entry point `sparse_mvmr_cutlass_sm90_full` mirrors
// `sparse_mvmr_cutlass_sm80_full` exactly (same algorithm, same
// host-pre-transposed aT, same single-alloc sentinel-zero-row input_b
// pad, same (mt, k) grid, same scatter-accumulate epilogue). It exists
// as a distinct symbol so the Python dispatcher can route sm_90 hardware
// explicitly (compute-capability query on the device) without disturbing
// the frozen sm_80 path. On a Hopper device this launches a kernel whose
// SASS for sm_90 issues Ampere-class cp.async + mma.sync (m16n8k16
// HMMA), which Hopper executes correctly. No per-file gencode pragma:
// the arch is driven entirely by TORCH_CUDA_ARCH_LIST at build time,
// exactly as the vvor sm90 .cu (setup.py globs *.cu and applies the
// arch flags uniformly).
//// The frozen sparse_mvmr_cutlass_sm80.cuh is *included* (via the
// sm90 .cuh) for the templated MvmrCutlassSm80FullOp — never modified.

#include "sparse_mvmr_cutlass_sm90.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cute/tensor.hpp>

#include "gather_tensor.hpp"

namespace sparse_engines_cuda {

namespace {

// element-templated sm_90 mvmr config (== Sm80 FpropConfigT).
// fp16 reproduces the prior path; bf16 is the new instantiation.
template <class Element>
using Sm90MvmrConfigFor = MvmrCutlassSm80FpropConfigT<Element>;
using Sm90MvmrFullIndexT = int32_t;

// Maps a chunk S-slot s → real input row b_idx_seg[chunk_off + s] when
// (chunk_off + s) < valid_len, else → row 0 (a guaranteed in-bounds
// real input_b row). Re-declared in this TU so the sm_90 kernel's
// ComposedLayout is instantiated independently of the sm_80 TU (the
// sm_80 functor lives in an anonymous namespace in
// sparse_mvmr_cutlass_sm80.cu and is not visible here). Identical
// semantics + `operator()(I) const` ABI to the sm_80
// MvmrSegmentClampedGather — 1:1 mirror (kernel-side virtual-zero
// sentinel; the host (N_b+1,W) alloc/copy/zero_ is dropped on both
// twins. Exactness argument: see the sm_80 functor's header note — the
// OOB column's GEMM value is never scattered (run_chunk's
// `s < chunk_len` guard) so clamping OOB to in-bounds row 0 is
// bit-identical to a host-zero-row).
template <class Index>
struct Sm90MvmrSegmentClampedGather {
  CUTE_HOST_DEVICE constexpr
  Sm90MvmrSegmentClampedGather(Index const* idx, int valid_len)
      : idx_(idx), valid_len_(valid_len) {}

  template <typename I>
  CUTE_HOST_DEVICE constexpr Index
  operator()(I i) const {
    // OOB → row 0 (in-bounds; never scattered — run_chunk guard).
    return (static_cast<int>(i) < valid_len_) ? idx_[i] : Index(0);
  }

  CUTE_HOST_DEVICE friend void print(Sm90MvmrSegmentClampedGather const&) {
    cute::print("Sm90MvmrSegClamped");
  }

  Index const* idx_;
  int          valid_len_;
};

template <class Element>
__global__
__launch_bounds__(Sm90MvmrConfigFor<Element>::MaxThreadsPerBlock,
                  Sm90MvmrConfigFor<Element>::MinBlocksPerMultiprocessor)
void mvmr_cutlass_sm90_full_kernel(
    const Element*         __restrict__ aT_ptr,       // (n_o_k, M_full, C_full) row-major (pre-transposed W)
    const Sm90MvmrFullIndexT*  __restrict__ b_idx_ptr,    // (T,) sorted-by-k
    const Sm90MvmrFullIndexT*  __restrict__ o_idx_ptr,    // (T,) output-row idx
    const int64_t*         __restrict__ seg_offs_ptr, // (n_o_k + 1,)
    const Element*         __restrict__ input_b_ptr,  // (N_b, C_full) row-major (no host sentinel row)
    float*                 __restrict__ o_ptr,        // (n_o, M_full) fp32 (G=1 squeezed)
    int M_full,
    int C_full,
    int C_seg_padded
) {
  using namespace cute;
  using Sm90MvmrConfig = Sm90MvmrConfigFor<Element>;
  using Sm90MvmrFullOp = MvmrCutlassSm90FullOp<Sm90MvmrConfig>;

  constexpr int TileM = int(Sm90MvmrConfig::TileM::value);
  constexpr int S_TILE = int(Sm90MvmrConfig::TileN::value);

  const int mt = blockIdx.x;   // M-tile
  const int k  = blockIdx.y;   // kernel offset (k-segment)

  const int m_start = mt * TileM;

  const int64_t seg_start = seg_offs_ptr[k];
  const int64_t seg_end   = seg_offs_ptr[k + 1];
  const int     seg_len   = static_cast<int>(seg_end - seg_start);
  if (seg_len <= 0) return;   // empty segment → o already zero

  // Affine W[k] m-tile slice from the pre-transposed aT (n_o_k, M_full,
  // C_full) row-major: W[m, c] = aT[k, m_start+m, c]. C-contiguous
  // (stride (C_full, 1)), exactly the affine A of the single-tile entry.
  const Element* w_k = aT_ptr
      + static_cast<int64_t>(k) * M_full * C_full
      + static_cast<int64_t>(m_start) * C_full;
  auto sW = make_shape(Int<TileM>{}, C_seg_padded);
  auto dW = make_stride(C_full, _1{});
  Tensor mW = make_tensor(make_gmem_ptr(w_k), make_layout(sW, dW));

  const int n_chunks = (seg_len + S_TILE - 1) / S_TILE;
  extern __shared__ char smem_buf[];
  Sm90MvmrFullOp op;

  for (int ci = 0; ci < n_chunks; ++ci) {
    const int chunk_off = ci * S_TILE;
    const int chunk_len = (seg_len - chunk_off) < S_TILE
                              ? (seg_len - chunk_off) : S_TILE;

    const Sm90MvmrFullIndexT* b_idx_chunk = b_idx_ptr + seg_start + chunk_off;
    const Sm90MvmrFullIndexT* o_idx_chunk = o_idx_ptr + seg_start + chunk_off;

    // B chunk view: (S_TILE, C_seg_padded). B[s, c] = input_b[gather(s),
    // c]. Source input_b is (N_b, C_full) row-major → C contiguous
    // (stride 1), S strided by C_full + IndexedGather. The clamp uses
    // chunk_len so padded S-slots (s ≥ chunk_len) clamp to in-bounds
    // row 0; that column's GEMM value is never scattered (run_chunk's
    // `s < chunk_len` guard) ⇒ OOB contribution exactly 0, no host
    // sentinel row needed. Contraction axis C stays stride-1 (the
    // gathered-path structural finding) → the single-tile Config composes
    // directly.
    auto sB = make_shape(Int<S_TILE>{}, C_seg_padded);
    auto dB = make_stride(C_full, _1{});
    Tensor mB = example::make_gather_tensor(
        make_gmem_ptr(input_b_ptr), sB, dB,
        Sm90MvmrSegmentClampedGather<Sm90MvmrFullIndexT>{
            b_idx_chunk, chunk_len});

    op.run_chunk(mW, mB, C_seg_padded,
                 o_idx_chunk, chunk_len,
                 o_ptr, m_start, M_full, smem_buf);
  }
}

// dtype-templated sm_90 launch tail shared by `_full` and
// `_full_prestaged` (mirrors the sm_80 twin). Element ∈ {half_t, bfloat16_t}.
template <class Element, class C10>
static void launch_mvmr_sm90_full(
    const at::Tensor& aT, const at::Tensor& b_idx, const at::Tensor& o_idx,
    const at::Tensor& seg_offs, const at::Tensor& inb_2d, at::Tensor& o,
    int M_full, int C_full, int C_seg_padded, int M_tiles, int n_o_k
) {
  using Sm90MvmrConfig = Sm90MvmrConfigFor<Element>;
  using Sm90MvmrFullOp = MvmrCutlassSm90FullOp<Sm90MvmrConfig>;

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t smem_size = Sm90MvmrFullOp::kSmemBytes;

  cudaError_t attr_err = cudaFuncSetAttribute(
      mvmr_cutlass_sm90_full_kernel<Element>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  TORCH_CHECK(attr_err == cudaSuccess,
      "cudaFuncSetAttribute (mvmr sm90 full) failed: ",
      cudaGetErrorString(attr_err));

  dim3 grid(M_tiles, n_o_k, 1);   // (mt, k)
  mvmr_cutlass_sm90_full_kernel<Element>
      <<<grid, Sm90MvmrConfig::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const Element*>(aT.template data_ptr<C10>()),
          b_idx.data_ptr<Sm90MvmrFullIndexT>(),
          o_idx.data_ptr<Sm90MvmrFullIndexT>(),
          seg_offs.data_ptr<int64_t>(),
          reinterpret_cast<const Element*>(inb_2d.template data_ptr<C10>()),
          o.data_ptr<float>(),
          M_full, C_full, C_seg_padded);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
      "mvmr_cutlass_sm90_full kernel launch failed: ", cudaGetErrorString(err));
}

static void dispatch_mvmr_sm90_full(
    const at::Tensor& aT, const at::Tensor& b_idx, const at::Tensor& o_idx,
    const at::Tensor& seg_offs, const at::Tensor& inb_2d, at::Tensor& o,
    int M_full, int C_full, int C_seg_padded, int M_tiles, int n_o_k
) {
  if (aT.scalar_type() == at::kHalf) {
    launch_mvmr_sm90_full<cutlass::half_t, c10::Half>(
        aT, b_idx, o_idx, seg_offs, inb_2d, o,
        M_full, C_full, C_seg_padded, M_tiles, n_o_k);
  } else {  // at::kBFloat16 (guarded by the host fns)
    launch_mvmr_sm90_full<cutlass::bfloat16_t, c10::BFloat16>(
        aT, b_idx, o_idx, seg_offs, inb_2d, o,
        M_full, C_full, C_seg_padded, M_tiles, n_o_k);
  }
}

} // namespace

at::Tensor sparse_mvmr_cutlass_sm90_full(
    at::Tensor a,         // (n_o_k, 1, C_full, M_full) fp16 — affine weight W[k]
    at::Tensor b_idx,     // (T,) int — input-row idx
    at::Tensor input_b,   // (N_b, 1, C_full) or (N_b, C_full) fp16
    at::Tensor o_idx,     // (T,) int — output-row idx
    at::Tensor seg_offs,  // (n_o_k + 1,) int64 — per-k segment offsets
    int64_t n_o           // number of output points
) {
  TORCH_CHECK(a.is_cuda() && input_b.is_cuda(),
      "sparse_mvmr_cutlass_sm90_full: a / input_b must be CUDA");
  TORCH_CHECK(b_idx.is_cuda() && o_idx.is_cuda() && seg_offs.is_cuda(),
      "sparse_mvmr_cutlass_sm90_full: idx / seg_offs tensors must be CUDA");
  // fp16 OR bf16 (both operands same dtype); fp32 rejected.
  TORCH_CHECK(a.scalar_type() == input_b.scalar_type(),
      "sparse_mvmr_cutlass_sm90_full: a and input_b must share dtype (got ",
      a.scalar_type(), " and ", input_b.scalar_type(), ")");
  TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
      "sparse_mvmr_cutlass_sm90_full: fp16/bf16 only (got ", a.scalar_type(), ")");
  TORCH_CHECK(b_idx.scalar_type() == at::kInt && o_idx.scalar_type() == at::kInt,
      "sparse_mvmr_cutlass_sm90_full: b_idx / o_idx must be int32");
  TORCH_CHECK(seg_offs.scalar_type() == at::kLong,
      "sparse_mvmr_cutlass_sm90_full: seg_offs must be int64");
  TORCH_CHECK(b_idx.is_contiguous() && o_idx.is_contiguous() &&
              seg_offs.is_contiguous(),
      "sparse_mvmr_cutlass_sm90_full: idx / seg_offs must be contiguous");
  TORCH_CHECK(a.dim() == 4 && a.size(1) == 1,
      "sparse_mvmr_cutlass_sm90_full: a must be (n_o_k, 1, C_full, M_full)");

  at::Tensor inb_2d = input_b;
  if (inb_2d.dim() == 3) {
    TORCH_CHECK(inb_2d.size(1) == 1,
        "sparse_mvmr_cutlass_sm90_full: input_b G dim must be 1");
    inb_2d = inb_2d.select(/*dim=*/1, /*index=*/0);
  }
  inb_2d = inb_2d.contiguous();
  TORCH_CHECK(inb_2d.dim() == 2,
      "sparse_mvmr_cutlass_sm90_full: input_b must be 2-D after squeeze");

  // Tiles are dtype-invariant — read from the concrete fp16 config alias.
  constexpr int M_TILE = int(MvmrCutlassSm90FpropConfig::TileM::value);
  constexpr int C_TILE = int(MvmrCutlassSm90FpropConfig::TileK::value);

  const int n_o_k = static_cast<int>(a.size(0));
  const int C_full = static_cast<int>(a.size(2));
  const int M_full = static_cast<int>(a.size(3));

  TORCH_CHECK(static_cast<int>(inb_2d.size(1)) == C_full,
      "sparse_mvmr_cutlass_sm90_full: input_b C (", inb_2d.size(1),
      ") must match a's C_full (", C_full, ")");
  TORCH_CHECK(M_full % M_TILE == 0,
      "sparse_mvmr_cutlass_sm90_full: M_full=", M_full,
      " must be a multiple of TileM=", M_TILE);
  TORCH_CHECK(C_full % C_TILE == 0,
      "sparse_mvmr_cutlass_sm90_full: C_full=", C_full,
      " must be a multiple of TileK=", C_TILE);
  TORCH_CHECK(seg_offs.numel() == n_o_k + 1,
      "sparse_mvmr_cutlass_sm90_full: seg_offs must have n_o_k+1 elements (got ",
      seg_offs.numel(), ", n_o_k=", n_o_k, ")");
  TORCH_CHECK(b_idx.numel() == o_idx.numel(),
      "sparse_mvmr_cutlass_sm90_full: b_idx / o_idx length mismatch");

  // Pre-transpose the affine weight a (n_o_k, 1, C_full, M_full) →
  // aT (n_o_k, M_full, C_full) C-contiguous (the mvmr analog of vvor's
  // pre-gather staging; one transpose+contiguous). C_full enforced a
  // TileK multiple above ⇒ no per-segment C padding (C_seg_padded ==
  // C_full). Identical to the sm_80 twin.
  auto aT = a.select(/*dim=*/1, /*index=*/0)        // (n_o_k, C_full, M_full)
             .transpose(1, 2)                       // (n_o_k, M_full, C_full)
             .contiguous();

  // No host sentinel-zero-row (1:1 with the sm_80 twin). The
  // kernel-side Sm90MvmrSegmentClampedGather clamps OOB S-slots to
  // in-bounds row 0; run_chunk's `s < chunk_len` scatter guard makes
  // those columns' contribution exactly 0 by structural exclusion. The
  // real (N_b, C_full) input_b is passed directly — the (N_b+1,W)
  // at::empty/narrow-copy/zero_ (a per-call ~170 µs copy) is deleted.

  const int M_tiles = M_full / M_TILE;
  const int C_seg_padded = C_full;   // C_full is a TileK multiple

  auto options_o = at::TensorOptions().dtype(torch::kFloat32).device(a.device());
  auto o = torch::zeros({n_o, 1, M_full}, options_o);

  // dtype-dispatched launch (fp16 → prior path).
  dispatch_mvmr_sm90_full(aT, b_idx, o_idx, seg_offs, inb_2d, o,
                          M_full, C_full, C_seg_padded, M_tiles, n_o_k);

  return o;
}

// ─── sm_90 twin of the repack-skipping host entry ──────────────────────────
//// 1:1 with `sparse_mvmr_cutlass_sm80_full_prestaged` (see its header
// note for the full rationale): `_full` minus the unconditional
// `a.select(1,0).transpose(1,2).contiguous()` repack. Accepts the
// already-(n_o_k, M_full, C_full)-C-contiguous pre-staged buffer `aT`
// and feeds it straight to `mvmr_cutlass_sm90_full_kernel` (kernel
// device body UNCHANGED — same frozen Sm80 cp.async-Unpredicated op
// cross-compiled for sm_90, same `make_stride` layout). M_full /
// C_full read from `aT`'s own shape. grad_b's transposed-weight case
// is served by the caller's staging choice (drop the `.transpose(1,2)`
// for grad_b), NOT a host transpose flag and NOT a host `.contiguous()`.
at::Tensor sparse_mvmr_cutlass_sm90_full_prestaged(
    at::Tensor aT,        // (n_o_k, M_full, C_full) fp16 C-contiguous
    at::Tensor b_idx,     // (T,) int — input-row idx
    at::Tensor input_b,   // (N_b, 1, C_full) or (N_b, C_full) fp16
    at::Tensor o_idx,     // (T,) int — output-row idx
    at::Tensor seg_offs,  // (n_o_k + 1,) int64 — per-k segment offsets
    int64_t n_o           // number of output points
) {
  TORCH_CHECK(aT.is_cuda() && input_b.is_cuda(),
      "sparse_mvmr_cutlass_sm90_full_prestaged: aT / input_b must be CUDA");
  TORCH_CHECK(b_idx.is_cuda() && o_idx.is_cuda() && seg_offs.is_cuda(),
      "sparse_mvmr_cutlass_sm90_full_prestaged: idx / seg_offs tensors must be CUDA");
  // fp16 OR bf16 (both operands same dtype); fp32 rejected.
  TORCH_CHECK(aT.scalar_type() == input_b.scalar_type(),
      "sparse_mvmr_cutlass_sm90_full_prestaged: aT and input_b must share dtype "
      "(got ", aT.scalar_type(), " and ", input_b.scalar_type(), ")");
  TORCH_CHECK(aT.scalar_type() == at::kHalf || aT.scalar_type() == at::kBFloat16,
      "sparse_mvmr_cutlass_sm90_full_prestaged: fp16/bf16 only (got ",
      aT.scalar_type(), ")");
  TORCH_CHECK(b_idx.scalar_type() == at::kInt && o_idx.scalar_type() == at::kInt,
      "sparse_mvmr_cutlass_sm90_full_prestaged: b_idx / o_idx must be int32");
  TORCH_CHECK(seg_offs.scalar_type() == at::kLong,
      "sparse_mvmr_cutlass_sm90_full_prestaged: seg_offs must be int64");
  TORCH_CHECK(b_idx.is_contiguous() && o_idx.is_contiguous() &&
              seg_offs.is_contiguous(),
      "sparse_mvmr_cutlass_sm90_full_prestaged: idx / seg_offs must be contiguous");
  TORCH_CHECK(aT.dim() == 3,
      "sparse_mvmr_cutlass_sm90_full_prestaged: aT must be 3-D "
      "(n_o_k, M_full, C_full)");
  TORCH_CHECK(aT.is_contiguous(),
      "sparse_mvmr_cutlass_sm90_full_prestaged: aT must be C-contiguous "
      "(pre-staged; this entry skips the internal repack by contract)");

  at::Tensor inb_2d = input_b;
  if (inb_2d.dim() == 3) {
    TORCH_CHECK(inb_2d.size(1) == 1,
        "sparse_mvmr_cutlass_sm90_full_prestaged: input_b G dim must be 1");
    inb_2d = inb_2d.select(/*dim=*/1, /*index=*/0);
  }
  inb_2d = inb_2d.contiguous();
  TORCH_CHECK(inb_2d.dim() == 2,
      "sparse_mvmr_cutlass_sm90_full_prestaged: input_b must be 2-D after squeeze");

  constexpr int M_TILE = int(MvmrCutlassSm90FpropConfig::TileM::value);
  constexpr int C_TILE = int(MvmrCutlassSm90FpropConfig::TileK::value);

  const int n_o_k = static_cast<int>(aT.size(0));
  const int M_full = static_cast<int>(aT.size(1));
  const int C_full = static_cast<int>(aT.size(2));

  TORCH_CHECK(static_cast<int>(inb_2d.size(1)) == C_full,
      "sparse_mvmr_cutlass_sm90_full_prestaged: input_b C (", inb_2d.size(1),
      ") must match aT's C_full (", C_full, ")");
  TORCH_CHECK(M_full % M_TILE == 0,
      "sparse_mvmr_cutlass_sm90_full_prestaged: M_full=", M_full,
      " must be a multiple of TileM=", M_TILE);
  TORCH_CHECK(C_full % C_TILE == 0,
      "sparse_mvmr_cutlass_sm90_full_prestaged: C_full=", C_full,
      " must be a multiple of TileK=", C_TILE);
  TORCH_CHECK(seg_offs.numel() == n_o_k + 1,
      "sparse_mvmr_cutlass_sm90_full_prestaged: seg_offs must have n_o_k+1 elements (got ",
      seg_offs.numel(), ", n_o_k=", n_o_k, ")");
  TORCH_CHECK(b_idx.numel() == o_idx.numel(),
      "sparse_mvmr_cutlass_sm90_full_prestaged: b_idx / o_idx length mismatch");

  // *** The ONLY difference vs `sparse_mvmr_cutlass_sm90_full` ***
  // `_full` here does:
  //   auto aT = a.select(1,0).transpose(1,2).contiguous();
  // `_prestaged` skips it — the caller's `aT` already IS that buffer.

  const int M_tiles = M_full / M_TILE;
  const int C_seg_padded = C_full;   // C_full is a TileK multiple

  auto options_o = at::TensorOptions().dtype(torch::kFloat32).device(aT.device());
  auto o = torch::zeros({n_o, 1, M_full}, options_o);

  // same dtype-dispatched launch as `_full`.
  dispatch_mvmr_sm90_full(aT, b_idx, o_idx, seg_offs, inb_2d, o,
                          M_full, C_full, C_seg_padded, M_tiles, n_o_k);

  return o;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
  m.impl("sparse_mvmr_cutlass_sm90_full",
         &sparse_mvmr_cutlass_sm90_full);
  m.impl("sparse_mvmr_cutlass_sm90_full_prestaged",
         &sparse_mvmr_cutlass_sm90_full_prestaged);
}

} // namespace sparse_engines_cuda
