// Single-tile (M_TILE, S_TILE) CUTLASS GEMM for mvmr (affine).
//// One host entry point over the templated kernel body
// (MvmrCutlassSm80SingleTileOp in the .cuh):
////   single_tile — sparse_mvmr_cutlass_sm80_single_tile(W_seg, B_seg, C_seg_padded)
//     Caller pre-gathers + pads. W_seg/B_seg are C-contig affine buffers.
//     W_seg (M_TILE, C_seg) = weight tile W[k] (affine, segment-id k only).
//     B_seg (S_TILE, C_seg) = pre-gathered input rows for this tile.
//     Returns a (M_TILE, S_TILE) fp32 tile with
//        C[m, s] = sum_c W_seg[m, c] * B_seg[s, c]
//     i.e. the mvmr GEMM core with the contraction over the CHANNEL axis.
//// The gathered entry adds the kernel-side b_idx IndexedGather on the
// input operand; the full entry adds the outer (k, m-tile, ...) grid + the
// scatter-accumulate-by-a_idx epilogue. The single-tile entry is a
// dense-write epilogue, no gather, no scatter — it exercises the GEMM core
// + the contraction-axis-C orientation only.

#include "sparse_mvmr_cutlass_sm80.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cute/tensor.hpp>

// IndexedGather + CustomStride + make_gather_tensor are vendored from the
// CUTLASS examples/common directory (already on the kernel's include path
// via setup.py — the same include the vvor gathered path uses).
#include "gather_tensor.hpp"

namespace sparse_engines_cuda {

namespace {

using MvmrConfig = MvmrCutlassSm80FpropConfig;
using MvmrOp     = MvmrCutlassSm80SingleTileOp<MvmrConfig>;

// Static-tile pointer-based kernel entrypoint. Builds the CuTe tensors
// from raw fp16 / fp32 pointers + dynamic C_seg, then invokes the op.
__global__
__launch_bounds__(MvmrConfig::MaxThreadsPerBlock, MvmrConfig::MinBlocksPerMultiprocessor)
void mvmr_cutlass_sm80_single_tile_kernel(
    const cutlass::half_t* __restrict__ W_ptr,  // (M_TILE, C_seg) row-major
    const cutlass::half_t* __restrict__ B_ptr,  // (S_TILE, C_seg) row-major
    float*                 __restrict__ O_ptr,  // (M_TILE, S_TILE) row-major
    int C_seg
) {
  using namespace cute;

  // Build CuTe gmem tensors. Shape (M, C) / (S, C) with stride (C, 1) =
  // row-major. Mode-0 dimension is static (TileM / TileN); mode-1
  // dimension C_seg is dynamic (the channel contraction length).
  auto sA = make_shape(MvmrConfig::TileM{}, C_seg);
  auto dA = make_stride(C_seg, _1{});
  Tensor mA = make_tensor(make_gmem_ptr(W_ptr), make_layout(sA, dA));

  auto sB = make_shape(MvmrConfig::TileN{}, C_seg);
  auto dB = make_stride(C_seg, _1{});
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(sB, dB));

  auto sC = make_shape(MvmrConfig::TileM{}, MvmrConfig::TileN{});
  auto dC = make_stride(MvmrConfig::TileN{}, _1{});
  Tensor mC = make_tensor(make_gmem_ptr(O_ptr), make_layout(sC, dC));

  extern __shared__ char smem_buf[];
  MvmrOp op;
  // k_tile_count = C_seg / TileK. Caller must round C_seg up to a
  // multiple of TileK before this dispatch (Python wrapper pads with
  // zeros). MainloopSm80CpAsyncUnpredicated assumes no residue.
  int k_tile_count = C_seg / int(MvmrConfig::TileK::value);
  op(mA, mB, mC, k_tile_count, smem_buf);
}

} // namespace

at::Tensor sparse_mvmr_cutlass_sm80_single_tile(
    at::Tensor W_seg,    // (M_TILE, C_seg_padded) fp16, row-major contig
    at::Tensor B_seg,    // (S_TILE, C_seg_padded) fp16, row-major contig
    int64_t C_seg_padded
) {
  TORCH_CHECK(W_seg.is_cuda() && B_seg.is_cuda(),
      "sparse_mvmr_cutlass_sm80: W_seg and B_seg must be CUDA tensors");
  TORCH_CHECK(W_seg.scalar_type() == at::kHalf && B_seg.scalar_type() == at::kHalf,
      "sparse_mvmr_cutlass_sm80: single_tile supports fp16 only");
  TORCH_CHECK(W_seg.is_contiguous() && B_seg.is_contiguous(),
      "sparse_mvmr_cutlass_sm80: W_seg / B_seg must be contiguous (row-major)");
  TORCH_CHECK(W_seg.dim() == 2 && B_seg.dim() == 2,
      "sparse_mvmr_cutlass_sm80: W_seg and B_seg must be 2-D");

  constexpr int M_TILE = int(MvmrConfig::TileM::value);
  constexpr int S_TILE = int(MvmrConfig::TileN::value);
  constexpr int C_TILE = int(MvmrConfig::TileK::value);

  TORCH_CHECK(W_seg.size(0) == M_TILE,
      "sparse_mvmr_cutlass_sm80: W_seg.size(0)=", W_seg.size(0),
      " must equal Config::TileM=", M_TILE);
  TORCH_CHECK(B_seg.size(0) == S_TILE,
      "sparse_mvmr_cutlass_sm80: B_seg.size(0)=", B_seg.size(0),
      " must equal Config::TileN=", S_TILE);
  TORCH_CHECK(W_seg.size(1) == C_seg_padded && B_seg.size(1) == C_seg_padded,
      "sparse_mvmr_cutlass_sm80: C_seg_padded must match W_seg/B_seg dim-1");
  TORCH_CHECK(C_seg_padded % C_TILE == 0 && C_seg_padded > 0,
      "sparse_mvmr_cutlass_sm80: C_seg_padded must be positive multiple of ",
      C_TILE, " (got ", C_seg_padded, ")");

  auto options_c = at::TensorOptions().dtype(torch::kFloat32).device(W_seg.device());
  auto O_tile = torch::zeros({M_TILE, S_TILE}, options_c);

  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr size_t smem_size = sizeof(MvmrConfig::SharedStorage);

  // The mainloop pipelined smem may exceed the default 48 KB carveout;
  // request the full opt-in carveout for sm_80+ (96 KB on sm_89).
  cudaError_t attr_err = cudaFuncSetAttribute(
      mvmr_cutlass_sm80_single_tile_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  TORCH_CHECK(attr_err == cudaSuccess,
      "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));

  mvmr_cutlass_sm80_single_tile_kernel
      <<<dim3(1, 1, 1), MvmrConfig::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const cutlass::half_t*>(W_seg.data_ptr<c10::Half>()),
          reinterpret_cast<const cutlass::half_t*>(B_seg.data_ptr<c10::Half>()),
          O_tile.data_ptr<float>(),
          static_cast<int>(C_seg_padded));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
      "mvmr_cutlass_sm80 kernel launch failed: ", cudaGetErrorString(err));

  return O_tile;
}

// ── Kernel-side IndexedGather on the B (input) operand ──────────────────────
//// The single-tile entry pre-gathers B Python-side (`stage_one_tile` does
// `input_b[b_idx]`). This entry moves that gather INSIDE the CUTLASS
// mainloop via a composed `IndexedGather` custom-stride layout on B,
// mirroring the vvor gathered path
// (sparse_vvor_cutlass_sm80.cu :: ..._single_tile_gathered).
//// ── Why mvmr's B-gather does NOT need vvor's transposing 2nd Config ──
//// The vvor gathered path gathered along the *contraction* axis K: its B
// view (N_TILE, K_seg) has stride (_1, C_full) so the first non-unit
// stride — the one make_gather_tensor replaces — is mode-1 = K = the
// contraction axis. With the gathered dim ON the contraction axis,
// consecutive K-elements live in disparate gmem rows, so the K-vector
// cp.async (`Layout<Shape<_1,_8>>`) cannot compose with the gather tensor
// (CuTe Copy_Traits correctly rejects the vectorization). vvor resolved
// this with a 2nd Config: M/N-major smem + a transposing SM75_U16x8_LDSM_T
// smem-copy atom.
//// mvmr's B-operand gathers along the **S/triplet axis** (the NON-
// contraction, N-tile dimension): B[s, c] = input_b[b_idx[s], c]. The
// source `input_b` is (N_b, C_full) row-major, so C is gmem-contiguous
// (stride 1) and S has the non-unit stride (C_full). make_gather_tensor
// therefore replaces the **mode-0 (S)** stride with IndexedGather and
// leaves **C (the contraction axis) stride _1, fully gmem-contiguous** —
// exactly as in the affine single-tile B. The cp.async vector still runs
// along the contiguous C axis (the single-tile `(_32,_4)×(_1,_8)`
// GmemTiledCopyB layout); the gather only re-points the per-S-row base
// address. Smem stays C-contig (LDSM_N), the TN MMA atom is unchanged.
// So the single-tile MvmrCutlassSm80FpropConfig + MvmrCutlassSm80SingleTileOp
// compose directly with the gather tensor — no transposing 2nd Config.
//// Padded-K caveat (inherited from the vvor gathered path): C_seg is
// padded by the host to a TileK multiple with zeros on W_seg's channel
// axis; B is read from `input_b` whose C axis already spans C_full ≥ C_TILE,
// so the padded channels read real-but-unused input values — they are
// masked to zero by W_seg's zero-padded channels (W[m, c]=0 for padded c ⇒
// the product is exactly 0). The S-axis padded slots (b_idx sentinel rows
// past seg_len) likewise contribute via real input rows; the reference
// mirrors the SAME b_idx (post-pad) so both sides agree on padded-slot
// values, identical to the vvor gathered contract.

namespace {

using MvmrGatherIndexT = int32_t;

__global__
__launch_bounds__(MvmrConfig::MaxThreadsPerBlock, MvmrConfig::MinBlocksPerMultiprocessor)
void mvmr_cutlass_sm80_single_tile_gathered_kernel(
    const cutlass::half_t* __restrict__ W_ptr,        // (M_TILE, C_seg) row-major affine
    const cutlass::half_t* __restrict__ input_b_ptr,  // (N_b, C_full) row-major
    const MvmrGatherIndexT* __restrict__ b_idx_ptr,   // (S_TILE,) gathered S rows
    float*                  __restrict__ O_ptr,       // (M_TILE, S_TILE) row-major
    int C_full,
    int c_start,
    int C_seg
) {
  using namespace cute;
  using example::IndexedGather;
  using example::CustomStride;

  // W stays affine, exactly as the single-tile entry: (M_TILE, C_seg)
  // row-major, C-contig.
  auto sA = make_shape(MvmrConfig::TileM{}, C_seg);
  auto dA = make_stride(C_seg, _1{});
  Tensor mA = make_tensor(make_gmem_ptr(W_ptr), make_layout(sA, dA));

  // B gather tensor. base = input_b_ptr + c_start  →  B[0,0] sees
  // input_b[b_idx[0], c_start]. shape = (S_TILE, C_seg).
  //   stride mode-0 (S) = C_full, wrapped by IndexedGather{b_idx_ptr}
  //                       → s-step adds  b_idx_ptr[s] * C_full
  //   stride mode-1 (C) = _1  (channel-contig within an input_b row)
  // make_gather_tensor replaces the first non-unit stride — mode-0 (S),
  // stride C_full — with CustomStride{IndexedGather, C_full}; mode-1's
  // stride _1 is left intact so the contraction axis C stays gmem-contig.
  auto sB = make_shape(MvmrConfig::TileN{}, C_seg);
  auto dB = make_stride(C_full, _1{});
  auto gather_B = IndexedGather<MvmrGatherIndexT>{b_idx_ptr};
  Tensor mB = example::make_gather_tensor(
      make_gmem_ptr(input_b_ptr + c_start), sB, dB, gather_B);

  // Output is plain affine (M_TILE, S_TILE) row-major fp32 (dense write).
  auto sC = make_shape(MvmrConfig::TileM{}, MvmrConfig::TileN{});
  auto dC = make_stride(MvmrConfig::TileN{}, _1{});
  Tensor mC = make_tensor(make_gmem_ptr(O_ptr), make_layout(sC, dC));

  extern __shared__ char smem_buf[];
  MvmrOp op;
  int k_tile_count = C_seg / int(MvmrConfig::TileK::value);
  op(mA, mB, mC, k_tile_count, smem_buf);
}

} // namespace

at::Tensor sparse_mvmr_cutlass_sm80_single_tile_gathered(
    at::Tensor W_seg,      // (M_TILE, C_seg_padded) fp16, row-major contig (affine)
    at::Tensor input_b,    // (N_b, 1, C_full) or (N_b, C_full) fp16, row-major
    at::Tensor b_idx_seg,  // (S_TILE,) int32 — gathered input-row indices
    int64_t c_start,
    int64_t C_seg_padded
) {
  TORCH_CHECK(W_seg.is_cuda() && input_b.is_cuda(),
      "sparse_mvmr_cutlass_sm80_gathered: W_seg / input_b must be CUDA");
  TORCH_CHECK(b_idx_seg.is_cuda(),
      "sparse_mvmr_cutlass_sm80_gathered: b_idx_seg must be CUDA");
  TORCH_CHECK(W_seg.scalar_type() == at::kHalf &&
              input_b.scalar_type() == at::kHalf,
      "sparse_mvmr_cutlass_sm80_gathered: fp16 only");
  TORCH_CHECK(b_idx_seg.scalar_type() == at::kInt,
      "sparse_mvmr_cutlass_sm80_gathered: b_idx_seg must be int32");
  TORCH_CHECK(b_idx_seg.is_contiguous(),
      "sparse_mvmr_cutlass_sm80_gathered: b_idx_seg must be contiguous");
  TORCH_CHECK(W_seg.is_contiguous(),
      "sparse_mvmr_cutlass_sm80_gathered: W_seg must be contiguous (row-major)");
  TORCH_CHECK(W_seg.dim() == 2,
      "sparse_mvmr_cutlass_sm80_gathered: W_seg must be 2-D");

  // Accept either (N_b, C_full) 2-D or (N_b, 1, C_full) 3-D for input_b;
  // squeeze the singleton G=1 dim if present so the kernel sees 2-D.
  at::Tensor inb_2d = input_b;
  if (inb_2d.dim() == 3) {
    TORCH_CHECK(inb_2d.size(1) == 1,
        "sparse_mvmr_cutlass_sm80_gathered: input_b G dim must be 1");
    inb_2d = inb_2d.select(/*dim=*/1, /*index=*/0);
  }
  inb_2d = inb_2d.contiguous();
  TORCH_CHECK(inb_2d.dim() == 2,
      "sparse_mvmr_cutlass_sm80_gathered: input_b must be 2-D after squeeze");

  constexpr int M_TILE = int(MvmrConfig::TileM::value);
  constexpr int S_TILE = int(MvmrConfig::TileN::value);
  constexpr int C_TILE = int(MvmrConfig::TileK::value);

  const int C_full = static_cast<int>(inb_2d.size(1));

  TORCH_CHECK(W_seg.size(0) == M_TILE,
      "sparse_mvmr_cutlass_sm80_gathered: W_seg.size(0)=", W_seg.size(0),
      " must equal Config::TileM=", M_TILE);
  TORCH_CHECK(W_seg.size(1) == C_seg_padded,
      "sparse_mvmr_cutlass_sm80_gathered: C_seg_padded must match W_seg dim-1");
  TORCH_CHECK(b_idx_seg.numel() == S_TILE,
      "sparse_mvmr_cutlass_sm80_gathered: b_idx_seg must have length S_TILE=",
      S_TILE, " (got ", b_idx_seg.numel(), ")");
  TORCH_CHECK(C_seg_padded % C_TILE == 0 && C_seg_padded > 0,
      "sparse_mvmr_cutlass_sm80_gathered: C_seg_padded must be a positive "
      "multiple of ", C_TILE, " (got ", C_seg_padded, ")");
  TORCH_CHECK(c_start >= 0 && c_start + C_seg_padded <= C_full,
      "sparse_mvmr_cutlass_sm80_gathered: [c_start, c_start+C_seg_padded) "
      "out of range (c_start=", c_start, ", C_seg_padded=", C_seg_padded,
      ", C_full=", C_full, ")");

  auto options_c = at::TensorOptions().dtype(torch::kFloat32).device(W_seg.device());
  auto O_tile = torch::zeros({M_TILE, S_TILE}, options_c);

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t smem_size = sizeof(MvmrConfig::SharedStorage);

  cudaError_t attr_err = cudaFuncSetAttribute(
      mvmr_cutlass_sm80_single_tile_gathered_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  TORCH_CHECK(attr_err == cudaSuccess,
      "cudaFuncSetAttribute (mvmr gathered) failed: ",
      cudaGetErrorString(attr_err));

  mvmr_cutlass_sm80_single_tile_gathered_kernel
      <<<dim3(1, 1, 1), MvmrConfig::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const cutlass::half_t*>(W_seg.data_ptr<c10::Half>()),
          reinterpret_cast<const cutlass::half_t*>(inb_2d.data_ptr<c10::Half>()),
          b_idx_seg.data_ptr<MvmrGatherIndexT>(),
          O_tile.data_ptr<float>(),
          C_full,
          static_cast<int>(c_start),
          static_cast<int>(C_seg_padded));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
      "mvmr_cutlass_sm80_gathered kernel launch failed: ",
      cudaGetErrorString(err));

  return O_tile;
}

// ── Full mvmr op — outer ragged-K-segment grid + scatter epilogue ───────────
//// gridDim = (M_tiles, n_o). One CTA per (m-tile, k-segment). The CTA
// reads its k-segment bounds from seg_offs[k]/seg_offs[k+1], then loops
// over ceil(seg_len / S_TILE) S-chunks; per chunk it builds the
// S-gathered B view (b_idx slice, clamped to the segment via
// SegmentClampedGather) over the affine C-contiguous W[k] m-tile slice
// (from the host-pre-transposed aT (K,M,C) buffer), runs the
// (M_TILE, S_TILE, C) GEMM, and scatter-accumulates each result column
// into o[o_idx[t], m] via atomicAdd with prev_out run-length coalescing
// (MvmrCutlassSm80FullOp::run_chunk). Empty segments (seg_len == 0) do
// zero chunk iterations and return — o was zero-initialised by the
// host, matching the reference.
//// Padded-S resolution: kernel-side virtual zero, no host sentinel row.
// SegmentClampedGather clamps any chunk S-slot whose in-chunk offset ≥
// chunk_len to in-bounds real row 0; that column's GEMM value is computed
// but the scatter loop stops at the real chunk_len so it is never emitted
// into o. OOB contribution is thus exactly 0 by structural exclusion —
// bit-identical to a host-appended zero row (which only ever zeroed those
// same unused columns), with the per-call host (N_b+1,W) alloc/copy/zero_
// removed. Arbitrary / empty seg_len just work.

namespace {

// the full mvmr op is templated on the operand element type
// (half_t / bfloat16_t). TileM/N/K + the fp32 scatter-accumulate epilogue are
// dtype-invariant; only the weight/input element type and the GEMM atom (via
// the Config) change. fp16 reproduces the prior path byte-for-byte.
template <class Element>
using MvmrFullConfigFor = MvmrCutlassSm80FpropConfigT<Element>;
using MvmrFullIndexT = int32_t;

// Maps a chunk S-slot s → real input row b_idx_seg[chunk_off + s] when
// (chunk_off + s) < valid_len, else → row 0 (a guaranteed in-bounds real
// input_b row — see the exactness argument below). Drop-in for
// example::IndexedGather inside make_gather_tensor's CustomStride (same
// operator()(I) const ABI). The functor is rebuilt per chunk with the
// chunk's base offset folded into the index pointer by the caller, so it
// only needs the per-chunk valid length.
//// ── Kernel-side virtual-zero sentinel (drops the host (N_b+1,W) row) ──
//// An earlier approach had the host append a physical all-zero row at
// index N_b (`at::empty({N_b+1,W})` + narrow-copy + 1-row `zero_`) and
// OOB slots gathered THAT row, so the GEMM column for an OOB S-slot
// evaluated to exactly 0.0. That host alloc + full-tensor narrow-copy was
// a per-call ~170 µs copy. It is removable *without* changing the OOB
// result by one bit, because the OOB column's GEMM value is **never read**:
////   `MvmrCutlassSm80FullOp::run_chunk`'s scatter-accumulate epilogue
//   iterates `for (int s = 0; s < chunk_len && s < S_TILE_I; ++s)` —
//   every OOB slot (s ≥ chunk_len) is structurally excluded from the
//   atomicAdd into `o`. The GEMM is column-independent (sC[m,s] depends
//   only on B[s,:]), so an OOB column's value cannot perturb any valid
//   column. The host zero row therefore only ever made an *unused*
//   column numerically zero — irrelevant to the scattered output.
//// So OOB slots may clamp to ANY in-bounds real row; row 0 is always
// in-bounds here: the kernel early-returns when seg_len ≤ 0, so a CTA
// that reaches a chunk has ≥ 1 real triplet ⇒ N_b ≥ 1 ⇒ row 0 exists.
// The GEMM reads real (unused) values for OOB columns; the scatter
// guard makes their contribution to `o` **exactly 0 by structural
// exclusion** — bit-identical to the host-zero-row behavior, with the
// host (N_b+1,W) alloc/copy/zero_ deleted.
template <class Index>
struct MvmrSegmentClampedGather {
  CUTE_HOST_DEVICE constexpr
  MvmrSegmentClampedGather(Index const* idx, int valid_len)
      : idx_(idx), valid_len_(valid_len) {}

  template <typename I>
  CUTE_HOST_DEVICE constexpr Index
  operator()(I i) const {
    // OOB → row 0 (in-bounds; the GEMM value for this column is computed
    // but never scattered — run_chunk's `s < chunk_len` guard).
    return (static_cast<int>(i) < valid_len_) ? idx_[i] : Index(0);
  }

  CUTE_HOST_DEVICE friend void print(MvmrSegmentClampedGather const&) {
    cute::print("MvmrSegClamped");
  }

  Index const* idx_;
  int          valid_len_;
};

template <class Element>
__global__
__launch_bounds__(MvmrFullConfigFor<Element>::MaxThreadsPerBlock,
                  MvmrFullConfigFor<Element>::MinBlocksPerMultiprocessor)
void mvmr_cutlass_sm80_full_kernel(
    const Element*         __restrict__ aT_ptr,       // (n_o_k, M_full, C_full) row-major (pre-transposed W)
    const MvmrFullIndexT*  __restrict__ b_idx_ptr,    // (T,) sorted-by-k
    const MvmrFullIndexT*  __restrict__ o_idx_ptr,    // (T,) output-row idx
    const int64_t*         __restrict__ seg_offs_ptr, // (n_o_k + 1,)
    const Element*         __restrict__ input_b_ptr,  // (N_b, C_full) row-major (no host sentinel row)
    float*                 __restrict__ o_ptr,        // (n_o, M_full) fp32 (G=1 squeezed)
    int M_full,
    int C_full,
    int C_seg_padded
) {
  using namespace cute;
  using MvmrConfig = MvmrFullConfigFor<Element>;
  using MvmrFullOp = MvmrCutlassSm80FullOp<MvmrConfig>;

  constexpr int TileM = int(MvmrConfig::TileM::value);
  constexpr int S_TILE = int(MvmrConfig::TileN::value);

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
  // We pass the padded
  // C_seg view; aT's real C extent is C_full ≥ C_seg_padded so the
  // mainloop reads in-bounds (no C padding needed when C_full is a
  // TileK multiple, which the host enforces).
  const Element* w_k = aT_ptr
      + static_cast<int64_t>(k) * M_full * C_full
      + static_cast<int64_t>(m_start) * C_full;
  auto sW = make_shape(Int<TileM>{}, C_seg_padded);
  auto dW = make_stride(C_full, _1{});
  Tensor mW = make_tensor(make_gmem_ptr(w_k), make_layout(sW, dW));

  const int n_chunks = (seg_len + S_TILE - 1) / S_TILE;
  extern __shared__ char smem_buf[];
  MvmrFullOp op;

  for (int ci = 0; ci < n_chunks; ++ci) {
    const int chunk_off = ci * S_TILE;
    const int chunk_len = (seg_len - chunk_off) < S_TILE
                              ? (seg_len - chunk_off) : S_TILE;

    const MvmrFullIndexT* b_idx_chunk = b_idx_ptr + seg_start + chunk_off;
    const MvmrFullIndexT* o_idx_chunk = o_idx_ptr + seg_start + chunk_off;

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
        MvmrSegmentClampedGather<MvmrFullIndexT>{
            b_idx_chunk, chunk_len});

    op.run_chunk(mW, mB, C_seg_padded,
                 o_idx_chunk, chunk_len,
                 o_ptr, m_start, M_full, smem_buf);
  }
}

// dtype-templated launch tail shared by `_full` and
// `_full_prestaged` (both feed the same kernel; only the weight-staging
// before this differs). Element ∈ {half_t, bfloat16_t}; C10 the matching
// ATen type. aT is the (n_o_k, M_full, C_full) C-contiguous staged weight.
template <class Element, class C10>
static void launch_mvmr_full(
    const at::Tensor& aT, const at::Tensor& b_idx, const at::Tensor& o_idx,
    const at::Tensor& seg_offs, const at::Tensor& inb_2d, at::Tensor& o,
    int M_full, int C_full, int C_seg_padded, int M_tiles, int n_o_k
) {
  using MvmrConfig = MvmrFullConfigFor<Element>;
  using MvmrFullOp = MvmrCutlassSm80FullOp<MvmrConfig>;

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t smem_size = MvmrFullOp::kSmemBytes;

  cudaError_t attr_err = cudaFuncSetAttribute(
      mvmr_cutlass_sm80_full_kernel<Element>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  TORCH_CHECK(attr_err == cudaSuccess,
      "cudaFuncSetAttribute (mvmr full) failed: ", cudaGetErrorString(attr_err));

  dim3 grid(M_tiles, n_o_k, 1);   // (mt, k)
  mvmr_cutlass_sm80_full_kernel<Element>
      <<<grid, MvmrConfig::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const Element*>(aT.template data_ptr<C10>()),
          b_idx.data_ptr<MvmrFullIndexT>(),
          o_idx.data_ptr<MvmrFullIndexT>(),
          seg_offs.data_ptr<int64_t>(),
          reinterpret_cast<const Element*>(inb_2d.template data_ptr<C10>()),
          o.data_ptr<float>(),
          M_full, C_full, C_seg_padded);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
      "mvmr_cutlass_sm80_full kernel launch failed: ", cudaGetErrorString(err));
}

// Dispatch the launch on the operand dtype (fp16 → prior path).
static void dispatch_mvmr_full(
    const at::Tensor& aT, const at::Tensor& b_idx, const at::Tensor& o_idx,
    const at::Tensor& seg_offs, const at::Tensor& inb_2d, at::Tensor& o,
    int M_full, int C_full, int C_seg_padded, int M_tiles, int n_o_k
) {
  if (aT.scalar_type() == at::kHalf) {
    launch_mvmr_full<cutlass::half_t, c10::Half>(
        aT, b_idx, o_idx, seg_offs, inb_2d, o,
        M_full, C_full, C_seg_padded, M_tiles, n_o_k);
  } else {  // at::kBFloat16 (guarded by the host fns)
    launch_mvmr_full<cutlass::bfloat16_t, c10::BFloat16>(
        aT, b_idx, o_idx, seg_offs, inb_2d, o,
        M_full, C_full, C_seg_padded, M_tiles, n_o_k);
  }
}

} // namespace

at::Tensor sparse_mvmr_cutlass_sm80_full(
    at::Tensor a,         // (n_o_k, 1, C_full, M_full) fp16 — affine weight W[k]
    at::Tensor b_idx,     // (T,) int — input-row idx
    at::Tensor input_b,   // (N_b, 1, C_full) or (N_b, C_full) fp16
    at::Tensor o_idx,     // (T,) int — output-row idx
    at::Tensor seg_offs,  // (n_o_k + 1,) int64 — per-k segment offsets
    int64_t n_o           // number of output points
) {
  TORCH_CHECK(a.is_cuda() && input_b.is_cuda(),
      "sparse_mvmr_cutlass_sm80_full: a / input_b must be CUDA");
  TORCH_CHECK(b_idx.is_cuda() && o_idx.is_cuda() && seg_offs.is_cuda(),
      "sparse_mvmr_cutlass_sm80_full: idx / seg_offs tensors must be CUDA");
  // fp16 OR bf16 (both operands same dtype); fp32 rejected
  // (no SM80 TC atom of this shape → fp32 conv stays on the Triton path).
  TORCH_CHECK(a.scalar_type() == input_b.scalar_type(),
      "sparse_mvmr_cutlass_sm80_full: a and input_b must share dtype (got ",
      a.scalar_type(), " and ", input_b.scalar_type(), ")");
  TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
      "sparse_mvmr_cutlass_sm80_full: fp16/bf16 only (got ", a.scalar_type(), ")");
  TORCH_CHECK(b_idx.scalar_type() == at::kInt && o_idx.scalar_type() == at::kInt,
      "sparse_mvmr_cutlass_sm80_full: b_idx / o_idx must be int32");
  TORCH_CHECK(seg_offs.scalar_type() == at::kLong,
      "sparse_mvmr_cutlass_sm80_full: seg_offs must be int64");
  TORCH_CHECK(b_idx.is_contiguous() && o_idx.is_contiguous() &&
              seg_offs.is_contiguous(),
      "sparse_mvmr_cutlass_sm80_full: idx / seg_offs must be contiguous");
  TORCH_CHECK(a.dim() == 4 && a.size(1) == 1,
      "sparse_mvmr_cutlass_sm80_full: a must be (n_o_k, 1, C_full, M_full)");

  at::Tensor inb_2d = input_b;
  if (inb_2d.dim() == 3) {
    TORCH_CHECK(inb_2d.size(1) == 1,
        "sparse_mvmr_cutlass_sm80_full: input_b G dim must be 1");
    inb_2d = inb_2d.select(/*dim=*/1, /*index=*/0);
  }
  inb_2d = inb_2d.contiguous();
  TORCH_CHECK(inb_2d.dim() == 2,
      "sparse_mvmr_cutlass_sm80_full: input_b must be 2-D after squeeze");

  constexpr int M_TILE = int(MvmrConfig::TileM::value);
  constexpr int C_TILE = int(MvmrConfig::TileK::value);

  const int n_o_k = static_cast<int>(a.size(0));
  const int C_full = static_cast<int>(a.size(2));
  const int M_full = static_cast<int>(a.size(3));

  TORCH_CHECK(static_cast<int>(inb_2d.size(1)) == C_full,
      "sparse_mvmr_cutlass_sm80_full: input_b C (", inb_2d.size(1),
      ") must match a's C_full (", C_full, ")");
  TORCH_CHECK(M_full % M_TILE == 0,
      "sparse_mvmr_cutlass_sm80_full: M_full=", M_full,
      " must be a multiple of TileM=", M_TILE);
  TORCH_CHECK(C_full % C_TILE == 0,
      "sparse_mvmr_cutlass_sm80_full: C_full=", C_full,
      " must be a multiple of TileK=", C_TILE);
  TORCH_CHECK(seg_offs.numel() == n_o_k + 1,
      "sparse_mvmr_cutlass_sm80_full: seg_offs must have n_o_k+1 elements (got ",
      seg_offs.numel(), ", n_o_k=", n_o_k, ")");
  TORCH_CHECK(b_idx.numel() == o_idx.numel(),
      "sparse_mvmr_cutlass_sm80_full: b_idx / o_idx length mismatch");

  // Pre-transpose the affine weight a (n_o_k, 1, C_full, M_full) →
  // aT (n_o_k, M_full, C_full) C-contiguous so each W[k] m-tile slice is
  // C-contiguous (the mvmr analog of vvor's pre-gather staging; one
  // transpose+contiguous). C_full is enforced a TileK multiple above so
  // no per-segment C padding is needed (C_seg_padded == C_full).
  auto aT = a.select(/*dim=*/1, /*index=*/0)        // (n_o_k, C_full, M_full)
             .transpose(1, 2)                       // (n_o_k, M_full, C_full)
             .contiguous();

  // No host sentinel-zero-row. The kernel-side
  // MvmrSegmentClampedGather clamps OOB S-slots to in-bounds row 0;
  // run_chunk's `s < chunk_len` scatter guard makes those columns'
  // contribution exactly 0 by structural exclusion (see the functor's
  // header note). The real (N_b, C_full) input_b is passed directly —
  // the (N_b+1,W) at::empty/narrow-copy/zero_ (a per-call ~170 µs copy)
  // is deleted.

  const int M_tiles = M_full / M_TILE;
  const int C_seg_padded = C_full;   // C_full is a TileK multiple

  // Output o (n_o, G=1, M_full) fp32, zero-initialised. Empty segments &
  // un-touched output rows stay exactly zero. The kernel addresses it
  // squeezed as (n_o, M_full) (G=1).
  auto options_o = at::TensorOptions().dtype(torch::kFloat32).device(a.device());
  auto o = torch::zeros({n_o, 1, M_full}, options_o);

  // dtype-dispatched launch (fp16 → prior path).
  dispatch_mvmr_full(aT, b_idx, o_idx, seg_offs, inb_2d, o,
                     M_full, C_full, C_seg_padded, M_tiles, n_o_k);

  return o;
}

// ─── Repack-skipping host entry (`_full` minus the internal
//     select/transpose/contiguous weight repack) ───────────────────────────
//// `sparse_mvmr_cutlass_sm80_full` re-materializes a fresh
// (n_o_k, M_full, C_full) C-contiguous weight buffer on EVERY call via
//   `a.select(1,0).transpose(1,2).contiguous()`   (lines 543-545 above)
// regardless of the caller's input layout. For the fused conv
// Function this repack is wasted: the caller already stages the weight
// C-contiguous once at the autograd-Function boundary. For grad_b it is
// an irreducible ~14 MB DtoD copy per iter (the post-fusion residual).
//// `_full_prestaged` is `_full` byte-for-byte EXCEPT the repack: it
// accepts the already-(n_o_k, M_full, C_full)-C-contiguous buffer `aT`
// — exactly the layout `_full` produces post-repack and exactly what
// `mvmr_cutlass_sm80_full_kernel` consumes via
// `make_stride(C_full, _1{})` (kernel device code UNCHANGED) — and
// feeds it straight to the launch. M_full / C_full are read from `aT`'s
// own shape (`aT.size(1)` / `aT.size(2)`); in `_full` they came from
// the 4-D `a` arg (`a.size(3)` / `a.size(2)`) — same two integers, just
// sourced from the pre-staged buffer instead of the pre-repack tensor.
//// The grad_b "transposed weight" case needs NO host transpose flag and
// NO host `.contiguous()`: the kernel only ever reads a
// (K, M_full, C_full) C-contiguous buffer, so fwd-vs-grad_b is purely a
// question of which weight axes the CALLER maps onto M_full / C_full
// when it stages. The fused Function stages
//   fwd:    weight.select(1,0).transpose(1,2).contiguous()  → (K, M_w, C_w)
//   grad_b: weight.select(1,0).contiguous()                 → (K, C_w, M_w)
// (i.e. for grad_b it simply drops the `.transpose(1,2)` it would have
// applied for fwd). Both are valid pre-staged (K, M_full, C_full)
// C-contiguous inputs to this one entry; no transposed-weight
// distinction reaches C++.
//// Everything else (input_b squeeze/contiguous, the dtype/dim/device
// TORCH_CHECKs, the M_full % TileM / C_full % TileK tile-multiple
// checks, seg_offs handling, the no-host-sentinel kernel-side
// clamped gather, the zero-init output, the cudaFuncSetAttribute +
// kernel launch) is replicated verbatim from `_full` so this is
// `_full`-minus-the-repack and nothing else. Kernel `.cuh` / device
// body untouched.
at::Tensor sparse_mvmr_cutlass_sm80_full_prestaged(
    at::Tensor aT,        // (n_o_k, M_full, C_full) fp16 C-contiguous —
                          //   the pre-staged weight (what `_full`'s
                          //   repack would have produced)
    at::Tensor b_idx,     // (T,) int — input-row idx
    at::Tensor input_b,   // (N_b, 1, C_full) or (N_b, C_full) fp16
    at::Tensor o_idx,     // (T,) int — output-row idx
    at::Tensor seg_offs,  // (n_o_k + 1,) int64 — per-k segment offsets
    int64_t n_o           // number of output points
) {
  TORCH_CHECK(aT.is_cuda() && input_b.is_cuda(),
      "sparse_mvmr_cutlass_sm80_full_prestaged: aT / input_b must be CUDA");
  TORCH_CHECK(b_idx.is_cuda() && o_idx.is_cuda() && seg_offs.is_cuda(),
      "sparse_mvmr_cutlass_sm80_full_prestaged: idx / seg_offs tensors must be CUDA");
  // fp16 OR bf16 (both operands same dtype); fp32 rejected.
  TORCH_CHECK(aT.scalar_type() == input_b.scalar_type(),
      "sparse_mvmr_cutlass_sm80_full_prestaged: aT and input_b must share dtype "
      "(got ", aT.scalar_type(), " and ", input_b.scalar_type(), ")");
  TORCH_CHECK(aT.scalar_type() == at::kHalf || aT.scalar_type() == at::kBFloat16,
      "sparse_mvmr_cutlass_sm80_full_prestaged: fp16/bf16 only (got ",
      aT.scalar_type(), ")");
  TORCH_CHECK(b_idx.scalar_type() == at::kInt && o_idx.scalar_type() == at::kInt,
      "sparse_mvmr_cutlass_sm80_full_prestaged: b_idx / o_idx must be int32");
  TORCH_CHECK(seg_offs.scalar_type() == at::kLong,
      "sparse_mvmr_cutlass_sm80_full_prestaged: seg_offs must be int64");
  TORCH_CHECK(b_idx.is_contiguous() && o_idx.is_contiguous() &&
              seg_offs.is_contiguous(),
      "sparse_mvmr_cutlass_sm80_full_prestaged: idx / seg_offs must be contiguous");
  // The repack-skip's load-bearing precondition: `aT` IS the
  // (n_o_k, M_full, C_full) C-contiguous buffer the kernel reads. `_full`
  // guaranteed this via its own `.contiguous()`; here the caller owns it.
  TORCH_CHECK(aT.dim() == 3,
      "sparse_mvmr_cutlass_sm80_full_prestaged: aT must be 3-D "
      "(n_o_k, M_full, C_full)");
  TORCH_CHECK(aT.is_contiguous(),
      "sparse_mvmr_cutlass_sm80_full_prestaged: aT must be C-contiguous "
      "(pre-staged; this entry skips the internal repack by contract)");

  at::Tensor inb_2d = input_b;
  if (inb_2d.dim() == 3) {
    TORCH_CHECK(inb_2d.size(1) == 1,
        "sparse_mvmr_cutlass_sm80_full_prestaged: input_b G dim must be 1");
    inb_2d = inb_2d.select(/*dim=*/1, /*index=*/0);
  }
  inb_2d = inb_2d.contiguous();
  TORCH_CHECK(inb_2d.dim() == 2,
      "sparse_mvmr_cutlass_sm80_full_prestaged: input_b must be 2-D after squeeze");

  constexpr int M_TILE = int(MvmrConfig::TileM::value);
  constexpr int C_TILE = int(MvmrConfig::TileK::value);

  // M_full / C_full come from the pre-staged buffer's own shape (in
  // `_full` they were a.size(3) / a.size(2) — the same two integers,
  // sourced from the pre-repack 4-D tensor instead).
  const int n_o_k = static_cast<int>(aT.size(0));
  const int M_full = static_cast<int>(aT.size(1));
  const int C_full = static_cast<int>(aT.size(2));

  TORCH_CHECK(static_cast<int>(inb_2d.size(1)) == C_full,
      "sparse_mvmr_cutlass_sm80_full_prestaged: input_b C (", inb_2d.size(1),
      ") must match aT's C_full (", C_full, ")");
  TORCH_CHECK(M_full % M_TILE == 0,
      "sparse_mvmr_cutlass_sm80_full_prestaged: M_full=", M_full,
      " must be a multiple of TileM=", M_TILE);
  TORCH_CHECK(C_full % C_TILE == 0,
      "sparse_mvmr_cutlass_sm80_full_prestaged: C_full=", C_full,
      " must be a multiple of TileK=", C_TILE);
  TORCH_CHECK(seg_offs.numel() == n_o_k + 1,
      "sparse_mvmr_cutlass_sm80_full_prestaged: seg_offs must have n_o_k+1 elements (got ",
      seg_offs.numel(), ", n_o_k=", n_o_k, ")");
  TORCH_CHECK(b_idx.numel() == o_idx.numel(),
      "sparse_mvmr_cutlass_sm80_full_prestaged: b_idx / o_idx length mismatch");

  // *** The ONLY difference vs `sparse_mvmr_cutlass_sm80_full` ***
  // `_full` here does:
  //   auto aT = a.select(1,0).transpose(1,2).contiguous();
  // `_prestaged` skips it — the caller's `aT` already IS that buffer.

  const int M_tiles = M_full / M_TILE;
  const int C_seg_padded = C_full;   // C_full is a TileK multiple

  auto options_o = at::TensorOptions().dtype(torch::kFloat32).device(aT.device());
  auto o = torch::zeros({n_o, 1, M_full}, options_o);

  // same dtype-dispatched launch as `_full` (the only
  // difference between the two entries is the weight-staging above).
  dispatch_mvmr_full(aT, b_idx, o_idx, seg_offs, inb_2d, o,
                     M_full, C_full, C_seg_padded, M_tiles, n_o_k);

  return o;
}

TORCH_LIBRARY_IMPL(sparse_engines_cuda, CUDA, m) {
  m.impl("sparse_mvmr_cutlass_sm80_single_tile",
         &sparse_mvmr_cutlass_sm80_single_tile);
  m.impl("sparse_mvmr_cutlass_sm80_single_tile_gathered",
         &sparse_mvmr_cutlass_sm80_single_tile_gathered);
  m.impl("sparse_mvmr_cutlass_sm80_full",
         &sparse_mvmr_cutlass_sm80_full);
  m.impl("sparse_mvmr_cutlass_sm80_full_prestaged",
         &sparse_mvmr_cutlass_sm80_full_prestaged);
}

} // namespace sparse_engines_cuda
