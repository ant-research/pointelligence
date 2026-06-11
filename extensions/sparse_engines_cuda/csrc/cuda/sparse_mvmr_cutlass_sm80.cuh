#ifndef SPARSE_ENGINES_CUDA_SPARSE_MVMR_CUTLASS_SM80_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_MVMR_CUTLASS_SM80_CUH

// ─── Tier-2 CUTLASS skeleton for mvmr (affine) ──────────────────────────────
//// Single-tile (M_TILE, S_TILE, K=C_seg) GEMM via CUTLASS 3.x CuTe +
// CollectiveMma<MainloopSm80CpAsyncUnpredicated<PIPE>, ...>. This is the
// *skeleton*: affine layouts only, no gather, no scatter. The structure
// is COPIED from the proven vvor single-tile template
// (sparse_vvor_cutlass_sm80.{cuh,cu}); the only semantic change is the
// contraction axis.
//// ── mvmr vs vvor (the one structural difference, gotten exactly right) ──
////   vvor (the template) contracts over seg_len (K):
//       grad_w[k] (M,C) = Σ_{t∈seg k} gradout[i[t]] (M-vec) ⊗ input[j[t]] (C-vec)
////   mvmr (this op) contracts over CHANNELS (C):
//       o[a_idx[t]] (M-vec) += W[k] (M,C) @ input[b_idx[t]] (C-vec),
//       summed over triplets t in segment k.
////   Authoritative reference: sparse_engines/mvmr_triton.py ::
//   sparse_matrix_vector_multiplication_reduction. The grouped kernel's
//   core GEMM is  out (L, M) = block_b (L, C) @ block_a (C, M),  i.e. it
//   contracts over C. W[k] is **affine/dense** (indexed by segment id k
//   only — NOT gathered). input is gathered by b_idx (the gathered op's
//   problem). Output is scatter-accumulated by a_idx (the full op's
//   problem). The single-tile op exercises ONLY the contraction-axis-C
//   GEMM core with a dense-write epilogue.
//// Single-tile contract (the vvor single-tile analog, contraction = C):
//   A (weight tile)  : (M_TILE, C_seg) row-major  → strides (C_seg, 1)
//   B (input tile)   : (S_TILE, C_seg) row-major  → strides (C_seg, 1)
//   C (output tile)  : (M_TILE, S_TILE) row-major → strides (S_TILE, 1)
//       C[m, s] = Σ_c A[m, c] * B[s, c]
// Both A and B have C (contraction) as the inner contiguous dim — the
// host wrapper does `index_select(...).t().contiguous()` to produce
// this memory layout (single index_select + view, cheap), exactly as
// the vvor single-tile wrapper does for its K axis.
//// CRITICAL inherited lesson (the +1,+1-shift bug seen in vvor):
// `GmemTiledCopyA/B` thread×val layout MUST be
//   Layout<Shape<_32,_4>,Stride<_4,_1>>{} , Layout<Shape<_1,_8>>{}
// → Tiler_MN = (TileM/2, TileK). The `(_16,_8)×(_1,_8)` form produces
// Tiler_MN=(16,64), overshoots TileK=32 by 2×, and threads with K_idx≥4
// clobber threads K_idx<4's smem writes. We use the `(_32,_4)` form
// verbatim from the vvor .cuh (see its comment block lines 80–94).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cutlass/numeric_types.h>   // cutlass::half_t / cutlass::bfloat16_t
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>

namespace sparse_engines_cuda {

using namespace cute;

// ── element→MMA-atom selector (mvmr TU) ────────────────────
// Re-declared in this translation unit (the vvor .cuh's `Sm80MmaAtomFor` is
// not visible here; both compile independently). half_t → fp16 HMMA atom
// (byte-identical to the prior hardcode), bfloat16_t → bf16 sibling
// (same tile shape / fp32 accumulator). No fp32 SM80 TC atom of this shape.
template <class Element> struct Sm80MvmrMmaAtomFor;
template <> struct Sm80MvmrMmaAtomFor<cutlass::half_t> {
  using type = SM80_16x8x16_F32F16F16F32_TN;
};
template <> struct Sm80MvmrMmaAtomFor<cutlass::bfloat16_t> {
  using type = SM80_16x8x16_F32BF16BF16F32_TN;
};

// Per-instantiation tile config. Mirrors VvorCutlassSm80FpropConfig
// verbatim — the GEMM shape is identical (single (M_TILE, S_TILE) tile,
// contraction axis tiled by TileK). At enc4 fp16 (M=C=512), TileM=64,
// TileN(=S_TILE)=64, TileK(=C-tile)=32.
template <class Element_ = cutlass::half_t>
struct MvmrCutlassSm80FpropConfigT {
  using TileM = _64;   // output M (weight rows / o channels)
  using TileN = _64;   // S_TILE (input-triplet rows for this tile)
  using TileK = _32;   // C-tile along the channel contraction axis
  using PIPE  = _3;    // 3-stage cp.async pipeline (matches example 59)

  using ElementA   = Element_;              // weight element (fp16/bf16)
  using ElementB   = Element_;              // input element
  using ElementAcc = float;                 // fp32 accumulator
  using ElementC   = float;                 // fp32 output (o tile)

  using TileShape  = Shape<TileM, TileN, TileK>;

  // SM80 16-bit-input fp32-accum HMMA atom (m16n8k16), selected from the
  // element type (half_t → F32F16F16F32, bfloat16_t → F32BF16BF16F32). The
  // fp16 instantiation is byte-identical to the prior hardcode.
  using TiledMma = TiledMMA<
      MMA_Atom<typename Sm80MvmrMmaAtomFor<Element_>::type>,
      Layout<Shape<_2, _2, _1>>,      // 2x2 warps -> 4 warps = 128 threads
      Tile<_32, _32, Underscore>>;    // pad warp tile to (32, 32, K)

  static constexpr int MaxThreadsPerBlock     = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int Stages                 = PIPE::value;

  // ── Gmem tiled-copy for A (weight tile, C-major) ─────────────────────
  // A is (M_TILE, C_TILE) row-major in memory -- C is the leading dim
  // (contiguous). The (_32,_4)×(_1,_8) form is the validated layout
  // (see the .cuh header note on the +1,+1-shift bug). Do NOT change to
  // (_16,_8) — that form overshoots TileK and clobbers smem.
  using GmemTiledCopyA = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementA>{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},   // 32 thr M, 4 thr C
      Layout<Shape<_1, _8>>{}));                  // 8×fp16 = 128b per thread

  using GmemTiledCopyB = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementB>{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _8>>{}));

  // ── Shared-memory layouts (C-contig, Swizzle<3,3,3> for fp16) ────────
  using SmemLayoutAtomA = decltype(
      composition(Swizzle<3, 3, 3>{},
                  Layout<Shape<_8, _32>,
                         Stride<_32, _1>>{}));   // 8x32, C-major (C=stride 1)
  using SmemCopyAtomA   = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;

  using SmemLayoutAtomB = decltype(
      composition(Swizzle<3, 3, 3>{},
                  Layout<Shape<_8, _32>,
                         Stride<_32, _1>>{}));
  using SmemCopyAtomB   = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;

  // ── Epilogue smem layout (M_TILE, S_TILE) fp32, row-major (S-contig) ─
  using SmemLayoutOut = Layout<Shape<TileM, TileN>,
                               Stride<TileN, _1>>;

  // 128 threads cooperatively store the (64, 64) fp32 tile (16B vectors).
  using GmemTiledCopyC = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cute::uint128_t>, ElementC>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _4>, Stride<_4, _1>>{}));
  using SmemCopyAtomC = Copy_Atom<UniversalCopy<cute::uint32_t>, ElementC>;

  // Per-CTA shared storage. mainloop carries A + B with PIPE stages;
  // epilogue overwrites the same buffer with the fp32 output tile.
  union SharedStorage {
    struct {
      ElementA sA[size(TileM{}) * size(TileK{}) * size(PIPE{})];
      ElementB sB[size(TileN{}) * size(TileK{}) * size(PIPE{})];
    } mainloop;

    struct {
      ElementC sC[size(TileM{}) * size(TileN{})];
    } epilogue;
  };
};

// fp16 alias = the exact prior type (byte-identical codegen); bf16
// is the new bf16 instantiation. The sm_90 mvmr config aliases this Fprop
// config (sparse_mvmr_cutlass_sm90.cuh), so both dtypes flow to sm_90 too.
using MvmrCutlassSm80FpropConfig     = MvmrCutlassSm80FpropConfigT<cutlass::half_t>;
using MvmrCutlassSm80FpropConfigBf16 = MvmrCutlassSm80FpropConfigT<cutlass::bfloat16_t>;

// ── Kernel body ─────────────────────────────────────────────────────────
//// Operates on already-tile-shaped gmem tensors:
//   gA : (M_TILE, C_seg) row-major fp16, C-contiguous  (weight tile)
//   gB : (S_TILE, C_seg) row-major fp16, C-contiguous  (input tile)
//   gC : (M_TILE, S_TILE) row-major fp32 (zero-initialized by host)
//// Single-CTA launch. C_seg is the (padded) channel contraction length.
// Structurally identical to VvorCutlassSm80SingleTileOp — only the
// operand semantics (contraction over C, affine weight) differ; the
// CUTLASS plumbing is unchanged.
template <class Config>
struct MvmrCutlassSm80SingleTileOp {
  using TileM     = typename Config::TileM;
  using TileN     = typename Config::TileN;
  using TileK     = typename Config::TileK;
  using PIPE      = typename Config::PIPE;
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementC  = typename Config::ElementC;
  using TileShape = typename Config::TileShape;
  using TiledMma  = typename Config::TiledMma;
  using SharedStorage = typename Config::SharedStorage;

  template <class TensorA, class TensorB, class TensorC>
  __device__ void
  operator()(TensorA gA_full,    // (M_TILE, C_seg)
             TensorB gB_full,    // (S_TILE, C_seg)
             TensorC gC,         // (M_TILE, S_TILE)
             int     k_tile_count,
             char*   smem_buf) const {
    using namespace cute;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::MainloopSm80CpAsyncUnpredicated<PIPE::value>,
        TileShape,
        ElementA,
        Underscore,    // ignore stride; pass full cute::Tensor instead
        ElementB,
        Underscore,
        TiledMma,
        typename Config::GmemTiledCopyA,
        typename Config::SmemLayoutAtomA,
        typename Config::SmemCopyAtomA,
        cute::identity,
        typename Config::GmemTiledCopyB,
        typename Config::SmemLayoutAtomB,
        typename Config::SmemCopyAtomB,
        cute::identity>;

    TiledMma tiled_mma;
    auto accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    clear(accum);

    // Tile gA / gB into C-tiles of width TileK. Single-(M, S) tile case:
    // M_TILE and S_TILE already match TileM/TileN exactly, so there's
    // only one (M, S) tile but k_tile_count C-tiles.
    auto gA = local_tile(gA_full, make_tile(TileM{}, TileK{}), make_coord(0, _));  // (TileM, TileK, c')
    auto gB = local_tile(gB_full, make_tile(TileN{}, TileK{}), make_coord(0, _));  // (TileN, TileK, c')

    auto k_tile_iter = cute::make_coord_iterator(k_tile_count);

    CollectiveMainloop collective_mma;
    collective_mma(
        accum,
        gA,
        gB,
        accum,
        k_tile_iter, k_tile_count,
        Underscore{},   // no residue predication for the single-tile case
        threadIdx.x,
        smem_buf);

    // ── Epilogue: dense write of the accumulator through smem to gmem ──
    // (Dense write; the scatter-accumulate by a_idx is the full op's problem.)
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sC = make_tensor(make_smem_ptr(&storage.epilogue.sC[0]),
                            typename Config::SmemLayoutOut{});

    auto smem_tiled_copy_C = make_tiled_copy_C(typename Config::SmemCopyAtomC{}, tiled_mma);
    auto smem_thr_copy_C   = smem_tiled_copy_C.get_slice(threadIdx.x);
    auto tCrC              = smem_thr_copy_C.retile_S(accum);
    auto tCsC              = smem_thr_copy_C.partition_D(sC);
    copy(smem_tiled_copy_C, tCrC, tCsC);

    __syncthreads();

    typename Config::GmemTiledCopyC gmem_tiled_copy_C;
    auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(threadIdx.x);
    auto tDsC            = gmem_thr_copy_C.partition_S(sC);
    auto tDgC            = gmem_thr_copy_C.partition_D(gC);
    copy(gmem_tiled_copy_C, tDsC, tDgC);
  }
};

// ── Full mvmr op — ragged K-segment grid + scatter epilogue ─────────────────
//// Drop-in replacement for sparse_matrix_vector_multiplication_reduction
// (the Triton-grouped mvmr). One CTA per (m-tile, k-segment); the CTA
// loops over S-chunks of its segment internally, running one
// (M_TILE, S_TILE, C) GEMM per chunk and scatter-accumulating each
// result column into o[o_idx[t]] via atomicAdd.
//// ── The two genuinely-new surfaces vs the single-tile/gathered ops ──
////  1. Ragged grid + per-segment S-chunk loop. mvmr's segment length is
//     the GEMM's *N/S* dimension (NOT the contraction — that is C, fixed
//     and tiled by TileK inside the mainloop). A long segment therefore
//     needs ceil(seg_len / S_TILE) separate GEMMs, each over a different
//     S-chunk of the segment's triplet slice. The CTA walks them in a
//     loop; an empty segment (seg_len == 0) does zero iterations and
//     returns (o was zero-initialized by the host). No host-side chunk
//     table / no extra sync — exactly the grouped_mma.cuh structure
//     (one worker per (k, ...) that loops triplets internally), lifted
//     onto the CUTLASS GEMM core.
////  2. The scatter-accumulate epilogue. vvor wrote a *dense* grad_w[k]
//     tile (output indexed by k directly). mvmr scatters into
//     o[o_idx[t]] — output rows selected per triplet, many triplets per
//     row. We lift the proven epilogue from
//     sparse_mvmr_grouped_mma.cuh verbatim in spirit: per output-M one
//     thread holds an fp32 register accumulator `o_acc`, walks the
//     chunk's S columns, and emits an `atomicAdd(o_ptr + …, o_acc)` only
//     at an output-index boundary within the chunk (the `prev_out`
//     run-length coalescing trick), accumulating consecutive same-o_idx
//     columns in-register first. The op output buffer is fp32
//     (mvmr_triton.py:128) so the fp32-atomicAdd precondition holds.
//// Padded S-slots (the last partial S-tile, slots s ≥ chunk_len) clamp
// (SegmentClampedGather kernel-side virtual zero) to in-bounds real input
// row 0 — NO host-appended sentinel row. The scatter loop stops at
// chunk_len, so an OOB column's GEMM value is computed but never emitted.
// OOB contribution is therefore exactly 0 by structural exclusion (a
// host-zero-row only ever zeroed these same unused columns; the per-call
// host (N_b+1,W) alloc/copy/zero_ is dropped). No garbage scattered.
//// W stays affine: the host pre-transposes a (K,1,C,M) → aT (K,M,C)
// C-contiguous once (the mvmr analog of vvor's pre-gather staging), so
// the W[k] (M_TILE, C) tile is C-contiguous exactly like the single-tile
// A and the single-tile MvmrCutlassSm80FpropConfig + the IndexedGather-on-S
// B view compose directly — no new Config.
template <class Config>
struct MvmrCutlassSm80FullOp {
  using TileM     = typename Config::TileM;
  using TileN     = typename Config::TileN;   // == S_TILE
  using TileK     = typename Config::TileK;   // == C_TILE (contraction)
  using PIPE      = typename Config::PIPE;
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementC  = typename Config::ElementC;
  using TileShape = typename Config::TileShape;
  using TiledMma  = typename Config::TiledMma;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm80CpAsyncUnpredicated<PIPE::value>,
      TileShape,
      ElementA,
      Underscore,
      ElementB,
      Underscore,
      TiledMma,
      typename Config::GmemTiledCopyA,
      typename Config::SmemLayoutAtomA,
      typename Config::SmemCopyAtomA,
      cute::identity,
      typename Config::GmemTiledCopyB,
      typename Config::SmemLayoutAtomB,
      typename Config::SmemCopyAtomB,
      cute::identity>;

  // smem holds the collective's SharedStorage during the mainloop, then
  // the (TileM, TileN) fp32 tile during the scatter epilogue. Disjoint
  // in time → size to the max (mirrors VvorCutlassSm80FullOp).
  static constexpr size_t kMainloopSmem =
      sizeof(typename CollectiveMainloop::SharedStorage);
  static constexpr size_t kEpilogueSmem =
      sizeof(ElementC) * size_t(size(TileM{})) * size_t(size(TileN{}));
  static constexpr size_t kSmemBytes =
      kMainloopSmem > kEpilogueSmem ? kMainloopSmem : kEpilogueSmem;

  // Run ONE (M_TILE, S_TILE, C) GEMM for a single S-chunk:
  //   gW : (TileM, C_seg_padded)  affine C-contig  (W[k] m-tile slice)
  //   gB : (TileN, C_seg_padded)  S-gathered, C-contig (input chunk)
  // and scatter-accumulate the result columns into o via atomicAdd.
  //  //   o_idx_chunk : pointer into o_idx at the chunk's first triplet
  //   chunk_len   : real triplet count in this chunk (≤ TileN)
  //   o_ptr       : (n_o, M_full) fp32 output (G=1 squeezed)
  //   m_start     : this CTA's M-tile origin
  //   M_full      : output row stride
  template <class TensorW, class TensorB>
  __device__ void
  run_chunk(TensorW gW,
            TensorB gB,
            int     C_seg_padded,
            const int32_t* __restrict__ o_idx_chunk,
            int     chunk_len,
            float*  __restrict__ o_ptr,
            int     m_start,
            int     M_full,
            char*   smem_buf) const {
    using namespace cute;

    TiledMma tiled_mma;
    auto accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    clear(accum);

    auto gWt = local_tile(gW, make_tile(TileM{}, TileK{}), make_coord(0, _));
    auto gBt = local_tile(gB, make_tile(TileN{}, TileK{}), make_coord(0, _));

    int k_tile_count = C_seg_padded / int(TileK::value);
    if (k_tile_count > 0) {
      auto k_tile_iter = cute::make_coord_iterator(k_tile_count);
      CollectiveMainloop collective_mma;
      collective_mma(
          accum,
          gWt,
          gBt,
          accum,
          k_tile_iter, k_tile_count,
          Underscore{},
          threadIdx.x,
          smem_buf);
    }

    // ── Stream accum → smem as a dense (TileM, TileN) fp32 tile ──────────
    // SmemLayoutOut is M-major Stride<TileN,_1> ⇒ sC(m, s) addressable.
    Tensor sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementC*>(smem_buf)),
                            typename Config::SmemLayoutOut{});
    auto smem_tiled_copy_C = make_tiled_copy_C(typename Config::SmemCopyAtomC{}, tiled_mma);
    auto smem_thr_copy_C   = smem_tiled_copy_C.get_slice(threadIdx.x);
    auto tCrC              = smem_thr_copy_C.retile_S(accum);
    auto tCsC              = smem_thr_copy_C.partition_D(sC);

    __syncthreads();   // mainloop done with smem; safe to reuse for sC
    copy(smem_tiled_copy_C, tCrC, tCsC);
    __syncthreads();

    // ── Scatter-accumulate epilogue (lifted from grouped_mma.cuh) ───────
    // One thread per output-M row owns an fp32 register accumulator and
    // walks the chunk's S columns; emit an atomicAdd only at an output-
    // index boundary (prev_out run-length coalescing). The sC tile is
    // M-major: sC[m, s] = O[m_start+m, s] for triplet (chunk first + s).
    constexpr int M_TILE_I = int(TileM::value);
    constexpr int S_TILE_I = int(TileN::value);
    ElementC* sC_raw = reinterpret_cast<ElementC*>(smem_buf);

    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    // M_TILE_I (=64) ≤ nthr (=128): threads [0, M_TILE_I) each own one m.
    for (int m = tid; m < M_TILE_I; m += nthr) {
      const int out_m = m_start + m;
      int   prev_out = -1;
      float o_acc    = 0.0f;
#pragma unroll 1
      for (int s = 0; s < chunk_len && s < S_TILE_I; ++s) {
        const int out_idx = o_idx_chunk[s];
        if (out_idx != prev_out && prev_out >= 0) {
          atomicAdd(o_ptr + static_cast<int64_t>(prev_out) * M_full + out_m,
                    o_acc);
          o_acc = 0.0f;
        }
        // sC is M-major Stride<TileN,_1> ⇒ element (m, s) at m*S_TILE + s.
        o_acc   += sC_raw[m * S_TILE_I + s];
        prev_out = out_idx;
      }
      if (prev_out >= 0) {
        atomicAdd(o_ptr + static_cast<int64_t>(prev_out) * M_full + out_m,
                  o_acc);
      }
    }
    __syncthreads();   // sC reused next chunk iteration
  }
};

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_MVMR_CUTLASS_SM80_CUH
