#ifndef SPARSE_ENGINES_CUDA_SPARSE_VVOR_CUTLASS_SM80_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_VVOR_CUTLASS_SM80_CUH

// ─── Cycle-4 §1.11 G14 — Tier-2 CUTLASS skeleton for vvor (Task 1, affine) ──
//
// Single-tile (M_TILE, N_TILE, K=seg_len) GEMM via CUTLASS 3.x
// CuTe + CollectiveMma<MainloopSm80CpAsyncUnpredicated<PIPE>, ...>. This
// is the *skeleton* per pre-reg §6 day-3 GO/NO-GO: affine layouts only,
// the segment's grad_output / input rows are PRE-GATHERED on the Python
// side into contiguous (M_TILE, K_seg) and (N_TILE, K_seg) buffers.
//
// Followups (Task 2+): replace the affine A/B layouts with
// `make_gather_tensor(..., IndexedGather{idx_ptr})` so the gather
// happens during cp.async load instead of as a separate Python step.
//
// Layouts:
//   A (grad_out segment): (M_TILE, K_seg) row-major  → strides (K_seg, 1)
//   B (input    segment): (N_TILE, K_seg) row-major  → strides (K_seg, 1)
//   C (grad_weight tile): (M_TILE, N_TILE) row-major → strides (N_TILE, 1)
//
// Both A and B have K (contraction) as the inner contiguous dim — matches
// example 59's "filter" + "activation" layouts (K = inner). The host-side
// Python wrapper does `index_select(...).t().contiguous()` to produce this
// memory layout; that's a single index_select + view, cheap.

#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>

namespace sparse_engines_cuda {

using namespace cute;

// Per-instantiation tile config. M_TILE / N_TILE picked to match
// example-59-style 128-element major-mode + 32-element minor-mode tile.
// For Task 1 we use a single (M_TILE, N_TILE) tile per launch — the outer
// (k, mt, ct) grid is added in Task 3.
//
// At enc4 fp16 (M=C=512), M_TILE=64, N_TILE=64 gives Mt=Ct=8 — but Task 1
// computes a single tile only, so we have the Python wrapper slice one
// (m_start..m_start+M_TILE, c_start..c_start+N_TILE) face of grad_weight.
struct VvorCutlassSm80FpropConfig {
  using TileM = _64;
  using TileN = _64;
  using TileK = _32;   // K-tile along seg_len contraction axis
  using PIPE  = _3;    // 3-stage cp.async pipeline (matches example 59)

  using ElementA   = cutlass::half_t;       // grad_output element
  using ElementB   = cutlass::half_t;       // input element
  using ElementAcc = float;                 // fp32 accumulator
  using ElementC   = float;                 // fp32 output (grad_weight tile)

  using TileShape  = Shape<TileM, TileN, TileK>;

  // SM80 fp16-input fp32-accum HMMA atom. m16n8k16 is the canonical fp16
  // tensor-core atom on sm_80/sm_89.
  using TiledMma = TiledMMA<
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      Layout<Shape<_2, _2, _1>>,      // 2x2 warps -> 4 warps = 128 threads
      Tile<_32, _32, Underscore>>;    // pad warp tile to (32, 32, K)

  static constexpr int MaxThreadsPerBlock     = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int Stages                 = PIPE::value;

  // ── Gmem tiled-copy for A (filter-shaped, K-major) ───────────────────
  // A is (M_TILE, K_TILE) row-major in memory -- K is the leading dim
  // (contiguous). 32 threads loading 4 fp16 each = 128b vector load.
  // Layout: 8 threads along M-axis, 4 threads along K-axis (within one
  // 32-thread row), vector size 4 along K. That covers (8, 4*4)=(8, 16)
  // elements per row; tile loop walks the M-mode.
  // Match example 59 / vendored gather example pattern.
  // 32 thr along M × 4 thr along K × 8-element K-vector per thread =
  // (TileM=64, TileK=32) coverage in 2 passes along M. Earlier code
  // used (16, 8) threads × val=8 → Tiler_MN=(16, 64), which overshoots
  // TileK=32 by 2× and produces the cycle-4 (+1, +1) shift bug because
  // threads with K_idx≥4 clobber threads K_idx<4's smem writes.
  using GmemTiledCopyA = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementA>{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},   // 32 thr M, 4 thr K
      Layout<Shape<_1, _8>>{}));                  // 8×fp16 = 128b per thread

  using GmemTiledCopyB = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementB>{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _8>>{}));

  // ── Shared-memory layouts ────────────────────────────────────────────
  // Use the same SmemLayoutAtom pattern as example 59 (Swizzle<3,3,3>
  // for fp16; the 8x16 tile + swizzle gives bank-conflict-free smem
  // access for ldmatrix). For TileM=64, TileK=32, the atom tiles into
  // (4 atoms in M) x (2 atoms in K).
  using SmemLayoutAtomA = decltype(
      composition(Swizzle<3, 3, 3>{},
                  Layout<Shape<_8, _32>,
                         Stride<_32, _1>>{}));   // 8x32, K-major (K=stride 1)
  using SmemCopyAtomA   = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;

  using SmemLayoutAtomB = decltype(
      composition(Swizzle<3, 3, 3>{},
                  Layout<Shape<_8, _32>,
                         Stride<_32, _1>>{}));
  using SmemCopyAtomB   = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;

  // ── Epilogue smem layout (M_TILE, N_TILE) fp32, row-major (N-contig) ─
  // Default `Layout<Shape<...>>` is column-major; we want N-contiguous to
  // match the gmem grad_weight tile layout (row-major). Explicit strides.
  using SmemLayoutOut = Layout<Shape<TileM, TileN>,
                               Stride<TileN, _1>>;

  // 128 threads cooperatively store the (64, 64) fp32 tile. Each thread
  // writes 4 fp32 per row (uint128_t = 16B = 4 fp32). 8 threads along N
  // within a row, 16 along M; per pass covers 16 rows × 32 cols. 8 passes
  // total for (64, 64) = 4096 fp32. Explicit value-layout strides
  // (Stride<_4, _1>) so CuTe picks the **N axis** (mode-1) as the
  // contiguous vectorized dim — N is stride 1 in SmemLayoutOut.
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

// ── Kernel body ─────────────────────────────────────────────────────────
//
// Operates on already-tile-shaped gmem tensors:
//   gA : (M_TILE, K_seg) row-major fp16, K-contiguous
//   gB : (N_TILE, K_seg) row-major fp16, K-contiguous
//   gC : (M_TILE, N_TILE) row-major fp32 (zero-initialized by host)
//
// Single-CTA launch. K_seg is dynamic (= segment length, varies per k).
template <class Config>
struct VvorCutlassSm80SingleTileOp {
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
  operator()(TensorA gA_full,    // (M_TILE, K_seg)
             TensorB gB_full,    // (N_TILE, K_seg)
             TensorC gC,         // (M_TILE, N_TILE)
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

    // Tile gA / gB into K-tiles of width TileK. Single-(M, N) tile case:
    // M_TILE and N_TILE already match TileM/TileN exactly, so there's
    // only one (M, N) tile but k_tile_count K-tiles.
    //
    // local_tile produces shape (TileM, TileK, k_tile_count) for gA.
    auto gA = local_tile(gA_full, make_tile(TileM{}, TileK{}), make_coord(0, _));  // (TileM, TileK, k')
    auto gB = local_tile(gB_full, make_tile(TileN{}, TileK{}), make_coord(0, _));  // (TileN, TileK, k')

    auto k_tile_iter = cute::make_coord_iterator(k_tile_count);

    CollectiveMainloop collective_mma;
    collective_mma(
        accum,
        gA,
        gB,
        accum,
        k_tile_iter, k_tile_count,
        Underscore{},   // no residue predication for Task 1
        threadIdx.x,
        smem_buf);

    // ── Epilogue: stream the accumulator through smem to gmem ───────────
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

// ── Task 2 config: M/N-contig smem + ldmatrix-T (for K-mode gather) ─────────
//
// Task 1 has K-contig smem because A/B are stored with K as the contiguous
// mode (Python pre-gather produces K-major buffers). For Task 2 we read
// directly from grad_output (N_o, M_full) row-major + input (N_b, C_full)
// row-major, so the M and N axes are gmem-contig in source. Composed
// IndexedGather makes the K axis a per-K-element gather; loads along the
// K axis are NOT contig and cannot use 128b cp.async.
//
// Resolution: vectorize cp.async along M (resp. N), which IS gmem-contig,
// and make smem M/N-major (the non-gathered axis is the inner smem mode).
// The SM80 m16n8k16 fp16 MMA atom still needs K-contig fragments in
// registers, so we use ldmatrix-T (SM75_U16x8_LDSM_T) to transpose during
// smem→reg load — same trick the canonical sm_80 NT example uses.
//
// All other Task 1 settings (TileM/N/K, TiledMma, PIPE, ElementA/B/C,
// SmemLayoutOut, GmemTiledCopyC, SmemCopyAtomC) are inherited unchanged.
struct VvorCutlassSm80GatherConfig {
  using TileM = _64;
  using TileN = _64;
  using TileK = _32;
  using PIPE  = _3;

  using ElementA   = cutlass::half_t;
  using ElementB   = cutlass::half_t;
  using ElementAcc = float;
  using ElementC   = float;

  using TileShape  = Shape<TileM, TileN, TileK>;

  using TiledMma = TiledMMA<
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      Layout<Shape<_2, _2, _1>>,
      Tile<_32, _32, Underscore>>;

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int Stages = PIPE::value;

  // ── Gmem tiled-copy for A (M-major, K-gathered) ──────────────────────
  // A view shape  = (TileM=64, TileK=32). gmem stride = (1, M_full × gather).
  // M is the contig mode (stride 1) → vectorize cp.async along M with
  // 8-fp16 (uint128_t) vectors.
  // Per thread: (V_M=8, V_K=1). Thread layout 8×16 covers full (64, 16);
  // 2 passes along K cover TileK=32. Total threads = 8 × 16 = 128.
  using GmemTiledCopyA = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementA>{},
      Layout<Shape<_8, _16>, Stride<_1, _8>>{},  // 8 thr M (stride 1 in tid),
                                                  // 16 thr K (stride 8 in tid)
      Layout<Shape<_8, _1>>{}));                  // V_M=8, V_K=1

  using GmemTiledCopyB = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementB>{},
      Layout<Shape<_8, _16>, Stride<_1, _8>>{},
      Layout<Shape<_8, _1>>{}));

  // ── Shared-memory layouts (M-major / N-major) ────────────────────────
  // Atom shape (8 elements of inner mode = M, 8 elements of outer mode = K).
  // M is the stride-1 mode → smem is M-major. Swizzle keeps the 16-element
  // ldmatrix-T fetch bank-conflict-free.
  using SmemLayoutAtomA = decltype(
      composition(Swizzle<3, 3, 3>{},
                  Layout<Shape<_32, _8>,
                         Stride<_1, _32>>{}));  // M-major: M=stride 1, K=stride 32
  using SmemCopyAtomA   = Copy_Atom<SM75_U16x8_LDSM_T, ElementA>;

  using SmemLayoutAtomB = decltype(
      composition(Swizzle<3, 3, 3>{},
                  Layout<Shape<_32, _8>,
                         Stride<_1, _32>>{}));
  using SmemCopyAtomB   = Copy_Atom<SM75_U16x8_LDSM_T, ElementB>;

  // Epilogue — reuse Task 1's M-major Stride-(TileN, 1) output layout.
  using SmemLayoutOut = Layout<Shape<TileM, TileN>,
                               Stride<TileN, _1>>;

  using GmemTiledCopyC = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cute::uint128_t>, ElementC>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _4>, Stride<_4, _1>>{}));
  using SmemCopyAtomC = Copy_Atom<UniversalCopy<cute::uint32_t>, ElementC>;

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

// ── Task 3: full-vvor op — Unpredicated mainloop + sentinel-zero-row ────────
//
// PADDED-SLOT RESOLUTION: **Option 3 (sentinel-zero-row)** from the design
// doc / pre-reg. Option 1 (K-axis predicated `MainloopSm80CpAsync`) was
// attempted first and is the documented R1 blocker: the predicated mainloop
// does an *in-place* `gA = cute::domain_offset(make_coord(0, k_residue, 0),
// gA)` (sm80_mma_multistage.hpp:500). For a `ComposedLayout` gather tensor
// `domain_offset` returns a layout whose `Offset` TYPE changes
// (`offset() + layout_b()(coord)`: ArithmeticTuple<C<0>,C<0>> →
// non-constant) so the `operator=` back into `gA` does not type-check
// ("no operator = matches these operands"). Templating the existing
// SingleTileOp on the predicated mainloop without a CUTLASS-internal
// rewrite is therefore infeasible inside the task budget → fall back to
// Option 3 per the pre-reg's explicit ">1h ⇒ Option 2/3" guidance.
//
// Option 3 keeps the proven Task-2 `MainloopSm80CpAsyncUnpredicated` +
// IndexedGather path verbatim. The only addition is a gather index
// functor (`SegmentClampedGather`, defined in the .cu) that returns a
// guaranteed-zero SENTINEL ROW for any K slot k ≥ seg_len. The host
// appends one all-zero row to grad_output (row index N_o) and input
// (row index N_b); a clamped padded slot indexes that row, so its
// outer-product contribution is exactly 0 — no garbage injected, no
// symmetric-padding trick, works for arbitrary / empty seg_len. Cost:
// one extra gmem row per a/b (≤ M_full + C_full fp16 = a few KB).
//
// One CTA computes one (k, mt, ct) tile. K_seg_padded is rounded up to a
// TileK multiple by the host; the Unpredicated mainloop sees a clean
// K-tile count. Empty segments (seg_len == 0) → all K slots clamp to the
// sentinel zero row → the (TileM, TileN) tile is exactly zero (matches
// the reference, which has no triplets to sum for that k).
template <class Config>
struct VvorCutlassSm80FullOp {
  using TileM     = typename Config::TileM;
  using TileN     = typename Config::TileN;
  using TileK     = typename Config::TileK;
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

  // smem holds the collective's own SharedStorage during the mainloop,
  // then the (TileM, TileN) fp32 tile during the epilogue. They are
  // disjoint in time → size the dynamic allocation to the max.
  static constexpr size_t kMainloopSmem =
      sizeof(typename CollectiveMainloop::SharedStorage);
  static constexpr size_t kEpilogueSmem =
      sizeof(ElementC) * size_t(size(TileM{})) * size_t(size(TileN{}));
  static constexpr size_t kSmemBytes =
      kMainloopSmem > kEpilogueSmem ? kMainloopSmem : kEpilogueSmem;

  // gA_full / gB_full : (TileM, K_seg_padded) / (TileN, K_seg_padded).
  // K_seg_padded is a TileK multiple; padded K slots already clamp to the
  // sentinel zero row via the gather functor. gC : (TileM,TileN) fp32.
  template <class TensorA, class TensorB, class TensorC>
  __device__ void
  operator()(TensorA gA_full,
             TensorB gB_full,
             TensorC gC,
             int     K_seg_padded,
             char*   smem_buf) const {
    using namespace cute;

    TiledMma tiled_mma;
    auto accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    clear(accum);

    auto gA = local_tile(gA_full, make_tile(TileM{}, TileK{}), make_coord(0, _));  // (TileM, TileK, k')
    auto gB = local_tile(gB_full, make_tile(TileN{}, TileK{}), make_coord(0, _));  // (TileN, TileK, k')

    int k_tile_count = K_seg_padded / int(TileK::value);

    if (k_tile_count > 0) {
      auto k_tile_iter = cute::make_coord_iterator(k_tile_count);
      CollectiveMainloop collective_mma;
      collective_mma(
          accum,
          gA,
          gB,
          accum,
          k_tile_iter, k_tile_count,
          Underscore{},   // Unpredicated — sentinel-zero-row handles tail
          threadIdx.x,
          smem_buf);
    }
    // k_tile_count == 0 only if K_seg_padded == 0 (no triplets AND no
    // pad) → accum stays zero-cleared; epilogue writes a zero tile.

    // ── Epilogue: stream the accumulator through smem to gmem ───────────
    Tensor sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementC*>(smem_buf)),
                            typename Config::SmemLayoutOut{});

    auto smem_tiled_copy_C = make_tiled_copy_C(typename Config::SmemCopyAtomC{}, tiled_mma);
    auto smem_thr_copy_C   = smem_tiled_copy_C.get_slice(threadIdx.x);
    auto tCrC              = smem_thr_copy_C.retile_S(accum);
    auto tCsC              = smem_thr_copy_C.partition_D(sC);

    __syncthreads();   // mainloop done reading smem; safe to reuse for sC
    copy(smem_tiled_copy_C, tCrC, tCsC);
    __syncthreads();

    typename Config::GmemTiledCopyC gmem_tiled_copy_C;
    auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(threadIdx.x);
    auto tDsC            = gmem_thr_copy_C.partition_S(sC);
    auto tDgC            = gmem_thr_copy_C.partition_D(gC);
    copy(gmem_tiled_copy_C, tDsC, tDgC);
  }
};

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_VVOR_CUTLASS_SM80_CUH
