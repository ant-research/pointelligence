#ifndef SPARSE_ENGINES_CUDA_SPARSE_MVMR_CUTLASS_SM90_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_MVMR_CUTLASS_SM90_CUH

// ─── Hopper sm_90 mvmr variant ──────────────────────────────────────────────
//
// Build-only locally (dev box is sm_89 / RTX 5880 Ada — cannot RUN sm_90).
// Correctness is validated on the H200 cluster cell (the vvor sm_90 analog).
// This header is the mvmr twin of sparse_vvor_cutlass_sm90.cuh — same
// rationale, same structure.
//
// ── Why this is the Sm80-Unpredicated op recompiled for sm_90, NOT a
//    genuine Sm90 WGMMA collective ──────────────────────────────────────────
//
// The vvor sm_90 work established the authoritative finding:
// vendored CUTLASS 4.3.4 has **no** `MainloopSm90CpAsyncUnpredicated`
// (and no WGMMA collective with a simple single-CTA `operator()`) usable
// for the K-mode `ComposedLayout` gather these grouped backward kernels
// need. Both Sm90 cp.async GMMA collectives
// (`MainloopSm90CpAsyncGmmaWarpSpecialized`,
// `MainloopSm90CpAsyncGmmaRmemAWarpSpecialized`) are producer/consumer
// warp-specialized (driven through a `MainloopPipeline`, no single-CTA
// `operator()`) and unconditionally `domain_offset` the K mode — the
// exact residue-shift that blocks the K-mode gather. The only
// vendored Sm90 *gather* example (52_hopper_gather_scatter_fusion)
// gathers along the strided (M/N) dim only, never the K (contraction)
// mode. See sparse_vvor_cutlass_sm90.cuh for the full citation trail.
//
// Net: the supported Hopper path is the proven Sm80
// `MainloopSm80CpAsyncUnpredicated` + `SegmentClampedGather`
// sentinel-zero-row op (the frozen full-op stack in
// sparse_mvmr_cutlass_sm80.{cuh,cu}) **cross-compiled for sm_90**. On a
// Hopper device this issues sm_80-class `cp.async` + `mma.sync`
// (m16n8k16 HMMA) PTX, forward-compatible with sm_90 and executed
// correctly by Hopper. It is **bit-identical in algorithm** to the
// sm_89 mvmr full op: same contraction-axis-C `CollectiveMma`, same
// S-axis `make_gather_tensor` + `MvmrSegmentClampedGather`
// sentinel-zero-row, same scatter-accumulate-by-o_idx prev_out-coalesced
// atomicAdd epilogue, same (mt, k) grid.
//
// Limitation, flagged for the dispatcher + H200 cell (identical to the
// vvor caveat): this path does NOT engage Hopper WGMMA. The H200
// cell measures whether the cp.async (Ampere-MMA-on-Hopper) path still
// beats Triton-grouped mvmr on H200. A genuine WGMMA mvmr variant
// remains an open follow-up.
//
// Structurally this header re-exposes the frozen Sm80 full op + config
// under sm_90-named aliases; the `.cu` provides the sm_90-dispatched
// host entry point `sparse_mvmr_cutlass_sm90_full` (the mvmr drop-in,
// mirroring vvor's sm90 `full`-only surface 1:1). No algorithm change →
// nothing to re-verify for parity beyond "sm_89 still green + sm_90
// cross-compiles". The frozen full-op .cuh is *included*, never modified.

#include "sparse_mvmr_cutlass_sm80.cuh"

namespace sparse_engines_cuda {

// The Hopper config + op are the proven Ampere ones (algorithm is
// arch-agnostic; the contraction-axis-C orientation + sentinel-zero-row
// S-gather + scatter epilogue the full-op work established are inherited
// by sm_90 unchanged). Aliased under sm90 names so call sites + future
// genuine-WGMMA work have a stable seam to specialize without touching
// the frozen sm_80 path.
using MvmrCutlassSm90FpropConfig = MvmrCutlassSm80FpropConfig;

template <class Config>
using MvmrCutlassSm90FullOp = MvmrCutlassSm80FullOp<Config>;

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_MVMR_CUTLASS_SM90_CUH
