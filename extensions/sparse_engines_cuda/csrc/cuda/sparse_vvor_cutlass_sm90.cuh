#ifndef SPARSE_ENGINES_CUDA_SPARSE_VVOR_CUTLASS_SM90_CUH
#define SPARSE_ENGINES_CUDA_SPARSE_VVOR_CUTLASS_SM90_CUH

// ─── Cycle-4 §1.12 G14/G18 — Hopper sm_90 vvor backward variant (Task 4) ────
//
// Build-only locally (dev box is sm_89 / RTX 5880 Ada — cannot RUN sm_90).
// Correctness is validated on the H200 cluster cell (separate pre-reg doc).
//
// ── Why this is the Sm80-Unpredicated op recompiled for sm_90, NOT a
//    genuine Sm90 WGMMA collective ──────────────────────────────────────────
//
// The pre-reg (cycle4_tier2_cutlass_vvor.md §5 Task-4 row + §"Risks" R2)
// and the cycle-4 conclusion (cycle_4_tier2_g14_complete.md §"design-doc
// R1") together fix the Hopper design surface. The genuine Sm90 GMMA
// (WGMMA) cp.async collectives in vendored CUTLASS 4.3.4 are
// `MainloopSm90CpAsyncGmmaWarpSpecialized` and
// `MainloopSm90CpAsyncGmmaRmemAWarpSpecialized`
// (include/cutlass/gemm/dispatch_policy.hpp:225/238). BOTH are:
//
//   1. **Producer/consumer warp-specialized** — a `load()` method driven
//      by a DMA producer warp through a `MainloopPipeline`
//      (`PipelineAsync<Stages>`, producer_acquire / producer_commit) and a
//      separate `mma()` method run by consumer warpgroups
//      (consumer_wait / consumer_release). There is NO simple `operator()`
//      single-CTA entry point analogous to Sm80's
//      `MainloopSm80CpAsyncUnpredicated::operator()`. Driving them requires
//      either a hand-written ~200-line producer/consumer pipeline scheduler
//      (deadlock-prone; unverifiable on sm_89 hardware) or the heavyweight
//      `cutlass::gemm::device::GemmUniversalAdapter` device API.
//
//   2. **Unconditionally `domain_offset`-ing the K mode** — both Sm90
//      cp.async collectives do, in `load()`
//      (sm90_mma_multistage_gmma_ss_warpspecialized.hpp:223):
//          Tensor gA = domain_offset(make_coord(0, get<2>(residue_mnk), 0),
//                                    gA_in);
//      This is the *same* residue-shift the design-doc R1 blocker flagged
//      for Sm80 `MainloopSm80CpAsync`. There is **no
//      `MainloopSm90CpAsyncUnpredicated`** — the unpredicated escape hatch
//      that made the Sm80 BEAT kernel possible does not exist on Sm90.
//
//   3. The only vendored Sm90 *gather* example
//      (examples/52_hopper_gather_scatter_fusion, via gather_kernel.cuh
//      `GemmGather`) gathers along the **strided dimension only** (M for
//      row-major A, N for column-major B) — explicitly *not* the K
//      (contraction) mode. The vvor backward is a per-k-segment
//      outer-product reduction whose gather is intrinsically K-mode, so
//      example 52's Sm90 GemmGather is structurally inapplicable.
//
// Net: a genuine WGMMA Sm90 vvor-with-K-gather kernel is a real research
// fork (hand-written warp-specialized pipeline, or a CUTLASS-internal
// rewrite to expose an unpredicated Sm90 cp.async mainloop) that exceeds a
// build-only / algorithmically-equivalent / sm_89-non-regressing
// deliverable and cannot be validated on the local sm_89 box.
//
// Per pre-reg R2 — *"Hopper TMA can't encode IndexedGather → Hopper path
// uses cp.async, losing ~10-20% peak. Acceptable; WGMMA alone likely beats
// Triton."* — the R2-compliant Hopper path is exactly the proven Sm80
// `MainloopSm80CpAsyncUnpredicated` + `SegmentClampedGather`
// sentinel-zero-row op, cross-compiled for sm_90. On Hopper this issues
// sm_80-class `cp.async` + `mma.sync` (m16n8k16 HMMA) PTX, which is
// forward-compatible with sm_90 and runs correctly (Hopper executes
// Ampere-class tensor-core instructions). It is **bit-identical in
// algorithm** to the sm_89 BEAT kernel: same K-mode `make_gather_tensor`
// ComposedLayout, same `SegmentClampedGather` sentinel-zero-row, same
// fp32 deterministic epilogue accumulation, same (k, mt, ct) grid.
//
// Limitation, flagged for the main agent + H200 cell: this path does NOT
// engage Hopper WGMMA. The pre-reg's sm_90 ≤ 0.95× threshold anticipated
// WGMMA. The H200 cell measures whether the cp.async (Ampere-MMA-on-Hopper)
// path still beats Triton-grouped on H200 — the sm_89 margin (0.2435×
// Triton, ~4.1×) makes this very likely even without WGMMA, but a genuine
// WGMMA variant remains an open follow-up if the H200 cp.async ratio lands
// above the Beat band.
//
// Structurally this header therefore re-exposes the Sm80 full op + config
// under sm_90-named aliases; the `.cu` provides the sm_90-dispatched host
// entry point. No algorithm change → nothing to re-verify for parity
// beyond "sm_89 still green + sm_90 cross-compiles".

#include "sparse_vvor_cutlass_sm80.cuh"

namespace sparse_engines_cuda {

// The Hopper config + op are the proven Ampere ones (algorithm is
// arch-agnostic; the sentinel-zero-row resolution the cycle-4 conclusion
// established is, per its own §"design-doc R1" text, inherited by sm_90).
// Aliased under sm90 names so call sites + future genuine-WGMMA work have
// a stable seam to specialize without touching the sm_80 path.
using VvorCutlassSm90Config = VvorCutlassSm80GatherConfig;

template <class Config>
using VvorCutlassSm90FullOp = VvorCutlassSm80FullOp<Config>;

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_SPARSE_VVOR_CUTLASS_SM90_CUH
