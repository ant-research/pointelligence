#include <Python.h>
#include <torch/library.h>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
	static struct PyModuleDef module_def = {
		PyModuleDef_HEAD_INIT,
		"_C", /* name of module */
		NULL, /* module documentation, may be NULL */
		-1,   /* size of per-interpreter state of the module,
	             or -1 if the module keeps state in global variables. */
		NULL, /* methods */
	};
	return PyModule_Create(&module_def);
}
}

// Defines the operators
TORCH_LIBRARY(sparse_engines_cuda, m) {
	m.def("sparse_matrix_vector_multiplication_reduction(Tensor a, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, int n) -> Tensor");
	m.def("sparse_vector_vector_outer_product_reduction(Tensor a, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, int n) -> Tensor");

	// Grouped CUDA kernels (weight-reuse hand-CUDA path).
	// Same semantics as the non-grouped variants but require a_idx (mvmr) /
	// o_idx (vvor) sorted ascending by kernel offset, plus a seg_offs
	// array of segment boundaries computed by sparse_engines._seg_offs.
	m.def("sparse_mvmr_grouped_mma(Tensor a, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, Tensor seg_offs, int n) -> Tensor");
	m.def("sparse_vvor_grouped_mma(Tensor grad_out, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, Tensor seg_offs, int n) -> Tensor");

	// WMMA-direct vvor. Same I/O shape as
	// sparse_vvor_grouped_mma but the inner loop uses wmma::mma_sync on
	// m16n16k16 fp16/bf16 tiles. Requires M % 16 == 0 and C % 16 == 0.
	// fp32 / TF32 not supported by this atom — caller should dispatch
	// fp32 inputs to sparse_vvor_grouped_mma instead.
	m.def("sparse_vvor_grouped_wmma(Tensor grad_out, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, Tensor seg_offs, int n) -> Tensor");

	// WMMA-direct mvmr (forward) — the mvmr analogue of the vvor WMMA path.
	// Same I/O shape as sparse_mvmr_grouped_mma; the inner loop contracts the
	// channel axis with wmma::mma_sync on m16n16k16 fp16/bf16 tiles and
	// atomicAdd-scatters each triplet column to out[o_idx]. Requires
	// M % 16 == 0 and C % 16 == 0; fp32 dispatches to sparse_mvmr_grouped_mma.
	m.def("sparse_mvmr_grouped_wmma(Tensor a, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, Tensor seg_offs, int n) -> Tensor");

	// Cooperative-warp split-K WMMA vvor.
	// Adds an extra `w` parameter for the T-axis slice count per tile.
	m.def("sparse_vvor_grouped_wmma_coop(Tensor grad_out, Tensor a_idx, Tensor b, Tensor b_idx, Tensor o_idx, Tensor seg_offs, int n, int w) -> Tensor");

	// CUTLASS Ampere single-tile vvor skeleton.
	// Pre-gathered (M_TILE, K_seg) / (N_TILE, K_seg) row-major fp16 inputs;
	// returns fp32 (M_TILE, N_TILE) tile. Affine layouts only — no IndexedGather.
	// Caller pads K_seg up to a multiple of Config::TileK (currently 32).
	m.def("sparse_vvor_cutlass_sm80_single_tile(Tensor A_seg, Tensor B_seg, int k_seg_padded) -> Tensor");

	// CUTLASS Ampere single-tile vvor with kernel-side K-mode IndexedGather.
	// Takes the FULL grad_output / input
	// tensors plus per-K-element row-index arrays (i_idx/j_idx, int32) and
	// channel-tile offsets (m_start, c_start). The CollectiveMma's cp.async
	// loads gather along the K axis via ComposedLayout + IndexedGather. Caller
	// pads K_seg up to a multiple of Config::TileK (currently 32) using sentinel
	// indices (i_idx and j_idx must have length K_seg_padded; padded slots must
	// not contribute to the reference either).
	m.def("sparse_vvor_cutlass_sm80_single_tile_gathered(Tensor grad_output, Tensor input_b, Tensor i_idx_seg, Tensor j_idx_seg, int m_start, int c_start, int k_seg_padded) -> Tensor");

	// Full CUTLASS Ampere vvor backward.
	// Outer (k, mt, ct) grid scheduler — drop-in replacement for the
	// sparse_vvor_grouped_* paths. One CTA per (k-segment, M-tile, C-tile);
	// each reads its segment bounds from seg_offs[k]/seg_offs[k+1] and runs
	// a K-mode-gathered, K-axis-PREDICATED CollectiveMma over the segment's
	// triplet slice (no index padding — MainloopSm80CpAsync handles the
	// partial last K-tile + empty segments natively). Returns fp32
	// grad_weight shaped (n_o, 1, M, C) like the other grouped paths.
	m.def("sparse_vvor_cutlass_sm80_full(Tensor grad_output, Tensor a_idx, Tensor input_b, Tensor b_idx, Tensor seg_offs, int n_o) -> Tensor");

	// Hopper sm_90 vvor backward.
	// Same call signature + algorithm as sparse_vvor_cutlass_sm80_full;
	// distinct symbol so the Python dispatcher can route sm_90 hardware.
	// Build-only locally — correctness validated on the H200 cluster cell.
	// This is the proven Sm80 cp.async-Unpredicated +
	// sentinel-zero-row op cross-compiled for sm_90 (no WGMMA; see
	// sparse_vvor_cutlass_sm90.cuh for the full rationale).
	m.def("sparse_vvor_cutlass_sm90_full(Tensor grad_output, Tensor a_idx, Tensor input_b, Tensor b_idx, Tensor seg_offs, int n_o) -> Tensor");

	// CUTLASS Ampere single-tile mvmr skeleton.
	// Affine layouts only — no IndexedGather, no scatter. Pre-gathered
	// (M_TILE, C_seg) weight tile + (S_TILE, C_seg) input tile, both fp16
	// row-major C-contiguous; returns the fp32 (M_TILE, S_TILE) tile
	//   O[m, s] = sum_c W_seg[m, c] * B_seg[s, c]
	// i.e. the mvmr GEMM core with the contraction over the CHANNEL axis
	// (vs vvor's seg_len contraction). Caller pads C_seg up to a multiple
	// of Config::TileK (currently 32). The b_idx input gather and the
	// a_idx scatter-accumulate epilogue are deferred to the gathered/full
	// entries below.
	m.def("sparse_mvmr_cutlass_sm80_single_tile(Tensor W_seg, Tensor B_seg, int c_seg_padded) -> Tensor");

	// CUTLASS Ampere single-tile mvmr with kernel-side
	// IndexedGather on the B (input) operand. W stays affine; B is read
	// directly from input_b and gathered along the S/triplet axis inside
	// the CollectiveMma mainloop via a composed IndexedGather custom-stride
	// layout (mirrors the proven vvor gathered path). Because mvmr gathers
	// along S (non-contraction) while C (contraction) stays gmem-contig,
	// the single-tile Config composes directly — no transposing 2nd Config
	// (the vvor ldmatrix-T pitfall does not apply here). Returns the
	// fp32 (M_TILE, S_TILE) tile  O[m, s] = sum_c W_seg[m, c]
	//   * input_b[b_idx_seg[s], c_start + c]. Caller pads C_seg up to a
	// TileK multiple (currently 32) and clamps/pads b_idx_seg to S_TILE.
	m.def("sparse_mvmr_cutlass_sm80_single_tile_gathered(Tensor W_seg, Tensor input_b, Tensor b_idx_seg, int c_start, int c_seg_padded) -> Tensor");

	// CUTLASS Ampere FULL mvmr op — drop-in
	// replacement for sparse_matrix_vector_multiplication_reduction (the
	// Triton-grouped mvmr). Outer ragged-K-segment grid (one CTA per
	// (m-tile, k-segment); the CTA loops over S-chunks of its segment) +
	// the kernel-side S-gather (input_b gathered by b_idx, clamped to
	// the segment via a sentinel zero row — the proven vvor-full
	// machinery) + the scatter-accumulate epilogue (fp32 register accum
	// → atomicAdd into o[o_idx[t]] with prev_out run-length coalescing,
	// lifted from sparse_mvmr_grouped_mma.cuh). a is the affine weight
	// (n_o_k, 1, C_full, M_full) (pre-transposed host-side to C-contig);
	// returns o (n_o, 1, M_full) fp32. Handles seg_len % S_TILE != 0 and
	// empty segments. C_full / M_full must be TileK / TileM multiples.
	m.def("sparse_mvmr_cutlass_sm80_full(Tensor a, Tensor b_idx, Tensor input_b, Tensor o_idx, Tensor seg_offs, int n_o) -> Tensor");

	// Repack-skipping twin of sparse_mvmr_cutlass_sm80_full.
	// `aT` is the already-(n_o_k, M_full, C_full)-C-contiguous pre-staged
	// weight (exactly what `_full`'s internal
	// select(1,0).transpose(1,2).contiguous() would produce, and exactly
	// what the kernel consumes); this entry SKIPS that per-call repack so
	// the fused conv Function can stage the weight once and pass it
	// straight through (fwd + the grad_b residual). Kernel device body
	// identical to `_full`; bit-exact vs `_full` given the same staged
	// buffer. grad_b's transposed-weight case is a caller staging choice
	// (drop the .transpose(1,2)), NOT a host transpose flag.
	m.def("sparse_mvmr_cutlass_sm80_full_prestaged(Tensor aT, Tensor b_idx, Tensor input_b, Tensor o_idx, Tensor seg_offs, int n_o) -> Tensor");

	// Hopper sm_90 mvmr full.
	// Same call signature + algorithm as sparse_mvmr_cutlass_sm80_full;
	// distinct symbol so the Python dispatcher can route sm_90 hardware.
	// Build-only locally — correctness validated on the H200 cluster cell.
	// This is the proven frozen Sm80 cp.async-Unpredicated + S-gather +
	// scatter-accumulate op cross-compiled for sm_90 (no WGMMA; see
	// sparse_mvmr_cutlass_sm90.cuh for the full rationale).
	m.def("sparse_mvmr_cutlass_sm90_full(Tensor a, Tensor b_idx, Tensor input_b, Tensor o_idx, Tensor seg_offs, int n_o) -> Tensor");

	// sm_90 twin of sparse_mvmr_cutlass_sm80_full_prestaged
	// (same repack-skip; build-only locally, executed on the H200 cell).
	m.def("sparse_mvmr_cutlass_sm90_full_prestaged(Tensor aT, Tensor b_idx, Tensor input_b, Tensor o_idx, Tensor seg_offs, int n_o) -> Tensor");

}
