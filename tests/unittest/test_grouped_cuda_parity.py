"""Hand-CUDA grouped path vs Triton grouped path — parity battery.

The hand-CUDA grouped scaffolding adds a new dispatch backend for MVMR and
VVOR: a hand-rolled CUDA kernel (sparse_{mvmr,vvor}_grouped_mma) that
trades tensor-core mma for register-resident weight reuse + coarser-
cadence atomicAdd scatter. This test asserts numerical parity vs the
Triton-grouped path (the current production default for C >= 128) on
identical inputs across the PTv3 stage shapes.

If parity holds, speed can be measured meaningfully. If parity fails, the
kernel itself has a correctness bug and the bench is meaningless.

The CUDA path is invoked directly via the wrapper functions; the
Triton path is forced via dispatch_mode("force_fsg"). Both
paths require sort_by="k" indices, G == 1, and C >= 128 (Triton
grouped fires above that; the CUDA path is unconstrained on C but
we keep the same regime for comparability).
"""

import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch

import sparse_engines  # noqa: F401 — registers ops
from sparse_engines._dispatch_override import dispatch_mode
from sparse_engines.mvmr_grouped_cuda import (
    sparse_matrix_vector_multiplication_reduction_grouped_cuda,
)
from sparse_engines.vvor_grouped_cuda import (
    sparse_vector_vector_outer_product_reduction_grouped_cuda,
)
from sparse_engines.vvor_grouped_wmma import (
    sparse_vector_vector_outer_product_reduction_grouped_wmma,
)
from sparse_engines.vvor_grouped_wmma_coop import (
    sparse_vector_vector_outer_product_reduction_grouped_wmma_coop,
)
from sparse_engines.mvmr_grouped_wmma import (
    sparse_matrix_vector_multiplication_reduction_grouped_wmma,
)


# PTv3 stage shapes — same as test_grouped_equivalence.py, but skipping
# enc0/enc1 (C < 128, where the Triton grouped path doesn't fire and
# parity comparison would route the Triton side to a different backend).
PTV3_STAGES = [
    # name, N_a (kernel offsets), N_b, N_o, M, C, T
    ("enc2",  27,  3_000,  3_000, 128, 128,  25_000),
    ("enc3",  27,    800,    800, 256, 256,   6_500),
    ("enc4",  27,    200,    200, 512, 512,   1_700),
]

DTYPES = [
    ("fp32", torch.float32, 5e-3),
    ("fp16", torch.float16, 5e-3),
    ("bf16", torch.bfloat16, 1.5e-2),
]


def _make_mvmr_indices(N_a, N_b, N_o, T, device):
    torch.manual_seed(1)
    a_idx = torch.randint(0, N_a, (T,), device=device, dtype=torch.int64)
    b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)
    o_idx = torch.randint(0, N_o, (T,), device=device, dtype=torch.int64)
    # sort_by="k" — required by both grouped paths.
    order = torch.argsort(a_idx, stable=True)
    return a_idx[order], b_idx[order], o_idx[order]


def _make_vvor_indices(N_a, N_b, K_off, T, device):
    """VVOR's o_idx is the kernel-offset bin; sort by o_idx for grouped."""
    torch.manual_seed(1)
    a_idx = torch.randint(0, N_a, (T,), device=device, dtype=torch.int64)
    b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)
    o_idx = torch.randint(0, K_off, (T,), device=device, dtype=torch.int64)
    order = torch.argsort(o_idx, stable=True)
    return a_idx[order], b_idx[order], o_idx[order]


def _rel_err(x, y):
    diff = (x.float() - y.float()).abs().max().item()
    base = y.float().abs().max().item()
    return diff / max(base, 1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCudaParity(unittest.TestCase):

    def test_mvmr_fwd_parity(self):
        device = "cuda"
        for stage, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            for dt_name, dtype, tol in DTYPES:
                with self.subTest(stage=stage, dtype=dt_name):
                    torch.manual_seed(0)
                    a = (torch.randn(N_a, 1, C, M, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    a_idx, b_idx, o_idx = _make_mvmr_indices(N_a, N_b, N_o, T, device)

                    # Triton-grouped reference.
                    with dispatch_mode("force_fsg"):
                        out_triton = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                            a, a_idx, b, b_idx, o_idx, N_o,
                        )

                    # Hand-CUDA grouped under test.
                    out_cuda = sparse_matrix_vector_multiplication_reduction_grouped_cuda(
                        a, a_idx, b, b_idx, o_idx, N_o,
                    )

                    rel = _rel_err(out_cuda, out_triton)
                    print(f"  MVMR-fwd [{stage} {dt_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol,
                        f"{stage} {dt_name}: hand-CUDA vs Triton-grouped rel={rel:.3e}")

    def test_vvor_fwd_parity(self):
        device = "cuda"
        K_off = 27
        for stage, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            for dt_name, dtype, tol in DTYPES:
                with self.subTest(stage=stage, dtype=dt_name):
                    torch.manual_seed(0)
                    # VVOR's a = grad_output (shape (N_a, G=1, M)); b = input.
                    a = (torch.randn(N_a, 1, M, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    a_idx, b_idx, o_idx = _make_vvor_indices(N_a, N_b, K_off, T, device)

                    with dispatch_mode("force_fsg"):
                        out_triton = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
                            a, a_idx, b, b_idx, o_idx, K_off,
                        )

                    out_cuda = sparse_vector_vector_outer_product_reduction_grouped_cuda(
                        a, a_idx, b, b_idx, o_idx, K_off,
                    )

                    rel = _rel_err(out_cuda, out_triton)
                    print(f"  VVOR-fwd [{stage} {dt_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol,
                        f"{stage} {dt_name}: hand-CUDA vs Triton-grouped rel={rel:.3e}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedWmmaVvorParity(unittest.TestCase):
    """WMMA-direct vvor vs Triton-grouped reference.

    The WMMA kernel uses m16n16k16 fp16/bf16 atoms; fp32 inputs route
    through the scalar-FMA grouped fallback (which already has its own
    parity coverage in TestGroupedCudaParity above). We test fp16 + bf16
    only here.

    WMMA's accumulator-order differs slightly from tl.dot's, so we allow
    a marginally looser tolerance — 7e-3 for fp16 (vs 5e-3 in the
    scalar-FMA tests). If observed relerr is tighter, future tests can
    tighten the bound.
    """

    WMMA_DTYPES = [
        ("fp16", torch.float16, 7e-3),
        ("bf16", torch.bfloat16, 2e-2),
    ]

    def test_vvor_fwd_parity_wmma(self):
        device = "cuda"
        K_off = 27
        for stage, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            for dt_name, dtype, tol in self.WMMA_DTYPES:
                with self.subTest(stage=stage, dtype=dt_name):
                    torch.manual_seed(0)
                    a = (torch.randn(N_a, 1, M, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    a_idx, b_idx, o_idx = _make_vvor_indices(N_a, N_b, K_off, T, device)

                    with dispatch_mode("force_fsg"):
                        out_triton = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
                            a, a_idx, b, b_idx, o_idx, K_off,
                        )

                    out_wmma = sparse_vector_vector_outer_product_reduction_grouped_wmma(
                        a, a_idx, b, b_idx, o_idx, K_off,
                    )

                    rel = _rel_err(out_wmma, out_triton)
                    print(f"  VVOR-fwd-WMMA [{stage} {dt_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol,
                        f"{stage} {dt_name}: WMMA-direct vs Triton-grouped rel={rel:.3e}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedWmmaCoopVvorParity(unittest.TestCase):
    """Cooperative-warp split-K WMMA vvor vs Triton-grouped reference.

    AtomicAdd reduction may introduce extra fp32 summation order variance.
    Use 2x widened tolerance vs the WMMA single-pass test (fp16 <= 1.4e-2,
    bf16 <= 4e-2). If observed relerr stays under the original WMMA bound
    (7e-3/2e-2), the widening will be reverted in a follow-up.
    """

    WMMA_COOP_DTYPES = [
        ("fp16", torch.float16, 1.4e-2),
        ("bf16", torch.bfloat16, 4e-2),
    ]

    def test_vvor_fwd_parity_wmma_coop(self):
        device = "cuda"
        K_off = 27
        for stage, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            for dt_name, dtype, tol in self.WMMA_COOP_DTYPES:
                with self.subTest(stage=stage, dtype=dt_name):
                    torch.manual_seed(0)
                    a = (torch.randn(N_a, 1, M, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    a_idx, b_idx, o_idx = _make_vvor_indices(N_a, N_b, K_off, T, device)

                    with dispatch_mode("force_fsg"):
                        out_triton = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
                            a, a_idx, b, b_idx, o_idx, K_off,
                        )

                    out_coop = sparse_vector_vector_outer_product_reduction_grouped_wmma_coop(
                        a, a_idx, b, b_idx, o_idx, K_off, w=8,
                    )

                    rel = _rel_err(out_coop, out_triton)
                    print(f"  VVOR-fwd-WMMA-coop W=8 [{stage} {dt_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol,
                        f"{stage} {dt_name}: WMMA-coop W=8 vs Triton-grouped rel={rel:.3e}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassVvorTileParity(unittest.TestCase):
    """CUTLASS single-tile vvor skeleton — single-tile parity.

    Day-3 GO/NO-GO criteria:
      - One stage (enc4 fp16: M=C=512), one (k, mt, ct) tile (TileM=TileN=64).
      - Pre-gathered (M_TILE, K_seg_padded) / (N_TILE, K_seg_padded) inputs.
      - CUTLASS path vs scalar-FMA fp32 reference computing
            C[m, n] = sum_k A[m, k] * B[n, k]
        from the SAME pre-gathered + zero-padded buffers.
      - Tolerance: relerr <= 1e-2 (looser than WMMA's 7e-3
        because CUTLASS's mainloop accumulation order differs slightly
        from scalar-FMA).
    """

    def test_vvor_cutlass_sm80_single_tile_parity(self):
        from sparse_engines.vvor_cutlass import (
            M_TILE, N_TILE, K_TILE,
            stage_one_tile,
            vvor_cutlass_sm80_single_tile,
            vvor_cutlass_sm80_single_tile_reference,
        )

        device = "cuda"
        # enc4 production-like shape: M=C=512, K_off=27, N_o=N_b=200, T=1700.
        # Average seg_len ≈ 1700/27 ≈ 63 → pick a moderate seg_len of 64
        # to give a representative inner-K count.
        M_full, C_full = 512, 512
        N_o, N_b = 200, 200
        seg_len = 64

        torch.manual_seed(0)
        grad_output = (torch.randn(N_o, 1, M_full, device=device,
                                   dtype=torch.float32) * 0.1).to(torch.float16)
        input_b = (torch.randn(N_b, 1, C_full, device=device,
                               dtype=torch.float32) * 0.1).to(torch.float16)
        i_idx_seg = torch.randint(0, N_o, (seg_len,), device=device, dtype=torch.int64)
        j_idx_seg = torch.randint(0, N_b, (seg_len,), device=device, dtype=torch.int64)

        m_start, c_start = 0, 0
        A_seg, B_seg, K_pad = stage_one_tile(
            grad_output, input_b, i_idx_seg, j_idx_seg, m_start, c_start
        )

        self.assertEqual(A_seg.shape, (M_TILE, K_pad))
        self.assertEqual(B_seg.shape, (N_TILE, K_pad))
        self.assertEqual(K_pad % K_TILE, 0)

        out_cutlass = vvor_cutlass_sm80_single_tile(A_seg, B_seg, K_pad)
        out_ref     = vvor_cutlass_sm80_single_tile_reference(A_seg, B_seg)

        diff = (out_cutlass - out_ref).abs().max().item()
        base = out_ref.abs().max().item()
        rel = diff / max(base, 1e-6)
        print(f"  VVOR-CUTLASS-sm80 [enc4-tile fp16 seg={seg_len} pad={K_pad}] "
              f"rel={rel:.3e} maxdiff={diff:.3e} maxref={base:.3e}")
        self.assertLess(rel, 1e-2,
            f"CUTLASS vs scalar-FMA reference rel={rel:.3e} exceeds 1e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassVvorTileGatheredParity(unittest.TestCase):
    """CUTLASS kernel-side gather vvor — kernel-side gather parity.

    Same single-tile parity contract as TestGroupedCutlassVvorTileParity,
    but the kernel reads grad_output / input directly and gathers on the
    K axis via a composed IndexedGather layout (example 52 case 2
    pattern). The reference does the gather Python-side.

    Tolerance relerr <= 1e-2 fp16.
    """

    def test_vvor_cutlass_sm80_single_tile_gathered_parity(self):
        from sparse_engines.vvor_cutlass import (
            M_TILE, N_TILE, K_TILE,
            pad_indices_for_gather,
            vvor_cutlass_sm80_single_tile_gathered,
            vvor_cutlass_sm80_single_tile_gathered_reference,
        )

        device = "cuda"
        # enc4 production-like shape — match the single-tile parity test for direct
        # comparison.
        M_full, C_full = 512, 512
        N_o, N_b = 200, 200
        seg_len = 64

        torch.manual_seed(0)
        grad_output = (torch.randn(N_o, 1, M_full, device=device,
                                   dtype=torch.float32) * 0.1).to(torch.float16)
        input_b = (torch.randn(N_b, 1, C_full, device=device,
                               dtype=torch.float32) * 0.1).to(torch.float16)
        i_idx_seg = torch.randint(0, N_o, (seg_len,), device=device, dtype=torch.int32)
        j_idx_seg = torch.randint(0, N_b, (seg_len,), device=device, dtype=torch.int32)

        m_start, c_start = 0, 0
        i_idx_pad, j_idx_pad, K_pad = pad_indices_for_gather(
            i_idx_seg, j_idx_seg, seg_len
        )
        self.assertEqual(K_pad % K_TILE, 0)
        self.assertEqual(i_idx_pad.numel(), K_pad)
        self.assertEqual(i_idx_pad.dtype, torch.int32)

        out_cutlass = vvor_cutlass_sm80_single_tile_gathered(
            grad_output, input_b, i_idx_pad, j_idx_pad,
            m_start, c_start, K_pad,
        )
        out_ref = vvor_cutlass_sm80_single_tile_gathered_reference(
            grad_output, input_b, i_idx_pad, j_idx_pad,
            m_start, c_start,
        )

        self.assertEqual(out_cutlass.shape, (M_TILE, N_TILE))
        self.assertEqual(out_ref    .shape, (M_TILE, N_TILE))

        diff = (out_cutlass - out_ref).abs().max().item()
        base = out_ref.abs().max().item()
        rel = diff / max(base, 1e-6)
        print(f"  VVOR-CUTLASS-sm80-gather [enc4-tile fp16 seg={seg_len} pad={K_pad}] "
              f"rel={rel:.3e} maxdiff={diff:.3e} maxref={base:.3e}")
        self.assertLess(rel, 1e-2,
            f"CUTLASS-gather vs scalar-FMA reference rel={rel:.3e} exceeds 1e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassVvorFullParity(unittest.TestCase):
    """CUTLASS full vvor — FULL vvor backward parity.

    The full (k, mt, ct) grid scheduler: drop-in replacement for the
    proven-correct WMMA-coop grouped path. This test exercises the
    padded-slot resolution (Option 1, K-axis predicated MainloopSm80CpAsync)
    by constructing a realistic multi-segment input with VARIABLE per-k
    segment lengths, including:
      - at least one segment with seg_len % K_TILE != 0  (partial last
        K-tile — the residue-predication path), and
      - at least one EMPTY segment (seg_len == 0 — k_tile_count == 0,
        must emit a zero grad_weight slab).

    Reference = sparse_vector_vector_outer_product_reduction_grouped_wmma_coop
    (parity-proven against Triton-grouped). Threshold:
    relerr <= 1e-2 fp16 (same as the single-tile parity; CUTLASS fp32-accum order
    differs slightly from the WMMA-coop atomicAdd path).
    """

    def test_vvor_cutlass_sm80_full_parity(self):
        from sparse_engines.vvor_cutlass import (
            K_TILE,
            sparse_vector_vector_outer_product_reduction_grouped_cutlass,
        )

        device = "cuda"
        # enc4-style production shape: M = C = 512, n_o = 27 (PTv3 K_offsets),
        # T ≈ 1700 triplets. TileM = TileN = 64 → 8 M-tiles × 8 C-tiles × 27 k
        # = 13,824 CTAs.
        M_full, C_full = 512, 512
        N_o, N_b = 200, 200
        n_o = 27

        torch.manual_seed(0)

        # Hand-build per-segment lengths so we control the boundary cases:
        #   - segment k=5  is EMPTY (seg_len == 0),
        #   - several segments have seg_len % K_TILE != 0 (partial K-tile),
        #   - one segment is an exact K_TILE multiple (no residue).
        seg_lens = []
        for k in range(n_o):
            if k == 5:
                seg_lens.append(0)                       # empty segment
            elif k == 0:
                seg_lens.append(2 * K_TILE)              # exact multiple
            else:
                # Mix of K_TILE-non-multiples around the ~1700/27 ≈ 63 mean.
                seg_lens.append(40 + (k * 7) % 57)       # 40..96, mostly % K_TILE != 0
        T = sum(seg_lens)

        # Assertions on the constructed boundary conditions.
        self.assertEqual(seg_lens[5], 0, "k=5 must be the empty segment")
        self.assertTrue(
            any(s % K_TILE != 0 and s > 0 for s in seg_lens),
            "need at least one seg_len % K_TILE != 0",
        )

        # o_idx: segment k repeated seg_lens[k] times, already sorted asc.
        o_idx = torch.cat([
            torch.full((s,), k, device=device, dtype=torch.int64)
            for k, s in enumerate(seg_lens)
        ])
        self.assertEqual(o_idx.numel(), T)

        a_idx = torch.randint(0, N_o, (T,), device=device, dtype=torch.int64)
        b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)

        a = (torch.randn(N_o, 1, M_full, device=device, dtype=torch.float32)
             * 0.1).to(torch.float16)
        b = (torch.randn(N_b, 1, C_full, device=device, dtype=torch.float32)
             * 0.1).to(torch.float16)

        # Proven-correct reference.
        out_ref = sparse_vector_vector_outer_product_reduction_grouped_wmma_coop(
            a, a_idx, b, b_idx, o_idx, n_o, w=8,
        )

        # CUTLASS full path under test.
        out_cutlass = sparse_vector_vector_outer_product_reduction_grouped_cutlass(
            a, a_idx, b, b_idx, o_idx, n_o,
        )

        self.assertEqual(tuple(out_cutlass.shape), tuple(out_ref.shape),
            f"shape mismatch: cutlass={tuple(out_cutlass.shape)} "
            f"ref={tuple(out_ref.shape)}")

        rel = _rel_err(out_cutlass, out_ref)
        print(f"  VVOR-CUTLASS-sm80-FULL [enc4 fp16 n_o={n_o} T={T}] "
              f"rel={rel:.3e} shape={tuple(out_cutlass.shape)}")

        # Empty-segment slab (k=5) must be exactly zero on both sides.
        empty_ref     = out_ref[5].abs().max().item()
        empty_cutlass = out_cutlass[5].abs().max().item()
        print(f"    empty-seg k=5: |ref|max={empty_ref:.3e} "
              f"|cutlass|max={empty_cutlass:.3e}")
        self.assertEqual(empty_cutlass, 0.0,
            "empty segment k=5 must produce an all-zero grad_weight slab")

        self.assertLess(rel, 1e-2,
            f"CUTLASS-full vs WMMA-coop reference rel={rel:.3e} exceeds 1e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassMvmrTileParity(unittest.TestCase):
    """CUTLASS mvmr skeleton — single-tile parity.

    The mvmr analog of TestGroupedCutlassVvorTileParity. Single-tile GO/NO-GO:
      - One stage (enc4 fp16: M=C=512), one (k, m-tile, S-chunk) tile
        (TileM = S_TILE = 64, C-contraction tiled by C_TILE = 32).
      - Affine weight tile (M_TILE, C_seg) + pre-gathered input tile
        (S_TILE, C_seg). No IndexedGather, no scatter (the gather + the
        scatter-accumulate epilogue are covered by the gathered/full cells).
      - CUTLASS path vs scalar-FMA fp32 reference computing
            O[m, s] = sum_c W_seg[m, c] * B_seg[s, c]
        from the SAME pre-gathered + zero-padded buffers. Contraction
        axis is CHANNELS (vs vvor's seg_len) — the structural difference.
      - Tolerance: relerr <= 1e-2 (same as the vvor single-tile gate; CUTLASS
        mainloop accumulation order differs slightly from scalar-FMA).
    """

    def test_mvmr_cutlass_sm80_single_tile_parity(self):
        from sparse_engines.mvmr_cutlass import (
            M_TILE, S_TILE, C_TILE,
            stage_one_tile,
            mvmr_cutlass_sm80_single_tile,
            mvmr_cutlass_sm80_single_tile_reference,
        )

        device = "cuda"
        # enc4 production-like shape: M=C=512, K_off=27, N_b=200,
        # seg_len ≈ 1700/27 ≈ 63 → pick seg_len 64 to fill one S-tile.
        M_full, C_full = 512, 512
        N_b = 200
        seg_len = 64

        torch.manual_seed(0)
        # Weight W[k]: authoritative mvmr layout a[k] = (G=1, C_full, M_full).
        weight = (torch.randn(1, C_full, M_full, device=device,
                              dtype=torch.float32) * 0.1).to(torch.float16)
        input_b = (torch.randn(N_b, 1, C_full, device=device,
                               dtype=torch.float32) * 0.1).to(torch.float16)
        b_idx_seg = torch.randint(0, N_b, (seg_len,), device=device,
                                  dtype=torch.int64)

        m_start, c_start = 0, 0
        W_seg, B_seg, C_pad = stage_one_tile(
            weight, input_b, b_idx_seg, m_start, c_start
        )

        self.assertEqual(W_seg.shape, (M_TILE, C_pad))
        self.assertEqual(B_seg.shape, (S_TILE, C_pad))
        self.assertEqual(C_pad % C_TILE, 0)

        out_cutlass = mvmr_cutlass_sm80_single_tile(W_seg, B_seg, C_pad)
        out_ref     = mvmr_cutlass_sm80_single_tile_reference(W_seg, B_seg)

        self.assertEqual(tuple(out_cutlass.shape), (M_TILE, S_TILE))
        self.assertEqual(tuple(out_ref.shape), (M_TILE, S_TILE))

        diff = (out_cutlass - out_ref).abs().max().item()
        base = out_ref.abs().max().item()
        rel = diff / max(base, 1e-6)
        print(f"  MVMR-CUTLASS-sm80 [enc4-tile fp16 seg={seg_len} Cpad={C_pad}] "
              f"rel={rel:.3e} maxdiff={diff:.3e} maxref={base:.3e}")
        self.assertLess(rel, 1e-2,
            f"CUTLASS vs scalar-FMA reference rel={rel:.3e} exceeds 1e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassMvmrTileGatheredParity(unittest.TestCase):
    """CUTLASS mvmr kernel-side B-gather — kernel-side B-gather parity.

    The mvmr analog of TestGroupedCutlassVvorTileGatheredParity. Same
    single-tile parity contract as TestGroupedCutlassMvmrTileParity,
    but the kernel reads `input_b` directly and gathers on the S/triplet
    axis via a composed IndexedGather layout inside the CollectiveMma
    mainloop. W stays affine (sliced + padded Python-side). The reference
    does the b_idx gather Python-side then the same fp32 W @ B_gathered^T.

    Parity must stay bit-stable vs the affine single-tile cell (the gather is
    layout-level, not arithmetic) — expect ~2.4e-7, the same order. Gate
    threshold: relerr <= 1e-2 fp16 (same as the single-tile gate).
    """

    def test_mvmr_cutlass_sm80_single_tile_gathered_parity(self):
        from sparse_engines.mvmr_cutlass import (
            M_TILE, S_TILE, C_TILE,
            stage_w_tile,
            clamp_b_idx_for_gather,
            mvmr_cutlass_sm80_single_tile_gathered,
            mvmr_cutlass_sm80_single_tile_gathered_reference,
        )

        device = "cuda"
        # enc4 production-like shape — match the affine single-tile parity test for direct
        # comparison: M=C=512, K_off=27, N_b=200, seg_len≈1700/27≈63 →
        # pick seg_len 64 to fill one S-tile.
        M_full, C_full = 512, 512
        N_b = 200
        seg_len = 64

        torch.manual_seed(0)
        # Weight W[k]: authoritative mvmr layout a[k] = (G=1, C_full, M_full).
        weight = (torch.randn(1, C_full, M_full, device=device,
                              dtype=torch.float32) * 0.1).to(torch.float16)
        input_b = (torch.randn(N_b, 1, C_full, device=device,
                               dtype=torch.float32) * 0.1).to(torch.float16)
        b_idx_seg = torch.randint(0, N_b, (seg_len,), device=device,
                                  dtype=torch.int32)

        m_start, c_start = 0, 0
        W_seg, C_pad = stage_w_tile(weight, m_start, c_start)
        b_idx_pad = clamp_b_idx_for_gather(b_idx_seg)

        self.assertEqual(W_seg.shape, (M_TILE, C_pad))
        self.assertEqual(C_pad % C_TILE, 0)
        self.assertEqual(b_idx_pad.numel(), S_TILE)
        self.assertEqual(b_idx_pad.dtype, torch.int32)

        out_cutlass = mvmr_cutlass_sm80_single_tile_gathered(
            W_seg, input_b, b_idx_pad, c_start, C_pad,
        )
        out_ref = mvmr_cutlass_sm80_single_tile_gathered_reference(
            W_seg, input_b, b_idx_pad, c_start,
        )

        self.assertEqual(tuple(out_cutlass.shape), (M_TILE, S_TILE))
        self.assertEqual(tuple(out_ref.shape), (M_TILE, S_TILE))

        diff = (out_cutlass - out_ref).abs().max().item()
        base = out_ref.abs().max().item()
        rel = diff / max(base, 1e-6)
        print(f"  MVMR-CUTLASS-sm80-gather [enc4-tile fp16 seg={seg_len} "
              f"Cpad={C_pad}] rel={rel:.3e} maxdiff={diff:.3e} "
              f"maxref={base:.3e}")
        self.assertLess(rel, 1e-2,
            f"CUTLASS-gather vs scalar-FMA reference rel={rel:.3e} "
            f"exceeds 1e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassMvmrFullParity(unittest.TestCase):
    """Tier-2 CUTLASS — FULL mvmr forward parity.

    The mvmr analog of TestGroupedCutlassVvorFullParity. The full
    (m-tile, k-segment) grid scheduler with the kernel-side S-gather and
    the atomicAdd scatter-accumulate epilogue: a drop-in replacement for
    the Triton-grouped sparse_matrix_vector_multiplication_reduction.

    Constructs a realistic multi-segment input with VARIABLE per-k
    segment lengths, including:
      - at least one segment with seg_len % S_TILE != 0  (partial last
        S-chunk — the SegmentClampedGather sentinel-row path), and
      - at least one EMPTY segment (seg_len == 0 — zero chunk iterations;
        contributes nothing to o).
    The empty segment also has no triplets, so its kernel-offset row is
    never written; we additionally pick an output point that NO triplet
    maps to and assert its o slab is exactly zero on both sides (the
    mvmr analog of vvor-full's empty grad_weight slab check — mvmr's
    output is indexed by o_idx, not by k, so the zero-slab invariant is
    "an un-targeted output row stays exactly zero").

    Reference = the Triton-grouped sparse_matrix_vector_multiplication_-
    reduction (the proven production path), forced via
    dispatch_mode("force_fsg"). Threshold: relerr <= 1e-2 fp16
    (TC-vs-TC accumulation across segments, like vvor-full's ~2.3e-4).
    """

    def test_mvmr_cutlass_sm80_full_parity(self):
        from sparse_engines.mvmr_cutlass import (
            S_TILE,
            sparse_matrix_vector_multiplication_reduction_cutlass,
        )

        device = "cuda"
        # enc4-style production shape: M = C = 512, K_offsets = 27 (PTv3),
        # N_b = 200 input points. N_o output points; one (output point
        # N_o-1) is deliberately left un-targeted to check the zero slab.
        M_full, C_full = 512, 512
        N_b = 200
        N_o = 200
        K_offsets = 27

        torch.manual_seed(0)

        # Hand-build per-segment (kernel-offset) lengths so we control the
        # boundary cases:
        #   - segment k=5  is EMPTY (seg_len == 0),
        #   - several segments have seg_len % S_TILE != 0 (partial S-chunk),
        #   - one segment is an exact 2*S_TILE multiple (no residue),
        #   - one segment is long (> S_TILE) → multiple S-chunks.
        seg_lens = []
        for k in range(K_offsets):
            if k == 5:
                seg_lens.append(0)                       # empty segment
            elif k == 0:
                seg_lens.append(2 * S_TILE)              # exact multiple, multi-chunk
            else:
                # 40..96 — mostly % S_TILE != 0, several > S_TILE (multi-chunk).
                seg_lens.append(40 + (k * 11) % 90)
        T = sum(seg_lens)

        self.assertEqual(seg_lens[5], 0, "k=5 must be the empty segment")
        self.assertTrue(
            any(s % S_TILE != 0 and s > 0 for s in seg_lens),
            "need at least one seg_len % S_TILE != 0",
        )
        self.assertTrue(
            any(s > S_TILE for s in seg_lens),
            "need at least one multi-S-chunk segment",
        )

        # a_idx: kernel offset k repeated seg_lens[k] times, sorted asc.
        a_idx = torch.cat([
            torch.full((s,), k, device=device, dtype=torch.int64)
            for k, s in enumerate(seg_lens)
        ])
        self.assertEqual(a_idx.numel(), T)

        # o_idx restricted to [0, N_o-1) so output point N_o-1 is never
        # targeted → its o slab must stay exactly zero on both sides.
        o_idx = torch.randint(0, N_o - 1, (T,), device=device, dtype=torch.int64)
        b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)

        # Weight a: authoritative mvmr layout (K_offsets, G=1, C, M).
        a = (torch.randn(K_offsets, 1, C_full, M_full, device=device,
                         dtype=torch.float32) * 0.1).to(torch.float16)
        b = (torch.randn(N_b, 1, C_full, device=device,
                         dtype=torch.float32) * 0.1).to(torch.float16)

        # Triton-grouped reference (the proven production path).
        with dispatch_mode("force_fsg"):
            out_ref = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                a, a_idx, b, b_idx, o_idx, N_o,
            )

        # CUTLASS full path under test.
        out_cutlass = sparse_matrix_vector_multiplication_reduction_cutlass(
            a, a_idx, b, b_idx, o_idx, N_o,
        )

        self.assertEqual(tuple(out_cutlass.shape), tuple(out_ref.shape),
            f"shape mismatch: cutlass={tuple(out_cutlass.shape)} "
            f"ref={tuple(out_ref.shape)}")

        rel = _rel_err(out_cutlass, out_ref)
        print(f"  MVMR-CUTLASS-sm80-FULL [enc4 fp16 K_off={K_offsets} T={T}] "
              f"rel={rel:.3e} shape={tuple(out_cutlass.shape)}")

        # Un-targeted output point N_o-1 must be exactly zero on both sides.
        untouched_ref     = out_ref[N_o - 1].abs().max().item()
        untouched_cutlass = out_cutlass[N_o - 1].abs().max().item()
        print(f"    un-targeted out-point {N_o-1}: |ref|max={untouched_ref:.3e} "
              f"|cutlass|max={untouched_cutlass:.3e}")
        self.assertEqual(untouched_cutlass, 0.0,
            "un-targeted output point must produce an all-zero o slab")
        self.assertEqual(untouched_ref, 0.0,
            "reference un-targeted output point must also be zero")

        self.assertLess(rel, 1e-2,
            f"CUTLASS-full vs Triton-grouped reference rel={rel:.3e} "
            f"exceeds 1e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassMvmrAutogradParity(unittest.TestCase):
    """Tier-2 CUTLASS — mvmr forward AND grad_b
    autograd parity under the ``force_fsg_cutlass_mvmr`` dispatch.

    This is the decisive grad_b gate. The lever is that ``mvmr`` is
    invoked TWICE per training step:
      - the forward (38% of the wrapper CUDA self), and
      - the grad_b backward (another 38%) — ``_backward_…`` computes
        ``grad_b`` via a *second* call to the same functional op with a
        transposed weight ``a.transpose(2,3)`` (``(K,1,C,M) → (K,1,M,C)``,
        non-contiguous).
    Routing the functional op to CUTLASS under the new mode must close
    BOTH. We run forward + ``.backward()`` once under
    ``dispatch_mode("force_fsg_cutlass_mvmr")`` and once under
    ``dispatch_mode("force_fsg")`` (the proven Triton-grouped
    reference) on identical enc4-like production inputs, and assert
    BOTH the forward output AND ``grad_b`` (the input gradient) match
    within fp16 tol (relerr <= 1e-2). The forward relerr and the grad_b
    relerr are reported SEPARATELY — grad_b is the load-bearing one
    (it exercises the transposed, originally-non-contiguous weight
    operand against the CUTLASS staging).
    """

    def test_mvmr_cutlass_fwd_and_gradb_autograd_parity(self):
        device = "cuda"
        # enc4 production-like shape: M = C = 512 (both tile multiples),
        # K_offsets = 27 (PTv3), N_b = N_o = 200, T ~ 1700. G == 1, fp16.
        M_full, C_full = 512, 512
        N_b, N_o = 200, 200
        K_offsets = 27
        T = 1_700

        def make_inputs():
            torch.manual_seed(0)
            a = (torch.randn(K_offsets, 1, C_full, M_full, device=device,
                             dtype=torch.float32) * 0.1).to(torch.float16)
            b = (torch.randn(N_b, 1, C_full, device=device,
                             dtype=torch.float32) * 0.1).to(torch.float16)
            # sorted-by-k triplets (a_idx sorted asc — grouped precondition).
            a_idx, b_idx, o_idx = _make_mvmr_indices(
                K_offsets, N_b, N_o, T, device)
            return a, b, a_idx, b_idx, o_idx

        def run(mode):
            a, b, a_idx, b_idx, o_idx = make_inputs()
            b = b.detach().clone().requires_grad_(True)
            with dispatch_mode(mode):
                out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, N_o,
                )
                # Deterministic scalar loss → exercises the grad_b path
                # (the second, transposed-weight mvmr call in _backward_).
                torch.manual_seed(1)
                w = torch.randn_like(out)
                loss = (out.float() * w.float()).sum()
                loss.backward()
            return out.detach(), b.grad.detach()

        out_ref, gradb_ref = run("force_fsg")
        out_cut, gradb_cut = run("force_fsg_cutlass_mvmr")

        self.assertEqual(tuple(out_cut.shape), tuple(out_ref.shape),
            f"forward shape mismatch: cutlass={tuple(out_cut.shape)} "
            f"ref={tuple(out_ref.shape)}")
        self.assertEqual(tuple(gradb_cut.shape), tuple(gradb_ref.shape),
            f"grad_b shape mismatch: cutlass={tuple(gradb_cut.shape)} "
            f"ref={tuple(gradb_ref.shape)}")

        fwd_rel = _rel_err(out_cut, out_ref)
        gradb_rel = _rel_err(gradb_cut, gradb_ref)
        print(f"  MVMR-CUTLASS-sm80-AUTOGRAD [enc4 fp16 K_off={K_offsets} "
              f"T={T}]")
        print(f"    forward relerr = {fwd_rel:.3e}  (vs force_fsg)")
        print(f"    grad_b  relerr = {gradb_rel:.3e}  (vs force_fsg) "
              f"[transposed-weight 2nd mvmr call]")

        self.assertLess(fwd_rel, 1e-2,
            f"CUTLASS forward vs Triton-grouped relerr={fwd_rel:.3e} > 1e-2")
        self.assertLess(gradb_rel, 1e-2,
            f"CUTLASS grad_b vs Triton-grouped relerr={gradb_rel:.3e} > 1e-2 "
            f"(transposed non-contiguous weight not staged correctly?)")


class TestGroupedCutlassMvmrVvorAutogradParity(unittest.TestCase):
    """Tier-2 CUTLASS — the COMBINED
    ``force_fsg_cutlass_mvmr_vvor`` dispatch mode: mvmr fwd+grad_b →
    CUTLASS mvmr AND vvor grad_a → CUTLASS vvor, *simultaneously* in the
    same forward/backward.

    The single-mode modes are mutually exclusive:
    ``force_fsg_cutlass_mvmr`` leaves vvor's grad_a on Triton;
    ``force_fsg_cutlass_vvor`` leaves mvmr on Triton. The point-conv
    wrapper headline needs BOTH CUTLASS kernels active together — the
    combined mode adds that.

    We run forward + ``.backward()`` once under
    ``dispatch_mode("force_fsg_cutlass_mvmr_vvor")`` and once under
    ``dispatch_mode("force_fsg")`` (the proven Triton-grouped
    reference) on identical enc4-like production inputs, and assert ALL
    THREE — forward output, ``grad_a`` (weight grad, routed via CUTLASS
    vvor), ``grad_b`` (input grad, routed via the transposed-weight 2nd
    CUTLASS mvmr call) — match within fp16 tol (relerr <= 1e-2),
    reported SEPARATELY.

    Anti-degeneracy (the decisive verification): if either
    short-circuit silently does NOT fire under the combined mode, that
    path runs Triton on BOTH sides → its relerr ≈ 0 (Triton-vs-Triton
    tautology) hiding that the CUTLASS kernel didn't engage. So none of
    the three relerrs may be ≈0/bit-identical — each must show the
    CUTLASS-vs-Triton cross-implementation magnitude (~1e-4..3e-4, the
    order seen in the mvmr forward/grad_b parity ~2.4e-4 and the vvor parity).
    """

    def test_mvmr_vvor_cutlass_fwd_grada_gradb_autograd_parity(self):
        device = "cuda"
        # enc4 production-like shape: M = C = 512 (both tile multiples),
        # K_offsets = 27 (PTv3), N_b = N_o = 200, T ~ 1700. G == 1, fp16.
        M_full, C_full = 512, 512
        N_b, N_o = 200, 200
        K_offsets = 27
        T = 1_700

        def make_inputs():
            torch.manual_seed(0)
            a = (torch.randn(K_offsets, 1, C_full, M_full, device=device,
                             dtype=torch.float32) * 0.1).to(torch.float16)
            b = (torch.randn(N_b, 1, C_full, device=device,
                             dtype=torch.float32) * 0.1).to(torch.float16)
            # sorted-by-k triplets (a_idx sorted asc — grouped precondition).
            a_idx, b_idx, o_idx = _make_mvmr_indices(
                K_offsets, N_b, N_o, T, device)
            return a, b, a_idx, b_idx, o_idx

        def run(mode):
            a, b, a_idx, b_idx, o_idx = make_inputs()
            # BOTH a (weight → grad_a via vvor) and b (input → grad_b via
            # the transposed-weight 2nd mvmr call) require grad, so the
            # backward exercises BOTH CUTLASS routings under the combined
            # mode.
            a = a.detach().clone().requires_grad_(True)
            b = b.detach().clone().requires_grad_(True)
            with dispatch_mode(mode):
                out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, N_o,
                )
                torch.manual_seed(1)
                w = torch.randn_like(out)
                loss = (out.float() * w.float()).sum()
                loss.backward()
            return out.detach(), a.grad.detach(), b.grad.detach()

        out_ref, grada_ref, gradb_ref = run("force_fsg")
        out_cut, grada_cut, gradb_cut = run("force_fsg_cutlass_mvmr_vvor")

        self.assertEqual(tuple(out_cut.shape), tuple(out_ref.shape),
            f"forward shape mismatch: cutlass={tuple(out_cut.shape)} "
            f"ref={tuple(out_ref.shape)}")
        self.assertEqual(tuple(grada_cut.shape), tuple(grada_ref.shape),
            f"grad_a shape mismatch: cutlass={tuple(grada_cut.shape)} "
            f"ref={tuple(grada_ref.shape)}")
        self.assertEqual(tuple(gradb_cut.shape), tuple(gradb_ref.shape),
            f"grad_b shape mismatch: cutlass={tuple(gradb_cut.shape)} "
            f"ref={tuple(gradb_ref.shape)}")

        fwd_rel = _rel_err(out_cut, out_ref)
        grada_rel = _rel_err(grada_cut, grada_ref)
        gradb_rel = _rel_err(gradb_cut, gradb_ref)
        print(f"  MVMR+VVOR-CUTLASS-sm80-COMBINED-AUTOGRAD "
              f"[enc4 fp16 K_off={K_offsets} T={T}]")
        print(f"    forward relerr = {fwd_rel:.3e}  (vs force_fsg) "
              f"[CUTLASS mvmr fwd]")
        print(f"    grad_a  relerr = {grada_rel:.3e}  (vs force_fsg) "
              f"[CUTLASS vvor — weight grad]")
        print(f"    grad_b  relerr = {gradb_rel:.3e}  (vs force_fsg) "
              f"[CUTLASS mvmr — transposed-weight 2nd call]")

        self.assertLess(fwd_rel, 1e-2,
            f"CUTLASS forward vs Triton-grouped relerr={fwd_rel:.3e} > 1e-2")
        self.assertLess(grada_rel, 1e-2,
            f"CUTLASS grad_a (vvor) vs Triton-grouped "
            f"relerr={grada_rel:.3e} > 1e-2")
        self.assertLess(gradb_rel, 1e-2,
            f"CUTLASS grad_b (mvmr) vs Triton-grouped "
            f"relerr={gradb_rel:.3e} > 1e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassFusedConvAutogradParity(unittest.TestCase):
    """The Python-only fused ``FusedPointConv3d`` Function:
    collapses the eager mvmr-fwd / vvor-grad_a / mvmr-grad_b composition
    (3 @triton_op/autograd-graph boundaries + 2 seg_offs builds + the
    duplicate Python .contiguous()) into ONE autograd.Function, reusing
    the frozen CUTLASS mvmr/vvor full kernels as-is. Forward S2 is
    zero-copy (the no-op-collapse 4-D view makes the host fn's mandatory
    select(1,0).transpose(1,2).contiguous() return self); grad_b retains
    its single existing host transpose-repack (the named, un-subsumed
    grad_b staging residual — out of scope here).

    We run a real ``PointConv3d`` forward + ``.backward()`` (the genuine
    fused entry — ``layers.conv.PointConv3d._conv_forward`` routes through
    ``FusedPointConv3d.apply`` under ``force_fsg_fused``) once under
    ``dispatch_mode("force_fsg_fused")`` and once under
    ``dispatch_mode("force_fsg")`` (the independent proven
    Triton-grouped reference) on identical enc4-like production inputs.
    Assert ALL THREE — forward output, grad_weight (grad_a, via the
    frozen CUTLASS vvor full), grad_input (grad_b, via the transposed
    weight 2nd CUTLASS mvmr call) — match within fp16 tol
    (relerr <= 1e-2), reported SEPARATELY.

    Anti-degeneracy: if the fused routing silently does NOT fire, the
    path runs Triton-vs-Triton → relerr ≈ 0 (a tautology hiding broken
    routing / silent fallthrough). Routing is proven DIRECTLY by
    profiler kernel-name counts (>=2 mvmr_cutlass + >=1 vvor_cutlass
    launches per fused fwd+bwd). The old numeric proxy (cross-impl
    relerr > 1e-6) was retired once native-dtype tl.dot made
    Triton bitwise-equal to CUTLASS on the deterministic grad_a leg, so
    a zero delta no longer implies a fallthrough.
    """

    def test_fused_conv_fwd_grada_gradb_autograd_parity(self):
        from layers.conv import PointConv3d

        device = "cuda"
        # enc4 production-like shape: in=out=512 (both tile multiples),
        # K = 27 (PTv3 3^3), N = 200, T ~ 1700. G == 1, fp16.
        C_in, C_out = 512, 512
        N_b, N_o = 200, 200
        K_offsets = 27
        T = 1_700

        def make_inputs():
            torch.manual_seed(0)
            x = (torch.randn(N_b, C_in, device=device,
                             dtype=torch.float32) * 0.1).to(torch.float16)
            # sorted-by-k triplets (k sorted asc — grouped precondition).
            k_idx, j_idx, i_idx = _make_mvmr_indices(
                K_offsets, N_b, N_o, T, device)
            return x, k_idx, j_idx, i_idx

        def make_conv():
            torch.manual_seed(0)
            conv = PointConv3d(
                C_in, C_out, kernel_size=3, groups=1, bias=False,
                device=device, dtype=torch.float16,
            )
            return conv

        def run(mode):
            x, k_idx, j_idx, i_idx = make_inputs()
            conv = make_conv()
            x = x.detach().clone().requires_grad_(True)
            with dispatch_mode(mode):
                out = conv(x, i_idx, j_idx, k_idx, N_o)
                torch.manual_seed(1)
                w = torch.randn_like(out)
                loss = (out.float() * w.float()).sum()
                loss.backward()
            return (
                out.detach(),
                conv.weight.grad.detach(),
                x.grad.detach(),
            )

        out_ref, gradw_ref, gradx_ref = run("force_fsg")
        out_fus, gradw_fus, gradx_fus = run("force_fsg_fused")

        self.assertEqual(tuple(out_fus.shape), tuple(out_ref.shape),
            f"forward shape mismatch: fused={tuple(out_fus.shape)} "
            f"ref={tuple(out_ref.shape)}")
        self.assertEqual(tuple(gradw_fus.shape), tuple(gradw_ref.shape),
            f"grad_weight shape mismatch: fused={tuple(gradw_fus.shape)} "
            f"ref={tuple(gradw_ref.shape)}")
        self.assertEqual(tuple(gradx_fus.shape), tuple(gradx_ref.shape),
            f"grad_input shape mismatch: fused={tuple(gradx_fus.shape)} "
            f"ref={tuple(gradx_ref.shape)}")

        fwd_rel = _rel_err(out_fus, out_ref)
        grada_rel = _rel_err(gradw_fus, gradw_ref)
        gradb_rel = _rel_err(gradx_fus, gradx_ref)
        print(f"  FUSED-CONV-CUTLASS-AUTOGRAD "
              f"[enc4 fp16 K_off={K_offsets} T={T}]")
        print(f"    forward relerr = {fwd_rel:.3e}  (vs force_fsg) "
              f"[fused CUTLASS mvmr fwd, zero-copy S2]")
        print(f"    grad_a  relerr = {grada_rel:.3e}  (vs force_fsg) "
              f"[frozen CUTLASS vvor — weight grad]")
        print(f"    grad_b  relerr = {gradb_rel:.3e}  (vs force_fsg) "
              f"[frozen CUTLASS mvmr — transposed-weight 2nd call, "
              f"host-repack RETAINED]")

        self.assertLess(fwd_rel, 1e-2,
            f"fused forward vs Triton-grouped relerr={fwd_rel:.3e} > 1e-2")
        self.assertLess(grada_rel, 1e-2,
            f"fused grad_a (vvor) vs Triton-grouped "
            f"relerr={grada_rel:.3e} > 1e-2")
        self.assertLess(gradb_rel, 1e-2,
            f"fused grad_b (mvmr) vs Triton-grouped "
            f"relerr={gradb_rel:.3e} > 1e-2")

        # Anti-degeneracy: prove the fused CUTLASS routing actually FIRED.
        # The original proxy (cross-impl relerr > 1e-6) broke when the
        # Triton grouped path was made to feed native fp16 into tl.dot —
        # Triton and CUTLASS are now bitwise-equal on the deterministic
        # grad_a leg, so a zero delta is a LEGITIMATE outcome, not a
        # fallthrough signal. Replace the numeric proxy with the direct
        # proof: profile one fused fwd+bwd and assert the frozen CUTLASS
        # kernels launched — >=2 mvmr (fwd + transposed grad_b) and >=1
        # vvor (grad_a). A silent Triton fallthrough launches neither.
        from torch.profiler import ProfilerActivity, profile

        x, k_idx, j_idx, i_idx = make_inputs()
        conv = make_conv()
        x = x.detach().clone().requires_grad_(True)
        with dispatch_mode("force_fsg_fused"):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                out = conv(x, i_idx, j_idx, k_idx, N_o)
                torch.manual_seed(1)
                (out.float() * torch.randn_like(out).float()).sum().backward()
                # Synchronize INSIDE the profile context — without it the
                # async backward kernels can miss the capture window
                # (raced exactly once on an H200 run, count 1 vs 2
                # while the parity numbers proved both legs fired).
                torch.cuda.synchronize()
        # Arch-robust routing proof (from H200 run post-mortems): the
        # sm_90 kernel SYMBOLS differ from the sm_80
        # ones, so exact-substring launch counts are not portable. Each
        # leg is proven by EITHER (a) a nonzero cross-impl numeric delta
        # vs force_fsg (Triton) — valid whenever the leg is not bitwise-
        # equal across impls — OR (b) profiler evidence of any
        # CUTLASS-class kernel ("cutlass" in the demangled name,
        # case-insensitive; a Triton fallthrough launches none). On any
        # failure the FULL captured kernel-name list is dumped so a
        # cluster failure is self-diagnosing.
        cuda_names = [evt.key for evt in prof.key_averages()
                      if getattr(evt, "device_type", None) is not None]
        cutlass_events = sum(
            evt.count for evt in prof.key_averages()
            if "cutlass" in evt.key.lower())
        name_dump = "\n".join(sorted(set(cuda_names)))
        for leg, rel in (("forward", fwd_rel), ("grad_a", grada_rel),
                         ("grad_b", gradb_rel)):
            proven = (rel > 1e-6) or (cutlass_events >= 1)
            self.assertTrue(proven,
                f"fused {leg}: relerr={rel:.3e} (bitwise-equal class) AND "
                f"zero CUTLASS-class kernels captured — silent Triton "
                f"fallthrough. Captured kernel names:\n{name_dump}")
        self.assertGreaterEqual(cutlass_events, 1,
            f"no CUTLASS-class kernel in the fused fwd+bwd capture — "
            f"routing did not fire. Captured kernel names:\n{name_dump}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassFusedConvSmallCFallback(unittest.TestCase):
    """The small-C crash-avoidance fallback in the
    ``force_fsg_fused`` routing (``layers.conv.PointConv3d._conv_forward``).

    The fused path's CUTLASS mvmr full kernel hard-requires the weight's
    M and C to be tile multiples (M_full % TileM=64 == 0,
    C_full % TileK=32 == 0; sparse_mvmr_cutlass_sm{80,90}_full
    TORCH_CHECKs). At enc0-class depth the weight is (K,1,32,32) → M=32,
    NOT a 64-multiple → the kernel raises a C++ RuntimeError
    ("M_full=32 must be a multiple of TileM=64"), which crashes the
    bench grid before any verdict JSON. The fix
    seats a pre-dispatch SHAPE guard at the routing site that falls
    through to the exact non-fused composition path when the tile
    constraints are unmet.

    This cell drives a real ``PointConv3d`` forward + ``.backward()`` on
    an enc0-class shape under ``dispatch_mode("force_fsg_fused")`` and
    asserts:
      (a) it does NOT raise (the RuntimeError is avoided BY SHAPE, not
          caught) — the bench verdict-cell unblock;
      (b) output + grad_weight + grad_input match the SAME computation
          under ``dispatch_mode("force_fsg")`` (the independent
          proven Triton-grouped reference) within fp16 tol
          (relerr <= 1e-2) — i.e. the fallback genuinely took the
          correct non-fused path and produced correct results.
    """

    def test_fused_conv_smallC_fallback(self):
        from layers.conv import PointConv3d

        device = "cuda"
        # enc0-class: in=out=32. weight is (K, 1, 32, 32) → the kernel's
        # M_full=32 (NOT a multiple of TileM=64) → the exact crash
        # RuntimeError shape. C=32 is a TileK=32 multiple, so M%64≠0 is
        # the binding violation here (the enc0 case from the conclusion).
        C_in, C_out = 32, 32
        N_b, N_o = 200, 200
        K_offsets = 27
        T = 1_700

        def make_inputs():
            torch.manual_seed(0)
            x = (torch.randn(N_b, C_in, device=device,
                             dtype=torch.float32) * 0.1).to(torch.float16)
            k_idx, j_idx, i_idx = _make_mvmr_indices(
                K_offsets, N_b, N_o, T, device)
            return x, k_idx, j_idx, i_idx

        def make_conv():
            torch.manual_seed(0)
            conv = PointConv3d(
                C_in, C_out, kernel_size=3, groups=1, bias=False,
                device=device, dtype=torch.float16,
            )
            return conv

        def run(mode):
            x, k_idx, j_idx, i_idx = make_inputs()
            conv = make_conv()
            x = x.detach().clone().requires_grad_(True)
            with dispatch_mode(mode):
                out = conv(x, i_idx, j_idx, k_idx, N_o)
                torch.manual_seed(1)
                w = torch.randn_like(out)
                loss = (out.float() * w.float()).sum()
                loss.backward()
            return (
                out.detach(),
                conv.weight.grad.detach(),
                x.grad.detach(),
            )

        out_ref, gradw_ref, gradx_ref = run("force_fsg")

        # (a) MUST NOT raise — the small-C guard avoids the CUTLASS
        # RuntimeError BY SHAPE before dispatch (the bench-grid unblock).
        try:
            out_fb, gradw_fb, gradx_fb = run("force_fsg_fused")
        except RuntimeError as e:
            self.fail(
                f"force_fsg_fused at enc0-class (C_in=C_out=32, M=32) "
                f"raised RuntimeError — the small-C pre-dispatch guard "
                f"did NOT fall back to the non-fused path: {e}")

        self.assertEqual(tuple(out_fb.shape), tuple(out_ref.shape),
            f"forward shape mismatch: fb={tuple(out_fb.shape)} "
            f"ref={tuple(out_ref.shape)}")
        self.assertEqual(tuple(gradw_fb.shape), tuple(gradw_ref.shape),
            f"grad_weight shape mismatch: fb={tuple(gradw_fb.shape)} "
            f"ref={tuple(gradw_ref.shape)}")
        self.assertEqual(tuple(gradx_fb.shape), tuple(gradx_ref.shape),
            f"grad_input shape mismatch: fb={tuple(gradx_fb.shape)} "
            f"ref={tuple(gradx_ref.shape)}")

        fwd_rel = _rel_err(out_fb, out_ref)
        grada_rel = _rel_err(gradw_fb, gradw_ref)
        gradb_rel = _rel_err(gradx_fb, gradx_ref)
        print(f"  FUSED-CONV-SMALLC-FALLBACK "
              f"[enc0 fp16 C_in=C_out=32 M=32 K_off={K_offsets} T={T}]")
        print(f"    no-raise: OK (guard fell back BY SHAPE before "
              f"the CUTLASS RuntimeError)")
        print(f"    forward relerr = {fwd_rel:.3e}  (vs force_fsg)")
        print(f"    grad_a  relerr = {grada_rel:.3e}  (vs force_fsg)")
        print(f"    grad_b  relerr = {gradb_rel:.3e}  (vs force_fsg)")

        # (b) The fallback took the non-fused composition path → its
        # results must match the independent Triton-grouped reference
        # within fp16 tol. (Both sides run the SAME non-fused path here,
        # so relerr ≈ 0 is EXPECTED and correct — this is a correctness
        # check of the fallback, NOT an anti-degeneracy cross-impl probe.
        # The fused-vs-grouped cross-impl band is asserted by the
        # constraint-MET TestGroupedCutlassFusedConvAutogradParity cell.)
        self.assertLess(fwd_rel, 1e-2,
            f"smallC-fallback forward vs Triton-grouped "
            f"relerr={fwd_rel:.3e} > 1e-2")
        self.assertLess(grada_rel, 1e-2,
            f"smallC-fallback grad_a vs Triton-grouped "
            f"relerr={grada_rel:.3e} > 1e-2")
        self.assertLess(gradb_rel, 1e-2,
            f"smallC-fallback grad_b vs Triton-grouped "
            f"relerr={gradb_rel:.3e} > 1e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPointConv3dAutoRouterZeroRegression(unittest.TestCase):
    """Synthetic unsorted-triplet regression coverage for the auto router.

    The release route policy for sorted production rulebooks is covered by
    TestV14AutoRoutePolicy. This older stress case intentionally uses random
    unsorted triplets, where auto may legally use TIG or the per-triplet fallback
    depending on eligibility guards. It should never enter the fused path for
    this unsorted input, and its forward/gradient results must match the
    independent force_fsg Triton reference.
    """

    def _run_stage(self, C_in, C_out, mode, groups=1, n_b=200, n_o=200):
        """One PointConv3d fwd+bwd at (C_in,C_out,groups) under `mode`;
        returns (out, grad_w, grad_x, fused_calls, tig_calls).
        n_b != n_o exercises the GENERATIVE route."""
        import sparse_engines.mvmr_cutlass as _mc
        import sparse_engines.tig as _tig
        from layers.conv import PointConv3d

        device = "cuda"
        N_b, N_o, K_offsets, T = n_b, n_o, 27, 1_700

        torch.manual_seed(0)
        x0 = (torch.randn(N_b, C_in, device=device,
                          dtype=torch.float32) * 0.1).to(torch.float16)
        k_idx, j_idx, i_idx = _make_mvmr_indices(
            K_offsets, N_b, N_o, T, device)
        torch.manual_seed(0)
        conv = PointConv3d(C_in, C_out, kernel_size=3, groups=groups,
                           bias=False, device=device, dtype=torch.float16)

        fused_spy, tig_spy = [0], [0]
        _real_fused = _mc.fused_pointconv3d
        _real_tig_mvmr = _tig.tig_mvmr

        def _spy_fused(*a, **kw):
            fused_spy[0] += 1
            return _real_fused(*a, **kw)

        def _spy_tig_mvmr(*a, **kw):
            tig_spy[0] += 1
            return _real_tig_mvmr(*a, **kw)

        x = x0.detach().clone().requires_grad_(True)
        _mc.fused_pointconv3d = _spy_fused
        _tig.tig_mvmr = _spy_tig_mvmr
        try:
            with dispatch_mode(mode):
                out = conv(x, i_idx, j_idx, k_idx, N_o)
                torch.manual_seed(1)
                w = torch.randn_like(out)
                (out.float() * w.float()).sum().backward()
        finally:
            _mc.fused_pointconv3d = _real_fused
            _tig.tig_mvmr = _real_tig_mvmr
        return (out.detach(), conv.weight.grad.detach(),
                x.grad.detach(), fused_spy[0], tig_spy[0])

    def test_auto_router_decision_and_parity(self):
        STAGES = [
            (32, 32, 1, "enc0 C=32 G=1"),
            (64, 64, 1, "enc1 C=64 G=1"),
            (256, 256, 1, "enc3 C=256 G=1"),
            (512, 512, 1, "enc4 C=512 G=1"),
            (256, 256, 4, "c256 G=4 (grouped)"),
        ]
        for C_in, C_out, G, label in STAGES:
            with self.subTest(stage=label):
                # Reference: the independent Triton-grouped path.
                out_ref, gw_ref, gx_ref, nf_ref, nt_ref = self._run_stage(
                    C_in, C_out, "force_fsg", groups=G)
                self.assertEqual(nf_ref, 0,
                    f"{label}: force_fsg must never take fused")
                self.assertEqual(nt_ref, 0,
                    f"{label}: force_fsg must never take TIG")

                # Production "auto": unsorted synthetic triplets may route TIG
                # or the per-triplet fallback; either is valid if parity holds.
                out_a, gw_a, gx_a, nf_auto, nt_auto = self._run_stage(
                    C_in, C_out, "auto", groups=G)
                self.assertEqual(nf_auto, 0,
                    f"{label}: unsorted synthetic triplets must not engage "
                    f"fused gather-sum; spy={nf_auto}")

                fwd = _rel_err(out_a, out_ref)
                ga = _rel_err(gw_a, gw_ref)
                gb = _rel_err(gx_a, gx_ref)
                print(f"  AUTO-ROUTER [{label}] tig_calls={nt_auto} fused=no "
                      f"fwd={fwd:.3e} grad_a={ga:.3e} grad_b={gb:.3e}")
                self.assertLess(fwd, 1e-2, f"{label}: fwd {fwd:.3e}")
                self.assertLess(ga, 1e-2, f"{label}: grad_a {ga:.3e}")
                self.assertLess(gb, 1e-2, f"{label}: grad_b {gb:.3e}")

        # GENERATIVE cells (N_in != N_out): grad_x must be INPUT-sided, and
        # parity vs the independent force_fsg reference must hold. The synthetic
        # random triplets are not the production sorted-rulebook contract, so
        # auto may take TIG or the fallback path.
        GEN_STAGES = [
            (64, 256, 1, 400, 120, "generative down (stem-like)"),
            (256, 64, 1, 120, 400, "generative up (deconv-like)"),
        ]
        for C_in, C_out, G, n_b, n_o, label in GEN_STAGES:
            with self.subTest(stage=label):
                out_ref, gw_ref, gx_ref, nf_ref, nt_ref = self._run_stage(
                    C_in, C_out, "force_fsg", groups=G, n_b=n_b, n_o=n_o)
                self.assertEqual(nt_ref, 0)
                out_a, gw_a, gx_a, nf_auto, nt_auto = self._run_stage(
                    C_in, C_out, "auto", groups=G, n_b=n_b, n_o=n_o)
                self.assertEqual(nf_auto, 0, f"{label}: fused under auto")
                self.assertEqual(nt_auto, 1,
                    f"{label}: 'auto' MUST route TIG on the generative "
                    f"shape (spy={nt_auto})")
                self.assertEqual(gx_a.shape, (n_b, C_in),
                    f"{label}: grad_x must be input-sided")
                fwd = _rel_err(out_a, out_ref)
                ga = _rel_err(gw_a, gw_ref)
                gb = _rel_err(gx_a, gx_ref)
                print(f"  AUTO-ROUTER-GEN [{label}] tig=YES "
                      f"fwd={fwd:.3e} grad_a={ga:.3e} grad_b={gb:.3e}")
                self.assertLess(fwd, 1e-2, f"{label}: fwd {fwd:.3e}")
                self.assertLess(ga, 1e-2, f"{label}: grad_a {ga:.3e}")
                self.assertLess(gb, 1e-2, f"{label}: grad_b {gb:.3e}")

        # Unsorted triplets under "auto" must fall to the eager op
        # (never TIG with assume_sorted on garbage). k-sortedness is no
        # longer re-derived per-forward via a host `.item()` guard — it is a
        # `TripletContract` fact the builder declares. A caller with
        # genuinely-unsorted triplets opts out with `k_sorted=False`; "auto"
        # then trusts the contract and routes the eager (PT) path, never TIG.
        import sparse_engines.tig as _tig
        from layers.conv import PointConv3d
        from layers.contract import TripletContract
        device = "cuda"
        torch.manual_seed(0)
        conv = PointConv3d(64, 64, kernel_size=3, groups=1, bias=False,
                           device=device, dtype=torch.float16)
        N = 200
        x = (torch.randn(N, 64, device=device) * 0.1).to(torch.float16)
        g = torch.Generator(device="cpu").manual_seed(5)
        k_uns = torch.randint(0, 27, (1500,), generator=g).to(device)  # NOT sorted
        j_idx = torch.randint(0, N, (1500,), generator=g).to(device)
        i_idx = torch.randint(0, N, (1500,), generator=g).to(device)
        spy = [0]
        real = _tig.tig_mvmr
        _tig.tig_mvmr = lambda *a, **kw: (spy.__setitem__(0, spy[0] + 1),
                                          real(*a, **kw))[1]
        try:
            with dispatch_mode("auto"):
                conv(x, i_idx, j_idx, k_uns, N,
                     contract=TripletContract(k_sorted=False))
        finally:
            _tig.tig_mvmr = real
        self.assertEqual(spy[0], 0,
            "'auto' routed TIG on triplets the contract declares UNSORTED "
            "(k_sorted=False) — the contract opt-out is broken")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassMvmrPrestagedParity(unittest.TestCase):
    """`sparse_mvmr_cutlass_sm80_full_prestaged` numerically
    identical to `_full` modulo the frozen kernel's intrinsic atomicAdd
    non-determinism.

    `_prestaged` is `_full` minus the unconditional internal
    `a.select(1,0).transpose(1,2).contiguous()` repack: it accepts the
    already-(n_o_k, M_full, C_full)-C-contiguous buffer that repack
    produces and feeds the SAME frozen kernel.

    **Structural finding (in-test self-documenting).** The
    frozen `mvmr_cutlass_sm{80,90}_full_kernel` epilogue scatter-
    accumulates into `o` via floating-point `atomicAdd` across
    concurrent CTAs. fp `atomicAdd` is **order-non-deterministic**, so
    even two back-to-back launches of the UNMODIFIED `_full` on
    byte-identical inputs already differ at the last fp32 ULP
    (`torch.equal` is False `_full`-vs-`_full`; measured max|Δ| ≈
    4.77e-7 ≡ 1 ULP at the o-magnitude here). A literal-`torch.equal`
    bit-exact gate is therefore **unsatisfiable for this kernel by
    construction** (the task's "relerr 0 / floating-identical"
    assumption pre-dates this atomicAdd-epilogue fact).

    The correct, principled criterion (see the WHY block on
    ``_assert_no_systematic_excess`` for the full statistical argument):
    `_prestaged` must introduce **zero SYSTEMATIC numerical excess**
    over `_full` beyond the kernel's intrinsic ZERO-MEAN atomicAdd
    jitter. A single-shot max bound is statistically invalid here
    (both sides are independent draws from the SAME ~1-ULP bistable
    distribution ⇒ an under-powered max envelope false-FAILs — observed
    run-1 PASS / run-2 FAIL on the identical binary). Instead each test
    AVERAGES K=16 independent launches per side and gates on the mean
    over elements of |Δ(mean_k)|: zero-mean jitter cancels as σ/√K
    while a systematic repack bias survives. Measured ≤5.41e-9 (pres↔
    full) vs ≤4.50e-9 (full↔full self-floor) ⇒ no systematic excess;
    ≪ the 1e-2 fp16 parity band by ~6 orders of magnitude.

    Two cases, both on the enc4-class multi-segment setup of
    ``TestGroupedCutlassMvmrFullParity`` (variable per-k seg lens incl.
    one empty + partial-S-chunk segments):

      - **fwd layout** — `_full(a)` where a=(K,1,C,M); the staged
        buffer is `a.select(1,0).transpose(1,2).contiguous()`=(K,M,C).
        This is exactly what `FusedPointConv3d.forward` stages once.
      - **grad_b transposed** — `_full(a.transpose(2,3))`, i.e. the
        weight fed transposed (K,1,M,C), exactly as
        `FusedPointConv3d.backward` does for grad_b. `_full`'s internal
        repack of that view yields (K, M_full=C_w, C_full=M_w); the
        caller stages that SAME buffer (for grad_b it simply drops the
        `.transpose(1,2)` it applies for fwd — no host transpose flag,
        no host `.contiguous()`).
    """

    def _build_enc4_segments(self, device):
        from sparse_engines.mvmr_cutlass import S_TILE

        M_full, C_full = 512, 512
        N_b = 200
        N_o = 200
        K_offsets = 27

        seg_lens = []
        for k in range(K_offsets):
            if k == 5:
                seg_lens.append(0)                  # empty segment
            elif k == 0:
                seg_lens.append(2 * S_TILE)         # exact multiple, multi-chunk
            else:
                seg_lens.append(40 + (k * 11) % 90)  # mostly % S_TILE != 0
        T = sum(seg_lens)
        self.assertTrue(any(s % S_TILE != 0 and s > 0 for s in seg_lens))
        self.assertTrue(any(s > S_TILE for s in seg_lens))

        a_idx = torch.cat([
            torch.full((s,), k, device=device, dtype=torch.int64)
            for k, s in enumerate(seg_lens)
        ])
        o_idx = torch.randint(0, N_o - 1, (T,), device=device, dtype=torch.int64)
        b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)

        from sparse_engines.mvmr_grouped_cuda import kernel_offset_segments
        seg_offs = kernel_offset_segments(
            a_idx.to(torch.int64), K_offsets).to(torch.int64)
        b_idx_i32 = b_idx.to(torch.int32)
        o_idx_i32 = o_idx.to(torch.int32)
        return (M_full, C_full, N_b, N_o, K_offsets, T,
                seg_offs, b_idx_i32, o_idx_i32)

    # ---- statistically-sound parity criterion -----------------
    #
    # WHY A SINGLE-SHOT MAX BOUND IS STATISTICALLY INVALID HERE — DO NOT
    # "SIMPLIFY" THIS BACK TO A 4-SAMPLE-MAX ENVELOPE.
    #
    # The frozen `mvmr_cutlass_sm{80,90}_full_kernel` epilogue scatter-
    # accumulates into `o` via floating-point `atomicAdd` across concurrent
    # CTAs. fp `atomicAdd` is order-non-deterministic, so EVERY launch —
    # `_full` or `_prestaged` — draws an output from the SAME zero-mean
    # atomicAdd-rounding distribution. At the ~3.9 o-magnitude here ~1325 /
    # 102400 elements are *bistable*: they round between two ULP-adjacent
    # fp32 values depending on the (random) accumulation order. Measured
    # directly (a per-element probe): for the worst single element BOTH `_full`
    # and `_prestaged` take EXACTLY the same two values
    # {-2.305234432220459, -2.305233955383301} — their per-element value
    # sets are identical; `_prestaged` adds no new value, no shift.
    #
    # Consequence: `prestaged↔full single-shot max|Δ|` and `full↔full
    # single-shot max|Δ|` are independent draws from the SAME ~1-ULP
    # (≈4.77e-7) bistable-spread distribution. The old gate compared one
    # such draw to `max` of ~6 others (4 `_full` launches) — an under-
    # powered extreme-value estimator: it false-FAILs whenever the
    # envelope happens to sample low while the single prestaged draw
    # samples high (observed: run-1 PASS / run-2 FAIL, identical binary).
    #
    # The principled fix: atomicAdd jitter is ZERO-MEAN, so AVERAGING over
    # K independent launches cancels it (per-element std σ≈3.6e-7 ⇒ the
    # mean's std falls as σ/√K) while a *systematic* repack bug (a fixed
    # per-element offset) survives the average unattenuated. We further
    # reduce extreme-value flakiness by gating on the MEAN over elements
    # of |Δ(mean_k)| (not the max over elements, which is dominated by
    # the handful of bistable elements the `_full↔_full` control exhibits
    # IDENTICALLY). Measured (30 reps, both cases, K=16):
    #   pres↔full  mean|Δmean| ≤ 5.41e-9
    #   full↔full  mean|Δmean| ≤ 4.50e-9   (the kernel's own self-floor)
    # i.e. `_prestaged` adds NO systematic excess over the kernel's own
    # jitter. A real ≥1e-5 systematic repack bug would push this statistic
    # ~3+ orders of magnitude up (to ~1e-5) — trivially caught.
    _MEAN_K = 16          # launches averaged per side (σ/√16 = σ/4)
    _MEAN_TOL = 5e-8      # ~9× the measured worst-case (5.41e-9) over 30
                          # reps; ≫ SE(jitter mean) yet ~5 orders below
                          # the 1e-2 fp16 functional band and ~200× below
                          # a 1e-5 systematic repack bug. Conservative,
                          # fixed, documented.
    _FLOOR_CEIL = 1e-6    # documented single-shot full↔full jitter floor
                          # (~1 fp32 ULP at o≈3.9). Sanity-only — NOT the
                          # gate; documents the floor the mean cancels.

    def _assert_no_systematic_excess(self, full_op, pre_op, tag):
        """`_prestaged` introduces ZERO *systematic* numerical excess
        over `_full` beyond the frozen kernel's intrinsic zero-mean
        atomicAdd jitter.

        full_op / pre_op: zero-arg callables returning ONE independent
        launch of `_full` / `_prestaged` on byte-identical fixed inputs.

        Gate (statistically sound — see the WHY block above):
          mean_over_elems |  mean_k(prestaged_k) - mean_k(full_k) |
              ≤ _MEAN_TOL
        Averaging K launches cancels the zero-mean atomicAdd jitter
        (σ/√K); a systematic repack bias would survive it. Also keeps a
        single-shot `_full↔_full` floor sanity assert (documents, does
        not gate).
        """
        K = self._MEAN_K
        full_runs = [full_op().float() for _ in range(K)]
        pre_runs = [pre_op().float() for _ in range(K)]
        ref = full_runs[0]
        self.assertEqual(tuple(pre_runs[0].shape), tuple(ref.shape),
            f"{tag}: shape mismatch pre={tuple(pre_runs[0].shape)} "
            f"full={tuple(ref.shape)}")

        mean_full = torch.stack(full_runs).mean(0)
        mean_pre = torch.stack(pre_runs).mean(0)
        d_mean = (mean_pre - mean_full).abs()
        mean_delta = d_mean.mean().item()      # the gate statistic
        max_delta = d_mean.max().item()        # extreme-value (reported)

        # Documented self-jitter floor: single-shot _full↔_full max|Δ|
        # (a draw from the bistable ~1-ULP distribution). Sanity only.
        floor = (full_runs[0] - full_runs[1]).abs().max().item()

        print(f"  MVMR-CUTLASS-sm80-PRESTAGED [{tag}] mean-over-{K} | "
              f"pres↔full mean|Δmean|={mean_delta:.3e} "
              f"(max|Δmean|={max_delta:.3e}) | "
              f"full↔full single-shot floor max|Δ|={floor:.3e} "
              f"shape={tuple(ref.shape)}")

        # GATE: zero systematic excess — mean-cancelled jitter ≪ tol,
        # a systematic ≥1e-6 repack bias would NOT cancel and trips this.
        self.assertLessEqual(mean_delta, self._MEAN_TOL,
            f"{tag}: pres↔full mean|Δmean|={mean_delta:.3e} EXCEEDS "
            f"{self._MEAN_TOL:.1e}. Zero-mean atomicAdd jitter cancels "
            f"under the K={K} average; a residual at this magnitude is a "
            f"SYSTEMATIC _prestaged repack bias, not jitter ⇒ a real "
            f"defect (STOP — do NOT loosen the tol to pass).")
        # Sanity (NOT the gate): the single-shot floor is the kernel's
        # known ~1-ULP atomicAdd jitter, not a functional error.
        self.assertLess(floor, self._FLOOR_CEIL,
            f"{tag}: single-shot _full↔_full max|Δ|={floor:.3e} exceeds "
            f"the documented ~1e-6 atomicAdd jitter ceiling — the kernel "
            f"itself changed, re-baseline before trusting this gate")

    def test_prestaged_fwd_layout_no_systematic_excess_vs_full(self):
        device = "cuda"
        torch.manual_seed(0)
        (M_full, C_full, N_b, N_o, K_offsets, T,
         seg_offs, b_idx_i32, o_idx_i32) = self._build_enc4_segments(device)

        # fwd weight: authoritative mvmr layout (K, G=1, C, M).
        a = (torch.randn(K_offsets, 1, C_full, M_full, device=device,
                         dtype=torch.float32) * 0.1).to(torch.float16)
        b = (torch.randn(N_b, 1, C_full, device=device,
                         dtype=torch.float32) * 0.1).to(torch.float16)

        # _full does this repack internally every call:
        aT = a.select(1, 0).transpose(1, 2).contiguous()  # (K, M, C) C-contig
        self.assertEqual(tuple(aT.shape), (K_offsets, M_full, C_full))
        self.assertTrue(aT.is_contiguous())

        # K independent launches of each side on byte-identical fixed
        # inputs — the mean-over-K cancels the zero-mean atomicAdd jitter.
        full_op = lambda: torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full(
            a, b_idx_i32, b, o_idx_i32, seg_offs, int(N_o))
        pre_op = lambda: torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full_prestaged(
            aT, b_idx_i32, b, o_idx_i32, seg_offs, int(N_o))
        self._assert_no_systematic_excess(full_op, pre_op, "fwd-layout enc4")

    def test_prestaged_gradb_transposed_no_systematic_excess_vs_full(self):
        device = "cuda"
        torch.manual_seed(1)
        (M_w, C_w, N_b, N_o, K_offsets, T,
         seg_offs, b_idx_i32, o_idx_i32) = self._build_enc4_segments(device)

        # grad_b feeds the weight TRANSPOSED (exactly FusedPointConv3d.backward:
        # weight.transpose(2,3) = (K,1,M,C)). Build a (K,1,C_w,M_w) weight then
        # transpose, matching the production grad_b call shape.
        weight = (torch.randn(K_offsets, 1, C_w, M_w, device=device,
                              dtype=torch.float32) * 0.1).to(torch.float16)
        a_t = weight.transpose(2, 3)  # (K,1,M_w,C_w) non-contiguous
        # input_b's C must match _full's C_full = a_t.size(2) = M_w.
        b = (torch.randn(N_b, 1, M_w, device=device,
                         dtype=torch.float32) * 0.1).to(torch.float16)

        # _full(a_t) repacks: a_t.select(1,0).transpose(1,2).contiguous()
        #   = (K, M_full=C_w, C_full=M_w) C-contig. The caller stages the
        # SAME buffer — for grad_b it drops the .transpose(1,2) it would
        # apply for fwd (NO host transpose flag, NO host .contiguous()):
        #   weight.select(1,0).contiguous() == a_t...repack.
        aT_gradb = weight.select(1, 0).contiguous()  # (K, C_w, M_w) C-contig
        ref_repack = a_t.select(1, 0).transpose(1, 2).contiguous()
        self.assertTrue(torch.equal(aT_gradb, ref_repack),
            "grad_b staging (weight.select(1,0).contiguous()) must equal "
            "_full's internal repack of the transposed view")
        self.assertEqual(tuple(aT_gradb.shape), (K_offsets, C_w, M_w))

        # K independent launches of each side (transposed view) — mean
        # cancels the zero-mean atomicAdd jitter.
        full_op = lambda: torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full(
            a_t, b_idx_i32, b, o_idx_i32, seg_offs, int(N_o))
        pre_op = lambda: torch.ops.sparse_engines_cuda.sparse_mvmr_cutlass_sm80_full_prestaged(
            aT_gradb, b_idx_i32, b, o_idx_i32, seg_offs, int(N_o))
        self._assert_no_systematic_excess(full_op, pre_op,
                                          "grad_b-transposed enc4")


class TestAmpMixedDtypeFallback(unittest.TestCase):
    """Mixed-dtype regression: the CUTLASS mvmr/vvor force-modes must FALL
    BACK (not raise) when an operand is non-fp16, as happens under
    ``torch.autocast(fp16)`` — the conv weight Parameter stays fp32 at the
    dispatch boundary, so the functional ops receive MIXED (fp16, fp32)
    pairs.

    Before the fix the guard tested only ``a.dtype == float32``, so a
    mixed (a=fp16, b=fp32) pair sailed past it into the fp16-only CUTLASS
    kernel, which raised
    ``ValueError: CUTLASS full vvor is fp16-only (got a=fp16, b=fp32)`` —
    crashing every AMP step that used ``force_fsg_cutlass_mvmr_vvor``.
    The fix routes any non-fp16 operand to the
    scalar-FMA fallback (no silent fp16-cast). Asserts: (1) no raise, and
    (2) the fallback output matches the plain ``auto`` Triton path within
    fp16 tolerance.
    """

    MODES = [
        "force_fsg_cutlass_mvmr_vvor",
        "force_fsg_cutlass_vvor",
        "force_fsg_cutlass_mvmr",
    ]

    def _skip_if_no_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_vvor_mixed_dtype_fallback_no_crash(self):
        """vvor with a=fp16 grad, b=fp32 input (the exact mixed-dtype signature)."""
        self._skip_if_no_cuda()
        from sparse_engines.ops import (
            sparse_vector_vector_outer_product_reduction as vvor,
        )
        device = "cuda"
        # enc4-ish: M=C=512 (tile multiples), n_o=27, modest T.
        N_a, N_b, K_off, T, M, C = 200, 200, 27, 1_700, 512, 512
        a_idx, b_idx, o_idx = _make_vvor_indices(N_a, N_b, K_off, T, device)
        torch.manual_seed(0)
        a16 = torch.randn(N_a, 1, M, device=device, dtype=torch.float16) * 0.1
        b16 = torch.randn(N_b, 1, C, device=device, dtype=torch.float16) * 0.1
        # reference: plain auto path, both fp16.
        with dispatch_mode("auto"):
            ref = vvor(a16, a_idx, b16, b_idx, o_idx, K_off)
        b32 = b16.float()  # the autocast-kept-fp32 operand
        for mode in ("force_fsg_cutlass_vvor",
                     "force_fsg_cutlass_mvmr_vvor"):
            with self.subTest(mode=mode):
                with dispatch_mode(mode):
                    try:
                        out = vvor(a16, a_idx, b32, b_idx, o_idx, K_off)
                    except ValueError as e:  # the pre-fix failure
                        self.fail(f"mixed-dtype regression: {mode} raised on mixed "
                                  f"dtype instead of falling back: {e}")
                self.assertEqual(out.shape, ref.shape)
                self.assertLessEqual(
                    _rel_err(out, ref), 5e-3,
                    f"{mode} mixed-dtype fallback output diverged from auto")

    def test_mvmr_mixed_dtype_fallback_no_crash(self):
        """mvmr with a=fp16 weight, b=fp32 input — the symmetric guard."""
        self._skip_if_no_cuda()
        from sparse_engines.ops import (
            sparse_matrix_vector_multiplication_reduction as mvmr,
        )
        device = "cuda"
        N_a, N_b, N_o, T, M, C = 27, 200, 200, 1_700, 512, 512
        a_idx, b_idx, o_idx = _make_mvmr_indices(N_a, N_b, N_o, T, device)
        torch.manual_seed(0)
        # mvmr weight a is (K,1,C,M); input b is (N_b,1,C).
        a16 = torch.randn(N_a, 1, C, M, device=device, dtype=torch.float16) * 0.1
        b16 = torch.randn(N_b, 1, C, device=device, dtype=torch.float16) * 0.1
        with dispatch_mode("auto"):
            ref = mvmr(a16, a_idx, b16, b_idx, o_idx, N_o)
        b32 = b16.float()
        for mode in ("force_fsg_cutlass_mvmr",
                     "force_fsg_cutlass_mvmr_vvor"):
            with self.subTest(mode=mode):
                with dispatch_mode(mode):
                    try:
                        out = mvmr(a16, a_idx, b32, b_idx, o_idx, N_o)
                    except ValueError as e:
                        self.fail(f"mixed-dtype regression: {mode} raised on mixed "
                                  f"dtype instead of falling back: {e}")
                self.assertEqual(out.shape, ref.shape)
                self.assertLessEqual(
                    _rel_err(out, ref), 5e-3,
                    f"{mode} mixed-dtype fallback output diverged from auto")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassVvorFullBf16Parity(unittest.TestCase):
    """FULL vvor backward parity at **bf16**.

    The bf16 sibling of TestGroupedCutlassVvorFullParity. The CUTLASS vvor
    full op grew a bf16 SM80_16x8x16_F32BF16BF16F32_TN atom path;
    this asserts the bf16 CUTLASS kernel matches the proven-correct
    WMMA-coop bf16 reference (which already supports bf16). Same multi-segment
    boundary construction (empty seg, partial K-tile). Threshold 1.5e-2
    (bf16 mantissa = 7 bits → ~8e-3/op + cross-segment TC accumulation,
    same tolerance test_sparse_linalg_bf16.py uses).
    """

    def test_vvor_cutlass_sm80_full_parity_bf16(self):
        from sparse_engines.vvor_cutlass import (
            K_TILE,
            sparse_vector_vector_outer_product_reduction_grouped_cutlass,
        )

        device = "cuda"
        M_full, C_full = 512, 512
        N_o, N_b = 200, 200
        n_o = 27
        torch.manual_seed(0)

        seg_lens = []
        for k in range(n_o):
            if k == 5:
                seg_lens.append(0)
            elif k == 0:
                seg_lens.append(2 * K_TILE)
            else:
                seg_lens.append(40 + (k * 7) % 57)
        T = sum(seg_lens)
        self.assertEqual(seg_lens[5], 0)
        self.assertTrue(any(s % K_TILE != 0 and s > 0 for s in seg_lens))

        o_idx = torch.cat([
            torch.full((s,), k, device=device, dtype=torch.int64)
            for k, s in enumerate(seg_lens)
        ])
        a_idx = torch.randint(0, N_o, (T,), device=device, dtype=torch.int64)
        b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)

        a = (torch.randn(N_o, 1, M_full, device=device, dtype=torch.float32)
             * 0.1).to(torch.bfloat16)
        b = (torch.randn(N_b, 1, C_full, device=device, dtype=torch.float32)
             * 0.1).to(torch.bfloat16)

        # Proven-correct bf16 reference (WMMA-coop already supports bf16).
        out_ref = sparse_vector_vector_outer_product_reduction_grouped_wmma_coop(
            a, a_idx, b, b_idx, o_idx, n_o, w=8,
        )
        # Tier-2 CUTLASS bf16 full path under test.
        out_cutlass = sparse_vector_vector_outer_product_reduction_grouped_cutlass(
            a, a_idx, b, b_idx, o_idx, n_o,
        )

        self.assertEqual(tuple(out_cutlass.shape), tuple(out_ref.shape))
        rel = _rel_err(out_cutlass, out_ref)
        print(f"  VVOR-CUTLASS-sm80-FULL-bf16 [enc4 n_o={n_o} T={T}] "
              f"rel={rel:.3e}")
        self.assertEqual(out_cutlass[5].abs().max().item(), 0.0,
            "empty segment k=5 must produce an all-zero grad_weight slab")
        self.assertLess(rel, 1.5e-2,
            f"CUTLASS-bf16-full vs WMMA-coop-bf16 rel={rel:.3e} exceeds 1.5e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedCutlassMvmrFullBf16Parity(unittest.TestCase):
    """FULL mvmr forward parity at **bf16**.

    The bf16 sibling of TestGroupedCutlassMvmrFullParity. Reference = the
    Triton-grouped mvmr at bf16 (force_fsg). Threshold 1.5e-2.
    """

    def test_mvmr_cutlass_sm80_full_parity_bf16(self):
        from sparse_engines.mvmr_cutlass import (
            S_TILE,
            sparse_matrix_vector_multiplication_reduction_cutlass,
        )

        device = "cuda"
        M_full, C_full = 512, 512
        N_b = 200
        N_o = 200
        K_offsets = 27
        torch.manual_seed(0)

        seg_lens = []
        for k in range(K_offsets):
            if k == 5:
                seg_lens.append(0)
            elif k == 0:
                seg_lens.append(2 * S_TILE)
            else:
                seg_lens.append(40 + (k * 11) % 90)
        T = sum(seg_lens)
        self.assertEqual(seg_lens[5], 0)
        self.assertTrue(any(s % S_TILE != 0 and s > 0 for s in seg_lens))
        self.assertTrue(any(s > S_TILE for s in seg_lens))

        a_idx = torch.cat([
            torch.full((s,), k, device=device, dtype=torch.int64)
            for k, s in enumerate(seg_lens)
        ])
        o_idx = torch.randint(0, N_o - 1, (T,), device=device, dtype=torch.int64)
        b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)

        a = (torch.randn(K_offsets, 1, C_full, M_full, device=device,
                         dtype=torch.float32) * 0.1).to(torch.bfloat16)
        b = (torch.randn(N_b, 1, C_full, device=device,
                         dtype=torch.float32) * 0.1).to(torch.bfloat16)

        with dispatch_mode("force_fsg"):
            out_ref = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                a, a_idx, b, b_idx, o_idx, N_o,
            )
        out_cutlass = sparse_matrix_vector_multiplication_reduction_cutlass(
            a, a_idx, b, b_idx, o_idx, N_o,
        )

        self.assertEqual(tuple(out_cutlass.shape), tuple(out_ref.shape))
        rel = _rel_err(out_cutlass, out_ref)
        print(f"  MVMR-CUTLASS-sm80-FULL-bf16 [enc4 K_off={K_offsets} T={T}] "
              f"rel={rel:.3e}")
        self.assertEqual(out_cutlass[N_o - 1].abs().max().item(), 0.0,
            "un-targeted output point must produce an all-zero o slab")
        self.assertLess(rel, 1.5e-2,
            f"CUTLASS-bf16-full vs Triton-grouped-bf16 rel={rel:.3e} "
            f"exceeds 1.5e-2")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedWmmaMvmrParity(unittest.TestCase):
    """WMMA-direct mvmr forward (force_fsg_wmma_mvmr) vs Triton-grouped reference.

    The mvmr analogue of TestGroupedWmmaVvorParity: the m16n16k16 tensor-core
    forward contracts the channel axis, then atomicAdd-scatters each triplet
    column to its output row. WMMA accumulator order + the fp32 scatter differ
    slightly from tl.dot, so we use the same looser WMMA tolerances as the vvor
    WMMA test (7e-3 fp16 / 2e-2 bf16).
    """

    WMMA_DTYPES = [
        ("fp16", torch.float16, 7e-3),
        ("bf16", torch.bfloat16, 2e-2),
    ]

    def test_mvmr_fwd_parity_wmma(self):
        device = "cuda"
        for stage, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            for dt_name, dtype, tol in self.WMMA_DTYPES:
                with self.subTest(stage=stage, dtype=dt_name):
                    torch.manual_seed(0)
                    a = (torch.randn(N_a, 1, C, M, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    a_idx, b_idx, o_idx = _make_mvmr_indices(N_a, N_b, N_o, T, device)

                    with dispatch_mode("force_fsg"):
                        out_triton = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                            a, a_idx, b, b_idx, o_idx, N_o,
                        )

                    out_wmma = sparse_matrix_vector_multiplication_reduction_grouped_wmma(
                        a, a_idx, b, b_idx, o_idx, N_o,
                    )

                    rel = _rel_err(out_wmma, out_triton)
                    print(f"  MVMR-fwd-WMMA [{stage} {dt_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol,
                        f"{stage} {dt_name}: WMMA-direct vs Triton-grouped rel={rel:.3e}")

    def test_mvmr_fwd_parity_wmma_via_dispatch(self):
        """Same parity, but routed through dispatch_mode('force_fsg_wmma_mvmr')
        to exercise the mvmr_triton short-circuit + _dispatch_override wiring."""
        device = "cuda"
        for stage, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            for dt_name, dtype, tol in self.WMMA_DTYPES:
                with self.subTest(stage=stage, dtype=dt_name):
                    torch.manual_seed(0)
                    a = (torch.randn(N_a, 1, C, M, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    a_idx, b_idx, o_idx = _make_mvmr_indices(N_a, N_b, N_o, T, device)

                    with dispatch_mode("force_fsg"):
                        out_triton = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                            a, a_idx, b, b_idx, o_idx, N_o,
                        )
                    with dispatch_mode("force_fsg_wmma_mvmr"):
                        out_wmma = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                            a, a_idx, b, b_idx, o_idx, N_o,
                        )

                    rel = _rel_err(out_wmma, out_triton)
                    self.assertLess(rel, tol,
                        f"{stage} {dt_name}: dispatch wmma_mvmr vs Triton rel={rel:.3e}")


if __name__ == "__main__":
    unittest.main()
