"""Edge-case correctness for MVMR / VVOR.

Probes paths the dispatcher must not crash on:
  - T = 0 (empty triplet array)
  - T = 1 (single triplet — falls below grouped threshold)
  - all triplets sharing one kernel offset (single non-empty segment,
    K-1 empty segments)
  - sort_by ≠ k (a_idx not sorted ascending → grouped refuses → must
    fall back without exception)
  - Group conv (G > 1) — grouped tl.dot path requires G=1; falls back
  - Forced-grouped path on a tiny input (correctness probe with
    threshold bypass)

Each is run in fp32, fp16, bf16. The expected behaviour is
correctness preserved across all paths; perf is irrelevant here.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import numpy as np
import torch
import sparse_engines
from sparse_engines._dispatch_override import dispatch_mode

from test_sparse_linalg_fp16 import mvmr_numpy_ref, vvor_numpy_ref


DTYPES = [
    ("fp32", torch.float32, 5e-3),
    ("fp16", torch.float16, 5e-3),
    ("bf16", torch.bfloat16, 1.5e-2),
]


def _rel_err(triton_out, ref_np):
    ref = torch.tensor(ref_np, device=triton_out.device)
    diff = (triton_out.float() - ref).abs().max().item()
    base = ref.abs().max().item()
    return diff / max(base, 1e-6)


class TestEdgeCases(unittest.TestCase):

    # ── T = 0 ──

    def test_T_zero_mvmr(self):
        """Empty triplet array: grouped should refuse (total_chunks=0),
        per-triplet kernel should produce zeros of shape (n_o, G, M)."""
        for dtype_name, dtype, _ in DTYPES:
            with self.subTest(dtype=dtype_name):
                device = "cuda"
                a = torch.randn(27, 1, 16, 16, device=device, dtype=dtype)
                b = torch.randn(8, 1, 16, device=device, dtype=dtype)
                a_idx = torch.empty(0, device=device, dtype=torch.int64)
                b_idx = torch.empty(0, device=device, dtype=torch.int64)
                o_idx = torch.empty(0, device=device, dtype=torch.int64)

                out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, 8,
                )
                self.assertEqual(out.shape, (8, 1, 16))
                self.assertEqual(out.abs().max().item(), 0.0)

    def test_T_zero_vvor(self):
        for dtype_name, dtype, _ in DTYPES:
            with self.subTest(dtype=dtype_name):
                device = "cuda"
                a = torch.randn(27, 1, 16, device=device, dtype=dtype)
                b = torch.randn(8, 1, 16, device=device, dtype=dtype)
                a_idx = torch.empty(0, device=device, dtype=torch.int64)
                b_idx = torch.empty(0, device=device, dtype=torch.int64)
                o_idx = torch.empty(0, device=device, dtype=torch.int64)

                out = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
                    a, a_idx, b, b_idx, o_idx, 27,
                )
                self.assertEqual(out.shape, (27, 1, 16, 16))
                self.assertEqual(out.abs().max().item(), 0.0)

    # ── T = 1 ──

    def test_T_one_mvmr(self):
        """Single triplet: T/K = 1/27 < 16 → grouped refuses, per-triplet
        path runs. Output should match the explicit single-triplet matmul."""
        for dtype_name, dtype, tol in DTYPES:
            with self.subTest(dtype=dtype_name):
                device = "cuda"
                torch.manual_seed(0)
                a = (torch.randn(27, 1, 16, 16, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                b = (torch.randn(8, 1, 16, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                a_idx = torch.tensor([5], device=device, dtype=torch.int64)
                b_idx = torch.tensor([3], device=device, dtype=torch.int64)
                o_idx = torch.tensor([2], device=device, dtype=torch.int64)

                out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, 8,
                )
                ref = mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, 8)
                rel = _rel_err(out, ref)
                self.assertLess(rel, tol)

    # ── Single segment (all triplets share kernel offset) ──

    def test_single_segment_mvmr(self):
        """All T triplets at kernel offset 13 (mid-K). Other 26 segments
        are empty: chunk_seg_offs has 26 zeros + a large segment + 0s.
        Grouped path must handle the empty-segment skip correctly."""
        for dtype_name, dtype, tol in DTYPES:
            with self.subTest(dtype=dtype_name):
                device = "cuda"
                torch.manual_seed(0)
                T = 1000
                a = (torch.randn(27, 1, 32, 32, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                b = (torch.randn(64, 1, 32, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                a_idx = torch.full((T,), 13, device=device, dtype=torch.int64)
                b_idx = torch.randint(0, 64, (T,), device=device, dtype=torch.int64)
                o_idx = torch.randint(0, 64, (T,), device=device, dtype=torch.int64)

                out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, 64,
                )
                ref = mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, 64)
                rel = _rel_err(out, ref)
                self.assertLess(rel, tol)

    # ── sort_by ≠ k fallback ──

    def test_unsorted_a_idx_falls_back(self):
        """a_idx not sorted ascending — grouped path must refuse and
        the per-triplet path must produce correct results."""
        for dtype_name, dtype, tol in DTYPES:
            with self.subTest(dtype=dtype_name):
                device = "cuda"
                torch.manual_seed(0)
                T = 500
                a = (torch.randn(27, 1, 16, 16, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                b = (torch.randn(32, 1, 16, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                # Sort by b_idx instead of a_idx — guarantees a_idx unsorted.
                a_idx = torch.randint(0, 27, (T,), device=device, dtype=torch.int64)
                b_idx = torch.randint(0, 32, (T,), device=device, dtype=torch.int64)
                o_idx = torch.randint(0, 32, (T,), device=device, dtype=torch.int64)
                order = torch.argsort(b_idx, stable=True)
                a_idx, b_idx, o_idx = a_idx[order], b_idx[order], o_idx[order]
                # Confirm unsorted (with very high probability T=500).
                if bool((a_idx[1:] >= a_idx[:-1]).all().item()):
                    self.skipTest("Random a_idx happens to be sorted; rerun.")

                out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, 32,
                )
                ref = mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, 32)
                rel = _rel_err(out, ref)
                self.assertLess(rel, tol)

    # ── Group conv (G > 1) fallback ──

    def test_group_conv_G2_falls_back(self):
        """G=2 — grouped path's tl.dot needs G=1 → must fall back to
        per-triplet which supports any G."""
        for dtype_name, dtype, tol in DTYPES:
            with self.subTest(dtype=dtype_name):
                device = "cuda"
                torch.manual_seed(0)
                T = 500
                G = 2
                a = (torch.randn(27, G, 16, 16, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                b = (torch.randn(32, G, 16, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                a_idx = torch.randint(0, 27, (T,), device=device, dtype=torch.int64)
                b_idx = torch.randint(0, 32, (T,), device=device, dtype=torch.int64)
                o_idx = torch.randint(0, 32, (T,), device=device, dtype=torch.int64)
                order = torch.argsort(a_idx, stable=True)
                a_idx, b_idx, o_idx = a_idx[order], b_idx[order], o_idx[order]

                out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, 32,
                )
                ref = mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, 32)
                rel = _rel_err(out, ref)
                self.assertLess(rel, tol)

    # ── Forced-grouped on tiny input (threshold bypass) ──

    def test_force_fsg_below_threshold(self):
        """T=200, K=27 → avg=7.4 < 16. Auto would route to per-triplet;
        force-grouped runs the grouped kernel anyway. Must still be
        correct (just maybe slower than per-triplet for this size)."""
        for dtype_name, dtype, tol in DTYPES:
            with self.subTest(dtype=dtype_name):
                device = "cuda"
                torch.manual_seed(0)
                T = 200
                a = (torch.randn(27, 1, 16, 16, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                b = (torch.randn(32, 1, 16, device=device, dtype=torch.float32)
                     * 0.1).to(dtype)
                a_idx = torch.randint(0, 27, (T,), device=device, dtype=torch.int64)
                b_idx = torch.randint(0, 32, (T,), device=device, dtype=torch.int64)
                o_idx = torch.randint(0, 32, (T,), device=device, dtype=torch.int64)
                order = torch.argsort(a_idx, stable=True)
                a_idx, b_idx, o_idx = a_idx[order], b_idx[order], o_idx[order]

                with dispatch_mode("force_fsg"):
                    out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                        a, a_idx, b, b_idx, o_idx, 32,
                    )
                ref = mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, 32)
                rel = _rel_err(out, ref)
                self.assertLess(rel, tol)


if __name__ == "__main__":
    unittest.main()
