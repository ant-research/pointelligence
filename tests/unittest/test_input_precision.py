"""Smoke test for the `precision_mode` flag on the grouped MVMR/VVOR.

``precision_mode("tf32")`` runs the grouped path's ``tl.dot`` at tf32
(tensor-core mma, ~1e-3 per-multiply rel error); ``"ieee"`` /
``"default"`` (under factory torch settings) run IEEE single-precision
(~1e-7). Default-resolution semantics live in
test_fp32_routing_precision.py.

We verify two things:
  1. ``precision_mode("ieee")`` produces tighter rel-error vs fp32-numpy
     reference at fp32 inputs than an explicit ``"tf32"`` scope.
  2. ``precision_mode("tf32")`` still lands within tf32's ~1e-3 budget.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import numpy as np
import torch
import sparse_engines
from sparse_engines._dispatch_override import precision_mode, dispatch_mode

from test_sparse_linalg_fp16 import mvmr_numpy_ref


class TestInputPrecision(unittest.TestCase):
    """The IEEE option should reduce tf32-induced rel error at fp32 inputs."""

    def test_mvmr_fp32_ieee_tightens_rel_err(self):
        device = "cuda"
        torch.manual_seed(0)
        N_a, N_b, N_o = 27, 8000, 8000
        G, M, C, T = 1, 256, 256, 50_000
        a = torch.randn(N_a, G, C, M, device=device, dtype=torch.float32) * 0.1
        b = torch.randn(N_b, G, C, device=device, dtype=torch.float32) * 0.1
        torch.manual_seed(1)
        a_idx = torch.randint(0, N_a, (T,), device=device, dtype=torch.int64)
        b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)
        o_idx = torch.randint(0, N_o, (T,), device=device, dtype=torch.int64)
        order = torch.argsort(a_idx, stable=True)
        a_idx, b_idx, o_idx = a_idx[order], b_idx[order], o_idx[order]

        ref = mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, N_o)
        ref_t = torch.tensor(ref, device=device)

        # Force grouped path so the precision flag matters.
        with dispatch_mode("force_fsg"):
            with precision_mode("tf32"):
                out_tf32 = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, N_o,
                )
            with precision_mode("ieee"):
                out_ieee = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                    a, a_idx, b, b_idx, o_idx, N_o,
                )

        rel_tf32 = (out_tf32 - ref_t).abs().max().item() / max(ref_t.abs().max().item(), 1e-6)
        rel_ieee = (out_ieee - ref_t).abs().max().item() / max(ref_t.abs().max().item(), 1e-6)

        print(f"\n  MVMR fp32 grouped × tf32 rel = {rel_tf32:.3e}")
        print(f"  MVMR fp32 grouped × ieee rel = {rel_ieee:.3e}")

        # ieee must be tighter than tf32, and tf32 must still be in budget.
        self.assertLess(rel_ieee, rel_tf32,
                         "IEEE precision should give tighter rel err than tf32")
        self.assertLess(rel_tf32, 5e-3,
                         "explicit tf32 scope should still be within 5e-3 budget")
        self.assertLess(rel_ieee, 1e-4,
                         "IEEE precision should reach <1e-4 rel err at fp32 inputs")


if __name__ == "__main__":
    unittest.main()
