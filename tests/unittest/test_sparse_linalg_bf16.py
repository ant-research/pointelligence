"""bf16 correctness for MVMR / VVOR kernels — forward AND backward.

bf16 has the same exponent range as fp32 but only 7 mantissa bits (vs
10 for fp16, 23 for fp32). Per-multiply rel error ≈ 1/2^7 = ~8e-3, so
end-to-end rel err of ~1e-2 is realistic; we set REL_TOL accordingly.

Tensor cores on sm_80+ accept bf16 inputs in `tl.dot` with `out_dtype=
tl.float32`, so the grouped path should fire identically to fp16.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import numpy as np
import torch
import sparse_engines

from test_sparse_linalg_fp16 import (
    mvmr_numpy_ref, vvor_numpy_ref,
    mvmr_grad_a_numpy_ref, mvmr_grad_b_numpy_ref,
    vvor_grad_a_numpy_ref, vvor_grad_b_numpy_ref,
)


class _MVMRVVORBf16Mixin:
    """Mixin holding parametrised bf16 tests. Subclasses set the shape."""

    N_a = 27
    N_b = 8000
    N_o = 8000
    G = 1
    M = 256
    C = 256
    T = 50_000

    REL_TOL = 1.5e-2  # bf16 mantissa is 7 bits → ~8e-3 per-op + accum

    def _device(self):
        return "cuda"

    def _make_indices(self, seed=1):
        device = self._device()
        torch.manual_seed(seed)
        a_idx = torch.randint(0, self.N_a, (self.T,), device=device, dtype=torch.int64)
        b_idx = torch.randint(0, self.N_b, (self.T,), device=device, dtype=torch.int64)
        o_idx = torch.randint(0, self.N_o, (self.T,), device=device, dtype=torch.int64)
        order = torch.argsort(a_idx, stable=True)
        return a_idx[order], b_idx[order], o_idx[order]

    def _rel_err(self, triton_out, ref_np):
        ref = torch.tensor(ref_np, device=self._device())
        max_abs = (triton_out.float() - ref).abs().max().item()
        max_val = ref.abs().max().item()
        return max_abs / max(max_val, 1e-6), max_abs, max_val

    def test_mvmr_fwd(self):
        device = self._device()
        torch.manual_seed(0)
        a = (torch.randn(self.N_a, self.G, self.C, self.M, device=device,
                          dtype=torch.float32) * 0.1).to(torch.bfloat16)
        b = (torch.randn(self.N_b, self.G, self.C, device=device,
                          dtype=torch.float32) * 0.1).to(torch.bfloat16)
        a_idx, b_idx, o_idx = self._make_indices()

        out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
            a, a_idx, b, b_idx, o_idx, self.N_o,
        )
        ref = mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, self.N_o)
        rel, max_abs, max_val = self._rel_err(out, ref)
        print(f"\n  [{self.__class__.__name__}] MVMR fwd  rel={rel:.3e}")
        self.assertLess(rel, self.REL_TOL)

    def test_vvor_fwd(self):
        device = self._device()
        torch.manual_seed(0)
        a = (torch.randn(self.N_a, self.G, self.M, device=device,
                          dtype=torch.float32) * 0.1).to(torch.bfloat16)
        b = (torch.randn(self.N_b, self.G, self.C, device=device,
                          dtype=torch.float32) * 0.1).to(torch.bfloat16)
        a_idx, b_idx, o_idx = self._make_indices()

        out = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
            a, a_idx, b, b_idx, o_idx, self.N_o,
        )
        ref = vvor_numpy_ref(a, a_idx, b, b_idx, o_idx, self.N_o)
        rel, max_abs, max_val = self._rel_err(out, ref)
        print(f"\n  [{self.__class__.__name__}] VVOR fwd  rel={rel:.3e}")
        self.assertLess(rel, self.REL_TOL)

    def test_mvmr_bwd(self):
        device = self._device()
        torch.manual_seed(0)
        a = (torch.randn(self.N_a, self.G, self.C, self.M, device=device,
                          dtype=torch.float32) * 0.1).to(torch.bfloat16).requires_grad_(True)
        b = (torch.randn(self.N_b, self.G, self.C, device=device,
                          dtype=torch.float32) * 0.1).to(torch.bfloat16).requires_grad_(True)
        a_idx, b_idx, o_idx = self._make_indices()

        out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
            a, a_idx, b, b_idx, o_idx, self.N_o,
        )
        grad_o = (torch.randn_like(out).float() * 0.1).to(out.dtype)
        out.backward(grad_o)

        grad_a_ref = mvmr_grad_a_numpy_ref(
            (self.N_a, self.G, self.C, self.M), a_idx, b, b_idx, o_idx, grad_o,
        )
        grad_b_ref = mvmr_grad_b_numpy_ref(
            a, a_idx, (self.N_b, self.G, self.C), b_idx, o_idx, grad_o,
        )
        rel_a, _, _ = self._rel_err(a.grad, grad_a_ref)
        rel_b, _, _ = self._rel_err(b.grad, grad_b_ref)
        print(f"\n  [{self.__class__.__name__}] MVMR bwd  "
              f"grad_a rel={rel_a:.3e}, grad_b rel={rel_b:.3e}")
        self.assertLess(rel_a, self.REL_TOL)
        self.assertLess(rel_b, self.REL_TOL)

    def test_vvor_bwd(self):
        device = self._device()
        torch.manual_seed(0)
        a = (torch.randn(self.N_a, self.G, self.M, device=device,
                          dtype=torch.float32) * 0.1).to(torch.bfloat16).requires_grad_(True)
        b = (torch.randn(self.N_b, self.G, self.C, device=device,
                          dtype=torch.float32) * 0.1).to(torch.bfloat16).requires_grad_(True)
        a_idx, b_idx, o_idx = self._make_indices()

        out = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
            a, a_idx, b, b_idx, o_idx, self.N_o,
        )
        grad_o = (torch.randn_like(out).float() * 0.1).to(out.dtype)
        out.backward(grad_o)

        grad_a_ref = vvor_grad_a_numpy_ref(
            (self.N_a, self.G, self.M), a_idx, b, b_idx, o_idx, grad_o,
        )
        grad_b_ref = vvor_grad_b_numpy_ref(
            a, a_idx, (self.N_b, self.G, self.C), b_idx, o_idx, grad_o,
        )
        rel_a, _, _ = self._rel_err(a.grad, grad_a_ref)
        rel_b, _, _ = self._rel_err(b.grad, grad_b_ref)
        print(f"\n  [{self.__class__.__name__}] VVOR bwd  "
              f"grad_a rel={rel_a:.3e}, grad_b rel={rel_b:.3e}")
        self.assertLess(rel_a, self.REL_TOL)
        self.assertLess(rel_b, self.REL_TOL)


class TestSparseLinAlgBf16Deep(_MVMRVVORBf16Mixin, unittest.TestCase):
    N_a, N_b, N_o = 27, 8000, 8000
    G, M, C, T = 1, 256, 256, 50_000


class TestSparseLinAlgBf16Shallow(_MVMRVVORBf16Mixin, unittest.TestCase):
    N_a, N_b, N_o = 27, 4000, 4000
    G, M, C, T = 1, 64, 64, 10_000


if __name__ == "__main__":
    unittest.main()
