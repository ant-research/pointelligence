"""Grouped-kernel-vs-per-triplet equivalence tests.

The strongest correctness signal: for each PTv3 stage shape × dtype ×
fwd/bwd, run BOTH the grouped tensor-core path and the legacy per-triplet
path on the SAME input and assert their outputs agree to within accum
tolerance.

This complements the numpy-reference tests because it doesn't depend on
the reference's behaviour. If we ever introduce a kernel bug that
matches the numpy ref's pattern (unlikely but possible), the equivalence
test would catch it because the legacy and grouped paths share no
implementation lines.

Forced via ``sparse_engines._dispatch_override.dispatch_mode``.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch
import sparse_engines
from sparse_engines._dispatch_override import dispatch_mode


# Cumulative PTv3 stage shapes in (N_a, N_b, N_o, M, C) form. T is set so
# avg_T_per_K >= 16 (grouped path fires) at every stage.
PTV3_STAGES = [
    # name, N_a (kernel offsets), N_b, N_o, M, C, T
    ("enc0",  27, 32_000, 32_000,  32,  32, 200_000),
    ("enc1",  27, 11_000, 11_000,  64,  64,  90_000),
    ("enc2",  27,  3_000,  3_000, 128, 128,  25_000),
    ("enc3",  27,    800,    800, 256, 256,   6_500),
    ("enc4",  27,    200,    200, 512, 512,   1_700),
]

DTYPES = [
    ("fp32", torch.float32, 5e-3),   # tf32 mma in tl.dot at fp32 inputs
    ("fp16", torch.float16, 5e-3),
    ("bf16", torch.bfloat16, 1.5e-2),
]


def make_indices(N_a, N_b, N_o, T, device):
    torch.manual_seed(1)
    a_idx = torch.randint(0, N_a, (T,), device=device, dtype=torch.int64)
    b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)
    o_idx = torch.randint(0, N_o, (T,), device=device, dtype=torch.int64)
    order = torch.argsort(a_idx, stable=True)         # sort_by="k"
    return a_idx[order], b_idx[order], o_idx[order]


def rel_err(x, y):
    """Relative error using max-abs(x-y) / max-abs(y), both fp32."""
    diff = (x.float() - y.float()).abs().max().item()
    base = y.float().abs().max().item()
    return diff / max(base, 1e-6)


class TestGroupedEquivalence(unittest.TestCase):

    def _run_mvmr(self, a, a_idx, b, b_idx, o_idx, n_o, mode):
        with dispatch_mode(mode):
            return sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                a, a_idx, b, b_idx, o_idx, n_o,
            )

    def _run_vvor(self, a, a_idx, b, b_idx, o_idx, n_o, mode):
        with dispatch_mode(mode):
            return sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
                a, a_idx, b, b_idx, o_idx, n_o,
            )

    # ── Forward ──

    def test_mvmr_fwd_equivalence(self):
        device = "cuda"
        for stage_name, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            for dtype_name, dtype, tol in DTYPES:
                with self.subTest(stage=stage_name, dtype=dtype_name):
                    torch.manual_seed(0)
                    a = (torch.randn(N_a, 1, C, M, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    a_idx, b_idx, o_idx = make_indices(N_a, N_b, N_o, T, device)

                    out_grp = self._run_mvmr(a, a_idx, b, b_idx, o_idx, N_o, "force_fsg")
                    out_pt  = self._run_mvmr(a, a_idx, b, b_idx, o_idx, N_o, "force_pt")
                    rel = rel_err(out_grp, out_pt)
                    print(f"  MVMR-fwd [{stage_name} {dtype_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol)

    def test_vvor_fwd_equivalence(self):
        device = "cuda"
        for stage_name, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            # VVOR's o_idx is the kernel offset (the K bin). For equivalence,
            # use the kernel-offset domain.
            K_off = 27
            for dtype_name, dtype, tol in DTYPES:
                with self.subTest(stage=stage_name, dtype=dtype_name):
                    torch.manual_seed(0)
                    a = (torch.randn(N_a, 1, M, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32)
                         * 0.1).to(dtype)
                    # Build triplets sorted by o_idx (K bin) so VVOR's grouped
                    # path can fire.
                    torch.manual_seed(1)
                    a_idx = torch.randint(0, N_a, (T,), device=device, dtype=torch.int64)
                    b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)
                    o_idx = torch.randint(0, K_off, (T,), device=device, dtype=torch.int64)
                    order = torch.argsort(o_idx, stable=True)
                    a_idx, b_idx, o_idx = a_idx[order], b_idx[order], o_idx[order]

                    out_grp = self._run_vvor(a, a_idx, b, b_idx, o_idx, K_off, "force_fsg")
                    out_pt  = self._run_vvor(a, a_idx, b, b_idx, o_idx, K_off, "force_pt")
                    rel = rel_err(out_grp, out_pt)
                    print(f"  VVOR-fwd [{stage_name} {dtype_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol)

    # ── Backward via autograd ──

    def test_mvmr_bwd_equivalence(self):
        device = "cuda"
        for stage_name, N_a, N_b, N_o, M, C, T in PTV3_STAGES:
            for dtype_name, dtype, tol in DTYPES:
                with self.subTest(stage=stage_name, dtype=dtype_name):
                    torch.manual_seed(0)
                    # Need TWO copies of (a, b) so both paths see identical
                    # inputs and accumulate independent grads.
                    a_data = (torch.randn(N_a, 1, C, M, device=device,
                                           dtype=torch.float32) * 0.1).to(dtype)
                    b_data = (torch.randn(N_b, 1, C, device=device,
                                           dtype=torch.float32) * 0.1).to(dtype)
                    a_idx, b_idx, o_idx = make_indices(N_a, N_b, N_o, T, device)

                    # Run grouped path
                    a_g = a_data.detach().clone().requires_grad_(True)
                    b_g = b_data.detach().clone().requires_grad_(True)
                    out_g = self._run_mvmr(a_g, a_idx, b_g, b_idx, o_idx, N_o, "force_fsg")
                    grad_o = (torch.randn_like(out_g).float() * 0.1).to(out_g.dtype)
                    out_g.backward(grad_o)

                    # Run per-triplet path with same grad_o
                    a_p = a_data.detach().clone().requires_grad_(True)
                    b_p = b_data.detach().clone().requires_grad_(True)
                    out_p = self._run_mvmr(a_p, a_idx, b_p, b_idx, o_idx, N_o, "force_pt")
                    out_p.backward(grad_o)

                    rel_a = rel_err(a_g.grad, a_p.grad)
                    rel_b = rel_err(b_g.grad, b_p.grad)
                    print(f"  MVMR-bwd [{stage_name} {dtype_name}] "
                          f"grad_a rel={rel_a:.3e}, grad_b rel={rel_b:.3e}")
                    self.assertLess(rel_a, tol, f"{stage_name} {dtype_name} grad_a")
                    self.assertLess(rel_b, tol, f"{stage_name} {dtype_name} grad_b")


if __name__ == "__main__":
    unittest.main()
