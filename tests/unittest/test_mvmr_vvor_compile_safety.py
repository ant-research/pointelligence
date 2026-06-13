"""torch.compile-safety tests for the mvmr/vvor sparse-engine op-pair.

These ops are ``@torch.library.triton_op`` with ``register_fake`` +
``register_autograd``. The intended steady state is that a model using
PointConv3d can be wrapped in ``torch.compile(fullgraph=True)`` WITHOUT a
graph break at the conv: Dynamo sees the registered fake (data-independent
[n_o, G, M] meta) and treats the op as an opaque-but-known leaf, and
Inductor lowers it without tracing into the body's Triton launches.

The one obstacle is the in-body host sync: the grouped path checks
``is_sorted_cached`` (a ``.item()``) and builds ``kernel_offset_segments_cached``
(a ``.data_ptr()`` memo) — both fail on a FakeTensor under trace. The
contract resolves it WITHOUT removing the safety net: a caller that knows its
triplets are k-sorted (every production conv path — ``build_triplets`` /
``MetaData`` emit ``sort_by="k"`` by construction) passes a precomputed
``seg_offs`` (built sync-free via ``torch.searchsorted``); the op then skips
the sync and the body traces. ``seg_offs=None`` keeps the exact eager behavior
(runtime sortedness check + PT fallback) for unsorted/undeclared callers.

This module asserts:
  1. eager parity — ``seg_offs=None`` vs a passed ``seg_offs`` agree (same
     grouped kernel, same segment offsets) within atomicAdd-order tolerance.
  2. compile — both ops, in BOTH forward directions (mvmr-as-fwd with vvor as
     its grad_a; vvor-as-fwd with mvmr as its grad), trace under
     ``torch.compile(fullgraph=True)`` for the FULL forward+backward and match
     eager. (vvor is a valid standalone forward primitive whose backward is
     mvmr, even though our models only reach it as mvmr's backward.)
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch

import sparse_engines  # noqa: F401 — registers the ops
from sparse_engines.ops import (
    sparse_matrix_vector_multiplication_reduction as mvmr,
    sparse_vector_vector_outer_product_reduction as vvor,
)
from sparse_engines._seg_offs import kernel_offset_segments


# enc3-class shape: grouped path fires (G==1, C>=64, avg_T_per_K>=16).
_K, _N, _M, _C, _T = 27, 800, 256, 256, 6_500


def _maxdiff(x, y):
    return (x.float() - y.float()).abs().max().item()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMvmrVvorCompileSafety(unittest.TestCase):
    def _mvmr_inputs(self, dtype):
        dev = "cuda"
        torch.manual_seed(0)
        a_idx = torch.randint(0, _K, (_T,), device=dev, dtype=torch.int64)
        order = torch.argsort(a_idx, stable=True)         # sort_by="k"
        a_idx = a_idx[order].contiguous()
        b_idx = torch.randint(0, _N, (_T,), device=dev, dtype=torch.int64)[order].contiguous()
        o_idx = torch.randint(0, _N, (_T,), device=dev, dtype=torch.int64)[order].contiguous()
        a = torch.randn(_K, 1, _C, _M, device=dev, dtype=dtype, requires_grad=True)
        b = torch.randn(_N, 1, _C, device=dev, dtype=dtype, requires_grad=True)
        seg = kernel_offset_segments(a_idx, _K)
        return a, a_idx, b, b_idx, o_idx, seg

    def _vvor_inputs(self, dtype):
        # vvor sorts by o_idx into K=n_o bins; a:[N,1,C], b:[N,1,M].
        dev = "cuda"
        torch.manual_seed(0)
        o_idx = torch.randint(0, _K, (_T,), device=dev, dtype=torch.int64)
        order = torch.argsort(o_idx, stable=True)
        o_idx = o_idx[order].contiguous()
        a_idx = torch.randint(0, _N, (_T,), device=dev, dtype=torch.int64)[order].contiguous()
        b_idx = torch.randint(0, _N, (_T,), device=dev, dtype=torch.int64)[order].contiguous()
        a = torch.randn(_N, 1, _C, device=dev, dtype=dtype, requires_grad=True)
        b = torch.randn(_N, 1, _M, device=dev, dtype=dtype, requires_grad=True)
        seg = kernel_offset_segments(o_idx, _K)
        return a, a_idx, b, b_idx, o_idx, seg

    def test_mvmr_eager_seg_offs_identity(self):
        # seg_offs=None and a passed seg_offs route to the SAME grouped kernel
        # with the SAME segment offsets — so the only possible difference is
        # the atomicAdd scatter ORDER between two separate launches (fp32
        # atomicAdd is non-associative), bounded well under fp32 accum noise.
        a, a_idx, b, b_idx, o_idx, seg = self._mvmr_inputs(torch.float32)
        o_none = mvmr(a, a_idx, b, b_idx, o_idx, _N)
        o_seg = mvmr(a, a_idx, b, b_idx, o_idx, _N, seg)
        self.assertLess(_maxdiff(o_none, o_seg), 1e-3)

    def test_vvor_eager_seg_offs_identity(self):
        a, a_idx, b, b_idx, o_idx, seg = self._vvor_inputs(torch.float32)
        o_none = vvor(a, a_idx, b, b_idx, o_idx, _K)
        o_seg = vvor(a, a_idx, b, b_idx, o_idx, _K, seg)
        self.assertLess(_maxdiff(o_none, o_seg), 1e-3)

    def test_mvmr_fullgraph_fwd_bwd(self):
        # mvmr forward → backward calls vvor (grad_a) + mvmr (grad_b); the
        # forward seg_offs propagates to both, so the {mvmr, vvor} dual closes
        # sync-free. fp32 keeps the scatter deterministic for a tight bound.
        a, a_idx, b, b_idx, o_idx, seg = self._mvmr_inputs(torch.float32)

        def f(a, b):
            return mvmr(a, a_idx, b, b_idx, o_idx, _N, seg)

        out_e = f(a, b)
        ge_a, ge_b = torch.autograd.grad(out_e.sum(), [a, b], retain_graph=True)

        torch._dynamo.reset()
        cf = torch.compile(f, fullgraph=True, dynamic=True)
        out_c = cf(a, b)
        gc_a, gc_b = torch.autograd.grad(out_c.sum(), [a, b])

        self.assertLess(_maxdiff(out_c, out_e), 5e-3)
        self.assertLess(_maxdiff(gc_a, ge_a), 5e-3)
        self.assertLess(_maxdiff(gc_b, ge_b), 5e-3)

    def test_vvor_fullgraph_fwd_bwd(self):
        # vvor as a standalone forward primitive: its backward is mvmr. Valid
        # computation even though our models only reach vvor as mvmr's grad_a.
        a, a_idx, b, b_idx, o_idx, seg = self._vvor_inputs(torch.float32)

        def f(a, b):
            return vvor(a, a_idx, b, b_idx, o_idx, _K, seg)

        out_e = f(a, b)
        ge_a, ge_b = torch.autograd.grad(out_e.sum(), [a, b], retain_graph=True)

        torch._dynamo.reset()
        cf = torch.compile(f, fullgraph=True, dynamic=True)
        out_c = cf(a, b)
        gc_a, gc_b = torch.autograd.grad(out_c.sum(), [a, b])

        self.assertLess(_maxdiff(out_c, out_e), 5e-3)
        self.assertLess(_maxdiff(gc_a, ge_a), 5e-3)
        self.assertLess(_maxdiff(gc_b, ge_b), 5e-3)


if __name__ == "__main__":
    unittest.main()
