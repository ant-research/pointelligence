"""torch.compile-safety + cross-engine parity for the fused PointConv3d op.

``force_fsg_fused`` used to route through ``FusedPointConv3d`` — a
``torch.autograd.Function`` whose ``.apply`` forces a Dynamo graph break on
every call (Inductor can't trace into an autograd.Function, so the whole
fullgraph compile fell back). The op is now the ``sparse_engines::
fused_pointconv3d`` ``custom_op`` (register_fake + register_autograd), so
Dynamo keeps it as an opaque-but-registered leaf — ZERO graph breaks — and the
surrounding trunk ops fuse up to its boundary.

What this module asserts (the bar these CUTLASS-backed ops can meet today):
  * the op contributes ZERO Dynamo graph breaks (``torch._dynamo.explain``);
  * fused (force_fsg_fused) fwd+bwd matches the independent TIG engine
    (force_tig) within fp16 tolerance, across G in {1, 4}.

NOT asserted: a full ``torch.compile(fullgraph=True)`` lowering through
Inductor. That additionally needs Meta/fake kernels on the underlying
``sparse_mvmr_cutlass_sm{80,90}_full_prestaged`` C++ ops AND a compile-guarded
sortedness check in the vvor grouped backward wrapper
(``vvor_cutlass.py`` ``(o_idx[1:] >= o_idx[:-1]).all().item()`` →
``aten._local_scalar_dense``). ``force_fsg_fused`` is an opt-in route, not the
production-compiled path (the default ``auto`` route uses the fully
fullgraph-safe ``tig_mvmr`` op), so that lowering is a tracked follow-up rather
than a blocker for removing the autograd.Function.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch

from sparse_engines.mvmr_cutlass import fused_pointconv3d


def _sorted_idx(K, T, dev, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.sort(torch.randint(0, K, (T,), generator=g))[0].to(dev).int()


def _maxreldiff(x, y):
    return ((x.float() - y.float()).abs().max()
            / (y.float().abs().max() + 1e-6)).item()


class TestV14AutoRoutePolicy(unittest.TestCase):
    def test_c512_training_falls_back_but_eval_can_fuse(self):
        from layers.conv import _auto_fused_gather_sum_width

        self.assertTrue(_auto_fused_gather_sum_width(64, grad_enabled=True))
        self.assertTrue(_auto_fused_gather_sum_width(128, grad_enabled=True))
        self.assertTrue(_auto_fused_gather_sum_width(256, grad_enabled=True))
        self.assertFalse(_auto_fused_gather_sum_width(512, grad_enabled=True))
        self.assertTrue(_auto_fused_gather_sum_width(512, grad_enabled=False))

    def test_force_fgs_is_short_for_force_fused_gather_sum(self):
        from sparse_engines._dispatch_override import current_mode, dispatch_mode

        with dispatch_mode("force_fgs"):
            self.assertEqual(current_mode(), "force_fused_gather_sum")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedConvCompileSafety(unittest.TestCase):
    def _fsg_vs_tig(self, G, N, K, T, Cg, Mg, seed=0):
        """One PointConv3d fwd+bwd run under force_fsg_fused and under force_tig
        (same inputs/weights) — both compute PointConv3d, so within fp16 tol the
        outputs and grads must agree. Self-contained (no hardcoded golden)."""
        from layers.conv import PointConv3d
        from sparse_engines._dispatch_override import dispatch_mode
        dev = "cuda"
        torch.manual_seed(seed)
        a = _sorted_idx(K, T, dev, seed)
        b = torch.randint(0, N, (T,), device=dev).int()
        o = torch.randint(0, N, (T,), device=dev).int()
        conv = PointConv3d(Cg * G, Mg * G, kernel_size=3, groups=G,
                           bias=False, device=dev, dtype=torch.float16)
        x0 = (torch.randn(N, Cg * G, device=dev) * 0.1).to(torch.float16)

        def run(mode):
            x = x0.detach().clone().requires_grad_(True)
            with dispatch_mode(mode):
                out = conv(x, o, b, a, N)
            torch.manual_seed(7)
            gout = torch.randn_like(out)
            (out.float() * gout.float()).sum().backward()
            return out.detach(), conv.weight.grad.detach().clone(), x.grad.detach()

        conv.zero_grad(set_to_none=True)
        out_f, gw_f, gx_f = run("force_fsg_fused")
        conv.zero_grad(set_to_none=True)
        out_t, gw_t, gx_t = run("force_tig")
        return (_maxreldiff(out_f, out_t), _maxreldiff(gw_f, gw_t),
                _maxreldiff(gx_f, gx_t))

    def test_fsg_matches_tig_g1(self):
        d_out, d_gw, d_gx = self._fsg_vs_tig(G=1, N=512, K=27, T=3000,
                                             Cg=64, Mg=64, seed=0)
        self.assertLess(d_out, 5e-2, f"fwd reldiff {d_out}")
        self.assertLess(d_gw, 5e-2, f"grad_w reldiff {d_gw}")
        self.assertLess(d_gx, 5e-2, f"grad_x reldiff {d_gx}")

    def test_fsg_matches_tig_grouped(self):
        d_out, d_gw, d_gx = self._fsg_vs_tig(G=4, N=512, K=27, T=2500,
                                             Cg=64, Mg=64, seed=1)
        self.assertLess(d_out, 5e-2, f"fwd reldiff {d_out}")
        self.assertLess(d_gw, 5e-2, f"grad_w reldiff {d_gw}")
        self.assertLess(d_gx, 5e-2, f"grad_x reldiff {d_gx}")

    def test_zero_graph_breaks(self):
        # The op is a registered custom_op leaf: Dynamo must NOT graph-break on
        # it (the autograd.Function it replaced broke on every .apply).
        dev = "cuda"
        torch.manual_seed(2)
        N, K, T, Cg, Mg = 1024, 27, 4000, 64, 64
        a = _sorted_idx(K, T, dev, 2)
        b = torch.randint(0, N, (T,), device=dev).int()
        o = torch.randint(0, N, (T,), device=dev).int()
        w = torch.randn(K, 1, Cg, Mg, device=dev, dtype=torch.float16,
                        requires_grad=True)
        x = torch.randn(N, 1, Cg, device=dev, dtype=torch.float16,
                        requires_grad=True)

        def f(w, x):
            return fused_pointconv3d(w, a, x, b, o, N)

        torch._dynamo.reset()
        explanation = torch._dynamo.explain(f)(w, x)
        self.assertEqual(
            explanation.graph_break_count, 0,
            f"fused_pointconv3d must contribute 0 graph breaks; got "
            f"{explanation.graph_break_count}")


if __name__ == "__main__":
    unittest.main()
