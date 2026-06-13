"""compile-safety + parity for the TIG hybrid mvmr op (build_hybrid=True path).

The hybrid flat+masked path used to route through `_TigMvmrHybrid`, a
`torch.autograd.Function` (the last autograd.Function in the conv stack). It is
bench/test-only (production builds `build_hybrid=False` → the flat `tig_mvmr`
op) and never torch.compile-reached, but an autograd.Function still forces a
Dynamo graph break wherever it appears. It is now the
`sparse_engines::tig_mvmr_hybrid` custom_op (register_fake + register_autograd):
the masked+residual forward reads the hybrid tensors, the backward is the SAME
flat-path grad as `tig_mvmr` (reconstruct flat TigIndex → tig_grad_weight/input).

Asserts:
  * hybrid fwd matches flat fwd within fp16 tol (same conv, different kernel
    structure), and hybrid grads match flat grads within the TIG grad kernels'
    fp32-atomic-accumulation noise band (the backward IS the flat backward, so
    the only difference is nondeterministic atomic ordering — measured ~1e-4,
    far below the fp16 step);
  * the op contributes ZERO Dynamo graph breaks.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch

from sparse_engines.tig import TigIndex, tig_mvmr


def _problem(n=2000, C=64, M=64, T=30000, K=27, seed=4, dev="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, n, (T,), generator=g).to(dev)
    j = torch.randint(0, n, (T,), generator=g).to(dev)
    k = torch.randint(0, K, (T,), generator=g).to(dev)
    weight = torch.randn(K, 1, C, M, generator=g).to(dev).to(torch.float16)
    feat = torch.randn(n, C, generator=g).to(dev).to(torch.float16)
    return weight, feat, i, j, k, n


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTigHybridCompileSafety(unittest.TestCase):
    def test_hybrid_matches_flat_fwd_and_grads(self):
        weight, feat, i, j, k, n = _problem()
        idx = TigIndex(i, j, k, n)  # build_hybrid=True (default)
        self.assertTrue(idx.has_hybrid)

        def run(mode):
            w = weight.detach().clone().requires_grad_(True)
            f = feat.detach().clone().requires_grad_(True)
            out = tig_mvmr(w, f, idx, mode=mode)
            torch.manual_seed(0)
            gout = torch.randn_like(out)
            (out.float() * gout.float()).sum().backward()
            return out.detach(), w.grad.detach(), f.grad.detach()

        out_h, gw_h, gf_h = run("hybrid")
        out_f, gw_f, gf_f = run("flat")

        def rel(a, b):
            return (a.float() - b.float()).abs().max().item() / (
                b.float().abs().max().item() + 1e-6)

        # forward: same conv, different kernel structure -> fp16-tol agreement
        self.assertLess(rel(out_h, out_f), 5e-3,
                        f"hybrid vs flat fwd reldiff {rel(out_h, out_f)}")
        # grads: hybrid backward IS the flat backward; the only difference is
        # the grad kernels' fp32-atomic ordering (noise ~1e-4 << fp16 step).
        self.assertLess(rel(gw_h, gw_f), 2e-3,
                        f"grad_w reldiff {rel(gw_h, gw_f)} exceeds atomic band")
        self.assertLess(rel(gf_h, gf_f), 2e-3,
                        f"grad_feat reldiff {rel(gf_h, gf_f)} exceeds atomic band")

    def test_zero_graph_breaks(self):
        weight, feat, i, j, k, n = _problem(seed=5)
        idx = TigIndex(i, j, k, n)
        w = weight.requires_grad_(True)
        f = feat.requires_grad_(True)

        def fn(w, f):
            return tig_mvmr(w, f, idx, mode="hybrid")

        torch._dynamo.reset()
        explanation = torch._dynamo.explain(fn)(w, f)
        self.assertEqual(
            explanation.graph_break_count, 0,
            f"tig_mvmr_hybrid must contribute 0 graph breaks; got "
            f"{explanation.graph_break_count}")


if __name__ == "__main__":
    unittest.main()
