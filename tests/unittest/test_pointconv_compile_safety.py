"""torch.compile-safety tests for PointConv3d (TripletContract API).

`_conv_forward` used to re-derive the triplet index's structural facts
(k-sortedness via `is_sorted_cached(k).item()`, exact-cover via
`exact_cover_cached(i/j).item()`) on EVERY forward — host syncs that break
Dynamo (graph break) and cost a CPU↔GPU round trip per call. The
TripletContract design carries those facts as a `TripletContract` produced once
by the triplet builder (in its `@torch.compiler.disable` region, where
`.item()` is free) and consumed by the forward WITHOUT re-derivation. The
forward then traces under `torch.compile(fullgraph=True)`.

This module asserts a real `PointConv3d` (and `GenerativePointConv3d`) forward
+ backward compiles fullgraph with ZERO graph breaks and matches eager, given
the contract its builder would supply. k-sortedness is the conv-path invariant
(every builder emits `sort_by="k"`), so `TripletContract.submanifold()` is the
default the radius-search builders produce.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch

import sparse_engines  # noqa: F401 — registers the ops
from layers.conv import PointConv3d
from layers.contract import TripletContract


def _maxdiff(x, y):
    return (x.float() - y.float()).abs().max().item()


def _submanifold_triplets(n, K, T, device, seed=0):
    """k-sorted submanifold rulebook (T != n): the radius-search contract."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randint(0, K, (T,), generator=g).to(device)
    k, _ = torch.sort(k)                          # sort_by="k"
    i = torch.randint(0, n, (T,), generator=g).to(device)
    j = torch.randint(0, n, (T,), generator=g).to(device)
    return i.contiguous(), j.contiguous(), k.contiguous()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPointConvCompileSafety(unittest.TestCase):
    def test_submanifold_conv_fullgraph_fwd_bwd(self):
        dev = "cuda"
        torch.manual_seed(0)
        n, K, T, Cin, Cout = 2000, 27, 6500, 32, 64
        i, j, k = _submanifold_triplets(n, K, T, dev)
        conv = PointConv3d(Cin, Cout, kernel_size=3, bias=True).to(dev)
        contract = TripletContract.submanifold()
        x = torch.randn(n, Cin, device=dev, requires_grad=True)

        def f(x):
            return conv(x, i, j, k, n, contract=contract)

        out_e = f(x)
        ge = torch.autograd.grad(out_e.sum(), [x, conv.weight, conv.bias],
                                 retain_graph=True)

        torch._dynamo.reset()
        cf = torch.compile(f, fullgraph=True, dynamic=True)
        out_c = cf(x)
        gc = torch.autograd.grad(out_c.sum(), [x, conv.weight, conv.bias])

        self.assertLess(_maxdiff(out_c, out_e), 5e-3)
        for a, b in zip(gc, ge):
            self.assertLess(_maxdiff(a, b), 5e-3)

    def test_conv_zero_graph_breaks(self):
        # Explicit regression guard: the conv contributes NO graph break.
        dev = "cuda"
        torch.manual_seed(0)
        n, K, T, Cin, Cout = 1500, 27, 5000, 32, 64
        i, j, k = _submanifold_triplets(n, K, T, dev, seed=1)
        conv = PointConv3d(Cin, Cout, kernel_size=3, bias=False).to(dev)
        contract = TripletContract.submanifold()
        x = torch.randn(n, Cin, device=dev, requires_grad=True)

        def f(x):
            return conv(x, i, j, k, n, contract=contract)

        torch._dynamo.reset()
        explanation = torch._dynamo.explain(f)(x)
        self.assertEqual(
            explanation.graph_break_count, 0,
            f"PointConv3d forward must have 0 graph breaks; got "
            f"{explanation.graph_break_count}")

    def test_generative_conv_fullgraph(self):
        # GenerativePointConv3d's CONV BODY (closed-form uniform_seg_len +
        # exact_cover_out via sites.to_contract()) must trace fullgraph. The
        # generator + sites.validate() are build-time prep (data-dependent
        # output shape — legitimately outside the compiled region), so we
        # pre-build sites and compile the _conv_forward with the contract,
        # mirroring how a model would compile the trunk with prep hoisted out.
        from layers.generative import SubdivisionGenerator
        from layers.metadata import MetaData
        from layers.conv import GenerativePointConv3d
        dev = "cuda"
        torch.manual_seed(0)
        g = 0.04
        vox = torch.unique(torch.randint(0, 40, (3000, 3), device=dev), dim=0)
        pts = (vox.float() + 0.5) * g
        nn_ = pts.shape[0]
        si = torch.zeros(nn_, dtype=torch.long, device=dev)
        m = MetaData(points=pts, sample_inds=si,
                     sample_sizes=torch.bincount(si), grid_size=g,
                     kernel_size=None, auto_build_triplets=False)
        conv = GenerativePointConv3d(16, 32, generator=SubdivisionGenerator(2),
                                     bias=False, device=dev,
                                     dtype=torch.float16)
        sites = conv.generator(m)              # build-time prep (hoisted out)
        sites.validate()
        contract = sites.to_contract()
        x = torch.randn(nn_, 16, device=dev, dtype=torch.float16,
                        requires_grad=True)

        def f(x):
            return conv._conv_forward(
                x, sites.i, sites.j, sites.k, sites.n_out, conv.weight,
                conv.bias, contract=contract)

        out_e = f(x)
        torch._dynamo.reset()
        cf = torch.compile(f, fullgraph=True, dynamic=True)
        out_c = cf(x)
        self.assertLess(_maxdiff(out_c, out_e), 5e-3)


if __name__ == "__main__":
    unittest.main()
