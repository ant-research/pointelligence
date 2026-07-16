"""Scheduler infra — CPU only, no operators. Validates segmentation, the
ForceFused override/oracle, refresh-cadence grouping, and geometry threading."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch
from internals.two_phase import (
    TwoPhaseOp, FusedOp, ForceFused, GeometryScheduler)


class AddIndexOp:
    """Separable: build_indices reads geom (an int), apply adds it to x.
    Advances geom by +1 so we can assert the cascade threads."""
    separable = True
    def __init__(self, tag): self.tag = tag; self.built_with = None
    def build_indices(self, geom):
        self.built_with = geom
        class B:  # duck-typed bundle
            pass
        b = B(); b.add = float(geom); b.next_geom = geom + 1
        return b
    def apply(self, x, b):
        return x + b.add


class SortByFeatureOp:
    """Non-separable: index (the permutation) depends on x -> a FusedOp seam.
    Mutates geom to a sentinel so we can assert downstream builds see it."""
    separable = False
    def forward(self, x, geom):
        perm = torch.argsort(x.sum(dim=1))      # feature-dependent index
        return x[perm], geom + 100              # geom mutated feature-dependently


def _interleaved(ops, x, geom):
    """Reference: build_i then apply_i, strictly in order (no hoist)."""
    for op in ops:
        if getattr(op, "separable", True):
            b = op.build_indices(geom); x = op.apply(x, b)
            ng = getattr(b, "next_geom", None); geom = ng if ng is not None else geom
        else:
            x, geom = op.forward(x, geom)
    return x, geom


def test_all_separable_matches_interleaved():
    x = torch.arange(6.0).reshape(3, 2)
    ops = [AddIndexOp("a"), AddIndexOp("b"), AddIndexOp("c")]
    got = GeometryScheduler().run(list(ops), x.clone(), geom=0)
    ref, _ = _interleaved([AddIndexOp("a"), AddIndexOp("b"), AddIndexOp("c")], x.clone(), 0)
    assert torch.equal(got, ref)


def test_geometry_cascade_threads_through_builds():
    """3 separable ops add geom=0,1,2 => x + 0 + 1 + 2 = x + 3."""
    x = torch.zeros(2, 2)
    got = GeometryScheduler().run([AddIndexOp("a"), AddIndexOp("b"), AddIndexOp("c")], x, geom=0)
    assert torch.allclose(got, torch.full((2, 2), 3.0))


def test_force_fused_equals_separable():
    """Wrapping separable ops in ForceFused changes only WHEN build runs;
    deterministic ops => bit-identical output."""
    x = torch.arange(6.0).reshape(3, 2)
    plain = GeometryScheduler().run([AddIndexOp("a"), AddIndexOp("b")], x.clone(), geom=0)
    forced = GeometryScheduler().run(
        [ForceFused(AddIndexOp("a")), ForceFused(AddIndexOp("b"))], x.clone(), geom=0)
    assert torch.equal(plain, forced)


def test_segmentation_count_and_seam_geometry():
    """A FusedOp mid-sequence splits into 2 compiled segments and its geom
    mutation reaches the downstream build."""
    x = torch.tensor([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    seam = SortByFeatureOp()
    tail = AddIndexOp("tail")
    n_segments = {"count": 0}
    def counting_compiler(fn):
        n_segments["count"] += 1
        return fn
    sched = GeometryScheduler()
    out = sched.run([AddIndexOp("head"), seam, tail], x.clone(), geom=0,
                    compile_segments=True, compiler=counting_compiler)
    assert n_segments["count"] == 2                       # head-seg, tail-seg
    assert tail.built_with == 101                         # 0 -> +1 (head) -> +100 (seam) = 101
