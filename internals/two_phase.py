"""Two-phase geometry/feature op protocol + segmented scheduler.

A network is a serial sequence of ops. A *separable* op (TwoPhaseOp) splits into
a feature-independent index build (build_indices(geom) -> bundle) and a break-free
feature compute (apply(x, bundle) -> x). A *non-separable* op (FusedOp) does a
feature-dependent build fused with compute (forward(x, geom) -> (x, geom)) and is a
scheduler seam.

The scheduler partitions the op-sequence at FusedOp boundaries into segments. Within
a segment it hoists every build to the front (phase 1, inside triplet_cache_scope)
then runs the break-free apply chain (phase 2, the torch.compile unit). FusedOps run
eagerly between segments and may advance geometry feature-dependently.

A bundle is any object with an optional ``next_geom`` attribute (the advanced
geometry for the next op); apply receives the same bundle build_indices returned.
"""
from __future__ import annotations
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import torch

from internals.triplet_cache import triplet_cache_scope


@runtime_checkable
class TwoPhaseOp(Protocol):
    separable: bool  # True

    def build_indices(self, geom: Any) -> Any: ...
    def apply(self, x: torch.Tensor, bundle: Any) -> torch.Tensor: ...


@runtime_checkable
class FusedOp(Protocol):
    separable: bool  # False

    def forward(self, x: torch.Tensor, geom: Any) -> tuple[torch.Tensor, Any]: ...


class ForceFused:
    """Adapt any TwoPhaseOp into a FusedOp: run its build+apply interleaved/eager
    at its sequence position. Forcing every op into this mode reproduces the exact
    interleaved baseline -- the off-path and the parity oracle."""
    separable = False

    def __init__(self, inner: TwoPhaseOp):
        self.inner = inner

    def forward(self, x: torch.Tensor, geom: Any) -> tuple[torch.Tensor, Any]:
        b = self.inner.build_indices(geom)
        ng = getattr(b, "next_geom", None)
        return self.inner.apply(x, b), (ng if ng is not None else geom)


def _default_compiler(fn: Callable) -> Callable:
    return torch.compile(fn, fullgraph=True, dynamic=True)


class GeometryScheduler:
    """Segmented two-pass executor (see module docstring)."""

    def run(self, ops: Sequence[Any], x: torch.Tensor, geom: Any, *,
            compile_segments: bool = False,
            compiler: Optional[Callable[[Callable], Callable]] = None) -> torch.Tensor:
        compiler = compiler or _default_compiler
        seg: list = []
        out = x
        for op in ops:
            if getattr(op, "separable", True):
                seg.append(op)
            else:
                out, geom = self._run_segment(seg, out, geom, compile_segments, compiler)
                seg = []
                out, geom = op.forward(out, geom)
        out, geom = self._run_segment(seg, out, geom, compile_segments, compiler)
        return out

    def _run_segment(self, seg: list, x: torch.Tensor, geom: Any,
                     do_compile: bool, compiler: Callable):
        if not seg:
            return x, geom
        bundles: list = []
        with triplet_cache_scope():                 # phase 1: hoisted builds
            for op in seg:
                b = op.build_indices(geom)
                bundles.append(b)
                ng = getattr(b, "next_geom", None)
                if ng is not None:
                    geom = ng

        def body(x: torch.Tensor) -> torch.Tensor:  # phase 2: break-free
            for op, b in zip(seg, bundles):
                x = op.apply(x, b)
            return x

        body_fn = compiler(body) if do_compile else body
        return body_fn(x), geom
