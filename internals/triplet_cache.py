"""Forward-scoped ambient triplet cache (POINTELLIGENCE building block).

Submanifold triplets (i, j, k) are a pure function of a point set's geometry,
the search radius, the kernel, the query set, and the sort order. Within one
forward the same geometry+kernel recurs (U-Net encoder/decoder cascade; PNT
stem-descent / head-ascent). This module provides a re-entrant, per-forward
cache that `build_triplets` consults directly, so every caller benefits with
no MetaData threading.

Usage — one line at the top of a model's forward:

    from internals.triplet_cache import triplet_cache_scope
    def forward(self, ...):
        with triplet_cache_scope():
            ...

Outside any scope the cache is inactive and `build_triplets` builds every call
(zero behavior change). Set POINTELLIGENCE_DISABLE_TRIPLET_CACHE=1 to disable
caching even inside a scope (clean A/B + debug escape hatch).
"""

import functools
import os
import threading
from contextlib import contextmanager

_state = threading.local()

# Read once at import (consistent with grid_indexing._GRID_OVERFLOW_CHECK).
_DISABLED = os.environ.get("POINTELLIGENCE_DISABLE_TRIPLET_CACHE", "0") == "1"


def _active_cache():
    """Return the active per-forward cache dict, or None if no scope is active
    (or caching is disabled by env)."""
    return getattr(_state, "cache", None)


@contextmanager
def triplet_cache_scope():
    """Install a forward-scoped triplet cache.

    Re-entrant: a nested scope yields the SAME dict as the enclosing scope
    (a model entering a scope and calling a submodule that also enters one
    share a single cache). On exit, the previous active cache is restored
    (None at the top level, so the dict is GC'd). Yields None when caching is
    disabled by POINTELLIGENCE_DISABLE_TRIPLET_CACHE.
    """
    if _DISABLED:
        yield None
        return
    prev = getattr(_state, "cache", None)
    if prev is not None:
        yield prev                       # re-entrant: reuse the outer dict
        return
    _state.cache = {}
    try:
        yield _state.cache
    finally:
        _state.cache = prev              # restore (None at top level)


_SELF = "__self__"        # query_points is points (self-query)
_RADIUS_NDIGITS = 9       # guard float-formatting jitter only; radii separated >>1e-9


def _kernel_descriptor(kernel_indexer):
    """Hashable identity for the kernel binning function, or None if it cannot
    be safely keyed (caller then SKIPS caching for that call).

    Production always passes partial(voxelize_3d, kernel_size=K) ->
    ("voxelize_3d", (), (("kernel_size", K),)). A non-partial (bare fn / lambda)
    is NOT safely keyable: build_triplets is public and accepts an arbitrary
    kernel_indexer; .func/.keywords would AttributeError, and all lambdas share
    "<lambda>" (false-hit). Return None for those -> no caching (safe)."""
    if isinstance(kernel_indexer, functools.partial):
        try:
            kw = tuple(sorted(kernel_indexer.keywords.items()))
            args = tuple(kernel_indexer.args)
            hash((kw, args))          # ensure hashable (e.g. a tensor kwarg is not)
        except TypeError:
            return None
        func = kernel_indexer.func
        return (getattr(func, "__qualname__", repr(func)), args, kw)
    return None


def triplet_key(points, query_points, neighbor_radius, radius_scaler,
                sort_by, return_num_neighbors, sample_inds,
                query_sample_inds, kernel_indexer):
    """Full output determinant of build_triplets, or None if uncacheable.

    Computed AFTER query-None normalization (so query_* mirror points/* on a
    self-query). data_ptr identity is sound within a forward (tensors stay
    alive; dict is fresh per scope). sample_inds is keyed by data_ptr (NOT a
    value fingerprint — tolist() would force a D2H sync and defeat the
    host-bound win); radius_search confines neighbors per batch element, so the
    batch partition is a genuine output determinant."""
    kd = _kernel_descriptor(kernel_indexer)
    if kd is None:
        return None
    q_self = query_points is points
    rs = None if radius_scaler is None else round(float(radius_scaler), _RADIUS_NDIGITS)
    return (
        points.data_ptr(), int(points.shape[0]), sample_inds.data_ptr(),
        _SELF if q_self else query_points.data_ptr(),
        _SELF if q_self else query_sample_inds.data_ptr(),
        round(float(neighbor_radius), _RADIUS_NDIGITS), rs,
        sort_by, bool(return_num_neighbors), kd,
    )
