import os
import importlib
import pytest
from functools import partial
from unittest import mock
import torch


def _fresh_module():
    import internals.triplet_cache as tc
    return importlib.reload(tc)


def test_active_cache_none_outside_scope():
    tc = _fresh_module()
    assert tc._active_cache() is None


def test_scope_installs_and_restores():
    tc = _fresh_module()
    assert tc._active_cache() is None
    with tc.triplet_cache_scope() as c:
        assert isinstance(c, dict)
        assert tc._active_cache() is c
    assert tc._active_cache() is None


def test_nested_scope_reuses_outer_dict():
    tc = _fresh_module()
    with tc.triplet_cache_scope() as outer:
        with tc.triplet_cache_scope() as inner:
            assert inner is outer
            inner["x"] = 1
        # inner exit must NOT clear the outer cache
        assert tc._active_cache() is outer
        assert outer["x"] == 1
    assert tc._active_cache() is None


def test_scope_restored_on_exception():
    tc = _fresh_module()
    with pytest.raises(RuntimeError):
        with tc.triplet_cache_scope():
            assert tc._active_cache() is not None
            raise RuntimeError("boom")
    assert tc._active_cache() is None


def test_kill_switch_disables_cache(monkeypatch):
    monkeypatch.setenv("POINTELLIGENCE_DISABLE_TRIPLET_CACHE", "1")
    tc = _fresh_module()
    with tc.triplet_cache_scope() as c:
        assert c is None
        assert tc._active_cache() is None


def _toy_inputs(n=512, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 3, generator=g) * 5.0
    sample_inds = torch.zeros(n, dtype=torch.long)
    sample_sizes = torch.tensor([n], dtype=torch.long)
    return pts.to(device), sample_inds.to(device), sample_sizes.to(device)


def _build(pts, si, ss, ks=3, sort_by="k"):
    from layers.triplets import (
        build_triplets, radius_scaler_for_kernel_size, voxelize_3d,
    )
    rs = radius_scaler_for_kernel_size(ks, 1.0, "ball")
    return build_triplets(
        points=pts, sample_inds=si, sample_sizes=ss,
        neighbor_radius=pts.new_tensor(0.0).item() + rs,  # grid_size=1.0
        kernel_indexer=partial(voxelize_3d, kernel_size=ks),
        sort_by=sort_by, return_num_neighbors=False, radius_scaler=rs,
    )


def _spy():
    return mock.patch("layers.triplets.radius_search",
                      wraps=__import__("layers.triplets", fromlist=["radius_search"]).radius_search)


def test_hit_returns_same_objects_and_skips_rebuild():
    tc = _fresh_module()
    pts, si, ss = _toy_inputs()
    with tc.triplet_cache_scope():
        with _spy() as spy:
            i1, j1, k1, _ = _build(pts, si, ss)
            i2, j2, k2, _ = _build(pts, si, ss)
            assert spy.call_count == 1            # 2nd call hit the cache
    assert i1 is i2 and j1 is j2 and k1 is k2     # same tensor objects


def test_no_scope_builds_every_call():
    tc = _fresh_module()
    pts, si, ss = _toy_inputs()
    with _spy() as spy:
        _build(pts, si, ss)
        _build(pts, si, ss)
        assert spy.call_count == 2                # no scope -> no caching


def test_key_distinct_on_kernel_size():
    tc = _fresh_module()
    pts, si, ss = _toy_inputs()
    with tc.triplet_cache_scope():
        with _spy() as spy:
            _build(pts, si, ss, ks=3)
            _build(pts, si, ss, ks=5)             # different kernel -> miss
            assert spy.call_count == 2


def test_key_distinct_on_sort_by():
    tc = _fresh_module()
    pts, si, ss = _toy_inputs()
    with tc.triplet_cache_scope():
        with _spy() as spy:
            _build(pts, si, ss, sort_by="k")
            _build(pts, si, ss, sort_by="j")      # different sort -> miss
            assert spy.call_count == 2


def test_key_distinct_on_sample_inds():
    # SAME points/N, DIFFERENT batch partition (different sample_inds object)
    # must MISS — radius_search confines neighbors per batch element. This is
    # the silent-wrong-result trap the audit caught.
    tc = _fresh_module()
    pts, si, ss = _toy_inputs()
    si2 = si.clone()
    with tc.triplet_cache_scope():
        with _spy() as spy:
            _build(pts, si, ss)
            _build(pts, si2, ss)
            assert spy.call_count == 2


def test_non_partial_kernel_indexer_is_uncacheable_but_safe():
    # A bare lambda is not a functools.partial: _kernel_descriptor returns None
    # so caching is skipped (no AttributeError, no false-hit) and it builds
    # every call.
    from layers.triplets import (
        build_triplets, radius_scaler_for_kernel_size, voxelize_3d,
    )
    tc = _fresh_module()
    pts, si, ss = _toy_inputs()
    rs = radius_scaler_for_kernel_size(3, 1.0, "ball")
    indexer = lambda *a, **k: voxelize_3d(*a, **k, kernel_size=3)  # noqa: E731
    with tc.triplet_cache_scope():
        with _spy() as spy:
            build_triplets(points=pts, sample_inds=si, sample_sizes=ss,
                           neighbor_radius=rs, kernel_indexer=indexer,
                           sort_by="k", return_num_neighbors=False, radius_scaler=rs)
            build_triplets(points=pts, sample_inds=si, sample_sizes=ss,
                           neighbor_radius=rs, kernel_indexer=indexer,
                           sort_by="k", return_num_neighbors=False, radius_scaler=rs)
            assert spy.call_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="build_triplets needs CUDA (Triton)")
def test_key_distinct_on_neighbor_radius():
    # Same points/kernel/sort but a different neighbor_radius (different grid
    # scale) must MISS: the radius sets which neighbors are found AND the grid
    # the kernel bins at. radius_scaler is held fixed so neighbor_radius is the
    # sole varying axis.
    from layers.triplets import (
        build_triplets, radius_scaler_for_kernel_size, voxelize_3d,
    )
    tc = _fresh_module()
    pts, si, ss = _toy_inputs()
    rs = radius_scaler_for_kernel_size(3, 1.0, "ball")
    ki = partial(voxelize_3d, kernel_size=3)
    with tc.triplet_cache_scope():
        with _spy() as spy:
            for grid in (1.0, 2.0):
                build_triplets(points=pts, sample_inds=si, sample_sizes=ss,
                               neighbor_radius=grid * rs, kernel_indexer=ki,
                               sort_by="k", return_num_neighbors=False,
                               radius_scaler=rs)
            assert spy.call_count == 2            # distinct radius -> miss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="build_triplets needs CUDA (Triton)")
def test_weakref_stale_entry_is_rebuilt():
    # A cache entry whose keyed `points` tensor has been freed (dead weakref)
    # must be treated as a MISS and rebuilt, not returned — guarding the
    # freed-then-recycled-data_ptr window. We poison the entry with an
    # already-dead weakref and confirm the next build rebuilds + overwrites.
    import weakref
    tc = _fresh_module()
    pts, si, ss = _toy_inputs()
    with tc.triplet_cache_scope() as cache:
        with _spy() as spy:
            _build(pts, si, ss)                   # build (1) + store
            assert spy.call_count == 1
            (key,) = list(cache.keys())
            cached_out, _pref = cache[key]
            dead = torch.empty(0)
            ref = weakref.ref(dead)
            del dead
            assert ref() is None                  # weakref is dead
            cache[key] = (cached_out, ref)        # poison the live entry
            _build(pts, si, ss)                   # dead ref -> rebuild (2)
            assert spy.call_count == 2
            # entry was overwritten with a LIVE weakref (self-healing)
            _, pref2 = cache[key]
            assert pref2() is not None
