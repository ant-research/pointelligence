import pytest
import torch

import internals.neighbors as neighbors_module


def _points(count):
    return torch.empty((count, 3), dtype=torch.float32)


def _install_backend_spies(monkeypatch):
    calls = []

    def make_spy(name):
        def spy(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name
        return spy

    monkeypatch.setattr(
        neighbors_module, "radius_search_sorted_grid8", make_spy("sorted8"))
    monkeypatch.setattr(
        neighbors_module, "radius_search_fixed_grid", make_spy("sorted27"))
    monkeypatch.setattr(
        neighbors_module, "radius_search_tiled", make_spy("tiled"))
    return calls


@pytest.mark.parametrize(
    "point_count,query_count,grid_size,radius",
    [
        (32, 16, None, 0.1),
        (1024, 256, 0.02, 0.01),
        (20001, 20001, 0.02, 0.2),
        (1000001, 500001, 1.0, 5.0),
    ],
)
def test_auto_always_selects_sorted8_materialized(
    monkeypatch, point_count, query_count, grid_size, radius
):
    calls = _install_backend_spies(monkeypatch)
    result = neighbors_module.radius_search(
        _points(point_count),
        _points(query_count),
        radius=radius,
        grid_size=grid_size,
        point_num_max=1,
        tiled_batch_threshold=10**9,
        tiled_radius_multiplier_threshold=0.0,
        fixed_grid_radius_multiplier_threshold=0.0,
    )
    assert result == "sorted8"
    assert len(calls) == 1
    name, args, kwargs = calls[0]
    assert name == "sorted8"
    assert args[-1] == "ball"
    assert kwargs == {}


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("sorted8_materialized", "sorted8"),
        ("sorted27_materialized", "sorted27"),
        ("sorted_grid8", "sorted8"),
        ("fixed_grid", "sorted27"),
    ],
)
def test_forced_sorted_backends(monkeypatch, backend, expected):
    calls = _install_backend_spies(monkeypatch)
    result = neighbors_module.radius_search(
        _points(32), _points(16), radius=0.1, backend=backend)
    assert result == expected
    assert len(calls) == 1
    name, args, kwargs = calls[0]
    assert name == expected
    assert args[-1] == "ball"
    assert kwargs == {}


def test_forced_tiled_backend(monkeypatch):
    calls = _install_backend_spies(monkeypatch)
    result = neighbors_module.radius_search(
        _points(32), _points(16), radius=0.1, backend="tiled")
    assert result == "tiled"
    assert calls[0][0] == "tiled"
    assert calls[0][2]["block_p"] == 4096


@pytest.mark.parametrize(
    "backend", [
        "lookup", "lookup_compact",
        "sorted8_fused", "sorted8_hybrid",
        "sorted27_fused", "sorted27_hybrid",
        "fixed_grid_fused", "fixed_grid_hybrid",
    ])
def test_legacy_shifted_lookup_is_not_a_production_backend(monkeypatch, backend):
    _install_backend_spies(monkeypatch)
    with pytest.raises(ValueError, match="backend must be"):
        neighbors_module.radius_search(
            _points(32), _points(16), radius=0.1, backend=backend)


def test_auto_dispatch_rejects_nonpositive_grid_size(monkeypatch):
    _install_backend_spies(monkeypatch)
    with pytest.raises(ValueError, match="grid_size must be positive"):
        neighbors_module.radius_search(
            _points(32), _points(16), radius=0.1, grid_size=0.0)
