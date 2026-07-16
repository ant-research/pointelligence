import itertools
import math

import pytest
import torch

from internals.neighbors import (
    radius_search_fixed_grid,
    radius_search_sorted_grid8,
    radius_search_sorted_grid8_segments,
)
from layers.triplets import voxelize_3d

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _canonical(result, num_points):
    neighbors, counts, *rest = result
    query = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts.long())
    key = query.to(torch.int64) * num_points + neighbors.to(torch.int64)
    order = torch.argsort(key)
    key = key[order]
    assert torch.unique(key).numel() == key.numel()
    distances = rest[0][order] if rest else None
    return key, counts, distances


@pytest.mark.parametrize("distance_type", ["ball", "chebyshev"])
@pytest.mark.parametrize("return_distances", [False, True])
@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
def test_sorted8_and_sorted27_match(
    distance_type, return_distances, count_dtype
):
    generator = torch.Generator(device="cuda").manual_seed(17)
    points = (torch.rand(3072, 3, device="cuda", generator=generator) - 0.5) * 4
    queries = points[::3].clone()
    kwargs = dict(
        points=points,
        queries=queries,
        radius=0.19,
        return_distances=return_distances,
        dtype_num_neighbors=count_dtype,
        distance_type=distance_type,
    )
    sorted27 = _canonical(radius_search_fixed_grid(**kwargs), points.shape[0])
    sorted8 = _canonical(radius_search_sorted_grid8(**kwargs), points.shape[0])
    assert torch.equal(sorted8[0], sorted27[0])
    assert torch.equal(sorted8[1], sorted27[1])
    assert sorted8[1].dtype == count_dtype
    if return_distances:
        torch.testing.assert_close(sorted8[2], sorted27[2], rtol=0, atol=0)


def test_sorted_grid_all_offsets_boundaries_and_batches():
    radius = 1.0
    eps = torch.finfo(torch.float32).eps
    offsets = torch.tensor(
        list(itertools.product((-1.0, 0.0, 1.0), repeat=3)),
        device="cuda",
    )
    points0 = offsets * (radius - 8 * eps)
    points0 = torch.cat((points0, torch.tensor([
        [radius, 0, 0], [radius + 8 * eps, 0, 0],
        [-radius, 0, 0], [-radius - 8 * eps, 0, 0],
    ], device="cuda")))
    points = torch.cat((points0, points0), dim=0)
    queries = torch.zeros((2, 3), device="cuda")
    sample = torch.cat((
        torch.zeros(points0.shape[0], device="cuda", dtype=torch.long),
        torch.ones(points0.shape[0], device="cuda", dtype=torch.long),
    ))
    query_sample = torch.arange(2, device="cuda", dtype=torch.long)
    kwargs = dict(
        points=points, queries=queries, radius=radius,
        sample_inds=sample, query_sample_inds=query_sample,
        distance_type="ball", return_distances=True,
    )
    sorted27 = _canonical(radius_search_fixed_grid(**kwargs), points.shape[0])
    sorted8 = _canonical(radius_search_sorted_grid8(**kwargs), points.shape[0])
    assert torch.equal(sorted8[0], sorted27[0])
    assert torch.equal(sorted8[1], sorted27[1])
    torch.testing.assert_close(sorted8[2], sorted27[2], rtol=0, atol=0)
    query_ids = torch.div(sorted8[0], points.shape[0], rounding_mode="floor")
    point_ids = sorted8[0].remainder(points.shape[0])
    assert torch.equal(query_ids, sample[point_ids])


@pytest.mark.parametrize(
    "search", [radius_search_sorted_grid8, radius_search_fixed_grid])
@pytest.mark.parametrize("num_points,num_queries", [(0, 0), (0, 3), (3, 0)])
def test_sorted_grid_empty_inputs(search, num_points, num_queries):
    points = torch.empty((num_points, 3), device="cuda")
    queries = torch.empty((num_queries, 3), device="cuda")
    neighbors, counts, distances = search(
        points, queries, 1.0, return_distances=True)
    assert neighbors.numel() == 0
    assert counts.shape == (num_queries,)
    assert distances.numel() == 0


def test_sorted_grid_entire_pipeline_runs_without_grad(monkeypatch):
    observed = []

    def checked(original):
        def wrapper(*args, **kwargs):
            observed.append(torch.is_grad_enabled())
            return original(*args, **kwargs)
        return wrapper

    for name in ("floor", "argsort", "searchsorted", "cumsum"):
        monkeypatch.setattr(torch, name, checked(getattr(torch, name)))

    points = torch.rand(256, 3, device="cuda", requires_grad=True)
    queries = points[::4]
    with torch.enable_grad():
        out27 = radius_search_fixed_grid(
            points, queries, 0.2, return_distances=True)
        out8 = radius_search_sorted_grid8(
            points, queries, 0.2, return_distances=True)
        assert torch.is_grad_enabled()

    assert observed
    assert not any(observed)
    assert not out27[2].requires_grad
    assert not out8[2].requires_grad


def _canonical_triplets(i, j, k, num_queries, num_points):
    key = (
        k.to(torch.int64) * (num_queries * num_points)
        + i.to(torch.int64) * num_points
        + j.to(torch.int64)
    )
    key = torch.sort(key).values
    assert torch.unique(key).numel() == key.numel()
    return key


@pytest.mark.parametrize("tap_stripes", [1, 8, 64])
@pytest.mark.parametrize("distance_type", ["ball", "chebyshev"])
@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "kernel_size,radius,kernel_grid_size",
    [
        (3, 0.05, 0.05),
        (15, math.sqrt(3) * 0.2 * 1.01, 0.05),
    ],
)
def test_sorted8_segments_match_radius_then_voxelize_and_sort(
    tap_stripes, distance_type, count_dtype, kernel_size, radius,
    kernel_grid_size,
):
    generator = torch.Generator(device="cuda").manual_seed(29 + kernel_size)
    point_parts = [
        torch.rand(1500, 3, device="cuda", generator=generator) * 1.8 - 0.9,
        torch.rand(1100, 3, device="cuda", generator=generator) * 1.4 - 0.5,
    ]
    query_parts = [
        torch.rand(400, 3, device="cuda", generator=generator) * 1.8 - 0.9,
        torch.rand(300, 3, device="cuda", generator=generator) * 1.4 - 0.5,
    ]
    points = torch.cat(point_parts)
    queries = torch.cat(query_parts)
    sample_inds = torch.cat([
        torch.full(
            (part.shape[0],), batch, dtype=torch.int32, device="cuda")
        for batch, part in enumerate(point_parts)
    ])
    query_sample_inds = torch.cat([
        torch.full(
            (part.shape[0],), batch, dtype=torch.int32, device="cuda")
        for batch, part in enumerate(query_parts)
    ])

    neighbors, counts = radius_search_sorted_grid8(
        points,
        queries,
        radius,
        sample_inds,
        query_sample_inds,
        dtype_num_neighbors=count_dtype,
        distance_type=distance_type,
    )
    i, j, seg_offs, prepared_counts = radius_search_sorted_grid8_segments(
        points,
        queries,
        radius,
        kernel_size,
        kernel_grid_size,
        sample_inds,
        query_sample_inds,
        dtype_num_neighbors=count_dtype,
        distance_type=distance_type,
        tap_stripes=tap_stripes,
    )

    baseline_i = torch.repeat_interleave(
        torch.arange(
            queries.shape[0], dtype=torch.int32, device="cuda"),
        counts,
    )
    baseline_k = voxelize_3d(
        kernel_size,
        points[neighbors] - queries[baseline_i],
        torch.tensor(kernel_grid_size, device="cuda"),
    )
    prepared_k = torch.repeat_interleave(
        torch.arange(kernel_size ** 3, dtype=torch.int32, device="cuda"),
        seg_offs.diff(),
    )

    baseline_key = _canonical_triplets(
        baseline_i, neighbors, baseline_k, queries.shape[0], points.shape[0])
    prepared_key = _canonical_triplets(
        i, j, prepared_k, queries.shape[0], points.shape[0])
    assert torch.equal(prepared_key, baseline_key)
    assert torch.equal(prepared_counts, counts)
    assert prepared_counts.dtype == count_dtype
    assert seg_offs.dtype == torch.int64
    assert seg_offs.shape == (kernel_size ** 3 + 1,)
    assert seg_offs[0] == 0
    assert seg_offs[-1] == i.numel()
    assert torch.equal(
        seg_offs.diff(),
        torch.bincount(baseline_k.long(), minlength=kernel_size ** 3),
    )
    assert torch.equal(
        torch.bincount(i.long(), minlength=queries.shape[0]).to(count_dtype),
        prepared_counts,
    )
    assert torch.equal(query_sample_inds[i.long()], sample_inds[j.long()])


def test_sorted8_segments_round_to_even_ties_match_voxelize():
    points = torch.tensor(
        [
            [-0.75, 0.00, 0.00],
            [-0.25, 0.00, 0.00],
            [0.25, 0.00, 0.00],
            [0.75, 0.00, 0.00],
            [0.00, -0.75, 0.25],
            [0.00, 0.75, -0.25],
        ],
        device="cuda",
    )
    queries = torch.zeros((1, 3), device="cuda")
    kernel_size = 5
    neighbors, counts = radius_search_sorted_grid8(
        points, queries, radius=1.0)
    i, j, seg_offs, prepared_counts = radius_search_sorted_grid8_segments(
        points,
        queries,
        radius=1.0,
        kernel_size=kernel_size,
        kernel_grid_size=0.5,
    )
    baseline_i = torch.zeros_like(neighbors, dtype=torch.int32)
    baseline_k = voxelize_3d(
        kernel_size,
        points[neighbors],
        torch.tensor(0.5, device="cuda"),
    )
    prepared_k = torch.repeat_interleave(
        torch.arange(kernel_size ** 3, dtype=torch.int32, device="cuda"),
        seg_offs.diff(),
    )
    assert torch.equal(counts, prepared_counts)
    assert torch.equal(
        _canonical_triplets(
            baseline_i, neighbors, baseline_k, 1, points.shape[0]),
        _canonical_triplets(i, j, prepared_k, 1, points.shape[0]),
    )


@pytest.mark.parametrize("num_points,num_queries", [(0, 0), (0, 3), (3, 0)])
def test_sorted8_segments_empty_inputs(num_points, num_queries):
    points = torch.empty((num_points, 3), device="cuda")
    queries = torch.empty((num_queries, 3), device="cuda")
    i, j, seg_offs, counts = radius_search_sorted_grid8_segments(
        points, queries, 0.2, 3, 0.2)
    assert i.dtype == torch.int32 and i.numel() == 0
    assert j.dtype == torch.int32 and j.numel() == 0
    assert torch.equal(seg_offs, torch.zeros(28, dtype=torch.int64, device="cuda"))
    assert counts.shape == (num_queries,)


@pytest.mark.parametrize(
    "kernel_size,kernel_grid_size",
    [(2, 0.1), (0, 0.1), (3, 0.0)],
)
def test_sorted8_segments_reject_invalid_kernel(
    kernel_size, kernel_grid_size
):
    points = torch.zeros((1, 3), device="cuda")
    with pytest.raises(ValueError):
        radius_search_sorted_grid8_segments(
            points,
            points,
            radius=0.2,
            kernel_size=kernel_size,
            kernel_grid_size=kernel_grid_size,
        )


@pytest.mark.parametrize("tap_stripes", [0, 3, 512])
def test_sorted8_segments_reject_invalid_tap_stripes(tap_stripes):
    points = torch.zeros((1, 3), device="cuda")
    with pytest.raises(ValueError, match="tap_stripes"):
        radius_search_sorted_grid8_segments(
            points, points, 0.2, 3, 0.2, tap_stripes=tap_stripes)


def test_sorted8_segments_matches_correctly_rounded_fp32_division():
    # 0.09 / float32(0.02) is 4.50000048. A reciprocal-multiply fast
    # division rounds this to the 4.5 tie and selects tap 4; voxelize_3d's
    # correctly-rounded division selects tap 5.
    points = torch.tensor([[0.09, 0.0, 0.0]], device="cuda")
    queries = torch.zeros((1, 3), device="cuda")
    kernel_size = 15
    kernel_grid_size = 0.02
    neighbors, counts = radius_search_sorted_grid8(
        points, queries, radius=0.1)
    i, j, seg_offs, prepared_counts = radius_search_sorted_grid8_segments(
        points,
        queries,
        radius=0.1,
        kernel_size=kernel_size,
        kernel_grid_size=kernel_grid_size,
    )
    baseline_k = voxelize_3d(
        kernel_size,
        points[neighbors],
        torch.tensor(kernel_grid_size, device="cuda"),
    )
    prepared_k = torch.repeat_interleave(
        torch.arange(kernel_size ** 3, dtype=torch.int32, device="cuda"),
        seg_offs.diff(),
    )
    assert torch.equal(counts, prepared_counts)
    assert torch.equal(i, torch.zeros_like(i))
    assert torch.equal(j.long(), neighbors.long())
    assert torch.equal(prepared_k, baseline_k)
