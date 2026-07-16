from functools import partial
from unittest import mock

import pytest
import torch

from internals.triplet_cache import triplet_cache_scope
from layers.contract import TripletContract
from layers.conv import PointConv3d
from layers.triplets import (
    build_triplets,
    build_triplets_segmented,
    radius_scaler_for_kernel_size,
    should_use_direct_segmented_triplets,
    voxelize_3d,
)
from sparse_engines._dispatch_override import dispatch_mode


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")


def _geometry(n=1800, batches=2, seed=41):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    points = torch.rand(n, 3, generator=generator, device="cuda") * 2 - 1
    sample_inds = torch.arange(n, device="cuda") // (n // batches)
    sample_inds.clamp_max_(batches - 1)
    sample_sizes = torch.bincount(sample_inds)
    return points, sample_inds, sample_sizes


def _canonical(i, j, k, num_queries, num_points):
    key = (
        k.long() * (num_queries * num_points)
        + i.long() * num_points
        + j.long()
    )
    return torch.sort(key).values


@pytest.mark.parametrize(
    "kernel_size,expected",
    [
        (1, True),
        (3, True),
        (5, True),
        (7, True),
        (15, True),
        ((7, 7, 5), False),
        (8, False),
    ],
)
def test_direct_segment_dispatch_is_shape_static(kernel_size, expected):
    assert should_use_direct_segmented_triplets(kernel_size) is expected


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7, 15])
def test_segmented_builder_matches_generic_builder(kernel_size):
    points, sample_inds, sample_sizes = _geometry()
    radius_scaler = radius_scaler_for_kernel_size(kernel_size)
    neighbor_radius = 0.02 * radius_scaler
    generic = build_triplets(
        points=points,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        neighbor_radius=neighbor_radius,
        kernel_indexer=partial(voxelize_3d, kernel_size=kernel_size),
        return_num_neighbors=True,
        radius_scaler=radius_scaler,
    )
    segmented = build_triplets_segmented(
        points=points,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        neighbor_radius=neighbor_radius,
        kernel_size=kernel_size,
        return_num_neighbors=True,
        radius_scaler=radius_scaler,
    )
    i, j, k, seg_offs, counts = segmented
    assert torch.equal(
        _canonical(*generic[:3], points.shape[0], points.shape[0]),
        _canonical(i, j, k, points.shape[0], points.shape[0]),
    )
    assert torch.equal(generic[3], counts)
    assert torch.equal(
        seg_offs.diff(),
        torch.bincount(k.long(), minlength=kernel_size ** 3),
    )
    assert i.dtype == j.dtype == k.dtype == torch.int32
    assert seg_offs.dtype == torch.int64


def test_segmented_builder_cache_reuses_all_outputs():
    import layers.triplets as triplets_module

    points, sample_inds, sample_sizes = _geometry()
    kernel_size = 3
    radius_scaler = radius_scaler_for_kernel_size(kernel_size)
    kwargs = dict(
        points=points,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        neighbor_radius=0.05 * radius_scaler,
        kernel_size=kernel_size,
        return_num_neighbors=True,
        radius_scaler=radius_scaler,
    )
    with mock.patch.object(
        triplets_module,
        "radius_search_sorted_grid8_segments",
        wraps=triplets_module.radius_search_sorted_grid8_segments,
    ) as spy:
        with triplet_cache_scope():
            first = build_triplets_segmented(**kwargs)
            second = build_triplets_segmented(**kwargs)
        assert spy.call_count == 1
    assert all(a is b for a, b in zip(first, second))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
def test_pointconv_prebuilt_segments_forward_backward_parity(
    dtype, kernel_size
):
    points, sample_inds, sample_sizes = _geometry(n=1000)
    radius_scaler = radius_scaler_for_kernel_size(kernel_size)
    i, j, k, seg_offs, _ = build_triplets_segmented(
        points=points,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        neighbor_radius=0.04 * radius_scaler,
        kernel_size=kernel_size,
        radius_scaler=radius_scaler,
    )
    conv = PointConv3d(
        16, 24, kernel_size=kernel_size, bias=False, dtype=dtype,
        device="cuda")
    feat_a = torch.randn(
        points.shape[0], 16, device="cuda", dtype=dtype, requires_grad=True)
    feat_b = feat_a.detach().clone().requires_grad_(True)
    grad_out = torch.randn(
        points.shape[0], 24, device="cuda", dtype=dtype)

    with dispatch_mode("force_tig"):
        out_a = conv(
            feat_a, i, j, k, points.shape[0],
            contract=TripletContract.submanifold())
        out_a.backward(grad_out)
        grad_feat_a = feat_a.grad.detach().clone()
        grad_weight_a = conv.weight.grad.detach().clone()
        conv.weight.grad = None

        out_b = conv(
            feat_b, i, j, k, points.shape[0],
            contract=TripletContract.submanifold(), seg_offs=seg_offs)
        out_b.backward(grad_out)
        grad_feat_b = feat_b.grad.detach().clone()
        grad_weight_b = conv.weight.grad.detach().clone()

    if dtype == torch.float32:
        rtol, atol = 2e-5, 2e-5
    else:
        rtol, atol = 3e-2, 3e-2
    torch.testing.assert_close(out_b, out_a, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        grad_feat_b, grad_feat_a, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        grad_weight_b, grad_weight_a, rtol=rtol, atol=atol)
