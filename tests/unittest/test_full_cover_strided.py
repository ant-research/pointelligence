import math
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from layers import MetaData, PointConv3d, conv_with_stride_full_cover
from layers.triplets import (
    build_full_cover_strided_rulebook,
    full_cover_radius_scaler,
    minimum_full_cover_kernel_size,
)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _edge_keys(i, j, n_points):
    return (i.long() * (n_points + 1) + j.long()).detach().cpu()


def _fixture_points(dtype=torch.float32):
    pts0 = torch.tensor([
        [0.001, 0.001, 0.001],
        [0.399, 0.399, 0.399],
        [0.401, 0.001, 0.001],
        [0.799, 0.399, 0.399],
    ], device=_device(), dtype=dtype)
    pts1 = pts0 + torch.tensor([2.0, 0.0, 0.0], device=_device(), dtype=dtype)
    points = torch.cat([pts0, pts1], dim=0)
    sample_inds = torch.cat([
        torch.zeros(pts0.shape[0], device=_device(), dtype=torch.long),
        torch.ones(pts1.shape[0], device=_device(), dtype=torch.long),
    ])
    sample_sizes = torch.bincount(sample_inds)
    return points, sample_inds, sample_sizes


def test_full_cover_radius_formula_stride8_grid005():
    scaler = full_cover_radius_scaler(8, 1e-2)
    radius = 0.05 * scaler
    assert scaler == pytest.approx(6.99748526, rel=1e-7)
    assert radius == pytest.approx(0.34987426, rel=1e-7)
    assert minimum_full_cover_kernel_size(scaler) == 15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_full_cover_requires_k15_for_stride8_grid005():
    points, sample_inds, sample_sizes = _fixture_points()
    with pytest.raises(ValueError, match="needs kernel_size >= 15"):
        build_full_cover_strided_rulebook(
            points,
            sample_inds,
            sample_sizes,
            stride=8,
            input_grid_size=0.05,
            kernel_size=13,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_full_cover_geometry_and_bruteforce_edges():
    points, sample_inds, sample_sizes = _fixture_points()
    rb = build_full_cover_strided_rulebook(
        points,
        sample_inds,
        sample_sizes,
        stride=8,
        input_grid_size=0.05,
        kernel_size=15,
    )

    assert torch.equal(rb.points, points[rb.center_source_indices.long()])
    assert rb.additional_center_source_indices.numel() > 0
    assert torch.all(rb.coverage_per_input >= 1)
    assert rb.selector_round_count >= 1
    assert torch.all(rb.sample_inds[:-1] <= rb.sample_inds[1:])
    assert torch.all(sample_inds[rb.j.long()] == rb.sample_inds[rb.i.long()])
    assert int(rb.k.min()) >= 0
    assert int(rb.k.max()) < 15 ** 3

    c0 = rb.initial_center_source_indices.long()
    c1 = rb.additional_center_source_indices.long()
    if c1.numel() > 0:
        dist_c1_c0 = torch.cdist(points[c1], points[c0])
        same_batch = sample_inds[c1].unsqueeze(1) == sample_inds[c0].unsqueeze(0)
        assert torch.all(dist_c1_c0[same_batch] > rb.radius)
    if c1.numel() > 1:
        d = torch.cdist(points[c1], points[c1])
        same = sample_inds[c1].unsqueeze(1) == sample_inds[c1].unsqueeze(0)
        eye = torch.eye(c1.numel(), device=points.device, dtype=torch.bool)
        mask = same & ~eye
        if mask.any():
            assert torch.all(d[mask] > rb.radius)

    d = torch.cdist(rb.points, points)
    same_batch = rb.sample_inds.unsqueeze(1) == sample_inds.unsqueeze(0)
    expected = (d <= rb.radius) & same_batch
    q, p = torch.nonzero(expected, as_tuple=True)
    got_keys = set(_edge_keys(rb.i, rb.j, points.shape[0]).tolist())
    exp_keys = set(_edge_keys(q, p, points.shape[0]).tolist())
    assert got_keys == exp_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_full_cover_deterministic_ties_and_ordering():
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.4, 0.4, 0.4],
        [0.8, 0.8, 0.8],
        [1.2, 1.2, 1.2],
    ], device="cuda")
    sample_inds = torch.zeros(points.shape[0], device="cuda", dtype=torch.long)
    sample_sizes = torch.bincount(sample_inds)
    rb1 = build_full_cover_strided_rulebook(
        points, sample_inds, sample_sizes, stride=8, input_grid_size=0.05,
        kernel_size=15)
    rb2 = build_full_cover_strided_rulebook(
        points, sample_inds, sample_sizes, stride=8, input_grid_size=0.05,
        kernel_size=15)
    assert torch.equal(rb1.center_source_indices, rb2.center_source_indices)
    assert torch.equal(rb1.i, rb2.i)
    assert torch.equal(rb1.j, rb2.j)
    assert torch.equal(rb1.k, rb2.k)
    assert torch.equal(rb1.center_source_indices, torch.sort(rb1.center_source_indices).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_full_cover_unpatch_rulebook_is_mirrored_and_sorted():
    points, sample_inds, sample_sizes = _fixture_points()
    rb = build_full_cover_strided_rulebook(
        points, sample_inds, sample_sizes, stride=8, input_grid_size=0.05,
        kernel_size=15)
    assert torch.equal(rb.k_upsample, torch.sort(rb.k_upsample).values)
    down_pairs = set(zip(rb.i.detach().cpu().tolist(), rb.j.detach().cpu().tolist()))
    up_pairs = set(zip(rb.j_upsample.detach().cpu().tolist(),
                       rb.i_upsample.detach().cpu().tolist()))
    assert up_pairs == down_pairs

    K = 15
    half = K // 2
    kd = rb.k.long()
    dz = kd % K - half
    dy = (kd // K) % K - half
    dx = kd // (K * K) - half
    ku_expected = ((-dx + half) * K + (-dy + half)) * K + (-dz + half)
    assert torch.equal(torch.sort(ku_expected).values, rb.k_upsample.long())


def _reference_pointconv(x, weight, bias, i, j, k, n_out):
    out = x.new_zeros((n_out, weight.shape[-1]))
    for edge in range(k.numel()):
        out[i[edge].long()] = out[i[edge].long()] + x[j[edge].long()].matmul(
            weight[k[edge].long(), 0])
    if bias is not None:
        out = out + bias
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_conv_with_stride_full_cover_forward_backward_reference(dtype):
    points, sample_inds, sample_sizes = _fixture_points()
    x = torch.randn(points.shape[0], 4, device="cuda", dtype=dtype,
                    requires_grad=True)
    conv = PointConv3d(4, 3, kernel_size=15, bias=True, device="cuda",
                       dtype=dtype)
    m = MetaData(points=points, sample_inds=sample_inds,
                 sample_sizes=sample_sizes, grid_size=0.05)
    out, m_out = conv_with_stride_full_cover(conv, x, m, stride=8)
    ref = _reference_pointconv(
        x, conv.weight, conv.bias, m_out.i, m_out.j, m_out.k,
        m_out.num_points())
    tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-5
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)

    grad = torch.randn_like(out)
    out.backward(grad, retain_graph=True)
    gx = x.grad.detach().clone()
    gw = conv.weight.grad.detach().clone()
    gb = conv.bias.grad.detach().clone()
    x.grad = None
    conv.weight.grad = None
    conv.bias.grad = None
    ref.backward(grad)
    torch.testing.assert_close(gx, x.grad, atol=tol, rtol=tol)
    torch.testing.assert_close(gw, conv.weight.grad, atol=tol, rtol=tol)
    torch.testing.assert_close(gb, conv.bias.grad, atol=tol, rtol=tol)

    assert m_out.parent.i_upsample is not None
    assert m_out.parent.full_cover_point_to_initial_center is not None
    assert m_out.parent.full_cover_telemetry["additional_center_count"] >= 1
