"""Tests for the ``recompute_k`` cached fast path of :class:`Upsample`.

``recompute_k`` reuses the cached ``(i_upsample, j_upsample)`` downsample edges
(the expensive radius search) but RE-BUCKETS ``k`` for a possibly-different
``kernel_size`` — so a large-kernel upsample can take the cached fast path even
though the stride conv cached ``k`` at its own (smaller) kernel size.

This geometry is tricky and easy to get wrong, so it is pinned here. The
load-bearing test is :func:`test_recompute_k_full_5cubed_utilization_at_fine_grid`:
the cached upsample offsets span ``±1.86·grid_fine`` (the stride conv's search
radius is built on the FINE grid), so the kernel must be quantized at the fine
grid for a 5³ kernel to actually fill all 5 buckets per axis. Quantizing at the
coarse grid collapses it to the centre 3³ — the exact bug this guards.
"""
import pytest
import torch

from layers.metadata import MetaData
from layers.upsample import Upsample, recompute_cached_upsample_k
from layers.triplets import voxelize_3d, handle_stride_and_build_triplets

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _hand_built_m_low(coarse_pts, fine_pts, i_up, j_up, grid_fine):
    """A coarse ``m_low`` whose ``.parent`` (fine) carries explicit cached
    ``i/j_upsample`` — built WITHOUT a radius search so ``recompute_k``'s
    geometry is tested in isolation (CPU, deterministic)."""
    n_f, n_c = fine_pts.shape[0], coarse_pts.shape[0]
    m_fine = MetaData(
        points=fine_pts,
        sample_inds=torch.zeros(n_f, dtype=torch.long),
        sample_sizes=torch.tensor([n_f]),
        grid_size=grid_fine,
        i_upsample=i_up.long(),
        j_upsample=j_up.long(),
    )
    return MetaData(
        points=coarse_pts,
        sample_inds=torch.zeros(n_c, dtype=torch.long),
        sample_sizes=torch.tensor([n_c]),
        grid_size=grid_fine * 2.0,        # coarse grid = 2× fine
        parent=m_fine,
    )


# ─── pure-geometry tests (no CUDA needed) ──────────────────────────────────

def test_recompute_k_orientation_and_grid():
    """``k = voxelize(source(coarse) − query(fine))`` at the FINE grid.
    One fine query at the origin; coarse sources at +x / −x / coincident."""
    fine = torch.tensor([[0.0, 0.0, 0.0]])
    coarse = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    m_low = _hand_built_m_low(coarse, fine,
                              torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]),
                              grid_fine=1.0)
    k = recompute_cached_upsample_k(m_low, kernel_size=5)
    # 5³, ks//2 = 2:  (+1,0,0)->(3,2,2)=87 ; (-1,0,0)->(1,2,2)=37 ; (0,0,0)->(2,2,2)=62
    assert k.tolist() == [87, 37, 62]


def test_recompute_k_full_5cubed_utilization_at_fine_grid():
    """THE load-bearing invariant. Cached upsample offsets span ±1.86·grid_fine.
    Fine-grid quantization → all 5 buckets/axis fill; coarse-grid quantization
    → the centre 3 only (the bug the fine-grid choice prevents)."""
    fine = torch.tensor([[0.0, 0.0, 0.0]])
    xs = [-1.8, -0.9, 0.0, 0.9, 1.8]            # within ±1.86·grid_fine
    coarse = torch.tensor([[x, 0.0, 0.0] for x in xs])
    i_up = torch.zeros(len(xs), dtype=torch.long)
    j_up = torch.arange(len(xs))
    m_low = _hand_built_m_low(coarse, fine, i_up, j_up, grid_fine=1.0)

    # correct: fine-grid quantization fills all 5 x-buckets
    k_fine = recompute_cached_upsample_k(m_low, kernel_size=5)
    kx_fine = (k_fine // 25).tolist()           # x bucket = k // (5*5)
    assert set(kx_fine) == {0, 1, 2, 3, 4}, kx_fine

    # the bug it prevents: coarse-grid quantization collapses to the centre 3
    offset = m_low.points[j_up] - m_low.parent.points[i_up]
    k_coarse = voxelize_3d(5, offset, grid_size=m_low.grid_size)   # COARSE (wrong)
    assert set((k_coarse // 25).tolist()) == {1, 2, 3}


def test_recompute_k_does_not_need_cached_k():
    """``recompute_k`` reads only ``i/j_upsample`` + coords, never ``k_upsample``,
    so it works when the stride conv never cached ``k`` at this kernel size."""
    m_low = _hand_built_m_low(torch.tensor([[1.0, 0.0, 0.0]]),
                              torch.tensor([[0.0, 0.0, 0.0]]),
                              torch.tensor([0]), torch.tensor([0]), grid_fine=1.0)
    assert m_low.parent.k_upsample is None
    assert recompute_cached_upsample_k(m_low, kernel_size=5).tolist() == [87]


@pytest.mark.parametrize("ks", [3, 5, 7])
def test_recompute_k_indices_in_range(ks):
    """Every recomputed bucket is a valid index in ``[0, ks³)``."""
    torch.manual_seed(0)
    coarse = (torch.rand(20, 3) - 0.5) * (2 * 1.86)     # ±1.86·grid_fine, grid_fine=1
    m_low = _hand_built_m_low(coarse, torch.zeros(1, 3),
                              torch.zeros(20, dtype=torch.long), torch.arange(20),
                              grid_fine=1.0)
    k = recompute_cached_upsample_k(m_low, kernel_size=ks)
    assert int(k.min()) >= 0 and int(k.max()) < ks ** 3


# ─── forward tests (need CUDA + the MVMR extension) ─────────────────────────

def _real_pyramid(grid_size=0.02, n=2000):
    torch.manual_seed(123)
    pts = torch.rand(n, 3, device=DEVICE) * 2.0
    si = torch.zeros(n, dtype=torch.long, device=DEVICE)
    ss = torch.tensor([n], device=DEVICE)
    m_high = MetaData(points=pts, sample_inds=si, sample_sizes=ss, grid_size=grid_size)
    return handle_stride_and_build_triplets(m_high, stride=2.0, kernel_size=(3, 3, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + MVMR ext")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_recompute_k_forward_finite_and_covered(dtype):
    """A 5³ cached-recompute_k upsample: finite forward+backward (incl. fp16),
    correct shape, and coverage that matches the cached edge set (every point
    WITH a cached edge gets output; the tiny uncovered fraction is inherent to
    the stride conv's radius — identical to today's scatter-average head)."""
    m_low = _real_pyramid()
    num_high = m_low.parent.points.shape[0]
    up = Upsample(16, 8, kernel_size=5, bias=False, receptive_field_scaler=0.216,
                  straight_recover=True, recompute_k=True).to(DEVICE).to(dtype)
    x = torch.randn(m_low.points.shape[0], 16, device=DEVICE, dtype=dtype,
                    requires_grad=True)
    x_high, _ = up(x, m_low)

    assert x_high.shape == (num_high, 8)
    assert torch.isfinite(x_high).all()

    # Coverage tracks the cached edges exactly: a point is "covered" iff it is a
    # query in i_upsample. Every covered point must get (generically non-zero)
    # output; only the inherent <2% uncovered tail (no cached edge — same set
    # scatter_avg zeros) may be zero.
    covered = torch.zeros(num_high, dtype=torch.bool, device=DEVICE)
    covered[m_low.parent.i_upsample] = True
    zero_out = x_high.float().norm(dim=1) == 0
    assert not (zero_out & covered).any(), "a point with cached edges got zero output"
    assert (~covered).float().mean().item() < 0.02

    x_high.float().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(up.conv.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + MVMR ext")
def test_recompute_k_reuses_cached_edges_no_search():
    """The cached path must not re-search: with ``recompute_k=True`` the output
    edges are the SAME SET as the cached ``(i_upsample, j_upsample)`` — they are
    re-sorted by the new k (so order differs), but the multiset is unchanged
    (no fresh radius search)."""
    m_low = _real_pyramid()
    up = Upsample(8, 8, kernel_size=5, bias=False, straight_recover=True,
                  recompute_k=True).to(DEVICE)
    x = torch.randn(m_low.points.shape[0], 8, device=DEVICE)
    with torch.no_grad():
        _, m_high = up(x, m_low)
    # same edge multiset (reordered by k), not a fresh search
    assert torch.equal(m_high.i.sort().values, m_low.parent.i_upsample.sort().values)
    assert torch.equal(m_high.j.sort().values, m_low.parent.j_upsample.sort().values)
    # k is sorted ascending (the sort_by="k" contract the grouped paths require)
    assert bool((m_high.k[1:] >= m_high.k[:-1]).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + MVMR ext")
def test_recompute_k_fused_path_C512_fwd_bwd():
    """in_channels >= 512 + fp16 routes the conv to the FUSED CUTLASS path, whose
    grouped VVOR backward builds per-kernel-offset segments and REQUIRES k sorted
    ascending. recompute_k re-buckets k, so without the re-sort this backward
    raises 'o_idx must be sorted ascending'. This is exactly the taper's first
    upsample conv (512->256, k=5); pins that recompute_k keeps the triplets
    properly k-sorted for the fused path."""
    m_low = _real_pyramid()
    up = Upsample(512, 256, kernel_size=5, bias=False, straight_recover=True,
                  recompute_k=True).to(DEVICE).half()
    x = torch.randn(m_low.points.shape[0], 512, device=DEVICE, dtype=torch.float16,
                    requires_grad=True)
    x_high, _ = up(x, m_low)
    assert x_high.shape == (m_low.parent.points.shape[0], 256)
    assert torch.isfinite(x_high).all()
    x_high.float().sum().backward()      # pre-fix: raises in the fused VVOR backward
    assert torch.isfinite(x.grad).all() and torch.isfinite(up.conv.weight.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
