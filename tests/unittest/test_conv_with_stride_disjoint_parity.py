"""Partition tests — conv_with_stride_disjoint is a true GRID PARTITION.

Supersedes the prior ball-parity battery (conv_with_stride_disjoint used to be a
fast *ball gather* asserted equal to conv_with_stride; it is now a genuine
non-overlapping, fully-covering cubic partition — the ViT-patchify operator). The
properties under test: every input point maps to exactly one output cell (100%
coverage, no orphaning), the kernel slot is the cell-grid sub-voxel index bounded
to [0, K**3), tokens sit at cell centers, and an Upsample(straight_recover,
recompute_k=False) recovers every input point from its cell-center token.
"""
import pytest
import torch

from layers import PointConv3d
from layers.conv import conv_with_stride_disjoint
from layers.upsample import Upsample
from layers.metadata import MetaData
from layers.triplets import handle_stride_disjoint_and_build_triplets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _meta(n=20000, scale=4.0, g0=0.02, nbatch=1):
    torch.manual_seed(0)
    pts = torch.rand(n, 3, device="cuda") * scale
    si = torch.randint(0, nbatch, (n,), device="cuda").sort().values
    return MetaData(points=pts, sample_inds=si, sample_sizes=torch.bincount(si),
                    grid_size=g0)


def test_partition_full_coverage_no_orphan():
    """Every input point is exactly one edge's source j (100% coverage) — the
    property the ball gather violated (it orphaned ~25-50% of corner points)."""
    for nbatch in (1, 3):
        m = _meta(nbatch=nbatch)
        n = m.points.shape[0]
        m = handle_stride_disjoint_and_build_triplets(m, stride=8, kernel_size=8)
        assert m.j.numel() == n, "one edge per input point"
        assert torch.unique(m.j).numel() == n, "every input point covered (no orphan)"
        assert torch.unique(m.parent.i_upsample).numel() == n, "upsample covers all points"


def test_partition_slot_k_bounded():
    """Slot k is the cell-grid sub-voxel index, bounded [0, K**3) with no clamp."""
    for K in (4, 8):
        m = _meta()
        m = handle_stride_disjoint_and_build_triplets(m, stride=K, kernel_size=K)
        assert int(m.k.min()) >= 0 and int(m.k.max()) < K ** 3


def test_partition_tokens_at_cell_centers():
    """Output token coords are the cell centers (cell_vox + 0.5) * cell_size."""
    g0, stride = 0.02, 8
    m = _meta(g0=g0)
    m = handle_stride_disjoint_and_build_triplets(m, stride=stride, kernel_size=stride)
    cell = g0 * stride
    frac = (m.points / cell) - torch.floor(m.points / cell)
    assert torch.allclose(frac, torch.full_like(frac, 0.5), atol=1e-4), "tokens at cell centers"


def test_stride_le_1_rejects():
    m = _meta()
    with pytest.raises(ValueError):
        handle_stride_disjoint_and_build_triplets(m, stride=1.0, kernel_size=8)


def test_noncubic_kernel_rejects():
    m = _meta()
    with pytest.raises(ValueError):
        handle_stride_disjoint_and_build_triplets(m, stride=8, kernel_size=(8, 8, 4))


def test_conv_and_unpatchify_roundtrip():
    """conv_with_stride_disjoint (patchify) → Upsample(straight_recover,
    recompute_k=False) (unpatchify) runs end-to-end and recovers every raw point
    (output rows == N_in, finite, no row left at the bias)."""
    m = _meta()
    n = m.points.shape[0]
    feat = torch.randn(n, 6, device="cuda")
    patch = PointConv3d(6, 256, kernel_size=8, bias=False).cuda()
    tok, m2 = conv_with_stride_disjoint(patch, feat, m, stride=8)
    assert tok.shape == (m2.points.shape[0], 256) and torch.isfinite(tok).all()
    up = Upsample(256, 64, kernel_size=8, bias=False,
                  straight_recover=True, recompute_k=False).cuda()
    out, m3 = up(tok, m2)
    assert out.shape == (n, 64), "every raw point recovered"
    assert torch.isfinite(out).all()
    assert torch.unique(m2.parent.i_upsample).numel() == n
