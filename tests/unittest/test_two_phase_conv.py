import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import pytest, torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _toy(n=4096, c=16, b=2, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pts = torch.rand(n, 3, generator=g, device="cuda") * 5.0
    x = torch.randn(n, c, generator=g, device="cuda")
    batch = torch.zeros(n, dtype=torch.long, device="cuda")
    batch[n // 2:] = 1
    return pts, x, batch


def test_convop_scheduler_matches_serial():
    """ConvOp adapter == serial conv_with_stride.

    Radius-search ordering within one kernel-tap segment is intentionally
    unspecified, so independently built rulebooks are compared as canonical
    (k, i, j) sets. PointConv3d also uses atomic reduction; feature output is
    therefore asserted allclose rather than bit-exact.
    """
    from layers.conv import PointConv3d, conv_with_stride
    from layers.metadata import MetaData
    from layers.two_phase_conv import ConvOp
    from internals.two_phase import GeometryScheduler

    pts, x, batch = _toy()
    conv = PointConv3d(16, 16, kernel_size=3).cuda()
    ss = torch.bincount(batch)

    # Serial reference (conv_with_stride mutates m and returns (y, m)).
    m_ref = MetaData(points=pts.clone(), sample_inds=batch.clone(),
                     sample_sizes=ss.clone(), grid_size=0.05)
    y_ref, m_ref_out = conv_with_stride(conv, x.clone(), m_ref, stride=2.0)

    # (a) build parity — exact semantic rulebook, unspecified within-tap order.
    m_idx = MetaData(points=pts.clone(), sample_inds=batch.clone(),
                     sample_sizes=ss.clone(), grid_size=0.05)
    bundle = ConvOp(conv, stride=2.0).build_indices(m_idx)
    num_points_in = pts.shape[0]
    num_points_out = bundle.meta.points.shape[0]

    def canonical(meta):
        key = (
            meta.k.long() * (num_points_out * num_points_in)
            + meta.i.long() * num_points_in
            + meta.j.long()
        )
        return torch.sort(key).values

    assert torch.equal(canonical(bundle.meta), canonical(m_ref_out))
    assert torch.equal(bundle.meta.seg_offs, m_ref_out.seg_offs)
    assert torch.equal(bundle.meta.points, m_ref_out.points)
    assert bundle.meta.contract == m_ref_out.contract

    # (b) full scheduler run — output matches up to the conv kernel's atomic noise.
    m_sched = MetaData(points=pts.clone(), sample_inds=batch.clone(),
                       sample_sizes=ss.clone(), grid_size=0.05)
    y = GeometryScheduler().run([ConvOp(conv, stride=2.0)], x.clone(), geom=m_sched)
    assert torch.allclose(y, y_ref, atol=1e-5)
