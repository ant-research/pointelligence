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
    """ConvOp adapter == serial conv_with_stride. The conv build (downsample +
    triplets) is deterministic, so the index bundle is asserted BIT-EXACT. PointConv3d's
    forward uses an atomic segment-reduce CUDA kernel whose accumulation order is NOT
    bitwise-reproducible (serial-vs-serial on identical inputs flips torch.equal,
    maxdiff ~1.5e-8; torch.use_deterministic_algorithms does not help), so the feature
    OUTPUT is asserted allclose, not equal."""
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

    # (a) build parity — bit-exact (the adapter's actual contribution).
    m_idx = MetaData(points=pts.clone(), sample_inds=batch.clone(),
                     sample_sizes=ss.clone(), grid_size=0.05)
    bundle = ConvOp(conv, stride=2.0).build_indices(m_idx)
    assert torch.equal(bundle.meta.i, m_ref_out.i)
    assert torch.equal(bundle.meta.j, m_ref_out.j)
    assert torch.equal(bundle.meta.k, m_ref_out.k)
    assert torch.equal(bundle.meta.points, m_ref_out.points)
    assert bundle.meta.contract == m_ref_out.contract

    # (b) full scheduler run — output matches up to the conv kernel's atomic noise.
    m_sched = MetaData(points=pts.clone(), sample_inds=batch.clone(),
                       sample_sizes=ss.clone(), grid_size=0.05)
    y = GeometryScheduler().run([ConvOp(conv, stride=2.0)], x.clone(), geom=m_sched)
    assert torch.allclose(y, y_ref, atol=1e-5)
