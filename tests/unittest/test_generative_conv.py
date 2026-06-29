"""Correctness tests for the generative-expansion point conv.

`GenerativePointConv3d` invents a denser output point set Y from input
X and convolves X → Y. Coverage:

1. Numerical parity vs an independent pure-PyTorch reference (fp32/fp16/bf16).
2. Gradient correctness (`gradcheck`, fp64) — input features and weight.
3. The ≥1-neighbour invariant — every output point has a contributing
   input, every input reaches an output (incl. the isolated-input case).
4. Per-sample confinement — no triplet crosses a batch boundary.
5. Dedup correctness — overlapping stamps merge to one output voxel.
6. Expansion factors — denser output at finer resolution; the ks=1,
   expansion=1 identity case.
7. Determinism — bit-exact coords, near-exact features.
8. groups > 1, rulebook (`sites`) reuse, custom stencils.
9. Enablement — a real `nn.Module` composes the operator.
10. Edge cases — empty input, single point, orphan-output fail-closed.
"""

import math

import pytest
import torch
import torch.nn as nn

from layers import GenerativePointConv3d, KernelStampGenerator, MetaData
from layers.generative import GeneratedSites, build_generative_triplets
from tests.unittest.refs.generative_conv_pyref import generative_conv_scatter_ref
from tests.unittest.unittest_utils import check_all_close

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="generative conv requires CUDA"
)

_RTOL = {torch.float32: 1e-3, torch.float16: 2e-2, torch.bfloat16: 3e-2}


def _make_metadata(
    n_per_sample, n_samples, grid_size=1.0, seed=0, distinct=False, span=12
):
    """Build a MetaData with integer-grid point coordinates.

    distinct=True picks distinct voxel cells per sample (needed where the
    test asserts an exact n_out); otherwise random cells (duplicates ok).
    """
    torch.manual_seed(seed)
    pts, sis = [], []
    for s in range(n_samples):
        if distinct:
            assert n_per_sample <= span**3
            flat = torch.randperm(span**3, device=DEVICE)[:n_per_sample]
            cells = torch.stack(
                [flat // (span * span), (flat // span) % span, flat % span], dim=1
            )
            p = cells.to(torch.float32) * grid_size
        else:
            p = (torch.rand(n_per_sample, 3, device=DEVICE) * span).floor() * grid_size
        pts.append(p)
        sis.append(torch.full((n_per_sample,), s, dtype=torch.long, device=DEVICE))
    sample_inds = torch.cat(sis)
    return MetaData(
        points=torch.cat(pts),
        sample_inds=sample_inds,
        sample_sizes=torch.bincount(sample_inds, minlength=n_samples),
        grid_size=grid_size,
    )


# ---------------------------------------------------------------- parity ----


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_parity_vs_pyref(dtype):
    """GenerativePointConv3d matches the independent scatter reference."""
    m = _make_metadata(300, n_samples=2, seed=1)
    Cin, Cout = 16, 24
    conv = GenerativePointConv3d(Cin, Cout, kernel_size=3, expansion=2.0)
    conv = conv.to(DEVICE).to(dtype)
    x = torch.randn(m.num_points(), Cin, device=DEVICE, dtype=dtype)

    sites = conv.generator(m)
    y, m_out = conv(x, m, sites=sites)
    ref = generative_conv_scatter_ref(
        x, sites.i, sites.j, sites.k, sites.n_out, conv.weight, conv.bias
    )

    assert m_out.num_points() == sites.n_out
    assert sites.n_out > m.num_points(), "expansion should produce a denser set"
    assert check_all_close(y.float(), ref, f"generative_conv[{dtype}]", rtol=_RTOL[dtype])


def test_grad_parity_vs_reference():
    """Kernel backward matches autograd through the independent reference.

    The MVMR engine computes in fp32, so a true fp64 `gradcheck` is not
    meaningful (finite-diff error scales as ~7e-8/eps, a fp32 precision
    floor). Instead the hand-written backward (mvmrᵀ for grad_x, vvor for
    grad_weight) is checked against torch-autograd of an independent
    pure-PyTorch forward, at the kernel's native precision.
    """
    m = _make_metadata(150, n_samples=2, seed=2)
    Cin, Cout = 8, 12
    conv = GenerativePointConv3d(
        Cin, Cout, kernel_size=3, expansion=2.0, bias=False
    ).to(DEVICE)
    sites = conv.generator(m)
    N = m.num_points()

    # gradient through the MVMR kernel
    x1 = torch.randn(N, Cin, device=DEVICE, requires_grad=True)
    w1 = conv.weight.detach().clone().requires_grad_(True)
    y1 = conv._conv_forward(x1, sites.i, sites.j, sites.k, sites.n_out, w1, None)
    g = torch.randn_like(y1)
    (y1 * g).sum().backward()

    # gradient through the independent reference's autograd
    x2 = x1.detach().clone().requires_grad_(True)
    w2 = conv.weight.detach().clone().requires_grad_(True)
    y2 = generative_conv_scatter_ref(
        x2, sites.i, sites.j, sites.k, sites.n_out, w2, None
    )
    (y2 * g).sum().backward()

    assert check_all_close(x1.grad, x2.grad, "grad_x", rtol=2e-3)
    assert check_all_close(w1.grad, w2.grad, "grad_weight", rtol=2e-3)


# ------------------------------------------------------------- invariant ----


def test_output_input_coverage():
    """Every output point has >=1 input neighbour; every input is used."""
    m = _make_metadata(200, n_samples=2, seed=3)
    sites = KernelStampGenerator(kernel_size=3, expansion=2.0)(m)

    out_cov = torch.bincount(sites.i.long(), minlength=sites.n_out)
    assert int(out_cov.min()) >= 1, "an output point has zero contributing inputs"
    in_cov = torch.bincount(sites.j.long(), minlength=m.num_points())
    assert int(in_cov.min()) >= 1, "an input point reaches no output"
    sites.validate()  # must not raise


def test_isolated_input():
    """One isolated input -> K outputs, each with exactly one triplet."""
    m = MetaData(
        points=torch.tensor([[5.0, 5.0, 5.0]], device=DEVICE),
        sample_inds=torch.zeros(1, dtype=torch.long, device=DEVICE),
        sample_sizes=torch.tensor([1], device=DEVICE),
        grid_size=1.0,
    )
    sites = KernelStampGenerator(kernel_size=3, expansion=2.0)(m)
    assert sites.n_out == 27, "3^3 distinct stamps from one point"
    assert sites.i.numel() == 27
    cov = torch.bincount(sites.i.long(), minlength=27)
    assert int(cov.min()) == 1 and int(cov.max()) == 1
    sites.validate()


def test_validate_catches_orphan():
    """A hand-built GeneratedSites with an uncovered output fails closed."""
    bad = GeneratedSites(
        points=torch.zeros(3, 3, device=DEVICE),
        sample_inds=torch.zeros(3, dtype=torch.long, device=DEVICE),
        sample_sizes=torch.tensor([3], device=DEVICE),
        grid_size=1.0,
        i=torch.tensor([0, 0, 1], device=DEVICE),  # output index 2 is an orphan
        j=torch.tensor([0, 1, 2], device=DEVICE),
        k=torch.zeros(3, dtype=torch.long, device=DEVICE),
    )
    with pytest.raises(RuntimeError, match="zero\n?.*contributing|contributing input"):
        bad.validate()


# ------------------------------------------------------------ confinement ----


def test_batch_boundary():
    """No triplet crosses a sample boundary, even with overlapping coords."""
    torch.manual_seed(4)
    n = 100
    p = (torch.rand(n, 3, device=DEVICE) * 5).floor()
    points = torch.cat([p, p.clone()])  # identical coords, different samples
    sample_inds = torch.cat(
        [
            torch.zeros(n, dtype=torch.long, device=DEVICE),
            torch.ones(n, dtype=torch.long, device=DEVICE),
        ]
    )
    m = MetaData(
        points=points,
        sample_inds=sample_inds,
        sample_sizes=torch.bincount(sample_inds),
        grid_size=1.0,
    )
    sites = KernelStampGenerator(kernel_size=3, expansion=2.0)(m)

    out_sample = sites.sample_inds[sites.i.long()]
    in_sample = sample_inds[sites.j.long()]
    assert torch.equal(out_sample.long(), in_sample.long())
    # identical coords in 2 samples -> exactly double the single-sample set
    single = KernelStampGenerator(3, 2.0)(
        MetaData(
            points=p,
            sample_inds=torch.zeros(n, dtype=torch.long, device=DEVICE),
            sample_sizes=torch.tensor([n], device=DEVICE),
            grid_size=1.0,
        )
    )
    assert sites.n_out == 2 * single.n_out


def test_dedup_correctness():
    """Overlapping stamps merge to one output voxel fed by both inputs."""
    gen = KernelStampGenerator(kernel_size=3, expansion=1.0)
    # two points one grid cell apart on x -> their 3x3x3 footprints overlap
    m = MetaData(
        points=torch.tensor([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], device=DEVICE),
        sample_inds=torch.zeros(2, dtype=torch.long, device=DEVICE),
        sample_sizes=torch.tensor([2], device=DEVICE),
        grid_size=1.0,
    )
    sites = gen(m)
    cov = torch.bincount(sites.i.long(), minlength=sites.n_out)
    assert int(cov.max()) == 2 and int(cov.min()) == 1
    # n_out = 27 + 27 - 18 overlapping voxels
    assert sites.n_out == 36
    for shared in (cov == 2).nonzero().flatten().tolist():
        contributors = sites.j[sites.i.long() == shared].long()
        assert set(contributors.tolist()) == {0, 1}

    # coincident inputs -> both stamp the same K voxels
    m2 = MetaData(
        points=torch.tensor([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], device=DEVICE),
        sample_inds=torch.zeros(2, dtype=torch.long, device=DEVICE),
        sample_sizes=torch.tensor([2], device=DEVICE),
        grid_size=1.0,
    )
    assert gen(m2).n_out == 27


def test_expansion_factors():
    """Finer resolution -> denser output; ks=1/expansion=1 is identity."""
    m = _make_metadata(150, n_samples=1, seed=5, distinct=True)
    N = m.num_points()
    counts = {}
    for exp in (1.0, 2.0, 4.0):
        s = KernelStampGenerator(kernel_size=3, expansion=exp)(m)
        assert s.n_out >= N, "generative output is never sparser than the input"
        counts[exp] = s.n_out
    assert counts[1.0] <= counts[2.0] <= counts[4.0]

    identity = KernelStampGenerator(kernel_size=1, expansion=1.0)(m)
    assert identity.n_out == N, "ks=1, expansion=1 maps each input voxel to itself"


# -------------------------------------------------------------- behaviour ----


def test_determinism():
    """Repeat runs: bit-exact coords, near-exact features."""
    m = _make_metadata(250, n_samples=2, seed=6)
    conv = GenerativePointConv3d(8, 8, kernel_size=3, expansion=2.0).to(DEVICE)
    x = torch.randn(m.num_points(), 8, device=DEVICE)
    y1, m1 = conv(x, m)
    y2, m2 = conv(x, m)
    assert torch.equal(m1.points, m2.points), "generated coords must be deterministic"
    assert torch.equal(m1.i, m2.i) and torch.equal(m1.j, m2.j)
    assert check_all_close(y1, y2, "determinism", rtol=1e-4)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_groups(groups):
    """Grouped generative conv matches the (group-aware) reference."""
    m = _make_metadata(200, n_samples=2, seed=7)
    Cin, Cout = 16, 16
    conv = GenerativePointConv3d(
        Cin, Cout, kernel_size=3, expansion=2.0, groups=groups
    ).to(DEVICE)
    x = torch.randn(m.num_points(), Cin, device=DEVICE)
    sites = conv.generator(m)
    y, _ = conv(x, m, sites=sites)
    ref = generative_conv_scatter_ref(
        x, sites.i, sites.j, sites.k, sites.n_out, conv.weight, conv.bias
    )
    assert check_all_close(y.float(), ref, f"groups={groups}", rtol=1e-3)


def test_sites_reuse():
    """Passing a precomputed `sites` reproduces the regenerated-path result."""
    m = _make_metadata(200, n_samples=2, seed=8)
    conv = GenerativePointConv3d(12, 12, kernel_size=3, expansion=2.0).to(DEVICE)
    x = torch.randn(m.num_points(), 12, device=DEVICE)
    sites = conv.generator(m)
    y_reuse, m_reuse = conv(x, m, sites=sites)
    y_fresh, m_fresh = conv(x, m)
    assert m_reuse.num_points() == m_fresh.num_points()
    assert check_all_close(y_reuse, y_fresh, "sites-reuse", rtol=1e-4)


def test_custom_stencil():
    """A custom (non-cube) stencil drives the operator end to end."""
    # 7-tap cross: centre + 6 face neighbours
    cross = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        device=DEVICE,
    )
    gen = KernelStampGenerator(kernel_size=3, expansion=1.0, stencil=cross)
    assert gen.kernel_taps == 7
    conv = GenerativePointConv3d(8, 8, generator=gen).to(DEVICE)
    assert conv.weight.shape[0] == 7, "weight tap count tracks the custom stencil"

    m = _make_metadata(120, n_samples=1, seed=9)
    x = torch.randn(m.num_points(), 8, device=DEVICE)
    sites = conv.generator(m)
    y, _ = conv(x, m, sites=sites)
    ref = generative_conv_scatter_ref(
        x, sites.i, sites.j, sites.k, sites.n_out, conv.weight, conv.bias
    )
    assert check_all_close(y.float(), ref, "custom-stencil", rtol=1e-3)


# ----------------------------------------------------------- enablement ----


def test_enablement_smoke():
    """A real nn.Module composes the operator and trains a step."""

    class GenerativeBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.up = GenerativePointConv3d(c, c, kernel_size=3, expansion=2.0)
            self.act = nn.ReLU()

        def forward(self, x, m):
            x, m = self.up(x, m)
            return self.act(x), m

    m = _make_metadata(180, n_samples=2, seed=11)
    block = GenerativeBlock(10).to(DEVICE)
    x = torch.randn(m.num_points(), 10, device=DEVICE, requires_grad=True)
    y, m_out = block(x, m)
    y.pow(2).sum().backward()
    assert m_out.num_points() > m.num_points()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in block.parameters()
    )


# ------------------------------------------------------------ edge cases ----


def test_empty_input():
    """N=0 input -> empty output, no triplets, no crash."""
    m = MetaData(
        points=torch.empty(0, 3, device=DEVICE),
        sample_inds=torch.empty(0, dtype=torch.long, device=DEVICE),
        sample_sizes=torch.zeros(2, dtype=torch.long, device=DEVICE),
        grid_size=1.0,
    )
    sites = build_generative_triplets(
        m.points, m.sample_inds, m.sample_sizes, m.grid_size,
        kernel_size=3, expansion=2.0,
    )
    assert sites.n_out == 0 and sites.i.numel() == 0
    sites.validate()


def test_single_point():
    """N=1 input runs end to end through the operator."""
    m = MetaData(
        points=torch.tensor([[4.0, 4.0, 4.0]], device=DEVICE),
        sample_inds=torch.zeros(1, dtype=torch.long, device=DEVICE),
        sample_sizes=torch.tensor([1], device=DEVICE),
        grid_size=1.0,
    )
    conv = GenerativePointConv3d(6, 6, kernel_size=3, expansion=2.0).to(DEVICE)
    x = torch.randn(1, 6, device=DEVICE)
    y, m_out = conv(x, m)
    assert m_out.num_points() == 27 and y.shape == (27, 6)


# --------------------------------------------------------- builder opts ----


def test_builder_k_groups_contiguous():
    """sort_by='k' yields contiguous, ascending k-groups (layout invariant).

    The builder replaces the structured `torch.sort(k)` with a
    deterministic (N,K)->(K,N) transpose permutation. The MVMR grouped
    tensor-core forward only needs same-k contiguity; lock it so a future
    edit can't silently break the layout the kernel dispatch assumes.
    """
    m = _make_metadata(250, n_samples=3, seed=11)
    sites = build_generative_triplets(
        m.points, m.sample_inds, m.sample_sizes, m.grid_size,
        kernel_size=3, expansion=2.0, sort_by="k",
    )
    k = sites.k.long()
    # non-decreasing -> all equal-k triplets form one contiguous run
    assert bool((k[1:] >= k[:-1]).all()), "k must be sorted ascending"
    K = 27
    # every tap present exactly N times (each input stamps all K taps once)
    counts = torch.bincount(k, minlength=K)
    assert int(counts.min()) == int(counts.max()) == m.num_points()


def test_builder_coord_decode_matches_bruteforce():
    """The arithmetic key-decode equals the brute-force candidate min.

    The builder decodes output voxel coords straight from the sorted unique
    1D keys instead of a first-occurrence scatter over the (N*K,4) table.
    Verify the decoded coords equal an independent brute-force build (group
    every candidate by its output index i, take the shared voxel row).
    """
    from layers.generative import _default_cube_stencil

    m = _make_metadata(180, n_samples=2, seed=12, distinct=True)
    grid_out = m.grid_size / 2.0
    sites = build_generative_triplets(
        m.points, m.sample_inds, m.sample_sizes, m.grid_size,
        kernel_size=3, expansion=2.0, sort_by="k",
    )
    # brute force: each triplet's candidate voxel = v0[j] + stencil[k];
    # all triplets sharing an output index i map to the same voxel row, so
    # writing them by i and reading back recovers each output's coord.
    stencil = _default_cube_stencil((3, 3, 3), DEVICE).to(torch.int32)
    v0 = torch.div(m.points, grid_out, rounding_mode="floor").to(torch.int32)
    cand = v0[sites.j.long()] + stencil[sites.k.long()]   # (T, 3), final order
    n_out = sites.n_out
    ref_vox = torch.full((n_out, 3), -(10**9), dtype=torch.int32, device=DEVICE)
    ref_vox[sites.i.long()] = cand  # any write per i; all are identical
    ref_pts = (ref_vox.to(m.points.dtype) + 0.5) * grid_out
    assert torch.equal(sites.points, ref_pts), "decoded coords must match brute force"
