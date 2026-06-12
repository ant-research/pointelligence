"""Parity gate: TIG at GENERATIVE shapes (N_in != N_out).

Two production-shaped index families, triplet math mirroring the production
builders (k-sorted):

- partition/stem: K=8^3=512 slots, fan-out 1 per input, fan-in = cell
  occupancy, N_out << N_in.
- fan-in-1 deconv: K=2^3=8, exactly one triplet per output, N_out > N_in.

Reference: autograd through the eager op under ``force_pt`` (per-triplet —
no shared code with TIG's kernels). fp32 parity at ``input_precision=
"ieee"`` must be tight (<=1e-5 rel); fp16 within the half-precision band
(<=5e-3 rel vs the fp32 reference).

Submanifold regression: an index built WITHOUT ``n_in`` must behave
byte-identically to the submanifold-era contract (N_in defaults to N) — covered by the existing
TIG suites; here we assert the default wiring directly.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sparse_engines.tig import TigIndex, tig_mvmr  # noqa: E402
from sparse_engines.ops import (  # noqa: E402
    sparse_matrix_vector_multiplication_reduction as eager_mvmr,
)
from sparse_engines._dispatch_override import dispatch_mode  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

device = "cuda"


def _partition_triplets(n_in: int, kk: int, n_cells: int, seed: int = 0):
    """Stem-like: every input point -> one (cell, slot); k-sorted."""
    g = torch.Generator(device=device).manual_seed(seed)
    i = torch.randint(0, n_cells, (n_in,), device=device, generator=g)
    k = torch.randint(0, kk, (n_in,), device=device, generator=g)
    j = torch.arange(n_in, device=device)
    order = torch.argsort(k)
    # ensure every output cell is hit at least once (dense output rows)
    i[order[:n_cells]] = torch.arange(n_cells, device=device)
    return i[order], j[order], k[order], n_cells, n_in


def _fanin1_triplets(n_out: int, kk: int, n_in: int, seed: int = 0):
    """Deconv-like: exactly one (parent, octant) per output; k-sorted."""
    g = torch.Generator(device=device).manual_seed(seed)
    i = torch.arange(n_out, device=device)
    j = torch.randint(0, n_in, (n_out,), device=device, generator=g)
    k = torch.randint(0, kk, (n_out,), device=device, generator=g)
    order = torch.argsort(k)
    return i[order], j[order], k[order], n_out, n_in


def _run_pair(i, j, k, n_out, n_in, kk, c, m, dtype, precision):
    """(out, grad_w, grad_x) for TIG-generative and the eager force_pt ref."""
    torch.manual_seed(0)
    w0 = torch.randn(kk, c, m, device=device) * (1.0 / math.sqrt(c))
    x0 = torch.randn(n_in, c, device=device)

    # reference: eager per-triplet at fp32 (the independent engine)
    w_ref = w0.clone().requires_grad_(True)
    x_ref = x0.clone().requires_grad_(True)
    # backward stays INSIDE the context: the reference grads must also run
    # per-triplet (scalar-FMA fp32, exact) — outside, auto routes the grad
    # legs to the grouped tl.dot path whose fp32 default is tf32 (~4e-4).
    with dispatch_mode("force_pt"):
        out_ref = eager_mvmr(w_ref.view(kk, 1, c, m), k, x_ref.view(n_in, 1, c),
                             j, i, n_out).view(n_out, m)
        out_ref.sum().backward()

    w_t = w0.clone().to(dtype).requires_grad_(True)
    x_t = x0.clone().to(dtype).requires_grad_(True)
    idx = TigIndex(i, j, k, n_out, num_kernel_offsets=kk,
                   build_hybrid=False, assume_sorted=True, n_in=n_in)
    assert idx.N == n_out and idx.N_in == n_in
    out_t = tig_mvmr(w_t, x_t, idx, mode="flat", input_precision=precision)
    out_t.sum().backward()

    return (out_ref, w_ref.grad, x_ref.grad), (out_t, w_t.grad, x_t.grad)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


CASES = [
    # (family, n_out, n_in, K, C, M) — production-network shapes, scaled
    ("partition_stem", 1900, 80000, 512, 7, 256),
    ("partition_stem", 1900, 80000, 512, 2, 256),
    ("fanin1_deconv", 31000, 7800, 8, 256, 128),
    ("fanin1_deconv", 31000, 7800, 8, 64, 64),
]


@pytest.mark.parametrize("family,n_out,n_in,kk,c,m", CASES)
@pytest.mark.parametrize("dtype,precision,tol", [
    (torch.float32, "ieee", 1e-5),
    (torch.float16, "tf32", 5e-3),
])
def test_generative_parity(subtests, family, n_out, n_in, kk, c, m,
                           dtype, precision, tol):
    if family == "partition_stem":
        i, j, k, n_out_, n_in_ = _partition_triplets(n_in, kk, n_out)
    else:
        i, j, k, n_out_, n_in_ = _fanin1_triplets(n_out, kk, n_in)
    ref, tig = _run_pair(i, j, k, n_out_, n_in_, kk, c, m, dtype, precision)

    # shape contract: grad_x must be input-sided (THE generative fix)
    assert tig[2].shape == (n_in_, c)
    assert tig[0].shape == (n_out_, m)

    for name, r, t in zip(("out", "grad_w", "grad_x"), ref, tig):
        with subtests.test(leg=name):
            d = _rel(t, r)
            assert d <= tol, f"{family} {name} rel {d:.3e} > {tol}"
            # anti-degeneracy: a real signal on both sides
            assert r.float().norm() > 0 and t.float().norm() > 0


@pytest.mark.parametrize("dtype,precision,tol", [
    (torch.float32, "ieee", 1e-5),
    (torch.float16, "tf32", 5e-3),
])
def test_fi1_and_dense_gemm_paths(subtests, dtype, precision, tol):
    """(a) FI1 plain-store forward (deconv, exact_cover_out),
    (b) FI1 grad_input (partition fan-out-1, exact_cover_in), (c) the
    partition dense-GEMM path — all vs the force_pt eager reference.
    The builders here genuinely satisfy the exactly-once contracts."""
    from sparse_engines.partition_gemm import partition_dense_mvmr

    # (a)+(b): fan-in-1 deconv with FI1 fwd; partition with FI1 grad_input
    for family, flags, (n_out, n_in, kk, c, m) in [
        ("fanin1_deconv", dict(exact_cover_out=True), (31000, 7800, 8, 64, 64)),
        ("partition_stem", dict(exact_cover_in=True), (1900, 80000, 512, 7, 256)),
    ]:
        if family == "partition_stem":
            i, j, k, n_o, n_i = _partition_triplets(n_in, kk, n_out)
        else:
            i, j, k, n_o, n_i = _fanin1_triplets(n_out, kk, n_in)
        torch.manual_seed(0)
        w0 = torch.randn(kk, c, m, device=device) / math.sqrt(c)
        x0 = torch.randn(n_i, c, device=device)
        w_ref = w0.clone().requires_grad_(True)
        x_ref = x0.clone().requires_grad_(True)
        with dispatch_mode("force_pt"):
            out_ref = eager_mvmr(w_ref.view(kk, 1, c, m), k,
                                 x_ref.view(n_i, 1, c), j, i, n_o
                                 ).view(n_o, m)
            out_ref.sum().backward()
        w_t = w0.clone().to(dtype).requires_grad_(True)
        x_t = x0.clone().to(dtype).requires_grad_(True)
        idx = TigIndex(i, j, k, n_o, num_kernel_offsets=kk,
                       build_hybrid=False, assume_sorted=True, n_in=n_i,
                       **flags)
        out_t = tig_mvmr(w_t, x_t, idx, mode="flat",
                         input_precision=precision)
        out_t.sum().backward()
        for name, r, t in zip(("out", "grad_w", "grad_x"),
                              (out_ref, w_ref.grad, x_ref.grad),
                              (out_t, w_t.grad, x_t.grad)):
            with subtests.test(path=f"fi1:{family}", leg=name):
                d = _rel(t, r)
                assert d <= tol, f"fi1 {family} {name} rel {d:.3e} > {tol}"
                assert r.float().norm() > 0 and t.float().norm() > 0

    # (c): the partition dense-GEMM path (exact for duplicates too)
    i, j, k, n_o, n_i = _partition_triplets(80000, 512, 1900)
    torch.manual_seed(0)
    w0 = torch.randn(512, 7, 256, device=device) / math.sqrt(7.0)
    x0 = torch.randn(n_i, 7, device=device)
    w_ref = w0.clone().requires_grad_(True)
    x_ref = x0.clone().requires_grad_(True)
    with dispatch_mode("force_pt"):
        out_ref = eager_mvmr(w_ref.view(512, 1, 7, 256), k,
                             x_ref.view(n_i, 1, 7), j, i, n_o).view(n_o, 256)
        out_ref.sum().backward()
    w_d = w0.clone().to(dtype).requires_grad_(True)
    x_d = x0.clone().to(dtype).requires_grad_(True)
    out_d = partition_dense_mvmr(w_d, x_d, i, j, k, n_o)
    out_d.sum().backward()
    for name, r, t in zip(("out", "grad_w", "grad_x"),
                          (out_ref, w_ref.grad, x_ref.grad),
                          (out_d, w_d.grad, x_d.grad)):
        with subtests.test(path="dense_gemm", leg=name):
            d = _rel(t, r)
            assert d <= tol, f"dense {name} rel {d:.3e} > {tol}"
            assert r.float().norm() > 0 and t.float().norm() > 0


def test_generation_from_one_autodetect():
    """The subdivision generator (ks=2, expansion=2 on
    voxel-centered inputs) stamps injectively -> exact_cover_out detected
    (n_out == N*K, host-side) and the op routes the FI1 plain-store
    forward under auto; a colliding dense stamp (ks=3 on a dense cloud)
    must NOT set the flag. Parity vs force_pt on the subdivision case."""
    import sparse_engines.tig as _tig
    from layers.conv import GenerativePointConv3d
    from layers.generative import SubdivisionGenerator
    from layers.metadata import MetaData
    from sparse_engines._dispatch_override import dispatch_mode

    torch.manual_seed(0)
    g = 0.04
    # voxel-centered coarse inputs (unique voxels)
    vox = torch.unique(torch.randint(0, 40, (5000, 3), device=device), dim=0)
    pts = (vox.float() + 0.5) * g
    n = pts.shape[0]
    si = torch.zeros(n, dtype=torch.long, device=device)
    m = MetaData(points=pts, sample_inds=si, sample_sizes=torch.bincount(si),
                 grid_size=g, kernel_size=None, auto_build_triplets=False)

    conv = GenerativePointConv3d(16, 32, generator=SubdivisionGenerator(2),
                                 bias=False, device=device,
                                 dtype=torch.float16)
    sites = conv.generator(m)
    assert sites.exact_cover_out, "subdivision must declare fan-in-1"
    assert sites.uniform_seg_len == n
    assert sites.n_out == n * 8
    sites.validate()
    # children tile the fine grid: all generated voxels distinct
    cv = torch.div(sites.points, sites.grid_size, rounding_mode="floor").long()
    keys = (cv[:, 0] * 2_000_003 + cv[:, 1]) * 2_000_003 + cv[:, 2]
    assert torch.unique(keys).numel() == sites.n_out, "children must be distinct"

    x = torch.randn(n, 16, device=device, dtype=torch.float16)
    fi1_spy = [0]
    real = _tig._tig_flat_fi1_kernel

    class _Spy:
        def __getitem__(self, grid):
            fi1_spy[0] += 1
            return real[grid]

    _tig._tig_flat_fi1_kernel = _Spy()
    try:
        with dispatch_mode("auto"):
            out, _ = conv(x, m, sites=sites)
    finally:
        _tig._tig_flat_fi1_kernel = real
    # FI1 routing is ARCH-GATED (wins on sm_8x, keeps the regular path on
    # Hopper): assert the route matches the gate, not unconditionally.
    expected = 1 if _tig._fi1_wins_here() else 0
    assert fi1_spy[0] == expected, (
        f"auto FI1 routing must follow the arch gate "
        f"(expected {expected}, got {fi1_spy[0]})")

    with dispatch_mode("force_pt"):
        out_ref, _ = conv(x, m, sites=sites)
    d = _rel(out, out_ref)
    assert d <= 5e-3, f"subdivision FI1 parity {d:.3e}"

    # colliding case: dense cloud, ks=3 -> merges happen -> flag False
    pts2 = torch.rand(4000, 3, device=device) * 0.6
    si2 = torch.zeros(4000, dtype=torch.long, device=device)
    m2 = MetaData(points=pts2, sample_inds=si2,
                  sample_sizes=torch.bincount(si2), grid_size=0.02,
                  kernel_size=None, auto_build_triplets=False)
    conv3 = GenerativePointConv3d(8, 8, kernel_size=3, expansion=2.0,
                                  bias=False, device=device,
                                  dtype=torch.float16)
    sites2 = conv3.generator(m2)
    assert not sites2.exact_cover_out, "colliding stamp must NOT set FI1"


def test_pt_fallback_advisory():
    """'auto' landing on the per-triplet engine must warn ONCE
    (optimization-opportunity advisory); the sorted production path must
    stay silent."""
    import warnings as _w
    from layers.conv import PointConv3d
    from sparse_engines._dispatch_override import dispatch_mode

    torch.manual_seed(0)
    n, t = 256, 1500
    conv = PointConv3d(16, 16, kernel_size=3, bias=False, device=device,
                       dtype=torch.float16)
    x = torch.randn(n, 16, device=device, dtype=torch.float16)
    g = torch.Generator(device="cpu").manual_seed(7)
    k_uns = torch.randint(0, 27, (t,), generator=g).to(device)  # unsorted
    i_idx = torch.randint(0, n, (t,), generator=g).to(device)
    j_idx = torch.randint(0, n, (t,), generator=g).to(device)

    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        with dispatch_mode("auto"):
            conv(x, i_idx, j_idx, k_uns, n)
    msgs = [str(r.message) for r in rec if r.category is RuntimeWarning]
    assert any("per-triplet fallback" in s for s in msgs), msgs

    # sorted path: silent
    k_s, order = torch.sort(k_uns)
    with _w.catch_warnings(record=True) as rec2:
        _w.simplefilter("always")
        with dispatch_mode("auto"):
            conv(x, i_idx[order], j_idx[order], k_s, n)
    assert not any("per-triplet fallback" in str(r.message) for r in rec2)


def test_submanifold_default_unchanged():
    """n_in omitted -> N_in == N; grad_x is output-sized (submanifold contract)."""
    n, kk, c, m = 4096, 27, 32, 32
    g = torch.Generator(device=device).manual_seed(1)
    t = 3 * n
    i = torch.randint(0, n, (t,), device=device, generator=g)
    j = torch.randint(0, n, (t,), device=device, generator=g)
    k = torch.randint(0, kk, (t,), device=device, generator=g)
    k, order = torch.sort(k)
    idx = TigIndex(i[order], j[order], k, n, num_kernel_offsets=kk,
                   build_hybrid=False, assume_sorted=True)
    assert idx.N_in == idx.N == n
    w = torch.randn(kk, c, m, device=device, dtype=torch.float16,
                    requires_grad=True)
    x = torch.randn(n, c, device=device, dtype=torch.float16,
                    requires_grad=True)
    out = tig_mvmr(w, x, idx, mode="flat")
    out.sum().backward()
    assert x.grad.shape == (n, c) and out.shape == (n, m)
