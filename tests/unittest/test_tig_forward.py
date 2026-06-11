"""Parity battery for the TIG forward engine.

Covers: exact pyref parity (fp64 oracle) on handcrafted edge cases,
flat-vs-hybrid mode agreement, dtype battery (fp16/bf16/fp32),
non-multiple channel shapes, heavy fan-in, empty taps / rows / rulebook,
index reuse, and production-path cross-check at a realistic shape.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sparse_engines.tig import TigIndex, tig_forward  # noqa: E402

device = "cuda"
K = 27

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="needs CUDA")


def pyref(weight, feat, i, j, k, n_out):
    """fp64 oracle: out[i] += feat[j] @ W[k], per triplet."""
    w = weight.double()
    if w.dim() == 4:
        w = w.select(1, 0)
    f = feat.double()
    out = torch.zeros(n_out, w.size(2), device=feat.device,
                      dtype=torch.float64)
    if i.numel():
        prod = torch.bmm(f[j].unsqueeze(1), w[k]).squeeze(1)
        out.index_add_(0, i.long(), prod)
    return out


def rel_fro(a, b):
    d = b.float().norm().clamp_min(1e-12)
    return ((a.float() - b.float()).norm() / d).item()


def rand_problem(N, C, M, T, seed=0, kmax=K):
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, N, (T,), generator=g).to(device)
    j = torch.randint(0, N, (T,), generator=g).to(device)
    k = torch.randint(0, kmax, (T,), generator=g).to(device)
    feat = torch.randn(N, C, device=device)
    weight = torch.randn(K, 1, C, M, device=device) * ((K * C) ** -0.5)
    return weight, feat, i, j, k


@requires_cuda
class TestTigPyrefParity:
    @pytest.mark.parametrize("mode", ["flat", "hybrid"])
    @pytest.mark.parametrize("N,C,M,T", [
        (8, 32, 32, 64),          # tiny
        (300, 32, 64, 4000),      # rectangular C != M
        (1000, 64, 32, 20000),    # heavy fan-in (T/N=20 over 27 taps)
    ])
    def test_fp32_ieee_matches_pyref(self, mode, N, C, M, T):
        weight, feat, i, j, k = rand_problem(N, C, M, T)
        idx = TigIndex(i, j, k, N)
        out = tig_forward(weight, feat, idx, mode=mode,
                           input_precision="ieee")
        ref = pyref(weight, feat, i, j, k, N)
        assert rel_fro(out, ref) < 1e-5

    @pytest.mark.parametrize("mode", ["flat", "hybrid"])
    def test_nonmultiple_channels(self, mode):
        weight, feat, i, j, k = rand_problem(257, 48, 80, 3000, seed=3)
        idx = TigIndex(i, j, k, 257)
        out = tig_forward(weight, feat, idx, mode=mode,
                           input_precision="ieee")
        ref = pyref(weight, feat, i, j, k, 257)
        assert rel_fro(out, ref) < 1e-5

    def test_zero_neighbor_rows_are_zero(self):
        N = 100
        weight, feat, i, j, k = rand_problem(N, 32, 32, 500, seed=1)
        i = i.clamp(max=49)  # rows 50.. get no triplets
        idx = TigIndex(i, j, k, N)
        for mode in ("flat", "hybrid"):
            out = tig_forward(weight, feat, idx, mode=mode,
                               input_precision="ieee")
            assert out[50:].abs().max().item() == 0.0
            assert rel_fro(out, pyref(weight, feat, i, j, k, N)) < 1e-5

    def test_few_taps_and_single_point(self):
        weight, feat, i, j, k = rand_problem(64, 32, 32, 300, seed=2, kmax=3)
        idx = TigIndex(i, j, k, 64)
        ref = pyref(weight, feat, i, j, k, 64)
        for mode in ("flat", "hybrid"):
            assert rel_fro(tig_forward(weight, feat, idx, mode=mode,
                                        input_precision="ieee"), ref) < 1e-5
        # single point, single triplet
        i1 = torch.zeros(1, dtype=torch.long, device=device)
        idx1 = TigIndex(i1, i1, i1, 1)
        out1 = tig_forward(weight, feat[:1], idx1, mode="flat",
                            input_precision="ieee")
        assert rel_fro(out1, pyref(weight, feat[:1], i1, i1, i1, 1)) < 1e-5

    def test_empty_rulebook(self):
        weight = torch.randn(K, 1, 32, 32, device=device)
        feat = torch.randn(10, 32, device=device)
        e = torch.empty(0, dtype=torch.long, device=device)
        idx = TigIndex(e, e, e, 10)
        out = tig_forward(weight, feat, idx, mode="flat",
                           input_precision="ieee")
        assert out.shape == (10, 32) and out.abs().max().item() == 0.0


@requires_cuda
class TestTigModesAndDtypes:
    def test_modes_agree_fp32_ieee(self):
        weight, feat, i, j, k = rand_problem(2000, 64, 64, 30000, seed=4)
        idx = TigIndex(i, j, k, 2000)
        a = tig_forward(weight, feat, idx, mode="flat",
                         input_precision="ieee")
        b = tig_forward(weight, feat, idx, mode="hybrid",
                         input_precision="ieee")
        assert rel_fro(a, b) < 1e-6

    @pytest.mark.parametrize("dtype,tol", [
        (torch.float16, 2e-3), (torch.bfloat16, 2e-2), (torch.float32, 2e-3)])
    @pytest.mark.parametrize("mode", ["flat", "hybrid"])
    def test_dtype_band_vs_pyref(self, dtype, tol, mode):
        weight, feat, i, j, k = rand_problem(1500, 64, 64, 20000, seed=5)
        idx = TigIndex(i, j, k, 1500)
        out = tig_forward(weight.to(dtype), feat.to(dtype), idx, mode=mode)
        assert out.dtype == dtype
        assert rel_fro(out, pyref(weight, feat, i, j, k, 1500)) < tol

    def test_index_reuse_two_feats(self):
        weight, feat, i, j, k = rand_problem(500, 32, 32, 5000, seed=6)
        idx = TigIndex(i, j, k, 500)
        feat2 = torch.randn_like(feat)
        for f in (feat, feat2):
            ref = pyref(weight, f, i, j, k, 500)
            assert rel_fro(tig_forward(weight, f, idx, mode="hybrid",
                                        input_precision="ieee"), ref) < 1e-5

    def test_determinism_band_fp16(self):
        # fp32 atomic accumulation: order nondeterminism stays in fp32
        # rounding noise, far below fp16 representation steps.
        weight, feat, i, j, k = rand_problem(1000, 64, 64, 50000, seed=7)
        idx = TigIndex(i, j, k, 1000)
        outs = [tig_forward(weight.half(), feat.half(), idx, mode="flat")
                for _ in range(3)]
        assert rel_fro(outs[0], outs[1]) < 1e-4
        assert rel_fro(outs[0], outs[2]) < 1e-4


def pyref_groups(weight, feat, i, j, k, n_out):
    """fp64 block-diagonal oracle: per group g, out[:, g] += feat[j][:, g] @ W[k, g]."""
    w = weight.double()                       # (K, G, Cg, Mg)
    K_, G, Cg, Mg = w.shape
    f = feat.double().view(-1, G, Cg)
    out = torch.zeros(n_out, G, Mg, device=feat.device, dtype=torch.float64)
    if i.numel():
        for g in range(G):
            prod = torch.bmm(f[j][:, g].unsqueeze(1), w[k][:, g]).squeeze(1)
            out.select(1, g).index_add_(0, i.long(), prod)
    return out.view(n_out, G * Mg)


@requires_cuda
class TestTigGroupsHybrid:
    """G>1 on the HYBRID (masked level-0 + flat residual) path — the
    group axis added in v1.2.0. Cells keep per-group M <= 64
    so auto-mode actually engages hybrid."""

    def make(self, G, Cg, Mg, N=900, T=12000, seed=None):
        gen = torch.Generator(device="cpu").manual_seed(
            seed if seed is not None else G * 1000 + Cg)
        i = torch.randint(0, N, (T,), generator=gen).to(device)
        j = torch.randint(0, N, (T,), generator=gen).to(device)
        k = torch.randint(0, K, (T,), generator=gen).to(device)
        feat = torch.randn(N, G * Cg, device=device)
        weight = torch.randn(K, G, Cg, Mg, device=device) * ((K * Cg) ** -0.5)
        return weight, feat, i, j, k, N

    @pytest.mark.parametrize("G,Cg,Mg", [(2, 32, 32), (4, 16, 32),
                                         (8, 32, 16), (4, 48, 40),
                                         (8, 8, 8)])  # 16-wide-tile cfg
    def test_hybrid_fp32_ieee_matches_oracle(self, G, Cg, Mg):
        weight, feat, i, j, k, N = self.make(G, Cg, Mg)
        idx = TigIndex(i, j, k, N)
        assert idx.has_hybrid
        out = tig_forward(weight, feat, idx, mode="hybrid",
                          input_precision="ieee")
        ref = pyref_groups(weight, feat, i, j, k, N)
        assert rel_fro(out, ref) < 1e-5

    @pytest.mark.parametrize("G", [2, 4, 8])
    def test_hybrid_agrees_with_flat(self, G):
        weight, feat, i, j, k, N = self.make(G, 32, 32, seed=G)
        idx = TigIndex(i, j, k, N)
        a = tig_forward(weight, feat, idx, mode="flat",
                        input_precision="ieee")
        b = tig_forward(weight, feat, idx, mode="hybrid",
                        input_precision="ieee")
        assert rel_fro(a, b) < 1e-6

    def test_auto_engages_hybrid_at_groups(self):
        # M <= 64 per group + has_hybrid -> auto must take the hybrid path
        # for G > 1 too (the G==1-only condition was lifted).
        weight, feat, i, j, k, N = self.make(4, 32, 32, seed=11)
        idx = TigIndex(i, j, k, N)
        out_auto = tig_forward(weight, feat, idx, mode="auto",
                               input_precision="ieee")
        ref = pyref_groups(weight, feat, i, j, k, N)
        assert rel_fro(out_auto, ref) < 1e-5

    def test_flat_packed_agrees_with_hybrid(self):
        # Cg==Mg==8, G=8 (G%4==0) -> flat routes the GP=4 packed kernel;
        # the hybrid path (masked + unpacked flat residual) is the
        # cross-check.
        weight, feat, i, j, k, N = self.make(8, 8, 8, seed=31)
        idx = TigIndex(i, j, k, N)
        a = tig_forward(weight, feat, idx, mode="flat",
                        input_precision="ieee")
        b = tig_forward(weight, feat, idx, mode="hybrid",
                        input_precision="ieee")
        assert rel_fro(a, b) < 1e-6
        assert rel_fro(a, pyref_groups(weight, feat, i, j, k, N)) < 1e-5

    @pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-3),
                                           (torch.bfloat16, 2e-2)])
    def test_hybrid_groups_dtype_bands(self, dtype, tol):
        weight, feat, i, j, k, N = self.make(4, 32, 32, N=1200, T=16000,
                                             seed=21)
        idx = TigIndex(i, j, k, N)
        out = tig_forward(weight.to(dtype), feat.to(dtype), idx,
                          mode="hybrid")
        ref = pyref_groups(weight, feat, i, j, k, N)
        assert rel_fro(out, ref) < tol


@requires_cuda
class TestTigVsProduction:
    def test_realistic_shape_vs_production_op(self):
        from functools import partial

        from layers import (build_triplets, radius_scaler_for_kernel_size,
                            voxelize_3d)
        from sparse_engines._dispatch_override import dispatch_mode
        from sparse_engines.mvmr_triton import (
            sparse_matrix_vector_multiplication_reduction as mvmr_op)

        torch.manual_seed(0)
        N, C, g = 30000, 128, 0.08
        coord = torch.rand(N, 3, device=device) * 5.0
        sample_inds = torch.zeros(N, device=device, dtype=torch.long)
        sample_sizes = torch.tensor([N], device=device)
        with torch.no_grad():
            i_t, j_t, k_t, _ = build_triplets(
                points=coord, sample_inds=sample_inds,
                sample_sizes=sample_sizes,
                neighbor_radius=g * radius_scaler_for_kernel_size(3),
                kernel_indexer=partial(voxelize_3d, kernel_size=3),
                radius_scaler=radius_scaler_for_kernel_size(3),
                return_num_neighbors=False)
        feat = torch.randn(N, C, device=device)
        weight = torch.randn(K, 1, C, C, device=device) * ((K * C) ** -0.5)
        with dispatch_mode("force_fsg"):
            ref32 = mvmr_op(weight, k_t, feat.unsqueeze(1), j_t, i_t,
                            N).select(1, 0)
            prod16 = mvmr_op(weight.half(), k_t, feat.half().unsqueeze(1),
                             j_t, i_t, N).select(1, 0)
        idx = TigIndex(i_t, j_t, k_t, N)
        prod_err = rel_fro(prod16, ref32)
        for mode in ("flat", "hybrid"):
            out16 = tig_forward(weight.half(), feat.half(), idx, mode=mode)
            err = rel_fro(out16, ref32)
            # TIG must be at least as close to the fp32 reference as the
            # production fp16 path itself (3x slack for noise).
            assert err < max(prod_err * 3, 2e-3), (mode, err, prod_err)
