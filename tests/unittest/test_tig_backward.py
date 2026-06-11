"""Backward parity battery for the TIG engine.

Oracle: analytic fp64 gradients of the mvmr op. Covers grad_input,
grad_weight, the autograd Function end-to-end (incl. needs_input_grad
gating), dtype bands, modes, edge cases, and the production-op
cross-check at a realistic shape.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sparse_engines.tig import (  # noqa: E402
    TigIndex, tig_backward_fused, tig_grad_input, tig_grad_weight,
    tig_mvmr)

device = "cuda"
K = 27

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="needs CUDA")


def rel_fro(a, b):
    return ((a.float() - b.float()).norm()
            / b.float().norm().clamp_min(1e-12)).item()


def rand_problem(N, C, M, T, seed=0, kmax=K):
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, N, (T,), generator=g).to(device)
    j = torch.randint(0, N, (T,), generator=g).to(device)
    k = torch.randint(0, kmax, (T,), generator=g).to(device)
    feat = torch.randn(N, C, device=device)
    weight = torch.randn(K, 1, C, M, device=device) * ((K * C) ** -0.5)
    go = torch.randn(N, M, device=device)
    return weight, feat, go, i, j, k


def ref_grads(weight, feat, go, i, j, k, n_out):
    """Analytic fp64: gx[j] += go[i] @ W[k]^T ; gW[k] += feat[j]^T ⊗ go[i]."""
    w = weight.double().select(1, 0)
    f, g = feat.double(), go.double()
    gx = torch.zeros_like(f)
    gw = torch.zeros_like(w)
    if i.numel():
        gx.index_add_(0, j.long(),
                      torch.bmm(g[i].unsqueeze(1),
                                w[k].transpose(1, 2)).squeeze(1))
        gw.index_add_(0, k.long(),
                      f[j].unsqueeze(2) * g[i].unsqueeze(1))
    return gx, gw.unsqueeze(1)


@requires_cuda
class TestTigBackwardPyref:
    @pytest.mark.parametrize("N,C,M,T", [
        (8, 32, 32, 64),
        (300, 32, 64, 4000),
        (1000, 64, 32, 20000),   # heavy fan-in
        (257, 48, 80, 3000),     # non-multiple channels
    ])
    def test_fp32_ieee_grads_match_ref(self, N, C, M, T):
        weight, feat, go, i, j, k = rand_problem(N, C, M, T, seed=N)
        idx = TigIndex(i, j, k, N)
        gx_ref, gw_ref = ref_grads(weight, feat, go, i, j, k, N)
        gx = tig_grad_input(weight, go, idx, input_precision="ieee")
        gw = tig_grad_weight(feat, go, idx, weight.shape,
                              input_precision="ieee")
        assert rel_fro(gx, gx_ref) < 1e-5
        assert rel_fro(gw, gw_ref) < 1e-5

    @pytest.mark.parametrize("N,C,M,T", [
        (300, 32, 64, 4000), (1000, 64, 32, 20000), (257, 48, 80, 3000)])
    def test_fused_backward_matches_split(self, N, C, M, T):
        weight, feat, go, i, j, k = rand_problem(N, C, M, T, seed=N + 1)
        idx = TigIndex(i, j, k, N)
        gx_ref, gw_ref = ref_grads(weight, feat, go, i, j, k, N)
        gw, gx = tig_backward_fused(weight, feat, go, idx, weight.shape,
                                     input_precision="ieee")
        assert rel_fro(gx, gx_ref) < 1e-5
        assert rel_fro(gw, gw_ref) < 1e-5

    def test_empty_rulebook_grads(self):
        weight = torch.randn(K, 1, 32, 32, device=device)
        feat = torch.randn(10, 32, device=device)
        go = torch.randn(10, 32, device=device)
        e = torch.empty(0, dtype=torch.long, device=device)
        idx = TigIndex(e, e, e, 10)
        assert tig_grad_input(weight, go, idx).abs().max().item() == 0.0
        assert tig_grad_weight(feat, go, idx,
                                weight.shape).abs().max().item() == 0.0


@requires_cuda
class TestTigAutogradFunction:
    @pytest.mark.parametrize("mode", ["flat", "hybrid"])
    def test_function_end_to_end_fp32_ieee(self, mode):
        weight, feat, go, i, j, k = rand_problem(500, 64, 64, 8000, seed=11)
        idx = TigIndex(i, j, k, 500)
        w = weight.clone().requires_grad_(True)
        f = feat.clone().requires_grad_(True)
        out = tig_mvmr(w, f, idx, mode=mode, input_precision="ieee")
        (out * go).sum().backward()
        gx_ref, gw_ref = ref_grads(weight, feat, go, i, j, k, 500)
        assert rel_fro(f.grad, gx_ref) < 1e-5
        assert rel_fro(w.grad, gw_ref) < 1e-5

    def test_needs_input_grad_gating(self):
        weight, feat, go, i, j, k = rand_problem(200, 32, 32, 2000, seed=12)
        idx = TigIndex(i, j, k, 200)
        f = feat.clone().requires_grad_(True)
        out = tig_mvmr(weight, f, idx)        # weight: no grad
        (out * go).sum().backward()
        assert f.grad is not None
        w = weight.clone().requires_grad_(True)
        out = tig_mvmr(w, feat, idx)          # feat: no grad
        (out * go).sum().backward()
        assert w.grad is not None

    @pytest.mark.parametrize("dtype,tol_x,tol_w", [
        (torch.float16, 2e-3, 2e-3), (torch.bfloat16, 2e-2, 2e-2)])
    def test_dtype_bands(self, dtype, tol_x, tol_w):
        weight, feat, go, i, j, k = rand_problem(1500, 64, 64, 20000, seed=13)
        idx = TigIndex(i, j, k, 1500)
        w = weight.to(dtype).requires_grad_(True)
        f = feat.to(dtype).requires_grad_(True)
        out = tig_mvmr(w, f, idx)
        (out * go.to(dtype)).sum().backward()
        gx_ref, gw_ref = ref_grads(weight, feat, go, i, j, k, 1500)
        assert rel_fro(f.grad, gx_ref) < tol_x
        assert rel_fro(w.grad, gw_ref) < tol_w


@requires_cuda
class TestTigBackwardVsProduction:
    def test_realistic_shape_grads_vs_production(self):
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
        go = torch.randn(N, C, device=device)

        # production fp32 reference grads
        with dispatch_mode("force_fsg"):
            wp = weight.clone().requires_grad_(True)
            fp = feat.clone().requires_grad_(True)
            out = mvmr_op(wp, k_t, fp.unsqueeze(1), j_t, i_t, N).select(1, 0)
            (out * go).sum().backward()

        idx = TigIndex(i_t, j_t, k_t, N)
        ws = weight.half().requires_grad_(True)
        fs = feat.half().requires_grad_(True)
        outs = tig_mvmr(ws, fs, idx)
        (outs * go.half()).sum().backward()

        # production fp16 grads (the band TIG must be within 3x of)
        with dispatch_mode("force_fsg"):
            wp16 = weight.half().requires_grad_(True)
            fp16 = feat.half().requires_grad_(True)
            out16 = mvmr_op(wp16, k_t, fp16.unsqueeze(1), j_t, i_t,
                            N).select(1, 0)
            (out16 * go.half()).sum().backward()

        for tig_g, prod16_g, ref_g in ((fs.grad, fp16.grad, fp.grad),
                                        (ws.grad, wp16.grad, wp.grad)):
            err_s = rel_fro(tig_g, ref_g)
            err_p = rel_fro(prod16_g, ref_g)
            assert err_s < max(err_p * 3, 2e-3), (err_s, err_p)


@requires_cuda
class TestTigGroups:
    """G>1 support (v1.2.0): block-diagonal math, flat path only."""

    @staticmethod
    def ref_groups(weight, feat, go, i, j, k, n_out):
        """fp64 oracle with groups: per (triplet, group) block-diagonal."""
        w = weight.double()           # (K, G, Cg, Mg)
        K_, G, Cg, Mg = w.shape
        f = feat.double().view(-1, G, Cg)
        g = go.double().view(-1, G, Mg)
        out = torch.zeros(n_out, G, Mg, device=feat.device,
                          dtype=torch.float64)
        gx = torch.zeros_like(f)
        gw = torch.zeros_like(w)
        for gi_ in range(G):
            prod = torch.bmm(f[j][:, gi_].unsqueeze(1),
                             w[k][:, gi_]).squeeze(1)
            out.select(1, gi_).index_add_(0, i.long(), prod)
            gx.select(1, gi_).index_add_(
                0, j.long(),
                torch.bmm(g[i][:, gi_].unsqueeze(1),
                          w[k][:, gi_].transpose(1, 2)).squeeze(1))
            gw.select(1, gi_).index_add_(
                0, k.long(),
                f[j][:, gi_].unsqueeze(2) * g[i][:, gi_].unsqueeze(1))
        return out.view(n_out, G * Mg), gx.view(-1, G * Cg), gw

    @pytest.mark.parametrize("G,Cg,Mg", [(2, 32, 32), (4, 16, 32),
                                         (2, 64, 48), (8, 32, 16),
                                         (8, 8, 8),    # GP=4 fwd/gi, GP=2 wg
                                         (2, 8, 8),    # GP=2, single pair
                                         (4, 16, 16)])  # GP=2 fwd/gi @ Cg=16
    def test_groups_fwd_bwd_fp32_ieee(self, G, Cg, Mg):
        N, T = 600, 9000
        gen = torch.Generator(device="cpu").manual_seed(G * 100 + Cg)
        i = torch.randint(0, N, (T,), generator=gen).to(device)
        j = torch.randint(0, N, (T,), generator=gen).to(device)
        k = torch.randint(0, K, (T,), generator=gen).to(device)
        feat = torch.randn(N, G * Cg, device=device)
        weight = torch.randn(K, G, Cg, Mg, device=device) * ((K * Cg) ** -0.5)
        go = torch.randn(N, G * Mg, device=device)
        idx = TigIndex(i, j, k, N)
        ref_out, ref_gx, ref_gw = self.ref_groups(weight, feat, go, i, j, k, N)
        w = weight.clone().requires_grad_(True)
        f = feat.clone().requires_grad_(True)
        out = tig_mvmr(w, f, idx, input_precision="ieee")
        (out * go).sum().backward()
        assert rel_fro(out, ref_out) < 1e-5
        assert rel_fro(f.grad, ref_gx) < 1e-5
        assert rel_fro(w.grad, ref_gw) < 1e-5

    @pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-3),
                                           (torch.bfloat16, 2e-2)])
    def test_groups_dtype_bands(self, dtype, tol):
        G, Cg, Mg, N, T = 4, 32, 32, 1200, 16000
        gen = torch.Generator(device="cpu").manual_seed(7)
        i = torch.randint(0, N, (T,), generator=gen).to(device)
        j = torch.randint(0, N, (T,), generator=gen).to(device)
        k = torch.randint(0, K, (T,), generator=gen).to(device)
        feat = torch.randn(N, G * Cg, device=device)
        weight = torch.randn(K, G, Cg, Mg, device=device) * ((K * Cg) ** -0.5)
        go = torch.randn(N, G * Mg, device=device)
        idx = TigIndex(i, j, k, N)
        ref_out, ref_gx, ref_gw = self.ref_groups(weight, feat, go, i, j, k, N)
        w = weight.to(dtype).requires_grad_(True)
        f = feat.to(dtype).requires_grad_(True)
        out = tig_mvmr(w, f, idx)
        (out * go.to(dtype)).sum().backward()
        assert rel_fro(out, ref_out) < tol
        assert rel_fro(f.grad, ref_gx) < tol
        assert rel_fro(w.grad, ref_gw) < tol

    @pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-3),
                                           (torch.bfloat16, 2e-2)])
    def test_packed_smallcg_dtype_bands(self, dtype, tol):
        """fp16/bf16 through the packed kernels at G=8, Cg==Mg==8
        (fwd + gi route GP=4 since G%4==0; wgrad routes GP=2) vs the
        fp64 oracle."""
        G, Cg, Mg, N, T = 8, 8, 8, 1200, 16000
        gen = torch.Generator(device="cpu").manual_seed(23)
        i = torch.randint(0, N, (T,), generator=gen).to(device)
        j = torch.randint(0, N, (T,), generator=gen).to(device)
        k = torch.randint(0, K, (T,), generator=gen).to(device)
        feat = torch.randn(N, G * Cg, device=device)
        weight = torch.randn(K, G, Cg, Mg, device=device) * ((K * Cg) ** -0.5)
        go = torch.randn(N, G * Mg, device=device)
        idx = TigIndex(i, j, k, N)
        ref_out, ref_gx, ref_gw = self.ref_groups(weight, feat, go, i, j, k, N)
        w = weight.to(dtype).requires_grad_(True)
        f = feat.to(dtype).requires_grad_(True)
        out = tig_mvmr(w, f, idx)
        (out * go.to(dtype)).sum().backward()
        assert rel_fro(out, ref_out) < tol
        assert rel_fro(f.grad, ref_gx) < tol
        assert rel_fro(w.grad, ref_gw) < tol

    @pytest.mark.parametrize("G,Cg,Mg", [(4, 16, 16),  # GP=2 @ Cg=Mg=16
                                         (8, 8, 8)])   # GP=4 @ Cg=Mg=8
    @pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5),
                                           (torch.float16, 1e-2)])
    def test_packed_new_routes_full_autograd(self, G, Cg, Mg, dtype, tol):
        """Pins the packed-kernel routes: fwd + grad_input pack
        GP=2 at Cg==Mg==16 (G even) and GP=4 at Cg==Mg==8 (G%4==0);
        wgrad keeps its prior routing (unpacked at Cg=16, GP=2 at Cg=8).
        Full autograd (tig_mvmr) vs the fp64 block-diagonal oracle,
        fp32-ieee tight band + fp16 representation band."""
        N, T = 1000, 14000
        gen = torch.Generator(device="cpu").manual_seed(G * 7 + Cg)
        i = torch.randint(0, N, (T,), generator=gen).to(device)
        j = torch.randint(0, N, (T,), generator=gen).to(device)
        k = torch.randint(0, K, (T,), generator=gen).to(device)
        feat = torch.randn(N, G * Cg, device=device)
        weight = torch.randn(K, G, Cg, Mg, device=device) * ((K * Cg) ** -0.5)
        go = torch.randn(N, G * Mg, device=device)
        idx = TigIndex(i, j, k, N)
        ref_out, ref_gx, ref_gw = self.ref_groups(weight, feat, go, i, j, k, N)
        w = weight.to(dtype).requires_grad_(True)
        f = feat.to(dtype).requires_grad_(True)
        prec = "ieee" if dtype == torch.float32 else "tf32"
        out = tig_mvmr(w, f, idx, input_precision=prec)
        (out * go.to(dtype)).sum().backward()
        assert rel_fro(out, ref_out) < tol
        assert rel_fro(f.grad, ref_gx) < tol
        assert rel_fro(w.grad, ref_gw) < tol

    @pytest.mark.parametrize("G,Cg,Mg", [(2, 32, 32), (4, 16, 32)])
    def test_groups_fused_backward_matches_split(self, G, Cg, Mg):
        """Fused one-pass backward at G>1 (assert lifted, group axis in
        the grid) — must match BOTH the split kernels and the fp64
        oracle. Stays unrouted by default; supported + tested only."""
        N, T = 700, 10000
        gen = torch.Generator(device="cpu").manual_seed(G * 31 + Cg)
        i = torch.randint(0, N, (T,), generator=gen).to(device)
        j = torch.randint(0, N, (T,), generator=gen).to(device)
        k = torch.randint(0, K, (T,), generator=gen).to(device)
        feat = torch.randn(N, G * Cg, device=device)
        weight = torch.randn(K, G, Cg, Mg, device=device) * ((K * Cg) ** -0.5)
        go = torch.randn(N, G * Mg, device=device)
        idx = TigIndex(i, j, k, N)
        _, ref_gx, ref_gw = self.ref_groups(weight, feat, go, i, j, k, N)
        gw_f, gx_f = tig_backward_fused(weight, feat, go, idx, weight.shape,
                                        input_precision="ieee")
        # vs fp64 oracle
        assert rel_fro(gx_f, ref_gx) < 1e-5
        assert rel_fro(gw_f, ref_gw) < 1e-5
        # vs the split path (same precision -> tight band)
        gx_s = tig_grad_input(weight, go, idx, input_precision="ieee")
        gw_s = tig_grad_weight(feat, go, idx, weight.shape,
                               input_precision="ieee")
        assert rel_fro(gx_f, gx_s) < 1e-6
        assert rel_fro(gw_f, gw_s) < 1e-6

    def test_groups_layer_level_vs_production(self):
        from functools import partial

        from layers import (PointConv3d, build_triplets,
                            radius_scaler_for_kernel_size, voxelize_3d)
        from sparse_engines._dispatch_override import (dispatch_mode,
                                                        precision_mode)

        torch.manual_seed(0)
        N, C, G, g = 8000, 128, 4, 0.16
        coord = torch.rand(N, 3, device=device) * 5.0
        si = torch.zeros(N, device=device, dtype=torch.long)
        ss = torch.tensor([N], device=device)
        with torch.no_grad():
            i_t, j_t, k_t, _ = build_triplets(
                points=coord, sample_inds=si, sample_sizes=ss,
                neighbor_radius=g * radius_scaler_for_kernel_size(3),
                kernel_indexer=partial(voxelize_3d, kernel_size=3),
                radius_scaler=radius_scaler_for_kernel_size(3),
                return_num_neighbors=False)
        pc = PointConv3d(C, C, kernel_size=3, groups=G, bias=True
                         ).to(device)
        feat = torch.randn(N, C, device=device)
        with dispatch_mode("force_pt"):
            f1 = feat.clone().requires_grad_(True)
            ref = pc(f1, i_t, j_t, k_t, N)
            ref.sum().backward()
            gref = f1.grad.clone()
            wref = pc.weight.grad.clone()
            pc.weight.grad = None
        with dispatch_mode("force_tig"), precision_mode("ieee"):
            f2 = feat.clone().requires_grad_(True)
            out = pc(f2, i_t, j_t, k_t, N)
            out.sum().backward()
        assert rel_fro(out, ref) < 1e-4
        assert rel_fro(f2.grad, gref) < 1e-4
        assert rel_fro(pc.weight.grad, wref) < 1e-4
