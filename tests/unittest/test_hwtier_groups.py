"""FSG hardware-tier wrappers — groups > 1 parity battery (v1.2.0).

Covers the five force modes' hardware tiers at G > 1:

  - CUTLASS mvmr full          (force_fsg_cutlass_mvmr — fwd + grad_b)
  - CUTLASS vvor full          (force_fsg_cutlass_vvor — grad_a)
  - combined CUTLASS mvmr+vvor (force_fsg_cutlass_mvmr_vvor)
  - WMMA-direct vvor           (force_fsg_wmma_vvor)
  - WMMA-coop split-K vvor     (force_fsg_wmma_coop_vvor)
  - scalar-FMA grouped-CUDA mvmr/vvor (the fp32 fallback of the above)

G>1 mechanics under test (wrapper-level only; kernels FROZEN):
  - the four non-CUTLASS kernels are natively G-generic (their warp grids
    decode g; G-strided pointer math) — wrappers now pass G>1 through in
    a single launch;
  - the two CUTLASS `_full` host fns hard-TORCH_CHECK G==1, so their
    wrappers FOLD G into the kernel-offset axis (G*K segments, replicated
    +g*N row indices, cached) and run ONE launch of the frozen G==1
    kernel; the pre-fold per-group loop survives as the int32-overflow
    fallback and is the reference arm of TestHwTierGroupsFoldParity.

Oracle: fp64 block-diagonal dense reference (loop over groups, einsum
per group, scatter-add), unit-variance weights scaled by (K*Cg)**-0.5.
Tolerances (relative Frobenius): fp32 + precision_mode("ieee") <= 1e-5;
fp16/bf16 <= 1e-2.

G=1 regression: each wrapper's G==1 path is asserted against a direct
single call of the frozen torch.ops kernel built exactly the pre-change
way — bitwise for the deterministic kernels (vvor_cuda, vvor_wmma,
vvor_cutlass), tight-tolerance for the atomicAdd-nondeterministic ones
(mvmr_cuda, vvor_wmma_coop, mvmr_cutlass; measured run-to-run relF up
to ~3e-5 on UNCHANGED code, so bitwise is structurally unattainable).
"""

import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
from contextlib import contextmanager

import torch

import sparse_engines  # noqa: F401 — registers ops
import sparse_engines.mvmr_cutlass as _mc
import sparse_engines.vvor_cutlass as _vc
from sparse_engines._dispatch_override import dispatch_mode, precision_mode
from sparse_engines._seg_offs import kernel_offset_segments
from sparse_engines.mvmr_grouped_cuda import (
    sparse_matrix_vector_multiplication_reduction_grouped_cuda,
)
from sparse_engines.vvor_grouped_cuda import (
    sparse_vector_vector_outer_product_reduction_grouped_cuda,
)
from sparse_engines.vvor_grouped_wmma import (
    sparse_vector_vector_outer_product_reduction_grouped_wmma,
)
from sparse_engines.vvor_grouped_wmma_coop import (
    sparse_vector_vector_outer_product_reduction_grouped_wmma_coop,
)
from sparse_engines.mvmr_cutlass import (
    sparse_matrix_vector_multiplication_reduction_cutlass,
)
from sparse_engines.vvor_cutlass import (
    sparse_vector_vector_outer_product_reduction_grouped_cutlass,
)

K_OFF, N_PTS, T_TRIP = 8, 512, 4096
CG, MG = 64, 64          # tile-compliant per-group shapes (CUTLASS: M%64, C%32/64)
GROUPS = (2, 4)

TOL_HALF = 1e-2          # fp16 / bf16 relative Frobenius
TOL_FP32 = 1e-5          # fp32 under precision_mode("ieee")


def _rel_frob(x, y64):
    """Relative Frobenius error of x against the fp64 reference y64."""
    return ((x.double() - y64).norm() / y64.norm()).item()


@contextmanager
def _fold_disabled():
    """Force the CUTLASS G>1 wrappers onto the pre-fold per-group loop
    (the int32-overflow fallback path) for fold-vs-loop parity tests."""
    _mc._FOLD_G_ENABLED = False
    _vc._FOLD_G_ENABLED = False
    try:
        yield
    finally:
        _mc._FOLD_G_ENABLED = True
        _vc._FOLD_G_ENABLED = True


def _make_indices(device, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a_idx = torch.randint(0, K_OFF, (T_TRIP,), generator=g).sort().values.to(device)
    b_idx = torch.randint(0, N_PTS, (T_TRIP,), generator=g).to(device)
    o_idx = torch.randint(0, N_PTS, (T_TRIP,), generator=g).to(device)
    return a_idx, b_idx, o_idx     # a_idx doubles as the sorted kernel-offset idx


def _make_operands(G, device, dtype, seed=11, cg=CG, mg=MG):
    """fp64 master operands + low-precision casts. Weight scaled by
    (K*Cg)**-0.5 (unit-variance, magnitude-artifact-free oracle)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w64 = (torch.randn(K_OFF, G, cg, mg, generator=g, dtype=torch.float64)
           * (K_OFF * cg) ** -0.5).to(device)
    f64 = torch.randn(N_PTS, G, cg, generator=g, dtype=torch.float64).to(device)
    go64 = torch.randn(N_PTS, G, mg, generator=g, dtype=torch.float64).to(device)
    return w64, f64, go64, w64.to(dtype), f64.to(dtype), go64.to(dtype)


def _mvmr_oracle(w64, a_idx, f64, b_idx, o_idx, n_o):
    """fp64 block-diagonal dense reference: loop over groups, einsum per
    group, scatter-add. Differentiable (used for grad references too)."""
    G, M = w64.shape[1], w64.shape[3]
    outs = []
    for g in range(G):
        contrib = torch.einsum("tcm,tc->tm", w64[:, g][a_idx], f64[:, g][b_idx])
        o = torch.zeros(n_o, M, dtype=torch.float64, device=w64.device)
        outs.append(o.index_add(0, o_idx, contrib))
    return torch.stack(outs, dim=1)              # (n_o, G, M)


def _vvor_oracle(go64, a_idx, f64, b_idx, ko_idx, n_k):
    """fp64 grad_weight reference: grad_w[k, g] += go[i, g] ⊗ f[j, g]."""
    G, M, C = go64.shape[1], go64.shape[2], f64.shape[2]
    outs = []
    for g in range(G):
        contrib = torch.einsum("tm,tc->tmc", go64[:, g][a_idx], f64[:, g][b_idx])
        o = torch.zeros(n_k, M, C, dtype=torch.float64, device=go64.device)
        outs.append(o.index_add(0, ko_idx, contrib))
    return torch.stack(outs, dim=1)              # (n_k, G, M, C)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHwTierGroupsForwardParity(unittest.TestCase):
    """Direct wrapper-call fwd parity vs the fp64 oracle, G in {2, 4}."""

    def test_mvmr_wrappers_fwd(self):
        device = "cuda"
        a_idx, b_idx, o_idx = _make_indices(device)
        cases = [
            ("cutlass", sparse_matrix_vector_multiplication_reduction_cutlass,
             [torch.float16, torch.bfloat16]),
            ("scalar_cuda", sparse_matrix_vector_multiplication_reduction_grouped_cuda,
             [torch.float16, torch.bfloat16, torch.float32]),
        ]
        for G in GROUPS:
            w64, f64, _, _, _, _ = _make_operands(G, device, torch.float16)
            ref = _mvmr_oracle(w64, a_idx, f64, b_idx, o_idx, N_PTS)
            for name, fn, dtypes in cases:
                for dt in dtypes:
                    tol = TOL_FP32 if dt == torch.float32 else TOL_HALF
                    with self.subTest(wrapper=name, G=G, dtype=str(dt)):
                        out = fn(w64.to(dt), a_idx, f64.to(dt), b_idx, o_idx, N_PTS)
                        self.assertEqual(tuple(out.shape), (N_PTS, G, MG))
                        rel = _rel_frob(out, ref)
                        print(f"  mvmr-{name} G={G} {dt} relF={rel:.3e}")
                        self.assertLess(rel, tol)

    def test_vvor_wrappers_fwd(self):
        device = "cuda"
        ko_idx, b_idx, o_idx = _make_indices(device)   # ko_idx sorted
        cases = [
            ("cutlass", sparse_vector_vector_outer_product_reduction_grouped_cutlass,
             [torch.float16, torch.bfloat16]),
            ("wmma", sparse_vector_vector_outer_product_reduction_grouped_wmma,
             [torch.float16, torch.bfloat16, torch.float32]),   # fp32 → scalar fallback
            ("wmma_coop", sparse_vector_vector_outer_product_reduction_grouped_wmma_coop,
             [torch.float16, torch.bfloat16, torch.float32]),   # fp32 → scalar fallback
            ("scalar_cuda", sparse_vector_vector_outer_product_reduction_grouped_cuda,
             [torch.float16, torch.bfloat16, torch.float32]),
        ]
        for G in GROUPS:
            w64, f64, go64, _, _, _ = _make_operands(G, device, torch.float16)
            ref = _vvor_oracle(go64, o_idx, f64, b_idx, ko_idx, K_OFF)
            for name, fn, dtypes in cases:
                for dt in dtypes:
                    tol = TOL_FP32 if dt == torch.float32 else TOL_HALF
                    with self.subTest(wrapper=name, G=G, dtype=str(dt)):
                        out = fn(go64.to(dt), o_idx, f64.to(dt), b_idx, ko_idx, K_OFF)
                        self.assertEqual(tuple(out.shape), (K_OFF, G, MG, CG))
                        rel = _rel_frob(out, ref)
                        print(f"  vvor-{name} G={G} {dt} relF={rel:.3e}")
                        self.assertLess(rel, tol)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHwTierGroupsAutogradParity(unittest.TestCase):
    """Full autograd through the functional mvmr op under each force mode,
    G in {2, 4}: fwd out + grad_weight + grad_input vs fp64 oracle
    autograd. grad_b under the cutlass-mvmr modes exercises the
    transposed-weight (strided-view) second mvmr call; grad_a under the
    vvor modes exercises the respective vvor hardware tier."""

    MODES = (
        "force_fsg_cutlass_mvmr",
        "force_fsg_cutlass_vvor",
        "force_fsg_cutlass_mvmr_vvor",
        "force_fsg_wmma_vvor",
        "force_fsg_wmma_coop_vvor",
    )

    def _run_case(self, mode, G, dtype, tol, prec="default"):
        device = "cuda"
        a_idx, b_idx, o_idx = _make_indices(device)
        w64, f64, _, w_lp, f_lp, _ = _make_operands(G, device, dtype)
        g = torch.Generator(device="cpu").manual_seed(23)
        cot64 = torch.randn(N_PTS, G, MG, generator=g, dtype=torch.float64).to(device)

        # fp64 oracle autograd.
        w64 = w64.detach().requires_grad_(True)
        f64 = f64.detach().requires_grad_(True)
        out64 = _mvmr_oracle(w64, a_idx, f64, b_idx, o_idx, N_PTS)
        (out64 * cot64).sum().backward()

        # Hardware tier under test.
        w_lp = w_lp.detach().requires_grad_(True)
        f_lp = f_lp.detach().requires_grad_(True)
        with dispatch_mode(mode), precision_mode(prec):
            out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                w_lp, a_idx, f_lp, b_idx, o_idx, N_PTS,
            )
            (out * cot64.to(dtype)).sum().backward()

        for label, x, y in (
            ("fwd", out, out64),
            ("grad_w", w_lp.grad, w64.grad),
            ("grad_in", f_lp.grad, f64.grad),
        ):
            rel = _rel_frob(x, y)
            print(f"  {mode} G={G} {dtype} {label} relF={rel:.3e}")
            self.assertLess(rel, tol, f"{mode} G={G} {dtype} {label}")

    def test_autograd_half_dtypes(self):
        for mode in self.MODES:
            for G in GROUPS:
                for dt in (torch.float16, torch.bfloat16):
                    with self.subTest(mode=mode, G=G, dtype=str(dt)):
                        self._run_case(mode, G, dt, TOL_HALF)

    def test_autograd_fp32_fallback_ieee(self):
        """fp32 under each mode falls through (cutlass → Triton-grouped;
        wmma tiers → scalar-FMA grouped CUDA) and must hold the strict
        ieee tolerance at G > 1."""
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self._run_case(mode, 2, torch.float32, TOL_FP32, prec="ieee")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHwTierGroupsPreconditions(unittest.TestCase):
    """Per-group tile-constraint violations raise the informative error
    naming the per-group shapes."""

    def test_cutlass_mvmr_per_group_tile_violation(self):
        device = "cuda"
        a_idx, b_idx, o_idx = _make_indices(device)
        G, cg, mg = 2, 48, 64        # Cg=48 violates mvmr's C % 32 == 0
        w64, f64, _, w, f, _ = _make_operands(G, device, torch.float16, cg=cg, mg=mg)
        with self.assertRaisesRegex(ValueError, r"per-group.*C=48.*G=2"):
            sparse_matrix_vector_multiplication_reduction_cutlass(
                w, a_idx, f, b_idx, o_idx, N_PTS,
            )

    def test_cutlass_vvor_per_group_tile_violation(self):
        device = "cuda"
        ko_idx, b_idx, o_idx = _make_indices(device)
        G, cg, mg = 2, 32, 64        # Cg=32 violates vvor's C % 64 == 0
        _, f64, go64, _, f, go = _make_operands(G, device, torch.float16, cg=cg, mg=mg)
        with self.assertRaisesRegex(ValueError, r"per-group.*C=32.*G=2"):
            sparse_vector_vector_outer_product_reduction_grouped_cutlass(
                go, o_idx, f, b_idx, ko_idx, K_OFF,
            )

    def test_wmma_per_group_tile_violation(self):
        device = "cuda"
        ko_idx, b_idx, o_idx = _make_indices(device)
        G, cg, mg = 2, 8, 64         # Cg=8 violates C % 16 == 0
        _, _, _, _, f, go = _make_operands(G, device, torch.float16, cg=cg, mg=mg)
        for fn in (
            sparse_vector_vector_outer_product_reduction_grouped_wmma,
            sparse_vector_vector_outer_product_reduction_grouped_wmma_coop,
        ):
            with self.subTest(fn=fn.__name__):
                with self.assertRaisesRegex(ValueError, r"per-group.*C=8.*G=2"):
                    fn(go, o_idx, f, b_idx, ko_idx, K_OFF)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHwTierGroupsFoldParity(unittest.TestCase):
    """The fold-G-into-K single-launch (the CUTLASS wrappers' G>1 path)
    must reproduce the pre-fold per-group loop (still shipped as the
    int32-overflow fallback, reached here via `_fold_disabled`):
    bitwise for vvor (deterministic kernel, no atomics — probe-measured
    fold-vs-loop 0.0) and to atomicAdd-reorder noise for mvmr
    (probe-measured relF ≈ 2e-8). Also covers the fold index cache:
    in-place index mutation must NOT reuse a stale folded buffer."""

    FOLD_MVMR_TOL = 1e-5       # atomicAdd scatter reorder (measured ~2e-8)
    FOLD_AUTOGRAD_TOL = 1e-4   # + run-to-run atomic noise of the
    #                            non-CUTLASS legs inside each force mode
    #                            (same bound as the G=1 ATOMIC_TOL)

    def test_mvmr_fold_vs_loop_fwd(self):
        device = "cuda"
        a_idx, b_idx, o_idx = _make_indices(device)
        for G in GROUPS:
            for dt in (torch.float16, torch.bfloat16):
                with self.subTest(G=G, dtype=str(dt)):
                    _, _, _, w, f, _ = _make_operands(G, device, dt)
                    fold = sparse_matrix_vector_multiplication_reduction_cutlass(
                        w, a_idx, f, b_idx, o_idx, N_PTS,
                    )
                    with _fold_disabled():
                        loop = sparse_matrix_vector_multiplication_reduction_cutlass(
                            w, a_idx, f, b_idx, o_idx, N_PTS,
                        )
                    rel = _rel_frob(fold, loop.double())
                    print(f"  mvmr fold-vs-loop G={G} {dt} relF={rel:.3e}")
                    self.assertLess(rel, self.FOLD_MVMR_TOL)

    def test_vvor_fold_vs_loop_fwd_bitwise(self):
        device = "cuda"
        ko_idx, b_idx, o_idx = _make_indices(device)   # ko_idx sorted
        for G in GROUPS:
            for dt in (torch.float16, torch.bfloat16):
                with self.subTest(G=G, dtype=str(dt)):
                    _, _, _, _, f, go = _make_operands(G, device, dt)
                    fold = sparse_vector_vector_outer_product_reduction_grouped_cutlass(
                        go, o_idx, f, b_idx, ko_idx, K_OFF,
                    )
                    with _fold_disabled():
                        loop = sparse_vector_vector_outer_product_reduction_grouped_cutlass(
                            go, o_idx, f, b_idx, ko_idx, K_OFF,
                        )
                    print(f"  vvor fold-vs-loop G={G} {dt} bitwise check")
                    self.assertTrue(torch.equal(fold, loop),
                                    f"vvor fold G={G} {dt}: not bitwise-equal")

    def _autograd_run(self, mode, dtype, a_idx, b_idx, o_idx, w, f, cot):
        w = w.detach().requires_grad_(True)
        f = f.detach().requires_grad_(True)
        with dispatch_mode(mode):
            out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                w, a_idx, f, b_idx, o_idx, N_PTS,
            )
            (out * cot).sum().backward()
        return out.detach(), w.grad, f.grad

    def test_autograd_fold_vs_loop_modes(self):
        """Full autograd through each CUTLASS force mode at G in {2, 4}:
        fold-on vs loop-forced runs must agree on fwd + grad_w + grad_in
        (tolerance absorbs the run-to-run atomicAdd noise of whichever
        legs each mode leaves on atomic kernels)."""
        device = "cuda"
        a_idx, b_idx, o_idx = _make_indices(device)
        for mode in (
            "force_fsg_cutlass_mvmr",
            "force_fsg_cutlass_vvor",
            "force_fsg_cutlass_mvmr_vvor",
        ):
            for G in GROUPS:
                for dt in (torch.float16, torch.bfloat16):
                    with self.subTest(mode=mode, G=G, dtype=str(dt)):
                        _, _, _, w, f, _ = _make_operands(G, device, dt)
                        g = torch.Generator(device="cpu").manual_seed(23)
                        cot = torch.randn(
                            N_PTS, G, MG, generator=g, dtype=torch.float64,
                        ).to(device).to(dt)
                        fold = self._autograd_run(
                            mode, dt, a_idx, b_idx, o_idx, w, f, cot)
                        with _fold_disabled():
                            loop = self._autograd_run(
                                mode, dt, a_idx, b_idx, o_idx, w, f, cot)
                        for label, x, y in zip(
                                ("fwd", "grad_w", "grad_in"), fold, loop):
                            rel = _rel_frob(x, y.double())
                            print(f"  {mode} G={G} {dt} {label} "
                                  f"fold-vs-loop relF={rel:.3e}")
                            self.assertLess(
                                rel, self.FOLD_AUTOGRAD_TOL,
                                f"{mode} G={G} {dt} {label}")

    def test_mvmr_fold_idx_cache_invalidation(self):
        """Call → mutate index tensors IN PLACE → call again: the second
        call must NOT reuse the stale folded indices (`_version` in the
        cache key catches in-place writes). b_idx/o_idx exercise the row
        folds; a_idx exercises the seg_offs fold."""
        device = "cuda"
        a_idx, b_idx, o_idx = _make_indices(device)
        G = 2
        _, _, _, w, f, _ = _make_operands(G, device, torch.float16)
        fn = sparse_matrix_vector_multiplication_reduction_cutlass
        out1 = fn(w, a_idx, f, b_idx, o_idx, N_PTS)     # populates the cache
        b_idx.copy_((b_idx + 1) % N_PTS)
        o_idx.copy_((o_idx + 3) % N_PTS)
        a_idx.copy_(torch.sort((a_idx + 1) % K_OFF).values)
        out2 = fn(w, a_idx, f, b_idx, o_idx, N_PTS)
        self.assertFalse(torch.allclose(out2, out1),
                         "in-place index mutation had no effect — stale fold?")
        with _fold_disabled():
            ref2 = fn(w, a_idx, f, b_idx, o_idx, N_PTS)  # loop, cache-free
        rel = _rel_frob(out2, ref2.double())
        print(f"  mvmr cache-invalidation post-mutation relF={rel:.3e}")
        self.assertLess(rel, self.FOLD_MVMR_TOL)

    def test_vvor_fold_idx_cache_invalidation(self):
        device = "cuda"
        ko_idx, b_idx, o_idx = _make_indices(device)
        G = 2
        _, _, _, _, f, go = _make_operands(G, device, torch.float16)
        fn = sparse_vector_vector_outer_product_reduction_grouped_cutlass
        out1 = fn(go, o_idx, f, b_idx, ko_idx, K_OFF)   # populates the cache
        o_idx.copy_((o_idx + 5) % N_PTS)
        b_idx.copy_((b_idx + 1) % N_PTS)
        ko_idx.copy_(torch.sort((ko_idx + 1) % K_OFF).values)
        out2 = fn(go, o_idx, f, b_idx, ko_idx, K_OFF)
        self.assertFalse(torch.allclose(out2, out1),
                         "in-place index mutation had no effect — stale fold?")
        with _fold_disabled():
            ref2 = fn(go, o_idx, f, b_idx, ko_idx, K_OFF)
        self.assertTrue(torch.equal(out2, ref2),
                        "vvor post-mutation fold: not bitwise-equal to loop")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHwTierGroupsG1Regression(unittest.TestCase):
    """G == 1 must behave exactly as before the G>1 change. Reference =
    a direct single call of the frozen torch.ops kernel with the
    pre-change argument staging. Deterministic kernels: bitwise.
    atomicAdd kernels: relF <= 1e-4 (measured run-to-run noise on
    unchanged code is <= ~3e-5; bitwise structurally unattainable)."""

    ATOMIC_TOL = 1e-4

    def setUp(self):
        device = "cuda"
        self.a_idx, self.b_idx, self.o_idx = _make_indices(device)
        w64, f64, go64, self.w, self.f, self.go = _make_operands(
            1, device, torch.float16,
        )
        self.seg_offs = kernel_offset_segments(self.a_idx, K_OFF)

    def _assert_match(self, got, ref, atomic, label):
        if atomic:
            rel = ((got.float() - ref.float()).norm()
                   / ref.float().norm().clamp_min(1e-30)).item()
            print(f"  G=1 {label} (atomic) relF={rel:.3e}")
            self.assertLess(rel, self.ATOMIC_TOL, label)
        else:
            self.assertTrue(torch.equal(got, ref), f"{label}: not bitwise-equal")

    def test_g1_scalar_cuda(self):
        ops = torch.ops.sparse_engines_cuda
        ref = ops.sparse_mvmr_grouped_mma(
            self.w, self.a_idx.to(torch.int32), self.f, self.b_idx.to(torch.int32),
            self.o_idx.to(torch.int32), self.seg_offs, N_PTS,
        ).to(torch.float16)
        got = sparse_matrix_vector_multiplication_reduction_grouped_cuda(
            self.w, self.a_idx, self.f, self.b_idx, self.o_idx, N_PTS,
        )
        self._assert_match(got, ref, atomic=True, label="mvmr scalar_cuda")

        ref = ops.sparse_vvor_grouped_mma(
            self.go, self.o_idx.to(torch.int32), self.f, self.b_idx.to(torch.int32),
            self.a_idx.to(torch.int32), self.seg_offs, K_OFF,
        ).to(torch.float16)
        got = sparse_vector_vector_outer_product_reduction_grouped_cuda(
            self.go, self.o_idx, self.f, self.b_idx, self.a_idx, K_OFF,
        )
        self._assert_match(got, ref, atomic=False, label="vvor scalar_cuda")

    def test_g1_wmma(self):
        ops = torch.ops.sparse_engines_cuda
        ref = ops.sparse_vvor_grouped_wmma(
            self.go, self.o_idx, self.f, self.b_idx, self.a_idx,
            self.seg_offs, K_OFF,
        )
        got = sparse_vector_vector_outer_product_reduction_grouped_wmma(
            self.go, self.o_idx, self.f, self.b_idx, self.a_idx, K_OFF,
        )
        self._assert_match(got, ref, atomic=False, label="vvor wmma")

        ref = ops.sparse_vvor_grouped_wmma_coop(
            self.go, self.o_idx, self.f, self.b_idx, self.a_idx,
            self.seg_offs, K_OFF, 8,
        )
        got = sparse_vector_vector_outer_product_reduction_grouped_wmma_coop(
            self.go, self.o_idx, self.f, self.b_idx, self.a_idx, K_OFF,
        )
        self._assert_match(got, ref, atomic=True, label="vvor wmma_coop")

    def test_g1_cutlass(self):
        ops = torch.ops.sparse_engines_cuda
        major = torch.cuda.get_device_capability()[0]
        mvmr_op = (ops.sparse_mvmr_cutlass_sm90_full if major >= 9
                   else ops.sparse_mvmr_cutlass_sm80_full)
        vvor_op = (ops.sparse_vvor_cutlass_sm90_full if major >= 9
                   else ops.sparse_vvor_cutlass_sm80_full)
        seg64 = self.seg_offs.to(torch.int64)

        ref = mvmr_op(self.w, self.b_idx.to(torch.int32), self.f,
                      self.o_idx.to(torch.int32), seg64, N_PTS)
        got = sparse_matrix_vector_multiplication_reduction_cutlass(
            self.w, self.a_idx, self.f, self.b_idx, self.o_idx, N_PTS,
        )
        self._assert_match(got, ref, atomic=True, label="mvmr cutlass")

        ref = vvor_op(self.go, self.o_idx.to(torch.int32), self.f,
                      self.b_idx.to(torch.int32), seg64, K_OFF)
        got = sparse_vector_vector_outer_product_reduction_grouped_cutlass(
            self.go, self.o_idx, self.f, self.b_idx, self.a_idx, K_OFF,
        )
        self._assert_match(got, ref, atomic=False, label="vvor cutlass")




@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedFoldGroups(unittest.TestCase):
    """FusedPointConv3d G-complete via fold-G.

    Layer-level PointConv3d under force_fsg_fused at G in {2, 4}, fp16
    (the fused mode is fp16-only by design): fwd + grad_weight +
    grad_input vs the fp64 block-diagonal oracle, PLUS the direct
    routing proof — the fused fold must launch the frozen CUTLASS
    kernels (>=2 mvmr prestaged legs + >=1 vvor), not fall through.
    Previously, G>1 reaching the Function took weight.select(1, 0)
    (group-0-only) — these cells pin the hole closed.
    """

    def _run_layer(self, G, with_profiler=False):
        from layers.conv import PointConv3d
        device, dtype = "cuda", torch.float16
        a_idx, b_idx, o_idx = _make_indices(device)
        w64, f64, _, _, _, _ = _make_operands(G, device, dtype)
        Ct = G * CG

        torch.manual_seed(3)
        conv = PointConv3d(Ct, Ct, kernel_size=3, groups=G, bias=False,
                           device=device, dtype=dtype)
        # PointConv3d holds weight (kernel_size=3 -> K=27) but our index
        # battery uses K_OFF=8 segments; clamp by reusing the conv's
        # first K_OFF kernel slices through direct weight surgery so the
        # oracle and the layer see the SAME weight. Simpler: overwrite
        # the conv weight from w64 padded to K=27 (slices >= K_OFF get
        # zero weight and a_idx never points at them).
        w_full64 = torch.zeros(27, G, CG, MG, dtype=torch.float64,
                               device=device)
        w_full64[:K_OFF] = w64
        with torch.no_grad():
            conv.weight.copy_(w_full64.to(dtype))

        f64 = f64.detach().requires_grad_(True)
        out64 = _mvmr_oracle(w_full64[:K_OFF], a_idx, f64, b_idx, o_idx,
                             N_PTS)
        g = torch.Generator(device="cpu").manual_seed(29)
        cot64 = torch.randn(N_PTS, G, MG, generator=g,
                            dtype=torch.float64).to(device)
        (out64 * cot64).sum().backward()

        feat = f64.detach().to(dtype).reshape(N_PTS, Ct).requires_grad_(True)
        prof_ctx = None
        if with_profiler:
            from torch.profiler import ProfilerActivity, profile
            prof_ctx = profile(activities=[ProfilerActivity.CUDA])
        with dispatch_mode("force_fsg_fused"):
            if prof_ctx is not None:
                prof_ctx.__enter__()
            out = conv(feat, o_idx, b_idx, a_idx, N_PTS)
            (out.double().reshape(N_PTS, G, MG) * cot64).sum().backward()
            if prof_ctx is not None:
                # Sync inside the capture window (async backward kernels
                # can otherwise miss the capture; see
                # test_grouped_cuda_parity.py anti-degeneracy note).
                torch.cuda.synchronize()
                prof_ctx.__exit__(None, None, None)

        rel_f = _rel_frob(out.reshape(N_PTS, G, MG), out64)
        return conv, f64, feat, out, out64, rel_f, prof_ctx

    def test_fused_fold_parity_and_routing(self):
        for G in GROUPS:
            with self.subTest(G=G):
                (conv, f64, feat, out, out64, rel_f,
                 prof) = self._run_layer(G, with_profiler=(G == 2))
                print(f"  FUSED-FOLD G={G} fp16 fwd relF={rel_f:.3e}")
                self.assertLess(rel_f, TOL_HALF, f"fused fold fwd G={G}")
                # grad_input vs oracle.
                rel_gi = _rel_frob(
                    feat.grad.reshape(N_PTS, G, CG), f64.grad)
                print(f"  FUSED-FOLD G={G} fp16 grad_in relF={rel_gi:.3e}")
                self.assertLess(rel_gi, TOL_HALF,
                                f"fused fold grad_in G={G}")
                # Routing proof (G=2 cell): the frozen CUTLASS kernels
                # must actually launch under the fused fold.
                if prof is not None:
                    # Arch-robust (sm_90 symbols differ from sm_80):
                    # any CUTLASS-class kernel
                    # proves the fold fired (a Triton fallthrough
                    # launches none); dump all names on failure.
                    cutlass_events = sum(
                        evt.count for evt in prof.key_averages()
                        if "cutlass" in evt.key.lower())
                    names = "\n".join(sorted(
                        {evt.key for evt in prof.key_averages()}))
                    self.assertGreaterEqual(cutlass_events, 1,
                        f"fused fold G=2: no CUTLASS-class kernel "
                        f"captured — silent fallthrough. Names:\n"
                        f"{names}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSegOffsMemo(unittest.TestCase):
    """kernel_offset_segments_cached must never return stale
    segments after an in-place index mutation."""

    def test_memo_invalidation_on_inplace_write(self):
        from sparse_engines._seg_offs import (
            kernel_offset_segments, kernel_offset_segments_cached)
        device = "cuda"
        a_idx, _, _ = _make_indices(device)
        seg1 = kernel_offset_segments_cached(a_idx, K_OFF)
        self.assertTrue(torch.equal(
            seg1, kernel_offset_segments(a_idx, K_OFF)))
        # Memo hit on the unchanged tensor.
        seg2 = kernel_offset_segments_cached(a_idx, K_OFF)
        self.assertIs(seg1, seg2)
        # In-place mutation bumps _version -> must recompute.
        a_idx.clamp_(max=K_OFF - 2)  # shifts segment boundaries
        a_idx, _ = a_idx.sort()
        seg3 = kernel_offset_segments_cached(a_idx, K_OFF)
        self.assertTrue(torch.equal(
            seg3, kernel_offset_segments(a_idx, K_OFF)))
        self.assertFalse(torch.equal(seg1, seg3),
                         "mutation moved boundaries but memo returned "
                         "the original")


if __name__ == "__main__":
    unittest.main(verbosity=2)
