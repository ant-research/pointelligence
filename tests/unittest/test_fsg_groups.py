"""FSG (full-segment grouped Triton) groups>1 parity tests — v1.2.0.

The G==1 gates in mvmr/vvor `_try_grouped_dispatch` were lifted for force
modes (auto still requires G==1). This file is the rigorous G>1 validation
against an fp64 block-diagonal dense oracle (loop over groups, einsum per
group, index_add scatter):

  - forward (mvmr grouped), G in {1, 2, 4, 8}, ragged per-group widths
    (Cg != Mg), all three dtypes;
  - vvor grouped (the grad_a engine) directly;
  - full autograd: out.sum().backward() under force_fsg — grad_a flows
    through grouped vvor, grad_b through a second grouped mvmr with the
    transposed weight — vs fp64-autograd oracle grads;
  - segment-walk edge cases: an EMPTY kernel-offset segment, a segment
    shorter than the smallest L_CHUNK, T not a multiple of any L_CHUNK,
    Cg/Mg not multiples of the tile sizes (masking correctness);
  - G=1 regression cells (the pre-lift behaviour must be unchanged).

Tolerances (relative Frobenius): fp32 + precision_mode("ieee") <= 1e-5;
fp16/bf16 <= 1e-2. Weights are unit-variance randn scaled by
(K*Cg)**-0.5 to avoid magnitude artifacts.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch
import triton

import sparse_engines
from sparse_engines._dispatch_override import dispatch_mode, precision_mode
from sparse_engines import mvmr_triton, vvor_triton
from sparse_engines._seg_offs import kernel_offset_segments

device = "cuda"

K_OFF = 27          # kernel offsets (k=3 conv)
N_FEAT = 2048       # feature rows
N_OUT = 2048        # output rows
T_DEFAULT = 24_001  # odd on purpose: not a multiple of any L_CHUNK

# (name, G, Cg, Mg) — per-group channel widths. Ragged (Cg != Mg) cells,
# a non-multiple-of-16 masking cell, a below-tl.dot-floor cell, and G=1
# regression cells.
CELLS = [
    ("G1_c64_m64",   1, 64, 64),   # G=1 regression (pre-lift path)
    ("G1_c16_m48",   1, 16, 48),   # G=1 regression, ragged
    ("G2_c16_m48",   2, 16, 48),
    ("G4_c16_m48",   4, 16, 48),
    ("G8_c16_m48",   8, 16, 48),
    ("G4_c24_m40",   4, 24, 40),   # Cg/Mg NOT multiples of 16 → tile masking
    ("G8_c8_m8",     8,  8,  8),   # below the sm_8x tl.dot 16-floor → masked
]

# (name, dtype, tolerance, use_ieee)
DTYPES = [
    ("fp32-ieee", torch.float32, 1e-5, True),
    ("fp16",      torch.float16, 1e-2, False),
    ("bf16",      torch.bfloat16, 1e-2, False),
]


# ── fp64 block-diagonal dense oracle ─────────────────────────────────────


def oracle_mvmr(a, a_idx, b, b_idx, o_idx, n_o):
    """out[i, g] += weight[k, g] applied to feat[j, g], all in fp64."""
    a64, b64 = a.double(), b.double()
    K, G, C, M = a64.shape
    o = torch.zeros(n_o, G, M, dtype=torch.float64, device=a.device)
    for g in range(G):
        contrib = torch.einsum("tcm,tc->tm", a64[a_idx, g], b64[b_idx, g])
        o[:, g].index_add_(0, o_idx, contrib)
    return o


def oracle_vvor(a, a_idx, b, b_idx, o_idx, n_o):
    """o[k, g] += a[a_idx, g] (outer) b[b_idx, g], all in fp64."""
    a64, b64 = a.double(), b.double()
    G = a64.shape[1]
    o = torch.zeros(n_o, G, a64.shape[2], b64.shape[2],
                    dtype=torch.float64, device=a.device)
    for g in range(G):
        contrib = torch.einsum("tm,tc->tmc", a64[a_idx, g], b64[b_idx, g])
        o[:, g].index_add_(0, o_idx, contrib)
    return o


def rel_fro(x, ref):
    """Relative Frobenius error, computed in fp64."""
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


# ── index builders ────────────────────────────────────────────────────────


def make_sorted_indices(T=T_DEFAULT, seed=1):
    """Random triplets, a_idx sorted ascending (sort_by='k')."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a_idx = torch.randint(0, K_OFF, (T,), generator=gen).sort().values
    b_idx = torch.randint(0, N_FEAT, (T,), generator=gen)
    o_idx = torch.randint(0, N_OUT, (T,), generator=gen)
    return a_idx.to(device), b_idx.to(device), o_idx.to(device)


def make_edge_indices(seed=2):
    """Segment-walk stress: explicit per-kernel-offset segment lengths
    including EMPTY segments, a length-3 segment (< smallest L_CHUNK=16),
    lengths that straddle chunk boundaries, and a total T that is not a
    multiple of any L_CHUNK."""
    seg_lens = [0, 3, 1000, 17, 0, 250, 1, 129, 0, 0, 31, 513, 64, 5,
                0, 999, 16, 2, 77, 0, 300, 128, 0, 33, 257, 0, 11]
    assert len(seg_lens) == K_OFF
    a_idx = torch.repeat_interleave(
        torch.arange(K_OFF), torch.tensor(seg_lens))
    T = a_idx.numel()  # 4174 — not a multiple of 16/32/64/128/256
    gen = torch.Generator(device="cpu").manual_seed(seed)
    b_idx = torch.randint(0, N_FEAT, (T,), generator=gen)
    o_idx = torch.randint(0, N_OUT, (T,), generator=gen)
    return a_idx.to(device), b_idx.to(device), o_idx.to(device)


def make_data(G, Cg, Mg, dtype, seed=0):
    torch.manual_seed(seed)
    w = (torch.randn(K_OFF, G, Cg, Mg, device=device)
         * (K_OFF * Cg) ** -0.5).to(dtype)
    feat = torch.randn(N_FEAT, G, Cg, device=device).to(dtype)
    return w, feat


class TestFsgGroups(unittest.TestCase):

    # ── 1. mvmr forward, grouped, vs fp64 oracle ─────────────────────────

    def test_mvmr_forward(self):
        a_idx, b_idx, o_idx = make_sorted_indices()
        for cell, G, Cg, Mg in CELLS:
            for dt_name, dtype, tol, ieee in DTYPES:
                with self.subTest(cell=cell, dtype=dt_name):
                    w, feat = make_data(G, Cg, Mg, dtype)
                    ref = oracle_mvmr(w, a_idx, feat, b_idx, o_idx, N_OUT)
                    with dispatch_mode("force_fsg"), \
                         precision_mode("ieee" if ieee else "default"):
                        out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                            w, a_idx, feat, b_idx, o_idx, N_OUT)
                    rel = rel_fro(out, ref)
                    print(f"  FSG mvmr-fwd [{cell} {dt_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol, f"{cell} {dt_name}")

    # ── 2. vvor forward (grad_a engine), grouped, vs fp64 oracle ─────────

    def test_vvor_forward(self):
        # vvor's o_idx is the kernel offset; sorted by it (sort_by="k").
        T = T_DEFAULT
        gen = torch.Generator(device="cpu").manual_seed(3)
        o_idx = torch.randint(0, K_OFF, (T,), generator=gen).sort().values.to(device)
        a_idx = torch.randint(0, N_OUT, (T,), generator=gen).to(device)
        b_idx = torch.randint(0, N_FEAT, (T,), generator=gen).to(device)
        for cell, G, Cg, Mg in CELLS:
            for dt_name, dtype, tol, ieee in DTYPES:
                with self.subTest(cell=cell, dtype=dt_name):
                    torch.manual_seed(0)
                    grad = (torch.randn(N_OUT, G, Mg, device=device)
                            * (K_OFF * Cg) ** -0.5).to(dtype)
                    feat = torch.randn(N_FEAT, G, Cg, device=device).to(dtype)
                    ref = oracle_vvor(grad, a_idx, feat, b_idx, o_idx, K_OFF)
                    with dispatch_mode("force_fsg"), \
                         precision_mode("ieee" if ieee else "default"):
                        out = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
                            grad, a_idx, feat, b_idx, o_idx, K_OFF)
                    rel = rel_fro(out, ref)
                    print(f"  FSG vvor-fwd [{cell} {dt_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol, f"{cell} {dt_name}")

    # ── 3. full autograd: grad_a (grouped vvor) + grad_b (grouped mvmr
    #      with transposed weight) vs fp64-autograd oracle ────────────────

    def test_autograd(self):
        a_idx, b_idx, o_idx = make_sorted_indices()
        for cell, G, Cg, Mg in CELLS:
            for dt_name, dtype, tol, ieee in DTYPES:
                with self.subTest(cell=cell, dtype=dt_name):
                    w_data, feat_data = make_data(G, Cg, Mg, dtype)

                    # fp64 oracle grads via autograd on the dense reference
                    w64 = w_data.double().requires_grad_(True)
                    f64 = feat_data.double().requires_grad_(True)
                    oracle_mvmr(w64, a_idx, f64, b_idx, o_idx, N_OUT).sum().backward()

                    w = w_data.detach().clone().requires_grad_(True)
                    feat = feat_data.detach().clone().requires_grad_(True)
                    with dispatch_mode("force_fsg"), \
                         precision_mode("ieee" if ieee else "default"):
                        out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                            w, a_idx, feat, b_idx, o_idx, N_OUT)
                        out.sum().backward()

                    rel_w = rel_fro(w.grad, w64.grad)
                    rel_f = rel_fro(feat.grad, f64.grad)
                    print(f"  FSG autograd [{cell} {dt_name}] "
                          f"grad_w rel={rel_w:.3e} grad_f rel={rel_f:.3e}")
                    self.assertLess(rel_w, tol, f"{cell} {dt_name} grad_w")
                    self.assertLess(rel_f, tol, f"{cell} {dt_name} grad_f")

    # ── 4. segment-walk edge cases ────────────────────────────────────────

    def test_edge_segments_forward(self):
        a_idx, b_idx, o_idx = make_edge_indices()
        for cell, G, Cg, Mg in [("G4_c16_m48", 4, 16, 48),
                                ("G4_c24_m40", 4, 24, 40),
                                ("G8_c8_m8",   8,  8,  8)]:
            for dt_name, dtype, tol, ieee in DTYPES:
                with self.subTest(cell=cell, dtype=dt_name):
                    w, feat = make_data(G, Cg, Mg, dtype)
                    ref = oracle_mvmr(w, a_idx, feat, b_idx, o_idx, N_OUT)
                    with dispatch_mode("force_fsg"), \
                         precision_mode("ieee" if ieee else "default"):
                        out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                            w, a_idx, feat, b_idx, o_idx, N_OUT)
                    rel = rel_fro(out, ref)
                    print(f"  FSG edge-fwd [{cell} {dt_name}] rel={rel:.3e}")
                    self.assertLess(rel, tol, f"{cell} {dt_name}")

    def test_edge_segments_autograd_fp32_ieee(self):
        a_idx, b_idx, o_idx = make_edge_indices()
        G, Cg, Mg = 4, 24, 40
        w_data, feat_data = make_data(G, Cg, Mg, torch.float32)
        w64 = w_data.double().requires_grad_(True)
        f64 = feat_data.double().requires_grad_(True)
        oracle_mvmr(w64, a_idx, f64, b_idx, o_idx, N_OUT).sum().backward()
        w = w_data.detach().clone().requires_grad_(True)
        feat = feat_data.detach().clone().requires_grad_(True)
        with dispatch_mode("force_fsg"), precision_mode("ieee"):
            out = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                w, a_idx, feat, b_idx, o_idx, N_OUT)
            out.sum().backward()
        rel_w = rel_fro(w.grad, w64.grad)
        rel_f = rel_fro(feat.grad, f64.grad)
        print(f"  FSG edge-autograd [G4_c24_m40 fp32-ieee] "
              f"grad_w rel={rel_w:.3e} grad_f rel={rel_f:.3e}")
        self.assertLess(rel_w, 1e-5)
        self.assertLess(rel_f, 1e-5)

    # ── 5. dispatch proof: the grouped path actually FIRES at G>1 under
    #      force_fsg (no silent per-triplet fallback), and correctly
    #      refuses unsorted indices ─────────────────────────────────────────

    def test_grouped_dispatch_fires_at_g_gt_1(self):
        a_idx, b_idx, o_idx = make_sorted_indices()
        G, Cg, Mg = 8, 16, 48
        w, feat = make_data(G, Cg, Mg, torch.float16)
        T = a_idx.numel()
        o = torch.zeros(N_OUT, G, Mg, dtype=torch.float32, device=device)
        with dispatch_mode("force_fsg"):
            fired = mvmr_triton._try_grouped_dispatch(
                w, a_idx, feat, b_idx, o, o_idx, T, G, Mg, Cg)
        self.assertTrue(fired, "grouped mvmr did not fire at G=8 force_fsg")

        grad = torch.randn(N_OUT, G, Mg, device=device, dtype=torch.float16)
        o_w = torch.zeros(K_OFF, G, Mg, Cg, dtype=torch.float32, device=device)
        # vvor's grouped path keys on sorted o_idx; reuse a_idx (sorted) there.
        with dispatch_mode("force_fsg"):
            fired_v = vvor_triton._try_grouped_dispatch(
                grad, o_idx, feat, b_idx, o_w, a_idx, T, G, Mg, Cg, K_OFF)
        self.assertTrue(fired_v, "grouped vvor did not fire at G=8 force_fsg")

    # ── 6. pinned-config direct launches of the multi-group-per-program
    #      (BLOCK_SIZE_G > 1) branch — autotune may or may not pick those
    #      configs, so the branch is validated deterministically here,
    #      including a G NOT divisible by BLOCK_SIZE_G (g_ok masking) ────

    @staticmethod
    def _launch_mvmr_pinned(w, a_idx, feat, b_idx, o_idx, n_o,
                            L_CHUNK, BSG, BSM, BSC, num_warps=4):
        from sparse_engines.mvmr_triton_kernel import (
            sparse_matrix_vector_multiplication_reduction_grouped_kernel as K)
        K_off, G, C, M = w.shape
        o = torch.zeros(n_o, G, M, dtype=torch.float32, device=device)
        seg_offs = kernel_offset_segments(a_idx, K_off)
        seg_lens = seg_offs[1:] - seg_offs[:-1]
        total_chunks = int(((seg_lens + L_CHUNK - 1) // L_CHUNK).sum().item())
        grid = (total_chunks * triton.cdiv(G, BSG)
                * triton.cdiv(M, BSM) * triton.cdiv(C, BSC),)
        K.fn[grid](w, feat, b_idx, o, o_idx, seg_offs, K_off, G, M, C,
                   L_CHUNK=L_CHUNK, BLOCK_SIZE_G=BSG, BLOCK_SIZE_M=BSM,
                   BLOCK_SIZE_C=BSC, INPUT_PRECISION="ieee",
                   num_warps=num_warps)
        return o

    @staticmethod
    def _launch_vvor_pinned(grad, a_idx, feat, b_idx, o_idx, n_o,
                            L_CHUNK, BSG, BSM, BSC, num_warps=4):
        from sparse_engines.vvor_triton_kernel import (
            sparse_vector_vector_outer_product_reduction_grouped_kernel as K)
        G, M, C = grad.shape[1], grad.shape[2], feat.shape[2]
        o = torch.zeros(n_o, G, M, C, dtype=torch.float32, device=device)
        seg_offs = kernel_offset_segments(o_idx, n_o)
        seg_lens = seg_offs[1:] - seg_offs[:-1]
        total_chunks = int(((seg_lens + L_CHUNK - 1) // L_CHUNK).sum().item())
        grid = (total_chunks * triton.cdiv(G, BSG)
                * triton.cdiv(M, BSM) * triton.cdiv(C, BSC),)
        K.fn[grid](grad, a_idx, feat, b_idx, o, seg_offs, n_o, G, M, C,
                   L_CHUNK=L_CHUNK, BLOCK_SIZE_G=BSG, BLOCK_SIZE_M=BSM,
                   BLOCK_SIZE_C=BSC, INPUT_PRECISION="ieee",
                   num_warps=num_warps)
        return o

    def test_multigroup_block_mvmr_pinned(self):
        a_idx, b_idx, o_idx = make_sorted_indices()
        # (G, Cg, Mg, BSG, BSM, BSC) — includes G=3 with BSG=2 (g_ok mask)
        # and tile sizes not covering Cg/Mg exactly.
        for G, Cg, Mg, BSG, BSM, BSC in [
            (8, 16, 48, 8, 16, 16),
            (4, 24, 40, 4, 16, 16),
            (3, 16, 16, 2, 16, 16),
            (8,  8,  8, 4, 16, 16),
        ]:
            for dt_name, dtype, tol in [("fp32-ieee", torch.float32, 1e-5),
                                        ("fp16", torch.float16, 1e-2)]:
                with self.subTest(G=G, Cg=Cg, Mg=Mg, BSG=BSG, dtype=dt_name):
                    w, feat = make_data(G, Cg, Mg, dtype)
                    ref = oracle_mvmr(w, a_idx, feat, b_idx, o_idx, N_OUT)
                    out = self._launch_mvmr_pinned(
                        w, a_idx, feat, b_idx, o_idx, N_OUT,
                        L_CHUNK=64, BSG=BSG, BSM=BSM, BSC=BSC)
                    rel = rel_fro(out, ref)
                    print(f"  FSG mvmr-BSG{BSG} [G{G}_c{Cg}_m{Mg} {dt_name}]"
                          f" rel={rel:.3e}")
                    self.assertLess(rel, tol)

    def test_multigroup_block_vvor_pinned(self):
        T = T_DEFAULT
        gen = torch.Generator(device="cpu").manual_seed(5)
        o_idx = torch.randint(0, K_OFF, (T,), generator=gen).sort().values.to(device)
        a_idx = torch.randint(0, N_OUT, (T,), generator=gen).to(device)
        b_idx = torch.randint(0, N_FEAT, (T,), generator=gen).to(device)
        for G, Cg, Mg, BSG, BSM, BSC in [
            (8, 16, 48, 8, 16, 16),
            (4, 24, 40, 4, 16, 16),
            (3, 16, 16, 2, 16, 16),
        ]:
            for dt_name, dtype, tol in [("fp32-ieee", torch.float32, 1e-5),
                                        ("fp16", torch.float16, 1e-2)]:
                with self.subTest(G=G, Cg=Cg, Mg=Mg, BSG=BSG, dtype=dt_name):
                    torch.manual_seed(0)
                    grad = (torch.randn(N_OUT, G, Mg, device=device)
                            * (K_OFF * Cg) ** -0.5).to(dtype)
                    feat = torch.randn(N_FEAT, G, Cg, device=device).to(dtype)
                    ref = oracle_vvor(grad, a_idx, feat, b_idx, o_idx, K_OFF)
                    out = self._launch_vvor_pinned(
                        grad, a_idx, feat, b_idx, o_idx, K_OFF,
                        L_CHUNK=64, BSG=BSG, BSM=BSM, BSC=BSC)
                    rel = rel_fro(out, ref)
                    print(f"  FSG vvor-BSG{BSG} [G{G}_c{Cg}_m{Mg} {dt_name}]"
                          f" rel={rel:.3e}")
                    self.assertLess(rel, tol)

    def test_grouped_dispatch_refuses_unsorted(self):
        a_idx, b_idx, o_idx = make_sorted_indices()
        a_idx_unsorted = a_idx.flip(0)
        G, Cg, Mg = 4, 16, 48
        w, feat = make_data(G, Cg, Mg, torch.float16)
        T = a_idx.numel()
        o = torch.zeros(N_OUT, G, Mg, dtype=torch.float32, device=device)
        with dispatch_mode("force_fsg"):
            fired = mvmr_triton._try_grouped_dispatch(
                w, a_idx_unsorted, feat, b_idx, o, o_idx, T, G, Mg, Cg)
        self.assertFalse(fired, "grouped mvmr must refuse unsorted a_idx")


if __name__ == "__main__":
    unittest.main()
