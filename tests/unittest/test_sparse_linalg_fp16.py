"""fp16 correctness for MVMR / VVOR kernels — forward AND backward.

Compares fp16 Triton output against an fp32-numpy reference. Tolerance is
set so we catch the per-block accumulation drift that the old fp16-accum
path suffered from (~9 bits lost when reducing 128 elements at C=512),
without flagging legitimate fp16 input rounding.

Coverage:
- Forward MVMR / VVOR at a deep-stage shape (M=256, C=256, T=50k) and a
  shallow-stage shape (M=64, C=64, T=10k). The deep shape exercises the
  C-axis reduction width that exposed the old precision bug; the shallow
  shape catches regressions on the lower-cost code paths and is closer
  to enc0/enc1 in the per-stage bench.
- Backward of MVMR via PyTorch autograd. Backward dW (weight grad) is
  computed by autograd routing through VVOR; backward dB (input grad)
  via the MVMR-with-transposed-weight path. Both are then compared
  cell-by-cell against fp32-numpy references derived from the forward
  formula.
- Backward of VVOR via PyTorch autograd. Routes through MVMR-transposed
  internally for both grad_a and grad_b.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import numpy as np
import torch
import sparse_engines


# ─────────────────────────────────────────────────────────────────────────────
# fp32-numpy references — both forward and backward forms.
# ─────────────────────────────────────────────────────────────────────────────

def mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, n):
    """Forward MVMR: o[o_idx] += a[a_idx] @ b[b_idx] (per-group)."""
    a = a.detach().float().cpu().numpy()
    b = b.detach().float().cpu().numpy()
    ai = a_idx.cpu().numpy()
    bi = b_idx.cpu().numpy()
    oi = o_idx.cpu().numpy()
    G, M = a.shape[1], a.shape[3]
    o = np.zeros((n, G, M), dtype=np.float32)
    for t in range(ai.shape[0]):
        o[oi[t]] += np.einsum("gc,gcm->gm", b[bi[t]], a[ai[t]])
    return o


def vvor_numpy_ref(a, a_idx, b, b_idx, o_idx, n):
    """Forward VVOR: o[o_idx] += a[a_idx] (outer) b[b_idx] (per-group)."""
    a = a.detach().float().cpu().numpy()
    b = b.detach().float().cpu().numpy()
    ai = a_idx.cpu().numpy()
    bi = b_idx.cpu().numpy()
    oi = o_idx.cpu().numpy()
    G, M = a.shape[1], a.shape[2]
    C = b.shape[2]
    o = np.zeros((n, G, M, C), dtype=np.float32)
    for t in range(ai.shape[0]):
        o[oi[t]] += np.einsum("gm,gc->gmc", a[ai[t]], b[bi[t]])
    return o


def mvmr_grad_a_numpy_ref(a_shape, a_idx, b, b_idx, o_idx, grad_o):
    """∂L/∂a[k, g, c, m] from forward MVMR.

    grad_a[a_idx[t]] += b[b_idx[t]] (outer) grad_o[o_idx[t]]
    Equivalent to a forward VVOR with (a=b, b=grad_o, o_idx=a_idx).
    """
    b = b.detach().float().cpu().numpy()
    grad_o = grad_o.detach().float().cpu().numpy()
    ai = a_idx.cpu().numpy()
    bi = b_idx.cpu().numpy()
    oi = o_idx.cpu().numpy()
    K, G, C, M = a_shape
    grad_a = np.zeros((K, G, C, M), dtype=np.float32)
    for t in range(ai.shape[0]):
        grad_a[ai[t]] += np.einsum("gc,gm->gcm", b[bi[t]], grad_o[oi[t]])
    return grad_a


def mvmr_grad_b_numpy_ref(a, a_idx, b_shape, b_idx, o_idx, grad_o):
    """∂L/∂b[j, g, c] from forward MVMR.

    grad_b[b_idx[t]] += a[a_idx[t]] @ grad_o[o_idx[t]]^T
    Equivalent to a forward MVMR with weight transposed and o_idx ↔ b_idx.
    """
    a = a.detach().float().cpu().numpy()
    grad_o = grad_o.detach().float().cpu().numpy()
    ai = a_idx.cpu().numpy()
    bi = b_idx.cpu().numpy()
    oi = o_idx.cpu().numpy()
    N_b, G, C = b_shape
    grad_b = np.zeros((N_b, G, C), dtype=np.float32)
    for t in range(ai.shape[0]):
        grad_b[bi[t]] += np.einsum("gcm,gm->gc", a[ai[t]], grad_o[oi[t]])
    return grad_b


def vvor_grad_a_numpy_ref(a_shape, a_idx, b, b_idx, o_idx, grad_o):
    """∂L/∂a[i, g, m] from forward VVOR.

    Forward: o[o_idx, g, m, c] += a[a_idx, g, m] * b[b_idx, g, c]
    grad_a[a_idx[t], g, m] += sum_c(grad_o[o_idx[t], g, m, c] * b[b_idx[t], g, c])
    """
    b = b.detach().float().cpu().numpy()
    grad_o = grad_o.detach().float().cpu().numpy()
    ai = a_idx.cpu().numpy()
    bi = b_idx.cpu().numpy()
    oi = o_idx.cpu().numpy()
    N_a, G, M = a_shape
    grad_a = np.zeros((N_a, G, M), dtype=np.float32)
    for t in range(ai.shape[0]):
        grad_a[ai[t]] += np.einsum("gmc,gc->gm", grad_o[oi[t]], b[bi[t]])
    return grad_a


def vvor_grad_b_numpy_ref(a, a_idx, b_shape, b_idx, o_idx, grad_o):
    """∂L/∂b[j, g, c] from forward VVOR.

    grad_b[b_idx[t], g, c] += sum_m(grad_o[o_idx[t], g, m, c] * a[a_idx[t], g, m])
    """
    a = a.detach().float().cpu().numpy()
    grad_o = grad_o.detach().float().cpu().numpy()
    ai = a_idx.cpu().numpy()
    bi = b_idx.cpu().numpy()
    oi = o_idx.cpu().numpy()
    N_b, G, C = b_shape
    grad_b = np.zeros((N_b, G, C), dtype=np.float32)
    for t in range(ai.shape[0]):
        grad_b[bi[t]] += np.einsum("gmc,gm->gc", grad_o[oi[t]], a[ai[t]])
    return grad_b


# ─────────────────────────────────────────────────────────────────────────────
# Test class — parametrised over (deep, shallow) shapes via subclassing.
# ─────────────────────────────────────────────────────────────────────────────


class _MVMRVVORFp16Mixin:
    """Mixin holding parametrised tests. Concrete subclasses set the shape."""

    # Shape parameters (override in subclass).
    N_a = 27           # number of kernel offsets
    N_b = 8000
    N_o = 8000
    G = 1
    M = 256
    C = 256
    T = 50_000

    REL_TOL = 5e-3     # for fp16 inputs, ~1e-3 input rounding + tf32 mma

    def _device(self):
        return "cuda"

    def _make_indices(self, seed=1):
        device = self._device()
        torch.manual_seed(seed)
        a_idx = torch.randint(0, self.N_a, (self.T,), device=device, dtype=torch.int64)
        b_idx = torch.randint(0, self.N_b, (self.T,), device=device, dtype=torch.int64)
        o_idx = torch.randint(0, self.N_o, (self.T,), device=device, dtype=torch.int64)
        order = torch.argsort(a_idx, stable=True)         # sort_by="k"
        return a_idx[order], b_idx[order], o_idx[order]

    def _rel_err(self, triton_out, ref_np):
        ref = torch.tensor(ref_np, device=self._device())
        max_abs = (triton_out.float() - ref).abs().max().item()
        max_val = ref.abs().max().item()
        return max_abs / max(max_val, 1e-6), max_abs, max_val

    # ── Forward correctness ──

    def test_mvmr_fwd(self):
        device = self._device()
        torch.manual_seed(0)
        a = (torch.randn(self.N_a, self.G, self.C, self.M, device=device,
                          dtype=torch.float32) * 0.1).half()
        b = (torch.randn(self.N_b, self.G, self.C, device=device,
                          dtype=torch.float32) * 0.1).half()
        a_idx, b_idx, o_idx = self._make_indices()

        out_triton = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
            a, a_idx, b, b_idx, o_idx, self.N_o,
        )
        out_ref = mvmr_numpy_ref(a, a_idx, b, b_idx, o_idx, self.N_o)
        rel, max_abs, max_val = self._rel_err(out_triton, out_ref)
        print(f"\n  [{self.__class__.__name__}] MVMR fwd  rel={rel:.3e} "
              f"(max_abs={max_abs:.3e}, max_val={max_val:.3e})")
        self.assertLess(rel, self.REL_TOL)

    def test_vvor_fwd(self):
        device = self._device()
        torch.manual_seed(0)
        a = (torch.randn(self.N_a, self.G, self.M, device=device,
                          dtype=torch.float32) * 0.1).half()
        b = (torch.randn(self.N_b, self.G, self.C, device=device,
                          dtype=torch.float32) * 0.1).half()
        a_idx, b_idx, o_idx = self._make_indices()

        out_triton = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
            a, a_idx, b, b_idx, o_idx, self.N_o,
        )
        out_ref = vvor_numpy_ref(a, a_idx, b, b_idx, o_idx, self.N_o)
        rel, max_abs, max_val = self._rel_err(out_triton, out_ref)
        print(f"\n  [{self.__class__.__name__}] VVOR fwd  rel={rel:.3e} "
              f"(max_abs={max_abs:.3e}, max_val={max_val:.3e})")
        self.assertLess(rel, self.REL_TOL)

    # ── Backward correctness via PyTorch autograd ──

    def test_mvmr_bwd(self):
        device = self._device()
        torch.manual_seed(0)
        a = (torch.randn(self.N_a, self.G, self.C, self.M, device=device,
                          dtype=torch.float32) * 0.1).half().requires_grad_(True)
        b = (torch.randn(self.N_b, self.G, self.C, device=device,
                          dtype=torch.float32) * 0.1).half().requires_grad_(True)
        a_idx, b_idx, o_idx = self._make_indices()

        out_triton = sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
            a, a_idx, b, b_idx, o_idx, self.N_o,
        )
        # Forward-output buffer is fp32; grad_o must match. We synthesise it
        # in fp16 for realism then upcast (mirrors training where loss.backward()
        # produces fp16 grads in autocast contexts).
        grad_o = (torch.randn_like(out_triton).float() * 0.1).to(out_triton.dtype)
        out_triton.backward(grad_o)

        # Reference grads (fp32 numpy).
        grad_a_ref = mvmr_grad_a_numpy_ref(
            (self.N_a, self.G, self.C, self.M), a_idx, b, b_idx, o_idx, grad_o,
        )
        grad_b_ref = mvmr_grad_b_numpy_ref(
            a, a_idx, (self.N_b, self.G, self.C), b_idx, o_idx, grad_o,
        )
        rel_a, _, _ = self._rel_err(a.grad, grad_a_ref)
        rel_b, _, _ = self._rel_err(b.grad, grad_b_ref)
        print(f"\n  [{self.__class__.__name__}] MVMR bwd  "
              f"grad_a rel={rel_a:.3e}, grad_b rel={rel_b:.3e}")
        self.assertLess(rel_a, self.REL_TOL, "MVMR grad_a rel err too high")
        self.assertLess(rel_b, self.REL_TOL, "MVMR grad_b rel err too high")

    def test_vvor_bwd(self):
        device = self._device()
        torch.manual_seed(0)
        a = (torch.randn(self.N_a, self.G, self.M, device=device,
                          dtype=torch.float32) * 0.1).half().requires_grad_(True)
        b = (torch.randn(self.N_b, self.G, self.C, device=device,
                          dtype=torch.float32) * 0.1).half().requires_grad_(True)
        a_idx, b_idx, o_idx = self._make_indices()

        out_triton = sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
            a, a_idx, b, b_idx, o_idx, self.N_o,
        )
        grad_o = (torch.randn_like(out_triton).float() * 0.1).to(out_triton.dtype)
        out_triton.backward(grad_o)

        grad_a_ref = vvor_grad_a_numpy_ref(
            (self.N_a, self.G, self.M), a_idx, b, b_idx, o_idx, grad_o,
        )
        grad_b_ref = vvor_grad_b_numpy_ref(
            a, a_idx, (self.N_b, self.G, self.C), b_idx, o_idx, grad_o,
        )
        rel_a, _, _ = self._rel_err(a.grad, grad_a_ref)
        rel_b, _, _ = self._rel_err(b.grad, grad_b_ref)
        print(f"\n  [{self.__class__.__name__}] VVOR bwd  "
              f"grad_a rel={rel_a:.3e}, grad_b rel={rel_b:.3e}")
        self.assertLess(rel_a, self.REL_TOL, "VVOR grad_a rel err too high")
        self.assertLess(rel_b, self.REL_TOL, "VVOR grad_b rel err too high")


class TestSparseLinAlgFp16Deep(_MVMRVVORFp16Mixin, unittest.TestCase):
    """Deep-stage shape (M=256, C=256, T=50k) — exercises the C-axis
    reduction width where the old fp16-accum lost precision."""
    N_a, N_b, N_o = 27, 8000, 8000
    G, M, C, T = 1, 256, 256, 50_000


class TestSparseLinAlgFp16Shallow(_MVMRVVORFp16Mixin, unittest.TestCase):
    """Shallow-stage shape (M=64, C=64, T=10k) — closer to enc0/enc1 of the
    PTv3 stage profile. Catches regressions on the lower-cost paths where
    deep-stage tests don't trigger."""
    N_a, N_b, N_o = 27, 4000, 4000
    G, M, C, T = 1, 64, 64, 10_000


if __name__ == "__main__":
    unittest.main()
