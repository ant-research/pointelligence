"""Engine routing must not change fp32 numerics.

The per-triplet engine computes fp32 eagerly (IEEE exact); the tl.dot
engines historically defaulted ``input_precision="tf32"``, so the
dispatcher's engine choice silently swapped IEEE fp32 for tf32 (10-bit
mantissa — integer-valued features land on a power-of-two grid; observed
as a fan-in-1 deconv corrupting voxel keys once auto-detection routed it
off the per-triplet fallback).

Contract pinned here: fp32 defaults to IEEE everywhere, matching
``torch.matmul``'s convention. tf32 is opt-in via the standard global
knob (``torch.backends.cuda.matmul.allow_tf32``) or a scoped
``precision_mode("tf32")``. Half-precision paths are untouched.
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch

from layers.conv import PointConv3d
from sparse_engines._dispatch_override import (
    current_precision, dispatch_mode, precision_mode)


def _fanin1_rulebook(n_out, n_in, K, device, seed=0):
    """Exactly-once output cover (each output row written once), k-sorted."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randperm(n_out, generator=g).to(device).int()
    j = torch.randint(0, n_in, (n_out,), generator=g).to(device).int()
    k = torch.randint(0, K, (n_out,), generator=g).to(device).int()
    order = k.argsort()
    return i[order].contiguous(), j[order].contiguous(), k[order].contiguous()


def _int_keys(n, device, seed=1):
    """Integer-valued fp32 features ~[2^22, 2^23): exact in fp32, NOT
    representable in tf32 (10-bit mantissa => multiples of 4096 only)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(4_200_000, 8_300_000, (n, 1), generator=g).to(device).float()


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestFp32RoutingPrecision(unittest.TestCase):

    def _identity_conv(self, K_side):
        conv = PointConv3d(1, 1, kernel_size=K_side, bias=False).cuda()
        with torch.no_grad():
            conv.weight.fill_(1.0)
        return conv

    def test_deconv_auto_matches_pt_exactly(self):
        """Fan-in-1 deconv (the auto-detected exactly-once route): under
        default settings, auto must be bit-exact vs the per-triplet
        reference on integer keys."""
        n_out, n_in = 20000, 2500
        i, j, k = _fanin1_rulebook(n_out, n_in, 8, "cuda")
        feat = _int_keys(n_in, "cuda")
        conv = self._identity_conv(2)
        with dispatch_mode("force_pt"):
            ref = conv(feat, i, j, k, n_out)
        with dispatch_mode("auto"):
            out = conv(feat, i, j, k, n_out)
        self.assertTrue(torch.equal(out, ref),
                        "auto-routed fp32 deconv must match the per-triplet "
                        "engine exactly (no silent tf32)")

    def test_forced_engines_match_pt_exactly(self):
        """Single-nonzero-slot submanifold conv: every forced tl.dot engine
        must also be IEEE-exact at fp32 by default."""
        n = 8000
        T = 40000
        g = torch.Generator(device="cpu").manual_seed(2)
        i = torch.randint(0, n, (T,), generator=g).cuda().int()
        j = torch.randint(0, n, (T,), generator=g).cuda().int()
        k = torch.randint(0, 27, (T,), generator=g).cuda().int()
        order = k.argsort()
        i, j, k = i[order].contiguous(), j[order].contiguous(), k[order].contiguous()
        feat = _int_keys(n, "cuda", seed=3).repeat(1, 64)  # C=64 (grouped floor)
        conv = PointConv3d(64, 64, kernel_size=3, bias=False).cuda()
        with torch.no_grad():
            conv.weight.zero_()
            # slot 13 = identity matrix: output rows are sums of gathered
            # integers; keep fan-in sums < 2^24 by masking to one slot.
            conv.weight[13] = torch.eye(64, device="cuda")
        with dispatch_mode("force_pt"):
            ref = conv(feat, i, j, k, n)
        for mode in ("force_fsg", "force_tig", "auto"):
            with dispatch_mode(mode):
                out = conv(feat, i, j, k, n)
            self.assertTrue(torch.equal(out, ref),
                            f"{mode}: fp32 default must be IEEE-exact vs PT")

    def test_default_follows_torch_global_knob(self):
        """'default' precision resolves from torch's standard tf32 opt-in."""
        prior = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            self.assertEqual(current_precision(), "ieee")
            torch.backends.cuda.matmul.allow_tf32 = True
            self.assertEqual(current_precision(), "tf32")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prior

    def test_scoped_tf32_opt_in(self):
        """precision_mode('tf32') forces tf32 regardless of the global knob."""
        prior = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with precision_mode("tf32"):
                self.assertEqual(current_precision(), "tf32")
            with precision_mode("ieee"):
                self.assertEqual(current_precision(), "ieee")
            self.assertEqual(current_precision(), "ieee")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prior


if __name__ == "__main__":
    unittest.main()
