"""The strided builder's TripletContract.k_sorted must be HONEST about sort_by.

`handle_stride_and_build_triplets` is called with sort_by="k" by every conv path
(→ k-sorted triplets → contract.k_sorted=True, the assume-sorted TIG invariant)
and with sort_by="i" by max_pool3d (→ i-sorted triplets). Stamping k_sorted=True
on the i-sorted pool rulebook would be a dishonest contract: a conv consuming it
would silently route the assume-k-sorted TIG path on unsorted-by-k data. The
pool feeds indexed_segment_reduce (never a conv) so it is latent today, but the
contract must still tell the truth — k_sorted == (sort_by == "k").
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import unittest
import torch

from layers.metadata import MetaData
from layers.triplets import handle_stride_and_build_triplets


def _grid_meta(g=0.04, dev="cuda", seed=0):
    torch.manual_seed(seed)
    vox = torch.unique(torch.randint(0, 30, (2000, 3), device=dev), dim=0)
    pts = (vox.float() + 0.5) * g
    si = torch.zeros(pts.shape[0], dtype=torch.long, device=dev)
    return MetaData(points=pts, sample_inds=si,
                    sample_sizes=torch.bincount(si), grid_size=g,
                    kernel_size=None, auto_build_triplets=False)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestContractHonestKSorted(unittest.TestCase):
    def test_conv_sort_k_is_ksorted(self):
        m = handle_stride_and_build_triplets(
            _grid_meta(), stride=2, kernel_size=3, sort_by="k")
        self.assertTrue(m.contract.k_sorted)
        # and the data really is k-sorted
        self.assertTrue(bool((m.k[1:] >= m.k[:-1]).all()))

    def test_pool_sort_i_is_not_ksorted(self):
        m = handle_stride_and_build_triplets(
            _grid_meta(), stride=2, kernel_size=3, sort_by="i")
        self.assertFalse(
            m.contract.k_sorted,
            "i-sorted pool rulebook must NOT declare k_sorted=True")


if __name__ == "__main__":
    unittest.main()
