import unittest

import torch

from sparse_engines.ops import indexed_distance as indexed_distance_triton
from sparse_engines_cuda.ops import indexed_distance as indexed_distance_cuda
from unittest_utils import check_all_close


def indexed_distance_torch(a, a_idx, b, b_idx):
    return torch.nn.functional.pairwise_distance(
        a[a_idx, :],
        b[b_idx, :],
        p=2.0,
        keepdim=False,
    )


class TestIndexedDistance(unittest.TestCase):
    def setUp(self):
        seed = 9527
        torch.manual_seed(seed)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        channels = 3
        length_a, length_b = 1024, 3721
        n = 1024 * 1024 * 64

        self.a = torch.randn(length_a, channels).to(self.device)
        self.b = torch.randn(length_b, channels).to(self.device)
        self.a_idx = torch.randint(low=0, high=length_a, size=(n,), device=self.device)
        self.b_idx = torch.randint(low=0, high=length_b, size=(n,), device=self.device)

    def test_torch_vs_triton_vs_native(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # warm-up run
        indexed_distance_triton(self.a, self.a_idx, self.b, self.b_idx)

        start.record()
        o_triton = indexed_distance_triton(self.a, self.a_idx, self.b, self.b_idx)
        end.record()
        torch.cuda.synchronize()
        t_triton = start.elapsed_time(end)
        print(f"triton:\t {t_triton:.4e}")

        # warm-up run
        indexed_distance_torch(self.a, self.a_idx, self.b, self.b_idx)

        start.record()
        o_torch = indexed_distance_torch(self.a, self.a_idx, self.b, self.b_idx)
        end.record()
        torch.cuda.synchronize()
        t_torch = start.elapsed_time(end)
        print(f"torch:\t {t_torch:.4e}")

        # warm-up run
        indexed_distance_cuda(self.a, self.a_idx, self.b, self.b_idx)

        start.record()
        o_cuda = indexed_distance_cuda(self.a, self.a_idx, self.b, self.b_idx)
        end.record()
        torch.cuda.synchronize()
        t_cuda = start.elapsed_time(end)
        print(f"cuda:\t {t_cuda:.4e}")

        self.assertTrue(check_all_close(o_torch, o_triton, "o", 1e-2))
        self.assertTrue(check_all_close(o_torch, o_cuda, "o", 1e-2))


if __name__ == "__main__":
    unittest.main()
