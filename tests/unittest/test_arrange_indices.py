import unittest

import torch

from internals.indexing import arrange_indices as arrange_indices_cuda
from internals.indexing import arange_cached, cumsum_exclusive


def arrange_indices_torch(indices, num_indices=None, num_shifts=1, mask=None):
    device = indices.device

    if mask is not None:
        indices_of_indices = torch.nonzero(mask).squeeze(dim=-1)
        indices = indices[indices_of_indices]
    else:
        indices_of_indices = arange_cached(indices.shape[0], device=device)

    if num_shifts > 1:
        indices_of_indices = torch.div(
            indices_of_indices, num_shifts, rounding_mode="trunc"
        )

    if num_indices is None:
        num_indices = torch.max(indices).item() + 1

    sorter = torch.argsort(indices)
    bucket_slots = torch.empty_like(sorter, dtype=sorter.dtype, device=device)
    bucket_slots.index_copy_(0, sorter, arange_cached(sorter.shape[0], device=device))

    bucket_sizes = torch.bincount(indices, minlength=num_indices)
    bucket_splits = cumsum_exclusive(bucket_sizes)

    # indices_of_indices: [0, 1, 2, 3, 4, 5] --> indices_arranged: [0, 1, 3, 4, 2, 5]
    indices_arranged = torch.empty_like(
        indices_of_indices, dtype=indices_of_indices.dtype, device=device
    )
    indices_arranged.index_copy_(0, bucket_slots.to(torch.int64), indices_of_indices)

    return indices_arranged, bucket_sizes, bucket_splits


class TestArrangeIndices(unittest.TestCase):
    def setUp(self):
        seed = 9527
        torch.manual_seed(seed)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        num_indices = 1024 * 1024
        n = num_indices * 8

        self.indices = torch.randint(
            low=0, high=num_indices, size=(n,), device=self.device
        )

    def test_torch_vs_cuda(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # warm-up
        arrange_indices_cuda(self.indices)

        start.record()
        o_cuda = arrange_indices_cuda(self.indices)
        end.record()
        torch.cuda.synchronize()
        t_cuda = start.elapsed_time(end)
        print(f"cuda:\t {t_cuda:.4e}")

        # warm-up
        arrange_indices_torch(self.indices)

        start.record()
        o_torch = arrange_indices_torch(self.indices)
        end.record()
        torch.cuda.synchronize()
        t_torch = start.elapsed_time(end)
        print(f"torch:\t {t_torch:.4e}")

        self.assertTrue(torch.equal(self.indices[o_cuda[0]], self.indices[o_torch[0]]))
        self.assertTrue(torch.equal(o_cuda[1], o_torch[1]))
        self.assertTrue(torch.equal(o_cuda[2], o_torch[2]))


if __name__ == "__main__":
    unittest.main()
