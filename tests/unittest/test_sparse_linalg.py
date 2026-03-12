import unittest

import torch
import numpy as np
import sparse_engines
import sparse_engines_cuda

from internals.indexing import cumsum_exclusive, repeat_interleave_indices
from unittest_utils import check_all_close


def sparse_matrix_vector_multiplication_reduction_numpy(a, a_idx, b, b_idx, o_idx, n):
    device = a.device
    a, a_idx, b, b_idx, o_idx = [
        x.detach().cpu().numpy() for x in (a, a_idx, b, b_idx, o_idx)
    ]

    k, g, c, m = a_idx.shape[0], a.shape[1], a.shape[2], a.shape[3]
    assert a_idx.shape[0] == b_idx.shape[0]
    assert a_idx.shape[0] == o_idx.shape[0]
    assert a.shape[1] == b.shape[1]
    assert a.shape[2] == b.shape[2]

    o = np.zeros(shape=(n, g, m), dtype=np.float32)
    for i in range(k):
        ai = a[a_idx[i], ...]
        bi = b[b_idx[i], ...]
        o[o_idx[i], :] += np.matmul(np.expand_dims(bi, -2), ai).squeeze(-2)

    return torch.tensor(o, dtype=torch.float32, device=device, requires_grad=False)


def sparse_vector_vector_outer_product_reduction_numpy(a, a_idx, b, b_idx, o_idx, n):
    device = a.device
    a, a_idx, b, b_idx, o_idx = [
        x.detach().cpu().numpy() for x in (a, a_idx, b, b_idx, o_idx)
    ]

    assert a_idx.shape[0] == b_idx.shape[0]
    assert a_idx.shape[0] == o_idx.shape[0]
    assert a.shape[1] == b.shape[1]

    k, g, m, c = a_idx.shape[0], a.shape[1], a.shape[-1], b.shape[-1]

    o = np.zeros(shape=(n, g, m, c), dtype=np.float32)
    for i in range(k):
        ai = a[a_idx[i], ...]
        bi = b[b_idx[i], ...]
        o[o_idx[i], ...] += np.matmul(np.expand_dims(ai, -1), np.expand_dims(bi, -2))

    return torch.tensor(o, dtype=torch.float32, device=device, requires_grad=False)


class TestSparseLinAlg(unittest.TestCase):
    def setUp(self):
        seed = 9527
        torch.manual_seed(seed)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        low, high = 1, 32
        N_a, N_b, N_o, G, M, C = 5**3, 512, 1024, 1, 256, 128
        assert C % G == 0

        lengths_o = torch.randint(
            low=low, high=high + 1, size=(N_o,), device=self.device
        )
        lengths_cumsum_o, lengths_sum = cumsum_exclusive(lengths_o, return_sum=True)
        self.o_idx_o = repeat_interleave_indices(
            repeats=lengths_o,
            output_size=lengths_sum,
            may_contain_zero_repeats=False,
        ).to(torch.int32)

        self.a = torch.randn(
            (N_a, G, C, M), device=self.device, dtype=torch.float32, requires_grad=True
        )
        self.a_idx_o = torch.randint(
            low=0, high=N_a, size=(lengths_sum,), device=self.device, dtype=torch.int32
        )

        self.b = torch.randn(
            (N_b, G, C), device=self.device, dtype=torch.float32, requires_grad=True
        )
        self.b_idx_o = torch.randint(
            low=0, high=N_b, size=(lengths_sum,), device=self.device, dtype=torch.int32
        )

        self.do = torch.randn((N_o, G, M), device=self.device, dtype=torch.float32)

        self.N_a, self.N_b, self.N_o = N_a, N_b, N_o
        self.sort_by = "a"

    def impl(
        self,
        sparse_matrix_vector_multiplication_reduction_func,
        sparse_vector_vector_outer_product_reduction_func,
    ):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        if "o" == self.sort_by:
            a_idx = self.a_idx_o
            b_idx = self.b_idx_o
            o_idx = self.o_idx_o
        elif "a" == self.sort_by:
            a_idx, sorter = torch.sort(self.a_idx_o)
            b_idx = self.b_idx_o[sorter]
            o_idx = self.o_idx_o[sorter]
        else:
            assert "b" == self.sort_by
            b_idx, sorter = torch.sort(self.b_idx_o)
            a_idx = self.a_idx_o[sorter]
            o_idx = self.o_idx_o[sorter]
        end.record()
        torch.cuda.synchronize()
        t_sort = start.elapsed_time(end)

        start.record()
        o = sparse_matrix_vector_multiplication_reduction_func(
            self.a, a_idx, self.b, b_idx, o_idx, self.N_o
        )
        end.record()
        torch.cuda.synchronize()
        t_forward = start.elapsed_time(end)

        start.record()
        da = sparse_vector_vector_outer_product_reduction_func(
            self.b, b_idx, self.do, o_idx, a_idx, self.N_a
        )
        end.record()
        torch.cuda.synchronize()
        t_backward_da = start.elapsed_time(end)

        start.record()
        db = sparse_matrix_vector_multiplication_reduction_func(
            self.a.transpose(2, 3), a_idx, self.do, o_idx, b_idx, self.N_b
        )
        end.record()
        torch.cuda.synchronize()
        t_backward_db = start.elapsed_time(end)

        if o.requires_grad:
            o.backward(self.do)
            self.assertTrue(check_all_close(self.a.grad, da, "da", 1e-2, mute=True))
            self.assertTrue(check_all_close(self.b.grad, db, "db", 1e-2, mute=True))
            self.a.grad.zero_()
            self.b.grad.zero_()

        return t_sort, t_forward, t_backward_da, t_backward_db, o, da, db

    def impl_numpy(self):
        return self.impl(
            sparse_matrix_vector_multiplication_reduction_numpy,
            sparse_vector_vector_outer_product_reduction_numpy,
        )

    def impl_native(self):
        return self.impl(
            sparse_engines_cuda.ops.sparse_matrix_vector_multiplication_reduction,
            sparse_engines_cuda.ops.sparse_vector_vector_outer_product_reduction,
        )

    def impl_triton(self):

        return self.impl(
            sparse_engines.ops.sparse_matrix_vector_multiplication_reduction,
            sparse_engines.ops.sparse_vector_vector_outer_product_reduction,
        )

    def test_native_vs_triton(self):
        _, _, _, _, o, da, db = self.impl_numpy()

        # to make sure the triton kernels have been compiled
        _ = self.impl_triton()

        for sort_by in ["a", "o", "b"]:
            self.sort_by = sort_by

            print(f"Using sort by {sort_by}...")

            t_sort, t_forward, t_da, t_db, o_native, da_native, db_native = (
                self.impl_native()
            )
            print(
                f"native:\tt_sort: {t_sort:.4e}, t_forward: {t_forward:.4e} t_da: {t_da:.4e}  t_db: {t_db:.4e}"
            )
            t_sort, t_forward, t_da, t_db, o_triton, da_triton, db_triton = (
                self.impl_triton()
            )
            print(
                f"triton:\tt_sort: {t_sort:.4e}, t_forward: {t_forward:.4e} t_da: {t_da:.4e}  t_db: {t_db:.4e}"
            )

            self.assertTrue(check_all_close(o_native, o, "o", 1e-2))
            self.assertTrue(check_all_close(o_triton, o, "o", 1e-2))

            self.assertTrue(check_all_close(da_native, da, "da", 1e-2))
            self.assertTrue(check_all_close(da_triton, da, "da", 1e-2))

            self.assertTrue(check_all_close(db_native, db, "db", 1e-2))
            self.assertTrue(check_all_close(db_triton, db, "db", 1e-2))
            print()


if __name__ == "__main__":
    unittest.main()
