import math
import unittest
from functools import partial


import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa_torch
from sparse_engines_cuda.ops import sparse_scaled_dot_product_attention as sdpa_native
from sparse_engines.sdpa_keops import sparse_scaled_dot_product_attention as sdpa_keops
from sparse_engines.ops import (
    sparse_scaled_dot_product_attention_forward as sdpa_triton_forward,
)

from unittest_utils import check_all_close


def _attn_mask(q_x, k_x, d):
    return (q_x * k_x).sum(-1).abs() < d


class TestSparseAttentions(unittest.TestCase):
    def setUp(self):
        seed = 9527
        torch.manual_seed(seed)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        N, H, L, S = 4, 10, 16 * 64 + 15, 64 * 64 + 7
        E, Ev, Ex = 64, 64, 4

        self.q = torch.nn.Parameter(torch.randn((N, H, L, E), device=self.device))
        self.k = torch.nn.Parameter(torch.randn((N, H, S, E), device=self.device))
        self.v = torch.nn.Parameter(
            torch.randn((N, H, S, Ev), device=self.device).clamp(-1, 1)
        )
        self.d_o = torch.randn((N, H, L, Ev), device=self.device)

        self.q_x = torch.randn((N, 1, L, Ex), device=self.device)
        self.k_x = torch.randn((N, 1, S, Ex), device=self.device)

        # to make sure the mask is not effected by the float computation errors
        self.distance = 1000
        scale = 100
        self.q_x = (self.q_x * scale).floor_()
        self.k_x = (self.k_x * scale).floor_()

        attn_mask = _attn_mask(
            self.q_x[:, :, :, None, :], self.k_x[:, :, None, :, :], d=self.distance
        )
        ratio = torch.nonzero(attn_mask).shape[0] / attn_mask.numel()
        print(f"sparsity: {ratio * 100}%.")

    def impl_keops(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        attn_mask_generator = (
            self.q_x,
            self.k_x,
            partial(_attn_mask, d=self.distance),
        )
        o = sdpa_keops(
            self.q,
            self.k,
            self.v,
            attn_mask_generator=attn_mask_generator,
        )
        end.record()
        torch.cuda.synchronize()
        t_forward = start.elapsed_time(end)

        start.record()
        o.backward(gradient=self.d_o, inputs=(self.q, self.k, self.v))
        end.record()
        torch.cuda.synchronize()
        t_backward = start.elapsed_time(end)
        dq, dk, dv = (
            self.q.grad.clone(),
            self.k.grad.clone(),
            self.v.grad.clone(),
        )
        self.q.grad.zero_()
        self.k.grad.zero_()
        self.v.grad.zero_()

        return 0, t_forward, t_backward, o, dq, dk, dv

    def impl_torch(self, neg_inf=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        attn_mask = _attn_mask(
            self.q_x[:, :, :, None, :],
            self.k_x[:, :, None, :, :],
            d=self.distance,
        )
        if neg_inf is not None:
            attn_mask = attn_mask.to(torch.float32)
            attn_mask = (1 - attn_mask) * neg_inf
        end.record()
        torch.cuda.synchronize()
        t_attn_mask = start.elapsed_time(end)

        start.record()
        o = sdpa_torch(
            self.q,
            self.k,
            self.v,
            attn_mask=attn_mask,
        )
        end.record()
        torch.cuda.synchronize()
        t_forward = start.elapsed_time(end)

        start.record()
        o.backward(gradient=self.d_o, inputs=(self.q, self.k, self.v))
        end.record()
        torch.cuda.synchronize()
        t_backward = start.elapsed_time(end)
        dq, dk, dv = (
            self.q.grad.clone(),
            self.k.grad.clone(),
            self.v.grad.clone(),
        )
        self.q.grad.zero_()
        self.k.grad.zero_()
        self.v.grad.zero_()
        return t_attn_mask, t_forward, t_backward, o, dq, dk, dv

    def impl_native(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        attn_mask = _attn_mask(
            self.q_x[:, :, :, None, :],
            self.k_x[:, :, None, :, :],
            d=self.distance,
        )
        N, L, S = self.q_x.shape[0], self.q_x.shape[2], self.k_x.shape[2]
        q_idx = torch.arange(N * L, dtype=torch.int32, device=self.q_x.device)
        nonzero_idx = torch.nonzero(attn_mask.squeeze(1)).to(torch.int32)
        k_idx = nonzero_idx[:, 0] * S + nonzero_idx[:, 2]
        k_length = (
            attn_mask.flatten(0, 2).to(torch.int32).sum(dim=-1, dtype=torch.int32)
        )
        k_cumsum = torch.cumsum(k_length, dim=0, dtype=torch.int32)
        end.record()
        torch.cuda.synchronize()
        t_attn_mask = start.elapsed_time(end)

        start.record()
        o = sdpa_native(
            self.q,
            self.k,
            self.v,
            q_idx,
            k_idx,
            k_cumsum,
            1 / math.sqrt(self.q.shape[-1]),
        )
        end.record()
        torch.cuda.synchronize()
        t_forward = start.elapsed_time(end)

        start.record()
        o.backward(gradient=self.d_o, inputs=(self.q, self.k, self.v))
        end.record()
        torch.cuda.synchronize()
        t_backward = start.elapsed_time(end)
        dq, dk, dv = (
            self.q.grad.clone(),
            self.k.grad.clone(),
            self.v.grad.clone(),
        )
        self.q.grad.zero_()
        self.k.grad.zero_()
        self.v.grad.zero_()
        return t_attn_mask, t_forward, t_backward, o, dq, dk, dv

    def impl_triton(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        attn_mask = _attn_mask(
            self.q_x[:, :, :, None, :],
            self.k_x[:, :, None, :, :],
            d=self.distance,
        )
        N, L, S = self.q_x.shape[0], self.q_x.shape[2], self.k_x.shape[2]
        q_idx = torch.arange(N * L, dtype=torch.int32, device=self.q_x.device)
        nonzero_idx = torch.nonzero(attn_mask.squeeze(1)).to(torch.int32)
        k_idx = nonzero_idx[:, 0] * S + nonzero_idx[:, 2]
        k_length = (
            attn_mask.flatten(0, 2).to(torch.int32).sum(dim=-1, dtype=torch.int32)
        )
        k_cumsum = torch.cumsum(k_length, dim=0, dtype=torch.int32)
        end.record()
        torch.cuda.synchronize()
        t_attn_mask = start.elapsed_time(end)

        start.record()
        o, _ = sdpa_triton_forward(
            self.q,
            self.k,
            self.v,
            q_idx,
            k_idx,
            k_cumsum,
            1 / math.sqrt(self.q.shape[-1]),
        )
        end.record()
        torch.cuda.synchronize()
        t_forward = start.elapsed_time(end)
        start.record()
        end.record()
        torch.cuda.synchronize()
        t_backward = start.elapsed_time(end)
        dq, dk, dv = None, None, None
        return t_attn_mask, t_forward, t_backward, o, dq, dk, dv

    def test_keops_vs_native_vs_torch(self):
        t_attn_mask, t_forward, t_backward, o_keops, dq_keops, dk_keops, dv_keops = (
            self.impl_keops()
        )
        print(
            f"keops:\tt_attn_mask: {t_attn_mask:.4e} t_forward: {t_forward:.4e} t_backward: {t_backward:.4e}"
        )

        neg_inf = -1e38  # -float('inf')
        t_attn_mask, t_forward, t_backward, o_torch, dq_torch, dk_torch, dv_torch = (
            self.impl_torch(neg_inf=neg_inf)
        )
        print(
            f"torch:\tt_attn_mask: {t_attn_mask:.4e} t_forward: {t_forward:.4e} t_backward: {t_backward:.4e}"
        )

        self.assertTrue(check_all_close(o_keops, o_torch, "o", 1e-2))
        self.assertTrue(check_all_close(dv_keops, dv_torch, "dv", 1e-2))
        self.assertTrue(check_all_close(dq_keops, dq_torch, "dq", 1e-2))
        self.assertTrue(check_all_close(dk_keops, dk_torch, "dk", 1e-2))

        print()

        t_attn_mask, t_forward, t_backward, o_torch, dq_torch, dk_torch, dv_torch = (
            self.impl_torch()
        )
        print(
            f"torch:\tt_attn_mask: {t_attn_mask:.4e} t_forward: {t_forward:.4e} t_backward: {t_backward:.4e}"
        )

        (
            t_attn_mask,
            t_forward,
            t_backward,
            o_native,
            dq_native,
            dk_native,
            dv_native,
        ) = self.impl_native()
        print(
            f"native:\tt_attn_mask: {t_attn_mask:.4e} t_forward: {t_forward:.4e} t_backward: {t_backward:.4e}"
        )
        self.assertTrue(check_all_close(o_native, o_torch, "o", 1e-2))
        self.assertTrue(check_all_close(dv_native, dv_torch, "dv", 1e-2))
        self.assertTrue(check_all_close(dq_native, dq_torch, "dq", 1e-2))
        self.assertTrue(check_all_close(dk_native, dk_torch, "dk", 1e-2))

        # print()
        # t_attn_mask, t_forward, t_backward, o_triton, dq_triton, dk_triton, dv_triton= self.impl_triton()
        # print(f"triton:\tt_attn_mask: {t_attn_mask:.4e} t_forward: {t_forward:.4e} t_backward: {t_backward:.4e}")
        # self.assertTrue(check_all_close(o_triton, o_torch, "o", 1e-2))


if __name__ == "__main__":
    unittest.main()
