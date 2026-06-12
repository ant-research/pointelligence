"""The partition conv as ONE dense GEMM (exact im2col).

A DISJOINT grid partition (the patchify-stem pattern: every input point lands in
exactly one (cell, slot) pair) admits an EXACT dense im2col: scatter each
point's C features into row ``cell``, column block ``slot*C:(slot+1)*C`` of a
dense ``z`` (uniqueness by disjointness — no accumulation), then the whole
conv is ``out = z @ W.reshape(K*C, M)``. Autograd supplies the two backward
GEMMs (``grad_W = z^T g``, ``grad_z = g W^T`` + gather) for free.

This formulation fits ONLY the partition shape: N_out ≈ N_in/occupancy and
tiny C keep ``z`` small (28 MB fp16 at the production stem), and disjointness
makes the scatter exact. The earlier rejection of masked-tiling/im2col
formulations was for the general submanifold case (N_out = N_in, large C →
z explodes) and does not apply here.
"""
from __future__ import annotations

import torch

__all__ = ["partition_dense_mvmr"]


def partition_dense_mvmr(
    weight: torch.Tensor,
    feat: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    n_out: int,
) -> torch.Tensor:
    """``out[i_t] += feat[j_t] @ W[k_t]`` for a DISJOINT partition.

    The scatter ACCUMULATES (functional ``scatter_add`` — measured 6x
    faster than ``index_put(accumulate=True)`` at fp16 on Ada), so duplicate
    ``(i_t, k_t)`` pairs sum exactly like the mvmr semantics (linearity:
    ``W[k] @ (x1 + x2) == W[k] @ x1 + W[k] @ x2``) — the real partition
    builders produce rare duplicates at sub-voxel boundaries (the slot
    clamp under fp rounding), and dropping them is a silent error. The
    partition shape is what makes the DENSE z affordable (N_out ≈
    N_in/occupancy, tiny C); the formulation itself is exact for any
    triplet set.

    weight: (K, G=1, C, M) or (K, C, M); feat: (N_in, C). Returns
    (n_out, M) in feat.dtype. Differentiable in weight and feat.
    """
    if weight.dim() == 4:
        K, G, C, M = weight.shape
        if G != 1:
            raise ValueError("partition_dense_mvmr supports G == 1 only")
    else:
        K, C, M = weight.shape
    w2 = weight.reshape(K * C, M)

    cols = k.long().unsqueeze(1) * C + torch.arange(C, device=feat.device)
    flat = i.long().unsqueeze(1) * (K * C) + cols                  # (T, C)
    z = feat.new_zeros(n_out * K * C).scatter_add(
        0, flat.reshape(-1), feat[j.long()].reshape(-1))
    return z.view(n_out, K * C) @ w2
