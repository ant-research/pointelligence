"""Pure-PyTorch reference for the generative-expansion point conv.

Given a rulebook ``(i, j, k)`` and the conv weight, computes the scatter
convolution ``out[i] += x[j] @ W[k]`` via an independent code path
(``einsum`` + ``index_add_``) — distinct from the MVMR sparse engine —
used for numerical-parity checks of ``GenerativePointConv3d``.
"""

from typing import Optional

import torch
from torch import Tensor


def generative_conv_scatter_ref(
    x: Tensor,
    i: Tensor,
    j: Tensor,
    k: Tensor,
    n_out: int,
    weight: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Naive scatter reference for the generative conv (computed in fp32).

    Args:
        x: ``(N, Cin)`` input features.
        i, j, k: ``(T,)`` rulebook — output / input / kernel-tap indices.
        n_out: number of output points.
        weight: ``(K, G, Cin/G, Cout/G)`` conv weight (native Lane-Q layout).
        bias: optional ``(Cout,)``.

    Returns:
        ``(n_out, Cout)`` output features, fp32.
    """
    K, G, CinG, CoutG = weight.shape
    N, Cin = x.shape
    Cout = G * CoutG
    assert Cin == G * CinG, f"Cin {Cin} != G*CinG {G * CinG}"

    x3 = x.to(torch.float32).view(N, G, CinG)
    w = weight.to(torch.float32)

    xt = x3[j.long()]                                   # (T, G, CinG)
    wt = w[k.long()]                                    # (T, G, CinG, CoutG)
    contrib = torch.einsum("tgc,tgco->tgo", xt, wt)     # (T, G, CoutG)
    contrib = contrib.reshape(i.shape[0], Cout)         # (T, Cout)

    out = torch.zeros(n_out, Cout, dtype=torch.float32, device=x.device)
    out.index_add_(0, i.long(), contrib)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out
