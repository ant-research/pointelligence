"""Grouped VVOR via cooperative-warp split-K WMMA kernel.

Same algorithm as sparse_vvor_grouped_wmma but each (k, mt, ct) tile is
split across W=8 single-warp blocks along the T axis. Each block accumulates
a partial 16x16 fp32 grad_weight fragment and atomicAdds to the pre-zeroed
output. Goal: use more SMs at small-C stages (enc0: 108 -> 864 blocks at W=8).
"""

import torch
from torch import Tensor

from ._seg_offs import kernel_offset_segments
import sparse_engines_cuda._C  # ensure TORCH_LIBRARY static initializers run

from .vvor_grouped_cuda import (
    sparse_vector_vector_outer_product_reduction_grouped_cuda as _fallback_scalar_fma,
)

# Default W: 8 slices per tile. Tunable per stage; bench will inform whether
# enc4 needs lower W due to atomic contention.
_DEFAULT_W = 8


def sparse_vector_vector_outer_product_reduction_grouped_wmma_coop(
    a: Tensor, a_idx: Tensor, b: Tensor, b_idx: Tensor, o_idx: Tensor,
    n_o: int, *, w: int = _DEFAULT_W,
) -> Tensor:
    """Cooperative-warp split-K WMMA vvor.

    Same call signature as sparse_vector_vector_outer_product_reduction_grouped_wmma
    plus a keyword `w` controlling the split-K slice count per tile.
    """
    if a.dtype == torch.float32:
        return _fallback_scalar_fma(a, a_idx, b, b_idx, o_idx, n_o)

    a = a.contiguous()
    b = b.contiguous()

    M = a.shape[2]
    C = b.shape[2]
    G = a.shape[1]

    # G >= 1 supported natively: the frozen kernel's block grid is
    # K * G * (M/16) * (C/16) * W with G-strided loads/stores — no
    # per-group loop needed. M / C are PER-GROUP channel counts, so the
    # %16 gate below is a per-group gate.
    if M % 16 != 0 or C % 16 != 0:
        raise ValueError(
            "WMMA-coop vvor requires per-group M and C divisible by 16; "
            f"got per-group M={M}, C={C} (G={G})"
        )
    if not bool((o_idx[1:] >= o_idx[:-1]).all().item()):
        raise ValueError("o_idx must be sorted ascending for grouped path")

    input_dtype = a.dtype
    seg_offs = kernel_offset_segments(o_idx, n_o)

    o = torch.ops.sparse_engines_cuda.sparse_vvor_grouped_wmma_coop(
        a, a_idx, b, b_idx, o_idx, seg_offs, n_o, w
    )

    return o if o.dtype == input_dtype else o.to(input_dtype)
