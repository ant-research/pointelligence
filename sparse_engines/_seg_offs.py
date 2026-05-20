"""Segment-offset utility for the grouped MVMR / VVOR kernels.

The grouped kernels expect triplets sorted by kernel offset (the default
``sort_by="k"`` in ``layers/triplets.py``). This module turns that sort
into segment boundaries: ``seg_offs[k]`` is the start index in the
triplet arrays of the segment for kernel offset ``k``; ``seg_offs[K]``
equals the total triplet count ``T``.

The grouped kernel walks each segment in chunks of ``L_CHUNK`` triplets,
issues one ``tl.dot`` per chunk (LHS rows = L_CHUNK ≥ 16 → tensor cores
fire), and scatter-adds the partial result to the output indices.
"""

from typing import Tuple

import torch
from torch import Tensor


def kernel_offset_segments(idx_sorted: Tensor, num_kernel_offsets: int) -> Tensor:
    """Return ``seg_offs[K+1]`` for an ``idx_sorted`` array sorted ascending.

    ``seg_offs[k]`` is the position of the first triplet whose value ≥ k.
    Equivalently, ``seg_offs[k+1] - seg_offs[k]`` is the length of the
    segment for kernel offset ``k``. ``seg_offs[K] == T``.

    Computed via a single ``torch.searchsorted`` — no host sync, no extra
    allocation beyond the K+1 output.

    Args:
      idx_sorted: 1-D int tensor of length T, sorted non-decreasing,
        with values in [0, K).
      num_kernel_offsets: ``K`` — the number of kernel-offset bins
        (typically 27 for k=3 conv, 125 for k=5).

    Returns:
      ``seg_offs`` of shape ``(K+1,)`` int64 on the same device.
    """
    K = int(num_kernel_offsets)
    bins = torch.arange(K + 1, device=idx_sorted.device, dtype=idx_sorted.dtype)
    # `searchsorted(sorted_seq, values)` returns insertion points; since
    # idx_sorted is sorted ascending, this gives the start of each segment.
    return torch.searchsorted(idx_sorted, bins)


def chunk_grid_for_segments(
    seg_offs: Tensor, l_chunk: int
) -> Tuple[Tensor, int]:
    """Given segment offsets, return the per-segment chunk counts and the
    cumulative chunk-start offsets needed by the grouped kernel's grid.

    Returns ``(chunk_seg_offs, total_chunks)`` where:
      - ``chunk_seg_offs[k]`` = first chunk index belonging to kernel offset ``k``
      - ``chunk_seg_offs[K]`` = ``total_chunks``
      - ``total_chunks`` = sum over k of ``ceil(seg_len[k] / l_chunk)``

    Each program in the grouped kernel processes one chunk; it locates
    its kernel offset by searching ``chunk_seg_offs`` for its program id.
    """
    seg_lens = seg_offs[1:] - seg_offs[:-1]
    chunks_per_k = (seg_lens + l_chunk - 1) // l_chunk
    chunk_seg_offs = torch.zeros(
        seg_offs.numel(), device=seg_offs.device, dtype=seg_offs.dtype,
    )
    chunk_seg_offs[1:] = torch.cumsum(chunks_per_k, dim=0)
    total_chunks = int(chunk_seg_offs[-1].item())
    return chunk_seg_offs, total_chunks


def total_chunks_for_lchunks(
    seg_offs: Tensor, l_chunk_options: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute total chunk count for each L_CHUNK in ``l_chunk_options``.

    Used by the grouped kernel launchers to support an autotune palette
    that varies L_CHUNK. We sum on-device for all options first, then
    pay a single host sync to read them back as ints — vs. one sync per
    option if computed sequentially.

    The kernels themselves walk seg_offs on-the-fly (no chunk_seg_offs
    needed) so we don't need to materialise the cumsum.
    """
    seg_lens = seg_offs[1:] - seg_offs[:-1]
    sums = torch.stack([
        ((seg_lens + lc - 1) // lc).sum() for lc in l_chunk_options
    ])
    return tuple(int(x) for x in sums.cpu().tolist())
