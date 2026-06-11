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

import weakref
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


_SORTED_CACHE: dict = {}
_SORTED_CACHE_MAX = 256

_SEG_CACHE: dict = {}
_SEG_CACHE_MAX = 64


def kernel_offset_segments_cached(
    idx_sorted: Tensor, num_kernel_offsets: int
) -> Tensor:
    """Memoized ``kernel_offset_segments`` — the Triton grouped path
    rebuilds seg_offs 3x per training step (fwd mvmr, vvor grad_a,
    grad_b mvmr) on the same triplet structure; the CUTLASS path
    already shares it via the autograd ctx.

    Key: ``(data_ptr, _version, numel, K)`` + a weakref liveness check
    on the SOURCE tensor (alive + same _version + same data_ptr, else
    evict-and-rebuild) — the weakref closes the recycled-data_ptr
    aliasing hole a ptr/version-only key has (a new tensor allocated at
    a recycled address could otherwise alias a stale entry; the same
    hardening as the fold cache). ``inference_mode`` tensors (no
    version counter) bypass the cache. Residual documented assumption:
    version-bypassing writes (``.data`` / ``set_``) are invisible, as
    everywhere else.
    """
    try:
        ver = idx_sorted._version
    except RuntimeError:  # inference_mode tensor — no version counter
        return kernel_offset_segments(idx_sorted, num_kernel_offsets)
    key = (idx_sorted.data_ptr(), ver, idx_sorted.numel(),
           int(num_kernel_offsets))
    hit = _SEG_CACHE.get(key)
    if hit is not None:
        src, seg = hit
        alive = src()
        if (alive is not None and alive._version == ver
                and alive.data_ptr() == key[0]):
            return seg
        del _SEG_CACHE[key]
    seg = kernel_offset_segments(idx_sorted, num_kernel_offsets)
    if len(_SEG_CACHE) >= _SEG_CACHE_MAX:
        _SEG_CACHE.pop(next(iter(_SEG_CACHE)))
    _SEG_CACHE[key] = (weakref.ref(idx_sorted), seg)
    return seg


def is_sorted_cached(idx: Tensor) -> bool:
    """Sortedness check with a verdict memo — removes the per-call host
    sync from the grouped-dispatch hot path (the ``.item()`` here was
    the last remaining per-call sync; measured -11% isolated / -34%
    back-to-back at the C=512 regime).

    Key: ``(data_ptr, _version, numel)``. ``_version`` increments on any
    in-place write, so a mutated tensor re-checks; a NEW tensor at a
    recycled address re-checks unless it coincidentally shares numel AND
    the allocator-recycled pointer with version 0 — triplet index
    tensors are built once per rulebook and never written in place, so
    in practice every distinct rulebook re-checks once and training
    steps hit the memo. Bounded FIFO (256 entries).
    """
    key = (idx.data_ptr(), idx._version, idx.numel())
    hit = _SORTED_CACHE.get(key)
    if hit is not None:
        return hit
    verdict = bool((idx[1:] >= idx[:-1]).all().item())
    if len(_SORTED_CACHE) >= _SORTED_CACHE_MAX:
        _SORTED_CACHE.pop(next(iter(_SORTED_CACHE)))
    _SORTED_CACHE[key] = verdict
    return verdict


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
