"""Collision-free grid-index encoding and sorted-segment utilities."""

import os

import torch
import triton

from .grid_hash_triton_kernel import grid_hash_kernel_4d

# The grid-hash int64 overflow guard is a debug check: it recomputes the
# stride cumprod in float64 and reads it to host (a D2H sync) on EVERY
# reduce_indices_to_1d call. The builders are host-bound, so this per-call
# sync stalls the GPU under contention. Gate it behind a flag (default off);
# real grids never approach 2**63 (would need a ~2M-cell axis extent).
# (Repo-wide building block → POINTELLIGENCE_ prefix, not PNT_ which denotes
# the PNT architecture.)
_GRID_OVERFLOW_CHECK = os.environ.get("POINTELLIGENCE_DEBUG_GRID_OVERFLOW", "0") == "1"


def compute_grid_indices(points, grid_size, sample_inds=None, dtype=torch.int32):
    """make points to 3d grid indices"""
    grid_inds = points.div(grid_size, rounding_mode="floor").to(dtype)

    if sample_inds is not None:
        grid_inds = torch.cat((grid_inds, sample_inds.unsqueeze(-1)), dim=1)

    return grid_inds.contiguous()


def reduce_indices_to_1d(inds, inds_min=None, inds_stride=None, dtype=None):
    assert (
        inds.dtype is not torch.float16
    ), "reduce_inds_to_1d does not support half inds, because of the inds_stride[-1] > 32767 make inf!"
    inds_min = torch.amin(inds, dim=0, keepdim=True) if inds_min is None else inds_min
    if inds_stride is None:
        inds_step = torch.amax(inds, dim=0, keepdim=True).add_(1)
        if not (isinstance(inds_min, int) and 0 == inds_min):
            inds_step = inds_step.sub_(inds_min)
        inds_stride = inds_step.cumprod(dim=-1, dtype=torch.int64)
        if _GRID_OVERFLOW_CHECK:
            assert inds_step.cumprod(dim=-1, dtype=torch.float64)[0, -1] < 2**63 - 1
        dtype = torch.int64 if inds_stride[0, -1] > 2**31 - 1 else torch.int32
        inds_stride = inds_stride.to(dtype)
        inds_stride = inds_stride.roll(1, dims=-1)
        inds_stride[0, 0] = 1

    n, d = inds.shape
    if d == 4 and inds.is_contiguous() and inds.is_cuda:
        # Fused Triton kernel: replaces (sub, mul, sum) with a single launch
        inds_1d = torch.empty((n,), dtype=dtype, device=inds.device)
        grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
        grid_hash_kernel_4d[grid](
            inds, inds_1d, inds_min.reshape(-1), inds_stride.reshape(-1), n
        )
    else:
        if not (isinstance(inds_min, int) and 0 == inds_min):
            inds_ = (inds - inds_min).to(dtype).mul_(inds_stride)
        else:
            if dtype == inds.dtype:
                inds_ = inds.mul(inds_stride)
            else:
                inds_ = inds.to(dtype).mul_(inds_stride)
        inds_1d = torch.sum(inds_, dim=-1)
    return inds_1d, inds_min, inds_stride, dtype


def reduction_params_from_indices(*indices):
    """Build collision-free linearization parameters for several index sets.

    Every set that will be encoded or queried must contribute to the bounds.
    Otherwise an out-of-range coordinate can wrap into another in-range cell
    under the mixed-radix linearization.
    """
    if not indices:
        raise ValueError("at least one index tensor is required")
    inds_min = torch.stack([x.amin(dim=0) for x in indices]).amin(dim=0, keepdim=True)
    inds_max = torch.stack([x.amax(dim=0) for x in indices]).amax(dim=0, keepdim=True)
    inds_step = inds_max.sub(inds_min).add_(1)
    cumulative = inds_step.cumprod(dim=-1, dtype=torch.int64)
    if _GRID_OVERFLOW_CHECK:
        assert inds_step.cumprod(dim=-1, dtype=torch.float64)[0, -1] < 2**63 - 1
    dtype = torch.int64 if cumulative[0, -1] > 2**31 - 1 else torch.int32
    inds_stride = cumulative.to(dtype).roll(1, dims=-1)
    inds_stride[0, 0] = 1
    return inds_min, inds_stride, dtype


def build_sorted_grid_segments(
    grid_inds,
    *,
    return_inverse=False,
):
    """Group grid indices into sorted contiguous segments.

    Returns the permutation that sorts points by their collision-free grid key,
    the length of every occupied-cell segment, and optionally the segment index
    of every input point. Unlike the retired lookup API, this function does not
    retain unique keys or linearization state for later queries.
    """
    grid_keys, _, _, _ = reduce_indices_to_1d(grid_inds)
    sorted_keys, sorter = torch.sort(grid_keys)
    if return_inverse:
        _, inverse_sorted, counts = torch.unique_consecutive(
            sorted_keys, return_inverse=True, return_counts=True
        )
        inverse = torch.empty_like(inverse_sorted)
        inverse.index_copy_(0, sorter, inverse_sorted)
    else:
        _, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        inverse = None
    return sorter, counts, inverse
