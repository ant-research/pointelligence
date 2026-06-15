"""Convolution triplet (i, j, k) construction.

Builds the sparse connectivity that maps each (query i, neighbor j) pair
to a kernel weight index k via local voxelization. The triplets define
the data-weight routing for PointConv3d.
"""

from functools import partial
from typing import Tuple, Callable, Optional

import math
import weakref
import torch

from torch import Tensor
from torch.nn.common_types import _size_3_t
from torch.nn.modules.utils import _triple

from internals.neighbors import radius_search
from internals.indexing import cumsum_exclusive, repeat_interleave_indices
from internals.triplet_cache import _active_cache, triplet_key
from internals.grid_sample import grid_sample_filter

from .metadata import MetaData
from .downsample import downsample
from .contract import TripletContract


def _build_triplets_from_neighbor_pairs(
    points: Tensor,
    query_points: Tensor,
    neighbor_indices: Tensor,
    num_neighbors: Tensor,
    kernel_indexer: Callable,
    neighbor_radius: float,
    radius_scaler: float,
    sort_by: str,
    return_num_neighbors: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Shared post-radius-search triplet construction.

    Both ``build_triplets`` (which calls ``radius_search``) and
    the strided builder hand off to this helper for the
    cumsum + repeat_interleave + offset + kernel_indexer + sort +
    dtype-norm steps. Behavior matches the original inlined logic in
    ``build_triplets`` line-for-line.

    Internally wraps in ``torch.no_grad()`` — caller does NOT need
    to be inside a no_grad block.
    """
    with torch.no_grad():
        assert torch.min(num_neighbors) > 0, (
            "Neighborhood search failed for some points, consider increase the neighbor_radius. "
            "It is likely that this happens in an *upsample* phase, "
            "where the *query_points* are not a subset of the *points*."
        )

        num_neighbors_cumsum, repeats_sum = cumsum_exclusive(
            num_neighbors, return_sum=True
        )
        center_indices = repeat_interleave_indices(
            repeats_cumsum=num_neighbors_cumsum, output_size=repeats_sum
        )

        offsets = (points[neighbor_indices] - query_points[center_indices])

        i = center_indices
        j = neighbor_indices
        k = kernel_indexer(points=offsets, grid_size=neighbor_radius / radius_scaler)

        if "i" == sort_by:
            pass
        elif "j" == sort_by:
            j, sorter = torch.sort(j)
            i = i[sorter]
            k = k[sorter]
        elif "k" == sort_by:
            k, sorter = torch.sort(k)
            i = i[sorter]
            j = j[sorter]
        else:
            assert (
                False
            ), f'Unknown sort_by argument "{sort_by}", it should be i, j, or k!'

        # Normalize all triplet indices to a consistent dtype.
        # The upstream code paths produce mixed dtypes (i: int64 from cumsum,
        # j: int32 from radius_search, k: int64 from torch.sum promotion).
        # Triton's autotuner keys on pointer dtypes, so mixed dtypes cause
        # redundant cache entries and mid-training autotune stalls.
        # Use int32 when safe, int64 otherwise. We check point counts
        # (already CPU ints) as upper bounds instead of calling .max().item()
        # which would force GPU→CPU syncs.
        _INT32_MAX = 2147483647
        max_possible = max(query_points.shape[0], points.shape[0])
        idx_dtype = torch.int32 if max_possible <= _INT32_MAX else torch.int64
        i = i.to(idx_dtype)
        j = j.to(idx_dtype)
        k = k.to(idx_dtype)

        if return_num_neighbors:
            return i, j, k, num_neighbors
        else:
            return i, j, k, None


def voxelize_3d(
    kernel_size: _size_3_t,
    points: Tensor,
    grid_size: Optional[Tensor] = None,
) -> Tensor:
    with torch.no_grad():
        sizes = _triple(kernel_size)
        ks = sizes[0]

        kernel_offset = (points / grid_size).round().int()
        k_l = torch.clamp(kernel_offset + (ks // 2), min=0, max=ks - 1)
        multipliers = torch.tensor([ks * ks, ks, 1], device=points.device, dtype=torch.int32)
        indices = torch.sum(k_l * multipliers, dim=1)

    return indices


@torch.compiler.disable
def build_triplets(
    points: Tensor,
    sample_inds: Tensor,
    sample_sizes: Tensor,
    neighbor_radius: float,
    kernel_indexer: Callable,
    query_points: Optional[Tensor] = None,
    query_sample_inds: Optional[Tensor] = None,
    query_sample_sizes: Optional[Tensor] = None,
    sort_by: str = "k",
    return_num_neighbors=False,
    radius_scaler: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
    with torch.no_grad():
        # They have to be all None, or all not None
        if (
            query_points is None
            or query_sample_inds is None
            or query_sample_sizes is None
        ):
            assert query_points is None
            assert query_sample_inds is None
            assert query_sample_sizes is None
            query_points = points
            query_sample_inds = sample_inds
            query_sample_sizes = sample_sizes

        cache = _active_cache()
        key = None
        if cache is not None:
            key = triplet_key(
                points, query_points, neighbor_radius, radius_scaler,
                sort_by, return_num_neighbors, sample_inds,
                query_sample_inds, kernel_indexer,
            )
            if key is not None:           # None => uncacheable indexer; just build
                hit = cache.get(key)
                if hit is not None:
                    cached_out, pref = hit
                    if pref() is not None:   # keyed tensor alive => addr not recycled
                        return cached_out
                    # stale (data_ptr recycled): fall through, rebuild, overwrite

        grid_size = neighbor_radius / radius_scaler if radius_scaler is not None else None
        neighbor_indices, num_neighbors = radius_search(
            points=points,
            query_points=query_points,
            radius=neighbor_radius,
            sample_inds=sample_inds,
            sample_sizes=sample_sizes,
            query_sample_inds=query_sample_inds,
            query_sample_sizes=query_sample_sizes,
            grid_size=grid_size,
        )

    out = _build_triplets_from_neighbor_pairs(
        points=points,
        query_points=query_points,
        neighbor_indices=neighbor_indices,
        num_neighbors=num_neighbors,
        kernel_indexer=kernel_indexer,
        neighbor_radius=neighbor_radius,
        radius_scaler=radius_scaler,
        sort_by=sort_by,
        return_num_neighbors=return_num_neighbors,
    )
    if cache is not None and key is not None:
        # Weakref the primary keyed tensor `points`; on a hit a dead weakref is
        # treated as a miss, closing the freed-then-recycled-address window.
        # We guard ONLY `points`, not the other data_ptr-keyed tensors
        # (sample_inds / query / query_sample_inds): every builder here stores
        # those on the SAME MetaData (or its parent chain) as `points`, so they
        # share a lifetime and are co-pinned for the whole forward scope — a live
        # `points` implies its companions are live. A public-API caller pairing a
        # persistent `points` with a transient, recycled `sample_inds` of
        # different contents would be the only way to defeat this; such a caller
        # should weakref-guard all keyed tensors.
        cache[key] = (out, weakref.ref(points))
    return out


def radius_scaler_for_kernel_size(kernel_size: _size_3_t, receptive_field_scaler: float = 1.0, distance_type: str = "ball"):
    """Compute the radius_scaler that determines the neighborhood search radius.

    PointCNN++ decouples kernel resolution from the receptive field's physical
    span. kernel_size controls the fineness of the weight grid (e.g., 3^3 or
    5^3 kernel cells), while receptive_field_scaler controls how much physical
    space the search sphere covers, as a volume multiplier of the kernel_size^3
    cube. This decoupling allows using a fine kernel (large kernel_size) over
    a small region, or a coarse kernel over a large region, independently.

    The returned radius_scaler is used as:
        neighbor_radius = grid_size * radius_scaler

    And inside build_triplets, grid_size is recovered as:
        grid_size = neighbor_radius / radius_scaler

    so that voxelize_3d always discretizes at the original grid_size scale,
    regardless of receptive_field_scaler.

    Args:
        kernel_size: Number of kernel cells per dimension (e.g., 3 -> 3^3 = 27 weights)
        receptive_field_scaler: Volume multiplier. 1.0 means the search sphere has
            the same volume as the kernel_size^3 cube. >1.0 searches wider (more
            neighbors per kernel cell). Default: 1.0
        distance_type: "ball" (spherical search) or "cube" (cubic search)

    Returns:
        radius_scaler: Multiply by grid_size to get the physical search radius.
            For kernel_size=3, ball mode, rfs=1.0: radius_scaler ≈ 1.86
    """
    kernel_size = _triple(kernel_size)
    kernel_size_max = max(kernel_size)

    if distance_type == "cube":
        cube_edge_length = kernel_size_max * math.pow(receptive_field_scaler, 1 / 3)
        radius_scaler = cube_edge_length / 2.0
    else:
        volume = math.pow(kernel_size_max, 3) * receptive_field_scaler
        radius_scaler = math.pow(3 * volume / (4 * math.pi), 1 / 3)

    return radius_scaler


def _no_overlap_check(
    kernel_size: _size_3_t,
    receptive_field_scaler: float,
    distance_type: str,
    stride: float,
) -> float:
    """Enforce the no-overlap invariant and return the computed radius_scaler.

    Raises ``ValueError`` if ``2 * radius_scaler > stride`` — the
    ball-disjoint contract required by ``conv_with_stride_disjoint`` and
    ``handle_stride_disjoint_and_build_triplets``.
    """
    radius_scaler = radius_scaler_for_kernel_size(
        kernel_size, receptive_field_scaler, distance_type
    )
    if 2 * radius_scaler > stride:
        raise ValueError(
            f"conv_with_stride_disjoint no-overlap violated: "
            f"2 * radius_scaler ({2 * radius_scaler:.4f}) > stride ({stride:.4f}) "
            f"with kernel_size={kernel_size}, "
            f"receptive_field_scaler={receptive_field_scaler}, "
            f"distance_type={distance_type!r}. "
            f"Reduce receptive_field_scaler, increase stride, or use "
            f"handle_stride_and_build_triplets / conv_with_stride for "
            f"overlap-allowed strided conv."
        )
    return radius_scaler




@torch.compiler.disable
def handle_stride_and_build_triplets(
    m: MetaData,
    stride: float,
    kernel_size: _size_3_t = _triple(3),
    receptive_field_scaler: float = 1.0,
    sort_by: str = "k",
    return_num_neighbors=False,
    distance_type: str = "ball",
) -> MetaData:
    with torch.no_grad():
        if stride != 1:
            if (m.parent is not None and
                m.parent.points.shape == m.points.shape and
                m.parent.grid_size == m.grid_size and
                torch.equal(m.parent.points, m.points) and
                torch.equal(m.parent.sample_inds, m.sample_inds)):
                parent_meta = m.parent
            else:
                parent_meta = MetaData(
                    points=m.points,
                    sample_inds=m.sample_inds,
                    sample_sizes=m.sample_sizes,
                    grid_size=m.grid_size,
                    parent=m.parent,
                )
                m.parent = parent_meta

            points_ = parent_meta.points
            sample_inds_ = parent_meta.sample_inds
            sample_sizes_ = parent_meta.sample_sizes
            grid_size_ = parent_meta.grid_size

            radius_scaler = radius_scaler_for_kernel_size(kernel_size, receptive_field_scaler, distance_type)
            neighbor_radius = grid_size_ * radius_scaler

            m.points, m.sample_inds, m.grid_size, m.downsample_indices = downsample(
                m.points, m.sample_inds, m.grid_size, stride
            )
            m.sample_sizes = torch.bincount(m.sample_inds)

            m.i, m.j, m.k, m.num_neighbors = build_triplets(
                points=points_,
                sample_inds=sample_inds_,
                sample_sizes=sample_sizes_,
                neighbor_radius=neighbor_radius,
                kernel_indexer=partial(voxelize_3d, kernel_size=kernel_size),
                query_points=m.points,
                query_sample_inds=m.sample_inds,
                query_sample_sizes=m.sample_sizes,
                sort_by=sort_by,
                return_num_neighbors=return_num_neighbors,
                radius_scaler=radius_scaler,
            )

            parent_meta.i_upsample = m.j
            parent_meta.j_upsample = m.i
            parent_meta.k_upsample = m.k

        if m.empty_triplets():
            radius_scaler = radius_scaler_for_kernel_size(kernel_size, receptive_field_scaler, distance_type)
            neighbor_radius = m.grid_size * radius_scaler
            m.i, m.j, m.k, m.num_neighbors = build_triplets(
                points=m.points,
                sample_inds=m.sample_inds,
                sample_sizes=m.sample_sizes,
                neighbor_radius=neighbor_radius,
                kernel_indexer=partial(voxelize_3d, kernel_size=kernel_size),
                sort_by=sort_by,
                return_num_neighbors=return_num_neighbors,
                radius_scaler=radius_scaler,
            )

    # Submanifold / strided (overlapping) rulebook: variable per-tap neighbour
    # counts (T != n, T != n_in) ⇒ neither exact cover, not uniform. The
    # k_sorted flag is HONEST about the requested sort: conv builders pass the
    # default sort_by="k" (→ submanifold(), k_sorted=True), but max_pool3d calls
    # this with sort_by="i" — its triplets are i-sorted, so stamping
    # k_sorted=True would be a dishonest contract (a conv consuming it would
    # silently route the assume-k-sorted TIG path on i-sorted data). max_pool3d
    # feeds indexed_segment_reduce, never a conv (contract.py), so this is
    # latent today; deriving k_sorted from sort_by keeps the contract truthful
    # if that ever changes. Structural constant — no host reduction.
    m.contract = TripletContract(k_sorted=(sort_by == "k"))
    return m


@torch.compiler.disable
def handle_stride_disjoint_and_build_triplets(
    m: MetaData,
    stride: float,
    kernel_size: _size_3_t = _triple(3),
    sort_by: str = "k",
) -> MetaData:
    """DISJOINT strided conv as a true GRID PARTITION (no radius search).

    Each input point is assigned to exactly ONE output cell by integer
    voxelization at ``cell_size = stride * grid_size`` — a tiling that is
    disjoint *and* covering (every point lands in exactly one cell; no point is
    orphaned). The output token sits at the **cell (voxel) center**
    (``grid_sample_filter(reduction="center")``), and the kernel-bucket ``k`` is
    the point's **cell-grid-relative sub-voxel slot**
    ``floor((point - cell_origin)/cell_size * K) in [0, K)`` per axis (cubic
    kernel ``K = kernel_size``), flattened to ``[0, K**3)``. The slot is bounded
    by construction (no clamp / no aliasing) and decoupled from the token
    coordinate: the slot answers "which sub-position within the patch", the cell
    center answers "where is the patch".

    This is the point analogue of a ViT ``Conv2d(kernel=P, stride=P)`` patchify:
    a partition into non-overlapping cubic patches with the weight indexed by
    the sub-position. The cached ``(i, j, k)_upsample`` edges are the exact
    inverse map, so an ``Upsample(straight_recover=True, recompute_k=False)``
    recovers EVERY raw point from its cell-center token (the unpatchify) with
    full coverage — see ``layers/upsample.py``.

    Supersedes the prior ball-radius-search disjoint conv, which only gathered
    the inscribed sphere of the patch and orphaned ~25-50% of points at the
    corners. ``receptive_field_scaler`` / ``distance_type`` no longer apply (a
    partition has no search radius) and were removed from the signature.
    """
    if stride <= 1:
        raise ValueError(
            f"handle_stride_disjoint_and_build_triplets requires stride > 1 "
            f"(downsample case); got stride={stride}."
        )
    ks = _triple(kernel_size)
    if not (ks[0] == ks[1] == ks[2]):
        raise ValueError(
            f"partition disjoint conv requires a cubic kernel; got kernel_size={ks}")
    K = int(ks[0])

    with torch.no_grad():
        if (m.parent is not None and
            m.parent.points.shape == m.points.shape and
            m.parent.grid_size == m.grid_size and
            torch.equal(m.parent.points, m.points) and
            torch.equal(m.parent.sample_inds, m.sample_inds)):
            parent_meta = m.parent
        else:
            parent_meta = MetaData(
                points=m.points,
                sample_inds=m.sample_inds,
                sample_sizes=m.sample_sizes,
                grid_size=m.grid_size,
                parent=m.parent,
            )
            m.parent = parent_meta

        points_ = parent_meta.points
        sample_inds_ = parent_meta.sample_inds
        grid_size_ = parent_meta.grid_size
        cell_size = stride * grid_size_

        # Grid partition: each point -> its cell; token coord = cell CENTER.
        # `mapping` (lookup_inds) is per-point -> token row, aligned with the
        # returned cell-center coords (100% coverage by construction).
        points_c, sample_inds_c, ds_indices, mapping = grid_sample_filter(
            points=points_,
            grid_size=cell_size,
            sample_inds=sample_inds_,
            reduction="center",
            return_mapping=True,
        )
        # reduction="center" with sample_inds appends the batch index as a 4th
        # column ((batch+0.5)*cell_size); keep only the xyz cell centers.
        m.points = points_c[:, :3].contiguous()
        m.sample_inds = sample_inds_c
        m.grid_size = cell_size
        m.sample_sizes = torch.bincount(sample_inds_c)
        m.downsample_indices = ds_indices

        # Cell-grid-relative sub-voxel slot, bounded [0, K) per axis. cell_vox via
        # floor(pt/cell_size) matches grid_sample_filter's cell assignment
        # (compute_grid_indices, with_shifts=False).
        cell_vox = torch.div(points_, cell_size, rounding_mode="floor")
        frac = points_ / cell_size - cell_vox                 # [0, 1) per axis
        sub = torch.floor(frac * K).to(torch.long).clamp_(0, K - 1)
        kk = (sub[:, 0] * K + sub[:, 1]) * K + sub[:, 2]      # [0, K**3)

        n_pts = points_.shape[0]
        idx = torch.arange(n_pts, device=points_.device)
        if sort_by == "k":  # keep the k-ascending contract (cached k_upsample reuse)
            order = torch.argsort(kk)
            idx, kk = idx[order], kk[order]
        m.i = mapping.to(torch.long)[idx]   # output token per edge (one edge / point)
        m.j = idx                           # input point per edge
        m.k = kk
        m.num_neighbors = None

        parent_meta.i_upsample = m.j
        parent_meta.j_upsample = m.i
        parent_meta.k_upsample = m.k

    # Disjoint strided rulebook is the one builder that can be fan-out-1: each
    # input point contributes to AT MOST one output cell, so when every input
    # row appears in exactly one triplet (T == n_in) the partition-stem
    # exact_cover_in contract holds (routes FI1 grad_input + the dense-GEMM
    # partition engine). Prove it ONCE here with the same memoized bincount the
    # old forward used — inside this @compiler.disable region the .item() is
    # free and never traced (vs. per-forward in _conv_forward, which broke
    # compile). Gate on the free host checks first (T == n_in and T != n),
    # exactly mirroring the retired conv.py auto-detect. exact_cover_out stays
    # False (deconv/strided is never fan-in-1 here); not uniform (radius search).
    _eci = False
    if m.k is not None and m.k.numel() > 0:
        n_out = m.points.shape[0]
        n_in = points_.shape[0]
        if m.k.numel() == n_in and m.k.numel() != n_out:
            from sparse_engines._seg_offs import exact_cover_cached
            _eci = exact_cover_cached(m.j, n_in)
    m.contract = TripletContract(k_sorted=True, exact_cover_out=False,
                                 exact_cover_in=_eci, uniform_seg_len=None)
    return m
