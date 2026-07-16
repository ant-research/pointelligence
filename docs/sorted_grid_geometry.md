# Sorted-Grid Convolution Geometry

`v1.5.0` moves the geometry pipeline around `PointConv3d` to compact,
collision-free sorted grids. The release changes how neighborhoods,
convolution triplets, strided centers, and center-nearest downsampling are
prepared. It does not approximate the fixed-radius predicate and does not
change the meaning of a convolution triplet `(i, j, k)`.

## What Changed

| Stage | Previous path | v1.5.0 path | Semantic contract |
|---|---|---|---|
| Radius search | shifted-grid lookup, data-dependent dispatch, and large-query chunking | compact exact eight-cell sorted grid | identical unique neighbor sets and per-query counts |
| Triplet preparation | query-major neighbors, materialized `k`, then a global tap sort | direct tap-major `(i, j, seg_offs)` emission | identical canonical `(i, j, k)` triplets |
| Center-nearest downsampling | Torch selector plus an always-built inverse map | segmented Triton argmin and optional inverse map | identical stable representative and mapping |
| Overlapping strided convolution | no reusable full-cover API | observed-point centers plus a deterministic residual radius-net | every input has coverage, with no fallback edge |

The geometry-only two-phase protocol is also public: index construction can be
hoisted ahead of the feature-compute chain, while the convolution itself keeps
the same `TripletContract` and numerical path.

## Exact Sorted-Grid Radius Search

`radius_search(..., backend="auto")` now selects
`sorted8_materialized`. Points are sorted into half-open cells with side length
`2R`. A query's axis-aligned radius box intersects exactly two cells per axis,
so their Cartesian product is a complete, duplicate-free eight-cell candidate
cover. A compact Triton count/fill pass applies the existing exact ball or
Chebyshev predicate to those candidates.

The implementation:

- packs batch identity and integer coordinates into collision-free sorted keys;
- runs under `torch.no_grad()`;
- stores compact query/cell ranges and accepted edges, not candidate pairs;
- performs no query segmentation or result concatenation;
- has no data-dependent CPU dispatch or per-call point-count split;
- preserves empty-input, batching, count-dtype, and optional-distance behavior.

The supported backend names are:

- `auto` and `sorted8_materialized`: production exact-eight path;
- `sorted27_materialized`: symmetric 27-cell alternative using `R`-sized
  cells;
- `tiled`: explicit diagnostic/reference path.

`sorted_grid8` and `fixed_grid` remain source-compatible aliases for the two
sorted-grid paths. The removed shifted-grid lookup is not a production backend.
Ordering inside a query is intentionally unspecified; unique neighbor
membership and counts are the API contract.

## Tap-Segmented Triplets

Point convolution wants neighbors grouped by kernel tap. The sorted-eight path
can emit that layout directly:

```python
from internals.neighbors import radius_search_sorted_grid8_segments

i, j, seg_offs, num_neighbors = radius_search_sorted_grid8_segments(
    points,
    queries,
    radius,
    kernel_size=3,
    kernel_grid_size=grid_size,
    sample_inds=sample_inds,
    query_sample_inds=query_sample_inds,
)
```

`seg_offs` has one interval per kernel tap and describes where each tap's
triplets begin and end. It is not the same object as `k`: `k` stores one tap ID
per edge, while `seg_offs` stores only the segment boundaries. The direct path
builds the boundaries during compact fill, so consumers no longer need the
retired bucket-arrangement extension or another global sort. Explicit `k` is
still materialized at the public convolution boundary where compatibility or
diagnostics require it.

## Full-Cover Strided Convolution

Two reusable APIs add overlapping, full-coverage strided point convolution:

```python
from layers import PointConv3d, conv_with_stride_full_cover
from layers.triplets import build_full_cover_strided_rulebook

conv = PointConv3d(64, 128, kernel_size=15)
x_low, m_low = conv_with_stride_full_cover(
    conv,
    x,
    metadata,
    stride=8,
    radius_margin=1e-2,
    radius_backend="auto",
)
```

The builder first partitions each batch into stride cells and picks the
observed point nearest each occupied cell's geometric center. Those initial
centers are `C0`; output coordinates are never snapped to voxel centers. With
`cell_size = stride * input_grid_size`, the default search radius is

```text
R = sqrt(3) / 2 * cell_size * (1 + radius_margin)
```

If `C0` does not cover every input, a deterministic priority maximal radius-net
is selected from the uncovered points. These residual centers are also observed
input points, lie farther than `R` from `C0` and from each other, and use the
same radius as the initial centers. The final rulebook therefore has real
overlap and complete input coverage without clamping offsets or appending an
out-of-radius closure edge.

The two independent public controls are:

- `radius_margin` (default `1e-2`), which scales the analytical radius;
- `kernel_size`, which sets the representable integer offset range.

The safe odd kernel is `2 * ceil(radius_scaler) + 1`. For stride 8 the default
geometry requires K15. An insufficient or non-cubic kernel raises an error;
offsets are never clamped or aliased.

`build_full_cover_strided_rulebook` returns center coordinates and source
indices, batch metadata, forward and exact reverse rulebooks, neighbor counts,
radius telemetry, initial/residual center counts, selector rounds, the original
point-to-initial-center map, and per-input coverage. The high-level adapter
caches the exact reverse graph for matching unpatchification.

## Center-Nearest Grid Downsampling

`grid_sample_filter(..., reduction="center_nearest")` now reuses sorted grid
segments and selects the stable nearest observed point with a compact Triton
segmented argmin. The selector retains the lower source index on exact distance
ties. Batch identity is part of the integer grid key, and signed coordinates
use the same floor-based cell semantics, so positive and negative quadrants do
not split one logical batch/cell incorrectly.

The inverse point-to-cell map is built only when `return_mapping=True`. Callers
that need only the representative points avoid that allocation. The public
`center_nearest_impl="torch"` override remains as a reference and CPU fallback;
CUDA `auto` selects the Triton path.

## Two-Phase Convolution Geometry

`GeometryScheduler` and `ConvOp` separate feature-independent rulebook
construction from feature compute:

```python
from internals.two_phase import GeometryScheduler
from layers.two_phase_conv import ConvOp

ops = [ConvOp(conv0, stride=2), ConvOp(conv1, stride=1)]
y = GeometryScheduler().run(ops, x, metadata, compile_segments=True)
```

Within a separable segment, all geometry bundles are built first under the
forward-scoped triplet cache. The break-free `apply` chain can then be compiled
as one feature-compute segment. `ForceFused` is the parity oracle that restores
the original interleaved build/apply order. This is an execution-structure API;
v1.5.0 makes no standalone speed claim for it.

## Comparison-Fairness Audit

All reported comparisons are strict recipe-matched operator comparisons.

| Comparison | Held identical | Only changed variable |
|---|---|---|
| v1.4.0 shifted lookup versus v1.5.0 sorted-eight | exact real points/queries, batch IDs, radius, `sqrt(distance²) <= R` predicate, distance output, GPU, synchronization | candidate-index implementation |
| H200 backend selection | exact real points/queries and geometry contract | pre-rollout auto, sorted-27, or sorted-eight backend |
| Triplet preparation | real points, queries, sorted-eight search, kernel geometry, compatibility `k`, TIG-ready output | query-major post-sort versus direct tap segments |
| v1.4.0 versus v1.5.0 downsampling | exact transformed points, cell size, center-nearest objective, batch composition, public API | tagged lookup/reduction implementation versus segmented Triton selector and optional inverse |
| Full-cover treatment | K15, radius, initial center-nearest `C0`, features, convolution, GPU | missing-edge closure with no new tokens versus residual centers with no closure |

The full-cover control builds the same forward and reverse metadata as the
treatment. This matters: timing only its forward rulebook would unfairly charge
the treatment for unpatchification metadata.

## Correctness Scope

Correctness is established before timing. The release tests cover:

- positive and negative coordinates, boundaries, empty inputs, dense cells,
  and multi-batch isolation;
- exact neighbor-set and count equality against brute force and retained
  references for ball and Chebyshev distance;
- sorted-eight versus sorted-27 membership equality;
- canonical triplet, `seg_offs`, forward, input-gradient, and weight-gradient
  parity;
- center-nearest ties, signed coordinates, optional inverse mappings, and
  batch-aware keys;
- observed full-cover centers, deterministic ordering, radius-net separation,
  zero uncovered inputs, strict K15 offsets, exact reverse edges, and fp32,
  fp16, and bf16 convolution parity;
- regression coverage for historical strided convolution and upsampling APIs.

The focused release-candidate suite passed 132 tests, followed by a clean CUDA
extension build and Pointcept overlay build. The H200 radius gate independently
passed 97 CUDA tests before timing.

## Performance

The cross-version tables execute the public v1.4.0 tag and the v1.5.0
candidate in fresh processes over tensors prepared once from real ScanNet
scenes. Exact neighbor sets/counts and center-nearest geometry are checked
before timing. The main sweep uses 500 warmups and three measured blocks; every
cell above 5% IQR/median is replaced by its complete targeted recheck (1,000
warmups/five blocks, then 2,000 warmups/seven blocks where needed). Every final
cell is at or below 5% dispersion. The H200 backend-selection, direct-segment,
and full-cover studies retain their original 20-warmup, three-repetition
protocols. Arms are always sequential and no performance claim uses synthetic
points.

### Exact v1.4.0 shifted lookup → v1.5.0 sorted-eight — RTX 5880 Ada

The strict comparison uses the actual v1.4.0 shifted eight-cell lookup source
and CUDA bucket kernel from tag `8515aadac`, versus the v1.5.0 production
sorted-eight path. It covers four stages of the same real transformed ScanNet
B=2 workload and four radius/grid ratios (16 cells). Values are geometric means
within each radius regime; memory is incremental peak allocated memory above
the resident input tensors.

| `R / grid_size` | v1.4.0 shifted lookup | v1.5.0 sorted-eight | Speedup | Peak allocation v1.4.0 → v1.5.0 |
|---:|---:|---:|---:|---:|
| 1.00 | 0.714 ms | **0.429 ms** | **1.664x** | 8.28 → **7.26 MiB** (12.4% lower) |
| 1.86 | 0.780 ms | **0.426 ms** | **1.832x** | 26.97 → **8.29 MiB** (69.3% lower) |
| 2.75 | 1.019 ms | **0.450 ms** | **2.262x** | 64.56 → **10.01 MiB** (84.5% lower) |
| 4.50 | 1.768 ms | **0.541 ms** | **3.269x** | 198.02 → **15.76 MiB** (92.0% lower) |
| **All 16 cells** | **1.001 ms** | **0.459 ms** | **2.179x** | **41.11 → 9.87 MiB** (76.0% lower) |

All 16 cells have exact unique `(query, point)` set and per-query count parity.
This table deliberately forces the v1.4.0 shifted lookup because that is the
legacy algorithm being replaced. The tagged v1.4.0 `auto` dispatcher sometimes
selects tiled brute force; its tiled ball predicate compares `distance² <= R²`,
while its shifted lookup computes `sqrt(distance²) <= R`. Those two v1.4.0
paths can disagree by boundary ULPs (observed at `R/grid_size=4.5`), so a tagged
`auto` timing would fail the correctness-before-timing gate and is not silently
mixed into this causal table.

### Backend selection — H200

This separate pre-rollout study chooses between the new sorted-grid candidates;
it is not the v1.4.0 release comparison. The matrix contains seven production
dataset labels and four radius/grid ratios, for 28 cells.

| `R / grid_size` | Pre-rollout `auto` | Sorted-27 | Sorted-eight | Sorted-eight speedup vs pre-rollout `auto` |
|---:|---:|---:|---:|---:|
| 1.00 | 1.747 ms | 1.144 ms | **0.948 ms** | **1.843x** |
| 1.86 | 1.866 ms | 1.149 ms | **0.915 ms** | **2.040x** |
| 2.75 | 1.452 ms | 1.168 ms | **0.973 ms** | **1.493x** |
| 4.50 | 1.590 ms | 1.448 ms | **1.187 ms** | **1.340x** |
| **All 28 cells** | **1.657 ms** | **1.221 ms** | **1.001 ms** | **1.656x** |

Sorted-eight had no cell with IQR/median above 5% and won 27 of 28 cells against
sorted-27. The exception was the first measured ScanNet `R/grid_size=1` cell:
1.297 ms versus 1.224 ms, or 5.96% slower. The identical-shape replay and a
preceding independent run favored sorted-eight, but the exception is retained
rather than waived post hoc. Sorted-27 therefore remains an explicit
alternative.

Three pre-rollout-auto comparator cells exceeded 5% dispersion, so that
comparison is an aggregate backend-selection result, not a worst-cell or
cross-release claim. At `R/grid_size=4.5`, sorted-eight peak allocated memory
was at most 49.9% of shifted lookup and 12.5% on a geometric-mean ratio basis.
Across all 28 cells, geometric-mean incremental allocation was 48.76 MiB for
sorted-eight, 73.72 MiB for sorted-27, and 164.92 MiB for shifted lookup.

On the real ScanNet B=2, width=1 PointConv stem with 5,836,947 triplets:

| Builder | Complete stem | Rulebook build |
|---|---:|---:|
| Sorted-eight | **5.163 ms** | **2.824 ms** |
| Sorted-27 | 5.397 ms | 3.017 ms |
| Compact shifted lookup | 5.970 ms | 3.620 ms |
| Shifted lookup | 6.574 ms | 4.224 ms |

Downsampling and convolution compute were unchanged in this stem comparison.

### Direct tap-segmented triplet preparation — RTX 5880 Ada

The total includes search, compatibility-`k` materialization, and construction
of the TIG-ready index on both sides.

| Dataset | Workload | Points / edges | Query-major control | Direct segments | Speedup | Peak allocation control → direct |
|---|---|---:|---:|---:|---:|---:|
| ScanNet | K3, stride 2 | 328,326 / 1,084,685 | 1.041 ms | **0.919 ms** | **1.133x** | 96.07 → 53.58 MiB |
| ScanNet | K15, stride 8 | 328,326 / 1,025,950 | 0.877 ms | **0.641 ms** | **1.369x** | 90.08 → 36.18 MiB |
| nuScenes | K3, stride 2 | 45,630 / 70,479 | 0.791 ms | **0.610 ms** | **1.297x** | 10.59 → 10.33 MiB |
| nuScenes | K15, stride 8 | 45,630 / 90,542 | 0.746 ms | **0.500 ms** | **1.492x** | 8.10 → 9.13 MiB |
| Waymo | K3, stride 2 | 211,966 / 434,922 | 1.055 ms | **0.875 ms** | **1.205x** | 47.65 → 44.90 MiB |
| Waymo | K15, stride 8 | 211,966 / 539,987 | 0.790 ms | **0.588 ms** | **1.343x** | 48.35 → 30.02 MiB |

The geometric-mean speedup is **1.301x**. Total-pipeline IQR/median stayed below
5% in every cell. Memory is lower in five cells; the sparse nuScenes K15 cell is
the disclosed exception at +12.7%.

### Exact v1.4.0 → v1.5.0 center-nearest downsampling — RTX 5880 Ada

The control calls the complete public `grid_sample_filter` from v1.4.0; the
treatment calls the same API from v1.5.0. Eight transformed ScanNet scenes are
combined into batch sizes 1, 2, 4, and 8, then sampled at four real hierarchy
stages. Inputs and target cell sizes are identical. The tables show
complete-operation speedup and incremental peak-allocation reduction.

| Batch | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---:|---:|---:|---:|---:|
| 1 | 1.776x | 1.571x | 1.508x | 1.523x |
| 2 | 2.172x | 1.796x | 1.538x | 1.553x |
| 4 | 2.927x | 1.947x | 1.638x | 1.532x |
| 8 | **3.221x** | 2.185x | 1.683x | 1.527x |

| Batch | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---:|---:|---:|---:|---:|
| 1 | 31.2% | 25.3% | 24.2% | 33.8% |
| 2 | 30.6% | 25.5% | 24.9% | 23.9% |
| 4 | 29.1% | 25.5% | 25.0% | 24.4% |
| 8 | 32.7% | 26.2% | 25.0% | 24.5% |

Across all 16 cells, geometric-mean latency is 0.520 ms for v1.4.0 and
0.285 ms for v1.5.0: a **1.827x** speedup. Geometric-mean incremental peak
allocation falls from 7.19 MiB to 5.24 MiB, or **27.1%**; per-cell reductions
range from 23.9% to 33.8%.

The selected voxel keys and minimum center distances match in every cell. Two
of millions of representatives use different source indices (`B4/S0` and
`B8/S0`), and both are exact equal-distance ties in the same voxel. v1.5.0's
stable lower-source-index tie rule makes those choices deterministic; no
non-tied representative changes. This explicit equivalence check replaces an
incorrect assumption of bit-identical source indices.

### Full-cover treatment cost — RTX 5880 Ada

The strict control and treatment use K15, stride 8, the same analytical radius,
the same initial centers, C=64 fp16 convolution, and identical forward/reverse
metadata. The sweep covers eight real ScanNet scenes at 2 cm and 5 cm
voxelization (16 cells).

| Grid | Residual centers / initial | Selector rounds | Edge increase | Median rulebook ratio | Median incremental-memory ratio |
|---:|---:|---:|---:|---:|---:|
| 2 cm | 0–0.163% | 0–1 | 0–0.211% | 1.565x | 1.156x |
| 5 cm | 0–0.635% | 0–1 | 0–0.568% | 1.192x | 1.144x |

The treatment added 29 centers in total, replacing 70 out-of-radius closure
edges from the control. The rulebook overhead pays for proving coverage and,
when needed, selecting residual centers. Convolution forward+backward measured
0.988x at the median and 0.987x geometrically, but several roughly 1 ms cells
remained above the 5% dispersion threshold even after deeper reruns. We
therefore make no convolution-speed claim; the reliable result is structural:
the treatment's edge increase is at most 0.568%, while coverage becomes exact
by construction.

## Reproducing the Measurements

The release includes public benchmark entry points:

```shell
python benchmarks/operators/bench_geometry_version_compare.py --help
python benchmarks/operators/bench_resnet_stem_breakdown.py --help
python benchmarks/operators/bench_triplet_preparation_real.py --help
python benchmarks/operators/bench_grid_downsample_real.py --help
python benchmarks/operators/bench_full_cover_strided.py --help
```

The cross-version driver additionally requires detached v1.4.0 and candidate
worktrees; it records both exact SHAs and runs each revision in a fresh process.
Each real-scene benchmark requires an explicit dataset root or coordinate glob.
The full-cover benchmark exposes `--warmups`, `--reps`, and `--inner`; reported
latency is per operation and peak memory is incremental above the live baseline.

## Limitations

- Neighbor ordering is not stable and is not part of the API contract.
- Sorted-27 is retained for target-device studies; no unmeasured embedded-GPU
  dispatch rule is claimed.
- Full-cover residual selection contains synchronization required by its
  deterministic iterative control flow. It is an opt-in geometry treatment,
  not a free replacement for every strided convolution.
- Geometry-stage gains do not directly predict whole-network wall time; model
  speedup depends on how much time the workload spends constructing geometry.
