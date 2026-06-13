# Core Concepts

This guide covers the key ideas behind Pointelligence. Combined with the reference implementations in `models/`, these should be enough to build custom architectures from the library's primitives.

## Ragged Tensors

Point clouds have variable sizes per sample. Instead of padding to the largest sample, we concatenate all points into one flat tensor and track boundaries with a `sample_sizes` array:

```text
Padded (wasteful):                    Ragged (what we do):

Sample 1:  [P][P][.][.][.][.][.]      [P][P] [P][P][P][P][P][P][P] [P][P][P][P]
Sample 2:  [P][P][P][P][P][P][P]      \___/  \___________________/  \_________/
Sample 3:  [P][P][P][P][.][.][.]       S1(2)          S2(7)            S3(4)
                 [.] = wasted
                                      sample_sizes = [2, 7, 4]
```

All layers in the library operate on this ragged format. The `MetaData` dataclass carries the spatial bookkeeping (points, sample indices, grid size) through the network.

## Downsampling

Voxel-based downsampling via `grid_sample_filter` with `center_nearest` mode: partition space into a grid, keep the point closest to each voxel center. This preserves thin structures better than random selection.

The `grid_size` parameter is the fundamental spatial unit (analogous to pixel size). Strides and receptive fields are defined as multiples of `grid_size`. A smaller `grid_size` gives higher resolution at greater compute cost.

**Tip:** apply a preliminary downsample at `grid_size / 3` before the network to handle irregular point density.

## Upsampling

Unlike images where target pixel locations are known, point cloud upsampling targets must be specified explicitly. The standard practice is to reuse the pre-downsampled points as targets. The calling pipeline is responsible for retaining these.

The `Upsample` layer supports two modes for choosing which low-res points contribute to each high-res output.

### Fresh search (default)

Each high-res query searches among the low-res sources within a sphere of radius `grid_size_low * radius_scaler`. The low-res points are spaced at `grid_size_low` (one representative per voxel from `center_nearest` downsampling).

**Minimum radius.** `center_nearest` picks an existing point, not the voxel center. In an isolated voxel the representative can sit at one corner while a high-res query sits at the opposite corner — the full voxel diagonal. So the search radius must exceed this:

```
radius_min  = grid_size_low * sqrt(3)   ≈ 1.73 * grid_size_low
radius_used = grid_size_low * radius_scaler
```

For the default configuration (`kernel_size=3`, ball mode), `radius_scaler ≈ 1.86` (see the formula below), which clears the bound by ~7%. `Upsample.forward` emits a `UserWarning` when `receptive_field_scaler < ~0.81` (i.e. `radius_used < radius_min`).

### Cached inverse — `Upsample(straight_recover=True)`

`conv_with_stride` already builds triplets `(i, j, k)` during downsample, where `i` indexes the low-res query and `j` the high-res neighbor (per the `(i, j)` convention above). These are cached on the parent `MetaData` with `i` and `j` swapped — `i_upsample = m.j`, `j_upsample = m.i`, `k_upsample = m.k` — so they read as HR-query / LR-source for the reverse direction. With `straight_recover=True`, `Upsample` reuses these verbatim: no radius search at upsample time, and no warning check.

The cached triplets carry the **downsample step's** `kernel_size`, `receptive_field_scaler`, and `distance_type`. With `straight_recover=True`, the Upsample's own arguments for those are not applied — the neighborhood geometry and the `k`-binning are inherited from the matching downsample. The Upsample's `kernel_size` must therefore match the downsample's, otherwise the convolution weight tensor is shaped for a different `k`-range than the cached indices.

Coverage is inherited too: with the default `radius_scaler ≈ 1.86` on the downsample, the cached path gives the same `sqrt(3)`-clearing coverage as the fresh-search path. Enable `straight_recover=True` when the decoder is structurally symmetric with the encoder (same `kernel_size`, neighborhood radius, and `distance_type`) and the matching downsample triplet lives on `m_low.parent`. Otherwise leave it `False` (the default) and let `Upsample` search independently.

Note: strided **downsampling** does not have this concern — the low-res query points are a subset of the high-res source points, so every query trivially finds at least itself as a neighbor.

## Neighborhoods: (i, j) Pairs

Fixed-radius search (not KNN) finds all neighbors within a radius of each query point. The result is a flat list of `(i, j)` pairs — query point `i`, neighbor point `j`:

```text
             [-SAMPLE 1-]   [--------SAMPLE 2--------]   [----SAMPLE 3----]
Query (i):     0    1        2  3      4      5           6       7    8
               |    |        |  |      |      |           |       |    |
               |-.  |-.      |  |-.-.  |-.-.  |-.         |--.    |-.  |-.
               | |  | |      |  | | |  | | |  | |         |  |    | |  | |
               v v  v v      |  v v v  v v v  v v         v  v    v v  v v
Neighbor (j):  0 1  0 1      2  3 4 5  3 4 5  3 4         6  7    6 7  6 8

Flattened:
  i: [0 0  1 1  2  3 3 3  4 4 4  5 5  6 6  7 7  8 8]
  j: [0 1  0 1  2  3 4 5  3 4 5  3 4  6 7  6 7  6 8]
```

Points are globally indexed across the batch, but edges never cross sample boundaries.

## Convolution Triplets: (i, j, k)

To route each neighbor through the right kernel weight, we extend `(i, j)` to `(i, j, k)` where `k` is the kernel weight index:

```text
  i: [0 0  1 1  2  3 3 3  4 4 4  5 5  ...]
  j: [0 1  0 1  2  3 4 5  3 4 5  3 4  ...]
  k: [4 5  5 4  4  4 6 8  2 4 5  1 3  ...]
```

How `k` is computed: the neighbor's position relative to the query point is transformed into a local coordinate frame, then discretized into a `kernel_size^3` voxel grid. The voxel index becomes `k`. This is done by `voxelize_3d` in `layers/triplets.py`.

### `kernel_size` vs `receptive_field_scaler`

PointCNN++ decouples kernel resolution from the receptive field's physical span — unlike voxel-based convolution where the two are coupled to the voxel grid resolution.

- **`kernel_size`** (e.g., 3 or 5): how many cells partition the kernel weight grid. A 3^3 kernel has 27 distinct weights; a 5^3 kernel has 125.
- **`receptive_field_scaler`**: a volume multiplier that controls how much physical space the search sphere covers, relative to the `kernel_size^3` cube.

The search radius is computed by `radius_scaler_for_kernel_size()`:

```
search_volume = kernel_size^3 * receptive_field_scaler
radius_scaler = (3 * search_volume / (4 * pi)) ^ (1/3)      # for ball mode
search_radius = grid_size * radius_scaler
```

For kernel_size=3, receptive_field_scaler=1.0, ball mode: `radius_scaler = (3 * 27 / (4 * pi))^(1/3) ≈ 1.86`.

This means you can independently choose:
- A **5^3 kernel over a small region** (fine-grained weights, tight neighborhood)
- A **3^3 kernel over a large region** (coarse weights, wide neighborhood)

The voxelization always operates at `grid_size` scale (recovered inside `build_triplets` as `search_radius / radius_scaler`), so changing `receptive_field_scaler` only affects which points are found, not how they are binned into kernel cells.

## MVMR: The Convolution Operator

The actual convolution is a sparse **M**atrix-**V**ector **M**ultiplication and **R**eduction:

```
output[i] += weight[k] @ input[j]     for each triplet (i, j, k)
```

Where `weight` has shape `(K, G, C_in, C_out)`, `input` has shape `(N, G, C_in)`, and `output` has shape `(N_out, G, C_out)`. Triplets are sorted by `k` so consecutive threads access the same weight block — keeping the GPU saturated with dense arithmetic.

The backward pass uses **VVOR** (Vector-Vector Outer product and Reduction) to compute weight gradients: `grad_weight[k] += input[j] outer grad_output[i]`.

Both are implemented as autotuned Triton kernels in `sparse_engines/`.

## Putting It Together

```python
from layers.conv import PointConv3d, conv_with_stride
from layers.metadata import MetaData

# Wrap your point cloud in MetaData
m = MetaData(
    points=coords,           # (N, 3) float32
    sample_inds=sample_inds, # (N,) int64 — which sample each point belongs to
    sample_sizes=sizes,      # (B,) int64 — points per sample
    grid_size=0.05,
)

# Create a convolution layer
conv = PointConv3d(in_channels=64, out_channels=128, kernel_size=3)

# conv_with_stride handles triplet construction + convolution in one call
features_out, m_out = conv_with_stride(conv, features, m, stride=2)
# m_out now has downsampled points, updated grid_size, and new triplets
```

For full architecture examples, see `models/resnet.py`.

### torch.compile and the triplet contract

`PointConv3d.forward(input, i, j, k, n, contract=None)` takes an optional
`TripletContract` (`layers/contract.py`) describing the index's structural
facts — k-sortedness, exact-cover, uniform segments. The triplet builders
produce it (`MetaData._build_triplets`, `handle_stride*`, and
`GeneratedSites.to_contract()` all attach one to the `MetaData`), and
`conv_with_stride` threads `m.contract` through automatically — so the common
path needs no change. Because the forward READS these facts instead of
re-deriving them with host syncs (`.item()`), a `PointConv3d` traces cleanly
under `torch.compile(fullgraph=True)`:

```python
contract = m.contract                       # produced by the builder
compiled = torch.compile(conv, fullgraph=True)
out = compiled(features, m.i, m.j, m.k, m.num_points(), contract=contract)
```

`contract=None` defaults to `TripletContract.submanifold()` (k-sorted, no
exact cover) — the conv-path invariant. A caller that hand-builds
genuinely-**unsorted** triplets must pass `TripletContract(k_sorted=False)` to
opt into the eager per-triplet path (or use the `force_pt` dispatch mode);
a bare unsorted `MetaData` is otherwise treated as sorted. Set
`POINTELLIGENCE_DEBUG_CONTRACTS=1` to re-validate a contract against its data in eager.

## Caching triplets — the forward-scoped ambient cache (`internals/triplet_cache.py`)

Submanifold triplets `(i, j, k)` are a pure function of a point set's geometry,
search radius, kernel, query set, and sort order. Within a single forward the
*same* geometry+kernel recurs, and rebuilding the triplets each time repeats the
host-bound `radius_search` (the dominant preprocessing cost). The recurrences:

- **U-Net encoder/decoder cascade** — each resolution level is visited descending
  and again ascending (the decoder's `Upsample` restores `m.points` to the *exact*
  encoder-level tensor object via the `m.parent` chain), so the stride-1
  submanifold build at each level recurs.
- **Consecutive same-scale convs within a stage** — every stride-1 block in an
  encoder stage builds on the same un-reassigned `m.points`.
- **A stem that descends from, and a head that ascends back to, the input
  resolution** — the entry-scale build recurs on the way back up.

A **forward-scoped ambient cache** captures all of these with no per-call
plumbing. `triplet_cache_scope()` (a re-entrant context manager that also works as
the `@triplet_cache_scope()` decorator) installs a thread-local, per-forward dict;
`build_triplets` consults it directly. Opt a model in with one line on its
`forward`:

```python
from internals.triplet_cache import triplet_cache_scope

class MyBackbone(nn.Module):
    @triplet_cache_scope()                  # one fresh cache per forward call
    def forward(self, data_dict):
        ...
```

**Key.** Each entry is keyed on the full output determinant of `build_triplets`:
`points`/`sample_inds`/query identities (by `data_ptr`), `neighbor_radius`,
`radius_scaler`, `sort_by`, `return_num_neighbors`, and a kernel descriptor.
`data_ptr` identity is sound *within* a forward (the dict is fresh per scope and
the keyed tensors stay alive on the activation/`parent` graph); the entry stores a
`weakref` to `points`, and a hit whose weakref is dead is treated as a miss —
closing the freed-then-recycled-address window. `sample_inds` is keyed by
`data_ptr`, not a value fingerprint, because `tolist()` would force a D2H sync and
defeat the host-bound win.

**Transparency.** The cache is numerically identical to no caching: a miss runs
the unchanged build; a hit returns exactly a prior identical-key build (the key is
a complete determinant, so it can never conflate different batch partitions,
radii, or kernels). Outside any scope the cache is off (build every call — the
default for code that doesn't opt in). Set `POINTELLIGENCE_DISABLE_TRIPLET_CACHE=1`
to disable it even inside a scope (clean A/B + debug).

**Scope of effect.** The consult lives inside `build_triplets`, so only the
submanifold-search path is cached. The strided/downsample builds
(`handle_stride_and_build_triplets`), the disjoint-partition patchify
(`grid_sample_filter`, no radius search), the `max_pool3d` build, and the
attention-context builders are *separate* code paths and are not cached (by
design — each produces a unique result per call). A backbone benefits only when it
actually revisits a scale through `build_triplets`; one whose reuse is
attention-only or already served by cached upsample edges sees few or no hits, and
opting in is harmless either way.
