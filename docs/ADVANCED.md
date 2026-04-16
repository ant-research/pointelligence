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

**Minimum radius for upsampling:** the `Upsample` layer has each high-res point search for nearby low-res points. The low-res points are spaced at `grid_size_low` (one representative per voxel from center_nearest downsampling). In the worst case, a high-res point sits at the corner of a low-res voxel — distance `sqrt(3)/2 * grid_size_low` (~0.87 * `grid_size_low`) from the nearest low-res point. So the search radius must exceed this:

```
radius_min  = grid_size_low * sqrt(3)/2   ≈ 0.87 * grid_size_low
radius_used = grid_size_low * radius_scaler
```

For the default configuration (kernel_size=3, ball mode), `radius_scaler ≈ 1.86` (see the formula below), which comfortably exceeds the minimum.

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
