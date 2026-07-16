# Fused Point Convolution

`v1.4.0` adds the fused point convolution engine for exact point-native sparse
convolution. The implementation lives in `sparse_engines/fused_point_conv.py`;
its production entry point is `fused_gather_sum_conv3d`, and the scheduler it
uses is the fused gather-sum schedule.

The operator keeps the same triplet semantics used by PointCNN++: each triplet
`(i, j, k)` means input point `j` contributes to output point `i` through kernel
slot `k`.

## Why This Helps

Real point neighborhoods are often non-injective. Multiple input points can
contribute to the same `(output, kernel-slot)` bucket. A flat triplet schedule
handles those contributions independently and then reduces them, which repeats
work when several inputs share the same destination bucket.

The fused gather-sum schedule first accumulates the bucket's input features, and
then applies the slot weight once:

```text
bucket(i, k) = sum over j in N(i, k) of x[j]
out[i] += bucket(i, k) @ W[k]
```

Here **fused** has a precise meaning: the schedule fuses the gather/reduce work
for all triplets sharing the same `(output, kernel-slot)` bucket before the
channel matmul. A flat schedule would materialize or process each `(i, j, k)`
contribution separately and then reduce multiple contributions into `out[i]`.
The fused point conv path instead builds one bucket row for `(i, k)`, gathers the
corresponding input features `x[j]`, sums them in the kernel, and contracts that
sum with `W[k]` once.

This is **not** fusing the entire forward/backward training step into one kernel.
Forward, grad-input, and grad-weight remain separate kernels/passes, but they
share the same bucketed gather-sum idea:

- forward fuses per-bucket input gathering and summation before `bucket @ W[k]`;
- grad-input applies the same fused gather-sum schedule to the transposed
  rulebook;
- grad-weight fuses per-bucket feature summation before the outer-product
  reduction against output gradients.

This reduces redundant multiply-reduce work without changing which points
contribute to the convolution.

## Operator Semantics

The fused point conv path preserves the exact point-native neighborhood used by
the existing operator:

- no voxelization is introduced;
- no neighbor contribution is dropped;
- the output is parity-checked against the flat reference path;
- the optimization changes scheduling, not the mathematical convolution.

The same bucket structure applies to the three training kernels:

- **Forward**: sum the input features in each `(i, k)` bucket, multiply by
  `W[k]`, and accumulate into output `i`.
- **Grad-input**: run the same gather-sum structure on the transposed rulebook.
- **Grad-weight**: reduce bucketed feature sums against output gradients, using
  the deterministic backward infrastructure introduced in `v1.3.0`.

## Relationship To VVOR

`v1.3.0` and `v1.4.0` both improve the weight-gradient side, but they attack
different levels of the same VVOR computation. `v1.3.0` introduced segment VVOR:
once the triplets for a kernel slot are known, one program reduces a weight
segment without atomics, making wide-channel weight gradients deterministic and
faster.

`v1.4.0` keeps that deterministic backward foundation and changes what is fed
into it for non-injective point neighborhoods. Instead of forming an outer
product for every triplet `(i, j, k)`, fused point conv first sums the input
features that share the same `(i, k)` bucket and then contributes one
outer-product term for that bucket:

```text
grad_W[k] += bucket(i, k)^T @ grad_out[i]
where bucket(i, k) = sum over j in N(i, k) of x[j]
```

So the v1.4 VVOR win is part of the fused gather-sum idea: it is the
weight-gradient version of multiplicity reduction. It is not a separate
mathematical operator and not a replacement for v1.3 segment VVOR; it composes
with the v1.3 deterministic reduction path by reducing the number of
outer-product contributions that path has to process.

## Dispatch Policy

The release default is the production `auto` route. It selects the best known
composition of TIG, fused point conv, cached rulebooks, and width-sensitive
overflow handling for the supported real-data shapes. The fused point conv path
is used where it is consistently faster; the router falls back to the older TIG
path where the fused schedule is not the measured winner.

`force_fused_gather_sum` and its short spelling `force_fgs` are diagnostic
benchmark modes. They are useful for isolating the fused point conv kernel, but
release numbers should be reported from the shipped `auto` route.

The route stays compile-safe: policy decisions depend on static operator
properties such as channel width, dtype, kernel size, and the submanifold
contract, not on per-iteration tensor lengths.

## Large-Radius Strided Stems

`v1.4.0` also includes a scheduling-only improvement for the large-radius tiled
radius search used by strided ResNet-style stems such as `conv7x7x7, stride=2`.
This path keeps the same radius-neighborhood rulebook and changes only how the
candidate points are blocked for the tiled search. It is separate from fused
point conv: fused point conv reduces multiplicity inside within-stage
convolutions, while the large-radius stem change reduces overhead in the
rulebook construction that precedes a strided stem convolution.

This optimization is allowed under the same semantic contract as the fused
engine: the set of neighbor triplets is unchanged, point contributions are not
dropped, and validation uses exact rulebook parity plus normal floating-point
output tolerance before timing.

### v1.5.0 geometry follow-up

The paragraph above describes the v1.4.0 release state. In v1.5.0, the
production radius-search `auto` route is the compact exact-eight sorted-grid
backend for both ordinary and large-radius calls. It preserves the same exact
neighbor contract while removing the shifted-grid candidate expansion and its
large-query chunk/concatenate workaround. The v1.4 tiled backend remains an
explicit diagnostic path, not the automatic strided-stem route. Against the
exact v1.4.0 shifted lookup, the new path measures a 2.179x geometric-mean
speedup and 76.0% lower geometric-mean incremental peak allocation on the
16-cell real-ScanNet matrix. See
[`sorted_grid_geometry.md`](sorted_grid_geometry.md) for that cross-release
table and the separate matched H200 stem/backend-selection results.

## Performance Summary

The release numbers below use the shipped `auto` route on real ScanNet scenes.
They compare `v1.4.0` with `v1.3.0` under the same point-native triplet semantics;
no voxelization or neighbor dropping is introduced. Forward-only and
forward-plus-backward are reported separately because the fused gather-sum path
changes both the forward schedule and the VVOR-side weight-gradient schedule.

### Operator-Level Speed

The strongest v1.4.0 operator gains are at `C=64`--`C=256`, where real point
neighborhoods have enough `(output, kernel-slot)` multiplicity for gather-sum to
avoid repeated channel contractions. `C=512` is near the TIG crossover, so the
production route falls back where TIG is faster.

| Scope | Card | Data / batch | Precision | v1.4.0 vs v1.3.0 |
|---|---|---|---|---:|
| Conv forward | RTX 5880 Ada | real ScanNet, val B=12 | fp16 | 1.85-3.93x at C64-C256; C512 1.08x |
| Conv fwd+bwd | RTX 5880 Ada | real ScanNet, train B=6 | fp16 | 1.73-3.21x at C64-C256; C512 0.96x |
| Conv forward | H200 | real ScanNet, val B=24 | fp16 | 1.63-2.30x at C64-C256; C512 1.00x |
| Conv fwd+bwd | H200 | real ScanNet, train B=12 | fp16 | 1.40-1.57x at C64-C256; C512 0.98x |

The bf16 route is intentionally conservative in this release and should be read
as a TIG-band policy, not as the fused-path headline.

### Whole-Backbone Speed

End-to-end gains depend on how much of the iteration is spent in eligible
point-conv stages and rulebook construction. Conv-heavy ResUNet benefits most.
ResNet-style models are more diluted, but the final release also includes a
semantic-preserving large-radius stem scheduling update so the large-batch
training rows are no longer a regression.

| Scope | Card | Data / batch | Precision | v1.4.0 vs v1.3.0 |
|---|---|---|---|---:|
| ResNet34 x1 train | RTX 5880 Ada | real ScanNet, B=6 | fp16 | 1.05x |
| ResNet34 x2 train | RTX 5880 Ada | real ScanNet, B=6 | fp16 | 1.06x |
| ResNet34 x1 train | H200 | real ScanNet, B=12 | fp16 | 0.99x |
| ResNet34 x2 train | H200 | real ScanNet, B=12 | fp16 | 1.00x |
| ResUNet train | RTX 5880 Ada | real ScanNet, B=4 | fp16 | 1.94x |
| ResUNet train | H200 | real ScanNet, B=12 | fp16 | 2.59x |

The public table reports `v1.4.0` against `v1.3.0`, the immediately previous
release with a comparable fp16 operator stack. Earlier-release fp16 comparisons
should be regenerated from the original release tags, or from the first release
that shipped a comparable fp16 path, before being used as public claims. Small-
batch validation rows are also intentionally not the headline: fixed overheads
and non-convolution work dominate more strongly there.

## Benchmarking Protocol

`v1.4.0` performance numbers should be read under the same protocol used for the
prior operator releases:

- real scenes only;
- production preprocessing;
- forward-only and forward-plus-backward reported separately;
- Ada and H200 reported separately;
- parity gate before timing;
- warmup-discarded median timing;
- no synthetic-data performance headline.

The public release report compares PointCNN++ versions against each other on the
same point-native operator semantics.

## Relationship To Earlier Releases

`v1.4.0` is a continuation of the same point-native triplet operator line, not
a new mathematical convolution and not a voxelized substitute. The releases are
layered as follows:

- `v1.1.0` reorganized the per-triplet execution into full-segment grouped
  generation. It improved how triplets are staged and grouped, but the compute
  model still fundamentally paid work per triplet contribution.
- `v1.2.0` introduced TIG, the triplet implicit-GEMM execution path. TIG made
  the forward path much faster by turning the flat triplet stream into a better
  matrix-style schedule, while preserving the same `(i, j, k)` rulebook
  semantics.
- `v1.3.0` added segment VVOR, a deterministic no-atomic weight-gradient path.
  This closed the wide-channel backward bottleneck left after TIG: each weight
  segment is reduced by one writer rather than by contended atomics.
- `v1.4.0` adds fused point conv with gather-sum scheduling. It composes with
  the previous pieces: the router still uses TIG where TIG is the measured
  winner, keeps the deterministic backward infrastructure from `v1.3.0`, and
  uses fused point conv where non-injective neighborhoods make multiple triplets
  share the same `(output, kernel-slot)` bucket. Its VVOR-side improvement is
  this same bucketed gather-sum idea applied to `grad_W`: fewer bucketed
  outer-products reach the deterministic segment reducer.

The key new lever in `v1.4.0` is therefore **multiplicity reduction**: when
several triplets differ only by input point `j` but share `(i, k)`, their input
features are summed before the channel matmul. Earlier releases optimized the
flat triplet execution; `v1.4.0` reduces the amount of channel work needed for
non-injective point neighborhoods.
