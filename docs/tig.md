# TIG — triplet implicit GEMM conv kernels

*Released in `v1.2.0`.*

`v1.2.0` replaces the production execution backend of every `PointConv3d`
with **TIG** (triplet implicit GEMM) — the third kernel generation for the
MVMR/VVOR primitives — and makes it the automatic default at every shape,
group count, and dtype. Training steps (forward + backward) get **2–5×
faster than the v1.0.0 per-triplet baseline** and 1.1–2.7× faster than the
v1.1.0 grouped path at the layer level on real scans, with zero new
global-memory materialization and a fully sync-free, CUDA-graph-capturable
op path. This document explains the design — including the first attempt
that was *rejected*, and why — how it was benchmarked, and the results.

New to the convolution itself? Read `docs/ADVANCED.md` § MVMR first.
`docs/grouped_kernels.md` (v1.1.0) explains the previous generation and
defines the recurring vocabulary (triplets `(i, j, k)`, MVMR, VVOR, `C`/`M`,
the PTv3 `enc0…enc4` ladder, Triton / tensor cores / CUTLASS) — those
definitions are not repeated here.

## Background and terminology

A few terms are new in `v1.2.0`; they are defined once, here.

- **k-sorted triplet list** — `build_triplets` emits the triplet list in
  kernel-slot-major order: all triplets sharing weight slot `k` are
  contiguous. `v1.1.0` exploited this through a separately materialized
  segment-offset table; TIG goes further — the sorted list *is* the index.
- **Fan-in** — the number of input points contributing to one
  `(output point, kernel slot)` pair. An exact-match voxel convolution is
  *injective*: fan-in ≤ 1 by construction. The general point-native operator
  computed here uses radius-search neighborhoods, so fan-in ranges 0–8 over
  the 27 taps of a 3×3×3 kernel on real scans. This one property drives the
  whole design (next section).
- **Submanifold vs. strided/generative** — a *submanifold* convolution maps
  a point set onto itself (`N_in == N_out`, same coordinates) — the bulk of
  a ResNet/PTv3 trunk. Strided (downsampling) and generative (upsampling)
  convolutions change the point set; they keep the eager v1.1.0 path.
- **Groups `G`, per-group width `Cg = C / G`** — grouped convolution splits
  the channels into `G` independent blocks: a block-diagonal weight tensor
  `(K, G, Cg, Mg)`.
- **Implicit GEMM; weight- vs. output-stationary** — an *implicit GEMM* runs
  a matrix multiply over operands never materialized as dense matrices; rows
  are gathered on the fly through an index. *Output-stationary* programs own
  a tile of output rows and gather whatever feeds them; *weight-stationary*
  programs pin one weight matrix in registers and stream the triplets that
  use it. Which is right depends on the sparsity structure.

### The three generations

| Generation | Release | Execution structure |
|---|---|---|
| **PT** — per-triplet | v1.0.0 | one program walks L triplets, scalar/fma math |
| **FSG** — full-segment grouped | v1.1.0 | k-segment batched `tl.dot` tiles; C-reduction split across programs, partial-sum atomics; Triton + CUTLASS/WMMA tiers |
| **TIG** — triplet implicit GEMM | v1.2.0 | weight-stationary flat walk over the k-sorted triplet list; fp16/bf16 operands fed natively to tensor-core dot with an fp32 accumulator; the **C-reduction stays in registers** inside one program; **one** fp32 atomic per chunk×M-tile; sync-free upper-bound launch grid |

The FSG→TIG inversion is the headline: FSG splits the channel reduction
across programs (more parallelism per tile, more atomic traffic), TIG keeps
it in registers (minimal atomics, deeper per-program work, one autograd
node, 3 kernel launches per forward+backward).

## Why an implicit-GEMM path

### The first design — and why it was rejected

The obvious starting point was the approach specialized *injective*
voxel-convolution engines use: an **output-stationary masked-tiling**
schedule. Sort the output rows so
rows with similar tap-occupancy bitmasks share a tile, then run an implicit
GEMM where each tile computes only its active taps. For the exact-match
voxel operator this works beautifully — injectivity bounds each row's mask
by local occupancy, so a good row order makes tiles nearly mask-homogeneous
and the padding (tile slots computed but masked off) stays small.

We prototyped exactly this transform first, before writing any kernel: tile
heights `B1 ∈ {16, 32, 64, 128}` crossed with a family of row orderings,
from natural order through total and packed per-tap counts to gray-coded
occupancy masks, measured as *padding factor* — slots a tile schedule must
compute divided by real triplets. It does not transfer to the general
operator. On real ScanNet scans (three scenes × five stage grids), the
**best** ordering in the family still pads **2.3–3.6×**; per-fan-in-level
decomposition pads 1.5–2.7×, a hybrid (binary level-0 mask + a flat residual
pass) 1.4–1.9×, and even pure *binary* masks 1.6–2.5× — far above what the
same machinery achieves on exact-match voxel masks. The mechanism is
structural: **24–50% of `(output, tap)` groups — and 41–71% of the triplet
*mass* — have fan-in > 1** (up to 8), and expanding rows into per-slot tile
entries under that distribution produces row signatures combinatorially too
diverse for *any* single total order to make fixed-height tiles homogeneous.
Masked tiling is the right answer to the injective problem and the wrong
answer to the general one.

One measurement deserves emphasis: **synthetic near-injective clouds pad
only ~1.1×** (deduplicated uniform clouds have 1–16% of groups at
fan-in > 1), so a synthetic-only evaluation would have green-lit the masked
design that real data kills. Every execution-structure decision behind TIG
was therefore gated on real scans — the same reason the benchmark section
insists on them.

### The inversion

So invert the schedule. Make the kernel **weight-stationary**: pin one
weight matrix per program and walk the k-sorted triplet list *flat*, in
order, with no row reordering and no slot expansion. Every operand row a
program touches is a real triplet — **zero padding by construction**, at any
fan-in distribution. The price — scattered atomic output writes — is reduced
below to one fp32 atomic per chunk×M-tile.

Two facts complete the argument. First, padded MACs are not the right
arbiter anyway: the deep stages of a point backbone are tiny (the `c512`
regime below is 942 points / ~13.7K triplets), so wall-clock there is
**latency- and launch-governed**, and a schedule with fewer launches, no
host syncs, and deeper per-program work wins even against padding-free
competitors of equal MAC count. Second, the masked idea is not discarded
wholesale: 57–79% of real-scan triplet mass sits at fan-in level 0, and a
binary-masked level-0 pass with a *cached* index wins some large-N/small-C
forward cells — it survives as a non-default sub-path (`_tig_masked_kernel`)
kept off the training path by its index-transform cost. What the padding
data refutes is output-stationary masked tiling as *the general structure* —
not masking as a tool.

## Kernel design

TIG is pure Triton — three kernels plus packed variants, one autograd node,
three launches per forward+backward. No CUTLASS dependency in the TIG tier.

### Forward — the flat kernel

- **The sorted triplets are the index.** The kernel consumes the production
  triplet list verbatim (zero copy, no rulebook transform). The only
  per-call index work is **one `torch.searchsorted`** (~0.07–0.14 ms) to
  find the K segment boundaries.
- **Weight-stationary walk.** Each program owns one `(k, M-tile)` pair and a
  chunk of slot `k`'s segment: gather input rows through the `j` indices,
  multiply against the register-resident weight tile, scatter into `i`.
- **Native half-precision into the tensor cores.** fp16/bf16 operands are
  fed *directly* to the tensor-core `tl.dot`; only the accumulator is fp32
  (upcasting first demotes the MMA to TF32 at half rate — see the fairness
  note below).
- **The C-reduction stays in registers.** One program carries the full
  channel reduction in a register C-loop — where FSG splits it across
  programs and pays partial-sum atomic traffic, TIG pays **one fp32
  `atomicAdd` per (chunk, M-tile)** at the epilogue.

### Backward — two role swaps, no new index

**grad-input is the same kernel**: exchange the gather/scatter roles
(`i ↔ j`) and read the weight through a stride-transposed view — zero copy,
no transposed index, no second code path. **grad-weight is an implicit outer
product with Split-K**: chunks of each k-segment accumulate
`input ⊗ grad_output` partials in registers and combine via per-chunk fp32
atomics into `grad_weight[k]`. (A fused one-pass backward kernel exists but
measured slower at every stage — grad-input atomics dominate — so the
dispatcher routes the split form.)

### Sync-free launch — CUDA-graph capturable

The number of chunks per segment depends on device data; reading it back to
size the launch grid would cost a device→host sync per call. TIG instead
launches an **upper-bound grid** — `ceil(T / L) + K` programs — with
in-kernel early exit. No launch parameter depends on device data, no
`.cpu()`/`.item()` exists anywhere in forward or backward, and the op chain
is **CUDA-graph-capturable by construction**. The forward's one transient
allocation is an fp32 accumulation buffer (peak forward memory ≤1.07× the
previous path; forward+backward peak at parity).

### Groups > 1 — every engine tier

`v1.2.0` brings full `groups > 1` support to every tier (previously `G > 1`
silently fell back to per-triplet in every engine):

- **TIG**: a native group axis in all three kernels — per-group pointer
  offsets, block-diagonal math, `(K, G, Cg, Mg)` weights end-to-end — plus a
  **packed multi-group tile mode** for tiny per-group widths: at
  `Cg ∈ {8, 16}` a single per-group dot would pad up to the tensor-core tile
  floor, so the packed kernels place **2 or 4 groups side by side in one
  dot**. This fixes the one regime where the flat path lost: narrow-group
  shapes (`C=64, G=8`, i.e. `Cg = 8`) fell behind per-triplet on tile
  padding — the masked design's killer, returned — and now win (0.47× vs PT
  on Ada, 0.20× on H200; grid below).
- **FSG Triton**: native `G > 1` (per-group dot, multi-group block configs).
- **FSG CUTLASS tiers**: fold-G-into-K single-launch wrappers over the
  frozen kernels — 1.9–3.2× faster than a per-group loop — including the
  fused single-autograd-node mode.
- **WMMA / scalar tiers**: per-group loops (correctness coverage; never the
  `G > 1` winner).

## How it was benchmarked

Two benchmarks in `benchmarks/operators/` answer the v1.1.0 notes' two
questions: *where does the speedup come from?* and *does it survive
integration?*

### Operator-level — `bench_tig_groups.py`

- **Input data** — a real ScanNet v2 validation scene (`scene0011_00`),
  preprocessed through Pointcept's `GridSample(grid_size=0.02 m)` exactly as
  production training does. *Why real data:* the fan-in measurements above —
  synthetic clouds are near-injective and flatter tile-friendly schedules.
- **Shapes** — three `(C, grid)` regimes standing in for the encoder ladder:
  `c64` (C=64 at 0.04 m grid → 58,818 points / 781,094 triplets — shallow),
  `c256` (C=256 at 0.16 m → 3,895 / 56,841 — mid), `c512` (C=512 at 0.32 m →
  942 / 13,738 — deep). Triplets per point is ~13–15 at every level.
- **Grid** — `G ∈ {1, 2, 4, 8}` × `{fp16, bf16}` (fp32 at `G ∈ {1, 4}`),
  engines `force_pt` / `force_fsg` / `force_tig` plus every FSG hardware
  tier, layer-level through `PointConv3d` (per-call index build included on
  every side).
- **Timing** — CUDA events, 5 warmup + 12 measured iterations, median of 3
  runs per cell.
- **Fairness note** — before the head-to-head, the FSG kernels were given the
  same native-dtype operand feed TIG uses (they previously upcast fp16/bf16
  to fp32 before `tl.dot`, demoting the MMA to TF32 at half rate). The fix
  moved FSG only 1–5% — the generation gap is structural (segment walk,
  C-split atomics, three-op autograd composition), not a dtype artifact. The
  comparison below is against the *squeezed* FSG.

### End-to-end — `bench_resnet_grid.py`

Same harness and protocol as the v1.1.0 notes: PointConv3d-ResNet18/34/50
(`BasicBlock × [2,2,2,2]` / `[3,4,6,3]` / `Bottleneck × [3,4,6,3]`) ×
channel scales `{0.25, 0.5, 1.0, 2.0}×` × `{fp16, fp32, bf16}`, on the same
real scene (237,360 raw points → 164,833 after `GridSample(0.02 m)`),
forward+backward, CUDA events, 3 warmup + 8 measured iterations,
trim-median. The mode set grows to four: `tig` / `grouped` / `per_triplet` /
`auto`. As in v1.1.0 this is a single-binary intra-run A/B — same compiled
kernels, same scene, same RNG; only the dispatch path varies. Both cards
return: a workstation **RTX 5880 Ada** (`sm_89`) and a data-center **H200**
(`sm_90`).

## Performance — operator level

### Headline — TIG vs per-triplet (PT, v1.0.0) at G=1

Layer-level forward+backward time ratio vs **PT**, fp16, real ScanNet (lower
is better; reproduce with `bench_tig_groups.py`):

| | c64 (shallow) | c256 (mid) | c512 (deep) |
|---|---|---|---|
| Ada, G=1 | 0.48× | 0.25× | 0.26× |
| H200, G=1 | 0.24× | 0.17× | 0.17× |

In absolute terms the deep cells are sub-millisecond: `c512` fp16
forward+backward is 3.64 ms under PT vs 0.89 ms under TIG on Ada, 3.38 ms vs
0.59 ms on H200. The H200 ratios are uniformly stronger — TIG's deeper
per-program work scales with the newer card's tensor cores, while the
per-triplet baseline stays launch-bound on both.

### The full (shape, G) decision grid

TIG forward+backward as a fraction of PT, fp16 (from the v1.2.0 operator
decision tables; bf16 lands in the same bands, within ±0.03):

| Card | Shape | G=1 | G=2 | G=4 | G=8 |
|---|---|---|---|---|---|
| Ada | c64 | 0.48 | 0.78 | 0.63 | 0.47 |
| Ada | c256 | 0.24 | 0.38 | 0.52 | 0.71 |
| Ada | c512 | 0.25 | 0.33 | 0.47 | 0.62 |
| H200 | c64 | 0.24 | 0.42 | 0.34 | 0.20 |
| H200 | c256 | 0.17 | 0.26 | 0.39 | 0.55 |
| H200 | c512 | 0.17 | 0.26 | 0.40 | 0.56 |

Two reading notes. The ratio degrades gracefully with `G` at mid/deep shapes
(each group's matmul narrows) but **never crosses 1.0** — grouped
convolution is now strictly cheaper than its per-triplet execution
everywhere. And the `c64 / G=8` corner (`Cg = 8`, the shape that lost to PT
before the packed tiles landed) is now among the *strongest* cells: packing
2–4 groups per tensor-core dot recovers the occupancy a per-group dot wastes
at width 8.

Counting winners across all 30 (shape, G, dtype) cells per card — every
engine and hardware tier competing — **TIG takes 25/30 on Ada and 26/30 on
H200**, with runner-up margins of 1.0–1.4× (Ada) and 1.1–1.6× (H200) at the
mid/deep cells. Every half-precision exception goes to the fused FSG mode at
a near-tie (≤1.03× on Ada; one narrow-group bf16 cell at 1.15× on H200); the
remaining exceptions are shallow **fp32** cells (`c64`), where the fused FSG
path keeps a 1.06–1.40× edge — fp32 has no tensor-core dot to feed, so TIG's
native-operand advantage vanishes in the launch-bound corner. The
network-level fp32 grids below still come out in TIG's favor.

### vs FSG (v1.1.0), and FSG's residual edge

At the cells above, TIG is **1.1–2.7× faster than FSG forward+backward at
the layer level on real scans** (operator-level at G=1: 1.0–1.4× on Ada,
0.9–1.4× on H200, growing with C). FSG's one remaining strength is
**forward-only** at grouped mid/deep shapes: its C-split structure extracts
more parallelism per launch when there is no backward to amortize, leading
TIG's forward by 7–14% at most grouped `c256`/`c512` cells on Ada (one bf16
cell reaches 21%) and by 6–18%, scattered, on H200. Forward+backward, TIG
still leads every one of those cells; forward-only deployments can pin
`force_fsg` there.

## Performance — end-to-end

### Speedup grids — `auto` (v1.2.0) over the engine generations

End-to-end forward+backward speedup of `auto` over v1.0.0 (`per_triplet`) /
v1.1.0 (`grouped`), RTX 5880 Ada, fp16:

| | 0.25× | 0.5× | 1.0× | 2.0× |
|---|---|---|---|---|
| ResNet18 vs v1.0.0 / v1.1.0 | 1.04 / 1.01 | 1.03 / 1.07 | 1.00 / 0.99 | **1.57 / 1.11** |
| ResNet34 vs v1.0.0 / v1.1.0 | 1.00 / 1.00 | 1.06 / 1.05 | 1.17 / 1.07 | **1.75 / 1.08** |
| ResNet50 vs v1.0.0 / v1.1.0 | 1.01 / 1.00 | 1.07 / 1.04 | 1.06 / 1.01 | **1.41 / 1.00** |

(fp32 follows the same shape, up to 1.53× / 1.19× at 2.0×.) On **H200** the
same grid widens further — `auto` over v1.0.0 / v1.1.0, fp16:

| | 0.25× | 0.5× | 1.0× | 2.0× |
|---|---|---|---|---|
| ResNet18 | 1.02 / 1.02 | 1.08 / 1.03 | 1.19 / 1.07 | **1.71 / 1.05** |
| ResNet34 | 1.04 / 1.07 | 1.13 / 1.07 | 1.19 / 1.04 | **2.02 / 1.08** |
| ResNet50 | 1.01 / 1.04 | 1.02 / 1.07 | 1.14 / 1.10 | **1.53 / 1.08** |

### The headline cells, in milliseconds

Forward+backward wall time at channel scale 2.0×, fp16, real scene:

| Card | Depth | v1.0.0 `per_triplet` | v1.1.0 `grouped` | v1.2.0 `auto` | vs v1.0.0 | vs v1.1.0 |
|---|---|---:|---:|---:|---:|---:|
| Ada | ResNet18 | 223.92 ms | 159.13 ms | 142.94 ms | 1.57× | 1.11× |
| Ada | ResNet34 | 314.47 ms | 194.48 ms | 179.84 ms | 1.75× | 1.08× |
| Ada | ResNet50 | 293.15 ms | 208.62 ms | 208.20 ms | 1.41× | 1.00× |
| H200 | ResNet18 | 215.44 ms | 133.13 ms | 126.34 ms | 1.71× | 1.05× |
| H200 | ResNet34 | 327.69 ms | 175.20 ms | 162.12 ms | 2.02× | 1.08× |
| H200 | ResNet50 | 294.76 ms | 208.86 ms | 192.82 ms | 1.53× | 1.08× |

The other dtypes tell the same story: H200 ResNet34 × 2.0× runs
324.56 → 165.66 ms in bf16 (1.96×) and 301.10 → 169.17 ms in fp32 (1.78×).
Peak VRAM moves with the routing, not against it (973 MB under `auto` vs
1150 MB under `per_triplet` at H200 ResNet18 × 2.0× fp16).

### Winner analysis — and why `auto` beats even `force_tig`

Across the full 36-cell grid (3 depths × 4 scales × 3 dtypes) per card,
**TIG is the best single force mode at 30/36 cells on both cards**, and
`auto` matches or beats the best single force mode at **34/36 on both
cards**. The `auto` misses are small launch-bound corners — ResNet18 × 0.25×
on Ada (fp16, bf16), ResNet18 × 0.25× and ResNet50 × 0.50× bf16 on H200 —
all within 9% of the best mode.

`auto` frequently outruns *pure* `force_tig` because it routes **per
layer**, not per network: TIG for the submanifold convolutions, the eager
path for the strided downsample layers (which TIG does not cover) — e.g.
H200 ResNet34 × 2.0× fp16: `force_tig` 168.77 ms, `auto` 162.12 ms; Ada
ResNet34 × 1.0× fp16: 169.44 ms vs 163.62 ms.

### Why the operator-level speedup (2–6×) compresses to 1.4–2.0× end-to-end

Same explanation as the v1.1.0 notes, one generation later. The
operator-level ratio captures the MVMR/VVOR matmul cost — exactly what TIG
restructures. End-to-end wall time also includes `build_triplets`, radius
search, voxelization, the downsample layers, norm/activation, and autograd
traversal — identical in every mode, a **fixed floor** that compresses the
visible ratio. That is why the small-scale columns sit near 1.0× (the floor
dominates) and the engine delta emerges with width: at production-relevant
channel counts the v1.2.0 dispatcher delivers **1.4–2.0× over v1.0.0** end
to end, on a real scene a user can recognize in their own workload. The
floor is also the next frontier: index construction (`build_triplets`) is
now the largest remaining term at shallow stages.

## Numerical contract

TIG accumulates in fp32 regardless of operand dtype (native fp16/bf16 into
the tensor-core dot, fp32 accumulator, epilogue, and atomics). Measured fp16
outputs sit ~5–10× closer to the fp64 reference than the fp16 path of the
specialized voxel engine measured above. All parity batteries run against an
fp64 block-diagonal oracle — 186 tests + 275 subtests on Ada, 128 + 205 on
H200 — covering ragged per-group widths, empty kernel-offset segments,
non-power-of-two tiles, and mixed-dtype AMP boundaries (mixed pairs promote
losslessly).

## The automatic dispatcher

`"auto"` (the production default) now routes **TIG** whenever both guards
pass: **k-sorted triplets** (the `build_triplets` contract, verified by a
memoized check; callers that guarantee ordering pass `assume_sorted=True`)
and **submanifold conv** (`N_in == N_out`). Generative or strided
convolutions and unsorted callers take the eager v1.1.0 path unchanged —
including under `force_tig`, which is "auto minus the guards" plus the same
eager fallback for downsample layers (the fallback is why `force_tig` is
safe network-wide, and why `auto` still edges it out by routing per layer).
The previous auto rules — grouped at `C ≥ 128`, fused at `C ≥ 512` — are
retired: TIG strictly beat the legacy router at every measured cell, all
dtypes included, so no shape gate remains. Every engine stays explicitly
reachable for benchmarking, ablation, and rollback:

```python
from sparse_engines._dispatch_override import dispatch_mode

with dispatch_mode("force_pt"):    ...   # v1.0.0 per-triplet
with dispatch_mode("force_fsg"):   ...   # v1.1.0 grouped Triton
with dispatch_mode("force_tig"):   ...   # v1.2.0 TIG (== auto, minus the guards)
# plus the FSG hardware tiers: force_fsg_cutlass_{mvmr,vvor,mvmr_vvor},
# force_fsg_wmma[_coop]_vvor, force_fsg_fused
```

In short: **training — TIG everywhere** (what `auto` does); **forward-only
inference at grouped mid/deep shapes** — FSG can still lead by ~7–21%, pin
`force_fsg` for those layers if it matters; **unsorted triplets /
non-submanifold convs** — the eager path (PT or FSG by the v1.1.0
thresholds), automatically.

## Implementation map

| Layer | Path | Files |
|---|---|---|
| Kernels | TIG flat fwd / grad-input, Split-K weight-grad, packed variants, masked level-0 sub-path | `sparse_engines/tig.py` (`_tig_flat_kernel`, `_tig_flat_packed_kernel`, `_tig_wgrad_kernel`, `_tig_wgrad_packed_kernel`, `_tig_masked_kernel`) |
| Index | k-segment boundaries, sorted check, hybrid split | `sparse_engines/tig.py` (`TigIndex`), `sparse_engines/_seg_offs.py` |
| Wrapper | autograd node + launch configs | `sparse_engines/tig.py` (`tig_forward`, `tig_grad_input`, `tig_grad_weight`) |
| FSG G>1 | native-G Triton + fold-G CUTLASS/WMMA wrappers | `sparse_engines/mvmr_grouped_cuda.py`, `vvor_grouped_cuda.py`, `vvor_grouped_wmma{,_coop}.py`, `mvmr_cutlass.py`, `vvor_cutlass.py` |
| Dispatch | auto-router + force modes | `sparse_engines/_dispatch_override.py`, `layers/conv.py` |
| Tests | TIG parity (fp64 oracle); grouped parity, all tiers | `tests/unittest/test_tig_{forward,backward}.py`, `test_fsg_groups.py`, `test_hwtier_groups.py` |
| Bench | operator grid; end-to-end ResNet grid | `benchmarks/operators/bench_tig_groups.py`, `bench_resnet_grid.py` |

## Reproducing the numbers

The layer-level decision grid (point `TIG_BENCH_SCANNET` at a
Pointcept-preprocessed ScanNet val directory containing `<scene>/coord.npy`;
the release numbers use `scene0011_00`), the end-to-end backbone grid, and
the correctness batteries:

```bash
TIG_BENCH_SCANNET=/path/to/scannet_v2/val \
    python benchmarks/operators/bench_tig_groups.py

python benchmarks/operators/bench_resnet_grid.py \
    --dtype all --depths 18,34,50 --scales 0.25,0.5,1.0,2.0 \
    --scene-coord /path/to/scannet_v2/val/scene0011_00

pytest tests/unittest/test_tig_forward.py tests/unittest/test_tig_backward.py \
       tests/unittest/test_fsg_groups.py tests/unittest/test_hwtier_groups.py
```

Without `--scene-coord` / `TIG_BENCH_SCANNET` the benches fall back to a
synthetic cloud for smoke runs — see the fan-in discussion above for why
release numbers never come from synthetic data.

## Citation

```bibtex
@misc{li2025pointcnnperformantconvolutionnative,
  title         = {PointCNN++: Performant Convolution on Native Points},
  author        = {Lihan Li and Haofeng Zhong and Rui Bu and Mingchao Sun and Wenzheng Chen and Baoquan Chen and Yangyan Li},
  year          = {2025},
  eprint        = {2511.23227},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2511.23227}
}
```
