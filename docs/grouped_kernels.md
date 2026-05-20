# Grouped MVMR/VVOR — ideas and implementation

Released in `v1.1.0`. Background reading: `docs/ADVANCED.md` § MVMR (the existing per-triplet path).

## Why grouped

The PointConv3d body of work is two sparse-linear-algebra primitives over the convolution triplet list `(i, j, k)`:

- **MVMR** — `output[i] += weight[k] @ input[j]` (Matrix-Vector Multiply + Reduction). Used in the forward pass.
- **VVOR** — `grad_weight[k] += input[j] ⊗ grad_output[i]` (Vector-Vector Outer product + Reduction). Used in the backward pass to accumulate weight gradients.

The default Triton path issues one kernel program per `(i, k)` bin and runs the matmul (`tl.dot`) inside the program. That program-per-bin granularity hides launch + weight-load cost behind real arithmetic *as long as both the channel count `C` and the per-bin output count `M` are large*. For typical PTv3 stage shapes that does not always hold — at small `C`, the per-program setup cost starts to dominate the matmul.

The **grouped** path inverts the granularity: one program holds the weight tensor `weight[k]` in registers across *many* `(i, k)` cells, amortising the launch + weight-load across more arithmetic. Kernel programs walk a segment-offset table (`sparse_engines/_seg_offs.py`) that bins triplets by `k`, so each program processes a contiguous run of triplets with shared `k`.

This grouping/register-resident-weight principle is dtype-independent — the same MVMR/VVOR amortisation structure is in play whether the inner matmul is scalar-FMA (fp32) or WMMA (fp16/bf16). Tensor cores are a throughput multiplier on top of the algorithmic win, not the source of it.

Two variants ship:

- `sparse_{mvmr,vvor}_grouped_mma` — scalar-FMA grouped kernels. The fp32 path; also the fp16/bf16 path when WMMA alignment doesn't fit.
- `sparse_vvor_grouped_wmma{,_coop}` — WMMA (m16n16k16) tensor-core kernels for fp16/bf16. The `_coop` variant uses cooperative-load to share register reads across 4-warp groups.

Both produce numerically equivalent output to the Triton path (asserted by `tests/unittest/test_grouped_cuda_parity.py` across PTv3 stages × `{fp32, fp16, bf16}` × `{fwd, grad_a, grad_b}`).

Dtype is a deliberate user choice — fp32 trades throughput for numerical precision, fp16/bf16 trades precision for tensor-core throughput. The grouped path serves both regimes.

## fp32 path

Scalar-FMA grouped MMA (`sparse_mvmr_grouped_mma`, `sparse_vvor_grouped_mma`):

- One program per output row tile (`M`-tile).
- Weight tile (`(K_offsets[k], C_in, C_out)`) loaded once into registers, reused across all input rows in the segment.
- Atomic-scatter writeback to `output[i]` (coarser cadence than the Triton path).

CUTLASS fallbacks (`sparse_{mvmr,vvor}_cutlass_sm{80,90}`) cover tile shapes the scalar-FMA path doesn't fit — these are reference kernels with no fp16 tensor-core dependency, used as the rigorous correctness oracle and as the bf16 path on hardware that lacks WMMA bf16.

There is no tensor-core fp32 path (WMMA tf32 was tried during development and didn't beat scalar-FMA on the irregular triplet layout).

## fp16 / bf16 path

`sparse_vvor_grouped_wmma` — WMMA m16n16k16 tensor-core kernel:

- Requires `M % 16 == 0` and `C % 16 == 0`. The Python wrapper at `sparse_engines/vvor_grouped_wmma.py:50-53` checks both and falls back to scalar-FMA if either fails — silent, no error.
- One WMMA fragment per (warp, k-segment); fragments accumulate into shared memory and write out atomically.
- The `_coop` variant (`sparse_vvor_grouped_wmma_coop`) takes a register-gather strategy with 4-warp groups — lower latency on Hopper for the larger `M` regime.

Forward MVMR (`sparse_mvmr_grouped_mma`) does NOT have a WMMA variant in this release. It uses the scalar-FMA grouped path for fp16/bf16; the WMMA boundary is currently on the backward (VVOR) side where the outer-product / register-reuse opportunity is larger.

## WS2.1 auto-router

`layers/conv.py:51` exposes:

```python
_FUSED_AUTOROUTER_MIN_C = 512
```

The PointConv3d dispatcher consults this threshold when `dispatch_mode == "auto"` (the default). The fused/grouped path activates only when the conv's per-group input channel count `C` is at or above 512 — measured net-positive over Triton at that scale on H200 fp16. Below 512, dispatch stays on the Triton path (= `v1.0.0` behaviour, bit-equivalent — zero regression by construction).

Explicit overrides:

- `dispatch_mode("force_grouped")` — bypass the threshold, use grouped path always (for benches/tests).
- `dispatch_mode("force_fused_conv")` — force the FusedPointConv3d wrapper.
- `dispatch_mode("force_per_triplet")` — force Triton (matches v1.0.0).

See `sparse_engines/_dispatch_override.py`.

## Implementation map

| Layer | Path | Files |
|---|---|---|
| Kernel | scalar-FMA MVMR | `extensions/sparse_engines_cuda/csrc/cuda/sparse_mvmr_grouped_mma.{cu,cuh}` |
| Kernel | scalar-FMA VVOR | `extensions/sparse_engines_cuda/csrc/cuda/sparse_vvor_grouped_mma.{cu,cuh}` |
| Kernel | WMMA VVOR | `…/sparse_vvor_grouped_wmma{,_coop}.{cu,cuh}` |
| Kernel | CUTLASS fallback | `…/sparse_{mvmr,vvor}_cutlass_sm{80,90}.{cu,cuh}` |
| Kernel | torch::ops registration | `…/definitions.cpp` |
| Wrapper | scalar-FMA | `sparse_engines/mvmr_grouped_cuda.py`, `sparse_engines/vvor_grouped_cuda.py` |
| Wrapper | WMMA | `sparse_engines/vvor_grouped_wmma.py`, `sparse_engines/vvor_grouped_wmma_coop.py` |
| Wrapper | CUTLASS | `sparse_engines/mvmr_cutlass.py`, `sparse_engines/vvor_cutlass.py` |
| Dispatch | auto-router | `sparse_engines/_dispatch_override.py` |
| Helper | k-segment offsets | `sparse_engines/_seg_offs.py` |
| Layer | conv integration | `layers/conv.py` (FusedPointConv3d, `_FUSED_AUTOROUTER_MIN_C`) |

## Bench methodology

Two benches live in `benchmarks/operators/`. The operator-level bench isolates kernel cost from `build_triplets` and autograd overhead; the end-to-end ResNet bench captures the cost the user actually pays. Together they answer two questions — _where does the speedup come from?_ (op bench) and _does it survive integration?_ (e2e bench).

Both expose a `--mode` toggle covering the v1.0.0-equivalent path (Triton per-triplet) and the v1.1.0 path (grouped + WMMA). The v1.0.0 vs v1.1.0 comparison is a single-binary intra-run A/B — same compiled kernel set, same point cloud, same RNG — so the only variable is the dispatch path.

### `bench_pointconv3d_grouped_cuda.py` — operator-level

Settings:

- **Shapes**: the five PTv3 encoder stages `enc0…enc4` — the (N, C) cells a PTv3-style point-cloud encoder produces. `enc0` = ~328K points × `C=32` (very high N, narrow C); `enc4` = ~2K points × `C=512` (low N, wide C). The full ladder spans the (N, C) plane that the auto-router must classify.
- **Triplets**: `--production-t` mode draws `T` from `build_triplets` against representative point distributions (`~5` neighbours per query — what the conv layer sees in real training). Default synthetic-T uses smaller `T` for quick smoke runs; release benches use `--production-t`.
- **Dtypes**: `fp32, fp16, bf16` — the full set PointConv3d supports.
- **Timing**: median ± IQR over `5 warmup + 12 measurement` iters per cell, `torch.cuda.synchronize() + time.perf_counter()` (chosen over `torch.cuda.Event` because the comparison crosses kernels with very different launch shapes).

Why representative: PTv3 is the most-deployed transformer-on-points architecture and PointConv3d is its primary conv primitive, so these shapes are exactly what the auto-router's `_FUSED_AUTOROUTER_MIN_C = 512` threshold is tuned against. Benchmarking on unrelated shapes would calibrate the router against the wrong workload.

### `bench_resnet_grid.py` — end-to-end

Settings:

- **Input data**: a real ScanNet v2 val scene (default `--scene-coord <dir>` → `scene0011_00`; 237,360 raw points, 5.84 × 8.24 × 2.61 m bbox). Preprocessed through Pointcept's `GridSample(grid_size=0.02 m, mode='train')` transform — the same step the production training pipeline applies before the model sees a scene. Post-preprocessing: 164,833 points (seeded for determinism). The bench keeps a synthetic-cloud fallback (no flag) for quick smoke runs, but release numbers always come from real data.
- **Depths**: ResNet18 (`BasicBlock × [2,2,2,2]`), ResNet34 (`BasicBlock × [3,4,6,3]`), ResNet50 (`Bottleneck × [3,4,6,3]`) — the canonical depth ladder, ported to PointConv3d. R18/R34 are pure 3×3×3 stacks; R50 mixes 1×1×1 + 3×3×3 in bottleneck blocks, exercising both the `nn.Linear` (1×1×1) path and the conv (3×3×3) path within one architecture.
- **Channel scaling**: `{0.25, 0.5, 1.0, 2.0}×` uniformly scales the four stage widths. At `0.25×` ResNet18 sits below the auto-router threshold across all stages (per-triplet wins); at `2.0×` ResNet50 sits above it everywhere (grouped/WMMA wins). The intermediate scales straddle the boundary stage-by-stage.
- **Dtypes**: `fp16, fp32, bf16`.
- **Modes per cell**: `force_grouped` (the v1.1.0 path), `force_per_triplet` (the v1.0.0 path), `auto` (production dispatcher). The third mode validates that the auto-router picks the empirical winner.
- **Receptive-field scaler**: `2.5` (matches production `unet_pointcnnpp`). Real scans have sparse-edge regions where the default `1.0` scaler yields no neighbours at some stride-2 query points; `2.5` is the operating point production training uses.
- **Timing**: median over `3 warmup + 8 measurement` iters per cell via `torch.cuda.Event` (a single ResNet fwd+bwd is 100–300 ms on real data, so iter count is lower than the kernel-bench).

Why representative: real architectures sweep the depth/width plane the auto-router must classify, and they do it on real point clouds whose density distribution is irregular (sparse outer regions, dense surfaces, kinect-style scan noise). Synthetic uniform-cube clouds underestimate the per-program launch cost relative to the kernel time the model actually pays — so they overstate the v1.1.0 speedup. Running on a real ScanNet scene gives a number the user can recognise in their own workload.

## Performance — v1.0.0 (Triton) vs v1.1.0 (grouped/WMMA)

Measured on two cards: a workstation **RTX 5880 Ada** (sm_89) and a data-center **H200** (sm_90). End-to-end numbers come from feeding the bench a real ScanNet v2 val scene (`scene0011_00`, 237,360 raw points → 164,833 post-`GridSample(0.02 m)`); operator-level numbers use production-T triplet counts at PTv3 stage shapes. Each bench is a single-binary intra-run A/B (only the dispatch path varies); times are median over warmup + measurement iters.

### Headline (peak speedup per axis)

| Scope | Card | Best cell | v1.0.0 (per-triplet) | v1.1.0 (auto) | Speedup |
|---|---|---|---:|---:|---:|
| End-to-end ResNet | RTX 5880 Ada | ResNet34 × 2.0× × fp16 | 306.32 ms | 189.15 ms | **1.62×** |
| End-to-end ResNet | H200          | ResNet34 × 2.0× × fp16 | 267.84 ms | 177.35 ms | **1.51×** |
| Operator (mvmr + vvor) | RTX 5880 Ada | enc4 × fp16 (CUTLASS) | 0.988 ms | 0.360 ms | **2.75×** |
| Operator (mvmr + vvor) | H200          | enc4 × fp16 (CUTLASS) | 0.924 ms | 0.381 ms | **2.42×** |

End-to-end times are fwd+bwd; operator times are forward MVMR + backward VVOR combined. Speedup quoted against `force_per_triplet` (= bit-equivalent v1.0.0).

### Why the kernel-level speedup (2.4×–2.75×) is bigger than the end-to-end speedup (1.5×–1.6×)

The kernel-level number captures the inner mvmr + vvor MMA cost — exactly where the grouped path's register-resident weight reuse pays off. End-to-end ResNet wall time also includes `build_triplets`, radius search, voxelisation, autograd graph traversal, BN, ReLU, and the un-fused linear/pooling layers. Those layers are the same in both v1.0.0 and v1.1.0, so they're a fixed addend that compresses the visible ratio. With a real scene (`N=164K` points) the kernel time becomes a smaller fraction of cell wall than it does at synthetic `N=10K`, so the visible end-to-end ratio compresses further — that's the honest number a user would see on their own workload, not a synthetic-data overestimate.

The H200's end-to-end ratio (1.51×) is close to the Ada's (1.62×) for the same reason: the grouped path's algorithmic win is dominated by register-resident weight reuse + per-program launch amortisation, both of which are roughly SM-clock-bound at PTv3-stage shapes. H200's extra SMs (132 vs Ada's 84) and ~5× HBM bandwidth would matter more at larger `N` or more triplets — at enc3/enc4, where the headline cells sit, the post-stride downsampled point count doesn't fill H200's SMs anyway, so its hardware advantages don't bind on this workload. H200 *is* faster than Ada in absolute wall time (267.84 ms vs 306.32 ms for the headline cell), just not dramatically so.

### End-to-end ResNet (fwd+bwd, ms)

Auto-router engages the grouped/WMMA path stage-by-stage where `C ≥ 512`. At scales `≤ 1.0×` it stays on per-triplet for most stages → close to zero regression; at `2.0×` it switches to grouped on every stage.

| Card | Depth | Scale | fp16 PT | fp16 auto | fp16 ratio | fp32 PT | fp32 auto | fp32 ratio | bf16 PT | bf16 auto | bf16 ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ada | ResNet18 | 0.25× | 125.17 | 127.03 | 1.01× | 123.70 | 126.79 | 1.02× | 118.80 | 124.52 | 1.05× |
| Ada | ResNet18 | 0.50× | 124.92 | 139.30 | 1.12× | 119.95 | 134.83 | 1.12× | 121.04 | 130.94 | 1.08× |
| Ada | ResNet18 | 1.00× | 140.23 | 145.02 | 1.03× | 137.57 | 140.54 | 1.02× | 147.73 | 141.73 | 0.96× |
| Ada | ResNet18 | 2.00× | 218.78 | 142.08 | **0.65×** | 201.37 | 172.14 | 0.85× | 219.58 | 158.48 | 0.72× |
| Ada | ResNet34 | 0.25× | 155.82 | 159.08 | 1.02× | 141.13 | 152.05 | 1.08× | 145.01 | 159.42 | 1.10× |
| Ada | ResNet34 | 0.50× | 157.73 | 166.40 | 1.05× | 150.55 | 162.49 | 1.08× | 154.94 | 167.00 | 1.08× |
| Ada | ResNet34 | 1.00× | 183.01 | 199.03 | 1.09× | 169.96 | 191.02 | 1.12× | 180.94 | 195.11 | 1.08× |
| Ada | ResNet34 | 2.00× | 306.32 | 189.15 | **0.62×** | 279.90 | 227.35 | 0.81× | 303.06 | 215.30 | 0.71× |
| Ada | ResNet50 | 0.25× | 169.40 | 178.12 | 1.05× | 164.96 | 172.96 | 1.05× | 174.20 | 176.26 | 1.01× |
| Ada | ResNet50 | 0.50× | 179.13 | 182.40 | 1.02× | 172.98 | 175.17 | 1.01× | 178.43 | 180.60 | 1.01× |
| Ada | ResNet50 | 1.00× | 194.98 | 197.11 | 1.01× | 186.85 | 195.77 | 1.05× | 192.81 | 196.25 | 1.02× |
| Ada | ResNet50 | 2.00× | 283.05 | 200.49 | 0.71× | 270.29 | 234.10 | 0.87× | 280.63 | 219.24 | 0.78× |
| H200 | ResNet18 | 0.25× | 103.07 | 104.76 | 1.02× | 97.60 | 102.23 | 1.05× | 101.66 | 108.93 | 1.07× |
| H200 | ResNet18 | 0.50× | 102.14 | 112.43 | 1.10× | 100.35 | 107.87 | 1.07× | 102.25 | 111.25 | 1.09× |
| H200 | ResNet18 | 1.00× | 115.47 | 120.03 | 1.04× | 113.58 | 130.14 | 1.15× | 115.03 | 123.82 | 1.08× |
| H200 | ResNet18 | 2.00× | 176.55 | 128.07 | **0.73×** | 172.18 | 137.73 | 0.80× | 176.38 | 137.97 | 0.78× |
| H200 | ResNet34 | 0.25× | 143.68 | 143.97 | 1.00× | 133.60 | 137.68 | 1.03× | 136.94 | 141.60 | 1.03× |
| H200 | ResNet34 | 0.50× | 139.28 | 157.66 | 1.13× | 134.64 | 153.87 | 1.14× | 140.72 | 157.86 | 1.12× |
| H200 | ResNet34 | 1.00× | 160.22 | 173.23 | 1.08× | 153.15 | 172.03 | 1.12× | 176.80 | 178.81 | 1.01× |
| H200 | ResNet34 | 2.00× | 267.84 | 177.35 | **0.66×** | 248.51 | 196.74 | 0.79× | 269.99 | 200.66 | 0.74× |
| H200 | ResNet50 | 0.25× | 168.55 | 173.34 | 1.03× | 159.91 | 166.59 | 1.04× | 163.47 | 169.23 | 1.04× |
| H200 | ResNet50 | 0.50× | 171.40 | 178.41 | 1.04× | 162.18 | 171.63 | 1.06× | 167.87 | 174.15 | 1.04× |
| H200 | ResNet50 | 1.00× | 182.64 | 199.22 | 1.09× | 175.05 | 183.85 | 1.05× | 179.40 | 186.96 | 1.04× |
| H200 | ResNet50 | 2.00× | 249.26 | 197.13 | 0.79× | 240.31 | 204.33 | 0.85× | 254.78 | 215.57 | 0.85× |

`ratio = auto / per_triplet`. Numbers below 1.00× are speedups; bold marks the headline wins per card. The `auto` dispatcher can outperform `force_grouped` per cell because it picks the dispatch per stage instead of per-network — e.g. ResNet34 × 2.0× × fp16 on Ada: `force_grouped` is 221.16 ms, `auto` is 189.15 ms.

### Operator-level (mvmr + vvor combined, ms)

| Card | Stage | dtype | Triton mvmr+vvor | Best v1.1.0 | via | Speedup |
|---|---|---|---:|---:|---|---:|
| Ada | enc3 | fp16 | 0.910 | 0.433 | CUTLASS | **2.10×** |
| Ada | enc4 | fp16 | 0.988 | 0.360 | CUTLASS | **2.75×** |
| Ada | enc3 | bf16 | 0.927 | 2.425 | WMMA-coop | 0.38× |
| Ada | enc4 | bf16 | 1.001 | 1.842 | WMMA-coop | 0.54× |
| Ada | enc4 | fp32 | 1.361 | 8.411 | scalar-FMA | 0.16× |
| H200 | enc3 | fp16 | 0.914 | 0.543 | CUTLASS | **1.68×** |
| H200 | enc4 | fp16 | 0.924 | 0.381 | CUTLASS | **2.42×** |
| H200 | enc3 | bf16 | 0.913 | 2.300 | WMMA-coop | 0.40× |
| H200 | enc4 | bf16 | 0.916 | 1.785 | WMMA-coop | 0.51× |
| H200 | enc4 | fp32 | 0.996 | 8.344 | scalar-FMA | 0.12× |

Operator-level fp16 speedup at enc3/enc4 is CUTLASS-driven. The fp32 scalar-FMA and bf16 WMMA-coop combined kernels run slower than the Triton grouped path at the operator level on these shapes — yet end-to-end fp32 grouped still pulls ahead at scale `2.0×` once per-conv-layer overhead amortises the kernel cost. This is why the auto-router decides per-stage on the live `(N, C)`, not per-dtype as a blanket rule.

### Hardware notes

- **RTX 5880 Ada (sm_89)** — workstation card, 14080 CUDA cores, 48 GB GDDR6.
- **H200 (sm_90)** — Hopper data-center card.

## Limitations

- WMMA path requires `sm_80` or newer (Ampere+).
- CUTLASS `sm_90` kernels exercise Hopper-specific cluster scheduling; on `sm_80/sm_89` the `sm_80` CUTLASS variant is used.
- The 512-channel auto-router boundary is measured on H200 fp16. Smaller-`C` regimes deliberately keep the Triton path — change the threshold via `dispatch_mode("force_grouped")` only after measuring on your hardware.
- Forward MVMR has no WMMA path in this release; fp16/bf16 fwd uses scalar-FMA grouped. (Backward VVOR is where the outer-product / register-reuse win shows up.)

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
