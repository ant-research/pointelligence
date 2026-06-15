# Segment VVOR — the no-atomic weight-gradient kernel

*Released in `v1.3.0`.*

`v1.2.0` (`docs/tig.md`) made the **forward** fast: the TIG kernels fed fp16/bf16
operands straight to the tensor cores, kept the channel reduction in registers, and ran
2–5× faster than the v1.0.0 per-triplet baseline at the layer level on real scans. But it
left one half of the training step on an older footing. The **weight gradient** still ran
through an **atomic Split-K**: each kernel segment — all triplets that share one weight
slot `k` — was shattered across many programs that accumulated partial gradients with
`atomicAdd`. At the wide-channel and strided-stem stages that fragmentation, not compute
and not bandwidth, became the backward bottleneck.

`v1.3.0` closes the backward with the **segment VVOR** kernel: one program loops a weight
slot's entire segment with the outer-product accumulator held in registers and commits the
gradient with a single, non-atomic store. Each gradient element has exactly one writer, so
the result is **bitwise-deterministic** — a reproducibility property the atomic path never
had — and a lightweight **architecture-aware dispatch** routes to it only where it wins.
The deep-stage training step (forward + backward) goes up to **10.7× faster than v1.0.0 on
RTX 5880 Ada and ~21× on H200**, lifting the v1.2.0 figure by a further ~1.4×, at zero
accuracy cost. `v1.3.0` is a **backward-only** release: the forward and grad-input kernels
are the v1.2.0 code unchanged, so inference and validation numbers are identical to v1.2.0.

## Background and terminology

`docs/grouped_kernels.md` (v1.1.0) and `docs/tig.md` (v1.2.0) define the recurring
vocabulary — triplets `(i, j, k)`, MVMR / VVOR, `C` / `M`, fan-in, the k-sorted triplet
list, Triton / tensor cores. Those definitions are not repeated here. Two terms matter for
the backward:

- **VVOR (the weight gradient)** — a *vector-vector outer-product reduction*: for each
  weight slot `k`, sum `input ⊗ grad_output` over every triplet in that slot's segment.
- **k-segment** — the contiguous block of the k-sorted triplet list that shares one weight
  slot `k`. The weight gradient for slot `k` is the reduction over its k-segment.

### The execution generations

| Generation | Release | Weight-gradient (VVOR) execution |
|---|---|---|
| **PT** — per-triplet | v1.0.0 | scalar outer product, atomic accumulate |
| **FSG** — full-segment grouped | v1.1.0 | k-segment batched tiles; C-reduction split across programs, partial-sum atomics |
| **TIG** — triplet implicit GEMM | v1.2.0 | implicit-GEMM Split-K: per-chunk `tl.dot`, one fp32 atomic per chunk × output-tile |
| **segment VVOR** | **v1.3.0** | one program loops the whole k-segment in registers, single non-atomic store; **deterministic** |

## Why an atomic-free backward

The weight gradient admits two execution strategies. The **Split-K** strategy (v1.2.0)
splits each k-segment across many programs, each handling a chunk and adding its partial
gradient with atomics. This maximizes contraction parallelism — the right answer when
segments are long and channels are narrow. But at the wide-channel and high-volume stages
it inverts: the segment fragments into many small tiles, and the atomic traffic to combine
them dominates the wall-clock. The strided patchify-stem weight gradient — the worst case
— ran at a tiny fraction of compute peak, purely from contention.

The **segment** strategy inverts the schedule. One program takes a single
`(k, channel-tile, output-tile)`, iterates over the *entire* segment for that slot,
accumulates the outer product in registers, and writes the result **once** with a plain,
non-atomic store. There is no atomic epilogue and no cross-program combine. The cost is
that a short, wide segment must be long enough to amortize the per-program setup — which is
exactly the regime the dispatch selects for, and exactly the regime where the atomic path
was weakest. It is the weight-gradient analogue of the v1.2.0 plain-store forward.

## Kernel design

### The segment kernel

Each program owns one `(k, channel-tile, output-tile)` and walks slot `k`'s whole
k-segment, gathering the input and grad-output rows triplet by triplet and accumulating
`input ⊗ grad_output` in a register tile. At the end of the segment it stores the tile to
`grad_weight[k]` with a single non-atomic write. The forward's k-sorted triplet list is
consumed verbatim — the segment boundaries are the same `searchsorted` offsets the forward
already computes, so the backward adds no index-build cost.

### Determinism by construction

Because each program owns a disjoint `(k, channel-tile, output-tile)` and writes it once,
every weight-gradient element has exactly one writer. The result is **bitwise-reproducible**
— running the same step twice produces bit-identical weights — which the atomic Split-K
path could not guarantee (atomic accumulation order is nondeterministic).

### What is unchanged from v1.2.0

The forward (MVMR) and grad-input kernels are the v1.2.0 code, untouched. Only the
weight-gradient path changes, and only at the stages the dispatch routes to the segment
kernel; everywhere else the atomic Split-K path runs and is bit-identical to v1.2.0.

## Architecture-aware dispatch

The segment kernel wins only when the per-segment tile is large enough to be
tensor-core-efficient **and** the segment is short enough that its in-register loop beats
the atomic path's contraction parallelism — a joint condition on the channel width `C` and
the kernel volume `K` (27 for a 3×3×3 kernel, 125 for 5×5×5). Crucially the crossover is
**architecture-dependent**: on Hopper, much cheaper global atomics speed the Split-K path,
so the segment kernel must be more selective there. The gate is static — it depends only on
`C` and `K`, never on a per-iteration tensor length — so it is `torch.compile`-safe:

```
segment  iff  C ≥ C_wide   OR   (C ≥ C_hi  AND  K ≥ K_hi)

      (C_wide, C_hi) = (256, 128)  on sm_89 (Ada)
                     = (512, 256)  on sm_90 (Hopper)
       K_hi = 125

otherwise:  atomic Split-K  (the v1.2.0 path, bit-identical to v1.2.0)
```

This mirrors the forward-side plain-store path, which is also architecture-gated for the
same reason. The thresholds were measured on real scans at the production batch and verified
to route correctly on both cards — no stage where the dispatch picks the slower kernel.
Tuning them on synthetic data or at batch size one gives the wrong gate: the
segment-vs-atomic crossover moves with batch size and with the real fan-in distribution.

## How it was benchmarked

Real ScanNet v2 scenes only, never synthetic. (Synthetic point clouds are near-injective:
they get the fan-in distribution and the atomic-contention fan-in wrong — the exact
quantities this kernel restructures — so a synthetic evaluation systematically mis-measures
the backward.) Scenes are batched to the **production training batch**, because the
weight-gradient dispatch is a training-time decision. Both cards report: a workstation **RTX
5880 Ada** (`sm_89`) and a data-center **H200** (`sm_90`).

### Operator-level — `bench_v13_tig_kernels.py`, `bench_v13_dispatch_retune.py`

Kernel-isolated forward / grad-input / grad-weight, segment vs. atomic, per stage and
dtype, batched to the training batch (B=6 on Ada / B=12 on H200).
`bench_v13_dispatch_retune.py` sweeps the segment-vs-atomic crossover to derive and verify
the gate. CUDA events, warmup discarded, median of repeated runs.

### End-to-end — `bench_v13_versions.py`

The full v1.0.0 → v1.3.0 training step (forward + backward) and validation step
(forward-only) at each encoder stage, on the same real scenes. Training batch B=6 / B=12,
validation batch B=12 / B=24 (Ada / H200). A single-binary intra-run A/B — same compiled
kernels, same scenes; only the engine generation varies.

## Performance — weight-gradient kernel

### Headline — segment vs. atomic at the deep stages

At the wide-channel and high-volume stages, where the atomic path bottlenecked, the segment
kernel is **2.4–3.4× faster than the v1.2.0 atomic Split-K** (kernel-isolated, fp16, real
ScanNet, production batch). The strided patchify-stem weight-gradient kernel itself goes
4.2 → 0.3 ms (~16×) where it dominates the backward.

### The full (stage × architecture) grid

Segment vs. atomic, kernel-isolated, fp16. The ratio is `atomic / segment` time — greater
than 1 means the segment kernel is faster.

| stage | RTX 5880 Ada (B=6) | H200 (B=12) |
|---|---|---|
| c64  k3 | atomic (gated) | atomic (gated) |
| c128 k3 | atomic (gated) | atomic (gated) |
| c256 k3 | **1.30×** | atomic (gated) |
| c512 k3 | **3.24×** | **2.31×** |
| c256 k5 | **2.77×** | **1.65×** |
| c512 k5 | **2.48×** | **3.03×** |

"atomic (gated)" means the dispatch correctly keeps the atomic path there — the segment
kernel would regress, so those stages stay bit-identical to v1.2.0. The architecture
divergence is the dispatch doing its job: on Ada the gate routes `c256 k3` (and `c128 k5`)
to the segment kernel, where they win; on Hopper it routes the same shapes to the atomic
path, because there the segment kernel would *lose* (`c256 k3` measures 0.66×, `c128 k5`
0.47× if forced). One fixed threshold would mis-route one card or the other; the arch-aware
gate is correct on both.

## Performance — end-to-end

### Speedup grids — `v1.3.0` over the engine generations

Full conv training step (forward + backward) speedup over the v1.0.0 per-triplet baseline,
real ScanNet, fp16, across the encoder ladder (3×3×3 kernel). The segment backward only
changes the stages the gate routes to it; the rest are unchanged from v1.2.0.

**RTX 5880 Ada (sm_89), train B=6:**

| stage | v1.1.0 | v1.2.0 | v1.3.0 |
|---|---|---|---|
| c64 k3  | 2.27× | 2.28× | 2.28× |
| c128 k3 | 3.88× | 3.54× | 3.54× |
| c256 k3 | 4.43× | 5.60× | **6.25×** |
| c512 k3 | 4.35× | 7.30× | **10.66×** |

**H200 (sm_90), train B=12:**

| stage | v1.1.0 | v1.2.0 | v1.3.0 |
|---|---|---|---|
| c64 k3  | 5.98× | 5.39× | 5.39× |
| c128 k3 | 7.43× | 8.46× | 8.45× |
| c256 k3 | 7.47× | 12.30× | 12.29× |
| c512 k3 | 7.98× | 15.41× | **21.03×** |

The deep `c512` stage — where the atomic backward was weakest — is where the segment kernel
pays off most: **+46% over v1.2.0 on Ada (7.30× → 10.66×), +37% on H200 (15.41× →
21.03×)**. The architecture-aware dispatch is visible at `c256 k3`: on Ada the segment
kernel fires and v1.3.0 pulls ahead of v1.2.0 (5.60× → 6.25×); on Hopper the same stage is
gated to atomic, so v1.3.0 correctly equals v1.2.0 rather than regressing. Validation
(forward-only) is unchanged from v1.2.0 — the segment kernel never executes during
inference.

### Whole-ResNet (fwd+bwd) — the realistic backbone impact

Whole-network training step (PointConv3d-ResNet18/34/50, forward+backward) at width
scale 2.0×, fp16, vs the v1.0.0 per-triplet baseline — all four generations measured in
the same batched real-ScanNet config (`bench_v13_resnet.py`):

| Depth | Card | v1.1.0 | v1.2.0 | v1.3.0 |
|---|---|---|---|---|
| ResNet18 | RTX 5880 Ada (B=6) | 1.42× | 1.61× | 1.62× |
| ResNet34 | RTX 5880 Ada (B=6) | 1.79× | 2.05× | **2.11×** |
| ResNet50 | RTX 5880 Ada (B=6) | 1.53× | 1.73× | 1.77× |
| ResNet18 | H200 (B=12) | 1.85× | 2.35× | 2.36× |
| ResNet34 | H200 (B=12) | 2.44× | 3.16× | **3.18×** |
| ResNet50 | H200 (B=12) | 2.05× | 2.56× | 2.63× |

At the whole-network level v1.3.0 is only a few percent over v1.2.0 (e.g. ResNet34 ×2.0:
2.05× → 2.11× on Ada, 3.16× → 3.18× on H200), and that is the *expected* result for a
backward-only release — see the compression note below. The dramatic numbers are
inherently per-stage; this table is the honest whole-network figure alongside them.

### Why the kernel-level win is bigger than the end-to-end win

The kernel-level grid above isolates the weight-gradient matmul — exactly what the segment
kernel restructures. A full training step is three passes — forward, grad-input,
grad-weight — and v1.3.0 changes only grad-weight, and only at the wide-channel stages the
gate routes to the segment kernel. So a per-kernel 2.4–3.4× backward win (or the +37–46%
deep-*stage* training lift over v1.2.0) dilutes to a few percent across a whole ResNet,
where most layers are narrow-C (gated to the unchanged atomic path) and grad-weight is
roughly a third of the backward. A full step also pays for `build_triplets`, neighborhood
search, voxelization, norm/activation, and autograd traversal — identical in every
generation, a fixed addend that compresses the ratio further. This is the same
"kernel-level > end-to-end" compression the v1.1/v1.2 notes describe, one step further:
v1.3.0 optimizes not one kernel but one half of one pass of it.

### Hardware notes

- **RTX 5880 Ada (`sm_89`)** — workstation card; segment gate `(C_wide, C_hi) = (256, 128)`.
- **H200 (`sm_90`)** — Hopper data-center card; tighter gate `(512, 256)` because its cheap
  global atomics speed the Split-K path the segment kernel competes against.

## Numerical contract

- **Segment path:** bitwise-deterministic weight gradient (single writer per element), fp32
  accumulation regardless of operand dtype, parity with the atomic path within the fp16/bf16
  bands and against an fp64 block-diagonal reference. Precisions: fp32 (IEEE / TF32), fp16,
  bf16; mixed-dtype AMP pairs promote losslessly.
- **Gated-atomic stages:** bit-identical to v1.2.0.
- **Forward / grad-input:** unchanged from v1.2.0.

## Limitations

- **Backward-only.** Inference / validation is the v1.2.0 forward, unchanged. The segment
  kernel only ever runs during training.
- **Gated regime.** The segment kernel fires only at wide-`C` / high-`K` stages; narrow-`C`
  low-`K` layers keep the v1.2.0 atomic Split-K (where it is faster).
- **Architecture-specific thresholds.** The gate is tuned on Ada (`sm_89`) and Hopper
  (`sm_90`); a new architecture should re-measure the crossover before trusting the gate.

## Implementation map

| Layer | Path | Files |
|---|---|---|
| Kernel | no-atomic segment weight-gradient | `sparse_engines/tig.py` (`_tig_vvor_seg_kernel`) |
| Dispatch | architecture-aware segment-vs-atomic gate | `sparse_engines/tig.py` (`_seg_gate_params`, `tig_grad_weight`) |
| Forward / grad-input | unchanged from v1.2.0 | `sparse_engines/tig.py` |
| Tests | segment-vs-atomic parity (fp64 oracle), mixed-dtype | `tests/unittest/test_tig_backward.py` |
| Bench | kernel grid; dispatch crossover; per-stage all-versions; whole-ResNet all-versions | `benchmarks/operators/bench_v13_tig_kernels.py`, `bench_v13_dispatch_retune.py`, `bench_v13_versions.py`, `bench_v13_resnet.py` |

## Reproducing the numbers

Point `TIG_BENCH_SCANNET` at a Pointcept-preprocessed ScanNet val directory (scene
subfolders each containing `coord.npy` / `color.npy` / `normal.npy`):

```bash
TIG_BENCH_SCANNET=/path/to/scannet_v2/val \
    python benchmarks/operators/bench_v13_tig_kernels.py     --batch 6      # Ada operator grid
    python benchmarks/operators/bench_v13_dispatch_retune.py --batch 6      # gate crossover
    python benchmarks/operators/bench_v13_versions.py        --train-batch 6 --val-batch 12
    python benchmarks/operators/bench_v13_resnet.py          --batch 6      # whole-ResNet grid

pytest tests/unittest/test_tig_backward.py
```

Use `--batch 12` (and `--train-batch 12 --val-batch 24` for the per-stage bench) for the
H200 production batch. The
benches refuse to report release numbers from synthetic data — see "How it was benchmarked"
for why.

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
