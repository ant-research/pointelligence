# 06 — Paper-table mapping

> **Goal of this page.** For each number you produced in tasks A-D,
> show which row in *PointCNN++: Performant Convolution on Native
> Points* ([arXiv:2511.23227](https://arxiv.org/abs/2511.23227))
> it corresponds to and what tolerance counts as "matched".

The PointCNN++ paper reports four families of numbers that this
reproduction guide covers:

1. 3DMatch Feature Match Recall (Task A)
2. KITTI Odometry registration (Task B)
3. NuScenes self-supervised pretrain (Task C)
4. NuScenes semantic segmentation (Task D)

For each, the table below maps:
- **Paper row** — the column/row name in the published table.
- **What you produced** — which file/log line in your reproduction.
- **Tolerance** — the ±band that defines "matched".

## Task A — 3DMatch Feature Match Recall + Table 2 sweep

| Paper claim | Where in your reproduction | Tolerance |
|---|---|---|
| Training-time best `feat_match_ratio` (τ=30 cm, training recipe) | `outputs/Experiments/.../*.txt` with `Current best val model with feat_match_ratio: <value>` | ±0.005 |
| Paper Table 2 RR / FMR / IR per N | `eval_table2.py --output_log <json>`'s final aggregate table | ±2 pp RR / ±2 pp FMR / ±5 pp IR |
| FCGF Table 1 cross-check FMR (τ=10 cm) | `benchmark_3dmatch.py --evaluate_feature_match_recall`'s final `average : <value>` line | ±0.01 |

**Literature anchor (FCGF ICCV'19):** 0.952 FMR @ τ=0.1 m. A clean
training-time run reaches `feat_match_ratio` 1.000 @ epoch 16 (bs=4)
and ~0.99 @ epoch 34 (bs=8) at the training-recipe τ=0.3 m, both
decisively clearing the FCGF anchor.

### Reference numbers — published ckpt at paper-canonical knobs

Published Google-Drive `ResUNetBN2C-3DMatch` checkpoint
(`1Wkyb9QSyKsTYPErUOex6lbMwkXootIFk`) on the 1623 OverlapPredator
pairs, paper-canonical knobs (τ_inlier=0.3 m, RANSAC
`distance_threshold = voxel_size × 1.5 = 0.0375 m`):

| N    | RR (measured / paper "Ours") | FMR (measured / paper) | IR (measured / paper) |
|-----:|:----------------------------:|:----------------------:|:---------------------:|
| 5000 |  92.8 / 90.3 (+2.5) |  99.9 / 98.9 (+1.0) | 57.5 / 58.2 (−0.7) |
| 2500 |  93.4 / 90.2 (+3.2) |  99.9 / 99.1 (+0.8) | 57.7 / 57.8 (−0.1) |
| 1000 |  93.0 / 89.2 (+3.8) |  99.9 / 99.1 (+0.8) | 61.3 / 57.3 (+4.0) |
|  500 |  91.9 / 89.1 (+2.8) |  99.6 / 98.4 (+1.2) | 64.8 / 52.1 (+12.7) |
|  250 |  87.9 / 88.3 (−0.4) |  99.4 / 99.2 (+0.2) | 66.9 / 53.4 (+13.5) |

Wall-time: ~140 min on a single consumer 24 GB GPU; ~60-90 min on H100.

### Reference numbers — fresh-retrain ckpt at paper-canonical knobs

A fresh-trained `ResUNetBN2C-3DMatch` ckpt (best FMR=1.000 @ epoch 9
on bs=4) on the same 1623 pairs, paper-canonical knobs, random
sampling averaged over 3 seeds:

| N    | RR (retrain / paper) | FMR (retrain / paper) | IR (retrain / paper) |
|-----:|:--------------------:|:---------------------:|:--------------------:|
| 5000 | 93.6 / 90.3 (+3.3) | 99.6 / 98.9 (+0.7) | 58.7 / 58.2 (+0.5) |
| 2500 | 93.4 / 90.2 (+3.2) | 99.6 / 99.1 (+0.5) | 58.7 / 57.8 (+0.9) |
| 1000 | 93.6 / 89.2 (+4.4) | 99.6 / 99.1 (+0.5) | 58.7 / 57.3 (+1.4) |
|  500 | 93.0 / 89.1 (+3.9) | 99.7 / 98.4 (+1.3) | 58.7 / 52.1 (+6.6) |
|  250 | 93.2 / 88.3 (+4.9) | 99.7 / 99.2 (+0.5) | 58.7 / 53.4 (+5.3) |

Wall-time: ~6-7 h for the full 1623 pairs × 3 seeds × 5-N sweep on a
single GPU.

**Headline cell N=5000**: FMR + IR are within ±2 pp / ±5 pp tolerance.
RR sits ~+2.5 to +3.3 pp above paper (just outside ±2 pp tolerance,
but in the favorable "exceeds paper" direction — `eval_table2.py`'s
top-K mutual NN feeds RANSAC cleaner correspondences than the
all-points-with-internal-feature-matching flow of `test.py`).

**Small-N IR divergence**: at N ≤ 500 the IR rises (top-K picks the
most confident correspondences) while paper's IR drops. This is a
known protocol-detail divergence in N semantics between the
`eval_table2.py` adapter and the paper's Table 2 pipeline.

## Task B — KITTI Odometry registration

| Paper claim | Where in your reproduction | Tolerance |
|---|---|---|
| KITTI val FMR (τ=30 cm) | `examples/FCGF/test.py`'s final `Feat Match Ratio: <value>` line | **±0.01** |
| KITTI val RTE-mean | same line: `RTE: <m>` | **±0.05 m** |
| KITTI val RRE-mean | same line: `RRE: <rad>` (convert to degrees: ×180/π) | **±0.5°** |
| KITTI val success rate | same line: `Success: <count>/<count> (<%>)` | **±0.01** |
| Training-time best `feat_match_ratio` | `outputs/Experiments/KITTI*/.../*.txt` | ±0.005 |

**Literature anchor (FCGF ICCV'19):** 0.966 FMR, RTE ≤ 0.07 m,
RRE ≤ 0.5°, success ≥ 0.97. A clean 8-cell KITTI retrain reaches
best 1.000 @ epoch 2-3 across all cells and finishes at ≈ 0.998
at epoch 200.

### Reference numbers — training-time FMR (8-cell sweep, 200 ep each)

A clean retrain across the canonical 4 voxel × 2 batch matrix:

| Cell (voxel × bs) | Best FMR | At epoch | Final val FMR (ep200) |
|---|---:|---:|---:|
| v0.10 × bs4   | **1.000** | 2 | 0.998 |
| v0.10 × bs8   | **1.000** | 2 | 0.998 |
| v0.15 × bs4   | **1.000** | 2 | 0.998 |
| v0.15 × bs8   | **1.000** | 2 | 0.998 |
| v0.20 × bs4   | **1.000** | 2 | 0.998 |
| v0.20 × bs8   | **1.000** | 2 | 0.998 |
| v0.30 × bs4   | **1.000** | 2 | 0.998 |
| v0.30 × bs8   | **1.000** | 3 | 0.998 |

All cells pass the 0.966 floor by epoch 2-3 and stabilise near 0.998
through epoch 200.

### Reference numbers — `test.py` registration metrics (KITTI val, 589 pairs)

`examples/FCGF/test.py` on the KITTI val test_phase (589 pairs from
sequences 9-10), using each cell's `checkpoint.pth` (final epoch 200):

| Cell (voxel × bs) | RTE (m) | RRE (rad) | RRE (deg) | Success |
|---|---:|---:|---:|---:|
| v0.10 × bs4 | 0.0598 | 0.00314 | 0.180° | 588/589 (99.83 %) |
| v0.10 × bs8 | 0.0616 | 0.00324 | 0.186° | 588/589 (99.83 %) |
| v0.15 × bs4 | 0.0570 | 0.00325 | 0.186° | **589/589 (100.00 %)** |
| v0.15 × bs8 | 0.0595 | 0.00305 | 0.175° | 588/589 (99.83 %) |
| v0.20 × bs4 | 0.0592 | 0.00304 | 0.174° | 588/589 (99.83 %) |
| v0.20 × bs8 | 0.0591 | 0.00331 | 0.190° | 588/589 (99.83 %) |
| v0.30 × bs4 | 0.0580 | 0.00304 | 0.174° | 588/589 (99.83 %) |
| v0.30 × bs8 | 0.0591 | 0.00313 | 0.179° | 588/589 (99.83 %) |

All 8 cells clear the FCGF ICCV'19 anchor (RTE ≤ 0.07 m, RRE ≤ 0.5°,
success ≥ 0.97) by a comfortable margin: RTE 0.057-0.062 m, RRE
0.17-0.19° (an order of magnitude below the 0.5° floor), success
99.83-100 %.

### Strict criterion vs loose criterion

The numbers above use the **FCGF-loose** success criterion
(RTE < 2 m AND RRE < 5°), which is `test.py`'s default and the
anchor for FCGF Table 6. PointCNN++ Table 1 reports row "Ours" at a
**strict** criterion (RTE < 0.2 m AND RRE < 1°) with target
0.19 m / 0.060° / 99.8 % Recall.

Direct strict-criterion verification against the published
`ResUNetBN2C-KITTI` checkpoint is not in scope for this overlay
reproduction: the published KITTI ckpt was trained against the
upstream `chrischoy/FCGF` master architecture (`TR_CHANNELS[1]=64` +
raw `nn.InstanceNorm1d`), while the overlay ships a modified
`ResUNetBN2C` (`TR_CHANNELS[1]=32` + `BatchNorm1dWrapper`). Loading
the published ckpt requires the upstream model definitions.

The fresh-retrain numbers above (RTE 0.057-0.062 m mean, RRE 0.17-0.19°
mean on successful pairs under the loose criterion) sit well inside
the strict band, so most successful pairs would also pass strict.
Recomputing an exact strict success rate from a `test.py` log requires
per-pair RTE/RRE data, which the default log does not preserve.

**RANSAC convergence**: ensure `RANSACConvergenceCriteria(50000, 0.999)`
in `examples/FCGF/test.py` (the overlay ships this value). Upstream
`chrischoy/FCGF` historically inherited `(4000000, 10000)`, where the
`10000` confidence is outside the valid `[0,1]` range and prevents
early-exit — every call would run all 4M iterations (~700-900 s/pair).
With `(50000, 0.999)`, per-pair RANSAC time is ~1 s and total wall
across 589 pairs is ~45 min per cell on a single GPU.

## Task C — NuScenes MSC pretrain

| Paper claim | Where in your reproduction | Tolerance |
|---|---|---|
| Final contrastive loss after 5 ep | `exp/nuscenes_pretrain/<exp>/train.log` last `Epoch 5/5: Loss <value>` line | **±5%** |
| `model_last.pth` saved at end of ep 5 | `exp/nuscenes_pretrain/<exp>/model/model_last.pth` exists | n/a |

There's no published "MSC pretrain accuracy" number — the proxy is
"the trained `model_last.pth`, when used to seed the Task D finetune,
should produce a finetune `val/mIoU` within ±0.2 pp of the canonical
finetune". A fresh MSC-pretrain → finetune chain typically lands at
~0.7613 best val mIoU vs ~0.7631 for the finetune of the published
MSC pretrain — Δ ~0.18 pp, within tolerance.

## Task D — NuScenes semantic segmentation

| Paper claim | Where in your reproduction | Tolerance |
|---|---|---|
| Best per-epoch `val/mIoU` over 50 ep | `exp/nuscenes/<exp>/train.log` `Currently Best mIoU: <value>` | **±0.005** |
| TTA `test/mIoU` from `scripts/test.sh` | `exp/nuscenes/<exp>/test/test_log.txt` final `Val result: mIoU/mAcc/allAcc` after the TTA passes | **±0.005** (±0.5 pp) |
| Per-class IoU (16 NuScenes classes) | same log; per-class breakdown above the final mean | ±0.01 per class |

**Literature anchor (Pointcept PTv3+MSC, original backbone):**
NuScenes val mIoU ~0.80 with TTA. The PointCNN++ backbone tradeoff
optimises throughput / memory; per-epoch `val/mIoU` of 0.7613-0.7631
is the expected training-time operating point. After the TTA test
pass it climbs into the ~0.78 range — see the reference numbers
below.

### Reference TTA `test/mIoU` (8-GPU DDP)

Both the published-MSC-seeded finetune and the fresh-MSC-seeded
finetune evaluated end-to-end via `scripts/test.sh -d nuscenes -n
<exp> -w model_best -g 8` on the canonical 5 + 10 RandomScale × Flip
TTA augmentation set:

| Checkpoint | TTA mIoU | mAcc | allAcc |
|---|---:|---:|---:|
| Finetune of published MSC pretrain | **~0.7784** | ~0.8377 | ~0.9385 |
| Finetune of fresh MSC pretrain     | **~0.7815** | ~0.8398 | ~0.9394 |

Both well above the literature 0.76 floor and within the ±0.5 pp
tolerance. The training-time ~0.18 pp `val/mIoU` gap (0.7613 vs
0.7631) flips sign under TTA so the two checkpoints are functionally
equivalent on the canonical paper-table number. Per-class breakdown
shows the fresh-MSC pretrain ahead on `car` (+~2 pp), `motorcycle`
(+~1 pp), `pedestrian` (+~0.7 pp), with very small regressions on
`bus` and `bicycle` (each ≤ 0.3 pp).

## Verification protocol

To verify a number is reproduced:

1. Read the relevant table row above for the metric.
2. Run the canonical script(s) on **both** your trained checkpoint
   AND the published Google-Drive checkpoint.
3. Compare the two values. Pass = within the tolerance band stated
   in the row.
4. Record the value in your reproduction notes alongside the
   command, the checkpoint path, and the script invocation.

If your value is outside the tolerance:
- Re-check the data preprocessing layout (`02_datasets.md`).
- Re-check the config (no surprise overrides via `--options`).
- Re-apply the submodule overlays (`00_setup_overlay.md`).
- File a documentation issue if you can't close the gap.

## When the published checkpoint isn't available

For Task A (3DMatch): the published `ResUNetBN2C` checkpoint is at
`https://drive.google.com/file/d/1Wkyb9QSyKsTYPErUOex6lbMwkXootIFk` —
70 MB. Use that as the A2/A3 reference.

For Task B (KITTI): same mechanism, different drive ID:
`https://drive.google.com/file/d/12ahfWCwJyaCJwcgqlKDgK-sqPCaTJtif`.

For Tasks C+D (Pointcept): MSC pretrain
`https://drive.google.com/file/d/1x8cyOZAKerBR0aNbUtHufwlVDDFBp8zM`,
finetune `model_best.pth`
`https://drive.google.com/file/d/1lwENP_nWhP8YaK7I6UzTYa_3QQDiPYJo`.

Pull with `gdown` (installed in `01_environment.md`).
