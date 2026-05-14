# 03 — Task A: FCGF 3DMatch (single GPU)

> **Goal of this page.** Train an FCGF model with the PointCNN++
> backbone (`ResUNetBN2C`) on 3DMatch indoor pair fragments and confirm
> validation `feat_match_ratio` exceeds the FCGF ICCV'19 paper floor of
> 0.952. Wall: ~6-8 h on a single H100 (~3-4 h on an A100, ~12-16 h on
> consumer 24 GB).

## Prereqs

- `01_environment.md` complete (env + overlays applied + smoke tests pass).
- `02_datasets.md` 3DMatch section complete: `data/3dmatch_processed/indoor/{train,test}` exists.
- 1 GPU with at least 24 GB VRAM.

## Train command

```bash
cd examples/FCGF
export THREEDMATCH_ROOT=/abs/path/to/data/3dmatch_processed/indoor
export VOXEL_SIZE=0.025                 # canonical 3DMatch voxel grid
export BATCH_SIZE=4                     # we'll add bs=8 below
export MAX_EPOCH=40                     # canonical training length

# Single GPU, batch=4 cell:
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_3dmatch.sh 3dmatch_bs4 ""
```

What this does:
- Sets up `outputs/Experiments/ThreeDMatchNewPairDatasetPure-v0.025/HardestContrastiveLossTrainer/ResUNetBN2C/SGD-lr0.0006-e40-b4i1-modelnout64.../<TIMESTAMP>/`
- Loads `configs/indoor/{train_info,val_info}.pkl` for the canonical pair lists.
- Runs the `HardestContrastiveLossTrainer` for 40 epochs with lr=6e-4, model_n_out=64, conv1_kernel_size=5.
- Runs validation (FMR computation on `val_info.pkl` pairs) every epoch.
- Saves `checkpoint.pth` after each epoch and `best_val_checkpoint.pth` whenever validation FMR improves.

## Optionally: also run bs=8 for the second cell

If you have a second GPU available (or are running this in a multi-cell
job for cluster reproduction), launch the same command on a different
GPU with `BATCH_SIZE=8`:

```bash
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=8 \
  bash scripts/train_3dmatch.sh 3dmatch_bs8 ""
```

## What progress should look like

Validation runs at the end of each epoch. Expected `feat_match_ratio`
trajectory:

| Epoch | bs=4 expected FMR | bs=8 expected FMR |
|---:|---:|---:|
| 5  | ~0.8 | ~0.6 |
| 10 | ~0.95 | ~0.88 |
| 16 | **1.000** | ~0.94 |
| 25 | 1.000 | ~0.97 |
| 34 | 1.000 | **~0.99** |
| 40 | 1.000 | 0.99-1.00 |

Watch with:

```bash
# while training runs, in another shell:
tail -f outputs/Experiments/ThreeDMatchNewPairDatasetPure-v0.025/HardestContrastiveLossTrainer/ResUNetBN2C/SGD-lr0.0006-e40-b4i1-modelnout64*/<TIMESTAMP>/*.txt | \
  grep "feat_match_ratio\|Final"
```

You should see one "Final ... feat_match_ratio: 0.X" per validation
pass. If FMR stalls below 0.5 past epoch 5, the data pipeline is
probably mis-voxelised — re-check `pre_downsample_voxel_size=0.020` in
the dataloader and confirm the 3DMatch directory layout from
`02_datasets.md`.

## Pass criterion

Per-epoch `feat_match_ratio` clears the FCGF ICCV'19 paper floor of
**0.952** at τ=10 cm at least once during the 40-epoch run on both
bs=4 and bs=8.

A successful run produces a row in the per-epoch log with `Feat Match
Ratio: 0.95+` at some epoch ≤ 30. Typical end-state: bs=4 reaches
1.000 by epoch 16; bs=8 reaches ~0.99 by epoch 34.

## Eval — compare to the published checkpoint

After training, also pull the published Google-Drive checkpoint:

```bash
mkdir -p data/checkpoints/fcgf
cd data/checkpoints/fcgf
gdown 'https://drive.google.com/file/d/1Wkyb9QSyKsTYPErUOex6lbMwkXootIFk' \
      -O 3dmatch_pcnnpp_ResUNetBN2C.pth
cd ../../..
```

### Paper Table 2 reproduction (full RR / FMR / IR sweep)

The PointCNN++ paper Table 2 reports RR / FMR / IR at five keypoint
counts {5000, 2500, 1000, 500, 250}. Reproduce with `eval_table2.py`
(in the FCGF overlay's `add/scripts/`):

```bash
cd examples/FCGF
python scripts/eval_table2.py \
  --checkpoint /abs/path/to/data/checkpoints/fcgf/3dmatch_pcnnpp_ResUNetBN2C.pth \
  --threedmatch_root /abs/path/to/data/3dmatch_processed/indoor \
  --pair_list ./configs/indoor/3DMatch.pkl \
  --n_points 5000,2500,1000,500,250 \
  --hit_ratio_thresh 0.3 \
  --ransac_corr_dist 0.0375 \
  --output_log /tmp/eval_table2.json
# ETA: ~140 min on a single 5880 Ada / consumer 24 GB; ~60-90 min on H100.
```

The two non-default flags are paper-canonical:

- `--hit_ratio_thresh 0.3` (NOT 0.1) — PointCNN++ paper Table 2 IR / FMR
  uses the training-recipe τ_inlier of 0.3 m, not the FCGF / Predator
  literature standard 0.1 m. Using τ=0.1 produces IR / FMR roughly 14 pp
  lower than the paper-reported numbers; that's the FCGF / Predator
  cross-check column, not the PointCNN++ paper-canonical column.
- `--ransac_corr_dist 0.0375` — RANSAC `max_correspondence_distance =
  voxel_size × 1.5`. The default in test.py-style scripts is
  `voxel_size × 1.0 = 0.025 m`, which is too tight and depresses RR
  by ~12 pp.

### Expected output (published Google-Drive ckpt at paper-canonical knobs)

```
| N    | RR   | FMR  | IR   |
| 5000 | 92.8 | 99.9 | 57.5 |
| 2500 | 93.4 | 99.9 | 57.7 |
| 1000 | 93.0 | 99.9 | 61.3 |
|  500 | 91.9 | 99.6 | 64.8 |
|  250 | 87.9 | 99.4 | 66.9 |
```

vs paper Table 2 row "Ours" (90.3 / 98.9 / 58.2 at N=5000). Tolerance:
±2 pp RR, ±2 pp FMR, ±5 pp IR. The N=5000 cell matches within tolerance
on FMR + IR; RR is +2.5 pp above paper (favorable "we exceed paper"
direction). At N ≤ 500 the IR rises (mutual NN + top-K picks the most
confident correspondences) while paper's drops; this is a known
protocol-detail divergence on N semantics.

### FCGF Table 1 cross-check (FMR-only, τ=0.1)

For comparison against the **FCGF ICCV'19 paper**'s Table 1 (FMR =
0.952 ± 0.029 at τ=0.1), use `benchmark_3dmatch.py`:

```bash
cd examples/FCGF

# Phase 1: extract features for every test fragment.
mkdir -p /tmp/3dmatch_features
python scripts/benchmark_3dmatch.py \
  --source /abs/path/to/data/3dmatch_processed/indoor/test \
  --target /tmp/3dmatch_features \
  --voxel_size 0.025 \
  --extract_features \
  --with_cuda \
  -m outputs/Experiments/.../best_val_checkpoint.pth   # your checkpoint
# ETA: ~10-20 min on a single H100; produces ~480 .npz files.

# Phase 2: compute Feature Match Recall over the curated 1623 test pairs.
python scripts/benchmark_3dmatch.py \
  --source /abs/path/to/data/3dmatch_processed/indoor/test \
  --target /tmp/3dmatch_features \
  --voxel_size 0.025 \
  --evaluate_feature_match_recall \
  --num_rand_keypoints 5000
# ETA: ~5-10 min CPU-only; final line is "average : <FMR> +- <std>".
```

This script uses the FCGF/Predator standard τ=0.1 (hardcoded). Useful
for cross-checking against FCGF Table 1 (0.952) — NOT for matching
PointCNN++ paper Table 2 (use `eval_table2.py` above for that).

## Common failures

- **`MinkowskiEngine` import error** in `benchmark_3dmatch.py`: you
  didn't apply the overlay. Re-run `bash overlays/FCGF/apply.sh
  examples/FCGF`.
- **`AttributeError: 'Namespace' object has no attribute 'dataset'`**
  in `test.py`: same — the `--dataset` arg is added by the overlay.
- **FMR stuck at ~0.15**: feature pipeline mismatch. The trained
  checkpoint expects `pre_downsample_voxel_size=0.020` (data loader's
  `grid_sample_filter(reduction="center_nearest")`), not
  `np.unique(floor(xyz/voxel_size))`. If you write your own eval, use
  `IndoorPairDataset` from `lib/data_loaders.py`.

## What you should have at this point

- `outputs/Experiments/.../best_val_checkpoint.pth` exists and has a
  validation FMR ≥ 0.95 logged.
- `benchmark_3dmatch.py --evaluate_feature_match_recall` prints
  `average : 0.95+` for both your checkpoint and the published one.
- `06_paper_table_mapping.md` shows where this number lands in
  arXiv:2511.23227.
