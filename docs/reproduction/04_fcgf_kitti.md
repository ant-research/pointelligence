# 04 — Task B: FCGF KITTI (single GPU per cell)

> **Goal of this page.** Train an FCGF model with the PointCNN++
> backbone on KITTI Odometry pair scans and confirm validation
> `feat_match_ratio` exceeds the FCGF-paper KITTI floor of 0.966
> (τ=30 cm). Wall: ~25-30 h on a single H100 per cell (200 epochs).

## Prereqs

- `01_environment.md` complete.
- `02_datasets.md` KITTI section complete: `data/kitti_odometry_dataset/`
  exists with `dataset/sequences/` and `icp/` populated.
- 1 GPU with at least 24 GB VRAM per cell (the cluster reproduction
  uses 8 cells = 4 voxel sizes × 2 batch sizes × 1 GPU each).

## Train command (single cell)

```bash
cd examples/FCGF
export KITTI_PATH=/abs/path/to/data/kitti_odometry_dataset
export ICP_SUBDIR=$KITTI_PATH/icp
export VOXEL_SIZE=0.3                    # canonical FCGF KITTI voxel
export BATCH_SIZE=4                      # canonical batch
export MAX_EPOCH=200                     # canonical training length

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_kitti.sh kitti_v03_bs4 ""
```

Output goes to:
```
outputs/Experiments/KITTINMPairDataset-v0.3/HardestContrastiveLossTrainer/ResUNetBN2C/SGD-lr1e-1-e200-b4i1-modelnout64kitti_v03_bs4/<TIMESTAMP>/
├── checkpoint.pth                       # latest
├── best_val_checkpoint.pth              # best validation FMR
└── *.txt                                # training log
```

## Optional: full 8-cell sweep (paper-exact)

The reproduction matrix runs 4 voxel sizes × 2 batch sizes = 8 cells.
If you have 8 GPUs available on one node, launch them in parallel:

```bash
for cell in v015_bs4 v015_bs8 v01_bs4 v01_bs8 v02_bs4 v02_bs8 v03_bs4 v03_bs8; do
  voxel="${cell%%_*}"; voxel="0.${voxel#v}"   # v015 -> 0.15, v03 -> 0.3, etc.
  bs="${cell##*_bs}"
  gpu=$(echo $cell | tr -d 'v_bs' | tr '01234567' '01234567')   # rough mapping
  CUDA_VISIBLE_DEVICES=$gpu BATCH_SIZE=$bs VOXEL_SIZE=$voxel \
    bash scripts/train_kitti.sh "kitti_${cell}" "" &
done
wait
```

Each cell runs ~25-30 h. If you only have 1 GPU, pick `v03_bs4` —
that's the canonical FCGF-paper cell.

## What progress should look like

Validation runs at every epoch. Expected `feat_match_ratio` trajectory:

| Epoch | Expected FMR | Notes |
|---:|---:|---|
| 1   | 0.10-0.30 | initialization noise |
| 2   | **1.000** | best typically reaches saturation this fast |
| 5   | 0.97-1.00 | |
| 10  | 0.99-1.00 | |
| 65  | ≥0.9945 | non-init val passes sustain ≥0.9945 |
| 200 | ~0.998 final | final stabilises high |

Watch:

```bash
tail -f outputs/Experiments/KITTINMPairDataset*/HardestContrastiveLossTrainer/.../<TIMESTAMP>/*.txt | \
  grep -E "Feat Match Ratio|Saving checkpoint"
```

## Pass criterion

Best per-epoch `feat_match_ratio` ≥ 0.966 at some point during the
200 epochs. A well-converged 8-cell sweep reaches best 1.000 by
epoch 2 across all cells and finishes at ≈ 0.998 at epoch 200.

## Eval — compare to the published checkpoint via test.py

After training:

```bash
mkdir -p data/checkpoints/fcgf
cd data/checkpoints/fcgf
gdown 'https://drive.google.com/file/d/12ahfWCwJyaCJwcgqlKDgK-sqPCaTJtif' \
      -O kitti_pcnnpp_ResUNetBN2C.pth
cd ../../..
```

Run `examples/FCGF/test.py` against your trained checkpoint:

```bash
cd examples/FCGF

# Your trained checkpoint (per-cell). save_dir is the dir containing
# checkpoint.pth + config.json.
CUDA_VISIBLE_DEVICES=0 \
  python test.py \
    --save_dir outputs/Experiments/.../<TIMESTAMP> \
    --kitti_root $KITTI_PATH \
    --test_phase test \
    --dataset KITTINMPairDataset
# ETA: ~30 min single GPU; final line is
# "RTE: <m>, var: ..., RRE: <rad>, var: ..., Success: <count>/<count> (<%>)"
```

### ⚠️ The published Google-Drive KITTI ckpt won't load with this overlay

The KITTI checkpoint at the URL above was trained against the
**original `chrischoy/FCGF` master architecture**
(`TR_CHANNELS=[None, 64, 64, 64, 128]` + raw `nn.InstanceNorm1d`, no
running stats), not the PointCNN++ overlay's modified `ResUNetBN2C`
(`TR_CHANNELS[1]=32` + `BatchNorm1dWrapper`). Loading the published
ckpt into the overlay's `test.py` raises:

```
RuntimeError: Error(s) in loading state_dict for ResUNetBN2C:
  Missing key(s): norm1.bn.weight, ..., running_mean, running_var ...
  size mismatch for conv1_tr.weight: copying a param with shape
    torch.Size([96, 64, 1]) ... in current model is torch.Size([96, 32, 1])
```

The 3DMatch published ckpt at `1Wkyb9QSyKsTYPErUOex6lbMwkXootIFk`
DOES load, indicating only the 3DMatch ckpt was re-trained against
the post-modernization architecture; the KITTI ckpt was left at its
upstream `chrischoy/FCGF` state.

If you need to load the published KITTI ckpt directly, clone
`chrischoy/FCGF` master separately and load via its model code. The
reproduction route used here is to **train a fresh KITTI ckpt with the
overlay's code** (the `train_kitti.sh` command above) and evaluate
with the overlay's `test.py`. That ckpt loads cleanly and satisfies
the FCGF Table 6 anchor (RTE < 0.07 m, RRE < 0.5°, Success > 0.97) by
a comfortable margin — see `06_paper_table_mapping.md`.

### Strict criterion vs loose criterion

`test.py`'s default success criterion is FCGF Table 6 loose
(RTE < 2 m AND RRE < 5°). PointCNN++ paper Table 1 uses the strict
criterion (RTE < 0.2 m AND RRE < 1°). A fresh-trained KITTI ckpt
typically lands at RTE 0.057-0.062 m mean and RRE 0.17-0.19° mean on
successful pairs under the loose criterion — both well inside the
strict band, so most successful pairs would also pass strict.
Recomputing the exact strict-criterion success rate from a `test.py`
log requires per-pair RTE/RRE data, which the default log does not
preserve.

Tolerance: ±5 cm RTE, ±0.5° RRE, ±0.01 success rate.

## Common failures

- **`AttributeError: 'Namespace' object has no attribute 'dataset'`**:
  overlay isn't applied. Run `bash overlays/FCGF/apply.sh examples/FCGF`.
- **ICP cache race / training hangs at "Building ICP cache..."**:
  your `data/kitti_odometry_dataset/icp/` is empty. Pre-build it
  with `python scripts/build_icp_cache.py` before launching
  parallel cells (otherwise each cell tries to write the same
  `.npz` files and they race).
- **`scipy.spatial._ckdtree.cKDTree.query() unexpected keyword argument 'n_jobs'`**:
  scipy ≥ 1.6 deprecated `n_jobs`. The overlay's
  `0002-eval-scipy-workers-compat.patch` handles this — re-apply
  the overlay if you forgot to.
- **Validation FMR stays ~0.20 throughout**: voxel_size mismatch.
  KITTI canonical is 0.30 (with `pre_downsample_voxel_size=0.20`); if
  you accidentally used 3DMatch-scale 0.025, the model can't make
  sense of the data.

## What you should have at this point

- `best_val_checkpoint.pth` for at least one cell (canonical: v03_bs4).
- `test.py` printed `Feat Match Ratio: 0.96+` and reasonable RTE/RRE/success on KITTI val.
- Numbers within tolerance of the published checkpoint's `test.py` output.

Next: Tasks C+D in [`05_pointcept_msc_finetune.md`](./05_pointcept_msc_finetune.md).
