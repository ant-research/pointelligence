# 02 — Datasets

> **Goal of this page.** All three datasets (3DMatch, KITTI Odometry,
> NuScenes) on local disk, in the directory layout the canonical
> training scripts expect, with byte-counts you can verify before
> kicking off any training.

| Dataset | Used by | Raw size | Preprocessed size | Wall to download |
|---|---|---:|---:|---:|
| 3DMatch (OverlapPredator format) | FCGF (Task A) | ~1.6 GB | same | ~5 min on a fast link |
| KITTI Odometry sequences 0-10 | FCGF (Task B) | ~80 GB | ~80 GB + ~1.6 GB ICP cache | ~2-4 h |
| NuScenes (LiDAR + lidarseg) | Pointcept (Tasks C+D) | ~500 GB | + ~10 GB info pkls | 6-12 h depending on link |

This page documents each one separately. You can do them in parallel.

---

## 3DMatch (OverlapPredator format) — for Task A

### What we use

FCGF's 3DMatch training and feature-match-recall evaluation use the
**OverlapPredator-curated** version of 3DMatch — the Princeton
`vision.princeton.edu/projects/2016/3DMatch` URL is permanently 404
and redirects to a dead `www.cs.princeton.edu` path; do **not** try
`examples/FCGF/scripts/download_3dmatch_test.sh`, it won't work.

The OverlapPredator preprocessing converts the raw 3DMatch fragments
into per-fragment torch `.pth` tensor files plus per-fragment
`.info.txt` metadata, with a curated test pair list at
`configs/indoor/3DMatch.pkl` (1623 pairs across 8 test scenes).

### Download

```bash
mkdir -p data/3dmatch_processed
cd data/3dmatch_processed

# OverlapPredator's processed 3DMatch dataset.
gdown 'https://drive.google.com/file/d/1zsZbJSID5AL4diJuhC0gZDYJsz-PidhH'

# It downloads as indoor.tar.gz (~1.6 GB).
tar xzf indoor.tar.gz
rm indoor.tar.gz
cd ../..
```

### Expected on-disk layout

```
data/3dmatch_processed/
└── indoor/
    ├── icp/                                    # per-pair ICP cache (used during training)
    ├── train/
    │   ├── 7-scenes-chess/
    │   │   ├── cloud_bin_0.pth
    │   │   ├── cloud_bin_0.info.txt
    │   │   └── ... (60ish per scene)
    │   └── (40ish train scenes)
    └── test/
        ├── 7-scenes-redkitchen/
        │   ├── cloud_bin_0.pth
        │   └── ... (60 fragments per scene)
        └── (8 test scenes total)
```

### Verify

```bash
# Eight test scenes, each ~60 fragments, total ~1623 pairs in the curated list.
ls data/3dmatch_processed/indoor/test/ | wc -l            # → 8
ls data/3dmatch_processed/indoor/test/7-scenes-redkitchen/*.pth | wc -l   # → ~60
du -sh data/3dmatch_processed/indoor/                     # → ~1.6 GB
```

### Pair lists (already shipped)

The curated test/train/val pair lists ship with the FCGF overlay at
`examples/FCGF/configs/indoor/`:

```
examples/FCGF/configs/indoor/
├── 3DMatch.pkl                  # 1623 test pairs (the headline benchmark)
├── 3DLoMatch.pkl                # low-overlap variant
├── train_info.pkl               # training pairs
└── val_info.pkl                 # validation pairs (used by lib/trainer.py FMR)
```

You don't need to download or generate these — they're versioned in
the overlay `add/` tree.

---

## KITTI Odometry — for Task B

### What we use

KITTI Odometry sequences 00-10 (the labeled training set). FCGF KITTI
training uses sequences 0-7 for training, 8-10 for val/test. The
OverlapPredator splits live in
`examples/FCGF/configs/kitti/{train,val,test}_kitti.txt`.

### Download

The official portal is at https://www.cvlibs.net/datasets/kitti/eval_odometry.php.
You need an account (free). Download:

- **Velodyne laser data (~80 GB)** — the LiDAR sweeps
- **Calibration files**
- **Ground-truth poses (sequences 00-10)**

```bash
mkdir -p data/kitti_odometry_dataset/dataset
# Place the downloaded zips there, then unpack:
cd data/kitti_odometry_dataset
unzip data_odometry_velodyne.zip       # → dataset/sequences/<XX>/velodyne/
unzip data_odometry_calib.zip          # → dataset/sequences/<XX>/calib.txt
unzip data_odometry_poses.zip          # → poses/<XX>.txt
```

### Expected on-disk layout

```
data/kitti_odometry_dataset/
├── dataset/
│   └── sequences/
│       ├── 00/
│       │   ├── calib.txt
│       │   ├── times.txt
│       │   └── velodyne/
│       │       ├── 000000.bin     # ~2 MB each, ~4500 per sequence
│       │       └── ...
│       └── ... (sequences 00 through 21)
├── poses/
│   ├── 00.txt
│   └── ... (00-10 only; 11-21 have no GT)
└── icp/                            # WILL be created by build_icp_cache.py
```

### ICP cache (preprocessing step)

FCGF KITTI training computes per-pair ICP refinement for ground-truth
poses. Doing this on-the-fly during multi-cell parallel training causes
race conditions (each parallel process tries to write the same `.npz`
file in `data/kitti_odometry_dataset/icp/`), so the canonical workflow
is to **pre-compute** the cache once before training:

```bash
cd examples/FCGF
KITTI_PATH=/abs/path/to/data/kitti_odometry_dataset \
  python scripts/build_icp_cache.py
# ETA: ~30 min on a single CPU
# Output size: ~1.6 GB at $KITTI_PATH/icp/
```

Verify:

```bash
ls data/kitti_odometry_dataset/icp/ | wc -l   # → several thousand .npz
du -sh data/kitti_odometry_dataset/icp/        # → ~1.6 GB
```

If you skip this step, training will work but be unstably slow (one
process recomputes ICP, others wait or race).

### Verify dataset

```bash
ls data/kitti_odometry_dataset/dataset/sequences/ | wc -l   # → 22 (sequences 00-21)
ls data/kitti_odometry_dataset/dataset/sequences/00/velodyne/ | wc -l   # → ~4541
du -sh data/kitti_odometry_dataset/dataset/                  # → ~80 GB
```

---

## NuScenes (LiDAR + lidarseg) — for Tasks C + D

### What we use

NuScenes v1.0-trainval + v1.0-test for the LiDAR sequences, plus the
**`lidarseg`** annotations (separate download) for the per-point
semantic labels. Pointcept's NuScenes loader computes a sweeps-info
PKL from these once and uses it for the rest of training.

### Download

Official portal: https://www.nuscenes.org/nuscenes#download (free,
account required, accept the terms-of-use).

Download:

1. **Full dataset, v1.0-trainval** (~470 GB across 10 part files)
2. **Full dataset, v1.0-test** (~50 GB)
3. **NuScenes-lidarseg, all parts** (the per-point segmentation labels)
4. **Map expansion v1.3** (small, mostly for visualization)

```bash
mkdir -p data/nuscenes
cd data/nuscenes
# Place the downloaded tarballs here.
for f in v1.0-trainval0?_blobs.tgz; do tar xzf "$f"; done
tar xzf v1.0-trainval_meta.tgz
tar xzf v1.0-test_blobs.tgz
tar xzf v1.0-test_meta.tgz
tar xzf nuScenes-lidarseg-all-v1.0.tar.bz2
# (and similarly for the other parts)
cd ../..
```

### Expected on-disk layout

```
data/nuscenes/
├── samples/                     # keyframe LiDAR + camera + radar
├── sweeps/                      # non-keyframe LiDAR sweeps
├── lidarseg/                    # per-point semantic labels (from the lidarseg download)
├── maps/
└── v1.0-trainval/               # JSON metadata (scenes, samples, instances, ...)
    ├── scene.json
    ├── sample.json
    └── ...
```

### Preprocessing — generate the sweep info PKL

Pointcept needs a single info PKL that pre-computes sample → sweep
associations. Run once:

```bash
mkdir -p data/nuscenes_processed
ln -s "$(pwd)/data/nuscenes" data/nuscenes_processed/raw

python examples/Pointcept/pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py \
  --dataset_root data/nuscenes \
  --output_root data/nuscenes_processed \
  --max_sweeps 10
# ETA: ~30 min single-threaded
```

This produces:

```
data/nuscenes_processed/
├── raw -> data/nuscenes  (symlink)
└── info/
    ├── nuscenes_infos_10sweeps_train.pkl
    ├── nuscenes_infos_10sweeps_val.pkl
    └── nuscenes_infos_10sweeps_test.pkl
```

### Verify

```bash
du -sh data/nuscenes/                          # → ~500 GB
ls data/nuscenes/samples/LIDAR_TOP/ | wc -l    # → ~404,000 keyframes (training+val+test)
ls data/nuscenes_processed/info/               # → 3 .pkl files
du -sh data/nuscenes_processed/info/           # → ~10 GB
```

### Configure data_root in the Pointcept config

The PointCNN++ overlay configs at
`examples/Pointcept/configs/nuscenes/{msc,semseg}-pointcnnpp-base.py`
default to `data_root = 'data/nuscenes_processed'` (relative to
Pointcept submodule cwd). To point at an absolute path, edit the
config or pass via `--options data.train.data_root=...` when
running `scripts/train.sh`.

---

## Sanity check before kicking off any training

```bash
# 3DMatch
ls data/3dmatch_processed/indoor/{train,test} >/dev/null && echo "3DMatch OK"
# KITTI
test -d data/kitti_odometry_dataset/dataset/sequences/00/velodyne \
  && test -d data/kitti_odometry_dataset/icp \
  && echo "KITTI OK"
# NuScenes
test -f data/nuscenes_processed/info/nuscenes_infos_10sweeps_train.pkl \
  && echo "NuScenes OK"
```

All three should print `<DATASET> OK`. If any don't, re-read the
relevant section above.

Next: pick the task you want to reproduce.
- Task A — [`03_fcgf_3dmatch.md`](./03_fcgf_3dmatch.md) (single-GPU)
- Task B — [`04_fcgf_kitti.md`](./04_fcgf_kitti.md) (single-GPU)
- Tasks C+D — [`05_pointcept_msc_finetune.md`](./05_pointcept_msc_finetune.md) (8-GPU DDP)
