# 05 — Tasks C + D: Pointcept MSC pretrain → NuScenes finetune → TTA test (8-GPU)

> **Goal of this page.** Reproduce the Pointcept side of PointCNN++:
> a 5-epoch self-supervised MSC pretrain on NuScenes LiDAR
> (Task C), a 50-epoch semantic-segmentation finetune (Task D
> training-time `val/mIoU`), and a TTA test pass over the val set
> for the canonical `test/mIoU` number (Task D headline). Wall: ~1-2 h
> (pretrain) + ~30-36 h (finetune) + ~7-8 h (TTA test) on 8×H100.

The canonical MSC pretrain length is **5 epochs**, set at
`configs/nuscenes/msc-pointcnnpp-base.py:52` as `epoch = 5`. That's
what the published `model_last.pth` was trained with — do not extend.

## Prereqs

- `01_environment.md` complete + overlays applied.
- `02_datasets.md` NuScenes section complete:
  `data/nuscenes_processed/info/nuscenes_infos_10sweeps_{train,val,test}.pkl`
  exist; raw at `data/nuscenes_processed/raw/`.
- 8 GPUs on one node, each with at least 80 GB VRAM (H100 SXM or A100).
  Consumer 24 GB cards do not fit the canonical batch_size=48 finetune.
- Pointcept submodule's relative `data_root` resolves: from inside
  `examples/Pointcept`, the path `data/nuscenes_processed` should
  point at your processed NuScenes dir (use a symlink if the real
  data is elsewhere).

## Stage 1: MSC pretrain (~1-2 h)

```bash
cd examples/Pointcept

bash scripts/train.sh \
  -d nuscenes_pretrain \
  -c msc-pointcnnpp-base \
  -n msc_pretrain_$(date +%Y%m%d_%H%M%S) \
  -g 8
```

What this does:
- Reads `configs/nuscenes/msc-pointcnnpp-base.py` (epoch=5, batch_size=24
  global, voxel/grid 0.05 m, AdamW).
- Writes to `exp/nuscenes_pretrain/<EXP_NAME>/`.
- Runs 5 epochs of Masked Scene Contrast self-supervised pretrain.
- Saves `model/model_last.pth` at the end.

### Expected loss trajectory

| Epoch | Loss (approx) |
|---:|---:|
| 1 (start) | ~4.5 |
| 1 (end) | ~4.0 |
| 3 | ~3.7 |
| 5 | ~3.3 |

Final contrastive loss should land in the 3.2-3.4 range. If it stays
> 4.0 at end of epoch 5, the dataloader is probably not producing
valid positive pairs (verify `data_root` resolves and the info PKLs
from `02_datasets.md` are in place).

### Pass criterion

Run completes 5 epochs cleanly, final loss ≤ 3.5, `model_last.pth`
saved. A typical run finishes at 3.30-3.35.

## Stage 2: Semantic-segmentation finetune (~30-36 h)

```bash
cd examples/Pointcept

PRETRAIN_CKPT=exp/nuscenes_pretrain/<MSC_EXP_NAME>/model/model_last.pth

bash scripts/train.sh \
  -d nuscenes \
  -c semseg-pointcnnpp-base \
  -n semseg_$(date +%Y%m%d_%H%M%S) \
  -w "$PRETRAIN_CKPT" \
  -g 8
```

What this does:
- Reads `configs/nuscenes/semseg-pointcnnpp-base.py` (epoch=50,
  batch_size=48 global, CE + Focal + Lovász loss).
- Writes to `exp/nuscenes/<EXP_NAME>/`.
- Loads MSC-pretrained backbone weights via `-w`, finetunes for 50
  epochs.
- Runs `val/mIoU` evaluation at every epoch.
- Saves `model/model_last.pth` and `model/model_best.pth` (whenever
  `val/mIoU` improves).

### Expected `val/mIoU` trajectory

| Epoch | Expected val/mIoU |
|---:|---:|
| 5 | ~0.55 |
| 10 | ~0.65 |
| 20 | ~0.71 |
| 30 | ~0.73 |
| 40 | ~0.75 |
| 48 | **0.7613-0.7631** |
| 50 | ≈0.76 |

### Pass criterion

Best per-epoch `val/mIoU` ≥ 0.760 over 50 epochs. Finetune from the
published MSC pretrain typically peaks around 0.7631 @ epoch 48;
finetune from a fresh MSC pretrain reproduced with this guide
typically peaks around 0.7613 @ epoch 48 — a ~0.18 pp gap that
reflects normal stochastic variation in the pretrain. The TTA test
pass below disambiguates intrinsic finetune quality from TTA noise.

## Stage 3: TTA test pass (~7-8 h on 8-GPU DDP)

This is the **canonical headline number** — the one that appears in
arXiv:2511.23227 Table for NuScenes. Per-epoch `val/mIoU` is a noisy
proxy; the TTA test pass produces the clean number.

```bash
cd examples/Pointcept

bash scripts/test.sh \
  -d nuscenes \
  -n <SEMSEG_EXP_NAME> \
  -w model_best \
  -g 8
```

The `-g 8` enables DDP-test which splits the 6019 NuScenes val
keyframes across all 8 GPUs in parallel. Without DDP (`-g 1`), the
pass takes ~60 h instead of ~7-8 h — be sure to use `-g 8`.

### What `test.sh` actually does

`tools/test.py` runs the `SemSegTester` hook, which evaluates each
keyframe under the **full TTA augmentation set** baked into
`configs/nuscenes/semseg-pointcnnpp-base.py`:

| Augmentation block | Variants |
|---|---|
| `RandomScale` only | scales {0.9, 0.95, 1.0, 1.05, 1.1} → 5 |
| `RandomScale` × `RandomFlip` | each scale × {flip-X, flip-Y} → 10 |

Total: 15 augmentations per keyframe. The reported `test/mIoU` is
averaged over the augmentation outputs. **This is not the same number
as the `val/mIoU` printed during training** — that's the per-epoch
mIoU on a single (no-augmentation) val pass.

### Expected `test/mIoU`

The published `model_best.pth` test/mIoU is what the paper reports.
A clean TTA test pass on 8-GPU DDP produces:

| Checkpoint | `test/mIoU` (TTA) | mAcc | allAcc |
|---|---:|---:|---:|
| Finetune of published MSC pretrain | **~0.7784** | ~0.838 | ~0.939 |
| Finetune of MSC pretrain reproduced by this guide | **~0.7815** | ~0.840 | ~0.939 |

Tolerance: ±0.5 pp. Both numbers sit within tolerance of each other
and of the published anchor; the ~0.18 pp training-time `val/mIoU`
gap (0.7613 vs 0.7631) flips sign under TTA, so the two checkpoints
are functionally equivalent on the canonical table number.

## Common failures

- **`val/mIoU` stays ~0.40 throughout**: data_root path is wrong.
  The relative path `data/nuscenes_processed` doesn't resolve to
  your actual NuScenes dir. Either symlink, edit the config, or
  pass `--options data.train.data_root=...`.
- **TTA test pass takes >24 h**: you launched with `-g 1`. Kill and
  rerun with `-g 8` — DDP-test scales linearly.
- **OOM during finetune**: batch_size 48 global = 6 per GPU on 8
  GPUs × 80 GB. If you have <80 GB cards, halve `batch_size_per_gpu`
  in the config (paper number assumes the full canonical batch).

## What you should have at this point

- `exp/nuscenes_pretrain/<msc>/model/model_last.pth` (Task C output)
- `exp/nuscenes/<semseg>/model/model_best.pth` with best_val ≥ 0.76
  (Task D training metric)
- A `test/mIoU` number from `scripts/test.sh -g 8` within tolerance
  of the published checkpoint (Task D headline)

Next: [`06_paper_table_mapping.md`](./06_paper_table_mapping.md) for
how these numbers map to the arXiv:2511.23227 paper table.
