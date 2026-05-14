# Reproducing PointCNN++

This guide walks an external user from `git clone` to "matched the paper
Table" for the four reproduction targets in
[*PointCNN++: Performant Convolution on Native Points*](https://arxiv.org/abs/2511.23227)
(CVPR 2026). It assumes no internal-cluster access and no prior context
with this codebase.

The four reproduction targets:

| # | Task | Hardware | Wall ETA | Headline metric |
|---|---|---|---:|---|
| A | FCGF 3DMatch (registration features on RGB-D fragments) | 1 GPU | ~6-8 h | `feat_match_ratio` ≥ 0.952 |
| B | FCGF KITTI (registration features on LiDAR scans) | 1 GPU | ~25-30 h | `feat_match_ratio` ≥ 0.966 |
| C | Pointcept MSC pretrain (LiDAR self-supervised) | 8 GPU | ~1-2 h | contrastive loss converging |
| D | Pointcept NuScenes semseg finetune + TTA test | 8 GPU | ~30-36 h train + 7-8 h TTA | `test/mIoU` (TTA) within ±0.5 pp of published checkpoint |

GPUs are H100/A100 80 GB-class for the cluster column. Consumer 24-49 GB
single-GPU walltimes are roughly 3-5× longer; specific notes appear in
each task page.

## Read in order

1. [`00_setup_overlay.md`](./00_setup_overlay.md) — apply the PointCNN++
   overlay onto upstream-pristine FCGF / Pointcept submodules.
   *(One-time, ~2 minutes after the clone.)*
2. [`01_environment.md`](./01_environment.md) — conda env, pip install,
   CUDA wheel build, dependency pitfalls. ~30-60 min on a fresh box.
3. [`02_datasets.md`](./02_datasets.md) — where to download 3DMatch,
   KITTI Odometry, NuScenes; raw layout; preprocessing commands;
   what the on-disk layout looks like after preprocessing. **The
   Princeton 3DMatch URL is permanently 404 — the doc points at the
   working OverlapPredator mirror instead.**
4. [`03_fcgf_3dmatch.md`](./03_fcgf_3dmatch.md) — single-GPU train +
   eval for Task A. Anchor numbers: training-time best
   `feat_match_ratio` should reach ≈ 1.000 by epoch 16; the FCGF
   paper's literature floor is 0.952.
5. [`04_fcgf_kitti.md`](./04_fcgf_kitti.md) — single-GPU train + eval
   for Task B. Anchor: best `feat_match_ratio` ≈ 1.000 by epoch 2,
   final ≈ 0.998 across all 8 voxel/batch cells; literature floor 0.966.
6. [`05_pointcept_msc_finetune.md`](./05_pointcept_msc_finetune.md) —
   8-GPU DDP MSC pretrain (5 ep) → finetune (50 ep) → TTA test.
   Anchor: best per-epoch `val/mIoU` ≈ 0.7613-0.7631; TTA `test/mIoU`
   ≈ 0.778-0.782.
7. [`06_paper_table_mapping.md`](./06_paper_table_mapping.md) — which
   number you produced corresponds to which row in arXiv:2511.23227,
   with tolerance bounds.

## What "reproduced" means here

Two layers:

- **Metric** — your trained checkpoint, evaluated by the canonical
  script in the submodule, lands within tolerance of the published
  Google-Drive reference checkpoint evaluated by the same script.
  Tolerances: ±0.01 FMR, ±0.5° RRE, ±5 cm RTE, ±0.5 pp mIoU.
- **Experience** — every step in this guide is concrete, has an ETA,
  and produces a verifiable output (file count, expected log line,
  byte count).

If the docs ever say "should produce X" and you produce something
different, that's a documentation bug — please file an issue with the
exact step number and the divergent output.

## Reference checkpoints (Google Drive)

These are the targets that "reproduction" matches against. Download
each one as part of the relevant task's eval step:

- FCGF KITTI (`ResUNetBN2C`):
  <https://drive.google.com/file/d/12ahfWCwJyaCJwcgqlKDgK-sqPCaTJtif>
- FCGF 3DMatch (`ResUNetBN2C`):
  <https://drive.google.com/file/d/1Wkyb9QSyKsTYPErUOex6lbMwkXootIFk>
- Pointcept MSC pretrain (`model_last.pth`):
  <https://drive.google.com/file/d/1x8cyOZAKerBR0aNbUtHufwlVDDFBp8zM>
- Pointcept finetune (`model_best.pth`):
  <https://drive.google.com/file/d/1lwENP_nWhP8YaK7I6UzTYa_3QQDiPYJo>

Pull with `gdown` (see `01_environment.md` for the install).

## Citation

If you use any of this for a paper or a project, please cite:

```bibtex
@misc{li2025pointcnnperformantconvolutionnative,
  title={PointCNN++: Performant Convolution on Native Points},
  author={Lihan Li and Haofeng Zhong and Rui Bu and Mingchao Sun
          and Wenzheng Chen and Baoquan Chen and Yangyan Li},
  year={2025},
  eprint={2511.23227},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.23227}
}
```
