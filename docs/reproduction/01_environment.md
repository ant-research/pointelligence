# 01 — Environment setup

> **Goal of this page.** A fresh Linux box has CUDA-capable Python and
> the `internals.*` / `extensions.sparse_engines_cuda` packages
> importable. Total wall: ~30-60 min on a clean machine, dominated by
> the CUDA wheel build.

## Hardware + driver minimums

| Component | Minimum | Tested with |
|---|---|---|
| GPU | NVIDIA Ampere (sm_80) or newer; 24 GB VRAM | H100 SXM, A100 80 GB, RTX 5880 Ada (49 GB) |
| CUDA driver | 12.4 | 12.9 (driver 575.x) |
| Disk | 80 GB free for KITTI alone; 600 GB for everything | |
| Python | 3.10 | 3.10.18 |

If you don't have an Ampere-or-newer GPU, the `internals.*` kernels
still build (they target sm_80 by default) but won't run on Volta /
Turing / Maxwell.

## Conda + Python

```bash
# Pick a name that doesn't collide with anything else on your box.
conda create -n pcnnpp python=3.10 -y
conda activate pcnnpp
```

Optional: install the bottom-of-the-stack toolchain via conda if your
system gcc is too old to build the CUDA extensions:

```bash
conda install cuda-toolkit cudnn 'gcc=13.2' 'gxx=13.2' ninja \
              google-sparsehash -c nvidia -c conda-forge -y
```

## PyTorch

This codebase is tested against torch 2.6+. CUDA 12.4+ wheels work.

```bash
# Pick the wheel that matches your CUDA driver. Examples:
pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.6,<2.10' triton
# or for cu121:
# pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.6,<2.10' triton
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print something like `2.9.1+cu128 True NVIDIA RTX 5880 Ada Generation`.

## pointelligence install (editable)

From the repo root:

```bash
pip install -e .
```

This makes `internals.*` (the PointCNN++ kernels, used by both submodule
overlays) and `extensions.sparse_engines_cuda` importable. It will
build the CUDA extension from source on first install — typically
5-15 min depending on your machine.

Verify the imports work:

```bash
python -c "import internals.grid_sample; import internals.neighbors; \
           import internals.indexing; print('internals OK')"
python -c "import extensions.sparse_engines_cuda; print('extensions OK')"
```

If the second one fails with `ImportError: cannot find sparse_engines_cuda`,
the CUDA build didn't complete. Pin `TORCH_CUDA_ARCH_LIST` to your GPU's
compute capability (e.g. `export TORCH_CUDA_ARCH_LIST="8.0+PTX"` for
A100; `9.0+PTX` for H100; `8.9+PTX` for RTX 4090 / 5880 Ada) and re-run
`pip install -e . --force-reinstall --no-build-isolation`.

## Python deps the submodules need

Run once after `pip install -e .` succeeds:

```bash
# Pointcept deps
pip install h5py tensorboard wandb open3d nuscenes-devkit \
            scipy torch-geometric \
            scikit-learn timm addict yapf plyfile ftfy peft \
            torch-scatter torch-cluster

# FCGF deps (mostly overlapping but includes future_fstrings + pykeops)
pip install future_fstrings pykeops einops easydict joblib

# Tools used by the reproduction guide
pip install gdown   # for downloading the Google-Drive reference checkpoints
```

A few notes:

- **`torch_scatter` / `torch_cluster`**: must match your torch version
  exactly. If `pip install` picks the wrong wheel index, fetch from
  PyTorch Geometric's wheel index manually:
  `pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-${TORCH_VER}+cu${CUDA_VER}.html`.
- **`future_fstrings`**: needed for FCGF's `model/resunet.py` first line
  encoding directive. If you skip it, you'll get
  `SyntaxError: unknown encoding: future_fstrings` on the first FCGF
  import.
- **`scipy>=1.6`**: enables `cKDTree.query(..., workers=-1)` parallelism.
  FCGF's `lib/eval.py` overlay patch handles the older `n_jobs` arg via
  fallback, so older scipy still works — just slower.

## Apply the submodule overlays

Per [`00_setup_overlay.md`](./00_setup_overlay.md):

```bash
git submodule update --init --recursive
bash overlays/Pointcept/apply.sh examples/Pointcept
bash overlays/FCGF/apply.sh examples/FCGF
```

After this, `examples/Pointcept` and `examples/FCGF` contain
upstream-pristine code with the PointCNN++ overlay applied.

## End-to-end smoke test

```bash
# 1. Pointcept config loads (the canonical PointCNN++ NuScenes config).
PYTHONPATH=$(pwd)/examples/Pointcept python -c "
from pointcept.utils.config import Config
cfg = Config.fromfile('examples/Pointcept/configs/nuscenes/semseg-pointcnnpp-base.py')
print('Pointcept config OK; epoch =', cfg.epoch, 'batch_size =', cfg.batch_size)
"

# 2. FCGF model imports cleanly.
PYTHONPATH=$(pwd)/examples/FCGF python -c "
from model.resunet import ResUNetBN2C
m = ResUNetBN2C(in_channels=1, out_channels=64, conv1_kernel_size=5, normalize_feature=True, voxel_size=0.025, D=3)
print('FCGF ResUNetBN2C OK; params =', sum(p.numel() for p in m.parameters()))
"
```

Both prints should land. If either errors, compare your dependency
versions against the table at the top of this page.

## What you should have at this point

- `pip list | grep -E "torch |triton|open3d|pointelligence"` —
  all installed.
- `git -C examples/Pointcept rev-parse HEAD` → `96e109d…`
- `git -C examples/FCGF rev-parse HEAD` → `0612340…`
- The two configs above load and the FCGF model instantiates.
- `nvidia-smi` shows your GPU(s) with the expected VRAM and driver.

Next: [`02_datasets.md`](./02_datasets.md).
