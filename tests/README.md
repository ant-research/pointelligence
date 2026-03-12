# Test Suite

Benchmarks for PointCNN++ backbone (ResNet18), single-conv layers, and several unittests.

## Data preparation

Point cloud data is expected under a **sample data directory** with this layout:

```
<data_dir>/
  scale_<name1>/          # e.g. scale_10k, scale_50k
    <id>_coord.npy        # required: (N, 3) or (N, >=3), first 3 cols = x,y,z
    <id>_color.npy        # optional: (N, C) per-point features (single_conv only)
  scale_<name2>/
    ...
```

You can download the sampledata needed through: `https://drive.google.com/file/d/1zsZbJSID5AL4diJuhC0gZDYJsz-PidhH/view?usp=sharing`.

## Running tests

Default data path in the scripts is `.../Pointcept/data/sampledata`. Override with `--data_dir`:

```bash
# ResNet18 backbone
python tests/backbone/test_resnet18_benchmark.py --data_dir /path/to/sampledata

# Single conv (normal / degraded)
python tests/single_conv/test_single_conv.py --data_dir /path/to/sampledata
```

Run from the repo root so `pointcnnpp` and `sparse_engines_cuda` are importable (or install the package and run from any directory).
