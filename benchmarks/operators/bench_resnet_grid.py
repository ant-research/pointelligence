"""ResNet end-to-end bench across depths × channel-scales × kernel-modes.

Sweeps the (N, C) plane that real architectures encounter:
- ResNet depths: 18 (BasicBlock), 34 (BasicBlock), 50 (Bottleneck)
- Channel scaling: 0.25× / 0.5× / 1.0× / 2.0× — covers the spectrum from
  "tiny C" (where per-triplet wins) to "fat C" (where grouped wins).
- Kernel mode: grouped (tensor-core, default) vs per-triplet (legacy).

For every (depth, scale, dtype, mode) cell, runs forward+backward on a
synthetic point cloud and times the median. Output is a table that
shows: which mode wins for that cell, by how much, and whether the
default dispatcher would pick the winner.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \\
        .venv/bin/python benchmarks/operators/bench_resnet_grid.py
    # add --dtype fp32 or --dtype bf16 to focus on one dtype
    # --depths 18,34 to subset
    # --scales 1.0,2.0 to subset
"""
from __future__ import annotations
import argparse
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

warnings.filterwarnings("ignore", category=FutureWarning)

import torch

from models.resnet import resnet18, resnet34, resnet50
from sparse_engines._dispatch_override import dispatch_mode

device = "cuda"

DEPTHS = [
    ("resnet18", resnet18, "BasicBlock × [2,2,2,2]"),
    ("resnet34", resnet34, "BasicBlock × [3,4,6,3]"),
    ("resnet50", resnet50, "Bottleneck × [3,4,6,3]"),
]

SCALES = [0.25, 0.5, 1.0, 2.0]
DTYPES = [
    ("fp16", torch.float16),
    ("fp32", torch.float32),
    ("bf16", torch.bfloat16),
]
MODES = [
    ("tig",           "force_tig"),  # v1.2.0 engine.
    ("grouped",       "force_fsg"),  # v1.1.0 engine.
    ("per_triplet",   "force_pt"),   # v1.0.0 engine.
    ("auto",          "auto"),  # Production dispatcher (v1.2.0: routes TIG).
]


def time_loop(fn, n_warmup=3, n_iters=8):
    """Trim-median timing. Lower iter count than the per-kernel bench
    because each ResNet fwd+bwd takes 50-500 ms — 8 iters is ~3 s/cell."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    return ts[n // 2]


def synthesise_pointcloud(N: int, in_channels: int, dtype, device):
    """Random points in a [0, 1]³ box — simple geometric distribution
    that gives roughly uniform triplet density across stages."""
    torch.manual_seed(0)
    points = torch.rand(N, 3, device=device)
    sample_sizes = torch.tensor([N], device=device, dtype=torch.long)
    grid_size = 1.0 / 256
    feat = torch.randn(N, in_channels, device=device, dtype=dtype)
    return points, sample_sizes, grid_size, feat


_SCENE_CACHE: dict = {}


def load_real_scene(scene_coord_path: str, in_channels: int, dtype, device,
                    grid_size: float = 0.02):
    """Load a real preprocessed point cloud from <scene_coord_path>.

    Expected layout: a directory under Pointcept's per-scene preprocessing
    (e.g., ScanNet v2 val sceneXXXX_NN/) containing at least `coord.npy`
    of shape (N, 3) float32 in meters. The reference scene called out in
    .claude/skills/training-infra/references/local-dataset-mirror.md is
    `scannet_v2/val/scene0011_00` (237,360 points, ~6×8×3 m bbox,
    preprocessed at 0.02 m voxel size).

    Reuses Pointcept's `GridSample` transform (the same preprocessing the
    production train/val data pipeline applies before the model sees a
    scene) — train-mode, one representative per voxel, seeded for
    determinism. `grid_size` defaults to 0.02 m to match the preprocessing
    voxel size; the bench's conv1 stride and per-stage cascade are then
    consistent with what production training sees.
    """
    cache_key = (scene_coord_path, grid_size)
    if cache_key in _SCENE_CACHE:
        points, sample_sizes = _SCENE_CACHE[cache_key]
        N = points.shape[0]
        # Re-roll the feature for this cell's dtype (kernel binary is the
        # same; only the dtype-cast feature tensor needs refreshing).
        torch.manual_seed(0)
        feat = torch.randn(N, in_channels, device=device, dtype=dtype)
        return points, sample_sizes, grid_size, feat

    import os, sys
    if os.path.isdir(scene_coord_path):
        coord_path = os.path.join(scene_coord_path, "coord.npy")
    else:
        coord_path = scene_coord_path
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(
            f"--scene-coord: {coord_path} not found (expected a directory "
            f"containing coord.npy, or the path to coord.npy itself)"
        )
    # Make Pointcept's GridSample importable when bench is run with bare
    # PYTHONPATH=.; the overlay's build tree is the canonical Pointcept
    # source for this repo.
    pointcept_root = os.path.join(REPO, "build", "Pointcept")
    if pointcept_root not in sys.path:
        sys.path.insert(0, pointcept_root)
    import numpy as np
    from pointcept.datasets.transform import GridSample

    coord_np = np.load(coord_path).astype(np.float32)
    assert coord_np.ndim == 2 and coord_np.shape[1] == 3, (
        f"coord.npy must be (N, 3); got {coord_np.shape}"
    )
    N_before = coord_np.shape[0]

    # Apply the production GridSample (train mode → 1 rep per voxel,
    # randomly chosen; we seed numpy so the choice is reproducible).
    np.random.seed(0)
    data = {"coord": coord_np, "index_valid_keys": ["coord"]}
    data = GridSample(grid_size=grid_size, mode="train")(data)
    coord_np = data["coord"]

    points = torch.from_numpy(coord_np).to(device)
    N = points.shape[0]
    sample_sizes = torch.tensor([N], device=device, dtype=torch.long)
    bbox = (points.max(0).values - points.min(0).values).tolist()
    print(f"  loaded real scene via GridSample: N={N:,} "
          f"(pre={N_before:,} at grid_size={grid_size} m), "
          f"bbox={[round(b,2) for b in bbox]} m, in_channels={in_channels}")
    _SCENE_CACHE[cache_key] = (points, sample_sizes)
    # Per-point feature: a single-channel synthetic projection of coord
    # (cheap, reproducible, and decoupled from per-cell RNG state). Matches
    # the synthetic path's `in_channels=1` shape so the kernel binary is
    # identical across the two modes.
    torch.manual_seed(0)
    feat = torch.randn(N, in_channels, device=device, dtype=dtype)
    return points, sample_sizes, grid_size, feat


def bench_one_cell(depth_name, factory, scale, dtype, mode_label, mode_arg,
                   scene_coord=None):
    """One (depth, scale, dtype, mode) cell. Returns (e2e_fwd_ms, e2e_fwdbwd_ms, peak_vram_mb)."""
    in_channels = 1

    if scene_coord is not None:
        points, sample_sizes, grid_size, feat = load_real_scene(
            scene_coord, in_channels, dtype, device,
        )
        # Real ScanNet scenes have sparse-edge regions where the default
        # receptive_field_scaler=1.0 yields no neighbors at some stride-2
        # query points. Production unet_pointcnnpp uses 2.5 — match it so
        # the bench reflects what a real conv would do on this density.
        rfs = 2.5
    else:
        N = 10_000
        points, sample_sizes, grid_size, feat = synthesise_pointcloud(
            N, in_channels, dtype, device,
        )
        rfs = 1.0

    model = factory(in_channels=in_channels, width_multiplier=scale).to(device).to(dtype)
    target = torch.randint(0, 1000, (1,), device=device)

    def fwd():
        with torch.no_grad():
            with dispatch_mode(mode_arg):
                model(feat, points, sample_sizes, grid_size,
                      receptive_field_scaler=rfs)

    def fwdbwd():
        with dispatch_mode(mode_arg):
            for p in model.parameters():
                p.grad = None
            x = feat.detach().clone().requires_grad_(True)
            out = model(x, points, sample_sizes, grid_size,
                        receptive_field_scaler=rfs)
            loss = torch.nn.functional.cross_entropy(out, target)
            loss.backward()

    torch.cuda.reset_peak_memory_stats(device)
    try:
        fwd_ms = time_loop(fwd)
        fb_ms = time_loop(fwdbwd)
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        msg = str(exc)
        if "out of memory" in msg.lower():
            print(f"  OOM at {depth_name} × {scale} × {dtype} × {mode_label}; skipping")
            torch.cuda.empty_cache()
            return None, None, None
        raise
    finally:
        del model, points, sample_sizes, feat, target
        torch.cuda.empty_cache()

    return fwd_ms, fb_ms, vram_mb


def run_grid(depths, scales, dtypes, modes, scene_coord=None):
    print(f"ResNet end-to-end bench grid — {torch.cuda.get_device_name(0)}")
    if scene_coord is not None:
        print(f"Input data: real scene at {scene_coord}")
    else:
        print(f"Input data: synthetic (random points in [0,1]^3, N=10,000)")
    print(f"Depths: {[d[0] for d in depths]}, "
          f"scales: {scales}, "
          f"dtypes: {[d[0] for d in dtypes]}, "
          f"modes: {[m[0] for m in modes]}\n")

    # Stored as: (depth_name, scale, dtype_name, mode_label) -> dict
    results = {}

    for dtype_name, dtype in dtypes:
        for depth_name, factory, depth_desc in depths:
            print(f"\n## {dtype_name} × {depth_name}  ({depth_desc})\n")
            print("| scale | mode | fwd ms | fwd+bwd ms | peak VRAM (MB) |")
            print("|---:|:---|---:|---:|---:|")
            for scale in scales:
                for mode_label, mode_arg in modes:
                    fw, fb, vram = bench_one_cell(
                        depth_name, factory, scale, dtype, mode_label, mode_arg,
                        scene_coord=scene_coord,
                    )
                    if fw is None:
                        print(f"| {scale:.2f}× | {mode_label} | OOM | OOM | OOM |")
                    else:
                        print(f"| {scale:.2f}× | {mode_label} | "
                              f"{fw:.2f} | {fb:.2f} | {vram:.0f} |")
                        results[(depth_name, scale, dtype_name, mode_label)] = {
                            "fwd_ms": round(fw, 3),
                            "fb_ms": round(fb, 3),
                            "vram_mb": round(vram, 0),
                        }
    return results


def winner_analysis(results):
    """Per (depth, scale, dtype) cell, identify which mode wins and how
    much the dispatcher's "auto" choice deviates from the empirical
    optimum."""
    print("\n\n## Winner analysis — kernel-mode head-to-head per cell\n")
    print("| depth | scale | dtype | best mode (fwd+bwd) | best ms | "
          "auto ms | auto picks best? | auto/best ratio |")
    print("|---|---:|---|---|---:|---:|:---:|---:|")

    cells = {}
    for (depth, scale, dt, mode), vals in results.items():
        cells.setdefault((depth, scale, dt), {})[mode] = vals["fb_ms"]

    for (depth, scale, dt), modes in sorted(cells.items()):
        if "auto" not in modes:
            continue
        # Best of grouped/per_triplet (excluding auto, since auto IS one of them).
        candidates = {k: v for k, v in modes.items() if k != "auto"}
        if not candidates:
            continue
        best_mode = min(candidates, key=candidates.get)
        best_ms = candidates[best_mode]
        auto_ms = modes["auto"]
        # Tolerance: a 5% margin counts as "auto picks best" — atomic-add
        # jitter alone can swing 5% between consecutive runs.
        picks_best = (auto_ms - best_ms) / best_ms < 0.05
        ratio = auto_ms / best_ms
        flag = "✅" if picks_best else "❌"
        print(f"| {depth} | {scale:.2f}× | {dt} | "
              f"**{best_mode}** ({modes[best_mode]:.2f} ms) | {best_ms:.2f} | "
              f"{auto_ms:.2f} | {flag} | {ratio:.2f}× |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "all"],
                         default="fp16")
    parser.add_argument("--depths", default="18,34,50")
    parser.add_argument("--scales", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--scene-coord", default=None,
                        help="Path to a real scene's coord.npy (or the scene directory "
                             "containing coord.npy). When set, replaces the synthetic "
                             "10K-point random cloud with the real scene. Reference: any "
                             "Pointcept-preprocessed ScanNet val scene directory containing "
                             "{coord,color,...}.npy — release benches used scene0011_00.")
    args = parser.parse_args()

    if args.dtype == "all":
        dtypes = DTYPES
    else:
        dtypes = [d for d in DTYPES if d[0] == args.dtype]

    selected_depths = set(args.depths.split(","))
    depths = [d for d in DEPTHS if d[0].replace("resnet", "") in selected_depths]

    selected_scales = [float(s) for s in args.scales.split(",")]
    scales = [s for s in SCALES if s in selected_scales]

    if not depths or not scales or not dtypes:
        print(f"Empty selection: depths={depths}, scales={scales}, dtypes={dtypes}")
        sys.exit(1)

    results = run_grid(depths, scales, dtypes, MODES, scene_coord=args.scene_coord)
    winner_analysis(results)


if __name__ == "__main__":
    main()
