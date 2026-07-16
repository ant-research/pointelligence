"""Correctness-first grid-downsampling benchmark on transformed ScanNet scenes.

The only causal variable is center-nearest selection:
  control   -- torch tensor materialization and segmented reductions
  treatment -- compact Triton segmented argmin with inverse mapping elided

Each workload is built from real scenes after Pointcept CenterShift and
train-mode GridSample. Arms run sequentially with synchronized timing.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "build", "Pointcept"))

from internals.grid_indexing import build_sorted_grid_segments, compute_grid_indices
from internals.grid_sample import (
    _center_nearest_indices_torch,
    grid_sample_filter,
)
from internals.grid_sample_triton_kernel import center_nearest_segment_indices

transform_path = os.path.join(
    REPO, "build", "Pointcept", "pointcept", "datasets", "transform.py"
)
transform_spec = importlib.util.spec_from_file_location(
    "pointcept_transform_for_benchmark", transform_path
)
transform_module = importlib.util.module_from_spec(transform_spec)
sys.modules[transform_spec.name] = transform_module
transform_spec.loader.exec_module(transform_module)
CenterShift = transform_module.CenterShift
GridSample = transform_module.GridSample


SCENES = [
    "scene0011_00",
    "scene0015_00",
    "scene0046_00",
    "scene0231_00",
    "scene0019_00",
    "scene0025_00",
    "scene0030_00",
    "scene0050_00",
]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def transformed_batch(root: str, batch: int, input_grid: float):
    points = []
    sizes = []
    for index, scene in enumerate(SCENES[:batch]):
        coord = np.load(os.path.join(root, scene, "coord.npy")).astype(
            np.float32, copy=True
        )
        data = {"coord": coord, "index_valid_keys": ["coord"]}
        data = CenterShift(apply_z=True)(data)
        np.random.seed(1701 + index)
        data = GridSample(grid_size=input_grid, mode="train")(data)
        coord = np.ascontiguousarray(data["coord"])
        points.append(torch.from_numpy(coord))
        sizes.append(coord.shape[0])
    points_cuda = torch.cat(points).cuda()
    sizes_cuda = torch.tensor(sizes, device="cuda", dtype=torch.long)
    sample_inds = torch.repeat_interleave(
        torch.arange(batch, device="cuda"), sizes_cuda
    )
    return points_cuda, sample_inds


def stage_input(points, sample_inds, grid_size, stage):
    for _ in range(stage):
        points, sample_inds, _, _ = grid_sample_filter(
            points,
            grid_size=grid_size * 2,
            sample_inds=sample_inds,
            center_nearest_impl="torch",
        )
        grid_size *= 2
    return points, sample_inds, grid_size


def stats(values):
    return (
        statistics.median(values),
        float(np.percentile(values, 75) - np.percentile(values, 25)),
    )


def timed(fn, warmups, runs, iterations):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    wall_ms = []
    device_ms = []
    for _ in range(runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_wall = time.perf_counter()
        start_event.record()
        for _ in range(iterations):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - start_wall) * 1000 / iterations)
        device_ms.append(start_event.elapsed_time(end_event) / iterations)
    return stats(wall_ms), stats(device_ms)


def peak_delta_mb(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return (peak - baseline) / (1024 * 1024)


def prepare_selector(points, sample_inds, target_grid):
    grid_inds = compute_grid_indices(points, target_grid, sample_inds)
    segments = build_sorted_grid_segments(grid_inds, return_inverse=True)
    return grid_inds, segments


def benchmark_workload(points, sample_inds, input_grid, warmups, runs, iterations):
    target_grid = input_grid * 2
    arms = {}
    for implementation in ("torch", "triton"):
        arms[implementation] = lambda implementation=implementation: (
            grid_sample_filter(
                points,
                grid_size=target_grid,
                sample_inds=sample_inds,
                return_mapping=False,
                center_nearest_impl=implementation,
            )
        )

    expected = arms["torch"]()
    actual = arms["triton"]()
    torch.cuda.synchronize()
    for index in (0, 1, 2):
        if not torch.equal(expected[index], actual[index]):
            raise AssertionError(f"full-output parity failed at tuple element {index}")

    grid_inds, segments = prepare_selector(points, sample_inds, target_grid)
    sorter, counts, lookup_inds = segments
    selected = torch.empty_like(counts, dtype=sorter.dtype)
    torch_selected = _center_nearest_indices_torch(
        points, grid_inds, target_grid, sorter, counts, lookup_inds
    )
    triton_selected = center_nearest_segment_indices(
        points,
        grid_inds,
        sorter,
        counts,
        target_grid,
        out=selected.clone(),
    )
    torch.cuda.synchronize()
    if not torch.equal(torch_selected, triton_selected):
        raise AssertionError("isolated selector parity failed")

    selector_arms = {
        "torch": lambda: _center_nearest_indices_torch(
            points, grid_inds, target_grid, sorter, counts, lookup_inds
        ),
        "triton": lambda: center_nearest_segment_indices(
            points,
            grid_inds,
            sorter,
            counts,
            target_grid,
            out=selected,
        ),
    }
    segment_arms = {
        "inverse": lambda: build_sorted_grid_segments(grid_inds, return_inverse=True),
        "minimal": lambda: build_sorted_grid_segments(grid_inds, return_inverse=False),
    }

    result = {
        "points": points.shape[0],
        "outputs": expected[0].shape[0],
    }
    for label, fn in arms.items():
        result[f"total_{label}"] = timed(fn, warmups, runs, iterations)
        result[f"peak_{label}"] = peak_delta_mb(fn)
    for label, fn in selector_arms.items():
        result[f"selector_{label}"] = timed(fn, warmups, runs, iterations)
    for label, fn in segment_arms.items():
        result[f"segments_{label}"] = timed(fn, warmups, runs, iterations)
    return result


def fmt(value):
    (wall, wall_iqr), (device, device_iqr) = value
    return (
        f"wall {wall:.4f}±{wall_iqr:.4f} ms; "
        f"device {device:.4f}±{device_iqr:.4f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get("TIG_BENCH_SCANNET"),
        help=(
            "ScanNet validation root containing <scene>/coord.npy; "
            "defaults to TIG_BENCH_SCANNET"
        ),
    )
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--stages", default="0,1,2,3")
    parser.add_argument("--input-grid", type=float, default=0.02)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not args.root:
        parser.error("--root or TIG_BENCH_SCANNET is required")

    print(torch.cuda.get_device_name(0))
    print("correctness before timing; real transformed ScanNet only")
    for batch in parse_csv_ints(args.batches):
        base_points, base_sample_inds = transformed_batch(
            args.root, batch, args.input_grid
        )
        for stage in parse_csv_ints(args.stages):
            points, sample_inds, grid_size = stage_input(
                base_points, base_sample_inds, args.input_grid, stage
            )
            result = benchmark_workload(
                points,
                sample_inds,
                grid_size,
                args.warmups,
                args.runs,
                args.iterations,
            )
            control = result["total_torch"][0][0]
            treatment = result["total_triton"][0][0]
            speedup = control / treatment
            print(
                f"B={batch} stage={stage} N={result['points']:,} "
                f"M={result['outputs']:,} speedup={speedup:.3f}x"
            )
            print(f"  total torch:   {fmt(result['total_torch'])}")
            print(f"  total triton:  {fmt(result['total_triton'])}")
            print(f"  selector torch:{fmt(result['selector_torch'])}")
            print(f"  selector triton:{fmt(result['selector_triton'])}")
            print(f"  segments inverse:{fmt(result['segments_inverse'])}")
            print(f"  segments minimal:{fmt(result['segments_minimal'])}")
            print(
                f"  peak transient: torch={result['peak_torch']:.2f} MiB "
                f"triton={result['peak_triton']:.2f} MiB"
            )
        del base_points, base_sample_inds
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
