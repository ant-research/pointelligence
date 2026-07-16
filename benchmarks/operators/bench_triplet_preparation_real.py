"""Real-scene benchmark for direct kernel-segment triplet construction.

Control and treatment use identical points, queries, radius, kernel, and
sorted-eight neighbor search. The control emits query-major neighbors and then
constructs, sorts, and gathers (i, j, k). The treatment emits directly into
kernel-tap segments and hands those segments to TIG without re-derivation.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from internals.grid_sample import grid_sample_filter
from internals.neighbors import (
    radius_search_sorted_grid8,
    radius_search_sorted_grid8_segments,
)
from layers.triplets import voxelize_3d
from sparse_engines.tig import TigIndex


DATASETS = {
    "scannet": (
        os.environ.get("RADIUS_BENCH_SCANNET_GLOB"),
        0.02,
    ),
    "nuscenes": (
        os.environ.get("RADIUS_BENCH_NUSCENES_GLOB"),
        0.05,
    ),
    "waymo": (
        os.environ.get("RADIUS_BENCH_WAYMO_GLOB"),
        0.05,
    ),
}
WORKLOADS = {
    "k1_stride1": (1, 1, None),
    "k3_stride2": (3, 2, None),
    "k5_stride2": (5, 2, None),
    "k7_stride2": (7, 2, None),
    "k15_stride8": (15, 8, math.sqrt(3.0) * 4.0 * 1.01),
    "k15_stride2_stress": (15, 2, math.sqrt(3.0) * 4.0 * 1.01),
}


def load_coord(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        coord = np.load(path)
    elif path.endswith(".bin"):
        raw = np.fromfile(path, dtype=np.float32)
        columns = 5 if raw.size % 5 == 0 else 4
        coord = raw.reshape(-1, columns)[:, :3]
    else:
        raise ValueError(f"unsupported coordinate file: {path}")
    return torch.from_numpy(np.asarray(coord, dtype=np.float32)).cuda()


def production_geometry(paths: list[str], grid_size: float, stride: int):
    point_parts = []
    sample_parts = []
    for batch, path in enumerate(paths):
        raw = load_coord(path)
        raw_sample = torch.zeros(
            raw.shape[0], dtype=torch.long, device="cuda")
        points, _, _, _ = grid_sample_filter(
            points=raw,
            grid_size=grid_size,
            sample_inds=raw_sample,
            reduction="center_nearest",
            return_mapping=True,
        )
        point_parts.append(points)
        sample_parts.append(torch.full(
            (points.shape[0],), batch, dtype=torch.long, device="cuda"))
    points = torch.cat(point_parts)
    sample_inds = torch.cat(sample_parts)
    queries, query_sample_inds, _, _ = grid_sample_filter(
        points=points,
        grid_size=grid_size * stride,
        sample_inds=sample_inds,
        reduction="center_nearest",
        return_mapping=True,
    )
    return points, queries, sample_inds, query_sample_inds


def control_search(points, queries, sample_inds, query_sample_inds, radius):
    return radius_search_sorted_grid8(
        points,
        queries,
        radius,
        sample_inds,
        query_sample_inds,
        dtype_num_neighbors=torch.int64,
    )


def control_post(
    points, queries, neighbors, counts, kernel_size, kernel_grid_size
):
    i = torch.repeat_interleave(
        torch.arange(
            queries.shape[0], dtype=torch.int32, device=queries.device),
        counts,
    )
    k = voxelize_3d(
        kernel_size,
        points[neighbors] - queries[i],
        torch.tensor(kernel_grid_size, device=points.device),
    )
    k, order = torch.sort(k)
    return i[order].to(torch.int32), neighbors[order].to(torch.int32), k.to(
        torch.int32)


def direct_search(
    points, queries, sample_inds, query_sample_inds, radius, kernel_size,
    kernel_grid_size, tap_stripes,
):
    return radius_search_sorted_grid8_segments(
        points,
        queries,
        radius,
        kernel_size,
        kernel_grid_size,
        sample_inds,
        query_sample_inds,
        dtype_num_neighbors=torch.int64,
        tap_stripes=tap_stripes,
    )


def materialize_k(seg_offs, kernel_size, output_size):
    return torch.repeat_interleave(
        torch.arange(
            kernel_size ** 3, dtype=torch.int32, device=seg_offs.device),
        seg_offs.diff(),
        output_size=output_size,
    )


def quantiles(values):
    ordered = sorted(values)
    median = statistics.median(ordered)
    q25, q75 = np.percentile(np.asarray(ordered), [25, 75])
    return {
        "values_ms": values,
        "median_ms": float(median),
        "iqr_ms": float(q75 - q25),
        "iqr_fraction": float((q75 - q25) / median) if median else 0.0,
    }


def timed(fn, warmups, runs, inner):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0 / inner)
    return quantiles(values)


def peak_memory(fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    result = {
        "peak_allocated_delta_mib": (
            torch.cuda.max_memory_allocated() - base_allocated) / 2 ** 20,
        "peak_reserved_delta_mib": (
            torch.cuda.max_memory_reserved() - base_reserved) / 2 ** 20,
    }
    del out
    return result


def canonical(i, j, k, num_queries, num_points):
    key = (
        k.long() * (num_queries * num_points)
        + i.long() * num_points
        + j.long()
    )
    key = torch.sort(key).values
    if torch.unique(key).numel() != key.numel():
        raise RuntimeError("duplicate triplet")
    return key


def benchmark_cell(
    dataset, paths, grid_size, workload, warmups, runs, inner, tap_stripes
):
    kernel_size, stride, radius_scaler = WORKLOADS[workload]
    if radius_scaler is None:
        volume = float(kernel_size ** 3)
        radius_scaler = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    radius = grid_size * radius_scaler
    points, queries, sample_inds, query_sample_inds = production_geometry(
        paths, grid_size, stride)

    control_neighbors, control_counts = control_search(
        points, queries, sample_inds, query_sample_inds, radius)
    control_i, control_j, control_k = control_post(
        points, queries, control_neighbors, control_counts, kernel_size,
        grid_size)
    direct_i, direct_j, direct_seg, direct_counts = direct_search(
        points, queries, sample_inds, query_sample_inds, radius, kernel_size,
        grid_size, tap_stripes)
    direct_k = materialize_k(
        direct_seg, kernel_size, direct_i.shape[0])

    control_key = canonical(
        control_i, control_j, control_k, queries.shape[0], points.shape[0])
    direct_key = canonical(
        direct_i, direct_j, direct_k, queries.shape[0], points.shape[0])
    if not torch.equal(control_key, direct_key):
        raise RuntimeError(
            f"triplet mismatch for {dataset}/{workload}")
    if not torch.equal(control_counts, direct_counts):
        raise RuntimeError(
            f"count mismatch for {dataset}/{workload}")
    if not torch.equal(
        direct_seg.diff(),
        torch.bincount(
            control_k.long(), minlength=kernel_size ** 3),
    ):
        raise RuntimeError(
            f"segment mismatch for {dataset}/{workload}")
    del control_key, direct_key

    def control_search_fn():
        return control_search(
            points, queries, sample_inds, query_sample_inds, radius)

    def control_post_fn():
        return control_post(
            points, queries, control_neighbors, control_counts, kernel_size,
            grid_size)

    def direct_search_fn():
        return direct_search(
            points, queries, sample_inds, query_sample_inds, radius,
            kernel_size, grid_size, tap_stripes)

    def direct_k_fn():
        return materialize_k(
            direct_seg, kernel_size, direct_i.shape[0])

    def control_index_fn():
        return TigIndex(
            control_i, control_j, control_k, queries.shape[0],
            kernel_size ** 3, build_hybrid=False, assume_sorted=True,
            n_in=points.shape[0])

    def direct_index_fn():
        return TigIndex.from_flat(
            direct_i, direct_j, direct_seg,
            n_out=queries.shape[0], n_in=points.shape[0],
            num_kernel_offsets=kernel_size ** 3,
            exact_cover_out=False, exact_cover_in=False,
            uniform_seg_len=-1)

    def control_total_fn():
        neighbors, counts = control_search_fn()
        i, j, k = control_post(
            points, queries, neighbors, counts, kernel_size, grid_size)
        return TigIndex(
            i, j, k, queries.shape[0], kernel_size ** 3,
            build_hybrid=False, assume_sorted=True, n_in=points.shape[0])

    def direct_total_fn():
        i, j, seg, _ = direct_search_fn()
        # The public PointConv contract still carries explicit k for fallback
        # engines and diagnostics, so include its compatibility materialization
        # in the fair production-ready total.
        _ = materialize_k(seg, kernel_size, i.shape[0])
        return TigIndex.from_flat(
            i, j, seg,
            n_out=queries.shape[0], n_in=points.shape[0],
            num_kernel_offsets=kernel_size ** 3,
            exact_cover_out=False, exact_cover_in=False,
            uniform_seg_len=-1)

    stages = {}
    for name, fn in (
        ("control_search", control_search_fn),
        ("control_post", control_post_fn),
        ("control_tig_index", control_index_fn),
        ("direct_search_segments", direct_search_fn),
        ("direct_materialize_k", direct_k_fn),
        ("direct_tig_index", direct_index_fn),
        ("control_ready_total", control_total_fn),
        ("direct_ready_total", direct_total_fn),
    ):
        stages[name] = timed(fn, warmups, runs, inner)

    memory = {
        "control_ready_total": peak_memory(control_total_fn),
        "direct_ready_total": peak_memory(direct_total_fn),
    }
    control_median = stages["control_ready_total"]["median_ms"]
    direct_median = stages["direct_ready_total"]["median_ms"]
    row = {
        "dataset": dataset,
        "paths": paths,
        "workload": workload,
        "points": points.shape[0],
        "queries": queries.shape[0],
        "edges": direct_i.shape[0],
        "kernel_size": kernel_size,
        "stride": stride,
        "grid_size": grid_size,
        "radius": radius,
        "radius_scaler": radius_scaler,
        "tap_stripes": tap_stripes,
        "stages": stages,
        "memory": memory,
        "direct_over_control": direct_median / control_median,
        "speedup": control_median / direct_median,
        "dispersion_warning": [
            name for name, result in stages.items()
            if result["iqr_fraction"] > 0.05
        ],
    }
    del points, queries, sample_inds, query_sample_inds
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="scannet,nuscenes,waymo")
    parser.add_argument("--workloads", default="k3_stride2,k15_stride8")
    parser.add_argument("--batch-scenes", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--tap-stripes", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets.split(","):
        pattern, grid_size = DATASETS[dataset]
        if not pattern:
            env_name = f"RADIUS_BENCH_{dataset.upper()}_GLOB"
            raise RuntimeError(
                f"{dataset}: set {env_name} to a real-data coordinate glob")
        paths = sorted(glob.glob(pattern))[:args.batch_scenes]
        if len(paths) != args.batch_scenes:
            raise RuntimeError(
                f"{dataset}: expected {args.batch_scenes} scenes, got {len(paths)}")
        for workload in args.workloads.split(","):
            row = benchmark_cell(
                dataset, paths, grid_size, workload,
                args.warmups, args.runs, args.inner, args.tap_stripes)
            rows.append(row)
            if not args.quiet:
                print(json.dumps(row, sort_keys=True))

    result = {
        "device": torch.cuda.get_device_name(),
        "warmups": args.warmups,
        "runs": args.runs,
        "inner": args.inner,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
