"""Real-scene benchmark for full-cover strided point convolution.

Fair comparison:
  control:   K15, same radius, same center-nearest C0, closure edges, no added tokens
  treatment: K15, same radius, same center-nearest C0, residual R-net centers

This script refuses synthetic input for timing. Use --quick only for development
smokes; performance reports should use the default warmup/repetition settings.
"""
import argparse
import json
import os
import statistics
import sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from internals.indexing import cumsum_exclusive, repeat_interleave_indices
from internals.neighbors import radius_search
from layers import PointConv3d
from layers.triplets import (
    _center_nearest_sources,
    _strict_voxelize_offsets,
    build_full_cover_strided_rulebook,
    full_cover_radius_scaler,
)


def _load_voxelized_scene(root, scene, voxel_size):
    path = os.path.join(root, scene, "coord.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = np.load(path)
    grid = np.round(raw / voxel_size).astype(np.int64)
    keys = grid[:, 0] * 1_000_000_000 + grid[:, 1] * 1_000_000 + grid[:, 2]
    _, uniq = np.unique(keys, return_index=True)
    return torch.from_numpy((grid[uniq] * voxel_size).astype(np.float32)).cuda()


def _sync():
    torch.cuda.synchronize()


def _time_cuda(fn, warmups, reps, inner):
    for _ in range(warmups):
        fn()
    _sync()
    times = []
    peaks = []
    for _ in range(reps):
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / inner)
        peak = max(0, torch.cuda.max_memory_allocated() - baseline)
        peaks.append(peak / (1024 ** 2))
    return {
        "median_ms": statistics.median(times),
        "iqr_ms": (np.percentile(times, 75) - np.percentile(times, 25)).item(),
        "runs_ms": times,
        "incremental_peak_mib_median": statistics.median(peaks),
    }


def _build_control_rulebook(points, sample_inds, sample_sizes, stride, grid_size,
                            kernel_size, radius_margin, backend):
    radius_scaler = full_cover_radius_scaler(stride, radius_margin)
    radius = grid_size * radius_scaler
    c0, point_to_c0 = _center_nearest_sources(points, sample_inds, stride * grid_size)
    centers = points[c0].contiguous()
    center_sample = sample_inds[c0].contiguous()
    neighbors, counts = radius_search(
        points=points,
        query_points=centers,
        radius=radius,
        sample_inds=sample_inds,
        query_sample_inds=center_sample,
        backend=backend,
    )
    offsets, total = cumsum_exclusive(counts, return_sum=True)
    i = repeat_interleave_indices(repeats_cumsum=offsets, output_size=total, may_contain_zero_repeats=False).long()
    j = neighbors.long()
    coverage = torch.zeros(points.shape[0], device=points.device, dtype=torch.long)
    if j.numel() > 0:
        coverage.index_add_(0, j, torch.ones(j.numel(), device=points.device, dtype=torch.long))
    missing = torch.nonzero(coverage == 0).squeeze(1).long()
    closure_edges = missing.numel()
    if closure_edges > 0:
        i = torch.cat([i, point_to_c0[missing].long()])
        j = torch.cat([j, missing])
        coverage[missing] += 1
        counts = counts + torch.bincount(
            point_to_c0[missing].long(), minlength=centers.shape[0]
        ).to(counts.dtype)
    k, _ = _strict_voxelize_offsets(
        points[j] - centers[i], grid_size=grid_size, kernel_size=kernel_size,
        context="full-cover benchmark control")
    k, order = torch.sort(k)
    i_sorted = i[order].int()
    j_sorted = j[order].int()

    k_up, _ = _strict_voxelize_offsets(
        centers[i] - points[j], grid_size=grid_size, kernel_size=kernel_size,
        context="full-cover benchmark control reverse")
    k_up, up_order = torch.sort(k_up)
    return {
        "points": centers,
        "sample_inds": center_sample,
        "i": i_sorted,
        "j": j_sorted,
        "k": k.int(),
        "num_neighbors": counts,
        "i_upsample": j[up_order].int(),
        "j_upsample": i[up_order].int(),
        "k_upsample": k_up.int(),
        "point_to_initial_center": point_to_c0,
        "coverage_per_input": coverage,
        "radius": radius,
        "radius_scaler": radius_scaler,
        "initial_centers": int(c0.numel()),
        "added_centers": 0,
        "closure_edges": int(closure_edges),
        "edge_count": int(k.numel()),
    }


def _conv_step(conv, feat, i, j, k, n):
    x = feat.detach().clone().requires_grad_(True)
    out = conv(x, i, j, k, n)
    loss = out.float().square().mean()
    loss.backward()
    return out


def _add_comparison(results):
    c = results["control"]
    t = results["treatment"]
    c_rule = c["rulebook"]["median_ms"]
    t_rule = t["rulebook"]["median_ms"]
    c_fb = c["forward_backward"]["median_ms"]
    t_fb = t["forward_backward"]["median_ms"]
    c_peak = c["rulebook"]["incremental_peak_mib_median"]
    t_peak = t["rulebook"]["incremental_peak_mib_median"]
    results["comparison"] = {
        "center_delta": int(t["initial_centers"] + t["added_centers"] - c["initial_centers"]),
        "edge_delta": int(t["edge_count"] - c["edge_count"]),
        "rulebook_overhead_ms": t_rule - c_rule,
        "rulebook_ratio": t_rule / c_rule if c_rule else None,
        "forward_backward_overhead_ms": t_fb - c_fb,
        "forward_backward_ratio": t_fb / c_fb if c_fb else None,
        "rulebook_peak_overhead_mib": t_peak - c_peak,
        "rulebook_peak_ratio": t_peak / c_peak if c_peak else None,
        "control_fanout": c["edge_count"] / max(1, c["initial_centers"]),
        "treatment_fanout": t["edge_count"] / max(1, t["initial_centers"] + t["added_centers"]),
    }
    return results


def _split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _bench_one(args, scene, grid_size):
    points = _load_voxelized_scene(args.data_root, scene, grid_size)
    sample_inds = torch.zeros(points.shape[0], device="cuda", dtype=torch.long)
    sample_sizes = torch.bincount(sample_inds)
    feat = torch.randn(points.shape[0], args.channels, device="cuda", dtype=torch.float16)
    conv = PointConv3d(args.channels, args.channels, args.kernel_size, bias=False,
                       device="cuda", dtype=torch.float16)

    control_holder = {}
    treatment_holder = {}

    def build_control():
        control_holder["rb"] = _build_control_rulebook(
            points, sample_inds, sample_sizes, args.stride, grid_size,
            args.kernel_size, args.radius_margin, args.backend)

    def build_treatment():
        treatment_holder["rb"] = build_full_cover_strided_rulebook(
            points, sample_inds, sample_sizes, stride=args.stride,
            input_grid_size=grid_size, kernel_size=args.kernel_size,
            radius_margin=args.radius_margin, radius_backend=args.backend)

    build_control()
    build_treatment()
    c = control_holder["rb"]
    t = treatment_holder["rb"]
    assert c["radius"] == t.radius
    assert c["initial_centers"] == int(t.initial_center_source_indices.numel())
    assert torch.all(c["coverage_per_input"] >= 1)
    assert c["i_upsample"].numel() == c["i"].numel()
    assert int(c["num_neighbors"].sum().item()) == c["i"].numel()
    assert torch.all(c["k"][1:] >= c["k"][:-1])
    assert torch.all(c["k_upsample"][1:] >= c["k_upsample"][:-1])
    forward_reverse_key = c["j"].long() * c["points"].shape[0] + c["i"].long()
    cached_reverse_key = (
        c["i_upsample"].long() * c["points"].shape[0]
        + c["j_upsample"].long()
    )
    assert torch.equal(
        torch.sort(forward_reverse_key).values,
        torch.sort(cached_reverse_key).values,
    )
    assert torch.all(t.coverage_per_input >= 1)

    def control_conv():
        rb = control_holder["rb"]
        return _conv_step(conv, feat, rb["i"], rb["j"], rb["k"], rb["points"].shape[0])

    def treatment_conv():
        rb = treatment_holder["rb"]
        return _conv_step(conv, feat, rb.i, rb.j, rb.k, rb.points.shape[0])

    results = {
        "scene": scene,
        "n_points": int(points.shape[0]),
        "grid_size": grid_size,
        "stride": args.stride,
        "kernel_size": args.kernel_size,
        "channels": args.channels,
        "backend": args.backend,
        "radius_margin": args.radius_margin,
        "warmups": args.warmups,
        "reps": args.reps,
        "inner": args.inner,
        "control": {
            "initial_centers": c["initial_centers"],
            "added_centers": 0,
            "closure_edges": c["closure_edges"],
            "edge_count": c["edge_count"],
            "rulebook": _time_cuda(build_control, args.warmups, args.reps, args.inner),
            "forward_backward": _time_cuda(control_conv, args.warmups, args.reps, args.inner),
        },
        "treatment": {
            "initial_centers": int(t.initial_center_source_indices.numel()),
            "added_centers": int(t.additional_center_source_indices.numel()),
            "added_center_fraction": float(t.additional_center_source_indices.numel()) / max(1, int(t.initial_center_source_indices.numel())),
            "selector_rounds": t.selector_round_count,
            "edge_count": int(t.k.numel()),
            "rulebook": _time_cuda(build_treatment, args.warmups, args.reps, args.inner),
            "forward_backward": _time_cuda(treatment_conv, args.warmups, args.reps, args.inner),
        },
    }
    return _add_comparison(results)


def _summary(rows):
    def med(key):
        vals = [row["comparison"][key] for row in rows if row["comparison"][key] is not None]
        return statistics.median(vals) if vals else None
    return {
        "num_cases": len(rows),
        "median_rulebook_overhead_ms": med("rulebook_overhead_ms"),
        "median_rulebook_ratio": med("rulebook_ratio"),
        "median_forward_backward_overhead_ms": med("forward_backward_overhead_ms"),
        "median_forward_backward_ratio": med("forward_backward_ratio"),
        "median_rulebook_peak_overhead_mib": med("rulebook_peak_overhead_mib"),
        "median_rulebook_peak_ratio": med("rulebook_peak_ratio"),
        "total_added_centers": sum(row["treatment"]["added_centers"] for row in rows),
        "total_closure_edges": sum(row["control"]["closure_edges"] for row in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        required=True,
        help="ScanNet validation root containing <scene>/coord.npy",
    )
    ap.add_argument("--scene", default="scene0011_00")
    ap.add_argument("--scenes", help="Comma-separated scene names. Overrides --scene.")
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--grid-sizes", help="Comma-separated grid sizes. Overrides --grid-size.")
    ap.add_argument("--stride", type=float, default=8.0)
    ap.add_argument("--kernel-size", type=int, default=15)
    ap.add_argument("--radius-margin", type=float, default=1e-2)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--warmups", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--inner",
        type=int,
        default=1,
        help="Operations per measured sample; latency is reported per operation.",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.quick:
        args.warmups = 2
        args.reps = 1
    if args.inner < 1:
        ap.error("--inner must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    scenes = _split_csv(args.scenes) or [args.scene]
    grid_sizes = [float(x) for x in _split_csv(args.grid_sizes)] or [args.grid_size]
    rows = []
    for scene in scenes:
        for grid_size in grid_sizes:
            try:
                row = _bench_one(args, scene, grid_size)
            except Exception as exc:
                row = {
                    "scene": scene,
                    "grid_size": grid_size,
                    "backend": args.backend,
                    "channels": args.channels,
                    "kernel_size": args.kernel_size,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print(json.dumps(row, indent=2, sort_keys=True))
    completed = [row for row in rows if "comparison" in row]
    payload = {"results": rows, "summary": _summary(completed) if completed else {"num_cases": 0}}
    if len(rows) > 1:
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
