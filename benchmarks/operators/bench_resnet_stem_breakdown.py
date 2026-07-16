"""Real-data ResNet stem timing breakdown.

This diagnostic isolates the first ResNet point convolution:
``conv7x7x7, stride=2, Cin=1``.  The full ResNet breakdown showed this stem
dominates e2e latency, so this script splits it into downsample, triplet
building, and conv compute without changing the shipped model semantics.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from layers import MetaData, voxelize_3d, radius_scaler_for_kernel_size  # noqa: E402
from layers.downsample import downsample  # noqa: E402
from layers.triplets import build_triplets, _build_triplets_from_neighbor_pairs  # noqa: E402
from layers.contract import TripletContract  # noqa: E402
from models.resnet import resnet34  # noqa: E402
from sparse_engines._dispatch_override import dispatch_mode  # noqa: E402
from internals.neighbors import (  # noqa: E402
    radius_search, radius_search_fixed_grid,
    radius_search_sorted_grid8, radius_search_strided_grid,
    radius_search_tiled,
)

device = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET", "")
SCENES = [
    "scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00",
    "scene0030_00", "scene0046_00", "scene0050_00", "scene0064_00",
    "scene0086_00", "scene0095_00", "scene0100_00", "scene0131_00",
    "scene0144_00", "scene0153_00", "scene0164_00", "scene0169_00",
    "scene0187_00", "scene0193_00", "scene0207_00", "scene0217_00",
    "scene0221_00", "scene0011_01", "scene0019_01", "scene0249_00",
]


def load_coord(root: str, scene: str) -> torch.Tensor:
    pth = os.path.join(root, scene, "coord.pth")
    if os.path.exists(pth):
        return torch.load(pth, map_location=device).float()
    npy = os.path.join(root, scene, "coord.npy")
    if os.path.exists(npy):
        import numpy as np
        return torch.from_numpy(np.load(npy)).to(device).float()
    raise FileNotFoundError(f"missing coord.pth/coord.npy for {scene} under {root}")


@dataclass
class _EventPair:
    label: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class EventMeter:
    def __init__(self) -> None:
        self._pairs: list[_EventPair] = []

    def time(self, label: str, fn: Callable):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        self._pairs.append(_EventPair(label, start, end))
        return out

    def elapsed_ms(self) -> dict[str, float]:
        torch.cuda.synchronize()
        out: dict[str, float] = defaultdict(float)
        for pair in self._pairs:
            out[pair.label] += pair.start.elapsed_time(pair.end)
        return dict(out)


def dedup_first(coord: torch.Tensor, gs: float) -> torch.Tensor:
    g = (coord / gs).long()
    uniq, inv = torch.unique(g, dim=0, return_inverse=True)
    first = torch.full((uniq.size(0),), coord.size(0), device=device, dtype=torch.long)
    first.scatter_reduce_(
        0, inv, torch.arange(coord.size(0), device=device), reduce="amin")
    return coord[first.sort().values]


def batched(raws: list[torch.Tensor], gs: float, batch: int):
    coords, sizes = [], []
    for coord in raws[:batch]:
        down = dedup_first(coord, gs)
        coords.append(down)
        sizes.append(down.size(0))
    return torch.cat(coords, 0), torch.tensor(sizes, device=device)


def _sample_inds(sample_sizes: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(0, sample_sizes.numel(), device=sample_sizes.device),
        sample_sizes,
    )


def build_triplets_backend(
    backend: str,
    *,
    points: torch.Tensor,
    sample_inds: torch.Tensor,
    sample_sizes: torch.Tensor,
    neighbor_radius: float,
    kernel_size,
    grid_size: float,
    query_points: torch.Tensor,
    query_sample_inds: torch.Tensor,
    query_sample_sizes: torch.Tensor,
    radius_scaler: float,
):
    sort_by = "i" if backend.endswith("_i") else "k"
    base_backend = backend[:-2] if backend.endswith("_i") else backend
    if base_backend == "default":
        return build_triplets(
            points=points,
            sample_inds=sample_inds,
            sample_sizes=sample_sizes,
            neighbor_radius=neighbor_radius,
            kernel_indexer=partial(voxelize_3d, kernel_size=kernel_size),
            query_points=query_points,
            query_sample_inds=query_sample_inds,
            query_sample_sizes=query_sample_sizes,
            sort_by=sort_by,
            return_num_neighbors=True,
            radius_scaler=radius_scaler,
        )
    if base_backend == "tiled":
        neighbor_indices, num_neighbors = radius_search_tiled(
            points=points,
            queries=query_points,
            radius=neighbor_radius,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
            dtype_num_neighbors=torch.int64,
        )
    elif base_backend == "sorted8":
        neighbor_indices, num_neighbors = radius_search_sorted_grid8(
            points=points,
            queries=query_points,
            radius=neighbor_radius,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
            dtype_num_neighbors=torch.int64,
        )
    elif base_backend == "strided_grid":
        neighbor_indices, num_neighbors = radius_search_strided_grid(
            points=points,
            queries=query_points,
            radius=neighbor_radius,
            cell_size=grid_size * 2,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
            dtype_num_neighbors=torch.int64,
        )
    elif base_backend == "fixed_grid":
        neighbor_indices, num_neighbors = radius_search_fixed_grid(
            points=points,
            queries=query_points,
            radius=neighbor_radius,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
            dtype_num_neighbors=torch.int64,
        )
    elif base_backend == "radius_grid":
        neighbor_indices, num_neighbors = radius_search(
            points=points,
            query_points=query_points,
            radius=neighbor_radius,
            sample_inds=sample_inds,
            query_sample_inds=query_sample_inds,
            sample_sizes=sample_sizes,
            query_sample_sizes=query_sample_sizes,
            grid_size=grid_size,
            tiled_radius_multiplier_threshold=10**9,
        )
    else:
        raise ValueError(f"unknown builder backend: {backend}")
    return _build_triplets_from_neighbor_pairs(
        points=points,
        query_points=query_points,
        neighbor_indices=neighbor_indices,
        num_neighbors=num_neighbors,
        kernel_indexer=partial(voxelize_3d, kernel_size=kernel_size),
        neighbor_radius=neighbor_radius,
        radius_scaler=radius_scaler,
        sort_by=sort_by,
        return_num_neighbors=True,
    )


def split_stem(
    conv,
    feat: torch.Tensor,
    points: torch.Tensor,
    sample_sizes: torch.Tensor,
    grid_size: float,
    receptive_field_scaler: float,
    mode: str,
    builder_backend: str = "default",
) -> tuple[torch.Tensor, MetaData, dict[str, float], dict[str, float | int]]:
    meter = EventMeter()
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    parent = MetaData(
        points=points,
        sample_inds=_sample_inds(sample_sizes),
        sample_sizes=sample_sizes,
        grid_size=grid_size,
        receptive_field_scaler=receptive_field_scaler,
    )
    m = MetaData(
        points=points,
        sample_inds=parent.sample_inds,
        sample_sizes=sample_sizes,
        grid_size=grid_size,
        receptive_field_scaler=receptive_field_scaler,
        parent=parent,
    )
    radius_scaler = radius_scaler_for_kernel_size(
        conv.kernel_size_3, receptive_field_scaler, "ball")
    neighbor_radius = parent.grid_size * radius_scaler

    def do_downsample():
        return downsample(m.points, m.sample_inds, m.grid_size, 2)

    m.points, m.sample_inds, m.grid_size, m.downsample_indices = meter.time(
        "downsample", do_downsample)
    m.sample_sizes = torch.bincount(m.sample_inds)

    def do_build():
        return build_triplets_backend(
            builder_backend,
            points=parent.points,
            sample_inds=parent.sample_inds,
            sample_sizes=parent.sample_sizes,
            neighbor_radius=neighbor_radius,
            kernel_size=conv.kernel_size_3,
            grid_size=parent.grid_size,
            query_points=m.points,
            query_sample_inds=m.sample_inds,
            query_sample_sizes=m.sample_sizes,
            radius_scaler=radius_scaler,
        )

    m.i, m.j, m.k, m.num_neighbors = meter.time("build_triplets", do_build)
    m.contract = TripletContract(k_sorted=not builder_backend.endswith("_i"))

    with dispatch_mode(mode):
        y = meter.time(
            "conv_compute",
            lambda: conv(feat, m.i, m.j, m.k, m.num_points(), contract=m.contract),
        )
    total_end.record()
    rows = meter.elapsed_ms()
    rows["stem_total"] = total_start.elapsed_time(total_end)
    rows["unattributed"] = rows["stem_total"] - sum(
        value for key, value in rows.items() if key != "stem_total")
    stats = {
        "n_in": int(points.size(0)),
        "n_out": int(m.num_points()),
        "triplets": int(m.i.numel()),
        "neighbors_mean": float(m.num_neighbors.float().mean().item()),
        "neighbors_max": int(m.num_neighbors.max().item()),
        "radius_scaler": float(radius_scaler),
        "neighbor_radius": float(neighbor_radius),
    }
    return y, m, rows, stats


def parity_check(
    model,
    feat: torch.Tensor,
    points: torch.Tensor,
    sizes: torch.Tensor,
    grid_size: float,
    receptive_field_scaler: float,
    mode: str,
) -> dict[str, float]:
    from layers import conv_with_stride

    m_ref = MetaData(
        points=points,
        sample_inds=_sample_inds(sizes),
        sample_sizes=sizes,
        grid_size=grid_size,
        receptive_field_scaler=receptive_field_scaler,
    )
    with torch.no_grad(), dispatch_mode(mode):
        ref, _ = conv_with_stride(
            model.conv1, feat, m_ref, 2, receptive_field_scaler=receptive_field_scaler)
        got, _, _, _ = split_stem(
            model.conv1, feat, points, sizes, grid_size, receptive_field_scaler,
            mode, "default")
    diff = (ref - got).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = ref.abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    ok = bool(torch.allclose(ref, got, rtol=1e-2, atol=3e-3))
    if not ok:
        raise RuntimeError(
            f"stem parity failed for mode={mode}: "
            f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    return {"max_abs": max_abs, "max_rel": max_rel}


def _median_iqr(vals: list[float]) -> tuple[float, float]:
    vals = sorted(vals)
    med = statistics.median(vals)
    if len(vals) >= 3:
        return med, vals[-1] - vals[0]
    return med, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="2,6")
    ap.add_argument("--widths", default="1.0,2.0")
    ap.add_argument("--modes", default="force_tig,auto,force_pt")
    ap.add_argument("--builder-backends", default="default")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--grid-size", type=float, default=0.02)
    ap.add_argument("--rfs", type=float, default=2.5)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not SCANNET:
        raise SystemExit("Set TIG_BENCH_SCANNET to a ScanNet val directory")

    raws = []
    for scene in SCENES:
        raws.append(load_coord(SCANNET, scene))

    torch.manual_seed(0)
    rows = []
    for batch in [int(x) for x in args.batches.split(",") if x]:
        points, sizes = batched(raws, args.grid_size, batch)
        feat = torch.randn(points.size(0), 1, device=device, dtype=torch.float16)
        for width in [float(x) for x in args.widths.split(",") if x]:
            model = resnet34(
                num_classes=40, in_channels=1, width_multiplier=width
            ).to(device).half().eval()
            for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
                parity = None
                if not args.skip_parity:
                    parity = parity_check(
                        model, feat, points, sizes, args.grid_size, args.rfs, mode)
                    print(f"parity B={batch} W={width:g} mode={mode} {parity}")
                backends = [x.strip() for x in args.builder_backends.split(",") if x.strip()]
                backend_ref = None
                for backend in backends:
                    for _ in range(args.warmup):
                        with torch.no_grad():
                            split_stem(
                                model.conv1, feat, points, sizes, args.grid_size,
                                args.rfs, mode, backend)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    acc: dict[str, list[float]] = defaultdict(list)
                    stat = None
                    y_last = None
                    for _ in range(args.iters):
                        with torch.no_grad():
                            y_last, _, timing, stat = split_stem(
                                model.conv1, feat, points, sizes, args.grid_size,
                                args.rfs, mode, backend)
                        for key, value in timing.items():
                            acc[key].append(float(value))
                    backend_parity = None
                    if backend_ref is None:
                        backend_ref = y_last
                    else:
                        diff = (backend_ref - y_last).abs()
                        backend_parity = {
                            "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                            "ok": bool(torch.allclose(
                                backend_ref, y_last, rtol=1e-2, atol=3e-3)),
                        }
                        if not backend_parity["ok"]:
                            raise RuntimeError(
                                f"backend parity failed: default-vs-{backend} "
                                f"max_abs={backend_parity['max_abs']:.3e}")
                    out = {
                        "batch": batch,
                        "width": width,
                        "mode": mode,
                        "builder_backend": backend,
                        "dtype": "fp16",
                        "parity": parity,
                        "backend_parity": backend_parity,
                        "stats": stat,
                        "timing_ms": {},
                        "memory_bytes": {
                            "peak_allocated": torch.cuda.max_memory_allocated(),
                            "peak_reserved": torch.cuda.max_memory_reserved(),
                        },
                    }
                    for key, vals in sorted(acc.items()):
                        med, iqr = _median_iqr(vals)
                        out["timing_ms"][key] = {"median": med, "iqr": iqr}
                    rows.append(out)
                    t = out["timing_ms"]
                    print(
                        f"B={batch} W={width:g} mode={mode} backend={backend} "
                        f"total={t['stem_total']['median']:.3f}ms "
                        f"down={t['downsample']['median']:.3f} "
                        f"build={t['build_triplets']['median']:.3f} "
                        f"conv={t['conv_compute']['median']:.3f} "
                        f"T={stat['triplets'] if stat else -1}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
