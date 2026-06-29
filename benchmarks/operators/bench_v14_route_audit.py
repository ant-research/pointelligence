"""v1.4.0 route audit on REAL ScanNet scenes.

This harness answers one narrow release question: does the shipped ``auto`` route
match the best-known v1.4 composition, instead of accidentally measuring a stale
TIG-only default or an over-forced fused route?

It times the same PointConv3d layer under:
  - force_tig: v1.3-style fallback/reference route
  - force_fused_gather_sum: diagnostic route, fuses every eligible k=3/fp16/Cin=Cout/G1
  - auto: shipped v1.4 route (C<=256 train fused, C512 train TIG, no-grad fused)

Real data only. Synthetic inputs are intentionally not used for timing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from layers import (PointConv3d, build_triplets, radius_scaler_for_kernel_size,
                    voxelize_3d)
from layers.conv import _auto_fused_gather_sum_width
from sparse_engines._dispatch_override import dispatch_mode
from sparse_engines import fused_point_conv as fsc

DEVICE = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET", "")
SCENES = [
    "scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00",
    "scene0030_00", "scene0046_00", "scene0050_00", "scene0064_00",
    "scene0086_00", "scene0095_00", "scene0100_00", "scene0131_00",
    "scene0144_00", "scene0153_00", "scene0164_00", "scene0169_00",
    "scene0187_00", "scene0193_00", "scene0207_00", "scene0217_00",
    "scene0221_00", "scene0011_01", "scene0019_01", "scene0249_00",
]
STAGES = [("c64", 64, 0.04, 3), ("c128", 128, 0.08, 3),
          ("c256", 256, 0.16, 3), ("c512", 512, 0.32, 3)]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
MODES = [("tig", "force_tig"), ("fused_force", "force_fused_gather_sum"),
         ("auto", "auto")]


def dedup_first(coord, gs):
    g = (coord / gs).long()
    uniq, inv = torch.unique(g, dim=0, return_inverse=True)
    first = torch.full((uniq.size(0),), coord.size(0), device=DEVICE, dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(coord.size(0), device=DEVICE), reduce="amin")
    return coord[first.sort().values]


def batched(raws, grid, B):
    coords, sizes = [], []
    for c in raws[:B]:
        d = dedup_first(c, grid)
        coords.append(d)
        sizes.append(d.size(0))
    coord = torch.cat(coords, 0)
    sample_sizes = torch.tensor(sizes, device=DEVICE)
    sample_inds = torch.repeat_interleave(torch.arange(B, device=DEVICE), sample_sizes)
    return coord, sample_inds, sample_sizes


def timed(fn, it, warm):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(it):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / it


def expected_auto_route(C, dtype_name, regime):
    if dtype_name != "fp16":
        return "tig"
    return "fused" if _auto_fused_gather_sum_width(C, grad_enabled=(regime == "train")) else "tig"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-batch", type=int, default=6)
    ap.add_argument("--val-batch", type=int, default=12)
    ap.add_argument("--dtypes", default="fp16", help="comma-separated: fp16,bf16")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--out", default="/tmp/v14_route_audit.json")
    args = ap.parse_args()

    dtype_names = [d for d in args.dtypes.split(",") if d]
    raws = [torch.from_numpy(np.load(f"{SCANNET}/{s}/coord.npy")).float().to(DEVICE)
            for s in SCENES[:max(args.train_batch, args.val_batch)]]
    rows = []

    print("route audit: auto is shipped v1.4; force_fused_gather_sum is diagnostic")
    print(f"fused knobs: MAXLVL policy dynamic, fold by width; arch={torch.cuda.get_device_name(0)}")
    for regime, B in (("train", args.train_batch), ("val", args.val_batch)):
        print(f"\n=== {regime.upper()} B={B} ===")
        for label, C, grid, ks in STAGES:
            coord, si, ss = batched(raws, grid, B)
            with torch.no_grad():
                i, j, k, _ = build_triplets(
                    points=coord, sample_inds=si, sample_sizes=ss,
                    neighbor_radius=grid * radius_scaler_for_kernel_size(ks),
                    kernel_indexer=partial(voxelize_3d, kernel_size=ks),
                    radius_scaler=radius_scaler_for_kernel_size(ks),
                    return_num_neighbors=False)
            N = coord.size(0)
            for dtype_name in dtype_names:
                dt = DTYPES[dtype_name]
                pc = PointConv3d(C, C, kernel_size=ks, bias=False).to(DEVICE).to(dt)
                feat0 = (torch.randn(N, C, device=DEVICE) * 0.1).to(dt)
                route = expected_auto_route(C, dtype_name, regime)
                print(f"  {label:4s} {dtype_name:4s} N={N:7d} T={int(i.numel()):8d} "
                      f"auto->{route:5s} maxlvl={fsc._maxlvl_for_width(C)} "
                      f"fold={int(fsc._fold_for_width(C))} b1={fsc._seg_b1_for_width(C, 0 if dtype_name == 'fp16' else 1)} | ",
                      end="", flush=True)
                base = None
                for mode_name, mode in MODES:
                    def run_train():
                        pc.zero_grad(set_to_none=True)
                        x = feat0.detach().clone().requires_grad_(True)
                        with dispatch_mode(mode):
                            out = pc(x, i, j, k, N)
                            out.float().sum().backward()
                    def run_val():
                        with torch.no_grad(), dispatch_mode(mode):
                            pc(feat0, i, j, k, N)
                    fn = run_train if regime == "train" else run_val
                    try:
                        ms = timed(fn, args.iters, args.warmup)
                    except Exception as exc:
                        ms = float("nan")
                        print(f"{mode_name}=ERR({type(exc).__name__}) ", end="")
                    if base is None:
                        base = ms
                    rows.append(dict(regime=regime, B=B, stage=label, C=C, N=N,
                                     T=int(i.numel()), dtype=dtype_name,
                                     expected_auto=route, mode=mode_name, ms=ms))
                    print(f"{mode_name}={ms:.3f}ms({ms/base:.2f}x) ", end="")
                print()
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
