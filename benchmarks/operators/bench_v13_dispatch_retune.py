"""v1.3.0 dispatch re-tune on REAL data at the production TRAIN batch size.

The seg-vs-atomic wgrad gate was originally set from a synthetic/forced microbench
and regressed c128 low-K wgrad 2.6x on real data (single-scene). This sweep
re-derives the crossover the RIGHT way: real ScanNet scenes BATCHED at the train
batch size (default B=6 on local Ada; pass --batch 12 for the H200 regime), across
the stage ladder x kernel-size(K) x dtype. Batching multiplies N and T ~Bx, which
lengthens segments and shifts the seg-vs-atomic crossover vs batch-1.

Emits, per (stage, K, dtype): batched N/T, seg vs atomic wgrad time, the win
ratio, and the route the CURRENT gate picks vs the route the data says is optimal
-> the recommended (C, K) gate. Real data only (CLAUDE.md hard rule).

Run: TIG_BENCH_SCANNET=/path/to/scannet_v2/val CUDA_VISIBLE_DEVICES=1 \
     .venv/bin/python benchmarks/operators/bench_v13_dispatch_retune.py --batch 6
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import torch
import triton

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from layers import build_triplets, radius_scaler_for_kernel_size, voxelize_3d
from sparse_engines.tig import (TigIndex, tig_grad_weight, _seg_vvor_cfg, _vvor_cfg,
                                 _SEG_VVOR_MIN_C, _SEG_VVOR_MIN_M,
                                 _SEG_VVOR_MIN_PROGRAMS, _seg_gate_params,
                                 _SEG_VVOR_HIGHK_MIN_K)

device = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET", "")
# distinct rooms (not multi-view dups of one room)
SCENES = ["scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00",
          "scene0030_00", "scene0046_00", "scene0050_00", "scene0064_00",
          "scene0086_00", "scene0095_00", "scene0100_00", "scene0131_00"]


def dedup_first(coord, grid_size):
    g = (coord / grid_size).long()
    uniq, inv = torch.unique(g, dim=0, return_inverse=True)
    first = torch.full((uniq.size(0),), coord.size(0), device=device, dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(coord.size(0), device=device), reduce="amin")
    return coord[first.sort().values]


def batched_cloud(scene_coords, grid, B):
    """Concat B dedup'd scenes into one batched cloud with per-scene segments."""
    coords, sizes = [], []
    for c in scene_coords[:B]:
        d = dedup_first(c, grid)
        coords.append(d)
        sizes.append(d.size(0))
    coord = torch.cat(coords, 0)
    sample_sizes = torch.tensor(sizes, device=device)
    sample_inds = torch.repeat_interleave(
        torch.arange(B, device=device), sample_sizes)
    return coord, sample_inds, sample_sizes


def timed(fn, it=30, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


STAGES = [  # (label, C=M, grid, kernel_size)
    ("c64  k3",  64, 0.04, 3), ("c128 k3", 128, 0.08, 3),
    ("c256 k3", 256, 0.16, 3), ("c512 k3", 512, 0.32, 3),
    ("c128 k5", 128, 0.14, 5), ("c256 k5", 256, 0.24, 5),
    ("c512 k5", 512, 0.32, 5),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()
    B = args.batch
    raws = [torch.from_numpy(np.load(f"{SCANNET}/{s}/coord.npy")).float().to(device)
            for s in SCENES[:B]]
    print(f"dispatch re-tune | REAL ScanNet | B={B} scenes | sm_89")
    rows = []
    for label, C, grid, ks in STAGES:
        coord, si, ss = batched_cloud(raws, grid, B)
        with torch.no_grad():
            i, j, k, _ = build_triplets(
                points=coord, sample_inds=si, sample_sizes=ss,
                neighbor_radius=grid * radius_scaler_for_kernel_size(ks),
                kernel_indexer=partial(voxelize_3d, kernel_size=ks),
                radius_scaler=radius_scaler_for_kernel_size(ks),
                return_num_neighbors=False)
        K = ks ** 3
        N = coord.size(0)
        idx = TigIndex(i, j, k, N, num_kernel_offsets=K)
        T = int(idx.T)
        wshape = (K, 1, C, C)
        _wide_c, _highk_min_c = _seg_gate_params()  # arch-aware (sm_89/sm_90)
        cur_gate = (C >= _SEG_VVOR_MIN_C and C >= _SEG_VVOR_MIN_M
                    and (C >= _wide_c or (C >= _highk_min_c and K >= _SEG_VVOR_HIGHK_MIN_K)))
        import sparse_engines.tig as _tig
        for dt in (torch.float16, torch.bfloat16):
            feat = torch.randn(N, C, device=device, dtype=dt)
            go = torch.randn(N, C, device=device, dtype=dt)
            # FORCE seg (zero the gate) to measure its TRUE time even where the
            # current gate routes atomic; restore the gate after. The arch-aware
            # (wide_c, highk_min_c) live in _SEG_GATE_CC_CACHE (computed per
            # device), so forcing seg means caching (0, 0) for this device in
            # addition to zeroing the module-level floors.
            _dev = torch.cuda.current_device()
            _save = (_tig._SEG_VVOR_MIN_C, _tig._SEG_VVOR_MIN_M,
                     _tig._SEG_VVOR_HIGHK_MIN_K, _tig._SEG_VVOR_MIN_PROGRAMS)
            _save_cache = _tig._SEG_GATE_CC_CACHE.get(_dev)
            _tig._SEG_VVOR_MIN_C = _tig._SEG_VVOR_MIN_M = 0
            _tig._SEG_VVOR_HIGHK_MIN_K = _tig._SEG_VVOR_MIN_PROGRAMS = 0
            _tig._SEG_GATE_CC_CACHE[_dev] = (0, 0)
            try:
                t_seg = timed(lambda: tig_grad_weight(feat, go, idx, wshape))
            finally:
                (_tig._SEG_VVOR_MIN_C, _tig._SEG_VVOR_MIN_M,
                 _tig._SEG_VVOR_HIGHK_MIN_K, _tig._SEG_VVOR_MIN_PROGRAMS) = _save
                if _save_cache is None:
                    _tig._SEG_GATE_CC_CACHE.pop(_dev, None)
                else:
                    _tig._SEG_GATE_CC_CACHE[_dev] = _save_cache
            t_atm = timed(lambda: tig_grad_weight(feat, go, idx, wshape,
                                                  wgrad_cfg=_vvor_cfg(C, C, dt)))
            win = t_atm / t_seg
            optimal = "seg" if t_seg < t_atm else "atomic"
            picked = "seg" if cur_gate else "atomic"
            flag = "" if optimal == picked else "  <-- MISROUTE"
            rows.append(dict(stage=label, B=B, N=N, T=T, C=C, K=K,
                             dtype=str(dt).split(".")[-1], seg=t_seg, atomic=t_atm,
                             win=win, optimal=optimal, picked=picked, segT_over_K=T // K))
            print(f"  {label:9s} {str(dt).split('.')[-1]:4s} N={N:7d} T={T:8d} K={K:3d} "
                  f"segT/K={T//K:6d} | auto {t_seg:.3f} atomic {t_atm:.3f} win {win:.2f}x "
                  f"opt={optimal:6s} gate={picked:6s}{flag}")
    out = f"/tmp/v13_dispatch_retune_b{B}.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
