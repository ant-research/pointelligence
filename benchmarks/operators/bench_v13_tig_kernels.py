"""v1.3.0 operator-level bench: TIG kernels isolated, on REAL ScanNet.

Reports forward (mvmr), grad_input (mvmr), and grad_weight (vvor) per kernel,
separating the v1.3.0 no-atomic SEGMENT wgrad (auto) from the v1.2.0 atomic
Split-K wgrad (forced). The seg-vs-atomic ratio IS the v1.3.0-over-v1.2.0
backward delta. Kernel-isolated on purpose: the eager fwd+bwd wall-clock in
bench_tig_groups is dominated by per-op autograd/dispatch overhead (~0.57 ms at
c512) that vanishes under torch.compile, which spuriously flatters FSG.

Covers the k=3 stage ladder AND the high-K patchify stem (kernel_size=4 ->
K=64; kernel_size=5 -> K=125; the strided stem is where seg's win headlines).

Real data only (CLAUDE.md hard rule): scene0011_00, GridSample per stage.
Run: TIG_BENCH_SCANNET=/path/to/scannet_v2/val \
     CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/operators/bench_v13_tig_kernels.py
"""
from __future__ import annotations
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
from sparse_engines.tig import (TigIndex, tig_forward, tig_grad_input,
                                 tig_grad_weight, _seg_vvor_cfg, _vvor_cfg,
                                 _SEG_VVOR_MIN_C, _SEG_VVOR_MIN_M,
                                 _SEG_VVOR_MIN_PROGRAMS, _seg_gate_params,
                                 _SEG_VVOR_HIGHK_MIN_K)

device = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET", "")
SCENE = "scene0011_00"


def dedup_first(coord, grid_size):
    g = (coord / grid_size).long()
    uniq, inv = torch.unique(g, dim=0, return_inverse=True)
    first = torch.full((uniq.size(0),), coord.size(0), device=device, dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(coord.size(0), device=device), reduce="amin")
    return coord[first.sort().values]


def timed(fn, it=50, warm=10):
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


# (label, total channels C=M, GridSample grid, kernel_size -> K)
STAGES = [
    ("c64  k3",  64, 0.04, 3),
    ("c128 k3", 128, 0.08, 3),
    ("c256 k3", 256, 0.16, 3),
    ("c512 k3", 512, 0.32, 3),
    # high-K stem regime (the seg sweet spot): wide kernel at coarse grid
    ("c256 K125(k5)", 256, 0.24, 5),
    ("c512 K125(k5)", 512, 0.32, 5),
]


def main():
    raw = torch.from_numpy(np.load(f"{SCANNET}/{SCENE}/coord.npy")).float().to(device)
    rows = []
    print(f"scene={SCENE} raw_pts={raw.shape[0]}  (real data)")
    for label, C, grid, ks in STAGES:
        coord = dedup_first(raw, grid)
        N = coord.size(0)
        si = torch.zeros(N, device=device, dtype=torch.long)
        ss = torch.tensor([N], device=device)
        with torch.no_grad():
            i, j, k, _ = build_triplets(
                points=coord, sample_inds=si, sample_sizes=ss,
                neighbor_radius=grid * radius_scaler_for_kernel_size(ks),
                kernel_indexer=partial(voxelize_3d, kernel_size=ks),
                radius_scaler=radius_scaler_for_kernel_size(ks),
                return_num_neighbors=False)
        K = ks ** 3
        idx = TigIndex(i, j, k, N, num_kernel_offsets=K)
        T = int(idx.T)
        wshape = (K, 1, C, M := C)
        scfg = _seg_vvor_cfg(C, M, torch.float16)
        n_prog = K * triton.cdiv(C, scfg["BC"]) * triton.cdiv(M, scfg["BM"])
        _wide_c, _highk_min_c = _seg_gate_params()  # arch-aware (sm_89/sm_90)
        seg_gate = (C >= _SEG_VVOR_MIN_C and M >= _SEG_VVOR_MIN_M and T > 0
                    and n_prog >= _SEG_VVOR_MIN_PROGRAMS
                    and (C >= _wide_c or (C >= _highk_min_c and K >= _SEG_VVOR_HIGHK_MIN_K)))
        for dt in (torch.float16, torch.bfloat16):
            feat = torch.randn(N, C, device=device, dtype=dt)
            go = torch.randn(N, M, device=device, dtype=dt)
            w = (torch.randn(K, 1, C, M, device=device, dtype=dt) * ((K * C) ** -0.5))
            t_fwd = timed(lambda: tig_forward(w, feat, idx))
            t_gin = timed(lambda: tig_grad_input(w, go, idx))
            t_seg = timed(lambda: tig_grad_weight(feat, go, idx, wshape))
            t_atm = timed(lambda: tig_grad_weight(feat, go, idx, wshape,
                                                  wgrad_cfg=_vvor_cfg(C, M, dt)))
            r = dict(stage=label, N=N, T=T, C=C, K=K, dtype=str(dt).split(".")[-1],
                     seg_gate=seg_gate, fwd=t_fwd, grad_input=t_gin,
                     gwgt_seg=t_seg, gwgt_atomic=t_atm, seg_vs_atomic=t_atm / t_seg)
            rows.append(r)
            print(f"  {label:16s} {r['dtype']:5s} N={N:6d} T={T:7d} K={K:3d} | "
                  f"fwd {t_fwd:.3f}  gin {t_gin:.3f}  gwgt: seg {t_seg:.3f} "
                  f"atomic {t_atm:.3f}  seg-win {t_atm/t_seg:.2f}x  "
                  f"{'[SEG]' if seg_gate else '[atomic-only]'}")
    out = os.path.join("/tmp", "v13_tig_kernels.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
