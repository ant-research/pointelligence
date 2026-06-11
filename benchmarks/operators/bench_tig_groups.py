"""TIG groups>1 profile + PT-baseline cells (v1.2.0).

Layer-level PointConv3d on a real ScanNet scene, fwd and f+b, fp16/bf16:
shapes (C_total, grid) x G in {1,2,4,8}; engines force_pt (the v1.0.0
baseline and the previous G>1 incumbent), force_fsg (G==1 only -
dispatch refuses G>1), force_tig. Reproduces the layer-level tables in
docs/tig.md: where G>1 should route, and the TIG-vs-PT ratios.

Set TIG_BENCH_SCANNET to a Pointcept-preprocessed ScanNet val directory
(containing <scene>/coord.npy); release benches used scene0011_00.

Run: TIG_BENCH_SCANNET=/path/to/scannet_v2/val CUDA_VISIBLE_DEVICES=3 \
    .venv/bin/python benchmarks/operators/bench_tig_groups.py
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from layers import (PointConv3d, build_triplets,
                    radius_scaler_for_kernel_size, voxelize_3d)
from sparse_engines._dispatch_override import dispatch_mode

device = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET")
SCENE = "scene0011_00"
SHAPES = [("c64", 64, 0.04), ("c256", 256, 0.16), ("c512", 512, 0.32)]
GROUPS = [1, 2, 4, 8]


def dedup_first(coord, grid_size):
    g = (coord / grid_size).long()
    g = g - g.min(0).values
    uniq, inv = torch.unique(g, dim=0, return_inverse=True)
    first = torch.full((uniq.size(0),), coord.size(0), device=device,
                       dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(coord.size(0), device=device),
                          reduce="amin")
    return coord[first.sort().values]


def time_med(fn, n_warmup=5, n_iters=12, runs=3):
    meds = []
    for _ in range(runs):
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
        meds.append(ts[len(ts) // 2])
    meds.sort()
    return meds[len(meds) // 2]


def main():
    if SCANNET is None:
        sys.exit("Set TIG_BENCH_SCANNET to a Pointcept-preprocessed ScanNet "
                 "val directory (containing <scene>/coord.npy).")
    raw = torch.from_numpy(
        np.load(f"{SCANNET}/{SCENE}/coord.npy")).float().to(device)
    rows = []
    for label, C, g in SHAPES:
        coord = dedup_first(raw, g)
        N = coord.size(0)
        si = torch.zeros(N, device=device, dtype=torch.long)
        ss = torch.tensor([N], device=device)
        with torch.no_grad():
            i_t, j_t, k_t, _ = build_triplets(
                points=coord, sample_inds=si, sample_sizes=ss,
                neighbor_radius=g * radius_scaler_for_kernel_size(3),
                kernel_indexer=partial(voxelize_3d, kernel_size=3),
                radius_scaler=radius_scaler_for_kernel_size(3),
                return_num_neighbors=False)
        T = i_t.numel()
        torch.manual_seed(0)
        feat32 = torch.randn(N, C, device=device)
        target32 = torch.randn(N, C, device=device)
        print(f"\n== {label} N={N} T={T} C={C} grid={g} ==")
        for G in GROUPS:
            for dt_label, dtype in [("fp16", torch.float16),
                                    ("bf16", torch.bfloat16)]:
                feat = feat32.to(dtype)
                target = target32.to(dtype)
                pc = PointConv3d(C, C, kernel_size=3, groups=G, bias=False
                                 ).to(device).to(dtype)
                base = {}
                for mode in ("force_pt", "force_fsg", "force_tig"):
                    def fwd(mode=mode):
                        with dispatch_mode(mode), torch.no_grad():
                            pc(feat, i_t, j_t, k_t, N)

                    def fb(mode=mode):
                        with dispatch_mode(mode):
                            for p in pc.parameters():
                                p.grad = None
                            f = feat.detach().clone().requires_grad_(True)
                            out = pc(f, i_t, j_t, k_t, N)
                            ((out - target).pow(2).sum()).backward()
                    try:
                        fwd(); fb()
                        f_ms, b_ms = time_med(fwd), time_med(fb)
                    except Exception as exc:
                        print(f"  G={G} {dt_label} {mode:10s} ERROR "
                              f"{type(exc).__name__}")
                        continue
                    if mode == "force_pt":
                        base = {"f": f_ms, "b": b_ms}
                    r = {"shape": label, "N": N, "T": T, "C": C, "G": G,
                         "dtype": dt_label, "mode": mode,
                         "fwd_ms": round(f_ms, 4), "fb_ms": round(b_ms, 4),
                         "fwd_vs_pt": round(f_ms / base["f"], 3) if base else None,
                         "fb_vs_pt": round(b_ms / base["b"], 3) if base else None}
                    rows.append(r)
                    print(f"  G={G} {dt_label} {mode:10s} fwd {f_ms:8.3f} "
                          f"f+b {b_ms:8.3f}"
                          + (f"  vs PT {r['fwd_vs_pt']}/{r['fb_vs_pt']}x"
                             if mode != "force_pt" and base else ""))
                del pc
                torch.cuda.empty_cache()
    with open("/tmp/tig_groups_bench.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\nwrote /tmp/tig_groups_bench.json")


if __name__ == "__main__":
    main()
