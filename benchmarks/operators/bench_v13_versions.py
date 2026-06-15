"""v1.3.0 EXTENDED release benchmark: all four generations on REAL ScanNet at the
production batch sizes, train (fwd+bwd) and val (fwd-only) regimes.

Generations (comparison-fair: identical shapes/batches across all four):
  v1.0.0  PT   per-triplet            dispatch_mode("force_pt")
  v1.1.0  FSG  full-segment grouped   dispatch_mode("force_fsg")
  v1.2.0  TIG  (atomic Split-K bwd)   force_tig + seg gate disabled
  v1.3.0  TIG  (no-atomic seg bwd)    force_tig + seg gate on  <-- this release extends here

Batch sizes (project_release_benchmark_batch_spec): train B=6 local / 12 H200;
val B=12 local / 24 H200. Real data only. The val regime is fwd under no_grad; the
train regime is fwd+bwd. Layer-level (PointConv3d) to match the v1.1/v1.2 release
methodology; the kernel-isolated true seg win is in bench_v13_tig_kernels.py.

Run: TIG_BENCH_SCANNET=/path/to/scannet_v2/val CUDA_VISIBLE_DEVICES=1 \
     .venv/bin/python benchmarks/operators/bench_v13_versions.py --train-batch 6 --val-batch 12
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

import sparse_engines.tig as _tig
from layers import (PointConv3d, build_triplets, radius_scaler_for_kernel_size,
                    voxelize_3d)
from sparse_engines._dispatch_override import dispatch_mode

device = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET", "")
SCENES = ["scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00",
          "scene0030_00", "scene0046_00", "scene0050_00", "scene0064_00",
          "scene0086_00", "scene0095_00", "scene0100_00", "scene0131_00",
          "scene0144_00", "scene0153_00", "scene0164_00", "scene0169_00",
          "scene0187_00", "scene0193_00", "scene0207_00", "scene0217_00",
          "scene0221_00", "scene0233_00", "scene0244_00", "scene0249_00"]

STAGES = [("c64 k3", 64, 0.04, 3), ("c128 k3", 128, 0.08, 3),
          ("c256 k3", 256, 0.16, 3), ("c512 k3", 512, 0.32, 3)]


def dedup_first(coord, gs):
    g = (coord / gs).long()
    uniq, inv = torch.unique(g, dim=0, return_inverse=True)
    first = torch.full((uniq.size(0),), coord.size(0), device=device, dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(coord.size(0), device=device), reduce="amin")
    return coord[first.sort().values]


def batched(raws, grid, B):
    cs, sizes = [], []
    for c in raws[:B]:
        d = dedup_first(c, grid); cs.append(d); sizes.append(d.size(0))
    coord = torch.cat(cs, 0)
    ss = torch.tensor(sizes, device=device)
    si = torch.repeat_interleave(torch.arange(B, device=device), ss)
    return coord, si, ss


def timed(fn, it=30, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


ARMS = [("v1.0 PT", "force_pt", False), ("v1.1 FSG", "force_fsg", False),
        ("v1.2 TIG-atomic", "force_tig", True), ("v1.3 TIG-seg", "force_tig", False)]


def seg_disabled():
    """Context: force the v1.2.0 atomic backward by raising the seg gate's
    channel floor above any real C. Arch-aware-safe: _SEG_VVOR_WIDE_C was
    removed by the arch-aware gate refactor (it is now computed per-device by
    _seg_gate_params()); the gate's outer `C >= _SEG_VVOR_MIN_C` guard alone
    routes everything to atomic when raised."""
    class _C:
        def __enter__(self):
            self.s = _tig._SEG_VVOR_MIN_C
            _tig._SEG_VVOR_MIN_C = 10 ** 9
        def __exit__(self, *a):
            _tig._SEG_VVOR_MIN_C = self.s
    return _C()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-batch", type=int, default=6)
    ap.add_argument("--val-batch", type=int, default=12)
    args = ap.parse_args()
    raws = [torch.from_numpy(np.load(f"{SCANNET}/{s}/coord.npy")).float().to(device)
            for s in SCENES[:max(args.train_batch, args.val_batch)]]
    rows = []
    for regime, B in (("train", args.train_batch), ("val", args.val_batch)):
        print(f"\n===== {regime.upper()} regime  B={B}  (real ScanNet, sm_89) =====")
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
            line = f"  {label:9s} N={N:7d} T={int(i.numel()):8d} | "
            base = None
            for dt in (torch.float16,):
                pc = PointConv3d(C, C, kernel_size=ks, bias=True).to(device).to(dt)
                feat0 = torch.randn(N, C, device=device, dtype=dt)
                for aname, mode, force_atomic in ARMS:
                    def run_train():
                        f = feat0.clone().requires_grad_(True)
                        out = pc(f, i, j, k, N)
                        out.sum().backward()
                        pc.zero_grad(set_to_none=True)
                    def run_val():
                        with torch.no_grad():
                            pc(feat0, i, j, k, N)
                    fn = run_train if regime == "train" else run_val
                    ctx_seg = seg_disabled() if force_atomic else _nullctx()
                    with dispatch_mode(mode), ctx_seg:
                        try:
                            ms = timed(fn)
                        except Exception as e:
                            ms = float("nan")
                            print(f"    {aname} {label}: ERR {type(e).__name__}")
                    rows.append(dict(regime=regime, B=B, stage=label, N=N, C=C, K=ks**3,
                                     dtype="fp16", arm=aname, ms=ms))
                    if base is None:
                        base = ms
                    line += f"{aname.split()[0]} {ms:.2f}({base/ms:.2f}x) "
            print(line)
    out = f"/tmp/v13_versions_t{args.train_batch}_v{args.val_batch}.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
