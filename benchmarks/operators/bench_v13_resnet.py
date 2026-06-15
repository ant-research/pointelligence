"""v1.3.0 EXTENDED release benchmark — WHOLE-ResNet end-to-end, all four generations
on REAL ScanNet at the production batch sizes.

This is the backbone-level companion to bench_v13_versions.py (which is per-conv-stage).
It mirrors the v1.1.0 / v1.2.0 release methodology (PointConv3d-ResNet18/34/50 ×
channel-scales × dtypes, forward+backward) but at the v1.3.0 production-aligned config:
REAL ScanNet scenes batched to the production batch (train B=6 local / 12 H200), and ALL
FOUR generations re-run in that same config so v1.0.0 is measured here too — not compared
against the old single-scene numbers.

Generations (comparison-fair: identical shapes / batches / scenes across all four):
  v1.0.0  PT   per-triplet            dispatch_mode("force_pt")
  v1.1.0  FSG  full-segment grouped   dispatch_mode("force_fsg")
  v1.2.0  TIG  (atomic Split-K bwd)   force_tig + seg gate disabled
  v1.3.0  TIG  (no-atomic seg bwd)    force_tig + seg gate on

Real geometry only (never synthetic — see the fan-in discussion in docs/segment_vvor.md);
per-point input feature is a reproducible 1-channel projection, exactly as the v1.1/v1.2
ResNet bench. The model widens internally per width_multiplier.

Run: TIG_BENCH_SCANNET=/path/to/scannet_v2/val CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
     .venv/bin/python benchmarks/operators/bench_v13_resnet.py --batch 6   # Ada
     ( --batch 12 for the H200 production batch )
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch

import sparse_engines.tig as _tig
from models.resnet import resnet18, resnet34, resnet50
from sparse_engines._dispatch_override import dispatch_mode

device = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET", "")
# 24 distinct real ScanNet v2 val scenes (same list as bench_v13_versions.py) — enough
# for the H200 val batch (24); train uses the first --batch of them.
SCENES = ["scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00",
          "scene0030_00", "scene0046_00", "scene0050_00", "scene0064_00",
          "scene0086_00", "scene0095_00", "scene0100_00", "scene0131_00",
          "scene0144_00", "scene0153_00", "scene0164_00", "scene0169_00",
          "scene0187_00", "scene0193_00", "scene0207_00", "scene0217_00",
          "scene0221_00", "scene0063_00", "scene0077_00", "scene0249_00"]

# Input voxel size — the Pointcept GridSample preprocessing grid the model's conv1
# sees; the ResNet strides down internally from here.
INPUT_GRID = 0.02
# Production unet_pointcnnpp receptive-field scaler (real scenes have sparse-edge
# regions where rfs=1.0 yields no neighbors at stride-2 query points).
RFS = 2.5

DEPTHS = [("resnet18", resnet18, "BasicBlock × [2,2,2,2]"),
          ("resnet34", resnet34, "BasicBlock × [3,4,6,3]"),
          ("resnet50", resnet50, "Bottleneck × [3,4,6,3]")]
SCALES = [0.25, 0.5, 1.0, 2.0]
DTYPES = [("fp16", torch.float16), ("fp32", torch.float32), ("bf16", torch.bfloat16)]
ARMS = [("v1.0", "force_pt", False), ("v1.1", "force_fsg", False),
        ("v1.2", "force_tig", True), ("v1.3", "force_tig", False)]


def dedup_first(coord, gs):
    """One representative point per voxel at grid size gs (GridSample-equivalent)."""
    g = (coord / gs).long()
    uniq, inv = torch.unique(g, dim=0, return_inverse=True)
    first = torch.full((uniq.size(0),), coord.size(0), device=device, dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(coord.size(0), device=device), reduce="amin")
    return coord[first.sort().values]


def batched(raws, B):
    """Batch the first B scenes (deduped at the input grid) into one (coord, sample_sizes)."""
    cs, sizes = [], []
    for c in raws[:B]:
        d = dedup_first(c, INPUT_GRID); cs.append(d); sizes.append(d.size(0))
    coord = torch.cat(cs, 0)
    ss = torch.tensor(sizes, device=device, dtype=torch.long)
    return coord, ss


def timed(fn, it=8, warm=3):
    """Trim-median wall time (ms). ResNet fwd+bwd is 50-500 ms → few iters suffice."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); s.record()
        fn()
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def seg_disabled():
    """Force the v1.2.0 atomic backward by raising the seg MIN_C floor above any shape."""
    class _C:
        def __enter__(self):
            self.s = _tig._SEG_VVOR_MIN_C
            _tig._SEG_VVOR_MIN_C = 10 ** 9
        def __exit__(self, *a):
            _tig._SEG_VVOR_MIN_C = self.s
    return _C()


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=6, help="scenes per batch (Ada 6 / H200 12)")
    ap.add_argument("--dtype", choices=["fp16", "fp32", "bf16", "all"], default="fp16")
    ap.add_argument("--depths", default="18,34,50")
    ap.add_argument("--scales", default="0.25,0.5,1.0,2.0")
    args = ap.parse_args()

    dtypes = DTYPES if args.dtype == "all" else [d for d in DTYPES if d[0] == args.dtype]
    sel_d = set(args.depths.split(","))
    depths = [d for d in DEPTHS if d[0].replace("resnet", "") in sel_d]
    sel_s = [float(s) for s in args.scales.split(",")]
    scales = [s for s in SCALES if s in sel_s]

    raws = [torch.from_numpy(np.load(f"{SCANNET}/{s}/coord.npy")).float().to(device)
            for s in SCENES[:args.batch]]
    coord, ss = batched(raws, args.batch)
    N = coord.size(0)
    print(f"WHOLE-ResNet v1.0->v1.3 | {torch.cuda.get_device_name(0)} | "
          f"REAL ScanNet B={args.batch} | N={N:,} (input grid {INPUT_GRID} m) | "
          f"train fwd+bwd | speedup vs v1.0")

    rows = []
    for dtname, dt in dtypes:
        for dname, factory, ddesc in depths:
            print(f"\n===== {dtname} × {dname}  ({ddesc}) =====")
            for scale in scales:
                feat0 = torch.randn(N, 1, device=device, dtype=dt)
                target = torch.randint(0, 1000, (args.batch,), device=device)
                line = f"  {scale:.2f}× | "
                base = None
                for aname, mode, force_atomic in ARMS:
                    try:
                        model = factory(in_channels=1, width_multiplier=scale).to(device).to(dt)
                    except Exception as e:
                        print(f"    {aname} build ERR {type(e).__name__}: {e}"); continue

                    def run_train():
                        for p in model.parameters():
                            p.grad = None
                        x = feat0.clone().requires_grad_(True)
                        out = model(x, coord, ss, INPUT_GRID, receptive_field_scaler=RFS)
                        torch.nn.functional.cross_entropy(out, target).backward()

                    ctx_seg = seg_disabled() if force_atomic else _nullctx()
                    torch.cuda.reset_peak_memory_stats(device)
                    try:
                        with dispatch_mode(mode), ctx_seg:
                            ms = timed(run_train)
                        vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        ms, vram = float("nan"), float("nan")
                        msg = "OOM" if "out of memory" in str(e).lower() else type(e).__name__
                        print(f"    {aname} {dname} {scale:.2f}× {dtname}: {msg}")
                        torch.cuda.empty_cache()
                    finally:
                        del model
                        torch.cuda.empty_cache()
                    rows.append(dict(card=torch.cuda.get_device_name(0), B=args.batch,
                                     depth=dname, scale=scale, dtype=dtname, arm=aname,
                                     ms=round(ms, 3), vram_mb=round(vram, 0)))
                    if base is None and ms == ms:  # first finite = v1.0 baseline
                        base = ms
                    spd = (base / ms) if (base and ms == ms) else float("nan")
                    line += f"{aname} {ms:.1f}({spd:.2f}x) "
                print(line)

    out = f"/tmp/v13_resnet_b{args.batch}.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
