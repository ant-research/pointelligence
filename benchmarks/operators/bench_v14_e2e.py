"""v1.4.0 end-to-end release benchmark: a REAL-scene ResNet backbone forward and
forward+backward, version-to-version, at small and large batch.

Arms (same backbone, only the conv operator changes via dispatch_mode):
  v1.0  per-triplet     force_pt
  v1.3  TIG (seg bwd)   force_tig
  v1.4  auto best-known route (eligible k=3/G=1/M=C/fp16 convs route to
        fused gather-sum where the release table wins; C512 train and all
        ineligible convs fall through to TIG)

Eager (the fused operator is a plain autograd.Function; torch.compile of the
fused path is a separate lane). Real ScanNet scenes, voxel-dedup batched. The
rulebook / TIG index is naturally reused across each stage's sibling convs, so
this is the deployment-faithful view of the operator speedup.

Run: TIG_BENCH_SCANNET=/path/to/scannet_v2/val \
     .venv/bin/python benchmarks/operators/bench_v14_e2e.py --batches 2,6 --depth 34
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from sparse_engines._dispatch_override import dispatch_mode
from models.resnet import resnet18, resnet34, resnet50

device = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET", "")
SCENES = [
    "scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00",
    "scene0030_00", "scene0046_00", "scene0050_00", "scene0064_00",
    "scene0086_00", "scene0095_00", "scene0100_00", "scene0131_00",
    "scene0144_00", "scene0153_00", "scene0164_00", "scene0169_00",
    "scene0187_00", "scene0193_00", "scene0207_00", "scene0217_00",
    "scene0221_00", "scene0011_01", "scene0019_01", "scene0249_00"]
_FACTORY = {18: resnet18, 34: resnet34, 50: resnet50}
BASE_ARMS = [("v1.0", "force_pt"), ("v1.3", "force_tig"), ("v1.4", "auto")]


def dedup_first(coord, gs):
    g = (coord / gs).long()
    uniq, inv = torch.unique(g, dim=0, return_inverse=True)
    first = torch.full((uniq.size(0),), coord.size(0), device=device, dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(coord.size(0), device=device), reduce="amin")
    return coord[first.sort().values]


def batched(raws, gs, B):
    cs, sizes = [], []
    for c in raws[:B]:
        d = dedup_first(c, gs); cs.append(d); sizes.append(d.size(0))
    return torch.cat(cs, 0), torch.tensor(sizes, device=device)


def timed(fn, it=10, warm=4):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="2,6", help="comma-sep batch sizes")
    ap.add_argument("--depth", type=int, default=34, choices=[18, 34, 50])
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--rfs", type=float, default=2.5)
    ap.add_argument("--gs", type=float, default=0.02)
    ap.add_argument("--out", default="/tmp/v14_e2e.json")
    ap.add_argument("--include-force-fused", action="store_true",
                    help="also time the diagnostic force_fused_gather_sum route")
    args = ap.parse_args()
    arms = list(BASE_ARMS)
    if args.include_force_fused:
        arms.append(("v1.4-force", "force_fused_gather_sum"))
    batches = [int(b) for b in args.batches.split(",")]
    raws = [torch.from_numpy(np.load(f"{SCANNET}/{s}/coord.npy")).float().to(device)
            for s in SCENES[:max(batches)]]
    rows = []
    for B in batches:
        points, sizes = batched(raws, args.gs, B)
        N = points.size(0)
        torch.manual_seed(0)
        feat = torch.randn(N, 1, device=device, dtype=torch.float16)
        model = _FACTORY[args.depth](in_channels=1, width_multiplier=args.width).to(device).to(torch.float16)
        target = torch.randint(0, 1000, (B,), device=device)
        print(f"\n=== resnet{args.depth}x{args.width}  B={B}  N={N}  (real ScanNet) ===")
        for regime in ("val", "train"):
            line = f"  {regime:5s} | "
            base = None
            for aname, mode in arms:
                def run():
                    if regime == "val":
                        with torch.no_grad(), dispatch_mode(mode):
                            model(feat, points, sizes, args.gs, receptive_field_scaler=args.rfs)
                    else:
                        for p in model.parameters():
                            p.grad = None
                        x = feat.detach().clone().requires_grad_(True)
                        with dispatch_mode(mode):
                            out = model(x, points, sizes, args.gs, receptive_field_scaler=args.rfs)
                            torch.nn.functional.cross_entropy(out, target).backward()
                try:
                    ms = timed(run)
                except Exception as e:
                    ms = float("nan")
                    print(f"    {aname} {regime}: ERR {type(e).__name__}: {e}")
                rows.append(dict(depth=args.depth, width=args.width, B=B, N=N,
                                 regime=regime, arm=aname, ms=ms))
                if base is None:
                    base = ms
                line += f"{aname} {ms:.1f}({base/ms:.2f}x) "
            print(line)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
