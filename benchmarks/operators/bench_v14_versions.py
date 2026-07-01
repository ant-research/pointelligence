"""v1.4.0 release benchmark: version-to-version speed of the SAME point-native
submanifold conv across all five generations on REAL ScanNet, at the production
batch sizes, in both the train (fwd+bwd) and val (fwd-only) regimes.

Engine arms (controlled diagnostic: identical real scenes, shapes, batches,
triplets, weights, and dtype; only the operator schedule changes):
  PT          per-triplet                    dispatch_mode("force_pt")
  FSG         full-segment grouped           dispatch_mode("force_fsg")
  TIG-atomic  TIG with atomic Split-K bwd     force_tig + seg gate disabled
  TIG-seg     TIG with segment VVOR bwd       force_tig + seg gate on
  AUTO        best-known v1.4 route           dispatch_mode("auto")

These arms are useful for isolating scheduler generations inside one codebase.
They are not, by themselves, proof that an old release tag shipped an equivalent
fp16/AMP path. Public release-to-release claims should use only comparable
release-supported paths; for v1.4.0 the public table reports the shipped `auto`
route against v1.3.0, the immediately previous comparable fp16 operator stack.
The v1.4 arm is the shipped auto router: eligible C<=256 fp16 submanifold convs
use fused gather-sum, C512 fwd+bwd and all ineligible shapes fall back to TIG. The
standalone fused gather-sum operator remains available via
``--include-standalone-fused`` as a diagnostic ceiling/failure-mode arm only. A
per-stage parity gate checks fused gather-sum forward against the v1.3 forward
BEFORE any timing is reported.

Batch sizes (the standing release-benchmark batch spec): train B=6 local / 12 H200;
val B=12 local / 24 H200. Real data only.

Run: TIG_BENCH_SCANNET=/path/to/scannet_v2/val \
     .venv/bin/python benchmarks/operators/bench_v14_versions.py --train-batch 6 --val-batch 12
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
from sparse_engines.fused_point_conv import fused_gather_sum_conv3d

device = "cuda"
SCANNET = os.environ.get("TIG_BENCH_SCANNET", "")
SCENES = [
    "scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00",
    "scene0030_00", "scene0046_00", "scene0050_00", "scene0064_00",
    "scene0086_00", "scene0095_00", "scene0100_00", "scene0131_00",
    "scene0144_00", "scene0153_00", "scene0164_00", "scene0169_00",
    "scene0187_00", "scene0193_00", "scene0207_00", "scene0217_00",
    "scene0221_00", "scene0011_01", "scene0019_01", "scene0249_00"]

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


# Dispatch-mode arms. AUTO is the shipped v1.4 route; the standalone
# fused operator is a diagnostic arm gated by --include-standalone-fused.
ARMS = [("PT schedule", "force_pt", False), ("FSG schedule", "force_fsg", False),
        ("TIG atomic schedule", "force_tig", True), ("TIG segment schedule", "force_tig", False),
        ("AUTO v1.4 route", "auto", False)]


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


def fused_weight_bias(pc):
    """Leaf (K,C,M) weight + (M,) bias holding the SAME values as the
    PointConv3d layer, so the fused arm computes the identical convolution and
    its backward times grad_weight + grad_bias on real leaves (matching the
    layer arms, which backprop into pc.weight / pc.bias)."""
    # PointConv3d weight layout is (K, G, in/G, out/G); G==1 here → (K, C, M).
    wf = pc.weight.squeeze(1).detach().clone().contiguous().requires_grad_(True)
    bf = pc.bias.detach().clone().requires_grad_(True)
    return wf, bf


def parity_check(pc, feat0, wf, bf, i, j, k, N):
    """v1.4 forward vs v1.3 forward (force_tig, seg on) on identical inputs.
    Returns max abs diff / max abs ref (fp16 reduction-order tolerance)."""
    with torch.no_grad():
        with dispatch_mode("force_tig"):
            ref = pc(feat0, i, j, k, N)
        fused = fused_gather_sum_conv3d(feat0, wf, i, j, k, N) + bf
        denom = ref.abs().max().item() + 1e-6
        return (fused - ref).abs().max().item() / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-batch", type=int, default=6)
    ap.add_argument("--val-batch", type=int, default=12)
    ap.add_argument("--dtypes", default="fp16,bf16", help="comma-sep: fp16,bf16")
    ap.add_argument("--parity-tol", type=float, default=5e-2,
                    help="max rel fwd diff v1.4-vs-v1.3 before a cell is flagged")
    ap.add_argument("--include-standalone-fused", action="store_true",
                    help="also time raw fused gather-sum as a diagnostic arm")
    args = ap.parse_args()
    global DTYPES
    DTYPES = [{"fp16": torch.float16, "bf16": torch.bfloat16}[d]
              for d in args.dtypes.split(",")]
    raws = [torch.from_numpy(np.load(f"{SCANNET}/{s}/coord.npy")).float().to(device)
            for s in SCENES[:max(args.train_batch, args.val_batch)]]
    rows = []
    parity = []
    for regime, B in (("train", args.train_batch), ("val", args.val_batch)):
        print(f"\n===== {regime.upper()} regime  B={B}  (real ScanNet) =====")
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
            for dt in DTYPES:
                dtn = "bf16" if dt == torch.bfloat16 else "fp16"
                line = f"  {label:9s} {dtn} N={N:7d} T={int(i.numel()):8d} | "
                base = None
                pc = PointConv3d(C, C, kernel_size=ks, bias=True).to(device).to(dt)
                feat0 = torch.randn(N, C, device=device, dtype=dt)
                wf, bf = fused_weight_bias(pc)

                # Parity gate BEFORE timing — fused must match the v1.3 forward.
                rel = parity_check(pc, feat0, wf, bf, i, j, k, N)
                ok = rel <= args.parity_tol
                parity.append(dict(regime=regime, stage=label, C=C, rel=rel, pass_=ok))
                if not ok:
                    print(f"    !! PARITY FLAG {label} {regime}: rel={rel:.2e} "
                          f"> tol={args.parity_tol:.0e}")

                # Dispatch-mode arms. These are scheduler diagnostics, not old
                # release tags.
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
                            print(f"    {aname} {label}: ERR {type(e).__name__}: {e}")
                    rows.append(dict(regime=regime, B=B, stage=label, N=N, C=C, K=ks**3,
                                     dtype=dtn, arm=aname, ms=ms))
                    if base is None:
                        base = ms
                    line += f"{aname.split()[0]} {ms:.2f}({base/ms:.2f}x) "

                if args.include_standalone_fused:
                    # Raw fused gather-sum diagnostic arm (same triplets). This is not
                    # the public v1.4 release route because C512 train may fall back.
                    def run_train_v14_fused():
                        f = feat0.clone().requires_grad_(True)
                        out = fused_gather_sum_conv3d(f, wf, i, j, k, N) + bf
                        out.sum().backward()
                        wf.grad = None
                        bf.grad = None
                    def run_val_v14_fused():
                        with torch.no_grad():
                            fused_gather_sum_conv3d(feat0, wf, i, j, k, N)
                    fn14 = run_train_v14_fused if regime == "train" else run_val_v14_fused
                    try:
                        ms14 = timed(fn14)
                    except Exception as e:
                        ms14 = float("nan")
                        print(f"    v1.4 FUSED-RAW {label}: ERR {type(e).__name__}: {e}")
                    rows.append(dict(regime=regime, B=B, stage=label, N=N, C=C, K=ks**3,
                                     dtype=dtn, arm="v1.4 FUSED-RAW", ms=ms14))
                    line += f"FUSED-RAW {ms14:.2f}({base/ms14:.2f}x)"
                print(line)
    out = f"/tmp/v14_versions_t{args.train_batch}_v{args.val_batch}.json"
    json.dump({"timings": rows, "parity": parity}, open(out, "w"), indent=2)
    n_fail = sum(1 for p in parity if not p["pass_"])
    print(f"\nparity: {len(parity)-n_fail}/{len(parity)} cells within tol "
          f"({n_fail} flagged)")
    print(f"wrote {out}")


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
