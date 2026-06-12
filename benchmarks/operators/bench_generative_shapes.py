"""Engine bench at the GENERATIVE conv shapes: patchify stem + fan-in-1 deconv.

Both production generative convs are excluded from TIG by the submanifold
gate (N_in != N_out) and run the eager engines, whose routing thresholds
(`_GROUPED_MIN_C = 128` etc.) were calibrated at K=27 submanifold square-ish
shapes. This bench measures the engines at the two real generative shapes:

- **Workload A — patchify stem** (the ViT Conv2d(P,P) patch-embed analog):
  one K**3-slot grid-partition strided conv, fan-out 1 per input point,
  fan-in = cell occupancy. Production: K=8 (512 slots), C_in tiny (2 outdoor /
  7 indoor with occupancy), M = trunk_dim.
- **Workload B — unpatchify deconv** (a multi-stage unpatchify head stage /
  inverse/transposed sparse-conv analog): 2**3 upsample over a nested grid
  hierarchy, **fan-in exactly 1** (each fine cell has one parent + one octant
  slot) — zero reduction in the forward.

Triplets are built standalone with the SAME math as the production builders
(grid partition for the stem; octant bit-shift for the fan-in-1 hierarchy),
sorted k-ascending (the production contract), and shared across all engines per cell
— the engine is the only variable.

A bytes-moved roofline column (minimal forward traffic / measured d2d copy
bandwidth) bounds how far each engine sits from "extreme".

Usage:
    PYTHONPATH=. python benchmarks/operators/bench_generative_shapes.py \\
        --scene-coord <dir-with-coord.npy> [--dtype all] [--output out.json]
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

warnings.filterwarnings("ignore", category=FutureWarning)

import torch

from layers.conv import PointConv3d
from sparse_engines._dispatch_override import dispatch_mode

device = "cuda"

DTYPES = [
    ("fp16", torch.float16),
    ("fp32", torch.float32),
    ("bf16", torch.bfloat16),
]
# TIG is structurally N/A at these shapes (submanifold-gated); force_tig
# falls through to the eager op where (mode != auto/force_pt, k-sorted)
# routes grouped — i.e. it degenerates to force_fsg. Not a distinct column.
MODES = [
    ("per_triplet", "force_pt"),       # PT engine.
    ("grouped",     "force_fsg"),      # FSG engine (sorted-k segment GEMM).
    ("auto",        "auto"),           # Production dispatcher.
    ("tig_gen",     "__tig_gen__"),    # generative TIG (direct call).
    ("tig_x",       "__tig_x__"),      # extreme — dense-GEMM partition
                                       # (stem) / FI1 store (deconv).
]

GRID_SIZE = 0.02  # base voxel size (m) — indoor ScanNet preprocessing.


def time_loop(fn, n_warmup=5, n_iters=12):
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
    return ts[len(ts) // 2]


def load_points(scene_coord_path: str | None, n_synth: int = 240_000):
    """Voxelized point coords at GRID_SIZE. Real scene if given, else a
    synthetic box matched to ScanNet scene0011_00's rough density."""
    if scene_coord_path:
        import numpy as np
        coord = np.load(os.path.join(scene_coord_path, "coord.npy"))
        pts = torch.from_numpy(coord).float().to(device)
    else:
        torch.manual_seed(0)
        pts = torch.rand(n_synth, 3, device=device) * torch.tensor(
            [6.0, 8.0, 3.0], device=device)
    # One representative per GRID_SIZE voxel (production preprocessing shape).
    vox = torch.div(pts, GRID_SIZE, rounding_mode="floor").long()
    key = (vox[:, 0] * 2_000_003 + vox[:, 1]) * 2_000_003 + vox[:, 2]
    uniq, inv = torch.unique(key, return_inverse=True)
    first_idx = torch.zeros(uniq.numel(), dtype=torch.long, device=device)
    first_idx.scatter_reduce_(0, inv, torch.arange(pts.shape[0], device=device),
                              reduce="amin", include_self=False)
    return pts[first_idx].contiguous()


def _cell_key(vox: torch.Tensor) -> torch.Tensor:
    return (vox[:, 0] * 2_000_003 + vox[:, 1]) * 2_000_003 + vox[:, 2]


def build_stem_triplets(pts: torch.Tensor, K: int, stride: int):
    """Grid-partition triplets — the production patchify builder math
    (cell id via floor(P/cell_size); sub-voxel slot in [0, K**3); k-sorted)."""
    cell_size = stride * GRID_SIZE
    cell_vox = torch.div(pts, cell_size, rounding_mode="floor")
    _, i = torch.unique(_cell_key(cell_vox.long()), return_inverse=True)
    n_out = int(i.max().item()) + 1
    frac = pts / cell_size - cell_vox
    sub = torch.floor(frac * K).long().clamp_(0, K - 1)
    k = (sub[:, 0] * K + sub[:, 1]) * K + sub[:, 2]
    j = torch.arange(pts.shape[0], device=device)
    order = torch.argsort(k)
    return i[order].contiguous(), j[order].contiguous(), k[order].contiguous(), n_out


def build_deconv_triplets(pts: torch.Tensor, fine_size: float):
    """Fan-in-1 upsample triplets for one 2**3 stage (coarse = 2*fine_size).

    Exact-nesting hierarchy (`floor(P/fine) >> 1 == floor(P/coarse)` per
    axis): i = each fine cell, j = its coarse parent, k = octant slot,
    k-sorted — the production unpatchify-hierarchy contract."""
    fine_vox = torch.div(pts, fine_size, rounding_mode="floor").long()
    fk = _cell_key(fine_vox)
    uniq_f, inv_f = torch.unique(fk, return_inverse=True)
    n_fine = uniq_f.numel()
    rep = torch.zeros(n_fine, dtype=torch.long, device=device)
    rep.scatter_reduce_(0, inv_f, torch.arange(pts.shape[0], device=device),
                        reduce="amin", include_self=False)
    fine_cells = fine_vox[rep]                      # one row per fine cell
    coarse_vox = fine_cells >> 1
    _, j = torch.unique(_cell_key(coarse_vox), return_inverse=True)
    n_coarse = int(j.max().item()) + 1
    oct_bits = fine_cells & 1
    k = oct_bits[:, 0] * 4 + oct_bits[:, 1] * 2 + oct_bits[:, 2]
    i = torch.arange(n_fine, device=device)
    order = torch.argsort(k)
    return i[order].contiguous(), j[order].contiguous(), k[order].contiguous(), \
        n_coarse, n_fine


def measure_bandwidth_gbs(n_bytes: int = 1 << 30) -> float:
    """Achievable d2d bandwidth (GB/s) via a large contiguous copy."""
    src = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    ms = time_loop(lambda: dst.copy_(src), n_warmup=3, n_iters=8)
    return (2 * n_bytes) / (ms * 1e-3) / 1e9


def roofline_fwd_ms(n_in, n_out, c, m, kk, dtype_bytes, bw_gbs):
    """Minimal forward traffic: read x once, read W once, write out once."""
    bytes_min = (n_in * c + kk * c * m + n_out * m) * dtype_bytes
    return bytes_min / (bw_gbs * 1e9) * 1e3


def bench_cell(c_in, m_out, kk, i, j, k, n_out, dtype, mode_str):
    n_in_feat = int(j.max().item()) + 1
    torch.manual_seed(0)
    conv = PointConv3d(c_in, m_out, kernel_size=round(kk ** (1 / 3)),
                       bias=False).to(device).to(dtype)
    x = torch.randn(n_in_feat, c_in, device=device, dtype=dtype)

    if mode_str == "__tig_x__":
        # The extreme paths, exploiting the builders' contracts
        # this bench constructs by hand (a production caller passes the
        # equivalent flags from the disjoint builders): K=512 partition ->
        # dense-GEMM im2col; K=8 fan-in-1 deconv -> FI1 plain-store fwd.
        from sparse_engines.tig import TigIndex, tig_mvmr
        from sparse_engines.partition_gemm import partition_dense_mvmr

        w = conv.weight

        if kk == 512:
            def x_call(feat):
                return partition_dense_mvmr(w, feat, i, j, k, n_out)
        else:
            def x_call(feat):
                idx = TigIndex(i, j, k, n_out, num_kernel_offsets=kk,
                               build_hybrid=False, assume_sorted=True,
                               n_in=n_in_feat, exact_cover_out=True)
                return tig_mvmr(w, feat, idx, mode="flat")

        fwd_ms = time_loop(lambda: x_call(x))
        x_g = x.clone().requires_grad_(True)

        def fwdbwd():
            x_g.grad = None
            w.grad = None
            x_call(x_g).sum().backward()

        torch.cuda.reset_peak_memory_stats()
        fb_ms = time_loop(fwdbwd)
        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        return fwd_ms, fb_ms, peak_mb

    if mode_str == "__tig_gen__":
        # The generative-TIG engine column, called directly (independent
        # of the auto-router wiring). TigIndex is built INSIDE the timed fn
        # — the production dispatch site builds it per call, so its
        # searchsorted cost is part of the engine's honest price.
        from sparse_engines.tig import TigIndex, tig_mvmr

        w = conv.weight  # (K, 1, C, M)

        def tig_call(feat):
            idx = TigIndex(i, j, k, n_out, num_kernel_offsets=kk,
                           build_hybrid=False, assume_sorted=True,
                           n_in=n_in_feat)
            return tig_mvmr(w, feat, idx, mode="flat")

        fwd_ms = time_loop(lambda: tig_call(x))
        x_g = x.clone().requires_grad_(True)

        def fwdbwd():
            x_g.grad = None
            w.grad = None
            tig_call(x_g).sum().backward()

        torch.cuda.reset_peak_memory_stats()
        fb_ms = time_loop(fwdbwd)
        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        return fwd_ms, fb_ms, peak_mb

    with dispatch_mode(mode_str):
        fwd_ms = time_loop(lambda: conv(x, i, j, k, n_out))

        x_g = x.clone().requires_grad_(True)

        def fwdbwd():
            x_g.grad = None
            conv.weight.grad = None
            conv(x_g, i, j, k, n_out).sum().backward()

        torch.cuda.reset_peak_memory_stats()
        fb_ms = time_loop(fwdbwd)
        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    return fwd_ms, fb_ms, peak_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-coord", default=None,
                    help="dir with coord.npy (real ScanNet scene); synthetic if absent")
    ap.add_argument("--dtype", default="all", choices=["all", "fp16", "fp32", "bf16"])
    ap.add_argument("--output", default=None, help="JSON output path")
    args = ap.parse_args()

    pts = load_points(args.scene_coord)
    n_raw = pts.shape[0]
    bw = measure_bandwidth_gbs()
    dev = torch.cuda.get_device_name(0)
    print(f"# bench_generative_shapes  device={dev}  N_voxelized={n_raw}  "
          f"d2d_bw={bw:.0f} GB/s  source="
          f"{'real:' + args.scene_coord if args.scene_coord else 'synthetic'}",
          flush=True)

    dtypes = [(n, t) for n, t in DTYPES if args.dtype in ("all", n)]

    # ── cell definitions ────────────────────────────────────────────────
    cells = []  # (workload, label, c_in, m_out, K^3, i, j, k, n_out, n_in)
    for c_in, m_out in [(7, 256), (2, 256), (7, 512)]:
        i, j, k, n_out = build_stem_triplets(pts, K=8, stride=8)
        cells.append(("stem", f"stem8x8x8_c{c_in}_m{m_out}",
                      c_in, m_out, 512, i, j, k, n_out, n_raw))
    # Deconv stages mirror a production unpatchify head (256,128,64,64) over
    # 0.16->0.08->0.04->0.02; plus one fat-channel probe at the raw level.
    stage_dims = [(256, 128, 0.08), (128, 64, 0.04), (64, 64, 0.02),
                  (256, 256, 0.02)]
    for c_in, m_out, fine in stage_dims:
        i, j, k, n_coarse, n_fine = build_deconv_triplets(pts, fine)
        tag = "fat" if (c_in, m_out) == (256, 256) else "s"
        cells.append(("deconv", f"deconv2x2x2_{tag}{fine}_c{c_in}_m{m_out}",
                      c_in, m_out, 8, i, j, k, n_fine, n_coarse))

    results = []
    for wl, label, c_in, m_out, kk, i, j, k, n_out, n_in in cells:
        t_per_k = i.numel() / kk
        print(f"\n## {label}  (T={i.numel()}, N_in={n_in}, N_out={n_out}, "
              f"T/K={t_per_k:.0f})", flush=True)
        print(f"| dtype | mode | fwd ms | fwd+bwd ms | peak MB | "
              f"roofline-fwd ms | fwd/roofline |", flush=True)
        print("|---|---|---:|---:|---:|---:|---:|", flush=True)
        for dt_name, dt in dtypes:
            rf = roofline_fwd_ms(n_in, n_out, c_in, m_out, kk,
                                 dt.itemsize, bw)
            for mode_name, mode_str in MODES:
                fwd_ms, fb_ms, peak = bench_cell(
                    c_in, m_out, kk, i, j, k, n_out, dt, mode_str)
                print(f"| {dt_name} | {mode_name} | {fwd_ms:.3f} | {fb_ms:.3f} "
                      f"| {peak:.0f} | {rf:.3f} | {fwd_ms / rf:.1f}x |",
                      flush=True)
                results.append(dict(
                    workload=wl, label=label, c_in=c_in, m_out=m_out, K3=kk,
                    T=int(i.numel()), n_in=int(n_in), n_out=int(n_out),
                    dtype=dt_name, mode=mode_name, fwd_ms=fwd_ms,
                    fwdbwd_ms=fb_ms, peak_mb=peak, roofline_fwd_ms=rf,
                    fwd_over_roofline=fwd_ms / rf))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(dict(device=dev, n_voxelized=int(n_raw), d2d_bw_gbs=bw,
                           grid_size=GRID_SIZE,
                           scene=args.scene_coord or "synthetic",
                           results=results), f, indent=1)
        print(f"\n# JSON written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
