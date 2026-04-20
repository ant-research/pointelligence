"""Comprehensive kernel profiling for MVMR/VVOR on H200.

Produces a structured breakdown of where time is spent so we can
target the right optimization (persistent kernels, TMA, FP8, atomic reduction).

Profiles:
  1. Isolated MVMR/VVOR kernel timing at ResNet18 channel dimensions
  2. Triplet building cost (radius_search + voxelize + sort)
  3. Roofline metrics: FLOPs, bytes transferred, arithmetic intensity
  4. Atomic contention: fan-in ratio (avg triplets per output point)
  5. Per-stage breakdown of ResNet18 forward + backward
"""

import argparse
import gc
import json
import sys
import os
import time
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cuda_timer(fn, warmup=5, iters=20):
    """Time a CUDA callable with events, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    # trim top/bottom 20%
    lo = len(times) // 5
    hi = len(times) - lo
    return sum(times[lo:hi]) / (hi - lo)


def fmt_ms(ms):
    return f"{ms:.3f}" if ms < 100 else f"{ms:.1f}"


def print_table(title, rows, headers):
    """Pretty-print a table."""
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, v in enumerate(row):
            col_w[i] = max(col_w[i], len(str(v)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(fmt.format(*headers))
    print("-" * sum(col_w + [2 * (len(col_w) - 1)]))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print()


# ---------------------------------------------------------------------------
# 1. Isolated kernel profiling
# ---------------------------------------------------------------------------

def profile_mvmr_vvor(device="cuda", dtype=torch.float16):
    """Profile MVMR and VVOR kernels at channel dims matching ResNet18 stages."""
    from sparse_engines.mvmr_triton import sparse_matrix_vector_multiplication_reduction
    from sparse_engines.vvor_triton import sparse_vector_vector_outer_product_reduction

    # ResNet18 stages: (C_in, C_out, K, approx T, approx N_in, approx N_out)
    # K=27 for 3x3x3, K=343 for 7x7x7
    configs = [
        # label,          G, C_in/G, C_out/G, K,   T
        ("conv1 7x7x7",   1,    1,    64, 343, 100_000),
        ("layer1 3x3x3",  1,   64,    64,  27, 200_000),
        ("layer2 3x3x3",  1,   64,   128,  27, 100_000),
        ("layer2b 3x3x3", 1,  128,   128,  27, 100_000),
        ("layer3 3x3x3",  1,  128,   256,  27,  50_000),
        ("layer3b 3x3x3", 1,  256,   256,  27,  50_000),
        ("layer4 3x3x3",  1,  256,   512,  27,  25_000),
        ("layer4b 3x3x3", 1,  512,   512,  27,  25_000),
    ]

    rows_mvmr = []
    rows_vvor = []

    for label, G, Ci, Co, K, T in configs:
        N_in = T // 15   # rough: ~15 triplets per input point
        N_out = T // 15

        # weight: (K, G, Ci, Co)
        weight = torch.randn(K, G, Ci, Co, device=device, dtype=dtype)
        inp = torch.randn(N_in, G, Ci, device=device, dtype=dtype)
        grad_out = torch.randn(N_out, G, Co, device=device, dtype=dtype)

        # Build synthetic sorted triplet indices
        k_idx = torch.sort(torch.randint(0, K, (T,), device=device))[0]
        j_idx = torch.randint(0, N_in, (T,), device=device)
        i_idx = torch.randint(0, N_out, (T,), device=device)

        # --- MVMR: output[i] += weight[k] @ input[j] ---
        def run_mvmr():
            sparse_matrix_vector_multiplication_reduction(
                weight, k_idx, inp, j_idx, i_idx, N_out
            )

        mvmr_ms = cuda_timer(run_mvmr)

        # Roofline for MVMR:
        # FLOPs: T * G * Ci * Co * 2  (multiply + add)
        # Bytes: T * (G*Ci*Co + G*Ci + G*Co) * elem_size  (weight + input + output read/write)
        elem = 2 if dtype == torch.float16 else 4
        flops = T * G * Ci * Co * 2
        # Loads: weight[k] loaded once per unique k in the L-block (approx T loads),
        # input[j] loaded, output[i] atomically written
        bytes_moved = T * (G * Ci * Co + G * Ci) * elem + T * G * Co * 4  # output always fp32
        ai = flops / bytes_moved if bytes_moved > 0 else 0
        bw_gb = (bytes_moved / 1e9) / (mvmr_ms / 1e3) if mvmr_ms > 0 else 0
        tflops = (flops / 1e12) / (mvmr_ms / 1e3) if mvmr_ms > 0 else 0

        # Fan-in: avg triplets per output point
        fan_in = T / N_out

        rows_mvmr.append([
            label, f"{T//1000}K", f"{G}x{Ci}x{Co}", f"K={K}",
            fmt_ms(mvmr_ms), f"{ai:.1f}", f"{bw_gb:.0f}", f"{tflops:.2f}",
            f"{fan_in:.1f}",
        ])

        # --- VVOR: grad_weight[k] += input[j] outer grad_out[i] ---
        def run_vvor():
            sparse_vector_vector_outer_product_reduction(
                inp, j_idx, grad_out, i_idx, k_idx, K
            )

        vvor_ms = cuda_timer(run_vvor)

        flops_vvor = T * G * Ci * Co * 2
        bytes_vvor = T * (G * Ci + G * Co) * elem + T * G * Ci * Co * 4
        ai_vvor = flops_vvor / bytes_vvor if bytes_vvor > 0 else 0
        bw_vvor = (bytes_vvor / 1e9) / (vvor_ms / 1e3) if vvor_ms > 0 else 0
        tflops_vvor = (flops_vvor / 1e12) / (vvor_ms / 1e3) if vvor_ms > 0 else 0
        fan_in_k = T / K  # avg triplets per weight offset

        rows_vvor.append([
            label, f"{T//1000}K", f"{G}x{Ci}x{Co}", f"K={K}",
            fmt_ms(vvor_ms), f"{ai_vvor:.1f}", f"{bw_vvor:.0f}", f"{tflops_vvor:.2f}",
            f"{fan_in_k:.0f}",
        ])

        del weight, inp, grad_out, k_idx, j_idx, i_idx
        torch.cuda.empty_cache()

    hdrs = ["Stage", "T", "G*Ci*Co", "Kernel", "ms", "AI(F/B)", "BW(GB/s)", "TFLOPS", "FanIn"]
    print_table("MVMR Forward Kernel", rows_mvmr, hdrs)
    print_table("VVOR Backward Kernel (grad_weight)", rows_vvor,
                ["Stage", "T", "G*Ci*Co", "Kernel", "ms", "AI(F/B)", "BW(GB/s)", "TFLOPS", "FanIn/K"])

    return rows_mvmr, rows_vvor


# ---------------------------------------------------------------------------
# 2. Triplet building profiling
# ---------------------------------------------------------------------------

def profile_triplet_building(device="cuda"):
    """Profile radius_search + voxelize + sort at different scales."""
    from layers.metadata import MetaData

    scales = [
        ("5K pts",   5_000, 2, 0.05),
        ("10K pts", 10_000, 2, 0.05),
        ("20K pts", 20_000, 2, 0.05),
        ("50K pts", 50_000, 2, 0.1),
    ]

    rows = []
    for label, npts, batch_size, grid_size in scales:
        coords = torch.rand(npts * batch_size, 3, device=device)
        sample_sizes = torch.tensor([npts] * batch_size, device=device, dtype=torch.long)
        sample_inds = torch.repeat_interleave(
            torch.arange(batch_size, device=device), sample_sizes
        )

        def build():
            m = MetaData(
                points=coords, sample_inds=sample_inds,
                sample_sizes=sample_sizes, grid_size=grid_size,
                kernel_size=(3, 3, 3), sort_by="k",
            )
            return m

        # warmup
        m = build()
        T = m.i.numel()
        avg_neighbors = T / (npts * batch_size)

        ms = cuda_timer(build, warmup=3, iters=10)

        rows.append([label, f"bs={batch_size}", f"gs={grid_size}", f"T={T//1000}K",
                      f"avg_nb={avg_neighbors:.1f}", fmt_ms(ms)])

        del coords, sample_sizes, sample_inds, m
        torch.cuda.empty_cache()

    print_table("Triplet Building (radius_search + voxelize + sort)",
                rows, ["Scale", "Batch", "GridSize", "Triplets", "AvgNeighbors", "ms"])
    return rows


# ---------------------------------------------------------------------------
# 3. Per-stage ResNet18 breakdown
# ---------------------------------------------------------------------------

class TimingHook:
    """Attaches forward hooks to named modules to measure per-layer time."""
    def __init__(self):
        self.times = defaultdict(list)
        self._handles = []
        self._starts = {}

    def attach(self, model, names_and_modules):
        for name, mod in names_and_modules:
            self._handles.append(
                mod.register_forward_pre_hook(self._make_pre(name))
            )
            self._handles.append(
                mod.register_forward_hook(self._make_post(name))
            )

    def _make_pre(self, name):
        def hook(mod, inp):
            torch.cuda.synchronize()
            self._starts[name] = time.perf_counter()
        return hook

    def _make_post(self, name):
        def hook(mod, inp, out):
            torch.cuda.synchronize()
            self.times[name].append((time.perf_counter() - self._starts[name]) * 1000)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()

    def summary(self):
        result = {}
        for name, ts in self.times.items():
            ts.sort()
            lo = len(ts) // 5
            hi = len(ts) - lo
            if hi > lo:
                result[name] = sum(ts[lo:hi]) / (hi - lo)
            elif ts:
                result[name] = ts[0]
        return result


def profile_resnet18_stages(device="cuda", dtype=torch.float16, num_points=10000):
    """Profile ResNet18 per-stage with forward and backward breakdown."""
    from models import resnet18
    from layers.downsample import downsample
    from tests.test_harness import generate_synthetic_batch

    batch_size = 2
    in_channels = 64

    batch_data = generate_synthetic_batch(num_points, batch_size, in_channels,
                                          dtype=torch.float32, device=device)
    coords = batch_data['coords']
    features = batch_data['features'].to(dtype=dtype)
    sample_sizes = torch.tensor(batch_data['batch_sizes'], device=device, dtype=torch.long)

    # Pre-downsample like the benchmark does
    sample_inds = torch.repeat_interleave(
        torch.arange(batch_size, device=device), sample_sizes
    )
    grid_size = 0.05
    coords, sample_inds, new_gs, ds_idx = downsample(coords, sample_inds, grid_size, 1.0)
    features = features[ds_idx]
    sample_sizes = torch.bincount(sample_inds)

    model = resnet18(in_channels=in_channels).to(device=device, dtype=dtype)

    # ---- Forward-only per-stage timing ----
    named_stages = [
        ("conv1", model.conv1),
        ("bn1+relu", model.bn1),
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
        ("avgpool+fc", model.avgpool),
    ]

    # Warmup
    model.eval()
    for _ in range(3):
        with torch.no_grad():
            model(features, coords, sample_sizes, new_gs)
    torch.cuda.synchronize()

    # Timed runs with hooks
    hook = TimingHook()
    hook.attach(model, named_stages)

    iters = 15
    total_fwd_times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(features, coords, sample_sizes, new_gs)
        torch.cuda.synchronize()
        total_fwd_times.append((time.perf_counter() - t0) * 1000)

    hook.remove()
    stage_times = hook.summary()

    total_fwd_times.sort()
    lo = len(total_fwd_times) // 5
    hi = len(total_fwd_times) - lo
    total_fwd = sum(total_fwd_times[lo:hi]) / (hi - lo)

    rows_fwd = []
    accounted = 0
    for name, _ in named_stages:
        ms = stage_times.get(name, 0)
        pct = (ms / total_fwd * 100) if total_fwd > 0 else 0
        rows_fwd.append([name, fmt_ms(ms), f"{pct:.1f}%"])
        accounted += ms
    overhead = total_fwd - accounted
    rows_fwd.append(["[overhead/triplets]", fmt_ms(overhead), f"{overhead/total_fwd*100:.1f}%"])
    rows_fwd.append(["TOTAL", fmt_ms(total_fwd), "100%"])

    print_table(f"ResNet18 Forward Breakdown ({num_points}pts/cloud, {dtype})",
                rows_fwd, ["Stage", "ms", "% of total"])

    # ---- Forward + backward total ----
    model.train()
    features_grad = features.clone().detach().requires_grad_(True)

    # warmup
    for _ in range(3):
        out = model(features_grad, coords, sample_sizes, new_gs)
        out.sum().backward()
        model.zero_grad(set_to_none=True)
        features_grad = features.clone().detach().requires_grad_(True)

    train_times = []
    for _ in range(iters):
        features_grad = features.clone().detach().requires_grad_(True)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = model(features_grad, coords, sample_sizes, new_gs)
        out.sum().backward()
        e.record()
        torch.cuda.synchronize()
        train_times.append(s.elapsed_time(e))
        model.zero_grad(set_to_none=True)

    train_times.sort()
    lo = len(train_times) // 5
    hi = len(train_times) - lo
    avg_train = sum(train_times[lo:hi]) / (hi - lo)

    print(f"  Forward+Backward total: {fmt_ms(avg_train)} ms")
    print(f"  Backward only (est):    {fmt_ms(avg_train - total_fwd)} ms")
    print(f"  Backward/Forward ratio: {(avg_train - total_fwd) / total_fwd:.2f}x")

    return {
        "forward_ms": total_fwd,
        "training_ms": avg_train,
        "backward_est_ms": avg_train - total_fwd,
        "stages": stage_times,
    }


# ---------------------------------------------------------------------------
# 4. Atomic contention analysis
# ---------------------------------------------------------------------------

def profile_atomic_contention(device="cuda"):
    """Measure actual fan-in distribution from real triplet builds."""
    from layers.metadata import MetaData

    configs = [
        ("10K gs=0.05", 10_000, 2, 0.05),
        ("10K gs=0.1",  10_000, 2, 0.1),
        ("50K gs=0.1",  50_000, 2, 0.1),
    ]

    rows = []
    for label, npts, bs, gs in configs:
        coords = torch.rand(npts * bs, 3, device=device)
        ss = torch.tensor([npts] * bs, device=device, dtype=torch.long)
        si = torch.repeat_interleave(torch.arange(bs, device=device), ss)

        m = MetaData(points=coords, sample_inds=si, sample_sizes=ss,
                     grid_size=gs, kernel_size=(3, 3, 3), sort_by="k")

        T = m.i.numel()
        N_out = coords.shape[0]
        # Fan-in per output point
        counts = torch.bincount(m.i, minlength=N_out)
        avg_fan = counts.float().mean().item()
        max_fan = counts.max().item()
        std_fan = counts.float().std().item()
        # Fan-in per weight index
        K = 27
        k_counts = torch.bincount(m.k, minlength=K)
        avg_k = k_counts.float().mean().item()
        max_k = k_counts.max().item()

        rows.append([label, f"T={T//1000}K", f"N={N_out//1000}K",
                      f"{avg_fan:.1f}", f"{max_fan}", f"{std_fan:.1f}",
                      f"{avg_k:.0f}", f"{max_k}"])

        del coords, ss, si, m
        torch.cuda.empty_cache()

    print_table("Atomic Contention Analysis",
                rows,
                ["Config", "Triplets", "N_out", "AvgFanIn_i", "MaxFanIn_i",
                 "StdFanIn_i", "AvgFanIn_k", "MaxFanIn_k"])
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kernel profiling for H200 optimization")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--points", type=int, default=10000, help="Points per cloud for ResNet18 profiling")
    parser.add_argument("--skip-resnet", action="store_true", help="Skip ResNet18 stage breakdown")
    parser.add_argument("--skip-kernels", action="store_true", help="Skip isolated kernel profiling")
    parser.add_argument("--skip-triplets", action="store_true", help="Skip triplet building profiling")
    parser.add_argument("--skip-contention", action="store_true", help="Skip atomic contention analysis")
    parser.add_argument("--json", type=str, default=None, help="Write results to JSON file")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"dtype: {args.dtype}, points: {args.points}")

    results = {}

    if not args.skip_kernels:
        results["mvmr"], results["vvor"] = profile_mvmr_vvor(device, dtype)

    if not args.skip_triplets:
        results["triplets"] = profile_triplet_building(device)

    if not args.skip_contention:
        results["contention"] = profile_atomic_contention(device)

    if not args.skip_resnet:
        results["resnet18"] = profile_resnet18_stages(device, dtype, args.points)

    if args.json:
        # Convert to serializable format
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {kk: vv if not isinstance(vv, float) else round(vv, 3)
                                    for kk, vv in v.items()}
            else:
                serializable[k] = str(v)
        with open(args.json, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nResults written to {args.json}")

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
