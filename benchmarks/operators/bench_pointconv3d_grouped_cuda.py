"""Engine-level A/B: hand-CUDA grouped vs Triton grouped (PointConv3d).

Measures kernel-time MVMR + VVOR at the engine level — raw kernel calls
on synthetic triplets, not the conv-wrapper level. This isolates the
kernel cost from build_triplets / autograd overhead.

The two engines under test:

- per_triplet / Triton grouped — the baseline path. One Triton program
  per (i, k) pair, register-resident weight per program; `tl.dot` does
  the (M, L) @ (L, C) matmul when both M and C are deep enough.
- grouped / hand-CUDA — the v1.1.0 path. A hand-written CUDA kernel
  that holds weights register-resident across multiple `(i, k)` cells
  per warp, amortising the launch + weight-load cost; the WMMA variant
  (`sparse_vvor_grouped_wmma`) uses m16n16k16 tensor-core mma for
  fp16/bf16 when `M % 16 == 0` and `C % 16 == 0`, with a scalar-FMA
  fallback otherwise.

Methodology:
  - 5 warmup + 12 measurement runs.
  - torch.cuda.synchronize() + time.perf_counter() for cross-kernel
    comparisons (NOT torch.cuda.Event — see PROGRESS.md Notes).
  - Median + IQR over the 12 measurement runs.
  - 3 process invocations recommended; this script measures one
    invocation. Use a driver script for cross-run aggregation.

Usage:
  python benchmarks/operators/bench_pointconv3d_grouped_cuda.py
  python benchmarks/operators/bench_pointconv3d_grouped_cuda.py --stages enc4
  python benchmarks/operators/bench_pointconv3d_grouped_cuda.py --dtype fp16
  python benchmarks/operators/bench_pointconv3d_grouped_cuda.py --json out.json
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

warnings.filterwarnings("ignore", category=FutureWarning)

import torch

import sparse_engines  # registers ops
from sparse_engines._dispatch_override import dispatch_mode
from sparse_engines.mvmr_grouped_cuda import (
    sparse_matrix_vector_multiplication_reduction_grouped_cuda,
)
from sparse_engines.vvor_grouped_cuda import (
    sparse_vector_vector_outer_product_reduction_grouped_cuda,
)
from sparse_engines.vvor_grouped_wmma import (
    sparse_vector_vector_outer_product_reduction_grouped_wmma,
)
from sparse_engines.vvor_grouped_wmma_coop import (
    sparse_vector_vector_outer_product_reduction_grouped_wmma_coop,
)
from sparse_engines.vvor_cutlass import (
    sparse_vector_vector_outer_product_reduction_grouped_cutlass,
)
from sparse_engines.mvmr_cutlass import (
    sparse_matrix_vector_multiplication_reduction_cutlass,
)


# PTv3 stage shapes — same as test_grouped_cuda_parity.py.
# Synthetic-T mode (default): N matches PTv3 cumulative shapes; T is hand-
# chosen low (representative of synthetic-randint testing).
PTV3_STAGES = {
    "enc2": (27,  3_000,  3_000, 128, 128,  25_000),
    "enc3": (27,    800,    800, 256, 256,   6_500),
    "enc4": (27,    200,    200, 512, 512,   1_700),
}

# G11.6 (cycle-3 §1.8): production-T mode. T values match what
# build_triplets produces in the wrapper-level bench at PTv3 stage shapes
# (~5 neighbors per query). enc0/enc1 added with their small C (32/64)
# explicitly — even though the Triton grouped path's _GROUPED_MIN_C=128
# would normally route these to per-triplet, force_grouped lets us bench
# the grouped kernel directly for the kernel-vs-wrapper diagnostic.
PTV3_STAGES_PRODUCTION_T = {
    "enc0": (27, 328_326, 328_326,  32,  32, 1_640_000),
    "enc1": (27, 113_242, 113_242,  64,  64,   566_000),
    "enc2": (27,  30_689,  30_689, 128, 128,   153_000),
    "enc3": (27,   7_993,   7_993, 256, 256,    40_000),
    "enc4": (27,   1_950,   1_950, 512, 512,     9_750),
}

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_mvmr_indices(N_a, N_b, N_o, T, device):
    torch.manual_seed(1)
    a_idx = torch.randint(0, N_a, (T,), device=device, dtype=torch.int64)
    b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)
    o_idx = torch.randint(0, N_o, (T,), device=device, dtype=torch.int64)
    order = torch.argsort(a_idx, stable=True)
    return a_idx[order], b_idx[order], o_idx[order]


def _make_vvor_indices(N_a, N_b, K_off, T, device):
    torch.manual_seed(1)
    a_idx = torch.randint(0, N_a, (T,), device=device, dtype=torch.int64)
    b_idx = torch.randint(0, N_b, (T,), device=device, dtype=torch.int64)
    o_idx = torch.randint(0, K_off, (T,), device=device, dtype=torch.int64)
    order = torch.argsort(o_idx, stable=True)
    return a_idx[order], b_idx[order], o_idx[order]


def _composite_resort_vvor(a_idx, b_idx, o_idx, N_a):
    """Re-sort triplets by composite key (o_idx primary, a_idx secondary).

    Used by the cycle-2 §3 A/B test (post-FAIL refinement): does the
    hand-CUDA vvor close the 3.5x regression if we ensure consecutive
    triplets within a k-segment share a_idx (= out-point index in
    vvor's convention)?

    Composite key = o_idx * N_a + a_idx fits in int64 for production sizes.
    Stable single argsort on the composite key preserves o_idx-primary
    ascending order (required by kernel_offset_segments) while making
    a_idx-runs contiguous within each o_idx segment.
    """
    key = o_idx.to(torch.int64) * int(N_a) + a_idx.to(torch.int64)
    order = torch.argsort(key, stable=True)
    return a_idx[order], b_idx[order], o_idx[order]


def _time_fn(fn, n_warmup=5, n_iters=12):
    """Median + IQR ms over n_iters runs, after n_warmup discarded warmups.

    Uses torch.cuda.synchronize() + time.perf_counter() per PROGRESS.md
    Notes (NOT torch.cuda.Event — measurement overhead inflates short
    kernels).
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples_ms = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    n = len(samples_ms)
    median = samples_ms[n // 2]
    q1 = samples_ms[n // 4]
    q3 = samples_ms[(3 * n) // 4]
    return {"median_ms": median, "iqr_ms": q3 - q1, "min_ms": samples_ms[0], "max_ms": samples_ms[-1]}


def bench_engine_level(stage, dtype_name, dtype, device, stages_map=None):
    """mvmr + vvor at engine level. Returns dict per backend.

    stages_map: dict matching PTV3_STAGES shape. Defaults to synthetic-T
    PTV3_STAGES. Pass PTV3_STAGES_PRODUCTION_T to bench at production sizes.
    """
    if stages_map is None:
        stages_map = PTV3_STAGES
    K_off, N_b, N_o, M, C, T = stages_map[stage]

    # mvmr inputs: a = weight (K, G=1, C, M); b = input (N_b, G=1, C); -> out (N_o, G=1, M)
    a = (torch.randn(K_off, 1, C, M, device=device, dtype=torch.float32) * 0.1).to(dtype)
    b = (torch.randn(N_b, 1, C, device=device, dtype=torch.float32) * 0.1).to(dtype)
    a_idx_m, b_idx_m, o_idx_m = _make_mvmr_indices(K_off, N_b, N_o, T, device)

    # vvor inputs: a = grad_out (N_o, G=1, M); b = input (N_b, G=1, C); -> grad_weight (K, G=1, M, C)
    g_out = (torch.randn(N_o, 1, M, device=device, dtype=torch.float32) * 0.1).to(dtype)
    a_idx_v, b_idx_v, o_idx_v = _make_vvor_indices(N_o, N_b, K_off, T, device)

    results = {}

    # --- Triton grouped path (current production default for C >= 128) ---
    def triton_mvmr():
        with dispatch_mode("force_grouped"):
            sparse_engines.ops.sparse_matrix_vector_multiplication_reduction(
                a, a_idx_m, b, b_idx_m, o_idx_m, N_o,
            )

    def triton_vvor():
        with dispatch_mode("force_grouped"):
            sparse_engines.ops.sparse_vector_vector_outer_product_reduction(
                g_out, a_idx_v, b, b_idx_v, o_idx_v, K_off,
            )

    results["triton_grouped_mvmr"] = _time_fn(triton_mvmr)
    results["triton_grouped_vvor"] = _time_fn(triton_vvor)

    # --- Hand-CUDA grouped path (Branch B / G10 under test) ---
    def cuda_mvmr():
        sparse_matrix_vector_multiplication_reduction_grouped_cuda(
            a, a_idx_m, b, b_idx_m, o_idx_m, N_o,
        )

    def cuda_vvor():
        sparse_vector_vector_outer_product_reduction_grouped_cuda(
            g_out, a_idx_v, b, b_idx_v, o_idx_v, K_off,
        )

    results["cuda_grouped_mvmr"] = _time_fn(cuda_mvmr)
    results["cuda_grouped_vvor"] = _time_fn(cuda_vvor)

    # --- Hand-CUDA grouped path with composite (k, a)-sort (vvor only) ---
    # Cycle-2 §3 A/B refinement: re-sort triplets so consecutive items
    # within each k-segment share a_idx (out-point), enabling the kernel's
    # `prev_out`-gated grad_out_reg reuse to actually fire.
    a_idx_v_re, b_idx_v_re, o_idx_v_re = _composite_resort_vvor(
        a_idx_v, b_idx_v, o_idx_v, N_o,
    )

    def cuda_vvor_resorted():
        sparse_vector_vector_outer_product_reduction_grouped_cuda(
            g_out, a_idx_v_re, b, b_idx_v_re, o_idx_v_re, K_off,
        )

    results["cuda_grouped_vvor_resorted"] = _time_fn(cuda_vvor_resorted)

    # Also include the sort cost itself so we can attribute it correctly.
    def vvor_resort_only():
        _composite_resort_vvor(a_idx_v, b_idx_v, o_idx_v, N_o)

    results["composite_sort_overhead"] = _time_fn(vvor_resort_only)

    # --- Hand-CUDA WMMA-direct vvor (Tier-1.5 / cycle-3 §1) ---
    # WMMA atom is m16n16k16 fp16/bf16. fp32 routes to scalar-FMA path
    # inside the wrapper (so the bench measurement is the WMMA kernel
    # specifically only for fp16/bf16 dtype).
    if dtype in (torch.float16, torch.bfloat16):
        def wmma_vvor():
            sparse_vector_vector_outer_product_reduction_grouped_wmma(
                g_out, a_idx_v, b, b_idx_v, o_idx_v, K_off,
            )
        results["cuda_grouped_vvor_wmma"] = _time_fn(wmma_vvor)
    else:
        results["cuda_grouped_vvor_wmma"] = {
            "median_ms": float("nan"), "iqr_ms": float("nan"),
            "min_ms": float("nan"), "max_ms": float("nan"),
            "note": "WMMA atom not available for fp32; routes to scalar-FMA fallback",
        }

    # --- Coop-warp split-K WMMA vvor (cycle-3 §1.9a / G11.7a) ---
    # W=8 slices per tile: 8x more blocks than single-warp baseline. Goal:
    # reduce kernel under-utilization at small-C / large-T (enc0: 108 -> 864 blocks).
    if dtype in (torch.float16, torch.bfloat16):
        def wmma_coop_vvor():
            sparse_vector_vector_outer_product_reduction_grouped_wmma_coop(
                g_out, a_idx_v, b, b_idx_v, o_idx_v, K_off, w=8,
            )
        results["cuda_grouped_vvor_wmma_coop"] = _time_fn(wmma_coop_vvor)
    else:
        results["cuda_grouped_vvor_wmma_coop"] = {
            "median_ms": float("nan"), "iqr_ms": float("nan"),
            "min_ms": float("nan"), "max_ms": float("nan"),
            "note": "WMMA-coop atom not available for fp32",
        }

    # --- Tier-2 CUTLASS implicit-GEMM + K-mode IndexedGather vvor
    #     (cycle-4 §1.11 G14 Tasks 1-3). fp16-only: the C++ kernel
    #     TORCH_CHECKs at::kHalf. bf16/fp32 report NaN with a note,
    #     mirroring the WMMA fp32 pattern. This is the §2.2 metric.
    # The Tier-2 path additionally requires M and C to be multiples of
    # the 64x64 kernel tile (intrinsic kernel constraint, not a bug);
    # enc0 (M=C=32) is below tile size and reports NaN with a note.
    _M_v, _C_v = g_out.shape[2], b.shape[2]
    if dtype == torch.float16 and _M_v % 64 == 0 and _C_v % 64 == 0:
        def cutlass_vvor():
            sparse_vector_vector_outer_product_reduction_grouped_cutlass(
                g_out, a_idx_v, b, b_idx_v, o_idx_v, K_off,
            )
        results["cuda_grouped_vvor_cutlass"] = _time_fn(cutlass_vvor)
    else:
        _note = (
            "Tier-2 CUTLASS vvor is fp16-only (C++ TORCH_CHECK at::kHalf)"
            if dtype != torch.float16
            else f"Tier-2 CUTLASS vvor requires M,C %% 64 == 0; got M={_M_v}, C={_C_v}"
        )
        results["cuda_grouped_vvor_cutlass"] = {
            "median_ms": float("nan"), "iqr_ms": float("nan"),
            "min_ms": float("nan"), "max_ms": float("nan"),
            "note": _note,
        }

    # --- Tier-2 CUTLASS implicit-GEMM + S-mode IndexedGather + scatter
    #     mvmr (G22 plan §4 Tasks M1-M4; M4's force_grouped_cutlass_mvmr
    #     dispatch mode). fp16-only: the C++ kernel TORCH_CHECKs at::kHalf.
    #     bf16/fp32 report NaN with a note, mirroring the vvor-CUTLASS
    #     pattern. This is the §2.2 *engine* metric (M5).
    # mvmr's kernel tile is TileM=64 (M axis) / TileK=32 (C contraction
    # axis), so the constraint is M % 64 == 0 and C % 32 == 0 (vvor's is
    # M,C % 64 — different because vvor's tile is 64x64x32 over a
    # different contraction axis). enc0 (M=C=32) fails M % 64 and reports
    # NaN with a note, mirroring the vvor-CUTLASS enc0 handling.
    if dtype == torch.float16 and M % 64 == 0 and C % 32 == 0:
        def cutlass_mvmr():
            sparse_matrix_vector_multiplication_reduction_cutlass(
                a, a_idx_m, b, b_idx_m, o_idx_m, N_o,
            )
        results["cuda_grouped_mvmr_cutlass"] = _time_fn(cutlass_mvmr)
    else:
        _note_m = (
            "Tier-2 CUTLASS mvmr is fp16-only (C++ TORCH_CHECK at::kHalf)"
            if dtype != torch.float16
            else f"Tier-2 CUTLASS mvmr requires M %% 64 == 0 and C %% 32 == 0; "
                 f"got M={M}, C={C}"
        )
        results["cuda_grouped_mvmr_cutlass"] = {
            "median_ms": float("nan"), "iqr_ms": float("nan"),
            "min_ms": float("nan"), "max_ms": float("nan"),
            "note": _note_m,
        }

    # Combined fwd+bwd-ish: mvmr (fwd) + vvor (bwd grad_weight). Missing
    # grad_input (which is another mvmr with transposed weight), but
    # that's symmetric to the fwd mvmr time so combined = fwd + vvor +
    # ~fwd is a fair approximation. We report mvmr+vvor as the kernel-
    # level "fwdbwd" composite; the conv-wrapper fwdbwd time will be
    # higher due to autograd graph overhead.
    results["combined_mvmr_plus_vvor"] = {
        "triton_ms": results["triton_grouped_mvmr"]["median_ms"] + results["triton_grouped_vvor"]["median_ms"],
        "cuda_ms":   results["cuda_grouped_mvmr"]["median_ms"]   + results["cuda_grouped_vvor"]["median_ms"],
        "cuda_vs_triton_ratio": (
            (results["cuda_grouped_mvmr"]["median_ms"] + results["cuda_grouped_vvor"]["median_ms"]) /
            max(1e-6, results["triton_grouped_mvmr"]["median_ms"] + results["triton_grouped_vvor"]["median_ms"])
        ),
    }

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-t", action="store_true",
                    help="Use production T sizes (matching build_triplets output in the "
                         "wrapper bench). Also includes enc0/enc1 stages with small C. "
                         "Default: synthetic T (PTV3_STAGES from cycle-3 §1).")
    ap.add_argument("--stages", nargs="+", default=None,
                    help="PTv3 stages to bench. Default: all stages in the active map "
                         "(synthetic = enc{2,3,4}; production-t = enc{0,1,2,3,4}).")
    ap.add_argument("--dtype", nargs="+", default=["fp16"],
                    choices=list(DTYPES.keys()),
                    help="dtypes to bench (default: fp16 only — the cycle-2 §3 target)")
    ap.add_argument("--json", type=str, default=None, help="Write JSON results to this path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable — aborting.")
        sys.exit(2)
    device = "cuda"

    stages_map = PTV3_STAGES_PRODUCTION_T if args.production_t else PTV3_STAGES
    if args.stages is None:
        args.stages = list(stages_map.keys())
    else:
        for s in args.stages:
            if s not in stages_map:
                sys.exit(f"--stages {s!r} not in active stage map (use --production-t for enc0/enc1)")

    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"# GPU: {gpu_name} (sm_{cc[0]}{cc[1]})")
    print(f"# torch: {torch.__version__}")
    print(f"# mode: {'production-T (from build_triplets)' if args.production_t else 'synthetic-T (cycle-3 §1)'}")
    print(f"# stages: {args.stages}")
    print(f"# dtypes: {args.dtype}")
    print()

    all_results = {
        "gpu": gpu_name, "sm": f"sm_{cc[0]}{cc[1]}",
        "torch": torch.__version__,
        "mode": "production-T" if args.production_t else "synthetic-T",
        "cells": [],
    }

    for stage in args.stages:
        for dt_name in args.dtype:
            dtype = DTYPES[dt_name]
            r = bench_engine_level(stage, dt_name, dtype, device, stages_map=stages_map)
            cell = {"stage": stage, "dtype": dt_name, **r,
                    "T": stages_map[stage][5], "M": stages_map[stage][3], "C": stages_map[stage][4]}
            all_results["cells"].append(cell)

            m_t = r["triton_grouped_mvmr"]["median_ms"]
            v_t = r["triton_grouped_vvor"]["median_ms"]
            m_c = r["cuda_grouped_mvmr"]["median_ms"]
            v_c = r["cuda_grouped_vvor"]["median_ms"]
            cmb = r["combined_mvmr_plus_vvor"]
            print(f"[{stage} {dt_name}]")
            print(f"  mvmr  triton={m_t:7.3f}ms (IQR {r['triton_grouped_mvmr']['iqr_ms']:.3f})  "
                  f"cuda={m_c:7.3f}ms (IQR {r['cuda_grouped_mvmr']['iqr_ms']:.3f})  "
                  f"ratio cuda/triton={m_c/max(1e-6,m_t):.3f}x")
            print(f"  vvor  triton={v_t:7.3f}ms (IQR {r['triton_grouped_vvor']['iqr_ms']:.3f})  "
                  f"cuda={v_c:7.3f}ms (IQR {r['cuda_grouped_vvor']['iqr_ms']:.3f})  "
                  f"ratio cuda/triton={v_c/max(1e-6,v_t):.3f}x")
            v_c_re = r["cuda_grouped_vvor_resorted"]["median_ms"]
            sort_ovh = r["composite_sort_overhead"]["median_ms"]
            print(f"  vvor  cuda+composite-sort={v_c_re:7.3f}ms "
                  f"(IQR {r['cuda_grouped_vvor_resorted']['iqr_ms']:.3f})  "
                  f"ratio resorted/triton={v_c_re/max(1e-6,v_t):.3f}x  "
                  f"sort-overhead={sort_ovh:.3f}ms ({sort_ovh/max(1e-6,v_c_re)*100:.1f}% of resorted)")
            v_c_wmma = r["cuda_grouped_vvor_wmma"]["median_ms"]
            if v_c_wmma == v_c_wmma:  # not NaN (fp16/bf16)
                ratio_wmma = v_c_wmma / max(1e-6, v_t)
                pass_fail = (
                    "PASS Match" if ratio_wmma <= 1.15
                    else "AMBIGUOUS" if ratio_wmma <= 1.50
                    else "FAIL"
                )
                # Combined fwdbwd estimate with WMMA-direct vvor.
                v_c_wmma_iqr = r["cuda_grouped_vvor_wmma"]["iqr_ms"]
                combined_wmma_ms = m_c + v_c_wmma  # mvmr is cuda-FMA, vvor is wmma
                combined_wmma_ratio = combined_wmma_ms / max(1e-6, cmb["triton_ms"])
                print(f"  vvor  cuda-WMMA      ={v_c_wmma:7.3f}ms "
                      f"(IQR {v_c_wmma_iqr:.3f})  "
                      f"ratio wmma/triton={ratio_wmma:.3f}x  [{pass_fail} @ 1.15x/1.50x]")
                print(f"  combined (mvmr-cuda + vvor-wmma)={combined_wmma_ms:.3f}ms  "
                      f"ratio vs triton-combined={combined_wmma_ratio:.3f}x")
            else:
                print(f"  vvor  cuda-WMMA      = SKIPPED (fp32 routes to scalar-FMA fallback)")
            v_c_coop = r["cuda_grouped_vvor_wmma_coop"]["median_ms"]
            if v_c_coop == v_c_coop:  # not NaN
                ratio_coop = v_c_coop / max(1e-6, v_t)
                pf_coop = (
                    "BEAT" if ratio_coop <= 0.95
                    else "MATCH" if ratio_coop <= 1.10
                    else "PASS" if ratio_coop <= 1.50
                    else "FAIL"
                )
                v_c_coop_iqr = r["cuda_grouped_vvor_wmma_coop"]["iqr_ms"]
                print(f"  vvor  cuda-WMMA-coop ={v_c_coop:7.3f}ms "
                      f"(IQR {v_c_coop_iqr:.3f})  "
                      f"ratio coop/triton={ratio_coop:.3f}x  [{pf_coop} per §1.9a]")
            v_c_cut = r["cuda_grouped_vvor_cutlass"]["median_ms"]
            if v_c_cut == v_c_cut:  # not NaN (fp16 only)
                ratio_cut = v_c_cut / max(1e-6, v_t)
                # Pre-reg §2.2 decision bands (cycle-4 §1.11 G14).
                pf_cut = (
                    "BEAT" if ratio_cut <= 0.95
                    else "MATCH" if ratio_cut <= 1.00
                    else "PASS Match-band" if ratio_cut <= 1.15
                    else "AMBIGUOUS" if ratio_cut <= 1.50
                    else "FAIL"
                )
                v_c_cut_iqr = r["cuda_grouped_vvor_cutlass"]["iqr_ms"]
                print(f"  vvor  cuda-CUTLASS   ={v_c_cut:7.3f}ms "
                      f"(IQR {v_c_cut_iqr:.3f})  "
                      f"ratio cutlass/triton={ratio_cut:.3f}x  [{pf_cut} per §2.2]")
            else:
                print(f"  vvor  cuda-CUTLASS   = SKIPPED "
                      f"(Tier-2 CUTLASS vvor is fp16-only)")
            m_c_cut = r["cuda_grouped_mvmr_cutlass"]["median_ms"]
            if m_c_cut == m_c_cut:  # not NaN (fp16 only)
                ratio_m_cut = m_c_cut / max(1e-6, m_t)
                # Pre-reg §2.2 ENGINE bands (G22 plan §5; M5).
                pf_m_cut = (
                    "ENGINE-BEAT" if ratio_m_cut <= 0.95
                    else "ENGINE-MATCH" if ratio_m_cut <= 1.15
                    else "ENGINE-MISS"
                )
                m_c_cut_iqr = r["cuda_grouped_mvmr_cutlass"]["iqr_ms"]
                print(f"  mvmr  cuda-CUTLASS   ={m_c_cut:7.3f}ms "
                      f"(IQR {m_c_cut_iqr:.3f})  "
                      f"ratio cutlass/triton={ratio_m_cut:.3f}x  [{pf_m_cut} per §2.2]")
            else:
                print(f"  mvmr  cuda-CUTLASS   = SKIPPED "
                      f"(Tier-2 CUTLASS mvmr is fp16-only / tile constraint)")
            print(f"  mvmr+vvor combined  triton={cmb['triton_ms']:.3f}ms  "
                  f"cuda={cmb['cuda_ms']:.3f}ms  "
                  f"ratio cuda/triton={cmb['cuda_vs_triton_ratio']:.3f}x")
            print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"# wrote {args.json}")


if __name__ == "__main__":
    main()
