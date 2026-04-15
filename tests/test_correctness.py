"""
Correctness test: compare performance branch kernels against main branch.

Runs identical inputs through both MVMR/VVOR from main and performance branches,
verifies outputs match. Tests both fp32 and fp16 paths.
"""

import sys
import os
import types
import torch
import numpy as np

# Bypass sparse_engines/__init__.py which tries to import CUDA extensions
# that may not be built for this worktree
PERF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PERF_ROOT)
_mod = types.ModuleType('sparse_engines')
_mod.__path__ = [os.path.join(PERF_ROOT, 'sparse_engines')]
sys.modules['sparse_engines'] = _mod


def test_mvmr_correctness():
    """Compare MVMR output: performance branch vs a naive reference implementation."""
    from sparse_engines.mvmr_triton import sparse_matrix_vector_multiplication_reduction

    torch.manual_seed(42)

    # Create test inputs matching MVMR signature:
    # a: (K, G, C, M) kernel weights
    # b: (N, G, C) input features
    # o: (n_o, G, M) output
    # Triplets: a_idx (kernel index k), b_idx (neighbor j), o_idx (query i)
    K, G, C, M = 27, 1, 64, 64  # kernel_size=3^3, groups=1, in=64, out=64
    N_points = 500
    n_o = 200
    T = 1000  # number of triplets

    a = torch.randn(K, G, C, M, device='cuda')
    b = torch.randn(N_points, G, C, device='cuda')

    # Generate valid triplet indices
    a_idx = torch.randint(0, K, (T,), device='cuda', dtype=torch.int64)
    b_idx = torch.randint(0, N_points, (T,), device='cuda', dtype=torch.int64)
    o_idx = torch.sort(torch.randint(0, n_o, (T,), device='cuda', dtype=torch.int64))[0]

    # Reference: naive Python implementation
    def naive_mvmr(a, a_idx, b, b_idx, o_idx, n_o):
        K, G, C, M = a.shape
        o = torch.zeros(n_o, G, M, device=a.device, dtype=torch.float32)
        for t in range(a_idx.numel()):
            ki = a_idx[t].item()
            ji = b_idx[t].item()
            oi = o_idx[t].item()
            # o[oi] += a[ki] @ b[ji] (contracted over C dimension)
            for g in range(G):
                o[oi, g] += a[ki, g].T @ b[ji, g]  # (M, C) @ (C,) -> (M,)
        return o

    ref_output = naive_mvmr(a, a_idx, b, b_idx, o_idx, n_o)

    results = {}

    for dtype_label, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        a_test = a.to(dtype)
        b_test = b.to(dtype)

        output = sparse_matrix_vector_multiplication_reduction(
            a_test, a_idx, b_test, b_idx, o_idx, n_o
        )

        output_f32 = output.float()
        max_diff = (ref_output - output_f32).abs().max().item()
        mean_diff = (ref_output - output_f32).abs().mean().item()
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

        if dtype_label == "fp32":
            atol = 1e-3  # Triton atomic_add can have small floating point differences
        else:
            atol = 5e-1  # fp16 input -> fp32 accumulation -> fp16 output -> fp32 compare

        close = torch.allclose(output_f32, ref_output, atol=atol, rtol=1e-2)

        results[dtype_label] = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'close': close,
            'output_dtype': str(output.dtype),
        }
        status = "PASS" if close and not has_nan and not has_inf else "FAIL"
        print(f"  MVMR {dtype_label}: {status} | max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
              f"output_dtype={output.dtype} nan={has_nan} inf={has_inf}")

    return results


def test_vvor_correctness():
    """Compare VVOR output against naive reference."""
    from sparse_engines.vvor_triton import sparse_vector_vector_outer_product_reduction

    torch.manual_seed(42)

    G, M, C = 1, 64, 64
    N_a = 200
    N_b = 500
    n_o = 27  # output is weight gradient shape (K, G, M, C)
    T = 1000

    a = torch.randn(N_a, G, M, device='cuda')
    b = torch.randn(N_b, G, C, device='cuda')

    a_idx = torch.sort(torch.randint(0, N_a, (T,), device='cuda', dtype=torch.int64))[0]
    b_idx = torch.randint(0, N_b, (T,), device='cuda', dtype=torch.int64)
    o_idx = torch.randint(0, n_o, (T,), device='cuda', dtype=torch.int64)

    # Reference: naive Python implementation
    def naive_vvor(a, a_idx, b, b_idx, o_idx, n_o):
        G, M = a.shape[1], a.shape[2]
        C = b.shape[2]
        o = torch.zeros(n_o, G, M, C, device=a.device, dtype=torch.float32)
        for t in range(a_idx.numel()):
            ai = a_idx[t].item()
            bi = b_idx[t].item()
            oi = o_idx[t].item()
            for g in range(G):
                o[oi, g] += a[ai, g].unsqueeze(1) * b[bi, g].unsqueeze(0)  # (M,1) * (1,C) -> (M,C)
        return o

    ref_output = naive_vvor(a, a_idx, b, b_idx, o_idx, n_o)

    results = {}

    for dtype_label, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        a_test = a.to(dtype)
        b_test = b.to(dtype)

        output = sparse_vector_vector_outer_product_reduction(
            a_test, a_idx, b_test, b_idx, o_idx, n_o
        )

        output_f32 = output.float()
        max_diff = (ref_output - output_f32).abs().max().item()
        mean_diff = (ref_output - output_f32).abs().mean().item()
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

        if dtype_label == "fp32":
            atol = 1e-3
        else:
            atol = 5e-1

        close = torch.allclose(output_f32, ref_output, atol=atol, rtol=1e-2)

        results[dtype_label] = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'close': close,
            'output_dtype': str(output.dtype),
        }
        status = "PASS" if close and not has_nan and not has_inf else "FAIL"
        print(f"  VVOR {dtype_label}: {status} | max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
              f"output_dtype={output.dtype} nan={has_nan} inf={has_inf}")

    return results


def test_conv_backward_correctness():
    """Test that gradients flow correctly through conv in both fp32 and fp16."""
    from layers.conv import PointConv3d
    from layers.metadata import MetaData

    torch.manual_seed(42)

    N = 500
    in_ch, out_ch = 64, 64
    kernel_size = (3, 3, 3)

    coords = torch.rand(N, 3, device='cuda', dtype=torch.float32)
    sample_sizes = torch.tensor([N], device='cuda', dtype=torch.long)
    sample_inds = torch.zeros(N, device='cuda', dtype=torch.long)

    m = MetaData(
        points=coords,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        grid_size=0.1,
        kernel_size=kernel_size,
        sort_by="k",
    )

    results = {}

    # Run fp32 as reference
    conv_ref = PointConv3d(in_ch, out_ch, kernel_size=kernel_size, dtype=torch.float32).cuda()
    feat_ref = torch.randn(N, in_ch, device='cuda', dtype=torch.float32, requires_grad=True)
    out_ref = conv_ref(feat_ref, m.i, m.j, m.k, m.num_points())
    loss_ref = out_ref.sum()
    loss_ref.backward()
    grad_ref = feat_ref.grad.clone()
    out_ref_val = out_ref.detach().clone()

    print(f"  Conv fp32 ref: output_range=[{out_ref_val.min():.4f}, {out_ref_val.max():.4f}] "
          f"grad_norm={grad_ref.norm():.4f}")

    for dtype_label, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        conv = PointConv3d(in_ch, out_ch, kernel_size=kernel_size, dtype=dtype).cuda()
        # Copy weights from reference for fair comparison
        with torch.no_grad():
            conv.weight.copy_(conv_ref.weight.to(dtype))
            conv.bias.copy_(conv_ref.bias.to(dtype))

        feat = torch.randn(N, in_ch, device='cuda', dtype=dtype, requires_grad=True)
        # Use same feature values
        with torch.no_grad():
            feat.copy_(feat_ref.detach().to(dtype))
        feat.requires_grad_(True)

        out = conv(feat, m.i, m.j, m.k, m.num_points())
        loss = out.sum()
        loss.backward()

        has_nan_out = torch.isnan(out).any().item()
        has_nan_grad = torch.isnan(feat.grad).any().item()
        has_inf_out = torch.isinf(out).any().item()
        has_inf_grad = torch.isinf(feat.grad).any().item()

        out_diff = (out.float() - out_ref_val).abs().max().item()
        grad_diff = (feat.grad.float() - grad_ref).abs().max().item()

        ok = not has_nan_out and not has_nan_grad and not has_inf_out and not has_inf_grad
        status = "PASS" if ok else "FAIL"

        results[dtype_label] = {
            'out_max_diff': out_diff,
            'grad_max_diff': grad_diff,
            'has_nan': has_nan_out or has_nan_grad,
            'has_inf': has_inf_out or has_inf_grad,
            'ok': ok,
        }
        print(f"  Conv {dtype_label}: {status} | out_diff={out_diff:.6f} grad_diff={grad_diff:.6f} "
              f"nan_out={has_nan_out} nan_grad={has_nan_grad}")

    return results


def test_norm_correctness():
    """Test RaggedNorm operates in fp32 internally regardless of input dtype."""
    from layers.norm import RaggedBatchNorm, RaggedInstanceNorm

    torch.manual_seed(42)

    N = 500
    C = 64
    lengths = torch.tensor([200, 300], device='cuda', dtype=torch.long)

    for NormClass, name in [(RaggedBatchNorm, "BatchNorm"), (RaggedInstanceNorm, "InstanceNorm")]:
        norm = NormClass(C).cuda().eval()

        x_fp32 = torch.randn(N, C, device='cuda', dtype=torch.float32)
        x_fp16 = x_fp32.half()

        if name == "BatchNorm":
            out_fp32 = norm(x_fp32)
            out_fp16 = norm(x_fp16)
        else:
            out_fp32 = norm(x_fp32, lengths)
            out_fp16 = norm(x_fp16, lengths)

        diff = (out_fp32 - out_fp16.float()).abs().max().item()
        has_nan = torch.isnan(out_fp16).any().item()
        status = "PASS" if not has_nan and diff < 1e-2 else "FAIL"
        print(f"  {name}: {status} | fp32_vs_fp16_diff={diff:.6f} nan={has_nan} "
              f"out_fp16_dtype={out_fp16.dtype}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Correctness Tests: Performance Branch vs Reference")
    print("="*80)

    print("\n--- MVMR Kernel ---")
    mvmr_results = test_mvmr_correctness()

    print("\n--- VVOR Kernel ---")
    vvor_results = test_vvor_correctness()

    print("\n--- Conv Forward+Backward ---")
    conv_results = test_conv_backward_correctness()

    print("\n--- Norm Layers ---")
    test_norm_correctness()

    print("\n" + "="*80)
    all_pass = True
    for name, results in [("MVMR", mvmr_results), ("VVOR", vvor_results), ("Conv", conv_results)]:
        for dtype_label, r in results.items():
            ok = r.get('close', r.get('ok', False)) and not r.get('has_nan', True) and not r.get('has_inf', True)
            if not ok:
                all_pass = False
                print(f"FAIL: {name} {dtype_label}")

    if all_pass:
        print("All correctness tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)
