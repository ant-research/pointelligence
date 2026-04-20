"""Verify PointConv3d grouped backward correctness, triplet equivalence,
and ResNet end-to-end determinism.

Covers:
- Grouped backward (groups>1): weight/input grads correct shape and non-zero
- fp16 backward: runs without error, produces non-zero grads
- Triplet equivalence: dirty_triplets + handle_stride == inline rebuild
- ResNet18/50 forward/backward determinism and multi-batch
"""

import sys
import torch
import torch.nn as nn
from functools import partial

sys.path.insert(0, ".")

from layers.conv import PointConv3d, GeneralConv, conv_with_stride
from layers.metadata import MetaData
from layers.triplets import (
    build_triplets,
    handle_stride_and_build_triplets,
    voxelize_3d,
    radius_scaler_for_kernel_size,
)


# ─── Helpers ───

def _old_inline_triplet_rebuild(m):
    """The old BasicBlock inline triplet rebuild."""
    neighbor_radius = m.grid_size * radius_scaler_for_kernel_size(kernel_size=3)
    i, j, k, _ = build_triplets(
        points=m.points,
        sample_inds=m.sample_inds,
        sample_sizes=m.sample_sizes,
        neighbor_radius=neighbor_radius,
        kernel_indexer=partial(voxelize_3d, kernel_size=3),
        radius_scaler=radius_scaler_for_kernel_size(kernel_size=3),
        return_num_neighbors=False,
    )
    return i, j, k


def _make_metadata(points, sample_inds, sample_sizes, grid_size):
    return MetaData(
        points=points.clone(),
        sample_inds=sample_inds.clone(),
        sample_sizes=sample_sizes.clone(),
        grid_size=grid_size,
    )


# ─── Grouped backward tests ───

def test_grouped_backward():
    """Grouped backward: verify forward+backward runs correctly for groups>1.

    Checks that:
    - Forward+backward runs without error
    - Weight grad has correct shape and is non-zero
    - Input grad has correct shape and is non-zero
    - Gradients are deterministic (same input -> same grad)
    """
    torch.manual_seed(42)
    device = "cuda"

    for C_in, C_out, groups in [(64, 64, 4), (128, 128, 8), (32, 32, 32), (64, 128, 2)]:
        K = 27
        conv = PointConv3d(C_in, C_out, kernel_size=3, groups=groups, bias=True).to(device)

        N = 200
        points = torch.rand(N, 3, device=device)
        sample_inds = torch.zeros(N, dtype=torch.long, device=device)
        sample_sizes = torch.tensor([N], device=device)
        rs = radius_scaler_for_kernel_size(kernel_size=3)
        i, j, k, _ = build_triplets(
            points=points, sample_inds=sample_inds, sample_sizes=sample_sizes,
            neighbor_radius=(1 / 512) * rs,
            kernel_indexer=partial(voxelize_3d, kernel_size=3),
            radius_scaler=rs,
        )

        x = torch.randn(N, C_in, device=device, requires_grad=True)
        out = conv(x, i, j, k, N)
        loss = out.sum()
        loss.backward()

        assert conv.weight.grad is not None, f"No weight grad for groups={groups}"
        assert conv.weight.grad.shape == conv.weight.shape, \
            f"Wrong grad_w shape: {conv.weight.grad.shape}"
        assert conv.weight.grad.abs().max() > 0, f"Zero weight grad for groups={groups}"

        assert x.grad is not None, f"No input grad for groups={groups}"
        assert x.grad.shape == (N, C_in), f"Wrong grad_x shape: {x.grad.shape}"
        assert x.grad.abs().max() > 0, f"Zero input grad for groups={groups}"

        assert conv.bias.grad is not None and conv.bias.grad.shape == (C_out,)

        # Determinism check: run again with fresh tensors, compare
        grad_x_1 = x.grad.clone()
        grad_w_1 = conv.weight.grad.clone()
        conv.zero_grad()
        x2 = x.detach().clone().requires_grad_(True)
        out2 = conv(x2, i, j, k, N)
        out2.sum().backward()
        dx_det = (grad_x_1 - x2.grad).abs().max().item()
        dw_det = (grad_w_1 - conv.weight.grad).abs().max().item()
        # atomic_add in VVOR/MVMR kernels is non-deterministic at fp32 level
        assert dx_det < 1e-5, f"Input grad not deterministic for groups={groups}: diff={dx_det}"
        assert dw_det < 1e-4, f"Weight grad not deterministic for groups={groups}: diff={dw_det}"

        print(f"  [PASS] grouped backward groups={groups:>2}, C_in={C_in:>3}, C_out={C_out:>3}: "
              f"grad_w max={conv.weight.grad.abs().max():.2e}, grad_x max={x.grad.abs().max():.2e}")


def test_backward_fp16():
    """fp16 backward for both grouped and ungrouped."""
    torch.manual_seed(42)
    device = "cuda"

    for C_in, C_out, groups in [(64, 128, 1), (64, 64, 4)]:
        conv = PointConv3d(C_in, C_out, kernel_size=3, groups=groups, bias=True).to(device, dtype=torch.float16)

        N = 200
        points = torch.rand(N, 3, device=device)
        sample_inds = torch.zeros(N, dtype=torch.long, device=device)
        sample_sizes = torch.tensor([N], device=device)
        rs = radius_scaler_for_kernel_size(kernel_size=3)
        i, j, k, _ = build_triplets(
            points=points, sample_inds=sample_inds, sample_sizes=sample_sizes,
            neighbor_radius=(1 / 512) * rs,
            kernel_indexer=partial(voxelize_3d, kernel_size=3),
            radius_scaler=rs,
        )

        x = torch.randn(N, C_in, device=device, dtype=torch.float16, requires_grad=True)
        out = conv(x, i, j, k, N)
        out.sum().backward()

        assert conv.weight.grad is not None and conv.weight.grad.abs().max() > 0
        assert x.grad is not None and x.grad.abs().max() > 0

        print(f"  [PASS] fp16 backward groups={groups}, C_in={C_in}, C_out={C_out}")


# ─── Triplet equivalence tests ───

def test_triplet_equivalence_multi_stride():
    """Multiple sequential strides: stride=2 -> stride=2 (like ResNet layer2->layer3)."""
    torch.manual_seed(42)
    device = "cuda"

    N = 2000
    points = torch.rand(N, 3, device=device)
    sample_inds = torch.zeros(N, dtype=torch.long, device=device)
    sample_sizes = torch.tensor([N], device=device)
    grid_size = 1 / 512

    for stride_sequence in [(2,), (2, 2), (2, 2, 2)]:
        m_old = _make_metadata(points, sample_inds, sample_sizes, grid_size)
        m_new = _make_metadata(points, sample_inds, sample_sizes, grid_size)

        for s in stride_sequence:
            # Both paths: downsample
            m_old = handle_stride_and_build_triplets(m_old, stride=s, kernel_size=(3, 3, 3))
            m_new = handle_stride_and_build_triplets(m_new, stride=s, kernel_size=(3, 3, 3))

            # Old: inline rebuild
            i_old, j_old, k_old = _old_inline_triplet_rebuild(m_old)

            # New: dirty + handle_stride(stride=1)
            m_new.dirty_triplets()
            m_new = handle_stride_and_build_triplets(m_new, stride=1, kernel_size=(3, 3, 3))

            assert torch.equal(m_old.points, m_new.points), f"Points differ at stride={s}!"
            assert m_old.grid_size == m_new.grid_size, f"Grid size differs at stride={s}!"
            assert torch.equal(i_old, m_new.i), f"i differs at stride={s}!"
            assert torch.equal(j_old, m_new.j), f"j differs at stride={s}!"
            assert torch.equal(k_old, m_new.k), f"k differs at stride={s}!"

            # Also update m_old's triplets so next stride sees same state
            m_old.i, m_old.j, m_old.k = i_old, j_old, k_old

        strides_str = "->".join(str(s) for s in stride_sequence)
        print(f"  [PASS] stride sequence [{strides_str}]: "
              f"{m_new.num_points()} pts, {m_new.i.shape[0]} triplets")


def test_triplet_equivalence_multi_batch():
    """Multi-sample batch with unequal sizes."""
    torch.manual_seed(42)
    device = "cuda"

    sizes = [500, 200, 1000, 50]
    N = sum(sizes)
    points = torch.rand(N, 3, device=device)
    sample_sizes = torch.tensor(sizes, device=device)
    sample_inds = torch.repeat_interleave(
        torch.arange(len(sizes), device=device), sample_sizes
    )
    grid_size = 1 / 256

    m_old = _make_metadata(points, sample_inds, sample_sizes, grid_size)
    m_new = _make_metadata(points, sample_inds, sample_sizes, grid_size)

    # Stride=2 downsample
    m_old = handle_stride_and_build_triplets(m_old, stride=2, kernel_size=(3, 3, 3))
    m_new = handle_stride_and_build_triplets(m_new, stride=2, kernel_size=(3, 3, 3))

    # Old path
    i_old, j_old, k_old = _old_inline_triplet_rebuild(m_old)

    # New path
    m_new.dirty_triplets()
    m_new = handle_stride_and_build_triplets(m_new, stride=1, kernel_size=(3, 3, 3))

    assert torch.equal(i_old, m_new.i), "i differs (multi-batch)!"
    assert torch.equal(j_old, m_new.j), "j differs (multi-batch)!"
    assert torch.equal(k_old, m_new.k), "k differs (multi-batch)!"

    print(f"  [PASS] multi-batch {sizes}: {m_new.num_points()} pts, {m_new.i.shape[0]} triplets")


def test_triplet_equivalence_tiny_pointcloud():
    """Edge case: very few points (some may have 0 neighbors at coarse grid)."""
    torch.manual_seed(42)
    device = "cuda"

    for N in [10, 25, 50]:
        points = torch.rand(N, 3, device=device) * 0.01  # tightly clustered
        sample_inds = torch.zeros(N, dtype=torch.long, device=device)
        sample_sizes = torch.tensor([N], device=device)
        grid_size = 0.005

        m_old = _make_metadata(points, sample_inds, sample_sizes, grid_size)
        m_new = _make_metadata(points, sample_inds, sample_sizes, grid_size)

        m_old = handle_stride_and_build_triplets(m_old, stride=2, kernel_size=(3, 3, 3))
        m_new = handle_stride_and_build_triplets(m_new, stride=2, kernel_size=(3, 3, 3))

        if m_old.num_points() == 0:
            print(f"  [SKIP] N={N}: all points collapsed to 0 after stride=2")
            continue

        i_old, j_old, k_old = _old_inline_triplet_rebuild(m_old)

        m_new.dirty_triplets()
        m_new = handle_stride_and_build_triplets(m_new, stride=1, kernel_size=(3, 3, 3))

        assert torch.equal(i_old, m_new.i), f"i differs (N={N})!"
        assert torch.equal(j_old, m_new.j), f"j differs (N={N})!"
        assert torch.equal(k_old, m_new.k), f"k differs (N={N})!"

        print(f"  [PASS] tiny N={N}: {m_new.num_points()} pts after stride, {m_new.i.shape[0]} triplets")


# ─── Full ResNet end-to-end ───

def test_resnet18_forward_deterministic():
    """Full ResNet18 forward: same input + same weights => same output.

    Run the model twice with same seed to verify determinism, which
    indirectly validates that the BasicBlock path is stable.
    """
    from models import resnet18

    torch.manual_seed(42)
    device = "cuda"

    N = 500
    C_in = 3
    points = torch.rand(N, 3, device=device)
    sample_sizes = torch.tensor([N], device=device)
    x = torch.randn(N, C_in, device=device)
    grid_size = 1 / 256

    model = resnet18(in_channels=C_in).to(device)
    model.eval()

    with torch.no_grad():
        out1 = model(x, points, sample_sizes, grid_size)
        out2 = model(x, points, sample_sizes, grid_size)

    diff = (out1 - out2).abs().max().item()
    # atomic_add non-determinism in Triton kernels causes small fp differences
    assert diff < 1e-4, f"ResNet18 not deterministic! diff={diff}"
    print(f"  [PASS] ResNet18 deterministic: diff={diff:.2e}, shape={out1.shape}")


def test_resnet18_backward_deterministic():
    """Full ResNet18 forward+backward: verify gradients are deterministic."""
    from models import resnet18

    torch.manual_seed(42)
    device = "cuda"

    N = 500
    C_in = 3
    points = torch.rand(N, 3, device=device)
    sample_sizes = torch.tensor([N], device=device)
    grid_size = 1 / 256

    model = resnet18(in_channels=C_in).to(device)
    x_data = torch.randn(N, C_in, device=device)

    def run_fwd_bwd():
        model.zero_grad()
        x = x_data.clone().requires_grad_(True)
        out = model(x, points, sample_sizes, grid_size)
        loss = out.sum()
        loss.backward()
        return x.grad.clone(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    grads1_x, grads1_w = run_fwd_bwd()
    grads2_x, grads2_w = run_fwd_bwd()

    dx_diff = (grads1_x - grads2_x).abs().max().item()
    assert dx_diff < 1e-4, f"Input grad not deterministic! diff={dx_diff}"

    max_param_diff = 0.0
    for name in grads1_w:
        diff = (grads1_w[name] - grads2_w[name]).abs().max().item()
        max_param_diff = max(max_param_diff, diff)
        assert diff < 1e-3, f"Param {name} grad not deterministic! diff={diff}"

    print(f"  [PASS] ResNet18 backward deterministic: {len(grads1_w)} param grads, "
          f"max dx={dx_diff:.2e}, max dw={max_param_diff:.2e}")


def test_resnet18_multi_batch():
    """ResNet18 with multi-sample batch of varying sizes."""
    from models import resnet18

    torch.manual_seed(42)
    device = "cuda"

    sizes = [300, 150, 500]
    N = sum(sizes)
    C_in = 3
    points = torch.rand(N, 3, device=device)
    sample_sizes = torch.tensor(sizes, device=device)
    x = torch.randn(N, C_in, device=device, requires_grad=True)
    grid_size = 1 / 256

    model = resnet18(in_channels=C_in).to(device)

    out = model(x, points, sample_sizes, grid_size)
    assert out.shape == (len(sizes), 1000), f"Wrong output shape: {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == (N, C_in)

    print(f"  [PASS] ResNet18 multi-batch {sizes}: output={out.shape}, grad={x.grad.shape}")


def test_resnet50_forward_backward():
    """ResNet50 uses Bottleneck — verify forward+backward runs cleanly."""
    from models import resnet50

    torch.manual_seed(42)
    device = "cuda"

    N = 300
    C_in = 3
    points = torch.rand(N, 3, device=device)
    sample_sizes = torch.tensor([N], device=device)
    x = torch.randn(N, C_in, device=device, requires_grad=True)
    grid_size = 1 / 256

    model = resnet50(in_channels=C_in).to(device)
    out = model(x, points, sample_sizes, grid_size)
    loss = out.sum()
    loss.backward()

    assert out.shape == (1, 1000)
    assert x.grad is not None
    print(f"  [PASS] ResNet50 forward+backward: output={out.shape}")


if __name__ == "__main__":
    print("=== Grouped backward ===")
    test_grouped_backward()

    print("\n=== fp16 backward ===")
    test_backward_fp16()

    print("\n=== Triplet: sequential strides ===")
    test_triplet_equivalence_multi_stride()

    print("\n=== Triplet: multi-batch ===")
    test_triplet_equivalence_multi_batch()

    print("\n=== Triplet: tiny point clouds ===")
    test_triplet_equivalence_tiny_pointcloud()

    print("\n=== ResNet18: forward determinism ===")
    test_resnet18_forward_deterministic()

    print("\n=== ResNet18: backward determinism ===")
    test_resnet18_backward_deterministic()

    print("\n=== ResNet18: multi-batch ===")
    test_resnet18_multi_batch()

    print("\n=== ResNet50: forward+backward ===")
    test_resnet50_forward_backward()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
