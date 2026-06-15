"""Realistic-data smoke for conv_with_stride_disjoint on voxelized ScanNet.

Mirrors the gating + dataset path from
test_radius_search_strided_realistic.py. Tests that the production-
format input path (Pointcept GridSample voxelization at 0.02 m) works
end-to-end through conv_with_stride_disjoint at a stride-4 no-overlap-clean cell.
Production training would not normally use stride=4 (PT-v3 uses
stride=2, which violates no-overlap), but stride=4 demonstrates the
algorithmic + integration correctness on real cloud distributions.
"""
import os
import numpy as np
import pytest
import torch

from layers import PointConv3d
from layers.conv import conv_with_stride_disjoint
from layers.metadata import MetaData


SCANNET_VAL = "/path/to/data"
DATASET_AVAILABLE = os.path.isdir(SCANNET_VAL)

pytestmark = [
    pytest.mark.skipif(
        not DATASET_AVAILABLE,
        reason=f"ScanNet v2 mirror not mounted at {SCANNET_VAL}",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"),
]


def _load_voxelized_scene(scene_id, voxel_size=0.02):
    path = os.path.join(SCANNET_VAL, scene_id, "coord.npy")
    if not os.path.exists(path):
        pytest.skip(f"scene fixture not found: {path}")
    raw = np.load(path)
    grid = (raw / voxel_size).round().astype(np.int64)
    keys = grid[:, 0] * 1_000_000 + grid[:, 1] * 1_000 + grid[:, 2]
    _, uniq_idx = np.unique(keys, return_index=True)
    voxelized = (grid[uniq_idx] * voxel_size).astype(np.float32)
    return torch.from_numpy(voxelized).cuda().float()


def test_conv_with_stride_disjoint_real_scannet_stride4_clean():
    """conv_with_stride_disjoint at stride=4/ks=3/ball/scaler=1.0 on a voxelized
    ScanNet val scene runs end-to-end without crash, and the output
    point count matches the standard conv_with_stride path."""
    coord = _load_voxelized_scene("scene0011_00")
    n_in = coord.shape[0]
    c_in, c_out = 16, 32
    sample_inds = torch.zeros(n_in, dtype=torch.long, device="cuda")
    sample_sizes = torch.bincount(sample_inds).long()
    feat = torch.randn(n_in, c_in, device="cuda")

    m_a = MetaData(points=coord.clone(), sample_inds=sample_inds.clone(),
                    sample_sizes=sample_sizes.clone(), grid_size=0.02)
    conv = PointConv3d(c_in, c_out, kernel_size=3, bias=True).cuda()

    out, m_out = conv_with_stride_disjoint(conv, feat, m_a, stride=4.0)

    assert out.shape[0] == m_out.points.shape[0], "output rows == output points"
    assert out.shape[1] == c_out
    assert m_out.points.shape[0] < n_in, "downsample reduced point count"

    # Compare output count to the standard path on the same scene.
    from layers.conv import conv_with_stride
    m_b = MetaData(points=coord.clone(), sample_inds=sample_inds.clone(),
                    sample_sizes=sample_sizes.clone(), grid_size=0.02)
    out_strd, m_strd = conv_with_stride(conv, feat.clone(), m_b, stride=4.0,
                                          receptive_field_scaler=1.0)

    assert m_out.points.shape[0] == m_strd.points.shape[0], (
        f"downsampled point count differs: disjoint={m_out.points.shape[0]} "
        f"strd={m_strd.points.shape[0]}"
    )

    print(f"\n  scene0011_00: n_in={n_in} → n_out={m_out.points.shape[0]} "
          f"(stride=4, c_in={c_in}, c_out={c_out})")
