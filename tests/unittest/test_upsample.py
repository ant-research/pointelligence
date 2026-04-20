"""Correctness tests for the Upsample layer.

Verifies:
1. Every high-res query point receives a non-zero output (full coverage).
2. With center-only kernel weights, output ≈ nearest low-res feature (geometric correctness).
3. straight_recover path also achieves full coverage and non-trivial output.
4. Gradients flow through to inputs and weights.
5. Isolated voxel with corner-clustered points (worst-case radius margin).
"""
import pytest
import torch

from layers.metadata import MetaData
from layers.upsample import Upsample
from layers.triplets import handle_stride_and_build_triplets
from internals.indexing import repeat_interleave_indices


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 12345


def _make_point_cloud(num_samples=2, num_points_per_sample=2000, grid_size=0.02):
    """Create a synthetic point cloud with known structure."""
    torch.manual_seed(SEED)
    points_list = []
    for _ in range(num_samples):
        pts = torch.rand(num_points_per_sample, 3, device=DEVICE) * 2.0
        points_list.append(pts)

    points = torch.cat(points_list, dim=0)
    sample_sizes = torch.tensor(
        [p.shape[0] for p in points_list], dtype=torch.int64, device=DEVICE
    )
    sample_inds = repeat_interleave_indices(
        repeats=sample_sizes,
        output_size=points.shape[0],
        may_contain_zero_repeats=False,
    )
    return points, sample_inds, sample_sizes, grid_size


def _downsample_to_metadata(points, sample_inds, sample_sizes, grid_size, stride=2.0):
    """Downsample and return low-res metadata with parent link to high-res."""
    m_high = MetaData(
        points=points,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        grid_size=grid_size,
    )
    m = handle_stride_and_build_triplets(m_high, stride=stride, kernel_size=(3, 3, 3))
    return m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestUpsample:
    def test_full_coverage(self):
        """Every high-res point must receive a non-zero output."""
        points, sample_inds, sample_sizes, grid_size = _make_point_cloud()
        m_low = _downsample_to_metadata(points, sample_inds, sample_sizes, grid_size)

        in_ch, out_ch = 32, 16
        upsample = Upsample(in_ch, out_ch, kernel_size=3, straight_recover=False).to(DEVICE)

        torch.manual_seed(SEED)
        x_low = torch.randn(m_low.points.shape[0], in_ch, device=DEVICE)

        with torch.no_grad():
            x_high, m_high = upsample(x_low, m_low)

        num_high = m_low.parent.points.shape[0]
        assert x_high.shape == (num_high, out_ch)
        zero_rows = (x_high.norm(dim=1) == 0).sum().item()
        assert zero_rows == 0, f"{zero_rows}/{num_high} high-res points got zero output"

    def test_center_kernel_approximates_nearest(self):
        """With only the center kernel weight active (k=13 for ks=3), the output
        for each high-res point should be close to the feature of its nearest
        low-res neighbor. This validates geometric correctness of the triplet
        construction."""
        points, sample_inds, sample_sizes, grid_size = _make_point_cloud(
            num_samples=1, num_points_per_sample=2000
        )
        m_low = _downsample_to_metadata(points, sample_inds, sample_sizes, grid_size)

        C = 16
        upsample = Upsample(C, C, kernel_size=3, bias=False, straight_recover=False).to(DEVICE)

        # Set weights: only the center kernel cell (index 13 for 3x3x3) gets identity,
        # all others get zero. This makes the conv act like nearest-neighbor lookup.
        # Weight shape: (in_channels=C, out_channels/groups=C, kernel_size=27)
        with torch.no_grad():
            upsample.conv.weight.zero_()
            center_k = 13  # (1,1,1) in 3x3x3
            for c in range(C):
                upsample.conv.weight[c, c, center_k] = 1.0

        # Use features = spatial coordinates of low-res points (known ground truth)
        # So we can verify the output is close to the nearest low-res point's coords.
        low_pts = m_low.points
        high_pts = m_low.parent.points

        # Pad or project coords to C channels (use first 3, rest zero)
        x_low = torch.zeros(low_pts.shape[0], C, device=DEVICE)
        x_low[:, :3] = low_pts

        with torch.no_grad():
            x_high, _ = upsample(x_low, m_low)

        # For each high-res point, find the actual nearest low-res point
        # (brute force, since we only have ~hundreds of low-res points)
        # We check per-sample to respect sample boundaries.
        high_sample_inds = m_low.parent.sample_inds
        low_sample_inds = m_low.sample_inds

        diffs = []
        for s in range(m_low.parent.sample_sizes.shape[0]):
            hi_mask = high_sample_inds == s
            lo_mask = low_sample_inds == s
            hi_pts = high_pts[hi_mask]
            lo_pts = low_pts[lo_mask]

            # pairwise distance: [num_high_in_sample, num_low_in_sample]
            dist = torch.cdist(hi_pts.unsqueeze(0), lo_pts.unsqueeze(0)).squeeze(0)
            nearest_idx = dist.argmin(dim=1)
            nearest_coords = lo_pts[nearest_idx]

            # The output xyz channels should be close to the nearest low-res coords
            output_coords = x_high[hi_mask, :3]
            diff = (output_coords - nearest_coords).abs()
            diffs.append(diff)

        all_diffs = torch.cat(diffs, dim=0)
        mean_diff = all_diffs.mean().item()
        max_diff = all_diffs.max().item()

        # The center kernel only fires for neighbors at offset (0,0,0), i.e., in the
        # same voxel cell. The nearest low-res point might not be exactly that one,
        # but for the vast majority it should be very close (within a grid cell).
        # We check that the mean error is small relative to the grid size.
        grid_size_low = m_low.grid_size
        assert mean_diff < grid_size_low * 2, (
            f"Center-kernel output deviates too much from nearest low-res point: "
            f"mean_diff={mean_diff:.4f}, grid_size_low={grid_size_low:.4f}"
        )

    def test_straight_recover_full_coverage(self):
        """The cached triplet path (straight_recover=True) must also give full
        coverage — every high-res point receives non-zero output."""
        points, sample_inds, sample_sizes, grid_size = _make_point_cloud()
        m_low = _downsample_to_metadata(points, sample_inds, sample_sizes, grid_size)

        in_ch, out_ch = 32, 16
        upsample = Upsample(
            in_ch, out_ch, kernel_size=3, straight_recover=True
        ).to(DEVICE)

        torch.manual_seed(SEED)
        x_low = torch.randn(m_low.points.shape[0], in_ch, device=DEVICE)

        with torch.no_grad():
            x_high, m_high = upsample(x_low, m_low)

        num_high = m_low.parent.points.shape[0]
        assert x_high.shape == (num_high, out_ch)
        zero_rows = (x_high.norm(dim=1) == 0).sum().item()
        assert zero_rows == 0, f"straight_recover: {zero_rows}/{num_high} high-res points got zero output"

    def test_gradient_flow(self):
        """Gradients must flow through upsample back to input features and weights."""
        points, sample_inds, sample_sizes, grid_size = _make_point_cloud(
            num_samples=1, num_points_per_sample=1000
        )
        m_low = _downsample_to_metadata(points, sample_inds, sample_sizes, grid_size)

        in_ch, out_ch = 16, 8
        upsample = Upsample(in_ch, out_ch, kernel_size=3, straight_recover=False).to(DEVICE)

        x_low = torch.randn(
            m_low.points.shape[0], in_ch, device=DEVICE, requires_grad=True
        )

        x_high, _ = upsample(x_low, m_low)
        loss = x_high.sum()
        loss.backward()

        assert x_low.grad is not None, "No gradient on input features"
        assert x_low.grad.abs().sum() > 0, "Input gradient is all zeros"

        for name, param in upsample.named_parameters():
            assert param.grad is not None, f"No gradient on {name}"
            assert param.grad.abs().sum() > 0, f"Gradient on {name} is all zeros"

    def test_deterministic(self):
        """Same input must produce identical output across two runs."""
        points, sample_inds, sample_sizes, grid_size = _make_point_cloud()
        m_low = _downsample_to_metadata(points, sample_inds, sample_sizes, grid_size)

        in_ch, out_ch = 16, 8
        torch.manual_seed(SEED + 99)
        upsample = Upsample(in_ch, out_ch, kernel_size=3, straight_recover=False).to(DEVICE)

        x_low = torch.randn(m_low.points.shape[0], in_ch, device=DEVICE)

        with torch.no_grad():
            out1, _ = upsample(x_low, m_low)
            out2, _ = upsample(x_low, m_low)

        # CUDA reductions (scatter-add) are non-deterministic at float precision,
        # so we check near-equality rather than bitwise equality.
        assert torch.allclose(out1, out2, atol=1e-5), (
            f"Output differs beyond float tolerance: max_diff="
            f"{(out1 - out2).abs().max().item():.2e}"
        )


    def test_isolated_voxel_corner_case(self):
        """Worst-case geometry: isolated voxel where center_nearest picks a
        corner point, and a high-res query sits at the opposite corner.

        This tests the tightest radius margin (~7% of grid_size_low).
        The search radius must still cover the full voxel diagonal."""
        grid_size_high = 0.1
        stride = 2.0
        grid_size_low = grid_size_high * stride  # 0.2
        eps = 0.001

        # Voxel [0, 0.2)^3: all source points near corner (0,0,0)
        # center_nearest will pick from these — all far from the opposite corner
        corner_cluster = torch.tensor([
            [eps, eps, eps],
            [2*eps, eps, eps],
            [eps, 2*eps, eps],
        ], device=DEVICE, dtype=torch.float32)

        # High-res query at the opposite corner of the SAME voxel
        opposite_corner = torch.tensor(
            [[grid_size_low - eps] * 3], device=DEVICE, dtype=torch.float32
        )

        # A distant isolated voxel (so no cross-voxel neighbor rescue)
        far_voxel = torch.tensor(
            [[10 * grid_size_low + eps] * 3], device=DEVICE, dtype=torch.float32
        )

        points = torch.cat([corner_cluster, opposite_corner, far_voxel])
        n = points.shape[0]
        sample_sizes = torch.tensor([n], dtype=torch.int64, device=DEVICE)
        sample_inds = torch.zeros(n, dtype=torch.int64, device=DEVICE)

        m_low = _downsample_to_metadata(
            points, sample_inds, sample_sizes, grid_size_high, stride=stride
        )

        # Verify the geometry: the opposite corner point should be far from
        # its same-voxel low-res representative
        opp_pt = opposite_corner[0]
        dists = torch.norm(m_low.points - opp_pt.unsqueeze(0), dim=1)
        min_dist = dists.min().item()
        import math
        voxel_diag = math.sqrt(3) * grid_size_low
        assert min_dist > 0.9 * voxel_diag, (
            f"Test setup error: opposite corner should be ~{voxel_diag:.4f} from "
            f"low-res point, got {min_dist:.4f}"
        )

        # The actual test: upsample must succeed with full coverage
        in_ch, out_ch = 8, 4
        upsample = Upsample(
            in_ch, out_ch, kernel_size=3, straight_recover=False
        ).to(DEVICE)
        x_low = torch.randn(m_low.points.shape[0], in_ch, device=DEVICE)

        with torch.no_grad():
            x_high, _ = upsample(x_low, m_low)

        zero_rows = (x_high.norm(dim=1) == 0).sum().item()
        assert zero_rows == 0, (
            f"Isolated voxel corner case: {zero_rows} high-res points got zero output. "
            f"min_dist={min_dist:.4f}, voxel_diag={voxel_diag:.4f}"
        )

    def test_small_receptive_field_warns(self):
        """receptive_field_scaler < ~0.81 makes radius < voxel diagonal.
        The warning must fire."""
        points, sample_inds, sample_sizes, grid_size = _make_point_cloud(
            num_samples=1, num_points_per_sample=500
        )
        m_low = _downsample_to_metadata(points, sample_inds, sample_sizes, grid_size)

        in_ch, out_ch = 8, 4
        upsample = Upsample(
            in_ch, out_ch, kernel_size=3,
            receptive_field_scaler=0.5,  # well below the 0.81 threshold
            straight_recover=False,
        ).to(DEVICE)

        x_low = torch.randn(m_low.points.shape[0], in_ch, device=DEVICE)

        with pytest.warns(UserWarning, match="smaller than the worst-case voxel diagonal"):
            with torch.no_grad():
                upsample(x_low, m_low)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
