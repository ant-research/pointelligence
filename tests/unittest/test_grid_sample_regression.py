import torch

from internals.grid_sample import grid_sample_filter


def test_batched_center_reduction_returns_xyz_and_preserves_batch_isolation():
    points = torch.tensor(
        [
            [-0.90, -0.80, -0.70],
            [-0.10, -0.20, -0.30],
            [0.10, 0.20, 0.30],
            [-0.75, -0.65, -0.55],
            [-0.25, -0.35, -0.45],
        ],
        dtype=torch.float32,
    )
    sample_inds = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)

    centers, center_batches, indices, mapping = grid_sample_filter(
        points,
        grid_size=1.0,
        sample_inds=sample_inds,
        reduction="center",
        return_mapping=True,
    )

    expected_centers = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    assert centers.shape == (3, 3)
    assert torch.equal(centers, expected_centers)
    assert torch.equal(center_batches, torch.tensor([0, 0, 1]))
    assert torch.equal(sample_inds[indices], center_batches)
    assert torch.equal(mapping, torch.tensor([0, 0, 1, 2, 2]))


def test_center_reduction_uses_half_open_cells_across_zero():
    eps = 1e-4
    points = torch.tensor(
        [
            [-eps, 0.25, 0.25],
            [-0.75, 0.25, 0.25],
            [eps, 0.25, 0.25],
        ],
        dtype=torch.float32,
    )

    centers, _, _, mapping = grid_sample_filter(
        points,
        grid_size=1.0,
        reduction="center",
        return_mapping=True,
    )

    # The two negative-x points share [-1, 0); crossing zero enters [0, 1).
    assert torch.equal(
        centers,
        torch.tensor(
            [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32
        ),
    )
    assert torch.equal(mapping, torch.tensor([0, 0, 1]))
