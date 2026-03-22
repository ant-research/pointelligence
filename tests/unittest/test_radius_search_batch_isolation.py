import pytest
import torch

from internals.indexing import cumsum_inclusive_zero_prefixed
from internals.neighbors import radius_search


def test_radius_search_respects_sample_boundaries_with_overlapping_coordinates():
    if not torch.cuda.is_available():
        pytest.skip("radius_search lookup regression requires CUDA")

    device = torch.device("cuda")
    points = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.05, 0.05],
            [0.10, 0.10, 0.10],
            [0.15, 0.15, 0.15],
            [0.00, 0.00, 0.00],
            [0.05, 0.05, 0.05],
            [0.10, 0.10, 0.10],
            [0.15, 0.15, 0.15],
        ],
        dtype=torch.float32,
        device=device,
    )
    sample_inds = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=device)
    queries = torch.tensor(
        [
            [0.05, 0.05, 0.05],
            [0.15, 0.15, 0.15],
            [0.05, 0.05, 0.05],
            [0.15, 0.15, 0.15],
        ],
        dtype=torch.float32,
        device=device,
    )
    query_sample_inds = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)

    neighbors, num_neighbors = radius_search(
        points=points,
        query_points=queries,
        radius=0.11,
        sample_inds=sample_inds,
        query_sample_inds=query_sample_inds,
    )
    neighbor_offsets = cumsum_inclusive_zero_prefixed(num_neighbors)
    expected_ranges = [(0, 4), (0, 4), (4, 8), (4, 8)]

    for query_index, (start, end) in enumerate(expected_ranges):
        neighbor_slice = neighbors[
            neighbor_offsets[query_index] : neighbor_offsets[query_index + 1]
        ]
        assert neighbor_slice.numel() > 0
        assert torch.all((neighbor_slice >= start) & (neighbor_slice < end))
