import pytest
import torch

from internals.neighbors import radius_search_tiled
from internals.neighbors import radius_search_sorted_grid8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")


def _keys(result, num_points):
    neighbors, counts = result[:2]
    query = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts.long())
    return torch.sort(query.long() * num_points + neighbors.long()).values


def test_sorted_grid_bounds_prevent_spatial_and_batch_key_aliases():
    eps = torch.finfo(torch.float32).eps
    one_batch = torch.tensor([
        [-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0],
        [1.0 + 8 * eps, 0, 0],
    ], device="cuda")
    points = torch.cat((one_batch, one_batch), dim=0)
    queries = torch.zeros((2, 3), device="cuda")
    sample = torch.repeat_interleave(
        torch.arange(2, device="cuda"), one_batch.shape[0])
    query_sample = torch.arange(2, device="cuda")

    candidate = radius_search_sorted_grid8(
        points, queries, 1.0, sample, query_sample,
        return_distances=True)
    candidate_keys = _keys(candidate, points.shape[0])
    assert torch.unique(candidate_keys).numel() == candidate_keys.numel()
    assert torch.equal(candidate[1], torch.tensor(
        [3, 3], device="cuda", dtype=candidate[1].dtype))
    query_ids = torch.div(
        candidate_keys, points.shape[0], rounding_mode="floor")
    point_ids = candidate_keys.remainder(points.shape[0])
    assert torch.equal(query_ids, sample[point_ids])

    expected = []
    expected_counts = []
    for batch_id in range(2):
        batch_points = points[sample == batch_id]
        neighbors, counts = radius_search_tiled(
            batch_points, queries[batch_id:batch_id + 1], 1.0)
        expected.append(neighbors + batch_id * one_batch.shape[0])
        expected_counts.append(counts)
    expected_result = (
        torch.cat(expected), torch.cat(expected_counts))
    assert torch.equal(candidate_keys, _keys(
        expected_result, points.shape[0]))
