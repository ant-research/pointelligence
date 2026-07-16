import pytest
import torch

from internals.grid_indexing import build_sorted_grid_segments, reduce_indices_to_1d
from internals.grid_sample import grid_sample_filter


def test_sorted_grid_segments_can_skip_inverse_mapping():
    grid_inds = torch.tensor(
        [[-1, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]],
        dtype=torch.int32,
    )
    sorter, counts, inverse = build_sorted_grid_segments(grid_inds, return_inverse=True)
    sorter_without, counts_without, inverse_without = build_sorted_grid_segments(
        grid_inds, return_inverse=False
    )

    assert inverse_without is None
    assert torch.equal(sorter_without, sorter)
    assert torch.equal(counts_without, counts)
    assert torch.equal(torch.bincount(inverse), counts)
    keys, _, _, _ = reduce_indices_to_1d(grid_inds)
    for segment in range(counts.numel()):
        members = torch.nonzero(inverse == segment).flatten()
        assert torch.unique(keys[members]).numel() == 1


def test_explicit_triton_selector_rejects_cpu():
    with pytest.raises(ValueError, match="requires CUDA"):
        grid_sample_filter(
            torch.zeros((1, 3)),
            grid_size=1.0,
            center_nearest_impl="triton",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("return_mapping", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_center_nearest_matches_torch_with_negative_multibatch_points(
    return_mapping,
    dtype,
):
    generator = torch.Generator(device="cuda").manual_seed(31)
    points = (torch.randn((20000, 3), generator=generator, device="cuda") * 3.0).to(
        dtype
    )
    sample_inds = torch.arange(points.shape[0], device="cuda") % 4

    expected = grid_sample_filter(
        points,
        grid_size=0.2,
        sample_inds=sample_inds,
        return_mapping=return_mapping,
        center_nearest_impl="torch",
    )
    actual = grid_sample_filter(
        points,
        grid_size=0.2,
        sample_inds=sample_inds,
        return_mapping=return_mapping,
        center_nearest_impl="triton",
    )
    torch.cuda.synchronize()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    if return_mapping:
        assert torch.equal(actual[3], expected[3])
    else:
        assert actual[3] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_center_nearest_handles_ties_and_long_dense_segments():
    generator = torch.Generator(device="cuda").manual_seed(47)
    dense = torch.rand((1025, 3), generator=generator, device="cuda") * 0.98 + 0.01
    tie = torch.tensor(
        [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
        device="cuda",
    )
    points = torch.cat((tie, dense + 2.0, -dense - 2.0), dim=0)
    sample_inds = torch.cat(
        (
            torch.zeros((tie.shape[0] + dense.shape[0],), device="cuda"),
            torch.ones((dense.shape[0],), device="cuda"),
        )
    ).to(torch.long)

    expected = grid_sample_filter(
        points,
        grid_size=1.0,
        sample_inds=sample_inds,
        center_nearest_impl="torch",
    )
    actual = grid_sample_filter(
        points,
        grid_size=1.0,
        sample_inds=sample_inds,
        center_nearest_impl="triton",
    )
    torch.cuda.synchronize()

    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[0], expected[0])
    # The first cell has an exact tie; the deterministic lower source index wins.
    assert actual[2][0].item() == 0
