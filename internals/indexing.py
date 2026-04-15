"""Index manipulation utilities for ragged tensor operations.

Provides cumsum_exclusive, repeat_interleave_indices, arrange_indices,
and other primitives used to manage variable-length batched data without
padding.
"""

import torch

from .constants import Constants

from sparse_engines_cuda.ops import bucket_arrange


def cumsum_inclusive(x, dim=0):
    # x:                1 2 3 4
    # cumsum_inclusive: 1 3 6 10
    return torch.cumsum(x, dim=dim)


def cumsum_exclusive(x, dim=0, return_sum=False):
    # x:                1 2 3 4
    # exclusive_cumsum: 0 1 3 6
    inclusive_cumsum = torch.cumsum(x, dim=dim)
    exclusive_cumsum = inclusive_cumsum - x
    if return_sum:
        assert dim == 0
        return exclusive_cumsum, inclusive_cumsum[-1]
    else:
        return exclusive_cumsum


def cumsum_inclusive_zero_prefixed(x):
    zero = Constants.get_zero(x.device, x.dtype)
    return torch.cat((zero, torch.cumsum(x, dim=0)), dim=0)


def arange_cached(end, device, dtype=torch.int64):
    constant_range = Constants.get_range(device, dtype)
    if end < constant_range.numel():
        return constant_range[:end]
    else:
        return torch.arange(end, dtype=dtype, device=device)


def repeat_interleave_indices(
    *,
    repeats=None,
    repeats_cumsum=None,
    output_size=None,
    may_contain_zero_repeats=True,
    fill_values=None
):
    # eg.
    # repeat_interleave_indices
    # input: repeats:[2, 1, 3]
    # output: [0, 0, 1, 2, 2, 2]
    # use fill_values like torch.repeat_interleave(more faster)
    # input: repeats:[2, 1, 3], fill_values:[3, 2, 1]
    # output: [3, 3, 2, 1, 1, 1]

    if repeats_cumsum is None:
        assert (
            repeats is not None
        ), "repeats should be provided when repeats_cumsum is not provided!"
        repeats_cumsum, repeats_sum = cumsum_exclusive(repeats, return_sum=True)
        if output_size is None:
            output_size = repeats_sum
        else:
            assert (
                output_size == repeats_sum
            ), "output_size should match with sum(repeats)!"

    assert (
        output_size is not None
    ), "output_size should be provided when repeats is not provided!"

    dtype, device = repeats_cumsum.dtype, repeats_cumsum.device
    indices_for_repeat = torch.zeros((output_size + 1,), dtype=dtype, device=device)
    if may_contain_zero_repeats or fill_values is not None:
        if fill_values is None:
            indices_for_repeat.index_add_(
                dim=0,
                index=repeats_cumsum[1:],
                source=Constants.get_one(device, dtype).expand(
                    repeats_cumsum.numel() - 1
                ),
            )
        else:
            fill_value_roll = fill_values.roll(1, dims=0)
            fill_value_roll[0] = 0
            fill_value_sub = fill_values - fill_value_roll
            indices_for_repeat.index_add_(
                dim=0, index=repeats_cumsum, source=fill_value_sub
            )
    else:
        indices_for_repeat.index_fill_(dim=0, index=repeats_cumsum[1:], value=1)
    indices_for_repeat.cumsum_(dim=0)

    return indices_for_repeat[:-1]


def arrange_indices(indices, num_indices=None, num_shifts=1, mask=None):
    # input: indices:[0, 1, 2, 1, 1, 2], num_indices:3
    # output: indices_arranged: [0, 1, 3, 4, 2, 5] bucket_sizes: [1, 3, 2] bucket_splits:[0, 1, 4]
    # indices[indices_arranged] = [0, 1, 1, 1, 2, 2]

    device = indices.device

    if mask is not None:
        indices_of_indices = torch.nonzero(mask).squeeze(dim=-1)
        indices = indices[indices_of_indices]
    else:
        indices_of_indices = arange_cached(indices.shape[0], device=device)

    if num_shifts > 1:
        indices_of_indices = torch.div(
            indices_of_indices, num_shifts, rounding_mode="trunc"
        )

    if num_indices is None:
        num_indices = torch.max(indices).item() + 1

    # bucket_sizes same as unique counts, eg. indices:[0, 1, 2, 1, 1, 2] bucket_sizes:[1, 3, 2]
    # bucket_slots unique number increment ID eg. indices:[0, 1, 2, 1, 1, 2] bucket_slots:[0, 0, 0, 1, 2, 1]
    bucket_sizes, bucket_slots = bucket_arrange(indices, num_indices)
    # get split id by bucket_sizes eg. bucket_sizes:[1, 3, 1] bucket_splits:[0, 1, 4]
    bucket_splits = cumsum_exclusive(bucket_sizes)

    # eg. indices:[0, 1, 2, 1, 1, 2] bucket_splits:[0, 1, 4] bucket_slots:[0, 0, 0, 1, 2, 1]
    # bucket_slots: [0, 0, 0, 1, 2, 1] + [0, 1, 4, 1, 1, 4] = [0, 1, 4, 2, 3, 5]
    bucket_slots += bucket_splits[indices.to(torch.int64)]

    # indices_of_indices: [0, 1, 2, 3, 4, 5] --> indices_arranged: [0, 1, 3, 4, 2, 5]
    indices_arranged = torch.empty_like(
        indices_of_indices, dtype=indices_of_indices.dtype, device=device
    )
    indices_arranged.index_copy_(0, bucket_slots.to(torch.int64), indices_of_indices)

    return indices_arranged, bucket_sizes, bucket_splits
