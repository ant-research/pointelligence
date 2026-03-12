import datetime
import math
import sys

import numpy as np
import torch
import torch.utils.benchmark as benchmark

from internals.grid_sample import grid_sample_filter
from internals.indexing import repeat_interleave_indices


def batch_grid_sample(points, num_points, grid_size, reduction):
    sample_inds = repeat_interleave_indices(
        repeats=num_points, output_size=points.shape[0], may_contain_zero_repeats=False
    )

    filtered = grid_sample_filter(
        points,
        grid_size,
        sample_inds=sample_inds,
        reduction=reduction,
        return_mapping=True,
    )

    return filtered[0], filtered[1]


def batch_grid_sample_center_nearest(points, num_points, grid_size):
    return batch_grid_sample(points, num_points, grid_size, "center_nearest")


def batch_grid_sample_average(points, num_points, grid_size):
    return batch_grid_sample(points, num_points, grid_size, "mean")


def batch_grid_sample_average_per_sample(points, num_points, grid_size):
    points_filtered_list = list()
    num_points_filtered_list = list()
    num_points_np = num_points.cpu().numpy()
    offset = 0
    for num_point in num_points_np:
        points_sample = points[offset : offset + num_point, :]
        offset += num_point

        filtered = grid_sample_filter(
            points_sample,
            grid_size,
            reduction="mean",
            return_mapping=True,
        )
        points_sample_filtered = filtered[0]
        points_filtered_list.append(points_sample_filtered)
        num_points_filtered_list.append(points_sample_filtered.shape[0])

    points_filtered = torch.cat(points_filtered_list, dim=0)
    dtype = torch.int64
    device = points.device
    num_points_filtered = torch.tensor(
        num_points_filtered_list, dtype=dtype, device=device
    )
    return points_filtered, num_points_filtered


def batch_grid_sample_average_sort(points, num_points, grid_size):
    device = points.device
    sample_inds = repeat_interleave_indices(
        repeats=num_points, output_size=points.shape[0], may_contain_zero_repeats=False
    )
    points_min = torch.segment_reduce(points, "min", lengths=num_points)
    points_max = torch.segment_reduce(points, "max", lengths=num_points)
    points_origin = points_min.div(grid_size, rounding_mode="floor") * grid_size

    dtype = torch.int32
    grid_sizes = (
        (points_max - points_origin).div(grid_size, rounding_mode="floor").to(dtype)
    )  # for each sample
    grid_sizes = torch.amax(grid_sizes, dim=0, keepdims=True) + 1  # max in the batch
    grid_sizes_prefix_1 = torch.cat(
        (torch.tensor([[1]], dtype=dtype, device=device), grid_sizes), dim=1
    )
    grid_sizes_cumprod = torch.cumprod(grid_sizes_prefix_1, dim=1)

    points_origin_repeated = points_origin[sample_inds]
    offsets = points - points_origin_repeated
    grid_inds = offsets.div(grid_size, rounding_mode="floor").to(dtype)
    grid_sample_inds = torch.cat((grid_inds, sample_inds.unsqueeze(dim=-1)), dim=1)
    grid_idx = torch.sum(grid_sample_inds * grid_sizes_cumprod, dim=1)

    grid_idx_sorted, sort_indices = torch.sort(grid_idx)
    grid_idx_sorted_unique, counts = torch.unique_consecutive(
        grid_idx_sorted, return_counts=True
    )

    points_sorted = points[sort_indices]
    points_filtered = torch.segment_reduce(points_sorted, "mean", lengths=counts)

    sample_inds_filtered = torch.div(
        grid_idx_sorted_unique, grid_sizes_cumprod[0, -1], rounding_mode="trunc"
    )
    _, num_points_filtered = torch.unique_consecutive(
        sample_inds_filtered, return_counts=True
    )

    return points_filtered, num_points_filtered


def unique_consecutive(x, return_counts=False):
    unique_mask = np.ediff1d(x, to_end=1) != 0
    x_unique = x[unique_mask]
    if not return_counts:
        return x_unique
    nonzero_indices = np.nonzero(unique_mask)[0]
    counts = np.ediff1d(nonzero_indices, to_begin=[nonzero_indices[0] + 1])
    return x_unique, counts.astype(np.int32)


# The numpy implementation mostly follows the torch implementation
def batch_grid_sample_average_numpy_sort(points, num_points, grid_size):
    batch_size = num_points.size
    sample_inds = np.repeat(np.arange(batch_size), num_points, axis=0)
    num_points_cumsum = np.concatenate(([0], np.cumsum(num_points)[:-1]), axis=0)
    points_min = np.minimum.reduceat(points, num_points_cumsum)
    points_max = np.maximum.reduceat(points, num_points_cumsum)
    points_origin = np.floor(points_min * (1 / grid_size)) * grid_size

    grid_sizes = np.floor(
        (points_max - points_origin) * (1 / grid_size)
    )  # for each sample
    grid_sizes = np.amax(grid_sizes, axis=0, keepdims=True) + 1  # max in the batch
    grid_sizes_prefix_1 = np.concatenate((np.array([[1]]), grid_sizes), axis=1)
    grid_sizes_cumprod = np.cumprod(grid_sizes_prefix_1, axis=1)

    points_origin_repeated = np.repeat(points_origin, num_points, axis=0)
    offsets = points - points_origin_repeated
    grid_inds = np.floor(offsets * (1 / grid_size))
    grid_sample_inds = np.concatenate(
        (grid_inds, np.expand_dims(sample_inds, axis=-1)), axis=1
    )
    grid_idx = np.sum(grid_sample_inds * grid_sizes_cumprod, axis=1)

    sort_indices = np.argsort(grid_idx)
    grid_idx_sorted = grid_idx[sort_indices]
    grid_idx_sorted_unique, unique_counts = unique_consecutive(
        grid_idx_sorted, return_counts=True
    )
    points_sorted = points[sort_indices]
    unique_counts_cumsum = np.concatenate(([0], np.cumsum(unique_counts)[:-1]), axis=0)
    points_filtered = np.add.reduceat(points_sorted, unique_counts_cumsum)
    points_filtered /= np.expand_dims(unique_counts, axis=-1)

    sample_inds_filtered = grid_idx_sorted_unique // grid_sizes_cumprod[0, -1]
    _, num_points_filtered = unique_consecutive(
        sample_inds_filtered, return_counts=True
    )

    return points_filtered, num_points_filtered


def sort_by_sample_inds(output, is_sorted=False):
    points, sample_inds = output
    if is_sorted:
        _, num_points = torch.unique_consecutive(sample_inds, return_counts=True)
        return points, num_points

    sample_inds, sort_indices = torch.sort(sample_inds)
    # We can compute num_points with diff+nonzero, which is faster, but with more LoC..
    _, num_points = torch.unique_consecutive(sample_inds, return_counts=True)
    points = points[sort_indices]

    return points, num_points


def sort_by_grid_inds(output, points, num_points, grid_size):
    points_filtered, num_points_filtered = output

    device = points.device
    if not isinstance(points_filtered, torch.Tensor):
        points_filtered = torch.tensor(points_filtered, device=device)
    if not isinstance(num_points_filtered, torch.Tensor):
        num_points_filtered = torch.tensor(
            num_points_filtered, dtype=torch.int64, device=device
        )

    points_min = torch.segment_reduce(points, "min", lengths=num_points)
    points_max = torch.segment_reduce(points, "max", lengths=num_points)
    points_origin = points_min.div(grid_size, rounding_mode="floor") * grid_size

    dtype = torch.int32
    grid_sizes = (
        (points_max - points_origin).div(grid_size, rounding_mode="floor").to(dtype)
    )  # for each sample
    grid_sizes = torch.amax(grid_sizes, dim=0, keepdims=True) + 1  # max in the batch
    grid_sizes_prefix_1 = torch.cat(
        (torch.tensor([[1]], dtype=dtype, device=device), grid_sizes), dim=1
    )
    grid_sizes_cumprod = torch.cumprod(grid_sizes_prefix_1, dim=1)

    sample_inds_filtered = repeat_interleave_indices(
        repeats=num_points_filtered,
        output_size=points_filtered.shape[0],
        may_contain_zero_repeats=False,
    )
    points_origin_repeated = points_origin[sample_inds_filtered]
    offsets = points_filtered - points_origin_repeated
    grid_inds = offsets.div(grid_size, rounding_mode="floor").to(dtype)

    grid_sample_inds = torch.cat(
        (grid_inds, sample_inds_filtered.unsqueeze(dim=-1)), dim=1
    )
    grid_idx = torch.sum(grid_sample_inds * grid_sizes_cumprod, dim=1)

    _, sort_indices = torch.sort(grid_idx)
    points_filtered_sorted = points_filtered[sort_indices]

    return points_filtered_sorted, num_points_filtered


def test():
    seed = 20211201
    np.random.seed(seed + 1)  # like kp kernel init
    torch.manual_seed(seed + 10)
    debug = True if sys.gettrace() else False
    if debug:
        batch_size, low_sample, high_sample = 4, 8, 32
        num_labels, num_channels = 8, 128
        scene_width, scene_height, grid_size = 32, 6, 0.08
        low_grid, high_grid = 1, 8
    else:
        batch_size, low_sample, high_sample = 4, 1024, 32768
        num_labels, num_channels = 8, 128
        scene_width, scene_height, grid_size = 32, 6, 0.08
        low_grid, high_grid = 1, 8
    number = 10
    device = "cuda:0"
    # device = 'cpu'
    device = device if torch.cuda.is_available() else "cpu"

    print(datetime.datetime.now(), ": benchmark started!")

    grid_size_x = math.ceil(scene_width / grid_size)
    grid_size_y = math.ceil(scene_width / grid_size)
    grid_size_z = math.ceil(scene_height / grid_size)
    grid_size_max = grid_size_x * grid_size_y * grid_size_z

    num_grids = np.random.randint(low=low_sample, high=high_sample, size=(batch_size,))
    points_list = list()
    for num_sample_grids in num_grids:
        grids = np.random.choice(grid_size_max, size=(num_sample_grids,), replace=False)
        num_grid_points = np.random.randint(
            low=low_grid, high=high_grid, size=(num_sample_grids,)
        )
        grids_x = grids // (grid_size_y * grid_size_z)
        grids_y = (grids // grid_size_z) % grid_size_y
        grids_z = grids % grid_size_z
        girds_xyz = np.stack((grids_x, grids_y, grids_z), axis=1)
        grids_xyz_repeat = np.repeat(girds_xyz, num_grid_points, axis=0)
        grids_xyz_inside = 0.1 + 0.8 * np.random.random(grids_xyz_repeat.shape)
        sample_points = (grids_xyz_inside + grids_xyz_repeat) * grid_size

        grid_labels = np.random.randint(
            low=0, high=num_labels, size=(num_sample_grids,)
        )
        # make sure the majority label is unique, to favor the equality check
        grid_labels_list = list()
        for grid_label, num_grid_point in zip(grid_labels, num_grid_points):
            major_labels = [grid_label] * (num_grid_point // 2 + 1)
            minor_labels = list(
                np.random.choice(num_labels, num_grid_point - len(major_labels))
            )
            grid_labels_list += major_labels + minor_labels
        sample_labels = np.array(grid_labels_list)

        shuffle = np.arange(sample_points.shape[0])
        np.random.shuffle(shuffle)
        points_list.append(sample_points[shuffle])

    points = torch.tensor(
        np.concatenate(points_list, axis=0), dtype=torch.float32, device=device
    )
    num_points = torch.tensor(
        np.array([x.shape[0] for x in points_list]), dtype=torch.int64, device=device
    )
    num_total_points = num_points.sum()
    features_ = torch.randn((num_total_points, num_channels), device=device)
    print(
        datetime.datetime.now(),
        ": benchmark data prepared (%d points)!" % points.shape[0],
    )

    points_np = points.cpu().numpy()
    num_points_np = num_points.to(torch.int32).cpu().numpy()
    augments_np = {
        "points": points_np,
        "num_points": num_points_np,
        "grid_size": grid_size,
    }
    augments_np_str = ", ".join([key + "=" + key for key in augments_np.keys()])

    output_average_numpy = batch_grid_sample_average_numpy_sort(**augments_np)
    output_average_numpy = sort_by_grid_inds(
        output_average_numpy, points, num_points, grid_size
    )
    print(datetime.datetime.now(), ": average_numpy results prepared!")

    augments = {
        "points": points,
        "num_points": num_points,
        "grid_size": grid_size,
    }
    augments_str = ", ".join([key + "=" + key for key in augments.keys()])
    output_average_sort = batch_grid_sample_average_sort(**augments)
    print(
        "max_diff(points_average_numpy, points_average_sort):",
        (output_average_numpy[0] - output_average_sort[0]).abs().max(),
    )
    assert output_average_numpy[0].allclose(output_average_sort[0], atol=1e-5)
    assert output_average_numpy[1].equal(output_average_sort[1])

    output_average_per_sample = batch_grid_sample_average_per_sample(**augments)
    output_average_per_sample = sort_by_grid_inds(
        output_average_per_sample, points, num_points, grid_size
    )
    print(
        "max_diff(points_average_numpy, points_average_per_sample):",
        (output_average_numpy[0] - output_average_per_sample[0]).abs().max(),
    )
    assert output_average_numpy[0].allclose(output_average_per_sample[0], atol=1e-5)
    assert output_average_numpy[1].equal(output_average_per_sample[1])

    output_average = batch_grid_sample_average(**augments)
    output_average = sort_by_sample_inds(output_average, True)
    output_average = sort_by_grid_inds(output_average, points, num_points, grid_size)
    print(
        "max_diff(points_average_numpy, points_average):",
        (output_average_numpy[0] - output_average[0]).abs().max(),
    )
    assert output_average_numpy[0].allclose(output_average[0], atol=1e-5)
    assert output_average_numpy[1].equal(output_average[1])

    output_center_nearest = batch_grid_sample_center_nearest(**augments)
    output_center_nearest = sort_by_sample_inds(output_center_nearest, True)
    assert output_average_numpy[1].equal(output_center_nearest[1])
    print(
        datetime.datetime.now(),
        ": sanity check for batch_grid_sample_center_nearest passed!",
    )

    t_subsample_numpy = benchmark.Timer(
        stmt="batch_grid_sample_average_numpy_sort(" + augments_np_str + ")",
        setup="from __main__ import batch_grid_sample_average_numpy_sort",
        globals=augments_np,
        description="Batch Subsample Average Numpy",
    )
    print(t_subsample_numpy.timeit(number))

    t_subsample_sort = benchmark.Timer(
        stmt="batch_grid_sample_average_sort(" + augments_str + ")",
        setup="from __main__ import batch_grid_sample_average_sort",
        globals=augments,
        description="Batch Subsample Average Sort",
    )
    print(t_subsample_sort.timeit(number))

    t_subsample = benchmark.Timer(
        stmt="batch_grid_sample_average_per_sample(" + augments_str + ")",
        setup="from __main__ import batch_grid_sample_average_per_sample",
        globals=augments,
        description="Batch Subsample Average per Sample",
    )
    print(t_subsample.timeit(number))

    t_subsample_average = benchmark.Timer(
        stmt="batch_grid_sample_average(" + augments_str + ")",
        setup="from __main__ import batch_grid_sample_average",
        globals=augments,
        description="Batch Subsample Average",
    )
    print(t_subsample_average.timeit(number))

    t_subsample_center_nearest = benchmark.Timer(
        stmt="batch_grid_sample_center_nearest(" + augments_str + ")",
        setup="from __main__ import batch_grid_sample_center_nearest",
        globals=augments,
        description="Batch Subsample Center Nearest",
    )
    print(t_subsample_center_nearest.timeit(number))


if __name__ == "__main__":
    test()
