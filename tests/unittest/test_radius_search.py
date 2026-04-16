import datetime
import math
import sys

import numpy as np
import torch
import torch.utils.benchmark as benchmark

from scipy import spatial

from internals.neighbors import radius_search
from internals.neighbors import radius_search_brute_force
from internals.neighbors import radius_search_tiled
from internals.neighbors import repeat_interleave_indices
from internals.indexing import cumsum_inclusive_zero_prefixed


def batch_radius_search_numpy(points, num_points, queries, num_queries, radius):
    neighbors_list = list()
    num_neighbors_list = list()
    points_offset = 0
    queries_offset = 0
    for num_point, num_query in zip(num_points, num_queries):
        kdtree = spatial.KDTree(points[points_offset : points_offset + num_point])
        neighbors = kdtree.query_ball_point(
            queries[queries_offset : queries_offset + num_query], radius
        )
        neighbors_flat = np.array([y + points_offset for x in neighbors for y in x])
        neighbors_list.append(neighbors_flat)
        num_neighbors_list.append([len(x) for x in neighbors])
        points_offset += num_point
        queries_offset += num_query
    neighbors = np.concatenate(neighbors_list, axis=0)
    num_neighbors_zero_prefixed = np.concatenate([[0]] + num_neighbors_list, axis=0)
    num_neighbors_cumsum = np.cumsum(num_neighbors_zero_prefixed, axis=0)
    return neighbors, num_neighbors_cumsum


def batch_radius_search_lookup(
    points, num_points, queries, num_queries, radius, point_num_max
):
    points_sample_inds = repeat_interleave_indices(
        repeats=num_points, output_size=points.shape[0], may_contain_zero_repeats=False
    )
    query_sample_inds = repeat_interleave_indices(
        repeats=num_queries,
        output_size=queries.shape[0],
        may_contain_zero_repeats=False,
    )

    neighbors, num_neighbors = radius_search(
        points,
        queries,
        radius,
        points_sample_inds,
        query_sample_inds,
        point_num_max=point_num_max,
    )
    num_neighbors_cumsum = cumsum_inclusive_zero_prefixed(num_neighbors)

    return neighbors, num_neighbors_cumsum


def batch_radius_search_tiled(
    points, num_points, queries, num_queries, radius, point_num_max
):
    points_sample_inds = repeat_interleave_indices(
        repeats=num_points, output_size=points.shape[0], may_contain_zero_repeats=False
    )
    query_sample_inds = repeat_interleave_indices(
        repeats=num_queries,
        output_size=queries.shape[0],
        may_contain_zero_repeats=False,
    )

    neighbors, num_neighbors = radius_search_tiled(
        points,
        queries,
        radius,
        points_sample_inds,
        query_sample_inds,
    )
    num_neighbors_cumsum = cumsum_inclusive_zero_prefixed(num_neighbors)

    return neighbors, num_neighbors_cumsum


def batch_radius_search_brute_force(
    points, num_points, queries, num_queries, radius, point_num_max
):
    neighbors_list = list()
    num_neighbors_list = list()
    points_offset = 0
    queries_offset = 0
    for num_point, num_query in zip(
        num_points.cpu().numpy(), num_queries.cpu().numpy()
    ):
        points_sample = points[points_offset : points_offset + num_point]
        queries_sample = queries[queries_offset : queries_offset + num_query]
        neighbors, num_neighbors = radius_search_brute_force(
            points_sample, queries_sample, radius
        )
        neighbors_list.append(neighbors + points_offset)
        num_neighbors_list.append(num_neighbors)
        points_offset += num_point
        queries_offset += num_query
    neighbors = torch.cat(neighbors_list, dim=0).cpu().numpy()
    zero = torch.tensor([0], device=points.device)
    num_neighbors_zero_prefixed = (
        torch.cat([zero] + num_neighbors_list, dim=0).cpu().numpy()
    )
    num_neighbors_cumsum = np.cumsum(num_neighbors_zero_prefixed, axis=0)
    return neighbors, num_neighbors_cumsum


def compute_iou(
    neighbors_a, num_neighbors_cumsum_a, neighbors_b, num_neighbors_cumsum_b
):
    if isinstance(neighbors_a, torch.Tensor):
        neighbors_a = neighbors_a.cpu().numpy()
    if isinstance(num_neighbors_cumsum_a, torch.Tensor):
        num_neighbors_cumsum_a = num_neighbors_cumsum_a.cpu().numpy()
    if isinstance(neighbors_b, torch.Tensor):
        neighbors_b = neighbors_b.cpu().numpy()
    if isinstance(num_neighbors_cumsum_b, torch.Tensor):
        num_neighbors_cumsum_b = num_neighbors_cumsum_b.cpu().numpy()

    assert num_neighbors_cumsum_a.shape == num_neighbors_cumsum_b.shape
    num_points = num_neighbors_cumsum_a.shape[0] - 1
    intersection = 0
    union = 0
    for i in range(num_points):
        s_a, e_a = num_neighbors_cumsum_a[i], num_neighbors_cumsum_a[i + 1]
        s_b, e_b = num_neighbors_cumsum_b[i], num_neighbors_cumsum_b[i + 1]
        set_a = set(neighbors_a[s_a:e_a])
        set_b = set(neighbors_b[s_b:e_b])
        intersection += len(set_a.intersection(set_b))
        union += len(set_a.union(set_b))
    if union == 0:
        print("No neighbors at all...")
    IoU = 1.0 if union == 0 else intersection / union
    return IoU


def test():
    seed = 20211201
    np.random.seed(seed + 1)  # like kp kernel init
    torch.manual_seed(seed + 10)
    debug = True if sys.gettrace() else False
    if debug:
        batch_size, low_sample, high_sample = 4, 8, 32
        scene_width, scene_height, radius = 32, 6, 0.2
        low_grid, high_grid = 0, 3
    else:
        batch_size, low_sample, high_sample = 4, 1024, 16384
        scene_width, scene_height, radius = 32, 6, 0.2
        low_grid, high_grid = 0, 32
    number = 10
    device = "cuda:0"
    # device = 'cpu'
    device = device if torch.cuda.is_available() else "cpu"

    print(datetime.datetime.now(), ": benchmark started!")

    grid_size = 2 * radius
    grid_size_x = math.ceil(scene_width / grid_size)
    grid_size_y = math.ceil(scene_width / grid_size)
    grid_size_z = math.ceil(scene_height / grid_size)
    grid_size_max = grid_size_x * grid_size_y * grid_size_z

    np.random.seed(9527)
    num_grids = np.random.randint(
        low=low_sample // 2, high=high_sample // 2, size=(batch_size,)
    )
    points_list = list()
    queries_list = list()
    # construct carefully such that:
    # 1. no point lies on the surface of any queries
    # 2. some points are neighbors of two queries
    for num_sample_grids in num_grids:
        grids = np.random.choice(grid_size_max, size=(num_sample_grids,), replace=False)
        grids_x = grids // (grid_size_y * grid_size_z)
        grids_y = (grids // grid_size_z) % grid_size_y
        grids_z = grids % grid_size_z

        offset = 1.6  # in (1, 2) such that the two balls overlap while their centers being in two grids
        grids_xx = np.concatenate((grids_x, grids_x), axis=0)
        grids_yy = np.concatenate((grids_y, grids_y), axis=0)
        grids_zz = np.concatenate((grids_z, grids_z + offset / 2), axis=0)
        queries = np.stack((grids_xx, grids_yy, grids_zz), axis=1) * grid_size

        num_grid_points = np.random.randint(
            low=low_grid, high=high_grid, size=(num_sample_grids * 2,)
        )
        grids_xyz_repeat = np.repeat(queries, num_grid_points, axis=0)
        grids_xyz_jitter = np.random.randn(grids_xyz_repeat.shape[0], 3)
        radius_limit = 0.9 * (offset - 1) * radius  # multiply 0.9 to make it safe...
        grids_xyz_jitter = np.clip(grids_xyz_jitter, -radius_limit, radius_limit)
        grid_points = grids_xyz_repeat + grids_xyz_jitter

        grids_xyz_share = (
            np.stack((grids_x, grids_y, grids_z + offset / 4), axis=1) * grid_size
        )
        num_grid_share_points = np.random.randint(
            low=low_grid, high=high_grid, size=(num_sample_grids,)
        )
        grids_xyz_share_repeat = np.repeat(
            grids_xyz_share, num_grid_share_points, axis=0
        )
        grids_xyz_share_jitter = np.random.randn(grids_xyz_share_repeat.shape[0], 3)
        radius_share_limit = (
            0.9 * ((2 - offset) / 2) * radius
        )  # multiply 0.9 to make it safe...
        grids_xyz_share_jitter = np.clip(
            grids_xyz_share_jitter, -radius_share_limit, radius_share_limit
        )
        grid_share_points = grids_xyz_share_repeat + grids_xyz_share_jitter
        points = np.concatenate((grid_points, grid_share_points), axis=0)

        # shuffe and remove some queries, such that some points won't be in any neighbor list
        shuffle_queries = np.random.choice(
            queries.shape[0], size=(int(queries.shape[0] * 0.9),), replace=False
        )
        queries_list.append(queries[shuffle_queries])

        # shuffe and remove some points, such that some queries will have no neighbors
        shuffle_points = np.random.choice(
            points.shape[0], size=(int(points.shape[0] * 0.9),), replace=False
        )
        points_list.append(points[shuffle_points])

    points = torch.tensor(
        np.concatenate(points_list, axis=0), dtype=torch.float32, device=device
    )
    num_points = torch.tensor(
        np.array([x.shape[0] for x in points_list]), dtype=torch.int64, device=device
    )

    queries = torch.tensor(
        np.concatenate(queries_list, axis=0), dtype=torch.float32, device=device
    )
    num_queries = torch.tensor(
        np.array([x.shape[0] for x in queries_list]), dtype=torch.int64, device=device
    )

    scale = torch.tensor([[scene_width, scene_width, scene_height]], device=device)
    avg = (low_grid + high_grid) // 2
    num_points_random = torch.randint(
        low=low_sample * avg, high=high_sample * avg, size=(batch_size,), device=device
    )
    points_random = torch.rand(torch.sum(num_points_random), 3, device=device) * scale

    num_queries_random = torch.randint(
        low=low_sample, high=high_sample, size=(batch_size,), device=device
    )
    queries_random = torch.rand(torch.sum(num_queries_random), 3, device=device) * scale

    input_list = [
        ("grids: queries < points", points, num_points, queries, num_queries, 1),
        ("grids: queries > points", queries, num_queries, points, num_points, 1),
        (
            "random: queries < points",
            points_random,
            num_points_random,
            queries_random,
            num_queries_random,
            1,
        ),
        (
            "random: queries > points",
            queries_random,
            num_queries_random,
            points_random,
            num_points_random,
            1,
        ),
        ("grids: queries < points", points, num_points, queries, num_queries, 4),
        ("grids: queries > points", queries, num_queries, points, num_points, 4),
        (
            "random: queries < points",
            points_random,
            num_points_random,
            queries_random,
            num_queries_random,
            4,
        ),
        (
            "random: queries > points",
            queries_random,
            num_queries_random,
            points_random,
            num_points_random,
            4,
        ),
    ]
    print(datetime.datetime.now(), ": benchmark data prepared!")

    for config, points, num_points, queries, num_queries, num_splits in input_list:
        points_np = points.cpu().numpy()
        num_points_np = num_points.cpu().numpy()
        queries_np = queries.cpu().numpy()
        num_queries_np = num_queries.cpu().numpy()

        config = config + " (%d vs. %d)" % (queries.shape[0], points.shape[0])
        augments_np = {
            "points": points_np,
            "num_points": num_points_np,
            "queries": queries_np,
            "num_queries": num_queries_np,
            "radius": radius,
        }
        augments_np_str = ", ".join([key + "=" + key for key in augments_np.keys()])
        neighbors_numpy, num_neighbors_cumsum_numpy = batch_radius_search_numpy(
            **augments_np
        )
        print(
            datetime.datetime.now(),
            ": numpy results prepared!-(%s) num_splits: %d" % (config, num_splits),
        )

        point_num_max = max(points.shape[0], queries.shape[0]) // num_splits
        augments = {
            "points": points,
            "num_points": num_points,
            "queries": queries,
            "num_queries": num_queries,
            "radius": radius,
            "point_num_max": point_num_max,
        }
        augments_str = ", ".join([key + "=" + key for key in augments.keys()])

        neighbors_lookup, num_neighbors_cumsum_lookup = batch_radius_search_lookup(
            **augments
        )
        iou = compute_iou(
            neighbors_numpy,
            num_neighbors_cumsum_numpy,
            neighbors_lookup,
            num_neighbors_cumsum_lookup,
        )
        assert iou > 1 - 1e-3, "IoU: %f" % iou
        print(
            datetime.datetime.now(),
            ": batch_radius_search_lookup passed!-(%s)" % config,
        )

        t_radius_search_numpy = benchmark.Timer(
            stmt="batch_radius_search_numpy(" + augments_np_str + ")",
            setup="from __main__ import batch_radius_search_numpy",
            globals=augments_np,
            description="Batch Radius Search Numpy-(%s)" % config,
        )
        print(t_radius_search_numpy.timeit(1))

        t_radius_search_lookup = benchmark.Timer(
            stmt="batch_radius_search_lookup(" + augments_str + ")",
            setup="from __main__ import batch_radius_search_lookup",
            globals=augments,
            description="Batch Radius Search Native-(%s)" % config,
        )
        print(t_radius_search_lookup.timeit(number))

        neighbors_tiled, num_neighbors_cumsum_tiled = batch_radius_search_tiled(
            **augments
        )
        iou = compute_iou(
            neighbors_numpy,
            num_neighbors_cumsum_numpy,
            neighbors_tiled,
            num_neighbors_cumsum_tiled,
        )
        assert iou > 1 - 1e-3, "IoU: %f" % iou
        print(
            datetime.datetime.now(),
            ": batch_radius_search_tiled passed!-(%s)" % config,
        )

        t_radius_search_tiled = benchmark.Timer(
            stmt="batch_radius_search_tiled(" + augments_str + ")",
            globals={**augments, "batch_radius_search_tiled": batch_radius_search_tiled},
            description="Batch Radius Search Tiled-(%s)" % config,
        )
        print(t_radius_search_tiled.timeit(number))

        if debug:  # too slow, so run it only in debug mode, where #points is small...
            neighbors_brute_force, num_neighbors_cumsum_brute_force = (
                batch_radius_search_brute_force(**augments)
            )
            iou = compute_iou(
                neighbors_numpy,
                num_neighbors_cumsum_numpy,
                neighbors_brute_force,
                num_neighbors_cumsum_brute_force,
            )
            assert iou > 1 - 1e-3, "IoU: %f" % iou
            print(
                datetime.datetime.now(),
                ": batch_radius_search_brute_force passed!-(%s)" % config,
            )

            t_radius_search_brute_force = benchmark.Timer(
                stmt="batch_radius_search_brute_force(" + augments_str + ")",
                setup="from __main__ import batch_radius_search_brute_force",
                globals=augments,
                description="Batch Radius Search Native-(%s)" % config,
            )
            print(t_radius_search_brute_force.timeit(number))


if __name__ == "__main__":
    test()
