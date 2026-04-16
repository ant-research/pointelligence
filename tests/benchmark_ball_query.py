"""Head-to-head benchmark: Pointelligence radius_search vs Pointcept ball_query."""

import time
import numpy as np
import torch
import pointops

from internals.neighbors import radius_search_lookup
from internals.indexing import repeat_interleave_indices


def make_data(num_pts, batch_size, device, scene_scale=10.0):
    """Generate batched point cloud data in both formats."""
    torch.manual_seed(42)

    # Per-batch sizes (slightly varied)
    base = num_pts // batch_size
    sizes = torch.full((batch_size,), base, dtype=torch.int64, device=device)
    sizes[-1] = num_pts - base * (batch_size - 1)

    points = torch.rand(num_pts, 3, device=device) * scene_scale

    # Pointelligence format: flat sample_inds
    sample_inds = repeat_interleave_indices(
        repeats=sizes, output_size=num_pts, may_contain_zero_repeats=False
    ).to(torch.int32)

    # Pointcept format: cumulative offset
    offset = sizes.cumsum(0).to(torch.int32)

    return points, sample_inds, offset


def bench(fn, warmup=5, repeats=20):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times), np.min(times)


def run_pointelligence(points, queries, radius, sample_inds, query_sample_inds):
    return radius_search_lookup(points, queries, radius, sample_inds, query_sample_inds)


def run_pointcept(xyz, new_xyz, radius, offset, new_offset, nsample):
    return pointops.ball_query(nsample, radius, 0.0, xyz, offset, new_xyz, new_offset)


def main():
    device = "cuda:0"
    batch_size = 4

    configs = [
        (50000, 0.2, "50K r=0.2"),
        (50000, 0.5, "50K r=0.5"),
        (100000, 0.2, "100K r=0.2"),
        (200000, 0.2, "200K r=0.2"),
        (50000, 1.0, "50K r=1.0"),
    ]

    print(f"{'Config':<20} {'nsample':>8} {'Ours (ms)':>10} {'Pointcept (ms)':>15} {'Speedup':>8} {'Our nbrs':>10} {'PC valid':>10}")
    print("-" * 95)

    for num_pts, radius, label in configs:
        points, sample_inds, offset = make_data(num_pts, batch_size, device)
        queries = points  # self-neighborhood
        query_sample_inds = sample_inds
        new_offset = offset

        # Run ours first to get actual neighbor counts
        neighbors, num_neighbors = run_pointelligence(
            points, queries, radius, sample_inds, query_sample_inds
        )
        max_nbrs = num_neighbors.max().item()
        total_nbrs = neighbors.numel()
        avg_nbrs = total_nbrs / num_pts if num_pts > 0 else 0

        # Use typical nsample values that Pointcept models use
        for nsample in [32, 64, min(max_nbrs, 128)]:
            if nsample < 1:
                continue

            # Benchmark ours
            t_ours, t_ours_min = bench(
                lambda: run_pointelligence(points, queries, radius, sample_inds, query_sample_inds)
            )

            # Benchmark Pointcept
            t_pc, t_pc_min = bench(
                lambda ns=nsample: run_pointcept(points, queries, radius, offset, new_offset, ns)
            )

            # Count valid (non-padded) neighbors from Pointcept
            idx_pc, _ = run_pointcept(points, queries, radius, offset, new_offset, nsample)
            pc_valid = (idx_pc >= 0).sum().item()

            speedup = t_pc / t_ours if t_ours > 0 else float("inf")

            tag = f"{label} ns={nsample}"
            print(
                f"{tag:<20} {nsample:>8} {t_ours:>10.3f} {t_pc:>15.3f} {speedup:>7.2f}x {total_nbrs:>10} {pc_valid:>10}"
            )

        print()


if __name__ == "__main__":
    main()
