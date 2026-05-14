#!/usr/bin/env python
"""Build the KITTI ICP cache by driving the actual KITTINMPairDataset.

Replaces precompute_kitti_icp.py. The previous script duplicated the
pair-generation logic from lib/data_loaders.py and silently drifted out
of sync, leaving 7+ pairs uncached and causing 8-way ICP races on the
cluster (jobs 282875826, 282938012). This script eliminates the
duplication: it instantiates KITTINMPairDataset and walks every index of
every phase, forcing the dataloader's own ICP load/compute/save block to
populate the cache. Pair-set parity is guaranteed by construction.

Run once on a workstation with the full KITTI mirror, then tar the icp/
directory and ship to OSS for cluster consumption.

Usage:
    python scripts/build_icp_cache.py --kitti_root /path/to/kitti_odometry_dataset
"""
import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_root', required=True)
    parser.add_argument('--icp_subdir', default='icp',
                        help='ICP cache subdir under kitti_root (default: icp)')
    parser.add_argument('--phases', nargs='+',
                        default=['train', 'val', 'test'])
    args = parser.parse_args()

    os.environ['ICP_SUBDIR'] = args.icp_subdir

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_config
    from lib.data_loaders import KITTINMPairDataset

    saved_argv = sys.argv
    sys.argv = ['build_icp_cache']
    config = get_config()
    sys.argv = saved_argv

    config.kitti_root = args.kitti_root
    config.voxel_size = 0.3

    total_pairs = 0
    total_misses_filled = 0
    t_start = time.time()

    for phase in args.phases:
        log.info(f"=== Phase: {phase} ===")
        ds = KITTINMPairDataset(
            phase=phase, config=config, transform=None,
            random_rotation=False, random_scale=False)

        misses_before = sum(
            1 for d, t0, t1 in ds.files
            if not os.path.exists(f'{ds.icp_path}/{d}_{t0}_{t1}.npy'))
        log.info(f"  {phase}: {len(ds)} pairs, {misses_before} cache misses to fill")

        if misses_before == 0:
            log.info(f"  {phase}: all pairs cached, skipping iteration")
            total_pairs += len(ds)
            continue

        for i in range(len(ds)):
            try:
                _ = ds[i]
            except Exception as e:
                log.warning(f"  {phase}[{i}] = {ds.files[i]}: {e}")
            if (i + 1) % 50 == 0:
                log.info(f"  {phase}: {i + 1}/{len(ds)} processed")

        misses_after = sum(
            1 for d, t0, t1 in ds.files
            if not os.path.exists(f'{ds.icp_path}/{d}_{t0}_{t1}.npy'))
        filled = misses_before - misses_after
        log.info(f"  {phase}: filled {filled}, remaining misses {misses_after}")
        total_pairs += len(ds)
        total_misses_filled += filled
        if misses_after > 0:
            log.error(f"  {phase}: {misses_after} pairs still missing after iteration!")

    elapsed = time.time() - t_start
    log.info(f"Done: {total_pairs} pairs across {len(args.phases)} phases, "
             f"{total_misses_filled} misses filled, {elapsed:.0f}s total. "
             f"Cache dir: {os.path.join(args.kitti_root, args.icp_subdir)}")


if __name__ == '__main__':
    main()
