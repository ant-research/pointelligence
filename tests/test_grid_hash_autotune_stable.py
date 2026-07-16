"""Regression test: grid_hash_kernel_4d must not re-autotune on point-cloud N.

Point-cloud augmentation produces a fresh point count `n` every training
iteration. If `grid_hash_kernel_4d` keys its `@triton.autotune` on `n`, the
autotuner re-tunes every iteration (~900 do_bench trials, each zeroing a
256 MB L2-flush buffer), costing seconds/iter. This is the bug that made
PNT-v0cx (PointConv3d -> radius_search -> reduce_indices_to_1d ->
grid_hash_kernel_4d) ~30x slower than the spconv v0c/v0cd path instead of
the expected <=~2x.

The autotuner caches one config per distinct key tuple. With a constant
(n-independent) key, repeated calls with varying `n` must NOT grow the
cache beyond the single entry established on the first call.
"""

import os
import sys

import pytest
import torch

PERF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PERF_ROOT)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grid_hash_kernel_4d is a CUDA Triton kernel"
)


def test_grid_hash_autotune_does_not_retune_on_varying_n():
    from internals.grid_indexing import reduce_indices_to_1d
    from internals.grid_hash_triton_kernel import grid_hash_kernel_4d

    def _call(n):
        inds = torch.randint(
            0, 1024, (n, 4), dtype=torch.int32, device="cuda"
        ).contiguous()
        reduce_indices_to_1d(inds)
        torch.cuda.synchronize()

    # Warm the autotuner once; this establishes the (single) cache entry.
    _call(4096)
    warmed = len(grid_hash_kernel_4d.cache)
    assert warmed >= 1, "autotuner cache should hold the warmed config"

    # Simulate ~30 training iters, each with a distinct point count.
    for n in range(5000, 5030):
        _call(n)

    grew_by = len(grid_hash_kernel_4d.cache) - warmed
    assert grew_by == 0, (
        f"grid_hash_kernel_4d re-autotuned {grew_by} extra time(s) across "
        f"varying n -> autotune key still depends on point-cloud N "
        f"(cache grew {warmed} -> {len(grid_hash_kernel_4d.cache)})"
    )
