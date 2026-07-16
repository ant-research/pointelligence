"""Compare public geometry APIs across two repository revisions on real data.

The driver prepares identical transformed ScanNet workloads once, launches each
repository revision in a fresh Python process, checks exact output parity, and
only then runs synchronized timing and incremental-memory measurements.  This
keeps imports, Triton caches, and implementation-specific module state isolated.

Example::

    python benchmarks/operators/bench_geometry_version_compare.py \
        --baseline-repo /path/to/v1.4.0 \
        --candidate-repo /path/to/v1.5.0 \
        --root /path/to/scannet/val \
        --pointcept-root build/Pointcept \
        --output /tmp/geometry_versions.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import torch
from torch.utils.cpp_extension import load

SCENES = [
    "scene0011_00",
    "scene0015_00",
    "scene0046_00",
    "scene0231_00",
    "scene0019_00",
    "scene0025_00",
    "scene0030_00",
    "scene0050_00",
]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _git_sha(repo: str) -> str:
    return subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()


def _load_transforms(pointcept_root: str):
    if pointcept_root not in sys.path:
        sys.path.insert(0, pointcept_root)
    transform_path = os.path.join(
        pointcept_root, "pointcept", "datasets", "transform.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pointcept_transform_for_version_benchmark", transform_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Pointcept transforms from {transform_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CenterShift, module.GridSample


def _transformed_scenes(root: str, pointcept_root: str, input_grid: float):
    CenterShift, GridSample = _load_transforms(pointcept_root)
    scenes = []
    for index, scene in enumerate(SCENES):
        coord = np.load(os.path.join(root, scene, "coord.npy")).astype(
            np.float32, copy=True
        )
        data = {"coord": coord, "index_valid_keys": ["coord"]}
        data = CenterShift(apply_z=True)(data)
        np.random.seed(1701 + index)
        data = GridSample(grid_size=input_grid, mode="train")(data)
        scenes.append(torch.from_numpy(np.ascontiguousarray(data["coord"])))
    return scenes


def _batch(scenes: list[torch.Tensor], batch: int):
    sizes = torch.tensor([scene.shape[0] for scene in scenes[:batch]])
    points = torch.cat(scenes[:batch])
    sample_inds = torch.repeat_interleave(torch.arange(batch), sizes)
    return points, sample_inds


def _prepare_workloads(args, path: str):
    candidate_repo = os.path.abspath(args.candidate_repo)
    sys.path.insert(0, candidate_repo)
    from internals.grid_sample import grid_sample_filter

    scenes = _transformed_scenes(args.root, args.pointcept_root, args.input_grid)
    workloads = {"downsample": [], "radius": []}
    radius_batches = set(_csv_ints(args.radius_batches))
    stages = _csv_ints(args.stages)
    ratios = _csv_floats(args.radius_ratios)

    for batch in _csv_ints(args.batches):
        points, sample_inds = _batch(scenes, batch)
        points = points.cuda()
        sample_inds = sample_inds.cuda()
        grid_size = args.input_grid
        for stage in stages:
            label = f"B{batch}_S{stage}"
            workloads["downsample"].append(
                {
                    "label": label,
                    "points": points.cpu(),
                    "sample_inds": sample_inds.cpu(),
                    "grid_size": grid_size,
                }
            )
            if batch in radius_batches:
                queries, query_sample_inds, _, _ = grid_sample_filter(
                    points,
                    grid_size=grid_size * 2,
                    sample_inds=sample_inds,
                    reduction="center_nearest",
                    return_mapping=False,
                )
                for ratio in ratios:
                    workloads["radius"].append(
                        {
                            "label": f"{label}_R{ratio:g}",
                            "points": points.cpu(),
                            "sample_inds": sample_inds.cpu(),
                            "queries": queries.cpu(),
                            "query_sample_inds": query_sample_inds.cpu(),
                            "grid_size": grid_size,
                            "radius_ratio": ratio,
                        }
                    )
            if stage != stages[-1]:
                points, sample_inds, _, _ = grid_sample_filter(
                    points,
                    grid_size=grid_size * 2,
                    sample_inds=sample_inds,
                    reduction="center_nearest",
                    return_mapping=False,
                )
                grid_size *= 2

    torch.save(workloads, path)
    return workloads


def _stats(values: list[float]):
    return {
        "median": statistics.median(values),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "samples": values,
    }


def _timed(fn, warmups: int, runs: int, iterations: int):
    for _ in range(warmups):
        result = fn()
        del result
    torch.cuda.synchronize()
    wall_ms = []
    device_ms = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            result = fn()
            del result
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000 / iterations)
        device_ms.append(start.elapsed_time(end) / iterations)
    return {"wall_ms": _stats(wall_ms), "device_ms": _stats(device_ms)}


def _peak_memory(fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del result
    torch.cuda.synchronize()
    mib = 1024 * 1024
    return {
        "allocated_mib": (peak_allocated - base_allocated) / mib,
        "reserved_mib": (peak_reserved - base_reserved) / mib,
    }


def _canonical_downsample(points, sample_inds, indices, grid_size):
    chosen_points = points[indices]
    spatial_keys = torch.floor(chosen_points / grid_size).to(torch.int64)
    keys = torch.cat(
        [spatial_keys, sample_inds[indices, None].to(torch.int64)],
        dim=1,
    ).numpy()
    selected = indices.to(torch.int64).numpy()
    centers = (spatial_keys.to(points.dtype) + 0.5) * grid_size
    distance_sq = torch.sum((chosen_points - centers).square(), dim=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0], keys[:, 3]))
    order_tensor = torch.from_numpy(order.copy()).to(torch.int64)
    return (
        torch.from_numpy(keys[order].copy()),
        torch.from_numpy(selected[order].copy()),
        distance_sq[order_tensor],
    )


def _prepare_legacy_bucket_extension(repo: str):
    """Load the tagged bucket kernel without installing either repository."""
    source_root = os.path.join(repo, "extensions", "sparse_engines_cuda", "csrc")
    bucket_cpp = os.path.join(source_root, "bucket_arrange.cpp")
    if not os.path.isfile(bucket_cpp):
        return
    sources = [
        os.path.join(source_root, "definitions.cpp"),
        bucket_cpp,
        os.path.join(source_root, "cuda", "bucket_arrange_kernel.cu"),
    ]
    sha = _git_sha(repo)[:12]
    build_directory = os.path.join(tempfile.gettempdir(), f"geometry_bucket_{sha}")
    os.makedirs(build_directory, exist_ok=True)
    load(
        name=f"geometry_bucket_{sha}",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        is_python_module=False,
        build_directory=build_directory,
        verbose=False,
    )

    package = types.ModuleType("sparse_engines_cuda")
    package.__path__ = []
    ops = types.ModuleType("sparse_engines_cuda.ops")

    def bucket_arrange(bucket_indices, num_buckets):
        return torch.ops.sparse_engines_cuda.bucket_arrange(bucket_indices, num_buckets)

    ops.bucket_arrange = bucket_arrange
    package.ops = ops
    sys.modules["sparse_engines_cuda"] = package
    sys.modules["sparse_engines_cuda.ops"] = ops


def _worker(args):
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    _prepare_legacy_bucket_extension(repo)
    from internals.grid_sample import grid_sample_filter
    from internals import neighbors as neighbors_module

    workloads = torch.load(args.workloads, map_location="cpu", weights_only=False)
    results = {"repo": repo, "sha": _git_sha(repo), "operation": args.operation}
    correctness = []
    timings = []

    selected_labels = {item for item in args.labels.split(",") if item}
    for item in workloads[args.operation]:
        if selected_labels and item["label"] not in selected_labels:
            continue
        points = item["points"].cuda()
        sample_inds = item["sample_inds"].cuda()
        if args.operation == "downsample":
            target_grid = item["grid_size"] * 2

            def fn():
                return grid_sample_filter(
                    points,
                    grid_size=target_grid,
                    sample_inds=sample_inds,
                    reduction="center_nearest",
                    return_mapping=False,
                )

            output = fn()
            keys, selected, distance_sq = _canonical_downsample(
                points.cpu(), sample_inds.cpu(), output[2].cpu(), target_grid
            )
            correctness.append(
                {
                    "label": item["label"],
                    "keys": keys,
                    "selected": selected,
                    "distance_sq": distance_sq,
                }
            )
            del output
        else:
            queries = item["queries"].cuda()
            query_sample_inds = item["query_sample_inds"].cuda()
            sample_sizes = torch.bincount(sample_inds)
            query_sample_sizes = torch.bincount(query_sample_inds)
            radius = item["grid_size"] * item["radius_ratio"]

            if args.radius_control == "v14_lookup" and hasattr(
                neighbors_module, "radius_search_lookup"
            ):

                def fn():
                    return neighbors_module.radius_search_lookup(
                        points=points,
                        queries=queries,
                        radius=radius,
                        sample_inds=sample_inds,
                        query_sample_inds=query_sample_inds,
                        return_distances=False,
                        dtype_num_neighbors=torch.int64,
                        distance_type="ball",
                    )

            else:

                def fn():
                    return neighbors_module.radius_search(
                        points=points,
                        query_points=queries,
                        radius=radius,
                        sample_inds=sample_inds,
                        query_sample_inds=query_sample_inds,
                        return_distances=False,
                        sample_sizes=sample_sizes,
                        query_sample_sizes=query_sample_sizes,
                        distance_type="ball",
                        grid_size=item["grid_size"],
                    )

            neighbors, counts = fn()
            query_ids = torch.repeat_interleave(
                torch.arange(queries.shape[0], device="cuda"), counts
            )
            pairs = (query_ids * points.shape[0] + neighbors).sort().values.cpu()
            correctness.append(
                {"label": item["label"], "pairs": pairs, "counts": counts.cpu()}
            )
            del neighbors, counts, query_ids, pairs

        if args.phase == "timing":
            measurement = {
                "label": item["label"],
                "points": points.shape[0],
                "timing": _timed(fn, args.warmups, args.runs, args.iterations),
                "peak_memory": _peak_memory(fn),
            }
            if args.operation == "radius":
                measurement["queries"] = queries.shape[0]
                measurement["radius_ratio"] = item["radius_ratio"]
            timings.append(measurement)
        torch.cuda.empty_cache()

    torch.save(correctness, args.correctness_output)
    results["measurements"] = timings
    with open(args.json_output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def _run_worker(args, repo, operation, phase, workdir):
    prefix = f"{Path(repo).name}_{operation}_{phase}"
    json_output = os.path.join(workdir, f"{prefix}.json")
    correctness_output = os.path.join(workdir, f"{prefix}.pt")
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--repo",
        repo,
        "--operation",
        operation,
        "--phase",
        phase,
        "--workloads",
        os.path.join(workdir, "workloads.pt"),
        "--json-output",
        json_output,
        "--correctness-output",
        correctness_output,
        "--radius-control",
        args.radius_control,
        "--labels",
        args.labels,
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
        "--iterations",
        str(
            args.downsample_iterations
            if operation == "downsample"
            else args.radius_iterations
        ),
    ]
    subprocess.run(command, check=True)
    with open(json_output, encoding="utf-8") as handle:
        payload = json.load(handle)
    parity = torch.load(correctness_output, map_location="cpu", weights_only=False)
    return payload, parity


def _assert_parity(operation, baseline, candidate):
    if [item["label"] for item in baseline] != [item["label"] for item in candidate]:
        raise AssertionError(f"{operation}: workload labels differ")
    details = []
    for left, right in zip(baseline, candidate):
        if operation == "downsample":
            if not torch.equal(left["keys"], right["keys"]):
                raise AssertionError(
                    f"downsample parity failed for {left['label']} field keys"
                )
            torch.testing.assert_close(
                left["distance_sq"], right["distance_sq"], rtol=0, atol=0
            )
            changed = int(torch.count_nonzero(left["selected"] != right["selected"]))
            details.append(
                {"label": left["label"], "equal_distance_tie_changes": changed}
            )
        else:
            for field in ("pairs", "counts"):
                if not torch.equal(left[field], right[field]):
                    raise AssertionError(
                        f"radius parity failed for {left['label']} field {field}"
                    )
            details.append({"label": left["label"], "exact_set_match": True})
    return details


def _comparisons(baseline, candidate):
    by_label = {item["label"]: item for item in candidate["measurements"]}
    out = []
    for old in baseline["measurements"]:
        new = by_label[old["label"]]
        old_ms = old["timing"]["device_ms"]["median"]
        new_ms = new["timing"]["device_ms"]["median"]
        old_alloc = old["peak_memory"]["allocated_mib"]
        new_alloc = new["peak_memory"]["allocated_mib"]
        old_reserved = old["peak_memory"]["reserved_mib"]
        new_reserved = new["peak_memory"]["reserved_mib"]
        out.append(
            {
                "label": old["label"],
                "speedup": old_ms / new_ms,
                "allocated_ratio": new_alloc / old_alloc if old_alloc else None,
                "reserved_ratio": new_reserved / old_reserved if old_reserved else None,
            }
        )
    return out


def _driver(args):
    for repo in (args.baseline_repo, args.candidate_repo):
        if not os.path.isdir(os.path.join(repo, ".git")) and not os.path.isfile(
            os.path.join(repo, ".git")
        ):
            raise ValueError(f"not a Git worktree: {repo}")
    with tempfile.TemporaryDirectory(prefix="geometry-version-bench-") as workdir:
        workload_path = os.path.join(workdir, "workloads.pt")
        workloads = _prepare_workloads(args, workload_path)
        payload = {
            "device": torch.cuda.get_device_name(0),
            "baseline": {
                "repo": args.baseline_repo,
                "sha": _git_sha(args.baseline_repo),
            },
            "candidate": {
                "repo": args.candidate_repo,
                "sha": _git_sha(args.candidate_repo),
            },
            "method": {
                "data": "real ScanNet validation scenes after production CenterShift and train GridSample",
                "correctness_before_timing": True,
                "sequential_fresh_processes": True,
                "warmups": args.warmups,
                "runs": args.runs,
                "downsample_iterations": args.downsample_iterations,
                "radius_iterations": args.radius_iterations,
                "radius_control": args.radius_control,
            },
            "workload_counts": {key: len(value) for key, value in workloads.items()},
            "operations": {},
        }
        operations = [item for item in args.operations.split(",") if item]
        invalid = set(operations) - {"radius", "downsample"}
        if invalid:
            raise ValueError(f"invalid operations: {sorted(invalid)}")
        for operation in operations:
            _, baseline_correctness = _run_worker(
                args, args.baseline_repo, operation, "correctness", workdir
            )
            _, candidate_correctness = _run_worker(
                args, args.candidate_repo, operation, "correctness", workdir
            )
            parity_details = _assert_parity(
                operation, baseline_correctness, candidate_correctness
            )
            baseline, _ = _run_worker(
                args, args.baseline_repo, operation, "timing", workdir
            )
            candidate, _ = _run_worker(
                args, args.candidate_repo, operation, "timing", workdir
            )
            payload["operations"][operation] = {
                "parity": "PASS",
                "parity_details": parity_details,
                "baseline": baseline["measurements"],
                "candidate": candidate["measurements"],
                "comparisons": _comparisons(baseline, candidate),
            }
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        print(json.dumps(payload, indent=2))


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-repo")
    parser.add_argument("--candidate-repo")
    parser.add_argument("--root")
    parser.add_argument("--pointcept-root")
    parser.add_argument("--output")
    parser.add_argument("--input-grid", type=float, default=0.02)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--radius-batches", default="2")
    parser.add_argument("--stages", default="0,1,2,3")
    parser.add_argument(
        "--radius-control",
        choices=("auto", "v14_lookup"),
        default="v14_lookup",
    )
    parser.add_argument("--radius-ratios", default="1,1.86,2.75,4.5")
    parser.add_argument("--operations", default="radius,downsample")
    parser.add_argument("--labels", default="")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--downsample-iterations", type=int, default=20)
    parser.add_argument("--radius-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--operation", choices=("radius", "downsample"))
    parser.add_argument("--phase", choices=("correctness", "timing"))
    parser.add_argument("--workloads")
    parser.add_argument("--json-output")
    parser.add_argument("--correctness-output")
    return parser


def main():
    args = _parser().parse_args()
    if args.worker:
        _worker(args)
        return
    required = ("baseline_repo", "candidate_repo", "root", "pointcept_root", "output")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    _driver(args)


if __name__ == "__main__":
    main()
