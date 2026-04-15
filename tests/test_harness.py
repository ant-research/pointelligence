"""
Shared test harness for PointCNN++ benchmarks.

Provides:
- TestMode: dtype x compile mode configuration
- Synthetic data generation for CI/quick validation
- Output validation (NaN/inf check, reference comparison)
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class TestMode:
    dtype: torch.dtype
    compile: bool
    label: str

    @property
    def dtype_str(self):
        return "fp16" if self.dtype == torch.float16 else "fp32"

    @property
    def compile_str(self):
        return "compiled" if self.compile else "eager"


# Standard 4-mode matrix
FP32_EAGER = TestMode(torch.float32, False, "fp32-eager")
FP32_COMPILED = TestMode(torch.float32, True, "fp32-compiled")
FP16_EAGER = TestMode(torch.float16, False, "fp16-eager")
FP16_COMPILED = TestMode(torch.float16, True, "fp16-compiled")

ALL_MODES = [FP32_EAGER, FP32_COMPILED, FP16_EAGER, FP16_COMPILED]

# Tolerances for cross-mode comparison (compared against fp32-eager reference)
TOLERANCES = {
    "fp32-eager": {"atol": 0.0, "rtol": 0.0},  # reference
    "fp32-compiled": {"atol": 1e-5, "rtol": 1e-5},
    "fp16-eager": {"atol": 5e-2, "rtol": 5e-2},
    "fp16-compiled": {"atol": 5e-2, "rtol": 5e-2},
}


def resolve_modes(dtype_arg: str, compile_arg: str) -> List[TestMode]:
    """Resolve CLI args into list of TestMode."""
    dtypes = [torch.float32, torch.float16] if dtype_arg == "all" else \
             [torch.float16] if dtype_arg == "fp16" else [torch.float32]
    compiles = [False, True] if compile_arg == "all" else \
               [True] if compile_arg == "compiled" else [False]

    modes = []
    for d in dtypes:
        for c in compiles:
            label = f"{'fp16' if d == torch.float16 else 'fp32'}-{'compiled' if c else 'eager'}"
            modes.append(TestMode(d, c, label))
    return modes


def generate_synthetic_pointcloud(num_points: int, feature_dim: int = 64,
                                  dtype: torch.dtype = torch.float32,
                                  device: str = 'cuda') -> Dict:
    """Generate a single synthetic point cloud with random coordinates in [0, 1]^3."""
    coords = torch.rand(num_points, 3, device=device, dtype=torch.float32)  # coords always fp32
    features = torch.randn(num_points, feature_dim, device=device, dtype=dtype)
    return {
        'coords': coords,
        'features': features,
        'num_points': num_points,
        'feature_dim': feature_dim,
    }


def generate_synthetic_batch(num_points_per_cloud: int, batch_size: int,
                             feature_dim: int = 64,
                             dtype: torch.dtype = torch.float32,
                             device: str = 'cuda') -> Dict:
    """Generate a batch of synthetic point clouds."""
    all_coords = []
    all_features = []
    batch_sizes = []

    for _ in range(batch_size):
        # Vary point count slightly per cloud for realism
        n = num_points_per_cloud + np.random.randint(-num_points_per_cloud // 10,
                                                      num_points_per_cloud // 10 + 1)
        n = max(100, n)
        pc = generate_synthetic_pointcloud(n, feature_dim, dtype, device)
        all_coords.append(pc['coords'])
        all_features.append(pc['features'])
        batch_sizes.append(n)

    return {
        'coords': torch.cat(all_coords, dim=0),
        'features': torch.cat(all_features, dim=0),
        'batch_sizes': batch_sizes,
        'total_points': sum(batch_sizes),
        'feature_dim': feature_dim,
    }


def validate_output(tensor: torch.Tensor, label: str,
                    reference: Optional[torch.Tensor] = None,
                    atol: Optional[float] = None,
                    rtol: Optional[float] = None) -> Dict:
    """Validate a tensor for NaN/inf and optionally compare against reference.

    Returns dict with validation results.
    """
    result = {
        'label': label,
        'has_nan': bool(torch.isnan(tensor).any()),
        'has_inf': bool(torch.isinf(tensor).any()),
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'max_abs': float(tensor.abs().max()) if tensor.numel() > 0 else 0.0,
    }

    if result['has_nan'] or result['has_inf']:
        nan_count = int(torch.isnan(tensor).sum())
        inf_count = int(torch.isinf(tensor).sum())
        result['error'] = f"NaN={nan_count}, Inf={inf_count} in {tensor.numel()} elements"

    if reference is not None and not result['has_nan'] and not result['has_inf']:
        ref = reference.float()
        val = tensor.float()
        diff = (ref - val).abs()
        result['max_diff'] = float(diff.max())
        result['mean_diff'] = float(diff.mean())

        if atol is not None and rtol is not None:
            close = torch.allclose(val, ref, atol=atol, rtol=rtol)
            result['within_tolerance'] = bool(close)
            if not close:
                result['tolerance_error'] = (
                    f"max_diff={result['max_diff']:.6f} "
                    f"(atol={atol}, rtol={rtol})"
                )

    return result


def compile_model(model_or_fn, fullgraph=False):
    """Wrap a model or function with torch.compile, with appropriate settings."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 256
    return torch.compile(model_or_fn, fullgraph=fullgraph)


def add_mode_args(parser):
    """Add --dtype and --compile arguments to an argparse parser."""
    parser.add_argument('--dtype', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'all'],
                        help='Data type for features/weights')
    parser.add_argument('--compile', type=str, default='eager',
                        choices=['eager', 'compiled', 'all'],
                        help='Execution mode')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of real data')
    parser.add_argument('--synthetic_points', type=int, default=10000,
                        help='Points per cloud for synthetic data')


def print_mode_comparison_table(mode_results: List[Dict]):
    """Print a comparison table across modes."""
    if not mode_results:
        return

    print(f"\n{'='*100}")
    print("Cross-Mode Comparison")
    print(f"{'='*100}")
    header = f"{'Mode':<20} {'Forward ms':<12} {'Backward ms':<13} {'Memory GB':<11} {'Valid':<8} {'Max Diff':<12}"
    print(header)
    print("-" * 100)

    for r in mode_results:
        label = r.get('mode_label', 'unknown')
        fwd = r.get('forward_ms', 0)
        bwd = r.get('backward_ms', 0)
        mem = r.get('memory_gb', 0)
        valid = r.get('valid', 'N/A')
        max_diff = r.get('max_diff', None)
        diff_str = f"{max_diff:.6f}" if max_diff is not None else "ref"
        print(f"{label:<20} {fwd:<12.2f} {bwd:<13.2f} {mem:<11.3f} {str(valid):<8} {diff_str:<12}")

    print(f"{'='*100}")
