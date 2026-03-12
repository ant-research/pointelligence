import torch
import torch.nn as nn
import time
import gc
import os
import sys
import numpy as np
import random
import datetime
import glob
from typing import Tuple

BATCH_SIZE = 2

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    from models import resnet18
    from layers.downsample import downsample
    POINTCNNPP_AVAILABLE = True
    print("PointCNNpp available")
except ImportError as e:
    print(f"PointCNNpp not available: {e}")
    POINTCNNPP_AVAILABLE = False

def load_real_pointcloud(coord_file, feature_dim=3, device='cuda'):
    coords = np.load(coord_file)
    if coords.shape[1] != 3:
        coords = coords[:, :3]
    
    num_points = coords.shape[0]
    features = np.random.randn(num_points, feature_dim).astype(np.float32)
    
    coords_tensor = torch.from_numpy(coords).float().to(device)
    features_tensor = torch.from_numpy(features).float().to(device)
    
    return coords_tensor, features_tensor

def load_batch_pointclouds(coord_files, feature_dim=3, device='cuda'):
    all_coords = []
    all_features = []
    batch_sizes = []
    
    for coord_file in coord_files:
        coords, features = load_real_pointcloud(coord_file, feature_dim, device)
        all_coords.append(coords)
        all_features.append(features)
        batch_sizes.append(coords.shape[0])
    
    coords = torch.cat(all_coords, dim=0)
    features = torch.cat(all_features, dim=0)
    batch_sizes = torch.tensor(batch_sizes, device=device, dtype=torch.long)
    
    return {
        'coords': coords,
        'features': features,
        'batch_sizes': batch_sizes.tolist(),
        'total_points': coords.shape[0],
        'feature_dim': feature_dim
    }

def test_resnet18(coords, features, grid_size=0.1, in_channels=64, device='cuda', warmup=True, sample_sizes=None, num_iters=10):
    if not POINTCNNPP_AVAILABLE:
        return None
    
    try:
        if sample_sizes is None:
            sample_sizes = torch.tensor([coords.shape[0]], device=device, dtype=torch.long)
        else:
            if isinstance(sample_sizes, list):
                sample_sizes = torch.tensor(sample_sizes, device=device, dtype=torch.long)
            else:
                sample_sizes = sample_sizes.to(device)
        
        original_num_points = coords.shape[0]

        sample_inds = torch.repeat_interleave(
            torch.arange(0, sample_sizes.numel(), device=device),
            sample_sizes
        )
        pre_downsample_grid_size = grid_size/2
        coords, sample_inds, _, indices = downsample(
            points=coords,
            sample_inds=sample_inds,
            grid_size=pre_downsample_grid_size,
            stride=1.0
        )
        features = features[indices]
        num_samples = sample_inds.max().item() + 1 if sample_inds.numel() > 0 else 0
        sample_sizes = torch.bincount(sample_inds, minlength=num_samples)
        
        model = resnet18(in_channels=in_channels).to(device)
        
        def forward_no_measure(x, points, sample_sizes, grid_size):
            with torch.no_grad():
                return model(x, points, sample_sizes, grid_size)
        
        def forward_backward_with_timing(x, points, sample_sizes, grid_size, measure_memory=False):
            """Execute forward + backward and return total time and peak memory"""
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            if measure_memory:
                torch.cuda.reset_peak_memory_stats()
            
            start.record()
            # Forward pass
            out = model(x, points, sample_sizes, grid_size)
            loss = out.sum()
            # Backward pass
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            
            total_time = start.elapsed_time(end)
            
            # Record peak memory
            peak_memory_gb = 0.0
            if measure_memory:
                peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Clear gradients
            model.zero_grad(set_to_none=True)
            
            return out, total_time, peak_memory_gb
        
        if warmup:
            try:
                model.train()
                features.requires_grad_(True)
                for i in range(3):
                    out, _, _ = forward_backward_with_timing(features, coords, sample_sizes, grid_size)
                    torch.cuda.synchronize()
                features.requires_grad_(False)
            except Exception as e:
                return None
        
        # 1. Forward-only timing (separate from memory)
        model.eval()
        forward_times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            forward_no_measure(features, coords, sample_sizes, grid_size)
            end_event.record()
            torch.cuda.synchronize()
            forward_times.append(start_event.elapsed_time(end_event))
        
        forward_times.sort()
        avg_forward = sum(forward_times[1:-1]) / len(forward_times[1:-1]) if len(forward_times) > 2 else (forward_times[0] if forward_times else 0.0)
        
        # 2. Forward + backward timing (separate from memory)
        model.train()
        features.requires_grad_(True)
        training_times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            out = model(features, coords, sample_sizes, grid_size)
            loss = out.sum()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            training_times.append(start_event.elapsed_time(end_event))
            model.zero_grad(set_to_none=True)
            features.detach_()
            features.requires_grad_(True)
        
        features.requires_grad_(False)
        training_times.sort()
        avg_training = sum(training_times[1:-1]) / len(training_times[1:-1]) if len(training_times) > 2 else (training_times[0] if training_times else 0.0)
        
        # 3. Forward + backward memory (separate from timing)
        model.train()
        features.requires_grad_(True)
        training_memories = []
        for _ in range(num_iters):
            out, total_time, peak_memory = forward_backward_with_timing(features, coords, sample_sizes, grid_size, measure_memory=True)
            training_memories.append(peak_memory)
            features.detach_()
            features.requires_grad_(True)
        
        features.requires_grad_(False)
        training_memories.sort()
        avg_training_memory = sum(training_memories[1:-1]) / len(training_memories[1:-1]) if len(training_memories) > 2 else (training_memories[0] if training_memories else 0.0)

        return {
            'inference_ms': avg_forward,
            'training_ms': avg_training,
            'input_size': original_num_points,
            'training_peak_memory_gb': avg_training_memory
        }
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_benchmark(sampledata_dir=None, device='cuda', grid_size=0.05):
    
    print(f"\n{'='*80}")
    print(f"Starting Benchmark Test (Device: {device})")
    print(f"Data Directory: {sampledata_dir}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(sampledata_dir):
        print(f"\nData directory does not exist: {sampledata_dir}")
        return None
    
    scale_bins = sorted([d for d in os.listdir(sampledata_dir) 
                        if d.startswith('scale_') and os.path.isdir(os.path.join(sampledata_dir, d))])
    
    if len(scale_bins) == 0:
        print(f"\nNo scale_* directories found")
        return None
    
    print(f"\nFound {len(scale_bins)} scale bins: {scale_bins}")
    
    all_results = []
    
    warmup_sizes = [50, 5000, 10000, 50000, 100000]
    
    if POINTCNNPP_AVAILABLE:
        for size in warmup_sizes:
            warmup_coords = torch.randn(size, 3, device=device)
            warmup_features = torch.randn(size, 64, device=device)
            warmup_sample_sizes = torch.tensor([size], device=device, dtype=torch.long)
            _ = test_pointcnnpp_resnet18(warmup_coords, warmup_features, device=device, warmup=False, grid_size=grid_size, sample_sizes=warmup_sample_sizes)
            torch.cuda.empty_cache()
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    for scale_idx, scale_bin in enumerate(scale_bins):
        print(f"\n{'='*80}")
        print(f"Scale Bin [{scale_idx+1}/{len(scale_bins)}]: {scale_bin}")
        print(f"{'='*80}")
        
        bin_dir = os.path.join(sampledata_dir, scale_bin)
        coord_files = glob.glob(os.path.join(bin_dir, '*_coord.npy'))
        
        if len(coord_files) == 0:
            print(f"  No point cloud files found, skipping")
            continue
        
        print(f"  Found {len(coord_files)} point cloud files")
        
        batch_files = coord_files[:BATCH_SIZE]
        
        batch_data = load_batch_pointclouds(batch_files, device=device)
        
        total_points = batch_data['total_points']
        feature_dim = batch_data['feature_dim']
        batch_sizes = batch_data['batch_sizes']
        
        print(f"  Points: {total_points:,}, Feature Dim: {feature_dim}")
        
        coords = batch_data['coords']
        features = batch_data['features']
        sample_sizes = torch.tensor(batch_sizes, device=device, dtype=torch.long)
        
        results = {
            'config_name': scale_bin,
            'num_clouds': len(batch_files),
            'points_per_cloud': total_points,
            'total_points': total_points,
            'original_points': total_points,
            'batch_sizes': batch_sizes,
            'grid_size': grid_size,
            'in_channels': feature_dim,
            'data_source': 'real',
            'scale_bin': scale_bin,
            'test_timestamp': datetime.datetime.now().isoformat(),
        }
        
        print(f"\n    Testing PointCNNpp ResNet18...")
        if POINTCNNPP_AVAILABLE:
            try:
                pointcnnpp_result = test_pointcnnpp_resnet18(
                    coords, features, grid_size, feature_dim,
                    device, warmup=True, sample_sizes=sample_sizes
                )
                if pointcnnpp_result:
                    results['pointcnnpp'] = pointcnnpp_result
                    print(f"      Completed - Inference: {pointcnnpp_result['inference_ms']:.2f}ms, "
                          f"Training: {pointcnnpp_result['training_ms']:.2f}ms, "
                          f"Training Memory: {pointcnnpp_result['training_peak_memory_gb']:.2f}GB")
                else:
                    print(f"      Failed: returned None")
                    results['pointcnnpp'] = None
            except Exception as e:
                print(f"      Failed: {e}")
                import traceback
                traceback.print_exc()
                results['pointcnnpp'] = None
        else:
            print("      Not available")
            results['pointcnnpp'] = None
        
        all_results.append(results)
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    return all_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PointCNNpp ResNet18 Benchmark')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Point cloud data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Compute device (cuda or cpu)')
    parser.add_argument('--grid_size', type=float, default=0.1,
                       help='Grid size')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*80)
    print("PointCNNpp ResNet18 Architecture Benchmark")
    print("="*80)
    print(f"Device: {device}")
    print(f"Data Directory: {args.data_dir if args.data_dir else 'Using synthetic data'}")
    print("="*80 + "\n")
    
    results = run_benchmark(sampledata_dir=args.data_dir, device=device, grid_size=args.grid_size)
    
    if results is None or len(results) == 0:
        print("\nBenchmark failed")
        return 1
    
    # Print statistics table
    print("\n" + "="*120)
    print("Performance Statistics Summary (Averaged across all input points)")
    print("="*120)
    
    # Collect valid results
    table_data = []
    for result in results:
        if result.get('pointcnnpp') is not None:
            pointcnnpp_data = result['pointcnnpp']
            table_data.append({
                'input_points': result.get('total_points', result.get('original_points', 0)),
                'inference_ms': pointcnnpp_data.get('inference_ms', 0.0),
                'training_ms': pointcnnpp_data.get('training_ms', 0.0),
                'training_memory_gb': pointcnnpp_data.get('training_peak_memory_gb', 0.0),
            })
    
    if len(table_data) == 0:
        print("No valid results to display")
    else:
        # Calculate averages across all input points
        avg_input_points = sum(d['input_points'] for d in table_data) / len(table_data)
        avg_inference_ms = sum(d['inference_ms'] for d in table_data) / len(table_data)
        avg_training_ms = sum(d['training_ms'] for d in table_data) / len(table_data)
        avg_training_memory_gb = sum(d['training_memory_gb'] for d in table_data) / len(table_data)
        
        # Print table header
        header = f"{'Input Points':<15} {'Inference (ms)':<18} {'Training (ms)':<18} {'Training Memory (GB)':<25}"
        print(header)
        print("-" * 100)
        
        # Print averaged row
        row = (f"{avg_input_points:<15,.0f} "
               f"{avg_inference_ms:<18.2f} "
               f"{avg_training_ms:<18.2f} "
               f"{avg_training_memory_gb:<25.2f}")
        print(row)
        
        print("="*100)
    
    print("\n" + "="*80)
    print("Benchmark completed!")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

