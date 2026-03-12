import torch
import torch.nn as nn
import time
import gc
import os
import sys
import numpy as np
import random
import glob
from functools import partial

if 'CUDA_LAUNCH_BLOCKING' not in os.environ:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    from layers.conv import PointConv3d
    from layers.metadata import MetaData
    from layers.downsample import downsample
    POINTCNN_AVAILABLE = True
    print("✓ pointcnn available")
except ImportError as e:
    print(f"pointcnn unavailable: {e}")
    POINTCNN_AVAILABLE = False
    downsample = None
    repeat_interleave_indices = None


def global_warmup(device='cuda', num_iterations=3):
    if device != 'cuda' or not torch.cuda.is_available():
        return
    
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        for _ in range(num_iterations):
            temp_tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(temp_tensor, temp_tensor)
            result.sum().item()
            del temp_tensor, result
        
        torch.cuda.synchronize()
        
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Global warmup warning: {e}")


def test_convolution(coords, features, voxel_size=0.025, in_channels=128, out_channels=128, kernel_size=3, device='cuda', warmup=True, batch_indices=None, batch_sizes=None, num_iters=5):
    if not POINTCNN_AVAILABLE:
        return None
    
    global_warmup(device=device)
    
    try:
        import time
        import torch
        import torch.nn as nn
        import gc
        
        if batch_sizes is not None:
            sample_sizes = torch.tensor(batch_sizes, device=device, dtype=torch.long)
        else:
            sample_sizes = torch.tensor([coords.shape[0]], device=device, dtype=torch.long)
        
        sample_inds = torch.repeat_interleave(
            torch.arange(0, sample_sizes.numel(), device=device),
            sample_sizes
        )
        sampled_coords, sampled_sample_inds, new_grid_size, _ = downsample(
            points=coords,
            sample_inds=sample_inds,
            grid_size=voxel_size,
            stride=1.0
        )
        sampled_features = features[sampled_sample_inds]
        sampled_sample_sizes = torch.bincount(sampled_sample_inds)
        
        actual_in_channels = sampled_features.shape[1] if len(sampled_features.shape) > 1 else in_channels
        
        kernel_size_3 = (kernel_size, kernel_size, kernel_size)
        
        def build_metadata():
            return MetaData(
                points=sampled_coords,
                sample_inds=sampled_sample_inds,
                sample_sizes=sampled_sample_sizes,
                grid_size=new_grid_size,
                kernel_size=kernel_size_3,
                sort_by="k"
            )
        
        conv = PointConv3d(
            actual_in_channels, out_channels, kernel_size=kernel_size_3
        ).to(device)

        test_metadata = build_metadata()
        try:
            with torch.no_grad():
                output = conv(sampled_features, test_metadata.i, test_metadata.j, test_metadata.k, test_metadata.num_points())
        except Exception as e:
            return None
        finally:
            if 'output' in locals():
                del output
            if 'test_metadata' in locals():
                del test_metadata
            torch.cuda.empty_cache()
            gc.collect()

        if warmup:
            for i in range(3):
                try:
                    warmup_metadata = build_metadata()
                    conv.eval()
                    with torch.no_grad():
                        output = conv(sampled_features, warmup_metadata.i, warmup_metadata.j, warmup_metadata.k, warmup_metadata.num_points())
                    del output, warmup_metadata
                    
                    warmup_metadata = build_metadata()
                    conv.train()
                    feat_grad = sampled_features.clone().detach().requires_grad_(True)
                    output = conv(feat_grad, warmup_metadata.i, warmup_metadata.j, warmup_metadata.k, warmup_metadata.num_points())
                    loss = output.sum()
                    loss.backward()
                    conv.zero_grad()
                    del output, loss, feat_grad, warmup_metadata
                    
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    pass

        forward_times = []
        for i in range(num_iters):
            torch.cuda.synchronize()
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            iter_metadata = build_metadata()
            feat_iter = sampled_features.clone().detach()
            total_start.record()

            with torch.no_grad():
                output = conv(feat_iter, iter_metadata.i, iter_metadata.j, iter_metadata.k, iter_metadata.num_points())
                
            total_end.record()
            torch.cuda.synchronize()
            total_time = total_start.elapsed_time(total_end)
            
            forward_times.append(total_time)
            del output, feat_iter, iter_metadata
            torch.cuda.empty_cache()
            gc.collect()

        backward_times = []
        for i in range(num_iters):
            try:
                torch.cuda.synchronize()
                backward_start = torch.cuda.Event(enable_timing=True)
                backward_end = torch.cuda.Event(enable_timing=True)
                iter_metadata = build_metadata()
                feat_grad = sampled_features.clone().detach().requires_grad_(True)
                backward_start.record()

                output = conv(feat_grad, iter_metadata.i, iter_metadata.j, iter_metadata.k, iter_metadata.num_points())
                loss = output.sum()
                loss.backward()
                
                backward_end.record()
                torch.cuda.synchronize()
                backward_time = backward_start.elapsed_time(backward_end)
                
                conv.zero_grad()
                
                backward_times.append(backward_time)
                del output, loss, feat_grad, iter_metadata
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                backward_times.append(0.0)

        forward_avg = sum(forward_times) / len(forward_times)
        valid_backward_times = [t for t in backward_times if t > 0]
        backward_avg = sum(valid_backward_times) / len(valid_backward_times) if valid_backward_times else 0.0

        forward_memory_gb = 0.0
        backward_memory_gb = 0.0
        
        if device == 'cuda' and torch.cuda.is_available():
            forward_metadata = build_metadata()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            conv.eval()
            feat_forward = sampled_features.clone().detach()
            with torch.no_grad():
                out_forward = conv(feat_forward, forward_metadata.i, forward_metadata.j, forward_metadata.k, forward_metadata.num_points())
            forward_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            del out_forward, feat_forward, forward_metadata
            torch.cuda.empty_cache()
            
            backward_metadata = build_metadata()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            conv.train()
            feat_backward = sampled_features.clone().detach().requires_grad_(True)
            out_backward = conv(feat_backward, backward_metadata.i, backward_metadata.j, backward_metadata.k, backward_metadata.num_points())
            loss_backward = out_backward.sum()
            loss_backward.backward()
            backward_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            conv.zero_grad()
            del out_backward, loss_backward, feat_backward, backward_metadata
            torch.cuda.empty_cache()

        num_points = sampled_coords.shape[0]
        
        print(f"\n[pointcnn_SingleConv] Input Points: {num_points/1e3:.1f}K")
        print(f"Forward: {forward_avg:.2f}ms, Backward: {backward_avg:.2f}ms")
        print(f"Forward Memory: {forward_memory_gb:.2f}GB, Backward Memory: {backward_memory_gb:.2f}GB")

        return {
            'forward_ms': forward_avg,
            'backward_ms': backward_avg,
            'forward_memory_gb': forward_memory_gb,
            'backward_memory_gb': backward_memory_gb,
            'memory_gb': backward_memory_gb,
            'input_size': num_points,
        }
        
    except Exception as e:
        print(f"pointcnn test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_real_pointcloud(coord_file, feature_dim=64, device='cuda'):
    coords = np.load(coord_file)
    num_points = coords.shape[0]
    
    coords_tensor = torch.from_numpy(coords).float().to(device)
    
    color_file = coord_file.replace('_coord.npy', '_color.npy')
    if os.path.exists(color_file):
        try:
            colors = np.load(color_file)
            if colors.shape[0] == num_points:
                features = torch.from_numpy(colors).float().to(device)
                feature_dim = colors.shape[1] if len(colors.shape) > 1 else 3
            else:
                features = torch.randn(num_points, feature_dim, device=device)
        except:
            features = torch.randn(num_points, feature_dim, device=device)
    else:
        features = torch.randn(num_points, feature_dim, device=device)
    
    return {
        'coords': coords_tensor,
        'features': features,
        'num_points': num_points,
        'feature_dim': features.shape[1] if len(features.shape) > 1 else feature_dim
    }


def load_batch_pointclouds(coord_files, feature_dim=64, device='cuda'):
    batch_coords = []
    batch_features = []
    batch_sizes = []
    batch_indices = []
    total_points = 0
    successful_loads = 0
    
    for batch_idx, coord_file in enumerate(coord_files):
        try:
            data = load_real_pointcloud(coord_file, feature_dim=feature_dim, device=device)
            num_points = data['num_points']
            
            if num_points == 0:
                print(f"    ⚠  Skipping empty file: {os.path.basename(coord_file)}")
                continue
            
            batch_coords.append(data['coords'])
            batch_features.append(data['features'])
            batch_sizes.append(num_points)
            
            batch_idx_tensor = torch.full((num_points,), successful_loads, dtype=torch.int32, device=device)
            batch_indices.append(batch_idx_tensor)
            
            total_points += num_points
            successful_loads += 1
            
        except Exception as e:
            print(f"Failed to load file {os.path.basename(coord_file)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(batch_coords) == 0:
        raise RuntimeError(f"Failed to load any point cloud files (total {len(coord_files)} files)")
    
    all_coords = torch.cat(batch_coords, dim=0)
    all_features = torch.cat(batch_features, dim=0)
    all_batch_indices = torch.cat(batch_indices, dim=0)
    
    unified_feature_dim = all_features.shape[1] if len(all_features.shape) > 1 else feature_dim
    
    return {
        'coords': all_coords,
        'features': all_features,
        'batch_indices': all_batch_indices,
        'batch_sizes': batch_sizes,
        'total_points': total_points,
        'num_batches': successful_loads,
        'feature_dim': unified_feature_dim
    }


def run_benchmark(sampledata_dir=None, device='cuda', voxel_size=0.05, in_channels=64, out_channels=128, kernel_size=3, batch_size=4, point_scale_factor=1.0, random_seed=None):
    
    print(f"\n{'='*80}")
    print(f"Starting single convolution layer benchmark test (Device: {device}, Batch Size: {batch_size})")
    print(f"Data Directory: {sampledata_dir}")
    print(f"Kernel Size: {kernel_size}, In Channels: {in_channels}, Out Channels: {out_channels}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(sampledata_dir):
        print(f"\n Data directory does not exist: {sampledata_dir}")
        return None
    
    scale_bins = sorted([d for d in os.listdir(sampledata_dir) 
                        if d.startswith('scale_') and os.path.isdir(os.path.join(sampledata_dir, d))])
    
    if len(scale_bins) == 0:
        print(f"\n No scale_* directories found")
        return None
    
    scale_bins = scale_bins[:9]
    print(f"\nFound {len(scale_bins)} scale bins (testing first 9): {scale_bins}")
    
    all_results = []
    
    for scale_idx, scale_bin in enumerate(scale_bins):
        print(f"\n{'='*80}")
        print(f"Scale bin [{scale_idx+1}/{len(scale_bins)}]: {scale_bin}")
        print(f"{'='*80}")
        
        bin_dir = os.path.join(sampledata_dir, scale_bin)
        coord_files = glob.glob(os.path.join(bin_dir, '*_coord.npy'))
        
        coord_files = sorted(coord_files)
        
        print(f"  Found {len(coord_files)} point cloud files in directory {bin_dir}")
        
        if len(coord_files) == 0:
            print(f"No point cloud files found, skipping")
            continue
        
        num_files_to_load = min(batch_size, len(coord_files))
        batch_files = coord_files[:num_files_to_load]
        
        print(f"\n  Preparing to load point cloud files (batch_size={num_files_to_load}, total {len(coord_files)} files in directory):")
        for i, f in enumerate(batch_files):
            print(f"    [{i+1}/{num_files_to_load}] {os.path.basename(f)}")
        
        try:
            batch_data = load_batch_pointclouds(batch_files, device=device, feature_dim=in_channels)
        except Exception as e:
            print(f"Failed to load point cloud data: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        total_points = batch_data['total_points']
        feature_dim = batch_data['feature_dim']
        batch_sizes = batch_data['batch_sizes']
        num_batches = batch_data['num_batches']
        
        print(f"\n  Batch info: Successfully loaded {num_batches} point clouds, total points: {total_points:,}, feature dimension: {feature_dim}")
        print(f"  Point counts per cloud: {batch_sizes}")
        
        coords = batch_data['coords']
        features = batch_data['features']
        batch_indices = batch_data['batch_indices']
        
        if point_scale_factor > 1.0:
            original_num_points = coords.shape[0]
            target_num_points = int(original_num_points * point_scale_factor)
            
            if target_num_points > original_num_points:
                if random_seed is not None:
                    np.random.seed(random_seed + scale_idx)
                else:
                    dynamic_seed = RANDOM_SEED + scale_idx + int(time.time() * 1000) % 100000
                    np.random.seed(dynamic_seed)
                indices = np.random.choice(original_num_points, target_num_points, replace=True)
                indices = np.sort(indices)
                
                coords = coords[indices]
                features = features[indices]
                
                if batch_indices is not None:
                    if isinstance(batch_indices, torch.Tensor):
                        original_batch_indices = batch_indices.detach().cpu().numpy()
                        batch_device = batch_indices.device
                    else:
                        original_batch_indices = np.asarray(batch_indices)
                        batch_device = torch.device('cpu')
                    
                    np_dtype = original_batch_indices.dtype
                    new_batch_indices_np = np.zeros(target_num_points, dtype=np_dtype)
                    
                    current_idx = 0
                    cumulative_sizes = np.cumsum([0] + batch_sizes)
                    for batch_idx, batch_size_val in enumerate(batch_sizes):
                        start_idx = cumulative_sizes[batch_idx]
                        end_idx = cumulative_sizes[batch_idx + 1]
                        
                        batch_mask = (indices >= start_idx) & (indices < end_idx)
                        batch_count = np.sum(batch_mask)
                        
                        if batch_count > 0:
                            new_batch_indices_np[current_idx:current_idx + batch_count] = batch_idx
                            current_idx += batch_count
                    
                    if current_idx < target_num_points:
                        new_batch_indices_np[current_idx:] = new_batch_indices_np[max(0, current_idx - 1)]
                    
                    batch_indices = torch.from_numpy(new_batch_indices_np).to(batch_device)
                    batch_sizes = [max(1, int(round(bs * point_scale_factor))) for bs in batch_sizes]
                
                print(f"  [Point cloud scaling] Sampling from {original_num_points:,} points to {target_num_points:,} points (scale factor: {point_scale_factor:.2f}x)")
                total_points = target_num_points
        
        if point_scale_factor is not None and abs(point_scale_factor - 1.0) > 1e-6:
            config_label = f"{scale_bin}_bs{batch_size}_scale{point_scale_factor:.2f}"
        else:
            config_label = f"{scale_bin}_bs{batch_size}"

        results = {
            'config_name': config_label,
            'total_points': total_points,
            'voxel_size': voxel_size,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'data_source': 'real',
            'scale_bin': scale_bin,
            'point_scale_factor': float(point_scale_factor) if point_scale_factor is not None else 1.0,
        }
        
        print(f"\n{'-'*80}")
        print(f"Single convolution layer test")
        print(f"{'-'*80}")
        
        print(f"\n    Testing pointcnn...")
        if POINTCNN_AVAILABLE:
            try:
                po_result = test_convolution(
                    coords, features, voxel_size, in_channels, out_channels, kernel_size,
                    device, warmup=True, batch_indices=batch_indices, batch_sizes=batch_sizes
                )
                if po_result:
                    results['pointops'] = po_result
                    print(f"      ✓ Complete - Forward: {po_result['forward_ms']:.2f}ms, "
                          f"Backward: {po_result['backward_ms']:.2f}ms")
            except Exception as e:
                print(f"      ✗ Failed: {e}")
                results['pointops'] = None
        else:
            print("      ⚠ Unavailable")
            results['pointops'] = None
        
        all_results.append(results)
        
        _clear_cuda_memory()
    
    print(f"\n{'='*80}")
    print(f"Test completed - Results summary")
    print(f"{'='*80}\n")
    
    print(f"\n{'='*120}")
    print(f"pointcnn Single Convolution Layer Performance Comparison Table")
    print(f"{'='*120}")
    
    header = (
        f"{'Config':<30} {'Method':<20} {'Input':<12} {'Forward':<10} {'Backward':<10} {'Memory':<8}"
    )
    print(header)
    print(f"{'-'*120}")
    
    for result in all_results:
        config_name = result['config_name']
        voxel_size_val = result['voxel_size']
        config_str = f"{config_name} (v={voxel_size_val})"
        
        methods = [
            ('pointcnn', 'pointops', 'Points'),
        ]
        
        for method_name, method_key, unit_type in methods:
            if method_key in result and result[method_key] is not None:
                res = result[method_key]
                input_size = res.get('input_size', 0)
                input_str = f"{input_size/1000:.1f}K {unit_type}"
                forward = res.get('forward_ms', 0)
                backward = res.get('backward_ms', 0)
                memory = res.get('memory_gb', 0)
                
                row = (
                    f"{config_str:<30} {method_name:<20} {input_str:<12} "
                    f"{forward:>8.2f}ms {backward:>8.2f}ms {memory:>6.2f}GB"
                )
                print(row)
            else:
                row = (
                    f"{config_str:<30} {method_name:<20} {'N/A':<12} "
                    f"{'N/A':>10} {'N/A':>10} {'N/A':>8}"
                )
                print(row)
        
        print(f"{'-'*120}")
    
    return all_results


def _clear_cuda_memory():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
    gc.collect()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='pointcnn Single Convolution Benchmark')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Real point cloud data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Compute device (cuda or cpu)')
    parser.add_argument('--voxel_size', type=float, default=0.025,
                       help='Voxel size')
    parser.add_argument('--in_channels', type=int, default=128,
                       help='Input channels')
    parser.add_argument('--out_channels', type=int, default=128,
                       help='Output channels')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='Kernel size')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*80)
    print("pointcnn Single Convolution Layer Benchmark")
    print("="*80)
    print(f"Device: {device}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Convolution Config: {args.in_channels} -> {args.out_channels}, kernel_size={args.kernel_size}")
    print("="*80 + "\n")
    
    all_results = []
    
    print("\n" + "="*80)
    print("="*80)
    print(f"Test: Batch Size = {args.batch_size}")
    print("="*80)
    print("="*80 + "\n")
    
    results = run_benchmark(
        sampledata_dir=args.data_dir, 
        device=device, 
        voxel_size=args.voxel_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        batch_size=args.batch_size
    )
    
    if results is not None and len(results) > 0:
        for r in results:
            r['batch_size'] = args.batch_size
        all_results.extend(results)
        print(f"\n✓ Batch Size={args.batch_size} test completed, {len(results)} scales")
    else:
        print(f"\n⚠ Batch Size={args.batch_size} test produced no results")
    
    if len(all_results) == 0:
        print("\n✗ Benchmark failed - no results produced")
        return 1
    
    print("\n" + "="*80)
    print("="*80)
    print(f"✓ Benchmark completed! Total {len(all_results)} test configurations completed")
    print("="*80)
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())