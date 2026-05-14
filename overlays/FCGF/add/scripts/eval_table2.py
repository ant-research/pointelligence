"""
eval_table2.py
--------------
Paper-canonical reproduction of PointCNN++ Table 2 on the 3DMatch test set
(1623 OverlapPredator-curated pairs across 8 scenes).

Correspondence protocol:
    * Mutual NN: keep correspondence (i, j) iff F0[i]'s NN in F1 is j AND
      F1[j]'s NN in F0 is i (bidirectional check).
    * Subsample mode (--corr_sampling, default 'random' to match paper
      Table 2 per Lihan 2026-05-08; 'topk' available for Lihan's
      test.py reproduction):
        - random         → uniform random pick of K mutual matches (paper)
        - topk           → K with smallest forward-NN feat dist (test.py)
        - score_weighted → prob ∝ 1/normalized(feat_dist) (FCGF trainer)
    * RR: RANSAC over the K correspondences (50k iter, conf 0.999),
      Predator-canonical RMSE on full src cloud ≤ 0.2 m success criterion.
    * RANSAC distance_threshold = voxel_size × 1.5 (Lihan 2026-05-07; the
      test.py default of × 1.0 is too tight and depresses RR by ~12 pp).
    * τ_inlier = 0.3 m (Lihan 2026-05-08 confirmed paper-canonical for
      Table 2 IR/FMR; FCGF/Predator literature uses 0.1 m which produces
      ~14 pp lower IR).

History (kept for context):
    * Pre-2026-05-07: one-way NN + score-weighted random (incorrect
      reading of Lihan's 2026-05-06 reply).
    * 2026-05-07: switched to mutual NN + top-K after `test_lihan.py`
      reproduced paper Table 2 IR/FMR within tolerance at N=5000.
    * 2026-05-07/08: Lihan confirmed three protocol fixes — τ_inlier=0.3
      (not 0.1), RANSAC dist_threshold=voxel_size×1.5 (not ×1.0), and
      sampling=random (not top-K, which she calls "fake increase" at
      small N because top-K picks the most-confident matches).
    * 2026-05-08: added --corr_sampling flag with default 'random'
      to match paper, while preserving 'topk' for test.py-reproduction
      cross-check.

Usage:
    python scripts/eval_table2.py \\
        --checkpoint /path/to/ResUNetBN2C-3DMatch.pth \\
        --threedmatch_root /path/to/3dmatch_processed/indoor \\
        --pair_list configs/indoor/3DMatch.pkl \\
        --output_log /tmp/eval_table2.json
"""
import os
import sys
import argparse
import json
import logging
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import open3d as o3d

# Make sure we can import from the FCGF package root.
_FCGF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FCGF_ROOT not in sys.path:
    sys.path.insert(0, _FCGF_ROOT)

from model import load_model
from lib.data_loaders import ThreeDMatchNewPairDatasetPure
from lib.eval import find_nn_gpu


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# Paper Table 2 row "Ours" (PointCNN++ arXiv:2511.23227 v3, percentages).
PAPER_TABLE2 = {
    5000: {'RR': 90.3, 'FMR': 98.9, 'IR': 58.2},
    2500: {'RR': 90.2, 'FMR': 99.1, 'IR': 57.8},
    1000: {'RR': 89.2, 'FMR': 99.1, 'IR': 57.3},
    500:  {'RR': 89.1, 'FMR': 98.4, 'IR': 52.1},
    250:  {'RR': 88.3, 'FMR': 99.2, 'IR': 53.4},
}
TOL_RR = 2.0
TOL_FMR = 2.0
TOL_IR = 5.0


# ──────────────────────────────────────────────────────────────
# Helpers mirrored from lib/trainer.py
# ──────────────────────────────────────────────────────────────

def apply_transform(pts, trans):
    """Apply a 4x4 rigid transform to a (N,3) numpy array."""
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.T + T


def extract_features_from_output(model_output):
    if isinstance(model_output, dict):
        for k in ('feat', 'F'):
            if k in model_output:
                return model_output[k]
        raise ValueError(f"dict output missing 'feat'/'F': keys={list(model_output)}")
    if hasattr(model_output, 'F'):
        return model_output.F
    if hasattr(model_output, 'feat'):
        return model_output.feat
    if hasattr(model_output, 'shape'):
        return model_output
    raise ValueError(f"Cannot extract features from type: {type(model_output)}")


def extract_coords_from_output(model_output, fallback_coords):
    """Mirror lib/trainer.py:100-120. Pull model output coords if present,
    else fall back to input. Returns CPU torch tensor or numpy array."""
    if isinstance(model_output, dict) and 'coord' in model_output:
        coord = model_output['coord']
        return coord.cpu() if hasattr(coord, 'cpu') else coord
    if hasattr(type(model_output), 'coord'):
        coord = model_output.coord
        return coord.cpu() if hasattr(coord, 'cpu') else coord
    return fallback_coords


def make_o3d_pcd(xyz_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np.astype(np.float64))
    return pcd


def find_corr_mutual(F0, F1, device, nn_max_n=500, subsample_size=5000,
                     sampling='random', rng=None):
    """
    Mutual NN + per-sampling-mode correspondence selection. Returns
    (src_idx, tgt_idx) 1-D numpy arrays of length min(M_mutual, subsample_size).

    Ports `find_corr(mutual=True)` from `debug_assets/test.py:68-146` while
    returning indices instead of xyz coords (so the caller can use the
    same indices for both IR/FMR computation and RANSAC-on-correspondences).

    Algorithm:
        1. Forward NN: for each i in F0, find j = argmin_j ||F0[i] - F1[j]||²
           (with squared-L2 distances).
        2. Backward NN: for each j in F1, find i' = argmin_{i'} ||F1[j] - F0[i']||².
        3. Mutual mask: keep i iff backward-NN(forward-NN(i)) == i.
        4. Subsample per `sampling` mode:
           - 'random'  → uniform random pick (paper Table 2 default; Lihan
                        2026-05-08: top-K is unfair to other baselines and
                        inflates small-N IR — "fake increase").
           - 'topk'    → argsort by forward-NN feat distance, take K smallest
                        (Lihan's `test.py:132` default; was Predator-style;
                        Lihan considers it Agent-style and unfair).
           - 'score_weighted' → prob ∝ 1/normalized(feat_dist), sample without
                        replacement (FCGF lib/trainer.py:594-616 trick).
    """
    F0_t = torch.from_numpy(F0).float().to(device) if not isinstance(F0, torch.Tensor) else F0
    F1_t = torch.from_numpy(F1).float().to(device) if not isinstance(F1, torch.Tensor) else F1

    # Forward NN (F0 -> F1) with distances.
    nn_result_01 = find_nn_gpu(F0_t, F1_t, nn_max_n=nn_max_n,
                               return_distance=True, dist_type='SquareL2')
    if isinstance(nn_result_01, tuple):
        nn_inds_01, nn_dists_01 = nn_result_01
    else:
        nn_inds_01 = nn_result_01
        nn_dists_01 = None

    if isinstance(nn_inds_01, torch.Tensor):
        nn_inds_01 = nn_inds_01.cpu().numpy()
    if nn_dists_01 is not None and isinstance(nn_dists_01, torch.Tensor):
        nn_dists_01 = nn_dists_01.cpu().numpy()
    if nn_dists_01 is not None:
        nn_dists_01 = np.asarray(nn_dists_01).squeeze()
        if nn_dists_01.ndim == 0:
            nn_dists_01 = nn_dists_01.reshape(1)
        if nn_dists_01.ndim > 1:
            nn_dists_01 = nn_dists_01.flatten()

    # Backward NN (F1 -> F0), distance not needed.
    nn_result_10 = find_nn_gpu(F1_t, F0_t, nn_max_n=nn_max_n,
                               return_distance=False, dist_type='SquareL2')
    if isinstance(nn_result_10, tuple):
        nn_inds_10 = nn_result_10[0]
    else:
        nn_inds_10 = nn_result_10
    if isinstance(nn_inds_10, torch.Tensor):
        nn_inds_10 = nn_inds_10.cpu().numpy()

    N0 = F0.shape[0] if hasattr(F0, 'shape') else len(F0)
    N1 = F1.shape[0] if hasattr(F1, 'shape') else len(F1)

    # Mutual check (vectorized): keep i iff nn_inds_10[nn_inds_01[i]] == i.
    valid_01 = (nn_inds_01 >= 0) & (nn_inds_01 < N1)
    mutual_mask = np.zeros(N0, dtype=bool)
    valid_idx = np.where(valid_01)[0]
    if len(valid_idx) > 0:
        j_vals = nn_inds_01[valid_idx]
        back_map = nn_inds_10[j_vals]
        mutual_mask[valid_idx] = (back_map == valid_idx)

    src_idx = np.where(mutual_mask)[0]
    tgt_idx = nn_inds_01[mutual_mask]

    # Subsample per `sampling` mode.
    if subsample_size > 0 and len(src_idx) > subsample_size:
        if sampling == 'topk':
            if nn_dists_01 is None:
                # Fallback (should not happen since we requested distances).
                keep = np.arange(len(src_idx))[:subsample_size]
            else:
                valid_dists = nn_dists_01[mutual_mask]
                keep = np.argsort(valid_dists)[:subsample_size]
        elif sampling == 'random':
            # Uniform random — paper Table 2 default per Lihan 2026-05-08.
            r = rng if rng is not None else np.random.default_rng(0)
            keep = r.choice(len(src_idx), size=subsample_size, replace=False)
        elif sampling == 'score_weighted':
            # FCGF lib/trainer.py:594-616 trick: prob ∝ 1/normalized(feat_dist).
            r = rng if rng is not None else np.random.default_rng(0)
            if nn_dists_01 is None:
                keep = r.choice(len(src_idx), size=subsample_size, replace=False)
            else:
                valid_dists = nn_dists_01[mutual_mask].astype(np.float64)
                eps = 1e-8
                dmin, dmax = valid_dists.min(), valid_dists.max()
                if dmax > dmin:
                    norm = (valid_dists - dmin) / (dmax - dmin + eps)
                else:
                    norm = np.ones_like(valid_dists)
                conf = 1.0 / (norm + eps)
                prob = conf / conf.sum()
                keep = r.choice(len(src_idx), size=subsample_size, replace=False, p=prob)
        else:
            raise ValueError(f"Unknown sampling={sampling}; use topk|random|score_weighted")
        src_idx = src_idx[keep]
        tgt_idx = tgt_idx[keep]

    return src_idx, tgt_idx


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='PointCNN++ paper-canonical Table 2 reproduction (RR/FMR/IR sweep).')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to ResUNetBN2C-3DMatch checkpoint .pth')
    parser.add_argument('--threedmatch_root',
                        default=os.environ.get('THREEDMATCH_ROOT', ''),
                        help='Root of 3dmatch_processed/indoor (contains test/, train/, ...)')
    parser.add_argument('--pair_list',
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs', 'indoor', '3DMatch.pkl'),
                        help='Path to 3DMatch.pkl pair list')
    parser.add_argument('--voxel_size', type=float, default=0.025)
    parser.add_argument('--pre_downsample_voxel_size', type=float, default=0.02)
    parser.add_argument('--n_points', type=str, default='5000,2500,1000,500,250',
                        help='Comma-separated keypoint counts to sweep.')
    parser.add_argument('--hit_ratio_thresh', type=float, default=0.1,
                        help='tau_inlier (m) for IR / FMR (paper-canonical).')
    parser.add_argument('--fmr_thresh', type=float, default=0.05,
                        help='tau_FMR (fraction-of-inliers) per-pair threshold.')
    parser.add_argument('--rr_thresh', type=float, default=0.2,
                        help='RR RMSE threshold (m).')
    parser.add_argument('--ransac_iter', type=int, default=50000)
    parser.add_argument('--ransac_corr_dist', type=float, default=0.05,
                        help='RANSAC max_correspondence_distance (m). '
                             'Predator/GeoTransformer convention: 0.05 m. '
                             'Lihan test.py uses voxel_size×1.0=0.025 m '
                             '(set --ransac_corr_dist 0.025 to match).')
    parser.add_argument('--ransac_confidence', type=float, default=0.999)
    parser.add_argument('--nn_max_n', type=int, default=500,
                        help='find_nn_gpu chunk size. Match Lihan test.py default 500.')
    parser.add_argument('--corr_sampling',
                        choices=['random', 'topk', 'score_weighted'],
                        default='random',
                        help='How to subsample mutual-NN matches down to N: '
                             '`random` = uniform random (paper Table 2 default — '
                             'Lihan 2026-05-08 confirmed top-K is unfair to '
                             'baselines and produces "fake increase" at small N); '
                             '`topk` = take K with smallest forward-NN feat dist '
                             '(Lihan test.py:132 default; Predator-style); '
                             '`score_weighted` = prob ∝ 1/normalized(feat_dist) '
                             '(FCGF lib/trainer.py:594-616 trick).')
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of random seeds. Default 3 since random/score_weighted '
                             'sampling has variance; reported numbers are mean over seeds. '
                             'Set to 1 with --corr_sampling topk for deterministic.')
    parser.add_argument('--max_pairs', type=int, default=-1,
                        help='Optional cap on dataset pairs (smoke-test only). -1 = full.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_log', default=None,
                        help='Path to write JSON summary.')
    return parser.parse_args()


def run_one_pair(model, item, ns, args, device, rng):
    """
    Run the full per-pair pipeline (forward + mutual NN + top-K per N +
    IR/FMR/RR computation).
    Returns dict[N] = {'IR': float, 'FMR_hit': bool, 'RR_hit': bool} per
    keypoint count, or None if the pair was unrenderable.
    """
    (src_pcd, tgt_pcd, src_feats, tgt_feats,
     _matches, tsfm, _b0, _b1, _info) = item

    # numpy-ify
    if isinstance(src_pcd, torch.Tensor):
        src_pcd = src_pcd.cpu().numpy()
    if isinstance(tgt_pcd, torch.Tensor):
        tgt_pcd = tgt_pcd.cpu().numpy()
    if isinstance(src_feats, torch.Tensor):
        src_feats = src_feats.cpu().numpy()
    if isinstance(tgt_feats, torch.Tensor):
        tgt_feats = tgt_feats.cpu().numpy()
    if isinstance(tsfm, torch.Tensor):
        tsfm = tsfm.cpu().numpy()

    src_pcd = src_pcd.astype(np.float32)
    tgt_pcd = tgt_pcd.astype(np.float32)
    tsfm = tsfm.astype(np.float64)

    N0, N1 = len(src_pcd), len(tgt_pcd)
    if N0 < 100 or N1 < 100:
        return None

    # Forward — DO NOT pre-subsample points. Use the full voxelized cloud.
    offset0 = torch.tensor([0, N0], dtype=torch.long, device=device)
    offset1 = torch.tensor([0, N1], dtype=torch.long, device=device)

    input0 = {
        'coord': torch.from_numpy(src_pcd).float().to(device),
        'feat':  torch.from_numpy(src_feats).float().to(device),
        'offset': offset0,
        'grid_size': args.voxel_size,
    }
    input1 = {
        'coord': torch.from_numpy(tgt_pcd).float().to(device),
        'feat':  torch.from_numpy(tgt_feats).float().to(device),
        'offset': offset1,
        'grid_size': args.voxel_size,
    }

    src_coord_t = torch.from_numpy(src_pcd).float().to(device)
    tgt_coord_t = torch.from_numpy(tgt_pcd).float().to(device)

    out0 = model(input0)
    F0 = extract_features_from_output(out0)
    if isinstance(F0, torch.Tensor):
        F0 = F0.detach().cpu().numpy()
    F0 = np.asarray(F0, dtype=np.float32)
    xyz0_model = extract_coords_from_output(out0, src_coord_t)
    if isinstance(xyz0_model, torch.Tensor):
        xyz0_model = xyz0_model.detach().cpu().numpy()
    xyz0_model = np.asarray(xyz0_model, dtype=np.float32)

    out1 = model(input1)
    F1 = extract_features_from_output(out1)
    if isinstance(F1, torch.Tensor):
        F1 = F1.detach().cpu().numpy()
    F1 = np.asarray(F1, dtype=np.float32)
    xyz1_model = extract_coords_from_output(out1, tgt_coord_t)
    if isinstance(xyz1_model, torch.Tensor):
        xyz1_model = xyz1_model.detach().cpu().numpy()
    xyz1_model = np.asarray(xyz1_model, dtype=np.float32)

    # Sanity: model output xyz must align with model output features (per row).
    # Mirrors the canonical lib/trainer.py:438-446 pattern.
    if xyz0_model.shape[0] != F0.shape[0] or xyz1_model.shape[0] != F1.shape[0]:
        raise ValueError(
            f'Model coord/feat row count mismatch: '
            f'xyz0={xyz0_model.shape[0]} vs F0={F0.shape[0]}, '
            f'xyz1={xyz1_model.shape[0]} vs F1={F1.shape[0]}')

    # Use the model's output coords for correspondence lookup.
    src_o3d = make_o3d_pcd(xyz0_model)
    tgt_o3d = make_o3d_pcd(xyz1_model)

    out_per_n = {}
    for N in ns:
        # Mutual NN + per-N subsample. Default sampling=random matches
        # paper Table 2 (Lihan 2026-05-08); use --corr_sampling topk
        # for Lihan's `test.py` reproduction (small-N IR inflated).
        src_idx, tgt_idx = find_corr_mutual(
            F0, F1, device, nn_max_n=args.nn_max_n, subsample_size=N,
            sampling=args.corr_sampling, rng=rng)
        if len(src_idx) < 4:
            # Too few mutual matches for RANSAC; record an empty pair.
            out_per_n[N] = {
                'IR': 0.0, 'FMR_hit': False, 'RR_hit': False,
                'rmse': float('inf'), 'num_corr': int(len(src_idx)),
            }
            continue

        xyz0_corr = xyz0_model[src_idx]
        xyz1_corr = xyz1_model[tgt_idx]

        # IR under T_gt: paper τ_inlier = 0.1 m.
        xyz0_warped = apply_transform(xyz0_corr, tsfm)
        per_corr_dist = np.sqrt(((xyz0_warped - xyz1_corr) ** 2).sum(1) + 1e-12)
        ir = float((per_corr_dist < args.hit_ratio_thresh).mean())
        fmr_hit = bool(ir > args.fmr_thresh)

        # RR via Open3D RANSAC on correspondences (paper-canonical).
        corres = o3d.utility.Vector2iVector(
            np.stack([src_idx, tgt_idx], axis=1).astype(np.int32))
        try:
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                src_o3d, tgt_o3d, corres,
                args.ransac_corr_dist,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(args.ransac_corr_dist)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    args.ransac_iter, args.ransac_confidence),
            )
            T_est = np.asarray(result.transformation, dtype=np.float64)
        except Exception as e:
            logger.warning(f'RANSAC failed: {e}; T_est=I')
            T_est = np.eye(4)

        # Predator-canonical RR (3DMatch benchmark): apply both T_gt and
        # T_est to the SOURCE cloud, then RMSE of the displacement diff.
        # Pair counts as a successful registration if RMSE ≤ 0.2 m.
        # Using the full xyz0_model cloud (model-output coords) so the
        # measurement is unbiased w.r.t. the correspondence subset.
        diff = (xyz0_model @ T_est[:3, :3].T + T_est[:3, 3]) - \
               (xyz0_model @ tsfm[:3, :3].T   + tsfm[:3, 3])
        rmse = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
        rr_hit = bool(rmse <= args.rr_thresh)

        out_per_n[N] = {
            'IR': ir,
            'FMR_hit': fmr_hit,
            'RR_hit': rr_hit,
            'rmse': rmse,
            'num_corr': int(len(src_idx)),
        }

    return out_per_n


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    ns = [int(x.strip()) for x in args.n_points.split(',') if x.strip()]
    logger.info(f'Keypoint counts: {ns}')

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    if not args.threedmatch_root:
        raise ValueError('threedmatch_root required (or set THREEDMATCH_ROOT env var).')
    if not os.path.isfile(args.pair_list):
        raise FileNotFoundError(f'Pair list not found: {args.pair_list}')

    # Load model.
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    config = checkpoint['config']
    logger.info(f'Checkpoint config: model={config.model}, '
                f'model_n_out={config.model_n_out}, '
                f'voxel_size={config.voxel_size}, '
                f'normalize_feature={config.normalize_feature}')

    Model = load_model(config.model)
    model = Model(
        in_channels=1,
        out_channels=config.model_n_out,
        bn_momentum=0.05,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3,
        voxel_size=config.voxel_size,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    logger.info('Model loaded.')

    # Build dataset.
    cfg = edict({
        'voxel_size': args.voxel_size,
        'pre_downsample_voxel_size': args.pre_downsample_voxel_size,
        'threedmatch_root': args.threedmatch_root,
        'augment_noise': 0.0,
        'rot_factor': 1.0,
        'use_random_scale': False,
        'use_random_rotation': False,
        'jitter_sigma': 0.0,
        'positive_pair_search_voxel_size_multiplier': 1.5,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'rotation_range': 360,
    })
    dataset = ThreeDMatchNewPairDatasetPure(
        phase='test',
        config=cfg,
        pre_downsample_voxel_size=args.pre_downsample_voxel_size,
        train_info=args.pair_list,
        data_augmentation=False,
    )
    logger.info(f'Dataset loaded: {len(dataset)} pairs.')

    n_pairs = len(dataset) if args.max_pairs < 0 else min(args.max_pairs, len(dataset))
    logger.info(f'Evaluating {n_pairs} pairs over {args.num_seeds} seed(s).')

    # Aggregator: per_seed[seed] -> per_scene[scene] -> per_N[N] -> list of dicts
    seeds_out = []
    skipped_total = 0
    t_start = time.time()

    with torch.no_grad():
        for seed in range(args.num_seeds):
            logger.info(f'=== Seed {seed} ===')
            rng = np.random.default_rng(seed)
            # Also seed the legacy global RNG for any code using np.random.* under the hood.
            np.random.seed(seed)
            torch.manual_seed(seed)

            per_scene = defaultdict(lambda: defaultdict(list))  # scene -> N -> [dict]
            skipped = 0

            for idx in tqdm(range(n_pairs), desc=f'seed={seed}'):
                try:
                    item = dataset[idx]
                except Exception as e:
                    logger.warning(f'idx={idx}: dataset error: {e}')
                    skipped += 1
                    continue

                # Scene name from pair_info: ('test/<scene>/cloud_bin_X.pth', ...).
                pair_info = item[8] if len(item) >= 9 else None
                if isinstance(pair_info, (tuple, list)) and len(pair_info) >= 1:
                    s = str(pair_info[0])
                    parts = s.split('/')
                    scene = parts[1] if len(parts) >= 2 else 'unknown'
                else:
                    scene = 'unknown'

                try:
                    out_per_n = run_one_pair(model, item, ns, args, device, rng)
                except Exception as e:
                    logger.warning(f'idx={idx} scene={scene}: pair failed: {e}')
                    skipped += 1
                    continue

                if out_per_n is None:
                    skipped += 1
                    continue

                for N, res in out_per_n.items():
                    per_scene[scene][N].append(res)

            seeds_out.append({'scene': per_scene, 'skipped': skipped})
            skipped_total += skipped

    wall_time = time.time() - t_start

    # ── Aggregate ────────────────────────────────────────────
    # Per-scene means (averaged across seeds), per N.
    all_scenes = sorted({sc for s in seeds_out for sc in s['scene'].keys()})
    scene_table = {sc: {N: {'IR': [], 'FMR': [], 'RR': [], 'n': 0} for N in ns}
                   for sc in all_scenes}
    for s in seeds_out:
        for sc in all_scenes:
            per_n = s['scene'].get(sc, {})
            for N in ns:
                rows = per_n.get(N, [])
                if not rows:
                    continue
                irs = [r['IR'] for r in rows]
                fmrs = [1.0 if r['FMR_hit'] else 0.0 for r in rows]
                rrs = [1.0 if r['RR_hit'] else 0.0 for r in rows]
                scene_table[sc][N]['IR'].append(float(np.mean(irs)))
                scene_table[sc][N]['FMR'].append(float(np.mean(fmrs)))
                scene_table[sc][N]['RR'].append(float(np.mean(rrs)))
                scene_table[sc][N]['n'] = len(rows)  # pairs per seed

    # Reduce across seeds (mean).
    final_scene = {}
    for sc in all_scenes:
        final_scene[sc] = {}
        for N in ns:
            d = scene_table[sc][N]
            final_scene[sc][N] = {
                'IR':  float(np.mean(d['IR']))  if d['IR']  else 0.0,
                'FMR': float(np.mean(d['FMR'])) if d['FMR'] else 0.0,
                'RR':  float(np.mean(d['RR']))  if d['RR']  else 0.0,
                'n':   d['n'],
            }

    # Cross-scene averages per N (the Table 2 column).
    avg_per_n = {}
    for N in ns:
        ir_vals = [final_scene[sc][N]['IR']  for sc in all_scenes if final_scene[sc][N]['n'] > 0]
        fmr_vals = [final_scene[sc][N]['FMR'] for sc in all_scenes if final_scene[sc][N]['n'] > 0]
        rr_vals = [final_scene[sc][N]['RR']  for sc in all_scenes if final_scene[sc][N]['n'] > 0]
        avg_per_n[N] = {
            'IR':  float(np.mean(ir_vals))  if ir_vals  else 0.0,
            'FMR': float(np.mean(fmr_vals)) if fmr_vals else 0.0,
            'RR':  float(np.mean(rr_vals))  if rr_vals  else 0.0,
        }

    # ── Print report ─────────────────────────────────────────
    print()
    for N in ns:
        print(f'=== Per-scene IR / FMR / RR @ N={N} ===')
        for sc in all_scenes:
            d = final_scene[sc][N]
            print(f'  {sc:30s}: IR={d["IR"]*100:6.2f}%, FMR={d["FMR"]*100:6.2f}%, '
                  f'RR={d["RR"]*100:6.2f}%  (n={d["n"]} pairs)')
        a = avg_per_n[N]
        print(f'  AVERAGE @ N={N}: IR={a["IR"]*100:6.2f}%, FMR={a["FMR"]*100:6.2f}%, '
              f'RR={a["RR"]*100:6.2f}%')
        print()

    print('=== Final aggregate table (percentages) ===')
    print('| N    |   RR  |  FMR  |   IR  |')
    print('|------|-------|-------|-------|')
    for N in ns:
        a = avg_per_n[N]
        print(f'| {N:4d} | {a["RR"]*100:5.1f} | {a["FMR"]*100:5.1f} | {a["IR"]*100:5.1f} |')
    print()

    # Cross-check vs paper.
    print('=== Cross-check vs PointCNN++ Table 2 row "Ours" ===')
    print('| N    | RR (paper / ours) | FMR (paper / ours) | IR (paper / ours) | RR_pass | FMR_pass | IR_pass |')
    print('|------|-------------------|--------------------|-------------------|---------|----------|---------|')
    cross = {}
    all_pass = True
    for N in ns:
        if N not in PAPER_TABLE2:
            continue
        a = avg_per_n[N]
        ours = {'RR': a['RR'] * 100, 'FMR': a['FMR'] * 100, 'IR': a['IR'] * 100}
        target = PAPER_TABLE2[N]
        rr_pass = abs(ours['RR'] - target['RR']) <= TOL_RR
        fmr_pass = abs(ours['FMR'] - target['FMR']) <= TOL_FMR
        ir_pass = abs(ours['IR'] - target['IR']) <= TOL_IR
        if not (rr_pass and fmr_pass and ir_pass):
            all_pass = False
        cross[N] = {
            'paper': target,
            'ours': ours,
            'delta': {k: ours[k] - target[k] for k in target},
            'pass': {'RR': rr_pass, 'FMR': fmr_pass, 'IR': ir_pass},
        }
        print(f'| {N:4d} | {target["RR"]:5.1f} / {ours["RR"]:5.1f}     '
              f'| {target["FMR"]:5.1f} / {ours["FMR"]:5.1f}      '
              f'| {target["IR"]:5.1f} / {ours["IR"]:5.1f}      '
              f'| {"PASS" if rr_pass else "FAIL":7s} '
              f'| {"PASS" if fmr_pass else "FAIL":8s} '
              f'| {"PASS" if ir_pass else "FAIL":7s} |')
    print()
    print(f'Tolerances: RR ±{TOL_RR}pp, FMR ±{TOL_FMR}pp, IR ±{TOL_IR}pp')
    print(f'Verdict: {"PASS — all cells within tolerance" if all_pass else "FAIL — at least one cell outside tolerance"}')
    print()

    # Total counts.
    total_pairs = sum(final_scene[sc][ns[0]]['n'] for sc in all_scenes)
    print(f'Total pairs scored (per seed): {total_pairs}')
    print(f'Total pairs skipped (sum across seeds): {skipped_total}')
    print(f'Wall time: {wall_time:.1f}s')

    # ── JSON output ──────────────────────────────────────────
    if args.output_log:
        os.makedirs(os.path.dirname(args.output_log) or '.', exist_ok=True)
        summary = {
            'args': vars(args),
            'paper_target': PAPER_TABLE2,
            'tolerances': {'RR': TOL_RR, 'FMR': TOL_FMR, 'IR': TOL_IR},
            'wall_time_s': wall_time,
            'num_pairs_per_seed': total_pairs,
            'num_skipped_total': skipped_total,
            'num_seeds': args.num_seeds,
            'per_scene': {sc: {str(N): final_scene[sc][N] for N in ns} for sc in all_scenes},
            'avg_per_n': {str(N): avg_per_n[N] for N in ns},
            'cross_check': {str(N): cross[N] for N in cross},
            'verdict': 'PASS' if all_pass else 'FAIL',
        }
        with open(args.output_log, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f'JSON summary written to: {args.output_log}')

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
