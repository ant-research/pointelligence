"""
MSC v1m2 (CSC) with PointCNN++ operators

This module provides a Masked Scene Contrast implementation using PointCNN++ operators,
replacing the pointops-based SparsePointConv3d and GridPoolWrapper.
"""

import random
from itertools import chain
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from timm.layers import trunc_normal_

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from internals.neighbors import radius_search
from internals.indexing import cumsum_exclusive, repeat_interleave_indices

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.utils.comm import get_world_size


def exponential_split_distance(distances, a=0.05*0.25):
    """
    Split distances into bins using exponential partitioning.
    
    Args:
        distances: [N] or [M] distance values (Euclidean distance)
        a: Base partitioning parameter, default 0.05*0.25 = 0.0125
    
    Returns:
        distance_bins: [N] or [M] distance bin indices
        Bin ranges:
        - [0, a)     -> bin 0 (closest)
        - [a, 2a)   -> bin 1
        - [2a, 4a)  -> bin 2
        - [4a, 6a)  -> bin 3
        - [6a, 10a) -> bin 4
        - [10a, 14a)-> bin 5
        - ... (exponential growth)
    """
    dist_abs = distances.abs()
    idx = 2 * torch.floor(torch.log((dist_abs + 2*a) / a) / np.log(2)) - 2
    idx = idx + ((3*(2**(idx//2)) - 2)*a <= dist_abs).float()
    idx = torch.clamp(idx, min=0)
    return idx.long()


@MODELS.register_module("MSC-v1m2-pointcnnpp")
class MaskedSceneContrast(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_in_channels,
        backbone_out_channels,
        mask_grid_size=0.1,
        mask_rate=0.4,
        view1_mix_prob=0,
        view2_mix_prob=0,
        matching_max_k=8,
        matching_max_radius=0.03,
        matching_max_pair=8192,
        nce_t=0.4,
        contrast_weight=1,
        reconstruct_weight=1,
        use_boundary_aware_loss=False,
        boundary_weight=2.0,
        boundary_radius=0.05,
        reconstruct_color=True,
        reconstruct_normal=True,
        partitions=4,
        r1=0.125,
        r2=2,
        use_hard_negative_mining=False,
        hard_neg_ratio=0.3,
        hard_neg_weight=1.5,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.mask_grid_size = mask_grid_size
        self.mask_rate = mask_rate
        self.view1_mix_prob = view1_mix_prob
        self.view2_mix_prob = view2_mix_prob
        self.matching_max_k = matching_max_k
        self.matching_max_radius = matching_max_radius
        self.matching_max_pair = matching_max_pair
        self.nce_t = nce_t
        self.contrast_weight = contrast_weight
        self.reconstruct_weight = reconstruct_weight
        self.reconstruct_color = reconstruct_color
        self.reconstruct_normal = reconstruct_normal

        # csc partition
        self.partitions = partitions
        self.r1 = r1
        self.r2 = r2

        self.mask_token = nn.Parameter(torch.zeros(1, backbone_in_channels))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.color_head = (
            nn.Linear(backbone_out_channels, 3) if reconstruct_color else None
        )
        self.normal_head = (
            nn.Linear(backbone_out_channels, 3) if reconstruct_normal else None
        )
        self.nce_criteria = torch.nn.CrossEntropyLoss(reduction="mean")

        self.use_boundary_aware_loss = use_boundary_aware_loss
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius
        
        # Hard negative mining parameters
        self.use_hard_negative_mining = use_hard_negative_mining
        self.hard_neg_ratio = hard_neg_ratio
        self.hard_neg_weight = hard_neg_weight
        
        self.use_exponential_split_matching = False
        self.exponential_split_a = 0.05 * 0.25
        # Distance bin configuration: {bin_id: (radius_multiplier, weight_multiplier)}
        # Closer distances: stricter (smaller radius, higher weight)
        # Farther distances: looser (larger radius, lower weight)
        self.distance_bin_config = {
            0: (0.5, 1.5),
            1: (0.7, 1.3),
            2: (1.0, 1.0),
            3: (1.2, 0.8),
            4: (1.5, 0.6),
            5: (2.0, 0.4),
        }
        self.default_bin_radius_mult = 2.5
        self.default_bin_weight_mult = 0.2

    @torch.no_grad()
    def _compute_bins(self, coord: torch.Tensor, batch: torch.Tensor, grid_size: float):
        """
        Compute bins using simple voxelization.
        Replaces GridPoolWrapper._compute_bins
        
        This computes grid indices by dividing coordinates by grid_size.
        """
        # Simple voxelization: divide coordinates by grid_size and floor
        grid_coords = torch.floor(coord / grid_size).long()
        return grid_coords

    @torch.no_grad()
    def _compute_indices_pointcnnpp(
        self, query_coord, key_coord, q_inds, p_inds, radius
    ):
        """
        Compute neighbor indices using PointCNN++ radius_search.
        Replaces manual distance-based search with optimized radius_search.
        
        Args:
            query_coord: [N, 3] query point coordinates
            key_coord: [M, 3] key point coordinates
            q_inds: [B] query point counts per batch
            p_inds: [B] key point counts per batch
            radius: search radius
        
        Returns:
            a_idx: query point indices [K]
            b_idx: key point indices [K]
            o_idx: query point indices (same as a_idx) [K]
        """
        device = query_coord.device
        
        # Build sample_inds for batch-aware search
        # sample_inds format: [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ...] for each batch
        if len(q_inds) == 0 or len(p_inds) == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty
        
        valid_q_batches = q_inds > 0
        valid_p_batches = p_inds > 0
        if not valid_q_batches.any() or not valid_p_batches.any():
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty
        
        query_sample_inds = torch.repeat_interleave(
            torch.arange(len(q_inds), device=device, dtype=torch.long),
            q_inds
        )
        key_sample_inds = torch.repeat_interleave(
            torch.arange(len(p_inds), device=device, dtype=torch.long),
            p_inds
        )
        
        # Use PointCNN++ radius_search for efficient neighbor search
        neighbors, num_neighbors = radius_search(
            points=key_coord,  # points to search in
            query_points=query_coord,  # query points
            radius=radius,
            sample_inds=key_sample_inds,
            query_sample_inds=query_sample_inds,
            return_distances=False,
            distance_type="ball",
        )
                
        # Convert radius_search output to (query_idx, key_idx) pairs
        # neighbors: [K] flattened neighbor indices (relative to key_coord)
        # num_neighbors: [N] number of neighbors for each query point
        
        if neighbors.numel() == 0:
            # Return empty tensors if no matches found
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty
        
        # Generate query indices by repeating each query point index according to its num_neighbors
        num_neighbors_cumsum, _ = cumsum_exclusive(num_neighbors, return_sum=True)
        query_indices = repeat_interleave_indices(
            repeats_cumsum=num_neighbors_cumsum,
            output_size=neighbors.numel(),
            may_contain_zero_repeats=False,
        )
        
        # neighbors are already key indices (relative to key_coord)
        key_indices = neighbors
        
        return query_indices, key_indices, query_indices

    @torch.no_grad()
    def generate_cross_masks(
        self, view1_origin_coord, view1_offset, view2_origin_coord, view2_offset
    ):
        view1_batch = offset2batch(view1_offset)
        view2_batch = offset2batch(view2_offset)

        view1_batch_count = view1_batch.bincount()
        view2_batch_count = view2_batch.bincount()
        view1_origin_coord_split = view1_origin_coord.split(list(view1_batch_count))
        view2_origin_coord_split = view2_origin_coord.split(list(view2_batch_count))
        union_origin_coord = torch.cat(
            list(
                chain.from_iterable(
                    zip(view1_origin_coord_split, view2_origin_coord_split)
                )
            )
        )
        union_offset = torch.cat(
            [view1_offset.unsqueeze(-1), view2_offset.unsqueeze(-1)], dim=-1
        ).sum(-1)
        union_batch = offset2batch(union_offset)

        key = self._compute_bins(union_origin_coord, union_batch, self.mask_grid_size)
        unique, cluster, counts = torch.unique(
            key, sorted=True, return_inverse=True, return_counts=True, dim=0
        )
        patch_num = unique.shape[0]
        patch_max_point = counts.max().item()
        patch2point_map = cluster.new_zeros(patch_num, patch_max_point)
        patch2point_mask = torch.lt(
            torch.arange(patch_max_point, device=counts.device).unsqueeze(0), counts.unsqueeze(-1)
        )
        sorted_cluster_value, sorted_cluster_indices = torch.sort(cluster)
        patch2point_map[patch2point_mask] = sorted_cluster_indices

        assert self.mask_rate <= 0.5
        patch_mask = torch.zeros(patch_num, device=union_origin_coord.device).int()
        rand_perm = torch.randperm(patch_num, device=union_origin_coord.device)
        mask_patch_num = int(patch_num * self.mask_rate)

        patch_mask[rand_perm[0:mask_patch_num]] = 1
        patch_mask[rand_perm[mask_patch_num : mask_patch_num * 2]] = 2
        point_mask = torch.zeros(
            union_origin_coord.shape[0], device=union_origin_coord.device
        ).int()
        point_mask[
            patch2point_map[patch_mask == 1][patch2point_mask[patch_mask == 1]]
        ] = 1
        point_mask[
            patch2point_map[patch_mask == 2][patch2point_mask[patch_mask == 2]]
        ] = 2

        point_mask_split = point_mask.split(
            list(
                torch.cat(
                    [view1_batch_count.unsqueeze(-1), view2_batch_count.unsqueeze(-1)],
                    dim=-1,
                ).flatten()
            )
        )
        view1_point_mask = torch.cat(point_mask_split[0::2]) == 1
        view2_point_mask = torch.cat(point_mask_split[1::2]) == 2
        return view1_point_mask, view2_point_mask

    @torch.no_grad()
    def match_contrastive_pair(
        self, view1_coord, view1_offset, view2_coord, view2_offset, max_k, max_radius,
        use_exponential_split=False, return_distance_bins=False
    ):
        """
        Match contrastive pairs based on Cartesian coordinates.
        Supports exponential split adaptive matching strategy.
        """
        if use_exponential_split:
            return self._match_with_exponential_split(
                view1_coord, view1_offset, view2_coord, view2_offset, 
                max_k, max_radius, return_distance_bins
            )
        else:
            b1 = offset2batch(view1_offset)
            b2 = offset2batch(view2_offset)
            q_inds = b1.bincount()
            p_inds = b2.bincount()

            a_idx, b_idx, o_idx = self._compute_indices_pointcnnpp(
                view1_coord, view2_coord, q_inds=q_inds, p_inds=p_inds, radius=max_radius
            )

            pairs = torch.stack([o_idx.to(torch.long), b_idx.to(torch.long)], dim=1)

            if max_k is not None and max_k > 0:
                o_unique, counts = torch.unique_consecutive(pairs[:, 0], return_counts=True)
                selector = torch.zeros(pairs.shape[0], dtype=torch.bool, device=pairs.device)
                start = 0
                for c in counts.tolist():
                    end = start + c
                    selector[start : start + min(c, max_k)] = True
                    start = end
                pairs = pairs[selector]

            if pairs.shape[0] > self.matching_max_pair:
                idx = torch.randperm(pairs.shape[0], device=pairs.device)[: self.matching_max_pair]
                pairs = pairs[idx]

            if return_distance_bins:
                matched_coord1 = view1_coord[pairs[:, 0]]
                matched_coord2 = view2_coord[pairs[:, 1]]
                distances = torch.norm(matched_coord1 - matched_coord2, p=2, dim=1)
                distance_bins = exponential_split_distance(distances, a=self.exponential_split_a)
                return pairs, distance_bins
            return pairs

    @torch.no_grad()
    def _match_with_exponential_split(
        self, view1_coord, view1_offset, view2_coord, view2_offset, 
        max_k, base_max_radius, return_distance_bins=False
    ):
        """
        Adaptive matching strategy based on exponential split.
        Uses different matching radius and weight for different distance ranges.
        """
        b1 = offset2batch(view1_offset)
        b2 = offset2batch(view2_offset)
        device = view1_coord.device
        
        all_pairs = []
        all_distance_bins = []
        
        for bin_id, (radius_mult, weight_mult) in self.distance_bin_config.items():
            adaptive_radius = base_max_radius * radius_mult
            
            q_inds = b1.bincount()
            p_inds = b2.bincount()
            
            try:
                a_idx, b_idx, o_idx = self._compute_indices_pointcnnpp(
                    view1_coord, view2_coord, q_inds=q_inds, p_inds=p_inds, radius=adaptive_radius
                )
                
                if a_idx.numel() == 0 or b_idx.numel() == 0:
                    continue
                
                pairs = torch.stack([o_idx.to(torch.long), b_idx.to(torch.long)], dim=1)
                
                if pairs.shape[0] == 0:
                    continue
                
                matched_coord1 = view1_coord[pairs[:, 0]]
                matched_coord2 = view2_coord[pairs[:, 1]]
                distances = torch.norm(matched_coord1 - matched_coord2, p=2, dim=1)
                distance_bins = exponential_split_distance(distances, a=self.exponential_split_a)
                
                bin_mask = (distance_bins == bin_id)
                if bin_mask.sum() > 0:
                    pairs = pairs[bin_mask]
                    all_pairs.append(pairs)
                    if return_distance_bins:
                        all_distance_bins.append(distance_bins[bin_mask])
            except (RuntimeError, ValueError, IndexError) as e:
                import warnings
                warnings.warn(f"Error matching bin {bin_id}: {str(e)}, skipping")
                continue
            except Exception as e:
                import warnings
                warnings.warn(f"Unknown error matching bin {bin_id}: {type(e).__name__}: {str(e)}")
                continue
        
        if len(all_pairs) == 0:
            return self.match_contrastive_pair(
                view1_coord, view1_offset, view2_coord, view2_offset,
                max_k, base_max_radius, use_exponential_split=False, return_distance_bins=return_distance_bins
            )
        
        all_pairs = torch.cat(all_pairs, dim=0)
        
        if max_k is not None and max_k > 0:
            o_unique, counts = torch.unique_consecutive(all_pairs[:, 0], return_counts=True)
            selector = torch.zeros(all_pairs.shape[0], dtype=torch.bool, device=device)
            start = 0
            for c in counts.tolist():
                end = start + c
                selector[start : start + min(c, max_k)] = True
                start = end
            all_pairs = all_pairs[selector]
            if return_distance_bins and len(all_distance_bins) > 0:
                all_distance_bins = torch.cat(all_distance_bins, dim=0)[selector]
        
        if all_pairs.shape[0] > self.matching_max_pair:
            idx = torch.randperm(all_pairs.shape[0], device=device)[: self.matching_max_pair]
            all_pairs = all_pairs[idx]
            if return_distance_bins and len(all_distance_bins) > 0:
                all_distance_bins = all_distance_bins[idx]
        
        if return_distance_bins:
            if len(all_distance_bins) == 0:
                matched_coord1 = view1_coord[all_pairs[:, 0]]
                matched_coord2 = view2_coord[all_pairs[:, 1]]
                distances = torch.norm(matched_coord1 - matched_coord2, p=2, dim=1)
                all_distance_bins = exponential_split_distance(distances, a=self.exponential_split_a)
            return all_pairs, all_distance_bins
        
        return all_pairs

    def compute_partitions(self, coord1, coord2):
        """
        Compute partition matrix for contrastive learning.
        """
        device = coord1.device
        partition_matrix = torch.zeros((coord1.shape[0], coord2.shape[0]), device=device)
        partition_matrix = partition_matrix - 1e7

        rel_trans = coord1.unsqueeze(1) - coord2.unsqueeze(0)
        mask_up = rel_trans[:, :, 2] > 0.0
        mask_down = rel_trans[:, :, 2] < 0.0

        distance_matrix = torch.sqrt(torch.sum(rel_trans.pow(2), dim=2) + 1e-7)

        # Partition 0 and 1: r1 < distance <= r2
        mask = (distance_matrix > self.r1) & (distance_matrix <= self.r2)
        partition_matrix[mask & mask_up] = 0
        partition_matrix[mask & mask_down] = 1

        # Partition 2 and 3: distance > r2
        mask = distance_matrix > self.r2
        partition_matrix[mask & mask_up] = 2
        partition_matrix[mask & mask_down] = 3

        return partition_matrix

    @torch.no_grad()
    def compute_boundary_mask(self, coord, batch, radius=None):
        """
        Compute boundary regions based on exact coordinates.
        
        Boundary points are defined as points with fewer neighbors than expected
        (may be at object edges or sparse regions).
        """
        if radius is None:
            radius = self.boundary_radius
        
        device = coord.device
        batch_counts = batch.bincount()
        boundary_mask = torch.zeros(coord.shape[0], dtype=torch.bool, device=device)
        
        start_idx = 0
        for batch_id, batch_size in enumerate(batch_counts):
            if batch_size == 0:
                continue
                
            end_idx = start_idx + batch_size
            batch_coord = coord[start_idx:end_idx]
            batch_list = torch.tensor([batch_size], device=device, dtype=torch.long)
            
            try:
                a_idx, b_idx, o_idx = self._compute_indices_pointcnnpp(
                    batch_coord, batch_coord, 
                    q_inds=batch_list, p_inds=batch_list, 
                    radius=radius
                )
                
                if o_idx.numel() > 0:
                    unique_queries, counts = torch.unique(o_idx, return_counts=True)
                    neighbor_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
                    neighbor_counts[unique_queries] = counts.long()
                    
                    neighbor_counts_float = neighbor_counts.float()
                    median_neighbors = neighbor_counts_float.median()
                    # Boundary points: neighbors < 60% of median (conservative threshold)
                    if median_neighbors > 0:
                        boundary_threshold = median_neighbors * 0.6
                        batch_boundary_mask = neighbor_counts_float < boundary_threshold
                    else:
                        batch_boundary_mask = neighbor_counts_float < 1.0
                else:
                    batch_boundary_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                
                boundary_mask[start_idx:end_idx] = batch_boundary_mask
            except Exception as e:
                # Fallback: detect boundaries based on local density
                if batch_size > 1:
                    distances = torch.cdist(batch_coord, batch_coord, p=2)
                    distances.fill_diagonal_(float('inf'))
                    k = min(5, batch_size - 1)
                    k_distances, _ = torch.topk(distances, k, dim=1, largest=False)
                    mean_k_distance = k_distances.mean(dim=1)
                    boundary_ratio = 0.3
                    k_boundary = max(1, int(batch_size * boundary_ratio))
                    _, top_k_indices = torch.topk(mean_k_distance, k_boundary, largest=True)
                    batch_boundary_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    batch_boundary_mask[top_k_indices] = True
                else:
                    batch_boundary_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                
                boundary_mask[start_idx:end_idx] = batch_boundary_mask
            
            start_idx = end_idx
        
        return boundary_mask

    def compute_contrastive_loss(
        self,
        view1_feat,
        view1_coord,
        view1_offset,
        view2_feat,
        view2_coord,
        view2_offset,
        match_index,
        view1_origin_coord=None,
        view2_origin_coord=None,
        view1_boundary_mask=None,
        view2_boundary_mask=None,
        distance_bins=None,
    ):
        assert view1_offset.shape == view2_offset.shape
        device = view1_feat.device
        loss = torch.tensor(0.0, device=device)
        pos_sim = torch.tensor(0.0, device=device)
        neg_sim = torch.tensor(0.0, device=device)
        large_num = 1e9

        view1_feat = view1_feat[match_index[:, 0]]
        view2_feat = view2_feat[match_index[:, 1]]
        view1_feat = view1_feat / (
            torch.norm(view1_feat, p=2, dim=1, keepdim=True) + 1e-7
        )
        view2_feat = view2_feat / (
            torch.norm(view2_feat, p=2, dim=1, keepdim=True) + 1e-7
        )

        view1_coord = view1_coord[match_index[:, 0]]
        view2_coord = view2_coord[match_index[:, 1]]
        
        distance_weights = None
        if distance_bins is not None:
            distance_weights = torch.ones(
                match_index.shape[0], device=view1_feat.device, dtype=torch.float32
            )
            for bin_id, (radius_mult, weight_mult) in self.distance_bin_config.items():
                bin_mask = (distance_bins == bin_id)
                if bin_mask.sum() > 0:
                    distance_weights[bin_mask] = weight_mult
            max_configured_bin = max(self.distance_bin_config.keys())
            unconfigured_mask = (distance_bins > max_configured_bin)
            if unconfigured_mask.sum() > 0:
                distance_weights[unconfigured_mask] = self.default_bin_weight_mult
        
        boundary_weights = None
        if self.use_boundary_aware_loss and view1_boundary_mask is not None and view2_boundary_mask is not None:
            view1_matched_boundary_mask = view1_boundary_mask[match_index[:, 0]]
            view2_matched_boundary_mask = view2_boundary_mask[match_index[:, 1]]
            
            boundary_mask = view1_matched_boundary_mask | view2_matched_boundary_mask
            boundary_weights = torch.ones(
                match_index.shape[0], device=device, dtype=torch.float32
            )
            boundary_weights[boundary_mask] = self.boundary_weight
        
        final_weights = torch.ones(
            match_index.shape[0], device=view1_feat.device, dtype=torch.float32
        )
        if distance_weights is not None:
            final_weights *= distance_weights
        if boundary_weights is not None:
            final_weights *= boundary_weights

        batch = offset2batch(view1_offset)[match_index[:, 0]]
        
        for batch_id in batch.unique():
            batch_mask = batch == batch_id
            sim = torch.mm(view1_feat[batch_mask], view2_feat[batch_mask].T)

            with torch.no_grad():
                pos_sim += torch.diagonal(sim).mean()
                
                n = sim.shape[0]
                if n > 1:
                    off_diagonal_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
                    neg_sim += sim[off_diagonal_mask].mean()

            labels = torch.arange(sim.shape[0], device=view1_feat.device).long()
            part = self.compute_partitions(
                view1_coord[batch_mask], view2_coord[batch_mask]
            )
            
            batch_weights = final_weights[batch_mask] if final_weights is not None else None
            
            for part_id in part.unique():
                part_mask = part == part_id
                part_mask.fill_diagonal_(True)
                
                # 计算基础 logits
                logits = torch.div(sim, self.nce_t) - large_num * (~part_mask).float()
                
                # Hard negative mining: 识别困难负样本并增强其 logits
                if self.use_hard_negative_mining and n > 1:
                    # 识别困难负样本：对于每个正样本，找到最相似的负样本
                    off_diagonal_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
                    # 只考虑当前 partition 内的负样本
                    partition_neg_mask = (~part_mask) & off_diagonal_mask
                    
                    if partition_neg_mask.sum() > 0:
                        # 对于每行，找到最相似的负样本（困难负样本）
                        num_hard_neg = max(1, int(n * self.hard_neg_ratio))
                        sim_masked = sim.clone()
                        sim_masked[~partition_neg_mask] = -1e10  # 将不在 partition 内的设为很小的值
                        sim_masked.fill_diagonal_(-1e10)  # 排除对角线
                        
                        # 选择每行的 top-k 困难负样本（向量化，无循环）
                        _, top_k_indices = torch.topk(sim_masked, k=min(num_hard_neg, n-1), dim=1)
                        
                        # 创建困难负样本掩码矩阵（向量化）
                        row_indices = torch.arange(n, device=sim.device).unsqueeze(1).expand(-1, top_k_indices.shape[1])
                        hard_neg_mask = torch.zeros_like(sim, dtype=torch.bool, device=sim.device)
                        hard_neg_mask[row_indices.flatten(), top_k_indices.flatten()] = True
                        
                        # 增强困难负样本的 logits（使其在 softmax 中占更大权重）
                        # 通过增加 logits 值来实现（相当于提高温度或权重）
                        logits[hard_neg_mask] = logits[hard_neg_mask] * self.hard_neg_weight
                
                # 计算损失
                if batch_weights is not None:
                    nce_criteria_none = torch.nn.CrossEntropyLoss(reduction='none')
                    partition_loss_per_sample = nce_criteria_none(logits, labels)
                    weighted_loss_per_sample = partition_loss_per_sample * batch_weights
                    partition_loss = weighted_loss_per_sample.mean()
                else:
                    partition_loss = self.nce_criteria(logits, labels)

                loss += partition_loss

        loss /= len(view1_offset) * self.partitions
            
        pos_sim /= len(view1_offset)
        neg_sim /= len(view1_offset)

        if get_world_size() > 1:
            dist.all_reduce(loss)
            dist.all_reduce(pos_sim)
            dist.all_reduce(neg_sim)
        return (
            loss / get_world_size(),
            pos_sim / get_world_size(),
            neg_sim / get_world_size(),
        )

    def forward(self, data_dict):
        view1_origin_coord = data_dict["view1_origin_coord"]
        view1_coord = data_dict["view1_coord"]
        view1_feat = data_dict["view1_feat"]
        view1_offset = data_dict["view1_offset"].int()

        view2_origin_coord = data_dict["view2_origin_coord"]
        view2_coord = data_dict["view2_coord"]
        view2_feat = data_dict["view2_feat"]
        view2_offset = data_dict["view2_offset"].int()

        view1_point_mask, view2_point_mask = self.generate_cross_masks(
            view1_origin_coord, view1_offset, view2_origin_coord, view2_offset
        )

        view1_mask_tokens = self.mask_token.expand(view1_coord.shape[0], -1)
        view1_weight = view1_point_mask.unsqueeze(-1).type_as(view1_mask_tokens)
        view1_feat = view1_feat * (1 - view1_weight) + view1_mask_tokens * view1_weight

        view2_mask_tokens = self.mask_token.expand(view2_coord.shape[0], -1)
        view2_weight = view2_point_mask.unsqueeze(-1).type_as(view2_mask_tokens)
        view2_feat = view2_feat * (1 - view2_weight) + view2_mask_tokens * view2_weight

        view1_data_dict = dict(
            origin_coord=view1_origin_coord,
            coord=view1_coord,
            feat=view1_feat,
            offset=view1_offset,
        )
        view2_data_dict = dict(
            origin_coord=view2_origin_coord,
            coord=view2_coord,
            feat=view2_feat,
            offset=view2_offset,
        )

        if "view1_grid_coord" in data_dict.keys():
            view1_data_dict["grid_coord"] = data_dict["view1_grid_coord"]
        if "view2_grid_coord" in data_dict.keys():
            view2_data_dict["grid_coord"] = data_dict["view2_grid_coord"]

        if random.random() < self.view1_mix_prob:
            view1_data_dict["offset"] = torch.cat(
                [view1_offset[1:-1:2], view1_offset[-1].unsqueeze(0)], dim=0
            )
        if random.random() < self.view2_mix_prob:
            view2_data_dict["offset"] = torch.cat(
                [view2_offset[1:-1:2], view2_offset[-1].unsqueeze(0)], dim=0
            )

        view1_feat = self.backbone(view1_data_dict)
        view2_feat = self.backbone(view2_data_dict)
        
        view1_boundary_mask = None
        view2_boundary_mask = None
        if self.use_boundary_aware_loss:
            view1_batch = offset2batch(view1_offset)
            view2_batch = offset2batch(view2_offset)
            view1_boundary_mask = self.compute_boundary_mask(
                view1_origin_coord, view1_batch
            )
            view2_boundary_mask = self.compute_boundary_mask(
                view2_origin_coord, view2_batch
            )
        
        match_result = self.match_contrastive_pair(
            view1_origin_coord,
            view1_offset,
            view2_origin_coord,
            view2_offset,
            max_k=self.matching_max_k,
            max_radius=self.matching_max_radius,
            use_exponential_split=True,
            return_distance_bins=True,
        )
        if isinstance(match_result, tuple):
            match_index, distance_bins = match_result
        else:
            match_index = match_result
            distance_bins = None
        
        nce_loss, pos_sim, neg_sim = self.compute_contrastive_loss(
            view1_feat,
            view1_origin_coord,
            view1_offset,
            view2_feat,
            view2_origin_coord,
            view2_offset,
            match_index,
            view1_origin_coord=view1_origin_coord,
            view2_origin_coord=view2_origin_coord,
            view1_boundary_mask=view1_boundary_mask,
            view2_boundary_mask=view2_boundary_mask,
            distance_bins=distance_bins,
        )
        loss = nce_loss * self.contrast_weight
        result_dict = dict(nce_loss=nce_loss, pos_sim=pos_sim, neg_sim=neg_sim)

        if self.color_head is not None:
            assert "view1_color" in data_dict.keys()
            assert "view2_color" in data_dict.keys()
            view1_color = data_dict["view1_color"]
            view2_color = data_dict["view2_color"]
            view1_color_pred = self.color_head(view1_feat[view1_point_mask])
            view2_color_pred = self.color_head(view2_feat[view2_point_mask])
            color_loss = (
                torch.sum((view1_color_pred - view1_color[view1_point_mask]) ** 2)
                + torch.sum((view2_color_pred - view2_color[view2_point_mask]) ** 2)
            ) / (view1_color_pred.shape[0] + view2_color_pred.shape[0])
            loss = loss + color_loss * self.reconstruct_weight
            result_dict["color_loss"] = color_loss

        if self.normal_head is not None:
            assert "view1_normal" in data_dict.keys()
            assert "view2_normal" in data_dict.keys()
            view1_normal = data_dict["view1_normal"]
            view2_normal = data_dict["view2_normal"]
            view1_normal_pred = self.normal_head(view1_feat[view1_point_mask])
            view2_normal_pred = self.normal_head(view2_feat[view2_point_mask])

            view1_normal_pred = view1_normal_pred / (
                torch.norm(view1_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            )
            view2_normal_pred = view2_normal_pred / (
                torch.norm(view2_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            )
            normal_loss = (
                torch.sum(view1_normal_pred * view1_normal[view1_point_mask])
                + torch.sum(view2_normal_pred * view2_normal[view2_point_mask])
            ) / (view1_normal_pred.shape[0] + view2_normal_pred.shape[0])
            loss = loss + normal_loss * self.reconstruct_weight
            result_dict["normal_loss"] = normal_loss

        result_dict["loss"] = loss
        return result_dict