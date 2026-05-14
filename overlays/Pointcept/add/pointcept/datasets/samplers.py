"""
类别平衡采样器（Class-Balanced Sampler）

功能：
1. 针对低IoU类别增加采样次数（oversampling）
2. 对高频类别减少采样次数（undersampling）
3. 平衡各个类别的训练难度
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
import os
import pickle
from pointcept.utils.logger import get_root_logger


class ClassBalancedSampler(Sampler):
    """
    类别平衡采样器（支持细粒度IoU分级）
    
    策略：
    1. 预先统计每个样本包含的类别
    2. 根据类别频率和IoU等级计算采样权重
    3. IoU等级分为：很低、较低、中等、较高、很高
    4. 每个IoU等级有不同的采样倍数
    5. 高频类别：减少采样次数
    """
    
    def __init__(
        self,
        dataset,
        num_classes=16,
        # IoU等级配置：根据验证结果分类
        # 格式：{等级名称: {'classes': [类别列表], 'multiplier': 采样倍数}}
        iou_levels=None,
        # 或者使用简化的配置方式（向后兼容）
        low_iou_classes=None,  # 已废弃，使用iou_levels代替
        low_iou_multiplier=2.0,  # 已废弃，使用iou_levels代替
        high_freq_threshold=0.15,  # 高频类别阈值（占总样本的比例）
        high_freq_multiplier=0.5,  # 高频类别采样倍数（减少采样）
        target_samples_per_class=None,  # 目标每个类别的样本数，None表示自动计算
        use_relative_adjustment=True,  # 是否使用相对调整策略（基于原始数量，而非最少类别）
        cache_dir=None,  # 缓存目录
        shuffle=True,
        seed=None,
    ):
        """
        Args:
            dataset: 数据集对象
            num_classes: 类别数量
            iou_levels: IoU等级配置字典，例如：
                {
                    'very_low': {'classes': [1], 'multiplier': 3.0},      # IoU < 0.5
                    'low': {'classes': [4, 7], 'multiplier': 2.5},        # 0.5 <= IoU < 0.7
                    'medium': {'classes': [8, 11, 12, 13], 'multiplier': 1.2},  # 0.7 <= IoU < 0.8
                    'high': {'classes': [0, 5, 6], 'multiplier': 1.0},    # 0.8 <= IoU < 0.9
                    'very_high': {'classes': [2, 3, 9, 10, 14, 15], 'multiplier': 0.8}  # IoU >= 0.9
                }
            high_freq_threshold: 高频类别阈值，超过此比例的类别会被减少采样
            high_freq_multiplier: 高频类别采样倍数，例如 0.5 表示采样减半
            target_samples_per_class: 目标每个类别的样本数，None表示自动计算
            use_relative_adjustment: 是否使用相对调整策略
                - True: 基于原始数量进行温和调整（推荐）
                - False: 基于最少类别数量进行平衡（传统方式）
            cache_dir: 缓存目录，用于保存类别统计信息
            shuffle: 是否打乱
            seed: 随机种子
        """
        self.dataset = dataset
        self.num_classes = num_classes
        
        # 处理IoU等级配置
        if iou_levels is not None:
            self.iou_levels = iou_levels
        else:
            # 向后兼容：使用旧的配置方式
            if low_iou_classes is not None:
                self.iou_levels = {
                    'low': {'classes': low_iou_classes, 'multiplier': low_iou_multiplier}
                }
            else:
                self.iou_levels = {}
        
        # 构建类别到IoU等级的映射
        self.class_to_iou_level = {}
        for level_name, level_config in self.iou_levels.items():
            for cls in level_config.get('classes', []):
                self.class_to_iou_level[cls] = level_name
        
        self.high_freq_threshold = high_freq_threshold
        self.high_freq_multiplier = high_freq_multiplier
        self.target_samples_per_class = target_samples_per_class
        self.use_relative_adjustment = use_relative_adjustment
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.seed = seed
        
        self.logger = get_root_logger()
        
        # 构建类别索引映射
        self.class_indices = self._build_class_indices()
        
        # 计算采样策略
        self.sampling_strategy = self._compute_sampling_strategy()
        
        self.logger.info(
            f"ClassBalancedSampler initialized: "
            f"total_samples={len(dataset)}, "
            f"iou_levels={len(self.iou_levels)} levels, "
            f"use_relative_adjustment={self.use_relative_adjustment}, "
            f"high_freq_threshold={self.high_freq_threshold}, "
            f"high_freq_multiplier={self.high_freq_multiplier}"
        )
    
    def _build_class_indices(self):
        """
        构建每个类别的样本索引映射
        
        返回:
            class_indices: dict, {class_id: [sample_indices]}
        """
        cache_file = None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.cache_dir, 
                f"class_indices_{len(self.dataset)}.pkl"
            )
        
        # 尝试从缓存加载
        if cache_file is not None and os.path.exists(cache_file):
            self.logger.info(f"Loading class indices from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    class_indices = pickle.load(f)
                self.logger.info("Successfully loaded class indices from cache")
                return class_indices
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, rebuilding...")
        
        # 构建类别索引
        self.logger.info("Building class indices... This may take a while...")
        class_indices = defaultdict(list)
        
        total_samples = len(self.dataset)
        for idx in range(total_samples):
            if (idx + 1) % 1000 == 0:
                self.logger.info(f"Processing {idx + 1}/{total_samples} samples...")
            
            try:
                # 获取数据（不应用transform，只获取segment信息）
                data = self.dataset.get_data(idx)
                segment = data.get('segment', None)
                
                if segment is not None:
                    # 统计该样本包含的类别
                    unique_classes = np.unique(segment)
                    for cls in unique_classes:
                        if 0 <= cls < self.num_classes:
                            class_indices[cls].append(idx)
            except Exception as e:
                self.logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        # 保存到缓存
        if cache_file is not None:
            self.logger.info(f"Saving class indices to cache: {cache_file}")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(class_indices, f)
                self.logger.info("Successfully saved class indices to cache")
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")
        
        # 打印统计信息
        self.logger.info("=" * 60)
        self.logger.info("Class indices statistics:")
        self.logger.info("-" * 60)
        for cls in sorted(class_indices.keys()):
            count = len(class_indices[cls])
            iou_level = self.class_to_iou_level.get(cls, 'normal')
            level_info = f" (IoU level: {iou_level})" if iou_level != 'normal' else ""
            self.logger.info(
                f"  Class {cls:2d}: {count:6d} samples{level_info}"
            )
        self.logger.info("=" * 60)
        
        return dict(class_indices)
    
    def _compute_sampling_strategy(self):
        """
        计算采样策略
        
        返回:
            sampling_strategy: dict, 包含采样配置
        """
        # 计算每个类别的样本数量
        class_counts = {
            cls: len(indices) 
            for cls, indices in self.class_indices.items()
        }
        
        total_samples = sum(class_counts.values())
        
        # 计算每个类别的频率
        class_frequencies = {
            cls: count / total_samples 
            for cls, count in class_counts.items()
        }
        
        # 确定目标采样数量
        if self.target_samples_per_class is None:
            # 使用最少类别的数量作为基准
            min_count = min(class_counts.values()) if class_counts else 1
            base_samples = max(1, min_count)
        else:
            base_samples = self.target_samples_per_class
        
        # 计算每个类别的采样数量
        sampling_counts = {}
        for cls in range(self.num_classes):
            if cls not in class_counts:
                sampling_counts[cls] = 0
                continue
            
            count = class_counts[cls]
            freq = class_frequencies[cls]
            
            # 选择调整策略
            if self.use_relative_adjustment:
                # 策略1：基于原始数量的相对调整（温和策略）
                # 以原始数量为基础，通过multiplier进行温和调整
                target_count = count
                
                # 1. 根据IoU等级调整采样倍数（相对于原始数量）
                iou_level = self.class_to_iou_level.get(cls, None)
                if iou_level is not None:
                    level_config = self.iou_levels[iou_level]
                    multiplier = level_config.get('multiplier', 1.0)
                    target_count = int(target_count * multiplier)
                    self.logger.info(
                        f"  Class {cls} (IoU level: {iou_level}): original={count} -> "
                        f"target={target_count} (x{multiplier})"
                    )
                
                # 2. 高频类别：减少采样（相对于调整后的数量）
                if freq > self.high_freq_threshold:
                    target_count = int(target_count * self.high_freq_multiplier)
                    self.logger.info(
                        f"  Class {cls} (high freq, {freq:.2%}): adjusted -> "
                        f"target={target_count} (x{self.high_freq_multiplier})"
                    )
            else:
                # 策略2：基于最少类别的绝对平衡（传统策略）
                # 以最少类别数量为基础，通过multiplier调整
                target_count = base_samples
                
                # 1. 根据IoU等级调整采样倍数
                iou_level = self.class_to_iou_level.get(cls, None)
                if iou_level is not None:
                    level_config = self.iou_levels[iou_level]
                    multiplier = level_config.get('multiplier', 1.0)
                    target_count = int(target_count * multiplier)
                    self.logger.info(
                        f"  Class {cls} (IoU level: {iou_level}): base={base_samples} -> "
                        f"target={target_count} (x{multiplier})"
                    )
                
                # 2. 高频类别：减少采样
                if freq > self.high_freq_threshold:
                    target_count = int(target_count * self.high_freq_multiplier)
                    self.logger.info(
                        f"  Class {cls} (high freq, {freq:.2%}): adjusted -> "
                        f"target={target_count} (x{self.high_freq_multiplier})"
                    )
            
            # 确保至少采样1个
            sampling_counts[cls] = max(1, target_count)
        
        strategy = {
            'base_samples': base_samples,
            'class_counts': class_counts,
            'class_frequencies': class_frequencies,
            'sampling_counts': sampling_counts,
        }
        
        # 打印采样策略摘要
        self.logger.info("=" * 60)
        self.logger.info("Sampling strategy summary:")
        self.logger.info("-" * 60)
        strategy_type = "Relative Adjustment (基于原始数量)" if self.use_relative_adjustment else "Absolute Balance (基于最少类别)"
        self.logger.info(f"Strategy: {strategy_type}")
        if not self.use_relative_adjustment:
            self.logger.info(f"Base samples per class: {base_samples}")
        self.logger.info("IoU level multipliers:")
        for level_name, level_config in sorted(self.iou_levels.items()):
            classes = level_config.get('classes', [])
            multiplier = level_config.get('multiplier', 1.0)
            self.logger.info(f"  {level_name:12s}: classes={classes}, multiplier={multiplier}x")
        self.logger.info(f"High freq threshold: {self.high_freq_threshold:.1%}")
        self.logger.info(f"High freq multiplier: {self.high_freq_multiplier}x")
        self.logger.info("-" * 60)
        for cls in sorted(sampling_counts.keys()):
            original = class_counts[cls]
            target = sampling_counts[cls]
            freq = class_frequencies[cls]
            ratio = target / original if original > 0 else 0
            iou_level = self.class_to_iou_level.get(cls, 'normal')
            level_str = f" [{iou_level}]" if iou_level != 'normal' else ""
            self.logger.info(
                f"  Class {cls:2d}{level_str:12s}: {original:6d} -> {target:6d} samples "
                f"(freq: {freq:5.2%}, ratio: {ratio:.2f}x)"
            )
        self.logger.info("=" * 60)
        
        return strategy
    
    def __iter__(self):
        """生成采样索引"""
        # 设置随机种子
        if self.seed is not None:
            np.random.seed(self.seed)
        
        sampled_indices = []
        
        # 对每个类别进行采样
        for cls in range(self.num_classes):
            if cls not in self.class_indices:
                continue
            
            indices = self.class_indices[cls]
            target_count = self.sampling_strategy['sampling_counts'][cls]
            
            if len(indices) == 0:
                continue
            
            # 采样策略
            if len(indices) >= target_count:
                # 情况1：样本足够，直接无放回采样
                sampled = np.random.choice(
                    indices, 
                    target_count, 
                    replace=False
                )
            else:
                # 情况2：样本不够（multiplier>1的情况）
                # 策略：全样本 + 随机采样剩余数量
                # 1. 先包含所有原始样本（无放回，保证所有样本都被使用）
                sampled = list(indices)
                
                # 2. 计算剩余需要采样的数量
                remaining_count = target_count - len(indices)
                
                # 3. 对剩余数量进行无放回随机采样
                # 这样可以增加困难样本的曝光，同时保证所有原始样本都被包含
                if remaining_count > 0:
                    additional_sampled = np.random.choice(
                        indices,
                        remaining_count,
                        replace=False
                    )
                    sampled.extend(additional_sampled.tolist())
            
            sampled_indices.extend(sampled)
        
        # 打乱顺序
        if self.shuffle:
            np.random.shuffle(sampled_indices)
        
        return iter(sampled_indices)
    
    def __len__(self):
        """返回采样后的数据集大小"""
        total = sum(self.sampling_strategy['sampling_counts'].values())
        return total

