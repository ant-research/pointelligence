import os
import copy
import torch
import torch.nn as nn
from collections import OrderedDict

from .default import HookBase
from .builder import HOOKS
from pointcept.utils.comm import is_main_process


@HOOKS.register_module()
class EMAHook(HookBase):
    """Exponential Moving Average Hook
    
    在训练过程中维护模型参数的指数移动平均，用于提升泛化性能。
    
    Args:
        decay (float): EMA衰减率，范围[0,1]。值越大，历史权重越大。
                      推荐值：0.996-0.998（对应论文中的alpha）
        decay_type (str): 衰减类型，'fixed'或'warmup'
        warmup_steps (int): Warmup步数，在warmup期间使用较小的decay
        update_freq (int): 更新频率，每N步更新一次EMA（减少开销）
        save_ema_model (bool): 是否保存EMA模型
        use_bn_recompute (bool): 是否重新计算BatchNorm统计量（推荐True）
    """
    
    def __init__(
        self,
        decay=0.996,
        decay_type="warmup",
        warmup_steps=10,
        update_freq=1,
        save_ema_model=True,
        use_bn_recompute=True,
    ):
        super().__init__()
        self.decay = decay
        self.decay_type = decay_type
        self.warmup_steps = warmup_steps
        self.update_freq = update_freq
        self.save_ema_model = save_ema_model
        self.use_bn_recompute = use_bn_recompute
        
        self.ema_model = None
        self.current_step = 0
        self.ema_decay = decay
        
    def before_train(self):
        """初始化EMA模型"""
        if is_main_process():
            self.trainer.logger.info(
                f"Initializing EMA Hook with decay={self.decay}, "
                f"decay_type={self.decay_type}, update_freq={self.update_freq}"
            )
        
        # 创建EMA模型的深拷贝
        # 注意：需要处理DDP模型的情况
        model = self.trainer.model
        if hasattr(model, 'module'):
            # DDP模型
            self.ema_model = copy.deepcopy(model.module)
        else:
            # 单GPU模型
            self.ema_model = copy.deepcopy(model)
        
        # EMA模型设置为评估模式，不参与训练
        self.ema_model.eval()
        
        # 冻结EMA模型参数，不计算梯度
        for param in self.ema_model.parameters():
            param.requires_grad = False
        
        self.current_step = 0
        
        if is_main_process():
            self.trainer.logger.info("EMA model initialized successfully")
    
    def _get_ema_decay(self):
        """获取当前步的EMA衰减率"""
        if self.decay_type == "warmup":
            # Warmup策略：在训练初期使用较小的decay
            # 公式：min(decay, (step + 1) / (step + warmup_steps))
            # 这确保在初期EMA更新更快，逐渐过渡到目标decay
            warmup_decay = min(
                self.decay,
                (self.current_step + 1) / (self.current_step + self.warmup_steps)
            )
            return warmup_decay
        else:
            # 固定decay
            return self.decay
    
    def after_step(self):
        """每步后更新EMA模型"""
        self.current_step += 1
        
        # 根据update_freq决定是否更新
        if self.current_step % self.update_freq != 0:
            return
        
        # 获取当前EMA衰减率
        self.ema_decay = self._get_ema_decay()
        
        # 更新EMA模型参数
        # 公式：x_ema = alpha * x_ema + (1 - alpha) * x
        # 其中 alpha = ema_decay
        model = self.trainer.model
        if hasattr(model, 'module'):
            # DDP模型
            current_state_dict = model.module.state_dict()
        else:
            # 单GPU模型
            current_state_dict = model.state_dict()
        
        ema_state_dict = self.ema_model.state_dict()
        
        # 更新每个参数
        for key in ema_state_dict.keys():
            if key in current_state_dict:
                # 只更新可训练参数（跳过running_mean, running_var等）
                if current_state_dict[key].requires_grad:
                    ema_state_dict[key].data.mul_(self.ema_decay).add_(
                        current_state_dict[key].data, alpha=1 - self.ema_decay
                    )
                else:
                    # 对于非可训练参数（如BN的running stats），直接复制
                    ema_state_dict[key].data.copy_(current_state_dict[key].data)
        
        self.ema_model.load_state_dict(ema_state_dict)
    
    def after_epoch(self):
        """每个epoch后，如果需要，重新计算BN统计量"""
        if not self.use_bn_recompute:
            return
        
        # 根据论文，重新计算BN统计量可以允许使用更大的averaging window
        # 这里我们可以在验证前重新计算BN统计量
        if is_main_process() and self.trainer.cfg.evaluate:
            self.trainer.logger.info("Recomputing BatchNorm statistics for EMA model...")
            self._recompute_bn_stats()
    
    def _recompute_bn_stats(self):
        """重新计算BatchNorm统计量
        
        根据论文，EMA模型中的BN统计量可能不准确，因为它们是基于
        训练过程中的running stats计算的。重新计算可以提升性能。
        """
        self.ema_model.train()  # 设置为训练模式以更新BN统计量
        
        # 使用训练数据的一个子集重新计算BN统计量
        # 为了效率，我们只使用少量样本
        num_samples = min(1000, len(self.trainer.train_loader.dataset))
        sample_indices = torch.randperm(len(self.trainer.train_loader.dataset))[:num_samples]
        
        with torch.no_grad():
            for idx in sample_indices:
                data_dict = self.trainer.train_loader.dataset[idx]
                # 构建batch
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].unsqueeze(0).cuda()
                
                # 前向传播以更新BN统计量
                _ = self.ema_model(data_dict)
        
        self.ema_model.eval()  # 恢复评估模式
    
    def after_train(self):
        """训练结束后保存EMA模型"""
        if not self.save_ema_model or not is_main_process():
            return
        
        save_path = os.path.join(self.trainer.cfg.save_path, "model")
        os.makedirs(save_path, exist_ok=True)
        
        ema_model_path = os.path.join(save_path, "model_ema.pth")
        self.trainer.logger.info(f"Saving EMA model to: {ema_model_path}")
        
        # 保存EMA模型
        torch.save(
            {
                "epoch": self.trainer.epoch + 1,
                "state_dict": self.ema_model.state_dict(),
                "ema_decay": self.ema_decay,
                "best_metric_value": self.trainer.best_metric_value,
            },
            ema_model_path + ".tmp",
        )
        os.replace(ema_model_path + ".tmp", ema_model_path)
        
        self.trainer.logger.info("EMA model saved successfully")
    
    def get_ema_model(self):
        """获取EMA模型（用于评估）"""
        return self.ema_model




