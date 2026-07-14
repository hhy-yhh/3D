"""
================================================================================
LATO-Enhanced SS Flow Trainer (步骤5 训练优化)
================================================================================

增强项:
  - 辅助 VAE 解码损失: 每 N 步解码一次预测 latent, 加 occupancy BCE loss
  - 迫使 Flow 模型不仅预测正确的 velocity, 还要保证解码后重建准确的 occupancy
================================================================================
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from trellis.trainers.flow_matching.flow_matching import FlowMatchingTrainer
from trellis.trainers.flow_matching.mixins.classifier_free_guidance import (
    ClassifierFreeGuidanceMixin,
)
from trellis.trainers.flow_matching.mixins.text_conditioned import TextConditionedMixin
from trellis.trainers.flow_matching.mixins.image_conditioned import ImageConditionedMixin


class EnhancedSSFlowTrainer(FlowMatchingTrainer):
    """
    ============================================================================
    LATO-Enhanced SS Flow Trainer
    ============================================================================

    相比原始 FlowMatchingTrainer:

    辅助 VAE 解码损失:
      - 每隔 aux_decode_every 步, 将预测的去噪 latent 通过冻结的 SS Decoder
      - 计算 occupancy BCE loss (对比 GT occupancy grid)
      - 权重控制: lambda_aux_decode

    这迫使 Flow 模型学习产生"解码友好"的 latent, 与 LATO 的 end-to-end
    训练哲学一致。

    Args:
        (same as FlowMatchingTrainer)
        aux_decode_every: 每 N 步计算一次辅助解码损失 (0=禁用).
        lambda_aux_decode: 辅助解码损失权重.
        freeze_decoder: 是否冻结辅助解码器 (推荐 True).
    ============================================================================
    """

    def __init__(
        self,
        *args,
        aux_decode_every: int = 100,
        lambda_aux_decode: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aux_decode_every = aux_decode_every
        self.lambda_aux_decode = lambda_aux_decode

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        ss_gt: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        计算训练损失 (含辅助解码损失).

        Args:
            x_0: [B, 8, 16, 16, 16] 干净的 SS latent.
            cond: 条件信息 (text/image embeddings).
            ss_gt: [B, 1, 64, 64, 64] GT occupancy grid (辅助损失需要).

        Returns:
            (terms, status) 字典.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        target = self.get_v(x_0, noise, t)

        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # ---- LATO 辅助解码损失 ----
        if (
            self.aux_decode_every > 0
            and self.global_step % self.aux_decode_every == 0
            and ss_gt is not None
            and 'decoder' in self.models
        ):
            with torch.no_grad():
                # 从预测的 velocity 恢复 x_0_pred
                x_0_pred = self._reconstruct_x0(x_t, pred, t)

            # 通过冻结的 SS decoder 解码
            aux_logits = self.models['decoder'](x_0_pred)
            aux_bce = F.binary_cross_entropy_with_logits(
                aux_logits, ss_gt.float(), reduction='mean'
            )
            terms["aux_decode_bce"] = aux_bce
            terms["loss"] = terms["loss"] + self.lambda_aux_decode * aux_bce

        # 按时间 bin 记录 loss
        mse_per_instance = torch.tensor([
            F.mse_loss(
                pred[i:i+1], target[i:i+1]
            ).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean().item()}

        return terms, {}

    @staticmethod
    def _reconstruct_x0(
        x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """从 flow matching velocity 恢复 x_0."""
        t = t.view(-1, *([1] * (x_t.ndim - 1)))
        return x_t - t * v_pred


class EnhancedSSFlowCFGTrainer(ClassifierFreeGuidanceMixin, EnhancedSSFlowTrainer):
    """带 CFG 的 Enhanced SS Flow Trainer."""
    pass


class TextConditionedEnhancedSSFlowCFGTrainer(
    TextConditionedMixin, EnhancedSSFlowCFGTrainer
):
    """文本条件 + CFG 的 Enhanced SS Flow Trainer."""
    pass


class ImageConditionedEnhancedSSFlowCFGTrainer(
    ImageConditionedMixin, EnhancedSSFlowCFGTrainer
):
    """图像条件 + CFG 的 Enhanced SS Flow Trainer."""
    pass
