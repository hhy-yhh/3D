"""
================================================================================
LATO SS Flow Trainer（v3 — 训练目标改为 LATO VoxelVAE coords）
================================================================================

v3 增强项:
  - 辅助 occupancy BCE 损失：用 LatoStructureHead 解码预测 latent → occ@128³
  - 训练目标从 TRELLIS SS Encoder latent (16³×8) 改为 LATO VoxelVAE coords
  - LatoStructureHead 与 SS Flow 联合训练

用法:
    trainer = LatoSSFlowCFGTrainer(
        models={'denoiser': ss_flow, 'structure_head': structure_head},
        dataset=dataset,
        lambda_occupancy=0.1,
        ...
    )
================================================================================
"""

from typing import Dict, Optional, Tuple

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


class LatoSSFlowTrainer(FlowMatchingTrainer):
    """
    ============================================================================
    LATO SS Flow Trainer（v3）
    ============================================================================

    v3 变化:
      - 训练目标从 TRELLIS SS Encoder latent → LATO VoxelVAE coords (128³)
      - 辅助 loss 使用 LatoStructureHead（可训练）而非 TRELLIS SS Decoder（冻结）
      - Occupancy BCE loss 在 128³ 分辨率计算

    Args:
        (same as FlowMatchingTrainer)
        lambda_occupancy: 辅助 occupancy BCE 损失权重 (0=禁用).
        aux_decode_every: 每 N 步计算一次辅助损失 (0=每步, -1=禁用).
    """

    def __init__(
        self,
        *args,
        lambda_occupancy: float = 0.1,
        aux_decode_every: int = 0,  # v3: 默认每步计算（轻量 head）
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_occupancy = lambda_occupancy
        self.aux_decode_every = aux_decode_every

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        ss_occupancy_128: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        计算训练损失（含辅助 occupancy BCE 损失）。

        Args:
            x_0: [B, 8, 16, 16, 16] 干净的 SS latent（flow matching 目标）。
            cond: 条件信息 (text/image embeddings)。
            ss_occupancy_128: [B, 1, 128, 128, 128] GT occupancy @ res128
                              （来自 LATO VoxelVAE.encode() 的 coords）。

        Returns:
            (terms, status) 字典。
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

        # ---- v3: LatoStructureHead 辅助 occupancy BCE loss @ 128³ ----
        should_decode = (
            self.lambda_occupancy > 0
            and 'structure_head' in self.training_models
            and ss_occupancy_128 is not None
        )
        if self.aux_decode_every < 0:
            should_decode = False
        elif self.aux_decode_every > 0:
            should_decode = should_decode and (self.global_step % self.aux_decode_every == 0)

        if should_decode:
            # 从预测的 velocity 恢复 x_0_pred
            x_0_pred = self._reconstruct_x0(x_t, pred, t)

            # LatoStructureHead: 16³ → 128³ occupancy
            # 🔧 使用 autocast (fp16) 节省显存 — 128³ 激活值巨大（~2GB @ fp32/B=4）
            with torch.autocast(device_type='cuda', enabled=self.fp16_mode is not None):
                occ_logits = self.training_models['structure_head'](x_0_pred)
                occ_bce = F.binary_cross_entropy_with_logits(
                    occ_logits, ss_occupancy_128.float(), reduction='mean'
                )
            terms["occ_bce_128"] = occ_bce
            terms["loss"] = terms["loss"] + self.lambda_occupancy * occ_bce

        # 按时间 bin 记录 loss
        mse_per_instance = torch.tensor([
            F.mse_loss(pred[i:i+1], target[i:i+1]).item()
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


class LatoSSFlowCFGTrainer(ClassifierFreeGuidanceMixin, LatoSSFlowTrainer):
    """带 CFG 的 LATO SS Flow Trainer（v3）。"""
    pass


class TextConditionedLatoSSFlowCFGTrainer(
    TextConditionedMixin, LatoSSFlowCFGTrainer
):
    """文本条件 + CFG 的 LATO SS Flow Trainer（v3）。"""
    pass


class ImageConditionedLatoSSFlowCFGTrainer(
    ImageConditionedMixin, LatoSSFlowCFGTrainer
):
    """图像条件 + CFG 的 LATO SS Flow Trainer（v3）。"""
    pass


# ============================================================================
# 向后兼容别名（v2 旧名 → v3 新名）
# ============================================================================
EnhancedSSFlowTrainer = LatoSSFlowTrainer
EnhancedSSFlowCFGTrainer = LatoSSFlowCFGTrainer
TextConditionedEnhancedSSFlowCFGTrainer = TextConditionedLatoSSFlowCFGTrainer
ImageConditionedEnhancedSSFlowCFGTrainer = ImageConditionedLatoSSFlowCFGTrainer
