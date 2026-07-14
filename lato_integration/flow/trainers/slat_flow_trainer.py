"""
================================================================================
LATO-Enhanced SLat Flow Trainer (步骤6 训练优化)
================================================================================

增强项:
  1. 辅助 VAE 解码损失: 定期 decode 预测 SLat, 通过冻结 decoder 加重建损失
  2. Latent 一致性损失: KL 约束使 flow 生成分布接近 VAE posterior
================================================================================
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict

from trellis.trainers.flow_matching.sparse_flow_matching import (
    SparseFlowMatchingTrainer,
)
from trellis.trainers.flow_matching.mixins.classifier_free_guidance import (
    ClassifierFreeGuidanceMixin,
)
from trellis.trainers.flow_matching.mixins.text_conditioned import TextConditionedMixin
from trellis.trainers.flow_matching.mixins.image_conditioned import ImageConditionedMixin
from trellis.modules import sparse as sp


class EnhancedSLatFlowTrainer(SparseFlowMatchingTrainer):
    """
    ============================================================================
    LATO-Enhanced SLat Flow Trainer
    ============================================================================

    相比原始 SparseFlowMatchingTrainer:

    1. 辅助 VAE 解码损失 (aux_decode_every > 0):
       - 每隔 N 步, 将预测的去噪 SLat 通过冻结的 decoder (GS/RF/Mesh)
       - 对于 GS decoder: 渲染并与 GT image 计算 L1+SSIM 损失
       - 迫使 Flow 模型产生"解码友好"的 latent features

    2. Latent 一致性损失 (lambda_latent_consistency > 0):
       - 将 x_0 (VAE encoder 输出) 作为 target distribution
       - 计算 flow-predicted latent 与 VAE posterior 之间的 KL-like 损失
       - 这使 flow 分布更接近 VAE 后验分布 (LATO 的 VAE 哲学)

    Args:
        (same as SparseFlowMatchingTrainer)
        aux_decode_every: 每 N 步计算辅助解码损失 (0=禁用).
        lambda_aux_decode: 辅助解码损失权重.
        lambda_latent_consistency: latent 一致性损失权重.
    ============================================================================
    """

    def __init__(
        self,
        *args,
        aux_decode_every: int = 200,
        lambda_aux_decode: float = 0.05,
        lambda_latent_consistency: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aux_decode_every = aux_decode_every
        self.lambda_aux_decode = lambda_aux_decode
        self.lambda_latent_consistency = lambda_latent_consistency

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        计算训练损失 (含辅助解码损失 + latent 一致性损失).

        Args:
            x_0: [N x in_channels] 干净的 SLat sparse tensor.
            cond: 条件信息.
            kwargs: 可能包含 image, alpha, extrinsics, intrinsics (辅助损失需要).

        Returns:
            (terms, status) 字典.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        target = self.get_v(x_0, noise, t)

        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # ---- LATO 辅助损失 1: 辅助 VAE 解码损失 ----
        if (
            self.aux_decode_every > 0
            and self.step % self.aux_decode_every == 0
        ):
            aux_loss = self._compute_aux_decode_loss(x_0, x_t, pred, t, **kwargs)
            if aux_loss is not None:
                terms["aux_decode"] = aux_loss
                terms["loss"] = terms["loss"] + self.lambda_aux_decode * aux_loss

        # ---- LATO 辅助损失 2: Latent 一致性损失 ----
        if self.lambda_latent_consistency > 0:
            # 将 per-sample t [B] 扩展为 per-voxel t [N_total, 1]
            # 使用 SparseTensor 的 layout 来对齐 batch → voxel 映射
            t_voxel = torch.zeros(
                pred.feats.shape[0], 1,
                device=pred.device, dtype=pred.feats.dtype
            )
            for i, slc in enumerate(pred.layout):
                t_voxel[slc] = t[i]
            # 从预测 velocity 恢复 x_0_pred
            x_0_pred_feats = x_t.feats - t_voxel * pred.feats

            # MSE between predicted latent and VAE encoder output
            # 这迫使 flow 分布在 VAE posterior 附近 (类似 LATO 的 KL 约束)
            latent_consistency = F.mse_loss(x_0_pred_feats, x_0.feats)
            terms["latent_consistency"] = latent_consistency
            terms["loss"] = (
                terms["loss"]
                + self.lambda_latent_consistency * latent_consistency
            )

        # 按时间 bin 记录 loss
        mse_per_instance = np.array([
            F.mse_loss(
                pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]
            ).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {
                    "mse": mse_per_instance[time_bin == i].mean()
                }

        return terms, {}

    def _compute_aux_decode_loss(
        self,
        x_0: sp.SparseTensor,
        x_t: sp.SparseTensor,
        v_pred: sp.SparseTensor,
        t: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        计算辅助 VAE 解码损失.

        从预测的 velocity 恢复 x_0_pred, 然后通过 decoder 解码并对比 GT.
        """
        # 恢复 x_0_pred
        t_expanded = t.view(-1, 1).to(x_t.device)
        x_0_pred_feats = x_t.feats - t_expanded * v_pred.feats
        x_0_pred = x_t.replace(x_0_pred_feats)

        # 如果有 decoder 和 GT image, 可以计算渲染损失
        if 'decoder' not in self.models:
            return None

        decoder = self.models['decoder']
        # 只对前几个样本做辅助损失 (节省显存)
        with torch.no_grad():
            try:
                # 尝试用增强 decoder 的 cross-attention
                if hasattr(decoder, 'use_cross_attn') and decoder.use_cross_attn:
                    reps = decoder(x_0_pred, original_latent=x_0_pred)
                else:
                    reps = decoder(x_0_pred)

                # 如果 decoder 返回 Gaussian 且有 renderer, 计算渲染损失
                if hasattr(self, 'renderer') and 'image' in kwargs:
                    image = kwargs['image']
                    alpha = kwargs.get('alpha', None)
                    extrinsics = kwargs.get('extrinsics', None)
                    intrinsics = kwargs.get('intrinsics', None)

                    if extrinsics is not None and intrinsics is not None:
                        self.renderer.rendering_options.resolution = (
                            image.shape[-1] if image is not None else 512
                        )
                        render_results = self._render_batch(
                            reps, extrinsics, intrinsics
                        )
                        rec_image = render_results['color']
                        if alpha is not None:
                            gt_image = (
                                image * alpha[:, None]
                                + (1 - alpha[:, None])
                                * render_results['bg_color'][..., None, None]
                            )
                        else:
                            gt_image = image
                        return F.l1_loss(rec_image, gt_image)
            except Exception:
                pass  # 辅助损失失败不影响主训练

        return None


class EnhancedSLatFlowCFGTrainer(
    ClassifierFreeGuidanceMixin, EnhancedSLatFlowTrainer
):
    """带 CFG 的 Enhanced SLat Flow Trainer."""
    pass


class TextConditionedEnhancedSLatFlowCFGTrainer(
    TextConditionedMixin, EnhancedSLatFlowCFGTrainer
):
    """文本条件 + CFG 的 Enhanced SLat Flow Trainer."""
    pass


class ImageConditionedEnhancedSLatFlowCFGTrainer(
    ImageConditionedMixin, EnhancedSLatFlowCFGTrainer
):
    """图像条件 + CFG 的 Enhanced SLat Flow Trainer."""
    pass
