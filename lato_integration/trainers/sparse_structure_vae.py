"""
Enhanced Sparse Structure VAE Trainer using LATO-style posterior.kl().

Inherits from TRELLIS's SparseStructureVaeTrainer and replaces the manual
KL loss computation with DiagonalGaussianDistribution.kl().
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from trellis.trainers.vae.sparse_structure_vae import (
    SparseStructureVaeTrainer as _SparseStructureVaeTrainer,
)


class EnhancedSparseStructureVaeTrainer(_SparseStructureVaeTrainer):
    """
    Enhanced Sparse Structure VAE Trainer with proper posterior.kl().

    The key enhancement is using DiagonalGaussianDistribution.kl()
    instead of the manual KL computation, which benefits from:
    - Clamped logvar for numerical stability
    - Support for non-standard prior distributions

    Args:
        (same as SparseStructureVaeTrainer)
    """

    def training_losses(
        self,
        ss: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses using DiagonalGaussianDistribution.kl().

        Args:
            ss: The [N x 1 x H x W x D] tensor of binary sparse structure.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        # 确保数据在 GPU 上
        encoder = self.training_models['encoder']
        if next(encoder.parameters()).is_cuda:
            ss = ss.cuda()
        
        # 确保为 float32
        ss = ss.float()
        
        # 直接前向传播（不使用 autocast）
        z, posterior = self.training_models['encoder'](
            ss, sample_posterior=True, return_raw=True
        )
        logits = self.training_models['decoder'](z)

        terms = edict(loss=0.0)
        
        # Dice Loss (默认)
        logits_sigmoid = F.sigmoid(logits)
        terms["dice"] = 1 - (
            (2 * (logits_sigmoid * ss).sum() + 1)
            / (logits_sigmoid.sum() + ss.sum() + 1)
        )
        terms["loss"] = terms["loss"] + terms["dice"]

        # KL Loss
        terms["kl"] = posterior.kl()
        if terms["kl"].ndim > 0:
            terms["kl"] = terms["kl"].mean()
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        return terms, {}
