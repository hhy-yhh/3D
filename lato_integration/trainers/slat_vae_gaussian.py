"""
Enhanced SLat VAE Gaussian Trainer using LATO-style posterior.kl()
and passing latent context to the enhanced decoder.

Inherits from TRELLIS's SLatVaeGaussianTrainer.
"""

from typing import Dict, List, Tuple

import torch
from easydict import EasyDict as edict

from trellis.trainers.vae.structured_latent_vae_gaussian import (
    SLatVaeGaussianTrainer as _SLatVaeGaussianTrainer,
)
from trellis.modules.sparse import SparseTensor
from trellis.representations import Gaussian
from trellis.utils.loss_utils import l1_loss, l2_loss, ssim, lpips


class EnhancedSLatVaeGaussianTrainer(_SLatVaeGaussianTrainer):
    """
    Enhanced SLat VAE Gaussian Trainer with posterior.kl() and latent context.

    Key enhancements:
    1. Uses DiagonalGaussianDistribution.kl() for proper KL loss
    2. Passes the original latent through to the decoder for cross-attention

    Args:
        (same as SLatVaeGaussianTrainer)
    """

    def training_losses(
        self,
        feats: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses with enhanced posterior handling.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        # Use enhanced encoder with DiagonalGaussianDistribution
        z, posterior = self.training_models['encoder'](
            feats, sample_posterior=True, return_raw=True
        )

        # Pass original latent for cross-attention in enhanced decoder
        decoder = self.training_models['decoder']
        if hasattr(decoder, 'use_cross_attn') and decoder.use_cross_attn:
            reps = decoder(z, original_latent=z)
        else:
            reps = decoder(z)

        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics)

        terms = edict(loss=0.0, rec=0.0)

        rec_image = render_results['color']
        gt_image = (
            image * alpha[:, None]
            + (1 - alpha[:, None]) * render_results['bg_color'][..., None, None]
        )

        if self.loss_type == 'l1':
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        if self.lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]
        terms["loss"] = terms["loss"] + terms["rec"]

        # Use DiagonalGaussianDistribution.kl() instead of manual formula
        terms["kl"] = posterior.kl()
        if terms["kl"].ndim > 0:
            terms["kl"] = terms["kl"].mean()
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss

        status = self._get_status(z, reps)

        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}
        return terms, status
