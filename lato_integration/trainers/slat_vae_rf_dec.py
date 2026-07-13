"""
Enhanced SLat VAE Radiance Field Decoder Trainer.

Inherits from TRELLIS's SLatVaeRadianceFieldDecoderTrainer and passes
latent context to the enhanced decoder for cross-attention.

Note: The RF decoder trainer typically only trains the decoder (frozen encoder),
so it doesn't compute KL loss. This trainer adds latent context passing.
"""

from typing import Dict, Tuple

import torch
from easydict import EasyDict as edict

from trellis.trainers.vae.structured_latent_vae_rf_dec import (
    SLatVaeRadianceFieldDecoderTrainer as _SLatVaeRadianceFieldDecoderTrainer,
)
from trellis.modules.sparse import SparseTensor
from trellis.utils.loss_utils import l1_loss, l2_loss, ssim, lpips


class EnhancedSLatVaeRadianceFieldDecoderTrainer(_SLatVaeRadianceFieldDecoderTrainer):
    """
    Enhanced SLat VAE Radiance Field Decoder Trainer with latent context.

    Key enhancement:
    - Passes the original latent through to the decoder for cross-attention

    Args:
        (same as SLatVaeRadianceFieldDecoderTrainer)
    """

    def training_losses(
        self,
        latents: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses with latent context for cross-attention.

        Args:
            latents: The [N x * x C] sparse latent tensor.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
        """
        # Pass original latent for cross-attention in enhanced decoder
        decoder = self.training_models['decoder']
        if hasattr(decoder, 'use_cross_attn') and decoder.use_cross_attn:
            reps = decoder(latents, original_latent=latents)
        else:
            reps = decoder(latents)

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

        return terms, {}
