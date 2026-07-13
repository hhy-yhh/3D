"""
Enhanced Radiance Field Decoder with cross-attention to the original latent.

Inherits from TRELLIS's SLatRadianceFieldDecoder and adds cross-attention
so decoder features can re-attend to the original VAE latent during decoding.
"""

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis.models.structured_latent_vae.decoder_rf import (
    SLatRadianceFieldDecoder as _SLatRadianceFieldDecoder,
)
from trellis.models.structured_latent_vae.decoder_rf import (
    ElasticSLatRadianceFieldDecoder as _ElasticSLatRadianceFieldDecoder,
)
from trellis.modules import sparse as sp
from trellis.representations import Strivec
from trellis.models.sparse_elastic_mixin import SparseTransformerElasticMixin

from .base import SparseTransformerCrossBase


class EnhancedSLatRadianceFieldDecoder(_SLatRadianceFieldDecoder):
    """
    Enhanced SLat Radiance Field Decoder with cross-attention to original latent.

    Follows the same pattern as EnhancedSLatGaussianDecoder: after the main
    self-attention transformer body, a cross-attention block attends back
    to the original latent features for better feature propagation.

    Args:
        (same as SLatRadianceFieldDecoder)
        use_cross_attn: Whether to enable cross-attention to latent.
        cross_attn_num_blocks: Number of cross-attention blocks.
    """

    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        use_cross_attn: bool = True,
        cross_attn_num_blocks: int = 2,
    ):
        super().__init__(
            resolution=resolution,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            representation_config=representation_config,
        )
        self.use_cross_attn = use_cross_attn
        self._latent_channels = latent_channels

        if use_cross_attn:
            # Project latent to context dimension for cross-attention
            self.latent_proj = sp.SparseLinear(latent_channels, model_channels)

            # Cross-attention block: decoder features attend back to latent
            self.cross_attn = SparseTransformerCrossBase(
                in_channels=model_channels,
                model_channels=model_channels,
                context_channels=model_channels,
                num_blocks=cross_attn_num_blocks,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                mlp_ratio=mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                pe_mode=pe_mode,
                use_fp16=use_fp16,
                use_checkpoint=use_checkpoint,
                qk_rms_norm=qk_rms_norm,
            )

            if use_fp16:
                self.cross_attn.convert_to_fp16()
                self.latent_proj.apply(
                    lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None
                )

    def forward(self, x: sp.SparseTensor, original_latent: Optional[sp.SparseTensor] = None) -> List[Strivec]:
        """
        Forward pass with optional cross-attention to the original latent.

        Args:
            x: Input latent sparse tensor.
            original_latent: The original VAE latent to cross-attend to.
                             If None, falls back to standard behavior.

        Returns:
            List of Strivec radiance field representations.
        """
        h = super(_SLatRadianceFieldDecoder, self).forward(x)  # SparseTransformerBase.forward

        # Apply cross-attention to original latent before output layer
        if self.use_cross_attn and original_latent is not None:
            ctx = self.latent_proj(original_latent)
            h = self.cross_attn(h, ctx)

        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)

    def convert_to_fp16(self) -> None:
        """Convert the torso of the model to float16."""
        super().convert_to_fp16()
        if self.use_cross_attn:
            self.cross_attn.convert_to_fp16()
            self.latent_proj.apply(
                lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None
            )

    def convert_to_fp32(self) -> None:
        """Convert the torso of the model to float32."""
        super().convert_to_fp32()
        if self.use_cross_attn:
            self.cross_attn.convert_to_fp32()
            self.latent_proj.apply(
                lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None
            )


class EnhancedElasticSLatRadianceFieldDecoder(SparseTransformerElasticMixin, EnhancedSLatRadianceFieldDecoder):
    """
    Enhanced SLat Radiance Field Decoder with elastic memory management and cross-attention.
    """
    pass
