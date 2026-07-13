"""
Enhanced Text-to-3D Pipeline with cross-attention decoders.

Inherits from TRELLIS's TrellisTextTo3DPipeline and overrides decode_slat()
to pass the original latent through to enhanced decoders for cross-attention.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from trellis.pipelines.trellis_text_to_3d import (
    TrellisTextTo3DPipeline as _TrellisTextTo3DPipeline,
)
from trellis.modules import sparse as sp


class EnhancedTrellisTextTo3DPipeline(_TrellisTextTo3DPipeline):
    """
    Enhanced Text-to-3D Pipeline using LATO-style cross-attention decoders.

    The key enhancement is in decode_slat(): the original latent (slat)
    is passed through to each enhanced decoder, enabling cross-attention
    from decoder features back to the original latent for better feature
    propagation during decoding.

    Usage:
        pipeline = EnhancedTrellisTextTo3DPipeline.from_pretrained(path)
        results = pipeline.run("a chair")
    """

    @staticmethod
    def from_pretrained(path: str) -> "EnhancedTrellisTextTo3DPipeline":
        """
        Load a pretrained model and wrap in enhanced pipeline.

        Args:
            path: The path to the model (local or Hugging Face).

        Returns:
            EnhancedTrellisTextTo3DPipeline instance.
        """
        pipeline = _TrellisTextTo3DPipeline.from_pretrained(path)
        enhanced = EnhancedTrellisTextTo3DPipeline()
        enhanced.__dict__ = pipeline.__dict__
        return enhanced

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent with cross-attention support.

        Passes the original latent (slat) to each decoder so they can
        cross-attend back to it during decoding.

        Args:
            slat: The structured latent sparse tensor.
            formats: The formats to decode to.

        Returns:
            dict: The decoded 3D representations.
        """
        ret = {}
        if 'mesh' in formats:
            decoder = self.models['slat_decoder_mesh']
            if hasattr(decoder, 'use_cross_attn') and decoder.use_cross_attn:
                ret['mesh'] = decoder(slat, original_latent=slat)
            else:
                ret['mesh'] = decoder(slat)

        if 'gaussian' in formats:
            decoder = self.models['slat_decoder_gs']
            if hasattr(decoder, 'use_cross_attn') and decoder.use_cross_attn:
                ret['gaussian'] = decoder(slat, original_latent=slat)
            else:
                ret['gaussian'] = decoder(slat)

        if 'radiance_field' in formats:
            decoder = self.models['slat_decoder_rf']
            if hasattr(decoder, 'use_cross_attn') and decoder.use_cross_attn:
                ret['radiance_field'] = decoder(slat, original_latent=slat)
            else:
                ret['radiance_field'] = decoder(slat)

        return ret

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with enhanced decoder support.

        Uses the enhanced SparseStructureDecoder if available.
        """
        # Sample occupancy latent (same as original)
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(
            num_samples, flow_model.in_channels, reso, reso, reso
        ).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples

        # Decode occupancy latent with enhanced decoder
        decoder = self.models['sparse_structure_decoder']
        logits = decoder(z_s)
        coords = torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].int()

        return coords
