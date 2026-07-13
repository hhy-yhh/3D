"""
Enhanced SparseStructure Encoder and Decoder with LATO-style VAE posterior.

Inherits from TRELLIS's original SparseStructureEncoder/Decoder and enhances
them with proper DiagonalGaussianDistribution posterior handling.
"""

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis.models.sparse_structure_vae import (
    SparseStructureEncoder as _SparseStructureEncoder,
    SparseStructureDecoder as _SparseStructureDecoder,
    norm_layer,
    ResBlock3d,
)
from trellis.modules.utils import zero_module

from .utils import DiagonalGaussianDistribution


class EnhancedSparseStructureEncoder(_SparseStructureEncoder):
    """
    Enhanced Sparse Structure Encoder with LATO-style VAE posterior.
    """

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False):
        h = self.input_layer(x)
        h = h.type(self.dtype)

        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)

        posterior = DiagonalGaussianDistribution(h, feat_dim=1)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        if return_raw:
            return z, posterior
        return z


class EnhancedSparseStructureDecoder(_SparseStructureDecoder):
    """
    Enhanced Sparse Structure Decoder with occupancy pruning.
    """

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
        use_pruning: bool = True,
    ):
        # 🔧 关键修复：必须在 super().__init__() 之前设置 use_pruning
        # 因为父类的 __init__ 会调用 self.convert_to_fp16()
        self.use_pruning = use_pruning

        super().__init__(
            out_channels=out_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            channels=channels,
            num_res_blocks_middle=num_res_blocks_middle,
            norm_type=norm_type,
            use_fp16=use_fp16,
        )

        if use_pruning:
            self.pruning_head = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv3d(out_channels, 1, 3, padding=1),
            )
            # 🔧 Freeze: SS VAE training never uses pruning_head
            # Unused params with requires_grad crash fp16_mode='inflat_all'
            for p in self.pruning_head.parameters():
                p.requires_grad_(False)
            # Parent's __init__ already called convert_to_fp16() before
            # pruning_head existed — manually convert it now.
            if use_fp16:
                for m in self.pruning_head:
                    if hasattr(m, 'weight'):
                        m.to(torch.float16)
        else:
            self.pruning_head = None

    def enable_pruning_grad(self):
        """Enable gradients on the pruning head (for mesh fine-tuning)."""
        if self.use_pruning and self.pruning_head is not None:
            for p in self.pruning_head.parameters():
                p.requires_grad_(True)

    def forward(self, x: torch.Tensor, return_pruning_logits: bool = False):
        h = self.input_layer(x)
        h = h.type(self.dtype)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)

        if self.use_pruning and return_pruning_logits and self.pruning_head is not None:
            pruning_logits = self.pruning_head(h)
            return h, pruning_logits

        return h
