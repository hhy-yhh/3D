"""
Enhanced SLatEncoder with LATO-style VAE posterior.

Inherits from TRELLIS's original SLatEncoder and uses
DiagonalGaussianDistribution for proper VAE posterior handling.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis.models.structured_latent_vae.encoder import SLatEncoder as _SLatEncoder
from trellis.modules import sparse as sp

from .utils import DiagonalGaussianDistribution


class EnhancedSLatEncoder(_SLatEncoder):
    """
    Enhanced SLat VAE Encoder with LATO-style DiagonalGaussianDistribution.

    Inherits from TRELLIS's original SLatEncoder. The key enhancement is
    using DiagonalGaussianDistribution which provides:
    - Clamped logvar [-30, 20] for numerical stability
    - Proper KL divergence computation via posterior.kl()
    - Clean sample() / mode() interface

    Args:
        (same as SLatEncoder)
    """

    def forward(self, x: sp.SparseTensor, sample_posterior: bool = True, return_raw: bool = False):
        """
        Forward pass with DiagonalGaussianDistribution posterior.

        Args:
            x: Input sparse tensor.
            sample_posterior: If True, sample from posterior; else return mode.
            return_raw: If True, return (z, posterior) tuple.

        Returns:
            z: Latent sparse tensor, or (z, posterior) if return_raw=True.
        """
        h = super(_SLatEncoder, self).forward(x)  # Call SparseTransformerBase.forward
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)

        # Use DiagonalGaussianDistribution for proper posterior handling
        posterior = DiagonalGaussianDistribution(h.feats, feat_dim=-1)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = h.replace(z)

        if return_raw:
            return z, posterior
        else:
            return z
