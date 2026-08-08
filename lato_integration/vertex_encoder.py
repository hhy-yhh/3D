"""
ConnectionHead adapted from LATO for edge/topology prediction.

Predicts connections (edges) between vertex pairs by concatenating
their features through a small MLP.
"""

import torch
import torch.nn as nn


class ConnectionHead(nn.Module):
    """
    Small MLP head for edge or connection logits between vertex pairs.

    Identical to LATO: vertex_encoder.py lines 55-68.

    Given features of two vertices [feat_u | feat_v], predicts the logit
    for whether an edge exists between them.

    Args:
        channels: Concatenated vertex pair feature dim (feat_dim * 2).
        out_channels: Output dimension (typically 1 for binary edge logit).
        mlp_ratio: Hidden layer multiplier.
    """

    def __init__(self, channels: int, out_channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
