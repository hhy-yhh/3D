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

    Adapted from LATO: vertex_encoder.py lines 55-68.

    Given features of two vertices [feat_u | feat_v], predicts the logit
    for whether an edge exists between them.

    Args:
        channels: Input feature dimension (per vertex, so 2*channels after concat).
        out_channels: Output dimension (typically 1 for binary edge logit).
        mlp_ratio: Hidden layer multiplier.
    """

    def __init__(self, channels: int, out_channels: int = 1, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_channels = int(channels * 2 * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Concatenated vertex pair features [P, channels * 2].

        Returns:
            Edge logits [P, out_channels].
        """
        return self.mlp(x)
