"""
LATO-compatible SLat Flow Model.
Inherits from SLatFlowModel with LATO-specific defaults.
"""

from .structured_latent_flow import SLatFlowModel


class LATOSLatFlowModel(SLatFlowModel):
    """
    LATO-compatible SLat Flow Model.
    
    Only changes 3 default parameters:
    - resolution: 64 -> 128 (higher resolution for LATO)
    - in_channels: 8 -> 16 (LATO latent dimension)
    - out_channels: 8 -> 16 (LATO latent dimension)
    """
    def __init__(
        self,
        resolution: int = 128,      # 原: 64
        in_channels: int = 16,      # 原: 8
        out_channels: int = 16,     # 原: 8
        **kwargs
    ):
        super().__init__(
            resolution=resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
