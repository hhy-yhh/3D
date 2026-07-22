"""
[DEPRECATED in v3] — 由 LATO VoxelVAE.encode() 替代。

v2 中的 EnhancedSLatEncoder 继承自 TRELLIS SLatEncoder，
在 v3 中 SLat 编码完全由 LATO VoxelVAE.encode() 处理（见 encode_lato_latent_v2.py）。

此文件保留以保持向后兼容，但不应在新代码中使用。
"""

# 如需 EnhancedSLatEncoder 的向后兼容，取消以下注释：
# from trellis.models.structured_latent_vae.encoder import SLatEncoder as _SLatEncoder
# from .utils import DiagonalGaussianDistribution
#
# class EnhancedSLatEncoder(_SLatEncoder):
#     """[DEPRECATED] Use LATO VoxelVAE.encode() instead."""
#     ...
