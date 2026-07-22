"""
LATO Integration for TRELLIS — v3: 全 LATO Encoder/Decoder + TRELLIS Flow 生成

架构（v3）:
  Encoder: LATO VoxelVAE.encode() — 替代 TRELLIS SS Encoder + SLat Encoder
  Decoder: LATO VoxelVAE.decode() + LatoStructureHead — 替代 TRELLIS SS Decoder + SLat Decoder
  Flow:   TRELLIS SS Flow + SLat Flow — 仅中间生成部分保留 TRELLIS

Key components:
  - LatoStructureHead: 轻量 3D CNN，替代 SS Decoder，16³→128³ 直接输出
  - DiagonalGaussianDistribution: FP16 安全 VAE 后验（保留给 latent consistency loss）
  - SparseTransformerCrossBase: 交叉注意力基类（保留给可能的扩展）
  - ConnectionHead: LATO 边预测头（推理用）

Usage:
    from lato_integration import LatoStructureHead, coords_from_occupancy
    from lato_integration.flow import EnhancedSSFlowModel, EnhancedSLatFlowModel
"""

# Foundation utilities (保留)
from .utils import DiagonalGaussianDistribution
from .base import SparseTransformerCrossBase
from .vertex_encoder import ConnectionHead

# === v3: LatoStructureHead — 替代 TRELLIS SS Decoder ===
from .structure_head import LatoStructureHead, coords_from_occupancy

# === 工具类（从 decoder_mesh.py 保留）===
from .decoder_mesh import SparsePredictionHead

# === Enhanced pipeline（保留，需更新）===
from .pipeline import EnhancedTrellisTextTo3DPipeline

__all__ = [
    # Foundation
    "DiagonalGaussianDistribution",
    "SparseTransformerCrossBase",
    "ConnectionHead",
    # v3: Structure Head
    "LatoStructureHead",
    "coords_from_occupancy",
    # Utility
    "SparsePredictionHead",
    # Pipeline
    "EnhancedTrellisTextTo3DPipeline",
]

# ========================================================================
# 以下模块已在 v3 中废弃（由 LATO VoxelVAE 替代）：
# ========================================================================
#
#   - lato_integration.encoder.EnhancedSLatEncoder
#     → 替代: LATO VoxelVAE.encode()（encode_lato_latent_v2.py）
#
#   - lato_integration.sparse_structure_vae.EnhancedSparseStructureEncoder
#     → 替代: LATO VoxelVAE.encode()（encode_lato_latent_v2.py）
#
#   - lato_integration.sparse_structure_vae.EnhancedSparseStructureDecoder
#     → 替代: LatoStructureHead（structure_head.py）
#
#   - lato_integration.decoder_gs / decoder_rf / decoder_mesh（Decoder 类）
#     → 替代: LATO VoxelVAE.decode()
#
#   - lato_integration.trainers.sparse_structure_vae / slat_vae_*
#     → 替代: 不再训练 VAE，只训练 Flow 模型
# ========================================================================
