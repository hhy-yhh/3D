"""
LATO Integration for TRELLIS — Enhanced Text-to-3D Generation

This package integrates LATO's (ICML 2026) superior VAE encode/decode
architecture into TRELLIS's ss_flow and slat_flow training pipelines.

Key enhancements:
- DiagonalGaussianDistribution: Proper VAE posterior with clamped logvar
- SparseTransformerCrossBase: Cross-attention to original latent in decoders
- Occupancy-guided pruning: Cleaner geometry at each resolution level
- ConnectionHead: Direct edge/topology prediction for mesh generation

Usage:
    # === 阶段一：VAE 优化 ===
    from lato_integration import (
        EnhancedSLatEncoder,
        EnhancedSLatGaussianDecoder,
        EnhancedSLatMeshDecoder,
        EnhancedTrellisTextTo3DPipeline,
    )

    # === 阶段二：Flow/DiT 优化 ===
    from lato_integration.flow import EnhancedSSFlowModel, EnhancedSLatFlowModel
    from lato_integration.flow.trainers import (
        EnhancedSSFlowTrainer,          # ss_flow: 辅助解码损失
        EnhancedSLatFlowTrainer,        # slat_flow: 辅助解码 + latent一致性
    )

    # For training:
    from lato_integration.trainers import EnhancedSLatVaeGaussianTrainer

    # For inference:
    pipeline = EnhancedTrellisTextTo3DPipeline.from_pretrained(path)
    results = pipeline.run("a wooden chair with armrests")
"""

# Foundation utilities
from .utils import DiagonalGaussianDistribution
from .base import SparseTransformerCrossBase
from .vertex_encoder import ConnectionHead

# Enhanced encoders
from .sparse_structure_vae import (
    EnhancedSparseStructureEncoder,
    EnhancedSparseStructureDecoder,
)
from .encoder import EnhancedSLatEncoder

# Enhanced decoders
from .decoder_gs import (
    EnhancedSLatGaussianDecoder,
    EnhancedElasticSLatGaussianDecoder,
)
from .decoder_rf import (
    EnhancedSLatRadianceFieldDecoder,
    EnhancedElasticSLatRadianceFieldDecoder,
)
from .decoder_mesh import (
    EnhancedSLatMeshDecoder,
    EnhancedElasticSLatMeshDecoder,
    EnhancedSparseSubdivideBlock3d,
    SparsePredictionHead,
)

# Enhanced pipeline
from .pipeline import EnhancedTrellisTextTo3DPipeline

__all__ = [
    # Foundation
    "DiagonalGaussianDistribution",
    "SparseTransformerCrossBase",
    "ConnectionHead",
    # Encoders
    "EnhancedSparseStructureEncoder",
    "EnhancedSparseStructureDecoder",
    "EnhancedSLatEncoder",
    # Decoders - Gaussian
    "EnhancedSLatGaussianDecoder",
    "EnhancedElasticSLatGaussianDecoder",
    # Decoders - Radiance Field
    "EnhancedSLatRadianceFieldDecoder",
    "EnhancedElasticSLatRadianceFieldDecoder",
    # Decoders - Mesh
    "EnhancedSLatMeshDecoder",
    "EnhancedElasticSLatMeshDecoder",
    "EnhancedSparseSubdivideBlock3d",
    "SparsePredictionHead",
    # Pipeline
    "EnhancedTrellisTextTo3DPipeline",
]
