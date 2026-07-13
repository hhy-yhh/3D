"""
Enhanced trainers that use LATO-style VAE posterior and cross-attention decoders.
"""

from .sparse_structure_vae import EnhancedSparseStructureVaeTrainer
from .slat_vae_gaussian import EnhancedSLatVaeGaussianTrainer
from .slat_vae_rf_dec import EnhancedSLatVaeRadianceFieldDecoderTrainer
from .slat_vae_mesh_dec import EnhancedSLatVaeMeshDecoderTrainer
