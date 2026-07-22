"""
[DEPRECATED in v3] — 由 LATO VoxelVAE + LatoStructureHead 替代。

v2 中此文件包含:
  - EnhancedSparseStructureEncoder → 替代: LATO VoxelVAE.encode()
  - EnhancedSparseStructureDecoder → 替代: LatoStructureHead (structure_head.py)

v3 中此文件仅保留 ResBlock3d 工具类和 re-export 以保持向后兼容。
"""

import torch.nn as nn

from trellis.models.sparse_structure_vae import ResBlock3d, norm_layer

# Re-export LatoStructureHead from the new location
from .structure_head import LatoStructureHead, coords_from_occupancy

__all__ = [
    "LatoStructureHead",
    "coords_from_occupancy",
    "ResBlock3d",
    "norm_layer",
]
