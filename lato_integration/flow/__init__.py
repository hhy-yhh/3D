"""
================================================================================
LATO-Enhanced Flow Models (阶段二优化)
================================================================================

本模块包含针对 TRELLIS ss_flow 和 slat_flow 训练的 LATO 风格优化。

区分标识:
  - 文件名前缀 "ss_"  → Sparse Structure Flow (步骤5)
  - 文件名前缀 "slat_" → Structured Latent Flow (步骤6)

快速选择:
  - 训练 ss_flow  → from lato_integration.flow import EnhancedSSFlowModel
  - 训练 slat_flow → from lato_integration.flow import EnhancedSLatFlowModel
================================================================================
"""
from .ss_flow import EnhancedSSFlowModel
from .slat_flow import EnhancedSLatFlowModel, EnhancedElasticSLatFlowModel

# LATOSLatFlowModel — 从 TRELLIS 模型库导入（需要 PYTHONPATH 包含 TRELLIS 根目录）
try:
    from trellis.models.lato_slat_flow import LATOSLatFlowModel
except ImportError:
    LATOSLatFlowModel = None