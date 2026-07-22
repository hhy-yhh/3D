"""
================================================================================
LATO-Enhanced Flow Trainers (阶段二训练优化)
================================================================================

区分标识:
  - ss_flow_trainer  → SS Flow 辅助解码训练器 (步骤5)
  - slat_flow_trainer → SLat Flow 辅助解码 + latent 一致性训练器 (步骤6)
================================================================================
"""
from .ss_flow_trainer import (
    # v3 names
    LatoSSFlowTrainer,
    LatoSSFlowCFGTrainer,
    TextConditionedLatoSSFlowCFGTrainer,
    ImageConditionedLatoSSFlowCFGTrainer,
    # v2 backward compat aliases
    EnhancedSSFlowTrainer,
    EnhancedSSFlowCFGTrainer,
    TextConditionedEnhancedSSFlowCFGTrainer,
    ImageConditionedEnhancedSSFlowCFGTrainer,
)
from .slat_flow_trainer import (
    EnhancedSLatFlowTrainer,
    EnhancedSLatFlowCFGTrainer,
    TextConditionedEnhancedSLatFlowCFGTrainer,
    ImageConditionedEnhancedSLatFlowCFGTrainer,
)
