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

# === v3: 自定义数据集（提供 ss_occupancy_128）===
from . import datasets as datasets

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
    # v10: Dynamic model builder
    "build_flow_model_from_config",
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


# ========================================================================
# v10: 动态模型构建器 — 从 config JSON 解析模型类名，自动选择正确架构
# ========================================================================

def build_flow_model_from_config(config_path: str, device=None, use_fp16=None):
    """
    从训练 config JSON 构建 Flow 模型（SS 或 SLat），自动解析模型类名。

    与 run_train.py 的 MODEL_REPLACEMENTS 保持同步，确保推理时加载的模型
    架构与训练时完全一致（例如 v10 的 EnhancedSLatFlowModel vs v8 的 LATOSLatFlowModel）。

    Args:
        config_path: 训练 config JSON 路径。
        device: torch device（默认 cuda）。
        use_fp16: 是否使用 FP16（默认从 config 读取；若 config 未设置，默认 True）。

    Returns:
        model 实例（已 .to(device)）。
    """
    import json
    import torch
    from .flow.slat_flow import EnhancedSLatFlowModel, EnhancedElasticSLatFlowModel
    from .flow.ss_flow import EnhancedSSFlowModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_cfg = cfg["models"]["denoiser"]
    name = model_cfg["name"]
    args = dict(model_cfg["args"])

    # 如果调用方未指定 use_fp16，从 config 取值；config 未设置则默认 True
    if use_fp16 is None:
        use_fp16 = args.get("use_fp16", True)
    args["use_fp16"] = use_fp16

    # 模型类名 → 类 映射（与 run_train.py MODEL_REPLACEMENTS 同步）
    MODEL_CLASS_MAP = {
        "SparseStructureFlowModel": EnhancedSSFlowModel,
        "EnhancedSSFlowModel": EnhancedSSFlowModel,
        "SLatFlowModel": EnhancedSLatFlowModel,
        "ElasticSLatFlowModel": EnhancedElasticSLatFlowModel,
        "LATOSLatFlowModel": EnhancedSLatFlowModel,  # v8→v10 自动升级
        "EnhancedSLatFlowModel": EnhancedSLatFlowModel,
        "EnhancedElasticSLatFlowModel": EnhancedElasticSLatFlowModel,
    }

    if name not in MODEL_CLASS_MAP:
        raise ValueError(
            f"不支持的模型类型: '{name}'。"
            f"支持的类型: {list(MODEL_CLASS_MAP.keys())}"
        )

    cls = MODEL_CLASS_MAP[name]
    print(f"  [ModelBuilder] '{name}' → {cls.__name__}")

    # 过滤 args 到模型构造函数支持的参数
    import inspect
    sig_params = set(inspect.signature(cls.__init__).parameters.keys())
    filtered_args = {k: v for k, v in args.items() if k in sig_params}
    skipped = set(args.keys()) - sig_params
    if skipped:
        print(f"  [ModelBuilder] 跳过不支持的参数: {skipped}")

    model = cls(**filtered_args).to(device)
    return model

