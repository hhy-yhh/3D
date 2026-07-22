"""
Sparse 工具类 — 从 LATO 移植。

v3 保留:
  - SparsePredictionHead: 用于 occupancy 预测的小 MLP
  - EnhancedSparseSubdivideBlock3d: 带 pruning 的上采样块

v3 废弃（由 LATO VoxelVAE.decode() 替代）:
  - EnhancedSLatMeshDecoder
  - EnhancedElasticSLatMeshDecoder
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from trellis.models.structured_latent_vae.decoder_mesh import (
    SparseSubdivideBlock3d as _SparseSubdivideBlock3d,
)
from trellis.modules import sparse as sp
from trellis.modules.sparse.linear import SparseLinear
from trellis.modules.sparse.nonlinearity import SparseGELU


class SparsePredictionHead(nn.Module):
    """
    用于 occupancy 预测的小 MLP（稀疏张量输入/输出）。

    Adapted from LATO: lato/models/lato_vae/lato_vae.py

    Args:
        channels: 输入特征通道数。
        out_channels: 输出通道数（默认 1 = occupancy）。
        mlp_ratio: 隐藏层倍数。
    """

    def __init__(self, channels: int, out_channels: int = 1, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            SparseLinear(channels, hidden_channels),
            SparseGELU(approximate="tanh"),
            SparseLinear(hidden_channels, out_channels),
        )

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return self.mlp(x)


class EnhancedSparseSubdivideBlock3d(_SparseSubdivideBlock3d):
    """
    带 occupancy pruning 的 3D 细分块。

    扩展 TRELLIS 的 SparseSubdivideBlock3d，增加 LATO 风格的
    occupancy 预测用于在每级分辨率过滤非表面体素。

    Args:
        (与 SparseSubdivideBlock3d 相同)
        use_pruning: 是否启用 occupancy pruning。
    """

    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        use_pruning: bool = True,
    ):
        super().__init__(
            channels=channels,
            resolution=resolution,
            out_channels=out_channels,
            num_groups=num_groups,
        )
        self.use_pruning = use_pruning
        if use_pruning:
            self.pruning_head = SparsePredictionHead(
                self.out_channels, out_channels=1
            )

    def forward(
        self,
        x: sp.SparseTensor,
        pruning: bool = False,
        training: bool = True,
        threshold: float = 0.5,
        force_no_prune: bool = False,
    ) -> Tuple[sp.SparseTensor, Optional[sp.SparseTensor]]:
        """
        Args:
            x: 输入稀疏张量。
            pruning: 是否应用 pruning。
            training: 是否训练模式。
            threshold: 推理时的 occupancy 阈值。
            force_no_prune: 强制不剪枝（用于最终层可视化）。

        Returns:
            (输出稀疏张量, 可选的 occupancy 概率)
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)

        if not pruning or not self.use_pruning:
            return h, None

        occ_prob = self.pruning_head(h)

        if not training and not force_no_prune:
            scores = torch.sigmoid(occ_prob.feats).squeeze(-1)
            if scores.numel() % 8 == 0:
                grouped_scores = scores.view(-1, 8)
                grouped_mask = grouped_scores >= threshold
                empty_parent_mask = grouped_mask.sum(dim=1) == 0
                if empty_parent_mask.any():
                    _, top_indices = torch.topk(
                        grouped_scores[empty_parent_mask], k=1, dim=1
                    )
                    parent_rows = torch.nonzero(empty_parent_mask, as_tuple=True)[0].unsqueeze(1)
                    grouped_mask[parent_rows, top_indices] = True
                occ_mask = grouped_mask.view(-1)
            else:
                occ_mask = scores >= threshold

            h = sp.SparseTensor(
                feats=h.feats[occ_mask],
                coords=h.coords[occ_mask],
            )

        return h, occ_prob
