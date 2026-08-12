"""
LatoStructureHead — 3D CNN 上采样头（PixelShuffle 可学习上采样）

将 SS Flow 的 dense 16³ 特征转为 res128 的 occupancy logits，
替代 TRELLIS SparseStructureDecoder + coords×2 hack。

v15: nearest-neighbor → 3D PixelShuffle 可学习上采样
    原: nn.Upsample(nearest) 硬复制，1 voxel → 512 相同 voxel，边界模糊
    新: Conv3d(in, out*8) → pixel_shuffle_3d(2)，子体素特征可学习分配

架构：3 阶段 2× 上采样
    16³ → [PixelShuffleUpsample] → 32³ → [PixelShuffleUpsample] → 64³ → [PixelShuffleUpsample] → 128³ → occupancy

用法:
    head = LatoStructureHead(in_channels=8, base_channels=256)
    occ_logits = head(ss_flow_output)  # [B, 1, 128, 128, 128]
    coords = coords_from_occupancy(occ_logits)  # [N, 4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from trellis.modules.spatial import pixel_shuffle_3d


class ResBlock3d(nn.Module):
    """轻量 3D ResBlock，用于上采样链中。"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return x + h


class UpsampleBlock3d(nn.Module):
    """
    单级可学习上采样块：3D PixelShuffle + ResBlock(s)。

    v15: nearest-neighbor → PixelShuffle
      - 旧: nn.Upsample(nearest) 硬复制 → Conv smooth（子体素特征完全相同）
      - 新: Conv(in, out*8) → pixel_shuffle_3d(2) (子体素特征由卷积学习分配)
      - 与 TRELLIS SparseStructureDecoder.UpsampleBlock3d 对齐

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数（PixelShuffle 后）。
        num_res_blocks: ResBlock 数量（默认 1，保持轻量）。
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 1):
        super().__init__()
        # Conv 产出 8× out_channels → pixel_shuffle scale=2 后还原为 out_channels
        self.conv = nn.Conv3d(in_channels, out_channels * 8, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.res_blocks = nn.ModuleList([
            ResBlock3d(out_channels) for _ in range(num_res_blocks)
        ])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = pixel_shuffle_3d(h, 2)  # channels×8 → spatial×2
        h = self.act(self.norm(h))
        for block in self.res_blocks:
            h = block(h)
        return h


class LatoStructureHead(nn.Module):
    """
    将 SS Flow 的 dense 16³ 特征上采样到 res128 的 occupancy logits。
    v15: 使用 3D PixelShuffle 可学习上采样（替代 nearest-neighbor 硬复制）。

    替代 TRELLIS 的 SparseStructureDecoder + coords×2 hack：
      - 旧: SS Flow(16³×8) → SS Decoder(TRELLIS 冻结) → occ@64³ → coords ×2 → 128³
      - 新: SS Flow(16³×8) → LatoStructureHead → occ@128³ → coords (直接!)

    参数量：~1-2M（比 SS Flow 的 ~145M 小得多），与 SS Flow 联合训练。

    Args:
        in_channels: SS Flow 输出通道数（通常 = out_channels = 8）。
        base_channels: 基础通道数，每级上采样后递减。
        num_res_blocks: 每级 ResBlock 数量。
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 256,
        num_res_blocks: int = 1,
    ):
        super().__init__()
        # 三级 2× PixelShuffle 上采样：16 → 32 → 64 → 128
        c0 = base_channels
        c1 = base_channels // 2   # 128
        c2 = base_channels // 4   # 64

        self.stage1 = UpsampleBlock3d(in_channels, c0, num_res_blocks)   # 16³ → 32³
        self.stage2 = UpsampleBlock3d(c0, c1, num_res_blocks)            # 32³ → 64³
        self.stage3 = UpsampleBlock3d(c1, c2, num_res_blocks)            # 64³ → 128³

        # 最终 1×1 卷积 → 单通道 occupancy logit
        self.out_conv = nn.Conv3d(c2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, 16, 16, 16] — SS Flow 输出。

        Returns:
            occupancy_logits: [B, 1, 128, 128, 128]
        """
        h = self.stage1(x)    # 16³ → 32³
        h = self.stage2(h)    # 32³ → 64³
        h = self.stage3(h)    # 64³ → 128³
        return self.out_conv(h)  # → [B, 1, 128, 128, 128]

    def convert_to_fp16(self) -> None:
        """转换为 fp16（PixelShuffle 使用可学习参数，需真正转换）。"""
        self.half()

    def convert_to_fp32(self) -> None:
        """转换为 fp32。"""
        self.float()


def coords_from_occupancy(
    logits: torch.Tensor,
    threshold: float = 2.0,
) -> torch.Tensor:
    """
    从 occupancy logits 提取稀疏坐标（INT 类型，兼容 TRELLIS pipeline）。

    Args:
        logits: [B, 1, H, W, D] occupancy logits。
        threshold: logit > threshold 视为 occupied。

    Returns:
        coords: [N, 4] tensor，列顺序 [batch_idx, x, y, z]（int）。
    """
    # argwhere 返回 [N, 5] 列: [B, C, X, Y, Z]，去掉 channel 列 → [N, 4]: [B, X, Y, Z]
    coords = torch.argwhere(logits > threshold)[:, [0, 2, 3, 4]]
    # argwhere 输出可能是 long，转为 int 与 TRELLIS 保持一致
    return coords.int()
