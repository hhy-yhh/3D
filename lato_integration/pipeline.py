"""
Enhanced Text-to-3D Pipeline — v3: 全 LATO Encoder/Decoder + TRELLIS Flow

v3 变化:
  - 移除 Gaussian/RadianceField decoder 路径（仅 LATO Mesh 输出）
  - 新增 sample_sparse_structure_lato() — 使用 LatoStructureHead
  - decode_slat() 简化：优先 LATO decode，fallback 到 TRELLIS mesh decoder
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from trellis.pipelines.trellis_text_to_3d import (
    TrellisTextTo3DPipeline as _TrellisTextTo3DPipeline,
)
from trellis.modules import sparse as sp


class EnhancedTrellisTextTo3DPipeline(_TrellisTextTo3DPipeline):
    """
    Enhanced Text-to-3D Pipeline — v3。

    核心变化（v3）:
      - sample_sparse_structure_lato(): SS Flow + LatoStructureHead → coords@128³
      - decode_slat(): 默认使用 LATO VoxelVAE decode
      - 不再有 GS/RF decode 路径
    """

    @staticmethod
    def from_pretrained(path: str) -> "EnhancedTrellisTextTo3DPipeline":
        """加载预训练模型并包装为增强管线。"""
        pipeline = _TrellisTextTo3DPipeline.from_pretrained(path)
        enhanced = EnhancedTrellisTextTo3DPipeline()
        enhanced.__dict__ = pipeline.__dict__
        return enhanced

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh'],
    ) -> dict:
        """
        Decode structured latent — v3: 优先 LATO VoxelVAE。

        如果有 lato_vae → 走 LATO decode 路径。
        否则 fallback 到 TRELLIS mesh decoder。
        """
        # LATO decode path (preferred in v3)
        if 'lato_vae' in self.models and 'mesh' in formats:
            return self.decode_slat_lato(slat)

        # Fallback: TRELLIS mesh decoder
        ret = {}
        if 'mesh' in formats and 'slat_decoder_mesh' in self.models:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        return ret

    def sample_sparse_structure_lato(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        v3: SS Flow + LatoStructureHead → coords@128³。

        替代 sample_sparse_structure()（后者使用 TRELLIS SS Decoder @ 64³）。
        输出 coords 已经是 res128，无需 ×2。

        Returns:
            coords: [N, 4] tensor [batch_idx, x, y, z]（int）。
        """
        from .structure_head import coords_from_occupancy

        flow_model = self.models['sparse_structure_flow_model']
        structure_head = self.models['lato_structure_head']
        reso = flow_model.resolution  # 16

        noise = torch.randn(
            num_samples, flow_model.in_channels, reso, reso, reso
        ).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples

        # LatoStructureHead: 16³ dense → 128³ occupancy → coords
        occ_logits = structure_head(z_s)
        coords = coords_from_occupancy(occ_logits)
        return coords

    @torch.no_grad()
    def run_lato(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
    ) -> dict:
        """
        v3 推理入口 — 使用 LatoStructureHead，无需 coords×2。

        与父类 run() 的区别：
          - sample_sparse_structure_lato() 替代 sample_sparse_structure()
          - 无 coords ×2
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)

        # 1. SS Flow + LatoStructureHead → coords@128³（直接！）
        coords = self.sample_sparse_structure_lato(
            cond, num_samples, sparse_structure_sampler_params
        )

        # 2. SLat Flow → LATO latent
        slat = self.sample_slat(cond, coords, slat_sampler_params)

        # 3. Decode
        return self.decode_slat(slat, formats)
