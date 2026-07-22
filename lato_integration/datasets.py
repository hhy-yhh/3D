"""
================================================================================
LATO 自定义数据集 — 提供 ss_occupancy_128 用于 LatoStructureHead 训练
================================================================================

扩展 TRELLIS 标准数据集，在返回 x_0 (SS latent) 的同时返回对应的
occupancy@128³ GT（由 LATO VoxelVAE coords 栅格化生成）。

用法:
    from lato_integration.datasets import LatoSSStructureLatent, TextConditionedLatoSSStructureLatent

    在 JSON config 中使用:
    "dataset": {
        "name": "TextConditionedLatoSSStructureLatent",
        "args": {
            "latent_model": "ss_enc_conv3d_16l8_fp16",
            "occupancy_dir": "ss_occupancy_128",
            ...
        }
    }
================================================================================
"""
import json
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image

from trellis.datasets.components import TextConditionedMixin, ImageConditionedMixin
from trellis.datasets.sparse_structure_latent import SparseStructureLatent


class LatoSSStructureLatent(SparseStructureLatent):
    """
    扩展 SparseStructureLatent，额外加载 occupancy@128³ 数据。

    额外参数:
        occupancy_dir: occupancy 数据子目录名（相对于 roots），
                       默认 "ss_occupancy_128"。
    """

    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        occupancy_dir: str = "ss_occupancy_128",
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        **kwargs,
    ):
        self.occupancy_dir = occupancy_dir
        super().__init__(
            roots=roots,
            latent_model=latent_model,
            min_aesthetic_score=min_aesthetic_score,
            normalization=normalization,
            **kwargs,
        )

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        # 加载 occupancy@128³ GT（来自 LATO VoxelVAE coords 栅格化）
        occ_path = os.path.join(root, self.occupancy_dir, f"{instance}.npz")
        if os.path.exists(occ_path):
            occ_data = np.load(occ_path)
            occ = torch.tensor(occ_data["occupancy"]).float()  # [1, 128, 128, 128]
            pack["ss_occupancy_128"] = occ
        else:
            # 数据不存在时设为 None，训练器会跳过辅助 loss
            pack["ss_occupancy_128"] = None

        return pack


class TextConditionedLatoSSStructureLatent(
    TextConditionedMixin, LatoSSStructureLatent
):
    """
    文本条件版本：在 LatoSSStructureLatent 的基础上加文本 captions。

    通过多重继承同时获得:
      - LatoSSStructureLatent: x_0 + ss_occupancy_128 加载
      - TextConditionedMixin: text cond + self.captions 管理

    MRO: TextConditionedMixin → LatoSSStructureLatent →
         SparseStructureLatent → SparseStructureLatentVisMixin →
         StandardDatasetBase → Dataset

    用法（JSON config）:
        "dataset": {
            "name": "TextConditionedLatoSSStructureLatent",
            "args": {
                "latent_model": "ss_enc_conv3d_16l8_fp16",
                "occupancy_dir": "ss_occupancy_128",
                "min_aesthetic_score": 4.5,
                "normalization": { ... }
            }
        }
    """

    def get_instance(self, root, instance):
        # LatoSSStructureLatent.get_instance: 加载 x_0 + ss_occupancy_128
        pack = LatoSSStructureLatent.get_instance(self, root, instance)
        # TextConditionedMixin.get_instance: 添加 cond (text captions)
        text = np.random.choice(self.captions[instance])
        pack["cond"] = text
        return pack


class ImageConditionedLatoSSStructureLatent(
    ImageConditionedMixin, LatoSSStructureLatent
):
    """
    图像条件版本：在 LatoSSStructureLatent 的基础上加图像条件。

    MRO: ImageConditionedMixin → LatoSSStructureLatent →
         SparseStructureLatent → SparseStructureLatentVisMixin →
         StandardDatasetBase → Dataset
    """

    def get_instance(self, root, instance):
        # LatoSSStructureLatent.get_instance: 加载 x_0 + ss_occupancy_128
        pack = LatoSSStructureLatent.get_instance(self, root, instance)
        # ImageConditionedMixin.get_instance: 添加 cond (image + extrinsics + intrinsics)
        text = np.random.choice(self.captions[instance]) if hasattr(self, 'captions') and instance in self.captions else None
        # Use ImageConditionedMixin's logic for image loading
        image_root = os.path.join(root, "renders_cond", instance)
        with open(os.path.join(image_root, "transforms.json")) as f:
            metadata = json.load(f)
        n_views = len(metadata["frames"])
        view = np.random.randint(n_views)
        metadata = metadata["frames"][view]

        image_path = os.path.join(image_root, metadata["file_path"])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]

        max_side = max(size) * 1.1
        crop_bbox = [
            center[0] - max_side / 2,
            center[1] - max_side / 2,
            center[0] + max_side / 2,
            center[1] + max_side / 2,
        ]
        image = image.crop(crop_bbox)
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        cond = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        pack["cond"] = cond
        pack["extrinsics"] = torch.tensor(metadata["extrinsics"])
        pack["intrinsics"] = torch.tensor(metadata["intrinsics"])

        return pack
