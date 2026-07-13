"""
================================================================================
encode_lato_latent_v2.py — 基于 LATO 官方代码，批量提取 VoxelVAE latent
================================================================================

与 v1 的差异：
  - 模型架构严格匹配 LATO 官方 (in_channels=15, hidden_dim=256, n_blocks=5)
  - 使用 LATO 官方 load_quantized_mesh_original 预处理（含 normal + VDF）
  - bf16 autocast（与官方一致）
  - VAE 按需移 GPU（节省显存）
  - 不做 decode / connection_head（只 encode）

用法:
    python lato_integration/encode_lato_latent_v2.py \
        --lato_ckpt /path/to/checkpoints/128to512/vae/vae_128to512.pt \
        --lato_config /path/to/configs/infer_vae_512.yaml \
        --data_dir /path/to/database \
        --output_dir /path/to/output \
        --resolution 128
================================================================================
"""

import os
import sys
import argparse
import traceback

import numpy as np
import torch
import yaml
import pandas as pd
from tqdm import tqdm

# ── 确保 LATO 在 sys.path 最前面 ──
_LATO_ROOT = os.environ.get("LATO_ROOT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "LATO"))
_LATO_ROOT = os.path.abspath(_LATO_ROOT)
for _p in list(sys.path):
    if os.path.abspath(_p) == _LATO_ROOT:
        sys.path.remove(_p)
sys.path.insert(0, _LATO_ROOT)

from lato.datasets.vertex_head import load_quantized_mesh_original
from lato.models.lato_vae.lato_vae import VoxelVAE
from lato.modules.sparse.basic import SparseTensor
from utils import load_pretrained_woself
from vertex_encoder import VoxelFeatureEncoder_active_pointnet


# ============================================================================
# 工具函数
# ============================================================================

MESH_EXTENSIONS = ('.obj', '.ply', '.stl', '.glb')


def find_mesh_file(data_dir, key):
    base_path = os.path.join(data_dir, "meshes", key)
    for ext in MESH_EXTENSIONS:
        path = base_path + ext
        if os.path.exists(path):
            return path
    if os.path.exists(base_path):
        return base_path
    return None


def make_models(config, checkpoint_path, device):
    """构建与 LATO 官方完全一致的模型架构。"""
    model_cfg = config["model"]

    voxel_encoder = VoxelFeatureEncoder_active_pointnet(
        in_channels=15,         # pos(3) + normal(3) + VDF(9)
        hidden_dim=256,
        out_channels=model_cfg.get("in_channels", 1024),
        scatter_type="mean",
        n_blocks=5,
        resolution=128,
    ).to(device)

    vae = VoxelVAE(
        in_channels=model_cfg["in_channels"],
        latent_dim=model_cfg["latent_dim"],
        encoder_blocks=model_cfg["encoder_blocks"],
        decoder_blocks_vtx=model_cfg["decoder_blocks_vtx"],
        num_heads=8,
        num_head_channels=64,
        mlp_ratio=4.0,
        attn_mode="swin",
        window_size=8,
        pe_mode="ape",
        use_fp16=False,
        use_checkpoint=False,
        qk_rms_norm=False,
        using_subdivide=True,
        using_attn=model_cfg.get("using_attn", False),
        attn_first=model_cfg.get("attn_first", True),
        pred_direction=model_cfg.get("pred_direction", False),
    )  # 先 CPU，encode 时临时移 GPU

    load_pretrained_woself(
        checkpoint_path=checkpoint_path,
        voxel_encoder=voxel_encoder,
        vae=vae,
    )

    # 确保 VAE 在 CPU
    vae.cpu()
    vae.eval()
    voxel_encoder.eval()

    return vae, voxel_encoder


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LATO latent 批量提取 (官方架构)")
    parser.add_argument("--lato_ckpt", type=str, required=True)
    parser.add_argument("--lato_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=65536,
                        help="点云采样数（LATO 默认 819200，可降低节省显存）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    is_cuda = (device.type == "cuda")
    print(f"[INFO] 设备: {device}")

    # ── 1. 加载配置 & 模型 ──
    with open(opt.lato_config, "r") as f:
        config = yaml.safe_load(f)
    latent_dim = config["model"]["latent_dim"]

    print(f"[INFO] 构建模型 (官方架构: in=15, hidden=256, blocks=5, latent={latent_dim}) ...")
    vae, voxel_encoder = make_models(config, opt.lato_ckpt, device)

    print(f"[INFO] VAE 设备: {next(vae.parameters()).device}, "
          f"voxel_encoder 设备: {next(voxel_encoder.parameters()).device}, "
          f"GPU 已用: {torch.cuda.memory_allocated()/1024**3:.1f} GiB")

    # ── 2. 读取 metadata ──
    metadata_path = os.path.join(opt.data_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    key_col = "sha256" if "sha256" in metadata.columns else metadata.columns[0]
    print(f"[INFO] metadata: {len(metadata)} 条, key={key_col}")

    # ── 3. 输出目录 ──
    lm_name = f"lato_vae_{latent_dim}dim_{opt.resolution}"
    latent_dir = os.path.join(opt.output_dir, "latents", lm_name)
    os.makedirs(latent_dir, exist_ok=True)

    latent_col = f"latent_{lm_name}"
    metadata[latent_col] = False

    success = skip = error = 0

    # ── 4. 逐模型处理 ──
    for idx, row in tqdm(list(metadata.iterrows()), desc="Encoding"):
        key = row[key_col]

        mesh_path = find_mesh_file(opt.data_dir, key)
        if mesh_path is None:
            tqdm.write(f"[SKIP] 未找到 mesh: {key}")
            skip += 1
            continue

        try:
            if opt.dry_run:
                continue

            # 4a. LATO 官方预处理 → 15 通道点云特征 (pos + normal + VDF)
            voxels, point_features = load_quantized_mesh_original(
                mesh_path,
                volume_resolution=opt.resolution,
                use_normals=True,
                pc_sample_number=opt.num_points,
            )
            voxels = voxels.to(device)
            point_features = point_features.to(device)  # [P, 15]

            if voxels.numel() == 0:
                tqdm.write(f"[SKIP] 无 active voxels: {key}")
                skip += 1
                continue

            # 4b. 构造 LATO 格式 coords [N, 4] (batch=0 + xyz)
            coords_4d = torch.cat([
                torch.zeros(len(voxels), 1, device=device),
                voxels,
            ], dim=1).int()

            # 4c. voxel_encoder: point_cloud [B, P, 15] → voxel feats [N, 1024]
            pts_batched = point_features.unsqueeze(0)  # [P,15] → [1, P, 15]
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda
            ):
                active_feats = voxel_encoder(
                    p=pts_batched,
                    sparse_coords=coords_4d,
                    res=opt.resolution,
                    bbox_size=(-0.5, 0.5),
                )

            del pts_batched, point_features

            # 4d. VAE encode: 临时移 GPU
            vae.to(device)
            sparse_in = SparseTensor(feats=active_feats, coords=coords_4d)

            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda
            ):
                latent, _ = vae.encode(sparse_in, sample_posterior=False)

            # VAE 移回 CPU
            vae.cpu()

            # 4e. 保存
            latent_feats = latent.feats.float().cpu().numpy().astype(np.float16)
            latent_coords = latent.coords.cpu().numpy().astype(np.int32)
            np.savez_compressed(
                os.path.join(latent_dir, f"{key}.npz"),
                coords=latent_coords,
                feats=latent_feats,
            )

            # 清理 GPU
            del active_feats, sparse_in, latent, latent_feats, latent_coords, voxels, coords_4d
            torch.cuda.empty_cache()

            metadata.at[idx, latent_col] = True
            if "num_voxels" in metadata.columns:
                metadata.at[idx, "num_voxels"] = len(latent_coords)
            success += 1

        except Exception as e:
            tqdm.write(f"[ERROR] {key}: {e}")
            traceback.print_exc()
            error += 1
            if is_cuda:
                if next(vae.parameters()).device.type != 'cpu':
                    vae.cpu()
                torch.cuda.empty_cache()

        # 每 10 个保存一次
        if (success + error) % 10 == 0 and (success + error) > 0:
            metadata.to_csv(os.path.join(opt.output_dir, "metadata.csv"), index=False)

    # ── 5. 最终保存 ──
    metadata.to_csv(os.path.join(opt.output_dir, "metadata.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"  完成: {success} 成功, {skip} 跳过, {error} 失败")
    print(f"  Latent 目录: {latent_dir}")
    print(f"  模型名: {lm_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
