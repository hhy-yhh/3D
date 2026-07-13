"""
================================================================================
encode_lato_latent.py — 使用 LATO VoxelVAE 提取 16-dim latent
================================================================================

用 LATO 预训练 VoxelVAE + VoxelFeatureEncoder 对 3D 模型提取 latent，
生成训练 TRELLIS SLat Flow 所需的 .npz 数据。

输出格式兼容 TRELLIS SLat dataset ({root}/latents/{model}/{sha256}.npz)。

用法:
    python encode_lato_latent.py \
        --lato_ckpt D:/code/LATO/ckpts/your_checkpoint.pt \
        --lato_config D:/code/LATO/configs/infer_vae_512.yaml \
        --data_dir D:/code/3D/database \
        --output_dir D:/code/3D/database/lato_latents \
        --resolution 128

依赖:
    - LATO 代码在 PYTHONPATH 中 (D:/code/LATO)
    - open3d, pyyaml, pandas, tqdm
================================================================================
"""

import os
import sys
import json
import argparse
import traceback

import numpy as np
import torch
import yaml
import pandas as pd
from tqdm import tqdm
import open3d as o3d

# ── flash_attn mock: encode 流程不用 flash_attn，但 LATO import 链会加载它 ──
#   如果环境中 flash_attn 和 torch 版本不匹配（如 torch 1.12 + flash_attn 2.x），
#   import flash_attn 会因 libc10_cuda.so 找不到而崩溃。这里提前 mock 绕过。
try:
    import flash_attn  # noqa: F401  (check if it works)
except ImportError:
    # flash_attn 未安装 → 需要 mock 整个包
    import types
    _mock_flash = types.ModuleType("flash_attn")
    sys.modules["flash_attn"] = _mock_flash
    sys.modules["flash_attn.flash_attn_interface"] = types.ModuleType("flash_attn.flash_attn_interface")
    sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")
except Exception:
    # flash_attn 安装了但加载失败（版本不匹配）→ mock 替换已加载的模块
    import types
    if "flash_attn" not in sys.modules:
        sys.modules["flash_attn"] = types.ModuleType("flash_attn")
    if "flash_attn.flash_attn_interface" not in sys.modules:
        sys.modules["flash_attn.flash_attn_interface"] = types.ModuleType("flash_attn.flash_attn_interface")
    if "flash_attn_2_cuda" not in sys.modules:
        sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")

# ── 确保 LATO 在 Python path 中 ──
_LATO_ROOT = os.environ.get("LATO_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..", "LATO"))
_LATO_ROOT = os.path.abspath(_LATO_ROOT)
if _LATO_ROOT not in sys.path:
    sys.path.insert(0, _LATO_ROOT)

from lato.modules.sparse import SparseTensor as LATOSparseTensor
from lato.models.lato_vae.lato_vae import VoxelVAE
from vertex_encoder import VoxelFeatureEncoder_active_pointnet
from utils import load_pretrained_woself


# ============================================================================
# 工具函数
# ============================================================================

def normalize_mesh(mesh: o3d.geometry.TriangleMesh,
                   bbox_min: float = -0.5,
                   bbox_max: float = 0.5) -> o3d.geometry.TriangleMesh:
    """将 mesh 归一化到 [-0.5, 0.5]³ 的包围盒中。"""
    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        return mesh
    aabb_min = vertices.min(axis=0)
    aabb_max = vertices.max(axis=0)
    center = (aabb_min + aabb_max) / 2
    scale = (aabb_max - aabb_min).max()
    if scale < 1e-8:
        scale = 1.0
    vertices = (vertices - center) / scale
    vertices = np.clip(vertices, bbox_min + 1e-6, bbox_max - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def get_active_voxels(mesh: o3d.geometry.TriangleMesh,
                      resolution: int = 128) -> torch.Tensor:
    """
    体素化 mesh 获取 active voxel 坐标 (整数索引)。

    Returns:
        [N, 3] int tensor of voxel grid indices in [0, resolution-1].
    """
    voxel_size = 1.0 / resolution
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=voxel_size,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    if len(voxels) == 0:
        return torch.empty(0, 3, dtype=torch.int32)
    return torch.from_numpy(voxels).int()


def sample_point_cloud(mesh: o3d.geometry.TriangleMesh,
                       n: int = 819200) -> torch.Tensor:
    """从 mesh 表面均匀采样点云。"""
    if n > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=n)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
    points = np.asarray(pcd.points).astype(np.float32)
    return torch.from_numpy(points)


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="使用 LATO VoxelVAE 提取 16-dim latent 作为 SLat Flow 训练数据"
    )
    parser.add_argument("--lato_ckpt", type=str, required=True,
                        help="LATO checkpoint 路径 (.pt)")
    parser.add_argument("--lato_config", type=str, required=True,
                        help="LATO VAE 配置文件 (如 infer_vae_512.yaml)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="包含 metadata.csv 和 meshes/ 的数据目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 latent 数据的目录")
    parser.add_argument("--resolution", type=int, default=128,
                        help="体素化分辨率 (默认 128)")
    parser.add_argument("--num_points", type=int, default=819200,
                        help="点云采样点数 (默认 819200)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mesh_subdir", type=str, default="meshes",
                        help="mesh 文件所在子目录")
    parser.add_argument("--mesh_ext", type=str, default=".obj",
                        help="mesh 文件扩展名")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印不实际处理")
    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ── 1. 加载 LATO 配置 ──
    with open(opt.lato_config, "r") as f:
        lato_cfg = yaml.safe_load(f)
    model_cfg = lato_cfg["model"]

    print(f"[INFO] LATO latent_dim={model_cfg['latent_dim']}, "
          f"encoder_blocks={len(model_cfg['encoder_blocks'])}, "
          f"decoder_blocks={len(model_cfg['decoder_blocks_vtx'])}")

    # ── 2. 构建 LATO 模型 ──
    vae = VoxelVAE(
        in_channels=model_cfg.get("in_channels", 1024),
        latent_dim=model_cfg["latent_dim"],
        encoder_blocks=model_cfg["encoder_blocks"],
        decoder_blocks_vtx=model_cfg["decoder_blocks_vtx"],
        attn_mode="swin",
        window_size=8,
        pe_mode="ape",
        using_subdivide=True,
        using_attn=model_cfg.get("using_attn", False),
    ).to(device)

    voxel_encoder = VoxelFeatureEncoder_active_pointnet(
        in_channels=3,
        hidden_dim=128,
        out_channels=model_cfg.get("in_channels", 1024),
        scatter_type="mean",
        n_blocks=3,
        resolution=opt.resolution,
    ).to(device)

    # ── 3. 加载预训练权重 ──
    result = load_pretrained_woself(
        opt.lato_ckpt,
        vae=vae,
        voxel_encoder=voxel_encoder,
    )
    print(f"[INFO] 加载 checkpoint: epoch={result.get('epoch', '?')}, "
          f"best_loss={result.get('best_loss', float('inf')):.4f}")

    vae.eval()
    voxel_encoder.eval()

    # ── 4. 读取 metadata ──
    metadata_path = os.path.join(opt.data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv 未找到: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    print(f"[INFO] metadata 共 {len(metadata)} 条记录")

    # 确定 key 列
    key_col = "sha256" if "sha256" in metadata.columns else metadata.columns[0]
    print(f"[INFO] 使用 key 列: {key_col}")

    # ── 5. 输出目录 ──
    latent_model_name = f"lato_vae_{model_cfg['latent_dim']}dim_{opt.resolution}"
    latent_output_dir = os.path.join(opt.output_dir, "latents", latent_model_name)
    os.makedirs(latent_output_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {latent_output_dir}")

    # ── 6. 逐个处理模型 ──
    latent_col = f"latent_{latent_model_name}"
    metadata[latent_col] = False

    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, row in tqdm(list(metadata.iterrows()), desc="Extracting latents"):
        key = row[key_col]
        mesh_path = os.path.join(opt.data_dir, opt.mesh_subdir, f"{key}{opt.mesh_ext}")

        if not os.path.exists(mesh_path):
            # 尝试不带扩展名
            alt_path = os.path.join(opt.data_dir, opt.mesh_subdir, str(key))
            if os.path.exists(alt_path):
                mesh_path = alt_path
            else:
                tqdm.write(f"[SKIP] mesh 未找到: {mesh_path}")
                skip_count += 1
                continue

        try:
            if opt.dry_run:
                print(f"  [DRY] 将处理: {key}")
                continue

            # 6a. 加载并归一化 mesh
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if len(mesh.vertices) == 0:
                tqdm.write(f"[SKIP] 空 mesh: {key}")
                skip_count += 1
                continue
            mesh = normalize_mesh(mesh)

            # 6b. 体素化
            active_coords = get_active_voxels(mesh, resolution=opt.resolution)
            if active_coords.numel() == 0:
                tqdm.write(f"[SKIP] 无 active voxels: {key}")
                skip_count += 1
                continue
            active_coords = active_coords.to(device)

            # 6c. 点云 → voxel features
            point_cloud = sample_point_cloud(mesh, n=opt.num_points).to(device)
            active_feats = voxel_encoder(
                point_cloud, active_coords, res=opt.resolution
            )

            # 6d. LATO VAE encode → 16-dim latent
            coords_4d = torch.cat([
                torch.zeros(len(active_coords), 1, dtype=torch.int32, device=device),
                active_coords,
            ], dim=1)
            sparse_input = LATOSparseTensor(feats=active_feats, coords=coords_4d)

            with torch.no_grad():
                latent, _ = vae.encode(sparse_input, sample_posterior=False)

            # 6e. 保存 .npz
            latent_coords = latent.coords.cpu().numpy().astype(np.int32)
            latent_feats = latent.feats.cpu().numpy().astype(np.float16)

            np.savez_compressed(
                os.path.join(latent_output_dir, f"{key}.npz"),
                coords=latent_coords,
                feats=latent_feats,
            )

            metadata.at[idx, latent_col] = True
            if "num_voxels" in metadata.columns:
                metadata.at[idx, "num_voxels"] = len(latent_coords)
            success_count += 1

        except Exception as e:
            tqdm.write(f"[ERROR] {key}: {e}")
            traceback.print_exc()
            error_count += 1
            continue

    # ── 7. 保存更新后的 metadata ──
    output_metadata_path = os.path.join(opt.output_dir, "metadata.csv")
    metadata.to_csv(output_metadata_path, index=False)
    print(f"\n[INFO] 完成! metadata 已保存到: {output_metadata_path}")

    # ── 8. 报告统计 ──
    print(f"\n{'='*60}")
    print(f"  处理结果")
    print(f"{'='*60}")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {error_count}")
    print(f"  Latent 目录: {latent_output_dir}")
    print(f"  Latent 模型名: {latent_model_name}")
    print(f"\n  下一步: 运行 stat_latent.py 计算 normalization 统计量")
    print(f"    python dataset_toolkits/stat_latent.py \\")
    print(f"        --output_dir {opt.output_dir} \\")
    print(f"        --model {latent_model_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
