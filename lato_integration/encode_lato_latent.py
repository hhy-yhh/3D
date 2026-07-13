"""
================================================================================
encode_lato_latent.py — LATO VoxelVAE latent 提取（mock 方案）
================================================================================

Mock flash_attn 和 torch_scatter（用 torch≥1.11 自带的 scatter_reduce 替代），
然后直接使用 LATO 原版 VoxelVAE + VoxelFeatureEncoder 的预训练权重。

用法:
    python lato_integration/encode_lato_latent.py \
        --lato_ckpt /path/to/ckpt.pt \
        --lato_config /path/to/infer_vae_512.yaml \
        --data_dir /path/to/database \
        --output_dir /path/to/output \
        --resolution 128
================================================================================
"""

import os
import sys
import types
import argparse
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import pandas as pd
from tqdm import tqdm

# ========================================================================
# 步骤 0: 在 import LATO 之前检查 / mock 不兼容的依赖
# ========================================================================

# ── 0a. flash_attn: 先试真 import，失败则 mock ──
_flash_ok = False
try:
    import flash_attn  # noqa: F401
    _flash_ok = True
except Exception:
    for _mod_name in ["flash_attn", "flash_attn.flash_attn_interface",
                       "flash_attn_2_cuda"]:
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = types.ModuleType(_mod_name)

# ── 0b. torch_scatter: 先试真 import，失败则 mock ──
_scatter_ok = False
try:
    import torch_scatter  # noqa: F401
    _scatter_ok = True
except Exception:
    def _scatter_mean_native(src, index, dim=-1, out=None, dim_size=None):
        """用 torch.scatter_reduce 替代 torch_scatter.scatter_mean。"""
        if dim != 0 and dim != -src.dim():
            src = src.transpose(0, dim)
        if out is None:
            if dim_size is None:
                dim_size = int(index.max().item()) + 1
            out_shape = list(src.shape)
            out_shape[0] = dim_size
            out = src.new_zeros(out_shape)
        index_exp = index
        if src.dim() > 1 and index.dim() < src.dim():
            index_exp = index.unsqueeze(-1).expand_as(src) if src.dim() > 1 else index
        out = out.scatter_reduce(
            0, index_exp, src, reduce="mean", include_self=False,
        )
        if dim != 0 and dim != -src.dim():
            out = out.transpose(0, dim)
        return out

    _torch_scatter = types.ModuleType("torch_scatter")
    _torch_scatter.scatter_mean = _scatter_mean_native
    sys.modules["torch_scatter"] = _torch_scatter

print(f"[DEPS] flash_attn={'OK' if _flash_ok else 'MOCK'}, "
      f"torch_scatter={'OK' if _scatter_ok else 'MOCK'}")

# ── 0c. 确保 LATO 路径在最前面 ──
_LATO_ROOT = os.environ.get("LATO_ROOT", os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "LATO"))
_LATO_ROOT = os.path.abspath(_LATO_ROOT)
for _p in list(sys.path):
    if os.path.abspath(_p) == _LATO_ROOT:
        sys.path.remove(_p)
sys.path.insert(0, _LATO_ROOT)

# ── 0d. 现在安全 import LATO 原版模块 ──
from lato.modules.sparse import SparseTensor as LATOSparseTensor
from lato.models.lato_vae.lato_vae import VoxelVAE
from vertex_encoder import VoxelFeatureEncoder_active_pointnet
from utils import load_pretrained_woself

# open3d 懒加载（避免 import 时 scipy/numpy 版本冲突）
_o3d = None
def _get_o3d():
    global _o3d
    if _o3d is None:
        import open3d as o3d
        _o3d = o3d
    return _o3d


# ============================================================================
# 体素化 / 点云工具
# ============================================================================

def normalize_mesh(mesh):
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return mesh
    c = (v.min(0) + v.max(0)) / 2
    s = (v.max(0) - v.min(0)).max()
    if s < 1e-8:
        s = 1.0
    v = np.clip((v - c) / s, -0.5 + 1e-6, 0.5 - 1e-6)
    o3d = _get_o3d()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    return mesh


def get_active_voxels(mesh, resolution=128):
    o3d = _get_o3d()
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1.0 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    voxels = np.array([v.grid_index for v in vg.get_voxels()])
    if len(voxels) == 0:
        return torch.empty(0, 3, dtype=torch.int32)
    return torch.from_numpy(voxels).int()


def sample_point_cloud(mesh, n=819200):
    if len(mesh.vertices) == 0:
        return torch.empty(0, 3)
    pcd = mesh.sample_points_uniformly(number_of_points=max(n, 100))
    return torch.from_numpy(np.asarray(pcd.points).astype(np.float32))


# ============================================================================
# 查找 mesh 文件（支持多种扩展名）
# ============================================================================

def find_mesh_file(data_dir, key):
    """查找 mesh 文件，支持 .obj, .ply, .stl 扩展名"""
    base_path = os.path.join(data_dir, "meshes", key)

    # 尝试带扩展名的
    extensions = ['.obj', '.ply', '.stl']
    for ext in extensions:
        path = base_path + ext
        if os.path.exists(path):
            return path

    # 尝试不带扩展名的
    if os.path.exists(base_path):
        return base_path

    return None


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LATO latent 提取 (mock 方案)")
    parser.add_argument("--lato_ckpt", type=str, required=True)
    parser.add_argument("--lato_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=819200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 依赖 mock: flash_attn → stub, torch_scatter → torch.scatter_reduce")

    # ── 1. 加载 LATO 模型 ──
    with open(opt.lato_config, "r") as f:
        lato_cfg = yaml.safe_load(f)
    model_cfg = lato_cfg["model"]
    latent_dim = model_cfg["latent_dim"]

    print(f"[INFO] 构建 VoxelVAE (latent_dim={latent_dim}) ...")
    vae = VoxelVAE(
        in_channels=model_cfg.get("in_channels", 1024),
        latent_dim=latent_dim,
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

    # 加载预训练权重
    result = load_pretrained_woself(
        opt.lato_ckpt, vae=vae, voxel_encoder=voxel_encoder,
    )
    print(f"[INFO] Checkpoint 加载: epoch={result.get('epoch', '?')}")
    vae.eval()
    voxel_encoder.eval()

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

        # 查找 mesh 文件（支持 .obj, .ply, .stl）
        mesh_path = find_mesh_file(opt.data_dir, key)
        if mesh_path is None:
            tqdm.write(f"[SKIP] 未找到: {key} (尝试了 .obj, .ply, .stl)")
            skip += 1
            continue

        try:
            if opt.dry_run:
                continue

            o3d = _get_o3d()
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if len(mesh.vertices) == 0:
                tqdm.write(f"[SKIP] 空 mesh: {key}")
                skip += 1
                continue
            mesh = normalize_mesh(mesh)

            # 4a. 体素化 → active coords
            active_coords = get_active_voxels(mesh, opt.resolution)
            if active_coords.numel() == 0:
                tqdm.write(f"[SKIP] 无 active voxels: {key}")
                skip += 1
                continue
            active_coords = active_coords.to(device)

            # 4b. 构造带 batch 维度的 coords（LATO 需要 [N, 4] 格式）
            coords_4d = torch.cat([
                torch.zeros(len(active_coords), 1, dtype=torch.int32, device=device),
                active_coords,
            ], dim=1)

            # 4c. 点云 → voxel features（LATO 预训练 VoxelFeatureEncoder）
            #     voxel_encoder 期望: p=[B,Np,3], sparse_coords=[N,4]
            pts = sample_point_cloud(mesh, opt.num_points).to(device)
            pts_batched = pts.unsqueeze(0)  # [P, 3] → [1, P, 3]
            active_feats = voxel_encoder(pts_batched, coords_4d, res=opt.resolution)

            # 4d. VAE encode → 16-dim latent
            sparse_in = LATOSparseTensor(feats=active_feats, coords=coords_4d)

            with torch.no_grad():
                latent, _ = vae.encode(sparse_in, sample_posterior=False)

            # 4d. 保存
            np.savez_compressed(
                os.path.join(latent_dir, f"{key}.npz"),
                coords=latent.coords.cpu().numpy().astype(np.int32),
                feats=latent.feats.cpu().numpy().astype(np.float16),
            )

            metadata.at[idx, latent_col] = True
            if "num_voxels" in metadata.columns:
                metadata.at[idx, "num_voxels"] = len(latent.coords)
            success += 1

        except Exception as e:
            tqdm.write(f"[ERROR] {key}: {e}")
            traceback.print_exc()
            error += 1

    # ── 5. 保存 metadata ──
    metadata.to_csv(os.path.join(opt.output_dir, "metadata.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"  完成: {success} 成功, {skip} 跳过, {error} 失败")
    print(f"  Latent 目录: {latent_dir}")
    print(f"  模型名: {lm_name}")
    print(f"\n  下一步:")
    print(f"    python dataset_toolkits/stat_latent.py \\")
    print(f"        --output_dir {opt.output_dir} --model {lm_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
