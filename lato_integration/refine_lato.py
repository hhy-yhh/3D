"""
================================================================================
refine_lato.py — 用 LATO 完整 encoder/decoder 精化草稿 mesh（Stage 2）
================================================================================

核心思路（贯穿本项目目标「用 LATO 替换 TRELLIS 达到更好效果」）：

  TRELLIS 负责生成草稿形状，LATO 完整 encode→decode 把草稿精化成干净、
  拓扑完整的 mesh。关键点：Stage 2 走的是 **LATO 原生路径**——

     草稿 mesh → load_quantized_mesh_original (体素化+点特征)
              → VoxelFeatureEncoder_active_pointnet → 1024 维几何特征
              → VAE.encode → 16 维 latent（LATO 原生分布！）
              → VAE.decode → 干净顶点层级
              → ConnectionHead → 精化 mesh

  现有管线把 SLat Flow 的「16 维语义 latent」直接喂给 VAE decoder，
  而 LATO VAE 训练时吃的是「PointNet 几何特征 encode 出的 16 维 latent」，
  两者分布不匹配 → decode 出的顶点云散乱 → mesh 碎片化（细小三角拼接）。
  Stage 2 让 VAE 拿到它训练时见过的输入类型，绕开分布不匹配。

  改编自 LATO 官方 scripts/infer_vae_512.py 的 reconstruct_mesh。
  零重训、零 fp16：voxel_encoder / VAE / ConnectionHead 全用 LATO 预训练权重。

用法（在 evaluate_3d_metrics.py / inference_lato.py 中调用）:
    from lato_integration.refine_lato import build_voxel_encoder, refine_mesh_with_lato
    voxel_encoder = build_voxel_encoder(device)
    load_pretrained_woself(ckpt, vae=vae, connection_head=conn, voxel_encoder=voxel_encoder)
    refined = refine_mesh_with_lato(draft_mesh, vae, voxel_encoder, conn, device, model_cfg)
================================================================================
"""

import os
import sys
import importlib.util

import numpy as np
import torch

# ── 路径设置（与其他 lato_integration 脚本一致）──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRELLIS_ROOT = os.environ.get(
    "TRELLIS_ROOT",
    os.path.abspath(os.path.join(_THIS_DIR, "..")),
)
_LATO_ROOT = os.environ.get(
    "LATO_ROOT",
    os.path.abspath(os.path.join(_TRELLIS_ROOT, "..", "LATO")),
)
for _p in [_TRELLIS_ROOT, _LATO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lato.datasets.vertex_head import load_quantized_mesh_original
from lato.modules.sparse.basic import SparseTensor
from lato_integration.inference_lato import predict_edges_batched, edges_to_mesh

# ── LATO 的 vertex_encoder 与本地 lato_integration/vertex_encoder.py 同名 ──
#    按绝对路径加载 LATO 版，避免 sys.path 解析到本地同名模块。
_lato_ve_path = os.path.join(_LATO_ROOT, "vertex_encoder.py")
if os.path.exists(_lato_ve_path):
    _spec = importlib.util.spec_from_file_location("lato_vertex_encoder", _lato_ve_path)
    _lato_ve = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_lato_ve)
    VoxelFeatureEncoder_active_pointnet = _lato_ve.VoxelFeatureEncoder_active_pointnet
else:
    # fallback：若 LATO 不在预期路径，走 sys.path 解析（需 LATO_ROOT 已设置）
    from vertex_encoder import VoxelFeatureEncoder_active_pointnet  # noqa: E402


def build_voxel_encoder(
    device,
    in_channels: int = 15,
    hidden_dim: int = 256,
    out_channels: int = 1024,
    scatter_type: str = "mean",
    n_blocks: int = 5,
    resolution: int = 128,
):
    """构建与 LATO 官方一致的 VoxelFeatureEncoder_active_pointnet。"""
    model = VoxelFeatureEncoder_active_pointnet(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        scatter_type=scatter_type,
        n_blocks=n_blocks,
        resolution=resolution,
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def refine_mesh_with_lato(
    draft_mesh,
    vae,
    voxel_encoder,
    connection_head,
    device,
    model_cfg,
    volume_resolution: int = 128,
    pc_sample_number: int = 65536,
    vertex_threshold: float = 0.5,
    edge_threshold: float = 0.45,
    k_neighbors: int = 32,
):
    """用 LATO 完整 encoder/decoder 精化草稿 mesh。

    Args:
        draft_mesh: trimesh.Trimesh，顶点需在 [-0.5, 0.5] 归一化空间
            （当前管线 extract_mesh_from_output 已归一化）。
        vae: LATO VoxelVAE（预训练）。
        voxel_encoder: LATO VoxelFeatureEncoder_active_pointnet（预训练）。
        connection_head: LATO ConnectionHead（预训练）。
        model_cfg: LATO VAE config 的 model 部分（取末级 resolution 归一化坐标）。
        volume_resolution: 体素化分辨率（与 GT latent 一致为 128）。
        pc_sample_number: 草稿 mesh 表面采样点数。
        vertex_threshold: VAE decode 的 inference_threshold（LATO 官方默认 0.5）。
        edge_threshold / k_neighbors: ConnectionHead 建 mesh 参数。

    Returns:
        trimesh.Trimesh（精化 mesh），任一步失败返回 None。
    """
    is_cuda = device.type == "cuda"

    # 1. 草稿 mesh → LATO 预处理：voxels[N,3] + point_features[P,15]（pos+normal+VDF）
    voxels, point_features = load_quantized_mesh_original(
        None,
        mesh_load=draft_mesh,
        volume_resolution=volume_resolution,
        use_normals=True,
        pc_sample_number=pc_sample_number,
    )
    if voxels is None or len(voxels) == 0:
        print("[refine_lato] 草稿 mesh 体素化为空，跳过精化")
        return None
    voxels = voxels.to(device)
    point_features = point_features.to(device)
    print(f"[refine_lato] 草稿体素化: {len(voxels)} voxels @ {volume_resolution}³, "
          f"点特征 {tuple(point_features.shape)}")

    # 2. coords_4d [N,4]（batch 列=0 + xyz），与 encode_lato_latent_v2.py 一致
    coords_4d = torch.cat([
        torch.zeros(len(voxels), 1, device=device),
        voxels,
    ], dim=1).int()

    # 3. voxel_encoder → 1024 维几何特征 → VAE.encode → 16 维 latent → decode
    pts_batched = point_features.unsqueeze(0)  # [1, P, 15]
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda
    ):
        active_feats = voxel_encoder(
            p=pts_batched,
            sparse_coords=coords_4d,
            res=volume_resolution,
            bbox_size=(-0.5, 0.5),
        )
        sparse_in = SparseTensor(feats=active_feats, coords=coords_4d)
        latent, _ = vae.encode(sparse_in, sample_posterior=False)
        decoded = vae.decode(
            latent,
            training=False,
            inference_threshold=vertex_threshold,
            vis_last_layer=False,
        )

    del pts_batched, active_feats, sparse_in, latent
    torch.cuda.empty_cache()

    # 4. decode → 顶点 → ConnectionHead → 精化 mesh（复用 inference_lato 的建 mesh）
    vertex_result = decoded[-1].get("vertex")
    if vertex_result is None:
        print("[refine_lato] 精化 decode 无 vertex 输出")
        return None
    vertex_coords_4d = vertex_result["coords"]
    vertex_feats = vertex_result["feats"]
    # 尽早释放 decode 层级，KDTree 建 mesh 阶段是 CPU 密集，GPU 显存不必一直占着
    del decoded
    torch.cuda.empty_cache()
    if vertex_coords_4d.shape[-1] == 4:
        vertex_coords_3d = vertex_coords_4d[:, 1:].float()
    else:
        vertex_coords_3d = vertex_coords_4d.float()
    if vertex_coords_3d.numel() == 0:
        print("[refine_lato] 精化顶点为空")
        return None

    # 坐标归一化：解码链 128→256→512，末级块 out_resolution = resolution * 2
    if vertex_coords_3d.max() > 1.0:
        last_res = model_cfg["decoder_blocks_vtx"][-1]["resolution"] * 2
        vertex_coords_3d = vertex_coords_3d / float(last_res) - 0.5

    edges = predict_edges_batched(
        connection_head, vertex_feats.float(), vertex_coords_3d.float(),
        threshold=edge_threshold, device=device, k_neighbors=k_neighbors,
    )
    if len(edges) == 0:
        print("[refine_lato] 精化边预测为空，跳过")
        return None
    mesh = edges_to_mesh(vertex_coords_3d.cpu().numpy(), edges)
    if mesh is None:
        print("[refine_lato] 精化 mesh 构建失败")
        return None
    print(f"[refine_lato] 精化完成: v={len(mesh.vertices)} f={len(mesh.faces)}")
    return mesh


if __name__ == "__main__":
    print("refine_lato.py — 供 evaluate_3d_metrics.py / inference_lato.py 调用，不独立运行。")
