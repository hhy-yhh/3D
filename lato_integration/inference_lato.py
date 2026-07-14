"""
================================================================================
inference_lato.py — TRELLIS + LATO 文本转 3D 推理脚本
================================================================================

完整推理管线:
  1. TRELLIS SS Flow + SS Decoder → occupancy coords (res 64)
  2. coords × 2 → res 128
  3. LATO SLat Flow (16-dim, 128-res) → structured latent
  4. LATO VoxelVAE.decode() → vertex hierarchy
  5. ConnectionHead → 边预测 → 三角面片化 → Mesh 导出

用法:
    python lato_integration/inference_lato.py \
        --trellis_pretrained microsoft/TRELLIS-text-base \
        --slat_ckpt outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt \
        --lato_ckpt D:/code/LATO/ckpts/your_checkpoint.pt \
        --lato_config D:/code/LATO/configs/infer_vae_512.yaml \
        --prompt "a brake caliper with 4 pistons" \
        --output output_mesh.obj

依赖:
    - TRELLIS (D:/code/TRELLIS_linux/3D/trellis)
    - LATO (D:/code/LATO)
    - networkx, trimesh, open3d
================================================================================
"""

import os
import sys
import argparse
import traceback

import numpy as np
import torch
import yaml

# ── 设置路径 ──
_TRELLIS_ROOT = os.environ.get(
    "TRELLIS_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "trellis"),
)
_LATO_ROOT = os.environ.get(
    "LATO_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "LATO"),
)
_TRELLIS_ROOT = os.path.abspath(_TRELLIS_ROOT)
_LATO_ROOT = os.path.abspath(_LATO_ROOT)

for _p in [_TRELLIS_ROOT, _LATO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── TRELLIS imports ──
from trellis.pipelines.trellis_text_to_3d import TrellisTextTo3DPipeline
from trellis.models.lato_slat_flow import LATOSLatFlowModel

# ── LATO imports ──
from lato.modules.sparse import SparseTensor as LATOSparseTensor
from lato.models.lato_vae.lato_vae import VoxelVAE
from vertex_encoder import ConnectionHead as LATOConnectionHead
from utils import load_pretrained_woself


# ============================================================================
# 工具函数
# ============================================================================

def trellis_to_lato_sparse(trellis_tensor) -> LATOSparseTensor:
    """将 TRELLIS SparseTensor 转换为 LATO SparseTensor。"""
    return LATOSparseTensor(
        feats=trellis_tensor.feats,
        coords=trellis_tensor.coords,
    )


def predict_edges_batched(
    connection_head: LATOConnectionHead,
    vertex_feats: torch.Tensor,
    vertex_coords: torch.Tensor,
    threshold: float = 0.45,
    device: torch.device = None,
    batch_size: int = 8192,
) -> list:
    """
    使用 ConnectionHead 预测顶点之间的边。

    对顶点对 (u, v) 使用双向打分:
      score = sigmoid(conn([feat_u | feat_v]) + conn([feat_v | feat_u]))

    为了支持大量顶点，采用最近邻候选 + 分批处理。

    Args:
        connection_head: LATO ConnectionHead 模块。
        vertex_feats: [N, C] 顶点特征。
        vertex_coords: [N, 3] 顶点坐标。
        threshold: 边存在的概率阈值。
        device: 计算设备。
        batch_size: 每批处理的顶点对数量。

    Returns:
        edges: [(u, v), ...] 预测的边列表。
    """
    if device is None:
        device = vertex_feats.device

    N = len(vertex_coords)
    if N < 2:
        return []

    # 使用 KNN 生成候选边（基于空间距离）
    # 简化: 对每个顶点找最近邻，连接阈值内顶点对
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertex_coords.cpu().numpy())

    # 构建 KDTree 查找邻居
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    edges = []
    all_edge_candidates = set()

    # 对每个顶点找 k 个最近邻
    k = min(32, N)
    for i in range(N):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(
            vertex_coords[i].cpu().numpy(), k
        )
        for j in idx:
            if j > i:
                all_edge_candidates.add((int(i), int(j)))

    if not all_edge_candidates:
        return []

    candidates = list(all_edge_candidates)
    u_list = [c[0] for c in candidates]
    v_list = [c[1] for c in candidates]

    connection_head = connection_head.to(device)
    connection_head.eval()

    probs = []

    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            batch_u = vertex_feats[torch.tensor(u_list[start:end], device=device)]
            batch_v = vertex_feats[torch.tensor(v_list[start:end], device=device)]

            logit_uv = connection_head(torch.cat([batch_u, batch_v], dim=-1))
            logit_vu = connection_head(torch.cat([batch_v, batch_u], dim=-1))
            prob = torch.sigmoid(logit_uv + logit_vu).squeeze(-1)
            probs.append(prob.cpu())

    probs = torch.cat(probs)
    edge_mask = probs > threshold

    edges = [
        (u_list[i], v_list[i])
        for i in range(len(candidates))
        if edge_mask[i].item()
    ]

    return edges


def edges_to_mesh(vertex_coords: np.ndarray, edges: list) -> "trimesh.Trimesh":
    """
    从顶点和边构建三角形 mesh。

    对每条边 (u, v)，查找 u 和 v 的公共邻居 w，形成三角形 (u, v, w)。
    """
    import networkx as nx
    import trimesh

    graph = nx.Graph()
    graph.add_nodes_from(range(len(vertex_coords)))
    graph.add_edges_from(edges)

    faces = []
    for u, v in edges:
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        common = neighbors_u & neighbors_v
        for w in common:
            if w > v:
                faces.append([int(u), int(v), int(w)])

    if len(faces) == 0:
        print("[WARN] 未找到三角面，尝试直接使用边构建 convex hull...")
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(vertex_coords)
            faces = hull.simplices.tolist()
        except Exception:
            print("[ERROR] 无法构建 mesh")
            return None

    faces = np.array(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertex_coords, faces=faces)
    mesh.remove_unreferenced_vertices()
    return mesh


# ============================================================================
# 推理主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TRELLIS + LATO 文本转 3D 推理"
    )
    # 模型路径
    parser.add_argument("--trellis_pretrained", type=str,
                        default="microsoft/TRELLIS-text-base",
                        help="TRELLIS 预训练 pipeline 路径")
    parser.add_argument("--slat_ckpt", type=str, required=True,
                        help="训练的 LATO SLat Flow checkpoint 路径")
    parser.add_argument("--lato_ckpt", type=str, required=True,
                        help="LATO checkpoint 路径 (.pt)")
    parser.add_argument("--lato_config", type=str, required=True,
                        help="LATO VAE 配置文件路径 (yaml)")

    # 推理参数
    parser.add_argument("--prompt", type=str, required=True,
                        help="文本描述")
    parser.add_argument("--output", type=str, default="output_mesh.obj",
                        help="输出 mesh 路径 (.obj)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss_steps", type=int, default=20,
                        help="SS Flow 采样步数")
    parser.add_argument("--slat_steps", type=int, default=20,
                        help="SLat Flow 采样步数")
    parser.add_argument("--cfg_strength", type=float, default=5.0,
                        help="CFG 强度")

    # LATO 参数
    parser.add_argument("--lato_threshold", type=float, default=0.2,
                        help="LATO VoxelVAE decode 的 inference_threshold")
    parser.add_argument("--edge_threshold", type=float, default=0.45,
                        help="ConnectionHead 边概率阈值")

    # SLat 归一化统计量（16-dim LATO latent 的 mean/std）
    parser.add_argument("--slat_stats", type=str, default=None,
                        help="SLat normalization stats JSON 文件路径 "
                             "(16-dim, 由 stat_latent.py 生成). "
                             "如果不指定，使用零均值/单位方差。")

    # 设备 & 精度
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_fp16", action="store_true", default=True,
                        help="使用 FP16 推理")

    opt = parser.parse_args()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")

    # ================================================================
    # 1. 加载 TRELLIS SS 管线（预训练权重）
    # ================================================================
    print("[1/5] 加载 TRELLIS SS pipeline ...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(opt.trellis_pretrained)
    print(f"  SS Flow: {type(pipeline.models['sparse_structure_flow_model']).__name__}")
    print(f"  SS Decoder: {type(pipeline.models['sparse_structure_decoder']).__name__}")

    # ================================================================
    # 2. 加载训练的 LATO SLat Flow
    # ================================================================
    print("[2/5] 加载 LATO SLat Flow ...")
    new_slat_flow = LATOSLatFlowModel(
        resolution=128,
        in_channels=16,
        out_channels=16,
        model_channels=768,
        cond_channels=768,
        num_blocks=12,
        num_heads=12,
        mlp_ratio=4,
        patch_size=2,
        num_io_res_blocks=2,
        io_block_channels=[128],
        pe_mode="ape",
        qk_rms_norm=True,
        use_fp16=opt.use_fp16,
    ).to(device)

    slat_ckpt = torch.load(opt.slat_ckpt, map_location=device, weights_only=True)
    # 兼容完整 checkpoint（含 model/state_dict 包装）和裸 state_dict
    if isinstance(slat_ckpt, dict):
        if 'state_dict' in slat_ckpt:
            slat_ckpt = slat_ckpt['state_dict']
        elif 'model' in slat_ckpt:
            slat_ckpt = slat_ckpt['model']
    new_slat_flow.load_state_dict(slat_ckpt)
    new_slat_flow.eval()
    print(f"  SLat Flow: LATOSLatFlowModel (128-res, 16-dim)")

    # ================================================================
    # 3. 加载 LATO VoxelVAE + ConnectionHead（预训练权重）
    # ================================================================
    print("[3/5] 加载 LATO VoxelVAE ...")
    with open(opt.lato_config, "r") as f:
        lato_cfg = yaml.safe_load(f)
    model_cfg = lato_cfg["model"]

    lato_vae = VoxelVAE(
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

    connection_head = LATOConnectionHead(
        channels=512 * 2,
        out_channels=1,
    ).to(device)

    result = load_pretrained_woself(
        opt.lato_ckpt,
        vae=lato_vae,
        connection_head=connection_head,
    )
    print(f"  VoxelVAE: latent_dim={model_cfg['latent_dim']}")
    print(f"  ConnectionHead: loaded (epoch={result.get('epoch', '?')})")

    lato_vae.eval()
    connection_head.eval()

    # ================================================================
    # 4. 组装管线
    # ================================================================
    print("[4/5] 组装管线 ...")

    # 替换 SLat Flow
    pipeline.models["slat_flow_model"] = new_slat_flow

    # 覆盖 slat_normalization 为 16-dim LATO 统计量
    # （原版 TRELLIS 是 8-dim，与 LATO 16-dim 不兼容）
    if opt.slat_stats is not None and os.path.exists(opt.slat_stats):
        import json
        with open(opt.slat_stats, "r") as f:
            stats = json.load(f)
        pipeline.slat_normalization = stats
        print(f"  SLat normalization (16-dim): mean={len(stats['mean'])} values, "
              f"std={len(stats['std'])} values")
    else:
        # 使用零均值/单位方差作为后备
        pipeline.slat_normalization = {
            "mean": [0.0] * 16,
            "std": [1.0] * 16,
        }
        print("  SLat normalization: using identity (mean=0, std=1)")

    # 添加 LATO VAE
    pipeline.models["lato_vae"] = lato_vae

    # 移除不需要的原版 decoder（节省显存）
    for key in ["slat_decoder_mesh", "slat_decoder_gs", "slat_decoder_rf"]:
        pipeline.models.pop(key, None)

    pipeline = pipeline.to(device)
    print("  管线已就绪")

    # ================================================================
    # 5. 推理
    # ================================================================
    print(f"[5/5] 推理: \"{opt.prompt}\" ...")
    print(f"  seed={opt.seed}, ss_steps={opt.ss_steps}, "
          f"slat_steps={opt.slat_steps}, cfg={opt.cfg_strength}")

    with torch.no_grad():
        outputs = pipeline.run(
            opt.prompt,
            seed=opt.seed,
            sparse_structure_sampler_params={
                "steps": opt.ss_steps,
                "cfg_strength": opt.cfg_strength,
            },
            slat_sampler_params={
                "steps": opt.slat_steps,
                "cfg_strength": opt.cfg_strength,
            },
            formats=["mesh"],
        )

    # ================================================================
    # 6. 后处理：LATO decode → mesh
    # ================================================================
    print("\n[后处理] 提取 mesh ...")

    if "lato_decoded" not in outputs:
        print("[ERROR] LATO decode 未产生输出!")
        print(f"  可用 keys: {list(outputs.keys())}")
        sys.exit(1)

    decoded = outputs["lato_decoded"]
    vertex_result = decoded[-1].get("vertex")
    if vertex_result is None:
        print("[ERROR] 未找到 vertex 结果!")
        print(f"  decoded[-1] keys: {list(decoded[-1].keys())}")
        sys.exit(1)

    # 提取顶点坐标和特征
    vertex_coords_4d = vertex_result["coords"]
    vertex_feats = vertex_result["feats"]

    if vertex_coords_4d.shape[-1] == 4:
        vertex_coords_3d = vertex_coords_4d[:, 1:].float()
    else:
        vertex_coords_3d = vertex_coords_4d.float()

    # 归一化坐标到 [-0.5, 0.5]
    if vertex_coords_3d.max() > 1.0:
        last_res = model_cfg["decoder_blocks_vtx"][-1]["resolution"]
        vertex_coords_3d = vertex_coords_3d / float(last_res) - 0.5

    print(f"  顶点数: {len(vertex_coords_3d)}")
    print(f"  特征维度: {vertex_feats.shape[-1]}")

    # 预测边
    print("  预测顶点边 ...")
    edges = predict_edges_batched(
        connection_head,
        vertex_feats.float(),
        vertex_coords_3d.float(),
        threshold=opt.edge_threshold,
        device=device,
    )
    print(f"  预测边数: {len(edges)}")

    # 三角面片化
    print("  三角面片化 ...")
    mesh = edges_to_mesh(
        vertex_coords_3d.cpu().numpy(),
        edges,
    )

    if mesh is None:
        print("[ERROR] Mesh 构建失败")
        sys.exit(1)

    # 保存
    mesh.export(opt.output)
    print(f"\n{'='*60}")
    print(f"  完成! Mesh 已保存到: {opt.output}")
    print(f"  顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
