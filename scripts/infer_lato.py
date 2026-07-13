"""
LATO-enhanced inference script.
使用官方 TRELLIS SS 管线 + 你训练的 LATO SLat Flow + 预训练 LATO VoxelVAE。

用法:
  python scripts/infer_lato.py \
      --prompt "a brake caliper with 4 pistons" \
      --ss_ckpt microsoft/TRELLIS-text-base \
      --slat_flow_ckpt outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt \
      --lato_ckpt D:/code/3D/LATO/ckpts/your_checkpoint.pt \
      --output_dir outputs/inference
"""

import os
import sys
import argparse
import torch
import numpy as np
import yaml

# === 路径 ===
TRELLIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRELLIS_ROOT)

# LATO 路径（根据实际情况调整）
LATO_ROOT = os.environ.get('LATO_ROOT', '/data/huanghaoyang/3D/LATO')
if os.path.exists(LATO_ROOT):
    sys.path.insert(0, LATO_ROOT)

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.models.lato_slat_flow import LATOSLatFlowModel

# LATO imports（如果 LATO 代码可用）
try:
    from lato.models.lato_vae.lato_vae import VoxelVAE
    from vertex_encoder import ConnectionHead
    from utils import load_pretrained_woself
    LATO_AVAILABLE = True
except ImportError:
    LATO_AVAILABLE = False
    print("⚠️ LATO 模块未找到，请设置 LATO_ROOT 环境变量")


def load_lato_vae(checkpoint_path, config_path, device):
    """加载 LATO 预训练 VoxelVAE + ConnectionHead"""
    if not LATO_AVAILABLE:
        raise ImportError("LATO 模块不可用")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)["model"]

    vae = VoxelVAE(
        in_channels=config["in_channels"],
        latent_dim=config["latent_dim"],
        encoder_blocks=config["encoder_blocks"],
        decoder_blocks_vtx=config["decoder_blocks_vtx"],
        num_heads=8, num_head_channels=64, mlp_ratio=4.0,
        attn_mode="swin", window_size=8, pe_mode="ape",
        use_fp16=False, use_checkpoint=False, qk_rms_norm=False,
        using_subdivide=True,
        using_attn=config.get("using_attn", False),
    ).to(device)
    connection_head = ConnectionHead(channels=512 * 2, out_channels=1, mlp_ratio=0.75).to(device)

    load_pretrained_woself(checkpoint_path, vae=vae, connection_head=connection_head)
    vae.eval()
    connection_head.eval()
    print("✅ LATO VAE + ConnectionHead 加载完成")
    return vae, connection_head


def predict_edges(connection_head, vertex_feats, vertex_coords, threshold=0.45, device="cuda", k_neighbors=None):
    """LATO ConnectionHead 预测边"""
    num_v = vertex_feats.shape[0]
    if num_v < 3:
        return np.empty((0, 2), dtype=np.int64)

    if k_neighbors and k_neighbors < num_v:
        dist = torch.cdist(vertex_coords, vertex_coords)
        _, indices = torch.topk(dist, k=k_neighbors + 1, dim=1, largest=False)
        neighbors = indices[:, 1:]
        src = torch.arange(num_v, device=device).unsqueeze(1).repeat(1, k_neighbors).flatten()
        dst = neighbors.flatten()
        mask = src < dst
        u_idx, v_idx = src[mask], dst[mask]
    else:
        u_idx, v_idx = torch.triu_indices(num_v, num_v, offset=1, device=device)

    probs = []
    with torch.no_grad():
        for s in range(0, u_idx.shape[0], 4096):
            e = min(s + 4096, u_idx.shape[0])
            bu, bv = u_idx[s:e], v_idx[s:e]
            logits_uv = connection_head(torch.cat([vertex_feats[bu], vertex_feats[bv]], dim=-1))
            logits_vu = connection_head(torch.cat([vertex_feats[bv], vertex_feats[bu]], dim=-1))
            probs.append(torch.sigmoid(logits_uv + logits_vu).squeeze(-1))

    if not probs:
        return np.empty((0, 2), dtype=np.int64)
    edge_mask = torch.cat(probs) > threshold
    return torch.stack([u_idx[edge_mask], v_idx[edge_mask]], dim=1).cpu().numpy()


def build_triangles(edges, num_vertices):
    """从边构建三角面片"""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(num_vertices))
    g.add_edges_from(edges)
    faces = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        for w in set(g.neighbors(u)) & set(g.neighbors(v)):
            if w > v:
                faces.append([u, v, w])
    return np.array(faces, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--ss_ckpt", type=str, default="microsoft/TRELLIS-text-base",
                        help="TRELLIS SS 管线 checkpoint（官方或自己训的）")
    parser.add_argument("--slat_flow_ckpt", type=str, required=True,
                        help="你训练的 LATO SLat Flow 权重 .pt")
    parser.add_argument("--lato_ckpt", type=str, required=True,
                        help="LATO 预训练 checkpoint .pt")
    parser.add_argument("--lato_config", type=str,
                        default=os.path.join(LATO_ROOT, "configs/infer_vae_512.yaml") if os.path.exists(LATO_ROOT) else "configs/infer_vae_512.yaml")
    parser.add_argument("--output_dir", type=str, default="./outputs/lato_inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ss_steps", type=int, default=20)
    parser.add_argument("--slat_steps", type=int, default=20)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 加载 TRELLIS SS 管线（预训练） ──
    print("加载 TRELLIS SS 管线...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.ss_ckpt)

    # ── 2. 加载你训练的 LATO SLat Flow ──
    print("加载 LATO SLat Flow...")
    slat_flow = LATOSLatFlowModel(
        resolution=128, in_channels=16, out_channels=16,
        model_channels=768, cond_channels=768,
        num_blocks=12, num_heads=12, patch_size=2,
        use_fp16=True
    ).to(device)
    slat_flow.load_state_dict(
        torch.load(args.slat_flow_ckpt, map_location=device), strict=False
    )
    slat_flow.eval()

    # ── 3. 加载 LATO VAE ──
    lato_vae, connection_head = load_lato_vae(args.lato_ckpt, args.lato_config, device)

    # ── 4. 替换 pipeline 模型 ──
    pipeline.models["slat_flow_model"] = slat_flow
    pipeline.models["lato_vae"] = lato_vae
    # 删除不用的原版 decoder（避免 key error）
    for k in ["slat_decoder_mesh", "slat_decoder_gs", "slat_decoder_rf"]:
        pipeline.models.pop(k, None)

    # ── 5. 推理 ──
    pipeline.cuda()
    print(f"\n生成: {args.prompt}")
    outputs = pipeline.run(
        args.prompt,
        num_samples=args.num_samples,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": args.cfg_strength,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.cfg_strength,
        },
        formats=["mesh"],
    )

    # ── 6. 后处理：LATO decoded → 三角面片化 → Mesh ──
    if "lato_decoded" in outputs:
        decoded = outputs["lato_decoded"]
        if isinstance(decoded, dict):
            # 获取最后一级的 vertex
            if 'vertex' in decoded:
                vertex = decoded["vertex"]
            else:
                # 可能是列表形式
                vertex = decoded[-1]["vertex"] if isinstance(decoded, list) else decoded
        else:
            vertex = decoded[-1]["vertex"] if isinstance(decoded, list) else decoded
        
        v_coords = vertex["coords"].float() / 512.0 - 0.5
        v_feats = vertex["feats"]

        edges = predict_edges(connection_head, v_feats, v_coords,
                              threshold=0.45, device=device)
        faces = build_triangles(edges, len(v_coords))

        import trimesh
        mesh = trimesh.Trimesh(vertices=v_coords.cpu().numpy(), faces=faces, process=False)
        if len(mesh.faces) > 0:
            trimesh.repair.fix_normals(mesh)

        obj_path = os.path.join(args.output_dir, "output.obj")
        mesh.export(obj_path)
        print(f"✅ Mesh 已保存: {obj_path}")
        print(f"   顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")

        # 保存 prompt
        with open(os.path.join(args.output_dir, "prompt.txt"), "w") as f:
            f.write(args.prompt)
    else:
        print("❌ 未产生 LATO decoded 输出，检查 pipeline models 字典")


if __name__ == "__main__":
    main()
