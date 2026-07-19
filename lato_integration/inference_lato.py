"""
================================================================================
inference_lato.py — TRELLIS + LATO 文本转 3D 推理脚本（v5）
================================================================================

完整推理管线:
  1. SS Flow（刹车卡钳训练）→ dense SS latent (16³×8)
  2. SS Decoder（冻结 TRELLIS 预训练）→ occupancy coords (res 64)
  3. coords × 2 → res 128
  4. SLat Flow（刹车卡钳训练, 16-dim, 128-res）→ structured latent
  5. LATO VoxelVAE.decode() → vertex hierarchy
  6. ConnectionHead → 边预测 → 三角面片化 → Mesh 导出 (.obj)

v5 更新:
  - 自动发现最新 checkpoint（--ss_dir / --slat_dir）
  - 从训练 config JSON 读取模型参数（不再硬编码）
  - 支持 --mode ss_only 快速验证 SS Flow
  - 更好的 checkpoint 格式兼容（state_dict / model / 裸 dict）
  - 更详细的错误提示

用法:
    # 完整推理（自动找最新 ckpt）
    python lato_integration/inference_lato.py \
        --ss_dir outputs/lato_ss_flow \
        --slat_dir outputs/lato_slat_flow \
        --lato_ckpt /path/to/LATO/checkpoints/128to512/vae/vae_128to512.pt \
        --lato_config /path/to/LATO/configs/infer_vae_512.yaml \
        --slat_stats /path/to/database_lato/lato_latents/latents/lato_vae_16dim_128/stats.json \
        --prompt "A brake caliper fixing interaxis 94.31 inner pad 98.47 pistons_num 4"

    # 指定具体 checkpoint
    python lato_integration/inference_lato.py \
        --ss_ckpt outputs/lato_ss_flow/ckpts/denoiser_step0500000.pt \
        --slat_ckpt outputs/lato_slat_flow/ckpts/denoiser_step0200000.pt \
        ...

    # SS-only 模式（只验证 SS Flow，不跑 SLat/LATO）:
    python lato_integration/inference_lato.py \
        --mode ss_only \
        --ss_dir outputs/lato_ss_flow \
        --prompt "A brake caliper"

依赖:
    - TRELLIS (trellis)
    - LATO (lato)
    - networkx, trimesh, open3d, scipy
================================================================================
"""

import os
import sys
import json
import glob
import argparse
import traceback

import numpy as np
import torch
import yaml

# ── 设置路径 ──
_TRELLIS_ROOT = os.environ.get(
    "TRELLIS_ROOT",
    os.path.join(os.path.dirname(__file__), ".."),
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

# ── LATO integration imports ──
from lato_integration.flow.ss_flow import EnhancedSSFlowModel

# ── LATO imports ──
from lato.modules.sparse import SparseTensor as LATOSparseTensor
from lato.models.lato_vae.lato_vae import VoxelVAE
from vertex_encoder import ConnectionHead as LATOConnectionHead
import importlib.util
_spec=importlib.util.spec_from_file_location("lato_utils","/data/huanghaoyang/3D/LATO/utils.py")
_lato_utils=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lato_utils)
load_pretrained_woself=_lato_utils.load_pretrained_woself


# ============================================================================
# 工具函数
# ============================================================================

def find_latest_ckpt(ckpt_dir: str, prefix: str = "denoiser_step") -> str:
    """在目录下找到 step 最大的 checkpoint。

    Args:
        ckpt_dir: checkpoint 目录路径。
        prefix: checkpoint 文件名前缀。

    Returns:
        最新 checkpoint 的完整路径，如果没有找到则返回 None。
    """
    if not os.path.isdir(ckpt_dir):
        return None
    pattern = os.path.join(ckpt_dir, f"{prefix}*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    # 按 step 数排序
    def _extract_step(path):
        name = os.path.basename(path)
        try:
            return int(name.replace(prefix, "").replace(".pt", ""))
        except ValueError:
            return 0
    files.sort(key=_extract_step)
    return files[-1]


def load_checkpoint(ckpt_path: str, device: torch.device) -> dict:
    """加载 checkpoint，兼容多种保存格式。

    支持的格式:
      - 裸 state_dict（直接是权重 dict）
      - {'state_dict': ...}
      - {'model': ...}
      - TRELLIS misc 格式 {'denoiser': ..., 'ema': ...}

    Returns:
        纯 state_dict。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint 格式异常: {type(ckpt)}")

    # 检查是否是 TRELLIS misc 格式（包含 step、denoiser 等）
    if 'denoiser' in ckpt and isinstance(ckpt['denoiser'], dict):
        # TRELLIS misc_*.pt 格式
        state = ckpt['denoiser']
        # 如果有 EMA，EMA 通常更好
        if 'ema' in ckpt and isinstance(ckpt['ema'], dict) and 'denoiser' in ckpt['ema']:
            state = ckpt['ema']['denoiser']
            print("  (使用 EMA 权重)")
        return state

    # 检查是否包含 state_dict / model
    if 'state_dict' in ckpt:
        return ckpt['state_dict']
    if 'model' in ckpt:
        return ckpt['model']

    # 假设就是裸 state_dict（key 是参数名如 "input_layer.weight"）
    # 检查是否有典型的参数 key
    sample_keys = list(ckpt.keys())[:3]
    if any('weight' in k or 'bias' in k or '.' in k for k in sample_keys):
        return ckpt

    raise ValueError(
        f"无法识别 checkpoint 格式。keys: {list(ckpt.keys())[:10]}...\n"
        f"  支持的格式: 裸 state_dict / {{'state_dict': ...}} / {{'model': ...}} / TRELLIS misc"
    )


def build_ss_flow_from_config(config_path: str, device: torch.device, use_fp16: bool = True):
    """从训练 config JSON 构建 EnhancedSSFlowModel。

    Args:
        config_path: lato_ss_flow.json 的路径。
        device: 计算设备。
        use_fp16: 是否使用 FP16。

    Returns:
        EnhancedSSFlowModel 实例。
    """
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    args = cfg['models']['denoiser']['args']

    model = EnhancedSSFlowModel(
        resolution=args['resolution'],
        in_channels=args['in_channels'],
        out_channels=args['out_channels'],
        model_channels=args['model_channels'],
        cond_channels=args['cond_channels'],
        num_blocks=args['num_blocks'],
        num_heads=args.get('num_heads'),
        mlp_ratio=args.get('mlp_ratio', 4),
        patch_size=args.get('patch_size', 1),
        pe_mode=args.get('pe_mode', 'ape'),
        qk_rms_norm=args.get('qk_rms_norm', False),
        use_fp16=use_fp16,
    ).to(device)
    return model


def build_slat_flow_from_config(config_path: str, device: torch.device, use_fp16: bool = True):
    """从训练 config JSON 构建 LATOSLatFlowModel。

    Args:
        config_path: lato_slat_flow.json 的路径。
        device: 计算设备。
        use_fp16: 是否使用 FP16。

    Returns:
        LATOSLatFlowModel 实例。
    """
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    args = cfg['models']['denoiser']['args']

    model = LATOSLatFlowModel(
        resolution=args['resolution'],
        in_channels=args['in_channels'],
        out_channels=args['out_channels'],
        model_channels=args['model_channels'],
        cond_channels=args['cond_channels'],
        num_blocks=args['num_blocks'],
        num_heads=args.get('num_heads'),
        mlp_ratio=args.get('mlp_ratio', 4),
        patch_size=args.get('patch_size', 2),
        num_io_res_blocks=args.get('num_io_res_blocks', 2),
        io_block_channels=args.get('io_block_channels'),
        pe_mode=args.get('pe_mode', 'ape'),
        qk_rms_norm=args.get('qk_rms_norm', False),
        use_fp16=use_fp16,
    ).to(device)
    return model


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
    k_neighbors: int = 32,
) -> list:
    """使用 ConnectionHead + KDTree 预测顶点之间的边。"""
    if device is None:
        device = vertex_feats.device

    N = len(vertex_coords)
    if N < 2:
        return []

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertex_coords.cpu().numpy())
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    all_edge_candidates = set()
    k = min(k_neighbors, N)
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
    """从顶点和边构建三角形 mesh。"""
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
        print("[WARN] 未找到三角面，尝试 convex hull 备选方案...")
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(vertex_coords)
            faces = hull.simplices.tolist()
        except Exception:
            print("[ERROR] Convex hull 也失败了，无法构建 mesh")
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
        description="TRELLIS + LATO 文本转 3D 推理（v5）"
    )

    # ── 模式 ──
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "ss_only"],
                        help="推理模式: full=完整管线, ss_only=仅 SS Flow（快速验证）")

    # ── 配置（替代硬编码参数）──
    parser.add_argument("--ss_config", type=str,
                        default=None,
                        help="SS Flow 训练 config JSON（默认: configs/generation/lato_ss_flow.json）")
    parser.add_argument("--slat_config", type=str,
                        default=None,
                        help="SLat Flow 训练 config JSON（默认: configs/generation/lato_slat_flow.json）")

    # ── SS Flow checkpoint（二选一：--ss_dir 自动发现 或 --ss_ckpt 指定）──
    parser.add_argument("--ss_dir", type=str,
                        default=None,
                        help="SS Flow 训练输出目录（自动发现最新 ckpt）")
    parser.add_argument("--ss_ckpt", type=str,
                        default=None,
                        help="SS Flow checkpoint 路径（指定具体文件）")
    parser.add_argument("--ss_stats", type=str, default=None,
                        help="SS normalization stats JSON（默认 identity）")

    # ── SLat Flow checkpoint（二选一：--slat_dir 或 --slat_ckpt）──
    parser.add_argument("--slat_dir", type=str,
                        default=None,
                        help="SLat Flow 训练输出目录（自动发现最新 ckpt）")
    parser.add_argument("--slat_ckpt", type=str,
                        default=None,
                        help="SLat Flow checkpoint 路径（指定具体文件）")
    parser.add_argument("--slat_stats", type=str,
                        default=None,
                        help="SLat normalization stats JSON（16-dim）")

    # ── LATO VAE（冻结预训练）──
    parser.add_argument("--lato_ckpt", type=str,
                        default=None,
                        help="LATO VoxelVAE checkpoint 路径")
    parser.add_argument("--lato_config", type=str,
                        default=None,
                        help="LATO VAE config yaml 路径")

    # ── TRELLIS 预训练部件 ──
    parser.add_argument("--trellis_pretrained", type=str,
                        default="microsoft/TRELLIS-text-base",
                        help="TRELLIS 预训练 pipeline（提供 SS Decoder + Samplers）")

    # ── 推理参数 ──
    parser.add_argument("--prompt", type=str,
                        default="A brake caliper fixing interaxis 94.31 inner pad 98.47 pistons_num 4",
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

    # ── LATO decode 参数 ──
    parser.add_argument("--lato_threshold", type=float, default=0.2,
                        help="LATO VoxelVAE decode inference_threshold")
    parser.add_argument("--edge_threshold", type=float, default=0.45,
                        help="ConnectionHead 边概率阈值（低=更多边, 高=更少边）")
    parser.add_argument("--k_neighbors", type=int, default=32,
                        help="KDTree 最近邻数（影响候选边数量）")

    # ── 设备 & 精度 ──
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_fp16", action="store_true",
                        help="禁用 FP16（调试用）")

    opt = parser.parse_args()

    # ── 自动设置默认路径 ──
    trellis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if opt.ss_config is None:
        opt.ss_config = os.path.join(trellis_root, "configs", "generation", "lato_ss_flow.json")
    if opt.slat_config is None:
        opt.slat_config = os.path.join(trellis_root, "configs", "generation", "lato_slat_flow.json")
    if opt.lato_config is None:
        opt.lato_config = os.path.join(_LATO_ROOT, "configs", "infer_vae_512.yaml")
    if opt.lato_ckpt is None:
        opt.lato_ckpt = os.path.join(_LATO_ROOT, "checkpoints", "128to512", "vae", "vae_128to512.pt")

    use_fp16 = not opt.no_fp16
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # ── 解析 SS checkpoint ──
    if opt.ss_ckpt:
        ss_ckpt_path = opt.ss_ckpt
    elif opt.ss_dir:
        ss_ckpt_path = find_latest_ckpt(os.path.join(opt.ss_dir, "ckpts"))
        if ss_ckpt_path is None:
            print(f"[ERROR] --ss_dir={opt.ss_dir} 的 ckpts/ 目录中未找到 checkpoint")
            sys.exit(1)
    else:
        ss_ckpt_path = None

    # ── 解析 SLat checkpoint ──
    if opt.slat_ckpt:
        slat_ckpt_path = opt.slat_ckpt
    elif opt.slat_dir:
        slat_ckpt_path = find_latest_ckpt(os.path.join(opt.slat_dir, "ckpts"))
        if slat_ckpt_path is None:
            print(f"[ERROR] --slat_dir={opt.slat_dir} 的 ckpts/ 目录中未找到 checkpoint")
            sys.exit(1)
    else:
        slat_ckpt_path = None

    print("=" * 70)
    print("TRELLIS + LATO 推理（v5）")
    print("=" * 70)
    print(f"  模式:      {opt.mode}")
    print(f"  设备:      {device}")
    print(f"  FP16:      {use_fp16}")
    print(f"  SS ckpt:   {ss_ckpt_path or '(未指定)'}")
    print(f"  SLat ckpt: {slat_ckpt_path or '(未指定)'}")
    print(f"  LATO ckpt: {opt.lato_ckpt}")
    print(f"  输出:      {opt.output}")
    print()

    # ================================================================
    # 1. 加载 TRELLIS SS 管线骨架（SS Decoder + Samplers，预训练权重）
    # ================================================================
    print("[1/6] 加载 TRELLIS 管线骨架（SS Decoder + Samplers）...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(opt.trellis_pretrained)
    print(f"  SS Decoder: {type(pipeline.models['sparse_structure_decoder']).__name__} (冻结)")

    # ================================================================
    # 2. 加载训练的 SS Flow
    # ================================================================
    if ss_ckpt_path is None:
        print("[2/6] 跳过 SS Flow（未指定 checkpoint），使用官方预训练")
    else:
        print(f"[2/6] 加载训练的 SS Flow ...")
        if not os.path.exists(ss_ckpt_path):
            print(f"[ERROR] SS checkpoint 不存在: {ss_ckpt_path}")
            sys.exit(1)

        if not os.path.exists(opt.ss_config):
            print(f"[ERROR] SS config 不存在: {opt.ss_config}")
            sys.exit(1)

        ss_flow = build_ss_flow_from_config(opt.ss_config, device, use_fp16)
        ss_state = load_checkpoint(ss_ckpt_path, device)
        missing, unexpected = ss_flow.load_state_dict(ss_state, strict=False)
        ss_flow.eval()

        # 读取到的 step 信息
        ckpt_step = os.path.basename(ss_ckpt_path).replace("denoiser_step", "").replace(".pt", "")
        print(f"  SS Flow: EnhancedSSFlowModel (from config: {opt.ss_config})")
        print(f"  Checkpoint step: {ckpt_step}")
        if missing:
            print(f"  ⚠️ Missing keys: {len(missing)} (expected for Enhanced model)")
        if unexpected:
            print(f"  ⚠️ Unexpected keys: {len(unexpected)}")

        pipeline.models["sparse_structure_flow_model"] = ss_flow

    # ================================================================
    # 3. 加载训练的 SLat Flow
    # ================================================================
    if opt.mode == "ss_only":
        print("[3/6] 跳过 SLat Flow（ss_only 模式）")
    elif slat_ckpt_path is None:
        print("[3/6] 跳过 SLat Flow（未指定 checkpoint），使用官方预训练")
    else:
        print(f"[3/6] 加载训练的 SLat Flow ...")
        if not os.path.exists(slat_ckpt_path):
            print(f"[ERROR] SLat checkpoint 不存在: {slat_ckpt_path}")
            sys.exit(1)

        if not os.path.exists(opt.slat_config):
            print(f"[ERROR] SLat config 不存在: {opt.slat_config}")
            sys.exit(1)

        slat_flow = build_slat_flow_from_config(opt.slat_config, device, use_fp16)
        slat_state = load_checkpoint(slat_ckpt_path, device)
        missing, unexpected = slat_flow.load_state_dict(slat_state, strict=False)
        slat_flow.eval()

        ckpt_step = os.path.basename(slat_ckpt_path).replace("denoiser_step", "").replace(".pt", "")
        print(f"  SLat Flow: LATOSLatFlowModel (from config: {opt.slat_config})")
        print(f"  Checkpoint step: {ckpt_step}")
        if missing:
            print(f"  ⚠️ Missing keys: {len(missing)}")
        if unexpected:
            print(f"  ⚠️ Unexpected keys: {len(unexpected)}")

        pipeline.models["slat_flow_model"] = slat_flow

    # ================================================================
    # 4. 加载 LATO VoxelVAE + ConnectionHead（冻结预训练）
    # ================================================================
    if opt.mode == "ss_only":
        print("[4/6] 跳过 LATO VAE（ss_only 模式）")
        lato_vae = None
        connection_head = None
        model_cfg = None
    else:
        print("[4/6] 加载 LATO VoxelVAE + ConnectionHead ...")
        if not os.path.exists(opt.lato_ckpt):
            print(f"[ERROR] LATO checkpoint 不存在: {opt.lato_ckpt}")
            sys.exit(1)
        if not os.path.exists(opt.lato_config):
            print(f"[ERROR] LATO config 不存在: {opt.lato_config}")
            sys.exit(1)

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
            channels=1024,
            out_channels=1,
            mlp_ratio=0.75,
        ).to(device)

        result = load_pretrained_woself(
            opt.lato_ckpt,
            vae=lato_vae,
            connection_head=connection_head,
        )
        lato_vae.eval()
        connection_head.eval()
        print(f"  VoxelVAE: latent_dim={model_cfg['latent_dim']}")
        print(f"  ConnectionHead: loaded (epoch={result.get('epoch', '?')})")

    # ================================================================
    # 5. 组装管线（normalization + LATO VAE 注入）
    # ================================================================
    print("[5/6] 组装管线 ...")

    # ── SS normalization ──
    if opt.ss_stats and os.path.exists(opt.ss_stats):
        with open(opt.ss_stats, "r") as f:
            pipeline.ss_normalization = json.load(f)
        print(f"  SS normalization: from {opt.ss_stats}")
    else:
        pipeline.ss_normalization = {"mean": [0.0] * 8, "std": [1.0] * 8}
        print("  SS normalization: identity (mean=0, std=1)")

    # ── SLat normalization ──
    if opt.slat_stats and os.path.exists(opt.slat_stats):
        with open(opt.slat_stats, "r") as f:
            pipeline.slat_normalization = json.load(f)
        print(f"  SLat normalization: from {opt.slat_stats}")
    else:
        pipeline.slat_normalization = {"mean": [0.0] * 16, "std": [1.0] * 16}
        print("  SLat normalization: identity (mean=0, std=1)")

    # ── 注入 LATO VAE ──
    if lato_vae is not None:
        pipeline.models["lato_vae"] = lato_vae
        pipeline.lato_inference_threshold = opt.lato_threshold

    # ── 移除不需要的原版 decoder（节省显存）──
    for key in ["slat_decoder_mesh", "slat_decoder_gs", "slat_decoder_rf"]:
        pipeline.models.pop(key, None)

    pipeline = pipeline.to(device)
    print("  管线已就绪")

    # ================================================================
    # 6. 推理
    # ================================================================
    print(f"\n[6/6] 推理: \"{opt.prompt}\"")
    print(f"  seed={opt.seed}, ss_steps={opt.ss_steps}, "
          f"slat_steps={opt.slat_steps}, cfg={opt.cfg_strength}")

    if opt.mode == "ss_only":
        # ── SS-only 模式：只跑 SS Flow + SS Decoder ──
        with torch.no_grad():
            cond = pipeline.get_cond([opt.prompt])
            torch.manual_seed(opt.seed)
            coords = pipeline.sample_sparse_structure(
                cond,
                num_samples=1,
                sampler_params={
                    "steps": opt.ss_steps,
                    "cfg_strength": opt.cfg_strength,
                },
            )
        print(f"\n{'='*60}")
        print(f"  SS-only 完成!")
        print(f"  Active voxels: {coords.shape[0]}")
        print(f"  Bbox min: {coords[:, 1:].min(dim=0).values.tolist()}")
        print(f"  Bbox max: {coords[:, 1:].max(dim=0).values.tolist()}")
        print(f"{'='*60}")
        return

    # ── Full 模式 ──
    try:
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
    except Exception as e:
        print(f"\n[ERROR] 推理失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================
    # 7. 后处理：LATO decode → mesh
    # ================================================================
    print("\n[后处理] 提取 mesh ...")

    if "lato_decoded" not in outputs:
        print("[ERROR] LATO decode 未产生输出!")
        print(f"  pipeline.models keys: {list(pipeline.models.keys())}")
        print(f"  outputs keys: {list(outputs.keys())}")
        print(f"\n  提示: 请确认 pipeline.models 包含 'lato_vae'")
        sys.exit(1)

    decoded = outputs["lato_decoded"]

    # LATO decode 返回的是 list of dicts（多级分辨率），取最后一级
    if isinstance(decoded, list):
        vertex_result = decoded[-1].get("vertex")
    elif isinstance(decoded, dict):
        vertex_result = decoded.get("vertex")
    else:
        print(f"[ERROR] 不支持的 lato_decoded 类型: {type(decoded)}")
        sys.exit(1)

    if vertex_result is None:
        print(f"[ERROR] 未找到 vertex 结果!")
        if isinstance(decoded, list):
            print(f"  decoded[-1] keys: {list(decoded[-1].keys())}")
        sys.exit(1)

    # 提取顶点坐标和特征
    vertex_coords_4d = vertex_result["coords"]
    vertex_feats = vertex_result["feats"]

    if vertex_coords_4d.shape[-1] == 4:
        # [batch, x, y, z] → [x, y, z]
        vertex_coords_3d = vertex_coords_4d[:, 1:].float()
    else:
        vertex_coords_3d = vertex_coords_4d.float()

    # 归一化坐标到 [-0.5, 0.5]
    if vertex_coords_3d.max() > 1.0:
        last_res = model_cfg["decoder_blocks_vtx"][-1]["resolution"]
        vertex_coords_3d = vertex_coords_3d / float(last_res) - 0.5

    print(f"  顶点数: {len(vertex_coords_3d)}")
    print(f"  特征维度: {vertex_feats.shape[-1]}")
    print(f"  坐标范围: [{vertex_coords_3d.min():.3f}, {vertex_coords_3d.max():.3f}]")

    # 预测边
    print(f"  预测顶点边 (threshold={opt.edge_threshold}, k={opt.k_neighbors})...")
    edges = predict_edges_batched(
        connection_head,
        vertex_feats.float(),
        vertex_coords_3d.float(),
        threshold=opt.edge_threshold,
        device=device,
        k_neighbors=opt.k_neighbors,
    )
    print(f"  预测边数: {len(edges)}")
    if len(edges) == 0:
        print("[ERROR] 未预测到任何边！尝试降低 --edge_threshold（如 0.3）")
        sys.exit(1)

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
    os.makedirs(os.path.dirname(opt.output) or ".", exist_ok=True)
    mesh.export(opt.output)
    print(f"\n{'='*60}")
    print(f"  ✅ 完成! Mesh 已保存到: {opt.output}")
    print(f"  顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
