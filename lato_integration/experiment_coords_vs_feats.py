"""
experiment_coords_vs_feats.py — 隔离碎片化根因：coords 还是 feats？(v2: 测 CD)

v2 改动：上一版只数 L0/L1/L2 顶点数，结果四组几乎一样——证明顶点数不是关键指标。
        本版每组 decode 后接 ConnectionHead 建 mesh、算 Chamfer Distance，
        直接对比「几何质量」，这才是区分 good/bad 的指标。

  4 组对照：
    A: GT coords + GT feats    → 基线（应 CD 低、无碎片）
    B: GT coords + SLat feats  → 只换 feats
    C: SS coords + GT feats    → 只换 coords
    D: SS coords + SLat feats  → 当前状态

  判读：
    A 低、B 高            → feats 是碎片化主因
    A 低、C 高            → coords 是碎片化主因
    A 本身也高            → 截断(52416→16384)/输入格式有问题，先修基线
    B、C 都高、D 更高     → 两者叠加

用法 (在 TRELLIS 目录下):
  python lato_integration/experiment_coords_vs_feats.py \
      --ss_ckpt "$SS_CKPT" --slat_ckpt "$SLAT_CKPT" \
      --slat_stats ... --lato_ckpt ... --lato_config ... --vae_ft_ckpt ... \
      --metadata .../metadata.csv \
      --gt_latents .../lato_vae_16dim_128 \
      --gt_meshes .../meshes \
      --sample_idx 0
"""

import os
import sys
import json
import csv
import ast
import argparse

import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree

# ── 路径设置（与 evaluate_3d_metrics.py 保持一致）──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRELLIS_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_LATO_ROOT = os.environ.get("LATO_ROOT", os.path.join(_TRELLIS_ROOT, "..", "LATO"))
for _p in [_TRELLIS_ROOT, _LATO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lato_integration.evaluate_3d_metrics import (
    load_pipeline, _build_prompt_from_row, compute_all_metrics,
)
from lato_integration.inference_lato import predict_edges_batched, edges_to_mesh


def load_gt_latent(npz_path):
    """加载 GT latent .npz：返回 (coords [N,3], feats [N,16] raw)。"""
    data = np.load(npz_path)
    coords = torch.tensor(data['coords']).int()
    if coords.shape[1] == 4:
        coords = coords[:, 1:]  # 剥离 batch 列 → [N,3]
    feats = torch.tensor(data['feats']).float()  # [N,16] raw（未归一化）
    return coords, feats


def to_4d(coords):
    """[N,3] → [N,4]（batch_col=0, x, y, z）。"""
    return torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1)


def build_mesh_from_decoded(decoded, connection_head, model_cfg, device,
                            edge_threshold=0.45, k_neighbors=32, mesh_mode="grid"):
    """从 VAE decode 输出构建 trimesh（逻辑同 evaluate_3d_metrics.extract_mesh_from_output）。

    mesh_mode: "grid"=格点相邻+四边形化（默认，出完整面）；"knn"=旧 KDTree 三角汤。
    """
    vertex_result = decoded[-1].get("vertex")
    if vertex_result is None:
        return None
    vertex_coords_4d = vertex_result["coords"]
    vertex_feats = vertex_result["feats"]
    if vertex_coords_4d.shape[-1] == 4:
        vertex_coords_int = vertex_coords_4d[:, 1:].long()
    else:
        vertex_coords_int = vertex_coords_4d.long()
    if vertex_coords_int.numel() == 0:
        return None

    if mesh_mode == "grid":
        from lato_integration.mesh_grid import build_mesh_from_grid
        last_res = model_cfg["decoder_blocks_vtx"][-1]["resolution"] * 2
        mesh = build_mesh_from_grid(
            vertex_coords_int, vertex_feats.float(), connection_head, device,
            last_res=last_res, edge_threshold=edge_threshold,
        )
        if mesh is not None and len(mesh.faces) > 1000:
            return mesh
        print(f"  [WARN] 格点建 mesh 过稀/失败 (f={len(mesh.faces) if mesh is not None else 0})，回退 KDTree")

    vertex_coords_3d = vertex_coords_int.float()
    if vertex_coords_3d.max() > 1.0:
        last_res = model_cfg["decoder_blocks_vtx"][-1]["resolution"] * 2
        vertex_coords_3d = vertex_coords_3d / float(last_res) - 0.5

    edges = predict_edges_batched(
        connection_head, vertex_feats.float(), vertex_coords_3d.float(),
        threshold=edge_threshold, device=device, k_neighbors=k_neighbors,
    )
    if len(edges) == 0:
        return None
    mesh = edges_to_mesh(vertex_coords_3d.cpu().numpy(), edges)
    return mesh


def feed_vae_and_eval(vae, connection_head, model_cfg, coords_4d, feats, gt_mesh,
                      device, label, threshold=0.2, edge_threshold=0.45,
                      k_neighbors=32, n_points=20000, use_fp16=False, mesh_mode="grid"):
    """decode → 建 mesh → 算 CD，打印 L0/L1/L2 + CD。"""
    from lato.modules.sparse import SparseTensor as LATOSparseTensor

    torch.cuda.empty_cache()
    if use_fp16:
        feats_in = feats.contiguous().half().to(device)
    else:
        feats_in = feats.contiguous().float().to(device)
    lato_slat = LATOSparseTensor(
        feats=feats_in,
        coords=coords_4d.contiguous().to(device),
    )
    with torch.no_grad():
        decoded = vae.decode(
            lato_slat, training=False,
            inference_threshold=threshold, vis_last_layer=False,
        )

    parts = [f"  {label}"]
    for i, level in enumerate(decoded):
        vr = level.get("vertex", {})
        vc = vr.get("coords")
        nv = vc.shape[0] if vc is not None else 0
        parts.append(f"L{i}={nv}")

    # 建 mesh（grid 模式快；knn 模式慢，~2-5 min，主要花在 KDTree 边候选）
    mesh = build_mesh_from_decoded(
        decoded, connection_head, model_cfg, device,
        edge_threshold=edge_threshold, k_neighbors=k_neighbors, mesh_mode=mesh_mode,
    )
    del decoded
    torch.cuda.empty_cache()

    if mesh is None:
        parts.append("CD=N/A(mesh失败)")
    else:
        try:
            metrics = compute_all_metrics(mesh, gt_mesh, n_points)
            parts.append(f"CD={metrics['chamfer_distance']:.4f}")
            parts.append(f"v={len(mesh.vertices)} f={len(mesh.faces)}")
        except Exception as e:
            parts.append(f"CD=ERR({e})")
        del mesh
    print("  ".join(parts))


def main():
    ap = argparse.ArgumentParser(description="隔离碎片化根因：coords vs feats（测 CD）")
    ap.add_argument("--ss_ckpt", required=True)
    ap.add_argument("--slat_ckpt", required=True)
    ap.add_argument("--slat_stats", default=None)
    ap.add_argument("--lato_ckpt", required=True)
    ap.add_argument("--lato_config", required=True)
    ap.add_argument("--vae_ft_ckpt", default=None)
    ap.add_argument("--trellis_pretrained", default="microsoft/TRELLIS-text-base")
    ap.add_argument("--metadata", required=True, help="训练集 metadata.csv")
    ap.add_argument("--gt_latents", required=True, help="GT latent .npz 目录")
    ap.add_argument("--gt_meshes", required=True, help="GT mesh 目录")
    ap.add_argument("--sample_idx", type=int, default=0, help="用第几条训练样本")
    ap.add_argument("--prompt", default=None, help="覆盖 prompt（可选）")
    ap.add_argument("--ss_threshold", type=float, default=2.0)
    ap.add_argument("--max_coords", type=int, default=16384)
    ap.add_argument("--ss_steps", type=int, default=20)
    ap.add_argument("--slat_steps", type=int, default=20)
    ap.add_argument("--cfg_strength", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lato_threshold", type=float, default=0.2)
    ap.add_argument("--edge_threshold", type=float, default=0.45)
    ap.add_argument("--k_neighbors", type=int, default=32)
    ap.add_argument("--only", default=None, choices=["A", "B", "C", "D"],
                    help="只跑单组实验（A/B/C/D），用于多卡并行或快速验证")
    ap.add_argument("--vae_fp16", action="store_true", default=False,
                    help="VAE decode 用 fp16（int32 上限翻倍，可喂更多 voxel）")
    ap.add_argument("--use_fp16", action="store_true", default=False)
    ap.add_argument("--mesh_mode", type=str, default="grid", choices=["grid", "knn"],
                    help="建 mesh 方式: grid=格点相邻+四边形化（默认），knn=旧 KDTree 三角汤")
    opt = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 选样本 + 构造 prompt ──
    with open(opt.metadata, encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    sample = samples[opt.sample_idx]
    sha256 = sample.get("sha256")  # .npz 文件名 = sha256
    fid = sample.get("file_identifier", sample.get("ID"))
    if opt.prompt:
        prompt = opt.prompt
    elif "captions" in sample and sample.get("captions"):
        caps = ast.literal_eval(sample["captions"])
        prompt = caps[0] if isinstance(caps, list) and len(caps) > 0 else str(caps)
    else:
        prompt = _build_prompt_from_row(sample)
    print(f"样本: file_identifier={fid} sha256={sha256}")
    print(f"prompt: {prompt}\n")

    # ── 2. 加载 GT latent ──
    npz_path = os.path.join(opt.gt_latents, f"{sha256}.npz")
    if not os.path.exists(npz_path):
        files = sorted(os.listdir(opt.gt_latents))
        npz_path = os.path.join(opt.gt_latents, files[opt.sample_idx])
        print(f"[WARN] 未找到 {sha256}.npz，改用 {files[opt.sample_idx]}")
    coords_gt, feats_gt = load_gt_latent(npz_path)
    # 截断到 max_coords（防 spconv int32 溢出）
    if opt.max_coords > 0 and coords_gt.shape[0] > opt.max_coords:
        coords_gt = coords_gt[:opt.max_coords]
        feats_gt = feats_gt[:opt.max_coords]
        print(f"[TRUNC] GT latent 截断到 {opt.max_coords} voxels（防 spconv int32 溢出）")
    print(f"GT latent: coords={tuple(coords_gt.shape)} feats={tuple(feats_gt.shape)} "
          f"feats_mean={feats_gt.mean():.4f} feats_std={feats_gt.std():.4f}\n")

    # ── 2.5 加载 GT mesh ──
    gt_path = sample.get("file_path", "")
    if not (gt_path and os.path.exists(gt_path)):
        gt_path = os.path.join(opt.gt_meshes, f"{fid}.stl")
    if not os.path.exists(gt_path):
        for ext in [".obj", ".ply", ".glb"]:
            alt = os.path.join(opt.gt_meshes, f"{fid}{ext}")
            if os.path.exists(alt):
                gt_path = alt
                break
    print(f"GT mesh: {gt_path}")
    gt_mesh = trimesh.load(gt_path, force="mesh")
    print(f"  GT mesh vertices={len(gt_mesh.vertices)} faces={len(gt_mesh.faces)}\n")

    # ── 3. 加载模型管线（保留 connection_head + model_cfg，mesh 建图要用）──
    print("加载管线 ...")
    pipeline, connection_head, model_cfg = load_pipeline(opt, device)
    vae = pipeline.models["lato_vae"]
    if opt.vae_fp16:
        vae = vae.half()
        pipeline.models["lato_vae"] = vae
        print("  VAE 已转 fp16（int32 上限翻倍）")
    flow_model = pipeline.models["sparse_structure_flow_model"]
    head = pipeline.models["lato_structure_head"]

    torch.manual_seed(opt.seed)
    cond = pipeline.get_cond([prompt])

    # ── 4. SS Flow → StructureHead → SS coords ──
    dtype = next(flow_model.parameters()).dtype
    noise = torch.randn(1, flow_model.in_channels, flow_model.resolution,
                        flow_model.resolution, flow_model.resolution,
                        dtype=dtype, device=device)
    ss_params = {**pipeline.sparse_structure_sampler_params,
                 "steps": opt.ss_steps, "cfg_strength": opt.cfg_strength}
    with torch.no_grad():
        z_s = pipeline.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **ss_params, verbose=False).samples
        occ_logits = head(z_s)
    coords_ss = torch.argwhere(occ_logits > opt.ss_threshold)[:, [0, 2, 3, 4]].int()
    if opt.max_coords > 0 and coords_ss.shape[0] > opt.max_coords:
        active = occ_logits[0, 0][coords_ss[:, 1], coords_ss[:, 2], coords_ss[:, 3]]
        _, topk = torch.topk(active, opt.max_coords)
        coords_ss = coords_ss[topk]
    print(f"SS coords: {coords_ss.shape[0]}\n")

    # 释放 SS Flow + StructureHead
    pipeline.models.pop("sparse_structure_flow_model", None)
    pipeline.models.pop("lato_structure_head", None)
    del flow_model, head, z_s, occ_logits
    torch.cuda.empty_cache()

    # ── 5. SLat Flow 采样（GT coords 和 SS coords 各跑一次）──
    slat_params = {"steps": opt.slat_steps, "cfg_strength": opt.cfg_strength}
    coords_gt_4d = to_4d(coords_gt).to(device)
    coords_ss_4d = coords_ss.to(device)

    print("SLat Flow 采样 @ GT coords ...")
    slat_gt = pipeline.sample_slat(cond, coords_gt_4d, sampler_params=slat_params)
    slat_feats_gt = slat_gt.feats

    print("SLat Flow 采样 @ SS coords ...")
    slat_ss = pipeline.sample_slat(cond, coords_ss_4d, sampler_params=slat_params)
    slat_feats_ss = slat_ss.feats

    # 释放 SLat Flow + CLIP
    pipeline.models.pop("slat_flow_model", None)
    pipeline.text_cond_model = None
    del cond, slat_gt, slat_ss
    torch.cuda.empty_cache()

    # ── 6. C 实验: GT feats 经最近邻映射到 SS coords ──
    tree = KDTree(coords_gt.numpy().astype(np.float64))
    _, nn_idx = tree.query(coords_ss[:, 1:].cpu().numpy().astype(np.float64))
    feats_gt_mapped = feats_gt[nn_idx]  # [N_ss, 16]

    # ── 7. 对照实验（每组建 mesh + 算 CD）──
    experiments = {
        "A": (coords_gt_4d, feats_gt, "A: GT coords + GT feats   "),
        "B": (coords_gt_4d, slat_feats_gt, "B: GT coords + SLat feats"),
        "C": (coords_ss_4d, feats_gt_mapped, "C: SS coords + GT feats   "),
        "D": (coords_ss_4d, slat_feats_ss, "D: SS coords + SLat feats "),
    }
    keys = [opt.only] if opt.only else ["A", "B", "C", "D"]

    print("\n" + "=" * 64)
    print(f"VAE decode 对照实验 (inference_threshold={opt.lato_threshold:.2f})")
    if opt.only:
        print(f"只跑 {opt.only} 组")
    else:
        print("每组含建 mesh + CD，预计共 ~8-20 分钟 ...")
    print("=" * 64)
    for k in keys:
        coords, feats, label = experiments[k]
        feed_vae_and_eval(vae, connection_head, model_cfg, coords, feats,
                          gt_mesh, device, label,
                          opt.lato_threshold, opt.edge_threshold, opt.k_neighbors,
                          use_fp16=opt.vae_fp16, mesh_mode=opt.mesh_mode)
    print("=" * 64)

    # ── 8. 判读提示 ──
    print("\n判读:")
    print("  A CD 低、B CD 高    → feats 是碎片化主因")
    print("  A CD 低、C CD 高    → coords 是碎片化主因")
    print("  A 本身 CD 就高      → 截断(52416→16384)/输入格式有问题，先修基线")
    print("  B、C 都高、D 更高   → 两者叠加")


if __name__ == "__main__":
    main()
