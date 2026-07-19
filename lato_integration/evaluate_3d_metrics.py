"""
================================================================================
evaluate_3d_metrics.py — TRELLIS+LATO 文本转3D 批量评估脚本
================================================================================

对测试集逐条推理，计算 Chamfer Distance / Hausdorff Distance / Normal Consistency。

用法:
    python lato_integration/evaluate_3d_metrics.py \
        --ss_ckpt outputs/lato_ss_flow/ckpts/denoiser_step1000000.pt \
        --slat_ckpt outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt \
        --lato_ckpt /path/to/LATO/checkpoints/128to512/vae/vae_128to512.pt \
        --lato_config /path/to/LATO/configs/infer_vae_512.yaml \
        --test_metadata /path/to/database_lato/test/metadata.csv \
        --gt_meshes /path/to/database_lato/meshes \
        --output_dir outputs/eval_results

依赖:
    - TRELLIS, LATO
    - open3d, trimesh, numpy, scipy
================================================================================
"""

import os
import sys
import json
import csv
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
import open3d as o3d
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm

# ── 路径 ──
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

from trellis.pipelines.trellis_text_to_3d import TrellisTextTo3DPipeline
from trellis.models.lato_slat_flow import LATOSLatFlowModel
from lato_integration.flow.ss_flow import EnhancedSSFlowModel
from lato.models.lato_vae.lato_vae import VoxelVAE
from vertex_encoder import ConnectionHead as LATOConnectionHead
if "utils" in sys.modules:
    del sys.modules["utils"]
import importlib.util
_spec = importlib.util.spec_from_file_location("lato_utils", "/data/huanghaoyang/3D/LATO/utils.py")
_lato_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lato_utils)
load_pretrained_woself = _lato_utils.load_pretrained_woself


# ============================================================================
# 3D 评估指标
# ============================================================================

def sample_points_from_mesh(mesh: trimesh.Trimesh, n_points: int = 50000) -> np.ndarray:
    """从 mesh 表面均匀采样点。"""
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def sample_points_and_normals(mesh: trimesh.Trimesh, n_points: int = 50000):
    """采样点和法线。"""
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx]
    return points, normals


def chamfer_distance(points_pred: np.ndarray, points_gt: np.ndarray) -> float:
    """
    Chamfer Distance (双向最近邻 L2 距离均值):
      CD = 1/|P| Σ min_{q∈Q} ||p - q||² + 1/|Q| Σ min_{p∈P} ||q - p||²
    """
    tree_gt = KDTree(points_gt)
    tree_pred = KDTree(points_pred)

    dist_p_to_q, _ = tree_gt.query(points_pred)  # pred → gt
    dist_q_to_p, _ = tree_pred.query(points_gt)   # gt → pred

    cd = (dist_p_to_q ** 2).mean() + (dist_q_to_p ** 2).mean()
    return float(cd)


def hausdorff_distance(points_pred: np.ndarray, points_gt: np.ndarray) -> float:
    """
    Hausdorff Distance (单向最大-最小距离，再取双向最大值):
      d_H = max{ max_{p∈P} min_{q∈Q} ||p - q||, max_{q∈Q} min_{p∈P} ||q - p|| }
    """
    tree_gt = KDTree(points_gt)
    tree_pred = KDTree(points_pred)

    dist_p_to_q, _ = tree_gt.query(points_pred)
    dist_q_to_p, _ = tree_pred.query(points_gt)

    return float(max(dist_p_to_q.max(), dist_q_to_p.max()))


def normal_consistency(
    points_pred: np.ndarray,
    normals_pred: np.ndarray,
    points_gt: np.ndarray,
    normals_gt: np.ndarray,
    k: int = 10,
) -> float:
    """
    Normal Consistency: 预测点到 GT 最近邻的法线夹角均值。
    NC = 1/|P| Σ |n_p · n_{q_nearest}|

    返回值范围 [0, 1]，越接近 1 越一致。
    """
    tree_gt = KDTree(points_gt)
    _, idx = tree_gt.query(points_pred, k=1)  # 每个 pred 点找最近的 gt 点
    nearest_normals = normals_gt[idx]

    # |dot product|, 法线方向可能相反
    dot = np.abs(np.sum(normals_pred * nearest_normals, axis=1))
    dot = np.clip(dot, 0.0, 1.0)
    return float(dot.mean())


def compute_all_metrics(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    n_points: int = 50000,
) -> dict:
    """计算全部三个指标。"""
    # 采样
    pred_pts_raw, pred_norms = sample_points_and_normals(pred_mesh, n_points)
    gt_pts_raw, gt_norms = sample_points_and_normals(gt_mesh, n_points)

    # 归一化到同一尺度：用 GT bbox 对角线统一缩放
    gt_scale = np.linalg.norm(gt_pts_raw.max(axis=0) - gt_pts_raw.min(axis=0))
    if gt_scale > 0:
        pred_pts = pred_pts_raw / gt_scale
        gt_pts = gt_pts_raw / gt_scale
    else:
        pred_pts, gt_pts = pred_pts_raw, gt_pts_raw

    cd = chamfer_distance(pred_pts, gt_pts)
    hd = hausdorff_distance(pred_pts, gt_pts)
    nc = normal_consistency(pred_pts, pred_norms, gt_pts, gt_norms)

    return {
        "chamfer_distance": cd,
        "hausdorff_distance": hd,
        "normal_consistency": nc,
    }


# ============================================================================
# 推理管线加载
# ============================================================================

def load_pipeline(opt, device):
    """加载完整推理管线。"""
    print("[1/4] 加载 TRELLIS 管线骨架（SS Decoder + Samplers）...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(opt.trellis_pretrained)

    print("[2/4] 加载训练好的 SS Flow ...")
    ss_flow = EnhancedSSFlowModel(
        resolution=16, in_channels=8, out_channels=8,
        model_channels=512, cond_channels=768,
        num_blocks=24, num_heads=16, mlp_ratio=4,
        patch_size=1, pe_mode="ape", qk_rms_norm=True,
        use_fp16=opt.use_fp16,
    ).to(device)
    ss_ckpt = torch.load(opt.ss_ckpt, map_location=device, weights_only=True)
    if isinstance(ss_ckpt, dict):
        ss_ckpt = ss_ckpt.get('state_dict', ss_ckpt.get('model', ss_ckpt))
    ss_flow.load_state_dict(ss_ckpt)
    ss_flow.eval()
    pipeline.models["sparse_structure_flow_model"] = ss_flow

    print("[3/4] 加载训练好的 SLat Flow ...")
    slat_flow = LATOSLatFlowModel(
        resolution=128, in_channels=16, out_channels=16,
        model_channels=384, cond_channels=768,
        num_blocks=12, num_heads=8, mlp_ratio=4,
        patch_size=2, num_io_res_blocks=2, io_block_channels=[128],
        pe_mode="ape", qk_rms_norm=True, use_fp16=opt.use_fp16,
    ).to(device)
    slat_ckpt = torch.load(opt.slat_ckpt, map_location=device, weights_only=True)
    if isinstance(slat_ckpt, dict):
        slat_ckpt = slat_ckpt.get('state_dict', slat_ckpt.get('model', slat_ckpt))
    slat_flow.load_state_dict(slat_ckpt)
    slat_flow.eval()
    pipeline.models["slat_flow_model"] = slat_flow

    # Normalization
    pipeline.ss_normalization = {"mean": [0.0]*8, "std": [1.0]*8}
    if opt.slat_stats and os.path.exists(opt.slat_stats):
        with open(opt.slat_stats, "r") as f:
            pipeline.slat_normalization = json.load(f)
    else:
        pipeline.slat_normalization = {"mean": [0.0]*16, "std": [1.0]*16}

    # LATO VAE
    print("[4/4] 加载 LATO VoxelVAE ...")
    with open(opt.lato_config, "r") as f:
        lato_cfg = yaml.safe_load(f)
    model_cfg = lato_cfg["model"]
    lato_vae = VoxelVAE(
        in_channels=model_cfg.get("in_channels", 1024),
        latent_dim=model_cfg["latent_dim"],
        encoder_blocks=model_cfg["encoder_blocks"],
        decoder_blocks_vtx=model_cfg["decoder_blocks_vtx"],
        attn_mode="swin", window_size=8, pe_mode="ape",
        using_subdivide=True,
        using_attn=model_cfg.get("using_attn", False),
    ).to(device)
    connection_head = LATOConnectionHead(channels=512, out_channels=1, mlp_ratio=0.75).to(device)
    load_pretrained_woself(opt.lato_ckpt, vae=lato_vae, connection_head=connection_head)
    lato_vae.eval()
    connection_head.eval()

    pipeline.models["lato_vae"] = lato_vae
    pipeline.lato_inference_threshold = opt.lato_threshold
    for key in ["slat_decoder_mesh", "slat_decoder_gs", "slat_decoder_rf"]:
        pipeline.models.pop(key, None)
    pipeline.to(device)

    return pipeline, connection_head, model_cfg


# ============================================================================
# Mesh 后处理（从 pipeline 输出提取）
# ============================================================================

def extract_mesh_from_output(outputs, connection_head, model_cfg, device, edge_threshold=0.45, k_neighbors=32):
    """从 pipeline.run() 输出提取 trimesh 对象。"""
    from lato_integration.inference_lato import predict_edges_batched, edges_to_mesh

    if "lato_decoded" not in outputs:
        return None

    decoded = outputs["lato_decoded"]
    vertex_result = decoded[-1].get("vertex")
    if vertex_result is None:
        return None

    vertex_coords_4d = vertex_result["coords"]
    vertex_feats = vertex_result["feats"]

    if vertex_coords_4d.shape[-1] == 4:
        vertex_coords_3d = vertex_coords_4d[:, 1:].float()
    else:
        vertex_coords_3d = vertex_coords_4d.float()

    if vertex_coords_3d.max() > 1.0:
        last_res = model_cfg["decoder_blocks_vtx"][-1]["resolution"]
        vertex_coords_3d = vertex_coords_3d / float(last_res) - 0.5

    edges = predict_edges_batched(
        connection_head, vertex_feats.float(), vertex_coords_3d.float(),
        threshold=edge_threshold, device=device, k_neighbors=k_neighbors,
    )

    if len(edges) == 0:
        return None

    mesh = edges_to_mesh(vertex_coords_3d.cpu().numpy(), edges)
    return mesh


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TRELLIS+LATO 批量评估")
    parser.add_argument("--ss_ckpt", type=str, required=True,
                        help="训练好的 SS Flow checkpoint")
    parser.add_argument("--slat_ckpt", type=str, required=True,
                        help="训练好的 SLat Flow checkpoint")
    parser.add_argument("--slat_stats", type=str, default=None,
                        help="SLat normalization stats JSON")
    parser.add_argument("--lato_ckpt", type=str, required=True,
                        help="LATO VAE checkpoint")
    parser.add_argument("--lato_config", type=str, required=True,
                        help="LATO VAE 配置文件")
    parser.add_argument("--trellis_pretrained", type=str,
                        default="microsoft/TRELLIS-text-base")
    parser.add_argument("--test_metadata", type=str, required=True,
                        help="测试集 metadata.csv 路径")
    parser.add_argument("--gt_meshes", type=str, required=True,
                        help="GT mesh 目录（STL/OBJ/PLY）")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results",
                        help="输出目录")
    parser.add_argument("--n_points", type=int, default=50000,
                        help="mesh 采样点数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss_steps", type=int, default=20)
    parser.add_argument("--slat_steps", type=int, default=20)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--lato_threshold", type=float, default=0.2)
    parser.add_argument("--edge_threshold", type=float, default=0.45)
    parser.add_argument("--k_neighbors", type=int, default=32,
                        help="KDTree 最近邻数，越小面越少")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_fp16", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="限制评估条数（0=全部）")
    parser.add_argument("--save_meshes", action="store_true", default=False,
                        help="保存生成的 mesh 文件到 output_dir/meshes/")
    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.output_dir, exist_ok=True)

    # ── 加载测试元数据 ──
    print(f"加载测试元数据: {opt.test_metadata}")
    with open(opt.test_metadata, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_samples = list(reader)
    print(f"  测试集样本数: {len(test_samples)}")

    if opt.limit > 0:
        test_samples = test_samples[:opt.limit]
        print(f"  限制为: {opt.limit} 条")

    # ── 加载管线（一次）──
    print("\n" + "=" * 60)
    pipeline, connection_head, model_cfg = load_pipeline(opt, device)
    print("=" * 60)

    # ── 预加载 GT mesh ──
    print("\n预加载 GT mesh ...")
    gt_meshes = {}
    sha256_col = "file_identifier" if "sha256" in test_samples[0] else None
    for sample in tqdm(test_samples, desc="Loading GT"):
        sha = sample[sha256_col] if sha256_col else sample.get("file_identifier", sample.get("ID"))
        gt_path = os.path.join(opt.gt_meshes, f"{sha}.stl")
        if not os.path.exists(gt_path):
            # 尝试其他扩展名
            for ext in [".obj", ".ply", ".glb"]:
                alt = os.path.join(opt.gt_meshes, f"{sha}{ext}")
                if os.path.exists(alt):
                    gt_path = alt
                    break
        try:
            gt_meshes[sha] = trimesh.load(gt_path, force="mesh")
        except Exception as e:
            print(f"  [WARN] 无法加载 {sha}: {e}")

    print(f"  已加载 {len(gt_meshes)} 个 GT mesh")

    # ── 逐条推理 + 评估 ──
    results = []
    failures = []
    captions_col = "captions" if "captions" in test_samples[0] else None

    print("\n" + "=" * 60)
    print("开始批量推理 + 评估")
    print("=" * 60)

    for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        sha = sample[sha256_col] if sha256_col else sample.get("file_identifier", sample.get("ID"))

        if sha not in gt_meshes:
            failures.append({"sha": sha, "error": "GT mesh not found"})
            continue

        # 获取 prompt
        if captions_col and sample.get(captions_col):
            import ast
            captions = ast.literal_eval(sample[captions_col])
            prompt = captions[0] if isinstance(captions, list) and len(captions) > 0 else str(captions)
        else:
            # 从参数列自动构造 prompt
            prompt = _build_prompt_from_row(sample)

        try:
            with torch.no_grad():
                            outputs = pipeline.run(
                    prompt,
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

            pred_mesh = extract_mesh_from_output(
                outputs, connection_head, model_cfg, device, opt.edge_threshold, opt.k_neighbors
            )

            if pred_mesh is None:
                failures.append({"sha": sha, "error": "Mesh extraction failed"})
                continue

            # 保存生成 mesh
            if opt.save_meshes:
                mesh_dir = os.path.join(opt.output_dir, "meshes")
                os.makedirs(mesh_dir, exist_ok=True)
                mesh_path = os.path.join(mesh_dir, f"{sha}.obj")
                pred_mesh.export(mesh_path)
                # 也保存顶点+面信息到 JSON 目录（方便对照）
                meta_path = os.path.join(mesh_dir, f"{sha}.json")
                with open(meta_path, "w") as f:
                    json.dump({
                        "sha": sha,
                        "prompt": prompt,
                        "num_vertices": len(pred_mesh.vertices),
                        "num_faces": len(pred_mesh.faces),
                    }, f, indent=2)

            gt_mesh = gt_meshes[sha]
            metrics = compute_all_metrics(pred_mesh, gt_mesh, opt.n_points)

            results.append({
                "sha": sha,
                "prompt": prompt,
                **metrics,
            })

        except Exception as e:
            failures.append({"sha": sha, "error": str(e)})
            continue

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    if results:
        cd_vals = [r["chamfer_distance"] for r in results]
        hd_vals = [r["hausdorff_distance"] for r in results]
        nc_vals = [r["normal_consistency"] for r in results]

        summary = {
            "num_success": len(results),
            "num_failures": len(failures),
            "chamfer_distance": {
                "mean": float(np.mean(cd_vals)),
                "std": float(np.std(cd_vals)),
                "min": float(np.min(cd_vals)),
                "max": float(np.max(cd_vals)),
                "median": float(np.median(cd_vals)),
            },
            "hausdorff_distance": {
                "mean": float(np.mean(hd_vals)),
                "std": float(np.std(hd_vals)),
                "min": float(np.min(hd_vals)),
                "max": float(np.max(hd_vals)),
                "median": float(np.median(hd_vals)),
            },
            "normal_consistency": {
                "mean": float(np.mean(nc_vals)),
                "std": float(np.std(nc_vals)),
                "min": float(np.min(nc_vals)),
                "max": float(np.max(nc_vals)),
                "median": float(np.median(nc_vals)),
            },
        }

        print(f"\n  成功: {summary['num_success']}, 失败: {summary['num_failures']}")
        print(f"\n  Chamfer Distance (↓):")
        print(f"    mean={summary['chamfer_distance']['mean']:.6f} ± {summary['chamfer_distance']['std']:.6f}")
        print(f"    median={summary['chamfer_distance']['median']:.6f}, min={summary['chamfer_distance']['min']:.6f}, max={summary['chamfer_distance']['max']:.6f}")
        print(f"\n  Hausdorff Distance (↓):")
        print(f"    mean={summary['hausdorff_distance']['mean']:.6f} ± {summary['hausdorff_distance']['std']:.6f}")
        print(f"    median={summary['hausdorff_distance']['median']:.6f}, min={summary['hausdorff_distance']['min']:.6f}, max={summary['hausdorff_distance']['max']:.6f}")
        print(f"\n  Normal Consistency (↑):")
        print(f"    mean={summary['normal_consistency']['mean']:.4f} ± {summary['normal_consistency']['std']:.4f}")
        print(f"    median={summary['normal_consistency']['median']:.4f}, min={summary['normal_consistency']['min']:.4f}, max={summary['normal_consistency']['max']:.4f}")

        # 保存
        results_path = os.path.join(opt.output_dir, "per_sample_results.json")
        summary_path = os.path.join(opt.output_dir, "summary.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  详细结果: {results_path}")
        print(f"  汇总: {summary_path}")
        if opt.save_meshes:
            mesh_dir = os.path.join(opt.output_dir, "meshes")
            print(f"  生成 mesh: {mesh_dir}/ ({len(results)} 个 .obj)")

    if failures:
        failures_path = os.path.join(opt.output_dir, "failures.json")
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"  失败记录: {failures_path}")
        for f in failures[:5]:
            print(f"    - {f['sha']}: {f['error']}")


def _build_prompt_from_row(row: dict) -> str:
    """从 metadata 行构造文本 prompt（当 captions 列不存在时使用）。"""
    parts = ["A brake caliper"]
    mapping = {
        "tangential_fixing_interaxis": "fixing interaxis",
        "tangential_pad_inner": "inner pad",
        "tangential_pad_outer": "outer pad",
        "radial_disc_internal_radius": "internal radius",
        "pistons_number": "pistons_num",
        "diameter_pistons_inlet": "inlet diameter",
        "diameter_pistons_central": "central diameter",
        "diameter_pistons_outlet": "outlet diameter",
        "pistons_effective_radius": "effective radius",
        "axial_disc_thickness": "disc thickness",
        "radial_space_external_radius": "external radius",
        "radial_space_internal_radius": "internal radius",
        "radial_space_cut": "radial cut",
        "axial_disc_distance": "disc distance",
        "tangential_space_dimension": "tangential dimension",
        "axial_space_dimension": "axial dimension",
        "radial_space_dimension": "radial dimension",
        "volume": "volume",
    }
    for col, label in mapping.items():
        if col in row and row[col]:
            try:
                val = float(row[col])
                parts.append(f"{label} {val:.2f}")
            except (ValueError, TypeError):
                parts.append(f"{label} {row[col]}")
    return " ".join(parts)


if __name__ == "__main__":
    main()
