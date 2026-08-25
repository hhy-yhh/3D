"""
experiment_coords_vs_feats.py — 隔离 L0=0 的根因：coords 还是 feats？

背景:
  VAE decode L0=0（顶点选择头不触发），导致 L1/L2 暴力细分 → 33 万顶点。
  本脚本用一条训练集样本做 4 组对照，定位是「coords 太糊」还是「feats 带噪」:

    A: GT coords + GT feats    → VAE  (基线，应触发 L0)
    B: GT coords + SLat feats  → VAE  (只换 feats：SLat Flow 在 GT coords 上采样)
    C: SS coords + GT feats    → VAE  (只换 coords：GT feats 经 NN 映射到 SS coords)
    D: SS coords + SLat feats  → VAE  (当前状态，L0=0)

  判断:
    A 应 L0>0（基线）。若 A 也 L0=0 → VAE 加载/输入格式有问题，先修这个。
    B 触发、D 不触发  → coords 是主因（feats 在正确拓扑上没问题）
    B 不触发          → feats 是主因（正确拓扑也救不回 L0）
    C 触发、D 不触发  → 进一步确认 feats 是主因
    C 不触发          → coords 单独就能破坏 L0

用法 (在 TRELLIS 目录下):
  python lato_integration/experiment_coords_vs_feats.py \
      --ss_ckpt "$SS_CKPT" --slat_ckpt "$SLAT_CKPT" \
      --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/stats.json \
      --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
      --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
      --vae_ft_ckpt outputs/vae_finetuned_bce/vae_finetuned.pt \
      --metadata /data/huanghaoyang/3D/database_lato/metadata.csv \
      --gt_latents /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128 \
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
from scipy.spatial import KDTree

# ── 路径设置（与 evaluate_3d_metrics.py 保持一致）──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRELLIS_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_LATO_ROOT = os.environ.get("LATO_ROOT", os.path.join(_TRELLIS_ROOT, "..", "LATO"))
for _p in [_TRELLIS_ROOT, _LATO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 复用 eval 的模型加载（含归一化修复 + 微调 VAE 覆盖）
from lato_integration.evaluate_3d_metrics import load_pipeline, _build_prompt_from_row


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


def feed_vae(vae, coords_4d, feats, device, label, threshold=0.2):
    """喂一组 (coords, feats) 给 VAE，打印 L0/L1/L2。"""
    from lato.modules.sparse import SparseTensor as LATOSparseTensor

    torch.cuda.empty_cache()  # 每组实验前清缓存，防累积
    lato_slat = LATOSparseTensor(
        feats=feats.contiguous().float().to(device),
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
    print("  ".join(parts))
    return decoded


def main():
    ap = argparse.ArgumentParser(description="隔离 L0=0 根因：coords vs feats")
    ap.add_argument("--ss_ckpt", required=True)
    ap.add_argument("--slat_ckpt", required=True)
    ap.add_argument("--slat_stats", default=None)
    ap.add_argument("--lato_ckpt", required=True)
    ap.add_argument("--lato_config", required=True)
    ap.add_argument("--vae_ft_ckpt", default=None)
    ap.add_argument("--trellis_pretrained", default="microsoft/TRELLIS-text-base")
    ap.add_argument("--metadata", required=True, help="训练集 metadata.csv")
    ap.add_argument("--gt_latents", required=True, help="GT latent .npz 目录")
    ap.add_argument("--sample_idx", type=int, default=0, help="用第几条训练样本")
    ap.add_argument("--prompt", default=None, help="覆盖 prompt（可选）")
    ap.add_argument("--ss_threshold", type=float, default=2.0)
    ap.add_argument("--max_coords", type=int, default=16384)
    ap.add_argument("--ss_steps", type=int, default=20)
    ap.add_argument("--slat_steps", type=int, default=20)
    ap.add_argument("--cfg_strength", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lato_threshold", type=float, default=0.2)
    ap.add_argument("--use_fp16", action="store_true", default=False)
    opt = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 选样本 + 构造 prompt ──
    with open(opt.metadata, encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    sample = samples[opt.sample_idx]
    sha256 = sample.get("sha256")  # .npz 文件名 = sha256（不是 file_identifier）
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
    # 🔧 GT latent 可能远超 spconv 的 int32 上限（本样本 52416 voxels），
    #    截断到 max_coords（与 SS coords 一致），否则 spconv 报 int32 overflow
    if opt.max_coords > 0 and coords_gt.shape[0] > opt.max_coords:
        coords_gt = coords_gt[:opt.max_coords]
        feats_gt = feats_gt[:opt.max_coords]
        print(f"[TRUNC] GT latent 截断到 {opt.max_coords} voxels（防 spconv int32 溢出）")
    print(f"GT latent: coords={tuple(coords_gt.shape)} feats={tuple(feats_gt.shape)} "
          f"feats_mean={feats_gt.mean():.4f} feats_std={feats_gt.std():.4f}\n")

    # ── 3. 加载模型管线 ──
    print("加载管线 ...")
    pipeline, connection_head, model_cfg = load_pipeline(opt, device)
    del connection_head, model_cfg  # 实验不需要边预测头，立即释放
    vae = pipeline.models["lato_vae"]
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

    # 🔧 释放 SS Flow + StructureHead（后续只用 coords），给 VAE decode 腾显存
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

    # 🔧 释放 SLat Flow + CLIP 文本模型（采样已完成，只留 VAE decode）
    pipeline.models.pop("slat_flow_model", None)
    pipeline.text_cond_model = None
    del cond
    torch.cuda.empty_cache()

    # ── 6. C 实验: GT feats 经最近邻映射到 SS coords ──
    tree = KDTree(coords_gt.numpy().astype(np.float64))
    _, nn_idx = tree.query(coords_ss[:, 1:].cpu().numpy().astype(np.float64))
    feats_gt_mapped = feats_gt[nn_idx]  # [N_ss, 16]

    # ── 7. 4 组对照实验 ──
    print("\n" + "=" * 64)
    print(f"VAE decode 对照实验 (inference_threshold={opt.lato_threshold:.2f})")
    print("=" * 64)
    feed_vae(vae, coords_gt_4d, feats_gt, device, "A: GT coords + GT feats   ", opt.lato_threshold)
    feed_vae(vae, coords_gt_4d, slat_feats_gt, device, "B: GT coords + SLat feats", opt.lato_threshold)
    feed_vae(vae, coords_ss_4d, feats_gt_mapped, device, "C: SS coords + GT feats   ", opt.lato_threshold)
    feed_vae(vae, coords_ss_4d, slat_feats_ss, device, "D: SS coords + SLat feats ", opt.lato_threshold)
    print("=" * 64)

    # ── 8. 判读提示 ──
    print("\n判读:")
    print("  A 应 L0>0（基线）；若 A 也 L0=0 → VAE 加载/输入格式有问题，先修这个")
    print("  B 触发、D 不触发 → coords 是主因")
    print("  B 不触发          → feats 是主因")
    print("  C 触发、D 不触发 → 进一步确认 feats 是主因")


if __name__ == "__main__":
    main()
