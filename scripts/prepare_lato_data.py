"""
步骤2：用 LATO 预训练 VoxelVAE 提取 latent，作为新 SLat Flow 的训练目标。

用法:
  python scripts/prepare_lato_data.py \
      --lato_ckpt /path/to/lato_checkpoint.pt \
      --lato_config /path/to/infer_vae_512.yaml \
      --data_dir /data/huanghaoyang/3D/database_lato/train \
      --output_dir /data/huanghaoyang/3D/TRELLIS/data/lato_latents \
      --device cuda
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# === LATO 路径 ===
LATO_ROOT = os.environ.get("LATO_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "3D", "LATO"))
sys.path.insert(0, LATO_ROOT)

from lato.models.lato_vae.lato_vae import VoxelVAE
from lato.modules.sparse.basic import SparseTensor
from vertex_encoder import VoxelFeatureEncoder_active_pointnet, ConnectionHead
from utils import load_pretrained_woself
import yaml


def load_lato(config_path, checkpoint_path, device):
    """加载 LATO VoxelVAE + VoxelFeatureEncoder"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["model"]

    vae = VoxelVAE(
        in_channels=cfg["in_channels"],
        latent_dim=cfg["latent_dim"],
        encoder_blocks=cfg["encoder_blocks"],
        decoder_blocks_vtx=cfg["decoder_blocks_vtx"],
        num_heads=8, num_head_channels=64, mlp_ratio=4.0,
        attn_mode="swin", window_size=8, pe_mode="ape",
        use_fp16=False, use_checkpoint=False, qk_rms_norm=False,
        using_subdivide=True,
        using_attn=cfg.get("using_attn", False),
    ).to(device)

    voxel_encoder = VoxelFeatureEncoder_active_pointnet(
        in_channels=15, hidden_dim=256, out_channels=1024,
        scatter_type="mean", n_blocks=5, resolution=128,
    ).to(device)

    connection_head = ConnectionHead(channels=512 * 2, out_channels=1, mlp_ratio=0.75).to(device)

    load_pretrained_woself(checkpoint_path, vae=vae, voxel_encoder=voxel_encoder,
                           connection_head=connection_head)
    vae.eval()
    voxel_encoder.eval()
    print("✅ LATO 模型加载完成")
    return vae, voxel_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lato_ckpt", type=str, required=True, help="LATO checkpoint .pt")
    parser.add_argument("--lato_config", type=str, default=os.path.join(LATO_ROOT, "configs/infer_vae_512.yaml"))
    parser.add_argument("--data_dir", type=str, required=True, help="TRELLIS 数据集目录（含 metadata.csv）")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_models", type=int, default=-1, help="-1 表示全部")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "lato_latents"), exist_ok=True)
    device = torch.device(args.device)

    # 1. 加载 LATO 模型
    vae, voxel_encoder = load_lato(args.lato_config, args.lato_ckpt, device)

    # 2. 读取 metadata
    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        # 尝试子目录
        metadata_path = os.path.join(args.data_dir, "train", "metadata.csv")
    df = pd.read_csv(metadata_path)
    if args.max_models > 0:
        df = df.head(args.max_models)
    print(f"共 {len(df)} 个模型待处理")

    # 3. 逐个提取 LATO latent
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取 LATO latent"):
        model_id = row.get("id", row.get("sha256", row.get("ID", str(idx))))

        # 解析 caption
        caption_raw = row.get("caption", row.get("captions", ""))
        if isinstance(caption_raw, str) and caption_raw.startswith("["):
            try:
                captions_list = json.loads(caption_raw)
                caption = np.random.choice(captions_list) if captions_list else ""
            except json.JSONDecodeError:
                caption = caption_raw
        else:
            caption = str(caption_raw)

        # 从数据集加载点云和 active voxels
        # 这里需要根据你的实际数据格式调整
        # 常用路径约定：
        #   {data_dir}/point_clouds/{model_id}.npy   — 点云
        #   {data_dir}/active_voxels/{model_id}.npy   — active voxel coords @ res 128
        pc_path = os.path.join(args.data_dir, "point_clouds", f"{model_id}.npy")
        coords_path = os.path.join(args.data_dir, "active_voxels", f"{model_id}.npy")

        try:
            point_cloud = torch.from_numpy(np.load(pc_path)).float().to(device)
            active_coords = torch.from_numpy(np.load(coords_path)).long().to(device)
        except FileNotFoundError:
            # 如果没有预提取的点云文件，跳过或从 mesh 实时计算
            print(f"  ⚠ 跳过 {model_id}：缺少点云/体素文件")
            continue

        # LATO encode
        with torch.no_grad():
            active_feats = voxel_encoder(
                p=point_cloud,
                sparse_coords=active_coords,
                res=128,
                bbox_size=(-0.5, 0.5),
            )
            sparse_input = SparseTensor(feats=active_feats, coords=active_coords.int())
            lato_latent, posterior = vae.encode(sparse_input)

        # 保存
        latent_path = os.path.join(args.output_dir, "lato_latents", f"{model_id}.pt")
        torch.save({
            "coords": lato_latent.coords.cpu(),        # [N, 4]
            "latent_feats": lato_latent.feats.cpu(),    # [N, 16]
            "caption": caption,
        }, latent_path)

        records.append({
            "id": model_id,
            "caption": caption,
            "latent_path": f"lato_latents/{model_id}.pt",
            "num_voxels": lato_latent.feats.shape[0],
        })

    # 4. 保存索引
    out_meta = pd.DataFrame(records)
    out_meta.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)
    print(f"\n✅ 完成！{len(records)} 条训练数据 → {args.output_dir}")


if __name__ == "__main__":
    main()
