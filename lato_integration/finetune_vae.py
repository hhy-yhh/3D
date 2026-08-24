"""
VoxelVAE Decoder Fine-Tuning v2 — occupancy BCE 监督剪枝头 + L0 顶点头

背景:
  旧版 (feats-L1) 只对齐顶点特征、不监督顶点选择，且从未解冻 vtx_head_64，
  导致 pruning_head 输出全高分 → 每级 8× 细分不剪枝 → 20 万顶点 / 1700 万面。

策略 (新版):
  用 decode(training=True) + gt_vertex_voxels_list（从 GT mesh 体素化的薄表面），
  对:
    - vtx_head_64   (L0 顶点选择)         —— BCE(vtx_feats, vertex_mask)
    - pruning_head  (L1/L2 细分剪枝)       —— BCE(occ_probs, prune_labels)
  做 occupancy BCE 监督，让剪枝头学会"每 8 个子体素只留表面上的 1-2 个"。

  gt_vertex_voxels_list 三个分辨率 [128, 256, 512]：
    [0] = 128³ 薄表面（⊆ latent coords，教 L0 区分表面/膨胀区）
    [1] = 256³ 薄表面（L1 剪枝 GT）
    [2] = 512³ 薄表面（L2 剪枝 GT）

用法:
  python lato_integration/finetune_vae.py \
      --lato_ckpt /path/to/LATO/checkpoints/128to512/vae/vae_128to512.pt \
      --lato_config /path/to/LATO/configs/infer_vae_512.yaml \
      --gt_latents /path/to/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/ \
      --gt_meshes /path/to/database_lato/meshes \
      --output_dir outputs/vae_finetuned_bce \
      --epochs 30
"""

import os, sys, argparse, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_TRELLIS_ROOT = os.environ.get("TRELLIS_ROOT", os.path.dirname(os.path.dirname(__file__)))
_LATO_ROOT = os.environ.get("LATO_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "LATO"))
for p in [os.path.abspath(_TRELLIS_ROOT), os.path.abspath(_LATO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

MESH_EXTENSIONS = (".obj", ".ply", ".stl", ".glb")


def load_vae(lato_ckpt: str, lato_config: str, device):
    """加载 VoxelVAE 预训练权重。"""
    from lato.models.lato_vae.lato_vae import VoxelVAE
    from lato_integration.vertex_encoder import ConnectionHead

    with open(lato_config, "r") as f:
        cfg = yaml.safe_load(f)["model"]

    vae = VoxelVAE(
        in_channels=cfg.get("in_channels", 1024),
        latent_dim=cfg["latent_dim"],
        encoder_blocks=cfg["encoder_blocks"],
        decoder_blocks_vtx=cfg["decoder_blocks_vtx"],
        attn_mode="swin", window_size=8, pe_mode="ape",
        using_subdivide=True,
        using_attn=cfg.get("using_attn", False),
    ).to(device)
    head = ConnectionHead(channels=1024, out_channels=1, mlp_ratio=0.75).to(device)

    import importlib.util
    spec = importlib.util.spec_from_file_location("lato_utils", os.path.join(_LATO_ROOT, "utils.py"))
    lato_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lato_utils)
    lato_utils.load_pretrained_woself(lato_ckpt, vae=vae, connection_head=head)

    return vae, head


def freeze_for_ft(vae, num_decoder_blocks: int = 2):
    """冻结 encoder 与大部分 decoder，仅保留:
       - vtx_head_64 / vtx_proj (L0 顶点选择，关键新增)
       - decoder_vtx / decoder_vtx_ca 最后 num_decoder_blocks 块 (含 pruning_head)
    """
    for p in vae.parameters():
        p.requires_grad = False

    # 1) L0 顶点头 + 投影（旧版从未训练，这是关键）
    for m in [vae.vtx_head_64, vae.vtx_proj]:
        for p in m.parameters():
            p.requires_grad = True

    # 2) 各级 subdivision 块（内含 pruning_head）
    for name in ["decoder_vtx", "decoder_vtx_ca"]:
        module = getattr(vae, name, None)
        if module is not None:
            for block in module[-num_decoder_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    trainable = [p for p in vae.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in vae.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")
    return trainable


def find_mesh_file(mesh_dir: str, key: str):
    base = os.path.join(mesh_dir, key)
    for ext in MESH_EXTENSIONS:
        p = base + ext
        if os.path.exists(p):
            return p
    return None


def voxelize_mesh_surface(mesh_path: str, resolution: int):
    """体素化 mesh 薄表面（open3d 三角面 → voxel grid）。返回 [N,3] int64，坐标 ∈ [0, resolution)。"""
    import open3d as o3d
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    if mesh_o3d is None or len(mesh_o3d.vertices) == 0:
        return None
    verts = np.clip(np.asarray(mesh_o3d.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh_o3d,
        voxel_size=1.0 / resolution,
        min_bound=[-0.5, -0.5, -0.5],
        max_bound=[0.5, 0.5, 0.5],
    )
    voxels = np.asarray([v.grid_index for v in vg.get_voxels()], dtype=np.int64)
    return voxels


def build_gt_vertex_voxels_list(mesh_path: str, resolutions, device):
    """返回 [gt_128, gt_256, gt_512]，每个 [N,4] int64 tensor (batch=0 + xyz)。"""
    gt_list = []
    for res in resolutions:
        v = voxelize_mesh_surface(mesh_path, res)
        if v is None or len(v) == 0:
            return None
        c4 = torch.cat([torch.zeros(len(v), 1), torch.from_numpy(v)], dim=1).long().to(device)
        gt_list.append(c4)
    return gt_list


def _flatten(coords_4d: torch.Tensor) -> torch.Tensor:
    coords_4d = coords_4d.long()
    return (
        coords_4d[:, 0] * (1024 ** 3)
        + coords_4d[:, 1] * (1024 ** 2)
        + coords_4d[:, 2] * 1024
        + coords_4d[:, 3]
    )


def bce_with_balance(logits: torch.Tensor, labels: torch.Tensor):
    """BCE with logits，按正负样本比自动加 pos_weight（缓解表面体素占比 ~1/8 的类不平衡）。"""
    num_pos = labels.sum().clamp(min=1.0)
    num_neg = (1.0 - labels).sum().clamp(min=1.0)
    pos_weight = num_neg / num_pos
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def main():
    parser = argparse.ArgumentParser(description="VoxelVAE Decoder Fine-Tuning (occupancy BCE)")
    parser.add_argument("--lato_ckpt", required=True)
    parser.add_argument("--lato_config", required=True)
    parser.add_argument("--gt_latents", required=True,
                        help="GT latent 目录，含 *.npz (coords [N,4] + 16-dim feats)")
    parser.add_argument("--gt_meshes", required=True,
                        help="GT mesh 目录（与 npz 同名，.obj/.ply/.stl/.glb）")
    parser.add_argument("--output_dir", default="outputs/vae_finetuned_bce")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--ft_decoder_blocks", type=int, default=2,
                        help="解冻 decoder_vtx/decoder_vtx_ca 最后 N 块（默认 2 = 全部）")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="输入 feats 加噪 std（0=干净 GT latent；>0 模拟 SLat Flow 误差增强鲁棒）")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.output_dir, exist_ok=True)

    # ── 1. 加载 ──
    print("[1/4] 加载预训练 VoxelVAE ...")
    vae, head = load_vae(opt.lato_ckpt, opt.lato_config, device)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    print("[2/4] 冻结 encoder，解冻 L0 头 + decoder 末 N 块 ...")
    trainable_params = freeze_for_ft(vae, num_decoder_blocks=opt.ft_decoder_blocks)
    optimizer = torch.optim.AdamW(trainable_params, lr=opt.lr, weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    from lato.modules.sparse import SparseTensor as LATOSparseTensor

    # ── 2. 数据 + 预计算 gt_vertex_voxels_list ──
    data_files = sorted([
        os.path.join(opt.gt_latents, f)
        for f in os.listdir(opt.gt_latents)
        if f.endswith(".npz") and f != "stats.json"
    ])
    print(f"  GT latents: {len(data_files)} 条")

    # 分辨率链 128 → 256 → 512（与 decoder_blocks_vtx [128, 256] 的 out_resolution 对应）
    resolutions = [128, 256, 512]

    print("[3/4] 预计算 gt_vertex_voxels_list（每样本体素化 128/256/512 薄表面）...")
    cache = {}  # key -> (npz_path, gt_list_on_cpu)
    valid_keys = []
    for npz_path in tqdm(data_files, desc="voxelize"):
        key = os.path.splitext(os.path.basename(npz_path))[0]
        mesh_path = find_mesh_file(opt.gt_meshes, key)
        if mesh_path is None:
            tqdm.write(f"  [SKIP] 无 mesh: {key}")
            continue
        gt_list_cpu = build_gt_vertex_voxels_list(mesh_path, resolutions, torch.device("cpu"))
        if gt_list_cpu is None:
            tqdm.write(f"  [SKIP] 体素化失败: {key}")
            continue
        cache[key] = (npz_path, gt_list_cpu)
        valid_keys.append(key)
    print(f"  有效样本: {len(valid_keys)} / {len(data_files)}")

    # ── 4. 训练 ──
    print(f"[4/4] Fine-tuning ({opt.epochs} epochs, noise_std={opt.noise_std}) ...")

    for epoch in range(opt.epochs):
        vae.train()
        epoch_loss = 0.0
        n_valid = 0

        for key in tqdm(valid_keys, desc=f"Epoch {epoch+1}"):
            npz_path, gt_list_cpu = cache[key]
            data = np.load(npz_path)
            coords = torch.from_numpy(data["coords"]).to(device)          # [N,4] int
            feats = torch.from_numpy(data["feats"].astype(np.float32)).to(device)  # [N,16]

            # 可选加噪（模拟 SLat Flow 误差；coords 保持干净用于 teacher forcing）
            if opt.noise_std > 0:
                feats = feats + torch.randn_like(feats) * opt.noise_std

            # gt[0]（128³ 薄表面）必须与输入 coords 有交集，否则 teacher forcing 为空
            if torch.isin(_flatten(coords), _flatten(gt_list_cpu[0].to(device))).sum() == 0:
                continue

            gt_list = [g.to(device) for g in gt_list_cpu]
            input_slat = LATOSparseTensor(coords=coords, feats=feats)

            results = vae.decode(input_slat, gt_vertex_voxels_list=gt_list, training=True)

            # L0：vtx_head_64 顶点选择
            l0_logits = results[0]["vtx_feats"].squeeze(-1)      # [N0]
            l0_labels = results[0]["vertex_mask"].float()         # [N0]
            if l0_labels.sum() == 0:
                continue
            loss = bce_with_balance(l0_logits, l0_labels)

            # L1 / L2：各级 pruning_head 剪枝
            for i in range(1, len(results)):
                vr = results[i].get("vertex")
                if vr is None:
                    continue
                occ_logits = vr["occ_probs"].squeeze(-1)          # [N_sub]
                prune_labels = vr["prune_labels"].float()          # [N_sub]
                if prune_labels.sum() == 0:
                    continue
                loss = loss + bce_with_balance(occ_logits, prune_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_valid += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_valid, 1)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % opt.save_every == 0:
            ckpt_path = os.path.join(opt.output_dir, f"vae_ft_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1, "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    final_path = os.path.join(opt.output_dir, "vae_finetuned.pt")
    torch.save({"model_state_dict": vae.state_dict()}, final_path)
    print(f"\n完成: {final_path}")


if __name__ == "__main__":
    main()
