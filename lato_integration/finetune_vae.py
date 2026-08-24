"""
VoxelVAE Decoder Fine-Tuning — 适配 SLat Flow 的 16-dim 语义特征

背景:
  VoxelVAE decoder 训练时对接自己 encoder 产出的精确 latent，
  推理时收到 SLat Flow 近似预测的 latent → L0 不触发。

策略:
  使用 GT latent（从 VoxelVAE.encode 产生），加噪声模拟 SLat Flow 误差，
  训练 decoder 在噪声输入下仍产出与干净输入一致的顶点特征（按顶点坐标对齐）。
  注意: 用 decode(training=False) 阈值选择顶点, 无需 gt_vertex_voxels_list;
       loss 作用在可微分的 feats 上（整数 coords 无梯度, 不能直接 L1）。

用法:
  python lato_integration/finetune_vae.py \
      --lato_ckpt /path/to/LATO/.../vae_128to512.pt \
      --lato_config /path/to/LATO/configs/infer_vae_512.yaml \
      --gt_latents /path/to/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/ \
      --output_dir outputs/vae_finetuned \
      --epochs 50
"""

import os, sys, json, argparse, yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

_TRELLIS_ROOT = os.environ.get("TRELLIS_ROOT", os.path.dirname(os.path.dirname(__file__)))
_LATO_ROOT = os.environ.get("LATO_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "LATO"))
for p in [os.path.abspath(_TRELLIS_ROOT), os.path.abspath(_LATO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


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


def freeze_for_ft(vae, num_layers_to_train: int = 2):
    """冻结 encoder，仅保留 decoder_vtx + decoder_vtx_ca 最后 N 层可训练。"""
    for p in vae.parameters():
        p.requires_grad = False

    for name in ["decoder_vtx", "decoder_vtx_ca"]:
        module = getattr(vae, name, None)
        if module is not None:
            for block in module[-num_layers_to_train:]:
                for p in block.parameters():
                    p.requires_grad = True

    trainable = [p for p in vae.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in vae.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")
    return trainable


def _coord_keys(coords: torch.Tensor) -> torch.Tensor:
    """[N, 3] int 坐标 -> [N] int64 唯一键 (x,y,z 均 < 1024)。"""
    return (coords[:, 0].long() * 1024 + coords[:, 1].long()) * 1024 + coords[:, 2].long()


def _coord_match(clean_coords: torch.Tensor, noisy_coords: torch.Tensor):
    """返回 (ic, inz) 行索引列表：clean 与 noisy 共有的顶点坐标。"""
    if clean_coords.numel() == 0 or noisy_coords.numel() == 0:
        return [], []
    kc = _coord_keys(clean_coords.cpu()).tolist()
    kn = _coord_keys(noisy_coords.cpu()).tolist()
    idx = {k: i for i, k in enumerate(kc)}
    ic, inz = [], []
    for j, k in enumerate(kn):
        if k in idx:
            ic.append(idx[k])
            inz.append(j)
    return ic, inz


def main():
    parser = argparse.ArgumentParser(description="VoxelVAE Decoder Fine-Tuning")
    parser.add_argument("--lato_ckpt", required=True)
    parser.add_argument("--lato_config", required=True)
    parser.add_argument("--gt_latents", required=True,
                        help="GT latent 目录，含 *.npz (coords + 16-dim feats)")
    parser.add_argument("--output_dir", default="outputs/vae_finetuned")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--ft_layers", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=0.15,
                        help="噪声标准差（模拟 SLat Flow MSE~0.22 的误差水平）")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.output_dir, exist_ok=True)

    # ── 加载 ──
    print("[1/3] 加载预训练 VoxelVAE...")
    vae, head = load_vae(opt.lato_ckpt, opt.lato_config, device)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    print("[2/3] 冻结 encoder，解冻 decoder 最后 N 层...")
    trainable_params = freeze_for_ft(vae, num_layers_to_train=opt.ft_layers)

    optimizer = torch.optim.AdamW(trainable_params, lr=opt.lr, weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    # ── 数据 ──
    data_files = sorted([
        os.path.join(opt.gt_latents, f)
        for f in os.listdir(opt.gt_latents)
        if f.endswith(".npz") and f != "stats.json"
    ])
    print(f"  GT latents: {len(data_files)} 条")

    from lato.modules.sparse import SparseTensor as LATOSparseTensor

    print(f"[3/3] Fine-tuning ({opt.epochs} epochs, noise_std={opt.noise_std})...")

    for epoch in range(opt.epochs):
        vae.train()
        epoch_loss = 0.0
        n_valid = 0

        for npz_path in tqdm(data_files, desc=f"Epoch {epoch+1}"):
            data = np.load(npz_path)
            coords = torch.from_numpy(data["coords"]).to(device)
            feats = torch.from_numpy(data["feats"].astype(np.float32)).to(device)

            # 干净输入 → 干净输出（教师, detach）
            clean_slat = LATOSparseTensor(coords=coords, feats=feats)
            with torch.no_grad():
                clean_out = vae.decode(clean_slat, training=False)
                if isinstance(clean_out, list):
                    clean_v = clean_out[-1].get("vertex", {})
                    clean_coords = clean_v.get("coords")
                    clean_feats = clean_v.get("feats")
                else:
                    continue

            if clean_coords is None or clean_coords.numel() == 0 or clean_feats is None:
                continue

            # 加噪输入 → 解码（梯度流经 decoder 末 N 层）
            # 注意: 这里 training=False 仅控制顶点选择逻辑, 不阻断梯度;
            #       decoder_vtx / decoder_vtx_ca 的可训练权重仍会收到梯度。
            noise = torch.randn_like(feats) * opt.noise_std
            noisy_slat = LATOSparseTensor(coords=coords, feats=feats + noise)
            noisy_out = vae.decode(noisy_slat, training=False)
            if isinstance(noisy_out, list):
                noisy_v = noisy_out[-1].get("vertex", {})
                noisy_coords = noisy_v.get("coords")
                noisy_feats = noisy_v.get("feats")
            else:
                continue

            if noisy_coords is None or noisy_coords.numel() == 0 or noisy_feats is None:
                continue

            # 一致性损失: 按顶点坐标对齐后, 让带噪输入的顶点特征逼近干净输入。
            # (顶点 coords 是整数索引, 无梯度; 可微分的是 feats)
            ic, inz = _coord_match(clean_coords, noisy_coords)
            if len(ic) > 0:
                ic = torch.tensor(ic, device=device)
                inz = torch.tensor(inz, device=device)
                loss = nn.functional.l1_loss(noisy_feats[inz], clean_feats[ic].detach())
            else:
                # 退化情形: 无共有顶点时用质心一致性兜底, 避免梯度饥饿
                loss = nn.functional.l1_loss(noisy_feats.mean(0), clean_feats.detach().mean(0))

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
