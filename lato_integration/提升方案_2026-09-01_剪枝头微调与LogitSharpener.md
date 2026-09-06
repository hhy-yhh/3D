# 提升方案：方案1 VAE 剪枝头微调 + 方案2 LogitSharpener

> **日期：** 2026-09-01
> **目标：** 改善「面多 / 表面粗糙 / 空隙」三个观感问题
> **前提：** CD 已 0.002（几何正确）——本次改动**不改 CD，只改观感**

---

## 背景：三个现象对应的根因

| 现象 | 主导代码点 | 证据/日志 |
|---|---|---|
| **面多** | VAE 剪枝头失效（对 SLat 带噪 latent 打分全高分 → 每级 8× 细分不剪枝） | `[VAE L2]/[VAE L1]` = **5.59×**（v17 实测，理想 ~1×）|
| **粗糙** | 16³ 结构源头粗（SS Flow resolution=16，PixelShuffle 拉到 128³） | 全样本均匀糙；壳 2.5×GT（SS th=2.0 产出 97,969 vs GT 38,577）|
| **空隙** | `--max_coords` 截断（active 97,969 → 只留 30,000，**丢 69%**，且 30K < GT 38.5K 欠覆盖）| `[SS] coords truncated` |

---

## 方案 1：VAE 剪枝头微调（`finetune_vae.py`，代码已就绪）

### 原理

VoxelVAE 的剪枝头（`decoder_vtx[i].upsample.pruning_head`）+ L0 顶点头（`vtx_head_64`）训练时吃 encoder 精确 latent，推理时吃 SLat 带噪 latent（MSE 0.2）→ 分布漂移 → 剪枝头输出全高分 → 每级 8× 细分不剪枝 → 20 万顶点 / 1700 万面。

`finetune_vae.py` 用 `decode(training=True)` + GT mesh 薄表面体素化（128/256/512 三级），对 3 个决策头做 occupancy BCE（只解冻 **2.7% 参数**），让剪枝头学会"每 8 个子体素只留表面上的 1-2 个"。

### Step A — 微调（~3h，GPU 空卡）

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python lato_integration/finetune_vae.py \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --gt_latents /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/ \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/vae_finetuned_bce \
    --max_coords 8000 \
    --epochs 30
```

**要点：**
- 启动应打印 `Trainable: ~4,470,787 / 165,884,963 (2.7%)`（确认只解冻 3 个决策头）
- OOM → `--max_coords 5000`
- 每 10 epoch 存 `vae_ft_epoch10/20.pt`，可提前验证；续训用 `--resume vae_ft_epoch20.pt`
- 最终产物：`outputs/vae_finetuned_bce/vae_finetuned.pt`

### Step B — 用微调权重重新评估（关键验证）

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SS_CKPT=$(ls outputs/lato_ss_flow_v6/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v10/ckpts/denoiser_step*.pt | sort -V | tail -1)

# 先单条
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" --slat_ckpt "$SLAT_CKPT" \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/stats.json \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --vae_ft_ckpt outputs/vae_finetuned_bce/vae_finetuned.pt \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/eval_ft_single \
    --ss_threshold 2.0 --max_coords 30000 --mesh_mode poisson --save_meshes --limit 1
```

全量 21 条：去掉 `--limit 1`，换 `--output_dir outputs/eval_ft_full`。

### 成功判据

| 指标 | 微调前 | 微调后（预期）|
|---|---|---|
| L2/L1 顶点比值 | 5.59× | **~1×** |
| 顶点数 | ~200,000 | **30,000~50,000** |
| 面数 | ~17,600,000 | 大幅下降 |
| CD / HD / NC | — | 保持或微变（本次目标不是 CD）|

> ⚠️ v17 当时评估时 CD 度量是坏的，本次是在**修复后的度量**下第一次有效验证。

---

## 方案 2：LogitSharpener（待实现）

### 原理（v14 方案 F）

StructureHead 输出的 occupancy logits **空间位置正确**（th>3 时 bbox 范围合理），但**边界模糊**（logits 被"摊大饼"，max ~9 而非锐利边界的 >50），导致壳 2.5×GT。用一个 **~150 参数**轻量 3D CNN 学"边缘锐化"（本质 3D unsharp mask：`y = x + net(x)`），让壳变薄、边界变锐。

**不动主模型**（SS/SLat Flow、StructureHead、VoxelVAE 全部冻结）。

### 文件清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `lato_integration/logit_sharpener.py` | `LogitSharpener` 模块 + `load_sharpener()` | 新建 |
| `lato_integration/train_logit_sharpener.py` | Step A 数据生成 + Step B 训练 | 新建 |
| `lato_integration/evaluate_3d_metrics.py` | 加 `--sharpener_ckpt` + 一行应用 | 修改 |

### LogitSharpener 模块设计（~125 参数）

```python
# logit_sharpener.py
"""LogitSharpener — 残差 3D CNN 锐化 occupancy logits（~125 参数）"""
import torch
import torch.nn as nn

class LogitSharpener(nn.Module):
    """y = x + net(x)：残差结构，只在需要的地方调整 logits。
    输入 [B,1,128,128,128] occupancy logits → 输出同形状锐化 logits。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=1),                    # 1→4 通道
            nn.SiLU(),
            nn.Conv3d(4, 4, kernel_size=3, padding=1, groups=4),  # depthwise 3×3×3
            nn.SiLU(),
            nn.Conv3d(4, 1, kernel_size=1),                    # 4→1
        )

    def forward(self, x):
        return x + self.net(x)


def load_sharpener(path, device):
    model = LogitSharpener().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
```

### train_logit_sharpener.py 设计

**Step A — 数据生成（~1h）**
- 加载 SS Flow + StructureHead（复用 `evaluate_3d_metrics.py::load_pipeline` 的加载逻辑，只需 ss 部分）
- 逐条跑 234 训练集 prompt（`database_lato/metadata.csv`，prompt 取法与评估脚本一致：`captions` 列 / `_build_prompt_from_row`）
- 保存每样本 `occ_logits [1,128,128,128]` 到 `output_dir/data/{key}.npz`
- GT occupancy 直接用现有 `ss_occupancy_128_v2/{key}.npz` 的 `occupancy`（float32 [1,128,128,128]，无需重新体素化）

**Step B — 训练（~10-20min）**
- `loss = BCE_with_logits(sharpener(logits), gt_occ)`，`pos_weight = n_neg/n_pos`（正样本仅 1.8%，类不平衡 ~1:55）
- AdamW lr=1e-3，~10-20 epoch
- 保存 `output_dir/sharpener.pt`

### evaluate_3d_metrics.py 集成（改动极小）

```python
# argparse 加：
#   parser.add_argument("--sharpener_ckpt", type=str, default=None,
#                       help="LogitSharpener 权重，在 StructureHead 后锐化 occupancy logits")

# main 循环里，head(z_s) 之后、取 coords 之前插入：
occ_logits = head(z_s)
if sharpener is not None:
    occ_logits = sharpener(occ_logits)   # 锐化后再取阈值
```

（`sharpener` 在 `--sharpener_ckpt` 存在时用 `load_sharpener()` 加载，否则为 None；SS Flow 是 fp32，logits 直接喂 sharpener 无需转换。）

### 运行命令

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SS_CKPT=$(ls outputs/lato_ss_flow_v6/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v10/ckpts/denoiser_step*.pt | sort -V | tail -1)

# 1) 生成数据 + 训练
python lato_integration/train_logit_sharpener.py \
    --ss_ckpt "$SS_CKPT" \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/logit_sharpener \
    --ss_threshold 2.0

# 2) 用 sharpener 评估测试集
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" --slat_ckpt "$SLAT_CKPT" \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/stats.json \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --sharpener_ckpt outputs/logit_sharpener/sharpener.pt \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/eval_sharp_single \
    --ss_threshold 2.0 --max_coords 30000 --mesh_mode poisson --save_meshes --limit 1
```

### 成功判据

| 指标 | 方案2 目标 |
|---|---|
| `[SS diag] active(>2.0)` | 从 ~97,969 显著下降（壳变薄，接近 GT 38,577）|
| 空隙 | 明显减少（截断前壳就薄，被丢的"低置信度真区域"变少）|
| CD / HD / NC | 保持或微变 |

---

## 执行顺序建议

```
① 方案1 Step A（~3h）── 可与方案2 并行（不同 GPU / 方案2 数据生成占 SS，注意不要挤同一张卡）
   │
   ├─ 方案1 Step B 评估 → 看 L2/L1 是否 ~1×、面数是否降 80%
   │
② 方案2 实现代码 → 同步服务器 → 数据生成 + 训练（~1h）
   │
   └─ 方案2 评估 → 看 active 数量是否下降、空隙是否减少
```

**历史教训（避免重蹈）：**
- v16 VoxelVAE 末层微调 → 无效；v18 LATO 精化 → 无效。**加功能要打中根因**，本次两个方案分别打"面多"和"空隙/壳厚"的根因。
- 方案2 如果 `active(>2.0)` 没降 → 说明 sharpener 没学到锐化，先看训练 loss 是否收敛、数据生成 logits 分布是否正常。
