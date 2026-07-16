# TRELLIS + LATO 文本转3D — 完整实现流程 v3（添加 SS Flow 训练）

> **目标：** TRELLIS 文本转3D 管线中，将 Sparse VAE Encoder/Decoder 替换为 LATO 的 VoxelVAE，**SS Flow 和 SLat Flow 均在刹车卡钳数据集上从零训练**。

**更新记录：**
- 2026-07-16：**v3 更新** — 新增 SS Flow 训练步骤（步骤2b + 步骤4a），训练模型数从 1 个变为 2 个
- 2026-07-14（下午）：修复 4 个训练启动 bug，验证单卡 RTX 4090 24GB 可运行配置
- 2026-07-14（上午）：修复 12 个 bug，新增多卡训练方案，调整默认模型参数

---

## 架构概览

```
原版 TRELLIS:
  Text → CLIP → SS Flow → SS Decoder → SLat Flow → SLat Decoder → Mesh/GS/RF

LATO 替换后:
  Text → CLIP → SS Flow(刹车卡钳训练) → SS Decoder(原版冻结) → coords[:,1:]×2
                 ↑ 新训练                        ↑ 不变              ↓
                                          SLat Flow(16-dim/128-res, 新训练)
                                                 ↓
                                          LATO VoxelVAE.decode()
                                                 ↓
                                          ConnectionHead → Mesh
```

**关键改动（v3）：**
- SS Flow：**在刹车卡钳数据上从零训练**（替代原版 TRELLIS 预训练权重），架构仍是 `SparseStructureFlowModel`
- SS Decoder：**完全不动**（冻结，使用 TRELLIS 预训练权重）
- SLat Flow：架构仍是 TRELLIS `SLatFlowModel`，参数改为 `resolution=128, in/out_channels=16`（适配 LATO latent 空间），**在刹车卡钳数据上从零训练**
- Decoder：**替换**为 LATO VoxelVAE + ConnectionHead（冻结，使用 LATO 预训练权重）
- 需训练 **2 个** 新模型（SS Flow + SLat Flow，各约 1M 步）
- 两者**可并行训练**（输入输出独立，SLat Flow 训练时用的 conditioning 是 ground truth SS latent，不依赖 SS Flow 的预测）

---

## 前置条件

### 硬件要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU | 1× 24GB (RTX 4090) | 4× RTX 4090 |
| 显存 | 20GB+ | 24GB |
| CUDA | 11.8+ | 12.5 |

> **注意：** 使用 `model_channels=384` 时单卡 24GB 可稳定运行。`model_channels=512` 需要多卡或更大的显存。`model_channels=768` 单卡必定 OOM。

### 目录结构

```
服务器: /data/huanghaoyang/3D/
├── TRELLIS/                        # TRELLIS 代码（含已修改的文件）
│   ├── trellis/
│   │   ├── models/
│   │   │   ├── lato_slat_flow.py           # ✅ 新增
│   │   │   └── __init__.py                 # ✅ 已修改
│   │   ├── pipelines/
│   │   │   └── trellis_text_to_3d.py       # ✅ 已修改
│   │   ├── trainers/
│   │   │   └── base.py                     # ✅ 已修改（跳过 init snapshot）
│   │   └── datasets/
│   │       └── structured_latent.py         # ✅ 已修改
│   ├── lato_integration/                   # 集成脚本
│   │   ├── run_train.py                    # ✅ 已修改
│   │   ├── flow/trainers/slat_flow_trainer.py  # ✅ 已修改
│   │   ├── flow/trainers/ss_flow_trainer.py    # ✅ 已修改
│   │   ├── encode_lato_latent_v2.py        # ✅ 已修改
│   │   └── inference_lato.py               # ✅ 已修改
│   ├── configs/generation/
│   │   └── lato_slat_flow.json             # ✅ 训练配置
│   └── dataset_toolkits/                   # 数据工具
│
├── LATO/                           # LATO 官方代码 + 预训练权重
│   ├── lato/
│   ├── configs/infer_vae_512.yaml
│   ├── vertex_encoder.py
│   ├── utils.py
│   └── checkpoints/128to512/vae/
│       └── vae_128to512.pt         # LATO 预训练 checkpoint
│
└── database_lato/                  # 训练数据
    ├── metadata.csv
    └── meshes/                     # 3D 模型文件 (.obj/.ply/.stl/.glb)
```

### 你需要有

- ✅ TRELLIS 代码（已修改 5 个文件）
- ✅ LATO 预训练权重 `vae_128to512.pt`（从 HuggingFace 下载）
- ✅ 训练数据集（3D 模型 `meshes/` + `metadata.csv`）
- ✅ GPU（建议 24GB+ 显存，多卡更佳）
- ✅ Conda 环境 `trellis_official`（torch 2.4.0 + CUDA 11.8）

### 下载 LATO Checkpoint

```bash
pip install huggingface_hub
hf download udbbdh/LATO checkpoints/128to512/vae/vae_128to512.pt --local-dir /data/huanghaoyang/3D/LATO
```

---

## 步骤总览

| 步骤 | 内容 | 脚本 | 预计时间 |
|------|------|------|----------|
| 1 | 确认代码修改 | 5 个文件 | 已就绪 |
| 2a | 提取 LATO latent | `encode_lato_latent_v2.py` | 取决于数据集大小 |
| 2b | **🆕 提取 SS latent** | `encode_ss_latent.py`（新增） | ~1 小时 |
| 3a | 计算 SLat normalization 统计量 | `stat_latent.py` | 5 分钟 |
| 3b | **🆕 计算 SS normalization 统计量** | `stat_latent.py` | 5 分钟 |
| 4a | **🆕 训练 SS Flow** | `run_train.py` | ~1M 步（1-5 天） |
| 4b | 训练 SLat Flow | `run_train.py` | ~1M 步（1-5 天） |
| 5 | 推理 | `inference_lato.py` | ~1 分钟/样本 |

---

## 步骤1：确认已修改的代码（5个文件）

### 1a. `trellis/models/lato_slat_flow.py` ✅ 已完成

```python
from .structured_latent_flow import SLatFlowModel

class LATOSLatFlowModel(SLatFlowModel):
    """LATO-compatible SLat Flow: resolution 64→128, channels 8→16."""
    def __init__(self, resolution=128, in_channels=16, out_channels=16, **kwargs):
        super().__init__(resolution=resolution, in_channels=in_channels,
                         out_channels=out_channels, **kwargs)
```

### 1b. `trellis/pipelines/trellis_text_to_3d.py` ✅ 已完成

四处关键修改：

**(1) `run()` 中 SS 坐标缩放（只乘空间维，不乘 batch 维）：**
```python
coords[:, 1:] = coords[:, 1:] * 2   # res 64 → 128，适配 LATO latent
```

**(2) 新增 `decode_slat_lato()` 方法：**
- 自动将 TRELLIS SparseTensor 转换为 LATO SparseTensor
- 调用 `lato_vae.decode()` 生成 vertex 层级
- `inference_threshold` 可从 pipeline 属性读取（默认 0.2）

**(3) `decode_slat()` 扩展：**
- 检测到 `lato_vae` 时自动走 LATO 路径

### 1c. `trellis/models/__init__.py` ✅ 已完成

已注册 `LATOSLatFlowModel`。

### 1d. `trellis/datasets/structured_latent.py` ✅ 已修改

三处修改：

**(1) `_loading_slat_dec()`：** 当 `pretrained_slat_dec=None` 且 `slat_dec_path=None` 时跳过加载（LATO 训练无兼容 decoder）

**(2) `visualize_sample()`：** decoder 不可用时返回空 dict，跳过可视化

**(3) `get_instance()`：** 自动剥离 LATO coords 的多余 batch 列 `[N,4]→[N,3]`

### 1e. `trellis/trainers/base.py` ✅ 已修改

**跳过 init/resume snapshot**（避免训练启动时因 snapshot 采样导致 OOM 或多卡 DDP hang）：
```python
if self.step == 0:
    # Skip init snapshot (random weights, not useful; avoids OOM/multi-GPU hang)
    pass
else:  # resume
    # Skip resume snapshot to avoid OOM
    pass
```

> 训练过程中的 snapshot（每 `i_sample` 步）不受影响，仍会正常保存采样结果。

---

## 步骤2：用 LATO VAE 提取训练数据

### 原理

```
3D Mesh → LATO 官方预处理(15ch) → VoxelFeatureEncoder → VoxelVAE.encode()
                    ↓                                              ↓
            pos+normal+VDF                                  .npz (coords[N,3] + feats[N,16])
```

> **注意：** v2 编码脚本保存 coords 为 `[N,4]`（LATO 格式 batch+xyz），步骤 1d 的 dataset 修复自动剥离为 `[N,3]`。无需重新编码。

### 运行命令

```bash
cd /data/huanghaoyang/3D/TRELLIS

python lato_integration/encode_lato_latent_v2.py \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --resolution 128 \
    --num_points 65536 \
    --device cuda
```

### 参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--lato_ckpt` | LATO 预训练 checkpoint | 必填 |
| `--lato_config` | LATO VAE 配置 yaml | 必填 |
| `--data_dir` | 包含 `metadata.csv` 和 `meshes/` 的数据目录 | 必填 |
| `--output_dir` | 输出目录（存放 `.npz` latent 文件） | 必填 |
| `--resolution` | 体素化分辨率 | 128（默认） |
| `--num_points` | 点云采样点数 | 65536（显存不足可降到 32768） |
| `--dry_run` | 仅打印不处理（先验证数据路径） | 可选 |

### 输出

```
lato_latents/
├── metadata.csv                              # 更新后的元数据（含 latent_lato_vae_16dim_128 列）
└── latents/
    └── lato_vae_16dim_128/
        ├── {sha256_1}.npz                    # coords[N,4] + feats[N,16]
        ├── {sha256_2}.npz
        └── ...
```

### v2 vs v1 差异

| | v1（废弃） | v2（当前） |
|---|---|---|
| 点云通道 | 3（只有位置） | **15**（pos+normal+VDF） |
| 模型结构 | hidden=128, blocks=3 | **hidden=256, blocks=5**（匹配官方） |
| 预处理 | 手写简化版 | **LATO 官方函数** |
| autocast | fp16 | **bf16**（与官方一致） |
| torch_scatter mock | 需要 | **不需要** |

---

## 步骤2b：🆕 用 TRELLIS SS Encoder 提取训练数据

### 原理

```
3D Mesh → TRELLIS SS Encoder → dense latent 16³×8
                                   ↓
                              .npz (dense volume [8,16,16,16])
```

SS Flow 的训练目标是：给定 CLIP 文本特征，预测这个 dense SS latent volume。
SS Decoder 随后把这个 dense volume 解码为 sparse occupancy（64³），再 scale 到 128-res 喂给 SLat Flow。

> **关键：** SS latent 和 LATO latent 是两个**独立**的东西。SS latent 用于 SS Flow 训练（dense 16³×8），LATO latent 用于 SLat Flow 训练（sparse 128³×16）。两者互不依赖，可并行提取。

### 运行命令

```bash
cd /data/huanghaoyang/3D/TRELLIS

python lato_integration/encode_ss_latent.py \
    --trellis_pretrained microsoft/TRELLIS-text-base \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir /data/huanghaoyang/3D/database_lato/ss_latents \
    --device cuda
```

### 参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--trellis_pretrained` | TRELLIS 预训练 pipeline（含 SS Encoder） | `microsoft/TRELLIS-text-base` |
| `--data_dir` | 包含 `metadata.csv` 和 `meshes/` 的数据目录 | 必填 |
| `--output_dir` | 输出目录（存放 `.npz` SS latent 文件） | 必填 |
| `--dry_run` | 仅打印不处理（先验证数据路径） | 可选 |

### 输出

```
ss_latents/
├── metadata.csv                                  # 更新后的元数据
└── latents/
    └── ss_enc_conv3d_16l8_fp16/
        ├── {sha256_1}.npz                        # dense [8,16,16,16] float16
        ├── {sha256_2}.npz
        └── ...
```

> **显存说明：** SS Encoder 只做一次前向推理（`@torch.no_grad()`），显存需求远低于训练。RTX 4090 24GB 单卡轻松运行。整个数据集 ~1 小时内完成。

---

## 步骤3a：计算 SLat normalization 统计量

```bash
cd /data/huanghaoyang/3D/TRELLIS

python dataset_toolkits/stat_latent.py \
    --output_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --model lato_vae_16dim_128 \
    --num_samples 50000
```

输出示例：
```
mean: [-2.17, -0.00, -0.13, -0.08, -0.53, 0.72, -1.14, 1.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
std:  [2.38, 2.39, 2.12, 2.17, 2.66, 2.37, 2.62, 2.68, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

产物：`lato_latents/latents/lato_vae_16dim_128/stats.json`

**将 `stats.json` 中的 `mean` 和 `std` 填入训练配置：**

编辑 `configs/generation/lato_slat_flow.json`，替换 `dataset.args.normalization` 中的 `mean` 和 `std`。

---

## 步骤3b：🆕 计算 SS normalization 统计量

```bash
cd /data/huanghaoyang/3D/TRELLIS

python dataset_toolkits/stat_latent.py \
    --output_dir /data/huanghaoyang/3D/database_lato/ss_latents \
    --model ss_enc_conv3d_16l8_fp16 \
    --num_samples 50000
```

产物：`ss_latents/latents/ss_enc_conv3d_16l8_fp16/stats.json`

> **注意：** 如果数据量不大（<1万），`--num_samples` 可以省略（默认全量计算）。

---

## 步骤4a：🆕 训练 SS Flow（刹车卡钳数据集）

### 训练配置

创建 `configs/generation/lato_ss_flow.json`（**基于原版 ss_flow_txt_dit_L_16l8_fp16.json，已为 RTX 4090 24GB 优化**）：

```json
{
    "models": {
        "denoiser": {
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": 16,
                "in_channels": 8,
                "out_channels": 8,
                "model_channels": 512,
                "cond_channels": 768,
                "num_blocks": 24,
                "num_heads": 16,
                "mlp_ratio": 4,
                "patch_size": 1,
                "pe_mode": "ape",
                "qk_rms_norm": true,
                "use_fp16": true,
                "use_checkpoint": true
            }
        }
    },
    "dataset": {
        "name": "TextConditionedSparseStructureLatent",
        "args": {
            "latent_model": "ss_enc_conv3d_16l8_fp16",
            "min_aesthetic_score": 4.5,
            "normalization": { /* ← 步骤3b 的 stats.json 真实值填这里 */ }
        }
    },
    "trainer": {
        "name": "TextConditionedFlowMatchingCFGTrainer",
        "args": {
            "max_steps": 1000000,
            "batch_size_per_gpu": 2,
            "batch_split": 1,
            "p_uncond": 0.1,
            "optimizer": {
                "name": "AdamW",
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.0
                }
            },
            "ema_rate": [0.9999],
            "fp16_mode": "inflat_all",
            "fp16_scale_growth": 0.001,
            "grad_clip": {
                "name": "AdaptiveGradClipper",
                "args": {
                    "max_norm": 1.0,
                    "clip_percentile": 95
                }
            },
            "i_log": 500,
            "i_sample": 10000,
            "i_save": 10000,
            "t_schedule": {
                "name": "logitNormal",
                "args": {
                    "mean": 1.0,
                    "std": 1.0
                }
            },
            "sigma_min": 1e-5,
            "text_cond_model": "openai/clip-vit-large-patch14"
        }
    }
}
```

### 关键参数说明

| 参数 | 值 | 原因 |
|------|-----|------|
| `model_channels` | **512** | SS Flow 是 dense 17³×8 体积，比 SLat 的 sparse 体素省显存，512 在 24GB 可跑 |
| `cond_channels` | **768** | 必须等于 CLIP ViT-L/14 输出维度 |
| `resolution` | **16** | 原版 TRELLIS SS latent 分辨率（dense 体积 16³×8） |
| `batch_size_per_gpu` | **2** | Dense 体积比 sparse 省显存，可以 batch=2 |
| `use_checkpoint` | **true** | 反向传播时重新计算中间激活，节省显存 |
| `use_fp16` | **true** | 混合精度训练 |

### 运行训练

**单卡：**
```bash
cd /data/huanghaoyang/3D/TRELLIS

CUDA_VISIBLE_DEVICES=2 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/ss_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow \
    --num_gpus 1 \
    --auto_retry 0
```

**多卡（SS Flow + SLat Flow 并行）：**
```bash
# 终端1：SS Flow（GPU 2,4）
CUDA_VISIBLE_DEVICES=2,4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/ss_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow \
    --num_gpus 2 \
    --auto_retry 0

# 终端2：SLat Flow（GPU 6,7）
CUDA_VISIBLE_DEVICES=6,7 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_slat_flow \
    --num_gpus 2 \
    --auto_retry 0
```

### 说明

| 项目 | 说明 |
|------|------|
| 训练步数 | 1,000,000 步 |
| 输入 | CLIP 文本特征（768-dim） |
| 输出 | dense SS latent（16³×8） |
| batch_size | 2 per GPU（dense 省显存） |
| 速度（单卡 512ch） | ~15,000 steps/h，ETA ~2.8 天 |
| 速度（2 卡 512ch） | ~30,000 steps/h，ETA ~1.4 天 |
| 预训练权重 | **不需要**，从零训练 |
| 断点续训 | 自动 resume |
| 输出 | `outputs/lato_ss_flow/ckpts/denoiser_step{step}.pt` |

### 为什么 SS Flow 显存比 SLat Flow 宽裕？

| | SS Flow | SLat Flow |
|---|---|---|
| 数据格式 | **Dense** 16³×8 = 32,768 个 float | **Sparse** 最多 16,384 个体素 × 16 维 |
| 模型输入 | 固定尺寸 dense tensor | 变长 sparse tensor |
| 计算方式 | 标准 3D 卷积 | Sparse 3D 卷积（开销更大） |
| batch_size | 2 | 1 |

> SS Flow 用标准 3D 卷积在 16³ 的小体积上做 dense flow matching，比 SLat Flow 的 sparse 128-res 操作更省显存。

### 显存不够（OOM）

| 优先级 | 措施 |
|--------|------|
| 1 | `model_channels: 512 → 384` |
| 2 | `batch_size_per_gpu: 2 → 1` |
| 3 | `use_checkpoint: true`（确保已开启） |

---

## 步骤4b：训练 SLat Flow（刹车卡钳数据集）

> **本节与原 v2 步骤4 相同。** SLat Flow 训练和 SS Flow 训练**完全独立**，可并行运行。

### 训练配置

`configs/generation/lato_slat_flow.json` 关键参数（**已为 RTX 4090 24GB 实测验证**）：

```json
{
    "models": {
        "denoiser": {
            "name": "LATOSLatFlowModel",
            "args": {
                "resolution": 128,
                "in_channels": 16,
                "out_channels": 16,
                "model_channels": 384,
                "cond_channels": 768,
                "num_blocks": 12,
                "num_heads": 8,
                "mlp_ratio": 4,
                "patch_size": 2,
                "num_io_res_blocks": 2,
                "io_block_channels": [128],
                "pe_mode": "ape",
                "qk_rms_norm": true,
                "use_fp16": true,
                "use_checkpoint": true
            }
        }
    },
    "dataset": {
        "name": "TextConditionedSLat",
        "args": {
            "latent_model": "lato_vae_16dim_128",
            "min_aesthetic_score": 4.5,
            "max_num_voxels": 16384,
            "normalization": { /* ← 步骤3a 的真实值填这里 */ },
            "pretrained_slat_dec": null
        }
    },
    "trainer": {
        "name": "TextConditionedSparseFlowMatchingCFGTrainer",
        "args": {
            "max_steps": 1000000,
            "batch_size_per_gpu": 1,
            "batch_split": 1,
            "p_uncond": 0.1,
            "text_cond_model": "openai/clip-vit-large-patch14"
        }
    }
}
```

### 关键参数说明

| 参数 | 值 | 原因 |
|------|-----|------|
| `model_channels` | **384** | RTX 4090 24GB 单卡上限，512 会 OOM |
| `cond_channels` | **768** | 必须等于 CLIP ViT-L/14 输出维度，否则矩阵乘法报错 |
| `max_num_voxels` | **16384** | 平衡数据覆盖和显存，32768 会 OOM |
| `batch_size_per_gpu` | **1** | 最小 batch，减少显存 |
| `batch_split` | **1** | 配合 batch_size=1 |
| `use_checkpoint` | **true** | 反向传播时重新计算中间激活，大幅节省显存 |
| `use_fp16` | **true** | 混合精度训练，权重和激活用 fp16 |
| `pretrained_slat_dec` | **null** | 不使用 TRELLIS 原版 decoder |

### 运行训练

**单卡：**
```bash
cd /data/huanghaoyang/3D/TRELLIS

CUDA_VISIBLE_DEVICES=2 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_slat_flow \
    --num_gpus 1 \
    --auto_retry 0
```

**多卡：**
```bash
CUDA_VISIBLE_DEVICES=2,4,6,7 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_slat_flow \
    --num_gpus 4 \
    --auto_retry 0
```

> **`CUDA_VISIBLE_DEVICES` 技巧：** 先用 `nvidia-smi` 查看各 GPU 的空闲情况，然后用这个环境变量选空闲的卡。程序内 GPU 编号从 0 重新映射。

### 说明

| 项目 | 说明 |
|------|------|
| 训练步数 | 1,000,000 步 |
| batch_size | 1 per GPU（4 卡 = 有效 batch 4） |
| 速度（单卡 384ch） | ~9,000 steps/h，ETA ~4.5 天 |
| 速度（4 卡 384ch） | ~36,000 steps/h，ETA ~1 天 |
| 预训练权重 | **不需要**，从零训练 |
| 断点续训 | 自动 resume |
| 输出 | `outputs/lato_slat_flow/ckpts/denoiser_step{step}.pt` |

### 显存不够（OOM）— RTX 4090 实测方案

| 优先级 | 措施 | 效果 |
|--------|------|------|
| 1 | `use_checkpoint: true` | 反向传播省 30-50% 显存 |
| 2 | `model_channels: 384 → 256` | 显存减半，模型容量下降 |
| 3 | `max_num_voxels: 16384 → 12288 → 8192` | 减少 voxel 数（但会过滤样本） |
| 4 | `batch_size_per_gpu: 1`（已是最小） | — |
| 5 | 换空闲 GPU（`CUDA_VISIBLE_DEVICES`） | 避免其他进程占用 |

> **实测数据：** RTX 4090 24GB，`model_channels=384` + `use_checkpoint=true` + `max_num_voxels=16384` + `batch_size=1`，单卡可稳定运行，显存使用 ~21GB。

### 关于 snapshot OOM

训练中的 snapshot（`i_sample=10000` 触发）做推理采样，显存消耗比训练更大。如果 snapshot OOM：
- 增加 `i_sample` 延迟采样（如改为 `50000`）
- 或跳过 init/resume snapshot（`base.py` 中已默认跳过）

---

## 步骤5：推理（文本 → 3D Mesh）

### 完整推理管线（v3）

```
Prompt → CLIP → SS Flow(刹车卡钳训练) → SS Decoder(冻结) → coords[:,1:]×2
                     ↓                                              ↓
              输出: dense 16³×8                            sparse coords 128-res
                                                                      ↓
                                                          SLat Flow(刹车卡钳训练)
                                                                      ↓
                                                          LATO VoxelVAE.decode()
                                                                      ↓
                                                          ConnectionHead → Mesh (.obj)
```

### 运行命令

```bash
cd /data/huanghaoyang/3D/TRELLIS

python lato_integration/inference_lato.py \
    --ss_ckpt outputs/lato_ss_flow/ckpts/denoiser_step1000000.pt \
    --slat_ckpt outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --ss_stats /data/huanghaoyang/3D/database_lato/ss_latents/latents/ss_enc_conv3d_16l8_fp16/stats.json \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents/latents/lato_vae_16dim_128/stats.json \
    --prompt "A brake caliper fixing interaxis 94.31 inner pad 98.47 pistons_num 4" \
    --seed 42 \
    --ss_steps 20 \
    --slat_steps 20 \
    --cfg_strength 5.0 \
    --output output_caliper.obj
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ss_ckpt` | **必填**（v3） | 🆕 你训练的 LATO SS Flow checkpoint |
| `--slat_ckpt` | **必填** | 你训练的 LATO SLat Flow checkpoint |
| `--lato_ckpt` | **必填** | LATO 预训练 checkpoint |
| `--lato_config` | **必填** | LATO VAE 配置 yaml |
| `--ss_stats` | 空 | 🆕 步骤3b 的 SS stats.json |
| `--slat_stats` | 空 | 步骤3a 的 SLat stats.json |
| `--prompt` | **必填** | 文本描述（刹车卡钳工程参数） |
| `--seed` | 42 | 随机种子 |
| `--ss_steps` | 20 | SS Flow 采样步数 |
| `--slat_steps` | 20 | SLat Flow 采样步数 |
| `--cfg_strength` | 5.0 | Classifier-Free Guidance 强度 |
| `--lato_threshold` | 0.2 | VoxelVAE decode 的 inference_threshold |
| `--edge_threshold` | 0.45 | ConnectionHead 边概率阈值 |
| `--output` | `output_mesh.obj` | 输出 mesh 路径 |

---

## 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `trellis/models/lato_slat_flow.py` | LATOSLatFlowModel 定义 | ✅ 已完成 |
| `trellis/pipelines/trellis_text_to_3d.py` | Pipeline：coords 缩放 + LATO decode | ✅ 已完成 |
| `trellis/models/__init__.py` | 模型注册 | ✅ 已完成 |
| `trellis/datasets/structured_latent.py` | Dataset：coords 适配 + 可视化跳过 | ✅ 已完成 |
| `trellis/trainers/base.py` | 跳过 init/resume snapshot | ✅ 已完成 |
| `configs/generation/lato_ss_flow.json` | 🆕 SS Flow 训练配置（512ch，RTX 4090 优化） | 🆕 需创建 |
| `configs/generation/lato_slat_flow.json` | SLat Flow 训练配置（384ch，RTX 4090 优化） | ✅ 已完成 |
| `lato_integration/run_train.py` | 训练启动脚本（SS Flow + SLat Flow 通用） | ✅ 已完成 |
| `lato_integration/flow/trainers/ss_flow_trainer.py` | SS Flow 训练器 | ✅ 已完成 |
| `lato_integration/flow/trainers/slat_flow_trainer.py` | SLat Flow 训练器 | ✅ 已完成 |
| `lato_integration/encode_lato_latent_v2.py` | LATO latent 提取（官方架构） | ✅ 已完成 |
| `lato_integration/encode_ss_latent.py` | 🆕 SS latent 提取（TRELLIS SS Encoder） | 🆕 需创建 |
| `lato_integration/inference_lato.py` | 端到端推理脚本（需更新支持 --ss_ckpt） | ⚠️ 需更新 |
| `dataset_toolkits/stat_latent.py` | Normalization 统计量计算 | ✅ 已有 |

---

## v3 待办事项

以下文件仍需创建或修改：

| 优先级 | 事项 | 说明 |
|--------|------|------|
| 🔴 高 | 创建 `configs/generation/lato_ss_flow.json` | 基于上文步骤4a 的 JSON 配置 |
| 🔴 高 | 创建 `lato_integration/encode_ss_latent.py` | 用 TRELLIS SS Encoder 提取刹车卡钳的 dense SS latent（参考 `encode_lato_latent_v2.py` 的结构） |
| 🔴 高 | 更新 `lato_integration/inference_lato.py` | 新增 `--ss_ckpt`、`--ss_stats` 参数，替换掉 `--trellis_pretrained` |
| 🟡 中 | 运行步骤2b（提取 SS latent） | 234 个 mesh × 1 次前向推理 ≈ 1 小时 |
| 🟡 中 | 运行步骤3b（计算 SS stats） | 几分钟 |
| 🟢 低 | 多卡并行训练 SS Flow + SLat Flow | 4 卡分两组：2 卡 SS + 2 卡 SLat |

---

## 常见问题

### Q: 显存不够（OOM）

- 步骤2：降低 `--num_points 32768`
- 步骤4：确保 `use_checkpoint: true`
- 步骤4：降低 `model_channels`（384→256）
- 步骤4：减少 `max_num_voxels`（16384→12288→8192，注意检查样本过滤情况）
- 步骤4：换空闲 GPU（`CUDA_VISIBLE_DEVICES`）
- 步骤4：多卡**不能**直接解决单卡 OOM（每卡各自处理自己的 batch）

### Q: 训练报 `mat1 and mat2 shapes cannot be multiplied (308x768 and 512x1024)`

`cond_channels` 必须等于 CLIP 模型输出维度。CLIP ViT-L/14 输出 **768**，所以 `cond_channels` 必须设为 **768**（不能是 512）。

### Q: 训练报 `AttributeError: '...Trainer' object has no attribute 'global_step'`

`slat_flow_trainer.py` 中 `self.global_step` 应改为 `self.step`（TRELLIS 基类使用 `step` 而非 `global_step`）。

### Q: 多卡训练 hang 住不动

通常是 init snapshot 时某个 GPU OOM 或不同步导致。确保 `base.py` 中已跳过 init/resume snapshot。

### Q: 推理时 ConnectionHead 边太少/太多

调整 `--edge_threshold`：
- 边太少（mesh 有洞）→ 降低阈值到 `0.3` 或 `0.35`
- 边太多（全是面）→ 提高阈值到 `0.5` 或 `0.6`

### Q: 推理时 `No module named 'lato'`

```bash
export LATO_ROOT="/data/huanghaoyang/3D/LATO"
```

### Q: 推理时 `No module named 'trellis'`

```bash
export TRELLIS_ROOT="/data/huanghaoyang/3D/TRELLIS"
export PYTHONPATH="/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
```

### Q: LATO checkpoint 加载失败

确保：
- LATO config 中 `latent_dim: 16`
- `in_channels: 1024`
- checkpoint 路径正确（HuggingFace 下载的 `vae_128to512.pt`）

### Q: 训练报 `AssertionError: No elastic module found`

从 `lato_slat_flow.json` 中删除 `trainer.args.elastic` 配置块（`LATOSLatFlowModel` 不走 elastic memory 路径）。

### Q: 训练报 `AttributeError: 'NoneType' object has no attribute 'split'`

确认 `pretrained_slat_dec` 设为 `null`（LATO 训练不需要 TRELLIS 原版 decoder 做可视化）。

### Q: 训练报 spconv ndim 不匹配

确认已应用 `structured_latent.py` 的 coords 剥离修复（`coords[:, 1:]`）。

### Q: 🆕 SS Flow 和 SLat Flow 可以同时训练吗？

**可以。** 两者完全独立：
- SS Flow 训练数据：SS latent（dense 16³×8）+ 文本 caption
- SLat Flow 训练数据：LATO latent（sparse coords + 16-dim feats）+ 文本 caption
- SLat Flow 训练时使用的 conditioning 是 **ground truth SS latent**（来自 SS Encoder），不依赖 SS Flow 的预测输出
- 如果 GPU 资源有限，**建议先训 SS Flow**（更快，~2.8天），再训 SLat Flow

### Q: 🆕 SS Flow 训练报 `KeyError: 'ss_enc_conv3d_16l8_fp16'`

数据集不包含 SS latent 文件。需要先运行步骤2b（`encode_ss_latent.py`）提取 SS latent，确认 `ss_latents/latents/ss_enc_conv3d_16l8_fp16/` 目录下有 `.npz` 文件。

### Q: 🆕 SS Flow 训练完怎么验证效果？

SS Flow 单独没有直观的可视化输出（输出是 dense latent 体积）。验证方式：
1. 端到端推理：用训好的 SS Flow + 训好的 SLat Flow + LATO VAE 跑一条完整管线
2. 检查训练 loss 曲线：`mse` 和 `bin_*` loss 是否收敛
3. 对比：用 TRELLIS 预训练 SS Flow vs 你训的 SS Flow，同样 prompt 生成 mesh 对比质量

---

## TRELLIS 原版 vs 本方案

| | 原版 TRELLIS | v2（只用 LATO） | v3（LATO + 刹车卡钳定制） |
|---|---|---|---|
| 需训练的模型数 | 5 个 × 1M 步 | 1 个 × 1M 步 | **2 个 × 1M 步** |
| 训练的模型 | 全部 | SLat Flow | **SS Flow + SLat Flow** |
| 使用预训练权重的 | 0 | SS Flow + SS Dec + LATO VAE | SS Dec + LATO VAE |
| SS Flow 来源 | 自己训练 | TRELLIS 预训练（通用物体） | **刹车卡钳数据训练** |
| SLat Flow 来源 | 自己训练 | 刹车卡钳数据训练 | 刹车卡钳数据训练 |
| 对刹车卡钳的空间理解 | 有（如包含卡钳数据） | 弱（依赖通用预训练） | **强（专门训练）** |
| 文本→结构精度 | — | 一般 | **高**（参数化控制） |
| 输出格式 | Mesh + 3DGS + RF | Mesh | Mesh |
| Latent 维度 | 8 | 16（更丰富） | 16（更丰富） |
| 分辨率 | 64 | 128（更精细） | 128（更精细） |
| SLat 模型宽度 | 768 | 384 | 384 |
| SS 模型宽度 | 1024 | —（不训练） | **512**（适配 24GB） |
| 训练方式 | 单卡 | 支持多卡 DDP | 支持多卡 DDP（两任务可并行） |

---

## 已修复的 Bug 清单（v2 更新）

### 第一轮修复（上午，12 个）

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `run_train.py` | `resolve_model` 过滤掉 `**kwargs` 中的父类参数 | 检测 `VAR_KEYWORD`，有则全量透传 |
| 2 | `lato_slat_flow.json` | `elastic` 配置触发断言 | 删除 elastic 块 |
| 3 | `structured_latent.py` | `pretrained_slat_dec=None` 导致 `from_pretrained(None)` | 加 None 检查跳过 |
| 4 | `structured_latent.py` | `visualize_sample` 无可视化 decoder 仍然尝试渲染 | 返回空 dict |
| 5 | `structured_latent.py` | LATO coords `[N,4]` 叠加 collate batch 变 `[N,5]` 导致 spconv ndim 不匹配 | 加载时剥离多余 batch 列 |
| 6 | `slat_flow_trainer.py` | latent consistency 的 t `[B]` 无法广播到 feats `[N_total,16]` | 用 sparse layout 逐样本扩展 |
| 7 | `trellis_text_to_3d.py` | `coords * 2` 连 batch 维一起乘 | 改为 `coords[:, 1:] *= 2` |
| 8 | `trellis_text_to_3d.py` | `decode_slat_lato` 硬编码 `inference_threshold=0.2` | 改为读 `self.lato_inference_threshold` |
| 9 | `inference_lato.py` | `_LATO_ROOT` 多一层 `..` | 修正路径 |
| 10 | `inference_lato.py` | `_TRELLIS_ROOT` 指向包目录而非父目录 | 修正路径 |
| 11 | `inference_lato.py` | checkpoint 加载不兼容完整 ckpt 格式 | 加 state_dict 解包 |
| 12 | `ss_flow_trainer.py` | 缺少 `import numpy as np` | 补充 import |

### 第二轮修复（下午，4 个 — 训练启动阶段）

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 13 | `lato_slat_flow.json` | `cond_channels=512` 与 CLIP ViT-L/14 输出 768 不匹配 → `mat1 and mat2 (308×768 vs 512×1024)` | `cond_channels: 512 → 768` |
| 14 | `base.py` | init/resume snapshot 导致 OOM 和多卡 DDP hang | 跳过 init/resume snapshot |
| 15 | `slat_flow_trainer.py` | `self.global_step` 不存在 → AttributeError | `global_step → step` |
| 16 | `lato_slat_flow.json` | 默认参数（512ch / 32K voxel / batch 4）在 RTX 4090 24GB 单卡 OOM | `model_channels→384`, `max_num_voxels→16384`, `batch→1`, `use_checkpoint: true` |
