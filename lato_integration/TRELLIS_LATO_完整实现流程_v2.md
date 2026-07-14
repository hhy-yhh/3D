# TRELLIS + LATO 文本转3D — 完整实现流程 v2（更新版）

> **目标：** TRELLIS 文本转3D 管线中，将 Sparse VAE Encoder/Decoder 替换为 LATO 的 VoxelVAE，SS Flow 和 SLat Flow 沿用 TRELLIS 原版架构。

**更新记录：**
- 2026-07-14：修复 12 个 bug，新增多卡训练方案，调整默认模型参数

---

## 架构概览

```
原版 TRELLIS:
  Text → CLIP → SS Flow → SS Decoder → SLat Flow → SLat Decoder → Mesh/GS/RF

LATO 替换后:
  Text → CLIP → SS Flow → SS Decoder(原版) → SLat Flow(16-dim/128-res)
                 ↑ 不变                          ↑ 架构不变，dim 适配
                       → LATO VoxelVAE.decode() → ConnectionHead → Mesh
                         ↑ 替换掉原版 SLat Decoder
```

**关键改动：**
- SS Flow + SS Decoder：**完全不动**（使用 TRELLIS 预训练权重）
- SLat Flow：架构仍是 TRELLIS `SLatFlowModel`，参数改为 `resolution=128, in/out_channels=16`（适配 LATO latent 空间）
- Decoder：**替换**为 LATO VoxelVAE + ConnectionHead（使用 LATO 预训练权重）
- 只需训练 **1 个** 新模型（SLat Flow，约 1M 步）

---

## 前置条件

### 硬件要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU | 1× 24GB (RTX 4090) | 4× RTX 4090 |
| 显存 | 20GB+ | 24GB |
| CUDA | 11.8+ | 12.5 |

> **注意：** 使用 `model_channels=512` 时单卡 24GB 可运行。若使用 `model_channels=768`，单卡大概率 OOM，需要多卡或减小模型。

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

- ✅ TRELLIS 代码（已修改 4 个文件）
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
| 1 | 确认代码修改 | 4 个文件 | 已就绪 |
| 2 | 提取 LATO latent | `encode_lato_latent_v2.py` | 取决于数据集大小 |
| 3 | 计算 normalization 统计量 | `stat_latent.py` | 5 分钟 |
| 4 | 训练 SLat Flow | `run_train.py` | ~1M 步（数小时到一天） |
| 5 | 推理 | `inference_lato.py` | ~1 分钟/样本 |

---

## 步骤1：确认已修改的代码（4个文件）

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

## 步骤3：计算 normalization 统计量

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

## 步骤4：训练 LATO SLat Flow

### 训练配置

`configs/generation/lato_slat_flow.json` 关键参数（**已为 RTX 4090 优化**）：

```json
{
    "models": {
        "denoiser": {
            "name": "LATOSLatFlowModel",
            "args": {
                "resolution": 128,
                "in_channels": 16,
                "out_channels": 16,
                "model_channels": 512,
                "cond_channels": 768,
                "num_blocks": 12,
                "num_heads": 8,
                "mlp_ratio": 4,
                "patch_size": 2,
                "num_io_res_blocks": 2,
                "io_block_channels": [128],
                "pe_mode": "ape",
                "qk_rms_norm": true,
                "use_fp16": true
            }
        }
    },
    "dataset": {
        "name": "TextConditionedSLat",
        "args": {
            "latent_model": "lato_vae_16dim_128",
            "min_aesthetic_score": 4.5,
            "max_num_voxels": 32768,
            "normalization": { /* ← 步骤3 的真实值填这里 */ },
            "pretrained_slat_dec": null
        }
    },
    "trainer": {
        "name": "TextConditionedSparseFlowMatchingCFGTrainer",
        "args": {
            "max_steps": 1000000,
            "batch_size_per_gpu": 4,
            "batch_split": 2,
            "p_uncond": 0.1,
            "text_cond_model": "openai/clip-vit-large-patch14"
        }
    }
}
```

### 运行训练

**单卡：**
```bash
cd /data/huanghaoyang/3D/TRELLIS

python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir outputs/lato_slat_flow \
    --num_gpus 1 \
    --auto_retry 0
```

**多卡（推荐）：**
```bash
python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir outputs/lato_slat_flow \
    --num_gpus 4 \
    --auto_retry 0
```

### 说明

| 项目 | 说明 |
|------|------|
| 训练步数 | 1,000,000 步 |
| batch_size | 4 per GPU（4 卡 = 有效 batch 16） |
| 预训练权重 | **不需要**，从零训练 |
| 断点续训 | 自动 resume（`--auto_retry 3` 崩溃自动重试 3 次） |
| 输出 | `outputs/lato_slat_flow/ckpts/denoiser_step{step}.pt` |

### 显存不够（OOM）

| 措施 | 效果 |
|------|------|
| `model_channels: 512 → 384` | 显存减半 |
| `batch_size_per_gpu: 4 → 2` | 减少 batch 显存 |
| `batch_split: 2 → 4` | 更多梯度累积 |
| `max_num_voxels: 32768 → 16384` | 减少 voxel 数 |
| `--num_gpus 8` | 分摊到更多 GPU |

### 关于 snapshot OOM

训练中的 snapshot（`i_sample=10000` 触发）做推理采样，显存消耗比训练更大。如果 snapshot OOM：
- 增加 `i_sample` 延迟采样（如改为 50000）
- 或换用 `model_channels=384` 的轻量配置

---

## 步骤5：推理（文本 → 3D Mesh）

### 完整推理管线

```
Prompt → CLIP → SS Flow → SS Decoder → coords[:,1:]×2 → SLat Flow(新训)
                                                             ↓
                                              LATO VoxelVAE.decode()
                                                             ↓
                                              ConnectionHead 边预测
                                                             ↓
                                              三角面片化 → Mesh (.obj)
```

### 运行命令

```bash
cd /data/huanghaoyang/3D/TRELLIS

python lato_integration/inference_lato.py \
    --trellis_pretrained microsoft/TRELLIS-text-base \
    --slat_ckpt outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents/latents/lato_vae_16dim_128/stats.json \
    --prompt "a Chinese style dragon" \
    --seed 42 \
    --ss_steps 20 \
    --slat_steps 20 \
    --cfg_strength 5.0 \
    --output output_mesh.obj
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--trellis_pretrained` | `microsoft/TRELLIS-text-base` | TRELLIS 预训练 pipeline（SS 部分） |
| `--slat_ckpt` | **必填** | 你训练的 LATO SLat Flow checkpoint |
| `--lato_ckpt` | **必填** | LATO 预训练 checkpoint |
| `--lato_config` | **必填** | LATO VAE 配置 yaml |
| `--slat_stats` | 空（用零均值/单位方差） | 步骤3 生成的 stats.json |
| `--prompt` | **必填** | 文本描述 |
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
| `configs/generation/lato_slat_flow.json` | SLat Flow 训练配置（512ch） | ✅ 已完成 |
| `lato_integration/run_train.py` | 训练启动脚本 | ✅ 已完成 |
| `lato_integration/flow/trainers/slat_flow_trainer.py` | SLat Flow 训练器 | ✅ 已完成 |
| `lato_integration/flow/trainers/ss_flow_trainer.py` | SS Flow 训练器 | ✅ 已完成 |
| `lato_integration/encode_lato_latent_v2.py` | LATO latent 提取（官方架构） | ✅ 已完成 |
| `lato_integration/inference_lato.py` | 端到端推理脚本 | ✅ 已完成 |
| `dataset_toolkits/stat_latent.py` | Normalization 统计量计算 | ✅ 已有 |

---

## 常见问题

### Q: 显存不够（OOM）

- 步骤2：降低 `--num_points 32768`
- 步骤4：减小 `batch_size_per_gpu` 和 `batch_split`
- 步骤4：降低 `model_channels`（768→512→384）
- 步骤4：减少 `max_num_voxels`
- 步骤4：增加 GPU 数量（`--num_gpus 4` 或 `--num_gpus 8`）

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

---

## TRELLIS 原版 vs 本方案

| | 原版 TRELLIS 从头训 | 本方案 |
|---|---|---|
| 需训练的模型数 | 5 个 × 1M 步 | **1 个 × 1M 步** |
| 使用预训练权重的 | 0 | SS Flow + SS Dec + LATO VAE |
| 输出格式 | Mesh + 3DGS + RF | Mesh |
| Latent 维度 | 8 | **16**（更丰富） |
| 分辨率 | 64 | **128**（更精细） |
| 推荐模型宽度 | 768 | **512**（适配 24GB 显存） |
| 训练方式 | 单卡 | 支持多卡 DDP |

---

## 已修复的 Bug 清单（v2 更新）

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
