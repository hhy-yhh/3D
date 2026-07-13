# TRELLIS + LATO 文本转3D — 完整实现流程

> **目标：** TRELLIS 文本转3D 管线中，将 Sparse VAE Encoder/Decoder 替换为 LATO 的 VoxelVAE，SS Flow 和 SLat Flow 沿用 TRELLIS 原版架构。

---

## 目录

- [架构概览](#架构概览)
- [前置条件](#前置条件)
- [步骤1：确认已修改的代码（3个文件）](#步骤1确认已修改的代码3个文件)
- [步骤2：用 LATO VAE 提取训练数据](#步骤2用-lato-vae-提取训练数据)
- [步骤3：计算 normalization 统计量](#步骤3计算-normalization-统计量)
- [步骤4：训练 LATO SLat Flow](#步骤4训练-lato-slat-flow)
- [步骤5：推理（文本 → 3D Mesh）](#步骤5推理文本--3d-mesh)
- [文件清单](#文件清单)

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

### 目录结构

```
D:\code\
├── TRELLIS_linux\3D\
│   ├── trellis\                    # TRELLIS 代码（含已修改的文件）
│   ├── lato_integration\           # 集成脚本（本目录）
│   ├── configs\generation\         # 训练配置
│   └── dataset_toolkits\           # 数据工具
│
├── LATO\                           # LATO 官方代码 + 预训练权重
│   ├── lato\
│   ├── configs\infer_vae_512.yaml
│   ├── vertex_encoder.py
│   ├── utils.py
│   └── ckpts\                      # LATO checkpoint.pt（需要你自己有）
│
└── database\                       # 训练数据
    ├── metadata.csv
    └── meshes\                     # 3D 模型文件
```

### 你需要有

- ✅ TRELLIS 预训练权重（`microsoft/TRELLIS-text-base`，含 SS Flow + SS Decoder）
- ✅ LATO 预训练权重（VoxelVAE + ConnectionHead）
- ✅ 训练数据集（3D 模型 `meshes/` + `metadata.csv`）
- ✅ GPU（建议 24GB+ 显存）

### 环境变量

```bash
export TRELLIS_ROOT="D:/code/TRELLIS_linux/3D/trellis"
export LATO_ROOT="D:/code/LATO"
export PYTHONPATH="$TRELLIS_ROOT:$LATO_ROOT:$PYTHONPATH"
```

---

## 步骤1：确认已修改的代码（3个文件）

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

三处关键修改：

**(1) `run()` 中 SS 坐标缩放：**
```python
coords = self.sample_sparse_structure(cond, num_samples, ...)
coords = coords * 2   # res 64 → 128，适配 LATO latent
```

**(2) 新增 `decode_slat_lato()` 方法：**
- 自动将 TRELLIS SparseTensor 转换为 LATO SparseTensor
- 调用 `lato_vae.decode()` 生成 vertex 层级
- LATO 未安装时自动回退到原版 decoder

**(3) `decode_slat()` 扩展：**
- 检测到 `lato_vae` 时自动走 LATO 路径
- 否则使用原版 TRELLIS decoder

### 1c. `trellis/models/__init__.py` ✅ 已完成

已注册 `LATOSLatFlowModel` 到 `lato_slat_flow` 模块。

---

## 步骤2：用 LATO VAE 提取训练数据

### 原理

```
3D Mesh → 体素化(res=128) → 点云采样 → VoxelFeatureEncoder → VoxelVAE.encode()
                                                                       ↓
                                          .npz (coords[N,4] + feats[N,16])
```

### 运行命令

```bash
cd D:/code/TRELLIS_linux/3D

python lato_integration/encode_lato_latent.py \
    --lato_ckpt D:/code/LATO/ckpts/your_checkpoint.pt \
    --lato_config D:/code/LATO/configs/infer_vae_512.yaml \
    --data_dir D:/code/database \
    --output_dir D:/code/database/lato_latents \
    --resolution 128 \
    --num_points 819200 \
    --device cuda
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--lato_ckpt` | LATO 预训练 checkpoint `.pt` 文件 |
| `--lato_config` | LATO VAE 配置 yaml（`infer_vae_512.yaml`） |
| `--data_dir` | 包含 `metadata.csv` 和 `meshes/` 的数据目录 |
| `--output_dir` | 输出目录（存放 `.npz` latent 文件） |
| `--resolution` | 体素化分辨率，默认 128 |
| `--num_points` | 点云采样点数，默认 819200 |
| `--dry_run` | 仅打印不处理（先验证数据路径） |

### 输出

```
lato_latents/
├── metadata.csv                          # 更新后的元数据（含 latent_lato_vae_16dim_128 列）
└── latents/
    └── lato_vae_16dim_128/
        ├── {sha256_1}.npz                # coords[N,4] + feats[N,16]
        ├── {sha256_2}.npz
        └── ...
```

> **注意：** 如果部分模型处理失败，脚本会跳过并继续，最后报告成功/跳过/失败数量。

---

## 步骤3：计算 normalization 统计量

```bash
cd D:/code/TRELLIS_linux/3D

python dataset_toolkits/stat_latent.py \
    --output_dir D:/code/database/lato_latents \
    --model lato_vae_16dim_128 \
    --num_samples 50000
```

输出示例：
```
mean: [-0.12, 0.03, -0.08, ...]  # 16 个值
std:  [2.35, 2.41, 2.18, ...]    # 16 个值
```

产物：`lato_latents/latents/lato_vae_16dim_128/stats.json`

**将 `stats.json` 中的 `mean` 和 `std` 填入训练配置：**

编辑 `configs/generation/lato_slat_flow.json`，替换 `dataset.args.normalization` 中的 `mean` 和 `std`（当前是占位值）。

---

## 步骤4：训练 LATO SLat Flow

### 训练配置

`configs/generation/lato_slat_flow.json` 关键参数：

```json
{
    "models": {
        "denoiser": {
            "name": "LATOSLatFlowModel",
            "args": {
                "resolution": 128,
                "in_channels": 16,
                "out_channels": 16,
                "model_channels": 768,
                "cond_channels": 768,
                "num_blocks": 12,
                "num_heads": 12,
                "patch_size": 2,
                "io_block_channels": [128],
                "use_fp16": true
            }
        }
    },
    "dataset": {
        "name": "TextConditionedSLat",
        "args": {
            "latent_model": "lato_vae_16dim_128",
            "max_num_voxels": 32768,
            "normalization": { /* ← 步骤3 的真实值填这里 */ }
        }
    },
    "trainer": {
        "name": "TextConditionedSparseFlowMatchingCFGTrainer",
        "args": {
            "max_steps": 1000000,
            "batch_size_per_gpu": 4,
            ...
        }
    }
}
```

### 运行训练

```bash
cd D:/code/TRELLIS_linux/3D

python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir D:/code/database/lato_latents \
    --output_dir outputs/lato_slat_flow \
    --num_gpus 1 \
    --auto_retry 3
```

### 关键说明

| 项目 | 说明 |
|------|------|
| 训练步数 | 1,000,000 步（约几小时到一天，取决于 GPU） |
| batch_size | 4（128-res 16-dim 显存较大，可据 GPU 调整） |
| 预训练权重 | **不需要**，从零训练（SS Flow + SS Decoder 用 TRELLIS 预训练，推理时加载） |
| 断点续训 | 自动 resume（`--auto_retry 3` 崩溃自动重试 3 次） |
| 输出 | `outputs/lato_slat_flow/ckpts/denoiser_step{step}.pt` |

---

## 步骤5：推理（文本 → 3D Mesh）

### 完整推理管线

```
Prompt → CLIP → SS Flow → SS Decoder → coords ×2 → SLat Flow(新训)
                                                              ↓
                                               LATO VoxelVAE.decode()
                                                              ↓
                                               ConnectionHead 边预测
                                                              ↓
                                               三角面片化 → Mesh (.obj)
```

### 运行命令

```bash
cd D:/code/TRELLIS_linux/3D

python lato_integration/inference_lato.py \
    --trellis_pretrained microsoft/TRELLIS-text-base \
    --slat_ckpt outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt \
    --lato_ckpt D:/code/LATO/ckpts/your_checkpoint.pt \
    --lato_config D:/code/LATO/configs/infer_vae_512.yaml \
    --slat_stats D:/code/database/lato_latents/latents/lato_vae_16dim_128/stats.json \
    --prompt "a brake caliper with 4 pistons" \
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

## 步骤总览

```
步骤1: 确认代码修改        → 已就绪（3 个文件已改好）
步骤2: 提取训练数据        → 跑 encode_lato_latent.py（取决于数据集大小）
步骤3: 计算统计量          → 跑 stat_latent.py（5 分钟）
步骤4: 训练 SLat Flow      → ~1M 步（唯一需要训练的，几小时到一天）
步骤5: 推理               → 跑 inference_lato.py（~1 分钟/样本）
```

### TRELLIS 原版 vs 本方案

| | 原版 TRELLIS 从头训 | 本方案 |
|---|---|---|
| 需训练的模型数 | 5 个 × 1M 步 | **1 个 × 1M 步** |
| 使用预训练权重的 | 0 | SS Flow + SS Dec + LATO VAE |
| 输出格式 | Mesh + 3DGS + RF | Mesh only |

---

## 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `trellis/models/lato_slat_flow.py` | LATOSLatFlowModel 定义 | ✅ 已完成 |
| `trellis/pipelines/trellis_text_to_3d.py` | Pipeline：coords 缩放 + LATO decode | ✅ 已完成 |
| `configs/generation/lato_slat_flow.json` | SLat Flow 训练配置 | ✅ 已完成 |
| `lato_integration/encode_lato_latent.py` | LATO latent 提取脚本 | ✅ 已完成 |
| `lato_integration/inference_lato.py` | 端到端推理脚本 | ✅ 已完成 |
| `dataset_toolkits/stat_latent.py` | Normalization 统计量计算 | ✅ 已有 |
| `lato_integration/run_train.py` | 训练启动脚本 | ✅ 已有 |

---

## 常见问题

### Q: 显存不够（OOM）

减小 `batch_size_per_gpu` 和 `batch_split`：

```json
"batch_size_per_gpu": 2,
"batch_split": 4,
```

或启用 elastic memory：
```json
"elastic": { "name": "LinearMemoryController", "args": { "target_ratio": 0.5 } }
```

### Q: 推理时 ConnectionHead 边太少/太多

调整 `--edge_threshold`：
- 边太少（mesh 有洞）→ 降低阈值到 `0.3` 或 `0.35`
- 边太多（全是面）→ 提高阈值到 `0.5` 或 `0.6`

### Q: LATO checkpoint 加载失败

LATO 的 `load_pretrained_woself` 会跳过 shape 不匹配的 key 并打印 warning。确保：
- LATO config 中 `latent_dim: 16`
- `in_channels: 1024`（VoxelFeatureEncoder 输出维度）

### Q: 推理时 `No module named 'lato'`

```bash
export PYTHONPATH="D:/code/LATO:$PYTHONPATH"
```

或在 `inference_lato.py` 中设置：
```bash
export LATO_ROOT="D:/code/LATO"
```
