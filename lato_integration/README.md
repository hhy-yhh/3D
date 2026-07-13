# LATO Integration for TRELLIS — 使用指南

> 将 LATO (ICML 2026) 的 VAE encode/decode 架构集成到 TRELLIS 的 ss_flow 和 slat_flow 训练管线中，提升文本转 3D 模型的生成质量。

---

## 目录

- [1. 项目结构](#1-项目结构)
- [2. 快速开始](#2-快速开始)
- [3. 阶段一：VAE 优化](#3-阶段一vae-优化)
  - [3.1 增强的 SS Encoder/Decoder](#31-增强的-ss-encoderdecoder)
  - [3.2 增强的 SLat Encoder](#32-增强的-slat-encoder)
  - [3.3 增强的 Gaussian Decoder](#33-增强的-gaussian-decoder)
  - [3.4 增强的 Radiance Field Decoder](#34-增强的-radiance-field-decoder)
  - [3.5 增强的 Mesh Decoder](#35-增强的-mesh-decoder)
- [4. 阶段二：Flow/DiT 优化](#4-阶段二flowdit-优化)
  - [4.1 增强的 SS Flow Model](#41-增强的-ss-flow-model)
  - [4.2 增强的 SLat Flow Model](#42-增强的-slat-flow-model)
  - [4.3 Flow 训练器](#43-flow-训练器)
- [5. 增强的 Pipeline](#5-增强的-pipeline)
- [6. 训练配置示例](#6-训练配置示例)
- [7. 常见问题](#7-常见问题)

---

## 1. 项目结构

```
lato_integration/
├── __init__.py                       # 统一导出
│
├── utils.py                          # DiagonalGaussianDistribution (VAE后验)
├── base.py                           # SparseTransformerCrossBase (交叉注意力)
├── vertex_encoder.py                 # ConnectionHead (边预测)
│
├── sparse_structure_vae.py           # [阶段一] 增强的 SS Encoder/Decoder
├── encoder.py                        # [阶段一] 增强的 SLat Encoder
├── decoder_gs.py                     # [阶段一] 增强的 Gaussian Decoder
├── decoder_rf.py                     # [阶段一] 增强的 Radiance Field Decoder
├── decoder_mesh.py                   # [阶段一] 增强的 Mesh Decoder
│
├── trainers/                         # [阶段一] VAE 训练器
│   ├── sparse_structure_vae.py       #   SS VAE 训练器
│   ├── slat_vae_gaussian.py          #   SLat Gaussian 训练器
│   ├── slat_vae_rf_dec.py            #   SLat RF Decoder 训练器
│   └── slat_vae_mesh_dec.py          #   SLat Mesh Decoder 训练器
│
├── flow/                             # [阶段二] Flow/DiT 优化
│   ├── ss_flow.py                    #   增强的 SS Flow Model
│   ├── slat_flow.py                  #   增强的 SLat Flow Model
│   └── trainers/
│       ├── ss_flow_trainer.py        #   SS Flow 训练器
│       └── slat_flow_trainer.py      #   SLat Flow 训练器
│
└── pipeline.py                       # 增强的 Text-to-3D Pipeline
```

---

## 2. 如何在 TRELLIS 上使用

### 前提: 设置 PYTHONPATH

```bash
export PYTHONPATH="/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
```

### 方式一：一键替换 (最推荐)

使用 `run_train.py` — 自动将 TRELLIS JSON 配置中的模型/训练器名映射到 LATO 增强版。

```bash
# 使用原始 TRELLIS JSON config, 自动替换为 LATO 增强模型
python lato_integration/run_train.py \
    --config configs/vae/ss_vae_conv3d_16l8_fp16.json \
    --data_dir ./data/ \
    --output_dir ./output/lato_ss_vae/

# 不想用 LATO? 加 --no_lato 回退到原始 TRELLIS
python lato_integration/run_train.py \
    --config configs/vae/ss_vae_conv3d_16l8_fp16.json \
    --data_dir ./data/ \
    --output_dir ./output/vanilla_ss_vae/ \
    --no_lato
```

**工作原理**: JSON config 中的 `"name": "SparseStructureEncoder"` 被自动映射为 `EnhancedSparseStructureEncoder`。

**支持的自动映射** (19 个类):

| JSON 中的 name | 自动映射到 |
|----------------|-----------|
| `SparseStructureEncoder` | `EnhancedSparseStructureEncoder` |
| `SparseStructureDecoder` | `EnhancedSparseStructureDecoder` |
| `SLatEncoder` | `EnhancedSLatEncoder` |
| `SLatGaussianDecoder` | `EnhancedSLatGaussianDecoder` |
| `SLatRadianceFieldDecoder` | `EnhancedSLatRadianceFieldDecoder` |
| `SLatMeshDecoder` | `EnhancedSLatMeshDecoder` |
| `SparseStructureFlowModel` | `EnhancedSSFlowModel` |
| `SLatFlowModel` | `EnhancedSLatFlowModel` |
| `SparseStructureVaeTrainer` | `EnhancedSparseStructureVaeTrainer` |
| `SLatVaeGaussianTrainer` | `EnhancedSLatVaeGaussianTrainer` |
| ... | (共 19 个映射) |

---

### 方式二：修改 JSON config (更灵活)

直接修改 TRELLIS JSON 配置, 填入完整的 LATO 类路径。适合需要精细控制增强参数时使用。

```json
{
    "models": {
        "encoder": {
            "name": "lato_integration.encoder.EnhancedSLatEncoder",
            "args": {
                "resolution": 64,
                "in_channels": 1024,
                "model_channels": 512,
                "latent_channels": 8,
                "num_blocks": 8,
                "num_heads": 8,
                "attn_mode": "swin",
                "use_fp16": true
            }
        },
        "decoder": {
            "name": "lato_integration.decoder_gs.EnhancedSLatGaussianDecoder",
            "args": {
                "resolution": 64,
                "model_channels": 512,
                "latent_channels": 8,
                "num_blocks": 4,
                "num_heads": 8,
                "use_cross_attn": true,
                "cross_attn_num_blocks": 2
            }
        }
    }
}
```

然后需要修改 `train.py` 的模型加载逻辑, 支持点号路径:

```python
# 在 train.py 的 main() 中, 将:
cls = getattr(models, name)
# 改为:
if '.' in name:
    import importlib
    module_path, cls_name = name.rsplit('.', 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
else:
    cls = getattr(models, name)
```

---

### 方式三：代码中手动导入 (最灵活)

完全绕过 TRELLIS 的 JSON 配置, 直接在 Python 脚本中构建模型和训练器。

```python
import sys
sys.path.insert(0, 'D:/code/TRELLIS-main/TRELLIS-main')
sys.path.insert(0, 'D:/code/TRELLIS-main')

from lato_integration import EnhancedSLatEncoder, EnhancedSLatMeshDecoder
from lato_integration.trainers import EnhancedSLatVaeMeshDecoderTrainer

# 创建模型
encoder = EnhancedSLatEncoder(
    resolution=64, in_channels=1024, model_channels=512,
    latent_channels=8, num_blocks=8, num_heads=8,
    attn_mode='swin', use_fp16=True,
)
decoder = EnhancedSLatMeshDecoder(
    resolution=64, model_channels=512, latent_channels=8,
    num_blocks=4, num_heads=8,
    use_cross_attn=True, use_pruning=True, use_edge_pred=True,
    representation_config={...},
)

# 创建训练器
trainer = EnhancedSLatVaeMeshDecoderTrainer(
    models={'encoder': encoder, 'decoder': decoder},
    dataset=dataset,
    output_dir='./output/',
    max_steps=500000,
    batch_size_per_gpu=4,
    optimizer={'name': 'AdamW', 'args': {'lr': 1e-4}},
)
trainer.run()
```

---

### 快速测试: 验证环境

```python
# test_lato.py — 运行此脚本确认环境 OK
import sys
sys.path.insert(0, 'D:/code/TRELLIS-main/TRELLIS-main')
sys.path.insert(0, 'D:/code/TRELLIS-main')

print("1. 检查 LATO 导入...")
from lato_integration import (
    EnhancedSparseStructureEncoder,
    EnhancedSparseStructureDecoder,
    EnhancedSLatEncoder,
    EnhancedSLatGaussianDecoder,
    EnhancedSLatRadianceFieldDecoder,
    EnhancedSLatMeshDecoder,
)
from lato_integration.flow import EnhancedSSFlowModel, EnhancedSLatFlowModel
from lato_integration.trainers import (
    EnhancedSparseStructureVaeTrainer,
    EnhancedSLatVaeGaussianTrainer,
)
print("   ✅ 所有导入成功")

print("\n2. 检查 DiagonalGaussianDistribution...")
from lato_integration import DiagonalGaussianDistribution
import torch
params = torch.randn(4, 16)
posterior = DiagonalGaussianDistribution(params)
z = posterior.sample()
kl = posterior.kl()
print(f"   ✅ sample shape={z.shape}, kl={kl.item():.4f}")

print("\n3. 检查原始 TRELLIS 导入...")
from trellis.models.sparse_structure_vae import SparseStructureEncoder
print(f"   ✅ SparseStructureEncoder 可用")

print("\n4. 检查继承关系...")
assert issubclass(EnhancedSparseStructureEncoder, SparseStructureEncoder)
print("   ✅ EnhancedSparseStructureEncoder 继承正确")

print("\n🎉 环境验证通过!")
```

---

## 3. 阶段一：VAE 优化

> **作用范围：** TRELLIS 训练步骤 1-4（VAE 编码器/解码器训练）
>
> **核心思想：** 改进 VAE 的压缩/重建质量 → 产生更好的 latent → Flow 模型学得更好

### 3.1 增强的 SS Encoder/Decoder

**文件：** `sparse_structure_vae.py`

| 类名 | 父类 | 功能 |
|------|------|------|
| `EnhancedSparseStructureEncoder` | `SparseStructureEncoder` | 将 64³ 占用网格压缩为 8×16×16×16 latent |
| `EnhancedSparseStructureDecoder` | `SparseStructureDecoder` | 从 latent 重建 64³ 占用网格 |

**LATO 增强点：**

| 特性 | 原始 | 增强后 |
|------|------|--------|
| VAE 后验 | `mean, logvar = chunk(h, 2)` | `DiagonalGaussianDistribution` — 带 clamp 的稳定后验 |
| KL 损失 | `0.5 * (mean² + exp(logvar) - logvar - 1)` | `posterior.kl()` — 支持非标准先验 |
| Decoder 输出 | 直接 logits | 可选 pruning head 输出 occupancy |

**使用示例：**

```python
from lato_integration import (
    EnhancedSparseStructureEncoder,
    EnhancedSparseStructureDecoder,
)

# 创建模型（参数与原始 SparseStructureEncoder 完全兼容）
encoder = EnhancedSparseStructureEncoder(
    in_channels=1,
    latent_channels=8,
    num_res_blocks=2,
    channels=[64, 128, 256],
    num_res_blocks_middle=2,
    norm_type="layer",
    use_fp16=True,
)

decoder = EnhancedSparseStructureDecoder(
    out_channels=1,
    latent_channels=8,
    num_res_blocks=2,
    channels=[64, 128, 256],
    norm_type="layer",
    use_fp16=True,
    use_pruning=True,  # LATO 增强：启用 pruning
)

# 前向传播
x = torch.randn(2, 1, 64, 64, 64)  # GT occupancy
z, posterior = encoder(x, sample_posterior=True, return_raw=True)
logits = decoder(z)

# KL 损失
kl_loss = posterior.kl()  # 替代手写 KL 公式

# 带 pruning 的输出
logits, pruning_logits = decoder(z, return_pruning_logits=True)
```

---

### 3.2 增强的 SLat Encoder

**文件：** `encoder.py`

| 类名 | 父类 | 功能 |
|------|------|------|
| `EnhancedSLatEncoder` | `SLatEncoder` | 将 DINOv2 特征 (1024维) 压缩为 8 维 sparse latent |

**LATO 增强点：**

| 特性 | 原始 | 增强后 |
|------|------|--------|
| VAE 后验 | 手动 `mean, logvar = chunk(feats, -1)` | `DiagonalGaussianDistribution(feats, feat_dim=-1)` |
| 采样 | 手动 `std = exp(0.5*logvar); z = mean + std*randn` | `posterior.sample()` |
| 返回值 | `(z, mean, logvar)` | `(z, posterior)` — 可直接调用 `.kl()` |

**使用示例：**

```python
from lato_integration import EnhancedSLatEncoder

encoder = EnhancedSLatEncoder(
    resolution=64,
    in_channels=1024,      # DINOv2 features
    model_channels=512,
    latent_channels=8,
    num_blocks=8,
    num_heads=8,
    attn_mode="swin",
    window_size=8,
    use_fp16=True,
)

# 前向传播
feats = sp.SparseTensor(feats=..., coords=...)  # DINOv2 sparse features
z, posterior = encoder(feats, sample_posterior=True, return_raw=True)

# KL 损失
kl_loss = posterior.kl()
```

---

### 3.3 增强的 Gaussian Decoder

**文件：** `decoder_gs.py`

| 类名 | 父类 | 功能 |
|------|------|------|
| `EnhancedSLatGaussianDecoder` | `SLatGaussianDecoder` | 从 SLat 解码为 3D Gaussian Splats |
| `EnhancedElasticSLatGaussianDecoder` | 同上 + Elastic | 低显存版本 |

**LATO 增强点：**

| 特性 | 原始 | 增强后 |
|------|------|--------|
| 注意力 | 仅 self-attention | self-attention + **cross-attention 回原始 latent** |
| Latent 传递 | 只用一次 | decoder 内部重新关注原始 latent 信息 |

**使用示例：**

```python
from lato_integration import EnhancedSLatGaussianDecoder

decoder = EnhancedSLatGaussianDecoder(
    resolution=64,
    model_channels=512,
    latent_channels=8,
    num_blocks=4,
    num_heads=8,
    attn_mode="swin",
    use_cross_attn=True,        # LATO 增强：启用 cross-attention
    cross_attn_num_blocks=2,    # cross-attention 层数
    representation_config={...},
)

# 前向传播（带 cross-attention）
gaussians = decoder(z, original_latent=z)  # 传入原始 latent

# 兼容模式（不加 cross-attention）
gaussians = decoder(z)  # 与原始 decoder 行为一致
```

---

### 3.4 增强的 Radiance Field Decoder

**文件：** `decoder_rf.py`

| 类名 | 父类 | 功能 |
|------|------|------|
| `EnhancedSLatRadianceFieldDecoder` | `SLatRadianceFieldDecoder` | 从 SLat 解码为 Strivec 辐射场 |
| `EnhancedElasticSLatRadianceFieldDecoder` | 同上 + Elastic | 低显存版本 |

**LATO 增强点：** 与 Gaussian Decoder 相同（cross-attention 回原始 latent）

**使用方式与 3.3 完全一致。**

---

### 3.5 增强的 Mesh Decoder

**文件：** `decoder_mesh.py` — **最重要的增强**

| 类名 | 功能 |
|------|------|
| `EnhancedSLatMeshDecoder` | 从 SLat 解码为 Mesh（FlexiCubes） |
| `EnhancedElasticSLatMeshDecoder` | 低显存版本 |
| `EnhancedSparseSubdivideBlock3d` | 带 occupancy pruning 的上采样块 |
| `SparsePredictionHead` | 用于 occupancy 预测的小 MLP |

**LATO 增强点（3个）：**

```
原始流程:  latent → SparseTransformerBase → Upsample(128) → Upsample(256) → FlexiCubes → Mesh

增强流程:  latent → SparseTransformerBase
                → [Cross-Attention 回 latent]          ← 增强1: 交叉注意力
                → SubdivideBlock + [Pruning Head]      ← 增强2: occupancy pruning
                → SubdivideBlock + [Pruning Head]      ← 增强2: occupancy pruning
                → FlexiCubes → Mesh
                → [ConnectionHead 预测边]              ← 增强3: 边拓扑预测
```

**使用示例：**

```python
from lato_integration import EnhancedSLatMeshDecoder

decoder = EnhancedSLatMeshDecoder(
    resolution=64,
    model_channels=512,
    latent_channels=8,
    num_blocks=4,
    num_heads=8,
    attn_mode="swin",
    # === LATO 增强开关 ===
    use_cross_attn=True,         # 增强1: cross-attention
    cross_attn_num_blocks=2,
    use_pruning=True,            # 增强2: occupancy pruning
    use_edge_pred=True,          # 增强3: edge prediction
    representation_config={...},
)

# 前向传播（全部增强）
meshes, pruning_data = decoder(
    z,
    original_latent=z,           # 用于 cross-attention
    return_pruning=True,         # 返回中间 pruning 数据（训练用）
)

# 预测边
edge_probs = decoder.predict_edges(vertex_feats, edge_candidates)
```

---

## 4. 阶段二：Flow/DiT 优化

> **作用范围：** TRELLIS 训练步骤 5-6（Flow Matching / DiT 训练）
>
> **核心思想：** 改进 DiT 架构 + 添加辅助损失 → Flow 模型生成更高质量的 latent

### 4.1 增强的 SS Flow Model

**文件：** `flow/ss_flow.py`

| 类名 | 父类 | 张量类型 |
|------|------|----------|
| `EnhancedSSFlowModel` | `SparseStructureFlowModel` | Dense 16³ |

**LATO 增强点：**

```
原始:  Linear → [Full Attention × N] → LayerNorm → Linear → Unpatchify

增强:  Linear → [IO ResBlocks]          ← 新增: 多尺度特征提取
            → [Swin Window Attention × N] ← 改进: 交替窗口注意力
            → LayerNorm → Linear → Unpatchify
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_swin_attn=True` | True | 使用 Swin window attention（奇偶 block 交替窗口偏移） |
| `window_size=4` | 4 | 窗口大小（16³ grid 上 window_size=4） |
| `num_io_res_blocks=0` | 0 | IO ResBlock 数量（0=禁用，1-2 推荐） |
| `io_block_channels=None` | None | IO block 通道列表，如 `[256, 512]` |

**使用示例：**

```python
from lato_integration.flow import EnhancedSSFlowModel

model = EnhancedSSFlowModel(
    resolution=16,
    in_channels=8,
    model_channels=1024,
    cond_channels=768,
    out_channels=8,
    num_blocks=24,
    num_heads=16,
    # === LATO 增强参数 ===
    use_swin_attn=True,
    window_size=4,
    num_io_res_blocks=2,
    io_block_channels=[512, 768],
)

# 前向传播（与原始接口完全一致）
output = model(noise, timestep, text_conditioning)
```

---

### 4.2 增强的 SLat Flow Model

**文件：** `flow/slat_flow.py` — **最显著的架构改进**

| 类名 | 父类 | 张量类型 |
|------|------|----------|
| `EnhancedSLatFlowModel` | `SLatFlowModel` | Sparse 64³ |
| `EnhancedElasticSLatFlowModel` | 同上 + Elastic | 低显存版本 |

**LATO 增强点（3个）：**

```
原始架构:                         增强架构:
  Input (8ch)                       Input (8ch)
    ↓                                 ↓
  IO: [128] × 2                    IO: [128→256→512] × 3    ← 增强1: 多级层级
    ↓                                 ↓
  PE (共享)                        PE (query独立 + ctx独立)  ← 增强2: 分离位置编码
    ↓                                 ↓
  Full Attention × N               Swin Window Attn × N     ← 增强3: 窗口注意力
    ↓                                 ↓
  IO: skip + upsample              IO: skip + upsample
    ↓                                 ↓
  Output (8ch)                     Output (8ch)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_swin_attn=True` | True | Swin window attention（与 LATO 骨干一致） |
| `window_size=8` | 8 | Swin 窗口大小 |
| `use_separate_cross_pe=True` | True | query/context 各自独立的位置编码 |
| `io_block_channels=None` | 自动 `[model//2]` | 增强后自动构建多层 hierarchy |
| `num_io_res_blocks=2` | 2 | 每层 IO block 数量 |

**使用示例：**

```python
from lato_integration.flow import EnhancedSLatFlowModel

model = EnhancedSLatFlowModel(
    resolution=64,
    in_channels=8,
    model_channels=1024,
    cond_channels=768,
    out_channels=8,
    num_blocks=24,
    num_heads=16,
    patch_size=2,
    # === LATO 增强参数 ===
    use_swin_attn=True,
    window_size=8,
    use_separate_cross_pe=True,
    num_io_res_blocks=3,                          # 3层IO (vs 原始2层)
    io_block_channels=[128, 256, 512],            # 多级hierarchy (vs 原始[128])
)

# 前向传播（与原始接口完全一致）
x = sp.SparseTensor(feats=noise, coords=active_voxels)
output = model(x, timestep, text_conditioning)
```

---

### 4.3 Flow 训练器

**文件：** `flow/trainers/`

#### SS Flow 训练器 (`ss_flow_trainer.py`)

| 类名 | 功能 |
|------|------|
| `EnhancedSSFlowTrainer` | 基础增强训练器 |
| `EnhancedSSFlowCFGTrainer` | + Classifier-Free Guidance |
| `TextConditionedEnhancedSSFlowCFGTrainer` | + 文本条件 |
| `ImageConditionedEnhancedSSFlowCFGTrainer` | + 图像条件 |

**LATO 增强：辅助 VAE 解码损失**

每 N 步将预测 latent 通过冻结的 SS Decoder 解码，计算 occupancy BCE loss：

```python
trainer = TextConditionedEnhancedSSFlowCFGTrainer(
    models={'denoiser': flow_model, 'decoder': ss_decoder},
    dataset=dataset,
    # === LATO 增强参数 ===
    aux_decode_every=100,       # 每100步计算一次辅助损失
    lambda_aux_decode=0.1,      # 辅助损失权重
    # === 原始参数 ===
    t_schedule={'name': 'logitNormal', 'args': {'mean': 0.0, 'std': 1.0}},
    sigma_min=1e-5,
    p_uncond=0.1,
    text_cond_model='openai/clip-vit-large-patch14',
)

# 训练数据需要包含 ss_gt (GT occupancy grid)
# data = {'x_0': ss_latent, 'ss_gt': occupancy_grid, 'caption': '...'}
```

#### SLat Flow 训练器 (`slat_flow_trainer.py`)

| 类名 | 功能 |
|------|------|
| `EnhancedSLatFlowTrainer` | 基础增强训练器 |
| `EnhancedSLatFlowCFGTrainer` | + CFG |
| `TextConditionedEnhancedSLatFlowCFGTrainer` | + 文本条件 |

**LATO 增强（2个）：**

1. **辅助 VAE 解码损失** — 定期 decode 预测 SLat 并计算渲染损失
2. **Latent 一致性损失** — KL-like 约束使 flow 分布接近 VAE posterior

```python
trainer = TextConditionedEnhancedSLatFlowCFGTrainer(
    models={'denoiser': flow_model, 'decoder': slat_decoder},
    dataset=dataset,
    # === LATO 增强参数 ===
    aux_decode_every=200,            # 每200步计算辅助解码损失
    lambda_aux_decode=0.05,          # 辅助解码损失权重
    lambda_latent_consistency=1e-4,  # latent 一致性损失权重
    # === 原始参数 ===
    t_schedule={'name': 'logitNormal', 'args': {'mean': 0.0, 'std': 1.0}},
    sigma_min=1e-5,
    p_uncond=0.1,
    text_cond_model='openai/clip-vit-large-patch14',
)
```

---

## 5. 增强的 Pipeline

**文件：** `pipeline.py`

| 类名 | 父类 | 功能 |
|------|------|------|
| `EnhancedTrellisTextTo3DPipeline` | `TrellisTextTo3DPipeline` | 文本/图像 → 3D 模型 |

**LATO 增强：** `decode_slat()` 自动传递原始 latent 给所有增强 decoder 做 cross-attention

```python
from lato_integration import EnhancedTrellisTextTo3DPipeline

# 从预训练模型加载
pipeline = EnhancedTrellisTextTo3DPipeline.from_pretrained(
    "JeffreyXiang/TRELLIS-image-large"
)

# 文本转 3D（自动使用 cross-attention）
results = pipeline.run(
    "a wooden chair with armrests",
    num_samples=1,
    seed=42,
    formats=['mesh', 'gaussian', 'radiance_field'],
)

# 访问结果
mesh = results['mesh'][0]          # MeshExtractResult
gaussian = results['gaussian'][0]  # Gaussian splats
```

---

## 6. 训练配置示例

### 完整训练流程（6 步）

```python
# ===================================================
# 步骤 1: SS VAE 训练 (阶段一优化)
# ===================================================
from lato_integration import (
    EnhancedSparseStructureEncoder,
    EnhancedSparseStructureDecoder,
)
from lato_integration.trainers import EnhancedSparseStructureVaeTrainer

ss_encoder = EnhancedSparseStructureEncoder(
    in_channels=1, latent_channels=8, num_res_blocks=2,
    channels=[64, 128, 256], use_fp16=True,
)
ss_decoder = EnhancedSparseStructureDecoder(
    out_channels=1, latent_channels=8, num_res_blocks=2,
    channels=[64, 128, 256], use_fp16=True, use_pruning=True,
)
trainer_1 = EnhancedSparseStructureVaeTrainer(
    models={'encoder': ss_encoder, 'decoder': ss_decoder},
    dataset=ss_dataset,
    loss_type='bce', lambda_kl=1e-6,
)
trainer_1.run()


# ===================================================
# 步骤 2: SLat VAE 训练 (阶段一优化)
# ===================================================
from lato_integration import EnhancedSLatEncoder, EnhancedSLatGaussianDecoder
from lato_integration.trainers import EnhancedSLatVaeGaussianTrainer

slat_encoder = EnhancedSLatEncoder(
    resolution=64, in_channels=1024, model_channels=512,
    latent_channels=8, num_blocks=8, num_heads=8,
    attn_mode='swin', use_fp16=True,
)
slat_gs_decoder = EnhancedSLatGaussianDecoder(
    resolution=64, model_channels=512, latent_channels=8,
    num_blocks=4, num_heads=8, use_cross_attn=True,
    representation_config={...},
)
trainer_2 = EnhancedSLatVaeGaussianTrainer(
    models={'encoder': slat_encoder, 'decoder': slat_gs_decoder},
    dataset=slat_dataset,
    lambda_kl=1e-6,
)
trainer_2.run()


# ===================================================
# 步骤 3-4: SLat Decoder 训练 (冻结 encoder)
# ===================================================
from lato_integration import (
    EnhancedSLatRadianceFieldDecoder,
    EnhancedSLatMeshDecoder,
)
from lato_integration.trainers import (
    EnhancedSLatVaeRadianceFieldDecoderTrainer,
    EnhancedSLatVaeMeshDecoderTrainer,
)

# ... 创建 decoder 和 trainer ...


# ===================================================
# 步骤 5: SS Flow 训练 (阶段二优化)
# ===================================================
from lato_integration.flow import EnhancedSSFlowModel
from lato_integration.flow.trainers import TextConditionedEnhancedSSFlowCFGTrainer

ss_flow = EnhancedSSFlowModel(
    resolution=16, in_channels=8, model_channels=1024,
    cond_channels=768, out_channels=8, num_blocks=24,
    num_heads=16, use_swin_attn=True, window_size=4,
    num_io_res_blocks=2, io_block_channels=[512, 768],
)
trainer_5 = TextConditionedEnhancedSSFlowCFGTrainer(
    models={'denoiser': ss_flow, 'decoder': ss_decoder},
    dataset=ss_latent_dataset,
    aux_decode_every=100, lambda_aux_decode=0.1,
    p_uncond=0.1, text_cond_model='openai/clip-vit-large-patch14',
)
trainer_5.run()


# ===================================================
# 步骤 6: SLat Flow 训练 (阶段二优化)
# ===================================================
from lato_integration.flow import EnhancedSLatFlowModel
from lato_integration.flow.trainers import TextConditionedEnhancedSLatFlowCFGTrainer

slat_flow = EnhancedSLatFlowModel(
    resolution=64, in_channels=8, model_channels=1024,
    cond_channels=768, out_channels=8, num_blocks=24,
    num_heads=16, patch_size=2,
    use_swin_attn=True, window_size=8,
    use_separate_cross_pe=True,
    num_io_res_blocks=3, io_block_channels=[128, 256, 512],
)
trainer_6 = TextConditionedEnhancedSLatFlowCFGTrainer(
    models={'denoiser': slat_flow, 'decoder': slat_gs_decoder},
    dataset=slat_latent_dataset,
    aux_decode_every=200, lambda_aux_decode=0.05,
    lambda_latent_consistency=1e-4,
    p_uncond=0.1, text_cond_model='openai/clip-vit-large-patch14',
)
trainer_6.run()
```

---

## 7. 常见问题

### Q: 增强类会破坏原有 checkpoint 兼容性吗？

部分会。架构有变化时（如 `use_swin_attn=True`、`io_block_channels` 变化），参数名/形状不同，需从头训练。以下情况兼容：

| 增强 | checkpoint 兼容 | 说明 |
|------|:---:|------|
| `EnhancedSparseStructureEncoder` (仅后验改进) | ✅ | 架构不变，只改 forward 逻辑 |
| `EnhancedSLatEncoder` (仅后验改进) | ✅ | 同上 |
| `EnhancedSLatGaussianDecoder` (cross_attn=False) | ✅ | `use_cross_attn=False` 时架构不变 |
| `EnhancedSLatGaussianDecoder` (cross_attn=True) | ❌ | 新增 cross-attn 层，参数不匹配 |
| `EnhancedSSFlowModel` (use_swin_attn=True) | ❌ | attention block 结构变化 |
| `EnhancedSLatFlowModel` (多级IO) | ❌ | IO block 通道数变化 |

### Q: 如何关闭增强回到原始行为？

```python
# 关闭所有 LATO 增强
decoder = EnhancedSLatGaussianDecoder(..., use_cross_attn=False)
flow = EnhancedSLatFlowModel(..., use_swin_attn=False, use_separate_cross_pe=False)
trainer = EnhancedSLatFlowTrainer(..., aux_decode_every=0, lambda_latent_consistency=0)
```

### Q: 辅助损失会增加多少训练时间？

- 辅助解码损失：约 +5-10%（取决于 decoder 复杂度和解码频率）
- Latent 一致性损失：约 +1-2%（仅多一次 MSE 前向计算）

### Q: 建议的训练配置？

| 显存 | 推荐配置 |
|------|----------|
| 24GB (RTX 4090) | Base 模型, `use_swin_attn=True`, `aux_decode_every=200` |
| 48GB (A6000) | Large 模型, 全部增强, `aux_decode_every=100` |
| 80GB (A100) | XLarge 模型, 全部增强, 3层IO hierarchy |
