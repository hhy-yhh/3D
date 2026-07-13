# TRELLIS + LATO 完整训练步骤

> **目标：文本 → Mesh（只需要 Mesh 输出，跳过 Radiance Field）**

---

## 回答：只需要 Mesh，还要训练全部吗？

**不需要全部。** 步骤3（Radiance Field Decoder）可以跳过，其他 5 步都必须训练：

| 步骤 | 内容 | Mesh 是否需要 | 原因 |
|:---:|---|---|---|
| 1 | SS VAE | ✅ 必须 | 为 SS Flow 提供 latent 训练数据 |
| 2 | SLat VAE Gaussian | ✅ 必须 | 训练 Encoder（Mesh Decoder 和 SLat Flow 共用） |
| 3 | SLat VAE RF Decoder | ❌ 跳过 | 仅辐射场输出需要 |
| 4 | SLat VAE Mesh Decoder | ✅ 必须 | 最终输出 Mesh |
| 5 | SS Flow | ✅ 必须 | 从文本生成 SS latent |
| 6 | SLat Flow | ✅ 必须 | 从 SS latent + 文本生成 SLat |

---

## 依赖关系

```
步骤1: SS VAE ──────────────────────────┐
     ↓                                   │
步骤2: SLat VAE Gaussian ─────────────┐  │
     ↓ (冻结 Encoder)                  │  │
步骤4: SLat VAE Mesh Decoder ────┐     │  │
     ↓                            │     │  │
步骤5: SS Flow ←── 依赖步骤1 latent  │  │
     ↓                            │     │  │
步骤6: SLat Flow ←── 依赖步骤2 Encoder + 步骤4 Mesh Decoder + 步骤5 SS Flow
     ↓
推理: 文本 → 3D Mesh
```

---

## 环境准备

```bash
# 设置 PYTHONPATH
export PYTHONPATH="/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"

# 公共参数
DATA_DIR="/data/huanghaoyang/3D/database_lato/train"
OUTPUT_BASE="/data/huanghaoyang/3D/TRELLIS/outputs"
GPU=1
```

---

## 阶段一：VAE 训练

### 步骤1：SS VAE（稀疏结构 VAE）

**作用：** 把 64³ 的 occupancy grid 压缩为 8×16³ 的 latent，供步骤5 SS Flow 训练使用。

**LATO 增强点：** `DiagonalGaussianDistribution` 替代手动 KL + `pruning_head` 辅助 occupancy 预测。

```bash
python lato_integration/run_train.py \
    --config configs/vae/ss_vae_conv3d_16l8_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_ss_vae \
    --num_gpus ${GPU} \
    --auto_retry 0
```

**产出模型：**
- `ss_enc_conv3d_16l8_fp16` — SS Encoder
- `ss_dec_conv3d_16l8_fp16` — SS Decoder

---

### 步骤2：SLat VAE Gaussian（结构化 Latent VAE — Encoder + Gaussian Decoder）

**作用：** 训练 SLat Encoder（将 DINOv2 1024维特征压缩为 8维 sparse latent）和 Gaussian Decoder。Encoder 是步骤4和步骤6的基础。

**LATO 增强点：** `DiagonalGaussianDistribution` + `cross-attn` 回原始 latent。

```bash
python lato_integration/run_train.py \
    --config configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_slat_vae_gs \
    --num_gpus ${GPU} \
    --auto_retry 0
```

**产出模型：**
- `dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16` — SLat Encoder（步骤4、6 共用）
- `slat_dec_gs_swin8_B_64l8gs32_fp16` — Gaussian Decoder（步骤6 辅助损失用）

---

### 步骤3：SLat VAE RF Decoder — ❌ 跳过

> 仅 Radiance Field 输出需要此步骤。Mesh 用户无需训练。

---

### 步骤4：SLat VAE Mesh Decoder（冻结 Encoder，只训练 Mesh Decoder）

**作用：** 冻结步骤2的 Encoder，训练 Mesh Decoder（FlexiCubes），是最终导出 Mesh 的关键模块。

**LATO 增强点（最密集）：** cross-attn + `EnhancedSparseSubdivideBlock3d` occupancy pruning + `ConnectionHead` 边预测。

**前置条件：** 步骤2 训练完成，有 Encoder checkpoint。

```bash
python lato_integration/run_train.py \
    --config configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_slat_vae_mesh \
    --num_gpus ${GPU} \
    --auto_retry 0
```

**产出模型：**
- `slat_dec_mesh_swin8_B_64l8_fp16` — Mesh Decoder

---

## 阶段二：Flow/DiT 训练

### 步骤5：SS Flow（文本 → SS Latent）

**作用：** 用步骤1冻结的 SS Encoder 产出的 latent 训练 DiT，实现从文本条件生成 SS latent。

**LATO 增强点：** Swin Window Attention（交替窗口偏移）+ IO ResBlocks + 辅助解码损失。

**前置条件：** 步骤1 训练完成。

```bash
python lato_integration/run_train.py \
    --config configs/generation/ss_flow_txt_dit_B_16l8_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_ss_flow \
    --num_gpus ${GPU} \
    --auto_retry 0
```

**产出模型：**
- `ss_flow_txt_dit_B_16l8_fp16` — SS Flow Model

---

### 步骤6：SLat Flow（SLat → 最终 Latent）

**作用：** 用步骤2冻结的 Encoder 产出的 SLat 训练 Sparse DiT，实现从文本条件 + SS latent 生成最终 SLat。

**LATO 增强点：** 多级 IO Hierarchy + 分离位置编码 + Swin Window Attention + 辅助解码损失 + Latent 一致性损失。

**前置条件：** 步骤2、步骤4、步骤5 训练完成。

```bash
python lato_integration/run_train.py \
    --config configs/generation/slat_flow_txt_dit_B_64l8p2_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_slat_flow \
    --num_gpus ${GPU} \
    --auto_retry 0
```

**产出模型：**
- `slat_flow_txt_dit_B_64l8p2_fp16` — SLat Flow Model

---

## 一键运行脚本

```bash
#!/bin/bash
set -e

export PYTHONPATH="/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"

DATA_DIR="/data/huanghaoyang/3D/database_lato/train"
OUTPUT_BASE="/data/huanghaoyang/3D/TRELLIS/outputs"
GPU=1

echo "========== 步骤1: SS VAE =========="
python lato_integration/run_train.py \
    --config configs/vae/ss_vae_conv3d_16l8_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_ss_vae \
    --num_gpus ${GPU} --auto_retry 0

echo "========== 步骤2: SLat VAE Gaussian =========="
python lato_integration/run_train.py \
    --config configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_slat_vae_gs \
    --num_gpus ${GPU} --auto_retry 0

echo "========== 步骤3: 跳过 (RF Decoder 不需要) =========="

echo "========== 步骤4: SLat VAE Mesh Decoder =========="
python lato_integration/run_train.py \
    --config configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_slat_vae_mesh \
    --num_gpus ${GPU} --auto_retry 0

echo "========== 步骤5: SS Flow =========="
python lato_integration/run_train.py \
    --config configs/generation/ss_flow_txt_dit_B_16l8_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_ss_flow \
    --num_gpus ${GPU} --auto_retry 0

echo "========== 步骤6: SLat Flow =========="
python lato_integration/run_train.py \
    --config configs/generation/slat_flow_txt_dit_B_64l8p2_fp16.json \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_BASE}/lato_slat_flow \
    --num_gpus ${GPU} --auto_retry 0

echo "========== 全部完成 =========="
```

---

## 训练顺序与依赖总结

```
步骤1 (SS VAE)
  │
  ├──→ 步骤5 (SS Flow) ──→ 步骤6 (SLat Flow) ──→ 推理
  │                              ↑
  └──→ 步骤2 (SLat VAE GS) ────→ │
          │                      │
          └──→ 步骤4 (Mesh Dec)──┘

步骤3 (RF Dec) → 跳过 ✗
```

**必须按顺序执行**，后一步依赖前一步产出的 checkpoint。`run_train.py` 会自动把 config 里的 `SparseStructureEncoder` 等类名映射到 LATO 增强版（`EnhancedSparseStructureEncoder` 等），无需手动改 JSON。
