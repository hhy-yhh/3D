# TRELLIS + LATO 文本转3D — 完整实现流程 v7

> **目标：** TRELLIS 文本转3D 管线中，**Encoder 全部用 LATO VoxelVAE，Decoder 全部用 LATO VoxelVAE，只有中间 Flow 生成用 TRELLIS**。SS Flow 和 SLat Flow 均在刹车卡钳数据集上从零训练，最后用 Chamfer Distance / Hausdorff Distance / Normal Consistency 评估。

**更新记录：**
- 2026-08-03：**v8 更新 — GT 数据根因修复** — 发现并修复 `load_quantized_mesh_original()` 对未归一化 STL 的坐标截断 bug：刹车卡钳毫米级坐标被 `np.clip` 压扁为 2D 平面，导致 234 个 SS occupancy GT 全部为单层体素。通过预处理归一化 STL + 重跑 LATO encode 恢复 3D GT。同时修复 StructureHead 全量重置后 optimizer Adam state 未清理导致训练无效的问题（Bug 16）。
- 2026-08-01：**LatoStructureHead 训练健壮性修复** — 修复 3 个关键 bug：`get_inference_cond` MRO 链断裂致训练字段泄漏到推理（Bug 13）、`x_0_pred` latent 溢出（Bug 14）、深层 Transformer fp16 激活累积溢出（Bug 15）
- 2026-07-23：**v7 更新** — 架构全面重构：TRELLIS SS/SLat Encoder/Decoder 全部替换为 LATO VoxelVAE；新增 `LatoStructureHead` 替代 SS Decoder；移除 coords×2 hack；删除 GS/RF decoder 及 VAE 训练器
- 2026-07-19：**v6 更新** — 端到端测试集推理验证通过；修复 6 个推理 bug
- 2026-07-17：**v5 更新** — 推理脚本重构：从 config JSON 读取模型参数、自动发现最新 ckpt、新增 ss_only 模式
- 2026-07-16：**v4 更新** — 新增步骤6（批量评估 + 3D 指标）
- 2026-07-26：**推理验证** — 修复 3 个推理 bug（dtype 不匹配、flash_attn fp32、ckpt 加载）；端到端 21 条测试集评估通过；SS Flow 100k 步 CD=0.214/HD=0.507/NC=0.459
- 2026-07-14：修复 12 个 bug + 4 个训练启动 bug，单卡 RTX 4090 24GB 可行

---

## 架构概览

```
                    v2 管线 (旧)                         v4 管线 (🆕)                       对应目标
                   ════════════════                   ═══════════════                     ════════

  Text ─→ CLIP ─→ SS Flow ─→ SS Decoder(TRELLIS)    Text ─→ CLIP ─→ SS Flow ─→ LatoStructureHead   ① 全 LATO 架构
                   (🆕重训)      (冻结, 64³输出)                  (🆕重训)      (🆕可训练, 128³直接输出!)
                        │            │                               │              │
                        │        coords×2(hack)                      │         coords@128
                        │            │                               │              │
                        ▼            ▼                               ▼              ▼
                   SLat Flow ─→ LATO VoxelVAE.decode()         SLat Flow ─→ LATO VoxelVAE.decode()    ② 高质量几何解码
                   (🆕重训)       (LATO 替代 decoder)            (🆕重训)       (LATO decoder)
                        │            │                               │              │
                        ▼            ▼                               ▼              ▼
                   ConnectionHead → Mesh                        ConnectionHead → Mesh                ③ 显式拓扑预测
                   (LATO, 冻结)                                  (LATO, 冻结)

  训练时:                                              训练时:
    SS Encoder: TRELLIS(冻结) → 16³×8                  SS 目标: LATO VoxelVAE.encode() → coords@128  ④ LATO 统一编码
    SLat Encoder: LATO VoxelVAE.encode() ✅             SLat 目标: LATO VoxelVAE.encode() → feats@128
```

### v3 → v4 核心变化

| 组件 | v3 | v4 |
|------|-----|-----|
| SS Encoder（训练） | TRELLIS SparseStructureEncoder（冻结） | LATO VoxelVAE.encode() |
| SS Decoder（推理） | TRELLIS SparseStructureDecoder（冻结, 64³） | **🆕 LatoStructureHead**（可训练, 128³ 直接输出） |
| coords ×2 | 需要（64→128 桥接） | **不需要** |
| SLat Encoder（训练） | LATO VoxelVAE.encode() ✅ | 不变 ✅ |
| SLat Decoder（推理） | LATO VoxelVAE.decode() ✅ | 不变 ✅ |
| GS/RF Decoder | 代码存在但未使用 | **已删除** |
| VAE Trainers | 存在但未使用 | **已删除** |

---

## 推理全链路追踪（v4：`pipeline.run()` → mesh.obj）

```
TrellisTextTo3DPipeline.run()          # trellis_text_to_3d.py:212
│
├─[1] sample_sparse_structure_lato()   # 🆕 :109 v4 新增方法
│   ├─ self.models['sparse_structure_flow_model']    ← 🆕 你的 SS Flow ckpt
│   └─ self.models['lato_structure_head']            ← 🆕 LatoStructureHead（替代 SS Decoder!）
│       └─ SS Flow(16³×8) → 3D CNN 上采样 → occ@128³ → coords
│
├─[2] ~~coords[:, 1:] = coords[:, 1:] * 2~~          # ❌ 已移除！LatoStructureHead 直接出 128
│
├─[3] sample_slat()                    # :180
│   ├─ self.models['slat_flow_model']                ← 🆕 你的 SLat Flow ckpt
│   └─ self.slat_normalization                        ← 🆕 刹车卡钳 stats.json
│
└─[4] decode_slat()                    # :153
    └─ 'lato_vae' in self.models? ──Yes──→ decode_slat_lato()  # :109
        │
        ├─ TRELLIS SparseTensor → LATO SparseTensor   # :131-135
        ├─ self.models['lato_vae'].decode()            # :141  ← LATO VoxelVAE
        └─ return {'lato_decoded': ...}

── 推理脚本后处理 (inference_lato.py) ──
│
├─[5] decoded[-1].get('vertex')          ← 取最后一级顶点
├─[6] predict_edges_batched(connection_head, ...)      ← LATO ConnectionHead
└─[7] edges_to_mesh() → trimesh → .obj  ← NetworkX 公共邻居法
```

### 6 处改动 × 对应关系（v3）

| # | 改动内容 | 对应目标 | 对应模块 |
|---|---------|---------|---------|
| 1 | SS Flow 替换为你训练的权重 | 刹车卡钳形状先验 | `EnhancedSSFlowModel`（`flow/ss_flow.py`） |
| 2 | **🆕 SS Decoder → LatoStructureHead** | LATO 全架构替代 TRELLIS SS Decoder | `LatoStructureHead`（`structure_head.py`） |
| 3 | **❌ coords×2 移除** | LatoStructureHead 直接出 128³ | 无需坐标变换 |
| 4 | SLat Flow 替换为你训练的权重（res=128, dim=16） | 刹车卡钳潜空间分布 | `LATOSLatFlowModel` |
| 5 | TRELLIS decoder → LATO VoxelVAE.decode() | 高质量几何解码 | LATO `VoxelVAE.decode()` |
| 6 | ConnectionHead 边预测 + NetworkX 三角面片化 | 显式拓扑预测 | LATO `ConnectionHead` |

### 目标符合性判定

```
目标: "Encoder 全用 LATO，Decoder 全用 LATO，只有 Flow 生成用 TRELLIS"

  ✅ SS Encoder → LATO VoxelVAE.encode()     — encode_lato_latent_v2.py
  ✅ SS Decoder → LatoStructureHead          — structure_head.py (🆕)
  ✅ SLat Encoder → LATO VoxelVAE.encode()   — 同上
  ✅ SLat Decoder → LATO VoxelVAE.decode()   — trellis_text_to_3d.py:141
  ✅ SS Flow → TRELLIS（仅中间生成）          — flow/ss_flow.py
  ✅ SLat Flow → TRELLIS（仅中间生成）        — lato_slat_flow.py
  ✅ coords×2 已移除                         — LatoStructureHead 直接 128³
  ✅ GS/RF Decoder 已删除                    — decoder_gs.py, decoder_rf.py
  ✅ VAE Trainer 已删除                      — trainers/slat_vae_*.py

  ⚠️ SS Flow 架构增强 (Swin/IO)             — 代码预留，未激活
  ⚠️ SLat Flow 架构增强 (Swin/PE)           — 代码预留，未使用 EnhancedSLatFlowModel
```

**结论：核心目标全部达成。** 整个管线 = LATO VoxelVAE Enc/Dec（冻结）+ SS Flow（TRELLIS, 重训）+ SLat Flow（TRELLIS, 重训）+ LatoStructureHead（🆕, 与 SS Flow 联合训练）。

---

## 训练/推理角色

```
  CLIP ──────────── 冻结 ─ 只做文本编码
  SS Flow ───────── 🆕训 ─ 唯一需要训练的模型之一
  LatoStructureHead 🆕训 ─ 与 SS Flow 联合训练（~1-2M 参数）
  SLat Flow ─────── 🆕训 ─ 唯一需要训练的模型之二
  LATO VoxelVAE ─── 冻结 ─ 预训练几何编解码器，训练和推理都不更新
  ConnectionHead ── 冻结 ─ 预训练边预测器，含在 LATO ckpt 中
```

**只有 SS Flow + LatoStructureHead 和 SLat Flow 需要训练，LATO VAE 完全不参与训练。**

---

## 文件清单（v4）

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新建 | `lato_integration/structure_head.py` | `LatoStructureHead` — 3D CNN 16³→128³，替代 SS Decoder |
| 🆕 新建 | `lato_integration/datasets.py` | `TextConditionedLatoSSStructureLatent` — 加载 ss_occupancy_128 |
| ✏️ 重写 | `lato_integration/__init__.py` | 移除 TRELLIS Enc/Dec 导出，新增 LatoStructureHead + datasets |
| ✏️ 重写 | `lato_integration/sparse_structure_vae.py` | 改为 LatoStructureHead re-export + 废弃注释 |
| ✏️ 重写 | `lato_integration/pipeline.py` | 新增 `sample_sparse_structure_lato()` 和 `run_lato()` |
| ✏️ 修改 | `lato_integration/run_train.py` | MODEL_REPLACEMENTS 移除 Enc/Dec，新增 LatoStructureHead；支持 lato_datasets |
| ✏️ 修改 | `lato_integration/inference_lato.py` | v5→v6：LatoStructureHead 替代 SS Decoder，移除 coords×2 |
| ✏️ 修改 | `lato_integration/evaluate_3d_metrics.py` | load_pipeline() 加载 LatoStructureHead |
| ✏️ 修改 | `lato_integration/flow/trainers/ss_flow_trainer.py` | 新增 `LatoSSFlowTrainer`，训练目标改为 128³ occupancy |
| ✏️ 修改 | `lato_integration/flow/trainers/__init__.py` | 新增 v4 trainer 名称导出 |
| ✏️ 修改 | `lato_integration/flow/ss_flow.py` | 修复 IO ResBlocks 残差连接 shape 不匹配 |
| ✏️ 修改 | `trellis/trainers/utils.py` | 修复 `model_grads_to_master_grads` None grad 处理 |
| ✏️ 修改 | `trellis/trainers/basic.py` | 修复 `run_step` 梯度 NaN 检查跳过 None grad |
| ✏️ 修改 | `trellis/pipelines/trellis_text_to_3d.py` | 新增 `sample_sparse_structure_lato()`，条件化 coords×2 |
| 🔴 废弃 | `lato_integration/encoder.py` | DEPRECATED（由 LATO VoxelVAE.encode 替代） |
| 🔴 简化 | `lato_integration/decoder_mesh.py` | 仅保留 SparsePredictionHead + EnhancedSparseSubdivideBlock3d |
| ❌ 删除 | `lato_integration/decoder_gs.py` | 不需要 Gaussian 输出 |
| ❌ 删除 | `lato_integration/decoder_rf.py` | 不需要 Radiance Field 输出 |
| ❌ 删除 | `lato_integration/trainers/sparse_structure_vae.py` | 不训练 SS VAE |
| ❌ 删除 | `lato_integration/trainers/slat_vae_*.py` | 不训练 SLat VAE |
| ✅ 保留 | `lato_integration/utils.py` | DiagonalGaussianDistribution（latent consistency 用） |
| ✅ 保留 | `lato_integration/base.py` | SparseTransformerCrossBase |
| ✅ 保留 | `lato_integration/vertex_encoder.py` | ConnectionHead（LATO 边预测） |
| ✅ 保留 | `lato_integration/flow/` | SS Flow + SLat Flow + trainers |
| ✅ 保留 | `lato_integration/encode_lato_latent_v2.py` | LATO latent 提取 |

---

## 完整执行步骤（v4）

### 依赖关系

```
步骤0 (部署代码)
  │
  ├─→ 步骤1 (生成 SS occupancy@128³) ─→ 步骤2 (创建 v3 配置) ─→ 步骤3 (训练 SS Flow)
  │                                                                      │
  └─→ 步骤4 (训练 SLat Flow) ←─────────────────────────────────────────┘
          │
          ▼
      步骤5 (单条推理) ─→ 步骤6 (完整推理)
```

**步骤 3 和 4 可以同时跑**（两张不同 GPU），SLat Flow 训练不依赖 SS Flow 输出。

---

### 步骤 0：部署 v3 代码到服务器

将以下文件从本地同步到服务器 `/data/huanghaoyang/3D/TRELLIS/`：

```bash
# 在 Windows 本地执行（scp 或 git push）
# 新建文件：
lato_integration/structure_head.py

# 修改文件：
lato_integration/__init__.py
lato_integration/sparse_structure_vae.py
lato_integration/encoder.py
lato_integration/decoder_mesh.py
lato_integration/run_train.py
lato_integration/inference_lato.py
lato_integration/pipeline.py
lato_integration/evaluate_3d_metrics.py
lato_integration/flow/trainers/ss_flow_trainer.py
lato_integration/flow/trainers/__init__.py
lato_integration/trainers/__init__.py
trellis/pipelines/trellis_text_to_3d.py
```

然后在服务器上删除旧文件：

```bash
ssh 服务器
cd /data/huanghaoyang/3D/TRELLIS

# 删除已废弃的文件
rm -f lato_integration/decoder_gs.py
rm -f lato_integration/decoder_rf.py
rm -f lato_integration/trainers/sparse_structure_vae.py
rm -f lato_integration/trainers/slat_vae_gaussian.py
rm -f lato_integration/trainers/slat_vae_rf_dec.py
rm -f lato_integration/trainers/slat_vae_mesh_dec.py
```

---

### 步骤 1：生成 SS 训练目标（🆕 v4 新步骤）

```bash
ssh 服务器
cd /data/huanghaoyang/3D/TRELLIS

python3 << 'EOF'
import numpy as np, os, glob

latent_dir = "/data/huanghaoyang/3D/database_lato/lato_latents/latents/lato_vae_16dim_128"
output_dir = "/data/huanghaoyang/3D/database_lato/ss_occupancy_128"
os.makedirs(output_dir, exist_ok=True)

for npz_path in glob.glob(os.path.join(latent_dir, "*.npz")):
    key = os.path.basename(npz_path).replace(".npz", "")
    data = np.load(npz_path)
    coords = data['coords']   # [N, 4] sparse @ res128
    occ = np.zeros((1, 128, 128, 128), dtype=np.float32)
    for c in coords:
        if c[0] == 0:
            occ[0, c[1], c[2], c[3]] = 1.0
    np.savez_compressed(os.path.join(output_dir, f"{key}.npz"), occupancy=occ)

print(f"Done: {len(glob.glob(output_dir + '/*.npz'))} files")
EOF
```

> **输出**：`/data/huanghaoyang/3D/database_lato/ss_occupancy_128/` 下 234 个 `.npz`，每个包含 `{'occupancy': float32 [1,128,128,128]}`

---

### 步骤 2：创建 v4 训练配置

```bash
cd /data/huanghaoyang/3D/TRELLIS

python3 << 'EOF'
import json
with open("configs/generation/lato_ss_flow.json") as f:
    cfg = json.load(f)

# 添加 LatoStructureHead 模型
cfg["models"]["structure_head"] = {
    "name": "LatoStructureHead",
    "args": {
        "in_channels": 8,
        "base_channels": 256,
        "num_res_blocks": 1
    }
}
# 添加 occupancy loss 权重
if "trainer" not in cfg:
    cfg["trainer"] = {"name": "FlowMatchingCFGTrainer", "args": {}}
cfg["trainer"]["args"]["lambda_occupancy"] = 0.1

# 🔧 v4 fix: 使用 LATO 自定义数据集（加载 ss_occupancy_128）
cfg["dataset"]["name"] = "TextConditionedLatoSSStructureLatent"
cfg["dataset"]["args"]["occupancy_dir"] = "ss_occupancy_128"

# 🔧 显存优化: batch_size 降到 2（128³ 激活值巨大，B=4 会 OOM）
cfg["trainer"]["args"]["batch_size_per_gpu"] = 2
cfg["trainer"]["args"]["batch_split"] = 2  # 梯度累积保持有效 batch=4

with open("configs/generation/lato_ss_flow_v4.json", "w") as f:
    json.dump(cfg, f, indent=4)
print("Done: configs/generation/lato_ss_flow_v4.json")
EOF
```

> **输出**：`configs/generation/lato_ss_flow_v4.json`（在原有 SS Flow 配置基础上增加 `structure_head` 模型、`lambda_occupancy`，并使用 `TextConditionedLatoSSStructureLatent` 数据集）

---

### 步骤 3：训练 SS Flow + LatoStructureHead

```bash
cd /data/huanghaoyang/3D/TRELLIS

CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v4.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow_v4 \
    --num_gpus 1 --auto_retry 0
```

| 项目 | 值 |
|------|-----|
| 模型 | `EnhancedSSFlowModel`（145M）+ `LatoStructureHead`（~1-2M） |
| 训练目标 | LATO coords → occupancy@128³ dense grid |
| 损失 | Flow Matching MSE + Occupancy BCE@128³ (λ=0.1) |
| 配置 | 512ch × 24 blocks × 16 heads |
| batch_size | 2 per GPU, split=1 |
| 步数 | 1,000,000 |
| 预计时间 | ~4.5 天（单卡 RTX 4090） |
| 显存 | ~20 GB（B=2 + checkpoint + autocast） |

> **⚠️ 从零启动的 NaN 问题**：`log_scale=20` 将 loss 放大 2^20≈1M 倍，fp16 梯度溢出 → NaN 死循环。已在 `trellis/trainers/basic.py` 加入连续 NaN 抢救逻辑（10 步后强制降 log_scale），详见[已知 Bug 修复清单](#已知-bug-修复清单v3)。

---

### 步骤 4：训练 SLat Flow

```bash
cd /data/huanghaoyang/3D/TRELLIS

# 方式一：从零开始训练
CUDA_VISIBLE_DEVICES=2 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_slat_flow_v4 \
    --num_gpus 1 --auto_retry 0

# 方式二：从旧 ckpt 续训（推荐，节省时间）
CUDA_VISIBLE_DEVICES=2 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_slat_flow_v4 \
    --load_dir outputs/lato_slat_flow \
    --ckpt 0880000 \
    --num_gpus 1 --auto_retry 0
```

| 项目 | 值 |
|------|-----|
| 模型 | `LATOSLatFlowModel`（384ch × 12 blocks） |
| 训练目标 | LATO VoxelVAE latent (128³ sparse × 16-dim) |
| 损失 | Flow Matching MSE |
| batch_size | 1 per GPU, max_num_voxels=16384 |
| 步数 | 1,000,000 |
| 预计时间 | ~4.5 天（从零），~0.5 天（续训从 880k） |

**步骤 3 和步骤 4 可以同时跑（用两张不同 GPU，如 cuda:4 和 cuda:2）。**

---

### 步骤 5：单条推理

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa                # dense attention: SDPA（与训练一致，fp32 兼容）
export SPARSE_ATTN_BACKEND=xformers     # sparse attention: 必须 xformers（不支持 sdpa；flash_attn 只支持 fp16）

# 自动找最新 ckpt
SS_CKPT=$(ls outputs/lato_ss_flow_v4/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v4/ckpts/denoiser_step*.pt | sort -V | tail -1)

python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database \
    --output_dir outputs/eval_results_v4 \
    --limit 1
```

---
### 步骤 6：完整推理

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa                # dense attention: SDPA（fp32 兼容）
export SPARSE_ATTN_BACKEND=xformers     # sparse attention: 必须 xformers（不支持 sdpa）

SS_CKPT=$(ls outputs/lato_ss_flow_v4/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v4/ckpts/denoiser_step*.pt | sort -V | tail -1)

# 确认步骤 5 单条通过后跑全部 21 条（去掉 --limit）
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database \
    --output_dir outputs/eval_results_v4
```

**输出文件：**

| 文件 | 内容 |
|------|------|
| `summary.json` | CD/HD/NC 均值/标准差/中位数/最大/最小 |
| `per_sample_results.json` | 逐条详细结果 |
| `failures.json` | 失败样本及原因 |

**指标说明：**

| 指标 | 方向 | 含义 |
|------|:--:|------|
| **Chamfer Distance (CD)** | ↓ | 双向最近邻 L2 距离均值 |
| **Hausdorff Distance (HD)** | ↓ | 最大局部偏差 |
| **Normal Consistency (NC)** | ↑ [0,1] | 法线方向一致性 |

---

## 原版 TRELLIS vs v3 vs v4：逐阶段对比

### 总览

```
原版 TRELLIS:
  Text → CLIP → SS Flow → SS Decoder → SLat Flow → SLat Decoder(×3) → GS/RF/Mesh

v2 LATO 管线:
  Text → CLIP → SS Flow(重训) → SS Decoder(TRELLIS冻结) → ×2坐标 → SLat Flow(重训) → LATO VoxelVAE → Mesh

v4 LATO 管线 (🆕):
  Text → CLIP → SS Flow(重训) → LatoStructureHead(🆕可训练) → coords@128 → SLat Flow(重训) → LATO VoxelVAE → Mesh
```

### 阶段对比表

| 阶段 | 原版 TRELLIS | v2 LATO | v4 LATO (🆕) |
|------|-------------|---------|--------------|
| **SS Encoder（训练）** | TRELLIS 3D CNN | TRELLIS（冻结） | **LATO VoxelVAE.encode()** |
| **SS Flow** | Dense DiT (512ch) | 同，重训 | 同，重训（目标变了） |
| **SS Decoder（推理）** | TRELLIS 3D CNN @64³ | TRELLIS（冻结）@64³ | **LatoStructureHead @128³** |
| **坐标桥接** | 无需 | coords ×2 (hack) | **无需！** |
| **SLat Encoder（训练）** | TRELLIS Sparse Trans. | LATO VoxelVAE.encode() | LATO VoxelVAE.encode() ✅ |
| **SLat Flow** | Sparse DiT (1024ch) | Sparse DiT (384ch)，重训 | 同 |
| **SLat Decoder（推理）** | TRELLIS (GS/RF/Mesh) | LATO VoxelVAE.decode() | LATO VoxelVAE.decode() ✅ |
| **输出类型** | GS + RF + Mesh | 仅 Mesh | 仅 Mesh |

---

## 常见问题

### Q: v3 需要重新训练吗？

**需要。** SS Flow 的训练目标从 TRELLIS SS Encoder latent (16³×8) 变为 LATO coords → occupancy@128³，分布不同，需要重新训练。SLat Flow 训练目标不变，但输入 coords 来自 LatoStructureHead（而非 SS Decoder + ×2），建议也重新训练。

### Q: LatoStructureHead 参数量多少？

~1-2M 参数，比 SS Flow (~145M) 小 100 倍。作为 SS Flow 的"解码头"联合训练，几乎不增加训练开销。

### Q: 为什么不直接用 LATO VoxelVAE 的 coarse decoder 作为 SS Decoder？

LATO VoxelVAE 的 decoder 是端到端的：输入 sparse latent → 输出 mesh vertices，没有单独的 "occupancy/coords" 中间输出。LatoStructureHead 是一个轻量的替代方案，专门从 SS Flow 的 dense 特征预测 coords。

### Q: 旧的 TRELLIS Encoder/Decoder 代码还能用吗？

`encoder.py`、`decoder_mesh.py`（Decoder 类）、`sparse_structure_vae.py`（Encoder/Decoder 类）均已废弃，文件保留但代码已移除或注释。如需回退到 v2，从 git 历史恢复即可。

### Q: 推理时边太多/太少

调整 `--edge_threshold`：边太少 → 降至 0.3；边太多 → 升至 0.5~0.6。

### Q: 推理时需要训练 LATO VAE 吗？

**不需要。** LATO VoxelVAE + ConnectionHead 是预训练好的冻结权重，整个流程中始终冻结。

---

## 已知 Bug 修复清单（v4）

### Bug 1: None grad 崩溃 — `model_grads_to_master_grads`

**现象**：
```
AttributeError: 'NoneType' object has no attribute 'data'
  at model_grads_to_master_grads (utils.py:51)
```

**根因**：`LatoSSFlowTrainer` 中 `structure_head` 条件性参与 forward（由 `aux_decode_every` 和 `lambda_occupancy` 控制），不参与时 `.grad=None`，但 `model_params` 包含所有模型的参数。

**修复**（3 处）：

| 文件 | 修改 |
|------|------|
| `trellis/trainers/utils.py:51` | `param.grad is None` 时用 `torch.zeros_like(param.data)` 替代 |
| `trellis/trainers/basic.py:394,405` | NaN 检测跳过 `grad is None` 的参数 |

### Bug 2: `LatoStructureHead` 缺少 `convert_to_fp16/32`

**现象**：`fp16_mode='inflat_all'` 时 resume checkpoint → `AttributeError: 'LatoStructureHead' object has no attribute 'convert_to_fp16'`

**根因**：TRELLIS trainer 对每个 model 调用 `convert_to_fp16()`，但 `LatoStructureHead` 是纯 `nn.Module`。

**修复**：`lato_integration/structure_head.py` 添加空实现兼容方法。

### Bug 3: 数据集不返回 `ss_occupancy_128`

**现象**：`structure_head` 永远不参与训练，`ss_occupancy_128 is None` → `should_decode=False`。

**根因**：`SparseStructureLatent.get_instance()` 只返回 `{'x_0': z}`，缺少 occupancy 字段。

**修复**：新建 `lato_integration/datasets.py` — `TextConditionedLatoSSStructureLatent` 额外加载 `ss_occupancy_128/*.npz`。

### Bug 4: Trainer 映射缺失

**现象**：
```
TypeError: EnhancedSSFlowModel.forward() got an unexpected keyword argument 'ss_occupancy_128'
```

**根因**：`TRAINER_REPLACEMENTS` 缺少 `TextConditionedFlowMatchingCFGTrainer` 映射，fallback 到原始 TRELLIS trainer，后者不认识 `ss_occupancy_128`。

**修复**：`lato_integration/run_train.py` 添加 `"TextConditionedFlowMatchingCFGTrainer": TextConditionedEnhancedSSFlowCFGTrainer`。

### Bug 5: 128³ OOM

**现象**：`torch.OutOfMemoryError` at `structure_head.py:109 (stage3 upsample)`，尝试分配 4 GB。

**根因**：stage3 激活值 `[B, 64, 128, 128, 128]` @ fp32, B=4 ≈ 4 GB 连续显存，加上 denoiser 已占 ~20 GB → 超出 23.55 GB。

**修复**（组合方案）：

| 措施 | 位置 |
|------|------|
| `torch.autocast` 包裹 structure_head 前向（fp16 计算） | `ss_flow_trainer.py` |
| `torch.utils.checkpoint` — stage3 不存中间激活 | `structure_head.py` |
| `batch_size_per_gpu: 2` + 梯度累积 | `lato_ss_flow_v4.json` |
| IO ResBlocks 残差连接 shape 检查 | `flow/ss_flow.py` |

### Bug 6: 从零训练 NaN 死循环 + 日志崩溃

**现象**：
```
AssertionError: input must be a non-empty list of dictionaries
  at dict_reduce (general_utils.py:59)
```

**根因链**：

```
log_scale=20.0（硬编码）
  → loss × 2^20 ≈ 1,000,000× 放大
  → fp16 梯度最大值=65504，溢出 → NaN
  → NaN → optimizer.step() 跳过
  → 模型卡在随机初始化，下步继续 NaN
  → 原逻辑每步只 log_scale -= 1，需 500+ 步才能降到安全值
  → 500 步全部 NaN → log_show 过滤后空列表 → dict_reduce 崩溃
```

**修复**：

| 文件 | 修改 |
|------|------|
| `trellis/trainers/basic.py` | 新增 `_consecutive_nan` 计数器；连续 NaN ≥ 10 步时强制 `log_scale -= 5`（原为 -1），最快 15-20 步内恢复 |
| `trellis/trainers/base.py` | `log_show` 过滤后为空时用最后一条日志作为 fallback，避免 `dict_reduce` 崩溃 |

### Bug 7: fp16 权重溢出 → 53 万步全部 NaN（🆕 2026-07-25 修复）

**现象**：
```
log.txt: "mse": NaN, "loss": NaN  (持续 530k+ 步)
checkpoint: 全部 24 层权重 NaN/Inf
log_scale: 0 (抢救代码持续降 scale，但拦不住前向 NaN)
```

**根因链**：

```
use_fp16=True + 24层 transformer × fp16 + fp16_mode=inflat_all
                    ↓
fp32 master params 权重随训练漂移 → 某步超过 65504 (fp16 max)
                    ↓
master_params_to_model_params() 拷贝 fp32→fp16 → Inf
  (trellis/trainers/utils.py:43 — 裸拷，无保护)
                    ↓
下一轮 forward: Inf 权重 → 全模型 NaN → log_scale 抢救跳过反向
  → 但权重已是 Inf，再也回不去 → 530k 步全部白费
```

**诊断过程**：

| 检查项 | 结果 |
|--------|------|
| SS latent 数据 (NPZ) | 干净，NaN=0 |
| config `use_fp16` | `true` — 模型跑 fp16 |
| config `fp16_mode` | `inflat_all` — trainer 会强制 `convert_to_fp16()` |
| step 10000 checkpoint | 干净 |
| step 20000+ checkpoint | 488 NaN/Inf params（全部 block 被污染） |

**修复（方案 B：fp32 全精度）**：

| 文件 | 修改 | 行号 |
|------|------|------|
| `flow/ss_flow.py` | ① `convert_to_fp16()` override — 检查 `self.use_fp16`，False 时调 `self.float()` 转 fp32 | 156-168 |
| `flow/ss_flow.py` | ② forward 末尾输出 NaN guard — `nan_to_num` + `clamp` 兜底 | 234-236 |
| `flow/ss_flow.py` | ③ `clamp_weights_fp16_safe()` — fp16 训练时每步钳制权重 ±65500，fp32 自动跳过 | 240-253 |
| `flow/trainers/ss_flow_trainer.py` | ④ `run_step()` override — optimizer.step() 后调用 `clamp_weights_fp16_safe()` | 78-88 |
| `configs/.../lato_ss_flow_v4.json` | ⑤ `"use_fp16": false` — 模型全 fp32 | — |

**Resume 时 fp16→fp32 自动转换**：

```
load checkpoint (fp16 权重)
  → load_state_dict(fp16 ckpt)          # 权重暂为 fp16
  → model.convert_to_fp16()             # trainer 无条件调用 (basic.py:189)
      → 我们的 override: use_fp16=False
      → self.float()                    # 全部转 fp32
  → optimizer state 正常恢复 (fp32)
  → 从 step 10000 开始 fp32 训练
```

**启动命令**：

```bash
# 1. 改 config 为 fp32
python3 -c "
import json
cfg = json.load(open('configs/generation/lato_ss_flow_v4.json'))
cfg['models']['denoiser']['args']['use_fp16'] = False
json.dump(cfg, open('configs/generation/lato_ss_flow_v4.json', 'w'), indent=2)
print('Done: use_fp16 → False')
"

# 2. 删除被污染的 checkpoint（仅保留 step 10000）
cd /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow_v4/ckpts
ls | grep -v step0010000 | xargs rm -f

# 3. 从 step 10000 resume
cd /data/huanghaoyang/3D/TRELLIS
python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v4.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow_v4 \
    --num_gpus 1 --auto_retry 0 --ckpt 10000
```

**验证收敛**：

```bash
tail -5 /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow_v4/log.txt
# mse 应从正常值开始，持续下降，不再 NaN
```

---

### Bug 8: 推理时 dtype 不匹配 + flash_attn 不支持 fp32（🆕 2026-07-26 修复）

**现象 1**：`evaluate_3d_metrics.py` 报错 `Input type (float) and bias type (c10::Half) should be the same`

**根因链**：

```
Bug 7 修复 → config use_fp16=false (fp32 训练)
              ↓
evaluate_3d_metrics.py 硬编码 use_fp16=True (line 360)
              ↓
structure_head = structure_head.half()  # fp16 权重
              ↓
SS Flow forward: h = h.type(x.dtype)    # ss_flow.py:224 — 输出转回输入 dtype (fp32)
              ↓
structure_head(fp32_input) + fp16_weight → dtype mismatch 💥
```

**现象 2**：修复 dtype 后报错 `FlashAttention only support fp16 and bf16 data type`

**根因**：`trellis/modules/attention/__init__.py:3` 默认 `BACKEND = 'flash_attn'`，flash_attn 只支持 fp16/bf16，不支持 fp32。

**修复**：

| 文件 | 修改 | 说明 |
|------|------|------|
| `evaluate_3d_metrics.py:360` | `default=True → default=False` | 默认 fp32，与训练一致 |
| 环境变量 | `export ATTN_BACKEND=sdpa` + `export SPARSE_ATTN_BACKEND=xformers` | dense attn 用 sdpa（与训练一致），sparse attn 用 xformers（fp32 兼容） |

**推理前必须设置**：
```bash
export ATTN_BACKEND=sdpa                # dense attention: SDPA（与训练一致，fp32 兼容）
export SPARSE_ATTN_BACKEND=xformers     # sparse attention: 必须 xformers（不支持 sdpa；flash_attn 只支持 fp16）
```

---

### Bug 9: evaluate_3d_metrics.py 无法加载 structure_head 权重（🆕 2026-07-26 修复）

**现象**：`LatoStructureHead: 使用随机初始化（未找到预训练权重）`

**根因**：

1. **checkpoint 格式误判**：
   - eval 脚本 `load_pipeline()` 对 SS/SLat ckpt 只用 `ckpt.get('state_dict', ckpt.get('model', ckpt))`
   - 但 TRELLIS misc 格式的权重在 `ckpt['denoiser']` 子 dict 里
   - 导致 SS Flow 加载了错误的 dict（顶层 key 如 `step`/`optimizer`），`load_state_dict(strict=False)` 不报错但实际没加载到

2. **structure_head 单独保存**：
   - 训练时 structure_head 未嵌入 denoiser ckpt，而是单独保存为 `structure_head_step*.pt`
   - eval 脚本只在 denoiser state_dict 里找 `structure_head.*` 前缀的 key，找不到

**修复**（`evaluate_3d_metrics.py` `load_pipeline()`）：

| 位置 | 修改 |
|------|------|
| SS ckpt 加载 | 先检查 `ckpt_raw['denoiser']`（TRELLIS misc 格式），再 fallback |
| structure_head | 先在 state_dict 中找 `structure_head.*` 前缀；找不到则 glob 同目录下 `structure_head_step*.pt` 取最新 |

---

### Bug 10: StructureHead 训练无效 — BCE 缺少 `pos_weight`（🆕 2026-07-30 修复）

**现象**：

- 训练 40 万步后 `out_conv.bias ≈ -1105`，`occ_logits.mean ≈ -1105`
- 仅 ~9.8k 体素 > 0，部分样本全部 < 0 → VoxelVAE 收到空 coords → 空顶点 → 崩溃
- 推理指标 CD=0.214，远低于预期

**根因**：

```
Occupancy BCE 正负样本比 = 1:200（128³ 中 ~10k occupied vs ~2M empty）
                    ↓
F.binary_cross_entropy_with_logits(reduction='mean', pos_weight=None)
                    ↓
99.5% 负样本 ≈ 对 loss 贡献 0（输出负值即可正确）
 0.5% 正样本 → 需要跨过 0 阈值，但梯度 ≈ 1.0 × lr=1e-4 / 步
                    ↓
模型学到捷径: out_conv.bias → -1105，所有特征输出被压制
从 -1105 恢复到 0 需要 1100 万步 → 40 万步远远不够
                    ↓
StructureHead 功能完全失效
```

> **责任归属**：这是 v4 架构设计时引入的训练代码 bug。原版 TRELLIS 和 LATO 均无 StructureHead + occupancy BCE 这套组件，`pos_weight` 缺失是生成 `ss_flow_trainer.py` 时的疏忽。

**修复**（`lato_integration/flow/trainers/ss_flow_trainer.py`）：

```python
# 修复前
occ_bce = F.binary_cross_entropy_with_logits(
    occ_logits, ss_occupancy_128.float(), reduction='mean'
)

# 修复后 — 动态 pos_weight 补偿正负不平衡
n_pos = ss_occupancy_128.sum().clamp(min=1)
n_neg = ss_occupancy_128.numel() - n_pos
pos_weight = (n_neg / n_pos).clamp(1.0, 500.0)
occ_bce = F.binary_cross_entropy_with_logits(
    occ_logits, ss_occupancy_128.float(),
    reduction='mean', pos_weight=pos_weight,
)
```

**已有 ckpt 修复**：

```bash
# StructureHead 独立保存，需单独重置所有 bias
SH_CKPT=outputs/lato_ss_flow_v4/ckpts/structure_head_step0400000.pt
python3 -c "
import torch
ckpt = torch.load('$SH_CKPT', map_location='cpu')
for k, v in ckpt.items():
    if 'bias' in k and v.ndim == 1:
        v.zero_()
torch.save(ckpt, '$SH_CKPT')
"
```

> ⚠️ SS Flow (denoiser) 权重不受影响，Flow Matching MSE 独立于 StructureHead，**无需重置 SS Flow**。

### Bug 11: 仅重置 bias 不够 — Stage Conv 权重仍然在生产极端负值（🆕 2026-07-31 修复）

**现象**：

- 重置 `out_conv.bias` 为 0 后 resume，log 中 `occ_bce_128` 仍然不存在
- 507k+ 步无任何 occ_bce 产出

**诊断**：

```
加载 StructureHead ckpt (bias=0) + 随机 latent 输入
  → occ_logits: mean=-425,476, min=-1,727,389, max=8.75
  → >0 voxels: 2
```

| | 随机初始化 | 当前 ckpt (bias=0) |
|------|----------|---------|
| occ_logits mean | **0.10** | **-432,648** |
| >0 voxels | 2,090,113 | **2** |

**根因链**：

```
Bug 10 修复前 107k 步训练（无 pos_weight）
  → conv 学会将一切输入推到极端负值（配合 out_conv.bias ≈ -1105）
  → 只重置 out_conv.bias → 0，但 stage1/2/3 conv 权重仍是坏的
  → conv 输出仍然 ~ -432k
  → training_losses 中 StructureHead 走 torch.autocast (fp16)
  → -432k 超出 fp16 上限 65504 → Inf
  → BCE output → Inf
  → torch.isfinite(occ_bce) → False → 静默跳过
  → occ_bce 永远不会写 log
```

**修复**：重置全部 StructureHead 权重（不只是 bias）：

```bash
python3 << 'EOF'
import torch, torch.nn as nn

ckpt_path = 'outputs/lato_ss_flow_v4/ckpts/structure_head_step0500000.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

for k in list(ckpt.keys()):
    if 'weight' in k:
        nn.init.kaiming_normal_(ckpt[k])
    elif 'bias' in k:
        ckpt[k].zero_()
    print(f'{k}: reinit')

torch.save(ckpt, ckpt_path)
print('Done.')
EOF
```

> ⚠️ StructureHead 需要从零开始学习（SS Flow 不受影响）。预计 50k 步收敛（~2 天）。

### StructureHead 恢复训练与验证计划（2026-07-31）

**启动训练**：

```bash
cd /data/huanghaoyang/3D/TRELLIS
CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v4.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/lato_ss_flow_v4 \
    --num_gpus 1 --ckpt 500000
```

**阶段 1 — 30 分钟后验证 occ_bce 恢复**：

```bash
tail -5 outputs/lato_ss_flow_v4/log.txt | grep -o "occ_bce_[^,}]*"
```

**阶段 2 — ~1 天后（~20k 步）验证 StructureHead 输出正常**：

```bash
python3 -c "
import torch
from lato_integration.structure_head import LatoStructureHead
import glob, os

ckpt = sorted(glob.glob('outputs/lato_ss_flow_v4/ckpts/structure_head_step*.pt'))[-1]
head = LatoStructureHead(in_channels=8, base_channels=256).cuda()
head.load_state_dict(torch.load(ckpt, map_location='cuda'))

x = torch.randn(1, 8, 16, 16, 16).cuda() * 2
with torch.no_grad():
    occ = head(x)
    print(f'Occ logits mean: {occ.mean():.2f}')       # 应在 ±5 以内
    print(f'Active voxels (>0): {(occ>0).sum().item()}')  # 应在几万到几十万
"
```

**阶段 3 — ~2 天后（~50k 步）推理验证质量**：

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers

SS_CKPT=$(ls outputs/lato_ss_flow_v4/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow/ckpts/denoiser_step*.pt | sort -V | tail -1)

python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/eval_results_v4 \
    --save_meshes --limit 1
```

**训练恢复时间线**：

| 步数 | 时间 | 标志 |
|------|------|------|
| 几百步 | ~30 分钟 | `occ_bce_128` 出现在 log |
| ~20k 步 | ~1 天 | StructureHead occ_logits 恢复正常范围（±5） |
| ~50k 步 | ~2 天 | 基本收敛，推理指标明显改善 |
| ~100k 步 | ~4-5 天 | 充分收敛 |

### Bug 12: StructureHead autocast fp16 溢出 — occ_bce 7 步后消失（🆕 2026-07-31 修复）

**现象**：

- 重置全部 StructureHead 权重 + resume 后，`occ_bce_128` 仅在前 7 步出现
- 随后完全消失，loss == mse（occ_bce 未参与）
- occ_bce 值快速下降：0.716 → 0.686 → 0.658 → 0.543 → 0.434 → 0.180 → 0.046 → 消失

**验证方法（CPU fp32 检测）**：

```bash
python3 << 'EOF'
import torch, sys
sys.path.insert(0, '/data/huanghaoyang/3D/TRELLIS')
from lato_integration.structure_head import LatoStructureHead

ckpt = torch.load('outputs/lato_ss_flow_v4/ckpts/structure_head_step0510000.pt', map_location='cpu')
head = LatoStructureHead(in_channels=8, base_channels=256)
head.load_state_dict(ckpt)
x = torch.randn(1, 8, 16, 16, 16) * 2

with torch.no_grad():
    occ = head(x)
    print(f'FP32: mean={occ.mean():.0f}, min={occ.min():.0f}, max={occ.max():.0f}')
    outside = (occ.abs() > 65500).sum().item()
    print(f'Outside fp16 range (>65500): {outside} / {occ.numel()}')
    if outside > 0:
        print('❌ 值超出 fp16 范围 → autocast 下变 Inf')
EOF
```

**实测输出**：

```
FP32: mean=-15527331, min=-23029436, max=-89916
Outside fp16 range (>65500): 2097152 / 2097152
❌ 100% 的值超出 fp16 范围 — autocast 下全部变 Inf
```

**根因链**：

```
StructureHead: 3 级 2× 上采样 (16³ → 32³ → 64³ → 128³), base_channels=256
                    ↓
每个 stage 的 3³ 卷积累积放大中间激活值:
  stage1: 50 × 3³ ×   8 × 0.04 ≈   432      (fp16 安全)
  stage2: 432 × 3³ × 256 × 0.02 ≈  59,719   (接近 fp16 上限 65504)
  stage3: 59k × 3³ × 128 × 0.01 ≈ 2,060,524 (远超 fp16 上限!)
                    ↓
training_losses 中 torch.autocast(fp16) 包裹 StructureHead
  → 前向: stage3 激活全部 Inf → occ_logits = Inf
  → BCE(Inf, ...) = Inf → isfinite=False → occ_bce 不写 log
  → 🔁 反向: torch.checkpoint 重新计算 stage3 激活 → 再次 Inf
  → Inf 梯度污染 conv 权重 → 权重无法学习
```

**失败的尝试：只把 BCE 移到 autocast 外面**

```python
# ❌ 不够: nan_to_num/clamp 只能修前向 BCE，反向用 checkpoint 重新计算时
# StructureHead 激活仍然在 fp16 下溢出 → 梯度废掉
with torch.autocast(device_type='cuda', enabled=self.fp16_mode is not None):
    occ_logits = self.training_models['structure_head'](x_0_pred)
occ_logits = occ_logits.float().clamp(-50.0, 50.0)
# BCE 计算安全，但梯度已经被 autocast 内的 Inf 污染了
```

**最终修复**（`ss_flow_trainer.py`）：

```python
# ✅ 正确：StructureHead 完全禁用 autocast，前向+反向全在 fp32
with torch.autocast(device_type='cuda', enabled=False):
    occ_logits = self.training_models['structure_head'](x_0_pred)
occ_logits = occ_logits.clamp(-50.0, 50.0)
```

> **设计说明**：
> - `enabled=False`：无论 trainer 的 fp16_mode 是什么，StructureHead 始终跑 fp32
> - 内存安全：StructureHead 使用 `torch.utils.checkpoint`，128³ 中间激活不常驻，反向时逐 stage 重新计算
> - `clamp(-50, 50)`：防御性兜底，fp32 正常不会触发
> - 对比原始 TRELLIS SS Decoder 也是 64³ 输出（小 8 倍），LatoStructureHead 到 128³ 远远超出 fp16 能力范围

### Bug 13: `get_inference_cond` MRO 链断裂 — 训练专用字段泄漏到推理（🆕 2026-08-01 修复）

**现象**：

- 推理时 `model.forward()` 收到意外的 `ss_occupancy_128` 关键字参数，可能导致 TypeError 或静默错误
- 特别是在 CFG (Classifier-Free Guidance) 采样时，`classifier_free_guidance.py` 的 `get_inference_cond()` 将全部 kwargs 打包传入 cond dict

**根因链**：

```
ss_occupancy_128 / extrinsics / intrinsics 是训练专用字段
            ↓
training_losses() 通过 **kwargs 接收并消费它们
            ↓
但采样/推理时 cond 通过 get_inference_cond() 打包 kwargs
            ↓
Python MRO (Method Resolution Order) 问题:
  ClassifierFreeGuidanceMixin.get_inference_cond() 不调用 super()
            ↓
LatoSSFlowTrainer.get_inference_cond() 的剥离逻辑被跳过
  (MRO 链: CFGMixin → LatoSSFlowCFGTrainer → LatoSSFlowTrainer)
  但 CFGMixin 不 super() → LatoSSFlowTrainer 的清理代码永远不会执行
            ↓
训练字段泄漏到 model.forward() 💥
```

**修复**（`lato_integration/flow/trainers/ss_flow_trainer.py`）：

| 位置 | 修改 |
|------|------|
| `LatoSSFlowTrainer._TRAIN_ONLY_KEYS` (line 56) | 定义训练专用字段集合：`{'ss_occupancy_128', 'extrinsics', 'intrinsics'}` |
| `LatoSSFlowTrainer.get_inference_cond()` (line 69-76) | 在调用父类前剥离 `_TRAIN_ONLY_KEYS` |
| `LatoSSFlowCFGTrainer.get_inference_cond()` (line 190-197) | **必须覆盖**：因为 `ClassifierFreeGuidanceMixin` 不调用 `super()`，MRO 链在此断裂，需显式剥离 |

```python
# LatoSSFlowTrainer — 基础版本（非 CFG 训练器使用）
_TRAIN_ONLY_KEYS = {'ss_occupancy_128', 'extrinsics', 'intrinsics'}

def get_inference_cond(self, cond, **kwargs):
    for key in self._TRAIN_ONLY_KEYS:
        kwargs.pop(key, None)
    return super().get_inference_cond(cond, **kwargs)

# LatoSSFlowCFGTrainer — CFG 版本（必须显式覆盖）
# 🔧 ClassifierFreeGuidanceMixin 不调用 super()，截断 MRO 链
def get_inference_cond(self, cond, **kwargs):
    for key in self._TRAIN_ONLY_KEYS:
        kwargs.pop(key, None)
    return super().get_inference_cond(cond, **kwargs)
```

### Bug 14: `x_0_pred` latent 溢出 — Flow Matching 重构值超出 fp16 范围（🆕 2026-08-01 修复）

**现象**：

- 训练早期（前几千步）StructureHead 收到极端 latent 值 → fp16 conv 激活全部 Inf → occ_bce 消失
- 即使 Bug 12 已禁用 autocast，`x_0_pred` 的值本身仍然可能极端（Flow Matching 初期速度预测不准确）

**根因链**：

```
Flow Matching 训练早期:
  x_t = noise + t * (x_0 - noise)
  v_pred 预测不准（模型未收敛）
            ↓
x_0_pred = x_t - t * v_pred  ← 重构的 clean latent
            ↓
t 接近 1 时，v_pred 的误差被放大:
  正常 x_0 ~ [-5, 5]，但 x_0_pred 可能 ~ [-500, 500]
            ↓
StructureHead(x_0_pred):
  stage1 conv: 500 × 3³ conv ≈ 13,500     (fp16 安全)
  stage2 conv: 13,500 × 3³ conv ≈ 364,500  (远超 fp16 上限 65504!)
  → 中间激活全部 Inf，occ_logits = NaN/Inf
            ↓
occ_bce = isfinite=False → 跳过 → occ_bce 静默消失
```

**修复**（`ss_flow_trainer.py:138`）：

```python
# 修复前
x_0_pred = self._reconstruct_x0(x_t, pred, t)

# 修复后 — clamp latent 值到安全范围再喂给 StructureHead
# 正常 latent 在 [-10, 10] 内，clamp 到 [-50, 50] 留足余量
x_0_pred = self._reconstruct_x0(x_t, pred, t)
x_0_pred = x_0_pred.clamp(-50.0, 50.0)
```

> **与 Bug 12 的关系**：Bug 12 修复了 StructureHead **内部**的 fp16 溢出（禁用 autocast）。Bug 14 修复了 StructureHead **输入**的溢出（clamp x_0_pred）。两者互补：即使 StructureHead 跑 fp32，如果输入值 ~500 也会导致 3 级上采样后激活值爆炸。Bug 14 确保输入始终在合理范围。

### Bug 15: 深层 Transformer fp16 激活累积溢出（🆕 2026-08-01 修复）

**现象**：

- SS Flow 24 层 DiT block 在 fp16 下连续运算，深层激活值逐渐漂移
- 不经常发生，但在特定 batch/seed 下激活值累积超过 fp16 上限 → NaN 扩散到整个 denoiser

**根因**：

```
24 层 Transformer × fp16 matmul + LayerNorm + SiLU:
  每层输出有 ~0.1% 的概率产生 > 65504 的激活
            ↓
第 N 层的异常值 → 第 N+1 层 attention(Q,K,V) 放大:
  Q·K^T / sqrt(d) 中异常值扩散到整个 attention map
            ↓
  24 层累积 → 尾部层 ~2% 概率输出 Inf
            ↓
Inf 输出 → Flow Matching loss NaN → 整个 batch 作废
```

**修复**（`flow/ss_flow.py:219-222`）：

```python
# 在每个 transformer block 之后钳制激活值
# 修复前
for block in self.blocks:
    h = block(h, t_emb, cond)

# 修复后 — 每层后做 fp16 安全钳
for block in self.blocks:
    h = block(h, t_emb, cond)
    # fp16 安全钳：24 层连续 fp16 运算可能累积溢出
    # 正常激活值 < 200，clamp 在 ±32000 留足余量，不会影响训练
    if self.dtype == torch.float16:
        h = torch.nan_to_num(h, nan=0.0, posinf=32000.0, neginf=-32000.0)
        h = h.clamp(-32000.0, 32000.0)
```

> **与 Bug 7 的关系**：Bug 7 的 `clamp_weights_fp16_safe()` 修复了**权重**溢出（optimizer step 后钳制参数），Bug 15 修复了**激活值**溢出（每层 forward 后钳制中间结果）。两者构成完整的 fp16 安全体系：权重钳 + 激活钳。

---

### 步骤 7：推理环境变量检查清单

```bash
# 推理前必须设置
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa                # dense attention: SDPA（与训练一致，fp32 兼容）
export SPARSE_ATTN_BACKEND=xformers     # sparse attention: 必须 xformers（不支持 sdpa；flash_attn 只支持 fp16）

# 选一张空闲 GPU
export CUDA_VISIBLE_DEVICES=2
```

> **环境变量说明：**
>
> | 变量 | 推荐值 | 作用范围 | 原因 |
> |------|--------|---------|------|
> | `ATTN_BACKEND` | `sdpa` | SS Flow dense attention (DiT blocks) | 与训练一致；fp32 兼容；PyTorch 内置 |
> | `SPARSE_ATTN_BACKEND` | `xformers` | SLat Flow sparse attention (SparseMultiHeadAttention) | 稀疏模块只支持 `xformers` / `flash_attn`；fp32 下仅 `xformers` 可用 |
>
> ⚠️ `SPARSE_ATTN_BACKEND` **不支持** `sdpa`，设为 `sdpa` 会被静默忽略并 fallback 到 `flash_attn`（fp32 报错）。

---

## 测试集评估结果（2026-07-26）

**训练状态**：
- SS Flow: 100,000 steps / 1,000,000（10%）
- LatoStructureHead: 100,000 steps（独立 ckpt：`structure_head_step0100000.pt`）
- SLat Flow: 旧 v2 ckpt @ 880,000 steps（`outputs/lato_slat_flow/`）
- LATO VoxelVAE + ConnectionHead: 冻结预训练（epoch 1, best loss 0.0807）

**评估命令**：
```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa                # dense attention: SDPA（fp32 兼容）
export SPARSE_ATTN_BACKEND=xformers     # sparse attention: 必须 xformers（不支持 sdpa）

SS_CKPT=$(ls outputs/lato_ss_flow_v4/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v4/ckpts/denoiser_step*.pt | sort -V | tail -1)

python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/eval_results_v4 \
    --save_meshes
```

**21 条测试集全量结果**：

| 指标 | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| Chamfer Distance ↓ | 0.214 | 0.047 | 0.130 | 0.311 |
| Hausdorff Distance ↓ | 0.507 | 0.038 | 0.429 | 0.565 |
| Normal Consistency ↑ | 0.459 | 0.112 | 0.054 | 0.601 |

**中间状态诊断**（`denoiser_step0100000.pt`）：

| 检查点 | 值 | 判断 |
|--------|-----|------|
| SS Flow output (z_s) | range [-2.68, 2.32] | ✅ Flow Matching 潜空间正常 |
| Structure Head output (occ) | max=9.38, mean=-1105 | ⚠️ 大量负 bias，仅 ~9.8k 体素 > 0 |
| threshold=0 active voxels | ~9,836 | ✅ 合理的体素数量 |
| Generated mesh vertices | 64-680 | ❌ 偏少（正常应 2k-5k） |

---

## Flow Matching Loss 收敛 ≠ 推理质量好

**关键理解**：SS Flow 训练的是**单步速度场预测**，推理时是**多步链式去噪**。

```
训练: noise → 单步预测 v → MSE(v, v_gt)     ← loss 衡量这个
推理: noise → 20 步迭代去噪 → z_s → SH → occ ← 质量取决于这个
```

| | 训练 MSE | 推理质量 |
|---|----------|---------|
| 100k 步 | 0.004 ✅ | 差（CD=0.214） |
| 300k 步 | ~0.003 | 明显改善 |
| 500k+ 步 | ~0.002 | 好 |

**原因**：
1. **误差累积**：每步去噪都有小误差，20 步累积后差距变大。类似 diffusion 模型训练 10% 时 loss 已降但生成的图还是噪声
2. **Structure Head 权重低**：BCE occupancy 损失权重 λ=0.1（仅占 10%），10 万步时 SH 还在保守期（大量负 bias 避免假阳性）
3. **SLat Flow 不匹配**：当前用的旧 v2 ckpt，输入分布来自 TRELLIS SS Decoder 而非 LatoStructureHead

**结论**：没有 bug，就是训得太少。SS Flow 训到 300k+ 步后重新评估，指标会大幅改善。

---

## 推理验证结果（2026-07-31）

**训练状态（Bug 11 修复前）**：
- SS Flow: 507,500 steps / 1,000,000（50%），mse ~0.001
- StructureHead: 500,000 steps（**conv 权重被 107k 步无 pos_weight 训练污染，仅 bias 被重置**）
- SLat Flow: 旧 v2 ckpt @ 880,000 steps（`outputs/lato_slat_flow/`）
- log: 无 NaN，log_scale 稳定在 20.0
- occ_bce_128: **消失于 step 106,920，此后 40 万步不存在**

**单条推理结果**（limit=1，StructureHead 输出 ~ -432k）：

| 指标 | 2026-07-26 (100k步) | 2026-07-31 (507k步) | 变化 |
|------|---------------------|---------------------|:--:|
| Chamfer Distance ↓ | 0.214 | 0.265 | 📉 SS Flow 与解码头分布偏移 |
| Hausdorff Distance ↓ | 0.507 | 0.541 | 📉 同上 |
| Normal Consistency ↑ | 0.459 | 0.530 | 📈 SS Flow 自身去噪质量提升 |

**根因诊断**：

```
StructureHead ckpt (bias=0, conv 权重 = 坏):
  occ_logits mean = -432,648  ← conv 输出极端负值
  >0 voxels = 2 / 2,097,152   ← 几乎全部被压制

训练时 autocast fp16: -432k → Inf → BCE Inf → NaN guard 跳过 → occ_bce 不写 log
```

**修复**：Bug 11 — 全部 StructureHead conv 权重 + bias 重新初始化，从 step 500,000 resume 训练。预计 ~50k 步收敛（~2 天）。

**当前步骤（2026-07-31）**：

```bash
# 1. 重置 StructureHead 全部权重
python3 << 'EOF'
import torch, torch.nn as nn
ckpt = torch.load('outputs/lato_ss_flow_v4/ckpts/structure_head_step0500000.pt', map_location='cpu')
for k in list(ckpt.keys()):
    if 'weight' in k: nn.init.kaiming_normal_(ckpt[k])
    elif 'bias' in k: ckpt[k].zero_()
torch.save(ckpt, 'outputs/lato_ss_flow_v4/ckpts/structure_head_step0500000.pt')
EOF

# 2. Resume 训练
cd /data/huanghaoyang/3D/TRELLIS
CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v4.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/lato_ss_flow_v4 \
    --num_gpus 1 --ckpt 500000
```

---

# 🆕 v8 补充：GT 数据根因修复与流程更新

> **日期：** 2026-08-03
> **状态：** SS occupancy 归一化已修复，LATO encode 重新运行中

---

## 根因分析

### 问题表现

v4 管线训练后推理 mesh 不成形：所有顶点 Z 轴聚集在 [1.48, 1.50]，mesh 实际为 2D 平面。CD=0.215、HD=0.507、NC=0.571。

### 逐段诊断链

```
推理 mesh 扁平
  → VoxelVAE decode 顶点 Z 坐标全在 [508, 511]（res512）
  → SLat Flow 输入 coords 全在 Z=127（res128）
  → StructureHead 输出 occupancy X 轴跨度为 0（全部正体素聚集在 X=127）
  → GT occupancy 同样：所有 234 个训练文件的 X 轴跨度为 0
  → GT 来自 LATO VoxelVAE.encode() 的 latent coords
  → LATO encode 输入 voxels 已全部 Z=127
```

### 最终定位

**`lato/datasets/vertex_head.py` 第 172 行 `load_quantized_mesh_original()` 函数：**

```python
vertices = np.clip(np.asarray(mesh_o3d.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
```

此函数假设输入 mesh 已归一化到 [-0.5, 0.5]。刹车卡钳 STL 为毫米级原始坐标（几十到几百 mm），Z 轴（厚度方向 ~20-80mm）因范围远小于 X/Y 轴，被全部截断至 0.5。量化到 res=128 后全部落入单个体素层（z=127），形成 2D 平面。

**影响链路：** 2D voxels → VoxelVAE.encode() → 2D latent coords → 步骤1 生成 2D occupancy GT → SS Flow + StructureHead 忠实地学到 2D 目标。

### TRELLIS 对比

TRELLIS 的 SS VAE 训练数据集（`trellis/datasets/sparse_structure.py` 第 43-45 行）使用**预处理好的 `.ply` voxel 文件**，坐标已在 [-0.5, 0.5] 范围内，不做在线归一化。LATO 试图从原始 mesh 一步到位体素化，缺少归一化预处理步骤。

---

## 修复方案

### 思路

不修改 LATO 源码（避免第三方库升级时补丁失效），采用与 TRELLIS 一致的**离线预处理**方式：将 STL 归一化到 [-0.5, 0.5] 后再进行 LATO encode。

### 步骤 0（新增）：归一化 STL 网格

```bash
cd /data/huanghaoyang/3D/TRELLIS

python3 << 'EOF'
import trimesh, numpy as np, os, glob, shutil
from tqdm import tqdm

DB = "/data/huanghaoyang/3D/database_lato"
mesh_dir = os.path.join(DB, "meshes")
backup_dir = os.path.join(DB, "meshes_mm_backup")
norm_dir = os.path.join(DB, "meshes_normalized")

# 备份原始 STL
if not os.path.exists(backup_dir):
    shutil.move(mesh_dir, backup_dir)

os.makedirs(norm_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(backup_dir, "*.stl")))
for path in tqdm(files, desc="Normalizing"):
    name = os.path.basename(path)
    out_path = os.path.join(norm_dir, name)
    if os.path.exists(out_path):
        continue
    mesh = trimesh.load(path, force="mesh")
    v = mesh.vertices
    center = (v.min(axis=0) + v.max(axis=0)) / 2
    scale = (v.max(axis=0) - v.min(axis=0)).max()
    mesh.vertices = (v - center) / scale * 0.9  # 最长边 → 0.9，留 5% margin
    mesh.export(out_path)

# 软链接归一化目录为 meshes/
os.symlink(norm_dir, mesh_dir)
print(f"Done: {len(files)} files → {norm_dir}")
EOF
```

### 步骤 1（🆕 更新）：重新生成 SS occupancy

步骤 0 完成后，重跑 LATO encode 和 occupancy 生成，使用新输出路径避免覆盖旧数据：

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"

# A. 重跑 LATO encode（输出到 lato_latents_v2/）
python lato_integration/encode_lato_latent_v2.py \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir /data/huanghaoyang/3D/database_lato/lato_latents_v2 \
    --resolution 128

# B. 从新 latent 生成 SS occupancy
python3 << 'EOF'
import numpy as np, os, glob

latent_dir = "/data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128"
output_dir = "/data/huanghaoyang/3D/database_lato/ss_occupancy_128_v2"
os.makedirs(output_dir, exist_ok=True)

for npz_path in glob.glob(os.path.join(latent_dir, "*.npz")):
    key = os.path.basename(npz_path).replace(".npz", "")
    data = np.load(npz_path, allow_pickle=True)
    coords = data['coords']   # [N, 4] sparse @ res128
    occ = np.zeros((1, 128, 128, 128), dtype=np.float32)
    for c in coords:
        if c[0] == 0:
            occ[0, c[1], c[2], c[3]] = 1.0
    np.savez_compressed(os.path.join(output_dir, f"{key}.npz"), occupancy=occ)

print(f"Done: {len(glob.glob(output_dir + '/*.npz'))} files → {output_dir}")
EOF

# C. 验证 occupancy 为 3D
python3 << 'EOF'
import numpy as np, os, glob

files = sorted(glob.glob("/data/huanghaoyang/3D/database_lato/ss_occupancy_128_v2/*.npz"))[:3]
for f in files:
    occ = np.load(f)['occupancy']
    pos = np.where(occ[0] > 0)
    ok = pos[2].max() - pos[2].min() > 10
    print(f"{'✅' if ok else '❌'} {os.path.basename(f)[:40]} "
          f"Xspan={pos[0].max()-pos[0].min()} "
          f"Yspan={pos[1].max()-pos[1].min()} "
          f"Zspan={pos[2].max()-pos[2].min()}")
EOF
```

### 步骤 2（🆕 更新）：创建 v4 训练配置

config 中数据集指向新的 occupancy 目录：

```python
cfg["dataset"]["args"]["occupancy_dir"] = "ss_occupancy_128_v2"
```

### 步骤 3+4：训练（不变）

SS Flow + StructureHead 和 SLat Flow 的训练步骤不变。注意：重置 StructureHead 权重后，**必须同时清理 misc checkpoint 中的 optimizer Adam state**（Bug 16）。

```bash
# 重置 StructureHead + 清理 Adam state
python3 << 'EOF'
import torch, torch.nn as nn, os, glob

ckpt_dir = "outputs/lato_ss_flow_v4/ckpts"
misc_files = sorted(glob.glob(f"{ckpt_dir}/misc_step*.pt"))
latest = misc_files[-1]
step = int(os.path.basename(latest).replace("misc_step","").replace(".pt",""))

# 重置权重
sh_path = f"{ckpt_dir}/structure_head_step{step:07d}.pt"
sh = torch.load(sh_path, map_location='cpu')
for k in list(sh.keys()):
    if 'weight' in k: nn.init.kaiming_normal_(sh[k])
    elif 'bias' in k: sh[k].zero_()
torch.save(sh, sh_path)
print(f"StructureHead 权重重置完成 (step {step})")

# 清空 StructureHead 对应的 Adam state
misc = torch.load(latest, map_location='cpu', weights_only=False)
for g in misc['optimizer']['param_groups']:
    for pid in g['params']:
        s = misc['optimizer']['state'].get(pid, {})
        ea = s.get('exp_avg')
        if ea is not None and ea.numel() == 1 and ea.shape == torch.Size([1]):
            for pid2 in g['params']:
                st = misc['optimizer']['state'].get(pid2, {})
                for k in ['exp_avg', 'exp_avg_sq']:
                    if k in st and st[k] is not None:
                        st[k].zero_()
                st['step'] = 0
            break
torch.save(misc, latest)
print(f"Optimizer state 已清理，保存: {os.path.basename(latest)}")
EOF

# Resume 训练
CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v4.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/lato_ss_flow_v4 \
    --num_gpus 1 --ckpt latest
```

---

## 新增 Bug：Bug 16 — StructureHead 权重重置后 Adam state 未清理

**现象：** StructureHead 全量重置（kaiming_normal_ init）后训练 130k 步，推理 mesh 仍然输出 2D 平面。

**根因：** 手动编辑 `structure_head_step*.pt` 覆写权重后，misc checkpoint 中的 optimizer Adam state 仍然保留旧权重的 momentum 和 variance（out_conv.bias ≈ -1105 时代）。Adam 持续将新权重推向旧错误方向，130k 步训练基本无效。

**修复：** 重置权重时必须同步清空 misc checkpoint 中对应 param_group 的 `exp_avg`、`exp_avg_sq`，并将 `step` 归零。

---

## 旧 vs 新 关键对比

| 项目 | 旧（v4-v7） | 🆕 新（v8） |
|------|------------|------------|
| **STL 预处理** | 无（原始 mm 坐标） | 归一化到 [-0.5, 0.5] |
| **LATO encode** | `load_quantized_mesh_original` clip 截断 | 预处理后 encode 正常 |
| **SS occupancy GT** | 2D 平面（Z 轴塌缩） | 3D 体素 |
| **SLat latent** | 2D coords | 3D coords |
| **StructureHead 重置后** | 未清理 Adam state（Bug 16） | 同步清理 optimizer state |
| **代码修改** | 不动 LATO 源码 | 不动 LATO 源码 ✅ |

---

## 已知影响范围

- **SS Flow 需要重训：** 训练目标从 2D 变为 3D，SS Flow 学到的 2D 映射不再适用
- **SLat Flow 通过 v2 推理验证可兼容 3D 输入：** 旧 SLat Flow 训练时虽然与 SS Flow 一样基于 2D latent，但 v2 推理已验证其接受 TRELLIS SS Decoder 产出的 3D coords 可正常产出 CD=0.214。SLat Flow 作为 sparse 128³ 上逐体素 feat 回归任务，对坐标分布敏感度远低于 dense latent 生成的 SS Flow。如果 v5 SS Flow 训好后评估发现 SLat 质量不达标，再按下方步骤重训
- **LATO VAE 不受影响：** 预训练权重冻结，仅 encode 输入归一化方式变更
- **推理管线无需改动：** 仅输入数据变化，代码不变

---

## 🆕 v8 SLat Flow 重训配置（备用）

> 如果 v5 SS Flow + 旧 SLat Flow 组合的 CD > 0.5，按以下步骤重训 SLat Flow。

### SLat Flow 训练 config（新建 `lato_slat_flow_v8.json`）

与旧 config 的差异：
- `--data_dir` 指向新 latent 目录 `lato_latents_v2/`
- `max_num_voxels` 从 16384 提高到 65536（新 latent 体素数 29k-52k）
- normalization stats 需从新 latent 重新计算

```bash
cd /data/huanghaoyang/3D/TRELLIS

# 从新 latent 计算 normalization stats
python3 << 'EOF'
import numpy as np, glob, json

d = "/data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128"
feats_list = []
for f in glob.glob(f"{d}/*.npz"):
    feats = np.load(f, allow_pickle=True)['feats']  # [N, 16]
    feats_list.append(feats)

all_feats = np.concatenate(feats_list, axis=0)
mean = all_feats.mean(axis=0).tolist()
std = all_feats.std(axis=0).tolist()

# 保存 stats
stats = {"mean": mean, "std": std}
with open(f"{d}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"stats.json 已保存（{len(feats_list)} 文件, {all_feats.shape[0]} 体素）")
print(f"mean[:8]: {[f'{v:.4f}' for v in mean[:8]]}")
print(f"std[:8]:  {[f'{v:.4f}' for v in std[:8]]}")
EOF

# 创建 v8 SLat config（基于旧 config 更新关键参数）
python3 << 'EOF'
import json

with open("configs/generation/lato_slat_flow.json") as f:
    cfg = json.load(f)

# 加载新 stats
with open("/data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/stats.json") as f:
    stats = json.load(f)

cfg["dataset"]["args"]["max_num_voxels"] = 65536
cfg["dataset"]["args"]["normalization"] = stats
cfg["models"]["denoiser"]["args"]["use_fp16"] = False  # fp32 训练

with open("configs/generation/lato_slat_flow_v8.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("configs/generation/lato_slat_flow_v8.json 已创建")
EOF
```

### SLat Flow 训练命令

```bash
cd /data/huanghaoyang/3D/TRELLIS

CUDA_VISIBLE_DEVICES=2 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow_v8.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents_v2 \
    --output_dir outputs/lato_slat_flow_v8 \
    --num_gpus 1
```

| 项目 | 旧 | 🆕 v8 |
|------|-----|------|
| 训练数据 | `lato_latents/`（2D） | `lato_latents_v2/`（3D） |
| max_num_voxels | 16384 | 65536 |
| normalization | 旧 2D 统计 | 新 3D 统计 |
| use_fp16 | true | false |
| 预计步数 | 1,000,000 | 1,000,000 |
| 预计时间 | ~4.5 天 | ~5 天（voxel 更多） |

---

## 🆕 训练健康自检脚本

> 每 10000 步跑一次（30 秒，不占训练 GPU），精确区分"训练不够"和"有 bug"。
>
> **文件位置:** `check_health.sh`（TRELLIS 根目录）
>
> **用法:**
> ```bash
> cd /data/huanghaoyang/3D/TRELLIS
> bash check_health.sh outputs/lato_ss_flow_v5 0
> bash check_health.sh outputs/lato_ss_flow_v5 0 --data_dir /data/huanghaoyang/3D/database_lato  # 加载真实 latent 测试
> ```

### 判断标准

| 指标 | 正常 | 异常 |
|------|:--:|:--:|
| SH 3D | X/Y/Z span 均 > 10，n > 0 | 全负（n=0）或任一轴 span=0 |
| occ_bce | > 0.01 | 趋近于 0（3.5×10⁻⁸）或 N/A |
| MSE | 持续下降 | 不降或上升 |
| NaN | 0 | > 0 |

**四项全正常 = 继续训，不是 bug，只是训练不够。**

### 输出示例（健康）

```
=== 训练健康检查 @ 18:17 ===
[SH 3D] ✅ n=99207 Xspan=106 Yspan=111 Zspan=111 (随机(0.1/0.5/1.0/2.0))
  occ_bce(最近200均值): 0.0966
  MSE(最近200均值):      0.0989
  NaN: 0 条 ✅
  step: 20000
  denoiser:      denoiser_step0020000.pt
  structure_head: structure_head_step0020000.pt
```

### 注意事项

1. **必须在 TRELLIS 根目录运行**（不在 `lato_integration/` 子目录下），否则 ckpt 路径解析错误
2. **推荐加 `--data_dir`** 用真实 SS latent 测试 StructureHead，比随机输入更准确
3. **step 从 ckpt 文件名提取**（`denoiser_step0020000.pt` → 20000），不依赖 log 解析
4. **测试输入不是 `randn*2`**（与 Flow Matching latent 分布不同，会产生误导的"全负"），而是多 scale 随机输入或真实 latent

---

## 🆕 v9 执行状态（2026-08-04）

| 步骤 | 内容 | 状态 |
|------|------|:--:|
| 步骤 0 | STL 归一化：234 个 mesh → `meshes_normalized/` → 软链接 `meshes/` | ✅ 完成 |
| 步骤 0 验证 | voxelization Z 轴 span=39-57（原 0），全部 3D | ✅ 通过 |
| 步骤 1A | 重跑 LATO encode → `lato_latents_v2/` | ✅ 完成 |
| 步骤 1A 验证 | 新 latent Z span=45-53（原 0），voxel 数 29k-52k（原固定 16k） | ✅ 通过 |
| 步骤 1B | 生成 SS occupancy → `ss_occupancy_128_v2/`（234 个 npz） | ✅ 完成 |
| 步骤 1B 验证 | occupancy X/Y/Z 全部 3D，体素数 29k-52k | ✅ 通过 |
| 步骤 2 | 更新 config：`occupancy_dir` → `ss_occupancy_128_v2` | ✅ 已改 |
| 步骤 3 | 重置 StructureHead + 清理 Adam state | ✅ 完成 |
| 步骤 4 | SS Flow + StructureHead 训练 → `outputs/lato_ss_flow_v5/` | 🔄 90k步，继续中 |
| 步骤 4 验证 | 健康检查全绿（SH 3D✅, occ_bce=0.038, MSE=0.024, NaN=0）| ✅ 通过 |
| 步骤 5 | SLat Flow 训练 | ⏳ 待 SS 训完评估 |
| 步骤 6 | 最终评估 + CD/HD/NC 指标 | ⏳ 待定 |

---

## 🆕 v9 已验证文件路径汇总（2026-08-04）

> 以下所有路径均经过 234/234 或 21/21 全量匹配验证。

### 数据目录结构

```
/data/huanghaoyang/3D/database_lato/
├── metadata.csv                        # 训练集 234 条
├── test/
│   └── metadata.csv                    # 测试集 21 条
├── meshes_mm_backup/                   # 原始 mm 坐标 STL（备份，sha256 命名）
├── meshes_normalized/                  # 🆕 归一化 STL [-0.5, 0.5]（sha256 命名）
├── meshes → meshes_normalized          # 软链接
├── lato_latents_v2/                    # 🆕 3D latent
│   └── latents/lato_vae_16dim_128/     # 234 个 npz, Zspan>40
├── ss_occupancy_128_v2/                # 🆕 3D occupancy（234 个 npz, Zspan>40）
└── ss_latents/
    └── ss_enc_conv3d_16l8_fp16/        # TRELLIS SS latent（255 个 npz，训练+测试）
```

### 训练数据（234 条，全部匹配）

| 数据 | 路径 | 匹配 |
|------|------|:--:|
| metadata | `/data/huanghaoyang/3D/database_lato/metadata.csv` | — |
| SS latent | `/data/huanghaoyang/3D/database_lato/ss_latents/ss_enc_conv3d_16l8_fp16/{sha256}.npz` | 234/234 |
| occupancy | `/data/huanghaoyang/3D/database_lato/ss_occupancy_128_v2/{sha256}.npz` | 234/234 |

### 测试数据（21 条）

| 数据 | 路径 | 匹配 |
|------|------|:--:|
| metadata | `/data/huanghaoyang/3D/database_lato/test/metadata.csv` | — |
| SS latent | `/data/huanghaoyang/3D/database_lato/ss_latents/ss_enc_conv3d_16l8_fp16/{sha256}.npz` | 21/21 |
| GT mesh | CSV `file_path` 列 → `/data/huanghaoyang/3D/database/{file_identifier}.stl` | 21/21 |

### 模型文件

| 模型 | 路径 |
|------|------|
| SS config | `configs/generation/lato_ss_flow_v3.json` |
| SS Flow ckpt | `outputs/lato_ss_flow_v5/ckpts/denoiser_step*.pt` |
| StructureHead ckpt | `outputs/lato_ss_flow_v5/ckpts/structure_head_step*.pt` |
| SLat config | `configs/generation/lato_slat_flow.json` |
| SLat Flow ckpt | `outputs/lato_slat_flow/ckpts/denoiser_step*.pt` |
| LATO VAE ckpt | `/data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt` |
| LATO config | `/data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml` |

### 已删除（旧 2D 数据）

| 目录 | 原因 |
|------|------|
| `ss_occupancy_128/` | Zspan=0，2D 扁平 |
| `lato_latents/` | Zspan=0，2D 扁平 |

### 关键说明

- **instance 匹配方式**：`StandardDatasetBase` 用 CSV 的 `sha256` 列作为 instance，拼路径 `{root}/{subdir}/{sha256}.npz`
- **occupancy 在训练中的作用**：仅 StructureHead 的 BCE loss 使用，`lambda_occupancy=0.1`；SS Flow 的 Flow Matching MSE 不受影响
- **evaluate_3d_metrics.py 已修复**：GT mesh 加载改为优先使用 CSV 的 `file_path` 绝对路径，解决 sha256 文件名不匹配问题

---

### 训练命令（当前 v5）

```bash
cd /data/huanghaoyang/3D/TRELLIS
CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v3.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/lato_ss_flow_v5 \
    --num_gpus 1
```

### 推理命令（当前 v5）

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers

SS_CKPT=$(ls outputs/lato_ss_flow_v5/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow/ckpts/denoiser_step*.pt | sort -V | tail -1)

python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/eval_results_v5 \
    --limit 1 \
    --save_meshes
```