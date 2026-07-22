# TRELLIS + LATO 文本转3D — 完整实现流程 v7

> **目标：** TRELLIS 文本转3D 管线中，**Encoder 全部用 LATO VoxelVAE，Decoder 全部用 LATO VoxelVAE，只有中间 Flow 生成用 TRELLIS**。SS Flow 和 SLat Flow 均在刹车卡钳数据集上从零训练，最后用 Chamfer Distance / Hausdorff Distance / Normal Consistency 评估。

**更新记录：**
- 2026-07-23：**v7 更新** — 架构全面重构：TRELLIS SS/SLat Encoder/Decoder 全部替换为 LATO VoxelVAE；新增 `LatoStructureHead` 替代 SS Decoder；移除 coords×2 hack；删除 GS/RF decoder 及 VAE 训练器
- 2026-07-19：**v6 更新** — 端到端测试集推理验证通过；修复 6 个推理 bug
- 2026-07-17：**v5 更新** — 推理脚本重构：从 config JSON 读取模型参数、自动发现最新 ckpt、新增 ss_only 模式
- 2026-07-16：**v4 更新** — 新增步骤6（批量评估 + 3D 指标）
- 2026-07-14：修复 12 个 bug + 4 个训练启动 bug，单卡 RTX 4090 24GB 可行

---

## 架构概览

```
                    v2 管线 (旧)                         v3 管线 (🆕)                       对应目标
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

### v2 → v3 核心变化

| 组件 | v2 | v3 |
|------|-----|-----|
| SS Encoder（训练） | TRELLIS SparseStructureEncoder（冻结） | LATO VoxelVAE.encode() |
| SS Decoder（推理） | TRELLIS SparseStructureDecoder（冻结, 64³） | **🆕 LatoStructureHead**（可训练, 128³ 直接输出） |
| coords ×2 | 需要（64→128 桥接） | **不需要** |
| SLat Encoder（训练） | LATO VoxelVAE.encode() ✅ | 不变 ✅ |
| SLat Decoder（推理） | LATO VoxelVAE.decode() ✅ | 不变 ✅ |
| GS/RF Decoder | 代码存在但未使用 | **已删除** |
| VAE Trainers | 存在但未使用 | **已删除** |

---

## 推理全链路追踪（v3：`pipeline.run()` → mesh.obj）

```
TrellisTextTo3DPipeline.run()          # trellis_text_to_3d.py:212
│
├─[1] sample_sparse_structure_lato()   # 🆕 :109 v3 新增方法
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

## 文件清单（v3）

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
| ✏️ 修改 | `lato_integration/flow/trainers/__init__.py` | 新增 v3 trainer 名称导出 |
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

## 完整执行步骤（v3）

### 依赖关系

```
步骤0 (部署代码)
  │
  ├─→ 步骤1 (生成 SS occupancy@128³) ─→ 步骤2 (创建 v3 配置) ─→ 步骤3 (训练 SS Flow)
  │                                                                      │
  └─→ 步骤4 (训练 SLat Flow) ←─────────────────────────────────────────┘
          │
          ▼
      步骤5 (单条推理) ─→ 步骤6 (批量评估)
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

### 步骤 1：生成 SS 训练目标（🆕 v3 新步骤）

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

### 步骤 2：创建 v3 训练配置

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

# 🔧 v3 fix: 使用 LATO 自定义数据集（加载 ss_occupancy_128）
cfg["dataset"]["name"] = "TextConditionedLatoSSStructureLatent"
cfg["dataset"]["args"]["occupancy_dir"] = "ss_occupancy_128"

# 🔧 显存优化: batch_size 降到 2（128³ 激活值巨大，B=4 会 OOM）
cfg["trainer"]["args"]["batch_size_per_gpu"] = 2
cfg["trainer"]["args"]["batch_split"] = 2  # 梯度累积保持有效 batch=4

with open("configs/generation/lato_ss_flow_v3.json", "w") as f:
    json.dump(cfg, f, indent=4)
print("Done: configs/generation/lato_ss_flow_v3.json")
EOF
```

> **输出**：`configs/generation/lato_ss_flow_v3.json`（在原有 SS Flow 配置基础上增加 `structure_head` 模型、`lambda_occupancy`，并使用 `TextConditionedLatoSSStructureLatent` 数据集）

---

### 步骤 3：训练 SS Flow + LatoStructureHead

```bash
cd /data/huanghaoyang/3D/TRELLIS

CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v3.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow_v3 \
    --num_gpus 1 --auto_retry 0
```

| 项目 | 值 |
|------|-----|
| 模型 | `EnhancedSSFlowModel`（145M）+ `LatoStructureHead`（~1-2M） |
| 训练目标 | LATO coords → occupancy@128³ dense grid |
| 损失 | Flow Matching MSE + Occupancy BCE@128³ (λ=0.1) |
| 配置 | 512ch × 24 blocks × 16 heads |
| batch_size | 4 per GPU |
| 步数 | 1,000,000 |
| 预计时间 | ~3 天（单卡 RTX 4090） |

---

### 步骤 4：训练 SLat Flow

```bash
cd /data/huanghaoyang/3D/TRELLIS

# 方式一：从零开始训练
CUDA_VISIBLE_DEVICES=2 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_slat_flow_v3 \
    --num_gpus 1 --auto_retry 0

# 方式二：从旧 ckpt 续训（推荐，节省时间）
CUDA_VISIBLE_DEVICES=2 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_slat_flow_v3 \
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

### 步骤 5：单条推理验证

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export SPARSE_ATTN_BACKEND=xformers

# 自动找最新 ckpt
SS_CKPT=$(ls outputs/lato_ss_flow_v3/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v3/ckpts/denoiser_step*.pt | sort -V | tail -1)

# 完整推理（SS + SLat + LATO → mesh）
python lato_integration/inference_lato.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --prompt "A brake caliper fixing interaxis 116.32 inner pad 222.29 pistons_num 6" \
    --seed 42 \
    --output output_v3.obj

# SS-only 模式（只验证 SS Flow + LatoStructureHead）
python lato_integration/inference_lato.py \
    --mode ss_only \
    --ss_ckpt "$SS_CKPT" \
    --prompt "A brake caliper"
```

**常用调参：**

```bash
# 边太少（mesh 有洞）→ 降低边阈值
--edge_threshold 0.3

# 边太多（噪声面过多）→ 提高边阈值
--edge_threshold 0.5

# VoxelVAE decode 保留更多/更少顶点
--lato_threshold 0.1   # 更多顶点
--lato_threshold 0.3   # 更少顶点
```

**全部参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `full` | `full`=完整管线, `ss_only`=仅 SS Flow + LatoStructureHead |
| `--ss_ckpt` | 无 | SS Flow checkpoint 路径（含 LatoStructureHead） |
| `--slat_ckpt` | 无 | SLat Flow checkpoint 路径 |
| `--ss_dir` | 无 | 自动发现：SS Flow 训练输出目录 |
| `--slat_dir` | 无 | 自动发现：SLat Flow 训练输出目录 |
| `--lato_ckpt` | `$LATO_ROOT/.../vae_128to512.pt` | LATO VAE checkpoint |
| `--lato_config` | `$LATO_ROOT/.../infer_vae_512.yaml` | LATO VAE config |
| `--prompt` | 刹车卡钳示例 | 文本描述 |
| `--output` | `output_mesh.obj` | 输出 mesh 路径 |
| `--seed` | 42 | 随机种子 |
| `--ss_steps` | 20 | SS Flow 采样步数 |
| `--slat_steps` | 20 | SLat Flow 采样步数 |
| `--cfg_strength` | 5.0 | CFG 强度 |
| `--lato_threshold` | 0.2 | VoxelVAE decode threshold |
| `--edge_threshold` | 0.45 | ConnectionHead 边概率阈值 |
| `--k_neighbors` | 32 | KDTree 最近邻数 |
| `--no_fp16` | false | 禁用 FP16（调试用） |

---

### 步骤 6：批量评估

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export SPARSE_ATTN_BACKEND=xformers

SS_CKPT=$(ls outputs/lato_ss_flow_v3/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v3/ckpts/denoiser_step*.pt | sort -V | tail -1)

# 先跑 1 条验证管线
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database \
    --output_dir outputs/eval_results_v3 \
    --limit 1

# 确认通过后跑全部 21 条（去掉 --limit）
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database \
    --output_dir outputs/eval_results_v3
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

## 原版 TRELLIS vs v2 vs v3：逐阶段对比

### 总览

```
原版 TRELLIS:
  Text → CLIP → SS Flow → SS Decoder → SLat Flow → SLat Decoder(×3) → GS/RF/Mesh

v2 LATO 管线:
  Text → CLIP → SS Flow(重训) → SS Decoder(TRELLIS冻结) → ×2坐标 → SLat Flow(重训) → LATO VoxelVAE → Mesh

v3 LATO 管线 (🆕):
  Text → CLIP → SS Flow(重训) → LatoStructureHead(🆕可训练) → coords@128 → SLat Flow(重训) → LATO VoxelVAE → Mesh
```

### 阶段对比表

| 阶段 | 原版 TRELLIS | v2 LATO | v3 LATO (🆕) |
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
