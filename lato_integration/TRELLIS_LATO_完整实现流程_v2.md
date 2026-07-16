# TRELLIS + LATO 文本转3D — 完整实现流程 v4

> **目标：** TRELLIS 文本转3D 管线中，将 Sparse VAE Encoder/Decoder 替换为 LATO 的 VoxelVAE，**SS Flow 和 SLat Flow 均在刹车卡钳数据集上从零训练**，最后用 Chamfer Distance / Hausdorff Distance / Normal Consistency 评估。

**更新记录：**
- 2026-07-16：**v4 更新** — 新增步骤6（批量评估 + 3D 指标），推理脚本支持 `--ss_ckpt`
- 2026-07-16：**v3 更新** — 新增 SS Flow 训练步骤，训练模型数从 1 个变为 2 个
- 2026-07-14（下午）：修复 4 个训练启动 bug，验证单卡 RTX 4090 24GB 可运行
- 2026-07-14（上午）：修复 12 个 bug，新增多卡训练方案

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

**关键改动：**
- SS Flow：**在刹车卡钳数据上从零训练**，架构 `SparseStructureFlowModel`（→ `EnhancedSSFlowModel`）
- SS Decoder：**完全不动**（冻结，TRELLIS 预训练权重）
- SLat Flow：架构 `LATOSLatFlowModel`，`resolution=128, in/out_channels=16`，**从零训练**
- Decoder：**替换**为 LATO VoxelVAE + ConnectionHead（冻结，LATO 预训练权重）
- 需训练 **2 个** 新模型，各约 1M 步，**可并行训练**

---

## 前置条件

### 硬件

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU | 1× 24GB (RTX 4090) | 4× RTX 4090 |
| 显存 | 20GB+ | 24GB |
| CUDA | 11.8+ | 12.5 |

### 目录结构

```
服务器: /data/huanghaoyang/3D/
├── TRELLIS/
│   ├── trellis/models/lato_slat_flow.py           # ✅ LATOSLatFlowModel
│   ├── trellis/models/__init__.py                 # ✅ 注册
│   ├── trellis/pipelines/trellis_text_to_3d.py    # ✅ coords 缩放 + LATO decode
│   ├── trellis/trainers/base.py                   # ✅ 跳过 init snapshot
│   ├── trellis/datasets/structured_latent.py       # ✅ coords 适配
│   ├── lato_integration/
│   │   ├── run_train.py                           # ✅ 训练入口
│   │   ├── inference_lato.py                      # ✅ 单条推理（v4: 支持 --ss_ckpt）
│   │   ├── evaluate_3d_metrics.py                 # 🆕 批量评估（CD/HD/NC）
│   │   ├── encode_lato_latent_v2.py               # ✅ LATO latent 提取
│   │   ├── flow/ss_flow.py                        # ✅ EnhancedSSFlowModel
│   │   ├── flow/slat_flow.py                      # ✅ LATOSLatFlowModel
│   │   └── flow/trainers/
│   │       ├── ss_flow_trainer.py                 # ✅ SS Flow 训练器
│   │       └── slat_flow_trainer.py               # ✅ SLat Flow 训练器
│   ├── configs/generation/
│   │   ├── lato_ss_flow.json                      # ✅ SS Flow 训练配置
│   │   └── lato_slat_flow.json                    # ✅ SLat Flow 训练配置
│   └── dataset_toolkits/stat_latent.py            # ✅ 统计量计算
│
├── LATO/
│   ├── lato/
│   ├── configs/infer_vae_512.yaml
│   ├── vertex_encoder.py
│   ├── utils.py
│   └── checkpoints/128to512/vae/vae_128to512.pt   # LATO 预训练
│
└── database_lato/                                  # 数据根目录
    ├── metadata.csv                                # 234 条（全量）
    ├── meshes/                                     # 234 个 .stl（GT）
    ├── ss_latents/ss_enc_conv3d_16l8_fp16/         # 255 npz（SS latent）
    ├── lato_latents/latents/lato_vae_16dim_128/    # 234 npz + stats.json
    ├── train/                                      # 训练子集
    │   ├── metadata.csv                            # 234 条（同根目录）
    │   ├── voxels/, features/, renders/
    │   ├── ss_latents/                             # 复制自根目录
    │   └── lato_latents/                           # 复制自根目录
    └── test/                                       # 测试子集
        ├── metadata.csv                            # 21 条
        ├── voxels/, features/, renders/
```

---

## 步骤总览

| 步骤 | 内容 | 脚本 | 预计时间 | 状态 |
|------|------|------|----------|------|
| 1 | 确认代码修改 | 5 个文件 | — | ✅ |
| 2a | 提取 LATO latent | `encode_lato_latent_v2.py` | ~30min | ✅ 234 npz |
| 2b | 确认 SS latent | — | — | ✅ 255 npz |
| 3a | SLat normalization | `stat_latent.py` | ~5min | ✅ stats.json |
| 3b | SS normalization | — | — | ⏭️ identity |
| 4a | 训练 SS Flow | `run_train.py` | ~2.8 天（单卡） | 🔄 |
| 4b | 训练 SLat Flow | `run_train.py` | ~4.5 天（单卡） | 🔄 |
| 5 | 单条推理验证 | `inference_lato.py` | ~30s/条 | ⏳ |
| 6 | 批量评估 | `evaluate_3d_metrics.py` | ~10min | ⏳ |

---

## 步骤1：代码修改（5个文件）

| 文件 | 改动 |
|------|------|
| `trellis/models/lato_slat_flow.py` | `LATOSLatFlowModel`（res=128, ch=16） |
| `trellis/models/__init__.py` | 注册 |
| `trellis/pipelines/trellis_text_to_3d.py` | coords `[:,1:]*2` + `decode_slat_lato()` + threshold 可配置 |
| `trellis/datasets/structured_latent.py` | None decoder 跳过 + coords 剥离 batch 列 |
| `trellis/trainers/base.py` | 跳过 init/resume snapshot |

---

## 步骤2a：提取 LATO latent

```bash
python lato_integration/encode_lato_latent_v2.py \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --resolution 128 --num_points 65536 --device cuda
```

输出：`lato_latents/latents/lato_vae_16dim_128/{sha256}.npz` × 234

---

## 步骤2b：确认 SS latent

数据已就绪：`ss_latents/ss_enc_conv3d_16l8_fp16/` × 255 npz

每个 npz: `{'mean': float32 [8,16,16,16]}`（dense 16³×8）

---

## 步骤3a：SLat normalization

```bash
python dataset_toolkits/stat_latent.py \
    --output_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --model lato_vae_16dim_128 --num_samples 50000
```

产物：`stats.json`，填入 `lato_slat_flow.json` 的 `dataset.args.normalization`

---

## 步骤3b：SS normalization — 跳过

SS Flow 使用 identity normalization（`mean=[0]*8, std=[1]*8`），与 TRELLIS 原版一致。

---

## 步骤4a：训练 SS Flow

```bash
cd /data/huanghaoyang/3D/TRELLIS

CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow \
    --num_gpus 1 --auto_retry 0
```

| 项目 | 值 |
|------|-----|
| 模型 | `EnhancedSSFlowModel`（→ `SparseStructureFlowModel`），145M 参数 |
| 配置 | 512ch × 24 blocks × 16 heads, cond=768 |
| 数据 | `TextConditionedSparseStructureLatent`, 234 条 |
| 输入/输出 | CLIP text → dense 16³×8 |
| batch_size | 2 per GPU |
| 步数 | 1,000,000 |
| 速度 | ~15,000 steps/h（单卡 4090），ETA ~2.8 天 |

**OOM 降级方案：** `model_channels: 512→384` → `batch_size: 2→1`

---

## 步骤4b：训练 SLat Flow

```bash
CUDA_VISIBLE_DEVICES=2 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents \
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_slat_flow \
    --num_gpus 1 --auto_retry 0
```

| 项目 | 值 |
|------|-----|
| 模型 | `LATOSLatFlowModel` |
| 配置 | 384ch × 12 blocks × 8 heads, cond=768, io_blocks=[128]×2 |
| 数据 | `TextConditionedSLat`, latent_model=`lato_vae_16dim_128` |
| 输入/输出 | CLIP text + GT SS latent → sparse 128³×16 |
| batch_size | 1 per GPU, max_num_voxels=16384 |
| 步数 | 1,000,000 |
| 速度 | ~9,000 steps/h（单卡 4090），ETA ~4.5 天 |

**续训：** 加 `--ckpt {step}` 从断点继续

---

## 步骤5：单条推理验证

```bash
python lato_integration/inference_lato.py \
    --ss_ckpt outputs/lato_ss_flow/ckpts/denoiser_step1000000.pt \
    --slat_ckpt outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents/latents/lato_vae_16dim_128/stats.json \
    --prompt "A brake caliper fixing interaxis 94.31 inner pad 98.47 pistons_num 4" \
    --seed 42 --output output_caliper.obj
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ss_ckpt` | **必填** | 训练的 SS Flow checkpoint |
| `--slat_ckpt` | **必填** | 训练的 SLat Flow checkpoint |
| `--lato_ckpt` | **必填** | LATO VAE checkpoint |
| `--lato_config` | **必填** | LATO VAE yaml |
| `--ss_stats` | 空（identity） | SS normalization |
| `--slat_stats` | 空（identity） | SLat normalization |
| `--prompt` | **必填** | 文本描述 |
| `--ss_steps` | 20 | SS Flow 采样步数 |
| `--slat_steps` | 20 | SLat Flow 采样步数 |
| `--cfg_strength` | 5.0 | CFG 强度 |
| `--lato_threshold` | 0.2 | VoxelVAE decode 阈值 |
| `--edge_threshold` | 0.45 | ConnectionHead 边阈值 |

---

## 步骤6：批量评估（🆕）

对测试集（21 条）逐条推理，计算 Chamfer Distance / Hausdorff Distance / Normal Consistency。

```bash
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt outputs/lato_ss_flow/ckpts/denoiser_step1000000.pt \
    --slat_ckpt outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents/latents/lato_vae_16dim_128/stats.json \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/eval_results \
    --limit 5   # 先测 5 条，确认没问题后去掉 --limit 跑全部 21 条
```

**指标：**

| 指标 | 方向 | 含义 |
|------|------|------|
| **Chamfer Distance** | ↓ | 生成 mesh 与 GT mesh 的双向最近邻 L2 距离均值 |
| **Hausdorff Distance** | ↓ | 生成 mesh 与 GT mesh 的最大局部偏差 |
| **Normal Consistency** | ↑ [0,1] | 对应最近邻点法线夹角的余弦绝对值均值 |

**输出：**
- `summary.json` — 均值/标准差/中位数/最大/最小
- `per_sample_results.json` — 逐条详细结果
- `failures.json` — 失败样本及原因

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test_metadata` | **必填** | 测试集 CSV |
| `--gt_meshes` | **必填** | GT STL/OBJ/PLY 目录 |
| `--n_points` | 50000 | 评估采样点数 |
| `--limit` | 0（全部） | 限制测试条数 |

---

## 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `trellis/models/lato_slat_flow.py` | LATOSLatFlowModel | ✅ |
| `trellis/pipelines/trellis_text_to_3d.py` | Pipeline：coords 缩放 + LATO decode | ✅ |
| `trellis/models/__init__.py` | 模型注册 | ✅ |
| `trellis/datasets/structured_latent.py` | Dataset：coords 适配 | ✅ |
| `trellis/trainers/base.py` | 跳过 init/resume snapshot | ✅ |
| `configs/generation/lato_ss_flow.json` | SS Flow 配置（512ch） | ✅ |
| `configs/generation/lato_slat_flow.json` | SLat Flow 配置（384ch） | ✅ |
| `lato_integration/run_train.py` | 训练入口（通用） | ✅ |
| `lato_integration/flow/ss_flow.py` | EnhancedSSFlowModel | ✅ |
| `lato_integration/flow/slat_flow.py` | LATOSLatFlowModel 增强 | ✅ |
| `lato_integration/flow/trainers/ss_flow_trainer.py` | SS Flow 训练器 | ✅ |
| `lato_integration/flow/trainers/slat_flow_trainer.py` | SLat Flow 训练器 | ✅ |
| `lato_integration/encode_lato_latent_v2.py` | LATO latent 提取 | ✅ |
| `lato_integration/inference_lato.py` | 单条推理 | ✅ v4 |
| `lato_integration/evaluate_3d_metrics.py` | 🆕 批量评估 | ✅ |
| `dataset_toolkits/stat_latent.py` | SLat normalization | ✅ |

---

## v4 当前状态

| 步骤 | 状态 | 说明 |
|------|------|------|
| 1. 代码修改 | ✅ | 5 个文件 |
| 2a. LATO latent | ✅ | 234 npz |
| 2b. SS latent | ✅ | 255 npz |
| 3a. SLat stats | ✅ | stats.json 已填入配置 |
| 3b. SS stats | ⏭️ | identity |
| 4a. 训练 SS Flow | 🔄 | 单卡 512ch，目标 1M 步 |
| 4b. 训练 SLat Flow | 🔄 | 单卡 384ch，续训中 |
| 5. 单条推理 | ⏳ | 两个都训完后 |
| 6. 批量评估 | ⏳ | 推理验证通过后 |

---

## 常见问题

### Q: 显存不够（OOM）

| 优先级 | 措施 |
|--------|------|
| 1 | `use_checkpoint: true` |
| 2 | `model_channels` 降一档（512→384, 384→256） |
| 3 | `batch_size_per_gpu` 降到 1 |
| 4 | SLat: `max_num_voxels` 降低（16384→12288→8192） |

### Q: 训练报 `mat1 and mat2 shapes cannot be multiplied`

`cond_channels` 必须 = CLIP 输出维度（768 for ViT-L/14）。

### Q: 训练报 `AttributeError: '...' object has no attribute 'global_step'`

`global_step` → `step`（TRELLIS 基类用 `step`）。

### Q: 多卡训练 hang 住

确保 `base.py` 已跳过 init/resume snapshot。

### Q: SS Flow 和 SLat Flow 可以同时训练吗？

**可以。** SLat Flow 训练时使用 SS Encoder 的 ground truth latent 作为 conditioning，不依赖 SS Flow 的预测。两者完全独立。

### Q: 推理时边太多/太少

调整 `--edge_threshold`：边太少 → 降至 0.3；边太多 → 升至 0.5~0.6。

### Q: LATO checkpoint 加载失败

确保 `latent_dim: 16, in_channels: 1024`，checkpoint 路径正确。
