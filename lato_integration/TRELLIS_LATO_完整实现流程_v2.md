# TRELLIS + LATO 文本转3D — 实现流程

> **目标：** Encoder/Decoder 全部用 LATO VoxelVAE，只有中间 Flow 生成用 TRELLIS。SS/SLat Flow 均在刹车卡钳数据集上从零训练。

**当前版本：v10 (2026-08-04)**

---

## 架构

```
Text → CLIP → SS Flow ──→ LatoStructureHead → coords@128³
              (训练)       (训练, 16³→128³)
                                │
                                ▼
              SLat Flow ──→ LATO VoxelVAE.decode() → ConnectionHead → Mesh
              (训练)         (冻结预训练)             (冻结预训练)
```

| 组件 | 来源 | 状态 |
|------|------|:--:|
| CLIP | `openai/clip-vit-large-patch14` | 冻结 |
| SS Flow | `EnhancedSSFlowModel` (512ch × 24 blocks) | **训练** |
| LatoStructureHead | 3D CNN 16³→128³ (~1-2M 参数) | **训练** |
| SLat Flow | `EnhancedSLatFlowModel` (384ch × 12 blocks, Swin) | **训练** |
| LATO VoxelVAE | 预训练 128→512 | 冻结 |
| ConnectionHead | LATO 预训练边预测器 | 冻结 |

---

## 当前训练

```
SS Flow v5 (GPU 4)                   SLat Flow v10 (GPU 7)
─────────────────────                ──────────────────────
config: lato_ss_flow_v3.json         config: lato_slat_flow_v9.json
data:   database_lato/               data:   lato_latents_v2/
model:  EnhancedSSFlowModel          model:  EnhancedSLatFlowModel
        + LatoStructureHead                   + Swin window attn (w=8)
        fp32 (use_fp16=false)                 + separate cross PE
target: ss_occupancy_128_v2 (3D)             fp16 (use_fp16=true)
loss:   FlowMSE + occBCE(λ=0.1)     target: 3D latent feats@128
                                     loss:   SparseFlowMSE + latent_cons(λ=1e-4)
```

---

## 数据目录

```
/data/huanghaoyang/3D/database_lato/
├── metadata.csv                          # 训练集 234 条
├── test/metadata.csv                     # 测试集 21 条
├── meshes → meshes_normalized            # 归一化 STL [-0.5, 0.5]
├── ss_occupancy_128_v2/                  # 3D occupancy (234 npz)
└── lato_latents_v2/                      # 3D LATO latent
    └── latents/lato_vae_16dim_128/
        ├── *.npz (234)                   # coords [N,4] + feats [N,16]
        └── stats.json                    # mean/std (16-dim)
```

---

## 模型文件

| 模型 | 路径 |
|------|------|
| SS config | `configs/generation/lato_ss_flow_v3.json` |
| SS Flow ckpt | `outputs/lato_ss_flow_v5/ckpts/denoiser_step*.pt` |
| StructureHead ckpt | `outputs/lato_ss_flow_v5/ckpts/structure_head_step*.pt` |
| SLat config | `configs/generation/lato_slat_flow_v9.json` |
| SLat Flow ckpt | `outputs/lato_slat_flow_v10/ckpts/denoiser_step*.pt` |
| SLat stats | `lato_latents_v2/latents/lato_vae_16dim_128/stats.json` |
| LATO VAE | `/data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt` |

---

## 命令

### 训练

```bash
cd /data/huanghaoyang/3D/TRELLIS

# SS Flow v5（GPU 4）
CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v3.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/lato_ss_flow_v5 \
    --num_gpus 1

# SLat Flow v10（GPU 7）
CUDA_VISIBLE_DEVICES=7 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow_v9.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents_v2 \
    --output_dir outputs/lato_slat_flow_v10 \
    --num_gpus 1
```

### 推理（评估）

**环境变量：**

| 变量 | 值 | 作用 | 原因 |
|------|------|------|------|
| `PYTHONPATH` | `LATO:TRELLIS` | 导入 LATO + TRELLIS 模块 | 两个项目不在同一目录 |
| `ATTN_BACKEND` | `sdpa` | SS Flow dense attention | fp32 兼容，PyTorch 内置 |
| `SPARSE_ATTN_BACKEND` | `xformers` | SLat Flow sparse attention | sparse 只支持 xformers/flash_attn；fp32 仅 xformers 可用 |

**命令：**

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers

SS_CKPT=$(ls outputs/lato_ss_flow_v5/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v10/ckpts/denoiser_step*.pt | sort -V | tail -1)

# 单条快速测试（调试用：--limit 1 --save_meshes）
# ⚠️ --slat_stats 必须传！不传 = identity fallback → VoxelVAE 拒解 → 散块
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/stats.json \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/eval_v10_test \
    --limit 1 --save_meshes

# 全量评估（21 条测试集，去掉 --limit）
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/stats.json \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/eval_v10_full \
    --save_meshes
```

### 健康检查

```bash
cd /data/huanghaoyang/3D/TRELLIS

# SS Flow（自动检测：目录下有 structure_head_step*.pt）
bash check_health.sh outputs/lato_ss_flow_v5 0 --data_dir /data/huanghaoyang/3D/database_lato

# SLat Flow（自动检测：无 structure_head ckpt）
bash check_health.sh outputs/lato_slat_flow_v10 0 --type slat

# 不占训练 GPU，每 10000 步跑一次（~30 秒）
```

#### 判断标准

**SS Flow：**

| 指标 | 正常 | 异常 |
|------|:--:|:--:|
| SH 3D | X/Y/Z span > 10, n > 0 | 全负或任一轴 span=0 |
| occ_bce | > 0.01 | 趋近 0 或不存在 |
| MSE | 持续下降 | 不降或上升 |
| NaN | 0 | > 0 |

**SLat Flow：**

| 指标 | 正常 | 异常 |
|------|:--:|:--:|
| SLat Forward | mean∈[-10,10], NaN=0, Inf=0 | NaN/Inf |
| MSE | 持续下降 | 不降或上升 |
| NaN | 0 | > 0 |

#### 成型判定（推理输出）

| 指标 | 散块 | 开始成型 | 充分收敛 |
|------|:--:|:--:|:--:|
| CD ↓ | >0.20 | 0.10-0.15 | <0.08 |
| 顶点数 | <500 | 1000-3000 | >3000 |
| 面数 | <200 | 500-2000 | >2000 |

---

## 推理全链路

```
Step 1 ─ CLIP 文本编码
│  功能: prompt → text embeddings [1, 77, 768] + null embedding (CFG)
│  文件: trellis/pipelines/trellis_text_to_3d.py → get_cond()
│        (内部调用 transformers.CLIPTextModel)
│
├─ Step 2 ─ SS Flow 去噪采样
│  功能: 随机噪声 → 20步 Flow Matching Euler 迭代 → dense latent [1, 8, 16³]
│  文件: trellis/pipelines/trellis_text_to_3d.py → sample_sparse_structure_lato()
│        trellis/pipelines/samplers/flow_euler.py → FlowEulerSampler.sample()
│        lato_integration/flow/ss_flow.py → EnhancedSSFlowModel.forward()
│
├─ Step 3 ─ LatoStructureHead → coords
│  功能: dense 16³ → 3级 2× 上采样 → occupancy logits [1, 1, 128³]
│        → argwhere(>0) → coords [N, 4] int @ res128
│  文件: lato_integration/structure_head.py → LatoStructureHead.forward()
│                                         → coords_from_occupancy()
│
├─ Step 4 ─ SLat Flow 去噪采样 + 反归一化
│  功能: coords + 随机噪声 → SparseTensor → 20步迭代去噪 → slat [N, 16]
│        → slat = slat * std + mean  (反归一化，恢复原始特征尺度)
│  文件: trellis/pipelines/trellis_text_to_3d.py → sample_slat()
│        lato_integration/flow/slat_flow.py → EnhancedSLatFlowModel.forward()
│        归一化/反归一化: trellis/datasets/structured_latent.py (训练)
│                         trellis_text_to_3d.py:236-238 (推理)
│
├─ Step 5 ─ TRELLIS → LATO SparseTensor 转换 → VoxelVAE decode
│  功能: TRELLIS SparseTensor → LATO SparseTensor (不同库，不同实现)
│        → VoxelVAE.decode(training=False) → 多级 vertex hierarchy
│  文件: trellis/pipelines/trellis_text_to_3d.py → decode_slat_lato()
│        LATO: lato/models/lato_vae/lato_vae.py → VoxelVAE.decode()
│
└─ Step 6 ─ ConnectionHead → 边预测 → 三角面片化 → Mesh
   功能: vertex_coords/feats → KDTree(k=32) → 候选边对
        → ConnectionHead → sigmoid → threshold(0.45) 过滤
        → NetworkX 公共邻居法 → 三角面 → trimesh.Trimesh → .obj
   文件: lato_integration/inference_lato.py → predict_edges_batched()
                                            → edges_to_mesh()
        LATO: vertex_encoder.py → ConnectionHead
```

---

## 关键注意事项

1. **`--slat_stats` 必须传** — 训练归一化和推理反归一化必须用同一组 stats，否则 VoxelVAE 收到错误尺度 feats → 散块（Bug 17）
2. **模型架构自动匹配** — `build_flow_model_from_config()` 从 config JSON 的 `name` 字段自动选择模型类，v8 ckpt 自动升级到 EnhancedSLatFlowModel
3. **SS fp32 / SLat fp16 互不干扰** — 两者之间只传递 int 坐标，无精度依赖
4. **Swin window attention 必须从零训练** — 架构变了（full→windowed + separate PE），不能从 v8 resume

---

## 历史 Bug 摘要

| # | 日期 | 描述 | 修复 |
|---|------|------|------|
| 1-6 | 07-14~08-01 | None grad/NaN/OOM/Adam state/dtype/ckpt 加载等 | 训练器健壮性修复 |
| 7 | 07-25 | fp16 权重溢出 → 53 万步 NaN | SS Flow 改 fp32 + 权重钳 |
| 8-9 | 07-26 | 推理 dtype 不匹配 + StructureHead ckpt 加载 | fp32 推理 + 独立 ckpt glob |
| 10-12 | 07-30~31 | StructureHead BCE 缺 pos_weight / 权重污染 / fp16 溢出 | 全量重置 + fp32 autocast 禁用 |
| 13-15 | 08-01 | MRO 断裂 / x_0 溢出 / Transformer fp16 激活溢出 | get_inference_cond 覆盖 + clamp |
| 16 | 08-03 | StructureHead 重置后 Adam state 未清 | 同步清理 exp_avg/exp_avg_sq |
| 17 | 08-04 | SLat Flow stats.json 缺失 → 反归一化无效 → 散块 | 生成 3D stats + 推理传 --slat_stats |
| 18 | 08-04 | evaluate_3d_metrics.py GT 路径不匹配 | 改用 CSV file_path |
| 19-20 | 08-04 | EnhancedSLatFlowModel dtype 不匹配 (ctx_pe + Swin blocks) | fp16 转换修复 |
| 21 | 08-04 | **SLat normalization std 错误 (0.05 应为 0.6~1.2)** → MSE 9.6 (应为 ~2) | stat_latent.py float16 累加溢出 + 用正确 stats 重算 config |
| 22 | 08-04 | check_health.sh MSE 误匹配 bin_* 嵌套值 + log entries 语义错误 + 早期训练误报 | grep 非贪婪匹配 + wc -l + ckpts/ 目录检测 |
| 23 | 08-04 | EnhancedSLatFlowModel._rebuild_blocks_with_swin() 未对 Swin blocks 做 xavier_uniform 初始化 | 添加 xavier_uniform + zero adaLN |
