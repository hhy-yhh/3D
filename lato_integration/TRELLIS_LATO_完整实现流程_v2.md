# TRELLIS + LATO 文本转3D — 实现流程

> **目标：** Encoder/Decoder 全部用 LATO VoxelVAE，只有中间 Flow 生成用 TRELLIS。SS/SLat Flow 均在刹车卡钳数据集上从零训练。

**当前版本：v11 (2026-08-07)**

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
| 24 | 08-06 | stats.json std 全部 ~0.05（正确 0.5~1.2）→ 推理反归一化无效 → 散块 | stat_latent.py 用 float64 重算 stats.json 覆盖 |
| 25 | 08-07 | evaluate_3d_metrics.py ConnectionHead channels=512，ckpt 期望 768 → 半数 MLP 参数未加载 | 待修复 |
| 26 | 08-07 | **LatoStructureHead 模块适配失败** — 16³→128³ nearest-neighbor 上采样，8× 压缩比下无法产生薄壳 occupancy（详见下方专项分析）| 架构需重新设计 |
| 27 | 08-07 | SLat Flow 训练停滞 — 48 万步 MSE 对数收敛至 ~0.26，缺 LR scheduler + batch_split=1 | 加 CosineAnnealingLR + batch_split=4 |

---

## 专项分析：LatoStructureHead 模块适配失败 (Bug 26)

### 现象

SS Flow MSE=0.007（充分收敛），但推理时 128³ 网格中 50 万 voxel > 0（24%），GT 卡钳仅 ~5000（0.24%）。mesh 碎片化，CD=0.267，水密=False。

### 逐模块排查结果

| 模块 | 代码正确？ | 适配通过？ | 结论 |
|------|:--:|:--:|------|
| SS Flow | ✅ | ✅ | MSE 0.007，充分收敛 |
| **LatoStructureHead** | ✅ | **❌** | 架构能力上限 |
| SLat Flow | ✅ | ✅ | MSE 0.26，对数收敛中 |
| VoxelVAE.decode() | ✅ | ✅ | GT latent 直通：57 万顶点，0% 稀疏 |
| ConnectionHead | ❌ | — | channels 512≠768，半数参数未加载 |

### 根因：物理上限，不是调参能解决的

```
TRELLIS 原版：16³ → 64³ (sparse transformer) → coords ×2 → 128³
             4× 压缩，sparse 操作保持边界锐度

LatoStructureHead：16³ → 128³ (nearest-neighbor upsample)
                   8× 压缩，每个 16³ 网格点膨胀为 8×8×8 硬块
```

`nn.Upsample(scale_factor=2, mode='nearest')` 三次叠加后，16³ 的 1 个 voxel 在 128³ 中变成 512 个完全相同的 voxel。3×3 卷积只能在块边界做模糊，无法在块内部刻画 1 voxel 厚的曲面。

**模型内部已经知道卡钳位置**（th>3 时 3075 个 voxel 的 X/Y/Z 范围合理），但 logits 被分散到周围 50 万 voxel 上，无法形成锐利边界。

### GT 直通验证

取任意训练集样本的 GT latent → VoxelVAE.decode()：

| 指标 | GT latent | 说明 |
|------|:--:|------|
| VAE 顶点 | 57.3 万 | 多级 subdivision 正常输出 |
| Level 0/1/2 | 0 / 7.2万 / 57.3万 | 层级结构正常 |
| 最近邻中位数 | 0.004 | 顶点密实均匀 |
| 稀疏率 (NN>0.03) | 0% | 无裂隙 |

**VAE 完全正常。** 问题锁死在 SS Flow → LatoStructureHead → coords 这一段。

### 正确的 coords 在哪里

```python
# 128³ 网格中 logits 分位数：
P0=-91.91  P50=-11.34  P90=-5.09  P95=0.22  P99=4.54  P99.9=6.66

# 不同阈值下的 voxel 数：
th>0 = 504,886  (24%)   ← 全是实心块
th>1 =  51,951  (2.5%)  ← 开始收敛
th>2 =  12,509  (0.6%)  ← 接近真实数量
th>3 =   3,075  (0.15%) ← 范围合理，卡钳形状
th>5 =     153          ← 太少了
```

模型在 th>3 时能产出正确数量级和空间范围的 coords，但正样本的 logits 被压在 0~3 之间出不去。这是因为 nearest-neighbor 上采样在粗糙网格上的信息瓶颈导致 sigmoid 永远不敢出高分。

### 修复方向

| 方向 | 说明 |
|------|------|
| **A. 改架构** | 将 nearest-neighbor 上采样替换为 3D pixel shuffle（可学习上采样），让模型学会在 8×8×8 块内分配子体素特征 |
| **B. 降压缩比** | 将 SS Flow 分辨率从 16³ 提升到 32³，StructureHead 只需 4× 上采样（32³→128³ = 2 阶段） |
| **C. 推理端阈值补偿** | 不改训练架构，推理时用 `--ss_threshold > 2.0` 硬过滤低置信度 voxel。缺点：可能丢掉正确但低置信度的薄壳区域 |
| **D. 回归 TRELLIS 路径** | 不在 128³ 做 dense occupancy，改为 16³→64³ sparse→coords×2（跟 TRELLIS 保持一致） |

---

## 专项分析：SLat Flow 训练对数收敛 (Bug 27)

### 现象

48.65 万步，MSE 缓慢对数下降而非真正停滞：

```
50k→100k:  0.51 → 0.44  (-31%)  快速
100k→200k: 0.44 → 0.34  (-24%)  较快
200k→300k: 0.34 → 0.28  (-18%)  放缓
300k→400k: 0.28 → 0.27  (-15%)  更慢
400k→480k: 0.27 → 0.26  (-15%)  对数收敛
最近 3000 步均值: 0.22
```

没有 NaN，万步波动仅 ±0.004。模型在 lr=1e-4 + batch_size=1 下已经学到了优化上限，需要更精细的优化。

### 原因

- `lato_slat_flow_v9.json` 无 `lr_scheduler` → lr 恒定 1e-4
- `batch_split: 1, batch_size_per_gpu: 1` → 单样本梯度方差大
- Swin window attention 在 fp16 下精度有限

### 修复（不改架构，不改 fp16）

在 `lato_slat_flow_v9.json` 加：

```json
"batch_split": 4,
"lr_scheduler": {
    "name": "CosineAnnealingLR",
    "args": { "T_max": 1000000, "eta_min": 1e-6 }
}
```

从当前 48.65 万步 ckpt resume 即可，不需重训。
