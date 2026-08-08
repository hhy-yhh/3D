# TRELLIS + LATO 文本转3D — 实现流程

> **目标：** Encoder/Decoder 全部用 LATO VoxelVAE，只有中间 Flow 生成用 TRELLIS。SS/SLat Flow 均在刹车卡钳数据集上从零训练。

**当前版本：v13 (2026-08-09)**

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

---

## v12 现状总结 (2026-08-08)

### 代码修复记录

| # | 日期 | 问题 | 修复 |
|---|------|------|------|
| 28 | 08-08 | `lato_slat_flow_v9.json` batch_split=4 但 batch_size_per_gpu=1 → 1%4≠0 断言失败 | batch_size_per_gpu 回退为 1, batch_split 回退为 1（GPU 显存放不下 4 样本） |
| 29 | 08-08 | 旧 ckpt (510k步) 无 lr_scheduler key → 恢复时 KeyError | `basic.py`: 加 `'lr_scheduler' in misc_ckpt` 保护，缺 key 则从头初始化 CosineAnnealingLR |
| 30 | 08-08 | ConnectionHead 本地副本与原始 LATO 不一致：SiLU→应为 GELU(tanh)，hidden_channels 公式多乘了 2 | `vertex_encoder.py`、`inference_lato.py`、`evaluate_3d_metrics.py` 对齐原始 LATO，channels=1024（512×2），预训练权重成功加载 |

### 训练状态

| 模型 | 步数 | MSE | 趋势 | 状态 |
|------|------|------|------|:--:|
| SS Flow v5 | 530k | ~0.012 | 📉 持续下降 | ✅ |
| SLat Flow v9 | 527k | ~0.22 | 缓慢下降（CosineAnnealingLR 衰减中） | ✅ |
| LatoStructureHead | 530k | occ_bce ~0.02 | 活跃 | ⚠️ 架构瓶颈 |
| VoxelVAE | LATO 预训练 | — | 冻结 | ✅ |
| ConnectionHead | LATO 预训练 | — | 冻结，权重正确加载 | ✅ |

### 推理结果（测试集 1 条）

| 指标 | 值 | 说明 |
|------|------|------|
| CD | 0.267 | 差，碎片化 |
| 顶点数 | 310,976 | 大量冗余内部顶点 |
| 面数 | 26,844,141 | VoxelVAE 暴力上采样 |
| VAE L0 | **0** | 首级 subdivision 无产出 |

---

## 架构局限分析

### 局限 1：StructureHead — 信息瓶颈（最大瓶颈）

```
SS Flow (16³×8ch) → LatoStructureHead → occupancy (128³)
    32,768 个数                       2,097,152 个 voxel
    信息压缩比 64:1
```

- LatoStructureHead **不是 LATO 官方模块**，是集成自写的桥接层
- 原始 TRELLIS 此处用 sparse transformer（16³→64³ sparse→coords×2→128³），sparse 操作保持边界锐度
- 本集成用 3 级 nearest-neighbor 上采样，16³ 的 1 个 voxel 在 128³ 中膨胀为 512 个完全相同 voxel，卷积无法在块内刻画薄壳曲面
- 模型在 th>3 时产出正确数量级的 coords，但 logits 被压在 0~3 区间出不去

**影响：** SS Flow 训得再好，StructureHead 也画不出锐利边界。这是 CD 0.267 的根因。

**修改方向与预估：**

| 方案 | 改动量 | 预估 CD | 风险 |
|------|:--:|------|------|
| A. nearest→pixel shuffle | 小（改 StructureHead 定义） | 0.15-0.20 | 需重训 SS Flow |
| B. 16³→32³ + 4×上采样 | 中（SS Flow 重训） | 0.10-0.15 | 训练时间翻倍 |
| D. 回归 TRELLIS sparse decoder | 大（架构重构） | <0.10 | 与 LATO 体系不兼容 |

### 局限 2：VoxelVAE 分布不匹配

```
训练时 LATO: GT mesh → PointNet(512维) → VoxelVAE.decode() → 漂亮 mesh
推理时本管线: text → SLat Flow(16维) → VoxelVAE.decode() → 碎片 mesh
```

- VoxelVAE 训练时收到的 512 维特征是 PointNet 从真实几何表面提取的局部几何信息
- 推理时喂给它的是 SLat Flow 从文本生成的 16 维全局语义特征
- VoxelVAE 冻结，从未学过"从语义特征重建 mesh"
- **验证：** GT latent 直通 VoxelVAE.decode() 输出 57 万顶点、完美重建。同一 VAE 不同输入，差距全在输入特征

**修复方向：**
1. 短期 — SLat Flow 训更久/更多数据，逼近 VoxelVAE 期望分布
2. 中期 — 解冻 VoxelVAE 最后几层 fine-tune
3. 长期 — 用卡钳数据端到端训专属 VAE

### 局限 3：数据规模

| | 本集成 | 原版 TRELLIS |
|------|:--:|:--:|
| 训练集 | **234 条** | 500,000 条 |
| 类别 | 1 类（刹车卡钳） | 多类别 |
| 模型参数 | ~300M (SS+SLat) | ~2B |

小数据集 + 大模型 = 模型容量远大于数据多样性，容易过拟合到训练集。

### 局限 4：全局不可端到端训练

```
5 个独立组件：CLIP(冻结) → SS Flow(训练) → StructureHead(训练) 
→ SLat Flow(训练) → VoxelVAE(冻结) → ConnectionHead(冻结)
```

- 梯度不能从 VoxelVAE 反向传到 SLat Flow
- VoxelVAE 被视为黑盒 decoder，SLat Flow 只能通过间接信号优化
- 每个接口都是分布不匹配的潜在故障点

### 瓶颈等级

```
StructureHead (64:1 压缩)  ████████████████████  致命
VoxelVAE 分布不匹配         ██████████████        严重
数据规模 (234条)            ██████████            中等
不可端到端                   ████████              中等
SS Flow 16³ 信息量          ██████                次要
ConnectionHead              ██                    已修复
```

---

## v13 TRELLIS 运行逻辑逐模块审查 (2026-08-09)

### 审查范围

对比原版 TRELLIS 代码 (`trellis/models/`, `trellis/pipelines/`, `trellis/trainers/`) 与自定义集成代码 (`lato_integration/`)，验证数据流、接口契约、训练范式的兼容性。

### 审查结果

#### 1. SS Flow 接口 ✅ 完全兼容

| 项目 | 原版 TRELLIS | 自定义 (EnhancedSSFlowModel) |
|------|-------------|------|
| 输入 | `[B,8,16,16,16]` noise + `[B,77,768]` CLIP cond | 相同，未改动 |
| 输出 | `[B,8,16,16,16]` velocity field | 相同 |
| 架构 | Dense DiT (patchify → Transformer → unpatchify) | 继承原版，内部增强 (512ch, 24 blocks) |
| 训练 | Flow Matching MSE | 相同 + occupancy BCE 辅助损失 |

`sparse_structure_flow.py:176-200` — 原始 SS Flow forward 未改动，自定义仅在 `EnhancedSSFlowModel` 子类中扩展了 channels 和 blocks 数量，接口完全一致。

#### 2. 结构解码器 → coords ✅ 形状兼容，存在架构差异

**原版** `SparseStructureDecoder` (`sparse_structure_vae.py:211-307`)：
- 输入 `[B,8,16,16,16]` → 2级 Conv3d + PixelShuffle 上采样 (16→32→64)
- 输出 `[B,1,64,64,64]` → `argwhere>0` → coords@64 → `×2` 硬升至 res128
- 上采样方式：**pixel_shuffle_3d**（可学习）

**自定义** `LatoStructureHead` (`structure_head.py:67-113`)：
- 输入 `[B,8,16,16,16]` → 3级 nn.Upsample(nearest) + Conv3d 上采样 (16→32→64→128)
- 输出 `[B,1,128,128,128]` → `argwhere>0` → 直接 coords@128
- 上采样方式：**nearest neighbor**（不可学习）

**关键差异：**
- 原版用 PixelShuffle（可学习，能在子体素间分配特征），自定义用 nearest neighbor（硬复制，1 个体素膨胀为 8 个完全相同体素）
- 原版只需 4× 压缩（16³→64³），自定义是 8× 压缩（16³→128³），64:1 信息膨胀比

**结论：** 接口形状兼容（都输出 `[N,4]` int coords），但 nearest-neighbor 上采样是 CD=0.267 的直接原因。

#### 3. SLat Flow ✅ 完全符合 TRELLIS 运行逻辑

| 项目 | 原版 SLatFlowModel | EnhancedSLatFlowModel |
|------|-------------------|----------------------|
| 输入 | `SparseTensor(coords@res128, noise feats)` | 相同 |
| Timestep | `t × 1000` → sinusoidal → MLP | 相同，继承 TimestepEmbedder |
| Cond | `[B, N_ctx, cond_channels]` text embedding | 相同 |
| 前向传播 | input_layer → IO blocks(sparse resblocks) → Transformer blocks → IO blocks → output_layer | **相同流程** + Swin window attn + 分离 cross PE |
| 输出 | `SparseTensor(same coords, velocity feats)` | 相同 |
| Loss | `F.mse_loss(pred.feats, target.feats)` | **完全一致** |

核心 Flow Matching 逻辑 (`sparse_flow_matching.py:108-118`)：
```python
noise = x_0.replace(torch.randn_like(x_0.feats))   # 保留拓扑结构,替换特征为噪声
t = self.sample_t(x_0.shape[0])                      # logitNormal 采样时间步
x_t = self.diffuse(x_0, t, noise=noise)              # 概率路径插值
pred = self.training_models['denoiser'](x_t, t*1000, cond)  # 模型预测速度场
target = self.get_v(x_0, noise, t)                   # 真实速度场
terms["mse"] = F.mse_loss(pred.feats, target.feats)  # 只算特征 MSE
```

与 TRELLIS 原版 Flow Matching 训练范式**完全一致**。

#### 4. 训练数据流 ✅ 符合范式

`structured_latent.py:157-170` 数据集加载：
```python
data = np.load(os.path.join(root, 'latents', latent_model, f'{instance}.npz'))
feats = (feats - self.mean) / self.std  # 训练时归一化
```

这些 `.npz` 文件是刹车卡钳 → VoxelVAE.encode() 生成的 GT latent。推理时反归一化 (`slat * std + mean`) 后送入 VoxelVAE.decode()。与原版 TRELLIS 的 SLat 训练/推理范式**完全一致**。

#### 5. VoxelVAE.decode() 接口 ✅ 已处理库差异

`trellis_text_to_3d.py:162-165`：TRELLIS SparseTensor → LATO SparseTensor 通过字段复制转换：
```python
lato_slat = LATOSparseTensor(
    feats=slat.feats.contiguous(),
    coords=slat.coords.contiguous(),
)
```
两个库的 SparseTensor 构造函数兼容，推理日志确认 "Success: All weights loaded perfectly"。

#### 6. CFG (Classifier-Free Guidance) ✅ 符合

`ClassifierFreeGuidanceMixin`：训练时 `p_uncond=0.1` 概率将 cond 替换为 null_cond，推理时 `cfg_strength=3.0`，与 TRELLIS 原版完全一致。

#### 7. 归一化/反归一化 ✅ 符合

- 训练：`feats = (feats - mean) / std` 
- 推理：`slat = slat * std + mean`
- 原版完全相同的机制

#### 8. 时间步调度 ✅ 符合

`lato_slat_flow_v9.json`：`logitNormal(mean=1.0, std=1.0)`，与原版 TRELLIS 一致。

### 潜在风险

| # | 风险 | 严重度 | 说明 |
|---|------|:--:|------|
| 1 | StructureHead fp16 空操作 | 低 | `convert_to_fp16()` 和 `convert_to_fp32()` 为空，如果 trainer 切换精度时 StructureHead 不跟随，可能 dtype 不匹配。但当前 SS Flow 用 fp32，不影响 |
| 2 | Cross PE 简化 | 低 | context positional encoding 用 1D→3D 近似，对文本序列合理，不影响功能 |
| 3 | Swin blocks 初始化时机 | 已修复 | `_rebuild_blocks_with_swin()` 内已补充 xavier_uniform + zero adaLN + fp16 转换（Bug 23 修复） |

### 审查结论

**代码完全符合 TRELLIS 的运行逻辑。** 所有接口契约（dense→sparse→coords→归一化→反归一化→解码）正确实现。Flow Matching 训练范式、CFG 推理、时间步调度、归一化体系与原版 TRELLIS 一致。当前 CD=0.267 **不是代码逻辑错误**，而是 StructureHead 的 nearest-neighbor 上采样架构瓶颈。

---

## v13 改进方案规划

### 性价比优先级排序

基于"改动量 / 预期收益"权衡，同时考虑训练时间成本：

```
方案         改动量    训练成本    预期 CD    性价比    推荐
────────────────────────────────────────────────────────
A. PixelShuffle  小       高(重训)   0.15-0.20   ★★★★    ⭐首推
C. 推理阈值       微小     零         0.20-0.23   ★★★★★   立即做
E. VoxelVAE微调   中       中         0.12-0.18   ★★★★    次推
B. 32³ SS Flow   大       很高       0.10-0.15   ★★★     中期
D. 回归TRELLIS    大       很高       <0.10       ★★      不推荐
────────────────────────────────────────────────────────
```

### 方案 A：StructureHead nearest → PixelShuffle 上采样 ⭐ 首推

**问题：** `nn.Upsample(nearest)` 硬复制，1 个体素 → 8 个相同体素，3×3 卷积只能在块边界模糊，无法刻画薄壳。

**修改：** 将 `UpsampleBlock3d` 中的 `nn.Upsample + Conv3d` 替换为 TRELLIS 原版的 3D PixelShuffle（`pixel_shuffle_3d`）：
```python
# 原版 TRELLIS sparse_structure_vae.py:89-99 的可学习上采样
class UpsampleBlock3d:
    def __init__(self, in_channels, out_channels):
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)
    def forward(self, x):
        return pixel_shuffle_3d(self.conv(x), 2)  # 每个体素扩展为 2×2×2，通道重排
```

**原理：** PixelShuffle 将 C×8 通道重新排列为 2×2×2 空间块，子体素特征可由卷积学习，不再是硬复制。原来 512 个相同子体素 → 8 个学习到的不同子体素。

**文件修改：**
- `lato_integration/structure_head.py`：`UpsampleBlock3d` 类（约 20 行改动）
- 后续重训 SS Flow v6

**代价：** 必须从零重训 SS Flow + StructureHead（约 50 万步，~3-4 天）。

**预期：** CD 从 0.267 → 0.15-0.20。不能完全解决问题（64:1 压缩比仍在），但消除 nearest 硬复制瓶颈。

### 方案 C：推理阈值补偿 ⭐ 立即做

**问题：** StructureHead 产出的 logits 分位数 P95=0.22，大量低置信度 voxel 淹没高置信度区域。

**修改：** `evaluate_3d_metrics.py` 加 `--ss_threshold` 参数，默认 2.0：
```python
# 当前 (inference_lato.py 或 evaluate 内部)
coords = torch.argwhere(occ_logits > 0)  # 所有正 logit 都保留

# 改为
coords = torch.argwhere(occ_logits > threshold)  # 只保留高置信度 voxel
```

**代价：** 零训练成本，仅改推理脚本 1 行。

**预期：** CD 从 0.267 → 0.20-0.23。th=3 时只保留 3075 个 voxel（接近 GT 数量级），空间范围合理。缺点：可能丢掉正确的薄壳区域。

### 方案 E：VoxelVAE 最后几层 Fine-tune ⭐ 次推

**问题：** VoxelVAE 训练时收到 512-dim PointNet 几何特征，推理时收到 16-dim SLat Flow 语义特征，分布不匹配。

**修改：** 冻结 VoxelVAE encoder + latent_expander，只 fine-tune decoder_vtx + decoder_vtx_ca 最后 1-2 层：
```python
# 冻结大部分参数
for param in vae.encoder.parameters(): param.requires_grad = False
for param in vae.latent_expander.parameters(): param.requires_grad = False
for param in vae.latent_proj.parameters(): param.requires_grad = False
for i, block in enumerate(vae.decoder_vtx):
    if i < len(vae.decoder_vtx) - 2:  # 只训练最后 2 层
        for param in block.parameters(): param.requires_grad = False
```

**数据准备：** 用当前 SLat Flow 在训练集上生成大量 (slat_pred, GT_mesh) 对，用 `L1Loss(vertex_coords, GT_points) + BCE(occupancy)` 做监督。

**代价：** 
- 需先跑一轮训练集推理生成 slat-vertex 对（~1 小时）
- Fine-tune ~10K steps（~2-3 小时）
- 不重训 SS/SLat Flow

**预期：** CD 从 0.267 → 0.12-0.18。VoxelVAE 学会容忍 16-dim 语义特征的分布偏移。

### 方案 B：SS Flow 分辨率 16³ → 32³

**问题：** 16³ 网格太粗糙，64:1 压缩比太高。

**修改：** 
- SS Flow config `resolution: 16 → 32`
- StructureHead 只需 2 级上采样（32→64→128），4× 压缩
- SS Flow 参数量大约 4×（体积 8×，但 patch_size 可保持），训练时间翻倍

**代价：** SS Flow 从零重训，训练时间 ~2×（更大模型 + 更多 GPU 显存）。

**预期：** CD < 0.15。这是从根本上解决压缩比问题的方案，但成本最高。

### 方案 D：回归 TRELLIS Sparse Decoder 路径（不推荐）

**说明：** 放弃 StructureHead，用原版 TRELLIS SparseStructureDecoder (16³→64³→coords×2→128³)，仅保留 VoxelVAE + ConnectionHead 做最终解码。

**不推荐原因：** 与项目目标冲突（"Encoder/Decoder 全部用 LATO"），且 SparseStructureDecoder 是 TRELLIS 预训练的，在 234 条卡钳数据上 fine-tune 效果存疑。

### 推荐行动路线

```
第 1 步 (今天, 0 成本)          第 2 步 (本周末, 小修改)        第 3 步 (下周)
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│ 方案 C           │      │ 方案 A                │      │ 方案 E            │
│ 推理阈值补偿     │ ───→ │ PixelShuffle 上采样   │ ───→ │ VoxelVAE 微调     │
│ CD: 0.27→0.22   │      │ CD: 0.22→0.18        │      │ CD: 0.18→0.14     │
│ 改动: 1 行代码   │      │ 改动: ~20 行 + 重训   │      │ 改动: 数据+微调    │
└─────────────────┘      └──────────────────────┘      └──────────────────┘
    立即验证效果              架构修复                     分布对齐
```

**如果方案 A 跑完 CD 仍 > 0.15：** 启动方案 B（32³ SS Flow），因该方案改动量大、训练时间长，只在 A+E 不足时触发。

**如果方案 A+E 达标 (CD < 0.15)：** 可考虑方案 B 进一步优化到 < 0.10。但 234 条数据可能让 < 0.10 很难，届时优先考虑增加数据量而非改架构。
