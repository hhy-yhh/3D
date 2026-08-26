# TRELLIS + LATO 文本转3D — 实现流程

> **目标：** Encoder/Decoder 全部用 LATO VoxelVAE，只有中间 Flow 生成用 TRELLIS。SS/SLat Flow 均在刹车卡钳数据集上从零训练。

**当前版本：v18 (2026-08-26)**

---

## 架构

```
Text → CLIP → SS Flow ──→ LatoStructureHead → coords@128³
              (训练)       (训练, 16³→128³)     ① PixelShuffle 可学习上采样
                                │
                                ▼
              SLat Flow ──→ LATO VoxelVAE.decode() → ConnectionHead → Mesh
              (训练)         (微调 decoder 末2层)     (冻结预训练)
                             ② 原"冻结"→现"微调"
```

| 组件 | 来源 | 状态 |
|------|------|:--:|
| CLIP | `openai/clip-vit-large-patch14` | 冻结 |
| SS Flow | `EnhancedSSFlowModel` (512ch × 24 blocks) | **训练** |
| LatoStructureHead | 3D CNN 16³→128³ | **训练**（① nearest → PixelShuffle） |
| SLat Flow | `EnhancedSLatFlowModel` (384ch × 12 blocks, Swin) | **训练** |
| LATO VoxelVAE | 预训练 128→512 | **微调**（② 冻结 encoder，微调 decoder 末 2 层） |
| ConnectionHead | LATO 预训练边预测器 | 冻结 |

---

## 当前训练

```
SS Flow v6 (GPU 4)                   SLat Flow v10 (GPU 7)
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
| SS Flow ckpt | `outputs/lato_ss_flow_v6/ckpts/denoiser_step*.pt` |
| StructureHead ckpt | `outputs/lato_ss_flow_v6/ckpts/structure_head_step*.pt` |
| SLat config | `configs/generation/lato_slat_flow_v9.json` |
| SLat Flow ckpt | `outputs/lato_slat_flow_v10/ckpts/denoiser_step*.pt` |
| SLat stats | `lato_latents_v2/latents/lato_vae_16dim_128/stats.json` |
| LATO VAE | `/data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt` |

---

## 命令

### 训练

```bash
cd /data/huanghaoyang/3D/TRELLIS

# SS Flow v6（GPU 4）
CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v3.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/lato_ss_flow_v6 \
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

SS_CKPT=$(ls outputs/lato_ss_flow_v6/ckpts/denoiser_step*.pt | sort -V | tail -1)
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

**自动检测训练类型：** 目录下有 `structure_head_step*.pt` → SS Flow，否则 → SLat Flow。

```bash
cd /data/huanghaoyang/3D/TRELLIS

# SS Flow v6（GPU 4）— 自动检测为 ss 类型
bash check_health.sh outputs/lato_ss_flow_v6 4 --data_dir /data/huanghaoyang/3D/database_lato

# SLat Flow v10（GPU 7）— 自动检测为 slat 类型（无 structure_head ckpt）
bash check_health.sh outputs/lato_slat_flow_v10 7

# 也可手动指定类型（--type ss|slat）
# 每 10000 步跑一次（~30 秒），不占训练 GPU
```

**v16 新增检查项：**

| 新增项 | 位置 | 说明 |
|--------|------|------|
| PixelShuffle 架构验证 | SS 专项 | 检查 conv 权重 `out_ch = base×8`，确认是 PixelShuffle 而非旧 nearest |
| StructureHead 权重匹配 | SS 专项 | 报告 "已加载 N/M 匹配"，缺失则警示架构不兼容 |
| lr_scheduler 配置 | SLat 专项 | 验证 CosineAnnealingLR 的 T_max 和 eta_min 是否正确 |
| 从零训练友好提示 | 通用 | 早期无 log/ckpt 时提示"从零训练，早期正常"，不误报 |

#### 判断标准

**SS Flow：**

| 指标 | 正常 | 异常 |
|------|:--:|:--:|
| SH Arch (PixelShuffle) | `out_ch = base×8` | `out_ch = base` (旧 nearest 架构) |
| SH Wt | 全部匹配 | 缺失参数（架构不兼容） |
| SH 3D | X/Y/Z span > 10, n > 0 | 全负或任一轴 span=0 |
| occ_bce | > 0.01 | 趋近 0 或不存在 |
| MSE | 持续下降 | 不降或上升 |
| NaN | 0 | > 0 |

**SLat Flow：**

| 指标 | 正常 | 异常 |
|------|:--:|:--:|
| lr_scheduler | CosineAnnealingLR 已配置 | 未配置 |
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
| SS Flow v6 | 530k | ~0.012 | 📉 持续下降 | ✅ |
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

---

## v13 方案 C 实施结果 (2026-08-09)

### 修改内容

4 个文件修改，将 StructureHead 推理阈值从 `0.0` 改为 `2.0`：

| 文件 | 修改 |
|------|------|
| `structure_head.py:128` | `coords_from_occupancy` 默认 threshold `0.0` → `2.0` |
| `evaluate_3d_metrics.py:397` | `--ss_threshold` 默认值 `0.0` → `2.0` |
| `inference_lato.py:425` | 新增 `--ss_threshold` 参数（默认 `2.0`） |
| `inference_lato.py:209-239` | `sample_ss_lato()` 接受并传递 `ss_threshold` |
| `trellis_text_to_3d.py:109-137` | `sample_sparse_structure_lato()` 新增 `ss_threshold` 参数 |

### 测试结果

| 策略 | 配置 | coords | CD | 结论 |
|------|------|--------|-----|------|
| 旧默认 | th=0, 不限 | 504,886 | 0.267 | 实心块 |
| 方案 C (v13 默认) | th=2.0, 不限 | 79,541 | OOM | 仍太多 |
| 阈值 + 截断 | th=1.0, top-16384 | 16,384 | **0.267** | 不变 |
| 阈值 + 截断 | th=2.0, top-16384 | 16,384 | 未跑 | — |

### 诊断日志

```
[SS diag] logits percentiles: P0=-73.14 | P50=-13.42 | P90=-3.63 | P95=0.86 | P99=4.40
[SS diag]   active(>0.0) = 123586 (5.9%)
[SS diag]   active(>1.0) = 101923 (4.9%)
[SS diag]   active(>2.0) =  79541 (3.8%)
[SS diag]   active(>5.0) =  11950 (0.6%)
[VAE L0] vertices=0                              ← 首级 subdivision 完全失败
```

### 方案 C 结论：❌ 架构瓶颈，阈值调参无效

阈值成功将 coords 从 50 万降到 16K，但 **CD 纹丝不动（0.267 → 0.267）**。原因：

1. **阈值只能过滤，不能纠正位置。** StructureHead 的 nearest-neighbor 上采样把高 logits 分散到了大块区域，取 top-K 只是从模糊块中挑最高分，位置仍然不精确
2. **logits 最高分位置也不在薄壳曲面上。** 即使只取 16K 个最高置信度 voxel（仅占 128³ 的 0.8%），mesh 仍然是碎块
3. **VAE L0=0 未改善。** 首级 subdivision 仍无产出，说明输入 VoxelVAE 的 sparse tensor 拓扑结构不支持 vertex hierarchy 的 subdivision

**方案 C 的价值：** 确认了问题不在 voxel 数量或阈值，而在 voxel 的**空间位置精度**——必须从架构层面改进。

---

## v14 新增方案：LogitSharpener（后训练锐化，不重训）

### 动机

方案 A（PixelShuffle）必须重训 SS Flow，用户不愿重训。需要一种**只训练极轻量模块、不改原有权重**的方案。

### 核心思路

```
SS Flow (冻结) → StructureHead (冻结) → LogitSharpener (训练，~1K 参数)
                                              │
                                    模糊 logits → 锐利 logits
```

StructureHead 输出的 logits **空间位置基本正确**（th>3 时 3075 个 voxel 的 bbox 范围合理），只是边界模糊。一个小型 3D CNN 可以学"边缘增强"——本质是 3D unsharp mask。

### 架构设计

```python
class LogitSharpener(nn.Module):
    """
    极轻量 3D 锐化模块：学习将模糊 occupancy logits 锐化为锐利边界。
    
    架构：3 层 3×3×3 depthwise-separable conv，参数量 ~500。
    输入 [B, 1, 128, 128, 128] → 输出 [B, 1, 128, 128, 128]
    """
    def __init__(self):
        # Layer 1: 1×1×1 → 4 channels
        # Layer 2: 3×3×3 depthwise conv
        # Layer 3: 4 → 1 output
        # Total: ~500 params
```

### 训练策略

| 步骤 | 说明 | 时间 |
|------|------|------|
| 1. 数据生成 | 冻结 SS Flow + StructureHead，推理 234 条训练集 → 保存 blur logits + GT occupancy | ~1 小时 |
| 2. 训练 | 只训 LogitSharpener，loss = BCE(sharpened_logits, GT_occupancy) | **~10-20 分钟** |
| 3. 评估 | 用训练好的 LogitSharpener 推理测试集 | ~1 分钟 |

### 与其他方案对比

| | 方案 C (阈值) | 方案 F (LogitSharpener) | 方案 A (PixelShuffle) |
|------|:--:|:--:|:--:|
| 改动量 | 1 行 | ~30 行新类 | ~20 行改架构 |
| 重训需求 | 无 | 只训 LogitSharpener | 重训 SS Flow (50万步) |
| 训练时间 | 0 | **10-20 分钟** | 3-4 天 |
| 预期 CD | 0.267 (无效) | **0.18-0.22** | 0.15-0.20 |
| 风险 | 低 | 低 | 中（ckpt 不兼容） |

### 为什么方案 F 可能有效

1. **信息已存在。** StructureHead 的 logits 在 th>3 时空间范围合理，高 logits 聚集在目标区域附近——只是太模糊
2. **锐化是局部的。** 3D unsharp mask 本质是 `output = input + α * (input - blurred_input)`，3×3×3 conv 只需学一个局部增强核
3. **GT 监督强。** 234 条 GT occupancy 直接提供逐体素监督信号，BCE loss 比 Flow Matching MSE 精确得多
4. **不改任何已训练权重。** SS Flow、StructureHead、SLat Flow、VoxelVAE 全部冻结

### 风险评估

- **如果 StructureHead logits 位置本身就错 → 无效。** 但 th>3 时的诊断表明位置基本正确
- **锐化可能引入噪声。** 过强的锐化在空白区域产生假阳性。用 L1/L2 正则化控制
- **不如方案 A 彻底。** 锐化是后处理，不能修复结构性问题。但成本低，值得先试

---

## v14 根因分析：生成模型质量差的宏观/微观原因

### 宏观原因（系统/架构层面）

#### 1. 管道碎片化，5 个独立黑盒串行

```
CLIP → SS Flow → StructureHead → SLat Flow → VoxelVAE → ConnectionHead → Mesh
  ❄️      🔥          🔥            🔥          ❄️            ❄️
```

梯度只能在 3 个训练模块（🔥）内部流动，组件间完全割裂。VoxelVAE 的误差信号无法反传给 SLat Flow，SLat Flow 只能通过间接的 MSE loss 猜测 VoxelVAE 喜欢什么输入。

#### 2. 数据规模差 2000 倍

| | 本集成 | 原版 TRELLIS |
|------|:--:|:--:|
| 训练集 | 234 | 500,000 |
| 类别数 | 1 | 多类 |
| 模型参数 | ~300M | ~2B |

大模型 + 小数据 = 模型容量远超数据多样性。模型学会了"记住"训练样本的特征分布，但从未学会泛化。生成时偏离训练分布一点点，下游 VoxelVAE 就解读为噪声。

#### 3. 架构目标冲突

```
TRELLIS 设计目标： 通用文本→3D 生成（需要多样性、语义理解）
LATO 设计目标：    特定形状的保拓扑解码（需要精确几何特征）

集成后的实际需求：  文本 → 精确的刹车卡钳几何
                               ↑
                     这个跨度，中间缺了"几何理解"这一步
```

SS/SLat Flow 擅长生成"语义正确"的粗糙结构，VoxelVAE 擅长把"几何精确"的 latent 解码为拓扑正确的 mesh。但桥接处（StructureHead + SLat Flow）输出的 latent 既不语义正确也不几何精确。

#### 4. 缺少端到端验证回路

原版 TRELLIS 有渲染器做可视化监督，LATO 有重建 loss 做几何监督。集成管道没有任何中间监督——只能在最终 CD 看到一个数字，无法知道哪一步出了什么问题。

---

### 微观原因（逐组件定位）

#### 1. StructureHead：nearest-neighbor 上采样 → 边界模糊（核心瓶颈）

```
16³ 的 1 个特征 voxel → nearest ×2 → ×2 → ×2 → 128³ 的 512 个完全相同的 voxel
```

3×3×3 卷积的感受野只能在块边界做平滑，无法在 8×8×8 硬块内部刻画 1-voxel 厚度的曲面。结果是 logits 被"摊大饼"：

```
P95 = 0.86   ← 95% 的 voxel 都没把握
P99 = 4.40   ← 最高 1% 也不够锐利
max = 9.34   ← 极端值太低（锐利边界通常 max > 50）
```

#### 2. VoxelVAE 输入分布漂移（隐藏瓶颈）

```
训练 VoxelVAE 时： GT mesh → PointNet(512-dim) → 512 维几何特征
                                        ↓ 每个顶点携带：法向量、曲率、局部几何
推理时：         text → SLat Flow(16-dim) → 16 维语义特征
                                        ↓ 每个 voxel 只有：全局语义方向
```

这不是"好坏"的差异，是**信息量 32:1 的降维**。VoxelVAE 从未学过从降维特征重建——它期望收到的 512-dim 里每个维度都携带具体几何信息。

**直接症状：VAE L0=0**

```
LATO VoxelVAE 解码流程：
L0: latent_expander(16-dim) → vertex prediction head → sigmoid > threshold?
    这一步期望在 64³ 空间中找到"哪些地方是顶点"
    但 16-dim 语义特征无法告诉它哪里是顶点 → threshold 不触发 → 0 个顶点
    
L1: 从 L0 顶点做 subdivision。但 L0=0 → 只能暴力 subdivide 所有活跃 coords
    → 38,869 顶点（无意义的全量采样）
    
L2: 继续 subdivide → 310,952 顶点 + 26,844,141 面（爆炸）
```

#### 3. StructureHead logits 空间位置对但精度不够

```
th>3 时 3075 个 voxel  → bbox 范围合理（卡钳形状）
但 logits 最大 = 9.34   → 足够"存在"但不够"精确"
```

模型知道卡钳在哪里（位置对），但画不出精确的曲面（位置不够对）。

#### 4. Flow Matching MSE 无法给几何反馈

SLat Flow 训练时：
```python
loss = MSE(pred.feats, gt.feats)  # 逐特征值比较
```

但 VoxelVAE 关心的不是 feats 的数值误差，而是 feats 隐含的几何结构。MSE=0.22 在数值上不算大，但对 VoxelVAE 来说，这 0.22 的误差可能意味着"顶点应该在这儿还是那儿"的巨大差异。

---

### 因果链

```
数据少（234条）
  ↓
SS Flow 学到的是模糊的平均特征（而非锐利的几何边界）
  ↓
StructureHead nearest-neighbor 把模糊放大 512 倍（1→512 个相同 voxel）
  ↓
SLat Flow 在模糊的 coords 上填充模糊的 feats
  ↓
VoxelVAE 收到 16-dim 模糊语义特征 + 位置模糊的 coords
  ↓
L0 无法检测顶点 → 暴力 subdivide → 31 万顶点 + 2684 万面 → CD=0.267
```

---

## v14 核心诊断：流程图/代码逻辑正确 ≠ 模块连接有效

### 两个层面的"通"

经过 v13 TRELLIS 逐模块审查，确认：
- **流程图正确** — SS Flow → StructureHead → coords → SLat Flow → VoxelVAE → ConnectionHead → Mesh 的设计逻辑无误
- **代码接口正确** — 所有模块间数据格式（dense tensor、SparseTensor、coords、feats）形状/类型匹配，能跑通不报错

但 **CD=0.267 不变**。问题在更深一层：

```
代码层面（✅ 通）：   模块 A 输出 → 模块 B 输入    格式正确，管道不漏
分布层面（❌ 不通）： 模块 A 产出的"内容"  ≠  模块 B 训练时见过的"内容"
```

### 逐接口诊断

| # | 接口 | 代码连接 | 分布匹配 | 详情 |
|---|------|:--:|:--:|------|
| 1 | SS Flow → StructureHead | ✅ | ✅ | 联合训练，16³ dense 特征分布一致 |
| 2 | **StructureHead → coords** | ✅ | **❌** | nearest-neighbor 上采样：1 个体素膨胀为 512 个相同体素，logits 被平均分摊。边界模糊的 coords 送入下游 |
| 3 | coords → SLat Flow | ✅ | **⚠️** | coords 位置本身就偏了，SLat Flow 在错误的拓扑上填 feats |
| 4 | **SLat Flow → VoxelVAE** | ✅ | **❌** | **16-dim 语义特征 vs 512-dim 几何特征**，信息量 32:1 降维 |
| 5 | VoxelVAE → ConnectionHead | ✅ | ✅ | 同体系预训练，无分布问题 |

### 接口 2 的深层分析：StructureHead → coords

```
StructureHead 要做的事：
    16³ × 8ch = 32,768 个值  →  在 128³ = 2,097,152 个 voxel 中找出 ~5000 个正确的

能做到吗？
    ✅ 位置对：模型知道卡钳在哪儿（th>3 时 bbox 范围合理）
    ❌ 精度不够：nearest×2×2×2 让 1 个特征点硬复制为 8×8×8=512 个相同 voxel
       3×3 卷积感受野只能覆盖块之间的过渡区域，无法在块内部刻画 1-voxel 厚的曲面
```

这不是代码 bug，是 **nn.Upsample(nearest) 的物理极限**。用 32,768 个信息点直接内插出 2,097,152 个点的锐利边界，数学上不可能。

### 接口 4 的深层分析：SLat Flow → VoxelVAE

```
训练时 VoxelVAE 收到的输入：
    GT mesh → PointNet(512-dim) → [每个顶点：位置、法向量、曲率、局部邻域几何]
    VoxelVAE 学到的映射： 丰富的局部几何特征 → 精确的顶点 subdivision

推理时 VoxelVAE 收到的输入：
    text → SLat Flow(16-dim) → [每个 voxel：全局语义方向，16 个维度的"卡钳-ness"]
    VoxelVAE 被要求做： 16 维语义 → 精确的顶点 subdivision
    
但实际上 VoxelVAE 从未学过这个映射。
它期望 512 维里每个维度都携带具体几何信息，
却收到 16 维"这是一个卡钳"的模糊语义信号。
```

这不是"特征质量差"，是 **特征的本体论差异**：
- 512-dim PointNet 特征描述的是 **"这个顶点处的局部几何是什么形状"**
- 16-dim SLat 特征描述的是 **"这个 voxel 大概是不是卡钳的一部分"**

两者完全不在同一个语义空间里。

### 两个接口的叠加效应

```
StructureHead 模糊 (接口 2) × VoxelVAE 分布漂移 (接口 4) = 复合故障

单独一个还能容忍：
  - 如果只有接口 2 坏：VoxelVAE 可以用 512-dim 几何特征从模糊 coords 中恢复形状
  - 如果只有接口 4 坏：锐利的 coords 可以补偿 16-dim 的语义贫乏

两个同时坏：
  - 模糊 coords + 语义贫乏 feats → VoxelVAE 完全无解 → L0=0 → 暴力 subdivide → CD=0.267
```

### 所以问题不在流程图，也不在代码逻辑

| 层面 | 状态 |
|------|:--:|
| 流程图设计 | ✅ 正确 |
| 代码接口（形状/类型/格式） | ✅ 通过 v13 逐行审查 |
| TRELLIS 训练范式 | ✅ 符合 |
| **跨模块分布对齐** | **❌ 2 个关键接口失效** |

这就是为什么改阈值、修 ConnectionHead、加 CosineAnnealingLR 都改变不了 CD——这些修的都是 **代码正确性**，而问题出在 **分布层面**，代码写对了也修不了。

---

## v15 设计与实现偏离分析 (2026-08-09)

### 设计目标回顾

```
Encoder/Decoder 全部用 LATO VoxelVAE，只有中间 Flow 生成用 TRELLIS
```

### 实际现状

```
Text → CLIP → SS Flow ──→ LatoStructureHead(自写) → coords
             (TRELLIS)   (既不是TRELLIS也不是LATO)
                                │
                                ▼
              SLat Flow ──→ VoxelVAE.decode() → ConnectionHead → Mesh
              (TRELLIS)     (LATO一半)            (LATO)
                            ↑
                      训练收512-dim PointNet几何特征
                      推理收16-dim Flow语义特征 ← 不对齐
```

### 三个设计偏离

| # | 设计目标 | 实际 | 后果 |
|---|---------|------|------|
| 1 | 结构解码用 LATO | 自写 LatoStructureHead (nearest 上采样) | 64:1 压缩，边界模糊，既非 TRELLIS 也非 LATO |
| 2 | LATO encoder 产出特征 | 没有 LATO encoder，特征来自 SLat Flow | 512-dim 几何 vs 16-dim 语义，VoxelVAE 收不认识的语言 |
| 3 | Decoder 用 LATO | 只用了 VoxelVAE 解码器一半，缺其配对编码器 | 解码器期望的几何特征无人提供 |

### 问题定性

**不是代码 bug，不是 TRELLIS 和 LATO 接口不兼容，而是设计适配不到位。**

- v13 已确认：代码逻辑正确，TRELLIS 和 LATO 的接口格式也对齐（张量形状、Flow Matching 范式、CFG 推理均与原版一致）
- 三个模块的适配层过于简陋，连接处存在"语言不通"——模块 A 说几何，模块 B 说语义，模块 C 两边都不搭

### 生成质量不佳的根因总结（设计视角）

```
问题一 ─ 结构解码是临时桥接，不是任何一个体系的标准模块
        │
        ├─ TRELLIS 原版用 SparseStructureDecoder (PixelShuffle 可学习上采样)
        ├─ LATO 体系根本没有这个模块（LATO 只做保拓扑编解码）
        └─ 自写的 nearest-neighbor 上采样 16³→128³ 信息冲淡 512 倍
           │
           ▼
问题二 ─ VoxelVAE 缺了配对编码器，特征来源不对
        │
        ├─ 训练时代: GT mesh → PointNet(512-dim) → VoxelVAE.encode → latent → decode
        ├─ 推理时代: text → CLIP → SS/SLat Flow(16-dim) → VoxelVAE.decode
        └─ 16-dim 语义 ≠ 512-dim 几何，解码器从未学过这个映射
           │
           ▼
问题三 ─ 两个偏离叠加，框架图虽窄但宽在错误的地方
        │
        ├─ 单独一个还勉强：模糊 coords + 好特征 或 好 coords + 弱特征 都能凑合
        ├─ 同时出现：模糊 coords + 错类型特征 → L0=0 → 暴力填充
        └─ 精度差 3 倍（CD=0.218 vs 充分收敛 0.08）
```

---

## v16 代码改造 + 训练流程 (2026-08-12)

### 改造目标

修复 v15 识别的两个模块不对齐问题：

```
问题一: StructureHead nearest-neighbor 硬复制 → 边界模糊
    → 改为 PixelShuffle 可学习上采样（与 TRELLIS 原版对齐）

问题二: VoxelVAE decoder 不适应 SLat Flow 特征
    → 加噪微调 decoder 最后 2 层，让 VoxelVAE 容忍 SLat Flow 误差
```

### 代码改动清单

| # | 文件 | 改动 | 原因 |
|---|------|------|------|
| 1 | `structure_head.py` | `UpsampleBlock3d` 从 `nn.Upsample(nearest)` 改为 `Conv3d(in, out*8) → pixel_shuffle_3d(2)` | 子体素特征可学习分配，不再硬复制 512 份 |
| 2 | `structure_head.py` | `convert_to_fp16/32` 从空操作改为 `self.half()/float()` | PixelShuffle 有可学习参数，需真实转换 |
| 3 | `finetune_vae.py` | **新文件** — VoxelVAE decoder 微调脚本 | 加噪训练最后 2 层，适配 SLat Flow 特征 |
| 4 | `lato_ss_flow_v3.json` | `lambda_occupancy: 1.0` → `0.1` | 辅助 loss 权重过高会拖慢主 loss 收敛 |
| 5 | `evaluate_3d_metrics.py` | 新增 `--vae_ft_ckpt` 参数 | 支持加载微调后的 VoxelVAE |
| 6 | `evaluate_3d_metrics.py` | StructureHead 权重加载加 ⚠️ 警告 | 旧 ckpt 与新架构不匹配时提醒用户 |

### 架构兼容性

| 组件 | 架构变了？ | 旧 ckpt 兼容？ | 策略 |
|------|:--:|:--:|------|
| SS Flow | ❌ | — | 从零训 |
| StructureHead | ✅ Conv3d(in, out*8) | ❌ 权重形状不同 | 从零训 |
| SLat Flow | ❌ | — | 从零训（v10） |
| VoxelVAE | ❌ (微调后覆盖) | — | 微调覆盖 decoder |

### 训练流程

**数据不变**，和 v5/v9 用同一份：

| 数据 | 路径 | 说明 |
|------|------|------|
| metadata | `database_lato/metadata.csv` | 训练集 234 条 |
| occupancy | `database_lato/ss_occupancy_128_v2/` | GT mesh → 体素化 |
| GT latents | `database_lato/lato_latents_v2/latents/lato_vae_16dim_128/` | GT mesh → VoxelVAE.encode() |

#### 环境变量

```bash
# SS Flow
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa

# SLat Flow
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export SPARSE_ATTN_BACKEND=xformers

# 推理（两个 backend 都设）
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
```

#### 第 1 步：SS Flow v6 + StructureHead 从零训练（GPU 4）

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa

CUDA_VISIBLE_DEVICES=4 python lato_integration/run_train.py \
    --config configs/generation/lato_ss_flow_v3.json \
    --data_dir /data/huanghaoyang/3D/database_lato \
    --output_dir outputs/lato_ss_flow_v6 \
    --num_gpus 1
```

#### 第 2 步（并行）：SLat Flow v10 从零训练（GPU 7）

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export SPARSE_ATTN_BACKEND=xformers

CUDA_VISIBLE_DEVICES=7 python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow_v9.json \
    --data_dir /data/huanghaoyang/3D/database_lato/lato_latents_v2 \
    --output_dir outputs/lato_slat_flow_v10 \
    --num_gpus 1
```

#### 第 3 步：VoxelVAE 微调（SS Flow 训完后）

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"

python lato_integration/finetune_vae.py \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --gt_latents /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/ \
    --output_dir outputs/vae_finetuned \
    --epochs 50
```

#### 第 4 步：全量推理评估

```bash
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers

SS_CKPT=$(ls outputs/lato_ss_flow_v6/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow_v10/ckpts/denoiser_step*.pt | sort -V | tail -1)

python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/stats.json \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --vae_ft_ckpt outputs/vae_finetuned/vae_finetuned.pt \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/eval_v6_v10 \
    --ss_threshold 2.0 \
    --max_coords 16384
```

### 时间线

```
GPU 4: SS Flow v6 从零训 ──────────────── 3-4 天 ──→ 完成
GPU 7: SLat Flow v10 从零训 ────────────── 3-4 天 ──→ 完成
                                                    │
                                            VoxelVAE 微调 (~3h)
                                                    │
                                                推理评估
```

---

## v16 改进总结（表格 + 当前流程图）

> 相对 v13 旧图，仅改两处模块内部实现，数据流/连线/分辨率均不变。

### 当前流程图

```
Text → CLIP → SS Flow ──→ LatoStructureHead → coords@128³
              (训练)       (训练, 16³→128³)     ① PixelShuffle 可学习上采样
                                │
                                ▼
              SLat Flow ──→ LATO VoxelVAE.decode() → ConnectionHead → Mesh
              (训练)         (微调 decoder 末2层)     (冻结预训练)
                             ② 原"冻结"→现"微调"
```

### 组件状态（当前）

| 组件 | 来源 | 当前状态 | 备注 |
|------|------|:--:|------|
| CLIP | `openai/clip-vit-large-patch14` | 冻结 | 不变 |
| SS Flow | `EnhancedSSFlowModel` (512ch×24) | **训练** | v6 从零重训 |
| LatoStructureHead | 3D CNN 16³→128³ | **训练** | ① nearest → PixelShuffle |
| SLat Flow | `EnhancedSLatFlowModel` (384ch×12, Swin) | **训练** | v10 从零重训 |
| LATO VoxelVAE | 预训练 128→512 | **微调** | ② 冻结 encoder，只微调 decoder 末 2 层 |
| ConnectionHead | LATO 预训练边预测器 | 冻结 | 不变 |

### 主要改进总表

| 优先级 | 模块（流程图位置） | 改进内容 | 改进目的 / 预期效果 |
|:--:|------|---------|-------------------|
| ⭐ | **LatoStructureHead** | nearest 上采样 → **PixelShuffle 可学习上采样** | 消除"1 体素硬复制 512 份"的边界模糊 → CD 0.267 → **0.15~0.20** |
| ⭐ | **VoxelVAE.decode()** | 冻结 → **微调 decoder 末 2 层**（加噪自蒸馏） | 适配 16-dim 语义特征，L0 正常触发、避免暴力细分 → CD → **0.12~0.18** |
| — | SS Flow config | `λ_occupancy` 1.0 → **0.1** | 降低辅助 loss 权重，加速主 loss 收敛 |
| — | 推理脚本 | 新增 `--vae_ft_ckpt` / `--ss_threshold` / `--max_coords` | 加载微调 VAE + 控制 coords 数量防 OOM |
| — | structure_head.py | `convert_to_fp16/32` 空操作 → 真实转换 | PixelShuffle 有可学习参数，需真实精度切换 |
| — | check_health.sh | 新增 PixelShuffle 架构 / 权重匹配检查 | 训练健康可见，确认架构正确 |

### 与旧图（v13）差异

| 位置 | v13 旧图 | v16 现图 |
|------|---------|---------|
| LatoStructureHead | `nn.Upsample(nearest)`，硬复制 | **PixelShuffle** 可学习上采样 |
| VoxelVAE.decode() | 冻结预训练 | **微调 decoder 末 2 层**（加噪自蒸馏） |

> 说明：表中"512 份"= 16³→128³ 每维放大 8 倍（128÷16=8），体积膨胀 8×8×8=512；与 VoxelVAE 的 128→512（解码分辨率）、PointNet 的 512-dim（特征维度）是三个不同的"512"，勿混淆。

---

## v17 VAE 剪枝头微调（BCE 方案）(2026-08-24)

### 背景与问题

v16 的 VoxelVAE 微调（feats-L1 加噪自蒸馏）**无效**——它只对齐顶点特征、不监督顶点选择，且从未解冻 `vtx_head_64`。评估结果：

```
[VAE L1] vertices=35754
[VAE L2] vertices=199964   # L2/L1 = 5.59×（理想 ≈ 1×）
v=199961  f=17646706  CD=0.2700
```

三个叠加 bug：

| # | 问题 | 状态 |
|---|---|---|
| ① | `decode_slat_lato` 没传 `vis_last_layer=False` → 末层 force_no_prune → 保证 8× 爆炸 | ✅ 已修 `trellis_text_to_3d.py` |
| ② | 坐标缩放 ÷256 应为 ÷512 → mesh 放大 2×，CD 虚高 | ✅ 已修 `evaluate_3d_metrics.py`/`inference_lato.py` |
| ③ | **pruning_head 未校准** → 每级细分 ~70% 子体素被保留，剪枝失效 | ⬅️ 本方案解决 |

③ 的本质：剪枝头在「encoder 精确 latent」上训练，推理却喂「SLat Flow 带噪 latent」（MSE≈0.17），分布漂移 → 剪枝头输出全高分 → 不剪枝。

### 方案

用 `decode(training=True) + gt_vertex_voxels_list`（GT mesh 薄表面体素化）做 occupancy BCE：

```
decode(training=True, gt_vertex_voxels_list=[gt_128, gt_256, gt_512])
  ├─ L0: BCE(vtx_head_64, vertex_mask)        # 顶点选择：表面=1 / 膨胀区=0
  ├─ L1: BCE(pruning_head[0], prune_labels)   # 128→256 剪枝
  └─ L2: BCE(pruning_head[1], prune_labels)   # 256→512 剪枝
```

只解冻 3 个决策头（**2.7% 参数**），各级之间隔着冻结的 decoder，梯度本就传不过去。

### 代码改动清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `finetune_vae.py` | **重写**：feats-L1 → occupancy BCE；新增 `voxelize_mesh_surface`（薄表面体素化）、`bce_with_balance`（pos_weight 平衡）；只解冻 3 个头；`--max_coords` 防 OOM |
| 2 | `inference_lato.py` | 新增 `--vae_ft_ckpt` 参数（覆盖加载微调权重） |
| 3 | `evaluate_3d_metrics.py` | 无需改（v16 已有 `--vae_ft_ckpt`） |

关键实现细节：

- `voxelize_mesh_surface()`：`o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds` 只体素化三角面（薄表面），**不做点云膨胀**，保证薄表面 ⊆ 厚壳 latent coords。
- `occ_probs`/`vtx_feats` 用 `BCE_with_logits`：源码确认 `SparsePredictionHead` 直接 `return self.mlp(x)`（无 sigmoid），`SparseVertexSubdivideBlock3d` 的 sigmoid 只在推理阈值时用，返回的 `occ_prob` 是原始 logits。
- `freeze_for_ft()` 只解冻 `vtx_head_64` + `decoder_vtx[i].upsample.pruning_head`（~450 万参数 / 2.7%）。

### 训练

```bash
cd /data/huanghaoyang/3D/TRELLIS
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python lato_integration/finetune_vae.py \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --gt_latents /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/ \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/vae_finetuned_bce \
    --max_coords 8000 \
    --epochs 30
```

- 产出：`vae_finetuned.pt`（最终）+ `vae_ft_epoch{N}.pt`（每 10 epoch 快照，可提前验证）。
- 耗时 ~2.5~3 小时（体素化 ~14 min + 训练 ~5.5 min/epoch）。
- 启动应打印 `Trainable: ~4,470,787 / 165,884,963 (2.7%)`。

### 断点续训 + 体素化缓存

`finetune_vae.py` 支持 `--resume`（断点续训）与体素化磁盘缓存：

- **`--resume <vae_ft_epoch{N}.pt>`**：恢复 模型权重 + optimizer 动量 + cosine 学习率相位。加载后手动把 scheduler 步进到 `start_epoch`，LR 与从头跑到该 epoch 完全一致。**只能从 `vae_ft_epoch{N}.pt` 续**（含 optimizer/epoch），不能从 `vae_finetuned.pt` 续（只有 `model_state_dict`）。
- **体素化缓存**：预计算循环先查 `output_dir/gt_cache/<key>.npz`，命中直接读，不再重跑 open3d 体素化（省 ~14 min）。第一次跑完自动生成。

续训命令：

```bash
python lato_integration/finetune_vae.py \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --gt_latents /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/ \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/vae_finetuned_bce \
    --max_coords 8000 \
    --epochs 30 \
    --resume outputs/vae_finetuned_bce/vae_ft_epoch10.pt
```

- **`--output_dir` 必须与原来一致**，否则找不到 `gt_cache` 和要续的 checkpoint。

### 生效方式

微调产物是「补丁」，加载顺序：预训练 `vae_128to512.pt` → `load_state_dict(微调权重, strict=False)` 覆盖 3 个头。

- **评估**：`evaluate_3d_metrics.py` 加 `--vae_ft_ckpt outputs/vae_finetuned_bce/vae_finetuned.pt`
- **生成**：`inference_lato.py` 加 `--vae_ft_ckpt ...`（v17 新增该参数）

### 判断是否生效

| 指标 | 微调前 | 微调后（预期） |
|---|---|---|
| L2/L1 顶点比值 | 5.59× | ~1× |
| 顶点数 | ~200,000 | 30,000~50,000 |
| 面数 | ~17,600,000 | 大幅下降 |

### 注意事项

- **OOM**：训练模式细分不剪枝（`lato_vae.py:94` 只在 `not training` 剪枝），窗口注意力 O(N×邻域)，需 `--max_coords` 压输入（8000 起，OOM 降到 5000）。
- **微调无效**：`--lr` 提到 3e-5；或加 `--noise_std 0.15` 模拟 SLat Flow 误差增强鲁棒。
- **断点续训**：已支持 `--resume` + 体素化落盘缓存，见上方「断点续训 + 体素化缓存」。
- **边界**：只修剪枝头，不修 SLat Flow latent 误差（MSE≈0.17 信息瓶颈），几何精度仍受 latent 质量限制，不会完美还原 GT。

---

## v18 用 LATO 完整 encoder/decoder 精化草稿 mesh (2026-08-26)

### 背景与动机

项目目标：**纯 TRELLIS 生成不理想，LATO 效果好 → 直接用 LATO 的 encoder/decoder 增强生成效果**（Encoder/Decoder 用 LATO，中间 Flow 生成保留 TRELLIS）。

排查发现（**修正 `诊断与修改总结_2026-08-26.md` 的结论**）：

1. **fp16 不需要**。`database_lato/lato_latents/latents/lato_vae_16dim_128/*.npz` 全部 234 个 GT latent 的 coords 都是 **16384**（编码硬上限），**不存在「52416 被截断」**。诊断总结的「fp16 喂 52416」前提不成立。
2. 实验 A（GT coords + GT feats 完美输入）在 16384 下 CD=0.307 → **瓶颈不在 voxel 数量**。
3. 真正的瓶颈是 **「VAE 输入分布不匹配」+「顶点云→mesh 粗糙」**（即旧文档「局限 2」）：
   - LATO VoxelVAE 训练时吃 **PointNet 1024 维几何特征 encode 出的 16 维 latent**；
   - 现有管线却把 **SLat Flow 的 16 维语义 latent** 直接喂给 decoder → VAE 从未学过从语义特征重建 → decode 出的顶点云散乱/多层 → 再用 KDTree 乱连边建 mesh → 「没有完整的面、全是细小三角」。

### 思路：两阶段「LATO 精化」

```
Stage 1（现有，不动）: TRELLIS 生成草稿 mesh
  Text → SS Flow → StructureHead → coords@128³ → SLat Flow → VAE.decode → ConnectionHead → 草稿 mesh

Stage 2（新增，LATO 完整 encoder/decoder）:
  草稿 mesh → load_quantized_mesh_original(体素化+点特征[P,15])
           → VoxelFeatureEncoder_active_pointnet → 1024 维几何特征
           → VAE.encode → 16 维 latent（LATO 原生分布！）
           → VAE.decode → 干净顶点层级 → ConnectionHead → 精化 mesh
```

Stage 2 让 VAE decoder 拿到它**训练时见过的输入类型**（PointNet 几何特征 → encode → decode），绕开分布不匹配——这是「让 LATO 发挥」缺失的 encoder 半边。

**零重训、零 fp16**：voxel_encoder / VAE / ConnectionHead 全用 LATO 预训练权重；Stage 2 输入经 LATO 预处理天然 ≤16384 voxel，不撞 spconv int32。

### 代码改动清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `lato_integration/refine_lato.py`（**新增**） | `build_voxel_encoder()`（构建 + `.eval()`）+ `refine_mesh_with_lato()`（草稿 mesh → 体素化 → PointNet → VAE.encode → decode → ConnectionHead），改编自 LATO 官方 `scripts/infer_vae_512.py` 的 `reconstruct_mesh`。LATO 的 `vertex_encoder` 用 `importlib` 按绝对路径加载，避开本地同名模块 |
| 2 | `lato_integration/evaluate_3d_metrics.py` | ① `load_pipeline` 在 `--refine_lato` 时加载 voxel_encoder（`load_pretrained_woself` 加 `voxel_encoder=`）；② 新增 `--refine_lato` 参数；③ 草稿 mesh 提取后精化，用精化 mesh 算指标/保存 |
| 3 | `lato_integration/inference_lato.py` | 同 evaluate：加载 voxel_encoder + `--refine_lato` + 草稿 mesh 构建后精化再导出 |

实现要点：
- `vertex_threshold` 跟随 `--lato_threshold`（默认 0.2），避免 LATO 官方 0.5 在原生路径触发 L0=0 → top-2 兜底 → 57万+ 顶点 → 建 mesh 极慢。
- 精化后尽早 `del decoded + empty_cache`，KDTree 建 mesh 阶段不占 GPU 显存。
- 默认关闭（`--refine_lato` 不加 = 行为与之前完全一致）。

### 用法

```bash
# 单条验证（仿原文档，只加 --refine_lato）
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" --slat_ckpt "$SLAT_CKPT" \
    --slat_stats /data/huanghaoyang/3D/database_lato/lato_latents_v2/latents/lato_vae_16dim_128/stats.json \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database_lato/meshes \
    --output_dir outputs/eval_refine_test --limit 1 --save_meshes --refine_lato
```

### 注意事项

- **voxel_encoder 权重必须加载成功**：日志 `--- Loading status for 'VoxelEncoder' ---` 必须是 `Success: All weights loaded perfectly`，否则随机初始化 → 精化结果是垃圾。
- **Stage 2 修 mesh 质量、不修形状**：LATO 重构的是草稿的「形状」——草稿形状对则精化干净完整，草稿形状烂则精化后仍是烂形状（只是变干净）。
- **性能**：每样本多一次 体素化(CPU) + PointNet encode + VAE encode→decode + KDTree 建 mesh，批量评估会明显变慢。

---

## v19 真正根因：CD 度量 bug + KDTree 三角汤 (2026-08-26 深夜)

### 结论（推翻 v18 方向）

v18 的「LATO encoder/decoder 精化」解决的是**不存在的问题**。排查后确认真正的根因有两个，都与生成本身无关：

1. **CD 评估坐标系不匹配**（主因，骗了所有人）
2. **KDTree 建 mesh 三角汤**（视觉主因）

### 问题 1：CD 坐标系 bug

```
decode 输出: 归一化空间 [-0.5, 0.5]（span ~1）
GT mesh:     原始毫米单位（span ~350，例如卡钳 355×212×126 mm）
```

旧 `compute_all_metrics` 把 pred 和 GT **都除以 GT 的毫米对角线（~430）** → pred 被缩成针尖大的点（span ~0.002）→ CD 量的是「大 GT 面到一个小点」的距离 → **恒 ~0.27，与生成质量完全无关**。

- 后果：所有实验（A/B/C/D、refine、grid、40K voxel）CD 全 ≈0.27——因为全在量同一个垃圾，怎么改都动不了。
- **验证**：用真实 GT mesh（毫米）+ 完美重建（和 GT 同形状）在旧方法下得 **0.26 分**，新方法得 **0.000**。旧 CD 是坏的，不是编的。
- **修复**：`_normalize_to_unit`（pred 和 GT 各自居中 + 缩放到单位 bbox 对角线）→ 修复后 GT 输入 0.0093、SS 生成路径 **0.0022**。

### 问题 2：KDTree 三角汤

`predict_edges_batched`（KDTree k=32 乱连边）+ 公共邻居三角枚举，对 512³ 上 20 万顶点 → **1790 万面**细小三角——这就是「没有完整的面、全是细小三角形」的直接来源。

- **修复**：`mesh_grid.py` 格点 6-邻域连边 + ConnectionHead 打分 + 四边形化 → **47 万完整面**。

### 问题 3：观感（方块 / 纸糊）

- grid 格点网格**非流形** → Laplacian 平滑炸出尖刺（射线），已撤销平滑。
- 512³ 粒度 → 表面阶梯/方块感。
- **修复**：**Poisson 表面重建**（open3d，流形、水密、光滑）→ 对流形网格再平滑（安全）→ 从「方块/纸糊」变「光滑卡钳」。

### 最终结果

| 场景 | CD | NC |
|---|---|---|
| A 组（GT 完美输入） | 0.0093 | — |
| SS 生成路径（grid 四边形化） | 0.0022 | 0.46 |
| SS 生成路径（Poisson 重建） | 0.0019 | **0.66** |

**结论：生成管线几何是正确的（CD 0.002）。「生成质量差」是 CD bug + 三角汤造成的假象，不是架构/训练/生成能力问题。**

### 代码改动清单

| 文件 | 改动 |
|---|---|
| `evaluate_3d_metrics.py` | `compute_all_metrics` 归一化修复 + pred/GT bbox/CD 分解诊断；`--mesh_mode grid/knn/poisson`；`--refine_lato`（已证无用但保留）；`extract_mesh_from_output` 支持三种模式 |
| `mesh_grid.py`（新增） | `predict_edges_grid`（格点 6-邻域候选边）、`build_mesh_from_grid`（四边形化）、`build_mesh_from_poisson`（Poisson + 平滑）、`_postprocess_mesh`（fix_normals + Laplacian） |
| `inference_lato.py` | 接入 `--mesh_mode` / poisson 分支 |
| `experiment_coords_vs_feats.py` | 接入 `--mesh_mode` / poisson 分支 |
| `refine_lato.py`（新增，已证无用） | LATO encoder/decoder 精化（v18 方案，无效） |

### 待办 / 可选
- 「纯 TRELLIS vs 当前 LATO 管线」对比**未做**——要回答「LATO 集成是否必要」需在修复指标下跑原版 TRELLIS 对照。
