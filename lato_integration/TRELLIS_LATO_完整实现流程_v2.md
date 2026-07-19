# TRELLIS + LATO 文本转3D — 完整实现流程 v5

> **目标：** TRELLIS 文本转3D 管线中，将 Sparse VAE Encoder/Decoder 替换为 LATO 的 VoxelVAE，**SS Flow 和 SLat Flow 均在刹车卡钳数据集上从零训练**，最后用 Chamfer Distance / Hausdorff Distance / Normal Consistency 评估。

**更新记录：**
- 2026-07-19：**v6 更新** — 端到端测试集推理验证通过；修复 6 个推理 bug；更新实际训练目录名和 working commands
- 2026-07-17：**v5 更新** — 推理脚本重构：从 config JSON 读取模型参数、自动发现最新 ckpt、新增 ss_only 模式
- 2026-07-16：**v4 更新** — 新增步骤6（批量评估 + 3D 指标），推理脚本支持 `--ss_ckpt`
- 2026-07-16：**v3 更新** — 新增 SS Flow 训练步骤，训练模型数从 1 个变为 2 个
- 2026-07-14：修复 12 个 bug + 4 个训练启动 bug，单卡 RTX 4090 24GB 可行

---

## 架构概览

```
                    原版 TRELLIS                        你的 LATO 管线                    对应目标
                   ════════════                       ═══════════════                   ════════

  Text ──→ CLIP ──→ SS Flow ──→ SS Decoder           (不变) (不变)  (🆕重训)  (不变)
                                    │                    │       │       │       │
                                    │                Text ─→ CLIP ─→ SS Flow ─→ SS Decoder   ① 刹车卡钳形状适配
                                    │                                        │
                                    ▼                                        ▼
                               SLat Flow                              coords ×2                ② 适配 LATO res=128
                         (res=64, dim=8, 1024ch)               (res=64 → 128 上采样)
                                    │                                        │
                                    │                                        ▼
                                    │                              SLat Flow (🆕重训)           ③ 刹车卡钳潜空间适配
                                    │                         (res=128, dim=16, 384ch)         + 匹配 LATO latent 维度
                                    │                                        │
                                    ▼                                        ▼
                          ┌─── SLat Decoder ───┐                   LATO VoxelVAE               ④ 更高质量的几何解码
                          │  GS  │  RF  │ Mesh │                   .decode()                   (cross-attn + pruning)
                          └───────────────────┘                        │
                               3 种输出                           ConnectionHead               ⑤ 显式拓扑预测
                                                                   (边预测 → 三角面片)
                                                                        │
                                                                        ▼
                                                                      Mesh                     ⑥ 仅需 mesh 输出
                                                                     (.obj)
```

## 代码逐行核查：改动 ↔ 目标 ↔ LATO 模块对应

### 推理全链路追踪（`pipeline.run()` → mesh.obj）

```
TrellisTextTo3DPipeline.run()          # trellis_text_to_3d.py:212
│
├─[1] sample_sparse_structure()        # :86
│   ├─ self.models['sparse_structure_flow_model']    ← 🆕 你的 SS Flow ckpt
│   └─ self.models['sparse_structure_decoder']       ← TRELLIS 预训练 (冻结)
│
├─[2] coords[:, 1:] = coords[:, 1:] * 2              # :234  🆕 桥梁
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
├─[5] decoded[-1].get('vertex')          # :693       ← 取最后一级顶点
├─[6] predict_edges_batched(connection_head, ...)      ← LATO ConnectionHead
│       # 函数定义 :241, 调用 :726, cat 操作 :291
└─[7] edges_to_mesh() → trimesh → .obj  # 定义 :307, 调用 :741  ← NetworkX 公共邻居法
```

### 7 处改动 × 对应关系

| # | 代码证据 | 改动内容 | 对应目标 | 对应 LATO 模块 |
|---|---------|---------|---------|---------------|
| 1 | `inference_lato.py:510` `pipeline.models["sparse_structure_flow_model"] = ss_flow` | SS Flow 替换为你训练的权重 | 刹车卡钳形状先验 | `EnhancedSSFlowModel`（`lato_integration/flow/ss_flow.py`，通过 `run_train.py:38` 映射） |
| 2 | `trellis_text_to_3d.py:105` `decoder = self.models['sparse_structure_decoder']` | SS Decoder 保持 TRELLIS 预训练冻结 | 复用官方 occupancy 解码 | 无改动（TRELLIS `SparseStructureDecoder`） |
| 3 | `trellis_text_to_3d.py:234` `coords[:, 1:] = coords[:, 1:] * 2` | 坐标 res 64→128 | 桥接 TRELLIS 和 LATO 分辨率差异 | 无对应模块（纯坐标变换） |
| 4 | `inference_lato.py:542` `pipeline.models["slat_flow_model"] = slat_flow` | SLat Flow 替换为你训练的权重（res=128, dim=16） | 刹车卡钳潜空间分布 + 匹配 LATO VAE 输入 | `LATOSLatFlowModel`（`trellis/models/lato_slat_flow.py`，仅改 3 个默认值） |
| 5 | `trellis_text_to_3d.py:141-145` `self.models['lato_vae'].decode(lato_slat, ...)` | TRELLIS decoder 替换为 LATO VoxelVAE | 高质量几何解码（cross-attn + pruning） | LATO `VoxelVAE.decode()` |
| 6 | `inference_lato.py:291-292` `connection_head(torch.cat([batch_u, batch_v]))` / 调用 `:726-733` | 顶点对 → 边概率双向打分 | 显式拓扑预测，替代 FlexiCubes | LATO `ConnectionHead`（`vertex_encoder.py`） |
| 7 | `inference_lato.py:307-338` `edges_to_mesh(vertex_coords, edges)` / 调用 `:741` | 边 → 三角面 NetworkX 公共邻居法 | 显式 mesh 构建 | 无对应模块（纯几何算法） |

### 目标符合性判定

```
目标: "将 Sparse VAE Encoder/Decoder 替换为 LATO 的 VoxelVAE，
       SS Flow 和 SLat Flow 均在刹车卡钳数据集上从零训练，
       最后用 CD/HD/NC 评估"

  ✅ Decoder 替换为 LATO VoxelVAE    — trellis_text_to_3d.py:141
  ✅ SS Flow 刹车卡钳从零训练         — inference_lato.py:496-510
  ✅ SLat Flow 刹车卡钳从零训练       — 同上 SLat 部分
  ✅ coords×2 桥接分辨率差异          — trellis_text_to_3d.py:234
  ✅ ConnectionHead 显式拓扑          — inference_lato.py:477
  ✅ CD/HD/NC 评估                   — evaluate_3d_metrics.py

  ⚠️ SS Flow 架构增强 (Swin/IO)     — 代码预留，未激活 (pass/num=0)
  ⚠️ SLat Flow 架构增强 (Swin/PE)   — 代码预留，未使用 EnhancedSLatFlowModel
```

**结论：核心目标全部达成。** 架构增强是预留扩展点，不影响目标功能。当前管线 = TRELLIS 的 SS 管线 + 刹车卡钳训练的 SS/SLat Flow + LATO VoxelVAE 解码器。

---

## 效果预期分析：LATO 增强版 vs 原版 TRELLIS

### 理论上不应该更好

核心矛盾很简单：

```
原版 TRELLIS:  ~10M 条训练数据  →  Flow 模型学到丰富的 3D 形状先验
你的 LATO 管线: 234 条训练数据  →  Flow 模型只见过刹车卡钳
```

### 瓶颈不在 Decoder，在 Flow

```
输入 Text ──→ [SS Flow] ──→ [SLat Flow] ──→ [Decoder] ──→ Mesh
                  ↑                ↑              ↑
              🆕你训练的       🆕你训练的      LATO 冻结（真正增强的地方）
              234条数据       234条数据
```

**Decoder 再好，也只能解码 Flow 给它的 latent。** 如果 Flow 产出了低质量 latent，LATO VoxelVAE 救不回来。

### 具体弱点逐条分析

| 弱点 | 原版 TRELLIS | 你的 LATO 管线 | 影响 |
|------|-------------|---------------|------|
| **训练数据量** | ~10M | **234**（差 4 个数量级） | 🔴 致命 |
| **SLat Flow 容量** | 1024ch × 24 blocks | **384ch × 12 blocks**（~1/3 参数量） | 🟡 中等 |
| **训推不一致** | 训练时 SLat 也见过 SS 预测值 | 训练时 SLat 只看 GT latent，推理时吃 SS 预测值→**没见过的分布** | 🔴 累积误差 |
| **Decoder 质量** | FlexiCubes 隐式等值面（有机形状偏向） | LATO cross-attn + pruning + 显式边预测 | 🟢 增强 |

### 唯一可能赢的场景

刹车卡钳是**机械零件**，形状空间小、规律性强，如果 234 条覆盖了主要变体，且 LATO 的显式拓扑对**锐利边缘+平面**的还原优于 FlexiCubes（FlexiCubes 天生偏向光滑有机形状），那 CD/HD 可能更好。

但这是小概率事件。

### 如果效果不好，在哪里？

按可能性从高到低：

1. **CD/HD 数值比原版差**（最可能）— Flow 模型欠拟合，产出的 latent 不能准确表达刹车卡钳形状，234 条数据不足以学到完整的条件分布 P(shape | text)
2. **生成形状"像但细节不对"** — 大体轮廓对（刹车卡钳形状简单），但活塞孔、固定耳等细节位置偏移，因为 SLat Flow 在 128³ 稀疏空间里只见过 234 种 voxel 排列
3. **多样性差** — 不同 prompt 生成的形状趋同，小数据 + CFG 引导导致模型只会"安全模式"
4. **特定样本崩坏** — 测试集中某些参数组合在训练集中没出现过 → 外推失败

### 结论

**大概率 CD/HD 比原版 TRELLIS 差，根本原因是 234 条数据喂不饱两个 Flow 模型。** 但这不代表工作没意义——证明了管线可行，下一步扩充数据就能真正发挥 LATO decoder 的优势。

### 评估函数正确性验证

| 函数 | 公式 | 实现 | 判定 |
|------|------|------|:--:|
| `chamfer_distance` | CD = 1/\|P\| Σ min\|p-q\|² + 1/\|Q\| Σ min\|q-p\|² | `KDTree.query`(L2) → `**2` → `.mean()` 双向 | ✅ |
| `hausdorff_distance` | d_H = max{max_p min_q\|p-q\|, max_q min_p\|q-p\|} | `KDTree.query`(L2) → `.max()` 双向取 max | ✅ |
| `normal_consistency` | NC = 1/\|P\| Σ \|n_p · n_nearest\| | `KDTree.query(k=1)` → `np.abs(dot)` → `.mean()` | ✅ |
| 归一化 | 统一到 GT bbox 对角线尺度 | `np.linalg.norm(bbox_diag)` 同时缩放 pred 和 gt | ✅ |

> 小瑕疵：`normal_consistency` 签名参数 `k=10` 实际未使用，query 硬编码 `k=1`。不影响正确性（NC 标准做法就是 k=1）。

### 训练/推理角色

```
  CLIP ──────────── 冻结 ─ 只做文本编码
  SS Flow ───────── 🆕训 ─ 唯一需要训练的模型之一
  SS Decoder ────── 冻结 ─ 仅做 occupancy 阈值化，不涉及几何质量
  SLat Flow ─────── 🆕训 ─ 唯一需要训练的模型之二
  LATO VoxelVAE ─── 冻结 ─ 预训练几何解码器，训练和推理都不更新
  ConnectionHead ── 冻结 ─ 预训练边预测器，含在 LATO ckpt 中
```

---

## 效果预期分析：这套管线到底能不能增强？

### 真实增强的部分 ✅

| 模块 | 为什么能增强 | 来源 |
|------|-------------|------|
| **LATO VoxelVAE** | cross-attention + occupancy pruning，ICML 2026 验证的几何重建质量 | LATO 论文 |
| **ConnectionHead** | 显式双向边预测，比 FlexiCubes 隐式等值面更不容易产生拓扑错误（洞、非流形边） | LATO 论文 |
| **coords ×2** | 分辨率 64→128，最终 mesh 顶点密度更高、细节更丰富 | 纯数学 |

### 不确定的部分 ⚠️

| 模块 | 风险 | 说明 |
|------|------|------|
| **SS Flow** | 架构等同原版（Swin/IO 全死代码），训练数据从 ~10M → **234 条** | 模型容量没变，数据少了 4 个数量级 |
| **SLat Flow** | 架构等同原版（`LATOSLatFlowModel` 只改 res/dim 默认值），模型还更小了（384ch vs 1024ch xlarge） | 如果原版容量是合理的那你的就偏小 |

### 核心瓶颈

```
管线 = SS Flow → SLat Flow → LATO VoxelVAE → Mesh
       ───────────────┬──────────────          │
              瓶颈在这里                      真正增强在这里
```

**Decoder 再好，也只能解码 Flow 模型生成的 latent。** 如果 Flow 模型因为数据不足产出了低质量 latent，LATO VoxelVAE 也救不回来。

### 234 条数据够不够？两种可能

| 场景 | 结果 | 条件 |
|------|------|------|
| ✅ **成功** | 你的管线 > 原版 TRELLIS | 刹车卡钳形状空间小、规律性强，234 条覆盖了主要变体 |
| ❌ **失败** | 你的管线 < 原版 TRELLIS | 刹车卡钳形状方差大，234 条不足以学到完整分布 |

### 唯一判断方式：跑步骤 6

```
步骤 6 不是走形式，是回答"有没有增强"的唯一方式。

  情况 A: CD/HD 比原版好  → 234 条够了，管线有效
  情况 B: CD/HD 比原版差  → 数据不够，需要：
                            1. 扩充数据集
                            2. 把 SS Flow 的 Swin/IO 增强真正实现
                            3. 把 SLat Flow 的 EnhancedSLatFlowModel 用上
```

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
│   │   ├── inference_lato.py                      # ✅ 单条推理（v5: 自动发现 ckpt + ss_only）
│   │   ├── evaluate_3d_metrics.py                 # 🆕 批量评估（CD/HD/NC）
│   │   ├── encode_lato_latent_v2.py               # ✅ LATO latent 提取
│   │   ├── flow/ss_flow.py                        # ✅ EnhancedSSFlowModel
│   │   ├── flow/slat_flow.py                      # ✅ EnhancedSLatFlowModel（增强预留）
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
| 4a | 训练 SS Flow | `run_train.py` | ~2.8 天（单卡） | ✅ 680k 步 |
| 4b | 训练 SLat Flow | `run_train.py` | ~4.5 天（单卡） | ✅ 880k 步 |
| 5 | 单条推理验证 | `inference_lato.py` | ~30s/条 | ✅ |
| 6 | 批量评估 | `evaluate_3d_metrics.py` | ~30s/条 | ✅ |

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
    --output_dir /data/huanghaoyang/3D/TRELLIS/outputs/lato_ss_flow_fast \
    --num_gpus 1 --auto_retry 0
```

> **实际训练目录:** `outputs/lato_ss_flow_fast`（非 `lato_ss_flow`），当前最新 ckpt: `denoiser_step0680000.pt`

| 项目 | 值 |
|------|-----|
| 模型 | `EnhancedSSFlowModel`（→ `SparseStructureFlowModel`），145M 参数 |
| 配置 | 512ch × 24 blocks × 16 heads, cond=768 |
| 数据 | `TextConditionedSparseStructureLatent`, 234 条 |
| 输入/输出 | CLIP text → dense 16³×8 |
| batch_size | 4 per GPU（配置值；OOM 时可降至 2 或 1） |
| 步数 | 1,000,000 |
| 速度 | ~15,000 steps/h（单卡 4090），ETA ~2.8 天 |

**OOM 降级方案：** `model_channels: 512→384` → `batch_size: 4→2→1`

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
| 实际最新 | `denoiser_step0880000.pt`（207MB） |

**续训：** 加 `--ckpt {step}` 从断点继续

---

## 步骤5：单条推理验证（v6）

### 环境准备（重要）

```bash
# LATO 必须在 TRELLIS 前面，否则 utils 模块冲突
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
# Sparse attention 只支持 xformers/flash_attn，xformers 兼容性更好
export SPARSE_ATTN_BACKEND=xformers
```

### 5a. 完整推理（SS + SLat + LATO → mesh）

```bash
cd /data/huanghaoyang/3D/TRELLIS

# 方式一：指定 checkpoint（推荐，明确可控）
SS_CKPT=$(ls outputs/lato_ss_flow_fast/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow/ckpts/denoiser_step*.pt | sort -V | tail -1)

python lato_integration/inference_lato.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --prompt "A brake caliper fixing interaxis 116.32 inner pad 222.29 outer pad 222.29 internal radius 130.55 pistons_num 6" \
    --seed 42 \
    --output output_caliper.obj

# 方式二：用 --ss_dir / --slat_dir 自动发现最新
python lato_integration/inference_lato.py \
    --ss_dir outputs/lato_ss_flow_fast \
    --slat_dir outputs/lato_slat_flow \
    --prompt "A brake caliper fixing interaxis 116.32 inner pad 222.29 pistons_num 6" \
    --seed 42 --output output_caliper.obj
```

> **注意：** `--lato_ckpt` 和 `--lato_config` 有自动默认值，路径匹配时可省略。
> `--slat_stats` 可不传（默认 identity），实际归一化参数在训练 config 中已固定，推理时由 pipeline 内置处理。

### 5b. SS-only 模式（快速验证 SS Flow，不跑 SLat/LATO）

训练中即可使用，不需要 SLat Flow 训练完成：

```bash
# 自动找最新 ckpt
python lato_integration/inference_lato.py \
    --mode ss_only \
    --ss_dir outputs/lato_ss_flow \
    --prompt "A brake caliper"

# 指定中间步数对比收敛效果
python lato_integration/inference_lato.py \
    --mode ss_only \
    --ss_ckpt outputs/lato_ss_flow/ckpts/denoiser_step0100000.pt \
    --prompt "A brake caliper"
```

SS-only 输出：active voxels 数量 + bbox 范围（判断 SS Flow 生成的稀疏结构是否合理）。

### 5c. 常用调参

```bash
# 边太少（mesh 有洞）→ 降低边阈值
--edge_threshold 0.3

# 边太多（噪声面过多）→ 提高边阈值
--edge_threshold 0.5

# VoxelVAE decode 保留更多/更少顶点
--lato_threshold 0.1   # 更多顶点
--lato_threshold 0.3   # 更少顶点
```

**全部参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `full` | `full`=完整管线, `ss_only`=仅 SS Flow |
| `--ss_dir` | 无 | SS Flow 训练输出目录，自动发现 `ckpts/` 下最新 |
| `--ss_ckpt` | 无 | 指定 SS Flow checkpoint 路径（优先于 `--ss_dir`） |
| `--slat_dir` | 无 | SLat Flow 训练输出目录，自动发现最新 |
| `--slat_ckpt` | 无 | 指定 SLat Flow checkpoint 路径（优先于 `--slat_dir`） |
| `--ss_config` | `configs/generation/lato_ss_flow.json` | SS Flow 训练 config（读取模型参数） |
| `--slat_config` | `configs/generation/lato_slat_flow.json` | SLat Flow 训练 config（读取模型参数） |
| `--lato_ckpt` | `$LATO_ROOT/checkpoints/128to512/vae/vae_128to512.pt` | LATO VAE checkpoint |
| `--lato_config` | `$LATO_ROOT/configs/infer_vae_512.yaml` | LATO VAE yaml |
| `--trellis_pretrained` | `microsoft/TRELLIS-text-base` | TRELLIS 预训练管线 |
| `--ss_stats` | identity | SS normalization stats JSON |
| `--slat_stats` | identity | SLat normalization stats JSON（16-dim） |
| `--prompt` | 刹车卡钳示例 | 文本描述 |
| `--output` | `output_mesh.obj` | 输出 mesh 路径 |
| `--seed` | 42 | 随机种子 |
| `--ss_steps` | 20 | SS Flow 采样步数 |
| `--slat_steps` | 20 | SLat Flow 采样步数 |
| `--cfg_strength` | 5.0 | CFG 强度 |
| `--lato_threshold` | 0.2 | VoxelVAE decode inference_threshold |
| `--edge_threshold` | 0.45 | ConnectionHead 边概率阈值 |
| `--k_neighbors` | 32 | KDTree 最近邻数（影响候选边数量） |
| `--no_fp16` | false | 禁用 FP16（调试用） |

---

## 步骤6：批量评估（v6 — 已验证通过）

对测试集（21 条）逐条推理，计算 Chamfer Distance / Hausdorff Distance / Normal Consistency。

### 环境准备

```bash
# 关键：LATO 必须在 TRELLIS 前面（否则 utils 模块冲突）
export PYTHONPATH="/data/huanghaoyang/3D/LATO:/data/huanghaoyang/3D/TRELLIS:$PYTHONPATH"
# Sparse attention 只支持 xformers 或 flash_attn，xformers 兼容性更好
export SPARSE_ATTN_BACKEND=xformers
```

### 推理前修复（已确认需要，v6）

`evaluate_3d_metrics.py` 和 `inference_lato.py` 有 6 处需要修复才能正常推理，详见 [v6 推理 bug 修复清单](#v6-推理-bug-修复清单)。

### 运行命令

```bash
cd /data/huanghaoyang/3D/TRELLIS

SS_CKPT=$(ls outputs/lato_ss_flow_fast/ckpts/denoiser_step*.pt | sort -V | tail -1)
SLAT_CKPT=$(ls outputs/lato_slat_flow/ckpts/denoiser_step*.pt | sort -V | tail -1)

# 先跑 1 条验证
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --slat_stats /tmp/slat_norm.json \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database \
    --output_dir outputs/eval_results \
    --limit 1

# 确认通过后去掉 --limit 跑全部 21 条
python lato_integration/evaluate_3d_metrics.py \
    --ss_ckpt "$SS_CKPT" \
    --slat_ckpt "$SLAT_CKPT" \
    --lato_ckpt /data/huanghaoyang/3D/LATO/checkpoints/128to512/vae/vae_128to512.pt \
    --lato_config /data/huanghaoyang/3D/LATO/configs/infer_vae_512.yaml \
    --slat_stats /tmp/slat_norm.json \
    --test_metadata /data/huanghaoyang/3D/database_lato/test/metadata.csv \
    --gt_meshes /data/huanghaoyang/3D/database \
    --output_dir outputs/eval_results
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--gt_meshes` | `/data/huanghaoyang/3D/database` | GT STL 按 `{ID}.stl` 命名，非 `{sha256}.stl` |
| `--slat_stats` | `/tmp/slat_norm.json` | 从训练 config 提取的归一化参数，**不能直接用 stats.json**（值与训练不同） |
| `--ss_ckpt` | `lato_ss_flow_fast` 下最新 | 实际训练目录名 |
| `--slat_ckpt` | `lato_slat_flow` 下最新 | 当前最新 880k 步 |

### 归一化参数提取

训练 config 中的值 ≠ `stats.json`（后者是后来重新统计的），推理必须用训练时的值：

```bash
python3 -c "import json; c=json.load(open('outputs/lato_slat_flow/config.json')); n=c['dataset']['args']['normalization']; json.dump(n,open('/tmp/slat_norm.json','w')); print('Done')"
```

### 指标说明

| 指标 | 方向 | 含义 |
|------|:--:|------|
| **Chamfer Distance (CD)** | ↓ | 双向最近邻 L2 距离均值 |
| **Hausdorff Distance (HD)** | ↓ | 最大局部偏差 |
| **Normal Consistency (NC)** | ↑ [0,1] | 法线方向一致性 |

### 输出

| 文件 | 内容 |
|------|------|
| `summary.json` | 均值/标准差/中位数/最大/最小 |
| `per_sample_results.json` | 逐条详细结果 |
| `failures.json` | 失败样本及原因 |

---

## v6 推理 bug 修复清单

训练完成后到跑通测试集推理，共发现 6 处需要在 evaluate/inference 脚本中修复的问题：

| # | 文件 | 行 | 问题 | 修复 |
|---|------|-----|------|------|
| 1 | `evaluate_3d_metrics.py` | 230 | `pipeline = pipeline.to(device)` 返回 None | 改为 `pipeline.to(device)` |
| 2 | `evaluate_3d_metrics.py` | 221 | `LATOConnectionHead(channels=512*2)` 与 LATO config 不符 | 改为 `channels=512` |
| 3 | `evaluate_3d_metrics.py` | 335 | GT 用 sha256 列名找文件，但 STL 按 ID 命名 | 改为 `file_identifier` |
| 4 | `evaluate_3d_metrics.py` | 61 | `from utils import load_pretrained_woself` 解析到 `lato_integration/utils.py` | importlib 直接加载 LATO utils.py |
| 5 | `inference_lato.py` | 88 | 同上 | importlib 直接加载 |
| 6 | `inference_lato.py` | 582 | `LATOConnectionHead(channels=512*2)` 同上 | 改为 `channels=512` |

**快速修复脚本（服务器上执行）：**

```bash
cd /data/huanghaoyang/3D/TRELLIS

# 1. pipeline.to() 不赋值
sed -i 's/pipeline = pipeline.to(device)/pipeline.to(device)/' lato_integration/evaluate_3d_metrics.py

# 2. ConnectionHead channels
sed -i 's/LATOConnectionHead(channels=512\*2,/LATOConnectionHead(channels=512,/' lato_integration/evaluate_3d_metrics.py
sed -i 's/LATOConnectionHead(channels=512 \* 2,/LATOConnectionHead(channels=512,/' lato_integration/inference_lato.py

# 3. GT 列名
sed -i 's/sha256_col = "sha256"/sha256_col = "file_identifier"/' lato_integration/evaluate_3d_metrics.py

# 4-5. utils 导入（Python 脚本处理）
python3 << 'EOF'
import re
for path in ["lato_integration/evaluate_3d_metrics.py", "lato_integration/inference_lato.py"]:
    with open(path) as f: content = f.read()
    old = 'from utils import load_pretrained_woself'
    new = '''import importlib.util
_spec=importlib.util.spec_from_file_location("lato_utils","/data/huanghaoyang/3D/LATO/utils.py")
_lato_utils=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lato_utils)
load_pretrained_woself=_lato_utils.load_pretrained_woself'''
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w') as f: f.write(content)
        print(f"Fixed: {path}")
    else:
        print(f"Already fixed: {path}")
EOF
```

---

## 原版 TRELLIS vs 你的 LATO 管线：逐阶段对比

### 总览

```
原版 TRELLIS (microsoft/TRELLIS-text-base):
  Text → CLIP → SS Flow → SS Decoder → SLat Flow → SLat Decoder(×3) → GS/RF/Mesh

你的 LATO 管线:
  Text → CLIP → SS Flow(重训) → SS Decoder(冻结) → ×2坐标 → SLat Flow(重训) → LATO VoxelVAE → ConnectionHead → Mesh
```

共 **5 个差异阶段**。

### 阶段 1：Text → CLIP 编码

| | 原版 | 你的 |
|------|------|------|
| 模型 | `openai/clip-vit-large-patch14` | **完全相同** |
| 输出 | `[B, 77, 768]` | `[B, 77, 768]` |
| 训练状态 | 冻结 | 冻结 |

**无差异。**

### 阶段 2：SS Flow — 稀疏结构生成

| | 原版 TRELLIS | 你的 LATO |
|------|------|------|
| 模型类 | `SparseStructureFlowModel` | `EnhancedSSFlowModel`（继承前者，预留 Swin/IO 扩展；通过 `run_train.py:38` 映射） |
| 架构 | Dense 3D DiT, full attention on 4096 tokens | **相同**（训练配置未启用 Swin/IO blocks；`num_io_res_blocks=0`） |
| 输入 | noise `[B,8,16,16,16]` | 同 |
| 输出 | velocity → denoised latent `[B,8,16,16,16]` | 同 |
| 参数量 | `model_channels=512`, 24 blocks, 16 heads | **512ch, 24 blocks, 16 heads** |
| batch_size | — | 4 per GPU |
| 训练数据 | ObjaverseXL (~10M 3D 模型) | **刹车卡钳 234 条** |
| 权重来源 | 官方预训练 | **从零训练**，目标 1M 步 |
| 训练器 | 原版 `FlowMatchingCFGTrainer` | **原版 TRELLIS trainer**（`TextConditionedFlowMatchingCFGTrainer` 未在 `run_train.py` TRAINER_REPLACEMENTS 中映射，fallback 到原版） |

**核心逻辑相同，差异在于训练数据领域专精。**

### 阶段 3：SS Decoder — occupancy 解码

| | 原版 | 你的 |
|------|------|------|
| 模型 | `SparseStructureDecoder`（3D CNN） | **完全相同，冻结** |
| 输入 | SS latent `[B,8,16,16,16]` | 同 |
| 输出 | occupancy logits → coords at res 64 | 同 |

**无差异。** SS Decoder 只做 `logits > 0` 阈值化，不涉及几何质量，无需重训。

### 阶段 4：坐标上采样（🆕 新增）

```
coords[:, 1:] = coords[:, 1:] * 2   # res 64 → 128
```

| | 原版 | 你的 |
|------|------|------|
| SLat 坐标分辨率 | **res 64**（不变） | **res 128**（×2） |
| 原因 | SLat Flow 原生 res=64 | LATO VoxelVAE 原生 res=128 |

### 阶段 5：SLat Flow — 结构化潜空间生成（差异最大）

| | 原版 TRELLIS (xlarge) | 你的 LATO |
|------|------|------|
| 模型类 | `SLatFlowModel` | `LATOSLatFlowModel`（仅改默认值：res=128, ch=16） |
| 分辨率 | **64** | **128** |
| Latent 维度 | **8** | **16** |
| 模型大小 | **1024ch, 24 blocks** | **384ch, 12 blocks** |
| IO blocks | `[256, 512]`（2 层不同通道） | `[128]`（1 层×2 次） |
| Attention | Full attention | Full attention |
| 数据类型 | Sparse Tensor（仅 active voxels） | 同（上限 16384 voxels） |
| Normalization | 通用 3D 数据集统计量 | **刹车卡钳 16-dim 统计量** |
| 训练数据 | ObjaverseXL (~10M) | **刹车卡钳 234 条** |
| 权重来源 | 官方预训练 | **从零训练**，目标 1M 步 |

**模型更小但分辨率更高。** 因为 res=128 下 active voxels 更多，但 384ch/12blocks 降低了每个 voxel 的计算量。latent dim 8→16 是为了匹配 LATO VoxelVAE 的输入。

### 阶段 6：解码器 → Mesh（完全替换）

| 对比维度 | 原版 TRELLIS | 你的 LATO |
|------|------|------|
| 解码器 | 3 个独立 Sparse Transformer decoder（GS/RF/Mesh） | **1 个 LATO VoxelVAE** + ConnectionHead |
| 架构 | Sparse Transformer × N blocks | 多级 subdivision decoder + cross-attention |
| Cross-attention | 无 | ✅ decoder ↔ latent |
| Occupancy pruning | 无 | ✅ 每级过滤非表面体素 |
| Mesh 方式 | FlexiCubes（隐式曲面 → 显式 mesh） | **显式顶点 → ConnectionHead 边预测 → 三角面片化** |
| 输出类型 | GS + RF + Mesh（3 种） | **仅 Mesh** |
| 训练状态 | 预训练冻结 | 冻结（LATO 预训练权重） |

**这是最大的差异。** LATO VoxelVAE 用显式拓扑预测取代了 TRELLIS 的隐式解码，几何质量更好，但也意味着你只能出 Mesh，不支持 Gaussian Splats 和 Radiance Field。

### 训练/推理状态总表

```
                    训练时                    推理时
─────────────────────────────────────────────────────
CLIP               冻结                      冻结
SS Flow            从零训练 🔄               加载你的 ckpt
SS Decoder         冻结                      冻结（TRELLIS 预训练）
SLat Flow          从零训练 🔄               加载你的 ckpt
LATO VoxelVAE      —（不参与训练）            冻结（LATO 预训练）
ConnectionHead     —（不参与训练）            冻结（LATO 预训练）
```

> **注意：** LATO VoxelVAE 和 ConnectionHead 只在推理时加载，训练阶段完全不涉及。你只需要训练 SS Flow 和 SLat Flow 两个模型。

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
| `lato_integration/flow/slat_flow.py` | EnhancedSLatFlowModel（增强预留） | ✅ |
| `lato_integration/flow/trainers/ss_flow_trainer.py` | SS Flow 训练器 | ✅ |
| `lato_integration/flow/trainers/slat_flow_trainer.py` | SLat Flow 训练器 | ✅ |
| `lato_integration/encode_lato_latent_v2.py` | LATO latent 提取 | ✅ |
| `lato_integration/inference_lato.py` | 单条推理（v5：自动发现 ckpt + ss_only 模式） | ✅ v5 |
| `lato_integration/evaluate_3d_metrics.py` | 🆕 批量评估 | ✅ |
| `dataset_toolkits/stat_latent.py` | SLat normalization | ✅ |

---

## v6 当前状态

| 步骤 | 状态 | 说明 |
|------|------|------|
| 1. 代码修改 | ✅ | 5 个文件 |
| 2a. LATO latent | ✅ | 234 npz |
| 2b. SS latent | ✅ | 255 npz |
| 3a. SLat stats | ✅ | stats.json 已填入配置 |
| 3b. SS stats | ⏭️ | identity |
| 4a. 训练 SS Flow | ✅ | `lato_ss_flow_fast`，680k 步（512ch, 24 blocks, 16 heads） |
| 4b. 训练 SLat Flow | 🔄 | `lato_slat_flow`，880k 步（384ch, 12 blocks, 8 heads），续训中目标 1M |
| 5. 单条推理 | ✅ | 已验证通过，速度 ~6s/条（含 SS + SLat sampling） |
| 6. 批量评估 | ✅ | 21 条端到端跑通，CD/HD/NC 指标正常输出 |

**Checkpoint 位置：**

| 模型 | 目录 | 最新 ckpt | 大小 |
|------|------|-----------|------|
| SS Flow | `outputs/lato_ss_flow_fast/ckpts/` | `denoiser_step0680000.pt` | ~145M |
| SLat Flow | `outputs/lato_slat_flow/ckpts/` | `denoiser_step0880000.pt` | 207M |
| LATO VAE | `/data/.../LATO/checkpoints/128to512/vae/` | `vae_128to512.pt` | 1.3G |

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

### Q: 推理时需要训练 LATO VAE 吗？

**不需要。** LATO VoxelVAE + ConnectionHead 是预训练好的冻结权重（`vae_128to512.pt`），推理时直接加载。你的管线只需要训练 **SS Flow** 和 **SLat Flow** 两个模型。LATO VAE 在整个流程中始终是冻结的，既不参与训练也不参与梯度回传。
