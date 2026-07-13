# TRELLIS + LATO Decoder 修改后运行步骤

> **改了什么：** SS Flow + SS Decoder 保留原版，SLat Flow 换 16-dim/128res，Decoder 换 LATO VoxelVAE。
> **只需训练 1 个新 SLat Flow（~1M 步），其余用预训练权重。**

---

## 前置条件

```
D:\code\3D\
├── TRELLIS\          # TRELLIS 代码（已改 lato_slat_flow.py + pipeline）
├── LATO\             # LATO 代码 + 预训练权重
│   └── ckpts\        # LATO checkpoint.pt（需要你自己有）
└── database\         # 训练数据
    ├── metadata.csv
    ├── ss_latent_*.npy
    ├── voxelized_*.npy
    └── ...
```

**你需要有：**
- TRELLIS 预训练权重（SS Flow + SS Decoder），或自己训好的
- LATO 预训练权重（VoxelVAE + VoxelFeatureEncoder + ConnectionHead）
- 训练数据集（3D 模型 + caption）

---

## 步骤1：修改 TRELLIS 代码（3 个文件）

### 1a. 新建 `trellis/models/lato_slat_flow.py`

```python
from .structured_latent_flow import SLatFlowModel

class LATOSLatFlowModel(SLatFlowModel):
    def __init__(self,
        resolution=128,
        in_channels=16,
        out_channels=16,
        **kwargs
    ):
        super().__init__(resolution=resolution, in_channels=in_channels,
                         out_channels=out_channels, **kwargs)
```

### 1b. 改 `trellis/pipelines/trellis_text_to_3d.py`

**`run()` 中，`sample_sparse_structure` 之后加一行：**

```python
coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
coords = coords * 2   # ← 新增：res 64 → 128
slat = self.sample_slat(cond, coords, slat_sampler_params)
```

**`decode_slat()` 替换为：**

```python
def decode_slat(self, slat, formats=['mesh']):
    ret = {}
    if 'mesh' in formats:
        ret['lato_decoded'] = self.models['lato_vae'].decode(
            slat, training=False, inference_threshold=0.2
        )
    return ret
```

**`run()` 最后一行改为：**

```python
return self.decode_slat(slat, formats=['mesh'])
```

> 原版的 `slat_decoder_mesh/gs/rf` 不再需要，从 models 字典中移除。

---

## 步骤2：用 LATO 提取 latent，生成训练数据

```bash
python encode_lato_latent.py \
    --lato_ckpt D:/code/3D/LATO/ckpts/your_checkpoint.pt \
    --lato_config D:/code/3D/LATO/configs/infer_vae_512.yaml \
    --data_dir D:/code/3D/database \
    --output_dir D:/code/3D/database/lato_latents
```

**脚本核心逻辑：**

```python
# 对每个模型：
for mesh in dataset:
    # ① 体素化 → active voxels @ res 128
    active_coords = get_active_voxels(mesh, resolution=128)

    # ② 点云采样 → LATO voxel_encoder
    point_cloud = sample_point_cloud(mesh, n=819200)
    active_feats = voxel_encoder(point_cloud, active_coords, res=128)

    # ③ LATO VAE encode → 16-dim latent
    sparse_input = SparseTensor(feats=active_feats, coords=active_coords)
    lato_latent, _ = vae.encode(sparse_input)

    # ④ 保存
    torch.save({
        'caption': caption,
        'coords': lato_latent.coords.cpu(),       # [N, 4]
        'latent_feats': lato_latent.feats.cpu(),   # [N, 16]
    }, f'{output_dir}/{model_id}.pt')
```

**产出：** `lato_latents/` 目录下每个模型一个 `.pt` 文件 + `metadata.csv` 索引。

---

## 步骤3：训练新 SLat Flow

### 3a. 新建 config JSON

```json
{
    "models": {
        "denoiser": {
            "name": "LATOSLatFlowModel",
            "args": {
                "resolution": 128,
                "in_channels": 16,
                "out_channels": 16,
                "model_channels": 768,
                "cond_channels": 768,
                "num_blocks": 12,
                "num_heads": 12,
                "mlp_ratio": 4,
                "patch_size": 2,
                "num_io_res_blocks": 2,
                "io_block_channels": [128],
                "pe_mode": "ape",
                "qk_rms_norm": true,
                "use_fp16": true
            }
        }
    },
    "dataset": {
        "name": "TextConditionedSLat",
        "args": {
            "latent_model": "lato_vae_16dim_128",
            "min_aesthetic_score": 0,
            "max_num_voxels": 65536,
            "normalization": {
                "mean": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "std":  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            }
        }
    },
    "trainer": {
        "name": "TextConditionedSparseFlowMatchingCFGTrainer",
        "args": {
            "max_steps": 1000000,
            "batch_size_per_gpu": 4,
            "batch_split": 2,
            "optimizer": {"name": "AdamW", "args": {"lr": 0.0001, "weight_decay": 0.0}},
            "ema_rate": [0.9999],
            "fp16_mode": "inflat_all",
            "fp16_scale_growth": 0.001,
            "grad_clip": {"name": "AdaptiveGradClipper", "args": {"max_norm": 1.0, "clip_percentile": 95}},
            "i_log": 500,
            "i_sample": 10000,
            "i_save": 10000,
            "p_uncond": 0.1,
            "t_schedule": {"name": "logitNormal", "args": {"mean": 1.0, "std": 1.0}},
            "sigma_min": 1e-5,
            "text_cond_model": "openai/clip-vit-large-patch14"
        }
    }
}
```

> `normalization` 的 mean/std 先填占位值。步骤2 生成数据后，用 `dataset_toolkits/stat_latent.py` 统计出真实值再填回来。

### 3b. 运行训练

```bash
export PYTHONPATH="D:/code/3D/TRELLIS:$PYTHONPATH"

python lato_integration/run_train.py \
    --config configs/generation/lato_slat_flow.json \
    --data_dir D:/code/3D/database/lato_latents \
    --output_dir D:/code/3D/TRELLIS/outputs/lato_slat_flow \
    --num_gpus 1 \
    --auto_retry 0
```

**这是唯一需要训练的步骤。** 跑 1M 步，约几小时到一天（取决于 GPU）。

---

## 步骤4：推理

```python
import torch, sys
sys.path.insert(0, 'D:/code/3D/TRELLIS')
sys.path.insert(0, 'D:/code/3D/LATO')

from trellis.models.lato_slat_flow import LATOSLatFlowModel
from trellis.pipelines import TrellisTextTo3DPipeline
from lato.models.lato_vae.lato_vae import VoxelVAE
from vertex_encoder import ConnectionHead, VoxelFeatureEncoder_active_pointnet
from lato.utils import load_pretrained_woself
import yaml, numpy as np

device = torch.device('cuda')

# ── 1. 加载 TRELLIS SS 管线（预训练） ──
pipeline = TrellisTextTo3DPipeline.from_pretrained('microsoft/TRELLIS-text-base')

# ── 2. 加载你训练的新 SLat Flow ──
new_slat_flow = LATOSLatFlowModel(
    resolution=128, in_channels=16, out_channels=16,
    model_channels=768, cond_channels=768,
    num_blocks=12, num_heads=12, patch_size=2,
).to(device)
ckpt = torch.load('outputs/lato_slat_flow/ckpts/denoiser_step1000000.pt',
                   map_location=device)
new_slat_flow.load_state_dict(ckpt)
new_slat_flow.eval()

# ── 3. 加载 LATO VAE（预训练） ──
config = yaml.safe_load(open('D:/code/3D/LATO/configs/infer_vae_512.yaml'))
lato_vae = VoxelVAE(
    in_channels=1024, latent_dim=16,
    encoder_blocks=config['model']['encoder_blocks'],
    decoder_blocks_vtx=config['model']['decoder_blocks_vtx'],
    attn_mode='swin', window_size=8, pe_mode='ape',
    using_subdivide=True,
).to(device)
connection_head = ConnectionHead(channels=512*2, out_channels=1).to(device)
load_pretrained_woself('D:/code/3D/LATO/ckpts/your_checkpoint.pt',
                       vae=lato_vae, connection_head=connection_head)
lato_vae.eval()
connection_head.eval()

# ── 4. 替换 pipeline 的模型 ──
pipeline.models['slat_flow_model'] = new_slat_flow
pipeline.models['lato_vae'] = lato_vae
# 删除不需要的原版 decoder
del pipeline.models['slat_decoder_mesh']
del pipeline.models['slat_decoder_gs']
del pipeline.models['slat_decoder_rf']

# ── 5. 推理 ──
pipeline.cuda()
outputs = pipeline.run(
    "a brake caliper with 4 pistons",
    seed=42,
    sparse_structure_sampler_params={"steps": 20, "cfg_strength": 5.0},
    slat_sampler_params={"steps": 20, "cfg_strength": 5.0},
)

# ── 6. 后处理：ConnectionHead 预测边 → 三角面片化 → Mesh ──
decoded = outputs['lato_decoded']
vertex_result = decoded[-1]['vertex']
vertex_coords = vertex_result['coords'].float() / 512.0 - 0.5
vertex_feats = vertex_result['feats']

# ConnectionHead 预测边
edges = predict_edges(connection_head, vertex_feats, vertex_coords,
                      threshold=0.45, device=device)

# 三角面片化
import networkx as nx
graph = nx.Graph()
graph.add_nodes_from(range(len(vertex_coords)))
graph.add_edges_from(edges)
faces = []
for u,v in edges:
    for w in set(graph.neighbors(u)) & set(graph.neighbors(v)):
        if w > v:
            faces.append([u,v,w])
faces = np.array(faces)

# 导出
import trimesh
mesh = trimesh.Trimesh(vertices=vertex_coords.cpu().numpy(), faces=faces)
mesh.export('output_mesh.obj')
print("完成！")
```

---

## 步骤总览

```
步骤1: 改 3 个文件                    → 10 分钟
步骤2: LATO encode 提取 latent       → 写脚本 + 跑数据（取决于数据集大小）
步骤3: 训练新 SLat Flow              → ~1M 步（唯一需要训练的）
步骤4: 推理                          → 加载 4 个模型权重，跑推理
```

| | 原版 TRELLIS 从头训 | 你的方案 |
|---|---|---|
| 需要训练的模型 | 5 个 × 1M 步 | **1 个 × 1M 步** |
| 使用预训练权重的 | 0 | SS Flow + SS Dec + LATO VAE |
| 最终输出 | Mesh + 3DGS + RF | Mesh only |
