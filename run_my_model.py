#export CUDA_VISIBLE_DEVICES=3
import os
os.environ['SPCONV_ALGO'] = 'native'

import torch
import imageio
from datetime import datetime
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# 配置
text_prompt = "A brake caliper fixing interaxis 129.13 inner pad 168.52 outer pad 168.52 internal radius 107.99 pistons_num 4 inlet diameter 32.00 central diameter 34.00 outlet diameter 36.00 effective radius 154.84 disc thickness 30.96 external radius 223.55 internal radius 190.58 radial cut 107.00 disc distance 39.32 tangential dimension 392.55 axial dimension 183.81 radial dimension 180.05 volume 0.002483"

# output_dir = f"/data/huanghaoyang/3D/TRELLIS/outputs/inference_slat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# output_dir = f"/data/huanghaoyang/3D/TRELLIS/outputs/inference{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir = f"/data/huanghaoyang/3D/TRELLIS/outputs/inference_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 结果保存到: {output_dir}")

# 加载模型
print("加载官方 Base 模型...")
pipeline = TrellisTextTo3DPipeline.from_pretrained('microsoft/TRELLIS-text-base')

# 1. 加载你的 SS Flow 权重（形状生成）
print("加载你的 SS Flow 权重...")
# ss_flow_path = '/data/huanghaoyang/3D/TRELLIS/outputs/ss_flow_training/ckpts/denoiser_step1000000.pt'  # 改为你最新保存的步数
ss_flow_path = '/data/huanghaoyang/3D/TRELLIS/outputs/ss_flow_training_new/ckpts/denoiser_step1000000.pt' 
ss_flow_ckpt = torch.load(ss_flow_path, map_location='cpu')
pipeline.models['sparse_structure_flow_model'].load_state_dict(ss_flow_ckpt, strict=False)
print("✅ SS Flow 加载成功")

# 2. 加载你的 SLAT Flow 权重（纹理生成）
# print("加载你的 SLAT Flow 权重...")
# slat_flow_path = '/data/huanghaoyang/3D/TRELLIS/outputs/slat_flow_training/ckpts/denoiser_step0250000.pt'  # 改为你最新保存的步数
# slat_flow_ckpt = torch.load(slat_flow_path, map_location='cpu')
# pipeline.models['slat_flow_model'].load_state_dict(slat_flow_ckpt, strict=False)
# print("✅ SLAT Flow 加载成功")

pipeline.cuda()
print("✅ 模型准备完成")

# 生成
print("\n开始生成...")
outputs = pipeline.run(
    text_prompt,
    seed=1,
    sparse_structure_sampler_params={
        "steps": 20,
        "cfg_strength": 5.0,
    },
    # slat_sampler_params={
    #     "steps": 20,
    #     "cfg_strength": 5.0,
    # },
)

# 保存结果
with open(os.path.join(output_dir, "prompt.txt"), "w", encoding="utf-8") as f:
    f.write(text_prompt)

# 保存视频
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(os.path.join(output_dir, "sample_gs.mp4"), video, fps=30)

# 保存 GLB
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export(os.path.join(output_dir, "sample.glb"))

# 保存 PLY
outputs['gaussian'][0].save_ply(os.path.join(output_dir, "sample.ply"))

print("\n✅ 完成！")
print(f"📁 结果目录: {output_dir}")