import os
os.environ['SPCONV_ALGO'] = 'native'

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# 加载官方预训练模型（对照实验）
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

# 用和之前相同的文本描述
outputs = pipeline.run(
    "a brake caliper with 4 pistons",
    seed=1,
)

# 保存结果
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("official_sample_gs.mp4", video, fps=30)

video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave("official_sample_rf.mp4", video, fps=30)

video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave("official_sample_mesh.mp4", video, fps=30)

glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export("official_sample.glb")

print("✅ 官方模型推理完成！")