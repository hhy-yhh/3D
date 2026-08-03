#!/bin/bash
# ============================================================================
# check_health.sh — 训练健康自检
# 用法: bash check_health.sh [ckpt_dir] [gpu_id] [--data_dir PATH]
# ============================================================================
CKPT_DIR=${1:-outputs/lato_ss_flow_v5}
GPU=${2:-0}
DATA_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --data_dir=*) DATA_DIR="${1#*=}"; shift ;;
        *) shift ;;
    esac
done

LOG="$CKPT_DIR/log.txt"
CKPT_SUBDIR="$CKPT_DIR/ckpts"

export CUDA_VISIBLE_DEVICES=$GPU
export CKPT_DIR="$CKPT_DIR"
export DATA_DIR="$DATA_DIR"

echo ""
echo "=== 训练健康检查 @ $(date +%H:%M) ==="

# ── 1. 步数：从 ckpt 文件名提取（最可靠） ──
DENOISER_CKPT=$(ls -1 "$CKPT_SUBDIR"/denoiser_step*.pt 2>/dev/null | sort -V | tail -1)
SH_CKPT=$(ls -1 "$CKPT_SUBDIR"/structure_head_step*.pt 2>/dev/null | sort -V | tail -1)

if [ -n "$DENOISER_CKPT" ]; then
    STEP=$(basename "$DENOISER_CKPT" | grep -oP 'step\K\d+')
    # 去前导零
    STEP=$((10#$STEP))
else
    STEP="N/A"
fi

# ── 2. StructureHead 3D 检查 ──
python3 << 'PYEOF'
import torch, glob, numpy as np, sys, os, json

ckpt_dir = os.environ.get("CKPT_DIR", "outputs/lato_ss_flow_v5") + "/ckpts"
data_dir = os.environ.get("DATA_DIR", "")
d = "cuda:0"

sys.path.insert(0, ".")
from lato_integration.structure_head import LatoStructureHead

sc = sorted(glob.glob(f"{ckpt_dir}/denoiser_step*.pt"))
shc = sorted(glob.glob(f"{ckpt_dir}/structure_head_step*.pt"))

if not sc:
    print("[SPARSE] ⏳ 无 checkpoint，跳过")
    exit(0)

sh = LatoStructureHead(in_channels=8, base_channels=256, num_res_blocks=1).to(d)
if shc:
    sh.load_state_dict(torch.load(shc[-1], map_location=d, weights_only=True), strict=False)
sh.eval()

# ── 搜真实 latent ──
real_latents = []
search_dirs = []

if data_dir:
    search_dirs += [
        os.path.join(data_dir, "lato_latents_v2", "latents", "lato_vae_16dim_128"),
        os.path.join(data_dir, "lato_latents", "latents", "lato_vae_16dim_128"),
    ]

# 从 config.json 推断 data_dir
cfg_path = os.path.join(os.environ.get("CKPT_DIR", ""), "config.json")
if os.path.exists(cfg_path):
    try:
        cfg_d = json.load(open(cfg_path)).get("data_dir", "")
        if cfg_d:
            search_dirs += [
                os.path.join(cfg_d, "lato_latents_v2", "latents", "lato_vae_16dim_128"),
                os.path.join(cfg_d, "lato_latents", "latents", "lato_vae_16dim_128"),
            ]
    except: pass

for dpath in search_dirs:
    if os.path.isdir(dpath):
        for f in sorted(glob.glob(os.path.join(dpath, "*.npz")))[:5]:
            try:
                arr = np.load(f, allow_pickle=True)
                if 'x_0' in arr:
                    real_latents.append(torch.tensor(arr['x_0']).float())
            except: pass
        if real_latents: break

# ── 测试 ──
torch.manual_seed(42)
results = []
inputs_used = ""

if real_latents:
    inputs_used = f"真实latent x{len(real_latents)}"
    for lat in real_latents:
        x = lat.unsqueeze(0).to(d)
        with torch.no_grad():
            occ = sh(x)
        pos = np.where(occ[0, 0].cpu().numpy() > 0)
        n = len(pos[0])
        if n > 0:
            results.append({
                "n": n, "xs": pos[2].max()-pos[2].min(),
                "ys": pos[1].max()-pos[1].min(), "zs": pos[0].max()-pos[0].min(),
            })
else:
    inputs_used = "随机输入(scale=0.1/0.5/1.0/2.0)"
    for scale in [0.1, 0.5, 1.0, 2.0]:
        torch.manual_seed(42)
        x = torch.randn(1, 8, 16, 16, 16, device=d) * scale
        with torch.no_grad():
            occ = sh(x)
        pos = np.where(occ[0, 0].cpu().numpy() > 0)
        n = len(pos[0])
        if n > 0:
            results.append({
                "n": n, "xs": pos[2].max()-pos[2].min(),
                "ys": pos[1].max()-pos[1].min(), "zs": pos[0].max()-pos[0].min(),
            })

if not results:
    print(f"[SH 3D] ❌ 全负 (输入: {inputs_used})")
else:
    n_avg = int(np.mean([r["n"] for r in results]))
    xs = int(np.mean([r["xs"] for r in results]))
    ys = int(np.mean([r["ys"] for r in results]))
    zs = int(np.mean([r["zs"] for r in results]))
    ok = min(xs, ys, zs) > 10
    print(f"[SH 3D] {'✅' if ok else '❌'} n={n_avg} Xspan={xs} Yspan={ys} Zspan={zs} ({inputs_used})")
PYEOF

# ── 3. loss（最近200步均值） ──
if [ -f "$LOG" ]; then
    OCC_AVG=$(grep "occ_bce_128" "$LOG" 2>/dev/null | tail -200 | grep -oP 'occ_bce_128"?\s*:\s*[\d.]+(?:e[+-]?\d+)?' | grep -oP '[\d.]+$' | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
    MSE_AVG=$(grep -oP '"mse"?\s*:\s*[\d.]+(?:e[+-]?\d+)?' "$LOG" 2>/dev/null | grep -oP '[\d.]+$' | tail -200 | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
    NAN_COUNT=$(grep -c "NaN" "$LOG" 2>/dev/null || echo 0)
else
    OCC_AVG="N/A"
    MSE_AVG="N/A"
    NAN_COUNT=0
fi

echo "  occ_bce(最近200均值): $OCC_AVG"
echo "  MSE(最近200均值):      $MSE_AVG"
echo "  NaN: $NAN_COUNT 条$([ $NAN_COUNT -gt 0 ] && echo ' ❌' || echo ' ✅')"
echo "  step: $STEP"
echo ""
echo "  denoiser:      $(basename "$DENOISER_CKPT" 2>/dev/null || echo '❌ 无')"
echo "  structure_head: $(basename "$SH_CKPT" 2>/dev/null || echo '❌ 无')"
echo ""
