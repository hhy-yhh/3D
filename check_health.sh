#!/bin/bash
# check_health.sh — 训练健康自检
# 用法: bash check_health.sh [ckpt_dir] [gpu_id] [--data_dir PATH]
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
export CUDA_VISIBLE_DEVICES=$GPU CKPT_DIR DATA_DIR

echo ""
echo "=== 训练健康检查 @ $(date +%H:%M) ==="

# ── 1. 步数 ──
DENOISER_CKPT=$(ls -1 "$CKPT_SUBDIR"/denoiser_step*.pt 2>/dev/null | sort -V | tail -1)
SH_CKPT=$(ls -1 "$CKPT_SUBDIR"/structure_head_step*.pt 2>/dev/null | sort -V | tail -1)
if [ -n "$DENOISER_CKPT" ]; then
    STEP=$(basename "$DENOISER_CKPT" | grep -oP 'step\K\d+')
    STEP=$((10#$STEP))
else
    STEP=$(grep -oP '^\d+' "$LOG" 2>/dev/null | tail -1)
    STEP=${STEP:-N/A}
fi

# ── 2. StructureHead 3D ──
python3 << 'PYEOF'
import torch, glob, numpy as np, sys, os, json
ckpt_dir = os.environ["CKPT_DIR"] + "/ckpts"
data_dir = os.environ.get("DATA_DIR", "")
d = "cuda:0"
sys.path.insert(0, ".")
from lato_integration.structure_head import LatoStructureHead

sc = sorted(glob.glob(f"{ckpt_dir}/denoiser_step*.pt"))
shc = sorted(glob.glob(f"{ckpt_dir}/structure_head_step*.pt"))
if not sc:
    print("[SH] ⏳ 无 checkpoint")
    exit(0)

sh = LatoStructureHead(in_channels=8, base_channels=256, num_res_blocks=1).to(d)
if shc:
    sh.load_state_dict(torch.load(shc[-1], map_location=d, weights_only=True), strict=False)
sh.eval()

real_latents = []
search = []
if data_dir:
    search += [os.path.join(data_dir, "lato_latents_v2/latents/lato_vae_16dim_128"),
               os.path.join(data_dir, "lato_latents/latents/lato_vae_16dim_128")]
cfg_path = os.path.join(os.environ["CKPT_DIR"], "config.json")
if os.path.exists(cfg_path):
    try:
        cfg_d = json.load(open(cfg_path)).get("data_dir", "")
        if cfg_d:
            search += [os.path.join(cfg_d, "lato_latents_v2/latents/lato_vae_16dim_128"),
                       os.path.join(cfg_d, "lato_latents/latents/lato_vae_16dim_128")]
    except: pass
for dpath in search:
    if os.path.isdir(dpath):
        for f in sorted(glob.glob(os.path.join(dpath, "*.npz")))[:5]:
            try:
                arr = np.load(f, allow_pickle=True)
                if 'x_0' in arr:
                    real_latents.append(torch.tensor(arr['x_0']).float())
            except: pass
        if real_latents: break

torch.manual_seed(42)
results, tag = [], ""
if real_latents:
    tag = f"真实latent x{len(real_latents)}"
    for lat in real_latents:
        x = lat.unsqueeze(0).to(d)
        with torch.no_grad(): occ = sh(x)
        pos = np.where(occ[0,0].cpu().numpy() > 0)
        if len(pos[0]) > 0:
            results.append({"n": len(pos[0]), "xs": pos[2].max()-pos[2].min(),
                           "ys": pos[1].max()-pos[1].min(), "zs": pos[0].max()-pos[0].min()})
else:
    tag = "随机(0.1/0.5/1.0/2.0)"
    for s in [0.1, 0.5, 1.0, 2.0]:
        torch.manual_seed(42)
        x = torch.randn(1, 8, 16, 16, 16, device=d) * s
        with torch.no_grad(): occ = sh(x)
        pos = np.where(occ[0,0].cpu().numpy() > 0)
        if len(pos[0]) > 0:
            results.append({"n": len(pos[0]), "xs": pos[2].max()-pos[2].min(),
                           "ys": pos[1].max()-pos[1].min(), "zs": pos[0].max()-pos[0].min()})

if not results:
    print(f"[SH 3D] ❌ 全负 ({tag})")
else:
    n_avg = int(np.mean([r["n"] for r in results]))
    xs, ys, zs = [int(np.mean([r[k] for r in results])) for k in ["xs","ys","zs"]]
    ok = min(xs, ys, zs) > 10
    print(f"[SH 3D] {'✅' if ok else '❌'} n={n_avg} Xspan={xs} Yspan={ys} Zspan={zs} ({tag})")
PYEOF

# ── 3. loss（最近200步均值） ──
if [ -f "$LOG" ]; then
    OCC_AVG=$(grep "occ_bce_128" "$LOG" 2>/dev/null | tail -200 | grep -oP 'occ_bce_128":\s*\K[\d.]+' | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
    MSE_AVG=$(grep -oP '"mse":\s*\K[\d.]+' "$LOG" 2>/dev/null | tail -200 | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
    NAN=$(grep -c "NaN" "$LOG" 2>/dev/null || true)
    NAN=${NAN:-0}
else
    OCC_AVG="N/A"; MSE_AVG="N/A"; NAN=0
fi

echo "  occ_bce(最近200均值): $OCC_AVG"
echo "  MSE(最近200均值):      $MSE_AVG"
echo "  NaN: $NAN 条$([ "$NAN" -gt 0 ] && echo ' ❌' || echo ' ✅')"
echo "  step: $STEP"
echo ""
echo "  denoiser:      $(basename "$DENOISER_CKPT" 2>/dev/null || echo '❌')"
echo "  structure_head: $(basename "$SH_CKPT" 2>/dev/null || echo '❌')"
echo ""
