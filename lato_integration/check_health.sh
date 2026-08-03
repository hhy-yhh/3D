#!/bin/bash
# ============================================================================
# check_health.sh — 训练健康自检（30 秒跑完，区分"训练不够"和"有 bug"）
# 用法: bash check_health.sh [ckpt_dir] [gpu_id] [--data_dir PATH]
#   bash check_health.sh outputs/lato_ss_flow_v5 0
#   bash check_health.sh outputs/lato_ss_flow_v5 0 --data_dir /data/huanghaoyang/3D/database_lato
# ============================================================================
CKPT_DIR=${1:-outputs/lato_ss_flow_v5}
GPU=${2:-0}
LOG="$CKPT_DIR/log.txt"
DATA_DIR=""

# 解析 --data_dir
for i in "$@"; do
    if [ "$i" = "--data_dir" ]; then
        shift
        DATA_DIR="$2"
    fi
    shift 2>/dev/null
done

export CUDA_VISIBLE_DEVICES=$GPU
export CKPT_DIR="$CKPT_DIR"
export DATA_DIR="$DATA_DIR"

echo ""
echo "═══════════════════════════════════════════"
echo "  训练健康检查: $CKPT_DIR"
echo "  $(date '+%H:%M')"
echo "═══════════════════════════════════════════"

# ── 1. 训练步数 ──
# 🔧 兼容多种 log 格式：行首数字 / JSON "step" / 纯数字行
if [ -f "$LOG" ]; then
    STEP=$(grep -oP '(?:^\d+(?=:)|"step":\s*\d+|(?<!\w)\d{4,}(?!\w))' "$LOG" 2>/dev/null | grep -oP '\d+' | tail -1)
else
    STEP=""
fi
echo ""
echo "  step: ${STEP:-N/A}"

# ── 2. StructureHead 3D？（优先用真实 latent，fallback 到多 scale 随机输入） ──
echo ""
python3 << 'PYEOF'
import torch, glob, numpy as np, sys, os

ckpt_dir = os.environ.get("CKPT_DIR", "outputs/lato_ss_flow_v5") + "/ckpts"
data_dir = os.environ.get("DATA_DIR", "")
d = "cuda:0"

sys.path.insert(0, ".")
from lato_integration.structure_head import LatoStructureHead

# 显示后端信息
try:
    import spconv
    print(f"  [SPARSE] Backend: spconv, Attention: xformers")
except:
    pass

sc = sorted(glob.glob(f"{ckpt_dir}/denoiser_step*.pt"))
shc = sorted(glob.glob(f"{ckpt_dir}/structure_head_step*.pt"))

if not sc:
    print("  ⏳ 无 denoiser checkpoint，跳过 StructureHead 检查")
    exit(0)

# 加载 StructureHead
sh = LatoStructureHead(in_channels=8, base_channels=256, num_res_blocks=1).to(d)
if shc:
    sd = torch.load(shc[-1], map_location=d, weights_only=True)
    sh.load_state_dict(sd, strict=False)
    print(f"  [加载: {os.path.basename(shc[-1])}]")
else:
    print("  [⚠️ 无 structure_head ckpt，使用随机初始化]")
sh.eval()

# ── 尝试加载真实 latent ──
real_latents = []
latent_dirs = []

# 从 data_dir 推导 latent 路径
if data_dir:
    latent_dirs += [
        os.path.join(data_dir, "lato_latents_v2", "latents", "lato_vae_16dim_128"),
        os.path.join(data_dir, "lato_latents", "latents", "lato_vae_16dim_128"),
    ]

# 也尝试从 CKPT_DIR 的 config.json 读取 data_dir
config_path = os.path.join(os.environ.get("CKPT_DIR", "outputs/lato_ss_flow_v5"), "config.json")
if os.path.exists(config_path):
    try:
        import json
        cfg = json.load(open(config_path))
        cfg_data = cfg.get("data_dir", "")
        if cfg_data:
            latent_dirs += [
                os.path.join(cfg_data, "lato_latents_v2", "latents", "lato_vae_16dim_128"),
                os.path.join(cfg_data, "lato_latents", "latents", "lato_vae_16dim_128"),
            ]
    except:
        pass

for lat_dir in latent_dirs:
    if os.path.isdir(lat_dir):
        npz_files = sorted(glob.glob(os.path.join(lat_dir, "*.npz")))[:5]
        for f in npz_files:
            try:
                data = np.load(f, allow_pickle=True)
                if 'x_0' in data:
                    real_latents.append(torch.tensor(data['x_0']).float())
            except:
                pass
        if real_latents:
            break

# ── 测试 ──
torch.manual_seed(42)
results = []

if real_latents:
    print(f"  [使用 {len(real_latents)} 个真实 latent 样本]")
    for lat in real_latents:
        x = lat.unsqueeze(0).to(d)
        with torch.no_grad():
            occ = sh(x)
        pos = np.where(occ[0, 0].cpu().numpy() > 0)
        n = len(pos[0])
        if n > 0:
            results.append({
                "n": n,
                "xs": pos[2].max() - pos[2].min(),
                "ys": pos[1].max() - pos[1].min(),
                "zs": pos[0].max() - pos[0].min(),
            })
else:
    print(f"  [无真实 latent，用多 scale 随机输入测试]")
    for scale in [0.1, 0.5, 1.0, 2.0]:
        torch.manual_seed(42)
        x = torch.randn(1, 8, 16, 16, 16, device=d) * scale
        with torch.no_grad():
            occ = sh(x)
        pos = np.where(occ[0, 0].cpu().numpy() > 0)
        n = len(pos[0])
        if n > 0:
            results.append({
                "n": n,
                "xs": pos[2].max() - pos[2].min(),
                "ys": pos[1].max() - pos[1].min(),
                "zs": pos[0].max() - pos[0].min(),
            })

if not results:
    print("  SH 3D: ❌  全负")
    if not real_latents:
        print("  ⚠️ 可能是测试输入不匹配（随机噪声 ≠ Flow Matching latent）")
        print("     用 --data_dir 指定数据集路径以加载真实 latent")
else:
    n_avg = int(np.mean([r["n"] for r in results]))
    xs = int(np.mean([r["xs"] for r in results]))
    ys = int(np.mean([r["ys"] for r in results]))
    zs = int(np.mean([r["zs"] for r in results]))
    ok = min(xs, ys, zs) > 10
    print(f"  SH 3D: {'✅' if ok else '❌'}  n={n_avg}  Xspan={xs}  Yspan={ys}  Zspan={zs}")
    if not ok:
        print("     → StructureHead 可能塌缩为 2D")
PYEOF

# ── 3. loss 趋势 ──
echo ""
if [ -f "$LOG" ]; then
    OCC_AVG=$(grep "occ_bce_128" "$LOG" 2>/dev/null | tail -200 | grep -oP 'occ_bce_128"?\s*:\s*\K[\d.]+(?:e[+-]?\d+)?' | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
    MSE_AVG=$(grep -oP '"mse"?\s*:\s*\K[\d.]+(?:e[+-]?\d+)?' "$LOG" 2>/dev/null | tail -200 | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
else
    OCC_AVG="N/A"
    MSE_AVG="N/A"
fi
echo "  occ_bce(最近200均值): $OCC_AVG"
echo "  MSE(最近200均值):      $MSE_AVG"

# ── 4. NaN ──
if [ -f "$LOG" ]; then
    NAN_COUNT=$(grep -c "NaN" "$LOG" 2>/dev/null || echo 0)
else
    NAN_COUNT=0
fi
NAN_COUNT=$(echo "$NAN_COUNT" | head -1)
if [ "$NAN_COUNT" -gt 0 ]; then
    echo "  NaN: $NAN_COUNT 条 ❌"
else
    echo "  NaN: 0 条 ✅"
fi
echo "  step: ${STEP:-N/A}"

# ── 5. checkpoint 文件 ──
echo ""
DENOISER_CKPT=$(ls -1 "$CKPT_DIR/ckpts/denoiser_step"*.pt 2>/dev/null | sort -V | tail -1)
SH_CKPT=$(ls -1 "$CKPT_DIR/ckpts/structure_head_step"*.pt 2>/dev/null | sort -V | tail -1)
echo "  最新 checkpoint:"
if [ -n "$DENOISER_CKPT" ]; then echo "    denoiser:      $(basename "$DENOISER_CKPT")"; else echo "    denoiser:      ❌ 无"; fi
if [ -n "$SH_CKPT" ]; then echo "    structure_head: $(basename "$SH_CKPT")"; else echo "    structure_head: ❌ 无"; fi

echo ""
echo "═══════════════════════════════════════════"
echo "  判断: SH 3D ✅ + occ_bce>0.01 + NaN=0 = 训练健康"
echo "  ⚠️ occ_bce 正常但 SH 全负 → 测试输入不匹配(加 --data_dir)"
echo "═══════════════════════════════════════════"
echo ""
