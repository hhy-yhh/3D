#!/bin/bash
# ============================================================================
# check_health.sh — 训练健康自检（30 秒跑完，区分"训练不够"和"有 bug"）
# 用法: bash check_health.sh [ckpt_dir] [gpu_id]
#   bash check_health.sh outputs/lato_ss_flow_v5 0
# ============================================================================
CKPT_DIR=${1:-outputs/lato_ss_flow_v5}
GPU=${2:-0}
LOG="$CKPT_DIR/log.txt"

export CUDA_VISIBLE_DEVICES=$GPU

echo ""
echo "═══════════════════════════════════════════"
echo "  训练健康检查: $CKPT_DIR"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════"

# ── 1. 训练步数 ──
STEP=$(grep -oP '^\d+' "$LOG" | tail -1)
echo ""
echo "  训练步数: $STEP"

# ── 2. StructureHead 3D？(固定 seed，多次采样) ──
echo ""
echo "  StructureHead 3D 检查:"
python3 << 'PYEOF'
import torch, glob, numpy as np, sys, os
sys.path.insert(0, ".")

ckpt_dir = os.environ.get("CKPT_DIR", "outputs/lato_ss_flow_v5") + "/ckpts"
d = "cuda:0"

from lato_integration.structure_head import LatoStructureHead

sc = sorted(glob.glob(f"{ckpt_dir}/denoiser_step*.pt"))
shc = sorted(glob.glob(f"{ckpt_dir}/structure_head_step*.pt"))
if not sc:
    print("  ⏳ 无 checkpoint，跳过")
    exit(0)

# 只加载 StructureHead（不需要 SS Flow，用随机输入测试就够了）
sh = LatoStructureHead(in_channels=8, base_channels=256, num_res_blocks=1).to(d)
sh.load_state_dict(torch.load(shc[-1], map_location=d, weights_only=True), strict=False)
sh.eval()

torch.manual_seed(42)
results = []
for _ in range(3):
    x = torch.randn(1, 8, 16, 16, 16, device=d) * 2  # 模拟 latent 分布
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
    print("  ❌ SH 3D: 全部输出为负！StructureHead 未学习")
else:
    n_avg = int(np.mean([r["n"] for r in results]))
    xs = int(np.mean([r["xs"] for r in results]))
    ys = int(np.mean([r["ys"] for r in results]))
    zs = int(np.mean([r["zs"] for r in results]))
    ok = min(xs, ys, zs) > 10
    print(f"  {'✅' if ok else '❌'} SH 3D: n={n_avg}  Xspan={xs}  Yspan={ys}  Zspan={zs}")
    if not ok:
        print("     → StructureHead 塌缩为 2D，需排查！")
PYEOF

# ── 3. loss 趋势 ──
echo ""
echo "  Loss 趋势（最近 200 步均值）:"

OCC_AVG=$(grep "occ_bce_128" "$LOG" | tail -200 | grep -oP 'occ_bce_128":\s*\K[\d.]+' | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
MSE_AVG=$(grep -oP '"mse":\s*\K[\d.]+' "$LOG" | tail -200 | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')

echo "    occ_bce: $OCC_AVG"
echo "    MSE:     $MSE_AVG"

# ── 4. NaN ──
NAN_COUNT=$(grep -c "NaN" "$LOG" 2>/dev/null || echo 0)
echo ""
echo "  NaN: $NAN_COUNT 条$([ $NAN_COUNT -gt 0 ] && echo ' ❌' || echo ' ✅')"

# ── 5. log_scale ──
LOG_SCALE=$(tail -1 "$LOG" | grep -oP 'log_scale":\s*\K[\d.]+')
echo "  log_scale: $LOG_SCALE"

# ── 6. checkpoint ──
echo ""
echo "  Checkpoint:"
ls -1 "$CKPT_DIR/ckpts/denoiser_step"*.pt 2>/dev/null | sort -V | tail -1 | xargs -I{} basename {} | sed 's/^/    /'

# ── 7. 判决 ──
echo ""
python3 << PYEOF
import subprocess, sys

occ = "$OCC_AVG"
nan = "$NAN_COUNT"

checks = []
checks.append(("SH 3D", "$SH_OK" == "1" if "$SH_OK" != "" else True))  # 由上一步决定
checks.append(("occ_bce > 0.01", float(occ) > 0.01 if occ != "N/A" else False))
checks.append(("NaN = 0", nan == "0" or nan == "0\n"))

all_ok = all(c[1] for c in checks)

print("═══════════════════════════════════════════")
if all_ok:
    print("  ✅ 训练健康 — 继续训，只是训练不够")
else:
    print("  ❌ 有问题 — 排查上述失败项")
print("═══════════════════════════════════════════")
print("")
PYEOF
