#!/bin/bash
# check_health.sh — 训练健康自检（支持 SS Flow / SLat Flow）
# v16: SS Flow v6 (PixelShuffle) + SLat Flow v10 (从零训)
# 用法:
#   bash check_health.sh [ckpt_dir] [gpu_id] [--type ss|slat] [--data_dir PATH]
#   bash check_health.sh outputs/lato_ss_flow_v6 0
#   bash check_health.sh outputs/lato_slat_flow_v10 0 --type slat

CKPT_DIR=${1:-outputs/lato_ss_flow_v6}
GPU=${2:-0}
TRAIN_TYPE=""
DATA_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type) TRAIN_TYPE="$2"; shift 2 ;;
        --type=*) TRAIN_TYPE="${1#*=}"; shift ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --data_dir=*) DATA_DIR="${1#*=}"; shift ;;
        *) shift ;;
    esac
done

LOG="$CKPT_DIR/log.txt"
CKPT_SUBDIR="$CKPT_DIR/ckpts"
export CUDA_VISIBLE_DEVICES=$GPU CKPT_DIR DATA_DIR TRAIN_TYPE

# ── 自动检测训练类型 ──
if [ -z "$TRAIN_TYPE" ]; then
    if ls "$CKPT_SUBDIR"/structure_head_step*.pt &>/dev/null; then
        TRAIN_TYPE="ss"
    elif ls "$CKPT_SUBDIR"/denoiser_step*.pt &>/dev/null; then
        TRAIN_TYPE="slat"
    else
        TRAIN_TYPE="ss"
    fi
fi

echo ""
echo "=== 训练健康检查 @ $(date +%H:%M) ==="
echo "  类型: $TRAIN_TYPE  | 目录: $CKPT_DIR  | GPU: $GPU"
echo ""

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

# ── 2. Loss 检查（通用） ──
if [ -f "$LOG" ]; then
    # 用 ^\d+:.*?"mse" 只匹配每行第一个 "mse"（loss.mse），避免误匹配 bin_* 里嵌套的 mse
    MSE_AVG=$(grep -oP '^\d+:.*?"mse":\s*\K[\d.e+\-]+' "$LOG" 2>/dev/null | tail -200 | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
    NAN=$(grep -c "NaN" "$LOG" 2>/dev/null || true)
    NAN=${NAN:-0}
    # 检查 loss 趋势：最近 50 步 vs 50-100 步前
    MSE_RECENT=$(grep -oP '^\d+:.*?"mse":\s*\K[\d.e+\-]+' "$LOG" 2>/dev/null | tail -50 | awk '{s+=$1;n++}END{if(n>0)printf "%.6f",s/n; else print "N/A"}')
    MSE_OLDER=$(grep -oP '^\d+:.*?"mse":\s*\K[\d.e+\-]+' "$LOG" 2>/dev/null | tail -100 | head -50 | awk '{s+=$1;n++}END{if(n>0)printf "%.6f",s/n; else print "N/A"}')
    LOG_LINES=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    LOG_LAST_STEP=$(grep -oP '^\d+' "$LOG" 2>/dev/null | tail -1)
    LOG_LAST_STEP=${LOG_LAST_STEP:-0}
else
    MSE_AVG="N/A"; MSE_RECENT="N/A"; MSE_OLDER="N/A"; NAN=0; LOG_LINES=0; LOG_LAST_STEP=0
fi

echo "  ── 基础指标 ──"
echo "  checkpoint step:  $STEP"
echo "  log 行数/最后step: $LOG_LINES / $LOG_LAST_STEP"
echo "  MSE (最近200均值): $MSE_AVG"
if [ "$MSE_RECENT" != "N/A" ] && [ "$MSE_OLDER" != "N/A" ]; then
    TREND=$(python3 -c "r=$MSE_RECENT; o=$MSE_OLDER; print('📉下降' if r<o*0.98 else ('📈上升' if r>o*1.02 else '➡️平稳'))" 2>/dev/null)
    echo "  MSE 趋势:          $TREND (最近50: $MSE_RECENT vs 前50: $MSE_OLDER)"
fi
echo "  NaN:               $NAN 条$([ "$NAN" -gt 0 ] && echo ' ❌' || echo ' ✅')"

# ── 3. SS Flow 专项检查 ──
if [ "$TRAIN_TYPE" = "ss" ]; then
    echo ""
    echo "  ── SS Flow 专项 ──"

    # Occ BCE
    if [ -f "$LOG" ]; then
        OCC_AVG=$(grep "occ_bce_128" "$LOG" 2>/dev/null | tail -200 | grep -oP 'occ_bce_128":\s*\K[\d.]+' | awk '{s+=$1;n++}END{if(n>0)printf "%.4f",s/n; else print "N/A"}')
        OCC_RECENT=$(grep "occ_bce_128" "$LOG" 2>/dev/null | tail -50 | grep -oP 'occ_bce_128":\s*\K[\d.]+' | awk '{s+=$1;n++}END{if(n>0)printf "%.6f",s/n; else print "N/A"}')
    else
        OCC_AVG="N/A"; OCC_RECENT="N/A"
    fi
    echo "  occ_bce (最近200均值): $OCC_AVG"
    if [ "$OCC_RECENT" != "N/A" ] && [ "$OCC_RECENT" != "0.0000" ]; then
        echo "  occ_bce (最近50):       $OCC_RECENT"
    elif [ "$OCC_RECENT" = "0.0000" ] || [ "$OCC_RECENT" = "N/A" ]; then
        echo "  ⚠️  occ_bce 未出现或趋零 — StructureHead 可能未参与训练"
    fi

    # StructureHead 3D 检查 (v16 PixelShuffle 架构)
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
    print("[SH 3D] ⏳ 无 checkpoint — 从零训练，早期正常")
    exit(0)

sh = LatoStructureHead(in_channels=8, base_channels=256, num_res_blocks=1).to(d)

# 验证 PixelShuffle 架构：conv 权重的 out_channels 应为 base*8
for name, p in sh.named_parameters():
    if "conv" in name and "weight" in p and p.ndim == 5:
        cout = p.shape[0]
        if cout % 8 == 0:
            print(f"[SH Arch] ✅ PixelShuffle: {name} out_ch={cout} (={cout//8}×8)")
        break

if shc:
    sd = torch.load(shc[-1], map_location=d, weights_only=True)
    miss, unexp = sh.load_state_dict(sd, strict=False)
    n_loaded = len(sd) - len(miss)
    print(f"[SH Wt] 已加载: {n_loaded}/{len(sd)} 匹配" + (f", 缺失: {len(miss)}" if miss else " (全部匹配)"))
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

# ── 4. SLat Flow 专项检查 ──
elif [ "$TRAIN_TYPE" = "slat" ]; then
    echo ""
    echo "  ── SLat Flow 专项 ──"

    # 检查模型 forward 是否正常
    python3 << 'PYEOF'
import torch, glob, sys, os, json
ckpt_dir = os.environ["CKPT_DIR"] + "/ckpts"
d = "cuda:0"
sys.path.insert(0, ".")

# 加载 config 获取模型参数
cfg_path = os.path.join(os.environ["CKPT_DIR"], "config.json")
model_args = {}
if os.path.exists(cfg_path):
    cfg = json.load(open(cfg_path))
    model_args = cfg.get("models", {}).get("denoiser", {}).get("args", {})

# 尝试加载模型
try:
    from lato_integration.flow.slat_flow import EnhancedSLatFlowModel
    model_cls = EnhancedSLatFlowModel
    use_enhanced = True
except:
    from trellis.models.structured_latent_flow import SLatFlowModel
    from trellis.models.lato_slat_flow import LATOSLatFlowModel
    try:
        model_cls = LATOSLatFlowModel
    except:
        model_cls = SLatFlowModel
    use_enhanced = False

# 过滤 args
import inspect
sig_params = set(inspect.signature(model_cls.__init__).parameters.keys())
defaults = {"resolution": 128, "in_channels": 16, "out_channels": 16, "model_channels": 384,
            "cond_channels": 768, "num_blocks": 12, "num_heads": 8, "mlp_ratio": 4,
            "patch_size": 2, "num_io_res_blocks": 2, "io_block_channels": [128],
            "pe_mode": "ape", "use_fp16": True, "use_checkpoint": True}
filtered = {**defaults, **{k: v for k, v in model_args.items() if k in sig_params}}
# 确保 io_block_channels 是 list（config JSON 里是 list，这里保持）
if "io_block_channels" in filtered and not isinstance(filtered.get("io_block_channels"), list):
    filtered.pop("io_block_channels", None)

# 移除 LATO 增强参数字段（如果模型不支持）
for k in list(filtered.keys()):
    if k not in sig_params:
        del filtered[k]

try:
    model = model_cls(**filtered).to(d).eval()
    model_loaded = True
except Exception as e:
    print(f"[SLat Model] ❌ 模型加载失败: {e}")
    model_loaded = False

if model_loaded:
    sc = sorted(glob.glob(f"{ckpt_dir}/denoiser_step*.pt"))
    if sc:
        ckpt = torch.load(sc[-1], map_location=d, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
        print(f"[SLat Model] ✅ 已加载: {os.path.basename(sc[-1])}")
    else:
        print(f"[SLat Model] ⚠️  无 checkpoint，使用随机权重测试")

    # 测试 forward：随机 sparse tensor
    try:
        from trellis.modules.sparse.basic import SparseTensor
        torch.manual_seed(42)
        N = 5000
        coords = torch.randint(0, 128, (N, 3))
        coords = torch.cat([torch.zeros(N, 1, dtype=torch.int32), coords.int()], dim=-1)
        feats = torch.randn(N, 16) * 2.0
        x = SparseTensor(feats=feats.to(d), coords=coords.to(d))
        t = torch.rand(1, device=d) * 1000
        cond = torch.randn(1, 77, 768, device=d)

        with torch.no_grad():
            out = model(x, t, cond)

        out_mean = out.feats.mean().item()
        out_std = out.feats.std().item()
        out_nan = torch.isnan(out.feats).sum().item()
        out_inf = torch.isinf(out.feats).sum().item()

        ok = out_nan == 0 and out_inf == 0 and abs(out_mean) < 100
        print(f"[SLat Forward] {'✅' if ok else '❌'} mean={out_mean:.4f} std={out_std:.4f} NaN={out_nan} Inf={out_inf} voxels={N}")
        if not ok:
            if out_nan > 0: print("  ⚠️  输出含 NaN — 权重可能已损坏")
            if out_inf > 0: print("  ⚠️  输出含 Inf — fp16 溢出或权重异常")
            if abs(out_mean) >= 100: print("  ⚠️  输出均值异常 — 模型可能未收敛")
    except Exception as e:
        print(f"[SLat Forward] ❌ 前向失败: {e}")
PYEOF

    # 检查 lr_scheduler 是否正确配置（v10 关键新增）
    echo ""
    echo "  ── 训练配置检查 ──"
    python3 << 'PYEOF'
import json, os
cfg_path = os.path.join(os.environ["CKPT_DIR"], "config.json")
if os.path.exists(cfg_path):
    cfg = json.load(open(cfg_path))
    lr_sch = cfg.get("trainer", {}).get("args", {}).get("lr_scheduler", {})
    if lr_sch:
        name = lr_sch.get("name", "?")
        t_max = lr_sch.get("args", {}).get("T_max", "?")
        eta_min = lr_sch.get("args", {}).get("eta_min", "?")
        print(f"  lr_scheduler:     ✅ {name}")
        print(f"    T_max:          {t_max}")
        print(f"    eta_min:        {eta_min}")
    else:
        print("  lr_scheduler:     ⚠️  未配置 (v10 建议 CosineAnnealingLR)")
else:
    print("  ⚠️  无 config.json")
PYEOF

    # SLat VAE 兼容性检查（用 stats.json 模拟 denormalize）
    echo ""
    echo "  ── VAE 兼容性预估 ──"
    python3 << 'PYEOF'
import json, os, sys, glob
data_dir = os.environ.get("DATA_DIR", "")
cfg_path = os.path.join(os.environ["CKPT_DIR"], "config.json")

# 尝试从 config 或 data_dir 找到 stats
stats = None
if os.path.exists(cfg_path):
    cfg = json.load(open(cfg_path))
    norm = cfg.get("dataset", {}).get("args", {}).get("normalization", {})
    if norm.get("mean") and norm.get("std"):
        stats = norm

if stats is None and data_dir:
    for stats_path in [
        os.path.join(data_dir, "lato_latents_v2/latents/lato_vae_16dim_128/stats.json"),
        os.path.join(data_dir, "lato_latents/latents/lato_vae_16dim_128/stats.json"),
    ]:
        if os.path.exists(stats_path):
            stats = json.load(open(stats_path))
            break

if stats:
    mean = stats["mean"]
    std = stats["std"]
    # 估算 denormalized 后的 feats 范围
    # 假设模型输出 ≈ 归一化尺度（mean~0, std~1）
    denorm_mean = [m for m in mean[:4]]
    denorm_std = [s for s in std[:4]]
    # 预期 VAE 输入范围: mean ± 3*std per channel
    low = [mean[i] - 3*std[i] for i in range(4)]
    high = [mean[i] + 3*std[i] for i in range(4)]
    ok = all(abs(v) < 10 for v in low + high)
    print(f"  stats mean[:4]: {[f'{v:.4f}' for v in denorm_mean]}")
    print(f"  stats std[:4]:  {[f'{v:.4f}' for v in denorm_std]}")
    print(f"  denorm 范围:     [{low[0]:.3f}, {high[0]:.3f}] ...")
    print(f"  VAE 安全:        {'✅ 范围内' if ok else '⚠️  超出 ±10'}")
else:
    print("  ⚠️  未找到 stats.json — 推理时需传 --slat_stats")
PYEOF
fi

# ── 5. 总结 ──
echo ""
echo "  ── 文件清单 ──"
echo "  denoiser:      $(basename "$DENOISER_CKPT" 2>/dev/null || echo '❌ 未找到')"
if [ "$TRAIN_TYPE" = "ss" ]; then
    echo "  structure_head: $(basename "$SH_CKPT" 2>/dev/null || echo '❌ 未找到')"
fi
echo "  log:           $([ -f "$LOG" ] && echo "$LOG ($LOG_LINES 行, step→$LOG_LAST_STEP)" || echo '❌ 未找到')"

echo ""
echo "  ── 判断 ──"
ISSUES=0
if [ "$NAN" -gt 0 ]; then echo "  ❌ 有 NaN"; ISSUES=$((ISSUES+1)); fi
if [ "$MSE_AVG" = "N/A" ]; then
    if [ -d "$CKPT_SUBDIR" ]; then
            echo "  ℹ️  无 log 数据 — 从零训练，早期 (< i_log 步) 或运行中，属正常现象"
    else
        echo "  ⚠️  无 log 且 ckpts/ 目录不存在 — 训练可能尚未初始化"
        ISSUES=$((ISSUES+1))
    fi
elif [ "$LOG_LINES" -lt 2 ]; then
    echo "  ⚠️  log 行数太少 (<2) — 训练刚启动或 log 写入异常"
fi
if [ "$TRAIN_TYPE" = "ss" ] && [ "$OCC_AVG" = "N/A" ]; then echo "  ⚠️  occ_bce 不存在 — StructureHead 可能未参与"; ISSUES=$((ISSUES+1)); fi
if [ "$TRAIN_TYPE" = "ss" ] && [ "$OCC_AVG" != "N/A" ] && [ "$OCC_AVG" != "0.0000" ]; then
    OCC_VAL=$(python3 -c "print(float('$OCC_AVG'))" 2>/dev/null || echo "0")
    if python3 -c "exit(0 if float('$OCC_VAL') < 0.001 else 1)" 2>/dev/null; then
        echo "  ⚠️  occ_bce 趋近 0 ($OCC_AVG) — 可能有 bug 或训练未开始"
        ISSUES=$((ISSUES+1))
    fi
fi
if [ "$TRAIN_TYPE" = "slat" ] && [ "$MSE_AVG" != "N/A" ]; then
    if python3 -c "exit(0 if float('$MSE_AVG') > 2.0 else 1)" 2>/dev/null; then
        echo "  ⚠️  MSE 过高 (>2.0) — 训练初期或有问题"
        ISSUES=$((ISSUES+1))
    fi
fi
if [ $ISSUES -eq 0 ]; then echo "  ✅ 全部指标正常 — 继续训练"; fi
echo ""
