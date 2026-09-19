#!/usr/bin/env python
"""repair_mesh.py — 对已有 mesh 文件做补洞/降面，并打印补前补后的客观指标

为什么单独做一个脚本：整条评估管线（SS Flow → SLat Flow → VAE decode → Poisson → 降面）
一条样本要十几分钟，而补洞本身只要几秒。调补洞参数时不该每次都重跑管线——
拿已经跑出来的 mesh 当输入，秒级迭代。

用法:
    # 单个文件：补前 → 补后 指标对比
    python lato_integration/repair_mesh.py \
        outputs/eval_decim_single/meshes/20250423_1800_838505.obj \
        --out outputs/repair_test/20250423_1800_838505.obj

    # 整个目录（批量补洞 + 汇总对比）
    python lato_integration/repair_mesh.py outputs/eval_decim_single/meshes/ \
        --out_dir outputs/repair_test/

    # 补完再平滑（抹降面造成的粗糙；先看补洞效果就别加）
    python lato_integration/repair_mesh.py in.obj --out out.obj --smooth_iters 3

    # 只补小洞，防大洞被硬填成平板
    python lato_integration/repair_mesh.py in.obj --out out.obj --max_loop 500

配合诊断工具看批量对比:
    python lato_integration/diag_mesh.py outputs/repair_test/ --json outputs/repair_compare.json
"""

import argparse
import glob
import os
import sys

import numpy as np
import trimesh

from lato_integration.diag_mesh import mesh_stats_from_mesh, fmt, quality_summary
from lato_integration.mesh_grid import repair_holes, decimate_mesh

EXTS = (".obj", ".stl", ".ply", ".glb", ".gltf", ".off")


def collect_one(path):
    return [path]


def collect_dir(d):
    files = []
    for e in EXTS:
        files += sorted(glob.glob(os.path.join(d, "*" + e)))
    return files


def main():
    ap = argparse.ArgumentParser(description="对已有 mesh 补洞/降面并输出指标对比")
    ap.add_argument("input", help="mesh 文件 或 目录")
    ap.add_argument("--out", default=None, help="输出文件路径（输入是单个文件时用）")
    ap.add_argument("--out_dir", default=None, help="输出目录（输入是目录时用）")
    ap.add_argument("--max_loop", type=int, default=0,
                    help=">0 时只补顶点数 ≤ 该值的边界环（防大洞被硬填成平板）")
    ap.add_argument("--smooth_iters", type=int, default=0,
                    help="补完后 Laplacian 平滑次数（0=不平滑，先看补洞效果就用 0）")
    ap.add_argument("--decimate", type=int, default=0,
                    help=">0 时补洞前先降面到该面数（注意：降面本身会造洞）")
    a = ap.parse_args()

    if os.path.isdir(a.input):
        files = collect_dir(a.input)
        if not files:
            print(f"[ERROR] {a.input} 下没有 mesh 文件", file=sys.stderr)
            sys.exit(1)
        out_dir = a.out_dir or os.path.join(os.path.dirname(a.input.rstrip("/\\")),
                                            os.path.basename(a.input.rstrip("/\\")) + "_repaired")
        os.makedirs(out_dir, exist_ok=True)
    else:
        files = [a.input]
        out_dir = None
        if a.out is None:
            print("[ERROR] 单个文件输入需要 --out", file=sys.stderr)
            sys.exit(1)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)

    before_stats, after_stats = [], []
    for f in files:
        try:
            mesh = trimesh.load(f, force="mesh", process=True)
        except Exception as e:
            print(f"[WARN] 载入失败 {f}: {e}", file=sys.stderr)
            continue
        if mesh is None or len(mesh.faces) == 0:
            print(f"[WARN] 空 mesh: {f}", file=sys.stderr)
            continue

        name = os.path.basename(f)
        print(f"\n{'=' * 70}\n{name}  v={len(mesh.vertices)} f={len(mesh.faces)}\n{'=' * 70}")
        before = mesh_stats_from_mesh(mesh, f"{name}:补前")
        before_stats.append(before)
        print("[补前]" + fmt(before))

        if a.decimate > 0:
            mesh = decimate_mesh(mesh, a.decimate)

        mesh, stats = repair_holes(mesh, max_loop_len=a.max_loop, smooth_iters=a.smooth_iters)

        after = mesh_stats_from_mesh(mesh, f"{name}:补后")
        after_stats.append(after)
        print("[补后]" + fmt(after))

        out_path = (os.path.join(out_dir, name) if out_dir else a.out)
        mesh.export(out_path)
        print(f"  → 已写出 {out_path}")

    # ── 批量汇总对比 ──
    if len(before_stats) > 1:
        b, af = quality_summary(before_stats), quality_summary(after_stats)
        print(f"\n{'=' * 70}\n汇总对比（{b['num_evaluated']} 个 mesh）\n{'=' * 70}")
        print(f"  {'指标':<20}{'补前':>14}{'补后':>14}")
        print(f"  {'watertight 比例':<18}{b['watertight_rate'] * 100:>13.0f}%"
              f"{af['watertight_rate'] * 100:>13.0f}%")
        for k in ("holes", "boundary_edges", "components", "non_manifold_edges",
                  "dihedral_mean", "dihedral_p90"):
            print(f"  {k:<20}{b[k]['mean']:>14.2f}{af[k]['mean']:>14.2f}")
        for k in ("faces", "vertices"):
            if k in before_stats[0]:
                bv = np.mean([s[k] for s in before_stats])
                av = np.mean([s[k] for s in after_stats])
                print(f"  {k:<20}{bv:>14.0f}{av:>14.0f}")


if __name__ == "__main__":
    main()
