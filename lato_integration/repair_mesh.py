#!/usr/bin/env python
"""repair_mesh.py — 对已有 mesh / 点云做重建、补洞、降面，并打印前后客观指标

为什么单独做一个脚本：整条评估管线（SS Flow → SLat Flow → VAE decode → 重建）
一条样本要十几分钟，而重建/补洞本身只要几秒到一分钟。调参时不该每次都重跑管线——
拿已经跑出来的产物当输入，秒级迭代。

输入可以是：
  - 一个 mesh 文件（.obj/.stl/...）：点云取它的顶点
  - 一个点云 npz：评估脚本用 --dump_coords 存下来的 decode 顶点（含 coords + last_res）
    ★ 建议先 dump 一次，之后所有重建方式都能拿它秒级对比，不用再跑管线

用法:
    # ① 先 dump 一次解码顶点（评估脚本里加 --dump_coords，跑一次管线）
    #    → outputs/coords/20250423_1800_838505.npz

    # ② 之后各种重建方式秒级对比（同一份点云，只换 --rebuild）
    for M in poisson ball alpha; do
      python lato_integration/repair_mesh.py outputs/coords/20250423_1800_838505.npz \
          --rebuild $M --out outputs/rebuild_$M/20250423_1800_838505.obj
    done

    # Poisson 换 depth 拿原生少面（GT 22.9k 面：d6≈16k, d7≈65k, d8≈260k, d9≈1.05M）
    python lato_integration/repair_mesh.py coords.npz \
        --rebuild poisson --depth 7 --out outputs/p7.obj

    # Ball Pivoting / Alpha Shape：只连该连的点、不填补空间 → 保内部通孔
    python lato_integration/repair_mesh.py coords.npz \
        --rebuild ball --radii 0.004,0.008,0.016 --out outputs/ball.obj
    python lato_integration/repair_mesh.py coords.npz \
        --rebuild alpha --alpha 0.01 --out outputs/alpha.obj

    # 只对已有 mesh 补洞（--repair 默认关，需要时显式加）
    python lato_integration/repair_mesh.py in.obj --out out.obj --repair

    # 批量（输入目录）
    python lato_integration/repair_mesh.py outputs/coords/ \
        --rebuild ball --out_dir outputs/rebuild_ball/
"""

import argparse
import glob
import os
import sys

import numpy as np
import trimesh

from lato_integration.diag_mesh import mesh_stats_from_mesh, fmt, quality_summary
from lato_integration.mesh_grid import (
    repair_holes, decimate_mesh, poisson_from_points, build_mesh_from_points,
)

EXTS = (".obj", ".stl", ".ply", ".glb", ".gltf", ".off")


def collect_dir(d):
    files = []
    for e in EXTS + (".npz",):
        files += sorted(glob.glob(os.path.join(d, "*" + e)))
    return files


def load_points(path):
    """从一个 npz 读出归一化点云 [-0.5, 0.5]。返回 (pts, 说明字符串)。"""
    d = np.load(path)
    key = None
    for k in ("coords", "points", "vertex_coords", "vertices"):
        if k in d:
            key = k
            break
    if key is None:
        raise ValueError(f"{path} 里找不到点云键（实际: {list(d.keys())}）")
    c = np.asarray(d[key], dtype=np.float64)
    if c.shape[-1] == 4:          # spconv 的 coords 第一列是 batch 索引
        c = c[:, 1:]
    last_res = float(d["last_res"]) if "last_res" in d else 512.0
    pts = c / last_res - 0.5
    return pts, f"{len(pts)} 点 (last_res={last_res:g})"


def main():
    ap = argparse.ArgumentParser(description="重建 / 补洞 / 降面 + 指标对比")
    ap.add_argument("input", help="mesh 文件、点云 npz，或目录")
    ap.add_argument("--out", default=None, help="输出文件（单个输入时用）")
    ap.add_argument("--out_dir", default=None, help="输出目录（目录输入时用）")

    ap.add_argument("--rebuild", default="none",
                    choices=["none", "poisson", "ball", "alpha"],
                    help="重建方式。none=不重建（只做补洞/降面）。"
                         "ball/alpha 只连该连的点、不填补空间 → 保内部通孔；"
                         "poisson 会填死内部空腔，但表面最光滑")
    ap.add_argument("--depth", type=int, default=8, help="[poisson] 八叉树深度（小=面少=更光滑）")
    ap.add_argument("--poisson_crop", type=float, default=0.1,
                    help="[poisson] 密度裁剪分位（低=少裁。注意：裁剪会开口造洞）")
    ap.add_argument("--radii", default="",
                    help="[ball] 球半径，逗号分隔（绝对单位）。留空=按平均最近邻距自动推")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="[alpha] α 值（绝对单位）。0=按平均最近邻距 × 3 自动推")

    ap.add_argument("--repair", action="store_true", default=False,
                    help="重建后用邻接面插值补洞（默认关）")
    ap.add_argument("--list_loops", action="store_true", default=False,
                    help="只列出所有边界环（顶点数 / 质心 / bbox），不补洞。"
                         "用来区分「表面缝隙」（小环，该补）和「镂空开口」（大环，该留），"
                         "然后据此定 --max_loop 阈值")
    ap.add_argument("--dump_loops", default=None,
                    help="把最大的几个边界环导成点云 PLY 到这个目录，用 CloudCompare 叠在"
                         "原 mesh 上看它们落在哪——判断某个大环是「镂空开口」还是「表面缺口」")
    ap.add_argument("--dump_loops_top", type=int, default=5,
                    help="[--dump_loops] 导出最大的几个环（默认 5）")
    ap.add_argument("--max_loop", type=int, default=0,
                    help="[--repair] >0 时只补顶点数 ≤ 该值的环（防大洞被硬填成平板）")
    ap.add_argument("--smooth_iters", type=int, default=0,
                    help="重建后 Laplacian 平滑次数（0=不平滑）")
    ap.add_argument("--decimate", type=int, default=0,
                    help=">0 时最后降面到目标面数（注意：降面本身会造洞）")
    a = ap.parse_args()

    if os.path.isdir(a.input):
        files = collect_dir(a.input)
        if not files:
            print(f"[ERROR] {a.input} 下没有可处理的文件", file=sys.stderr)
            sys.exit(1)
        base = os.path.basename(a.input.rstrip("/\\"))
        out_dir = a.out_dir or os.path.join(os.path.dirname(a.input.rstrip("/\\")),
                                            base + "_out")
        os.makedirs(out_dir, exist_ok=True)
    else:
        files = [a.input]
        out_dir = None
        if a.out is None and not (a.list_loops or a.dump_loops):
            print("[ERROR] 单个文件输入需要 --out（或改用 --out_dir）", file=sys.stderr)
            sys.exit(1)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)

    mode = a.rebuild
    radii = [float(x) for x in a.radii.split(",") if x.strip()] if a.radii else None

    in_stats, out_stats = [], []
    for f in files:
        name = os.path.basename(f)
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")

        # ── 载入：mesh 走 trimesh，npz 走点云 ──
        is_npz = f.lower().endswith(".npz")
        try:
            if is_npz:
                pts, desc = load_points(f)
                mesh = None
                print(f"  [输入] 点云 {desc}  bounds={np.round(pts.min(0), 3)}~{np.round(pts.max(0), 3)}")
            else:
                mesh = trimesh.load(f, force="mesh", process=True)
                if mesh is None or len(mesh.faces) == 0:
                    print(f"[WARN] 空 mesh: {f}", file=sys.stderr)
                    continue
                pts = np.asarray(mesh.vertices, dtype=np.float64)
                print(f"  [输入] mesh v={len(mesh.vertices)} f={len(mesh.faces)}")
                st = mesh_stats_from_mesh(mesh, f"{name}:输入")
                in_stats.append(st)
                print("[输入]" + fmt(st))
        except Exception as e:
            print(f"[WARN] 载入失败 {f}: {e}", file=sys.stderr)
            continue

        # ── 重建 ──
        if mode != "none":
            try:
                if mode == "poisson":
                    print(f"[mesh_grid] Poisson 重建, {len(pts)} 点, depth={a.depth}")
                    mesh = poisson_from_points(
                        pts, depth=a.depth, crop_density_quantile=a.poisson_crop,
                        smooth_iterations=max(a.smooth_iters, 0))
                else:
                    print(f"[mesh_grid] {mode} 重建, {len(pts)} 点")
                    mesh, _info = build_mesh_from_points(
                        pts, mode=mode, radii=radii, alpha=(a.alpha or None),
                        smooth_iterations=max(a.smooth_iters, 0))
            except Exception as e:
                print(f"[WARN] {mode} 重建失败: {e}", file=sys.stderr)
            if mesh is None or len(mesh.faces) == 0:
                print(f"  [WARN] {mode} 没产出 mesh，保留原 mesh")
                mesh = trimesh.load(f, force="mesh", process=True) if not is_npz else None

        if mesh is None:
            # 点云输入 + --rebuild none：直接把点云导成 PLY，用来肉眼判断
            # 「镂空的壁在不在点云里」——这决定了换重建方式有没有用。
            if is_npz and mode == "none":
                out_path = (os.path.join(out_dir, os.path.splitext(name)[0] + ".ply")
                            if out_dir else a.out)
                pc = trimesh.PointCloud(pts)
                pc.export(out_path)
                print(f"  → 已写出点云 {out_path}（{len(pts)} 点，直接用mesh查看器打开看镂空在不在）")
            continue

        # ── 只列边界环（不补）：区分「表面缝隙」和「镂空开口」──
        if a.list_loops:
            from lato_integration.mesh_grid import boundary_loops
            loops = sorted(boundary_loops(mesh), key=len, reverse=True)
            print(f"\n  边界环 {len(loops)} 个（按顶点数从大到小）——"
                  f"小环=表面缝隙（该补），大环=镂空开口（该留）：")
            print(f"  {'#':>4}  {'顶点数':>8}  {'质心 x,y,z':>30}  {'bbox dx,dy,dz':>24}")
            for i, lp in enumerate(loops):
                p = np.asarray(mesh.vertices, dtype=np.float64)[lp]
                c, sz = p.mean(axis=0), p.max(axis=0) - p.min(axis=0)
                print(f"  {i:>4}  {len(lp):>8}  "
                      f"[{c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}]".rjust(30)
                      + f"  [{sz[0]:.3f},{sz[1]:.3f},{sz[2]:.3f}]".rjust(24))
            print(f"\n  下一步：--max_loop 取一个介于「小环」和「大环」之间的值，"
                  f"补洞时就只补小环、放过镂空开口")
            continue

        # ── 导出边界环点云：叠在原 mesh 上看大环落在哪 ──
        if a.dump_loops:
            from lato_integration.mesh_grid import boundary_loops
            loops = sorted(boundary_loops(mesh), key=len, reverse=True)
            os.makedirs(a.dump_loops, exist_ok=True)
            V = np.asarray(mesh.vertices, dtype=np.float64)
            n = min(len(loops), a.dump_loops_top)
            print(f"\n  边界环 {len(loops)} 个，导出最大的 {n} 个到 {a.dump_loops}/：")
            for i, lp in enumerate(loops[:n]):
                p = V[lp]
                c, sz = p.mean(axis=0), p.max(axis=0) - p.min(axis=0)
                fn = f"loop_{i:02d}_n{len(lp)}.ply"
                trimesh.PointCloud(p).export(os.path.join(a.dump_loops, fn))
                print(f"    {fn:26s} 质心[{c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}]  "
                      f"bbox[{sz[0]:.3f},{sz[1]:.3f},{sz[2]:.3f}]")
            print(f"  → 用 CloudCompare 把 mesh + 这些 .ply 一起打开，"
                  f"看环是绕在镂空边缘（该留）还是横在表面缺块上（该补）")
            continue

        # ── 可选：补洞 / 降面 ──
        if a.repair:
            mesh, _ = repair_holes(mesh, max_loop_len=a.max_loop,
                                   smooth_iters=a.smooth_iters)
        if a.decimate > 0:
            mesh = decimate_mesh(mesh, a.decimate)

        st = mesh_stats_from_mesh(mesh, f"{name}:输出")
        out_stats.append(st)
        print("[输出]" + fmt(st))

        out_path = (os.path.join(out_dir, os.path.splitext(name)[0] + ".obj")
                    if out_dir else a.out)
        mesh.export(out_path)
        print(f"  → 已写出 {out_path}")

    # ── 批量汇总对比（点云输入时没有「输入」列，只列输出）──
    if len(out_stats) > 1:
        af = quality_summary(out_stats)
        b = quality_summary(in_stats) if in_stats else {}
        if af:
            print(f"\n{'=' * 72}\n汇总（{af['num_evaluated']} 个）\n{'=' * 72}")
            print(f"  {'指标':<20}{'输入':>14}{'输出':>14}")
            if b:
                print(f"  {'watertight 比例':<17}{b['watertight_rate'] * 100:>13.0f}%"
                      f"{af['watertight_rate'] * 100:>13.0f}%")
            for k in ("holes", "boundary_edges", "components", "non_manifold_edges",
                      "dihedral_mean"):
                bs = f"{b[k]['mean']:>14.2f}" if b else f"{'-':>14}"
                print(f"  {k:<20}{bs}{af[k]['mean']:>14.2f}")
            print(f"  {'faces':<20}{'':>14}"
                  f"{np.mean([x['faces'] for x in out_stats if 'faces' in x]):>14.0f}")


if __name__ == "__main__":
    main()
