#!/usr/bin/env python
"""try_mc.py — 用 marching cubes 从点云/网格重建 mesh（独立试验脚本，不改主流程）

目的：给"MC 到底行不行"一个可跑、可对比的答案。
输入是**离散点云**（MC 本身需要标量场，所以这里先从点云造场再提取等值面）。

两种造场方式（--mode）：
  occ  : 点体素化成 occupancy → 高斯平滑 → marching cubes。
         最"经典"的 MC，会封住内部空腔（和 Poisson 一样会填内部），表面偏方块。
  dist : 占据格点做欧氏距离变换(EDT) → 在距离=alpha 处提等值面。
         得到"贴着点云的等值面"，更容易保住凹腔/通孔；可能产生薄壳/双面。

用法：
    # 从生成网格采样点云来重建（零改动、先看效果）
    python lato_integration/try_mc.py outputs/eval_crop0_raw/meshes/xxx.obj \
        --mode occ --res 256 --sigma 1.5 --out outputs/mc_test/occ.obj

    python lato_integration/try_mc.py outputs/eval_crop0_raw/meshes/xxx.obj \
        --mode dist --res 256 --alpha 2.0 --out outputs/mc_test/dist.obj

    # 直接给解码点云 npz（含 coords [N,4] 或 [N,3]）
    python lato_integration/try_mc.py coords.npz --mode dist --res 256 --alpha 2.0 --out out.obj

跑完用 diag_mesh.py 对比：
    python lato_integration/diag_mesh.py outputs/mc_test/occ.obj outputs/mc_test/dist.obj

依赖：numpy, trimesh, scipy(ndimage), scikit-image(marching_cubes)
      缺则：pip install scikit-image scipy
"""
import argparse
import os
import sys

import numpy as np
import trimesh


def load_points(path, n_sample=300000, seed=0):
    """从网格采样点云，或直接读 npz 里的 coords/points。返回 [N,3] float64。"""
    if path.endswith(".npz"):
        d = np.load(path)
        for k in ("coords", "points", "vertex_coords", "v", "vertices"):
            if k in d:
                a = np.asarray(d[k], dtype=np.float64)
                return a[:, -3:] if a.shape[1] >= 3 else a
        raise ValueError(f"npz 里找不到点云键，实际键: {list(d.keys())}")

    m = trimesh.load(path, force="mesh", process=True)
    if m is None or len(m.faces) == 0:
        raise ValueError(f"无法从 {path} 载入网格")
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(m, n_sample, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def to_grid(points, res, margin=0.02):
    """把点云体素化到 res^3 的 0/1 网格，返回 (occ, lo, hi)。"""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = hi - lo
    span[span <= 0] = 1.0
    lo = lo - span * margin
    hi = hi + span * margin
    span = hi - lo

    occ = np.zeros((res, res, res), dtype=np.float32)
    idx = np.floor((points - lo) / span * res).astype(np.int64)
    np.clip(idx, 0, res - 1, out=idx)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return occ, lo, hi


def grid_to_world(verts_idx, lo, hi, res):
    """MC 顶点在 grid 索引空间 [0,res-1] → 世界坐标。"""
    span = hi - lo
    return lo + (verts_idx + 0.5) / res * span


def main():
    ap = argparse.ArgumentParser(description="marching cubes 试验重建（独立脚本）")
    ap.add_argument("input", help="输入：网格(.obj/.stl/.ply) 或 点云(.npz)")
    ap.add_argument("--mode", choices=["occ", "dist"], default="occ")
    ap.add_argument("--res", type=int, default=256, help="体素分辨率（256 起，越大越慢越占内存）")
    ap.add_argument("--sigma", type=float, default=1.5, help="[occ] occupancy 高斯平滑强度")
    ap.add_argument("--level", type=float, default=0.5, help="[occ] 等值面阈值")
    ap.add_argument("--alpha", type=float, default=2.0, help="[dist] 距离等值面（单位=体素）")
    ap.add_argument("--sample", type=int, default=300000, help="从网格采样的点数")
    ap.add_argument("--out", required=True, help="输出 .obj 路径")
    a = ap.parse_args()

    pts = load_points(a.input, a.sample)
    print(f"[try_mc] 输入点云 N={len(pts)}  bounds={np.round(pts.min(0),3).tolist()}~{np.round(pts.max(0),3).tolist()}")

    occ, lo, hi = to_grid(pts, a.res)
    print(f"[try_mc] 体素化 res={a.res}  占据格点={int(occ.sum())} ({occ.mean()*100:.2f}%)")

    # 造场
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    if a.mode == "occ":
        vol = gaussian_filter(occ, sigma=a.sigma)
        level = a.level
        print(f"[try_mc] 模式 occ: gaussian(sigma={a.sigma}) → MC level={level}")
    else:
        # 距离变换：每个空格点到最近占据格点的距离（单位=体素）
        vol = distance_transform_edt(occ < 0.5)
        level = a.alpha
        print(f"[try_mc] 模式 dist: EDT → MC level(alpha)={level}")

    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        print(f"[ERR] 需要 scikit-image: {e}", file=sys.stderr)
        sys.exit(2)

    verts_idx, faces, _, _ = marching_cubes(vol, level=level)
    verts = grid_to_world(verts_idx, lo, hi, a.res)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    mesh.export(a.out)
    print(f"[try_mc] 输出 {a.out}: v={len(mesh.vertices)} f={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} components={len(mesh.split(only_watertight=False))}")
    print(f"[try_mc] 对比: python lato_integration/diag_mesh.py {a.out}")


if __name__ == "__main__":
    main()
