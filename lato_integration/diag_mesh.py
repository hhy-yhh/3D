#!/usr/bin/env python
"""diag_mesh.py — 网格客观诊断（洞 / 碎块 / 非流形 / 粗糙度）

对生成的 mesh 或 GT mesh 输出一组可对比的客观指标，用于定位「孔洞多 / 表面粗糙」的根因。

用法:
    python lato_integration/diag_mesh.py a.obj b.stl ...
    python lato_integration/diag_mesh.py outputs/eval_x/meshes/            # 目录→自动收集
    python lato_integration/diag_mesh.py outputs/eval_x/meshes/ --json out.json
    python lato_integration/diag_mesh.py outputs/eval_x/meshes/*.obj

指标含义:
    watertight          是否封闭（无边界边）
    components          连通块数（>1 = 有游离碎块）
    boundary_edges      边界边数（= 洞的边；0 = 无洞）
    holes / largest_loops  边界环数 / 最大的几个环（顶点数）
    non_manifold_edges  非流形边数（>2 个面共享）
    dihedral_deg        相邻面二面角（度）；越小越光滑，和 GT 对比才有意义
    edge_len            边长分布（std/mean 大 = 面大小不均 = 「碎」感）

依赖: trimesh, numpy（与项目一致，无需额外安装）
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import trimesh

EXTS = (".obj", ".stl", ".ply", ".glb", ".gltf", ".off")


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for e in EXTS:
                files += sorted(glob.glob(os.path.join(p, "*" + e)))
        elif any(ch in p for ch in "*?["):
            files += sorted(glob.glob(p))
        elif os.path.exists(p):
            files.append(p)
        else:
            print(f"[WARN] 不存在: {p}", file=sys.stderr)
    # 去重保序
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def mesh_stats(path):
    try:
        m = trimesh.load(path, force="mesh", process=True)
    except Exception as e:
        return {"file": path, "error": f"load failed: {e}"}
    if m is None or len(getattr(m, "faces", [])) == 0:
        return {"file": path, "error": "empty mesh"}
    try:
        m.merge_vertices()
    except Exception:
        pass

    edges = m.edges_sorted
    u, c = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = u[c == 1]      # 洞的边
    non_manifold = u[c > 2]         # 非流形边

    # 边界环（洞）：边界边组成的连通分量
    adj = defaultdict(set)
    for a, b in boundary_edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, holes = set(), []
    for v in list(adj):
        if v in seen:
            continue
        stack, n = [v], 0
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            n += 1
            stack.extend(adj[x] - seen)
        holes.append(n)
    holes.sort(reverse=True)

    comps = m.split(only_watertight=False)
    largest_f = max((len(x.faces) for x in comps), default=0)

    if len(m.face_adjacency):
        ang = np.degrees(m.face_adjacency_angles)
    else:
        ang = np.array([0.0])
    el = m.edges_unique_length
    el = el if len(el) else np.array([0.0])

    return {
        "file": path,
        "vertices": int(len(m.vertices)),
        "faces": int(len(m.faces)),
        "watertight": bool(m.is_watertight),
        "winding_ok": bool(m.is_winding_consistent),
        "components": int(len(comps)),
        "largest_component_faces": int(largest_f),
        "boundary_edges": int(len(boundary_edges)),
        "non_manifold_edges": int(len(non_manifold)),
        "holes": int(len(holes)),
        "largest_loops": [int(x) for x in holes[:8]],
        "euler": int(len(m.vertices) - len(u) + len(m.faces)),
        "dihedral_mean": float(ang.mean()),
        "dihedral_p50": float(np.percentile(ang, 50)),
        "dihedral_p90": float(np.percentile(ang, 90)),
        "dihedral_p99": float(np.percentile(ang, 99)),
        "edge_len_mean": float(el.mean()),
        "edge_len_std": float(el.std()),
        "volume": float(m.volume),
        "bounds": np.round(m.bounds, 3).tolist(),
    }


def fmt(r):
    if "error" in r:
        return f"\n=== {r['file']}\n  [ERROR] {r['error']}"
    return (
        f"\n=== {r['file']}\n"
        f"  v={r['vertices']} f={r['faces']}  watertight={r['watertight']} winding_ok={r['winding_ok']}\n"
        f"  components={r['components']} (largest_f={r['largest_component_faces']})  euler={r['euler']}\n"
        f"  boundary_edges={r['boundary_edges']}  non_manifold_edges={r['non_manifold_edges']}\n"
        f"  holes={r['holes']}  largest_loops(verts)={r['largest_loops']}\n"
        f"  dihedral_deg: mean={r['dihedral_mean']:.1f} p50={r['dihedral_p50']:.1f} "
        f"p90={r['dihedral_p90']:.1f} p99={r['dihedral_p99']:.1f}\n"
        f"  edge_len: mean={r['edge_len_mean']:.4f} std={r['edge_len_std']:.4f}\n"
        f"  volume={r['volume']:.4f} bounds={r['bounds']}"
    )


def main():
    ap = argparse.ArgumentParser(description="网格客观诊断（洞/碎块/非流形/粗糙度）")
    ap.add_argument("paths", nargs="+", help="mesh 文件 / 目录 / glob")
    ap.add_argument("--json", default=None, help="把结果追加写入该 json（便于多次对比）")
    a = ap.parse_args()

    files = collect(a.paths)
    if not files:
        print("没有找到 mesh 文件", file=sys.stderr)
        sys.exit(1)

    results = [mesh_stats(p) for p in files]
    for r in results:
        print(fmt(r))

    ok = [r for r in results if "error" not in r]
    if len(ok) > 1:
        def arr(k):
            return np.array([r[k] for r in ok], dtype=float)
        print("\n---- 聚合 (%d 个有效) ----" % len(ok))
        for k in ("faces", "components", "boundary_edges", "non_manifold_edges",
                  "holes", "dihedral_mean", "dihedral_p90"):
            v = arr(k)
            print(f"  {k:22s} mean={v.mean():.2f} min={v.min():.2f} max={v.max():.2f}")

    if a.json:
        try:
            old = json.load(open(a.json)) if os.path.exists(a.json) else []
        except Exception:
            old = []
        json.dump(old + results, open(a.json, "w"), indent=2, default=float)
        print(f"\n已写入 {a.json}")


if __name__ == "__main__":
    main()
