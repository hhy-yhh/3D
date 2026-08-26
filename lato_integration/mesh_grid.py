"""
================================================================================
mesh_grid.py — 用 LATO 拓扑（格点相邻连边 + 四边形化）把解码顶点云变成干净 mesh
================================================================================

替换 inference_lato.py / evaluate_3d_metrics.py 里的 KDTree+三角枚举建 mesh。
后者在 512³ 上对 20 万顶点 KDTree(32近邻) 乱连边 + 公共邻居枚举三角，
必然产出上千万细小三角汤（"没有完整的面、全是细小三角形"）。

本模块的思路（对应用户目标"用 LATO 拓扑"）：
  1. 只在 512³ 格点上连 6-邻域（正交距离 1）的候选边 → 不跨层/跨点乱连
  2. ConnectionHead 打分过滤（LATO 连接头本来就是学"相邻顶点是否相连"）
  3. 4 顶点闭合的格点正方形 → quad → 2 三角 → 完整的面

产出：几十万面（而非千万），面完整、规整四边形。
若格点建 mesh 失败/过稀，调用方回退到 KDTree。
================================================================================
"""

import numpy as np
import torch


def predict_edges_grid(connection_head, vertex_coords_int, vertex_feats, threshold=0.45,
                       device=None, batch_size=8192):
    """格点 6-邻域候选边 + ConnectionHead 打分。

    Args:
        connection_head: LATO ConnectionHead（fp32）。
        vertex_coords_int: [N,3] int64 格点坐标（512³ 网格）。
        vertex_feats: [N, feat_dim] 顶点特征（fp32）。
        threshold: 边保留阈值。
        batch_size: ConnectionHead 打分批大小。
    Returns:
        list[(int, int)] 边（u<v）。为空则返回 []。
    """
    if device is None:
        device = vertex_feats.device
    N = vertex_coords_int.shape[0]
    if N < 2:
        return []

    # 1. 哈希：coord → index（spconv 坐标是整数格点，无重复）
    coords_np = vertex_coords_int.cpu().numpy()
    coord_to_idx = {}
    for i in range(N):
        coord_to_idx[(int(coords_np[i, 0]), int(coords_np[i, 1]), int(coords_np[i, 2]))] = i

    # 2. 候选边：+x/+y/+z 正交邻居，每对只生成一次（u<v）
    candidates = []
    for i in range(N):
        cx, cy, cz = int(coords_np[i, 0]), int(coords_np[i, 1]), int(coords_np[i, 2])
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            j = coord_to_idx.get((cx + d[0], cy + d[1], cz + d[2]))
            if j is not None:
                candidates.append((i, j))
    if not candidates:
        return []

    # 3. ConnectionHead 打分（与 predict_edges_batched 同款逻辑）
    u_list = [c[0] for c in candidates]
    v_list = [c[1] for c in candidates]
    connection_head = connection_head.to(device)
    connection_head.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            uu = torch.tensor(u_list[start:end], device=device)
            vv = torch.tensor(v_list[start:end], device=device)
            fu = vertex_feats[uu].float()
            fv = vertex_feats[vv].float()
            logit_uv = connection_head(torch.cat([fu, fv], dim=-1))
            logit_vu = connection_head(torch.cat([fv, fu], dim=-1))
            prob = torch.sigmoid(logit_uv + logit_vu).squeeze(-1)
            probs.append(prob.cpu())
    probs = torch.cat(probs)
    mask = probs > threshold
    edges = [(u_list[i], v_list[i]) for i in range(len(candidates)) if mask[i].item()]
    return edges


def build_mesh_from_grid(vertex_coords_int, vertex_feats, connection_head, device,
                         last_res=512, edge_threshold=0.45, batch_size=8192):
    """格点拓扑建 mesh：6-邻域边 → ConnectionHead 打分 → 四边形化 → 三角面。

    Args:
        vertex_coords_int: [N,3] int64 格点坐标。
        vertex_feats: [N, feat_dim] 顶点特征。
        connection_head: LATO ConnectionHead。
        last_res: 输出分辨率（坐标除以它再 -0.5 归一化，通常 = 512）。
        edge_threshold: 边保留阈值。
        batch_size: ConnectionHead 打分批大小。
    Returns:
        trimesh.Trimesh（顶点归一化到 [-0.5, 0.5]），失败返回 None。
    """
    import trimesh

    N = vertex_coords_int.shape[0]
    if N < 3:
        print("[mesh_grid] 顶点不足 3 个")
        return None

    edges = predict_edges_grid(connection_head, vertex_coords_int, vertex_feats,
                               threshold=edge_threshold, device=device, batch_size=batch_size)
    if not edges:
        print("[mesh_grid] 格点边预测为空")
        return None
    print(f"[mesh_grid] 格点候选边 {len(edges)} 条")

    edge_set = set(edges)
    coords_np = vertex_coords_int.cpu().numpy()
    coord_to_idx = {
        (int(coords_np[i, 0]), int(coords_np[i, 1]), int(coords_np[i, 2])): i
        for i in range(N)
    }

    def has_edge(a, b):
        if a > b:
            a, b = b, a
        return (a, b) in edge_set

    # 四边形化：每个顶点沿 3 个轴对找闭合的格点正方形
    quads_ordered = {}
    for i in range(N):
        cx, cy, cz = int(coords_np[i, 0]), int(coords_np[i, 1]), int(coords_np[i, 2])
        for d1, d2 in (((1, 0, 0), (0, 1, 0)),
                       ((1, 0, 0), (0, 0, 1)),
                       ((0, 1, 0), (0, 0, 1))):
            j1 = coord_to_idx.get((cx + d1[0], cy + d1[1], cz + d1[2]))
            j2 = coord_to_idx.get((cx + d2[0], cy + d2[1], cz + d2[2]))
            j12 = coord_to_idx.get((cx + d1[0] + d2[0], cy + d1[1] + d2[1], cz + d1[2] + d2[2]))
            if j1 is None or j2 is None or j12 is None:
                continue
            if has_edge(i, j1) and has_edge(i, j2) and has_edge(j1, j12) and has_edge(j2, j12):
                key = tuple(sorted((i, j1, j12, j2)))
                if key not in quads_ordered:
                    quads_ordered[key] = (i, j1, j12, j2)

    if not quads_ordered:
        print("[mesh_grid] 未找到闭合四边形，无法建 mesh")
        return None

    faces = []
    for q in quads_ordered.values():
        a, b, c, d = q  # 四边形按顺序 (i, j1, j12, j2)
        faces.append([a, b, c])
        faces.append([a, c, d])

    faces = np.array(faces, dtype=np.int64)
    vertices = coords_np.astype(np.float64) / float(last_res) - 0.5
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    print(f"[mesh_grid] 格点建 mesh: v={len(mesh.vertices)} f={len(mesh.faces)}")
    return mesh
