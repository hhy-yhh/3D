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


def build_mesh_from_poisson(vertex_coords_int, device, last_res=512,
                            depth=9, knn=30, crop_density_quantile=0.1,
                            smooth_iterations=2, smooth_lambda=0.5):
    """用 open3d Poisson 从解码顶点云重建光滑、水密、流形曲面。

    绕开「512³ 格点四边形」的方块感/非流形问题，直接从点云光滑重建。
    几何主体不变（CD 基本保持），观感从「方块」变「光滑卡钳」。

    Args:
        vertex_coords_int: [N,3] int64 格点坐标（512³ 网格）。
        device: 设备（仅用于日志，Poisson 在 CPU 跑）。
        last_res: 输出分辨率（归一化用，通常 512）。
        depth: Poisson 八叉树深度（~200K 点用 9-10）。
        knn: 法线估计近邻数。
        crop_density_quantile: 裁剪掉密度低于该分位数的顶点（去掉 Poisson 膨胀部分）。
    Returns:
        trimesh.Trimesh（顶点归一化到 [-0.5,0.5]），失败返回 None。
    """
    import open3d as o3d
    import trimesh
    import numpy as np

    N = vertex_coords_int.shape[0]
    if N < 10:
        print("[mesh_grid] 顶点过少，跳过 Poisson")
        return None

    coords_np = vertex_coords_int.cpu().numpy().astype(np.float64)
    pts = coords_np / float(last_res) - 0.5  # 归一化到 [-0.5, 0.5]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 1. 估计法线（PCA）
    k = min(knn, N - 1)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    # 2. 统一法线方向：远离质心（对居中形状近似 outward）
    center = pts.mean(axis=0)
    norms = np.asarray(pcd.normals).copy()
    dot = np.sum(norms * (pts - center), axis=1)
    norms[dot < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(norms)

    print(f"[mesh_grid] Poisson: {N} 点, depth={depth}, k={k}")
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # 3. 裁剪低密度膨胀部分（Poisson 常带一个向外鼓的气球）
    densities = np.asarray(densities)
    thresh = np.quantile(densities, crop_density_quantile)
    mesh_o3d.remove_vertices_by_mask(densities < thresh)

    verts = np.asarray(mesh_o3d.vertices)
    tris = np.asarray(mesh_o3d.triangles)
    if len(tris) == 0:
        print("[mesh_grid] Poisson 输出空面")
        return None
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    # Poisson 输出是流形网格 → 平滑安全（不会像非流形 grid 网格那样炸尖刺），
    # 抹平「纸糊」的细碎小面
    mesh = _postprocess_mesh(mesh, smooth_iterations=smooth_iterations, smooth_lambda=smooth_lambda)
    print(f"[mesh_grid] Poisson 重建: v={len(mesh.vertices)} f={len(mesh.faces)}")
    return mesh


def _postprocess_mesh(mesh, smooth_iterations=2, smooth_lambda=0.5):
    """修复法线 + 轻量 Laplacian 平滑。

    - fix_normals：统一面的绕向/法线，消除黑面、破碎感（NC 偏低的主因）。
    - 轻量平滑：把 512³ 格点阶梯抹平，表面从「方块」变「光滑」，几何主体不变（CD 几乎不变）。
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh
    try:
        import trimesh
        trimesh.repair.fix_normals(mesh)
        if smooth_iterations > 0:
            from trimesh.smoothing import filter_laplacian
            filter_laplacian(mesh, iterations=smooth_iterations, lamb=smooth_lambda)
        print(f"[mesh_grid] 后处理: fix_normals + 平滑({smooth_iterations}次, λ={smooth_lambda})")
    except Exception as e:
        print(f"[mesh_grid] 后处理跳过: {e}")
    return mesh


def build_mesh_from_grid(vertex_coords_int, vertex_feats, connection_head, device,
                         last_res=512, edge_threshold=0.45, batch_size=8192,
                         smooth_iterations=0, smooth_lambda=0.5):
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
    # 后处理：修复法线 + 轻量平滑（改善观感，几何主体不变）
    mesh = _postprocess_mesh(mesh, smooth_iterations=smooth_iterations, smooth_lambda=smooth_lambda)
    return mesh


def fill_mesh_holes(mesh, max_hole_size=None):
    """用 trimesh.repair.fill_holes 补洞（作用于最终 mesh，可选）。

    Args:
        mesh: trimesh.Trimesh。
        max_hole_size: trimesh 补洞的孔大小上限（None=默认）。
    Returns:
        trimesh.Trimesh（失败/无需补时原样返回）。
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh
    try:
        import trimesh
        before = len(mesh.faces)
        # 兼容不同 trimesh 版本：新版支持 max_hole_size，旧版不支持
        try:
            trimesh.repair.fill_holes(mesh, max_hole_size=max_hole_size)
        except TypeError:
            trimesh.repair.fill_holes(mesh)
        print(f"[mesh_grid] 补洞: faces {before} → {len(mesh.faces)}")
    except Exception as e:
        print(f"[mesh_grid] 补洞跳过: {e}")
    return mesh


# ============================================================================
# v25 新增：占据格点边界提取 —— 闭合体积 + 保内部空腔
# ============================================================================
#
# 动机（对应修改安排 P1「repair holes → achieve a closed volume」）：
#   Poisson 解的是指示函数，数学上必然是封闭曲面 → 内部通孔/空腔被填死；
#   降面（1.19M → 10 万）又会把水密的 Poisson 输出砍出 256 个洞、粗糙度 ×3。
#
# 本模块换一条路：decode 输出的 coords 本来就是 512³ 整数格点 = 体素，
#   直接提取「占据 ↔ 非占据」之间的面。集合的边界在数学上恒为闭合曲面，
#   且外表面与每个内部空腔的内表面会同时生成 → 闭合体积 + 内部空腔天然保留。
#
#   --cavity fill 则反过来：先把空腔填实再提边界，得到只有外表面的封闭体
#   （等价于 Poisson 的行为，用于对比）。

# 6 个方向的面 → 该面上 4 个角点相对体素 (i,j,k) 的偏移
_QUAD_OFFSETS = {
    (1, 0, 0): ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
    (-1, 0, 0): ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    (0, 1, 0): ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
    (0, -1, 0): ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    (0, 0, 1): ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
    (0, 0, -1): ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
}


def _voxel_boundary_faces(solid, lo):
    """从 0/1 体素集合提取边界面（占据 ↔ 非占据 之间的面）。

    Args:
        solid: bool ndarray [nx,ny,nz]。
        lo: 体素 (0,0,0) 对应的全局格点坐标 [3]。
    Returns:
        (vertices [V,3] float64 全局格点坐标, faces [F,3] int64)，格式为角点索引。
    """
    import numpy as np

    nx, ny, nz = solid.shape
    sp = np.pad(solid, 1, constant_values=False)
    core = sp[1:-1, 1:-1, 1:-1]
    # 角点格点维度 = 体素数 + 1
    cy_n, cz_n = ny + 1, nz + 1

    quad_ids = []
    for d, offs in _QUAD_OFFSETS.items():
        dx, dy, dz = d
        nb = sp[1 + dx: 1 + dx + nx, 1 + dy: 1 + dy + ny, 1 + dz: 1 + dz + nz]
        m = core & ~nb
        if not m.any():
            continue
        v = np.argwhere(m)  # [K,3]
        base = v + lo  # 体素的全局格点坐标
        ids = np.empty((len(v), 4), dtype=np.int64)
        for c, (ox, oy, oz) in enumerate(offs):
            ids[:, c] = ((base[:, 0] + ox) * cy_n + (base[:, 1] + oy)) * cz_n + (base[:, 2] + oz)
        quad_ids.append(ids)

    if not quad_ids:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    quad_ids = np.concatenate(quad_ids, axis=0)
    # 只保留用到的角点（全格点阵可能有上千万，必须稀疏化）
    uniq, inv = np.unique(quad_ids.ravel(), return_inverse=True)
    quads = inv.reshape(-1, 4)

    cz = uniq % cz_n
    cy = (uniq // cz_n) % cy_n
    cx = uniq // (cz_n * cy_n)
    verts = np.stack([cx, cy, cz], axis=1).astype(np.float64)  # 已是全局格点坐标

    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
    return verts, faces


def build_mesh_from_voxel(vertex_coords_int, last_res=512, cavity="keep",
                          min_component_voxels=0, smooth_iterations=0,
                          smooth_lambda=0.5, target_faces=0):
    """占据格点边界提取：闭合体积 + 保内部空腔（Poisson 做不到）。

    Args:
        vertex_coords_int: [N,3] int64 格点坐标（512³ 网格）。
        last_res: 输出分辨率（坐标除以它再 -0.5 归一化，通常 = 512）。
        cavity: "keep" = 保留内部通孔/空腔（提取集合边界，含空腔内表面）；
                "fill" = 填实空腔，只留外表面（等价 Poisson 的封闭行为）。
        min_component_voxels: >0 时丢掉小于该体素数的连通碎块。
        smooth_iterations: Laplacian 平滑次数（闭合流形上安全）。
        smooth_lambda: 平滑强度。
        target_faces: >0 时 quadric 降面到目标面数。
    Returns:
        trimesh.Trimesh（顶点归一化到 [-0.5,0.5]），失败返回 None。
    """
    import numpy as np
    import trimesh

    coords = np.asarray(vertex_coords_int.detach().cpu() if hasattr(vertex_coords_int, "detach")
                        else vertex_coords_int, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[0] < 4:
        print("[mesh_grid] 体素模式：顶点不足")
        return None
    coords = np.unique(coords, axis=0)

    lo = coords.min(axis=0) - 1
    hi = coords.max(axis=0) + 1
    lo = np.clip(lo, 0, last_res - 1)
    hi = np.clip(hi, 0, last_res - 1)
    shape = tuple((hi - lo + 1).tolist())

    solid = np.zeros(shape, dtype=bool)
    idx = coords - lo
    solid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    print(f"[mesh_grid] 体素模式: {len(coords)} 个占用体素, bbox={shape}, "
          f"cavity={cavity}, 填充率={solid.mean() * 100:.2f}%")

    if min_component_voxels > 0:
        solid = _drop_small_components(solid, min_component_voxels)
        print(f"[mesh_grid]   去碎块(<{min_component_voxels}): 剩 {int(solid.sum())} 体素")

    if cavity == "fill":
        solid = _fill_cavities(solid)

    verts, faces = _voxel_boundary_faces(solid, lo)
    if len(faces) == 0:
        print("[mesh_grid] 体素边界为空")
        return None

    verts = verts / float(last_res) - 0.5
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    print(f"[mesh_grid] 体素边界: v={len(mesh.vertices)} f={len(mesh.faces)} "
          f"watertight={mesh.is_watertight}")

    mesh = _postprocess_mesh(mesh, smooth_iterations=smooth_iterations,
                             smooth_lambda=smooth_lambda)
    if target_faces and target_faces > 0:
        mesh = decimate_mesh(mesh, target_faces)
    return mesh


def _drop_small_components(solid, min_voxels):
    """丢掉小于 min_voxels 的 26-邻域连通碎块。scipy 缺失时原样返回。"""
    import numpy as np
    try:
        from scipy import ndimage
    except Exception as e:
        print(f"[mesh_grid] 去碎块跳过（缺 scipy: {e}）")
        return solid
    st = np.ones((3, 3, 3), dtype=bool)
    lab, n = ndimage.label(solid, structure=st)
    if n <= 1:
        return solid
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.nonzero(sizes >= min_voxels)[0]
    print(f"[mesh_grid]   连通块 {n} → 保留 {len(keep)}")
    return np.isin(lab, keep)


def _fill_cavities(solid):
    """把不与外界连通的空腔填实（6-邻域连通判定）。scipy 缺失时原样返回。"""
    import numpy as np
    try:
        from scipy import ndimage
    except Exception as e:
        print(f"[mesh_grid] 填腔跳过（缺 scipy: {e}）")
        return solid
    sp = np.pad(solid, 1, constant_values=False)
    # 空腔 = 不与边缘连通的非占据体素
    st = ndimage.generate_binary_structure(3, 1)  # 6-邻域，与面邻接一致
    lab, n = ndimage.label(~sp, structure=st)
    outside = lab[0, 0, 0]
    filled = sp | ((lab != outside) & (lab != 0))
    print(f"[mesh_grid]   填腔: 空腔块 {n - 1} 个 → 封闭体")
    return filled[1:-1, 1:-1, 1:-1]


# ============================================================================
# v25 新增：邻接面插值补洞（repair holes → closed volume）
# ============================================================================


def boundary_loops(mesh):
    """找出所有边界环（只属于 1 个面的边串成的环）。

    Returns:
        list[list[int]]：每个环的顶点索引（首尾不重复）。
    """
    import numpy as np

    if mesh is None or len(mesh.faces) == 0:
        return []
    edges = mesh.edges_sorted
    u, c = np.unique(edges, axis=0, return_counts=True)
    bnd = u[c == 1]
    if len(bnd) == 0:
        return []

    adj = {}
    for a, b in bnd:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    loops, seen = [], set()
    for start in list(adj):
        if start in seen:
            continue
        loop, prev, cur = [], None, start
        while cur is not None and cur not in seen:
            seen.add(cur)
            loop.append(cur)
            nxt = [x for x in adj[cur] if x != prev]
            prev, cur = cur, (nxt[0] if nxt else None)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _earclip(poly):
    """简单多边形耳切三角化。poly: [M,2]；返回 [(i,j,k)] 索引三角形。"""
    import numpy as np

    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n))
    # 保证逆时针
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    if area < 0:
        idx.reverse()

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def inside(p, a, b, c):
        d1 = cross(a, b, p)
        d2 = cross(b, c, p)
        d3 = cross(c, a, p)
        return (d1 >= 0 and d2 >= 0 and d3 >= 0)

    tris, guard = [], 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        ear = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 0:  # 凹顶点
                continue
            if any(m not in (i0, i1, i2) and inside(poly[m], a, b, c) for m in idx):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            ear = True
            break
        if not ear:
            return []  # 非简单多边形 → 交给调用方回退
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def repair_holes(mesh, max_loop_len=0, smooth_iters=0):
    """邻接面插值补洞：对每个边界环用相邻面的法线拟合平面，再在平面上插值补面。

    与 trimesh.repair.fill_holes 的区别：后者用环的质心做扇形填充（不管相邻面的
    走向），这里用**相邻面的面积加权法线**决定补面所在平面，环顶点先投影到该平面
    再耳切三角化，补出来的面延续周围曲面的走向，不产生尖刺。

    Args:
        mesh: trimesh.Trimesh。
        max_loop_len: >0 时只补顶点数 ≤ 该值的环（防大洞被硬填成平板）。
        smooth_iters: 补完后对补出的顶点做几次 Laplacian 平滑（0=不做）。
    Returns:
        (mesh, stats dict)
    """
    import numpy as np
    import trimesh

    stats = {"loops": 0, "filled": 0, "skipped": 0, "added_faces": 0}
    if mesh is None or len(mesh.faces) == 0:
        return mesh, stats

    mesh = mesh.copy()
    mesh.merge_vertices()
    loops = boundary_loops(mesh)
    stats["loops"] = len(loops)
    if not loops:
        print("[mesh_grid] 补洞: 无边界环（已闭合）")
        return mesh, stats

    fnorm = np.asarray(mesh.face_normals, dtype=np.float64)
    farea = np.asarray(mesh.area_faces, dtype=np.float64)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    tri_all = np.asarray(mesh.faces, dtype=np.int64)

    # 顶点 → 相邻面的面积加权法线（bincount 向量化；百万面用 Python 循环要几分钟）
    contrib = farea[:, None] * fnorm  # [F,3]
    nv = len(V)
    vnorm = np.zeros((nv, 3), dtype=np.float64)
    for c in range(3):
        for k in range(3):
            vnorm[:, k] += np.bincount(tri_all[:, c], weights=contrib[:, k], minlength=nv)
    nz = np.linalg.norm(vnorm, axis=1)
    nz[nz == 0] = 1.0
    vnorm /= nz[:, None]

    new_verts = list(V)
    new_faces = []
    for loop in loops:
        if max_loop_len and len(loop) > max_loop_len:
            stats["skipped"] += 1
            continue
        pts = V[loop]
        n = vnorm[loop].mean(axis=0)
        ln = np.linalg.norm(n)
        if ln < 1e-8:  # 法线互相抵消 → 用环的最小二乘平面
            c = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - c)
            n = vt[-1]
        else:
            n = n / ln
        c = pts.mean(axis=0)

        # 环顶点投影到拟合平面
        d = ((pts - c) @ n)[:, None]
        proj = pts - d * n

        # 平面内 2D 基
        a = np.array([1.0, 0.0, 0.0])
        if abs(n @ a) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        poly = np.stack([(proj - c) @ u, (proj - c) @ v], axis=1)

        tris = _earclip(poly)
        if len(tris) != len(loop) - 2:
            # 耳切失败（非简单环）→ 用「投影到拟合平面的质心」扇形填充
            center_idx = len(new_verts)
            new_verts.append(c)
            tris_abs = [(loop[i], loop[(i + 1) % len(loop)], center_idx)
                        for i in range(len(loop))]
        else:
            # 环顶点投影到拟合平面，补面才会延续相邻曲面的走向
            for i, vi in enumerate(loop):
                new_verts[vi] = proj[i]
            tris_abs = [(loop[i], loop[j], loop[k]) for (i, j, k) in tris]

        new_faces.extend(tris_abs)
        stats["filled"] += 1

    if not new_faces:
        print(f"[mesh_grid] 补洞: 边界环 {stats['loops']} 个，全部超过 max_loop_len={max_loop_len} 未补")
        return mesh, stats

    # 原面 + 补面拼回去。漏掉这一步就会把整个原网格丢掉，只剩补丁。
    all_faces = np.concatenate([tri_all, np.asarray(new_faces, dtype=np.int64)], axis=0)
    out = trimesh.Trimesh(vertices=np.asarray(new_verts, dtype=np.float64),
                          faces=all_faces, process=False)
    # 不调 merge_vertices：V/tri_all 进函数时已经 merge 过，补面又按索引引用原顶点，
    # 再 merge 一次只会把几何重合但拓扑不同的顶点并掉，反而制造新的非流形边。
    out.remove_unreferenced_vertices()
    # remove_degenerate_faces 只有较新 trimesh 才有；旧版走 update_faces(nondegenerate_faces())
    try:
        out.remove_degenerate_faces()
    except AttributeError:
        try:
            out.update_faces(out.nondegenerate_faces())
        except Exception as e:
            print(f"[mesh_grid] 去退化面跳过: {e}")
    trimesh.repair.fix_normals(out)

    # 安全阀：补洞只该加面，不该减面。少了说明面索引错位 → 宁可原样返回。
    if len(out.faces) < len(mesh.faces):
        print(f"[mesh_grid] ⚠️ 补洞后 {len(out.faces)} 面 < 原 {len(mesh.faces)} 面，"
              f"索引错位 → 回退原 mesh")
        return mesh, stats

    if smooth_iters > 0:
        try:
            from trimesh.smoothing import filter_laplacian
            filter_laplacian(out, iterations=smooth_iters, lamb=0.5)
        except Exception as e:
            print(f"[mesh_grid] 补洞平滑跳过: {e}")

    stats["added_faces"] = len(out.faces) - len(mesh.faces)
    print(f"[mesh_grid] 补洞: 环 {stats['loops']} 个（补 {stats['filled']} / 跳过 {stats['skipped']}），"
          f"新增 {stats['added_faces']} 面 → watertight={out.is_watertight}")
    return out, stats


def decimate_mesh(mesh, target_faces):
    """用 open3d quadric 边坍缩把 mesh 压到 ~target_faces 面。

    target_faces 大于等于当前面数时跳过（不动）；失败回退原 mesh。
    Args:
        mesh: trimesh.Trimesh。
        target_faces: 目标面数。
    Returns:
        trimesh.Trimesh。
    """
    if mesh is None or len(mesh.faces) == 0 or target_faces is None or target_faces <= 0:
        return mesh
    current = len(mesh.faces)
    if current <= target_faces:
        print(f"[mesh_grid] 降面跳过: f={current} ≤ target={target_faces}")
        return mesh
    try:
        import numpy as np
        import open3d as o3d
        import trimesh
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
            triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
        )
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(target_faces))
        verts = np.asarray(o3d_mesh.vertices)
        tris = np.asarray(o3d_mesh.triangles)
        if len(tris) == 0:
            print("[mesh_grid] 降面输出空面，回退原 mesh")
            return mesh
        print(f"[mesh_grid] 降面: v {len(mesh.vertices)}→{len(verts)}, "
              f"f {current}→{len(tris)}")
        return trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    except Exception as e:
        print(f"[mesh_grid] 降面失败回退原 mesh: {e}")
        return mesh
