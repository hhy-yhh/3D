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
    import numpy as np

    N = vertex_coords_int.shape[0]
    if N < 10:
        print("[mesh_grid] 顶点过少，跳过 Poisson")
        return None

    coords_np = vertex_coords_int.cpu().numpy().astype(np.float64)
    pts = coords_np / float(last_res) - 0.5  # 归一化到 [-0.5, 0.5]
    return poisson_from_points(
        pts, depth=depth, knn=knn, crop_density_quantile=crop_density_quantile,
        smooth_iterations=smooth_iterations, smooth_lambda=smooth_lambda,
    )


def poisson_from_points(pts, depth=9, knn=30, crop_density_quantile=0.1,
                        smooth_iterations=2, smooth_lambda=0.5):
    """直接从归一化点云（[-0.5, 0.5]）做 Poisson 重建。

    与 build_mesh_from_poisson 的区别：那个从 512³ 整数格点坐标出发，这个收现成点云。
    用途：拿已有的 Poisson 输出 mesh 的顶点当点云，换个 depth 重新重建——
    不用重跑整条生成管线（十几分钟 → 一分钟），专门用于调 depth 拿「原生少面」。

    depth 与面数的关系（实测同一样本）：9 → 1.05M 面, 8 → ~260k, 7 → ~65k, 6 → ~16k
    （GT 是 22,892 面）。降 depth 天然得到少面，且 Poisson 输出**恒为闭合流形**，
    没有洞可补——这是比「先 119 万面再砍到 10 万再补洞」正确得多的路径。

    Args:
        pts: [N,3] 归一化点云。
        depth: 八叉树深度（小 = 面少 = 更光滑）。GT 量级用 6~7。
        knn: 法线估计近邻数。
        crop_density_quantile: 密度裁剪分位（低=少裁，保留薄孔/翅片）。
        smooth_iterations / smooth_lambda: 重建后 Laplacian 平滑。
    Returns:
        trimesh.Trimesh（已归一化），失败返回 None。
    """
    import open3d as o3d
    import trimesh
    import numpy as np

    pts = np.asarray(pts, dtype=np.float64)
    N = len(pts)
    if N < 10:
        print("[mesh_grid] 点云过少，跳过 Poisson")
        return None

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


def _pcd_with_normals(pts, knn=30):
    """建 open3d 点云 + PCA 估法线 + 统一朝外（远离质心）。

    Ball Pivoting / Alpha Shape 都要法线；统一朝外这一步对凹腔形状只是近似
    （凹腔处的点其实该朝内），但 open3d 的法线定向器在没有扫描视角时也只有这种
    粗略办法，试过 orient_normals_consistent_tangent_plane（对 40 万点太慢）。
    """
    import open3d as o3d
    import numpy as np

    pts = np.asarray(pts, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    k = min(knn, len(pts) - 1)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    center = pts.mean(axis=0)
    norms = np.asarray(pcd.normals).copy()
    dot = np.sum(norms * (pts - center), axis=1)
    norms[dot < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(norms)
    return pcd, k


def _auto_scale_multiplier(pcd, mult):
    """按点云平均最近邻距把「倍率」换算成绝对半径列表。"""
    import open3d as o3d
    import numpy as np

    try:
        mean_nn = float(pcd.compute_nearest_neighbor_distance().mean())
    except Exception:
        mean_nn = None
    if not mean_nn or mean_nn <= 0:
        # 退化：用 bbox 对角线的千分之一兜底
        bb = np.asarray(pcd.points)
        mean_nn = float(np.linalg.norm(bb.max(0) - bb.min(0))) / 1000.0
    return mean_nn, [mean_nn * m for m in mult]


def build_mesh_from_points(pts, mode="ball", radii=None, alpha=None, knn=30,
                           smooth_iterations=0, smooth_lambda=0.5, target_faces=0):
    """从归一化点云 [-0.5,0.5] 重建：ball（球旋转）/ alpha（α 形状）。

    为什么需要这两条：Poisson 解指示函数 → 必然封死内部通孔；而「占据格点边界提取」
    假设点云是实心体，对**散点**（本管线的 decode 输出只占 bbox 的 ~1.6%、碎成几百块）
    会产出海绵状曲面。这两条都只连「该连的点」，不填补空间，所以能保住内部凹腔。

    Args:
        pts: [N,3] 归一化点云。
        mode: "ball" 或 "alpha"。
        radii: [ball] 球半径列表（绝对单位）。None → 按平均最近邻距 × [1.5, 3, 6] 自动推。
        alpha: [alpha] α 值（绝对单位）。None → 平均最近邻距 × 3。
        knn: 法线估计近邻数。
        smooth_iterations / smooth_lambda: 重建后 Laplacian 平滑（0=不做）。
        target_faces: >0 时降面到目标面数。
    Returns:
        (trimesh.Trimesh 或 None, info dict)
    """
    import open3d as o3d
    import trimesh
    import numpy as np

    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 10:
        print("[mesh_grid] 点云过少")
        return None, {}

    pcd, k = _pcd_with_normals(pts, knn)
    mean_nn, auto_radii = _auto_scale_multiplier(pcd, [1.5, 3.0, 6.0])
    info = {"mean_nn": mean_nn, "knn": k, "mode": mode}

    if mode == "ball":
        use_radii = [float(r) for r in radii] if radii else auto_radii
        print(f"[mesh_grid] BallPivoting: {len(pts)} 点, 平均间距={mean_nn:.5f}, "
              f"radii={[round(r, 5) for r in use_radii]}")
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(use_radii))
        info["radii"] = use_radii
    elif mode == "alpha":
        use_alpha = float(alpha) if alpha else mean_nn * 3.0
        print(f"[mesh_grid] AlphaShape: {len(pts)} 点, 平均间距={mean_nn:.5f}, alpha={use_alpha:.5f}")
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, use_alpha)
        info["alpha"] = use_alpha
    else:
        raise ValueError(f"未知 mode: {mode}")

    tris = np.asarray(mesh_o3d.triangles)
    if len(tris) == 0:
        print(f"[mesh_grid] {mode} 输出空面（半径/α 不合适，调 --ball_radii / --alpha）")
        return None, info

    verts = np.asarray(mesh_o3d.vertices)
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    mesh.remove_unreferenced_vertices()
    print(f"[mesh_grid] {mode} 重建: v={len(mesh.vertices)} f={len(mesh.faces)}")

    mesh = _postprocess_mesh(mesh, smooth_iterations=smooth_iterations,
                             smooth_lambda=smooth_lambda)
    if target_faces and target_faces > 0:
        mesh = decimate_mesh(mesh, target_faces)
    return mesh, info


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


def clean_occupancy(occ, threshold=0.0, morph_close=0, morph_open=0,
                    upsample=1, blur_sigma=0.0):
    """把 occupancy 处理成一个干净的概率场 —— 「MC 建面」和「清理后喂 SLat」共用。

    存在的理由：生成的 occupancy 是「51 块拼成的海绵」（GT 是 1 块连续壳，
    实测样本 20250423_1800_838505 / th=3.0），这既是 MC 出 230 个组件的原因，
    也很可能是下游「糊团点云」的成因。形态学清理把它压回连续壳（51 → 1）。

    Args:
        occ: [D,H,W] numpy 数组（logits 或已是 [0,1] 场）。
        threshold: 二值化阈值（logits 用 --ss_threshold；已是概率场则用 0.5）。
        morph_close / morph_open: 闭/开运算次数（26-邻域立方体结构元）。0=不做。
        upsample / blur_sigma: 上采样倍数 / 高斯模糊 σ（按原始分辨率计）。实测收益低。
    Returns:
        (场, 阈值)：处理后的 [0,1] float32 场，以及应对它用的等值面阈值。
        未做任何处理时原样返回 (occ, threshold)。
    """
    if not (morph_close > 0 or morph_open > 0 or upsample > 1 or blur_sigma > 0):
        return occ, float(threshold)

    from scipy import ndimage
    vox0 = int((occ > float(threshold)).sum())
    b = occ > float(threshold)
    if morph_close > 0 or morph_open > 0:
        st = np.ones((3, 3, 3), bool)               # 26-邻域
        n_before = ndimage.label(b, structure=st)[1]
        if morph_close > 0:
            b = ndimage.binary_closing(b, st, iterations=int(morph_close))
        if morph_open > 0:
            b = ndimage.binary_opening(b, st, iterations=int(morph_open))
        n_after = ndimage.label(b, structure=st)[1]
        print(f"[mesh_grid] 形态学清理: 闭x{morph_close} 开x{morph_open} | "
              f"连通分量 {n_before} → {n_after} | voxel {vox0} → {int(b.sum())}")

    f = b.astype(np.float32)
    if upsample > 1:
        f = ndimage.zoom(f, float(upsample), order=1)     # 三线性插值
        print(f"[mesh_grid] 上采样 x{upsample}: {occ.shape} → {f.shape}")
    if blur_sigma > 0:
        sig = float(blur_sigma) * max(int(upsample), 1)
        f = ndimage.gaussian_filter(f, sig)
        print(f"[mesh_grid] 场高斯模糊: σ={sig:.2f}（{f.shape[0]}³ 网格）")
    return f, 0.5


def mesh_from_occupancy(occ_logits, threshold=0.0, smooth_iters=None,
                        smooth_lambda=0.5, morph_close=0, morph_open=0,
                        upsample=1, blur_sigma=0.0):
    """从 occupancy logits 直接 marching cubes 建面（跳过 Poisson 和 VAE decode）。

    为什么需要这条：Poisson 拟合连续指示函数 → 数学上必然封闭 → 内部镂空被桥接填死；
    而 ball/grid/alpha 等在 decode 出的「糊团点云」上全崩。MC 对「已知的 occupancy」
    取等值面，空腔就是空腔，不存在「跨过空隙」这个动作，所以结构上不可能填死镂空，
    也不依赖点云的局部连通性。

    实测（GT occupancy，样本 20250423_1800_838505）：
      components=3（Poisson 是 770）、euler=-30（GT -6、Poisson -1979）、
      holes=0 / boundary_edges=0 / non_manifold=0 / watertight=True、
      dihedral mean=9.3°（GT 6.4、Poisson 13.0、降面版 38.0）

    代价：分辨率锁在 occupancy 的 128³，出来是体素级阶梯（dihedral p50=0、p90=45），
    需要平滑；且丢掉了 VAE decode 的细节增益。

    Args:
        occ_logits: StructureHead 的原始输出 [B,1,D,H,W] 或 [D,H,W]（未过 sigmoid）。
        threshold: 等值面的 logits 阈值，应与 --ss_threshold 一致
                   （logits > threshold 视为 occupied）。
        smooth_iters: Laplacian 平滑次数（None/0 = 不平滑）。
    Returns:
        trimesh.Trimesh 或 None
    """
    import trimesh
    from skimage import measure

    occ = occ_logits.detach().float().cpu().numpy()
    occ = np.squeeze(occ).astype(np.float32)
    if occ.ndim != 3:
        print(f"[mesh_grid] occupancy 形状异常（期望 3D）: {occ.shape}")
        return None

    # 场层面的预处理（形态学 / 上采样 / 高斯模糊）—— 与「清理后喂 SLat」共用同一套逻辑
    occ, threshold = clean_occupancy(
        occ, threshold, morph_close=morph_close, morph_open=morph_open,
        upsample=upsample, blur_sigma=blur_sigma,
    )
    mesh_res = occ.shape[0]

    verts, faces, _, _ = measure.marching_cubes(occ, level=float(threshold))
    if len(faces) == 0:
        print(f"[mesh_grid] MC 没产出面（threshold={threshold} 太高？）")
        return None

    # MC 的顶点列顺序与 occupancy 轴序一致（x, y, z），除以分辨率归一到 [-0.5, 0.5]
    verts = verts / float(mesh_res) - 0.5
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    print(f"[mesh_grid] MC 重建: v={len(mesh.vertices)} f={len(mesh.faces)}")
    # MC 输出是闭合流形（watertight），平滑安全 —— 复用与 poisson/ball/voxel 同一套后处理
    # （fix_normals 修 MC 绕向可能整体朝内的问题 + Laplacian 抹体素阶梯）
    mesh = _postprocess_mesh(mesh,
                             smooth_iterations=(smooth_iters or 0),
                             smooth_lambda=smooth_lambda)
    return mesh


def refine_mesh_with_points(mesh, pts, max_move=None, mode="normal", verbose=True):
    """用高精度点云精修网格顶点位置 —— **拓扑（连接关系）完全不变**。

    为什么需要：MC 网格的拓扑是对的（保镂空、单连通、watertight），但它的顶点只能落在
    occupancy 的 128³ 格点上，所以有体素级方块感。而 VAE.decode 出的点云是 512³ 精度的。
    把 MC 顶点朝点云表面挪，方块感就减轻 —— 这是「MC 出拓扑 + VAE 出精度」的结合点，
    比在场层面做高斯模糊更根本（模糊只是把方块磨圆，这个是把顶点拉向真实表面）。

    ⚠️ **mode="nearest" 是错的，别用**：让每个顶点各自找「最近点」，多个顶点会找到**同一个**
    点、被拉到同一位置 → 边长为 0 → 三角形退化 → trimesh 合并重复顶点 → 拓扑崩坏。
    实测（样本 20250423_1800_838505）：v 40262→27583、components 1→22532、
    non_manifold 0→17524、watertight True→False。

    Args:
        mesh: trimesh.Trimesh（MC 输出，归一化坐标 [-0.5, 0.5]）
        pts: [N,3] 归一化点云（512³ 精度，同一个坐标空间）
        max_move: 最大位移（归一化单位），沿移动方向钳制。建议设。
        mode: 移动方向
              "normal"（默认，安全）= **只沿顶点法线**移动：取到最近点的向量在法线上的
                                    投影并钳制 —— 相邻顶点各走各的法线，不会横向碰撞
              "nearest"           = 直接移到最近点（**会崩拓扑，仅留作对照**）
    Returns:
        (mesh, stats dict)
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=np.float64)
    P = np.asarray(pts, dtype=np.float64)
    if len(P) == 0:
        return mesh, {"error": "空点云"}

    n0 = len(V)
    dd, idx = cKDTree(P).query(V)
    D = P[idx] - V                                  # 到最近点的向量

    if mode == "nearest":
        t = np.linalg.norm(D, axis=1)
        if max_move is not None and float(max_move) > 0:
            keep = t <= float(max_move)
            D[~keep] = 0.0
            n_clamped = int((~keep).sum())
        else:
            n_clamped = 0
        V_new = V + D
    else:                                            # "normal"：只沿法线走
        N = np.asarray(mesh.vertex_normals, dtype=np.float64)
        tn = (D * N).sum(axis=1)                     # 在法线上的投影分量
        n_clamped = 0
        if max_move is not None and float(max_move) > 0:
            m = float(max_move)
            n_clamped = int((np.abs(tn) > m).sum())
            tn = np.clip(tn, -m, m)
        V_new = V + N * tn[:, None]

    moved = np.linalg.norm(V_new - V, axis=1)
    mesh.vertices = V_new
    # 安全检查：顶点数不能变（变了说明发生了合并 → 拓扑已崩）
    n1 = len(mesh.vertices)
    if n1 != n0:
        print(f"  ⚠️ 精修后顶点数 {n0} → {n1}（发生了合并！拓扑可能已损坏，"
              f"请用 mode='normal' 且调小 max_move）")
    stats = {
        "n_verts": n0,
        "n_clamped": n_clamped,
        "move_mean": float(moved.mean()),
        "move_p95": float(np.percentile(moved, 95)),
        "move_max": float(moved.max()),
        "nn_mean": float(dd.mean()),
    }
    if verbose:
        print(f"[mesh_grid] 点云精修顶点(mode={mode}): {n0} 个顶点 | "
              f"位移 mean={stats['move_mean']:.5f} p95={stats['move_p95']:.5f} "
              f"max={stats['move_max']:.5f} | 被钳制 {n_clamped} 个"
              + (f"（max_move={max_move}）" if max_move else ""))
    return mesh, stats


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
    """找出所有边界环，**沿面的绕向做有向边追踪**，保证每个环都闭合。

    为什么不用「按顶点度数走」：网格有非流形边时，边界图上会出现度数 >2 的分叉点，
    按顶点游走会在分叉处走进死路，产出**非闭合的路径**。这种路径被当多边形填充后
    是一个对不上边界的游离补丁（表现为 components 变多、non_manifold 变多）。

    这里改成在**有向边**上走：每条边界边取「沿它所属面的绕向」的方向 u→v，
    下一步必须从 v 出发；分叉点（v 有多个出边）选**转弯最小**的那条（最直，
    贴着真实孔洞边界走）。因为每步都消耗一条有向边且只消耗一次，环必然闭合。

    另一个好处：环的方向天然就是「沿面绕向」，所以补面只要**反着绕**，
    合成后的网格绕向自动一致（见 repair_holes）。

    Returns:
        list[list[int]]：每个环的顶点索引（首尾不重复，方向沿面绕向）。
    """
    import numpy as np

    if mesh is None or len(mesh.faces) == 0:
        return []
    faces = np.asarray(mesh.faces, dtype=np.int64)
    V = np.asarray(mesh.vertices, dtype=np.float64)

    edges = np.asarray(mesh.edges_sorted, dtype=np.int64)
    uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    if len(bnd) == 0:
        return []
    bset = set((int(a), int(b)) for a, b in bnd)  # 已排序的 (小,大)

    # 每条边界边的有向形式：沿它所属面的绕向
    directed = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key in bset:
                directed.add((int(u), int(v)))

    out_map = {}
    for u, v in directed:
        out_map.setdefault(u, []).append((u, v))

    loops = []
    unused = set(directed)
    for start in directed:
        if start not in unused:
            continue
        loop, cur = [], start
        while cur in unused:
            unused.discard(cur)
            loop.append(cur[0])
            v = cur[1]
            cands = [e for e in out_map.get(v, []) if e in unused]
            if not cands:
                break
            if len(cands) == 1:
                cur = cands[0]
            else:
                # 分叉点：选与来向夹角最小（最直）的那条
                d_in = V[v] - V[cur[0]]
                n1 = float(np.linalg.norm(d_in))
                best, best_cos = None, -2.0
                for e in cands:
                    d = V[e[1]] - V[v]
                    n2 = float(np.linalg.norm(d))
                    cos = -1.0 if (n1 < 1e-12 or n2 < 1e-12) else float(d_in @ d / (n1 * n2))
                    if cos > best_cos:
                        best_cos, best = cos, e
                cur = best
        # 只有真正回到起点的环才算闭合
        if len(loop) >= 3 and (loop[-1], loop[0]) in directed:
            loops.append(loop)
    return loops


def _edge_to_face_normal(mesh):
    """{排序后的边 (u,v) → 该边所属面的法线}，只用于查边界边的相邻面走向。"""
    import numpy as np

    es = np.asarray(mesh.edges_sorted, dtype=np.int64)      # [3F,2]
    fn = np.asarray(mesh.face_normals, dtype=np.float64)    # [F,3]
    face_of_row = np.repeat(np.arange(len(mesh.faces)), 3)
    uniq, first = np.unique(es, axis=0, return_index=True)
    face_of_uniq = face_of_row[first]
    return {(int(a), int(b)): fn[fi] for (a, b), fi in zip(uniq, face_of_uniq)}


def _fill_loop_advancing(V, loop, edge_normals, w_dir=1.0, w_area=0.3, w_shape=0.2):
    """推进前沿补洞（Liepa 式）：从边界往里逐个长三角形。

    与「拟合平面 + 投影 + 耳切」的区别，正是它不喷射线的原因：
      · **顶点一个都不动**——投影法会把环上的原顶点搬到拟合平面上，
        等于位移了原有曲面，射线就是从那儿长出来的；
      · 每次在所有候选三角形里挑**与相邻面二面角最小**的那个，
        补出来的面顺着周围曲面的走向，而不是硬糊一块平板；
      · 纯 3D 就地三角化，不做 2D 投影 → 不存在投影自交。

    权重 = 方向项（与相邻面法线的夹角）+ 面积项 + 形状项（惩罚细长三角形）。

    Args:
        V: [N,3] 顶点坐标。
        loop: 边界环顶点索引（方向沿面绕向）。
        edge_normals: [len(loop),3]，edge_normals[j] = 边 (loop[j-1], loop[j])
                      的相邻面法线；无相邻面（新生成的边）用 NaN。
        w_dir / w_area / w_shape: 三项权重。
    Returns:
        list[(i,j,k)] 新三角形（已按与原面相反的绕向生成，合成后 winding 一致）。
    """
    import numpy as np

    V = np.asarray(V, dtype=np.float64)
    A = [int(x) for x in loop]
    nrm = np.asarray(edge_normals, dtype=np.float64).copy()  # [m,3]
    faces = []

    while len(A) > 3:
        m = len(A)
        idx = np.arange(m)
        A_np = np.asarray(A, dtype=np.int64)
        a = A_np[(idx - 1) % m]
        b = A_np
        c = A_np[(idx + 1) % m]

        Va, Vb, Vc = V[a], V[b], V[c]
        # 将要添加的三角形是 (a, c, b)（反向绕），法线按这个绕向算
        tn = np.cross(Vc - Va, Vb - Va)
        ln = np.linalg.norm(tn, axis=1)
        bad = ln < 1e-14
        ln = np.where(bad, 1.0, ln)
        tn = tn / ln[:, None]
        area = 0.5 * ln

        # 方向项：候选面法线 vs 相邻两面法线（NaN 表示无边相邻 → 不计）
        n_next = np.roll(nrm, -1, axis=0)
        d1 = np.arccos(np.clip(np.sum(tn * nrm, axis=1), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.sum(tn * n_next, axis=1), -1.0, 1.0))
        dev = np.nanmax(np.stack([d1, d2], axis=1), axis=1)
        dev = np.where(np.isnan(dev), 0.0, dev)

        # 形状项：周长²/面积，越大越细长
        per = (np.linalg.norm(Vb - Va, axis=1) + np.linalg.norm(Vc - Vb, axis=1)
               + np.linalg.norm(Va - Vc, axis=1))
        shape = per ** 2 / np.maximum(area, 1e-14)

        def _nz(x):
            mx = np.max(x)
            return x / mx if mx > 0 else x

        W = (w_dir * _nz(dev / np.pi) + w_area * _nz(area) + w_shape * _nz(shape))
        W = np.where(bad, np.inf, W)   # 退化三角形永不选
        if not np.isfinite(W).any():
            break
        i = int(np.argmin(W))

        ia, ib, ic = (i - 1) % m, i, (i + 1) % m
        faces.append((int(A_np[ia]), int(A_np[ic]), int(A_np[ib])))  # 反向绕

        new_n = tn[i]
        A = [A_np[k] for k in range(m) if k != ib]
        # nrm[j] 记的是边 (A[j-1], A[j]) 的相邻面法线。删掉顶点 ib 后：
        # 索引 < ib 的边不变，> ib 的整体前移一位（np.delete 正好做到），
        # 新边 (A[ib-1], A[ib+1]) 落到索引 ib % (m-1) 上，法线就是刚加的这个面。
        nrm = np.delete(nrm, ib, axis=0)
        nrm[ib % (m - 1)] = new_n

    if len(A) == 3:  # 最后收口
        faces.append((A[2], A[1], A[0]))
    return faces


def _fill_loops_once(mesh, loops, max_loop_len=0):
    """对给定的边界环做一轮邻接面插值填充，返回新的 Trimesh（原面 + 补面）。

    补面的绕向与环的绕向**相反**：环是沿「所属面的绕向」走的，共享边要在两个面里
    方向相反，合成后的网格绕向才一致（不然 winding_ok 会从 True 掉成 False）。
    """
    import numpy as np
    import trimesh

    V = np.asarray(mesh.vertices, dtype=np.float64)
    tri_all = np.asarray(mesh.faces, dtype=np.int64)

    # 边界边 → 相邻面法线（推进前沿的方向项要它）
    e2n = _edge_to_face_normal(mesh)

    new_faces = []
    filled = skipped = 0
    for loop in loops:
        if max_loop_len and len(loop) > max_loop_len:
            skipped += 1
            continue
        m = len(loop)
        en = np.full((m, 3), np.nan, dtype=np.float64)
        for j in range(m):
            u, v = loop[j - 1], loop[j]
            fn_ = e2n.get((u, v) if u < v else (v, u))
            if fn_ is not None:
                en[j] = fn_
        new_faces.extend(_fill_loop_advancing(V, loop, en))
        filled += 1

    if not new_faces:
        return None, filled, skipped

    all_faces = np.concatenate([tri_all, np.asarray(new_faces, dtype=np.int64)], axis=0)
    # 顶点原封不动（推进前沿不移动任何顶点），直接复用 V
    out = trimesh.Trimesh(vertices=V, faces=all_faces, process=False)
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
    return out, filled, skipped


def repair_holes(mesh, max_loop_len=0, smooth_iters=0, max_passes=3):
    """邻接面插值补洞：对每个边界环用相邻面的法线拟合平面，再在平面上插值补面。

    与 trimesh.repair.fill_holes 的区别：后者用环的质心做扇形填充（不管相邻面的
    走向），这里用**相邻面的面积加权法线**决定补面所在平面，环顶点先投影到该平面
    再耳切三角化，补出来的面延续周围曲面的走向，不产生尖刺。

    多轮迭代：一轮补完后非流形分叉点常被邻接的补面消解，边界会重新串成干净的环，
    所以补完重算边界再补，直到没有边界环 / 补不动 / 达到 max_passes。

    Args:
        mesh: trimesh.Trimesh。
        max_loop_len: >0 时只补顶点数 ≤ 该值的环（防大洞被硬填成平板）。
        smooth_iters: 补完后 Laplacian 平滑次数（0=不做）。
        max_passes: 最多补几轮。
    Returns:
        (mesh, stats dict)
    """
    import numpy as np
    import trimesh

    stats = {"loops": 0, "filled": 0, "skipped": 0, "added_faces": 0, "passes": 0}
    if mesh is None or len(mesh.faces) == 0:
        return mesh, stats

    cur = mesh.copy()
    cur.merge_vertices()
    base_faces = len(cur.faces)

    for p in range(max_passes):
        loops = boundary_loops(cur)
        stats["loops"] += len(loops)
        if not loops:
            break
        nxt, filled, skipped = _fill_loops_once(cur, loops, max_loop_len)
        stats["filled"] += filled
        stats["skipped"] += skipped
        stats["passes"] = p + 1
        if nxt is None:
            break
        # 安全阀：补洞只该加面，不该减面。少了说明面索引错位 → 宁可停手。
        if len(nxt.faces) <= len(cur.faces):
            if len(nxt.faces) < len(cur.faces):
                print("[mesh_grid] ⚠️ 补洞后反而少面，疑似索引错位 → 停手")
            break
        cur = nxt
        if cur.is_watertight:
            break

    if cur is mesh or len(cur.faces) <= base_faces:
        print(f"[mesh_grid] 补洞: 边界环 {stats['loops']} 个，未产生有效补面")
        return mesh, stats

    # 绕向只在真的不一致时才修（补面已按相反绕向生成，通常本来就一致；
    # 无条件 fix_normals 会在非水密网格上把整块补丁翻转，反而破坏 winding）
    if not cur.is_winding_consistent:
        print("[mesh_grid] 补洞后绕向不一致 → fix_normals 修正")
        trimesh.repair.fix_normals(cur)

    if smooth_iters > 0:
        try:
            from trimesh.smoothing import filter_laplacian
            filter_laplacian(cur, iterations=smooth_iters, lamb=0.5)
        except Exception as e:
            print(f"[mesh_grid] 补洞平滑跳过: {e}")

    stats["added_faces"] = len(cur.faces) - base_faces
    print(f"[mesh_grid] 补洞: {stats['passes']} 轮, 边界环 {stats['loops']} 个"
          f"（补 {stats['filled']} / 跳过 {stats['skipped']}），"
          f"新增 {stats['added_faces']} 面 → watertight={cur.is_watertight} "
          f"winding_ok={cur.is_winding_consistent}")
    return cur, stats


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
