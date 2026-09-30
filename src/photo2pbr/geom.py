# -*- coding: utf-8 -*-
"""geom —针孔相机 / 光线-盒求交 / 单图房间盒体估计。

约定
----
* 相机系: x 右, y 下, z 前（OpenCV 习惯），单位米。
* 世界系（房间系）: x 右, y 上, z 前。
* 深度 = 相机 z 向距离（不是欧氏距离）。
* 曼哈顿假设: 房间近似长方体（3 组正交平面）。

单视图硬事实
------------
站在房间里向前拍，**永远看不到自己背后那面墙**。
因此深度轴只有一端（远墙）有真实观测，另一端（近墙）必须用假设补齐，
并在结果里用 ``observed`` 如实标注，不许假装是"测出来的"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "intrinsics", "ray_dirs", "ray_box_t", "depth_map_of_box",
    "backproject", "point_normals", "rotation_world_to_cam",
    "estimate_room", "RoomEstimate",
]

FACE_NAMES = ("floor", "ceiling", "left", "right", "back", "front")


# ═══════════════════════════════════════════════════════════════════
# 相机
# ═══════════════════════════════════════════════════════════════════
def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """水平 FOV → 3x3 内参（方形像素）。"""
    f = 0.5 * float(width) / np.tan(np.radians(0.5 * float(fov_deg)))
    return np.array([[f, 0.0, (width - 1) * 0.5],
                     [0.0, f, (height - 1) * 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def ray_dirs(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """每个像素的相机系方向矢量，z=1（这样参数 t 就等于 z 深度）。"""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                       np.arange(height, dtype=np.float64))
    return np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """深度图 → 相机系点云（沿用 ray_dirs 的 z=1 约定）。"""
    h, w = depth.shape
    return ray_dirs(K, w, h) * np.asarray(depth, dtype=np.float64)[..., None]


def rotation_world_to_cam(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """世界(x右,y上,z前) → 相机(x右,y下,z前)：先翻 y，再 yaw / pitch。"""
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    F = np.diag([1.0, -1.0, 1.0])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return Rx @ Ry @ F


# ═══════════════════════════════════════════════════════════════════
# 光线-盒求交（合成参考实现 + 深度合成）
# ═══════════════════════════════════════════════════════════════════
def ray_box_t(origin, dirs, bmin, bmax):
    """光线与轴对齐盒求交（slab 法）。返回 (t, valid)。

    相机在盒内时取"第一面出射"的 t；在盒外取入射 t。
    """
    origin = np.asarray(origin, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    t0 = (bmin - origin) * inv
    t1 = (bmax - origin) * inv
    tmin = np.max(np.minimum(t0, t1), axis=-1)
    tmax = np.min(np.maximum(t0, t1), axis=-1)
    t = np.where(tmin > 1e-9, tmin, tmax)
    valid = (tmax > np.maximum(tmin, 1e-9)) & (t > 0.0)
    return t, valid


def depth_map_of_box(K: np.ndarray, width: int, height: int,
                     R_wc: np.ndarray, C, bmin, bmax):
    """合成一张"从盒内看到的盒"的深度图（用于自测/演示）。"""
    d_cam = ray_dirs(K, width, height)                     # (h,w,3) z=1
    d_world = d_cam @ np.asarray(R_wc, dtype=np.float64)   # R_wc.T @ d
    t, valid = ray_box_t(C, d_world, bmin, bmax)
    depth = np.where(valid, t, 0.0)
    hit = np.asarray(C, dtype=np.float64) + t[..., None] * d_world
    return depth, valid, hit


# ═══════════════════════════════════════════════════════════════════
# 法线 & 主轴
# ═══════════════════════════════════════════════════════════════════
def point_normals(pts: np.ndarray, valid: np.ndarray):
    """由深度点云求逐像素法线（中心差分叉积），翻向相机侧。"""
    pts = np.asarray(pts, dtype=np.float64)
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(dx, dy)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = ln[..., 0] > 1e-12
    n = n / np.where(ln > 1e-12, ln, 1.0)
    # 翻向相机：相机在原点 → 朝向相机的面法线满足 n·p < 0
    dot = np.sum(n * pts, axis=-1)
    n = np.where((dot > 0.0)[..., None], -n, n)
    mask = np.asarray(valid, bool) & ok          # （注意：不能在翻转前用 dot 当掩码）
    return n, mask


def _principal_axes(normals: np.ndarray) -> np.ndarray:
    """正交主轴：Σ n nᵀ 的特征分解（盒体的 3 组法线天然正交）。"""
    S = normals.T @ normals
    _, V = np.linalg.eigh(S)
    axes = np.ascontiguousarray(V[:, ::-1].T.copy())   # 行 = 主轴
    if np.linalg.det(axes) < 0.0:
        axes[2] = -axes[2]
    return axes


def _cluster_planes(vals: np.ndarray, tol: float, max_peaks: int = 2):
    """直方图峰 → 面簇中心。

    墙在坐标分布里是一个**尖峰**；分位数会把"斜着看过去的散点"也算进来，
    所以这里找峰，而不是取分位。返回 ≤max_peaks 个面的坐标（升序无关）。
    """
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if not np.isfinite(vmin) or vmax - vmin <= 1e-12:
        return [vmin] if np.isfinite(vmin) else []
    nb = 128
    hist, edges = np.histogram(vals, bins=nb, range=(vmin, vmax))
    binw = (vmax - vmin) / nb
    found = []
    for b in np.argsort(-hist):
        if hist[b] <= 0:
            break
        c = 0.5 * (edges[b] + edges[b + 1])
        if any(abs(c - f) < max(3.0 * binw, tol) for f in found):
            continue
        sel = np.abs(vals - c) < max(2.0 * binw, tol)
        if int(sel.sum()) < 10:
            continue
        found.append(float(vals[sel].mean()))
        if len(found) >= max_peaks:
            break
    return found


# ═══════════════════════════════════════════════════════════════════
# 房间估计
# ═══════════════════════════════════════════════════════════════════
@dataclass
class RoomEstimate:
    """房间盒体（**相机为原点**，单位米）。"""
    axes: np.ndarray                 # (3,3) 每行 = 房间轴在相机系下的方向
    bmin: np.ndarray                 # 房间坐标下界（相机原点）
    bmax: np.ndarray
    roles: Dict[str, int]            # {"horizontal": i, "vertical": j, "depth": k}
    observed: Dict[str, bool]        # 各端是否有真实观测（"axis{i}:lo"）
    coverage: Dict[str, float]
    face_masks: Dict[str, np.ndarray]
    face_normals: Dict[str, np.ndarray] = field(default_factory=dict)  # 相机系内法线
    confidence: str = "estimated"    # estimated | partial | assumed

    @property
    def dims(self) -> np.ndarray:
        return self.bmax - self.bmin


def _role_dim(i: int, roles: Dict[str, int], fallback: Tuple[float, float, float]) -> float:
    if i == roles["horizontal"]:
        return float(fallback[0])
    if i == roles["vertical"]:
        return float(fallback[1])
    return float(fallback[2])


def estimate_room(depth: np.ndarray, K: np.ndarray, valid: Optional[np.ndarray] = None,
                  *, stride: int = 3, min_plane_frac: float = 0.02,
                  front_depth_ratio: float = 1.0,
                  fallback_dims: Tuple[float, float, float] = (5.0, 2.8, 4.0)
                  ) -> RoomEstimate:
    """深度图 → 房间盒体（曼哈顿）。

    步骤:
        1. 点云法线 → Σnnᵀ 特征分解 → 3 组正交轴
        2. 投影到房间系 → 逐轴分位数求边界
        3. 逐平面覆盖率 → 判断哪端真的"看见"了
        4. 看不见的一端：深度轴用镜像假设、其余用默认尺寸补齐（observed=False）
    """
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    if valid is None:
        valid = depth > 1e-6
    valid = np.asarray(valid, bool) & (depth > 1e-6)

    pts = backproject(depth, K)
    n, nm = point_normals(pts, valid)
    nm = nm & valid
    if nm.sum() < 0.005 * h * w:
        raise ValueError("有效法线不足（<0.5%% 像素）：不是盒体房间或深度太差")

    ns = n[::stride, ::stride][nm[::stride, ::stride]]
    axes = _principal_axes(ns)

    P = pts[valid]
    pr = P @ axes.T
    N = n[valid]                       # 与 P 同序的逐点法线
    NOK = nm[valid]

    # ── 逐轴找"面簇"：只看"法线与该轴一致"的点（真正贴在这组墙面上的点），
    #    这样细窄的墙（斜视时只有几十像素）也能被认出来。
    vals_all = [pr[:, i] for i in range(3)]
    tol = np.zeros(3)
    cand_lo = np.full(3, np.nan)
    cand_hi = np.full(3, np.nan)
    cover_lo = np.zeros(3)
    cover_hi = np.zeros(3)
    for i in range(3):
        v = vals_all[i]
        tol[i] = max(0.02, 0.01 * float(v.max() - v.min()))
        along = NOK & (np.abs(N @ axes[i]) > 0.9)
        if int(along.sum()) < 20:
            continue
        vs = v[along]
        for c in _cluster_planes(vs, tol[i]):
            if c < 0.0:
                cand_lo[i] = c if np.isnan(cand_lo[i]) else min(cand_lo[i], c)
            else:
                cand_hi[i] = c if np.isnan(cand_hi[i]) else max(cand_hi[i], c)
        if not np.isnan(cand_lo[i]):
            cover_lo[i] = float(np.mean(np.abs(vs - cand_lo[i]) < tol[i]))
        if not np.isnan(cand_hi[i]):
            cover_hi[i] = float(np.mean(np.abs(vs - cand_hi[i]) < tol[i]))

    # 轴角色：竖轴 ≈ 相机 up；深度轴 ≈ 相机 forward；剩下是水平轴
    up_cam = np.array([0.0, -1.0, 0.0])
    fwd_cam = np.array([0.0, 0.0, 1.0])
    d_up = np.abs(axes @ up_cam)
    d_fwd = np.abs(axes @ fwd_cam)
    depth_ax = int(np.argmax(d_fwd))
    order_up = np.argsort(-d_up)
    vert = int(next(i for i in order_up if i != depth_ax))
    horiz = int(next(i for i in range(3) if i not in (vert, depth_ax)))
    roles = {"horizontal": horiz, "vertical": vert, "depth": depth_ax}

    obs = {}
    for i in range(3):
        obs["axis%d:lo" % i] = bool((not np.isnan(cand_lo[i]))
                                    and cover_lo[i] >= min_plane_frac)
        obs["axis%d:hi" % i] = bool((not np.isnan(cand_hi[i]))
                                    and cover_hi[i] >= min_plane_frac)

    lo = cand_lo.copy()
    hi = cand_hi.copy()

    # ── 深度轴：单视图看不到背后 → 只信有面簇的一端，另一端镜像假设
    i = depth_ax
    mirror_depth = False
    if not (obs["axis%d:lo" % i] and obs["axis%d:hi" % i]):
        have = [(abs(c), c) for c in (cand_lo[i], cand_hi[i]) if not np.isnan(c)]
        back_val = max(have)[1] if have else float(fallback_dims[2]) * 0.5
        front_val = -back_val * float(front_depth_ratio)
        lo[i], hi[i] = min(back_val, front_val), max(back_val, front_val)
        mirror_depth = True

    # ── 其余轴：缺端用默认尺寸补齐
    assumed_any = mirror_depth
    for i in range(3):
        if i == depth_ax:
            continue
        if np.isnan(lo[i]):
            base = hi[i] if not np.isnan(hi[i]) else 0.0
            lo[i] = base - _role_dim(i, roles, fallback_dims)
            assumed_any = True
        if np.isnan(hi[i]):
            base = lo[i] if not np.isnan(lo[i]) else 0.0
            hi[i] = base + _role_dim(i, roles, fallback_dims)
            assumed_any = True

    # ── 面命名
    names: Dict[Tuple[int, str], str] = {}
    s_v = float(axes[vert] @ np.array([0.0, 1.0, 0.0]))     # 相机 y 向下
    if s_v > 0.0:
        names[(vert, "hi")], names[(vert, "lo")] = "floor", "ceiling"
    else:
        names[(vert, "lo")], names[(vert, "hi")] = "floor", "ceiling"
    s_h = float(axes[horiz] @ np.array([1.0, 0.0, 0.0]))    # 相机 x 向右
    if s_h > 0.0:
        names[(horiz, "hi")], names[(horiz, "lo")] = "right", "left"
    else:
        names[(horiz, "lo")], names[(horiz, "hi")] = "right", "left"
    d = depth_ax
    if not np.isnan(cand_lo[d]) and np.isnan(cand_hi[d]):
        back_lo = True                                      # 真实观测在 lo 端
    elif not np.isnan(cand_hi[d]) and np.isnan(cand_lo[d]):
        back_lo = False
    else:
        back_lo = bool(abs(lo[d]) > abs(hi[d]))
    if back_lo:
        names[(d, "lo")], names[(d, "hi")] = "back", "front"
    else:
        names[(d, "hi")], names[(d, "lo")] = "back", "front"

    # ── 面掩膜：最近平面归属
    ends = [(0, "lo"), (0, "hi"), (1, "lo"), (1, "hi"), (2, "lo"), (2, "hi")]
    dist = np.stack([np.abs(pr[:, a] - (lo[a] if e == "lo" else hi[a]))
                     for (a, e) in ends], axis=1)
    thr = np.array([tol[a] for (a, _e) in ends]) * 2.0
    idx = np.argmin(dist, axis=1)
    ok = dist[np.arange(len(dist)), idx] < thr[idx]

    face_masks = {name: np.zeros((h, w), bool) for name in FACE_NAMES}
    py, px = np.nonzero(valid)
    for j, key in enumerate(ends):
        nm_ = names.get(key)
        if nm_ is None:
            continue
        sel = ok & (idx == j)
        face_masks[nm_][py[sel], px[sel]] = True

    # 面法线（相机系）：内法线 = ±e_axis，再旋到相机系
    face_normals: Dict[str, np.ndarray] = {}
    for key, name in names.items():
        a, e = key
        n_room = np.zeros(3)
        n_room[a] = 1.0 if e == "lo" else -1.0
        face_normals[name] = axes.T @ n_room

    coverage = {}
    for i in range(3):
        coverage["axis%d:lo" % i] = float(cover_lo[i])
        coverage["axis%d:hi" % i] = float(cover_hi[i])
    # 用了假设（镜像/默认尺寸）就必须如实降级，不许冒充"测出来的"
    confidence = "partial" if (mirror_depth or assumed_any) else "estimated"

    return RoomEstimate(axes=axes, bmin=lo, bmax=hi, roles=roles,
                        observed=obs, coverage=coverage,
                        face_masks=face_masks, face_normals=face_normals,
                        confidence=confidence)