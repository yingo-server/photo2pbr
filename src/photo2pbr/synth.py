# -*- coding: utf-8 -*-
"""synth —合成"房间照"（测试与演示用，不参与生产逻辑）。

用解析法生成:
    · 盒体房间的深度图（光线-盒求交，精确）
    · 每面材质（albedo 噪声）与兰伯特光照（可换太阳方向）
所以它是**带真值**的：几何、albedo、shading 都知道，测试才有意义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .core import linear_to_srgb
from .geom import depth_map_of_box, intrinsics, rotation_world_to_cam

__all__ = ["SynthRoom", "make_room"]

# 世界系: x 右, y 上, z 前 → 六个面的外法线（指向房间内部）
# 注意：顺序必须与 dist 的构造顺序一致 [xmin,ymin,zmin,xmax,ymax,zmax]
_FACE_ORDER = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
_FACE_NAMES = {"xmin": "left", "xmax": "right",
               "ymin": "floor", "ymax": "ceiling",
               "zmin": "front", "zmax": "back"}
_FACE_NORMALS = {"left": (1.0, 0.0, 0.0), "right": (-1.0, 0.0, 0.0),
                 "floor": (0.0, 1.0, 0.0), "ceiling": (0.0, -1.0, 0.0),
                 "front": (0.0, 0.0, 1.0), "back": (0.0, 0.0, -1.0)}


def _bilinear_noise(rng, h: int, w: int, gy: int = 6, gx: int = 8,
                    lo: float = 0.30, hi: float = 0.85) -> np.ndarray:
    """低分辨率噪声 + 双线性上采样 → 平滑材质（避免"最近邻色块"伪影）。"""
    g = rng.random((gy, gx, 3)) * (hi - lo) + lo
    ys = np.linspace(0.0, gy - 1.0, h)
    xs = np.linspace(0.0, gx - 1.0, w)
    y0 = np.clip(np.floor(ys).astype(int), 0, gy - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, gx - 2)
    fy = (ys - y0)[:, None, None]
    fx = (xs - x0)[None, :, None]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


def _value_noise(rng, h: int, w: int, gy: int, gx: int) -> np.ndarray:
    """单通道值噪声（双线性上采样）。"""
    g = rng.random((gy, gx))
    ys = np.linspace(0.0, gy - 1.0, h)
    xs = np.linspace(0.0, gx - 1.0, w)
    y0 = np.clip(np.floor(ys).astype(int), 0, gy - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, gx - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


def _wall_texture(rng, h: int, w: int, tint=(1.0, 1.0, 1.0)) -> np.ndarray:
    """程序化"灰泥/混凝土"墙面：多尺度斑块 + 软污渍 + 细颗粒 + 轻微色调。

    目标是"像真实照片的墙面"（低饱和、有颗粒与污渍），而不是彩虹色块。
    """
    from .material import box_filter

    base = 0.62 + 0.16 * (_value_noise(rng, h, w, 4, 5) - 0.5) * 2.0
    base += 0.10 * (_value_noise(rng, h, w, 12, 16) - 0.5) * 2.0
    base += 0.05 * (_value_noise(rng, h, w, 48, 64) - 0.5) * 2.0
    stain_raw = (_value_noise(rng, h, w, 3, 4) > 0.70).astype(np.float64)
    stain = box_filter(stain_raw, 9) * 3.2                      # 软污渍
    grain = rng.normal(0.0, 0.012, (h, w))
    v = np.clip(base - 0.16 * np.clip(stain, 0.0, 1.0) + grain, 0.22, 0.92)
    rgb = np.dstack([v * tint[0], v * tint[1], v * tint[2]])
    return np.clip(rgb, 0.0, 1.0)


@dataclass
class SynthRoom:
    image: np.ndarray          # sRGB float [0,1]
    depth: np.ndarray          # 相机 z 深度
    valid: np.ndarray
    K: np.ndarray
    R_wc: np.ndarray           # 世界→相机
    bmin: np.ndarray
    bmax: np.ndarray
    faces: Dict[str, np.ndarray]
    albedo_lin: np.ndarray
    shading_lin: np.ndarray
    sun_dir_world: np.ndarray


def make_room(width: int = 320, height: int = 240,
              dims: Tuple[float, float, float] = (4.0, 2.8, 5.0),
              fov_deg: float = 75.0, yaw_deg: float = 12.0, pitch_deg: float = 0.0,
              sun_el_deg: float = 35.0, sun_az_deg: float = 140.0,
              seed: int = 0, noise: float = 0.0,
              ambient: float = 0.28, key: float = 0.9) -> SynthRoom:
    """合成一张"站在房间中央拍的照片"（含真值）。"""
    rng = np.random.default_rng(seed)
    dims = np.asarray(dims, dtype=np.float64)
    bmin, bmax = -dims / 2.0, +dims / 2.0          # 相机在房间中心（世界原点）
    K = intrinsics(width, height, fov_deg)
    R_wc = rotation_world_to_cam(yaw_deg, pitch_deg)

    depth, valid, hit = depth_map_of_box(K, width, height, R_wc,
                                         np.zeros(3), bmin, bmax)
    dist = np.stack([np.abs(hit[..., i] - bmin[i]) for i in range(3)]
                    + [np.abs(hit[..., i] - bmax[i]) for i in range(3)], axis=-1)
    face_id = np.argmin(dist, axis=-1)

    el, az = np.radians(sun_el_deg), np.radians(sun_az_deg)
    L = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])

    albedo = np.zeros((height, width, 3), np.float64)
    shading = np.zeros((height, width), np.float64)
    faces: Dict[str, np.ndarray] = {}
    for fid, fkey in enumerate(_FACE_ORDER):
        m = (face_id == fid) & valid
        faces[_FACE_NAMES[fkey]] = m
        if not m.any():
            continue
        # 每面用程序化"灰泥"材质（近白，略带冷暖色调差别）
        tints = {"left": (1.00, 1.00, 1.01), "right": (1.00, 0.99, 0.97),
                 "floor": (1.02, 0.98, 0.92), "ceiling": (0.99, 0.99, 1.00),
                 "front": (1.00, 1.00, 1.00), "back": (0.99, 0.99, 0.98)}
        tex = _wall_texture(rng, height, width,
                            tint=tints.get(_FACE_NAMES[fkey], (1.0, 1.0, 1.0)))
        yy, xx = np.nonzero(m)
        albedo[yy, xx] = tex[yy, xx]
        n = np.array(_FACE_NORMALS[_FACE_NAMES[fkey]])
        lam = max(0.0, float(np.dot(n, L)))
        grad = 0.75 + 0.25 * (1.0 - yy.astype(np.float64) / height)   # 轻微纵深梯度
        shading[yy, xx] = (ambient + key * lam) * grad

    lin = np.clip(albedo * shading[..., None], 0.0, 1.0)
    image = linear_to_srgb(lin)
    if noise > 0.0:
        image = np.clip(image + rng.normal(0.0, noise, image.shape), 0.0, 1.0)
    return SynthRoom(image=image, depth=depth, valid=valid, K=K, R_wc=R_wc,
                     bmin=bmin, bmax=bmax, faces=faces,
                     albedo_lin=albedo, shading_lin=shading, sun_dir_world=L)