# -*- coding: utf-8 -*-
"""light —天空亮度模型 / 太阳盘检测。

模型
----
天空亮度随"仰角"变化，用最简的事实模型:
    L(e) = A + B·sin(e)        （e = 仰角，A = 地平线亮度）
    CIE 阴天特例: A = Lz/3, B = 2·Lz/3  →  zenith/horizon = 3
太阳: 画面中最亮的局部斑块 → 方向（像素→相机矢量）、相对强度、角径（→软硬）。
看不到天空（室内墙照）时：**如实标注 unconstrained**，环境退化为中性，不瞎编。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .material import box_filter

__all__ = ["fit_sky_luminance", "estimate_sun", "estimate_lighting",
           "equirect_from_lighting", "Lighting"]


# ═══════════════════════════════════════════════════════════════════
# 天空
# ═══════════════════════════════════════════════════════════════════
def _row_sin_elevation(K: np.ndarray, height: int) -> np.ndarray:
    fy, cy = float(K[1, 1]), float(K[1, 2])
    y = (np.arange(height, dtype=np.float64) - cy) / fy
    return -y / np.sqrt(y * y + 1.0)


def fit_sky_luminance(lin_lum: np.ndarray, K: np.ndarray, horizon_row: int,
                      sky_mask: Optional[np.ndarray] = None) -> Dict[str, object]:
    """把 L(e)=A+B·sin(e) 最小二乘拟合到天空区域。"""
    h, w = lin_lum.shape
    sin_e = _row_sin_elevation(K, h)
    rows = np.arange(h)
    sel = rows < int(horizon_row)
    if sky_mask is not None:
        sel &= np.asarray(sky_mask, bool).any(axis=1)
    if int(sel.sum()) < 4:
        return {"A": 0.0, "B": 0.0, "rmse": 0.0, "n_rows": 0,
                "ratio": 1.0, "confidence": "unconstrained"}

    Lrow = np.asarray(lin_lum, dtype=np.float64).mean(axis=1)
    x = sin_e[sel]
    y = Lrow[sel]
    B, A = np.polyfit(x, y, 1)
    pred = A + B * x
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    zenith = float(A + B)
    horizon = float(A)
    ratio = zenith / max(horizon, 1e-9)
    return {"A": float(A), "B": float(B), "rmse": rmse, "n_rows": int(sel.sum()),
            "ratio": float(ratio), "confidence": "fitted"}


# ═══════════════════════════════════════════════════════════════════
# 太阳
# ═══════════════════════════════════════════════════════════════════
def estimate_sun(lin_lum: np.ndarray, K: np.ndarray,
                 sky_mask: Optional[np.ndarray] = None,
                 win: int = 14) -> Dict[str, object]:
    """找最亮斑块 → 方向 / 相对强度 / 角径。"""
    L = np.asarray(lin_lum, dtype=np.float64)
    h, w = L.shape
    blur = box_filter(L, 4)
    if sky_mask is None:
        m = np.ones_like(L, bool)
    else:
        m = np.asarray(sky_mask, bool)
    mb = np.where(m, blur, -1.0e9)
    v0, u0 = np.unravel_index(int(np.argmax(mb)), mb.shape)
    if mb[v0, u0] <= -1.0e8:
        return {"confidence": "none"}

    y0, y1 = max(0, v0 - win), min(h, v0 + win + 1)
    x0, x1 = max(0, u0 - win), min(w, u0 + win + 1)
    sub = blur[y0:y1, x0:x1]
    sm = m[y0:y1, x0:x1]
    if not sm.any():
        sm = np.ones_like(sub, bool)
    bg = float(np.percentile(sub[sm], 40))
    wgt = np.clip(sub - bg, 0.0, None) * sm
    if float(wgt.sum()) <= 0.0:
        return {"confidence": "none"}

    yy, xx = np.mgrid[y0:y1, x0:x1]
    cu = float((wgt * xx).sum() / wgt.sum())
    cv = float((wgt * yy).sum() / wgt.sum())
    d2 = (xx - cu) ** 2 + (yy - cv) ** 2
    sigma = float(np.sqrt((wgt * d2).sum() / wgt.sum()) + 1e-9)
    peak = float(sub.max())
    ref = float(np.median(L[m])) if m.any() else peak
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d = np.array([(cu - cx) / fx, (cv - cy) / fy, 1.0])
    d = d / np.linalg.norm(d)
    ang_deg = float(np.degrees(2.0 * sigma / fx))
    return {"confidence": "detected", "u": cu, "v": cv, "sigma_px": sigma,
            "dir_cam": [float(d[0]), float(d[1]), float(d[2])],
            "rel_intensity": float(peak / (ref + 1e-9)),
            "angular_deg": ang_deg}


# ═══════════════════════════════════════════════════════════════════
# 组合
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Lighting:
    sky_zenith_rel: float = 1.0
    sky_horizon_rel: float = 1.0
    sky_ratio: float = 1.0
    sky_rmse: float = 0.0
    sky_confidence: str = "unconstrained"
    sun_dir_cam: Optional[Tuple[float, float, float]] = None
    sun_rel_intensity: float = 0.0
    sun_sigma_px: float = 0.0
    sun_angular_deg: float = 0.0
    sun_confidence: str = "none"
    sky_tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # —— 主光（室内/无天空时由面法线+shading 反解）——
    key_dir_cam: Optional[Tuple[float, float, float]] = None
    key_ambient: float = 0.0
    key_intensity: float = 0.0
    key_confidence: str = "none"      # fitted-faces | none

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 6)
        return d


def fit_key_light(face_shading, face_normals_cam):
    """由"各面平均 shading + 各面法线"反解主光方向。

    模型: s_f ≈ a + k·max(0, n_f·L)      （a 环境项，k 光强）
    方向在半球上网格搜索；a,k 对每个候选方向做线性最小二乘（k≥0）。
    比"找最亮像素"物理得多：亮斑位置不是光的方向，法线分布才是。
    """
    s = np.asarray(face_shading, float)
    N = np.asarray(face_normals_cam, float)
    if s.size < 2 or N.shape != (s.size, 3):
        return {"confidence": "none"}
    best = None
    for az_deg in range(0, 360, 15):
        az = np.radians(az_deg)
        for el_deg in range(-10, 91, 10):
            el = np.radians(el_deg)
            L = np.array([np.cos(el) * np.cos(az), -np.sin(el), np.cos(el) * np.sin(az)])
            lam = np.clip(N @ L, 0.0, None)
            A = np.stack([np.ones_like(lam), lam], axis=1)
            coef, *_ = np.linalg.lstsq(A, s, rcond=None)
            a0, k = float(coef[0]), float(coef[1])
            if k <= 0.0:
                k = 0.0
                a0 = float(np.mean(s))
            pred = a0 + k * lam
            rmse = float(np.sqrt(np.mean((s - pred) ** 2)))
            if best is None or rmse < best["rmse"]:
                best = {"rmse": rmse, "dir_cam": L.tolist(), "ambient": a0,
                        "intensity": k, "confidence": "fitted-faces"}
    if best is None:
        return {"confidence": "none"}
    return best



def estimate_lighting(lin_rgb: np.ndarray, K: np.ndarray, horizon_row: int,
                      sky_mask: Optional[np.ndarray] = None, *,
                      shading: Optional[np.ndarray] = None,
                      face_normals=None, face_means=None) -> Lighting:
    """一次拿到天空/太阳/主光；看不到天空就如实降级。

    · 有天空: 拟合 L(e)=A+B·sinе + 在天空里找太阳盘
    · 没天空: 用"各面法线 + 各面平均 shading"反解**主光方向**（fit_key_light）
    """
    lin = np.asarray(lin_rgb, dtype=np.float64)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    sky = fit_sky_luminance(lum, K, horizon_row, sky_mask)
    usable_sky = sky_mask is not None and bool(np.asarray(sky_mask, bool).any())
    sun = estimate_sun(lum, K, sky_mask) if usable_sky else {"confidence": "none"}

    key = {"confidence": "none"}
    if (not usable_sky and face_normals is not None and face_means is not None
            and len(face_means) >= 2):
        key = fit_key_light(face_means, face_normals)

    # 天色调（高空区域均值，归一化到亮度1）
    top = lin[: max(1, int(horizon_row * 0.5))]
    tint = top.reshape(-1, 3).mean(axis=0) if top.size else np.ones(3)
    tl = float(tint @ np.array([0.2126, 0.7152, 0.0722])) + 1e-9
    tint = np.round(tint / tl, 6)

    return Lighting(
        sky_zenith_rel=float(sky.get("A", 0.0) + sky.get("B", 0.0)),
        sky_horizon_rel=float(sky.get("A", 0.0)),
        sky_ratio=float(sky.get("ratio", 1.0)),
        sky_rmse=float(sky.get("rmse", 0.0)),
        sky_confidence=str(sky.get("confidence", "unconstrained")),
        sun_dir_cam=tuple(sun["dir_cam"]) if sun.get("confidence") == "detected" else None,
        sun_rel_intensity=float(sun.get("rel_intensity", 0.0)),
        sun_sigma_px=float(sun.get("sigma_px", 0.0)),
        sun_angular_deg=float(sun.get("angular_deg", 0.0)),
        sun_confidence=str(sun.get("confidence", "none")),
        sky_tint=(float(tint[0]), float(tint[1]), float(tint[2])),
        key_dir_cam=(tuple(key["dir_cam"]) if key.get("confidence") == "fitted-faces"
                     else None),
        key_ambient=float(key.get("ambient", 0.0)),
        key_intensity=float(key.get("intensity", 0.0)),
        key_confidence=str(key.get("confidence", "none")),
    )


# ═══════════════════════════════════════════════════════════════════
# 等距柱状环境贴图（反射环境用）
# ═══════════════════════════════════════════════════════════════════
def equirect_from_lighting(lighting: Lighting, size: Tuple[int, int] = (256, 128)
                           ) -> np.ndarray:
    """由光照模型生成等距柱状环境（线性 RGB，write_hdr 可直接写）。"""
    w, h = int(size[0]), int(size[1])
    v = (np.arange(h, dtype=np.float64) + 0.5) / h      # 0 顶 → 1 底
    elev = np.sin(np.radians(90.0 - 180.0 * v))          # +1 天顶 → -1 天底
    A = max(lighting.sky_horizon_rel, 1e-3)
    B = max(lighting.sky_zenith_rel - lighting.sky_horizon_rel, 0.0)
    prof = A + B * np.clip(elev, 0.0, 1.0)
    prof = np.where(elev >= 0.0, prof, 0.35 * A)        # 地面半球压暗
    img = prof[:, None, None] * np.asarray(lighting.sky_tint, np.float64)[None, None, :]
    img = np.repeat(img, w, axis=1)
    if lighting.sun_confidence == "detected" and lighting.sun_rel_intensity > 0.0:
        u_s = (np.arange(w, dtype=np.float64) + 0.5) / w
        phi_s = np.arctan2(lighting.sun_dir_cam[0], lighting.sun_dir_cam[2]) / (2 * np.pi) + 0.5
        dphi = np.minimum(np.abs(u_s - phi_s), 1.0 - np.abs(u_s - phi_s)) * 2 * np.pi
        s_sig = max(np.radians(lighting.sun_angular_deg) / 2.355, 1e-4)
        blob = np.exp(-0.5 * (dphi[None, :] / s_sig) ** 2) * 0.8
        img = img + blob[..., None] * max(lighting.sun_rel_intensity, 0.0)
    return np.clip(img, 0.0, None)