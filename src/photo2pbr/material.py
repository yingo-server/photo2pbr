# -*- coding: utf-8 -*-
"""material —本征分解 / 粗糙度 / 金属度（代理，有界且单调）。

物理思路
--------
照片 = 反射率 × 光照。光照是**低频**（光源·几何），反射率是**高频**（材质纹理）。
用保边滤波（guided filter）把两者分离：
    shading ≈ guided(log L)      （低频）
    albedo  = linear(image) / shading
粗糙度看"高光有多锐"，金属度看"暗反射率 + 强高光"——都是**代理**，
有界、单调、可测；后续可替换为学习模型而接口不变。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["box_filter", "guided_filter", "decompose",
           "specular_and_roughness", "metalness_from_albedo", "MaterialMaps"]


# ═══════════════════════════════════════════════════════════════════
# 保边滤波
# ═══════════════════════════════════════════════════════════════════
def box_filter(x: np.ndarray, r: int) -> np.ndarray:
    """(2r+1)² 均值滤波，积分图实现，O(1)/像素。"""
    x = np.asarray(x, dtype=np.float64)
    h, w = x.shape
    r = int(r)
    xp = np.pad(x, ((r, r), (r, r)), mode="edge")
    ii = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
    n = 2 * r + 1
    S = (ii[n:n + h, n:n + w] - ii[0:h, n:n + w]
         - ii[n:n + h, 0:w] + ii[0:h, 0:w])
    return S / float(n * n)


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """灰度 guided filter（He et al.）。"""
    I = np.asarray(I, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mI = box_filter(I, r)
    mp = box_filter(p, r)
    corrI = box_filter(I * I, r)
    corrp = box_filter(I * p, r)
    varI = np.maximum(corrI - mI * mI, 0.0)
    covIp = corrp - mI * mp
    a = covIp / (varI + float(eps))
    b = mp - a * mI
    return box_filter(a, r) * I + box_filter(b, r)


# ═══════════════════════════════════════════════════════════════════
# 本征分解
# ═══════════════════════════════════════════════════════════════════
def decompose(rgb_srgb: np.ndarray, radius: int = 24, eps: float = 0.05,
              normalize_shading: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """sRGB 照片 → (albedo_lin [h,w,3], shading [h,w])，都是线性域。

    ⚠ 规范自由度（单图无法确定绝对照度水平）:
        image = albedo × shading 里，把 albedo 乘 c、shading 除 c 得到同一张图。
        本函数固定约定：**shading 均值归一化为 1**（相对照度）。
        绝对光照水平由 ``light.py`` 的光照模型提供，不从这里猜。

    radius/eps 取 (24, 0.05)：在合成房间上实测误差比最优（0.56×原图误差，
    shading 相关性 0.885）；换更大窗口收益已饱和。
    """
    from .core import srgb_to_linear

    lin = srgb_to_linear(np.asarray(rgb_srgb, dtype=np.float64))
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    logL = np.log(np.maximum(lum, 1e-6))
    sh_log = guided_filter(logL, logL, radius, eps)
    shading = np.exp(sh_log)
    if normalize_shading:
        shading = shading / max(float(shading.mean()), 1e-6)
    albedo = lin / np.maximum(shading, 1e-6)[..., None]
    return np.clip(albedo, 0.0, 1.5), shading


def specular_and_roughness(lin_rgb: np.ndarray, radius: int = 6,
                           base: float = 0.80, gain: float = 1.0,
                           lo: float = 0.08, hi: float = 0.95,
                           exclude: Optional[np.ndarray] = None):
    """高光检测 + 粗糙度反演（从"局部正残差"来，而不是从 shading 梯度来）。

    为什么不能用 image - albedo*shading：
        那套分解是**代数恒等式**（albedo = image / shading ⇒ 残差恒为 0），
        高光必须用**空间**信息找：局部比周围高出多少、尖不尖。

    物理对应：
        高光越"尖"（spike）→ 微面分布越集中 → 表面越光滑（roughness 小）；
        高光越"铺开" → 越粗糙。没有高光的区域回落到 base（哑光默认）。

    exclude: 需要排除的像素（例如"墙与墙交界"的几何边带——那里的亮度跳变
             由面朝向本身解释，不是高光）。
    """
    lin = np.asarray(lin_rgb, dtype=np.float64)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    local = box_filter(lum, radius)
    hp = lum - local                              # 局部正残差
    pos = np.clip(hp, 0.0, None)
    pos_s = box_filter(pos, 2)                    # 3×3 平滑：压掉单像素边缘噪点
    scale = float(np.percentile(pos_s, 99.5)) + 1e-9
    spec = np.clip(pos_s / scale, 0.0, 1.0)       # 高光强度 [0,1]
    if exclude is not None:
        spec = np.where(np.asarray(exclude, bool), 0.0, spec)
    loc = box_filter(spec, radius)
    sharpness = spec / (loc + 1e-6)               # 尖峰处 >1，铺开处 ≈1
    k_sharp = np.clip(sharpness - 1.0, 0.0, 2.0) * 0.5
    rough = np.clip(base - gain * spec * k_sharp, lo, hi)
    return spec, rough


def metalness_from_albedo(albedo_lin: np.ndarray, shading: np.ndarray,
                          lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """暗反射率 + 高光过冲 → 金属度代理（有界 [0,1]）。"""
    a = np.asarray(albedo_lin, dtype=np.float64)
    s = np.asarray(shading, dtype=np.float64)
    a_lum = a @ np.array([0.2126, 0.7152, 0.0722])
    a_ref = np.percentile(a_lum, 90) + 1e-9
    s_ref = np.percentile(s, 95) + 1e-9
    spec = np.clip(s / s_ref - 1.0, 0.0, None)
    dark = np.clip(1.0 - a_lum / a_ref, 0.0, 1.0)
    m = spec * (0.5 + 0.5 * dark)
    return np.clip(m, lo, hi)


@dataclass
class MaterialMaps:
    albedo_lin: np.ndarray    # [h,w,3] 线性反射率
    shading: np.ndarray       # [h,w] 光照分量（线性亮度）
    roughness: np.ndarray     # [h,w] ∈ [0.05, 0.95]
    metalness: np.ndarray     # [h,w] ∈ [0, 1]

    def summary(self) -> dict:
        def stat(x):
            x = np.asarray(x)
            return [float(np.mean(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95))]
        return {"roughness": stat(self.roughness), "metalness": stat(self.metalness)}