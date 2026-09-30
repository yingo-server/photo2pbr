# -*- coding: utf-8 -*-
"""material + light 测试：本征分解、粗糙度/金属度单调性、天空拟合、太阳盘。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photo2pbr import core, geom, light as lightmod, material, synth  # noqa: E402


# ── 本征分解 ──────────────────────────────────────────────────────
def test_decompose_recovers_albedo_better_than_raw():
    """规范不变比较：归一化后 albedo 的形状误差必须显著优于原图。

    单图无法确定绝对照度水平（见 decompose 的规范自由度说明），
    所以这里比的是"形状"（各自除以均值），不是绝对刻度。
    """
    r = synth.make_room(width=200, height=150, dims=(4.0, 2.6, 5.0), fov_deg=100.0)
    alb, sh = material.decompose(r.image)
    m = r.valid
    gt = r.albedo_lin[m]
    raw = core.srgb_to_linear(r.image)[m]
    est = alb[m]

    def _norm(x):
        return x / (float(np.mean(x)) + 1e-9)

    err_raw = float(np.mean(np.abs(_norm(raw) - _norm(gt))))
    err_est = float(np.mean(np.abs(_norm(est) - _norm(gt))))
    assert err_est < 0.65 * err_raw, "本征分解没有改善：raw=%.4f est=%.4f" % (err_raw, err_est)   # 实测 0.50
    corr = float(np.corrcoef(sh[m], r.shading_lin[m])[0, 1])
    assert corr > 0.75, "shading 相关性太低：%.3f" % corr                                        # 实测 0.897


def test_decompose_shading_is_low_frequency():
    r = synth.make_room(width=160, height=120)
    _, sh = material.decompose(r.image)
    m = r.valid
    expect = r.shading_lin[m]
    assert float(np.corrcoef(sh[m], expect)[0, 1]) > 0.3


# ── 粗糙度 / 金属度 ───────────────────────────────────────────────
def _blob(h, w, sigma, cx=60.0, cy=40.0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / (sigma * sigma))


def test_specular_roughness_sharp_vs_soft():
    """高光反演：尖高光 → 更光滑；铺开高光 → 更粗糙；无高光处落到 base。"""
    h, w = 80, 120
    sharp = _blob(h, w, 2.0, cx=40.0, cy=40.0)
    soft = _blob(h, w, 9.0, cx=90.0, cy=40.0)
    lum = 0.35 + 0.45 * sharp + 0.45 * soft
    lin = np.dstack([lum, lum, lum])
    spec, rough = material.specular_and_roughness(lin, radius=6)
    assert spec.min() >= 0.0 and spec.max() <= 1.0
    assert float(spec.max()) > 0.5, "高光没检出来"
    r_sharp = float(rough[40, 40])
    r_soft = float(rough[40, 90])
    r_base = float(rough[2, 2])
    assert r_sharp < r_soft, "尖高光应更光滑：%.3f vs %.3f" % (r_sharp, r_soft)
    assert abs(r_base - 0.80) < 0.06, "无高光处应回到 base：%.3f" % r_base
    assert rough.min() >= 0.08 - 1e-9 and rough.max() <= 0.95 + 1e-9


def test_metalness_bounds_and_dark_patch_boost():
    h, w = 80, 120
    patch = _blob(h, w, 8.0) > 0.5
    albedo = np.full((h, w, 3), 0.8)
    albedo[patch] = 0.15
    shading = 1.0 + 0.8 * _blob(h, w, 8.0)
    m = material.metalness_from_albedo(albedo, shading)
    assert m.min() >= 0.0 - 1e-9 and m.max() <= 1.0 + 1e-9
    assert float(m[patch].mean()) > float(m[~patch].mean())


# ── 天空 / 太阳 ───────────────────────────────────────────────────
def _sky_canvas(h=200, w=160, A=0.3, B=0.6, horizon=140):
    """纯 L(e)=A+B·sin(e)（地平线以下另一常数）——用于验证拟合本身。"""
    K = geom.intrinsics(w, h, 60.0)
    sin_e = lightmod._row_sin_elevation(K, h)
    lum = np.zeros((h, w))
    for v in range(h):
        lum[v] = (A + B * sin_e[v]) if v < horizon else 0.15
    return lum, K, horizon


def test_sky_fit_recovers_linear_profile():
    lum, K, horizon = _sky_canvas()
    fit = lightmod.fit_sky_luminance(lum, K, horizon)
    assert fit["confidence"] == "fitted"
    assert abs(fit["A"] - 0.3) < 0.02, "A=%.4f" % fit["A"]
    assert abs(fit["ratio"] - 3.0) < 0.15, "ratio=%.3f" % fit["ratio"]


def test_sun_detection_position_and_strength():
    lum, K, horizon = _sky_canvas()
    u0, v0 = 110.0, 40.0
    y, x = np.mgrid[0:lum.shape[0], 0:lum.shape[1]].astype(np.float64)
    blob = 2.0 * np.exp(-0.5 * (((x - u0) ** 2 + (y - v0) ** 2) / (3.0 ** 2)))
    lum = lum + blob
    sky_mask = np.zeros_like(lum, bool)
    sky_mask[:horizon] = True
    s = lightmod.estimate_sun(lum, K, sky_mask)
    assert s["confidence"] == "detected"
    assert abs(s["u"] - u0) < 2.0 and abs(s["v"] - v0) < 2.0, "太阳位置偏离"
    assert 1.5 < s["sigma_px"] < 8.0
    assert s["rel_intensity"] > 1.0, "太阳亮度应远高于天空背景"


def test_lighting_unconstrained_without_sky():
    # 整幅都是"墙"（没有天空）→ 必须如实降级
    h, w = 60, 80
    K = geom.intrinsics(w, h, 60.0)
    img = np.full((h, w, 3), 0.2)
    lit = lightmod.estimate_lighting(img, K, horizon_row=0)
    assert lit.sky_confidence == "unconstrained"
    d = lit.to_dict()
    assert d["sun_confidence"] in ("none", "detected")


def test_equirect_env_is_nonnegative_and_shaped():
    h, w = 32, 64
    K = geom.intrinsics(w, h, 60.0)
    lum, K2, horizon = _sky_canvas(h, w)
    img = np.stack([lum] * 3, axis=-1)
    lit = lightmod.estimate_lighting(img, K2, horizon)
    env = lightmod.equirect_from_lighting(lit, size=(64, 32))
    assert env.shape == (32, 64, 3)
    assert env.min() >= 0.0
    assert env.max() > env.min()