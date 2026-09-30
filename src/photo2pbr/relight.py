# -*- coding: utf-8 -*-
"""relight —用估计出的场景**重新渲染**（验证 + 出图）。

同一套几何/材质/光照，可以做两件事:
    1. 同视角重渲 → 与输入对照，验证估计是否自洽（round-trip 误差）
    2. 换光重渲（太阳转 30°）→ 证明环境是"物理的"，不是贴图拼的

实现要点
--------
* 房间系：相机为原点（与 RoomEstimate 一致），轴 = axes 的行
* 逐像素：光线-盒求交 → 面 id + 面内 uv → 从"面贴图"采样 albedo
* 面贴图：由估计 albedo 按同一映射回填（没看见的 texel 用该面均值补）
* 光照：ambient + sun·max(0, n·L)，法线 = 该面朝向房间内部的方向
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import core, geom

FACES = ("left", "right", "floor", "ceiling", "back", "front")
DEFAULT_TEX = 256


# ═══════════════════════════════════════════════════════════════════
# 几何辅助
# ═══════════════════════════════════════════════════════════════════
def face_layout(axes: np.ndarray, bmin, bmax, roles: Dict[str, int]):
    """面 → (轴索引, 端, 面内切向 a1, a2)。端 'lo'/'hi' 指该轴上的边界。"""
    vert, horiz, depth = roles["vertical"], roles["horizontal"], roles["depth"]
    if float(axes[vert] @ np.array([0.0, 1.0, 0.0])) > 0.0:
        face_end = {"floor": (vert, "hi"), "ceiling": (vert, "lo")}
    else:
        face_end = {"floor": (vert, "lo"), "ceiling": (vert, "hi")}
    if float(axes[horiz] @ np.array([1.0, 0.0, 0.0])) > 0.0:
        face_end.update({"right": (horiz, "hi"), "left": (horiz, "lo")})
    else:
        face_end.update({"right": (horiz, "lo"), "left": (horiz, "hi")})
    back_is_lo = abs(bmin[depth]) > abs(bmax[depth])
    face_end["back"] = (depth, "lo" if back_is_lo else "hi")
    face_end["front"] = (depth, "hi" if back_is_lo else "lo")
    out = {}
    for name, (ax, end) in face_end.items():
        others = [i for i in range(3) if i != ax]
        out[name] = (ax, end, others[0], others[1])
    return out


def _key_of(ax: int, end: str) -> int:
    """与 dist 的构造顺序一致：[xlo, ylo, zlo, xhi, yhi, zhi]"""
    return ax if end == "lo" else ax + 3


# ═══════════════════════════════════════════════════════════════════
# 贴字（画进面贴图 → 渲染时透视/朝向/光照自动正确）
# ═══════════════════════════════════════════════════════════════════
def _load_font(path, size):
    """尽量找一个能用的 TTF；找不到退回 PIL 位图字体。"""
    from PIL import ImageFont

    cands = [path] if path else []
    cands += [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/system/fonts/NotoSansCJK-Regular.ttc",
        "/sdcard/render/fonts/AaXiuKai-2.ttf",
    ]
    for c in cands:
        try:
            if c and Path(c).exists():
                return ImageFont.truetype(str(c), int(size))
        except Exception:
            pass
    return ImageFont.load_default()


def _text_layout(layout, name, axes):
    """决定文字的两个轴怎么映射到墙面的 (u,v)（含 u/v 交换与正负）。

    返回 (mode, sx, sy)：
        mode="uv" → 文字 x 沿 u，文字 y 沿 v
        mode="vu" → 文字 x 沿 v，文字 y 沿 u（需要转置）
        sx: 文字 x 方向相对"画面右"是否同向；sy: 文字 y 相对"画面下"是否同向
    """
    _ax, _end, a1, a2 = layout[name]
    right = np.array([1.0, 0.0, 0.0])
    down = np.array([0.0, 1.0, 0.0])
    u_dir = np.asarray(axes[a1], float)
    v_dir = np.asarray(axes[a2], float)
    if abs(float(u_dir @ right)) >= abs(float(v_dir @ right)):
        # u 更像"水平方向"→ 文字 x 沿 u
        sx = 1 if float(u_dir @ right) >= 0.0 else -1
        sy = 1 if float(v_dir @ down) >= 0.0 else -1
        return "uv", sx, sy
    # v 更像"水平方向"→ 文字 x 沿 v（转置存储）
    sx = 1 if float(v_dir @ right) >= 0.0 else -1
    sy = 1 if float(u_dir @ down) >= 0.0 else -1
    return "vu", sx, sy


def apply_text(texes, layout, model, *, text, font_path=None, face="auto",
               size_frac: float = 0.42, ink=(0.06, 0.05, 0.05),
               ink_alpha: float = 0.95, margin: float = 0.10,
               angle_deg: float = 0.0, spacing: float = 0.15):
    """把文字画到某个面的贴图上（自动处理轴向/长宽比/旋转/多行/缩放）。

    渲染时按 uv 采样 → 透视/朝向自动正确；且文字随该面一起受光。
    angle_deg: 在**墙面平面内**旋转（沿墙倾斜）。
    """
    from PIL import Image, ImageDraw

    if not text:
        return None
    masks = model.get("face_masks") or {}
    if face == "auto":
        face = (max(masks, key=lambda k: float(np.mean(masks[k]))) if masks
                else "back")
    if face not in layout:
        face = "back"

    geo = model["geometry"]
    axes = np.asarray(geo["axes"], float)
    ext = (np.asarray(geo["bounds_max"], float)
           - np.asarray(geo["bounds_min"], float))
    mode, sx, sy = _text_layout(layout, face, axes)
    _ax, _end, a1, a2 = layout[face]
    if mode == "uv":
        w_x, w_y = float(ext[a1]), float(ext[a2])
    else:
        w_x, w_y = float(ext[a2]), float(ext[a1])
    aspect = max(0.2, w_x / max(w_y, 1e-9))         # 墙的物理长宽比（文字坐标系）
    tex = int(texes.shape[1])
    W_d = max(32, int(round(tex * aspect)))
    H_d = tex

    fs = max(8, int(size_frac * H_d))
    font = _load_font(font_path, fs)
    gap = int(fs * float(spacing))
    txt = str(text)
    # 1) 先渲染到"刚好包住文字"的紧致画布（多行用 multiline_*）
    probe = Image.new("L", (8, 8), 0)
    dp = ImageDraw.Draw(probe)
    try:
        box = dp.multiline_textbbox((0, 0), txt, font=font, spacing=gap)
    except Exception:
        box = (0, 0, fs * max(len(txt.splitlines()), 1), fs)
    tw, th = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    pad = max(2, int(0.12 * fs))
    tight = Image.new("L", (tw + 2 * pad, th + 2 * pad), 0)
    dt = ImageDraw.Draw(tight)
    try:
        dt.multiline_text((pad - box[0], pad - box[1]), txt, fill=255,
                          font=font, spacing=gap, align="center")
    except Exception:
        dt.text((pad, pad), txt, fill=255, font=font)
    # 2) 墙面平面内旋转
    if abs(float(angle_deg)) > 1e-6:
        tight = tight.rotate(float(angle_deg), resample=Image.BICUBIC, expand=True)
    # 3) 缩进到墙面画布（留边距）
    lim_w = (1.0 - 2.0 * margin) * W_d
    lim_h = (1.0 - 2.0 * margin) * H_d
    sc = min(1.0, lim_w / max(tight.width, 1), lim_h / max(tight.height, 1))
    if sc < 1.0:
        tight = tight.resize((max(1, int(tight.width * sc)),
                              max(1, int(tight.height * sc))), Image.LANCZOS)
    canvas = Image.new("L", (W_d, H_d), 0)
    canvas.paste(tight, ((W_d - tight.width) // 2, (H_d - tight.height) // 2))
    # 4) 压成方形存储（与采样时的拉伸互抵）
    canvas = canvas.resize((tex, tex), Image.LANCZOS)
    a = np.asarray(canvas, np.float64) / 255.0
    if mode == "vu":
        a = a.T                                  # 文字 x 沿 v → 转置
        if sx < 0:
            a = a[::-1, :]
        if sy < 0:
            a = a[:, ::-1]
    else:
        if sx < 0:
            a = a[:, ::-1]
        if sy < 0:
            a = a[::-1, :]
    idx = _key_of(*layout[face][:2])
    m = a[..., None] * float(ink_alpha)
    ink_arr = np.asarray(ink, float).reshape(1, 1, 3)
    texes[idx] = texes[idx] * (1.0 - m) + ink_arr * m
    return face


def _sample_tex(tex: np.ndarray, u, v):
    """双线性采样（u,v ∈ [0,1]，对单个面）→ 更少块状。"""
    T = int(tex.shape[0])
    x = np.clip(np.asarray(u, float) * (T - 1), 0, T - 1)
    y = np.clip(np.asarray(v, float) * (T - 1), 0, T - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, T - 1)
    y1 = np.minimum(y0 + 1, T - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy)
            + tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)


def _box_form_factors(layout, bmin, bmax, k: int = 8):
    """盒体内的面-面形状因子 F_ij（数值积分；凸盒内无遮挡）。

    返回 (names, F, A)：F[i, j] = 面 i 看到面 j 的比例；F 行和 ≤ 1。
    """
    items = sorted(layout.items(), key=lambda kv: _key_of(*kv[1][:2]))
    names = [kv[0] for kv in items]
    bmin = np.asarray(bmin, float)
    bmax = np.asarray(bmax, float)

    P, Nn, dA, fid = [], [], [], []
    for idx, (name, (ax, end, a1, a2)) in enumerate(items):
        plane = bmin[ax] if end == "lo" else bmax[ax]
        u = np.linspace(bmin[a1], bmax[a1], k, endpoint=False) + (
            (bmax[a1] - bmin[a1]) / (2 * k))
        v = np.linspace(bmin[a2], bmax[a2], k, endpoint=False) + (
            (bmax[a2] - bmin[a2]) / (2 * k))
        uu, vv = np.meshgrid(u, v)
        p = np.zeros((k * k, 3))
        p[:, ax] = plane
        p[:, a1] = uu.ravel()
        p[:, a2] = vv.ravel()
        n = np.zeros(3)
        n[ax] = 1.0 if end == "lo" else -1.0
        P.append(p)
        Nn.append(np.tile(n, (k * k, 1)))
        dA.append(np.full(k * k, ((bmax[a1] - bmin[a1]) * (bmax[a2] - bmin[a2]))
                           / (k * k)))
        fid.append(np.full(k * k, idx))
    P = np.concatenate(P)
    Nn = np.concatenate(Nn)
    dA = np.concatenate(dA)
    fid = np.concatenate(fid)

    F = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        mi = fid == i
        Pi, Ni = P[mi], Nn[mi]
        Ai = float(dA[mi].sum())
        for j in range(len(names)):
            mj = fid == j
            Pj, Nj, dAj = P[mj], Nn[mj], dA[mj]
            dv = Pj[None, :, :] - Pi[:, None, :]
            r2 = np.maximum(np.sum(dv * dv, axis=-1), 1e-6)
            r = np.sqrt(r2)
            cos_i = np.clip(np.sum(Ni[:, None, :] * dv, axis=-1) / r, 0.0, None)
            cos_j = np.clip(np.sum(Nj[None, :, :] * (-dv), axis=-1) / r, 0.0, None)
            F[i, j] = float(np.sum(cos_i * cos_j / (np.pi * r2) * dAj[None, :])
                            / max(Ai, 1e-9))
    # 数值保护：近壁处离散化会让行和略超 1 → 归一到 ≤1（能量守恒）
    row = F.sum(axis=1, keepdims=True)
    F = F / np.where(row > 1.0, row, 1.0)
    return names, F, dA


def _radiosity_solve(direct, rho, F, iters: int = 8):
    """一次能量分配：B = E + ρ·(F@B)，迭代收敛（房间里的多次互反射）。"""
    B = np.asarray(direct, float).copy()
    rho = np.asarray(rho, float)
    for _ in range(int(iters)):
        B = np.asarray(direct, float) + rho * (F @ B)
    return B


def _point_light_shading(pts, fid, layout, L_room, ambient, intensity, bmin, bmax,
                         match_face_means: bool = True):
    """近距光源（像窗口/灯）：1/r² 衰减 + 定向余弦 → 光斑与梯度。

    match_face_means=True 时，把每个面的**平均亮度**拉回到拟合光的一致水平
    （只改面内的分布，不改整面曝光）——这样"换灯"不会让画面整体忽明忽暗。
    """
    n_pix = np.zeros_like(pts)
    for name, (ax, end, _a1, _a2) in layout.items():
        m = fid == _key_of(ax, end)
        if not m.any():
            continue
        n = np.zeros(3)
        n[ax] = 1.0 if end == "lo" else -1.0
        n_pix[m] = n
    center = 0.5 * (np.asarray(bmin, float) + np.asarray(bmax, float))
    lamp = center + L_room * (0.25 * float(np.min(np.asarray(bmax) - np.asarray(bmin))))
    dv = lamp[None, None, :] - pts
    r2 = np.maximum(np.sum(dv * dv, axis=-1), 0.05 ** 2)
    dro = dv / np.sqrt(r2)[..., None]
    cos = np.clip(np.sum(n_pix * dro, axis=-1), 0.0, None)
    raw = cos / r2
    lam_dir = np.clip(np.sum(n_pix * L_room[None, None, :], axis=-1), 0.0, None)
    target = ambient + intensity * float(lam_dir.mean())
    scale = max(target - ambient, 0.0) / max(float(raw.mean()), 1e-9)
    shading = ambient + scale * raw
    if match_face_means:
        for name, (ax, end, _a1, _a2) in layout.items():
            m = fid == _key_of(ax, end)
            if not m.any():
                continue
            n = np.zeros(3)
            n[ax] = 1.0 if end == "lo" else -1.0
            tgt = ambient + intensity * max(0.0, float(np.dot(n, L_room)))
            got = float(shading[m].mean())
            if got > 1e-6:
                shading[m] *= tgt / got
    return shading


def _face_geometry(model: dict):
    geo = model["geometry"]
    axes = np.asarray(geo["axes"], float)
    bmin = np.asarray(geo["bounds_min"], float)
    bmax = np.asarray(geo["bounds_max"], float)
    w, h = int(geo["size"][0]), int(geo["size"][1])
    K = np.asarray(geo["K"], float)
    d_cam = geom.ray_dirs(K, w, h)
    d_room = d_cam @ axes.T
    t, valid = geom.ray_box_t(np.zeros(3), d_room, bmin, bmax)
    p_box = t[..., None] * d_room
    depth = model.get("depth")
    if depth is not None and np.asarray(depth).size:
        pts_cam = geom.backproject(np.asarray(depth, float), K)
        pts = pts_cam @ axes.T
        ok = np.asarray(depth, float) > 0.0
        pts = np.where(ok[..., None], pts, p_box)
    else:
        pts = p_box
    dist = np.stack([np.abs(pts[..., i] - bmin[i]) for i in range(3)]
                    + [np.abs(pts[..., i] - bmax[i]) for i in range(3)], axis=-1)
    fid = np.argmin(dist, axis=-1)
    return axes, bmin, bmax, w, h, K, pts, fid, valid


# ═══════════════════════════════════════════════════════════════════
# 面贴图（从估计 albedo 回填）
# ═══════════════════════════════════════════════════════════════════
def build_face_textures(model: dict, tex: int = DEFAULT_TEX):
    axes, bmin, bmax, w, h, K, pts, fid, valid = _face_geometry(model)
    layout = face_layout(axes, bmin, bmax, model["geometry"]["roles"])
    key_to_name = {}
    for name, (ax, end, _a1, _a2) in layout.items():
        key_to_name[_key_of(ax, end)] = name
    albedo = np.asarray(model["albedo"], float)
    texes = np.zeros((6, tex, tex, 3))
    cnts = np.zeros((6, tex, tex))
    for idx in range(6):
        name = key_to_name.get(idx)
        if name is None:
            continue
        m = fid == idx
        if not m.any():
            continue
        _ax, _end, a1, a2 = layout[name]
        u = (pts[..., a1][m] - bmin[a1]) / max(float(bmax[a1] - bmin[a1]), 1e-9)
        v = (pts[..., a2][m] - bmin[a2]) / max(float(bmax[a2] - bmin[a2]), 1e-9)
        ui = np.clip((u * (tex - 1)).round().astype(int), 0, tex - 1)
        vi = np.clip((v * (tex - 1)).round().astype(int), 0, tex - 1)
        np.add.at(texes, (idx, vi, ui), albedo[m])
        np.add.at(cnts, (idx, vi, ui), 1.0)
    for idx in range(6):
        c = cnts[idx]
        filled = c > 0
        mean = (texes[idx][filled].mean(axis=0) if filled.any()
                else np.array([0.5, 0.5, 0.5]))
        texes[idx][filled] /= c[filled][:, None]
        texes[idx][~filled] = mean
    return texes, layout, (bmin, bmax, pts, fid, w, h, axes)


# ═══════════════════════════════════════════════════════════════════
# 重渲
# ═══════════════════════════════════════════════════════════════════
def _rot_about_axis(v: np.ndarray, axis: int, deg: float) -> np.ndarray:
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    R = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R @ v


def relight(model: dict, *, tex: int = DEFAULT_TEX, sun_rot_deg: float = 0.0,
            ambient: Optional[float] = None, intensity: Optional[float] = None,
            light_model: str = "fitted", text: Optional[str] = None,
            font_path: Optional[str] = None, text_face: str = "auto",
            text_angle: float = 0.0, ink_alpha: float = 0.92,
            bounce: bool = True) -> np.ndarray:
    """用 model 重渲一张（同视角）。返回 sRGB float [h,w,3]。

    light_model: "fitted"=拟合光（每面常数，用于与输入对照）
                 "point" =近距光源（1/r²+余弦 → 光斑与梯度，更像窗口光）
    text:        把文字贴到某个面（透视/朝向自动正确，且随光照一起明暗）
    text_angle:  文字在墙面平面内旋转（沿墙倾斜）
    bounce:      point 模式下叠加"面与面互反射"（房间里光会来回弹）
    """
    texes, layout, (bmin, bmax, pts, fid, w, h, axes) = build_face_textures(model, tex)
    if text:
        apply_text(texes, layout, model, text=text, font_path=font_path,
                   face=text_face, angle_deg=text_angle, ink_alpha=ink_alpha)
    key_to_name = {}
    for name, (ax, end, _a1, _a2) in layout.items():
        key_to_name[_key_of(ax, end)] = name

    lit = model.get("lighting") or {}
    role_vert = model["geometry"]["roles"]["vertical"]
    if lit.get("sun_dir_cam"):
        L_cam = np.asarray(lit["sun_dir_cam"], float)
    elif lit.get("key_dir_cam"):
        L_cam = np.asarray(lit["key_dir_cam"], float)      # 室内：面法线反解的主光
    else:
        L_cam = np.array([0.35, -0.45, 0.82])
    L_cam = L_cam / max(float(np.linalg.norm(L_cam)), 1e-9)
    L_room = axes @ L_cam
    if sun_rot_deg:
        L_room = _rot_about_axis(L_room, role_vert, float(sun_rot_deg))
        L_room = L_room / max(np.linalg.norm(L_room), 1e-9)
    fitted_key = lit.get("key_confidence") == "fitted-faces"
    if ambient is None:
        ambient = (float(lit.get("key_ambient") or 0.25) if fitted_key
                   else float(max(lit.get("sky_horizon_rel", 0.3), 0.25)))
    if intensity is None:
        intensity = (float(lit.get("key_intensity") or 0.6) if fitted_key
                     else float(max(lit.get("sun_rel_intensity", 0.0), 0.6)))

    # —— 光照：默认"拟合光"（每面常数）；point=近距光源（光斑+梯度）
    if light_model == "point":
        shading = _point_light_shading(pts, fid, layout, L_room, ambient, intensity,
                                       bmin, bmax)
        if bounce:
            # 互反射：把"被照亮的面反弹回来的光"加到其它面上（盒体形状因子 + 辐射度迭代）
            names_, F, _A = _box_form_factors(layout, bmin, bmax, k=8)
            alb = np.asarray(model["albedo"], float)
            luma = alb @ np.array([0.2126, 0.7152, 0.0722])
            masks = model.get("face_masks") or {}
            direct_i, rho_i = [], []
            for nm in names_:
                m = fid == _key_of(*layout[nm][:2])
                direct_i.append(float(shading[m].mean()) if m.any() else 0.0)
                if nm in masks and np.asarray(masks[nm]).any():
                    rho_i.append(float(luma[np.asarray(masks[nm], bool)].mean()))
                else:
                    rho_i.append(0.5)
            B = _radiosity_solve(direct_i, rho_i, F)
            for nm, b_i, d_i in zip(names_, B, direct_i):
                m = fid == _key_of(*layout[nm][:2])
                if not m.any():
                    continue
                shading[m] = shading[m] * (b_i / d_i) if d_i > 1e-6 else b_i
    else:
        shading = np.zeros((h, w))
        for name, (ax, end, _a1, _a2) in layout.items():
            m = fid == _key_of(ax, end)
            if not m.any():
                continue
            n = np.zeros(3)
            n[ax] = 1.0 if end == "lo" else -1.0
            shading[m] = ambient + intensity * max(0.0, float(np.dot(n, L_room)))

    out = np.zeros((h, w, 3))
    for idx in range(6):
        name = key_to_name.get(idx)
        if name is None:
            continue
        m = fid == idx
        if not m.any():
            continue
        ax, end, a1, a2 = layout[name]
        u = (pts[..., a1][m] - bmin[a1]) / max(float(bmax[a1] - bmin[a1]), 1e-9)
        v = (pts[..., a2][m] - bmin[a2]) / max(float(bmax[a2] - bmin[a2]), 1e-9)
        alb = _sample_tex(texes[idx], u, v)          # 双线性采样
        out[m] = alb * shading[m][..., None]
    return core.linear_to_srgb(np.clip(out, 0.0, None))


# ═══════════════════════════════════════════════════════════════════
# bundle ↔ 对照表
# ═══════════════════════════════════════════════════════════════════
def load_bundle_model(bundle: str | Path):
    """从 bundle 里找缓存模型（cache/models/<key>/model.npz）。"""
    from . import pipeline
    bundle = Path(bundle)
    cands = sorted((bundle / "cache" / "models").glob("*/model.npz"))
    if not cands:
        return None
    return pipeline.load_model(cands[0].parent)


def compose_sheet(bundle: str | Path, out_path: str | Path,
                  width: int = 240) -> str:
    """把 输入 / 重渲 / 换光重渲 / 各贴图 / 深度 / 掩膜 拼成一张对照表。"""
    from PIL import Image, ImageDraw

    bundle = Path(bundle)
    model = load_bundle_model(bundle)
    if model is None:
        raise FileNotFoundError("bundle 里没有缓存模型：%s" % bundle)

    same = relight(model)
    moved = relight(model, sun_rot_deg=30.0)
    core.save_png(same, bundle / "relight.png")
    core.save_png(moved, bundle / "relight_sun30.png")

    import json
    scene = json.loads((bundle / "scene.json").read_text(encoding="utf-8"))
    src_name = scene.get("source", {}).get("path", "")
    tiles = []
    if src_name and (bundle / src_name).exists():
        img, _ = core.load_image(bundle / src_name)
        tiles.append(("input (photo)", img))
    tiles.append(("relight (estimated scene)", same))
    tiles.append(("relight, sun +30deg", moved))
    for fn, cap in (("albedo.png", "albedo (estimated)"),
                    ("shading.png", "shading (p99 normalized)"),
                    ("specular.png", "specular (highlight residual)"),
                    ("roughness.png", "roughness (inverted from spec)"),
                    ("metalness.png", "metalness"),
                    ("depth16.png", "depth (16bit)")):
        p = bundle / fn
        if p.exists():
            tiles.append((cap, _png_to_float(p)))
    masks = []
    for fn in ("masks/back.png", "masks/left.png", "masks/floor.png"):
        p = bundle / fn
        if p.exists():
            masks.append(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0)
    if masks:
        mm = np.dstack(masks + [np.zeros_like(masks[0])] * (3 - len(masks)))
        tiles.append(("masks (R=back G=left B=floor)", mm))

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    ar = model["geometry"]["size"][1] / max(model["geometry"]["size"][0], 1)
    tw = int(width)
    th = max(1, int(round(tw * ar)))
    cap_h = 14
    sheet = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + cap_h + 8) + 8),
                      (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (cap, arr) in enumerate(tiles):
        r, c = divmod(i, cols)
        x0 = 8 + c * (tw + 8)
        y0 = 8 + r * (th + cap_h + 8)
        a = np.clip(np.asarray(arr, np.float32), 0, 1)
        if a.ndim == 2:
            a = np.dstack([a] * 3)
        im = Image.fromarray((a * 255 + 0.5).astype(np.uint8)).resize((tw, th),
                                                                      Image.LANCZOS)
        sheet.paste(im, (x0, y0))
        draw.text((x0, y0 + th + 1), cap, fill=(0, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out_path))
    return str(out_path)


# ═══════════════════════════════════════════════════════════════════
# "给人看的"场景对照表（2 列 × 若干行，图大、话少）
# ═══════════════════════════════════════════════════════════════════
def _png_to_float(p: Path) -> np.ndarray:
    """读 PNG → float [0,1]，兼容 8bit 与 16bit（PIL 的 I;16 不能直接 convert）。"""
    from PIL import Image

    im = Image.open(p)
    a = np.asarray(im)
    if a.dtype == np.uint16:
        out = a.astype(np.float32) / 65535.0
    elif a.dtype == np.uint8:
        out = a.astype(np.float32) / 255.0
    else:
        out = a.astype(np.float32)
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[..., :3]
    return out


def draw_schematic(model: dict) -> np.ndarray:
    """把"房间盒体线框 + 相机 + 光方向"画成示意图（一眼看懂 3D）。"""
    from PIL import Image, ImageDraw

    geo = model["geometry"]
    w, h = int(geo["size"][0]), int(geo["size"][1])
    K = np.asarray(geo["K"], float)
    axes = np.asarray(geo["axes"], float)
    bmin = np.asarray(geo["bounds_min"], float)
    bmax = np.asarray(geo["bounds_max"], float)
    img = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(img)

    corners = []
    for bx in (bmin[0], bmax[0]):
        for by in (bmin[1], bmax[1]):
            for bz in (bmin[2], bmax[2]):
                corners.append(np.array([bx, by, bz]))

    def project(p_room):
        p_cam = axes.T @ p_room
        if p_cam[2] <= 5e-2:
            return None
        u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
        v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
        return float(u), float(v)

    # ── 画"站在房间里看到的结构"：远端矩形 + 四条向观者收敛的棱
    dep = int(geo["roles"]["depth"])
    back_val = bmin[dep] if abs(bmin[dep]) > abs(bmax[dep]) else bmax[dep]
    front_val = bmax[dep] if back_val == bmin[dep] else bmin[dep]
    others = [i for i in range(3) if i != dep]
    combos = [(bmin[others[0]], bmin[others[1]]),
              (bmax[others[0]], bmin[others[1]]),
              (bmax[others[0]], bmax[others[1]]),
              (bmin[others[0]], bmax[others[1]])]          # 环序，避免对角线
    back_corners = []
    for a_, b_ in combos:
        p = np.zeros(3)
        p[dep] = back_val
        p[others[0]] = a_
        p[others[1]] = b_
        back_corners.append(p)
    # 远端矩形（按环序连边）
    for i in range(4):
        p1 = project(back_corners[i])
        p2 = project(back_corners[(i + 1) % 4])
        if p1 and p2:
            d.line([p1, p2], fill=(150, 150, 150), width=2)
    # 四条纵向棱（从远端角向相机方向采样；出画就停，避免出现"大 X"）
    x_lim = (-0.6 * w, 1.6 * w)
    y_lim = (-0.6 * h, 1.6 * h)
    for bc in back_corners:
        fc = bc.copy()
        fc[dep] = front_val
        pts = []
        for t in np.linspace(0.0, 1.0, 60):
            p = bc + t * (fc - bc)
            pr = project(p)
            if pr is None:
                break
            inside = (x_lim[0] < pr[0] < x_lim[1]) and (y_lim[0] < pr[1] < y_lim[1])
            if not inside:
                break
            pts.append(pr)
        for i in range(len(pts) - 1):
            d.line([pts[i], pts[i + 1]], fill=(170, 170, 170), width=2)

    # 相机（主点）与光方向箭头
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(200, 60, 60))
    lit = model.get("lighting") or {}
    L_cam = None
    if lit.get("sun_dir_cam"):
        L_cam = np.asarray(lit["sun_dir_cam"], float)
    elif lit.get("key_dir_cam"):
        L_cam = np.asarray(lit["key_dir_cam"], float)
    if L_cam is not None:
        n = np.linalg.norm(L_cam[:2]) + 1e-9
        ex = cx + 90.0 * float(L_cam[0]) / n
        ey = cy + 90.0 * float(L_cam[1]) / n
        d.line([cx, cy, ex, ey], fill=(230, 150, 0), width=3)
        d.text((ex + 4, ey - 6), "key light", fill=(190, 110, 0))
    dims = bmax - bmin
    d.text((8, h - 16), "room %.1f x %.1f x %.1f m" % tuple(dims.tolist()),
           fill=(40, 40, 40))
    d.text((8, 6), "wireframe: estimated room box", fill=(80, 80, 80))
    return np.asarray(img, np.float32) / 255.0


def compose_light_study(bundle: str | Path, out_path: str | Path,
                        *, text: Optional[str] = None,
                        font_path: Optional[str] = None,
                        angles=(-40.0, -20.0, 0.0, 20.0, 40.0),
                        light_model: str = "point",
                        width: int = 300) -> str:
    """光的研究条带：同一个字/同一面墙，五档光角（用近距灯，光斑会扫过墙面）。"""
    from PIL import Image, ImageDraw

    bundle = Path(bundle)
    model = load_bundle_model(bundle)
    if model is None:
        raise FileNotFoundError("bundle 里没有缓存模型：%s" % bundle)

    imgs = []
    for a in angles:
        im = relight(model, text=text, font_path=font_path, sun_rot_deg=float(a),
                     light_model=light_model)
        imgs.append((("lamp %+d deg" % int(a)), im))
    ar = model["geometry"]["size"][1] / max(model["geometry"]["size"][0], 1)
    tw = int(width)
    th = max(1, int(round(tw * ar)))
    cap_h = 18
    n = len(imgs)
    sheet = Image.new("RGB", (n * (tw + 6) + 6, th + cap_h + 12), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    font = _load_font(font_path, 14)
    for i, (cap, arr) in enumerate(imgs):
        x0 = 6 + i * (tw + 6)
        a = np.clip(np.asarray(arr, np.float32), 0, 1)
        pil = Image.fromarray((a * 255 + 0.5).astype(np.uint8)).resize((tw, th),
                                                                      Image.LANCZOS)
        sheet.paste(pil, (x0, 6))
        try:
            d.text((x0, 6 + th + 2), cap, fill=(20, 20, 20), font=font)
        except Exception:
            d.text((x0, 6 + th + 2), cap, fill=(20, 20, 20))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out_path))
    return str(out_path)


def compose_scene_sheet(bundle: str | Path, out_path: str | Path,
                        *, text: Optional[str] = None,
                        font_path: Optional[str] = None,
                        text2: Optional[str] = None,
                        text_angle: float = 0.0,
                        width: int = 360) -> str:
    """输入照片 / 重渲 / 贴字 / 换光 / 近距灯 —— 一眼能看懂的那张。"""
    from PIL import Image, ImageDraw

    bundle = Path(bundle)
    model = load_bundle_model(bundle)
    if model is None:
        raise FileNotFoundError("bundle 里没有缓存模型：%s" % bundle)

    import json
    scene = json.loads((bundle / "scene.json").read_text(encoding="utf-8"))
    src_name = scene.get("source", {}).get("path", "")
    img_in = None
    if src_name and (bundle / src_name).exists():
        img_in, _ = core.load_image(bundle / src_name)

    img_re = relight(model)
    img_tx = (relight(model, text=text, font_path=font_path, text_angle=text_angle)
              if text else img_re)
    img_ts = (relight(model, text=text, font_path=font_path, sun_rot_deg=30.0,
                      text_angle=text_angle)
              if text else relight(model, sun_rot_deg=30.0))
    img_pt = relight(model, light_model="point", text=text, font_path=font_path,
                     text_angle=text_angle)

    # 第二张贴字：同一面墙、换个倾角（演示"沿墙旋转"）
    img_tx2 = None
    if text:
        img_tx2 = relight(model, text=text, font_path=font_path,
                          text_angle=(text_angle - 10.0))

    sch = draw_schematic(model)
    core.save_png(img_tx, bundle / "scene_text.png")
    core.save_png(img_ts, bundle / "scene_text_light30.png")
    core.save_png(img_pt, bundle / "scene_point_light.png")
    core.save_png(sch, bundle / "scene_wireframe.png")
    if img_tx2 is not None:
        core.save_png(img_tx2, bundle / "scene_text_wall2.png")

    panels = []
    if img_in is not None:
        panels.append(("input photo", img_in))
    panels.append(("estimated room: wireframe + light", sch))
    panels.append(("relight = estimated scene", img_re))
    panels.append(("text ON the wall", img_tx))
    panels.append(("same text, light moved 30deg", img_ts))
    if img_tx2 is not None:
        panels.append(("same text, tilted on the wall", img_tx2))
    panels.append(("text + nearby lamp (window-like)", img_pt))

    cols = 2
    rows = (len(panels) + cols - 1) // cols
    ar = model["geometry"]["size"][1] / max(model["geometry"]["size"][0], 1)
    tw = int(width)
    th = max(1, int(round(tw * ar)))
    cap_h = 20
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + cap_h + 10) + 10),
                      (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    font = _load_font(font_path, 15)
    for i, (name, arr) in enumerate(panels):
        r, c = divmod(i, cols)
        x0 = 10 + c * (tw + 10)
        y0 = 10 + r * (th + cap_h + 10)
        a = np.clip(np.asarray(arr, np.float32), 0, 1)
        if a.ndim == 2:
            a = np.dstack([a] * 3)
        im = Image.fromarray((a * 255 + 0.5).astype(np.uint8)).resize((tw, th),
                                                                     Image.LANCZOS)
        sheet.paste(im, (x0, y0))
        cap = "%d) %s" % (i + 1, name)
        try:
            draw.text((x0, y0 + th + 2), cap, fill=(20, 20, 20), font=font)
        except Exception:
            draw.text((x0, y0 + th + 2), cap, fill=(20, 20, 20))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out_path))
    return str(out_path)