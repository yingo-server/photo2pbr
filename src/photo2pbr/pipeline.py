# -*- coding: utf-8 -*-
"""pipeline —编排 + md5 建模缓存。

    photo ──► [缓存命中?] ──否──► build_model ──► models/<key>/model.npz
                        └──是──► load_model ──► export_bundle ──► out/
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import core, export, geom, light as lightmod, material

__all__ = ["run", "build_model", "load_model", "save_model", "Result", "model_key"]


def model_key(photo_path: str | Path, cfg: core.Config) -> str:
    return core.md5_texts(core.md5_file(photo_path), cfg.to_json(), core.VERSION)


def _depth_provider_default(photo_path: str | Path, cfg: core.Config):
    """可选的 onnx 单目深度。未配置/未安装 → None（走假设房间，降级不崩）。"""
    import os
    model_path = os.environ.get("PHOTO2PBR_DEPTH_ONNX", "")
    if not model_path or not Path(model_path).exists():
        return None
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return None

    def _provider(img_srgb: np.ndarray) -> np.ndarray:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        h, w = img_srgb.shape[:2]
        x = np.asarray(img_srgb, np.float32)[None].transpose(0, 3, 1, 2)
        name = inp.name
        out = sess.run(None, {name: x})[0]
        d = np.asarray(out).squeeze()
        d = (d - d.min()) / max(float(d.max() - d.min()), 1e-9)
        return d * 3.0 + 0.5                     # 相对深度 → 米（粗略标定）

    return _provider


def _boundary_mask(face_masks, r: int = 4):
    """面与面的交界带（几何边带）→ True；用于把"墙角亮度跳变"从高光里剔除。"""
    if not face_masks:
        return None
    from .material import box_filter

    lab = None
    for i, (_k, m) in enumerate(face_masks.items()):
        m = np.asarray(m, bool)
        if lab is None:
            lab = np.zeros(m.shape, np.int16)
        lab[m] = i + 1
    if lab is None:
        return None
    bd = np.zeros(lab.shape, bool)
    dx = lab[:, :-1] != lab[:, 1:]
    ax = (lab[:, :-1] > 0) | (lab[:, 1:] > 0)
    bd[:, :-1] |= dx & ax
    bd[:, 1:] |= dx & ax
    dy = lab[:-1, :] != lab[1:, :]
    ay = (lab[:-1, :] > 0) | (lab[1:, :] > 0)
    bd[:-1, :] |= dy & ay
    bd[1:, :] |= dy & ay
    return box_filter(bd.astype(np.float64), int(r)) > 0.0


def _assumed_room(cfg: core.Config) -> geom.RoomEstimate:
    """没有任何深度信息时的房间假设：相机居中，尺寸用默认值；如实标注 assumed。"""
    dims = np.array([cfg.default_room_width_m,
                     cfg.default_room_height_m,
                     cfg.default_room_depth_m], np.float64)
    return geom.RoomEstimate(
        axes=np.eye(3), bmin=-dims / 2.0, bmax=+dims / 2.0,
        roles={"horizontal": 0, "vertical": 1, "depth": 2},
        observed={}, coverage={}, face_masks={}, confidence="assumed")


# ═══════════════════════════════════════════════════════════════════
# 建模
# ═══════════════════════════════════════════════════════════════════
def build_model(photo_path: str | Path, cfg: core.Config,
                depth_provider=None) -> Dict[str, object]:
    img, (w, h) = core.load_image(photo_path, max_side=cfg.max_side)
    fov = core.fov_from_exif(photo_path) or cfg.fov_deg
    K = geom.intrinsics(w, h, fov)

    # 1) 几何优先：先拿到深度/房间，光照才知道"哪些像素其实是墙"
    provider = depth_provider if depth_provider is not None else _depth_provider_default(photo_path, cfg)
    depth = None
    room = _assumed_room(cfg)
    if provider is not None:
        try:
            depth = np.asarray(provider(img), np.float64)
            room = geom.estimate_room(depth, K, front_depth_ratio=cfg.front_depth_ratio,
                                      fallback_dims=(cfg.default_room_width_m,
                                                     cfg.default_room_height_m,
                                                     cfg.default_room_depth_m))
        except Exception:
            depth, room = None, _assumed_room(cfg)

    # 2) 材质（本征分解 + 高光/粗糙度反演 + 金属度代理）
    albedo, shading = material.decompose(img)
    # 墙与墙交界处有几何边带：那里的亮度跳变由面朝向解释，不该算"高光"
    exclude = _boundary_mask(room.face_masks, r=4)
    spec, rough = material.specular_and_roughness(core.srgb_to_linear(img),
                                                  exclude=exclude)
    metal = material.metalness_from_albedo(albedo, shading)

    # 3) 光照（天空 + 太阳）。有深度时：深度无效或远超中位数的像素才算"天空"，
    #    全屋表面 → 没有天空可用 → 如实降级（不再把墙当天空拟合）。
    horizon = int(cfg.horizon_frac * h)
    sky_mask = None
    if depth is not None:
        d = np.asarray(depth, float)
        med = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
        sky_mask = (d <= 1e-6) | ((med > 0) & (d > 3.0 * med))
        if float(sky_mask.mean()) < 0.01:
            sky_mask = np.zeros_like(sky_mask, bool)
    # 面法线 + 各面平均 shading → 交给光照去反解主光方向（无天空时）
    face_normals = []
    face_means = []
    for name, m in (room.face_masks or {}).items():
        cov = float(np.mean(m)) if getattr(m, "size", 0) else 0.0
        if cov < 0.01:
            continue
        n = (room.face_normals or {}).get(name)
        if n is None:
            continue
        face_normals.append(np.asarray(n, float))
        face_means.append(float(np.asarray(shading)[m].mean()))

    lighting = lightmod.estimate_lighting(core.srgb_to_linear(img), K, horizon,
                                          sky_mask=sky_mask, shading=shading,
                                          face_normals=face_normals,
                                          face_means=face_means)

    return {
        "meta": {"key": model_key(photo_path, cfg), "src": Path(photo_path).name,
                 "size": [w, h], "fov_deg": float(fov), "version": core.VERSION},
        "albedo": np.asarray(albedo, np.float32),
        "shading": np.asarray(shading, np.float32),
        "roughness": np.asarray(rough, np.float32),
        "metalness": np.asarray(metal, np.float32),
        "specular": np.asarray(spec, np.float32),
        "depth": None if depth is None else np.asarray(depth, np.float32),
        "face_masks": {k: np.asarray(v, bool) for k, v in room.face_masks.items()},
        "geometry": {"axes": room.axes, "bounds_min": room.bmin, "bounds_max": room.bmax,
                     "dims": room.dims, "roles": room.roles, "observed": room.observed,
                     "coverage": room.coverage, "confidence": room.confidence,
                     "K": K, "size": [w, h], "fov_deg": float(fov)},
        "lighting": lighting.to_dict(),
    }


def save_model(model: dict, mdir: Path) -> None:
    mdir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "albedo": np.asarray(model["albedo"], np.float16),
        "shading": np.asarray(model["shading"], np.float16),
        "roughness": np.asarray(model["roughness"], np.float16),
        "metalness": np.asarray(model["metalness"], np.float16),
        "specular": np.asarray(model.get("specular", np.zeros_like(model["roughness"])),
                               np.float16),
        "axes": np.asarray(model["geometry"]["axes"], np.float64),
        "bmin": np.asarray(model["geometry"]["bounds_min"], np.float64),
        "bmax": np.asarray(model["geometry"]["bounds_max"], np.float64),
        "K": np.asarray(model["geometry"]["K"], np.float64),
    }
    if model.get("depth") is not None:
        arrays["depth"] = np.asarray(model["depth"], np.float16)
    for name, m in (model.get("face_masks") or {}).items():
        arrays["mask_" + name] = np.packbits(np.asarray(m, bool))
    tmp = mdir / "model.npz.tmp"
    with open(tmp, "wb") as fh:          # 传文件对象：避免 numpy 给文件名自动补 .npz
        np.savez_compressed(fh, **arrays)
    tmp.replace(mdir / "model.npz")

    def _j(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k2: _j(x) for k2, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_j(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    with open(mdir / "model.json", "w", encoding="utf-8") as f:
        json.dump({"meta": model["meta"], "geometry": _j({
            k: v for k, v in model["geometry"].items() if k not in ("axes", "K")}),
            "lighting": _j(model["lighting"]),
            "masks": sorted((model.get("face_masks") or {}).keys())}, f,
            ensure_ascii=False, indent=2)


def load_model(mdir: Path) -> dict:
    with np.load(mdir / "model.npz", allow_pickle=False) as z:
        meta = json.loads((mdir / "model.json").read_text(encoding="utf-8"))
        geo = dict(meta["geometry"])
        geo["axes"] = z["axes"]
        geo["bounds_min"] = z["bmin"]
        geo["bounds_max"] = z["bmax"]
        geo["K"] = z["K"]
        geo["dims"] = np.asarray(geo["bounds_max"]) - np.asarray(geo["bounds_min"])
        masks = {}
        hw = (geo["size"][1], geo["size"][0])
        for name in meta.get("masks", []):
            key = "mask_" + name
            if key in z:
                masks[name] = np.unpackbits(z[key])[: hw[0] * hw[1]].reshape(hw).astype(bool)
        depth = z["depth"].astype(np.float64) if "depth" in z else None
        return {"meta": meta["meta"], "albedo": z["albedo"].astype(np.float64),
                "shading": z["shading"].astype(np.float64),
                "roughness": z["roughness"].astype(np.float64),
                "metalness": z["metalness"].astype(np.float64),
                "specular": z["specular"].astype(np.float64) if "specular" in z else None,
                "depth": depth, "face_masks": masks, "geometry": geo,
                "lighting": meta["lighting"]}


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Result:
    out_dir: str
    key: str
    hit: bool
    files: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def run(photo_path: str | Path, out_dir: str | Path, cfg: Optional[core.Config] = None,
        depth_provider=None) -> Result:
    cfg = cfg or core.Config()
    t0 = time.perf_counter()
    key = model_key(photo_path, cfg)
    mdir = cfg.resolved_cache_root() / "models" / key
    hit = bool((mdir / "model.npz").exists() and not cfg.force)

    t1 = time.perf_counter()
    if hit:
        model = load_model(mdir)
    else:
        model = build_model(photo_path, cfg, depth_provider=depth_provider)
        save_model(model, mdir)
    t2 = time.perf_counter()

    files = export.export_bundle(out_dir, model, str(photo_path), key)
    t3 = time.perf_counter()

    warnings = []
    if not hit:
        warnings.append("模型已重建（缓存未命中）")
    if model["geometry"]["confidence"] != "estimated":
        warnings.append("几何置信度=%s：%s" % (
            model["geometry"]["confidence"],
            "无深度输入，使用默认房间假设" if model["geometry"]["confidence"] == "assumed"
            else "部分墙面用假设补齐"))
    if model["lighting"].get("sky_confidence") == "unconstrained":
        warnings.append("画面中没有可用天空：环境光照退化为中性")

    return Result(out_dir=str(out_dir), key=key, hit=hit, files=files,
                  timings={"load_or_build_s": round(t1 - t0, 3),
                           "build_s": round(t2 - t1, 3),
                           "export_s": round(t3 - t2, 3),
                           "total_s": round(t3 - t0, 3)},
                  warnings=warnings)