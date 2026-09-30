# -*- coding: utf-8 -*-
"""core —配置 / 哈希缓存 / 图像 IO。

设计:
    · 缓存键 = md5(照片) + md5(配置 JSON) + VERSION  → 照片没换就不重建模
    · 所有可写目录默认落在 Windows 的 %LOCALAPPDATA%\\photo2pbr
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

VERSION = "0.1.0"

__all__ = [
    "VERSION", "Config",
    "md5_file", "md5_texts",
    "default_cache_root",
    "srgb_to_linear", "linear_to_srgb",
    "load_image", "save_png", "save_png16",
    "fov_from_exif",
]


# ═══════════════════════════════════════════════════════════════════
# 哈希
# ═══════════════════════════════════════════════════════════════════
def md5_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_texts(*texts: str) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def default_cache_root() -> Path:
    base = (os.environ.get("LOCALAPPDATA")
            or os.environ.get("XDG_CACHE_HOME")
            or str(Path.home() / ".cache"))
    return Path(base) / "photo2pbr"


# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """一次"建模"的全部外部参数（都会进缓存键）。"""
    fov_deg: float = 60.0              # 无 EXIF 时的水平视场角
    horizon_frac: float = 0.55         # 地平线(默认)在画面高度中的比例
    max_side: int = 1280               # 输入图长边上限（性能预算）
    front_depth_ratio: float = 1.0     # 单视图看不到相机背后 → 前后深度镜像比
    default_room_width_m: float = 5.0  # 缺失墙面时的默认尺寸
    default_room_height_m: float = 2.8
    default_room_depth_m: float = 4.0
    cache_root: str = ""               # 空 = 默认缓存目录
    force: bool = False                # 忽略缓存
    seed: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    def resolved_cache_root(self) -> Path:
        return Path(self.cache_root) if self.cache_root else default_cache_root()


# ═══════════════════════════════════════════════════════════════════
# 色彩
# ═══════════════════════════════════════════════════════════════════
def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


# ═══════════════════════════════════════════════════════════════════
# 图像 IO（Pillow）
# ═══════════════════════════════════════════════════════════════════
def load_image(path: str | Path, max_side: Optional[int] = None
               ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """读图 → float32 sRGB [0,1]，可按长边缩放。返回 (rgb, (w,h))。"""
    from PIL import Image

    im = Image.open(str(path)).convert("RGB")
    if max_side and max(im.size) > int(max_side):
        s = float(max_side) / float(max(im.size))
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                       Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0, (im.width, im.height)


def save_png(arr: np.ndarray, path: str | Path) -> str:
    """float [0,1]（灰度或 RGB）→ 8bit PNG。"""
    from PIL import Image

    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(str(path))
    return str(path)


def save_png16(arr: np.ndarray, path: str | Path) -> str:
    """float [0,1] 灰度 → 16bit 灰度 PNG（深度图等）。"""
    from PIL import Image

    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((a * 65535.0 + 0.5).astype(np.uint16)).save(str(path))
    return str(path)


# ═══════════════════════════════════════════════════════════════════
# EXIF（可选）
# ═══════════════════════════════════════════════════════════════════
def fov_from_exif(path: str | Path) -> Optional[float]:
    """从 EXIF 的 35mm 等效焦距估算水平 FOV（度）。失败返回 None。"""
    try:
        from PIL import Image

        exif = Image.open(str(path)).getexif()
        f35 = exif.get(41989)  # FocalLengthIn35mmFilm
        if not f35:
            return None
        f35 = float(f35)
        if f35 <= 0:
            return None
        return math.degrees(2.0 * math.atan(36.0 / (2.0 * f35)))
    except Exception:
        return None