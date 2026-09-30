# -*- coding: utf-8 -*-
"""core 测试：哈希、色彩往返、配置确定性。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photo2pbr import core  # noqa: E402


def test_md5_stability(tmp: Path = None):
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="p2p-"))
    f1 = d / "a.bin"
    f1.write_bytes(b"hello world" * 100)
    assert core.md5_file(f1) == core.md5_file(f1)
    f2 = d / "b.bin"
    f2.write_bytes(b"hello worlD" * 100)
    assert core.md5_file(f1) != core.md5_file(f2)


def test_md5_texts_deterministic():
    assert core.md5_texts("a", "b") == core.md5_texts("a", "b")
    assert core.md5_texts("a", "b") != core.md5_texts("ab", "")


def test_srgb_roundtrip():
    x = np.linspace(0.0, 1.0, 256)
    y = core.linear_to_srgb(core.srgb_to_linear(x))
    assert np.allclose(x, y, atol=1e-9)


def test_config_json_is_deterministic():
    a = core.Config()
    b = core.Config()
    assert a.to_json() == b.to_json()
    assert "fov_deg" in a.to_json()
    c = core.Config(fov_deg=70.0)
    assert c.to_json() != a.to_json()


def test_cache_root_is_pathlike():
    p = core.default_cache_root()
    assert isinstance(p, Path)
    assert str(p)


def test_save_png_roundtrip(tmp: Path = None):
    import tempfile

    from PIL import Image

    d = Path(tempfile.mkdtemp(prefix="p2p-"))
    arr = np.random.default_rng(0).random((16, 12, 3), dtype=np.float32)
    out = core.save_png(arr, d / "x.png")
    back = np.asarray(Image.open(out), dtype=np.float32) / 255.0
    assert back.shape == arr.shape
    assert np.abs(back - arr).max() <= 1.0 / 255.0 + 1e-6