# -*- coding: utf-8 -*-
"""端到端：合成照片 → pipeline → bundle；缓存命中；HDR 往返；Blender 脚本语法。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photo2pbr import core, export, pipeline, synth  # noqa: E402


def _make_photo(tmp: Path) -> Path:
    room = synth.make_room(width=160, height=120, fov_deg=80.0)
    src = tmp / "photo.png"
    core.save_png(room.image, src)
    return src


def test_pipeline_end_to_end_and_cache_hit(tmp: Path = None):
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="p2p-"))
    src = _make_photo(tmp)
    cfg = core.Config(cache_root=str(tmp / "cache"), max_side=160)

    res1 = pipeline.run(src, tmp / "out1", cfg)
    assert not res1.hit, "首次必须建模"
    out1 = Path(res1.out_dir)
    for name in ("scene.json", "sky.hdr", "blender_preview.py", "albedo.png",
                 "roughness.png", "metalness.png", "report.txt"):
        assert (out1 / name).exists(), "缺少 %s" % name

    scene = json.loads((out1 / "scene.json").read_text(encoding="utf-8"))
    assert scene["schema"] == "photo2pbr/scene/1"
    assert len(scene["room"]["dims"]) == 3
    assert scene["camera"]["fov_deg"] > 0
    assert "lighting" in scene

    res2 = pipeline.run(src, tmp / "out2", cfg)
    assert res2.hit, "第二次必须命中缓存（不重复建模）"
    assert res2.key == res1.key


def test_blender_preview_script_is_valid_python(tmp: Path = None):
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="p2p-"))
    src = _make_photo(tmp)
    cfg = core.Config(cache_root=str(tmp / "cache"), max_side=160)
    res = pipeline.run(src, tmp / "out", cfg)
    text = (Path(res.out_dir) / "blender_preview.py").read_text(encoding="utf-8")
    compile(text, "blender_preview.py", "exec")          # 必须语法合法
    assert "bpy" in text and "sky.hdr" in text


def test_hdr_roundtrip_8x4(tmp: Path = None):
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="p2p-"))
    img = np.zeros((4, 8, 3))
    img[..., 0] = np.linspace(0.1, 1.0, 8)[None, :]
    img[..., 1] = 0.25
    img[..., 2] = 0.5
    p = export.write_hdr(tmp / "t.hdr", img)

    data = Path(p).read_bytes()
    assert data.startswith(b"#?RADIANCE")
    assert b"-Y 4 +X 8" in data

    # 手写一个最小解码器，证明确实能读回来
    body = data.split(b"\n\n", 1)[1].split(b"\n", 1)[1]
    pos = 0
    rows = []
    for _ in range(4):
        assert body[pos] == 2 and body[pos + 1] == 2
        w = (body[pos + 2] << 8) | body[pos + 3]
        assert w == 8
        pos += 4
        chans = []
        for _c in range(4):
            vals = []
            while len(vals) < w:
                code = body[pos]
                pos += 1
                if code > 128:
                    n = code - 128
                    val = body[pos]
                    pos += 1
                    vals.extend([val] * n)
                else:
                    n = code
                    vals.extend(body[pos:pos + n])
                    pos += n
            chans.append(vals)
        rows.append(chans)
    assert len(rows) == 4
    # 第一个像素：R 通道约 0.1
    r0 = rows[0][0][0] / 256.0 * (2.0 ** (rows[0][3][0] - 128))
    assert abs(r0 - 0.1) < 0.02, "解码回来 R=%.4f" % r0


def test_pipeline_with_depth_provider_estimates_room(tmp: Path = None):
    """有深度输入（将来接 onnx 模型）→ 几何走"估计"路径，且如实标注 partial。"""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="p2p-"))
    room = synth.make_room(width=240, height=180, dims=(4.0, 2.6, 5.0),
                           fov_deg=100.0, yaw_deg=0.0)
    src = tmp / "photo.png"
    core.save_png(room.image, src)
    cfg = core.Config(cache_root=str(tmp / "cache"), max_side=240, fov_deg=100.0)
    res = pipeline.run(src, tmp / "out", cfg, depth_provider=lambda img: room.depth)
    scene = json.loads((Path(res.out_dir) / "scene.json").read_text(encoding="utf-8"))
    assert scene["room"]["confidence"] == "partial"
    dims = np.sort(np.asarray(scene["room"]["dims"], dtype=float))
    ref = np.sort(np.array([4.0, 2.6, 5.0]))
    assert float(np.max(np.abs(dims - ref))) < 0.4, "dims=%s" % dims.tolist()


def test_pipeline_force_rebuilds(tmp: Path = None):
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="p2p-"))
    src = _make_photo(tmp)
    cfg = core.Config(cache_root=str(tmp / "cache"), max_side=160)
    assert not pipeline.run(src, tmp / "a", cfg).hit
    assert pipeline.run(src, tmp / "b", cfg).hit
    cfg2 = core.Config(cache_root=str(tmp / "cache"), max_side=160, force=True)
    assert not pipeline.run(src, tmp / "c", cfg2).hit, "--force 必须重建"