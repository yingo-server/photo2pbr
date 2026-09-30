# -*- coding: utf-8 -*-
"""relight 测试：往返一致性（重渲 vs 输入）、换光生效、对照表可生成。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photo2pbr import core, pipeline, relight, synth  # noqa: E402

DIMS = (4.0, 2.6, 5.0)
FOV = 100.0


def _model(tmp: Path):
    room = synth.make_room(width=240, height=180, dims=DIMS, fov_deg=FOV, yaw_deg=0.0)
    src = tmp / "photo.png"
    core.save_png(room.image, src)
    cfg = core.Config(cache_root=str(tmp / "out" / "cache"), max_side=240, fov_deg=FOV)
    res = pipeline.run(src, tmp / "out", cfg, depth_provider=lambda img: room.depth)
    model = relight.load_bundle_model(Path(res.out_dir))
    assert model is not None
    return room, model, Path(res.out_dir)


def test_face_layout_covers_all_six_faces():
    room, model, _ = _model(Path(__import__("tempfile").mkdtemp(prefix="p2p-")))
    geo = model["geometry"]
    layout = relight.face_layout(np.asarray(geo["axes"]), np.asarray(geo["bounds_min"]),
                                 np.asarray(geo["bounds_max"]), geo["roles"])
    assert set(layout) == {"left", "right", "floor", "ceiling", "back", "front"}
    axes_used = [v[0] for v in layout.values()]
    assert sorted(axes_used) == [0, 0, 1, 1, 2, 2]


def test_relight_reproduces_input(tmp: Path = None):
    """往返一致性：重渲图必须与输入图在"形状"上接近（同视角、同光）。

    实测 0.069（未反解主光时 0.31）——阈值留一倍余量。
    """
    import tempfile

    room, model, _ = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    out = relight.relight(model)
    a = np.asarray(out, np.float64)
    b = np.asarray(room.image, np.float64)
    assert a.shape == b.shape

    def _norm(x):
        return x / (float(np.mean(x)) + 1e-9)

    err_n = float(np.mean(np.abs(_norm(a) - _norm(b))))
    assert err_n < 0.12, "往返误差过大：norm=%.4f" % err_n


def test_key_light_direction_close_to_truth(tmp: Path = None):
    """无天空室内：主光方向由面法线反解，必须接近合成真值（实测 6.5°）。"""
    import tempfile

    room, model, _ = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    lit = model["lighting"]
    assert lit.get("key_confidence") == "fitted-faces"
    v = np.asarray(lit["key_dir_cam"], float)
    v = v / np.linalg.norm(v)
    L_true = room.R_wc @ room.sun_dir_world
    L_true = L_true / np.linalg.norm(L_true)
    ang = float(np.degrees(np.arccos(np.clip(abs(float(v @ L_true)), -1.0, 1.0))))
    assert ang < 15.0, "主光方向偏差 %.1f°" % ang


def test_relight_sun_rotation_changes_image():
    import tempfile

    _, model, _ = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    a = relight.relight(model)
    b = relight.relight(model, sun_rot_deg=30.0)
    d = float(np.mean(np.abs(a - b)))
    assert d > 0.01, "换光后画面几乎没变（d=%.4f）" % d
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_apply_text_changes_texture():
    """贴字：必须真的改到某个面的贴图（并且知道是哪个面）。"""
    import tempfile

    _, model, _ = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    texes, layout, _g = relight.build_face_textures(model, tex=96)
    before = texes.copy()
    face = relight.apply_text(texes, layout, model, text="HI")
    assert face in layout
    assert float(np.abs(texes - before).max()) > 0.1, "贴图没有被改动"


def test_point_light_gives_gradient_and_is_finite():
    """近距光源：同一面内应出现亮度梯度（1/r²），且与拟合光明显不同。"""
    import tempfile

    _, model, _ = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    a = relight.relight(model, light_model="point")
    b = relight.relight(model)
    assert np.isfinite(a).all()
    assert float(np.mean(np.abs(a - b))) > 0.01, "近距光与拟合光几乎一样"
    g = np.asarray(a, float).mean(axis=-1)
    assert float(g.std()) > 0.02, "近距光没有产生梯度"


def test_apply_text_rotation_and_multiline():
    """贴字参数：旋转会改变字形分布；多行能渲染且不炸。"""
    import tempfile

    _, model, _ = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    t0, layout, _g = relight.build_face_textures(model, tex=96)
    t1 = t0.copy()
    t2 = t0.copy()
    relight.apply_text(t1, layout, model, text="HI")
    relight.apply_text(t2, layout, model, text="HI", angle_deg=-20.0)
    assert float(np.mean(np.abs(t1 - t2))) > 1e-4, "旋转没有生效"
    t3 = t0.copy()
    relight.apply_text(t3, layout, model, text="HI\nTHERE")
    changed = int((np.abs(t3 - t0).max(axis=-1) > 0.05).sum())
    assert float(np.abs(t3 - t0).max()) > 0.12, "多行没有渲染（diff 太小）"
    assert changed > 100, "多行改动像素太少：%d" % changed


def test_form_factors_and_bounce():
    """形状因子健全性 + 互反射：暗面应被"反弹光"提亮得更多。"""
    import tempfile

    _, model, _ = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    geo = model["geometry"]
    layout = relight.face_layout(np.asarray(geo["axes"]), np.asarray(geo["bounds_min"]),
                                 np.asarray(geo["bounds_max"]), geo["roles"])
    names, F, _A = relight._box_form_factors(layout, geo["bounds_min"], geo["bounds_max"])
    assert F.shape == (6, 6)
    assert float(F.min()) >= -1e-9
    assert float(F.sum(axis=1).max()) <= 1.001, "行和应 ≤1（能量守恒）"

    a = np.asarray(relight.relight(model, light_model="point", bounce=False), float)
    b = np.asarray(relight.relight(model, light_model="point", bounce=True), float)
    assert np.isfinite(b).all()
    assert float(np.mean(b)) > float(np.mean(a)), "互反射应整体提亮（多次反射加能量）"


def test_compose_light_study_writes_png():
    import tempfile

    _, _, outdir = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    p = relight.compose_light_study(outdir, outdir / "study.png", text="HI",
                                    angles=(-20.0, 0.0, 20.0), width=100)
    assert Path(p).exists() and Path(p).stat().st_size > 1000


def test_compose_scene_sheet_writes_png():
    import tempfile

    _, _, outdir = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    p = relight.compose_scene_sheet(outdir, outdir / "scene.png", text="HI", width=120)
    assert Path(p).exists() and Path(p).stat().st_size > 1000


def test_compose_sheet_writes_png():
    import tempfile

    _, _, outdir = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    p = relight.compose_sheet(outdir, outdir / "sheet.png", width=120)
    assert Path(p).exists() and Path(p).stat().st_size > 1000


def test_pipeline_marks_sky_unconstrained_when_all_room():
    """全屋表面（没有天空）→ 光照必须如实降级，不许把墙当天空。"""
    import json
    import tempfile

    room, _, outdir = _model(Path(tempfile.mkdtemp(prefix="p2p-")))
    scene = json.loads((outdir / "scene.json").read_text(encoding="utf-8"))
    assert scene["lighting"]["sky_confidence"] == "unconstrained"