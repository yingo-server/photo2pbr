# -*- coding: utf-8 -*-
"""geom 测试：解析光线-盒、合成房间的估计精度、噪声鲁棒性。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photo2pbr import geom, synth  # noqa: E402

FOV = 80.0
DIMS = (4.0, 2.6, 5.0)


def _s(x) -> float:
    """numpy≥2：非 0 维数组不能直接 float()，统一走 item()。"""
    return float(np.asarray(x).item())


def _b(x) -> bool:
    return bool(np.asarray(x).item())


def _room(fov: float = 100.0, yaw: float = 0.0):
    """fov100/yaw0：四侧墙 + 地板 + 天花板都在画面里（可验证几何精度）。"""
    return synth.make_room(width=240, height=180, dims=DIMS, fov_deg=fov,
                           yaw_deg=yaw, pitch_deg=0.0)


def test_ray_box_analytic():
    box = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    t, v = geom.ray_box_t([0.0, 0.0, 0.0], np.array([[[1.0, 0.0, 0.0]]]), *box)
    assert _b(v) and abs(_s(t) - 1.0) < 1e-12
    t, v = geom.ray_box_t([0.0, 0.0, 0.0], np.array([[[0.0, 0.0, 1.0]]]), *box)
    assert _b(v) and abs(_s(t) - 1.0) < 1e-12
    # 盒外：从 x=5 向 -x 打，命中面在 x=1 → t=4
    t, v = geom.ray_box_t([5.0, 0.0, 0.0], np.array([[[-1.0, 0.0, 0.0]]]), *box)
    assert _b(v) and abs(_s(t) - 4.0) < 1e-12
    # 平行于 x 的射线（永远进不了 x slab）→ 无效
    t, v = geom.ray_box_t([5.0, 0.0, 0.0], np.array([[[0.0, 0.0, 1.0]]]), *box)
    assert not _b(v)


def test_estimate_room_dims_and_axes():
    r = _room()
    est = geom.estimate_room(r.depth, r.K)
    dims = est.dims
    h = float(dims[est.roles["horizontal"]])
    v = float(dims[est.roles["vertical"]])
    d_back = float(max(abs(est.bmin[est.roles["depth"]]), abs(est.bmax[est.roles["depth"]])))
    assert abs(h - DIMS[0]) < 0.25, "水平尺寸 %.3f" % h
    assert abs(v - DIMS[1]) < 0.25, "高度 %.3f" % v
    assert abs(d_back - DIMS[2] / 2.0) < 0.25, "后墙距离 %.3f" % d_back

    # 主轴 = 世界轴在相机系下的方向（允许排列与正负号）
    expected = r.R_wc.T                       # 行 = 世界轴在相机系
    for row in est.axes:
        best = float(np.max(np.abs(expected @ row)))
        assert best > 0.9985, "主轴偏差过大 best=%.5f" % best
    for erow in expected:
        best = float(np.max(np.abs(est.axes @ erow)))
        assert best > 0.9985, "主轴不匹配 best=%.5f" % best


def test_estimate_room_rotation_recovered():
    """斜着拍（yaw=15°）：主轴仍要被准确还原。"""
    r = _room(yaw=15.0)
    est = geom.estimate_room(r.depth, r.K)
    expected = r.R_wc.T
    for row in est.axes:
        best = float(np.max(np.abs(expected @ row)))
        assert best > 0.998, "旋转主轴偏差过大 best=%.5f" % best
    v = float(est.dims[est.roles["vertical"]])
    assert abs(v - DIMS[1]) < 0.3, "斜视高度 %.3f" % v


def test_estimate_room_face_masks_cover_main_walls():
    r = _room()
    est = geom.estimate_room(r.depth, r.K)
    valid = float((r.valid).sum())
    for name in ("back", "floor"):
        cov = float(est.face_masks[name].sum()) / valid
        assert cov > 0.02, "%s 面覆盖率太低：%.4f" % (name, cov)
    # 深度轴永远看不到"相机背后" → 必须如实标注为 partial，且假想端 observed=False
    assert est.confidence == "partial"
    d = est.roles["depth"]
    fabricated = [k for k in ("axis%d:lo" % d, "axis%d:hi" % d) if not est.observed[k]]
    assert len(fabricated) == 1, "恰好应有一端是假设：%s" % est.observed


def test_estimate_room_noisy_depth_still_ok():
    r = _room()
    rng = np.random.default_rng(1)
    dn = r.depth * (1.0 + 0.002 * rng.standard_normal(r.depth.shape))
    est = geom.estimate_room(dn, r.K)
    dims = est.dims
    h = float(dims[est.roles["horizontal"]])
    v = float(dims[est.roles["vertical"]])
    assert abs(h - DIMS[0]) < 0.4, "含噪水平尺寸 %.3f" % h
    assert abs(v - DIMS[1]) < 0.4, "含噪高度 %.3f" % v