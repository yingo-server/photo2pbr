# -*- coding: utf-8 -*-
"""cli —命令行入口。

    photo2pbr run   <photo> --out out/scene1 [--fov 65] [--force]
    photo2pbr demo  --out dist/demo [--width 240 --height 180]
    photo2pbr selfcheck
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from . import core, pipeline, synth


def _p(msg: str) -> None:
    try:
        print(msg)
    except Exception:
        print(msg.encode("utf-8", "replace").decode("ascii", "replace"))


def _report(res: pipeline.Result) -> None:
    _p("out     : %s" % res.out_dir)
    _p("key     : %s" % res.key)
    _p("缓存    : %s" % ("命中（未重建模）" if res.hit else "未命中（已建模并入库）"))
    _p("耗时(s) : %s" % json.dumps(res.timings, ensure_ascii=False))
    for w in res.warnings:
        _p("注意    : %s" % w)


def cmd_run(a) -> int:
    cfg = core.Config(fov_deg=a.fov, max_side=a.max_side, force=a.force,
                      cache_root=a.cache or "")
    if not Path(a.photo).exists():
        _p("找不到照片: %s" % a.photo)
        return 2
    res = pipeline.run(a.photo, a.out, cfg)
    _report(res)
    return 0


def cmd_demo(a) -> int:
    # 演示用固定几何/视场角：合成与估计必须一致，否则等于在骗自己
    DEMO_DIMS = (4.0, 2.6, 5.0)
    DEMO_FOV = 100.0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    room = synth.make_room(width=a.width, height=a.height, dims=DEMO_DIMS,
                           fov_deg=DEMO_FOV, yaw_deg=0.0)
    src = out / "demo_input.png"
    core.save_png(room.image, src)
    _p("已合成演示照片: %s (%dx%d)  真值尺寸=%s  真值fov=%.0f"
       % (src, a.width, a.height, list(DEMO_DIMS), DEMO_FOV))

    cfg = core.Config(cache_root=str(out / "cache"), max_side=max(a.width, a.height),
                      force=True, fov_deg=DEMO_FOV)
    # 演示：把合成房间的真值深度喂给管线（等价于将来 onnx 单目深度模型的输出）
    res = pipeline.run(src, out, cfg, depth_provider=lambda img: room.depth)
    _report(res)

    # 顺手出两张对照表：场景（给人看）+ 贴图（给分析用）
    try:
        from .relight import compose_scene_sheet, compose_sheet, compose_light_study
        s1 = compose_scene_sheet(out, out / "sheet.png", text="OwnRender")
        s2 = compose_sheet(out, out / "sheet_maps.png")
        s3 = compose_light_study(out, out / "light_study.png", text="OwnRender")
        _p("场景对照表: %s" % s1)
        _p("贴图对照表: %s" % s2)
        _p("光研究  : %s" % s3)
    except Exception as exc:      # 出图失败不影响主流程
        _p("对照表生成失败（不影响结果）: %s" % exc)
    return 0


def cmd_selfcheck(a) -> int:
    import numpy as np
    from . import geom

    ok = True

    def check(name, cond):
        nonlocal ok
        _p("  [%s] %s" % ("OK" if cond else "!!", name))
        ok = ok and bool(cond)

    _p("photo2pbr selfcheck (v%s)" % core.VERSION)
    t, v = geom.ray_box_t([0.0, 0.0, 0.0], np.array([[[1.0, 0.0, 0.0]]]),
                          [-1, -1, -1], [1, 1, 1])
    check("光线-盒求交 t=1", abs(float(np.asarray(t).item()) - 1.0) < 1e-12
          and bool(np.asarray(v).item()))
    r = synth.make_room(width=240, height=180, dims=(4.0, 2.6, 5.0), fov_deg=100.0)
    est = geom.estimate_room(r.depth, r.K)
    dims = est.dims
    check("房间水平尺寸 ≈ 4.0m", abs(dims[est.roles["horizontal"]] - 4.0) < 0.25)
    check("房间高度 ≈ 2.6m", abs(dims[est.roles["vertical"]] - 2.6) < 0.25)
    _p("selfcheck: %s" % ("全部通过" if ok else "有失败"))
    return 0 if ok else 1


def cmd_preview(a) -> int:
    from .relight import compose_scene_sheet, compose_sheet

    out = Path(a.out) if a.out else (Path(a.bundle) / "sheet.png")
    try:
        s1 = compose_scene_sheet(a.bundle, out, text=(a.text or None),
                                 font_path=(a.font or None))
        s2 = compose_sheet(a.bundle, out.with_name(out.stem + "_maps.png"))
    except Exception as exc:
        _p("生成失败: %s" % exc)
        return 1
    _p("场景对照表: %s" % s1)
    _p("贴图对照表: %s" % s2)
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="photo2pbr",
                                 description="单张照片 → 符合物理条件的反射环境")
    ap.add_argument("--version", action="version", version="photo2pbr " + core.VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="处理一张照片")
    p_run.add_argument("photo")
    p_run.add_argument("--out", default="out/scene1")
    p_run.add_argument("--fov", type=float, default=60.0, help="无 EXIF 时的水平视场角")
    p_run.add_argument("--max-side", type=int, default=1280)
    p_run.add_argument("--force", action="store_true", help="忽略缓存，强制重建模")
    p_run.add_argument("--cache", default="", help="缓存根目录（默认 %%LOCALAPPDATA%%）")
    p_run.set_defaults(func=cmd_run)

    p_demo = sub.add_parser("demo", help="合成演示照片并端到端跑一遍")
    p_demo.add_argument("--out", default="dist/demo")
    p_demo.add_argument("--width", type=int, default=320)
    p_demo.add_argument("--height", type=int, default=240)
    p_demo.set_defaults(func=cmd_demo)

    p_sc = sub.add_parser("selfcheck", help="内建自检")
    p_sc.set_defaults(func=cmd_selfcheck)

    p_pv = sub.add_parser("preview", help="对已有 bundle 出对照表（场景 + 贴图）")
    p_pv.add_argument("--bundle", default="dist/demo")
    p_pv.add_argument("--out", default="")
    p_pv.add_argument("--text", default="OwnRender", help="贴到墙上的文字（空=不贴）")
    p_pv.add_argument("--font", default="", help="TTF 字体路径（可选中文字体）")
    p_pv.set_defaults(func=cmd_preview)

    a = ap.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())