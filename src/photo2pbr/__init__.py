# -*- coding: utf-8 -*-
"""photo2pbr — 单张照片 → 符合物理条件的反射环境。

模块分层（自下而上）:
    core      配置 / 哈希缓存 / 图像 IO
    geom      针孔相机 / 光线-盒求交 / 房间盒体估计
    material  本征分解 / 粗糙度 / 金属度
    light     天空亮度模型 / 太阳盘检测
    synth     合成"房间照"（测试与演示用，不参与生产逻辑）
    export    scene.json / 贴图 / sky.hdr / blender 预览
    pipeline  编排 + md5 建模缓存
    cli       命令行
"""
from .core import VERSION, Config  # noqa: F401

__all__ = ["VERSION", "Config"]
__version__ = VERSION