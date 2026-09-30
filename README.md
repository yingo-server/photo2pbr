# photo2pbr

**一张照片 → 符合物理条件的反射环境（房间盒体 + 材质 + 光照）。**

> Windows-first。用于给"墙面叠字 / 合成渲染"提供可复用的 PBR 场景与反射环境。
> 不需要高精度重建——需要的是**物理自洽**：光的方向、软硬、房间的反射、材质的粗糙度。

---

## 为什么

单张照片里没有视差，**做不出精确 3D**（3DGS 需要多视角）。
我们只抽取渲染真正需要的东西：

| 需要什么 | 怎么拿 |
|---|---|
| 房间形状 | 曼哈顿假设（三组正交平面）→ 从深度/透视线估计盒体 |
| 材质 | 本征分解：光照分量（低频）↔ 反射率分量（高频）→ albedo / roughness / metalness |
| 光照 | 天空亮度梯度拟合 + 太阳盘检测（位置、相对强度、角径→软硬） |
| 反射环境 | 导出 `sky.hdr`（可作 Blender/渲染器的 environment texture） |
| 可复用 | **照片 md5 + 配置 md5** → 建模只做一次（缓存） |

## 安装

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 快速开始

```bat
:: 端到端冒烟：合成一张"房间照" → 估计 → 导出 bundle
python -m photo2pbr demo --out dist\demo

:: 处理自己的照片
python -m photo2pbr run path\to\photo.jpg --out out\scene1
python -m photo2pbr run path\to\photo.jpg --out out\scene1 --fov 65 --force
```

## 输出 bundle

```
out/scene1/
├── scene.json          # 场景描述（schema photo2pbr/scene/1）
├── albedo.png          # 反射率（线性）
├── shading.png         # 光照分量（按 p99 归一显示）
├── specular.png        # 高光残差（局部正残差）
├── roughness.png       # 粗糙度（由高光反演：越尖越光滑）
├── metalness.png       # 金属度（代理）
├── depth16.png         # 深度（如有）
├── masks/*.png         # 六个面的归属掩膜
├── sky.hdr             # 辐射环境贴图（Radiance .hdr）
├── sheet.png           # 场景对照表（输入/重渲/贴字/换光/近距灯）
├── sheet_maps.png      # 贴图对照表（技术分析用）
├── scene_text.png      # 贴字重渲
├── scene_point_light.png  # 近距光源重渲
├── blender_preview.py  # 一键在 Blender 里复现（环境贴图 + 太阳角径）
└── report.txt          # 人读摘要
```

## 能力（v0.2）

* **重渲（relight）**：用估计出的几何/材质/光照把场景重新渲染一遍
  —— 既是"看图"，也是**自洽性验证**（实测往返误差 0.069）
* **贴字（text on wall）**：文字直接画进**墙面贴图**，渲染时按 uv 采样
  → 透视、朝向自动正确；且跟随该面的光照明暗（暗处的字自然变暗）
* **两种光源模型**：
  * `fitted`：由"面法线 + 各面平均 shading"反解的主光（实测方向误差 6.5°）
  * `point`：近距光源（1/r² + 余弦）→ 墙上出现**光斑与梯度**（更像窗口光）
* **对照表**：`sheet.png` 给人看（少而大）；`sheet_maps.png` 给分析用

## 架构

```
photo ──► core(缓存/IO) ──► geom(相机·盒体) ──► material(本征/PBR) ──► light(太阳/天空) ──► export(bundle)
                                  │                                       │
                             (可选) onnx 单目深度                      sky.hdr → 反射环境
```

* `geom.py`：针孔相机、光线-盒求交、深度→房间估计（PCA 求主轴 + 逐平面覆盖率，缺面用假设补齐并**如实标注**）
* `material.py`：guided filter 本征分解、粗糙度/金属度**代理**（有界、单调、可升级为学习模型）
* `light.py`：天空 `L(θ)=A+B·sinθ` 拟合、太阳盘检测（位置/强度/角径）
* 设计原则：**能用库就用库**（numpy 做数值、Pillow 做 IO、onnxruntime 做深度、Blender 做渲染），我们只写算法与编排

## 性能（Windows 目标）

| 指标 | 目标 |
|---|---|
| 冷启动 → 首张结果 | **≤ 90 s** |
| 连续构建（缓存命中） | **≤ 10 s / 张** |
| 缓存命中路径 | ≤ 300 ms |

## 测试

全部在 CI（`windows-latest` × Python 3.11/3.12）执行；本地可用轻量 runner：

```bat
python tools\run_tests.py
```

## 路线图

- [x] P0 骨架 / 缓存 / CI
- [x] P1 几何：射线-盒、房间盒体估计（深度驱动）
- [x] P1 材质：本征分解 + 粗糙度/金属度代理
- [x] P1 光照：天空拟合 + 太阳盘
- [ ] P2 单图（无深度）恢复：灭点/透视线 → 盒体比例
- [ ] P2 接入 onnxruntime 单目深度（可选依赖，缺省降级）
- [ ] P3 水渍/镜面区域分割（水面反射用）
- [ ] P4 与 OwnRender 渲染管线对接（叠字为准）

## 许可

MIT
