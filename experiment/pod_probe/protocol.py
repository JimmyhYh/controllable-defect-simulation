"""受控探针阶段 1：方案常量与路径。

只读复用 controlled_generation 的渲染与几何实现；本包不修改任何既有模块。
执行边界：所有运行步骤（建库、校准、渲染、推理、报告）一律由用户手动启动。
"""

from __future__ import annotations

import json
from pathlib import Path

VERSION = "pod_probe_stage1_v1"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mvtec_ad" / "leather"
SPLIT = ROOT / "data" / "split.json"
OUT = ROOT / "results" / "pod_probe" / "ksdd2_leather_stage1_v1"
OUT_STAGE2 = ROOT / "results" / "pod_probe" / "anolcd_stage2_v1"

# 尺度约定：整幅视场 102.4 mm；k(r) = 102.4 / r
FIELD_OF_VIEW_MM = 102.4
INPUT_SIZE = 256
K_MM_PER_PX = FIELD_OF_VIEW_MM / INPUT_SIZE          # 0.40
PIXEL_AREA_MM2 = K_MM_PER_PX ** 2                    # 0.16

# 建库与校准角色（沿用既有 split，不重抽）
BANK_IMAGES = 80          # R 前 80 张，与旧基线同口径
CALIB_QUANTILE = 0.95     # V49 上 max 分数的 linear 分位数

# 8 个水平 = 1 个无刺激基线 + 7 档 SEMU。
# 下限由 8-bit 可表示性决定：残差上限约为 A，A 低于 1/255 时图像不变，刺激不存在。
# leather 背景 Lb 仅 0.264—0.321，故同一幅度下只有小面积缺陷能把 SEMU 压到 1 附近
# （S 越小 C_jnd 越大）。因此形态同时覆盖小/中两种面积，由数据决定哪一档可达。
SEMU_LEVELS = (1.0, 1.6, 2.5, 4.0, 6.3, 10.0, 16.0)
POLARITY = 1.0            # 阶段 1 只做亮缺陷

RESIDUAL_FLOOR = 1.0 / 255.0   # 有效域与可表示性的共同下限
REPRESENT_MARGIN = 1.02        # 预算幅度须高于下限的余量
COVERAGE_WARN = 0.90           # 仅作记录阈值，不作失败条件

# 四个形态：两种外观 × 小/中两种面积。小面积用于把 SEMU 压到可表示下界附近，
# 中面积保证检测器有响应；两者都保留，避免单一面积决策失误导致曲线全饱和或全零。
SHAPES = {
    "mura_small": {
        "kind": "mura_like", "shape": "ellipse", "units": "px256",
        "center": [128.0, 128.0], "axes": [6.0, 6.0], "angle_deg": 0.0,
        "nominal_pixels": 113,
    },
    "mura_mid": {
        "kind": "mura_like", "shape": "ellipse", "units": "px256",
        "center": [128.0, 128.0], "axes": [12.0, 12.0], "angle_deg": 0.0,
        "nominal_pixels": 452,
    },
    "line_short": {
        "kind": "line_like", "shape": "line", "units": "px256",
        "center": [128.0, 128.0], "length": 16.0, "width": 2.0, "angle_deg": 0.0,
        "nominal_pixels": 32,
    },
    "line_mid": {
        "kind": "line_like", "shape": "line", "units": "px256",
        "center": [128.0, 128.0], "length": 48.0, "width": 4.0, "angle_deg": 0.0,
        "nominal_pixels": 192,
    },
}

AMPLITUDE_LIMIT = 0.25    # render() 的硬约束
AMP_TOLERANCE = 0.05      # 目标 SEMU 与实现 SEMU 的相对容差
AMP_MAX_ITER = 6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split() -> dict:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def background_paths() -> list[str]:
    """P20 固定顺序。"""
    return list(load_split()["splits"]["P"])


def bank_paths(n: int = BANK_IMAGES) -> list[str]:
    return list(load_split()["splits"]["R"])[:n]


def calib_paths() -> list[str]:
    return list(load_split()["splits"]["V"])


def stim_id(shape: str, semu: float, bg_index: int) -> str:
    return f"{shape}__semu{semu:g}__bg{bg_index:02d}"


def base_id(bg_index: int) -> str:
    return f"baseline__bg{bg_index:02d}"


def out_dir(stage: str) -> Path:
    return OUT / stage


def out2_dir(stage: str) -> Path:
    """阶段 2（AnoLCD）独立目录，不与阶段 1 混放。"""
    return OUT_STAGE2 / stage


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
