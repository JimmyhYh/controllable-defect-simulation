"""替代生成器实验：检验检测器是否记住了原生成器的空间结构。

设计
----
对阶段 2b 的每一条刺激逐条配对，保持三件事完全相同、只改一件事：

| 保持 | 依据 |
| --- | --- |
| 同一背景、同一位置、同一朝向 | 从原 index.json 抄 background_index 与固定几何 |
| **同一有效面积 S** | 抄原刺激的 S_mm2，反解半径使二值掩码面积相等 |
| **同一平均相对对比度 Cx** | 抄原刺激的 Cx_percent，反解幅度 A = Cx/100 · Lb |

| 改变 | 原版本 | 本实验 |
| --- | --- | --- |
| 边缘剖面 | 软边（渲染器衰减剖面） | **硬边阶跃**（二值掩码 × 常数幅度） |

因为 S 与 Cx 都相同，SEMU 在公式上完全相同。所以两版的检出率差异，
只能归因于空间结构（边缘梯度），不能归因于强度或面积。

用途：若某检测器在新版上响应显著下降，说明它对边缘剖面敏感，
即它记住的是原生成器的特征，而非"异常"本身。
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
STAGE2 = ROOT / "results" / "pod_probe" / "anolcd_stage2_v1"
OUT = ROOT / "results" / "pod_probe" / "alt_generator_v1"
ANOLCD = ROOT / "data" / "weak_localization" / "AnoLCD" / "display"

INPUT_SIZE = 256
PIXEL_AREA_MM2 = 0.16
SHAPES = ("mura_p25", "mura_med", "mura_p75", "line_med")
LINE_ASPECT = 112.0 / 4.0
CENTER = (128.0, 128.0)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _grid():
    yy, xx = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE].astype(np.float32)
    return yy, xx


GRID_Y, GRID_X = _grid()


def _ellipse_mask(radius: float) -> np.ndarray:
    cy, cx = CENTER[1], CENTER[0]
    return ((GRID_Y - cy) ** 2 + (GRID_X - cx) ** 2) <= radius ** 2


def _line_mask(length: float, width: float) -> np.ndarray:
    cy, cx = CENTER[1], CENTER[0]
    inside_y = np.abs(GRID_Y - cy) <= length / 2.0
    inside_x = np.abs(GRID_X - cx) <= width / 2.0
    return inside_y & inside_x


def _solve_ellipse_radius(target_px: int) -> float:
    lo, hi = 0.3, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _ellipse_mask(mid).sum() < target_px:
            lo = mid
        else:
            hi = mid
    # 二值面积随半径是阶梯函数，二分中点未必最优，取实际最接近目标的候选
    candidates = [lo, hi, 0.5 * (lo + hi)]
    return min(candidates, key=lambda r: abs(int(_ellipse_mask(r).sum()) - target_px))


def _solve_line_dims(target_px: int) -> tuple:
    width = np.sqrt(target_px / LINE_ASPECT)
    lo, hi = width * 0.5, width * 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _line_mask(mid * LINE_ASPECT, mid).sum() < target_px:
            lo = mid
        else:
            hi = mid
    candidates = [lo, hi, 0.5 * (lo + hi)]
    best = min(candidates, key=lambda w: abs(int(_line_mask(w * LINE_ASPECT, w).sum()) - target_px))
    return best * LINE_ASPECT, best


def hard_stimulus(background: np.ndarray, record: dict) -> dict:
    """按原记录的 S 与 Cx 造硬边版本。"""
    target_px = int(round(record["S_mm2"] / PIXEL_AREA_MM2))
    if record["shape"].startswith("mura"):
        radius = _solve_ellipse_radius(target_px)
        mask = _ellipse_mask(radius)
        geom = {"radius_px": radius}
    else:
        length, width = _solve_line_dims(target_px)
        mask = _line_mask(length, width)
        geom = {"length_px": length, "width_px": width}

    lb = float(np.median(_luma(background)))
    amplitude = float(record["Cx_percent"]) / 100.0 * lb
    image = background + amplitude * mask[..., None].astype(np.float32)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    # 实测核验：面积与平均对比度
    diff = _luma(image) - _luma(background)
    effective = diff > (0.5 / 255.0)
    inside = effective[mask]
    measured_s = float(effective.sum() * PIXEL_AREA_MM2)
    cx = 0.0
    if inside.any() and lb > 0:
        cx = float(np.mean(np.abs(diff[mask])) / lb * 100.0)
    return {
        "image": image,
        "mask": mask,
        "geometry": geom,
        "amplitude": amplitude,
        "S_mm2_measured": measured_s,
        "Cx_percent_measured": cx,
        "pixels_measured": int(effective.sum()),
        "clipped_fraction": float(np.mean((background + amplitude * mask[..., None]) > 1.0)),
    }


def render_all(overwrite: bool = False) -> dict:
    target = OUT / "stimuli"
    index_path = target / "index.json"
    if index_path.exists() and not overwrite:
        raise RuntimeError("alt stimuli already built; use --overwrite or a new branch")

    protocol = json.loads((STAGE2 / "stimuli" / "protocol.json").read_text(encoding="utf-8"))
    original = json.loads((STAGE2 / "stimuli" / "index.json").read_text(encoding="utf-8"))
    backgrounds = protocol["backgrounds"]

    target.mkdir(parents=True, exist_ok=True)
    records, problems = [], []
    bg_cache = {}

    for item in original["records"]:
        if item["kind"] != "stimulus":
            continue
        bgi = int(item["background_index"])
        if bgi not in bg_cache:
            path = ANOLCD / backgrounds[bgi]
            bg_cache[bgi] = np.asarray(
                Image.open(path).convert("RGB").resize(
                    (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR),
                dtype=np.float32) / np.float32(255.0)
        bg = bg_cache[bgi]

        try:
            made = hard_stimulus(bg, item)
        except Exception as exc:
            problems.append({"id": item["id"], "reason": str(exc)})
            continue

        new_id = "alt_" + item["id"]
        np.savez_compressed(target / f"{new_id}.npz",
                            image=made["image"].astype(np.float32))
        records.append({
            "id": new_id,
            "paired_original": item["id"],
            "kind": "stimulus",
            "shape": item["shape"],
            "target_SEMU": item["target_SEMU"],
            "background_index": bgi,
            "S_mm2_original": item["S_mm2"],
            "S_mm2_measured": made["S_mm2_measured"],
            "Cx_percent_original": item["Cx_percent"],
            "Cx_percent_measured": made["Cx_percent_measured"],
            "amplitude": made["amplitude"],
            "geometry": made["geometry"],
            "pixels_measured": made["pixels_measured"],
            "clipped_fraction": made["clipped_fraction"],
            "SEMU_original": item["SEMU_actual"],
        })

    # 基线原样复制，保证两版背景集完全一致
    for item in original["records"]:
        if item["kind"] != "baseline":
            continue
        bgi = int(item["background_index"])
        src = STAGE2 / "stimuli" / f"{item['id']}.npz"
        with np.load(src) as handle:
            image = np.array(handle["image"], dtype=np.float32)
        new_id = "alt_" + item["id"]
        np.savez_compressed(target / f"{new_id}.npz", image=image)
        records.append({"id": new_id, "paired_original": item["id"], "kind": "baseline",
                        "background_index": bgi, "Lb": item.get("Lb")})

    s_err = [abs(r["S_mm2_measured"] - r["S_mm2_original"]) / max(r["S_mm2_original"], 1e-9)
             for r in records if r["kind"] == "stimulus"]
    c_err = [abs(r["Cx_percent_measured"] - r["Cx_percent_original"])
             / max(abs(r["Cx_percent_original"]), 1e-9)
             for r in records if r["kind"] == "stimulus"]

    index = {
        "version": "alt_generator_v1",
        "purpose": "同 S 同 Cx，只改边缘剖面（软边→硬边）",
        "changed": "edge profile: soft -> hard step",
        "held_fixed": ["background_index", "center", "orientation", "S", "Cx"],
        "n_backgrounds": len(backgrounds),
        "n_stimulus": sum(1 for r in records if r["kind"] == "stimulus"),
        "n_baseline": sum(1 for r in records if r["kind"] == "baseline"),
        "S_rel_error_median": float(np.median(s_err)) if s_err else None,
        "S_rel_error_p90": float(np.percentile(s_err, 90)) if s_err else None,
        "Cx_rel_error_median": float(np.median(c_err)) if c_err else None,
        "Cx_rel_error_p90": float(np.percentile(c_err, 90)) if c_err else None,
        "problems": problems,
        "records": records,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: v for k, v in index.items() if k != "records"} | {
        "problems": len(problems)}


if __name__ == "__main__":
    print(json.dumps(render_all(), ensure_ascii=False, indent=1))
