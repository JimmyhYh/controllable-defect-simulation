"""阶段 2a：AnoLCD 真实缺陷的 SEMU 统计（纯 CPU，不加载模型）。

目的：得到真实缺陷的 S / Cx / SEMU 分布，用于确定 2b 的形态参数，
并作为主线判据 5.2 的"任务难度"一侧。

口径（与 SEMI D31-1102 一致）：
    S    = GT 阳性像素数 × PIXEL_AREA_MM2
    Lb   = 同图内 GT 外扩保护区之外的亮度均值
    Lo   = GT 内的亮度均值
    Cx   = (Lo − Lb) / Lb × 100   (%)
    SEMU = |Cx| / (1.97 / S^0.33 + 0.72)
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from . import protocol as P
from . import semu

ANOLCD = P.ROOT / "data" / "weak_localization" / "AnoLCD" / "display"
KINDS = ("mura", "line", "spot", "scrathes", "cylinder")
GUARD_PX = 8           # 背景环带外沿，256 尺度上约 3.2 mm
RING_INNER_PX = 2      # 环带内沿：跳过缺陷软边过渡
GT_FRACTION_MAX = 0.20 # 超过此占比视为"整块区域异常"，SEMU 的"背景"前提不成立
CX_MAX_PERCENT = 100.0 # 超过此对比度视为非 Mura 型（背景/缺陷取样失配），不适用 SEMU
SRC_SIZE = 1024


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def read_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(
        (P.INPUT_SIZE, P.INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255.0)


def read_mask(path) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L").resize(
        (P.INPUT_SIZE, P.INPUT_SIZE), Image.Resampling.NEAREST), dtype=np.uint8)
    return m > 127


def measure_one(image_path, mask_path, guard: int = GUARD_PX,
                ring_inner: int = RING_INNER_PX) -> dict:
    rgb = read_rgb(image_path)
    gt = read_mask(mask_path)
    y = luma(rgb)

    n_gt = int(gt.sum())
    if n_gt == 0:
        return {"error": "empty ground truth"}
    frac = n_gt / gt.size
    if frac > GT_FRACTION_MAX:
        return {"error": f"gt_fraction {frac:.3f} exceeds {GT_FRACTION_MAX}; "
                         f"no uniform background available for SEMU"}

    # 背景取紧邻缺陷的环带，而非全图外部：显示屏图常含大面积暗区，
    # 用全图外部会把 Lb 拉到近 0，使 Cx 失真。
    outer = binary_dilation(gt, iterations=guard)
    inner = binary_dilation(gt, iterations=ring_inner) if ring_inner > 0 else gt
    ring = outer & ~inner
    used_fallback = False
    if int(ring.sum()) < 50:                     # 小缺陷时环带过窄，退回 GT 外全部
        ring = ~gt
        used_fallback = True

    # 用中位数而非均值：显示屏图常含黑色非显示区，均值会被暗区拉低而失真
    lb = float(np.median(y[ring]))
    lo = float(np.median(y[gt]))
    s_mm2 = n_gt * P.PIXEL_AREA_MM2
    cx = semu.contrast_percent(lo, lb)
    if abs(cx) > CX_MAX_PERCENT:
        return {"error": f"|Cx| {abs(cx):.1f}% exceeds {CX_MAX_PERCENT}; "
                         f"ring/Lo sampling mismatch, not a Mura-type contrast"}
    return {
        "gt_pixels_256": n_gt,
        "gt_fraction": frac,
        "ring_pixels": int(ring.sum()),
        "S_mm2": s_mm2,
        "Lb": lb, "Lo": lo, "Cx_percent": cx,
        "C_jnd_percent": semu.c_jnd_percent(s_mm2),
        "SEMU": semu.semu_value(cx, s_mm2),
        "background_fallback": used_fallback,
    }


def _pct(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {"n": int(a.size), "min": float(a.min()), "p10": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4]), "max": float(a.max())}


def run(overwrite: bool = False) -> dict:
    out = P.out2_dir("real_semu")
    if (out / "complete.json").exists() and not overwrite:
        raise RuntimeError("real_semu already sealed; rename the old dir or pass overwrite")
    out.mkdir(parents=True, exist_ok=True)

    rows, problems, not_applicable = [], [], []
    for kind in KINDS:
        img_dir, gt_dir = ANOLCD / "test" / kind, ANOLCD / "ground_truth" / kind
        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = gt_dir / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                problems.append({"image": str(img_path), "reason": "mask missing"})
                continue
            try:
                m = measure_one(img_path, mask_path)
            except Exception as exc:                      # 记录后继续，不中断整批
                problems.append({"image": str(img_path), "reason": str(exc)})
                continue
            if "error" in m:
                not_applicable.append(
                    {"kind": kind, "image": img_path.name, "reason": m["error"]})
                continue
            rows.append({"kind": kind, "image": img_path.name, **m})

    by_kind = {}
    for kind in KINDS:
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        by_kind[kind] = {
            "S_mm2": _pct([r["S_mm2"] for r in sel]),
            "Cx_percent": _pct([r["Cx_percent"] for r in sel]),
            "SEMU": _pct([r["SEMU"] for r in sel]),
            "below_SEMU_1": int(sum(1 for r in sel if r["SEMU"] < 1.0)),
            "background_fallback": int(sum(1 for r in sel if r["background_fallback"])),
        }

    total = {
        "S_mm2": _pct([r["S_mm2"] for r in rows]),
        "Cx_percent": _pct([r["Cx_percent"] for r in rows]),
        "SEMU": _pct([r["SEMU"] for r in rows]),
        "below_SEMU_1": int(sum(1 for r in rows if r["SEMU"] < 1.0)),
        "background_fallback": int(sum(1 for r in rows if r["background_fallback"])),
    }

    summary = {"version": P.VERSION, "stage": "real_semu",
               "field_of_view_mm": P.FIELD_OF_VIEW_MM, "input_size": P.INPUT_SIZE,
               "k_mm_per_px": P.K_MM_PER_PX, "pixel_area_mm2": P.PIXEL_AREA_MM2,
               "guard_px": GUARD_PX, "guard_mm": GUARD_PX * P.K_MM_PER_PX,
               "source": str(ANOLCD), "n_images": len(rows), "problems": problems,
               "not_applicable": not_applicable,
               "not_applicable_by_kind": {k: sum(1 for x in not_applicable if x["kind"] == k)
                                          for k in KINDS},
               "by_kind": by_kind, "total": total,
               "note": ("AnoLCD display test 的 750 张异常图与作者 GT；"
                        "仅统计任务难度，不评价模型、不宣称独立验证；"
                        "Cx 用编码 RGB 加权代理，未做 sRGB 线性化")}
    P.write_json(out / "summary.json", summary)
    P.write_json(out / "per_image.json", {"rows": rows})
    with (out / "per_image.csv").open("w", encoding="utf-8") as fh:
        cols = ["kind", "image", "gt_pixels_256", "gt_fraction", "S_mm2", "Lb", "Lo",
                "Cx_percent", "C_jnd_percent", "SEMU", "background_fallback"]
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    P.write_json(out / "complete.json", {"stage": "real_semu", "n": len(rows),
                                         "problems": len(problems),
                                         "not_applicable": len(not_applicable)})
    return summary
