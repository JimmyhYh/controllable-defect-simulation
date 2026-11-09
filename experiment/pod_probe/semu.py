"""SEMU / JND 口径与被测幅度标定。

公式依据 SEMI D31-1102 附录 1（A1-1、A1-2）：
    C_jnd = 1.97 / S^0.33 + 0.72     单位 %，S 单位 mm^2
    SEMU  = |Cx| / C_jnd              Cx 单位 %（相对背景 100%）
本模块不做亮度线性化之外的任何归一化；输入为 [0,1] float 亮度。
"""

from __future__ import annotations

import numpy as np

from . import protocol as P


def c_jnd_percent(area_mm2: float) -> float:
    if area_mm2 <= 0:
        raise ValueError("area must be positive")
    return 1.97 / (area_mm2 ** 0.33) + 0.72


def contrast_percent(lo: float, lb: float) -> float:
    """带符号相对对比度（%）。Lo、Lb 为 [0,1] 亮度。"""
    if lb <= 0:
        raise ValueError("background luminance must be positive")
    return (lo - lb) / lb * 100.0


def semu_value(cx_percent: float, area_mm2: float) -> float:
    return abs(cx_percent) / c_jnd_percent(area_mm2)


def area_mm2(effective_pixels: int) -> float:
    return float(effective_pixels) * P.PIXEL_AREA_MM2


def measure_stimulus(background: np.ndarray, generated: np.ndarray, effective: np.ndarray) -> dict:
    """从渲染结果实测 Cx、S、SEMU。亮度取 RGB 加权（编码值代理，报告须标注）。"""
    def luma(rgb: np.ndarray) -> np.ndarray:
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    y_bg = luma(background)
    y_im = luma(generated)
    lb = float(y_bg.mean())
    lo = float(y_im[effective].mean()) if effective.any() else lb
    cx = contrast_percent(lo, lb)
    s = area_mm2(int(effective.sum()))
    return {
        "Lb": lb,
        "Lo": lo,
        "Cx_percent": cx,
        "S_mm2": s,
        "effective_pixels": int(effective.sum()),
        "C_jnd_percent": c_jnd_percent(s) if s > 0 else None,
        "SEMU": semu_value(cx, s) if s > 0 else 0.0,
    }


class NotReachable(RuntimeError):
    """目标 SEMU 低于该背景的可表示下限：再小的幅度会让有效区为空。"""


def calibrate_amplitude(render_fn, background: np.ndarray, spec_base: dict, target_semu: float,
                        max_iter: int = P.AMP_MAX_ITER, tol: float = P.AMP_TOLERANCE) -> dict:
    """二分搜索单边幅度，使实测 SEMU 命中目标。

    SEMU 对幅度单调递增（幅度同时抬高 Cx 并扩大有效面积），故二分可靠。
    下界由实际 alpha 峰值与 1/255 共同决定，避免压到量化下限以下得到空域。
    """
    probe = render_fn(background, {**spec_base, "amplitude": P.AMPLITUDE_LIMIT * 0.4})
    alpha_peak = float(np.asarray(probe["alpha"]).max())
    if alpha_peak <= 0:
        raise NotReachable("alpha peak is zero")
    lo_a = P.RESIDUAL_FLOOR / alpha_peak * 1.05
    hi_a = P.AMPLITUDE_LIMIT
    if lo_a >= hi_a:
        raise NotReachable("minimum representable amplitude exceeds limit")

    # 先看下界处的 SEMU；若已高于目标，说明目标低于可测下限
    out0 = render_fn(background, {**spec_base, "amplitude": P.POLARITY * lo_a})
    m0 = measure_stimulus(out0["background"], out0["image"], out0["M_eff"])
    if m0["effective_pixels"] <= 0 or m0["SEMU"] > target_semu * (1 + tol):
        raise NotReachable(f"SEMU at minimum amplitude is {m0['SEMU']:.3f} > target {target_semu}")

    best = None
    history = []
    for _ in range(max_iter):
        a = 0.5 * (lo_a + hi_a)
        out = render_fn(background, {**spec_base, "amplitude": P.POLARITY * a})
        stats = out["stats"]
        # 只接受两类硬失败；coverage 低于 0.90 是软边缺陷在低幅度下的必然结果，
        # 属于被测量的事实，记录而不判失败。
        if stats["outside_ROI_max_float_residual"] != 0:
            raise RuntimeError("residual outside ROI")
        m = measure_stimulus(out["background"], out["image"], out["M_eff"])
        if m["effective_pixels"] <= 0:
            lo_a = a
            continue
        err = (m["SEMU"] - target_semu) / target_semu
        history.append((a, m["SEMU"]))
        cand = {"amplitude": P.POLARITY * a, "measure": m, "stats": stats,
                "relative_error": err, "iterations": len(history)}
        if best is None or abs(err) < abs(best["relative_error"]):
            best = cand
        if abs(err) <= tol:
            break
        if m["SEMU"] < target_semu:
            lo_a = a
        else:
            hi_a = a

    best["history"] = [{"amplitude": round(a, 6), "SEMU": round(s, 6)} for a, s in history]
    best["within_tolerance"] = abs(best["relative_error"]) <= tol
    return best
