"""阶段 2b：AnoLCD 真实显示屏上的探针测量。

复用既有 AnoLCD PatchCore 库与冻结阈值（不重建、不重校准），
在 256 缩图上渲染，零额外插值。

分步入口：prepare → render → measure → report，各自独立，不互相串联。
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from controlled_generation.render import render

from . import protocol as P
from . import real_semu
from . import semu

# ---- 2a 实测分位决定形态（mura 的 S p25 / 中位 / p75；line 的 S 中位）----
SHAPES2 = {
    "mura_p25": {"kind": "mura_like", "shape": "ellipse", "units": "px256",
                 "center": [128.0, 128.0], "axes": [4.0, 4.0], "angle_deg": 0.0,
                 "nominal_pixels": 50},
    "mura_med": {"kind": "mura_like", "shape": "ellipse", "units": "px256",
                 "center": [128.0, 128.0], "axes": [7.0, 7.0], "angle_deg": 0.0,
                 "nominal_pixels": 154},
    "mura_p75": {"kind": "mura_like", "shape": "ellipse", "units": "px256",
                 "center": [128.0, 128.0], "axes": [12.0, 12.0], "angle_deg": 0.0,
                 "nominal_pixels": 452},
    "line_med": {"kind": "line_like", "shape": "line", "units": "px256",
                 "center": [128.0, 128.0], "length": 112.0, "width": 4.0, "angle_deg": 0.0,
                 "nominal_pixels": 460},
}
LEVELS2 = (1.0, 1.6, 2.5, 4.0, 6.3, 10.0, 16.0, 25.0)
N_BG = 60
BG_SEED = 20261027
TAU_FROZEN = 1.8928574562072753
RENDER_SEED_BASE = 20261027


def bg_list() -> list:
    """test/good 固定 60 张：未参与建库与校准，但属历史 Tfinal，仅作刺激背景。"""
    goods = sorted((real_semu.ANOLCD / "test" / "good").glob("*.png"))
    order = np.random.default_rng(BG_SEED).permutation(len(goods))
    return [goods[i] for i in sorted(order[:N_BG])]


def read_bg(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(
        (P.INPUT_SIZE, P.INPUT_SIZE), Image.Resampling.BILINEAR),
        dtype=np.float32) / np.float32(255.0)


def prepare() -> dict:
    out = P.out2_dir("stimuli")
    out.mkdir(parents=True, exist_ok=True)
    paths = bg_list()
    P.write_json(out / "protocol.json", {
        "version": P.VERSION, "stage": "2b",
        "field_of_view_mm": P.FIELD_OF_VIEW_MM, "input_size": P.INPUT_SIZE,
        "k_mm_per_px": P.K_MM_PER_PX, "pixel_area_mm2": P.PIXEL_AREA_MM2,
        "shapes": SHAPES2, "levels": list(LEVELS2), "n_backgrounds": N_BG,
        "bg_seed": BG_SEED, "tau_frozen": TAU_FROZEN,
        "backgrounds": [str(p.relative_to(real_semu.ANOLCD)) for p in paths],
        "detector": "existing AnoLCD PatchCore (80 imgs, input 256, coreset 0.01)",
        "note": "test/good 未参与建库与校准；但属历史 Tfinal，仅作背景，不评价性能",
    })
    return {"n_backgrounds": len(paths)}


def render_all(overwrite: bool = False) -> dict:
    out = P.out2_dir("stimuli")
    if (out / "index.json").exists() and not overwrite:
        raise RuntimeError("stimuli already built")
    paths = bg_list()
    records, problems, not_representable = [], [], []

    for i, path in enumerate(paths):
        rgb = read_bg(path)
        lb = float(np.median(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]))
        np.savez_compressed(out / f"{P.base_id(i)}.npz", image=rgb.astype(np.float32))
        records.append({"id": P.base_id(i), "kind": "baseline", "background_index": i,
                        "relative_path": str(path.relative_to(real_semu.ANOLCD)), "Lb": lb})

        for key, spec_base in SHAPES2.items():
            nominal = spec_base["nominal_pixels"] * P.PIXEL_AREA_MM2
            for level in LEVELS2:
                sid = P.stim_id(key, level, i)
                a_est = level * semu.c_jnd_percent(nominal) / 100.0 * lb
                if a_est < P.RESIDUAL_FLOOR * P.REPRESENT_MARGIN:
                    not_representable.append({"id": sid, "shape": key, "target_SEMU": level,
                                              "background_index": i, "A_estimated": a_est,
                                              "reason": "amplitude below 1/255"})
                    continue
                seed = RENDER_SEED_BASE + i * 100 + int(round(level * 4))
                spec = {k: v for k, v in spec_base.items() if k != "nominal_pixels"}
                try:
                    cal = semu.calibrate_amplitude(
                        lambda bg, s_, sd=seed: render(bg, s_, seed=sd), rgb, spec, float(level))
                except semu.NotReachable as exc:
                    not_representable.append({"id": sid, "shape": key, "target_SEMU": level,
                                              "background_index": i, "reason": str(exc)})
                    continue
                except RuntimeError as exc:
                    problems.append({"id": sid, "reason": str(exc)})
                    continue

                res = render(rgb, {**spec, "amplitude": cal["amplitude"]}, seed=seed)
                m = semu.measure_stimulus(res["background"], res["image"], res["M_eff"])
                np.savez_compressed(out / f"{sid}.npz",
                                    image=res["image"].astype(np.float32),
                                    M_eff=np.packbits(res["M_eff"]),
                                    M_eff_shape=np.array(res["M_eff"].shape, dtype=np.int32))
                records.append({"id": sid, "kind": "stimulus", "background_index": i,
                                "shape": key, "target_SEMU": level,
                                "amplitude": cal["amplitude"], "SEMU_actual": m["SEMU"],
                                "Cx_percent": m["Cx_percent"], "S_mm2": m["S_mm2"],
                                "effective_pixels": m["effective_pixels"], "Lb": m["Lb"],
                                "within_tolerance": cal["within_tolerance"],
                                "coverage": res["stats"]["effective_coverage"],
                                "outside_ROI": res["stats"]["outside_ROI_max_float_residual"]})

    index = {"version": P.VERSION, "shapes": list(SHAPES2), "levels": list(LEVELS2),
             "records": records, "problems": problems,
             "not_representable": not_representable,
             "count_stimuli": sum(1 for r in records if r["kind"] == "stimulus"),
             "count_baseline": sum(1 for r in records if r["kind"] == "baseline")}
    P.write_json(out / "index.json", index)
    P.write_json(out / "complete.json", {"n": len(records), "problems": len(problems),
                                         "not_representable": len(not_representable)})
    return {"stimuli": index["count_stimuli"], "baseline": index["count_baseline"],
            "problems": len(problems), "not_representable": len(not_representable)}


def measure_all(device: str = "cuda", overwrite: bool = False) -> dict:
    import sys
    out = P.out2_dir("scores")
    if (out / "scores.json").exists() and not overwrite:
        raise RuntimeError("scores already sealed")
    if str(P.ROOT) not in sys.path:
        sys.path.insert(0, str(P.ROOT))
    from localization_calibration import load_model, predict_map

    index = json.loads((P.out2_dir("stimuli") / "index.json").read_text(encoding="utf-8"))
    model = load_model("PatchCore")
    stim_dir = P.out2_dir("stimuli")

    rows = []
    for rec in index["records"]:
        with np.load(stim_dir / f"{rec['id']}.npz") as z:
            rgb = z["image"].astype(np.float32)
        score, heat = predict_map("PatchCore", model, rgb)
        peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
        row = {**rec, "score": float(score), "peak_xy": [int(peak[1]), int(peak[0])]}
        if rec["kind"] == "stimulus":
            shape = tuple(int(v) for v in np.load(stim_dir / f"{rec['id']}.npz")["M_eff_shape"])
            eff = np.unpackbits(np.load(stim_dir / f"{rec['id']}.npz")["M_eff"],
                                count=int(np.prod(shape))).reshape(shape).astype(bool)
            row["peak_in_mask"] = bool(eff[tuple(row["peak_xy"])[::-1]])
        rows.append(row)

    base = {r["background_index"]: r["score"] for r in rows if r["kind"] == "baseline"}
    for r in rows:
        r["baseline_score"] = base[r["background_index"]]
        r["delta_score"] = r["score"] - base[r["background_index"]]
        r["hit"] = bool(r["score"] > TAU_FROZEN)

    out.mkdir(parents=True, exist_ok=True)
    P.write_json(out / "scores.json", {"tau": TAU_FROZEN, "rows": rows,
                                       "baseline_fp": int(sum(1 for r in rows
                                                              if r["kind"] == "baseline" and r["hit"]))})
    P.write_json(out / "complete.json", {"n": len(rows)})
    return {"rows": len(rows), "baseline_fp": int(sum(1 for r in rows
                                                       if r["kind"] == "baseline" and r["hit"]))}


def _crossing(levels, rates, target: float):
    """线性插值求检出率达到 target 的 SEMU；未跨越时返回 None 并给出方向。"""
    for i in range(len(levels) - 1):
        a, b = rates[i], rates[i + 1]
        if a is None or b is None:
            continue
        if a < target <= b:
            if b == a:
                return float(levels[i])
            w = (target - a) / (b - a)
            return float(np.exp(np.log(levels[i]) * (1 - w) + np.log(levels[i + 1]) * w))
    if rates and rates[0] is not None and rates[0] >= target:
        return "below_min"
    return "above_max"


def report_all(n_boot: int = 1000) -> dict:
    out = P.out2_dir("report")
    out.mkdir(parents=True, exist_ok=True)
    scores = json.loads((P.out2_dir("scores") / "scores.json").read_text(encoding="utf-8"))
    rows = scores["rows"]
    stim = [r for r in rows if r["kind"] == "stimulus"]
    base = [r for r in rows if r["kind"] == "baseline"]

    cells, curves = [], {}
    for shape in SHAPES2:
        rates, semus = [], []
        for level in LEVELS2:
            sel = [r for r in stim if r["shape"] == shape and abs(r["target_SEMU"] - level) < 1e-12]
            n = len(sel)
            rate = (sum(1 for r in sel if r["hit"]) / n) if n else None
            cells.append({"shape": shape, "target_SEMU": level, "n": n,
                          "hits": int(sum(1 for r in sel if r["hit"])),
                          "detection_rate": rate,
                          "peak_in_mask_rate": (sum(1 for r in sel if r.get("peak_in_mask")) / n
                                                if n else None),
                          "SEMU_actual_median": (float(np.median([r["SEMU_actual"] for r in sel]))
                                                 if n else None)})
            rates.append(rate)
            semus.append(float(np.median([r["SEMU_actual"] for r in sel])) if n else None)
        curves[shape] = {"rates": rates, "semu": semus}

    # bootstrap：按背景有放回重采样，给 SEMU_50 / SEMU_90 的百分位区间
    rng = np.random.default_rng(20261027)
    # 只用有刺激数据的背景：部分背景 Lb 过低，所有档位都不可表示，不应计入重采样
    bg_ids = sorted({r["background_index"] for r in stim})
    boot = {shape: {"50": [], "90": []} for shape in SHAPES2}
    for _ in range(n_boot):
        pick = rng.choice(bg_ids, size=len(bg_ids), replace=True)
        for shape in SHAPES2:
            rates = []
            for level in LEVELS2:
                sel = [r for r in stim if r["shape"] == shape
                       and abs(r["target_SEMU"] - level) < 1e-12 and r["background_index"] in set(pick.tolist())]
                rates.append((sum(1 for r in sel if r["hit"]) / len(sel)) if sel else None)
            for q in ("50", "90"):
                v = _crossing(LEVELS2, rates, 0.5 if q == "50" else 0.9)
                if isinstance(v, float):
                    boot[shape][q].append(v)

    summary_cells = {}
    for shape in SHAPES2:
        c50 = _crossing(LEVELS2, curves[shape]["rates"], 0.5)
        c90 = _crossing(LEVELS2, curves[shape]["rates"], 0.9)
        b50 = boot[shape]["50"]
        b90 = boot[shape]["90"]
        summary_cells[shape] = {
            "SEMU_50": c50, "SEMU_90": c90,
            "SEMU_50_ci95": [float(np.percentile(b50, 2.5)), float(np.percentile(b50, 97.5))] if len(b50) > 20 else None,
            "SEMU_90_ci95": [float(np.percentile(b90, 2.5)), float(np.percentile(b90, 97.5))] if len(b90) > 20 else None,
            "boot_valid": len(b50),
            "monotonic": all(b >= a - 0.05 for a, b in zip([x for x in curves[shape]["rates"] if x is not None],
                                                           [x for x in curves[shape]["rates"] if x is not None][1:])),
        }

    real = json.loads((P.out2_dir("real_semu") / "summary.json").read_text(encoding="utf-8"))
    real_mura = real["by_kind"]["mura"]["SEMU"]
    verdict = {}
    for shape, info in summary_cells.items():
        s50 = info["SEMU_50"]
        if not isinstance(s50, float):
            verdict[shape] = f"无法定 SEMU_50（{s50}）"
            continue
        if real_mura["median"] < s50:
            tag = "A：真实缺陷中位 SEMU 低于检测器 50% 检出点 → 可探测性边界可解释零增益"
        elif isinstance(info["SEMU_90"], float) and real_mura["median"] > info["SEMU_90"]:
            tag = "B：真实缺陷中位 SEMU 高于 90% 检出点 → 缺陷在可靠检出区内，零增益应归因训练侧"
        else:
            tag = "C：过渡区，报连续值，不做二元结论"
        verdict[shape] = tag

    summary = {"version": P.VERSION, "stage": "2b", "tau": TAU_FROZEN,
               "baseline_fp": scores["baseline_fp"], "baseline_n": len(base),
               "usable_backgrounds": len({r["background_index"] for r in stim}),
               "cells": cells, "per_shape": summary_cells,
               "real_mura_SEMU": {"median": real_mura["median"], "p25": real_mura["p25"],
                                  "p75": real_mura["p75"], "n": real_mura["n"]},
               "verdict": verdict,
               "note": ("区间为背景重采样的百分位区间，非 MIL-HDBK-1823A 的严格 a90/95；"
                        "背景取自 test/good，未参与建库与校准但属历史 Tfinal；"
                        "Cx 未做 sRGB 线性化，SEMU 只作相对比较")}
    P.write_json(out / "summary.json", summary)
    P.write_json(out / "complete.json", {"stage": "2b_report"})
    return summary
