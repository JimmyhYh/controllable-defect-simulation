"""spot 形态探针：用于首次触发判据的 A 分支。

目的
----
阶段 2b 的四个形态最小只到 mura_p25（S = 8.0 mm²），而真实 spot 是 0.96 mm²，
小 8.3 倍。因此从未测过"检测器对 spot 尺度的 50% 检出点"，判据的 A 分支
（真实缺陷 SEMU 中位 < a50）一直没有被触发过。

本模块补一个 spot 尺度的形态，用同一批背景、同一套档位、同一个冻结 τ，
测出它的检出曲线与 a50，从而首次给出 A 分支的判定依据。

形态选择
--------
axes = 1.4 px → 名义面积 6.16 px → S ≈ 0.985 mm²，对应真实 spot 的 0.96 mm²。
等效直径约 2.8 px，在 stride 8 的特征图上占 0.35 个位置，与真实 spot 一致。

可表示性（已核算）：60 个背景 Lb 中位 0.194，axes=1.4 时 SEMU 上限中位约 51，
46/60 个背景能覆盖到真实 spot 的 SEMU 中位 11.64。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from controlled_generation.render import render

from . import protocol as P
from . import real_semu, semu
from .stage2 import BG_SEED, N_BG, RENDER_SEED_BASE, TAU_FROZEN, bg_list, read_bg

OUT = P.ROOT / "results" / "pod_probe" / "spot_probe_v1"

SHAPES_SPOT = {
    "spot_xs": {"kind": "mura_like", "shape": "ellipse", "units": "px256",
                "center": [128.0, 128.0], "axes": [1.4, 1.4], "angle_deg": 0.0,
                "nominal_pixels": 6.16},
}
LEVELS_SPOT = (1.0, 1.6, 2.5, 4.0, 6.3, 10.0, 16.0, 25.0)


def _out(stage: str) -> Path:
    return OUT / stage


def render_all(overwrite: bool = False) -> dict:
    target = _out("stimuli")
    if (target / "index.json").exists() and not overwrite:
        raise RuntimeError("spot stimuli already built")
    target.mkdir(parents=True, exist_ok=True)

    paths = bg_list()
    records, problems, not_repr = [], [], []
    stage2_stim = P.ROOT / "results" / "pod_probe" / "anolcd_stage2_v1" / "stimuli"

    for i, path in enumerate(paths):
        rgb = read_bg(path)
        lb = float(np.median(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
                             + 0.0722 * rgb[..., 2]))
        # 基线：直接复用 2b 的同背景原图，保证两批刺激的背景完全一致
        base_src = stage2_stim / f"{P.base_id(i)}.npz"
        if base_src.exists():
            with np.load(base_src) as handle:
                base_image = np.array(handle["image"], dtype=np.float32)
            np.savez_compressed(target / f"{P.base_id(i)}.npz", image=base_image)
            records.append({"id": P.base_id(i), "kind": "baseline", "background_index": i,
                            "Lb": lb})
        else:
            problems.append({"id": P.base_id(i), "reason": "baseline not found in stage2"})

        for key, spec_base in SHAPES_SPOT.items():
            nominal = spec_base["nominal_pixels"] * P.PIXEL_AREA_MM2
            for level in LEVELS_SPOT:
                sid = P.stim_id(key, level, i)
                a_est = level * semu.c_jnd_percent(nominal) / 100.0 * lb
                if a_est < P.RESIDUAL_FLOOR * P.REPRESENT_MARGIN:
                    not_repr.append({"id": sid, "shape": key, "target_SEMU": level,
                                     "background_index": i, "A_estimated": a_est,
                                     "reason": "amplitude below 1/255"})
                    continue
                seed = RENDER_SEED_BASE + i * 100 + int(round(level * 4))
                spec = {k: v for k, v in spec_base.items() if k != "nominal_pixels"}
                try:
                    cal = semu.calibrate_amplitude(
                        lambda bg, s_, sd=seed: render(bg, s_, seed=sd),
                        rgb, spec, float(level))
                except semu.NotReachable as exc:
                    not_repr.append({"id": sid, "shape": key, "target_SEMU": level,
                                     "background_index": i, "reason": str(exc)})
                    continue
                except RuntimeError as exc:
                    problems.append({"id": sid, "reason": str(exc)})
                    continue

                res = render(rgb, {**spec, "amplitude": cal["amplitude"]}, seed=seed)
                m = semu.measure_stimulus(res["background"], res["image"], res["M_eff"])
                np.savez_compressed(target / f"{sid}.npz",
                                    image=res["image"].astype(np.float32))
                records.append({
                    "id": sid, "kind": "stimulus", "background_index": i,
                    "shape": key, "target_SEMU": level,
                    "amplitude": cal["amplitude"], "SEMU_actual": m["SEMU"],
                    "Cx_percent": m["Cx_percent"], "S_mm2": m["S_mm2"],
                    "effective_pixels": m["effective_pixels"], "Lb": lb,
                    "within_tolerance": cal["within_tolerance"],
                    "coverage": res["stats"]["effective_coverage"]})

    index = {"version": "spot_probe_v1", "shapes": list(SHAPES_SPOT),
             "levels": list(LEVELS_SPOT), "n_backgrounds": N_BG, "bg_seed": BG_SEED,
             "tau_frozen": TAU_FROZEN, "records": records, "problems": problems,
             "not_representable": not_repr,
             "count_stimuli": len(records),
             "purpose": "首次触发判据 A 分支：测 spot 尺度的 a50"}
    (target / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    return {"stimuli": len(records), "problems": len(problems),
            "not_representable": len(not_repr)}


def measure_all(overwrite: bool = False) -> dict:
    from localization_calibration import load_model, predict_map

    target = _out("scores")
    record_path = target / "scores.json"
    if record_path.exists() and not overwrite:
        raise RuntimeError("spot scores already sealed")
    target.mkdir(parents=True, exist_ok=True)

    index = json.loads((_out("stimuli") / "index.json").read_text(encoding="utf-8"))
    model = load_model("PatchCore")
    tau = TAU_FROZEN

    rows = []
    for i, rec in enumerate(index["records"], 1):
        npz = _out("stimuli") / f"{rec['id']}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as handle:
            image = np.array(handle["image"], dtype=np.float32)
        score, _map = predict_map("PatchCore", model, image)
        rows.append({**{k: rec.get(k) for k in
                        ("id", "kind", "shape", "target_SEMU", "background_index",
                         "SEMU_actual", "S_mm2", "Cx_percent")},
                     "score": float(score), "hit": bool(score > tau)})

    baseline_fp = sum(1 for r in rows if r["kind"] == "baseline" and r["hit"])
    result = {"condition": "PatchCore", "tau": tau, "n": len(rows),
              "baseline_false_positive": baseline_fp,
              "baseline_n": sum(1 for r in rows if r["kind"] == "baseline")}
    record_path.write_text(json.dumps({**result, "rows": rows}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    return result


def report() -> dict:
    from . import pod as POD

    data = json.loads((_out("scores") / "scores.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["kind"] == "stimulus"]
    tau = data["tau"]

    cells = {}
    for level in LEVELS_SPOT:
        sel = [r for r in rows if abs(r["target_SEMU"] - level) < 1e-9]
        if not sel:
            continue
        cells["%g" % level] = {"n": len(sel),
                               "hits": sum(1 for r in sel if r["hit"]),
                               "rate": sum(1 for r in sel if r["hit"]) / len(sel),
                               "SEMU_median": float(np.median([r["SEMU_actual"] for r in sel]))}

    x = np.log(np.array([r["SEMU_actual"] for r in rows]))
    y = np.array([r["hit"] for r in rows], dtype=float)
    bg = [r["background_index"] for r in rows]
    beta, ok = POD.fit(x, y, "probit")
    a50 = POD.level_at(beta, 0.5) if beta[1] > 0 else None
    a90 = POD.level_at(beta, 0.9) if beta[1] > 0 else None
    b50, b90 = POD.bootstrap(x, y, bg) if ok and beta[1] > 0 else (np.array([]), np.array([]))

    rates = [c["rate"] for c in cells.values()]
    lo, hi = (min(rates), max(rates)) if rates else (None, None)
    a50_identifiable = bool(lo is not None and lo <= 0.5 < hi)

    # 真实 spot 的分布
    real = json.loads((P.ROOT / "results" / "pod_probe" / "anolcd_stage2_v1"
                       / "real_semu" / "summary.json").read_text(encoding="utf-8"))
    real_spot = real["by_kind"]["spot"]["SEMU"]
    real_med = float(real_spot["median"])
    real_S = float(real["by_kind"]["spot"]["S_mm2"]["median"])

    # 特征占位：等效直径 / stride 8
    stim_S = float(np.median([r["S_mm2"] for r in rows]))
    diameter_px = 2.0 * np.sqrt((stim_S / P.PIXEL_AREA_MM2) / np.pi)
    occupancy = diameter_px / 8.0

    # A 分支判定：a50 与真实中位比较；a50 不可估时用最高档率作下界判断
    if a50_identifiable:
        a50_value = a50
        a50_lower = None
        exceeds = bool(a50 is not None and a50 > real_med)
        basis = "point estimate"
    else:
        a50_value = None
        a50_lower = float(np.exp(x.max()))
        exceeds = bool(hi is not None and hi < 0.5)
        basis = "a50 not identifiable; highest-cell rate below 50% implies a50 above data range"

    if not exceeds:
        branch = "not_A"
        detail = "a50 不高于真实中位，未触发 A 分支"
    elif occupancy < 1.0:
        branch = "A2"
        detail = ("a50 高于真实缺陷 SEMU 中位，且该形态在特征图上占 "
                  f"{occupancy:.2f} 个位置（< 1）→ 面积过小导致特征不可表达，"
                  "不是对比度不足；补样本无效，应改检测器结构")
    else:
        branch = "A1"
        detail = ("a50 高于真实缺陷 SEMU 中位，且该形态占位 "
                  f"{occupancy:.2f} ≥ 1 → 特征可表达，属对比度不足；补样本可能有效")

    result = {
        "condition": "PatchCore", "tau": tau,
        "shape": list(SHAPES_SPOT),
        "n_stimulus": len(rows),
        "baseline_false_positive": data["baseline_false_positive"],
        "baseline_n": data["baseline_n"],
        "cells": cells,
        "lowest_cell_rate": lo, "highest_cell_rate": hi,
        "beta_probit": [float(v) for v in beta],
        "a50_fitted_insample": a50, "a90_fitted_insample": a90,
        "a50": a50 if a50_identifiable else None,
        "a90": a90 if a50_identifiable else None,
        "a50_ci95": ([float(np.percentile(b50, 2.5)), float(np.percentile(b50, 97.5))]
                     if len(b50) else None),
        "a90_95": float(np.percentile(b90, 5)) if len(b90) else None,
        "a50_identifiable": a50_identifiable,
        "stim_S_mm2_median": stim_S,
        "stim_equiv_diameter_px": float(diameter_px),
        "feature_occupancy": float(occupancy),
        "real_spot_SEMU_median": real_med,
        "real_spot_S_mm2_median": real_S,
        "branch": branch,
        "branch_basis": basis,
        "a50_lower_bound": a50_lower,
        "detail": detail,
    }

    (_out("report")).mkdir(parents=True, exist_ok=True)
    (_out("report") / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys

    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("render", "all"):
        print(json.dumps(render_all(), ensure_ascii=False))
    if step in ("measure", "all"):
        print(json.dumps(measure_all(), ensure_ascii=False))
    if step in ("report", "all"):
        r = report()
        keys = ("n_stimulus", "baseline_false_positive", "baseline_n",
                "lowest_cell_rate", "highest_cell_rate", "a50", "a90", "a90_95",
                "a50_identifiable", "feature_occupancy",
                "real_spot_SEMU_median", "branch", "detail")
        print(json.dumps({k: r.get(k) for k in keys}, ensure_ascii=False, indent=1))
