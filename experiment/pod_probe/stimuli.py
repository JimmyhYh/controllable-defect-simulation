"""刺激构造：按目标 SEMU 反推幅度，调用既有渲染实现，落盘可重算的数组。

只读复用 controlled_generation.render.render；本模块不改其行为。
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from controlled_generation.render import render

from . import protocol as P
from . import semu

RENDER_SEED_BASE = 20261026


def read_background(relative: str) -> tuple[np.ndarray, str]:
    path = P.DATA / relative
    raw = np.asarray(Image.open(path).convert("RGB").resize(
        (P.INPUT_SIZE, P.INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
    rgb = raw.astype(np.float32) / np.float32(255.0)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return rgb, digest


def spec_for(shape_key: str, amplitude: float) -> dict:
    base = dict(P.SHAPES[shape_key])
    base.pop("nominal_pixels", None)
    base["amplitude"] = amplitude
    return base


def build(overwrite: bool = False) -> dict:
    out = P.out_dir("stimuli")
    index_path = out / "index.json"
    if index_path.exists() and not overwrite:
        raise RuntimeError("stimuli already sealed; use verify-stimuli or a new branch")

    out.mkdir(parents=True, exist_ok=True)
    backgrounds = P.background_paths()
    records = []

    # 基线：P20 原图，只读一次，供两个形态复用
    for i, rel in enumerate(backgrounds):
        rgb, digest = read_background(rel)
        np.savez_compressed(out / f"{P.base_id(i)}.npz",
                            image=rgb.astype(np.float32), source_sha256=np.array([digest]))
        records.append({"id": P.base_id(i), "kind": "baseline", "background_index": i,
                        "relative_path": rel, "source_sha256": digest,
                        "Lb": float((0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean())})

    problems = []
    not_representable = []
    for i, rel in enumerate(backgrounds):
        rgb, digest = read_background(rel)
        lb = float((0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean())
        for shape_key in P.SHAPES:
            nominal_mm2 = P.SHAPES[shape_key]["nominal_pixels"] * P.PIXEL_AREA_MM2
            for level in P.SEMU_LEVELS:
                sid = P.stim_id(shape_key, level, i)
                # 可表示性预判：残差上限约为 A，A 低于 1/255 时图像不变，刺激不存在
                a_est = level * semu.c_jnd_percent(nominal_mm2) / 100.0 * lb
                if a_est < P.RESIDUAL_FLOOR * P.REPRESENT_MARGIN:
                    not_representable.append({
                        "id": sid, "shape": shape_key, "target_SEMU": level,
                        "background_index": i, "Lb": lb, "A_estimated": a_est,
                        "floor": P.RESIDUAL_FLOOR,
                        "reason": "amplitude below 1/255; image would be unchanged at encode level",
                    })
                    continue
                seed = RENDER_SEED_BASE + i * 100 + int(round(level * 4))
                try:
                    cal = semu.calibrate_amplitude(
                        lambda bg, spec, s=seed: render(bg, spec, seed=s),
                        rgb, spec_for(shape_key, 0.0), float(level))
                except semu.NotReachable as exc:
                    not_representable.append({
                        "id": sid, "shape": shape_key, "target_SEMU": level,
                        "background_index": i, "Lb": lb, "A_estimated": a_est,
                        "floor": P.RESIDUAL_FLOOR, "reason": str(exc),
                    })
                    continue
                except RuntimeError as exc:
                    problems.append({"id": sid, "reason": str(exc)})
                    continue

                res = render(rgb, spec_for(shape_key, cal["amplitude"]), seed=seed)
                m = semu.measure_stimulus(res["background"], res["image"], res["M_eff"])
                np.savez_compressed(
                    out / f"{sid}.npz",
                    image=res["image"].astype(np.float32),
                    M_eff=np.packbits(res["M_eff"]),
                    M_eff_shape=np.array(res["M_eff"].shape, dtype=np.int32),
                )
                records.append({
                    "id": sid, "kind": "stimulus", "background_index": i, "relative_path": rel,
                    "source_sha256": digest, "shape": shape_key, "target_SEMU": level,
                    "amplitude": cal["amplitude"], "render_seed": seed,
                    "SEMU_actual": m["SEMU"], "Cx_percent": m["Cx_percent"],
                    "S_mm2": m["S_mm2"], "effective_pixels": m["effective_pixels"],
                    "Lb": m["Lb"], "C_jnd_percent": m["C_jnd_percent"],
                    "relative_error": cal["relative_error"],
                    "within_tolerance": cal["within_tolerance"],
                    "amplitude_iterations": cal["iterations"],
                    "render_valid": res["stats"]["valid"],
                    "coverage_below_warn": res["stats"]["effective_coverage"] < P.COVERAGE_WARN,
                    "effective_coverage": res["stats"]["effective_coverage"],
                    "outside_ROI_max_float_residual": res["stats"]["outside_ROI_max_float_residual"],
                    "roi_clipped_channel_fraction": res["stats"]["roi_clipped_channel_fraction"],
                    "geometry": res["stats"]["geometry"],
                })

    index = {"version": P.VERSION, "field_of_view_mm": P.FIELD_OF_VIEW_MM,
             "k_mm_per_px": P.K_MM_PER_PX, "pixel_area_mm2": P.PIXEL_AREA_MM2,
             "semu_levels": list(P.SEMU_LEVELS), "shapes": list(P.SHAPES),
             "polarity": P.POLARITY, "render_seed_base": RENDER_SEED_BASE,
             "residual_floor": P.RESIDUAL_FLOOR, "coverage_warn": P.COVERAGE_WARN,
             "backgrounds": backgrounds, "records": records, "problems": problems,
             "not_representable": not_representable,
             "count_stimuli": sum(1 for r in records if r["kind"] == "stimulus"),
             "count_baseline": sum(1 for r in records if r["kind"] == "baseline"),
             "count_not_representable": len(not_representable)}
    P.write_json(index_path, index)
    P.write_json(out / "complete.json", {"stage": "stimuli", "records": len(records),
                                         "problems": len(problems),
                                         "not_representable": len(not_representable)})
    return index


def load_effective(path) -> np.ndarray:
    with np.load(path) as z:
        shape = tuple(int(v) for v in z["M_eff_shape"])
        return np.unpackbits(z["M_eff"], count=int(np.prod(shape))).reshape(shape).astype(bool)


def load_image(path) -> np.ndarray:
    with np.load(path) as z:
        return z["image"].astype(np.float32)


def verify() -> dict:
    out = P.out_dir("stimuli")
    index_path = out / "index.json"
    if not index_path.exists():
        raise RuntimeError("no sealed stimuli")
    index = __import__("json").loads(index_path.read_text(encoding="utf-8"))
    problems = []
    warnings = []

    for rec in index["records"]:
        npz = out / f"{rec['id']}.npz"
        if not npz.exists():
            problems.append({"id": rec["id"], "reason": "missing npz"})
            continue
        if rec["kind"] != "stimulus":
            continue
        if rec["effective_pixels"] <= 0:
            problems.append({"id": rec["id"], "reason": "empty effective region"})
        if rec["outside_ROI_max_float_residual"] != 0:
            problems.append({"id": rec["id"], "reason": "residual outside ROI"})
        if not rec["within_tolerance"]:
            problems.append({"id": rec["id"],
                             "reason": f"SEMU off target by {rec['relative_error']:.3%}"})
        if rec.get("coverage_below_warn"):
            warnings.append({"id": rec["id"],
                             "coverage": rec["effective_coverage"],
                             "note": "effective region shrinks at low amplitude; SEMU uses measured area"})

    return {"records": len(index["records"]), "stimuli": index["count_stimuli"],
            "not_representable": index.get("count_not_representable", 0),
            "coverage_warnings": len(warnings), "warnings": warnings[:20],
            "problems": problems, "passed": not problems}
