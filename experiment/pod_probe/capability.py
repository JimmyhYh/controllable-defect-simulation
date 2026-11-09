"""DRAEM-A/B 能力卡：对已缓存的探针刺激推理，并采基线误报。

与 PatchCore 行同口径：
- 刺激来自同一批 1167 个 npz（4 形态 × 8 档 + 60 基线）；
- 阈值沿用各自冻结值，不重新校准；
- 基线误报单独统计（特异度），与检出率一并给出。

跨检测器可比量只有检出率与 a50/a90。DRAEM 的 τ 在 softmax 概率域
（上限 1.0），其 score/τ 与 PatchCore 不可比，本模块不输出该比值。
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from localization_calibration import load_model, predict_map

ROOT = Path(__file__).resolve().parent.parent
STAGE2 = ROOT / "results" / "pod_probe" / "anolcd_stage2_v1"
OUT = ROOT / "results" / "pod_probe" / "capability_v1"

MODELS = {"A": "A20261020", "B": "B20261020"}


def finalize(codes=("A", "B")) -> dict:
    out = {}
    for cond in codes:
        path = OUT / f"draem_{cond}" / "scores.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        base_rate = data["baseline_false_positive_rate"]
        for shape, v in data["by_shape"].items():
            hi, lo = v["highest_cell_rate"], v["lowest_cell_rate"]
            v["no_response"] = bool(hi is not None and hi < 0.50)
            v["response_minus_baseline"] = (None if hi is None else hi - base_rate)
            v["a50_identifiable"] = bool(lo is not None and lo <= 0.5 < hi)
            v["a90_identifiable"] = bool(lo is not None and lo < 0.9 <= hi)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        summary = {k: v for k, v in data.items() if k != "rows"}
        (OUT / f"draem_{cond}" / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        out[cond] = summary
    return out


def frozen_tau(condition: str) -> float:
    path = ROOT / "results" / "weak_localization" / f"calibration_{condition}.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])


def load_records():
    idx = json.loads((STAGE2 / "stimuli" / "index.json").read_text(encoding="utf-8"))
    return idx["records"], idx


def run(cond: str, overwrite: bool = False) -> dict:
    if cond not in MODELS:
        raise ValueError(f"condition {cond} not in {list(MODELS)}")
    condition = MODELS[cond]
    target = OUT / f"draem_{cond}"
    record = target / "scores.json"
    if record.exists() and not overwrite:
        raise RuntimeError(f"draem {cond} already sealed; use --overwrite or a new branch")

    tau = frozen_tau(condition)
    records, idx = load_records()
    model = load_model(condition)

    rows = []
    skipped = 0
    for i, rec in enumerate(records, 1):
        npz = STAGE2 / "stimuli" / f"{rec['id']}.npz"
        if not npz.exists():
            skipped += 1
            continue
        # NpzFile 必须显式关闭，否则 1167 个句柄会累积并导致进程被系统终止
        with np.load(npz) as handle:
            image = np.array(handle["image"], dtype=np.float32)
        score, _map = predict_map(cond, model, image)
        del image
        if i % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        rows.append({
            "id": rec["id"],
            "kind": rec["kind"],
            "shape": rec.get("shape"),
            "target_SEMU": rec.get("target_SEMU"),
            "SEMU_actual": rec.get("SEMU_actual"),
            "background_index": rec.get("background_index"),
            "score": float(score),
            "hit": bool(score > tau),
        })

    baseline = [r for r in rows if r["kind"] == "baseline"]
    stim = [r for r in rows if r["kind"] == "stimulus"]
    base_fp = sum(1 for r in baseline if r["hit"])

    cells = {}
    for shape in sorted({r["shape"] for r in stim}):
        for level in sorted({r["target_SEMU"] for r in stim if r["shape"] == shape}):
            sel = [r for r in stim if r["shape"] == shape and r["target_SEMU"] == level]
            if not sel:
                continue
            cells["%s@%g" % (shape, level)] = {
                "n": len(sel),
                "hits": sum(1 for r in sel if r["hit"]),
                "detection_rate": sum(1 for r in sel if r["hit"]) / len(sel),
            }

    by_shape = {}
    for shape in sorted({r["shape"] for r in stim}):
        sel = [r for r in stim if r["shape"] == shape]
        rates = [c["detection_rate"] for k, c in cells.items() if k.startswith(shape + "@")]
        lo, hi = (min(rates), max(rates)) if rates else (None, None)
        by_shape[shape] = {
            "n": len(sel),
            "detection_rate_overall": sum(1 for r in sel if r["hit"]) / len(sel),
            "lowest_cell_rate": lo,
            "highest_cell_rate": hi,
            "no_response": bool(hi is not None and hi < 0.50),
            "saturated": bool(lo is not None and lo >= 0.80),
            "a50_identifiable": bool(lo is not None and lo <= 0.5 < hi),
            "a90_identifiable": bool(lo is not None and lo < 0.9 <= hi),
        }

    result = {
        "condition": condition,
        "tau": tau,
        "tau_domain": "softmax probability, max over pixels",
        "n_images": len(rows),
        "n_baseline": len(baseline),
        "baseline_false_positive": base_fp,
        "baseline_false_positive_rate": (base_fp / len(baseline)) if baseline else None,
        "n_stimulus": len(stim),
        "n_skipped_missing": skipped,
        "stimulus_detection_rate_overall": (
            sum(1 for r in stim if r["hit"]) / len(stim) if stim else None),
        "by_shape": by_shape,
        "cells": cells,
        "note": ("跨检测器只比检出率与 a50/a90；score/tau 与 PatchCore 不可比。"
                 "no_response 表示全档检出率 ≤5%，按协议 5.x 判据不适用。"),
    }

    target.mkdir(parents=True, exist_ok=True)
    (target / "scores.json").write_text(
        json.dumps({**result, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (target / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys

    codes = sys.argv[1:] or ["A", "B"]
    out = {}
    for c in codes:
        r = run(c)
        out[c] = {k: r[k] for k in ("tau", "baseline_false_positive", "n_baseline",
                                    "stimulus_detection_rate_overall")}
        out[c]["by_shape"] = {s: {"overall": round(v["detection_rate_overall"], 3),
                                  "low": v["lowest_cell_rate"], "high": v["highest_cell_rate"],
                                  "no_response": v["no_response"]}
                              for s, v in r["by_shape"].items()}
    print(json.dumps(out, ensure_ascii=False, indent=1))
