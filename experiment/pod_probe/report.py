"""汇总、判据与曲线。不重算任何分数，只读 scores.json。"""

from __future__ import annotations

import csv
import json

import numpy as np

from . import protocol as P


def _cell(rows: list[dict], shape: str, level: float) -> dict:
    sel = [r for r in rows if r["kind"] == "stimulus" and r["shape"] == shape
           and abs(r["target_SEMU"] - level) < 1e-12]
    n = len(sel)
    hits = sum(1 for r in sel if r["hit"])
    inmask = sum(1 for r in sel if r.get("peak_in_mask"))
    deltas = np.asarray([r["delta_score"] for r in sel], dtype=np.float64)
    semu = np.asarray([r["SEMU_actual"] for r in sel], dtype=np.float64)
    return {"shape": shape, "target_SEMU": level, "n": n, "n_expected": 20,
            "hits": hits, "detection_rate": hits / n if n else None,
            "peak_in_mask_rate": inmask / n if n else None,
            "delta_score_median": float(np.median(deltas)) if n else None,
            "SEMU_actual_median": float(np.median(semu)) if n else None}


def run() -> dict:
    score_path = P.out_dir("scores") / "scores.json"
    if not score_path.exists():
        raise RuntimeError("scores missing")
    data = json.loads(score_path.read_text(encoding="utf-8"))
    rows = data["rows"]

    base = [r for r in rows if r["kind"] == "baseline"]
    base_fp = sum(1 for r in base if r["hit"])
    base_rate = base_fp / len(base) if base else None

    cells = [_cell(rows, shape, level) for shape in P.SHAPES for level in P.SEMU_LEVELS]

    # 逐形态判读。n 不足的档位不参与（分母过小会让曲线低端不可信）。
    # 单调性给一个背景的容差：低检出率区间里 1/17 与 1/18 的差是采样噪声，不是回落。
    MIN_N = 10
    TOL = 1.0 / 20.0
    series, monotonic, crosses, saturated, flat_zero, insufficient = {}, {}, {}, {}, {}, {}
    for shape in P.SHAPES:
        seq = [(c["target_SEMU"], c["detection_rate"]) for c in cells
               if c["shape"] == shape and c["detection_rate"] is not None and c["n"] >= MIN_N]
        series[shape] = seq
        insufficient[shape] = len(seq) < 3
        monotonic[shape] = (not insufficient[shape]
                            and all(b[1] >= a[1] - TOL for a, b in zip(seq, seq[1:])))
        crosses[shape] = any(a[1] <= 0.20 and b[1] >= 0.80 for i, a in enumerate(seq)
                             for b in seq[i + 1:])
        saturated[shape] = bool(seq) and all(a[1] >= 0.80 for a in seq)
        flat_zero[shape] = bool(seq) and all(a[1] <= 0.20 for a in seq)

    usable = {s: bool(monotonic[s] and crosses[s]) for s in P.SHAPES}

    # 判据 3：基线失稳
    base_unstable = bool(base_rate is not None and base_rate > 0.30)

    passed = any(usable.values()) and not base_unstable

    out = P.out_dir("report")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "curve.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)

    summary = {
        "version": P.VERSION, "tau": data["tau"],
        "baseline_count": len(base), "baseline_false_positive": base_fp,
        "baseline_false_positive_rate": base_rate, "baseline_unstable": base_unstable,
        "monotonic": monotonic, "crosses_20_to_80": crosses, "usable": usable,
        "saturated": saturated, "flat_zero": flat_zero, "insufficient": insufficient,
        "min_n_per_cell": MIN_N, "monotonic_tolerance": TOL,
        "criterion_passed": passed, "cells": cells,
        "count_not_representable": _count_not_representable(),
        "note": ("leather domain, single polarity, up to 20 backgrounds, new 256 instance, "
                 "no training; SEMU below the 8-bit representable floor is not rendered. "
                 "saturated = all cells >= 0.80 (detector threshold below representable floor); "
                 "flat_zero = all cells <= 0.20 (no response or threshold above measurable range)"),
    }
    P.write_json(out / "summary.json", summary)
    _plot(cells, out / "curve.png")
    P.write_json(out / "complete.json", {"stage": "report", "criterion_passed": passed})
    return summary


def _count_not_representable() -> int:
    path = P.out_dir("stimuli") / "index.json"
    if not path.exists():
        return 0
    idx = json.loads(path.read_text(encoding="utf-8"))
    return int(idx.get("count_not_representable", 0))


def _plot(cells: list[dict], path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=160)
    for shape in P.SHAPES:
        seq = [c for c in cells if c["shape"] == shape
               and c["detection_rate"] is not None and c["n"] >= 10]
        if not seq:
            continue
        ax.plot([c["target_SEMU"] for c in seq], [c["detection_rate"] for c in seq],
                marker="o", markersize=3.5, linewidth=1.4, label=shape)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("target SEMU (log2)")
    ax.set_ylabel("detection rate (n=20 backgrounds)")
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, color="#888780", linewidth=0.6, linestyle=":")
    ax.grid(True, linewidth=0.4, color="#D3D1C7")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
